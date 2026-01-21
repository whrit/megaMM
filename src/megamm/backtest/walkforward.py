from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import pandas as pd
import torch

from ..config import AppConfig
from ..dataset import align_intersection, to_tensor
from ..features.schema import FEATURE_COLUMNS
from ..models.train import train_dense_hmm, resolve_device
from ..models.transition import estimate_transition_matrix_from_forward_backward
from ..models.predict import predict_one
from ..paths import ARTIFACTS
from .metrics import log_loss_3class, brier_3class, tail_metrics
from ..reporting.console import (
    console,
    create_progress,
    log_section,
    log_stats_table,
    log_metrics_comparison,
    log_info,
    log_success,
)
from ..reporting.stats import compute_walkforward_splits_stats, compute_class_balance
from ..reporting.charts import plot_walkforward_metrics, plot_calibration, plot_confusion_matrix

@dataclass(frozen=True)
class WFResult:
    k: int
    log_loss: float
    brier: float
    up_precision: float
    up_recall: float
    down_precision: float
    down_recall: float
    n_samples: int

def make_labels_from_returns(ret_next: pd.Series, up_thr: float, down_thr: float) -> pd.Series:
    y = pd.Series(index=ret_next.index, dtype="float64")
    y.loc[ret_next <= down_thr] = 0
    y.loc[(ret_next > down_thr) & (ret_next < up_thr)] = 1
    y.loc[ret_next >= up_thr] = 2
    return y

def walk_forward(features: Dict[str, pd.DataFrame], cfg: AppConfig, generate_charts: bool = True) -> List[WFResult]:
    """Run walk-forward evaluation with progress tracking and visualization.

    Args:
        features: Dictionary of ticker -> feature DataFrame
        cfg: Application config
        generate_charts: Whether to generate visualization charts

    Returns:
        List of WFResult for each K value
    """
    log_section("Walk-Forward Evaluation", "Expanding window cross-validation")

    aligned = align_intersection(features, min_len=cfg.eval.min_train_years * 252 + cfg.eval.test_block_days + 20)
    tickers = sorted(aligned.keys())
    dates = aligned[tickers[0]].index

    # Log data summary
    log_stats_table({
        "Tickers": len(tickers),
        "Date Range": f"{dates[0].date()} to {dates[-1].date()}",
        "Total Days": len(dates),
        "Features": len(FEATURE_COLUMNS),
        "Device": cfg.model.device,
    }, title="Dataset Summary")
    resolved_device = resolve_device(cfg.model.device)
    log_info(f"Requested device: {cfg.model.device}, using: {resolved_device}")

    up_thr = float(torch.log(torch.tensor(1.0 + cfg.thresholds.up_pct)).item())
    down_thr = float(torch.log(torch.tensor(1.0 + cfg.thresholds.down_pct)).item())

    labels = {}
    for tk in tickers:
        ret1 = aligned[tk]["ret_1"]
        y = make_labels_from_returns(ret1.shift(-1), up_thr, down_thr)
        labels[tk] = y

    min_train = cfg.eval.min_train_years * 252
    test_block = cfg.eval.test_block_days
    step = cfg.eval.step_days

    # Compute split statistics
    split_stats = compute_walkforward_splits_stats(len(dates), min_train, test_block, step)
    n_splits = split_stats["n_splits"]

    log_stats_table({
        "Min Training Days": min_train,
        "Test Block Days": test_block,
        "Step Days": step,
        "Total Splits": n_splits,
        "K Candidates": str(cfg.model.k_candidates),
    }, title="Walk-Forward Configuration")

    results: List[WFResult] = []
    all_predictions: Dict[int, Tuple[List[int], List[Tuple[float,float,float]]]] = {}

    # Total iterations = K candidates * splits
    total_iterations = len(cfg.model.k_candidates) * n_splits

    with create_progress() as progress:
        main_task = progress.add_task("[cyan]Walk-Forward Evaluation", total=total_iterations)

        for k in cfg.model.k_candidates:
            all_y: List[int] = []
            all_p: List[Tuple[float,float,float]] = []

            train_end_idx = min_train
            split_i = 0

            while train_end_idx + test_block < len(dates) - 2:
                split_i += 1
                split_start = time.time()

                train_dates = dates[:train_end_idx]
                test_dates = dates[train_end_idx:train_end_idx + test_block]

                # Update progress description for training phase
                progress.update(main_task, description=f"[cyan]K={k} Split {split_i}/{n_splits} [Training]")

                # Use configured device (with automatic fallback in train_dense_hmm)
                device = resolve_device(cfg.model.device)
                train_panel = to_tensor({tk: aligned[tk].loc[train_dates, FEATURE_COLUMNS] for tk in tickers}, device=str(device))
                tr = train_dense_hmm(train_panel.X, k=k, cfg=cfg.model, verbose=False)

                # Use actual device from trained model (may differ if fallback occurred)
                model_device = tr.device
                A = estimate_transition_matrix_from_forward_backward(tr.model, train_panel.X.to(model_device))

                # Update progress for prediction phase
                progress.update(main_task, description=f"[cyan]K={k} Split {split_i}/{n_splits} [Predicting]")

                # Run predictions for all test dates
                n_preds = len(test_dates) * len(tickers)
                pred_count = 0
                for d in test_dates:
                    prefix_dates = dates[dates <= d]
                    for tk in tickers:
                        y = labels[tk].loc[d]
                        if pd.isna(y):
                            pred_count += 1
                            continue
                        X_prefix = to_tensor({tk: aligned[tk].loc[prefix_dates, FEATURE_COLUMNS]}, device=model_device).X
                        pred = predict_one(tr.model, A, X_prefix, cfg.thresholds)
                        all_y.append(int(y))
                        all_p.append((pred.p_down, pred.p_mid, pred.p_up))
                        pred_count += 1

                        # Update description with prediction progress every 100 predictions
                        if pred_count % 100 == 0:
                            progress.update(main_task, description=f"[cyan]K={k} Split {split_i}/{n_splits} [Pred {pred_count}/{n_preds}]")

                train_end_idx += step
                progress.advance(main_task)

            # Store predictions for charts
            all_predictions[k] = (all_y, all_p)

            ll = log_loss_3class(all_y, all_p)
            br = brier_3class(all_y, all_p)
            tails = tail_metrics(all_y, all_p, threshold=0.5)

            results.append(WFResult(
                k=k,
                log_loss=ll,
                brier=br,
                up_precision=tails["up_precision"],
                up_recall=tails["up_recall"],
                down_precision=tails["down_precision"],
                down_recall=tails["down_recall"],
                n_samples=len(all_y),
            ))

            # Log per-K summary
            log_info(f"K={k}: Log Loss={ll:.4f}, Brier={br:.4f}, Samples={len(all_y):,}")

    # Display results table
    console.print()
    log_metrics_comparison([r.__dict__ for r in results])

    # Class balance for best K
    best_k = min(results, key=lambda r: r.log_loss).k
    best_y, best_p = all_predictions[best_k]
    balance = compute_class_balance(best_y)
    console.print()
    log_stats_table({
        "Down": f"{balance['classes']['Down']['count']:,} ({balance['classes']['Down']['pct']:.1f}%)",
        "Mid": f"{balance['classes']['Mid']['count']:,} ({balance['classes']['Mid']['pct']:.1f}%)",
        "Up": f"{balance['classes']['Up']['count']:,} ({balance['classes']['Up']['pct']:.1f}%)",
        "Imbalance Ratio": f"{balance['imbalance_ratio']:.2f}",
    }, title=f"Class Balance (K={best_k})")

    # Generate charts
    if generate_charts:
        console.print()
        log_info("Generating evaluation charts...")
        charts_dir = ARTIFACTS / "reports" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Metrics comparison chart
        results_df = pd.DataFrame([r.__dict__ for r in results])
        plot_walkforward_metrics(results_df, charts_dir / "walkforward_metrics.png")
        log_success(f"Saved {charts_dir / 'walkforward_metrics.png'}")

        # Calibration plot for best K
        plot_calibration(best_y, best_p, charts_dir / f"calibration_k{best_k}.png",
                        title=f"Probability Calibration (K={best_k})")
        log_success(f"Saved {charts_dir / f'calibration_k{best_k}.png'}")

        # Confusion matrix for best K
        import numpy as np
        y_pred = [np.argmax(p) for p in best_p]
        plot_confusion_matrix(best_y, y_pred, charts_dir / f"confusion_k{best_k}.png",
                             title=f"Confusion Matrix (K={best_k})")
        log_success(f"Saved {charts_dir / f'confusion_k{best_k}.png'}")

    return results
