from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import torch
from rich import print

from ..config import AppConfig
from ..dataset import align_intersection, to_tensor
from ..features.schema import FEATURE_COLUMNS
from ..models.train import train_dense_hmm, resolve_device
from ..models.transition import estimate_transition_matrix_from_forward_backward
from ..models.predict import predict_one
from .metrics import log_loss_3class, brier_3class, tail_metrics

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

def walk_forward(features: Dict[str, pd.DataFrame], cfg: AppConfig) -> List[WFResult]:
    aligned = align_intersection(features, min_len=cfg.eval.min_train_years * 252 + cfg.eval.test_block_days + 20)
    tickers = sorted(aligned.keys())
    dates = aligned[tickers[0]].index

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

    results: List[WFResult] = []

    for k in cfg.model.k_candidates:
        all_y: List[int] = []
        all_p: List[Tuple[float,float,float]] = []

        train_end_idx = min_train
        split_i = 0
        while train_end_idx + test_block < len(dates) - 2:
            split_i += 1
            train_dates = dates[:train_end_idx]
            test_dates = dates[train_end_idx:train_end_idx + test_block]

            # Use configured device (with automatic fallback in train_dense_hmm)
            device = resolve_device(cfg.model.device)
            train_panel = to_tensor({tk: aligned[tk].loc[train_dates, FEATURE_COLUMNS] for tk in tickers}, device=str(device))
            tr = train_dense_hmm(train_panel.X, k=k, cfg=cfg.model)
            # Use actual device from trained model (may differ if fallback occurred)
            model_device = tr.device
            A = estimate_transition_matrix_from_forward_backward(tr.model, train_panel.X.to(model_device))

            for d in test_dates:
                prefix_dates = dates[dates <= d]
                for tk in tickers:
                    y = labels[tk].loc[d]
                    if pd.isna(y):
                        continue
                    X_prefix = to_tensor({tk: aligned[tk].loc[prefix_dates, FEATURE_COLUMNS]}, device=model_device).X
                    pred = predict_one(tr.model, A, X_prefix, cfg.thresholds)
                    all_y.append(int(y))
                    all_p.append((pred.p_down, pred.p_mid, pred.p_up))

            train_end_idx += step
            print(f"K={k} completed split #{split_i}")

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

    return results
