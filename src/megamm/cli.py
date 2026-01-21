from __future__ import annotations
import time
import pandas as pd
import torch
import typer

from .config import load_config, load_universe
from .paths import ensure_dirs, ARTIFACTS
from .data.download import download_universe
from .features.build import build_features_for_universe
from .features.schema import FEATURE_COLUMNS
from .backtest.walkforward import walk_forward
from .dataset import align_intersection, to_tensor
from .models.train import train_dense_hmm, resolve_device
from .models.transition import estimate_transition_matrix_from_forward_backward
from .models.artifacts import save_best
from .engine.predict_latest import predict_latest
from .reporting.console import (
    console,
    log_section,
    log_stats_table,
    log_model_summary,
    log_prediction_summary,
    log_data_summary,
    log_success,
    log_info,
    log_warning,
    create_progress,
)
from .reporting.stats import format_prediction_summary
from .reporting.charts import (
    plot_transition_matrix,
    plot_prediction_distribution,
    plot_feature_correlations,
)

app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich")

@app.command()
def download(force: bool = typer.Option(False, help="Redownload even if cached")):
    """Download OHLCV data for all tickers in the universe."""
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()

    log_section("Data Download", f"Fetching OHLCV data for {len(tickers)} tickers")

    log_stats_table({
        "Tickers": len(tickers),
        "Interval": cfg.data.interval,
        "Period": cfg.data.period,
        "Force Redownload": force,
    }, title="Download Configuration")

    with create_progress() as progress:
        task = progress.add_task("[cyan]Downloading", total=len(tickers))
        raw = download_universe(tickers, cfg.data, force=force)
        progress.update(task, completed=len(tickers))

    # Summary statistics
    total_rows = sum(len(df) for df in raw.values())
    date_ranges = [(df.index.min(), df.index.max()) for df in raw.values() if len(df) > 0]
    if date_ranges:
        min_date = min(d[0] for d in date_ranges)
        max_date = max(d[1] for d in date_ranges)
    else:
        min_date = max_date = "N/A"

    console.print()
    log_stats_table({
        "Total Rows": f"{total_rows:,}",
        "Date Range": f"{min_date} to {max_date}",
        "Output Path": "artifacts/data/raw/",
    }, title="Download Summary")

    log_success("Data download complete")

@app.command()
def features(
    force: bool = typer.Option(False, help="Recompute even if cached"),
    plot_corr: bool = typer.Option(False, "--plot-corr", help="Generate feature correlation chart"),
):
    """Compute feature set from OHLCV data."""
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()

    log_section("Feature Engineering", f"Computing {len(FEATURE_COLUMNS)} features for {len(tickers)} tickers")

    log_stats_table({
        "Features": ", ".join(FEATURE_COLUMNS[:5]) + "...",
        "Vol Window": cfg.features.vol_window,
        "MA Window": cfg.features.ma_window,
        "RSI Window": cfg.features.rsi_window,
        "Z-Score Window": cfg.features.z_window,
        "Force Recompute": force,
    }, title="Feature Configuration")

    raw = download_universe(tickers, cfg.data, force=False)

    with create_progress() as progress:
        task = progress.add_task("[cyan]Computing features", total=len(tickers))

        def update_progress():
            progress.advance(task)

        build_features_for_universe(raw, cfg.features, force=force)
        progress.update(task, completed=len(tickers))

    # Load features for summary
    feat_files = list((ARTIFACTS / "data" / "features").glob("*.parquet"))
    total_rows = 0
    sample_df = None
    for f in feat_files:
        df = pd.read_parquet(f)
        total_rows += len(df)
        if sample_df is None:
            sample_df = df

    console.print()
    log_stats_table({
        "Tickers Processed": len(feat_files),
        "Total Feature Rows": f"{total_rows:,}",
        "Features per Row": len(FEATURE_COLUMNS),
        "Output Path": "artifacts/data/features/",
    }, title="Feature Summary")

    # Generate correlation chart if requested
    if plot_corr and sample_df is not None:
        charts_dir = ARTIFACTS / "reports" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        plot_feature_correlations(
            sample_df[FEATURE_COLUMNS],
            charts_dir / "feature_correlations.png",
            title="Feature Correlations (Sample Ticker)"
        )
        log_success(f"Saved {charts_dir / 'feature_correlations.png'}")

    log_success("Feature computation complete")

@app.command()
def walkforward(
    generate_charts: bool = typer.Option(True, "--charts/--no-charts", help="Generate evaluation charts"),
):
    """Run walk-forward evaluation and train best model."""
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()

    log_section("Walk-Forward Pipeline", f"Evaluating K candidates on {len(tickers)} tickers")

    # Load features
    feats = {}
    with create_progress() as progress:
        task = progress.add_task("[cyan]Loading features", total=len(tickers))
        for tk in tickers:
            p = ARTIFACTS / "data" / "features" / f"{tk}.parquet"
            if not p.exists():
                raise FileNotFoundError(f"Missing features for {tk}. Run `qt-hmm features` first.")
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            feats[tk] = df
            progress.advance(task)

    # Run walk-forward evaluation
    results = walk_forward(feats, cfg, generate_charts=generate_charts)

    # Save results
    rep = pd.DataFrame([r.__dict__ for r in results]).sort_values("log_loss")
    out_csv = ARTIFACTS / "reports" / "walkforward_metrics.csv"
    rep.to_csv(out_csv, index=False)
    log_success(f"Saved metrics to {out_csv}")

    # Train final model on full history
    best_k = int(rep.iloc[0]["k"])
    console.print()
    log_section("Final Model Training", f"Training on full history with K={best_k}")

    aligned = align_intersection(feats, min_len=cfg.eval.min_train_years * 252 + 20)

    # Use configured device (CUDA/MPS/CPU) with automatic fallback
    device = resolve_device(cfg.model.device)
    log_info(f"Requested device: {cfg.model.device}, using: {device}")

    panel = to_tensor(aligned, device=str(device))

    log_info("Training final model (this may take a moment)...")
    tr = train_dense_hmm(panel.X, k=best_k, cfg=cfg.model, verbose=True)

    # Use actual device from trained model (may differ if fallback occurred)
    model_device = tr.device
    A = estimate_transition_matrix_from_forward_backward(tr.model, panel.X.to(model_device))

    # Log model summary
    console.print()
    log_model_summary(
        k=best_k,
        train_seconds=tr.train_seconds,
        device=model_device,
        n_tickers=len(panel.tickers),
        n_dates=len(panel.dates),
        transition_matrix=A,
    )

    # Save transition matrix chart
    if generate_charts:
        charts_dir = ARTIFACTS / "reports" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        plot_transition_matrix(
            A.cpu().numpy() if hasattr(A, 'cpu') else A,
            charts_dir / "transition_matrix.png",
            title=f"Final Model Transition Matrix (K={best_k})"
        )
        log_success(f"Saved {charts_dir / 'transition_matrix.png'}")

    meta = {
        "best_k": best_k,
        "trained_at_unix": time.time(),
        "tickers": panel.tickers,
        "start": str(panel.dates[0].date()),
        "end": str(panel.dates[-1].date()),
        "train_seconds": tr.train_seconds,
        "device": model_device,
    }
    save_best(tr.model, A, cfg, meta)

    console.print()
    log_stats_table({
        "Best K": best_k,
        "Training Time": f"{tr.train_seconds:.2f}s",
        "Device": model_device,
        "Tickers": len(panel.tickers),
        "Date Range": f"{panel.dates[0].date()} to {panel.dates[-1].date()}",
        "Output": "artifacts/models/best/",
    }, title="Final Model Summary")

    log_success("Walk-forward pipeline complete")

@app.command()
def predict(
    plot_dist: bool = typer.Option(False, "--plot-dist", help="Generate prediction distribution chart"),
):
    """Generate predictions using the trained model."""
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()

    log_section("Prediction Generation", f"Generating predictions for {len(tickers)} tickers")

    with create_progress() as progress:
        task = progress.add_task("[cyan]Running predictions", total=1)
        df = predict_latest(tickers, cfg)
        progress.update(task, completed=1)

    # Compute and display summary
    summary = format_prediction_summary(df)
    console.print()
    log_prediction_summary(summary)

    # Generate distribution chart if requested
    if plot_dist and "p_up" in df.columns and "p_down" in df.columns:
        charts_dir = ARTIFACTS / "reports" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        plot_prediction_distribution(
            df[["p_down", "p_mid", "p_up"]].values,
            charts_dir / "prediction_distribution.png",
            title=f"Prediction Distribution ({summary.get('asof_date', 'Latest')})"
        )
        log_success(f"Saved {charts_dir / 'prediction_distribution.png'}")

    # Save predictions
    out_pq = ARTIFACTS / "predictions" / "predictions.parquet"
    out_csv = ARTIFACTS / "predictions" / "predictions.csv"
    df.to_parquet(out_pq)
    df.to_csv(out_csv, index=False)

    console.print()
    log_stats_table({
        "Tickers": len(df),
        "As-of Date": summary.get("asof_date", "N/A"),
        "Bullish": summary.get("bullish_count", "N/A"),
        "Neutral": summary.get("neutral_count", "N/A"),
        "Bearish": summary.get("bearish_count", "N/A"),
        "Output": "artifacts/predictions/",
    }, title="Prediction Output")

    log_success("Prediction generation complete")

@app.command()
def device():
    """Show device availability and resolved runtime device."""
    cfg = load_config()
    requested = cfg.model.device
    resolved = resolve_device(requested)

    log_section("Device Probe", "Detect GPU/MPS availability and resolve runtime device")
    log_stats_table({
        "Requested": requested,
        "Resolved": str(resolved),
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda or "N/A",
        "MPS Available": torch.backends.mps.is_available(),
    }, title="Device Availability")

    if requested == "cuda" and not torch.cuda.is_available():
        log_warning("CUDA requested but not available; falling back to CPU.")
    elif requested == "mps" and not torch.backends.mps.is_available():
        log_warning("MPS requested but not available; falling back to CPU.")
    elif requested == "auto":
        log_info("Auto selects CUDA if available, otherwise MPS, otherwise CPU.")

if __name__ == "__main__":
    app()
