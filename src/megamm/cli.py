from __future__ import annotations
import time
import pandas as pd
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
def walkforward():
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()

    feats = {}
    for tk in tickers:
        p = ARTIFACTS / "data" / "features" / f"{tk}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing features for {tk}. Run `qt-hmm features` first.")
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        feats[tk] = df

    print("Running walk-forward evaluation (this may take a while)...")
    results = walk_forward(feats, cfg)

    rep = pd.DataFrame([r.__dict__ for r in results]).sort_values("log_loss")
    out_csv = ARTIFACTS / "reports" / "walkforward_metrics.csv"
    rep.to_csv(out_csv, index=False)
    print(f"[green]Wrote[/green] {out_csv}")

    best_k = int(rep.iloc[0]["k"])
    print(f"Training best model on full aligned history with K={best_k}...")
    aligned = align_intersection(feats, min_len=cfg.eval.min_train_years * 252 + 20)

    # Use configured device (CUDA/MPS/CPU) with automatic fallback
    device = resolve_device(cfg.model.device)
    print(f"Using device: {device}")
    panel = to_tensor(aligned, device=str(device))
    tr = train_dense_hmm(panel.X, k=best_k, cfg=cfg.model)

    # Use actual device from trained model (may differ if fallback occurred)
    model_device = tr.device
    A = estimate_transition_matrix_from_forward_backward(tr.model, panel.X.to(model_device))

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
    print(f"[green]Saved best model[/green] to artifacts/models/best/ (trained on {model_device})")

@app.command()
def predict():
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()
    df = predict_latest(tickers, cfg)
    print(df.head(10))
    print("[green]Wrote[/green] artifacts/predictions/predictions.parquet and .csv")

if __name__ == "__main__":
    app()
