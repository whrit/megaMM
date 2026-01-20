from __future__ import annotations
import time
import pandas as pd
import typer
from rich import print

from .config import load_config, load_universe
from .paths import ensure_dirs, ARTIFACTS
from .data.download import download_universe
from .features.build import build_features_for_universe
from .backtest.walkforward import walk_forward
from .dataset import align_intersection, to_tensor
from .models.train import train_dense_hmm, resolve_device
from .models.transition import estimate_transition_matrix_from_forward_backward
from .models.artifacts import save_best
from .engine.predict_latest import predict_latest

app = typer.Typer(no_args_is_help=True)

@app.command()
def download(force: bool = typer.Option(False, help="Redownload even if cached")):
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()
    print(f"Downloading {len(tickers)} tickers...")
    download_universe(tickers, cfg.data, force=force)
    print("[green]Done[/green]. Raw data in artifacts/data/raw/")

@app.command()
def features(force: bool = typer.Option(False, help="Recompute even if cached")):
    cfg = load_config()
    tickers = load_universe()
    ensure_dirs()
    raw = download_universe(tickers, cfg.data, force=False)
    print("Computing features...")
    build_features_for_universe(raw, cfg.features, force=force)
    print("[green]Done[/green]. Features in artifacts/data/features/")

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
