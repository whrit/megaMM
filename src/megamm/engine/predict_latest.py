from __future__ import annotations
from typing import Dict, List
import pandas as pd
import torch
from ..config import AppConfig
from ..dataset import align_intersection, to_tensor
from ..features.schema import FEATURE_COLUMNS
from ..models.predict import predict_one
from ..models.artifacts import load_best
from ..models.train import resolve_device
from ..paths import ensure_dirs, ARTIFACTS, feat_path


def load_features(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for tk in tickers:
        p = feat_path(tk)
        if not p.exists():
            raise FileNotFoundError(f"Missing features for {tk}. Run `qt-hmm features` first.")
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        out[tk] = df
    return out


def get_model_device(model) -> str:
    """Get the device string for a pomegranate model."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def predict_latest(tickers: List[str], cfg: AppConfig) -> pd.DataFrame:
    """Generate predictions for all tickers using the best saved model.

    Uses the configured device for inference (GPU if available and configured).
    """
    device = resolve_device(cfg.model.device)
    model, A, _meta = load_best(device=str(device))
    ensure_dirs()

    # Get actual device from model (may differ if fallback occurred during load)
    model_device = get_model_device(model)

    feats = load_features(tickers)
    aligned = align_intersection(feats, min_len=800)
    tickers_sorted = sorted(aligned.keys())
    dates = aligned[tickers_sorted[0]].index
    asof = dates[-1]

    rows = []
    for tk in tickers_sorted:
        X_prefix = to_tensor({tk: aligned[tk].loc[dates, FEATURE_COLUMNS]}, device=model_device).X
        pred = predict_one(model, A, X_prefix, cfg.thresholds)
        rows.append({
            "asof": asof,
            "ticker": tk,
            "p_down": pred.p_down,
            "p_mid": pred.p_mid,
            "p_up": pred.p_up,
            "mu_r_t1": pred.mu_r_t1,
            "var_r_t1": pred.var_r_t1,
        })

    out = pd.DataFrame(rows).sort_values(["ticker"])
    (ARTIFACTS / "predictions").mkdir(parents=True, exist_ok=True)
    out.to_parquet(ARTIFACTS / "predictions" / "predictions.parquet", index=False)
    out.to_csv(ARTIFACTS / "predictions" / "predictions.csv", index=False)
    return out
