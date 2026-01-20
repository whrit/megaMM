from __future__ import annotations
import pandas as pd
from ..config import FeatureConfig
from .schema import FEATURE_COLUMNS

def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / (sd + 1e-12)

def normalize_features(feat: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = feat.copy()

    to_z = ["vol_20", "log_hl", "gap", "dlog_vol", "vol_z", "ma20_dist", "rsi14_scaled", "abs_ret_1"]
    for c in to_z:
        if c in out.columns:
            out[c] = rolling_zscore(out[c], cfg.z_window)

    out = out.clip(lower=-cfg.clip_z, upper=cfg.clip_z)
    return out[FEATURE_COLUMNS]
