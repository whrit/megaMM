from __future__ import annotations
from typing import Dict
import pandas as pd
from ..config import FeatureConfig
from ..paths import ensure_dirs, feat_path
from ..data.quality import repair_ohlcv, validate_ohlcv
from .compute import compute_features
from .normalize import normalize_features

def build_features_for_universe(raw: Dict[str, pd.DataFrame], cfg: FeatureConfig, force: bool = False) -> Dict[str, pd.DataFrame]:
    ensure_dirs()
    out: Dict[str, pd.DataFrame] = {}
    for tk, df in raw.items():
        p = feat_path(tk)
        if p.exists() and not force:
            f = pd.read_parquet(p)
            f.index = pd.to_datetime(f.index)
            f.sort_index(inplace=True)
            out[tk] = f
            continue

        df = repair_ohlcv(df)
        validate_ohlcv(df, ticker=tk)
        feat = compute_features(df, cfg)
        feat = normalize_features(feat, cfg)
        feat = feat.dropna(how="any")
        feat.to_parquet(p)
        out[tk] = feat

    return out
