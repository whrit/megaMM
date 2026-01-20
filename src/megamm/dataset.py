from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from .features.schema import FEATURE_COLUMNS

@dataclass(frozen=True)
class Panel:
    X: torch.Tensor            # (B, T, D)
    tickers: List[str]
    dates: pd.DatetimeIndex

def align_intersection(features: Dict[str, pd.DataFrame], min_len: int = 800) -> Dict[str, pd.DataFrame]:
    filtered = {tk: df for tk, df in features.items() if len(df) >= min_len}
    if len(filtered) < 5:
        raise ValueError(f"Too few tickers after length filter: {len(filtered)}. Lower min_len or add tickers.")

    tickers = sorted(filtered.keys())
    common = filtered[tickers[0]].index
    for tk in tickers[1:]:
        common = common.intersection(filtered[tk].index)

    if len(common) < min_len:
        raise ValueError(f"Intersection calendar too short: {len(common)} days. Consider lowering min_len or using last N years.")

    return {tk: filtered[tk].loc[common, FEATURE_COLUMNS].copy() for tk in tickers}

def to_tensor(aligned: Dict[str, pd.DataFrame], device: str = "cpu") -> Panel:
    tickers = sorted(aligned.keys())
    dates = aligned[tickers[0]].index
    X_np = np.stack([aligned[tk].loc[dates, FEATURE_COLUMNS].to_numpy(dtype=np.float32) for tk in tickers], axis=0)
    X = torch.tensor(X_np, device=device)
    return Panel(X=X, tickers=tickers, dates=dates)
