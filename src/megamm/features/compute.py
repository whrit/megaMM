from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import FeatureConfig
from .schema import FEATURE_COLUMNS

def _rsi_scaled(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(window).mean()
    roll_dn = down.rolling(window).mean()
    rs = roll_up / (roll_dn + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return (rsi / 50.0) - 1.0

def compute_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"].replace(0, np.nan)

    out["ret_1"] = np.log(close / close.shift(1))
    out["abs_ret_1"] = out["ret_1"].abs()
    out["ret_5"] = np.log(close / close.shift(5))
    out["ret_20"] = np.log(close / close.shift(20))

    out["vol_20"] = out["ret_1"].rolling(cfg.vol_window).std()

    out["log_hl"] = np.log(high / low)
    out["gap"] = np.log(open_ / close.shift(1))

    vol_log = np.log(vol)
    out["dlog_vol"] = vol_log.diff()
    vol_mu = vol_log.rolling(cfg.vol_lookback).mean()
    vol_sd = vol_log.rolling(cfg.vol_lookback).std()
    out["vol_z"] = (vol_log - vol_mu) / (vol_sd + 1e-12)

    ma = close.rolling(cfg.ma_window).mean()
    out["ma20_dist"] = (close / ma) - 1.0

    out["rsi14_scaled"] = _rsi_scaled(close, cfg.rsi_window)

    return out[FEATURE_COLUMNS]
