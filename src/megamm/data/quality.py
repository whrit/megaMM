from __future__ import annotations
import pandas as pd

def repair_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Fix minor OHLC inconsistencies from yfinance data."""
    df = df.copy()
    oc_min = df[["Open", "Close"]].min(axis=1)
    oc_max = df[["Open", "Close"]].max(axis=1)
    df["Low"] = df["Low"].clip(upper=oc_min)
    df["High"] = df["High"].clip(lower=oc_max)
    df["Low"] = df["Low"].clip(upper=df["High"])
    return df

def validate_ohlcv(df: pd.DataFrame, *, ticker: str = "") -> None:
    if df.empty:
        raise ValueError(f"{ticker}: empty dataframe")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: index not monotonic increasing")
    if df.index.duplicated().any():
        raise ValueError(f"{ticker}: duplicate timestamps")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            raise ValueError(f"{ticker}: missing column {c}")

    if (df["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")

    if (df["Low"] > df[["Open", "Close"]].min(axis=1)).any():
        raise ValueError(f"{ticker}: Low > min(Open,Close) on some rows")
    if (df["High"] < df[["Open", "Close"]].max(axis=1)).any():
        raise ValueError(f"{ticker}: High < max(Open,Close) on some rows")
    if (df["Low"] > df["High"]).any():
        raise ValueError(f"{ticker}: Low > High on some rows")
