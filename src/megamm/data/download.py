from __future__ import annotations
from typing import Dict, List
import pandas as pd
import yfinance as yf
from ..config import DataConfig
from ..paths import ensure_dirs, raw_path

def download_one(ticker: str, cfg: DataConfig) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        back_adjust=cfg.back_adjust,
        repair=cfg.repair,
        keepna=cfg.keepna,
    )
    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing expected columns {missing}. Got {list(df.columns)}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def download_universe(tickers: List[str], cfg: DataConfig, force: bool = False) -> Dict[str, pd.DataFrame]:
    ensure_dirs()
    out: Dict[str, pd.DataFrame] = {}
    for tk in tickers:
        p = raw_path(tk)
        if p.exists() and not force:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            out[tk] = df
            continue
        df = download_one(tk, cfg)
        df.to_parquet(p)
        out[tk] = df
    return out
