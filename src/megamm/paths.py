from __future__ import annotations
from pathlib import Path

ARTIFACTS = Path("artifacts")

def ensure_dirs() -> None:
    (ARTIFACTS / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "data" / "features").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "models").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "models" / "best").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "reports").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "predictions").mkdir(parents=True, exist_ok=True)

def raw_path(ticker: str) -> Path:
    return ARTIFACTS / "data" / "raw" / f"{ticker}.parquet"

def feat_path(ticker: str) -> Path:
    return ARTIFACTS / "data" / "features" / f"{ticker}.parquet"
