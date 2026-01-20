from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

@dataclass(frozen=True)
class DataConfig:
    interval: str = "1d"
    period: str = "max"
    auto_adjust: bool = True
    back_adjust: bool = False
    repair: bool = True
    keepna: bool = False

@dataclass(frozen=True)
class FeatureConfig:
    vol_window: int = 20
    vol_lookback: int = 60
    ma_window: int = 20
    rsi_window: int = 14
    z_window: int = 252
    clip_z: float = 8.0

@dataclass(frozen=True)
class ModelConfig:
    k_candidates: Tuple[int, ...] = (3, 4, 5, 6)
    max_iter: int = 500
    tol: float = 0.1
    init: str = "random"
    inertia: float = 0.0
    random_state: int = 42
    device: str = "cpu"

@dataclass(frozen=True)
class EvalConfig:
    min_train_years: int = 5
    test_block_days: int = 252
    step_days: int = 63

@dataclass(frozen=True)
class ThresholdConfig:
    up_pct: float = 0.05
    down_pct: float = -0.05

@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    eval: EvalConfig
    thresholds: ThresholdConfig

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def load_config(config_path: str | Path = "configs/config.json") -> AppConfig:
    p = Path(config_path)
    obj = load_json(p)

    data = DataConfig(**obj["data"])
    feat = FeatureConfig(**obj["features"])
    model_dict = dict(obj["model"])
    k_candidates = tuple(model_dict.pop("k_candidates"))
    model = ModelConfig(k_candidates=k_candidates, **model_dict)
    ev = EvalConfig(**obj["eval"])
    thr = ThresholdConfig(**obj["thresholds"])
    return AppConfig(data=data, features=feat, model=model, eval=ev, thresholds=thr)

def load_universe(universe_path: str | Path = "configs/universe.json") -> List[str]:
    p = Path(universe_path)
    obj = load_json(p)
    tickers = obj.get("tickers", [])
    if not tickers:
        raise ValueError("Universe file has no tickers.")
    return [str(t).upper().strip() for t in tickers]
