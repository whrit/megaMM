from __future__ import annotations
import json
from pathlib import Path
import torch
from ..config import AppConfig
from ..paths import ensure_dirs, ARTIFACTS
from .train import resolve_device


def save_best(model, A: torch.Tensor, cfg: AppConfig, meta: dict) -> None:
    """Save the best model, transition matrix, and config to artifacts."""
    ensure_dirs()
    best = ARTIFACTS / "models" / "best"
    best.mkdir(parents=True, exist_ok=True)
    # Always save model on CPU for portability
    model_cpu = model.cpu() if hasattr(model, 'cpu') else model
    A_cpu = A.cpu() if hasattr(A, 'cpu') else A
    torch.save({"model": model_cpu, "A": A_cpu}, best / "model.pt")
    (best / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (best / "config.json").write_text(json.dumps({
        "data": cfg.data.__dict__,
        "features": cfg.features.__dict__,
        "model": cfg.model.__dict__,
        "eval": cfg.eval.__dict__,
        "thresholds": cfg.thresholds.__dict__,
    }, indent=2), encoding="utf-8")


def load_best(device: str = "cpu") -> tuple[object, torch.Tensor, dict]:
    """Load the best model, optionally moving to a specific device.

    Args:
        device: Target device ('cpu', 'cuda', 'mps'). Uses resolve_device for validation.

    Returns:
        Tuple of (model, transition_matrix, metadata)
    """
    best = ARTIFACTS / "models" / "best"
    obj = torch.load(best / "model.pt", map_location="cpu", weights_only=False)
    meta = json.loads((best / "meta.json").read_text(encoding="utf-8"))

    model = obj["model"]
    A = obj["A"]

    # Move to target device if requested
    target = resolve_device(device)
    if target.type != "cpu":
        model = model.to(target)
        A = A.to(target)

    return model, A, meta
