from __future__ import annotations
from dataclasses import dataclass
import time
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
from ..config import ModelConfig

@dataclass(frozen=True)
class TrainResult:
    model: DenseHMM
    k: int
    train_seconds: float
    device: str


def resolve_device(requested: str) -> torch.device:
    """Resolve requested device string to a torch.device, with availability checks."""
    requested = requested.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested in ("cuda", "mps"):
        # Requested GPU but not available - fall back to CPU
        return torch.device("cpu")
    return torch.device("cpu")


def train_dense_hmm(X: torch.Tensor, k: int, cfg: ModelConfig) -> TrainResult:
    """Train a DenseHMM with GPU support for CUDA/MPS devices.

    Per pomegranate docs, the model and data must be on the same device.
    We create the model on CPU, move it to the target device, then fit.
    """
    t0 = time.time()
    device = resolve_device(cfg.device)

    # Create distributions and model on CPU first
    dists = [Normal(covariance_type="diag") for _ in range(k)]
    model = DenseHMM(
        dists,
        init=cfg.init,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        inertia=cfg.inertia,
        random_state=cfg.random_state,
        verbose=True,
    )

    # Move model and data to target device, then fit
    # For MPS, there can be issues with pomegranate's internal KMeans - fall back to CPU if needed
    try:
        model = model.to(device)
        X_device = X.to(device)
        model.fit(X_device)
    except RuntimeError as e:
        if device.type != "cpu":
            print(f"[yellow]Warning: {device} training failed ({e}), falling back to CPU[/yellow]")
            device = torch.device("cpu")
            model = model.to(device)
            X_device = X.to(device)
            model.fit(X_device)
        else:
            raise

    return TrainResult(model=model, k=k, train_seconds=time.time() - t0, device=str(device))
