from __future__ import annotations
import torch

def estimate_transition_matrix_from_forward_backward(model, X_train: torch.Tensor) -> torch.Tensor:
    fb = model.forward_backward(X_train)
    total = None

    if isinstance(fb, (tuple, list)) and len(fb) > 0:
        batch_like = fb[0]
        # pomegranate 1.x returns expected transition counts as 3D tensor [B, K, K]
        if isinstance(batch_like, torch.Tensor) and batch_like.ndim == 3:
            total = batch_like.sum(dim=0)  # sum across batches -> [K, K]
        elif isinstance(batch_like, (tuple, list)):
            for item in batch_like:
                M = None
                if isinstance(item, torch.Tensor) and item.ndim == 2:
                    M = item
                elif isinstance(item, (tuple, list)) and len(item) > 0 and isinstance(item[0], torch.Tensor) and item[0].ndim == 2:
                    M = item[0]
                if M is not None:
                    total = M if total is None else (total + M)

    if total is None:
        raise RuntimeError(
            "Could not parse forward_backward() output to extract transition counts. "
            "Inspect its structure and update qt_hmm/models/transition.py."
        )

    total = total.clamp_min(0.0)
    return total / (total.sum(dim=1, keepdim=True) + 1e-12)
