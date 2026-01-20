from __future__ import annotations
import torch
import math

def normal_tail_ge(a: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    z = (a - mu) / (sigma * math.sqrt(2.0) + 1e-12)
    return 0.5 * torch.special.erfc(z)

def normal_tail_le(a: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    z = (mu - a) / (sigma * math.sqrt(2.0) + 1e-12)
    return 0.5 * torch.special.erfc(z)
