from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from ..config import ThresholdConfig
from .params import get_return_mu_sigma
from .math import normal_tail_ge, normal_tail_le

@dataclass(frozen=True)
class Prediction:
    p_up: float
    p_mid: float
    p_down: float
    p_z_t: list[float]
    p_z_t1: list[float]
    mu_r_t1: float
    var_r_t1: float

def predict_one(model, A: torch.Tensor, X_prefix: torch.Tensor, thr: ThresholdConfig) -> Prediction:
    post = model.predict_proba(X_prefix)
    if isinstance(post, torch.Tensor):
        gamma = post[0]
    else:
        gamma = post[0]
    pi_t = gamma[-1]
    pi_t1 = pi_t @ A

    mus, sigmas = get_return_mu_sigma(model)

    up_thr = torch.tensor(math.log(1.0 + thr.up_pct), device=mus.device, dtype=mus.dtype)
    dn_thr = torch.tensor(math.log(1.0 + thr.down_pct), device=mus.device, dtype=mus.dtype)

    p_up_k = normal_tail_ge(up_thr, mus, sigmas)
    p_dn_k = normal_tail_le(dn_thr, mus, sigmas)

    p_up = float((pi_t1 * p_up_k).sum().item())
    p_dn = float((pi_t1 * p_dn_k).sum().item())
    p_mid = max(0.0, 1.0 - p_up - p_dn)

    mu_mix = (pi_t1 * mus).sum()
    second = (pi_t1 * (sigmas**2 + mus**2)).sum()
    var_mix = (second - mu_mix**2).clamp_min(0.0)

    return Prediction(
        p_up=p_up,
        p_mid=p_mid,
        p_down=p_dn,
        p_z_t=pi_t.detach().cpu().tolist(),
        p_z_t1=pi_t1.detach().cpu().tolist(),
        mu_r_t1=float(mu_mix.item()),
        var_r_t1=float(var_mix.item()),
    )
