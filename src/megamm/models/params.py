from __future__ import annotations
import torch
from ..features.schema import RETURN_DIM

def get_return_mu_sigma(model) -> tuple[torch.Tensor, torch.Tensor]:
    mus = []
    sigmas = []
    for d in model.distributions:
        mu = _get_tensor(d, ["means", "mean", "loc"])
        mu_r = mu[RETURN_DIM]

        cov = _get_tensor(d, ["covs", "cov", "variances", "var", "scale"])
        if cov.ndim == 2:
            cov = torch.diag(cov)
        var_r = cov[RETURN_DIM].clamp_min(1e-8)
        mus.append(mu_r)
        sigmas.append(torch.sqrt(var_r))
    return torch.stack(mus), torch.stack(sigmas)

def _get_tensor(obj, names: list[str]) -> torch.Tensor:
    for n in names:
        if hasattr(obj, n):
            t = getattr(obj, n)
            if isinstance(t, torch.Tensor):
                return t
    if hasattr(obj, "parameters"):
        params = getattr(obj, "parameters")
        if isinstance(params, dict):
            for n in names:
                if n in params and isinstance(params[n], torch.Tensor):
                    return params[n]
    raise AttributeError(f"Could not find tensor attribute in {names} on {type(obj)}")
