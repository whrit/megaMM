from __future__ import annotations
import math
from typing import List, Tuple, Dict

def log_loss_3class(y_true: List[int], p: List[Tuple[float,float,float]]) -> float:
    eps = 1e-12
    s = 0.0
    n = len(y_true)
    for y, (pd, pm, pu) in zip(y_true, p):
        probs = [max(eps, pd), max(eps, pm), max(eps, pu)]
        norm = sum(probs)
        probs = [v / norm for v in probs]
        s += -math.log(probs[y])
    return s / max(1, n)

def brier_3class(y_true: List[int], p: List[Tuple[float,float,float]]) -> float:
    s = 0.0
    n = len(y_true)
    for y, (pd, pm, pu) in zip(y_true, p):
        yv = [0.0, 0.0, 0.0]
        yv[y] = 1.0
        pv = [pd, pm, pu]
        s += sum((a - b) ** 2 for a, b in zip(pv, yv))
    return s / max(1, n)

def tail_metrics(y_true: List[int], p: List[Tuple[float,float,float]], threshold: float = 0.5) -> Dict[str, float]:
    def pr_for(label: int, idx: int):
        tp = fp = fn = 0
        for y, probs in zip(y_true, p):
            prob = probs[idx]
            pred = prob >= threshold
            actual = (y == label)
            if pred and actual:
                tp += 1
            if pred and (not actual):
                fp += 1
            if (not pred) and actual:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return prec, rec

    up_p, up_r = pr_for(2, 2)
    dn_p, dn_r = pr_for(0, 0)
    return {"up_precision": up_p, "up_recall": up_r, "down_precision": dn_p, "down_recall": dn_r}
