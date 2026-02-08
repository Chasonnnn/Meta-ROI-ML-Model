from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


@dataclass(frozen=True)
class LiftResult:
    topk_frac: float
    k: int
    overall_rate: float
    topk_rate: float
    lift: float
    capture: float


def compute_lift_at_k(y_true: np.ndarray, p: np.ndarray, topk_frac: float) -> LiftResult:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    k = int(np.ceil(n * topk_frac))
    k = max(1, min(k, n))
    order = np.argsort(-p)
    top = y[order[:k]]
    overall = float(y.mean()) if n else 0.0
    top_rate = float(top.mean()) if len(top) else 0.0
    lift = float(top_rate / overall) if overall > 0 else float("nan")
    capture = float(top.sum() / max(1, y.sum()))
    return LiftResult(
        topk_frac=float(topk_frac),
        k=int(k),
        overall_rate=overall,
        topk_rate=top_rate,
        lift=lift,
        capture=capture,
    )


def lift_curve(y_true: np.ndarray, p: np.ndarray, points: int = 50) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    order = np.argsort(-p)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = max(1, int(y_sorted.sum()))
    n = len(y_sorted)

    xs = np.linspace(1, n, num=points, dtype=int)
    pop_frac = (xs / n).tolist()
    capture = (cum_pos[xs - 1] / total_pos).astype(float).tolist()
    return {"population_frac": pop_frac, "positive_capture_frac": capture}


def calibration_bins(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)

    out = {"bins": []}
    n = len(y)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            out["bins"].append({"count": 0, "p_mean": None, "y_rate": None})
            continue
        p_mean = float(p[m].mean())
        y_rate = float(y[m].mean())
        out["bins"].append({"count": cnt, "p_mean": p_mean, "y_rate": y_rate})
        ece += abs(p_mean - y_rate) * (cnt / n)
    out["ece"] = float(ece)
    return out


def compute_core_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }

