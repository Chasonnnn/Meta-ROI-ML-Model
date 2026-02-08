from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class EvalMetrics:
    pr_auc: float
    brier: float
    roc_auc: float | None
    lift: dict[str, Any]
    ece: dict[str, Any]
    calibration: dict[str, Any]


def _as_float(x: float | np.floating) -> float:
    return float(x)


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, topk_frac: float) -> dict[str, Any]:
    n = len(y_true)
    k = max(1, int(round(n * topk_frac)))
    order = np.argsort(-y_prob)
    top = y_true[order][:k]
    total_pos = float(y_true.sum())
    captured = float(top.sum())
    capture_rate = captured / total_pos if total_pos > 0 else 0.0
    precision_topk = captured / k if k > 0 else 0.0
    base_rate = float(y_true.mean()) if n else 0.0
    lift = (precision_topk / base_rate) if base_rate > 0 else None
    return {
        "topk_frac": topk_frac,
        "k": k,
        "capture_rate": capture_rate,
        "precision_topk": precision_topk,
        "base_rate": base_rate,
        "lift": lift,
    }


def ece_binned(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict[str, Any]:
    # Equal-width bins on predicted probability.
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(y_prob, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    ece = 0.0
    per_bin = []
    n = len(y_true)
    for b in range(bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt == 0:
            per_bin.append({"bin": b, "count": 0})
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        w = cnt / n
        ece += w * abs(acc - conf)
        per_bin.append({"bin": b, "count": cnt, "acc": acc, "conf": conf})

    return {"bins": bins, "ece": float(ece), "per_bin": per_bin}


def evaluate_binary(
    y_true: np.ndarray, y_prob: np.ndarray, topk_frac: float = 0.1, ece_bins: int = 10
) -> EvalMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pr_auc = _as_float(average_precision_score(y_true, y_prob))
    brier = _as_float(brier_score_loss(y_true, y_prob))
    roc_auc = None
    try:
        roc_auc = _as_float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = None

    lift = lift_at_k(y_true, y_prob, topk_frac=topk_frac)
    ece = ece_binned(y_true, y_prob, bins=ece_bins)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=ece_bins, strategy="uniform")
    calibration = {
        "bins": ece_bins,
        "frac_pos": [float(x) for x in frac_pos],
        "mean_pred": [float(x) for x in mean_pred],
    }

    return EvalMetrics(
        pr_auc=pr_auc,
        brier=brier,
        roc_auc=roc_auc,
        lift=lift,
        ece=ece,
        calibration=calibration,
    )

