from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return {"psi": None, "bins": bins, "reason": "empty"}

    # Quantile bins from expected
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return {"psi": None, "bins": bins, "reason": "degenerate_edges"}

    # Ensure full coverage
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(1, exp_counts.sum())
    act_pct = act_counts / max(1, act_counts.sum())

    # Avoid div by zero
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, 1.0)
    act_pct = np.clip(act_pct, eps, 1.0)
    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return {"psi": psi, "bins": int(len(edges) - 1)}


def compute_psi_for_columns(
    expected_df: pd.DataFrame, actual_df: pd.DataFrame, cols: list[str], bins: int = 10
) -> dict[str, Any]:
    out: dict[str, Any] = {"bins": bins, "columns": {}}
    for c in cols:
        if c not in expected_df.columns or c not in actual_df.columns:
            continue
        out["columns"][c] = compute_psi(
            expected=expected_df[c].to_numpy(), actual=actual_df[c].to_numpy(), bins=bins
        )
    return out

