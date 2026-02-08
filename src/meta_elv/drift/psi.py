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


def build_psi_reference(expected: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """
    Build a reusable PSI reference from an "expected" distribution.

    Reference stores internal bin edges (finite quantile cutpoints) and expected bin proportions.
    This avoids serializing +/-inf edges into JSON.
    """
    expected = np.asarray(expected, dtype=float)
    expected = expected[~np.isnan(expected)]
    if len(expected) == 0:
        return {"ok": False, "bins": bins, "reason": "empty"}

    if bins < 2:
        raise ValueError("bins must be >= 2")

    # Internal quantile cutpoints (exclude 0 and 1).
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.unique(np.quantile(expected, qs))
    if len(edges) == 0:
        return {"ok": False, "bins": bins, "reason": "degenerate_edges"}

    idx = np.digitize(expected, edges, right=False)
    exp_counts = np.bincount(idx, minlength=len(edges) + 1)
    exp_pct = exp_counts / max(1, exp_counts.sum())
    return {
        "ok": True,
        "bins": int(len(edges) + 1),
        "edges": edges.astype(float).tolist(),
        "expected_pct": exp_pct.astype(float).tolist(),
    }


def compute_psi_from_reference(reference: dict[str, Any], actual: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    actual = actual[~np.isnan(actual)]
    if len(actual) == 0:
        return {"psi": None, "bins": reference.get("bins"), "reason": "empty_actual"}

    if not reference.get("ok", False):
        return {"psi": None, "bins": reference.get("bins"), "reason": "invalid_reference"}

    edges = np.asarray(reference.get("edges") or [], dtype=float)
    expected_pct = np.asarray(reference.get("expected_pct") or [], dtype=float)
    if len(expected_pct) != len(edges) + 1:
        return {"psi": None, "bins": reference.get("bins"), "reason": "shape_mismatch"}

    idx = np.digitize(actual, edges, right=False)
    act_counts = np.bincount(idx, minlength=len(edges) + 1)
    act_pct = act_counts / max(1, act_counts.sum())

    eps = 1e-6
    exp_pct = np.clip(expected_pct, eps, 1.0)
    act_pct = np.clip(act_pct, eps, 1.0)
    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return {"psi": psi, "bins": int(len(edges) + 1)}


def build_psi_reference_for_columns(
    expected_df: pd.DataFrame, cols: list[str], bins: int = 10
) -> dict[str, Any]:
    out: dict[str, Any] = {"bins": bins, "columns": {}}
    for c in cols:
        if c not in expected_df.columns:
            continue
        out["columns"][c] = build_psi_reference(expected_df[c].to_numpy(), bins=bins)
    return out


def compute_psi_for_columns_from_reference(
    reference: dict[str, Any], actual_df: pd.DataFrame, cols: list[str] | None = None
) -> dict[str, Any]:
    ref_cols = (reference.get("columns") or {}) if isinstance(reference, dict) else {}
    use_cols = cols if cols is not None else list(ref_cols.keys())
    out: dict[str, Any] = {"bins": reference.get("bins"), "columns": {}}
    for c in use_cols:
        if c not in actual_df.columns or c not in ref_cols:
            continue
        out["columns"][c] = compute_psi_from_reference(ref_cols[c], actual_df[c].to_numpy())
    return out


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
