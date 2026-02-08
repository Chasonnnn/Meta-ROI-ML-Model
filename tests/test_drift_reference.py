from __future__ import annotations

import numpy as np
import pandas as pd

from meta_elv.drift import build_psi_reference_for_columns, compute_psi_for_columns_from_reference


def test_psi_reference_roundtrip() -> None:
    rng = np.random.default_rng(7)
    train = pd.DataFrame(
        {
            "a": rng.normal(size=1000),
            "b": rng.uniform(size=1000),
        }
    )
    score = pd.DataFrame(
        {
            "a": rng.normal(loc=0.1, scale=1.1, size=800),
            "b": rng.uniform(low=0.05, high=0.95, size=800),
        }
    )

    ref = build_psi_reference_for_columns(train, cols=["a", "b"], bins=10)
    out = compute_psi_for_columns_from_reference(ref, score)
    cols = out.get("columns") or {}
    assert "a" in cols and cols["a"].get("psi") is not None
    assert "b" in cols and cols["b"].get("psi") is not None

