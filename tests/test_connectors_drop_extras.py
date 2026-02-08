from __future__ import annotations

from pathlib import Path

import pandas as pd

from meta_elv.connectors.meta_csv.loader import load_leads


def test_leads_loader_drops_non_canonical_columns(tmp_path: Path) -> None:
    p = tmp_path / "leads.csv"
    df = pd.DataFrame(
        [
            {
                "lead_id": "l1",
                "created_time": "2026-02-08T12:00:00Z",
                "campaign_id": "c1",
                "email": "person@example.com",  # should be dropped
            }
        ]
    )
    df.to_csv(p, index=False)

    res = load_leads(p)
    assert "email" not in res.df.columns
    assert "lead_id" in res.df.columns
    assert "created_time" in res.df.columns
    # Should warn about dropped columns
    assert any("Dropped non-canonical columns" in w for w in res.warnings)

