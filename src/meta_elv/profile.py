from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .table_builder import BuildTableResult
from .utils import write_json


def _null_pct(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(df) == 0:
        return {c: 0.0 for c in cols if c in df.columns}
    for c in cols:
        if c in df.columns:
            out[c] = float(df[c].isna().mean())
    return out


def build_data_profile(result: BuildTableResult) -> dict[str, Any]:
    df = result.table
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "join": {
            "strategy": result.join_strategy,
            "match_rate": result.join_match_rate,
        },
        "labeling": {
            "as_of_date": result.as_of_date.isoformat(),
            "counts": result.label_summary,
        },
        "null_pct": _null_pct(
            df,
            [
                "campaign_id",
                "adset_id",
                "ad_id",
                "lead_dow",
                "lead_hour",
                "label",
            ],
        ),
        "details": result.details,
    }

    # class balance (labeled only)
    labeled = df[df["label"].notna()] if "label" in df.columns else df.iloc[0:0]
    if len(labeled):
        profile["labeling"]["positive_rate_labeled"] = float(labeled["label"].mean())
        profile["labeling"]["n_labeled"] = int(len(labeled))
    else:
        profile["labeling"]["positive_rate_labeled"] = None
        profile["labeling"]["n_labeled"] = 0

    return profile


def write_data_profile(run_dir: Path, profile: dict[str, Any]) -> None:
    write_json(run_dir / "data_profile.json", profile)

    # Human-readable summary
    lines: list[str] = []
    lines.append("# Data Profile")
    lines.append("")
    lines.append(f"- Rows: {profile.get('rows')}")
    join = profile.get("join", {})
    lines.append(f"- Join strategy: {join.get('strategy')}")
    lines.append(f"- Join match rate: {join.get('match_rate'):.1%}" if join.get("match_rate") is not None else "- Join match rate: n/a")
    labeling = profile.get("labeling", {})
    counts = labeling.get("counts", {})
    lines.append(
        f"- Labels (pos/neg/unk): {counts.get('positive', 0)}/{counts.get('negative', 0)}/{counts.get('unknown', 0)}"
    )
    feats = ((profile.get("details") or {}).get("features") or {})
    if feats:
        lines.append(f"- Feature window days: {feats.get('feature_window_days')}")
        lines.append(f"- Feature lag days: {feats.get('feature_lag_days')}")
    pr = labeling.get("positive_rate_labeled")
    if pr is not None:
        lines.append(f"- Positive rate (labeled only): {pr:.2%}")
    nulls = profile.get("null_pct", {})
    if nulls:
        worst = sorted(nulls.items(), key=lambda kv: kv[1], reverse=True)[:8]
        lines.append("")
        lines.append("## Top Null%")
        for k, v in worst:
            lines.append(f"- {k}: {v:.1%}")

    (run_dir / "data_profile.md").write_text("\n".join(lines) + "\n")
