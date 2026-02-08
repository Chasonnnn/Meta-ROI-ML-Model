from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .connectors import meta_csv
from .connectors.base import ADS_REQUIRED, LEADS_REQUIRED, OUTCOMES_REQUIRED


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    details: dict[str, Any]


def _missing_cols(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def _null_pct(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(df)
    for c in cols:
        if c not in df.columns:
            continue
        if n == 0:
            out[c] = 0.0
        else:
            out[c] = float(df[c].isna().mean())
    return out


def _check_unique(df: pd.DataFrame, key: str) -> tuple[bool, int]:
    if key not in df.columns:
        return True, 0
    dupes = int(df[key].duplicated().sum())
    return dupes == 0, dupes


def validate_from_config(cfg: RunConfig) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"files": {}}

    def require_file(p: Path, label: str) -> None:
        if not p.exists():
            errors.append(f"Missing {label} file: {p}")

    require_file(cfg.paths.ads_path, "ads")
    require_file(cfg.paths.leads_path, "leads")
    if cfg.paths.outcomes_path is not None:
        require_file(cfg.paths.outcomes_path, "outcomes")
    if cfg.paths.lead_to_ad_map_path is not None:
        require_file(cfg.paths.lead_to_ad_map_path, "lead_to_ad_map")

    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings, details=details)

    ads_lr = meta_csv.load_ads(cfg.paths.ads_path)
    leads_lr = meta_csv.load_leads(cfg.paths.leads_path)
    warnings.extend(ads_lr.warnings)
    warnings.extend(leads_lr.warnings)

    ads = ads_lr.df
    leads = leads_lr.df

    ads_missing = _missing_cols(ads, ADS_REQUIRED)
    leads_missing = _missing_cols(leads, LEADS_REQUIRED)
    if ads_missing:
        errors.append(f"ads.csv missing required columns: {ads_missing}")
    if leads_missing:
        errors.append(f"leads.csv missing required columns: {leads_missing}")

    # outcomes.csv optional for scoring; required for supervised train/eval
    outcomes = None
    if cfg.paths.outcomes_path is not None:
        outcomes_lr = meta_csv.load_outcomes(cfg.paths.outcomes_path)
        warnings.extend(outcomes_lr.warnings)
        outcomes = outcomes_lr.df
        outcomes_missing = _missing_cols(outcomes, OUTCOMES_REQUIRED)
        if outcomes_missing:
            errors.append(f"outcomes.csv missing required columns: {outcomes_missing}")

    # Basic type checks
    if "date" in ads.columns:
        bad_date = int(pd.isna(ads["date"]).sum())
        if bad_date:
            errors.append(f"ads.csv has {bad_date} unparseable/null dates")
    if "created_time" in leads.columns:
        bad_ct = int(pd.isna(leads["created_time"]).sum())
        if bad_ct:
            errors.append(f"leads.csv has {bad_ct} unparseable/null created_time values")
    if outcomes is not None and "qualified_time" in outcomes.columns:
        # qualified_time may be null; but invalid strings become NaT too, so we only warn if raw was non-empty
        # (hard to distinguish post-normalization). We'll just report null pct in profile.
        pass

    # Numeric columns
    for col in ["impressions", "clicks", "spend"]:
        if col in ads.columns:
            bad_num = int(pd.isna(ads[col]).sum())
            if bad_num:
                warnings.append(f"ads.csv column {col} has {bad_num} null/unparseable values")

    # Uniqueness checks
    ok_leads, dup_leads = _check_unique(leads, "lead_id")
    if not ok_leads:
        errors.append(f"leads.csv has {dup_leads} duplicate lead_id values")
    if outcomes is not None:
        ok_out, dup_out = _check_unique(outcomes, "lead_id")
        if not ok_out:
            errors.append(f"outcomes.csv has {dup_out} duplicate lead_id values")

    # Summaries for human-readable report
    details["files"]["ads"] = {
        "path": str(cfg.paths.ads_path),
        "rows": int(len(ads)),
        "missing_required_columns": ads_missing,
        "null_pct": _null_pct(ads, ADS_REQUIRED),
    }
    details["files"]["leads"] = {
        "path": str(cfg.paths.leads_path),
        "rows": int(len(leads)),
        "missing_required_columns": leads_missing,
        "null_pct": _null_pct(leads, LEADS_REQUIRED + ["ad_id", "adset_id", "campaign_id"]),
    }
    if outcomes is not None:
        details["files"]["outcomes"] = {
            "path": str(cfg.paths.outcomes_path),
            "rows": int(len(outcomes)),
            "missing_required_columns": _missing_cols(outcomes, OUTCOMES_REQUIRED),
            "null_pct": _null_pct(outcomes, OUTCOMES_REQUIRED),
        }

    ok = not errors
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, details=details)


def render_validation_summary(result: ValidationResult) -> str:
    lines: list[str] = []
    if result.ok:
        lines.append("Validation: OK")
    else:
        lines.append("Validation: FAILED")

    if result.errors:
        lines.append("")
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"- {e}")

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"- {w}")

    files = result.details.get("files", {})
    if files:
        lines.append("")
        lines.append("File summary:")
        for name, info in files.items():
            lines.append(f"- {name}: {info.get('rows')} rows ({info.get('path')})")
            missing = info.get("missing_required_columns") or []
            if missing:
                lines.append(f"  missing: {missing}")
            nulls = info.get("null_pct") or {}
            if nulls:
                # show worst offenders
                worst = sorted(nulls.items(), key=lambda kv: kv[1], reverse=True)[:5]
                worst_s = ", ".join([f"{k}={v:.1%}" for k, v in worst])
                lines.append(f"  top null%: {worst_s}")

    return "\n".join(lines)
