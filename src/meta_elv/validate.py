from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .connectors import meta_csv
from .connectors.base import (
    AD_CREATIVES_REQUIRED,
    ADSET_TARGETING_REQUIRED,
    ADS_GEO_REQUIRED,
    ADS_PLACEMENT_REQUIRED,
    ADS_REQUIRED,
    LEADS_REQUIRED,
    OUTCOMES_REQUIRED,
)


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
    if cfg.paths.ads_placement_path is not None:
        require_file(cfg.paths.ads_placement_path, "ads_placement")
    if cfg.paths.ads_geo_path is not None:
        require_file(cfg.paths.ads_geo_path, "ads_geo")
    if cfg.paths.adset_targeting_path is not None:
        require_file(cfg.paths.adset_targeting_path, "adset_targeting")
    if cfg.paths.ad_creatives_path is not None:
        require_file(cfg.paths.ad_creatives_path, "ad_creatives")

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

    # Optional enrichment files (validated only if configured)
    def _validate_optional_csv(
        name: str,
        path: Path | None,
        loader_fn,
        required_cols: list[str],
        extra_null_cols: list[str] | None = None,
    ) -> None:
        nonlocal errors, warnings, details
        if path is None:
            return
        try:
            lr = loader_fn(path)
            warnings.extend(lr.warnings)
            df = lr.df
        except Exception as e:  # pragma: no cover
            errors.append(f"Failed to read {name}: {e}")
            return
        missing = _missing_cols(df, required_cols)
        if missing:
            errors.append(f"{name} missing required columns: {missing}")
        cols_for_nulls = required_cols + (extra_null_cols or [])
        info: dict[str, Any] = {
            "path": str(path),
            "rows": int(len(df)),
            "missing_required_columns": missing,
            "null_pct": _null_pct(df, cols_for_nulls),
        }
        # Join-quality sanity checks against ads.csv IDs
        try:
            if name in {"ads_placement", "ads_geo", "ad_creatives"} and "ad_id" in df.columns and "ad_id" in ads.columns:
                ads_ids = set(ads["ad_id"].dropna().astype("object").astype(str).tolist())
                if ads_ids:
                    cov = float(df["ad_id"].dropna().astype("object").astype(str).isin(ads_ids).mean())
                    info["mapping_coverage_to_ads"] = cov
                    if cov < 0.90:
                        warnings.append(f"{name}: low ad_id coverage vs ads.csv ({cov:.1%}).")
            if name == "adset_targeting" and "adset_id" in df.columns and "adset_id" in ads.columns:
                ads_ids = set(ads["adset_id"].dropna().astype("object").astype(str).tolist())
                if ads_ids:
                    cov = float(df["adset_id"].dropna().astype("object").astype(str).isin(ads_ids).mean())
                    info["mapping_coverage_to_ads"] = cov
                    if cov < 0.90:
                        warnings.append(f"{name}: low adset_id coverage vs ads.csv ({cov:.1%}).")
        except Exception:
            pass

        details["files"][name] = info

    _validate_optional_csv(
        "ads_placement",
        cfg.paths.ads_placement_path,
        meta_csv.load_ads_placement,
        ADS_PLACEMENT_REQUIRED,
    )
    _validate_optional_csv(
        "ads_geo",
        cfg.paths.ads_geo_path,
        meta_csv.load_ads_geo,
        ADS_GEO_REQUIRED,
    )
    _validate_optional_csv(
        "adset_targeting",
        cfg.paths.adset_targeting_path,
        meta_csv.load_adset_targeting,
        ADSET_TARGETING_REQUIRED,
    )
    _validate_optional_csv(
        "ad_creatives",
        cfg.paths.ad_creatives_path,
        meta_csv.load_ad_creatives,
        AD_CREATIVES_REQUIRED,
    )

    # Join-path sanity checks (warn loudly; table builder can still attempt name join fallback)
    has_join_ids = False
    for c in ["ad_id", "adset_id", "campaign_id"]:
        if c in leads.columns and leads[c].notna().any():
            has_join_ids = True
            break
    if not has_join_ids:
        if cfg.paths.lead_to_ad_map_path is None:
            warnings.append(
                "leads.csv has no ad_id/adset_id/campaign_id. "
                "Provide paths.lead_to_ad_map_path for stable joins, or expect name-based fallback warnings."
            )
        else:
            # mapping presence already checked above; validate minimal columns if we can read
            try:
                lead_map = meta_csv.load_lead_to_ad_map(cfg.paths.lead_to_ad_map_path).df
                if "lead_id" not in lead_map.columns:
                    errors.append("lead_to_ad_map.csv missing required column: lead_id")
                if not any(c in lead_map.columns for c in ["ad_id", "adset_id", "campaign_id"]):
                    errors.append(
                        "lead_to_ad_map.csv must include at least one of: ad_id, adset_id, campaign_id"
                    )
            except Exception as e:  # pragma: no cover
                errors.append(f"Failed to read lead_to_ad_map.csv: {e}")

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
            cov = info.get("mapping_coverage_to_ads")
            if isinstance(cov, (int, float)):
                lines.append(f"  mapping coverage vs ads.csv: {float(cov):.1%}")

    return "\n".join(lines)
