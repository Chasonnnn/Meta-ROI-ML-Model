from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig
from .connectors import meta_csv


@dataclass(frozen=True)
class BuildTableResult:
    table: pd.DataFrame
    as_of_date: date
    join_strategy: str
    join_match_rate: float
    label_summary: dict[str, int]
    warnings: list[str]
    details: dict[str, Any]


def _as_date(d: Any) -> date | None:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, str):
        return datetime.fromisoformat(d).date()
    raise TypeError(f"Unsupported date type: {type(d)}")


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    return n / d.replace(0, np.nan)


def _compute_rolling_features(
    ads: pd.DataFrame, key_col: str, window_days: int
) -> pd.DataFrame:
    metrics = ["impressions", "clicks", "spend"]
    g = (
        ads.groupby([key_col, "date"], as_index=False)[metrics]
        .sum()
        .sort_values([key_col, "date"])
        .reset_index(drop=True)
    )
    for m in metrics:
        g[f"{m}_sum_{window_days}d"] = (
            g.groupby(key_col)[m]
            .rolling(window_days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    g["active_day"] = (g["impressions"] > 0).astype(int)
    g[f"active_days_{window_days}d"] = (
        g.groupby(key_col)["active_day"]
        .rolling(window_days, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    g[f"ctr_{window_days}d"] = _safe_div(g[f"clicks_sum_{window_days}d"], g[f"impressions_sum_{window_days}d"])
    g[f"cpc_{window_days}d"] = _safe_div(g[f"spend_sum_{window_days}d"], g[f"clicks_sum_{window_days}d"])
    g[f"cpm_{window_days}d"] = _safe_div(g[f"spend_sum_{window_days}d"] * 1000.0, g[f"impressions_sum_{window_days}d"])

    if window_days >= 7:
        for m in metrics:
            g[f"{m}_sum_2d"] = (
                g.groupby(key_col)[m]
                .rolling(2, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
        # Trend: compare last 2d average spend to prior 5d average spend in the 7d window.
        spend_7 = g["spend_sum_7d"] if window_days == 7 else g[f"spend_sum_{window_days}d"]
        clicks_7 = g["clicks_sum_7d"] if window_days == 7 else g[f"clicks_sum_{window_days}d"]
        spend_2 = g["spend_sum_2d"]
        clicks_2 = g["clicks_sum_2d"]
        prev_spend = (spend_7 - spend_2).clip(lower=0.0)
        prev_clicks = (clicks_7 - clicks_2).clip(lower=0.0)
        g["spend_trend_ratio_2vprev"] = _safe_div(spend_2 / 2.0, (prev_spend / 5.0) + 1e-6)
        g["clicks_trend_ratio_2vprev"] = _safe_div(clicks_2 / 2.0, (prev_clicks / 5.0) + 1e-6)

    # Keep only feature columns + keys
    keep = [key_col, "date"] + [
        c for c in g.columns if c not in (metrics + ["active_day", key_col, "date"])
    ]
    return g[keep]


def _build_ads_entity_maps(ads: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build stable mapping tables from IDs to names using the most recent date.
    """
    ads2 = ads.copy()
    ads2["date"] = pd.to_datetime(ads2["date"], errors="coerce").dt.date.astype("object")
    ads2 = ads2.sort_values("date")

    ad_map = (
        ads2.dropna(subset=["ad_id"])
        .groupby("ad_id", as_index=False)
        .tail(1)[["ad_id", "ad_name", "adset_id", "adset_name", "campaign_id", "campaign_name"]]
        .drop_duplicates(subset=["ad_id"])
    )
    adset_map = (
        ads2.dropna(subset=["adset_id"])
        .groupby("adset_id", as_index=False)
        .tail(1)[["adset_id", "adset_name", "campaign_id", "campaign_name"]]
        .drop_duplicates(subset=["adset_id"])
    )
    campaign_map = (
        ads2.dropna(subset=["campaign_id"])
        .groupby("campaign_id", as_index=False)
        .tail(1)[["campaign_id", "campaign_name"]]
        .drop_duplicates(subset=["campaign_id"])
    )
    return {"ad": ad_map, "adset": adset_map, "campaign": campaign_map}


def _has_any_non_null(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def build_table(cfg: RunConfig) -> BuildTableResult:
    warnings: list[str] = []
    details: dict[str, Any] = {}

    ads_lr = meta_csv.load_ads(cfg.paths.ads_path)
    leads_lr = meta_csv.load_leads(cfg.paths.leads_path)
    warnings.extend(ads_lr.warnings)
    warnings.extend(leads_lr.warnings)
    ads = ads_lr.df
    leads = leads_lr.df

    # Parse dates/times (loader attempts this, but enforce)
    ads["date"] = pd.to_datetime(ads["date"], errors="coerce").dt.date.astype("object")
    leads["created_time"] = pd.to_datetime(leads["created_time"], errors="coerce", utc=True)

    if leads["created_time"].isna().any():
        bad = int(leads["created_time"].isna().sum())
        warnings.append(f"{bad} leads have unparseable created_time; they will be dropped")
        leads = leads.dropna(subset=["created_time"]).reset_index(drop=True)

    # Determine join strategy and ensure IDs exist on leads when possible.
    join_strategy = "id"
    if not (_has_any_non_null(leads, "ad_id") or _has_any_non_null(leads, "adset_id") or _has_any_non_null(leads, "campaign_id")):
        if cfg.paths.lead_to_ad_map_path is not None and Path(cfg.paths.lead_to_ad_map_path).exists():
            join_strategy = "mapping"
            map_lr = meta_csv.load_lead_to_ad_map(Path(cfg.paths.lead_to_ad_map_path))
            warnings.extend(map_lr.warnings)
            lead_map = map_lr.df
            if "lead_id" not in lead_map.columns:
                raise ValueError("lead_to_ad_map.csv must include lead_id")
            # Merge adds *_y columns; prefer mapping columns.
            leads = leads.merge(lead_map, on="lead_id", how="left", suffixes=("", "_map"))
        else:
            join_strategy = "name_fallback"
            warnings.append(
                "Leads are missing ad_id/adset_id/campaign_id and no lead_to_ad_map_path was provided. "
                "Falling back to name-based joins; results may be unreliable."
            )

    maps = _build_ads_entity_maps(ads)

    # If IDs exist, attach names and hierarchy fields using ads maps.
    if _has_any_non_null(leads, "ad_id"):
        leads = leads.merge(maps["ad"], on="ad_id", how="left", suffixes=("", "_from_ads"))
    elif _has_any_non_null(leads, "adset_id"):
        leads = leads.merge(maps["adset"], on="adset_id", how="left", suffixes=("", "_from_ads"))
    elif _has_any_non_null(leads, "campaign_id"):
        leads = leads.merge(maps["campaign"], on="campaign_id", how="left", suffixes=("", "_from_ads"))

    # Name-based join to recover IDs if still missing.
    if join_strategy == "name_fallback":
        # campaign
        if not _has_any_non_null(leads, "campaign_id") and _has_any_non_null(leads, "campaign_name"):
            # Map campaign_name -> campaign_id (most frequent id for that name)
            tmp = ads.dropna(subset=["campaign_id", "campaign_name"])
            name_to_id = (
                tmp.groupby(["campaign_name", "campaign_id"]).size().reset_index(name="n")
                .sort_values(["campaign_name", "n", "campaign_id"], ascending=[True, False, True])
                .drop_duplicates(subset=["campaign_name"])[["campaign_name", "campaign_id"]]
            )
            leads = leads.merge(name_to_id, on="campaign_name", how="left", suffixes=("", "_from_name"))
        # adset
        if not _has_any_non_null(leads, "adset_id") and _has_any_non_null(leads, "adset_name"):
            tmp = ads.dropna(subset=["adset_id", "adset_name"])
            name_to_id = (
                tmp.groupby(["adset_name", "adset_id"]).size().reset_index(name="n")
                .sort_values(["adset_name", "n", "adset_id"], ascending=[True, False, True])
                .drop_duplicates(subset=["adset_name"])[["adset_name", "adset_id"]]
            )
            leads = leads.merge(name_to_id, on="adset_name", how="left", suffixes=("", "_from_name"))
        # ad
        if not _has_any_non_null(leads, "ad_id") and _has_any_non_null(leads, "ad_name"):
            tmp = ads.dropna(subset=["ad_id", "ad_name"])
            name_to_id = (
                tmp.groupby(["ad_name", "ad_id"]).size().reset_index(name="n")
                .sort_values(["ad_name", "n", "ad_id"], ascending=[True, False, True])
                .drop_duplicates(subset=["ad_name"])[["ad_name", "ad_id"]]
            )
            leads = leads.merge(name_to_id, on="ad_name", how="left", suffixes=("", "_from_name"))

    # Compute join match rate (how many leads ended up with at least campaign_id)
    match = 0
    if "campaign_id" in leads.columns:
        match = int(leads["campaign_id"].notna().sum())
    join_match_rate = match / max(1, len(leads))
    details["join"] = {
        "strategy": join_strategy,
        "match_rate": join_match_rate,
        "rows": int(len(leads)),
    }

    # Outcomes / labels (optional for scoring; required for supervised training)
    outcomes_available = bool(
        cfg.paths.outcomes_path is not None and Path(cfg.paths.outcomes_path).exists()
    )
    if outcomes_available:
        out_lr = meta_csv.load_outcomes(Path(cfg.paths.outcomes_path))
        warnings.extend(out_lr.warnings)
        outcomes = out_lr.df
        outcomes["qualified_time"] = pd.to_datetime(outcomes["qualified_time"], errors="coerce", utc=True)
        outcomes = outcomes.drop_duplicates(subset=["lead_id"])
        leads = leads.merge(outcomes[["lead_id", "qualified_time"]], on="lead_id", how="left")
    else:
        # Inference-only mode: do not fabricate negative labels.
        leads["qualified_time"] = pd.NaT

    # as_of_date
    if cfg.label.as_of_date:
        as_of = _as_date(cfg.label.as_of_date)
        if as_of is None:
            raise ValueError("as_of_date could not be parsed")
    else:
        as_of = leads["created_time"].max().date()
    assert as_of is not None

    label_window = int(cfg.label.label_window_days)
    maturity_cutoff = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc) - timedelta(
        days=label_window
    )

    if not outcomes_available:
        # Without outcomes we cannot label; keep everything unknown.
        leads["label"] = np.nan
        leads["label_status"] = "unknown"
        label_summary = {
            "positive": 0,
            "negative": 0,
            "unknown": int(len(leads)),
        }
        details["labeling"] = {
            "as_of_date": as_of.isoformat(),
            "label_window_days": label_window,
            "require_label_maturity": bool(cfg.label.require_label_maturity),
            "maturity_cutoff_utc": maturity_cutoff.isoformat(),
            "outcomes_available": False,
            "counts": label_summary,
        }
    else:
        created = leads["created_time"]
        qualified_time = pd.to_datetime(leads["qualified_time"], errors="coerce", utc=True)

        is_positive = qualified_time.notna() & (qualified_time <= created + pd.to_timedelta(label_window, unit="D"))
        if cfg.label.require_label_maturity:
            is_mature = created <= maturity_cutoff
        else:
            is_mature = pd.Series([True] * len(leads), index=leads.index)
        is_negative = qualified_time.isna() & is_mature

        leads["label"] = np.where(is_positive, 1, np.where(is_negative, 0, np.nan))
        leads["label_status"] = np.where(
            is_positive, "positive", np.where(is_negative, "negative", "unknown")
        )

        label_summary = {
            "positive": int((leads["label_status"] == "positive").sum()),
            "negative": int((leads["label_status"] == "negative").sum()),
            "unknown": int((leads["label_status"] == "unknown").sum()),
        }

        details["labeling"] = {
            "as_of_date": as_of.isoformat(),
            "label_window_days": label_window,
            "require_label_maturity": bool(cfg.label.require_label_maturity),
            "maturity_cutoff_utc": maturity_cutoff.isoformat(),
            "outcomes_available": True,
            "counts": label_summary,
        }

    # Feature engineering
    leads["lead_date"] = leads["created_time"].dt.date.astype("object")
    leads["lead_hour"] = leads["created_time"].dt.hour.astype("int16")
    leads["lead_dow"] = leads["created_time"].dt.weekday.astype("int8")

    # Features are computed at the most granular available key per lead.
    # Because ads.csv is daily aggregates, we default to lagging features by 1 day.
    feature_window = int(cfg.features.feature_window_days)
    lag = int(cfg.features.feature_lag_days)
    leads["feature_end_date"] = (
        pd.to_datetime(leads["lead_date"]).dt.date.astype("object")
    )
    if lag:
        leads["feature_end_date"] = (
            pd.to_datetime(leads["feature_end_date"]) - pd.to_timedelta(lag, unit="D")
        ).dt.date.astype("object")

    # Precompute per-level rolling features
    feature_tables: dict[str, pd.DataFrame] = {}
    for level_key in ["ad_id", "adset_id", "campaign_id"]:
        if level_key in ads.columns:
            feature_tables[level_key] = _compute_rolling_features(ads, level_key, feature_window)

    def merge_features(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
        ft = feature_tables.get(key)
        if ft is None or key not in df.columns:
            return df
        ft2 = ft.copy()
        rename = {c: f"{prefix}{c}" for c in ft2.columns if c not in {key, "date"}}
        ft2 = ft2.rename(columns=rename)
        merged = df.merge(
            ft2,
            left_on=[key, "feature_end_date"],
            right_on=[key, "date"],
            how="left",
        )
        return merged.drop(columns=["date"])

    # Merge all levels then coalesce per-row: ad -> adset -> campaign.
    leads = merge_features(leads, "campaign_id", "campaign_")
    leads = merge_features(leads, "adset_id", "adset_")
    leads = merge_features(leads, "ad_id", "ad_")

    # Determine base feature names from any available feature table
    base_feature_names: list[str] = []
    for ft in feature_tables.values():
        base_feature_names = [c for c in ft.columns if c not in {"date", "ad_id", "adset_id", "campaign_id"}]
        if base_feature_names:
            break

    if not base_feature_names:
        warnings.append("No ads-derived feature tables could be computed; ads-derived features will be missing.")
    else:
        for f in base_feature_names:
            ad_c = f"ad_{f}"
            adset_c = f"adset_{f}"
            camp_c = f"campaign_{f}"
            s = None
            if ad_c in leads.columns:
                s = leads[ad_c]
            if s is None:
                s = pd.Series([np.nan] * len(leads), index=leads.index)
            if adset_c in leads.columns:
                s = s.fillna(leads[adset_c])
            if camp_c in leads.columns:
                s = s.fillna(leads[camp_c])
            leads[f] = s

        # Drop intermediate prefixed columns to keep the table compact
        protected = {
            "ad_id",
            "ad_name",
            "adset_id",
            "adset_name",
            "campaign_id",
            "campaign_name",
        }
        drop_cols = [
            c
            for c in leads.columns
            if c.startswith(("ad_", "adset_", "campaign_")) and c not in protected
        ]
        leads = leads.drop(columns=drop_cols)

    # Determine feature columns (exclude ids/timestamps/labels)
    non_feature = {
        "lead_id",
        "created_time",
        "qualified_time",
        "lead_date",
        "feature_end_date",
        "label",
        "label_status",
        "campaign_id",
        "campaign_name",
        "adset_id",
        "adset_name",
        "ad_id",
        "ad_name",
    }
    feat_cols = [c for c in leads.columns if c not in non_feature]

    details["features"] = {
        "feature_window_days": feature_window,
        "feature_lag_days": lag,
        "feature_columns": feat_cols,
    }

    # Final column order (stable-ish)
    base_cols = [
        "lead_id",
        "created_time",
        "lead_date",
        "campaign_id",
        "campaign_name",
        "adset_id",
        "adset_name",
        "ad_id",
        "ad_name",
        "lead_dow",
        "lead_hour",
        "feature_end_date",
        "label_status",
        "label",
    ]
    cols = [c for c in base_cols if c in leads.columns] + [c for c in leads.columns if c not in base_cols]
    leads = leads[cols]

    return BuildTableResult(
        table=leads,
        as_of_date=as_of,
        join_strategy=join_strategy,
        join_match_rate=join_match_rate,
        label_summary=label_summary,
        warnings=warnings,
        details=details,
    )
