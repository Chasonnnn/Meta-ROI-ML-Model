from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..base import LoadResult


def _first_present(cols: Iterable[str], candidates: list[str]) -> str | None:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        c = cols_l.get(cand.lower())
        if c is not None:
            return c
    return None


def _normalize_columns(df: pd.DataFrame, mapping: dict[str, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    rename: dict[str, str] = {}
    for canonical, candidates in mapping.items():
        if canonical in df.columns:
            continue
        found = _first_present(df.columns, candidates)
        if found is not None:
            rename[found] = canonical
        else:
            # Leave missing; validation will catch.
            pass
    if rename:
        warnings.append(f"Renamed columns: {rename}")
        df = df.rename(columns=rename)
    return df, warnings


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _id_to_str(x: object) -> str:
    # Avoid '123.0' for integer-like floats that come from CSV parsing.
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        return str(x)
    if isinstance(x, int):
        return str(int(x))
    return str(x)


def _coerce_id_columns_to_str(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    changed: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        s0 = df[c]
        non_null = s0.dropna()
        # Heuristic: if a sample is already strings, skip to avoid noisy warnings.
        sample = non_null.head(50).tolist()
        if sample and all(isinstance(v, str) for v in sample):
            continue
        # Preserve nulls as nulls.
        s = s0.where(s0.notna(), None)
        df[c] = s.map(lambda v: _id_to_str(v) if v is not None else None)
        changed.append(c)
    if changed:
        warnings.append(f"Coerced ID columns to strings: {changed}")
    return df, warnings

def _drop_non_canonical(df: pd.DataFrame, keep: list[str]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    keep_existing = [c for c in keep if c in df.columns]
    keep_set = set(keep_existing)
    extra = [c for c in df.columns if c not in keep_set]
    if extra:
        # Avoid leaking any potential PII fields downstream by default.
        warnings.append(f"Dropped non-canonical columns: {extra}")
        df = df[keep_existing].copy()
    else:
        df = df[keep_existing].copy()
    return df, warnings


def load_ads(path: Path) -> LoadResult:
    df = _read_csv(path)
    mapping = {
        "date": ["date", "date_start", "day"],
        "campaign_id": ["campaign_id", "campaign id", "Campaign ID"],
        "campaign_name": ["campaign_name", "campaign name", "Campaign name", "Campaign"],
        "adset_id": ["adset_id", "ad set id", "Ad Set ID", "adset id"],
        "adset_name": ["adset_name", "ad set name", "Ad Set Name", "adset name", "Ad Set"],
        "ad_id": ["ad_id", "ad id", "Ad ID"],
        "ad_name": ["ad_name", "ad name", "Ad Name", "Ad"],
        "impressions": ["impressions", "Impressions"],
        "clicks": ["clicks", "Clicks", "link_clicks", "Link Clicks"],
        "spend": ["spend", "amount_spent", "Amount Spent"],
    }
    df, warnings = _normalize_columns(df, mapping)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("object")

    for col in ["impressions", "clicks", "spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df, dropped_warnings = _drop_non_canonical(
        df,
        [
            "date",
            "campaign_id",
            "campaign_name",
            "adset_id",
            "adset_name",
            "ad_id",
            "ad_name",
            "impressions",
            "clicks",
            "spend",
        ],
    )
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["campaign_id", "adset_id", "ad_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)


def load_leads(path: Path) -> LoadResult:
    df = _read_csv(path)
    mapping = {
        "lead_id": ["lead_id", "lead id", "Lead ID", "id", "Id"],
        "created_time": ["created_time", "created time", "Created Time", "created_at", "created"],
        "ad_id": ["ad_id", "ad id", "Ad ID"],
        "adset_id": ["adset_id", "ad set id", "Ad Set ID"],
        "campaign_id": ["campaign_id", "campaign id", "Campaign ID"],
        "ad_name": ["ad_name", "ad name", "Ad Name", "Ad"],
        "adset_name": ["adset_name", "ad set name", "Ad Set Name", "Ad Set"],
        "campaign_name": ["campaign_name", "campaign name", "Campaign name", "Campaign"],
    }
    df, warnings = _normalize_columns(df, mapping)
    if "created_time" in df.columns:
        df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce", utc=True)
    # Keep only canonical columns by default (avoid accidental PII propagation).
    keep = [
        c
        for c in [
            "lead_id",
            "created_time",
            "campaign_id",
            "campaign_name",
            "adset_id",
            "adset_name",
            "ad_id",
            "ad_name",
        ]
        if c in df.columns
    ]
    df, dropped_warnings = _drop_non_canonical(df, keep)
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["lead_id", "campaign_id", "adset_id", "ad_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)


def load_outcomes(path: Path) -> LoadResult:
    df = _read_csv(path)
    mapping = {
        "lead_id": ["lead_id", "lead id", "Lead ID", "id", "Id"],
        "qualified_time": [
            "qualified_time",
            "qualified time",
            "Qualified Time",
            "qualified_at",
            "qualified_timestamp",
            "qualified",
        ],
    }
    df, warnings = _normalize_columns(df, mapping)
    if "qualified_time" in df.columns:
        df["qualified_time"] = pd.to_datetime(df["qualified_time"], errors="coerce", utc=True)
    keep = [c for c in ["lead_id", "qualified_time"] if c in df.columns]
    df, dropped_warnings = _drop_non_canonical(df, keep)
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["lead_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)


def load_lead_to_ad_map(path: Path) -> LoadResult:
    df = _read_csv(path)
    mapping = {
        "lead_id": ["lead_id", "lead id", "Lead ID", "id", "Id"],
        "ad_id": ["ad_id", "ad id", "Ad ID"],
        "adset_id": ["adset_id", "ad set id", "Ad Set ID"],
        "campaign_id": ["campaign_id", "campaign id", "Campaign ID"],
    }
    df, warnings = _normalize_columns(df, mapping)
    keep = [c for c in ["lead_id", "ad_id", "adset_id", "campaign_id"] if c in df.columns]
    df, dropped_warnings = _drop_non_canonical(df, keep)
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["lead_id", "campaign_id", "adset_id", "ad_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)


def load_ads_placement(path: Path) -> LoadResult:
    """
    Daily ads breakdown by placement. Canonical columns:
    - date, ad_id, placement, impressions, clicks, spend
    """
    df = _read_csv(path)
    mapping = {
        "date": ["date", "date_start", "day"],
        "ad_id": ["ad_id", "ad id", "Ad ID"],
        "placement": [
            "placement",
            "Placement",
            "platform_position",
            "Platform Position",
            "publisher_platform",
            "Publisher Platform",
        ],
        "impressions": ["impressions", "Impressions"],
        "clicks": ["clicks", "Clicks", "link_clicks", "Link Clicks"],
        "spend": ["spend", "amount_spent", "Amount Spent"],
    }
    df, warnings = _normalize_columns(df, mapping)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("object")
    for col in ["impressions", "clicks", "spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["date", "ad_id", "placement", "impressions", "clicks", "spend"] if c in df.columns]
    df, dropped_warnings = _drop_non_canonical(df, keep)
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["ad_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)


def load_ads_geo(path: Path) -> LoadResult:
    """
    Daily ads breakdown by geo. Canonical columns:
    - date, ad_id, geo, impressions, clicks, spend
    """
    df = _read_csv(path)
    mapping = {
        "date": ["date", "date_start", "day"],
        "ad_id": ["ad_id", "ad id", "Ad ID"],
        "geo": [
            "geo",
            "Geo",
            "country",
            "Country",
            "region",
            "Region",
            "dma",
            "DMA",
            "state",
            "State",
            "location",
            "Location",
        ],
        "impressions": ["impressions", "Impressions"],
        "clicks": ["clicks", "Clicks", "link_clicks", "Link Clicks"],
        "spend": ["spend", "amount_spent", "Amount Spent"],
    }
    df, warnings = _normalize_columns(df, mapping)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("object")
    for col in ["impressions", "clicks", "spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["date", "ad_id", "geo", "impressions", "clicks", "spend"] if c in df.columns]
    df, dropped_warnings = _drop_non_canonical(df, keep)
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["ad_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)


def load_adset_targeting(path: Path) -> LoadResult:
    """
    Static adset targeting table. Canonical columns:
    - adset_id, audience_keywords (free text / delimited list)
    """
    df = _read_csv(path)
    mapping = {
        "adset_id": ["adset_id", "ad set id", "Ad Set ID", "adset id"],
        "audience_keywords": [
            "audience_keywords",
            "Audience Keywords",
            "keywords",
            "Keywords",
            "interests",
            "Interests",
            "targeting",
            "Targeting",
        ],
    }
    df, warnings = _normalize_columns(df, mapping)
    if "audience_keywords" in df.columns:
        df["audience_keywords"] = df["audience_keywords"].astype("object")
    keep = [c for c in ["adset_id", "audience_keywords"] if c in df.columns]
    df, dropped_warnings = _drop_non_canonical(df, keep)
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["adset_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)


def load_ad_creatives(path: Path) -> LoadResult:
    """
    Static ad creatives table. Canonical columns:
    - ad_id, creative_type (e.g. image/video)
    """
    df = _read_csv(path)
    mapping = {
        "ad_id": ["ad_id", "ad id", "Ad ID"],
        "creative_type": [
            "creative_type",
            "Creative Type",
            "media_type",
            "Media Type",
            "asset_type",
            "Asset Type",
            "type",
            "Type",
        ],
    }
    df, warnings = _normalize_columns(df, mapping)
    if "creative_type" in df.columns:
        df["creative_type"] = df["creative_type"].astype("object")
    keep = [c for c in ["ad_id", "creative_type"] if c in df.columns]
    df, dropped_warnings = _drop_non_canonical(df, keep)
    warnings.extend(dropped_warnings)
    df, id_warnings = _coerce_id_columns_to_str(df, ["ad_id"])
    warnings.extend(id_warnings)
    return LoadResult(df=df, warnings=warnings)
