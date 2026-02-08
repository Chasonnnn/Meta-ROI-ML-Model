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
    return LoadResult(df=df, warnings=warnings)

