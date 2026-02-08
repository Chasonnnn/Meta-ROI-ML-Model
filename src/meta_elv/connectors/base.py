from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadResult:
    df: pd.DataFrame
    warnings: list[str]


# Canonical required columns (v1)
ADS_REQUIRED = [
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
]

LEADS_REQUIRED = ["lead_id", "created_time"]
OUTCOMES_REQUIRED = ["lead_id", "qualified_time"]


def load_ads(path: Path) -> LoadResult:  # pragma: no cover (implemented by specific connector)
    raise NotImplementedError


def load_leads(path: Path) -> LoadResult:  # pragma: no cover (implemented by specific connector)
    raise NotImplementedError


def load_outcomes(path: Path) -> LoadResult:  # pragma: no cover (implemented by specific connector)
    raise NotImplementedError


def load_lead_to_ad_map(path: Path) -> LoadResult:  # pragma: no cover (implemented by specific connector)
    raise NotImplementedError

