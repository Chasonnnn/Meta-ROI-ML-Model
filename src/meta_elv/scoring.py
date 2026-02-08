from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from .config import RunConfig


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_columns: dict[str, list[str]]


def load_model_bundle(model_path: Path) -> ModelBundle:
    obj = load(model_path)
    if isinstance(obj, dict) and "model" in obj and "feature_columns" in obj:
        return ModelBundle(model=obj["model"], feature_columns=obj["feature_columns"])
    # Back-compat: raw sklearn model saved directly
    return ModelBundle(model=obj, feature_columns={"numeric": [], "categorical": []})


def score_table(cfg: RunConfig, table: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    df = table.copy()
    num_cols = bundle.feature_columns.get("numeric", [])
    cat_cols = bundle.feature_columns.get("categorical", [])
    cols = [c for c in (num_cols + cat_cols) if c in df.columns]
    if not cols:
        raise RuntimeError("Model bundle does not specify feature columns compatible with this table.")

    prob = bundle.model.predict_proba(df[cols])[:, 1]
    df["p_qualified_14d"] = prob.astype(float)
    df["value_per_qualified"] = float(cfg.business.value_per_qualified)
    df["elv"] = df["p_qualified_14d"] * df["value_per_qualified"]
    df["score_rank"] = (-df["p_qualified_14d"]).rank(method="first").astype(int)
    return df


def _ads_spend_by_level(
    ads: pd.DataFrame, level: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    a = ads.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a[(a["date"] >= start_date) & (a["date"] <= end_date)]
    if level == "campaign":
        keys = ["campaign_id", "campaign_name"]
    elif level == "adset":
        keys = ["campaign_id", "campaign_name", "adset_id", "adset_name"]
    else:
        raise ValueError("level must be campaign or adset")
    out = a.groupby(keys, as_index=False)["spend"].sum().rename(columns={"spend": "spend"})
    return out


def compute_leaderboards(
    scored: pd.DataFrame,
    ads: pd.DataFrame,
    min_lead_date: pd.Timestamp | None = None,
    max_lead_date: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    df = scored.copy()
    df["lead_date"] = pd.to_datetime(df["lead_date"], errors="coerce")
    if min_lead_date is None:
        min_lead_date = df["lead_date"].min()
    if max_lead_date is None:
        max_lead_date = df["lead_date"].max()
    if pd.isna(min_lead_date) or pd.isna(max_lead_date):
        # Fallback: use full ads date range
        min_lead_date = pd.to_datetime(ads["date"], errors="coerce").min()
        max_lead_date = pd.to_datetime(ads["date"], errors="coerce").max()

    # Predicted ELV by segment
    campaign_elv = (
        df.dropna(subset=["campaign_id"]).groupby(["campaign_id", "campaign_name"], as_index=False)["elv"].sum()
    ).rename(columns={"elv": "predicted_elv"})
    adset_elv = (
        df.dropna(subset=["campaign_id", "adset_id"]).groupby(
            ["campaign_id", "campaign_name", "adset_id", "adset_name"], as_index=False
        )["elv"].sum()
    ).rename(columns={"elv": "predicted_elv"})

    spend_campaign = _ads_spend_by_level(ads, "campaign", min_lead_date, max_lead_date)
    spend_adset = _ads_spend_by_level(ads, "adset", min_lead_date, max_lead_date)

    lb_campaign = spend_campaign.merge(campaign_elv, on=["campaign_id", "campaign_name"], how="left")
    lb_campaign["predicted_elv"] = lb_campaign["predicted_elv"].fillna(0.0)
    lb_campaign["elv_per_spend"] = lb_campaign["predicted_elv"] / lb_campaign["spend"].replace(0, np.nan)
    lb_campaign = lb_campaign.sort_values("elv_per_spend", ascending=False)

    lb_adset = spend_adset.merge(
        adset_elv, on=["campaign_id", "campaign_name", "adset_id", "adset_name"], how="left"
    )
    lb_adset["predicted_elv"] = lb_adset["predicted_elv"].fillna(0.0)
    lb_adset["elv_per_spend"] = lb_adset["predicted_elv"] / lb_adset["spend"].replace(0, np.nan)
    lb_adset = lb_adset.sort_values("elv_per_spend", ascending=False)

    return {"campaign": lb_campaign, "adset": lb_adset}

