from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

from .config import RunConfig
from .drift import compute_psi_for_columns_from_reference


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    feature_columns: dict[str, list[str]]
    drift_reference: dict[str, Any] | None = None
    drift: dict[str, Any] | None = None


def load_model_bundle(model_path: Path) -> ModelBundle:
    obj = load(model_path)
    if isinstance(obj, dict) and "model" in obj and "feature_columns" in obj:
        return ModelBundle(
            model=obj["model"],
            feature_columns=obj["feature_columns"],
            drift_reference=obj.get("drift_reference"),
            drift=obj.get("drift"),
        )
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


def compute_drift_against_reference(
    table: pd.DataFrame, bundle: ModelBundle, *, bins: int | None = None
) -> dict[str, Any] | None:
    """
    Compute PSI drift for a score-only run relative to the training reference stored in the model bundle.
    """
    if not bundle.drift_reference:
        return None
    ref = bundle.drift_reference
    cols = None
    if bundle.drift and isinstance(bundle.drift, dict):
        cols = bundle.drift.get("psi_columns")
        if isinstance(cols, list):
            cols = [str(c) for c in cols]
        else:
            cols = None

    psi = compute_psi_for_columns_from_reference(ref, table, cols=cols)
    threshold = None
    if bundle.drift and isinstance(bundle.drift, dict):
        threshold = bundle.drift.get("psi_threshold")

    flagged = []
    if threshold is not None:
        try:
            thr = float(threshold)
            flagged = [
                c
                for c, r in (psi.get("columns") or {}).items()
                if (r or {}).get("psi") is not None and float(r["psi"]) >= thr
            ]
        except Exception:
            flagged = []

    return {
        "psi_reference_bins": ref.get("bins"),
        "psi_scoring_vs_train": psi,
        "psi_threshold": threshold,
        "flagged_columns": flagged,
        "n_flagged": int(len(flagged)),
    }


def _ads_spend_by_level(
    ads: pd.DataFrame, level: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    a = ads.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a[(a["date"] >= start_date) & (a["date"] <= end_date)]
    if level == "campaign":
        id_keys = ["campaign_id"]
        name_cols = ["campaign_name"]
    elif level == "adset":
        id_keys = ["campaign_id", "adset_id"]
        name_cols = ["campaign_name", "adset_name"]
    else:
        raise ValueError("level must be campaign or adset")

    a = a.dropna(subset=id_keys)
    agg: dict[str, tuple[str, str]] = {"spend": ("spend", "sum")}
    for c in name_cols:
        if c in a.columns:
            agg[c] = (c, "first")
    out = a.groupby(id_keys, as_index=False).agg(**agg)
    # Stable-ish column order: ids, names, spend
    cols = [c for c in id_keys + name_cols + ["spend"] if c in out.columns]
    return out[cols]


def compute_leaderboards(
    scored: pd.DataFrame,
    ads: pd.DataFrame,
    min_lead_date: pd.Timestamp | None = None,
    max_lead_date: pd.Timestamp | None = None,
    *,
    min_segment_leads: int = 30,
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

    if int(min_segment_leads) < 0:
        raise ValueError("min_segment_leads must be >= 0")

    def _segment_stats(level: str) -> pd.DataFrame:
        if level == "campaign":
            id_keys = ["campaign_id"]
            name_cols = ["campaign_name"]
        elif level == "adset":
            id_keys = ["campaign_id", "adset_id"]
            name_cols = ["campaign_name", "adset_name"]
        else:
            raise ValueError("level must be campaign or adset")

        d = df.dropna(subset=id_keys)
        agg: dict[str, tuple[str, str]] = {
            "lead_count": ("lead_id", "size") if "lead_id" in d.columns else ("elv", "size"),
            "predicted_elv": ("elv", "sum"),
            "avg_p_qualified_14d": ("p_qualified_14d", "mean"),
            "avg_elv": ("elv", "mean"),
        }
        for c in name_cols:
            if c in d.columns:
                agg[c] = (c, "first")
        if "label" in d.columns:
            agg["labeled_count"] = ("label", "count")
            agg["positive_rate_labeled"] = ("label", "mean")
        out = d.groupby(id_keys, as_index=False).agg(**agg)
        cols: list[str] = []
        for c in id_keys + name_cols + list(agg.keys()):
            if c in out.columns and c not in cols:
                cols.append(c)
        return out[cols]

    campaign_stats = _segment_stats("campaign")
    adset_stats = _segment_stats("adset")

    spend_campaign = _ads_spend_by_level(ads, "campaign", min_lead_date, max_lead_date)
    spend_adset = _ads_spend_by_level(ads, "adset", min_lead_date, max_lead_date)

    lb_campaign = spend_campaign.merge(campaign_stats, on=["campaign_id"], how="outer", suffixes=("", "_stats"))
    if "campaign_name" in lb_campaign.columns and "campaign_name_stats" in lb_campaign.columns:
        lb_campaign["campaign_name"] = lb_campaign["campaign_name"].fillna(lb_campaign["campaign_name_stats"])
        lb_campaign = lb_campaign.drop(columns=["campaign_name_stats"])
    lb_campaign["spend"] = lb_campaign["spend"].fillna(0.0)
    lb_campaign["predicted_elv"] = lb_campaign["predicted_elv"].fillna(0.0)
    lb_campaign["lead_count"] = lb_campaign["lead_count"].fillna(0).astype(int)
    lb_campaign["low_volume"] = lb_campaign["lead_count"] < int(min_segment_leads)
    lb_campaign["elv_per_spend"] = lb_campaign["predicted_elv"] / lb_campaign["spend"].replace(0, np.nan)
    lb_campaign = lb_campaign.sort_values(
        ["low_volume", "elv_per_spend", "predicted_elv", "spend"],
        ascending=[True, False, False, False],
    )

    lb_adset = spend_adset.merge(adset_stats, on=["campaign_id", "adset_id"], how="outer", suffixes=("", "_stats"))
    for name_col in ["campaign_name", "adset_name"]:
        stats_col = f"{name_col}_stats"
        if name_col in lb_adset.columns and stats_col in lb_adset.columns:
            lb_adset[name_col] = lb_adset[name_col].fillna(lb_adset[stats_col])
            lb_adset = lb_adset.drop(columns=[stats_col])
    lb_adset["spend"] = lb_adset["spend"].fillna(0.0)
    lb_adset["predicted_elv"] = lb_adset["predicted_elv"].fillna(0.0)
    lb_adset["lead_count"] = lb_adset["lead_count"].fillna(0).astype(int)
    lb_adset["low_volume"] = lb_adset["lead_count"] < int(min_segment_leads)
    lb_adset["elv_per_spend"] = lb_adset["predicted_elv"] / lb_adset["spend"].replace(0, np.nan)
    lb_adset = lb_adset.sort_values(
        ["low_volume", "elv_per_spend", "predicted_elv", "spend"],
        ascending=[True, False, False, False],
    )

    return {"campaign": lb_campaign, "adset": lb_adset}
