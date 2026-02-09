from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pathlib import Path

from meta_elv.config import (
    BusinessConfig,
    FeaturesConfig,
    LabelConfig,
    ModelConfig,
    PathsConfig,
    ReportingConfig,
    RunConfig,
    SplitsConfig,
)
from meta_elv.modeling.train import train_and_evaluate


def test_train_metrics_include_drift_psi() -> None:
    cfg = RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=Path("ads.csv"),  # unused by train_and_evaluate
            leads_path=Path("leads.csv"),  # unused by train_and_evaluate
            outcomes_path=None,
            lead_to_ad_map_path=None,
            ads_placement_path=None,
            ads_geo_path=None,
            adset_targeting_path=None,
            ad_creatives_path=None,
        ),
        label=LabelConfig(label_window_days=14, as_of_date=None, require_label_maturity=True),
        features=FeaturesConfig(ads_granularity="daily", feature_window_days=7, feature_lag_days=1),
        business=BusinessConfig(value_per_qualified=1.0),
        splits=SplitsConfig(train_frac=0.6, calib_frac=0.2, test_frac=0.2),
        model=ModelConfig(
            model_type="logreg",
            calibration_method="sigmoid",
            random_seed=7,
            lgbm_params={},
            logreg_params={"C": 1.0, "max_iter": 2000},
        ),
        reporting=ReportingConfig(topk_frac=0.10, ece_bins=10, min_segment_leads=30),
    )

    rng = np.random.default_rng(7)
    n = 150
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    created = [start + timedelta(hours=i) for i in range(n)]
    y = rng.integers(0, 2, size=n)

    table = pd.DataFrame(
        {
            "lead_id": [f"l{i:04d}" for i in range(n)],
            "created_time": [c.isoformat() for c in created],
            "label": y.astype(int),
            "label_status": np.where(y == 1, "positive", "negative"),
            # representative rolling features
            "impressions_sum_7d": rng.normal(1000, 200, size=n).clip(min=0),
            "clicks_sum_7d": rng.normal(25, 6, size=n).clip(min=0),
            "spend_sum_7d": rng.normal(80, 18, size=n).clip(min=0),
            "ctr_7d": rng.uniform(0.005, 0.03, size=n),
            "cpc_7d": rng.uniform(0.5, 4.0, size=n),
            "lead_hour": [c.hour for c in created],
            "lead_dow": [c.weekday() for c in created],
            "campaign_id": rng.choice(["c1", "c2", "c3"], size=n),
        }
    )

    artifacts = train_and_evaluate(cfg, table)
    assert "drift" in artifacts.metrics
    drift = artifacts.metrics["drift"]
    assert "psi_train_vs_test" in drift
    cols = (drift["psi_train_vs_test"] or {}).get("columns") or {}
    assert len(cols) > 0
