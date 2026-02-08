from __future__ import annotations

from pathlib import Path

import pandas as pd

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
from meta_elv.table_builder import build_table


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_no_outcomes_means_all_labels_unknown(tmp_path: Path) -> None:
    cfg = RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=tmp_path / "ads.csv",
            leads_path=tmp_path / "leads.csv",
            outcomes_path=None,  # inference-only
            lead_to_ad_map_path=None,
        ),
        label=LabelConfig(label_window_days=14, as_of_date="2026-02-08", require_label_maturity=True),
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

    ads = pd.DataFrame(
        [
            {
                "date": "2026-02-07",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "a1",
                "adset_name": "A1",
                "ad_id": "ad1",
                "ad_name": "AD1",
                "impressions": 1000,
                "clicks": 10,
                "spend": 25.0,
            }
        ]
    )
    _write_csv(cfg.paths.ads_path, ads)

    # Include both mature and recent leads; without outcomes, both should stay unknown.
    leads = pd.DataFrame(
        [
            {
                "lead_id": "l_recent",
                "created_time": "2026-02-03T12:00:00Z",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "a1",
                "adset_name": "A1",
                "ad_id": "ad1",
                "ad_name": "AD1",
            },
            {
                "lead_id": "l_mature",
                "created_time": "2026-01-10T12:00:00Z",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "a1",
                "adset_name": "A1",
                "ad_id": "ad1",
                "ad_name": "AD1",
            },
        ]
    )
    _write_csv(cfg.paths.leads_path, leads)

    res = build_table(cfg)
    t = res.table.set_index("lead_id")
    assert t.loc["l_recent", "label_status"] == "unknown"
    assert t.loc["l_mature", "label_status"] == "unknown"
    assert pd.isna(t.loc["l_recent", "label"])
    assert pd.isna(t.loc["l_mature", "label"])


def test_require_label_maturity_false_labels_recent_negatives(tmp_path: Path) -> None:
    cfg = RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=tmp_path / "ads.csv",
            leads_path=tmp_path / "leads.csv",
            outcomes_path=tmp_path / "outcomes.csv",
            lead_to_ad_map_path=None,
        ),
        label=LabelConfig(label_window_days=14, as_of_date="2026-02-08", require_label_maturity=False),
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

    ads = pd.DataFrame(
        [
            {
                "date": "2026-02-07",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "a1",
                "adset_name": "A1",
                "ad_id": "ad1",
                "ad_name": "AD1",
                "impressions": 1000,
                "clicks": 10,
                "spend": 25.0,
            }
        ]
    )
    _write_csv(cfg.paths.ads_path, ads)

    leads = pd.DataFrame(
        [
            {
                "lead_id": "l_recent",
                "created_time": "2026-02-03T12:00:00Z",
                "ad_id": "ad1",
            }
        ]
    )
    _write_csv(cfg.paths.leads_path, leads)

    # Not qualified (empty string => NaT).
    outcomes = pd.DataFrame([{"lead_id": "l_recent", "qualified_time": ""}])
    _write_csv(cfg.paths.outcomes_path, outcomes)

    res = build_table(cfg)
    t = res.table.set_index("lead_id")
    assert t.loc["l_recent", "label_status"] == "negative"
    assert int(t.loc["l_recent", "label"]) == 0
