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


def test_daily_ads_feature_lag_excludes_same_day(tmp_path: Path) -> None:
    cfg = RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=tmp_path / "ads.csv",
            leads_path=tmp_path / "leads.csv",
            outcomes_path=None,
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
        reporting=ReportingConfig(topk_frac=0.10, ece_bins=10),
    )

    # Ads spend is 1/day for 7 days, then huge spend on the lead day.
    ads_rows = []
    for d in pd.date_range("2026-02-01", "2026-02-08", freq="D"):
        spend = 1000.0 if d.date().isoformat() == "2026-02-08" else 1.0
        ads_rows.append(
            {
                "date": d.date().isoformat(),
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "a1",
                "adset_name": "A1",
                "ad_id": "ad1",
                "ad_name": "AD1",
                "impressions": 1000,
                "clicks": 10,
                "spend": spend,
            }
        )
    ads = pd.DataFrame(ads_rows)
    _write_csv(cfg.paths.ads_path, ads)

    leads = pd.DataFrame(
        [
            {
                "lead_id": "l1",
                "created_time": "2026-02-08T12:00:00Z",
                "campaign_id": "c1",
                "campaign_name": "C1",
                "adset_id": "a1",
                "adset_name": "A1",
                "ad_id": "ad1",
                "ad_name": "AD1",
            }
        ]
    )
    _write_csv(cfg.paths.leads_path, leads)

    res = build_table(cfg)
    t = res.table
    # Because feature_lag_days=1, feature_end_date should be 2026-02-07.
    assert str(t.loc[0, "feature_end_date"]) == "2026-02-07"

    # Rolling spend sum at 7d window ending 2026-02-07 should be 7.0 (1/day * 7).
    # If same-day leakage were present, this would include the 1000.0 spend from 2026-02-08.
    spend_sum_col = "spend_sum_7d"
    assert spend_sum_col in t.columns
    assert float(t.loc[0, spend_sum_col]) == 7.0

