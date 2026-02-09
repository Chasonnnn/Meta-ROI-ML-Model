from __future__ import annotations

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
from meta_elv.demo.generate import DemoDataSpec, generate_demo_data
from meta_elv.table_builder import build_table
from meta_elv.validate import validate_from_config


def test_demo_generator_emits_enrichment_csvs_and_table_features(tmp_path: Path) -> None:
    paths = generate_demo_data(tmp_path, DemoDataSpec(seed=7, days=14, leads_per_day=10, include_enrichments=True))

    cfg = RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=paths["ads"],
            leads_path=paths["leads"],
            outcomes_path=paths["outcomes"],
            lead_to_ad_map_path=None,
            ads_placement_path=paths.get("ads_placement"),
            ads_geo_path=paths.get("ads_geo"),
            adset_targeting_path=paths.get("adset_targeting"),
            ad_creatives_path=paths.get("ad_creatives"),
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

    vr = validate_from_config(cfg)
    assert vr.ok, f"demo data validation failed: {vr.errors}"

    res = build_table(cfg)
    t = res.table

    # Spot-check that enrichment-derived features exist.
    assert "placement_total_spend_7d" in t.columns
    assert "geo_total_spend_7d" in t.columns
    assert "aud_kw_count" in t.columns
    assert "creative_present" in t.columns

    # Ensure we got at least some non-null coverage.
    assert float(t["placement_total_spend_7d"].notna().mean()) > 0.2
    assert float(t["geo_total_spend_7d"].notna().mean()) > 0.2

