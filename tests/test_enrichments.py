from __future__ import annotations

import hashlib
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
from meta_elv.validate import validate_from_config


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _bucket(token: str, buckets: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) % buckets


def _base_cfg(tmp_path: Path) -> RunConfig:
    return RunConfig(
        schema_version=1,
        paths=PathsConfig(
            ads_path=tmp_path / "ads.csv",
            leads_path=tmp_path / "leads.csv",
            outcomes_path=None,
            lead_to_ad_map_path=None,
            ads_placement_path=None,
            ads_geo_path=None,
            adset_targeting_path=None,
            ad_creatives_path=None,
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


def test_placement_breakdown_respects_feature_lag(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg = RunConfig(
        **{
            **cfg.__dict__,
            "paths": PathsConfig(
                **{
                    **cfg.paths.__dict__,
                    "ads_placement_path": tmp_path / "ads_placement.csv",
                }
            ),
        }
    )

    # Minimal ads + lead
    ads_rows = []
    for d in pd.date_range("2026-02-01", "2026-02-08", freq="D"):
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
                "spend": 1.0,
            }
        )
    _write_csv(cfg.paths.ads_path, pd.DataFrame(ads_rows))

    leads = pd.DataFrame(
        [
            {
                "lead_id": "l1",
                "created_time": "2026-02-08T12:00:00Z",
                "ad_id": "ad1",
            }
        ]
    )
    _write_csv(cfg.paths.leads_path, leads)

    # Placement breakdown: small spend before lead date, huge spend on the lead date (should be excluded by lag=1).
    pb_rows = []
    for d in pd.date_range("2026-02-01", "2026-02-08", freq="D"):
        if d.date().isoformat() == "2026-02-08":
            pb_rows.append(
                {"date": d.date().isoformat(), "ad_id": "ad1", "placement": "stories", "impressions": 100, "clicks": 1, "spend": 1000.0}
            )
        else:
            pb_rows.append(
                {"date": d.date().isoformat(), "ad_id": "ad1", "placement": "feed", "impressions": 100, "clicks": 1, "spend": 1.0}
            )
    _write_csv(cfg.paths.ads_placement_path, pd.DataFrame(pb_rows))  # type: ignore[arg-type]

    res = build_table(cfg)
    row = res.table.iloc[0]

    assert str(row["feature_end_date"]) == "2026-02-07"
    # Rolling total spend ending at 2026-02-07 should be 7.0 (1/day x 7), not 1007.0.
    assert float(row["placement_total_spend_7d"]) == 7.0
    assert str(row["placement_top1"]) == "feed"


def test_targeting_keywords_hashed_features_are_stable_and_text_is_dropped(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg = RunConfig(
        **{
            **cfg.__dict__,
            "paths": PathsConfig(
                **{
                    **cfg.paths.__dict__,
                    "adset_targeting_path": tmp_path / "adset_targeting.csv",
                }
            ),
        }
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

    leads = pd.DataFrame([{"lead_id": "l1", "created_time": "2026-02-08T12:00:00Z", "ad_id": "ad1"}])
    _write_csv(cfg.paths.leads_path, leads)

    targeting = pd.DataFrame([{"adset_id": "a1", "audience_keywords": "foo, bar"}])
    _write_csv(cfg.paths.adset_targeting_path, targeting)  # type: ignore[arg-type]

    res = build_table(cfg)
    assert "audience_keywords" not in res.table.columns
    row = res.table.iloc[0]
    assert int(row["aud_kw_count"]) == 2

    b1 = _bucket("foo", 64)
    b2 = _bucket("bar", 64)
    assert float(row[f"aud_kw_hash_{b1:03d}"]) == 1.0
    assert float(row[f"aud_kw_hash_{b2:03d}"]) == 1.0


def test_creative_type_features(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg = RunConfig(
        **{
            **cfg.__dict__,
            "paths": PathsConfig(
                **{
                    **cfg.paths.__dict__,
                    "ad_creatives_path": tmp_path / "ad_creatives.csv",
                }
            ),
        }
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

    leads = pd.DataFrame([{"lead_id": "l1", "created_time": "2026-02-08T12:00:00Z", "ad_id": "ad1"}])
    _write_csv(cfg.paths.leads_path, leads)

    creatives = pd.DataFrame([{"ad_id": "ad1", "creative_type": "Video"}])
    _write_csv(cfg.paths.ad_creatives_path, creatives)  # type: ignore[arg-type]

    res = build_table(cfg)
    assert "creative_type" not in res.table.columns
    row = res.table.iloc[0]
    assert int(row["creative_present"]) == 1
    assert int(row["creative_is_video"]) == 1
    assert int(row["creative_is_image"]) == 0


def test_validate_optional_enrichment_files(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg = RunConfig(
        **{
            **cfg.__dict__,
            "paths": PathsConfig(
                **{
                    **cfg.paths.__dict__,
                    "ads_placement_path": tmp_path / "ads_placement.csv",
                }
            ),
        }
    )

    # Required core files
    _write_csv(
        cfg.paths.ads_path,
        pd.DataFrame(
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
        ),
    )
    _write_csv(cfg.paths.leads_path, pd.DataFrame([{"lead_id": "l1", "created_time": "2026-02-08T12:00:00Z"}]))

    # Missing required 'placement' column should fail validation.
    _write_csv(
        cfg.paths.ads_placement_path,  # type: ignore[arg-type]
        pd.DataFrame([{"date": "2026-02-07", "ad_id": "ad1", "spend": 1.0, "impressions": 10, "clicks": 1}]),
    )
    vr = validate_from_config(cfg)
    assert not vr.ok
    assert any("ads_placement missing required columns" in e for e in vr.errors)

