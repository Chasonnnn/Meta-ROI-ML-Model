from __future__ import annotations

from pathlib import Path

from meta_elv.config import load_config
from meta_elv.table_builder import build_table


def _write(p: Path, content: str) -> None:
    p.write_text(content.strip() + "\n")


def test_label_maturity_unknown_not_negative(tmp_path: Path) -> None:
    ads = tmp_path / "ads.csv"
    leads = tmp_path / "leads.csv"
    outcomes = tmp_path / "outcomes.csv"

    _write(
        ads,
        """
date,campaign_id,campaign_name,adset_id,adset_name,ad_id,ad_name,impressions,clicks,spend
2026-02-01,cmp_001,C1,as_001,A1,ad_001,AD1,100,2,10.0
2026-02-02,cmp_001,C1,as_001,A1,ad_001,AD1,100,2,10.0
""",
    )
    # One recent lead (should be unknown), one mature (should be negative), one positive.
    _write(
        leads,
        """
lead_id,created_time,ad_id
lead_recent,2026-02-10T12:00:00Z,ad_001
lead_mature,2026-01-20T12:00:00Z,ad_001
lead_pos,2026-01-20T13:00:00Z,ad_001
""",
    )
    _write(
        outcomes,
        """
lead_id,qualified_time
lead_pos,2026-01-25T12:00:00Z
""",
    )

    cfg_path = tmp_path / "config.yaml"
    _write(
        cfg_path,
        f"""
schema_version: 1
paths:
  ads_path: {ads}
  leads_path: {leads}
  outcomes_path: {outcomes}
  lead_to_ad_map_path: null
label:
  label_window_days: 14
  as_of_date: 2026-02-15
  require_label_maturity: true
features:
  ads_granularity: daily
  feature_window_days: 7
  feature_lag_days: 1
business:
  value_per_qualified: 1.0
splits:
  train_frac: 0.6
  calib_frac: 0.2
  test_frac: 0.2
model:
  model_type: logreg
  calibration_method: sigmoid
  random_seed: 7
  lgbm_params: {{}}
  logreg_params: {{}}
reporting:
  topk_frac: 0.10
  ece_bins: 10
""",
    )
    cfg = load_config(cfg_path)
    res = build_table(cfg)
    t = res.table.set_index("lead_id")
    assert t.loc["lead_recent", "label_status"] == "unknown"
    assert t.loc["lead_mature", "label_status"] == "negative"
    assert t.loc["lead_pos", "label_status"] == "positive"


def test_feature_lag_excludes_same_day_ads_metrics(tmp_path: Path) -> None:
    ads = tmp_path / "ads.csv"
    leads = tmp_path / "leads.csv"
    outcomes = tmp_path / "outcomes.csv"

    _write(
        ads,
        """
date,campaign_id,campaign_name,adset_id,adset_name,ad_id,ad_name,impressions,clicks,spend
2026-02-01,cmp_001,C1,as_001,A1,ad_001,AD1,100,1,0.0
2026-02-02,cmp_001,C1,as_001,A1,ad_001,AD1,100,1,1000.0
""",
    )
    _write(
        leads,
        """
lead_id,created_time,ad_id
lead_1,2026-02-02T12:00:00Z,ad_001
""",
    )
    _write(outcomes, "lead_id,qualified_time\nlead_1,\n")

    cfg_path = tmp_path / "config.yaml"
    _write(
        cfg_path,
        f"""
schema_version: 1
paths:
  ads_path: {ads}
  leads_path: {leads}
  outcomes_path: {outcomes}
  lead_to_ad_map_path: null
label:
  label_window_days: 14
  as_of_date: 2026-03-01
  require_label_maturity: true
features:
  ads_granularity: daily
  feature_window_days: 7
  feature_lag_days: 1
business:
  value_per_qualified: 1.0
splits:
  train_frac: 0.6
  calib_frac: 0.2
  test_frac: 0.2
model:
  model_type: logreg
  calibration_method: sigmoid
  random_seed: 7
  lgbm_params: {{}}
  logreg_params: {{}}
reporting:
  topk_frac: 0.10
  ece_bins: 10
""",
    )
    cfg = load_config(cfg_path)
    res = build_table(cfg)
    row = res.table.iloc[0]

    # With lag=1, feature_end_date = 2026-02-01, so spend_sum_7d should reflect spend on 2/1 (0.0), not 2/2 (1000.0)
    spend_col = "spend_sum_7d"
    assert spend_col in res.table.columns
    assert float(row[spend_col]) == 0.0

