from __future__ import annotations

from pathlib import Path

from meta_elv.config import load_config
from meta_elv.validate import validate_from_config


def _write(p: Path, content: str) -> None:
    p.write_text(content.strip() + "\n")


def test_validate_missing_required_columns(tmp_path: Path) -> None:
    ads = tmp_path / "ads.csv"
    leads = tmp_path / "leads.csv"
    outcomes = tmp_path / "outcomes.csv"

    # Missing spend column
    _write(
        ads,
        """
date,campaign_id,campaign_name,adset_id,adset_name,ad_id,ad_name,impressions,clicks
2026-01-01,cmp_001,C1,as_001,A1,ad_001,AD1,100,2
""",
    )
    _write(
        leads,
        """
lead_id,created_time,ad_id
lead_1,2026-01-02T12:00:00Z,ad_001
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
  as_of_date: 2026-01-20
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
    res = validate_from_config(cfg)
    assert not res.ok
    assert "spend" in "".join(res.errors)
