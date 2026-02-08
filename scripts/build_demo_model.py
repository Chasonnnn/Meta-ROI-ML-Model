from __future__ import annotations

import tempfile
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
from meta_elv.modeling.train import save_model_bundle, train_and_evaluate
from meta_elv.table_builder import build_table


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    asset_path = repo_root / "src" / "meta_elv" / "assets" / "demo_model.joblib"

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        paths = generate_demo_data(td_p, spec=DemoDataSpec(seed=7, days=60, leads_per_day=200))

        cfg = RunConfig(
            schema_version=1,
            paths=PathsConfig(
                ads_path=paths["ads"],
                leads_path=paths["leads"],
                outcomes_path=paths["outcomes"],
                lead_to_ad_map_path=None,
            ),
            label=LabelConfig(label_window_days=14, as_of_date=None, require_label_maturity=True),
            features=FeaturesConfig(ads_granularity="daily", feature_window_days=7, feature_lag_days=1),
            business=BusinessConfig(value_per_qualified=1.0),
            splits=SplitsConfig(train_frac=0.6, calib_frac=0.2, test_frac=0.2),
            # Keep the bundled demo model free of platform-specific dependencies (e.g., libomp for LightGBM on macOS).
            model=ModelConfig(
                model_type="logreg",
                calibration_method="sigmoid",
                random_seed=7,
                lgbm_params={},
                logreg_params={"C": 1.0, "max_iter": 2000},
            ),
            reporting=ReportingConfig(topk_frac=0.10, ece_bins=10),
        )

        table = build_table(cfg).table
        artifacts = train_and_evaluate(cfg, table)
        save_model_bundle(asset_path, artifacts.model_bundle)

    print(f"Wrote bundled demo model to: {asset_path}")


if __name__ == "__main__":
    main()
