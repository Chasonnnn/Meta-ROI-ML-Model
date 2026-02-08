from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    ads_path: Path
    leads_path: Path
    outcomes_path: Path | None
    lead_to_ad_map_path: Path | None


@dataclass(frozen=True)
class LabelConfig:
    label_window_days: int
    as_of_date: str | None  # YYYY-MM-DD
    require_label_maturity: bool


@dataclass(frozen=True)
class FeaturesConfig:
    ads_granularity: str  # daily
    feature_window_days: int
    feature_lag_days: int


@dataclass(frozen=True)
class BusinessConfig:
    value_per_qualified: float


@dataclass(frozen=True)
class SplitsConfig:
    train_frac: float
    calib_frac: float
    test_frac: float


@dataclass(frozen=True)
class ModelConfig:
    model_type: str  # lgbm | logreg
    calibration_method: str  # sigmoid | isotonic
    random_seed: int
    lgbm_params: dict[str, Any]
    logreg_params: dict[str, Any]


@dataclass(frozen=True)
class ReportingConfig:
    topk_frac: float
    ece_bins: int


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    paths: PathsConfig
    label: LabelConfig
    features: FeaturesConfig
    business: BusinessConfig
    splits: SplitsConfig
    model: ModelConfig
    reporting: ReportingConfig


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got: {type(raw)}")

    schema_version = int(raw.get("schema_version", 1))

    paths_raw = raw.get("paths", {}) or {}
    label_raw = raw.get("label", {}) or {}
    features_raw = raw.get("features", {}) or {}
    business_raw = raw.get("business", {}) or {}
    splits_raw = raw.get("splits", {}) or {}
    model_raw = raw.get("model", {}) or {}
    reporting_raw = raw.get("reporting", {}) or {}

    cfg = RunConfig(
        schema_version=schema_version,
        paths=PathsConfig(
            ads_path=Path(paths_raw["ads_path"]),
            leads_path=Path(paths_raw["leads_path"]),
            outcomes_path=_as_path(paths_raw.get("outcomes_path")),
            lead_to_ad_map_path=_as_path(paths_raw.get("lead_to_ad_map_path")),
        ),
        label=LabelConfig(
            label_window_days=int(label_raw.get("label_window_days", 14)),
            as_of_date=label_raw.get("as_of_date"),
            require_label_maturity=bool(label_raw.get("require_label_maturity", True)),
        ),
        features=FeaturesConfig(
            ads_granularity=str(features_raw.get("ads_granularity", "daily")),
            feature_window_days=int(features_raw.get("feature_window_days", 7)),
            feature_lag_days=int(features_raw.get("feature_lag_days", 1)),
        ),
        business=BusinessConfig(
            value_per_qualified=float(business_raw.get("value_per_qualified", 1.0)),
        ),
        splits=SplitsConfig(
            train_frac=float(splits_raw.get("train_frac", 0.6)),
            calib_frac=float(splits_raw.get("calib_frac", 0.2)),
            test_frac=float(splits_raw.get("test_frac", 0.2)),
        ),
        model=ModelConfig(
            model_type=str(model_raw.get("model_type", "lgbm")),
            calibration_method=str(model_raw.get("calibration_method", "sigmoid")),
            random_seed=int(model_raw.get("random_seed", 7)),
            lgbm_params=dict(model_raw.get("lgbm_params", {}) or {}),
            logreg_params=dict(model_raw.get("logreg_params", {}) or {}),
        ),
        reporting=ReportingConfig(
            topk_frac=float(reporting_raw.get("topk_frac", 0.10)),
            ece_bins=int(reporting_raw.get("ece_bins", 10)),
        ),
    )

    # Basic sanity checks
    total = cfg.splits.train_frac + cfg.splits.calib_frac + cfg.splits.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"splits fractions must sum to 1.0, got {total}")
    if cfg.features.feature_lag_days < 0:
        raise ValueError("feature_lag_days must be >= 0")
    if cfg.features.feature_window_days <= 0:
        raise ValueError("feature_window_days must be > 0")
    if cfg.label.label_window_days <= 0:
        raise ValueError("label_window_days must be > 0")
    return cfg

