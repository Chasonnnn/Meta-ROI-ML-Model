from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import RunConfig
from .calibration import CalibratedModel, fit_calibrator
from .metrics import calibration_bins, compute_core_metrics, compute_lift_at_k, lift_curve
from .split import TimeSplit, time_split


@dataclass(frozen=True)
class TrainArtifacts:
    model: Any
    model_bundle: dict[str, Any]
    metrics: dict[str, Any]
    split: TimeSplit


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover (older sklearn)
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    v1: numeric rolling + time features, plus low-cardinality IDs as categoricals.
    """
    exclude = {
        "lead_id",
        "created_time",
        "qualified_time",
        "lead_date",
        "feature_end_date",
        "label",
        "label_status",
        "p_qualified_14d",
        "value_per_qualified",
        "elv",
        "score_rank",
    }

    # Categorical: prefer campaign/adset IDs; avoid ad_id by default to limit cardinality.
    categorical = [c for c in ["campaign_id", "adset_id"] if c in df.columns]
    # Numeric: include all numeric dtypes not excluded.
    numeric = []
    for c in df.columns:
        if c in exclude or c in categorical:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)

    # Ensure we include the basic lead time features if present.
    for c in ["lead_dow", "lead_hour"]:
        if c in df.columns and c not in numeric:
            numeric.append(c)

    return numeric, categorical


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", _make_ohe(), categorical_cols))
    if not transformers:
        raise ValueError("No features found to train on.")

    return ColumnTransformer(transformers, remainder="drop")


def _build_estimator(cfg: RunConfig) -> Any:
    mt = cfg.model.model_type.lower().strip()
    if mt == "logreg":
        params = {"C": 1.0, "max_iter": 2000}
        params.update(cfg.model.logreg_params or {})
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 2000)),
            solver="saga",
        )
    if mt == "lgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import LightGBM. Install with: uv sync --extra lgbm. "
                "On macOS you may also need OpenMP (e.g., `brew install libomp`)."
            ) from e
        params = dict(cfg.model.lgbm_params or {})
        params.setdefault("random_state", int(cfg.model.random_seed))
        params.setdefault("n_estimators", 400)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("num_leaves", 31)
        params.setdefault("min_child_samples", 50)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.9)
        params.setdefault("reg_lambda", 1.0)
        return lgb.LGBMClassifier(**params)
    raise ValueError(f"Unknown model_type: {cfg.model.model_type}")


def _campaign_rate_baseline(train_df: pd.DataFrame) -> dict[str, float]:
    if "campaign_id" not in train_df.columns:
        return {}
    d = train_df.dropna(subset=["campaign_id", "label"])
    if not len(d):
        return {}
    rates = d.groupby("campaign_id")["label"].mean().to_dict()
    return {str(k): float(v) for k, v in rates.items()}


def _apply_campaign_rate_baseline(df: pd.DataFrame, rates: dict[str, float], default: float) -> np.ndarray:
    if "campaign_id" not in df.columns or not rates:
        return np.full(len(df), default, dtype=float)
    cid = df["campaign_id"].astype("object").astype(str)
    out = cid.map(rates).fillna(default).to_numpy(dtype=float)
    return out


def train_and_evaluate(cfg: RunConfig, table: pd.DataFrame, *, max_labeled_rows: int | None = None) -> TrainArtifacts:
    if "label" not in table.columns:
        raise ValueError("Table must include 'label' for training.")

    # Train/eval uses only labeled rows (unknowns excluded).
    labeled = table[table["label"].notna()].copy()
    if len(labeled) < 50:
        raise ValueError("Not enough labeled rows to train (need at least 50).")

    labeled["created_time"] = pd.to_datetime(labeled["created_time"], utc=True, errors="coerce")
    labeled = labeled.dropna(subset=["created_time"])

    # Optional cap for interactive environments (e.g., Hugging Face Spaces).
    if isinstance(max_labeled_rows, int) and max_labeled_rows > 0 and len(labeled) > max_labeled_rows:
        labeled = labeled.sort_values("created_time").iloc[-max_labeled_rows:].copy()

    split = time_split(
        labeled,
        time_col="created_time",
        train_frac=cfg.splits.train_frac,
        calib_frac=cfg.splits.calib_frac,
        test_frac=cfg.splits.test_frac,
    )

    train_df = labeled.loc[split.train_idx]
    calib_df = labeled.loc[split.calib_idx]
    test_df = labeled.loc[split.test_idx]

    y_train = train_df["label"].astype(int).to_numpy()
    y_calib = calib_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    numeric_cols, categorical_cols = _select_feature_columns(labeled)

    # Preprocess: scale numeric for logreg, not necessary for lgbm but fine; keep simple.
    scale_numeric = True
    pre = _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=scale_numeric)
    est = _build_estimator(cfg)
    base_model = Pipeline([("pre", pre), ("est", est)])

    X_train = train_df[numeric_cols + categorical_cols]
    X_calib = calib_df[numeric_cols + categorical_cols]
    X_test = test_df[numeric_cols + categorical_cols]

    base_model.fit(X_train, y_train)
    p_calib_raw = base_model.predict_proba(X_calib)[:, 1]
    calibrator = fit_calibrator(cfg.model.calibration_method, p_calib_raw, y_calib)
    model = CalibratedModel(base_model=base_model, calibrator=calibrator)

    p_test = model.predict_proba(X_test)[:, 1]

    # Metrics (model)
    metrics: dict[str, Any] = {}
    metrics["model"] = compute_core_metrics(y_test, p_test)
    lift = compute_lift_at_k(y_test, p_test, cfg.reporting.topk_frac)
    metrics["model"]["lift_at_k"] = lift.__dict__
    metrics["model"]["lift_curve"] = lift_curve(y_test, p_test, points=50)
    metrics["model"]["calibration"] = calibration_bins(y_test, p_test, bins=cfg.reporting.ece_bins)

    # Baseline: campaign historical qualification rate (train window)
    default_rate = float(y_train.mean()) if len(y_train) else 0.0
    rates = _campaign_rate_baseline(train_df)
    p_base = _apply_campaign_rate_baseline(test_df, rates, default=default_rate)
    metrics["baseline_campaign_rate"] = compute_core_metrics(y_test, p_base)
    lift_b = compute_lift_at_k(y_test, p_base, cfg.reporting.topk_frac)
    metrics["baseline_campaign_rate"]["lift_at_k"] = lift_b.__dict__

    # Bundle
    bundle = {
        "model": model,
        "feature_columns": {"numeric": numeric_cols, "categorical": categorical_cols},
        "calibration_method": cfg.model.calibration_method,
        "model_type": cfg.model.model_type,
    }

    metrics["data"] = {
        "n_labeled": int(len(labeled)),
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)),
        "positive_rate_labeled": float(labeled["label"].mean()),
    }
    metrics["split"] = {"train_end_time": split.train_end_time, "calib_end_time": split.calib_end_time}

    return TrainArtifacts(model=model, model_bundle=bundle, metrics=metrics, split=split)


def save_model_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, path)
