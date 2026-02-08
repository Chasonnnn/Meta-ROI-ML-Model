from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import RunConfig
from ..drift import compute_psi_for_columns
from .metrics import EvalMetrics, evaluate_binary


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        return LGBMClassifier
    except Exception:  # pragma: no cover
        return None


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict[str, Any]
    feature_columns: dict[str, list[str]]


def _select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    exclude = {
        "lead_id",
        "created_time",
        "qualified_time",
        "lead_date",
        "feature_end_date",
        "label_status",
        "label",
        "campaign_name",
        "adset_name",
        "ad_name",
    }
    cat_cols: list[str] = []
    for c in ["campaign_id", "adset_id"]:
        if c in df.columns and df[c].notna().any():
            cat_cols.append(c)

    num_cols: list[str] = []
    for c in df.columns:
        if c in exclude or c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    # Always include lead time features if present
    for c in ["lead_dow", "lead_hour"]:
        if c in df.columns and c not in num_cols:
            num_cols.append(c)

    return sorted(set(num_cols)), cat_cols


def _make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _make_estimator(cfg: RunConfig, num_cols: list[str], cat_cols: list[str]):
    pre = _make_preprocessor(num_cols, cat_cols)
    model_type = cfg.model.model_type.lower()
    if model_type == "logreg":
        params = {"class_weight": "balanced", "solver": "lbfgs"}
        params.update(cfg.model.logreg_params or {})
        est = LogisticRegression(random_state=cfg.model.random_seed, **params)
        return Pipeline([("pre", pre), ("est", est)])

    if model_type == "lgbm":
        LGBMClassifier = _try_import_lightgbm()
        if LGBMClassifier is None:
            raise RuntimeError(
                "lightgbm is not installed. Install with: uv sync --extra lgbm (or pip install lightgbm)."
            )
        params = dict(cfg.model.lgbm_params or {})
        params.setdefault("random_state", cfg.model.random_seed)
        params.setdefault("n_estimators", 400)
        params.setdefault("learning_rate", 0.05)
        est = LGBMClassifier(**params)
        return Pipeline([("pre", pre), ("est", est)])

    raise ValueError(f"Unknown model_type: {cfg.model.model_type}")


def _time_split(df: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df2 = df.sort_values("created_time").reset_index(drop=True)
    n = len(df2)
    n_train = int(round(n * cfg.splits.train_frac))
    n_calib = int(round(n * cfg.splits.calib_frac))
    n_train = max(1, min(n_train, n - 2))
    n_calib = max(1, min(n_calib, n - n_train - 1))
    n_test = n - n_train - n_calib
    if n_test <= 0:
        n_test = 1
        n_calib = max(1, n - n_train - n_test)
    train = df2.iloc[:n_train].copy()
    calib = df2.iloc[n_train : n_train + n_calib].copy()
    test = df2.iloc[n_train + n_calib :].copy()
    return train, calib, test


def train_model(cfg: RunConfig, table: pd.DataFrame, run_dir: Path) -> TrainResult:
    # Filter to supervised rows (matured labels)
    if cfg.label.require_label_maturity:
        sup = table[table["label_status"].isin(["positive", "negative"])].copy()
    else:
        sup = table[table["label"].notna()].copy()

    if len(sup) < 100:
        raise RuntimeError(f"Not enough labeled rows to train (have {len(sup)}).")

    num_cols, cat_cols = _select_feature_columns(sup)
    X_cols = num_cols + cat_cols
    y = sup["label"].astype(int).to_numpy()

    train_df, calib_df, test_df = _time_split(sup, cfg)

    X_train = train_df[X_cols]
    y_train = train_df["label"].astype(int).to_numpy()
    X_calib = calib_df[X_cols]
    y_calib = calib_df["label"].astype(int).to_numpy()
    X_test = test_df[X_cols]
    y_test = test_df["label"].astype(int).to_numpy()

    method = cfg.model.calibration_method.lower()
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be sigmoid or isotonic")

    # --- Baseline: calibrated logistic regression ---
    logreg_cfg = RunConfig(
        schema_version=cfg.schema_version,
        paths=cfg.paths,
        label=cfg.label,
        features=cfg.features,
        business=cfg.business,
        splits=cfg.splits,
        model=cfg.model.__class__(
            model_type="logreg",
            calibration_method=cfg.model.calibration_method,
            random_seed=cfg.model.random_seed,
            lgbm_params=cfg.model.lgbm_params,
            logreg_params=cfg.model.logreg_params,
        ),
        reporting=cfg.reporting,
    )
    logreg_est = _make_estimator(logreg_cfg, num_cols=num_cols, cat_cols=cat_cols)
    logreg_est.fit(X_train, y_train)
    logreg_cal = CalibratedClassifierCV(logreg_est, method=method, cv="prefit")
    logreg_cal.fit(X_calib, y_calib)
    logreg_prob = logreg_cal.predict_proba(X_test)[:, 1]
    logreg_em: EvalMetrics = evaluate_binary(
        y_true=y_test, y_prob=logreg_prob, topk_frac=cfg.reporting.topk_frac, ece_bins=cfg.reporting.ece_bins
    )

    # --- Baseline: campaign historical qualification rate ---
    campaign_baseline = None
    if "campaign_id" in train_df.columns and train_df["campaign_id"].notna().any():
        rates = train_df.groupby("campaign_id")["label"].mean()
        global_rate = float(train_df["label"].mean())
        pred = test_df["campaign_id"].map(rates).fillna(global_rate).to_numpy(dtype=float)
        campaign_baseline = evaluate_binary(
            y_true=y_test, y_prob=pred, topk_frac=cfg.reporting.topk_frac, ece_bins=cfg.reporting.ece_bins
        )

    # --- Main model (configured) ---
    base = _make_estimator(cfg, num_cols=num_cols, cat_cols=cat_cols)
    base.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(base, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)

    prob_test = calibrated.predict_proba(X_test)[:, 1]
    em: EvalMetrics = evaluate_binary(
        y_true=y_test, y_prob=prob_test, topk_frac=cfg.reporting.topk_frac, ece_bins=cfg.reporting.ece_bins
    )

    metrics: dict[str, Any] = {
        "n_labeled": int(len(sup)),
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_test": int(len(test_df)),
        "positive_rate_train": float(y_train.mean()),
        "eval_test": {
            "pr_auc": em.pr_auc,
            "brier": em.brier,
            "roc_auc": em.roc_auc,
            "lift": em.lift,
            "ece": em.ece,
            "calibration": em.calibration,
        },
        "baselines": {
            "logreg_calibrated": {
                "pr_auc": logreg_em.pr_auc,
                "brier": logreg_em.brier,
                "roc_auc": logreg_em.roc_auc,
                "lift": logreg_em.lift,
                "ece": logreg_em.ece,
                "calibration": logreg_em.calibration,
            },
            "campaign_rate": (
                None
                if campaign_baseline is None
                else {
                    "pr_auc": campaign_baseline.pr_auc,
                    "brier": campaign_baseline.brier,
                    "roc_auc": campaign_baseline.roc_auc,
                    "lift": campaign_baseline.lift,
                    "ece": campaign_baseline.ece,
                    "calibration": campaign_baseline.calibration,
                }
            ),
        },
        "features": {"numeric": num_cols, "categorical": cat_cols},
        "model": {"type": cfg.model.model_type, "calibration": method},
        "splits": {
            "train_frac": cfg.splits.train_frac,
            "calib_frac": cfg.splits.calib_frac,
            "test_frac": cfg.splits.test_frac,
            "train_end_created_time": str(train_df["created_time"].max()),
            "calib_end_created_time": str(calib_df["created_time"].max()),
            "test_end_created_time": str(test_df["created_time"].max()),
        },
    }

    # Minimal drift: PSI for up to 10 numeric features between train and test.
    drift_cols = [c for c in num_cols if c in train_df.columns][:10]
    metrics["drift_psi"] = compute_psi_for_columns(
        expected_df=train_df, actual_df=test_df, cols=drift_cols, bins=10
    )

    model_path = run_dir / "model.joblib"
    dump(
        {"model": calibrated, "feature_columns": {"numeric": num_cols, "categorical": cat_cols}},
        model_path,
    )
    return TrainResult(model_path=model_path, metrics=metrics, feature_columns={"numeric": num_cols, "categorical": cat_cols})
