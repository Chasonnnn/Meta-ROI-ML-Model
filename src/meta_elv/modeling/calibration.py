from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_proba(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1 - 1e-6)


@dataclass
class SigmoidCalibrator:
    lr: LogisticRegression

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = _clip_proba(p)
        x = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


@dataclass
class IsotonicCalibrator:
    iso: IsotonicRegression

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = _clip_proba(p)
        out = self.iso.predict(p)
        return np.clip(out, 0.0, 1.0)


def fit_calibrator(method: str, p_calib: np.ndarray, y_calib: np.ndarray) -> Any:
    method = method.lower().strip()
    if method == "sigmoid":
        p = _clip_proba(p_calib)
        x = np.log(p / (1.0 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x, y_calib)
        return SigmoidCalibrator(lr=lr)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(_clip_proba(p_calib), y_calib)
        return IsotonicCalibrator(iso=iso)
    raise ValueError(f"Unknown calibration_method: {method}")


@dataclass
class CalibratedModel:
    base_model: Any
    calibrator: Any

    def predict_proba(self, X: Any) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        cal = _clip_proba(cal)
        return np.column_stack([1.0 - cal, cal])

