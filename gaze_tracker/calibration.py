from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import (
    POLY_DEGREE,
    RIDGE_ALPHA,
    USE_RANSAC,
    RANSAC_MIN_SAMPLES,
    RANSAC_RESIDUAL_THRESHOLD,
    RANSAC_MAX_TRIALS,
)


@dataclass
class CalibrationResult:
    model_x: object
    model_y: object
    poly: PolynomialFeatures
    scaler: StandardScaler


class RegressionCalibrator:
    def __init__(self):
        self.samples: List[Tuple[np.ndarray, Tuple[float, float]]] = []
        self.result: Optional[CalibrationResult] = None

    def add_sample(self, features: np.ndarray, screen_xy: Tuple[float, float]) -> None:
        self.samples.append((features, screen_xy))

    def fit(self) -> bool:
        if len(self.samples) < 5:
            return False

        X = np.array([s[0] for s in self.samples], dtype=np.float64)
        y = np.array([s[1] for s in self.samples], dtype=np.float64)

        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
        X_poly = poly.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)

        base = Ridge(alpha=RIDGE_ALPHA)
        if USE_RANSAC and len(X_scaled) >= 10:
            ransac_kwargs = dict(
                min_samples=RANSAC_MIN_SAMPLES,
                residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
                max_trials=RANSAC_MAX_TRIALS,
                random_state=42,
            )
            try:
                model_x = RANSACRegressor(estimator=base, **ransac_kwargs)
            except TypeError:
                model_x = RANSACRegressor(base_estimator=base, **ransac_kwargs)

            ransac_kwargs["random_state"] = 24
            try:
                model_y = RANSACRegressor(estimator=base, **ransac_kwargs)
            except TypeError:
                model_y = RANSACRegressor(base_estimator=base, **ransac_kwargs)
        else:
            model_x = base
            model_y = Ridge(alpha=RIDGE_ALPHA)

        model_x.fit(X_scaled, y[:, 0])
        model_y.fit(X_scaled, y[:, 1])

        self.result = CalibrationResult(
            model_x=model_x, model_y=model_y, poly=poly, scaler=scaler
        )
        return True

    def predict(self, features: np.ndarray) -> Optional[Tuple[float, float]]:
        if self.result is None:
            return None
        X_poly = self.result.poly.transform(features.reshape(1, -1))
        X_scaled = self.result.scaler.transform(X_poly)
        x = float(self.result.model_x.predict(X_scaled)[0])
        y = float(self.result.model_y.predict(X_scaled)[0])
        return x, y
