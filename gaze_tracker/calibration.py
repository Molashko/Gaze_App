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
    # Обученные регрессоры для предсказания экранных координат X и Y.
    model_x: object
    model_y: object
    # Преобразователь полиномиальных признаков (высшие степени/комбинации).
    poly: PolynomialFeatures
    # Нормализатор признаков после полиномиального расширения.
    scaler: StandardScaler


class RegressionCalibrator:
    def __init__(self):
        # Храним пары: (вектор признаков взгляда, целевая нормализованная позиция на экране).
        self.samples: List[Tuple[np.ndarray, Tuple[float, float]]] = []
        # После обучения сюда кладется полный пайплайн инференса.
        self.result: Optional[CalibrationResult] = None

    def add_sample(self, features: np.ndarray, screen_xy: Tuple[float, float]) -> None:
        self.samples.append((features, screen_xy))

    def fit(self) -> bool:
        # Слишком мало данных -> модель будет шумной и нестабильной.
        if len(self.samples) < 5:
            return False

        # Формируем матрицу признаков X и вектор целей y.
        X = np.array([s[0] for s in self.samples], dtype=np.float64)
        y = np.array([s[1] for s in self.samples], dtype=np.float64)

        # Расширяем пространство признаков, чтобы модель ловила нелинейности взгляда.
        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
        X_poly = poly.fit_transform(X)
        # Масштабирование важно для устойчивой Ridge-регрессии на полиномиальных признаках.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)

        # Базовый регрессор с L2-регуляризацией.
        base = Ridge(alpha=RIDGE_ALPHA)
        if USE_RANSAC and len(X_scaled) >= 10:
            # Для X и Y строим независимые устойчивые модели;
            # разные random_state уменьшают риск одинаковых "ошибочных" подвыборок.
            model_x = self._build_ransac(base, random_state=42)
            model_y = self._build_ransac(base, random_state=24)
        else:
            # При малом количестве данных используем прямую Ridge без RANSAC.
            model_x = base
            model_y = Ridge(alpha=RIDGE_ALPHA)

        # Обучение по каждой координате отдельно.
        model_x.fit(X_scaled, y[:, 0])
        model_y.fit(X_scaled, y[:, 1])

        # Сохраняем полный набор объектов для последующего predict().
        self.result = CalibrationResult(
            model_x=model_x, model_y=model_y, poly=poly, scaler=scaler
        )
        return True

    def predict(self, features: np.ndarray) -> Optional[Tuple[float, float]]:
        # До калибровки предсказывать нечем.
        if self.result is None:
            return None
        # Важно повторить тот же порядок преобразований, что был при fit().
        X_poly = self.result.poly.transform(features.reshape(1, -1))
        X_scaled = self.result.scaler.transform(X_poly)
        x = float(self.result.model_x.predict(X_scaled)[0])
        y = float(self.result.model_y.predict(X_scaled)[0])
        return x, y

    @staticmethod
    def _build_ransac(base: Ridge, random_state: int) -> RANSACRegressor:
        # Общие параметры устойчивой регрессии к выбросам.
        kwargs = dict(
            min_samples=RANSAC_MIN_SAMPLES,
            residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
            max_trials=RANSAC_MAX_TRIALS,
            random_state=random_state,
        )
        try:
            # Новый API scikit-learn.
            return RANSACRegressor(estimator=base, **kwargs)
        except TypeError:
            # Обратная совместимость со старым API.
            return RANSACRegressor(base_estimator=base, **kwargs)
