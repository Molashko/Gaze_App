from typing import Optional, Tuple, Dict
import time

import numpy as np

from .calibration import RegressionCalibrator
from .filters import OneEuroFilter, MedianFilter
from .config import ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF, MEDIAN_WINDOW


class GazeTracker:
    def __init__(self, screen_w: int, screen_h: int):
        # Размеры экрана нужны для перевода нормализованных предсказаний в пиксели.
        self.screen_w = screen_w
        self.screen_h = screen_h
        # Калибратор связывает признаки взгляда с координатами экрана.
        self.calibrator = RegressionCalibrator()
        self.calibrated = False

        # Каскад сглаживания: медиана по окну + OneEuro по времени.
        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.median_filter = MedianFilter(MEDIAN_WINDOW)

    def reset(self) -> None:
        # Полный сброс состояния под новую калибровку/сеанс.
        self.calibrator = RegressionCalibrator()
        self.calibrated = False
        self.filter_x.reset()
        self.filter_y.reset()
        self.median_filter.reset()

    def build_feature_vector(self, gaze_features: Dict[str, float], head_pose: Tuple[float, float, float]) -> np.ndarray:
        # Собираем единый вектор признаков для модели калибровки/предсказания.
        yaw, pitch, roll = head_pose
        return np.array(
            [
                gaze_features["gaze_x_2d"],
                gaze_features["gaze_y_2d"],
                gaze_features["gaze_x_3d"],
                gaze_features["gaze_y_3d"],
                gaze_features["eye_aspect"],
                yaw,
                pitch,
                roll,
            ],
            dtype=np.float64,
        )

    def add_calibration_sample(self, feature_vec: np.ndarray, screen_xy: Tuple[float, float]) -> None:
        # screen_xy ожидается в нормализованном диапазоне [0..1].
        self.calibrator.add_sample(feature_vec, screen_xy)

    def finalize_calibration(self) -> bool:
        # Обучаем модель по накопленным сэмплам и фиксируем флаг готовности.
        self.calibrated = self.calibrator.fit()
        return self.calibrated

    def predict_screen(self, feature_vec: np.ndarray) -> Optional[Tuple[float, float]]:
        # Сначала получаем нормализованные координаты от модели.
        pred = self.calibrator.predict(feature_vec)
        if pred is None:
            return None
        x, y = pred
        # Жестко ограничиваем диапазон, чтобы исключить выбросы за экран.
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        # Перевод в пиксели экрана.
        return x * self.screen_w, y * self.screen_h

    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        # Сначала подавляем одиночные выбросы, затем применяем временную фильтрацию.
        x, y = self.median_filter(x, y)
        t = time.time()
        return self.filter_x(x, t), self.filter_y(y, t)
