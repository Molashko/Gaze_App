from typing import Optional, Tuple, Dict
from collections import deque
import time

import numpy as np

from .calibration import RegressionCalibrator
from .filters import OneEuroFilter, MedianFilter
from .config import ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF, MEDIAN_WINDOW
from .logging_utils import log


class GazeTracker:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.calibrator = RegressionCalibrator()
        self.calibrated = False

        self.filter_x = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.filter_y = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.median_filter = MedianFilter(MEDIAN_WINDOW)

        self.feature_history = deque(maxlen=10)

    def reset(self) -> None:
        self.calibrator = RegressionCalibrator()
        self.calibrated = False
        self.filter_x.reset()
        self.filter_y.reset()
        self.median_filter.reset()
        self.feature_history.clear()

    def build_feature_vector(self, gaze_features: Dict[str, float], head_pose: Tuple[float, float, float]) -> np.ndarray:
        yaw, pitch, roll = head_pose
        vec = np.array(
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
        return vec

    def add_calibration_sample(self, feature_vec: np.ndarray, screen_xy: Tuple[float, float]) -> None:
        self.calibrator.add_sample(feature_vec, screen_xy)

    def finalize_calibration(self) -> bool:
        self.calibrated = self.calibrator.fit()
        return self.calibrated

    def predict_screen(self, feature_vec: np.ndarray) -> Optional[Tuple[float, float]]:
        pred = self.calibrator.predict(feature_vec)
        if pred is None:
            return None
        x, y = pred
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        return x * self.screen_w, y * self.screen_h

    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        x, y = self.median_filter(x, y)
        t = time.time()
        return self.filter_x(x, t), self.filter_y(y, t)
