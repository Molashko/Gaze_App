from collections import deque
from typing import Deque, Tuple

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, te: float, cutoff: float) -> float:
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)

    def _exp_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        te = t - self.t_prev
        if te <= 0:
            te = 1e-6

        dx = (x - self.x_prev) / te
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx_hat = self._exp_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = self._exp_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class MedianFilter:
    def __init__(self, window: int = 15):
        self.window = window
        self.history_x: Deque[float] = deque(maxlen=window)
        self.history_y: Deque[float] = deque(maxlen=window)

    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        self.history_x.append(x)
        self.history_y.append(y)
        if len(self.history_x) < 3:
            return x, y
        return float(np.median(list(self.history_x))), float(np.median(list(self.history_y)))

    def reset(self) -> None:
        self.history_x.clear()
        self.history_y.clear()
  