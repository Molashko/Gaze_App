from collections import deque
from typing import Deque, Tuple

import numpy as np


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        # Параметры OneEuro:
        # min_cutoff — базовая плавность,
        # beta — адаптация к скорости,
        # d_cutoff — фильтрация производной.
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # Предыдущее состояние фильтра.
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, te: float, cutoff: float) -> float:
        # Преобразование частоты среза в коэффициент экспоненциального сглаживания.
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)

    def _exp_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def __call__(self, x: float, t: float) -> float:
        # Первый сэмпл инициализирует состояние без фильтрации.
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        # te = dt между измерениями.
        te = t - self.t_prev
        if te <= 0:
            # Защита от нулевого/отрицательного шага времени.
            te = 1e-6

        # Фильтруем оценку скорости сигнала.
        dx = (x - self.x_prev) / te
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx_hat = self._exp_smoothing(a_d, dx, self.dx_prev)

        # Чем выше скорость, тем выше cutoff и быстрее реакция фильтра.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = self._exp_smoothing(a, x, self.x_prev)

        # Обновляем внутреннее состояние.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self) -> None:
        # Сброс для новой сессии/калибровки.
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class MedianFilter:
    def __init__(self, window: int = 15):
        # Независимые окна по X и Y.
        self.history_x: Deque[float] = deque(maxlen=window)
        self.history_y: Deque[float] = deque(maxlen=window)

    def __call__(self, x: float, y: float) -> Tuple[float, float]:
        # Добавляем новую точку в окна.
        self.history_x.append(x)
        self.history_y.append(y)
        if len(self.history_x) < 3:
            # Пока данных мало — возвращаем "как есть".
            return x, y
        # Медиана хорошо отсекает одиночные выбросы.
        return float(np.median(list(self.history_x))), float(np.median(list(self.history_y)))

    def reset(self) -> None:
        # Полная очистка окон.
        self.history_x.clear()
        self.history_y.clear()
  