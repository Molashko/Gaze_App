from gaze_tracker.ui.app import main


if __name__ == "__main__":
    main()
import sys
import time
import threading
import subprocess
from typing import Optional, Tuple, List
from datetime import datetime
from collections import deque

import ctypes

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

import tkinter as tk
from tkinter import messagebox


# === LOGGING ===
LOG_FILE = "gaze_debug.log"

def log(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {message}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


# MediaPipe Face Mesh landmarks
# С refine_landmarks=True получаем 478 landmarks (468 face + 10 iris)
# Iris: 468-472 (правый глаз), 473-477 (левый глаз)
# Центр радужки: 468 (правый), 473 (левый)

# Правый глаз - ключевые landmarks
RIGHT_EYE = {
    "inner": 133,       # Внутренний угол (у носа)
    "outer": 33,        # Внешний угол
    "top_lid": 159,     # Верхнее веко (центр)
    "bottom_lid": 145,  # Нижнее веко (центр)
    "top_lid_inner": 160,
    "top_lid_outer": 158,
    "bottom_lid_inner": 144,
    "bottom_lid_outer": 153,
    "iris_center": 468,
    "iris_points": [468, 469, 470, 471, 472]
}

LEFT_EYE = {
    "inner": 362,       # Внутренний угол (у носа)
    "outer": 263,       # Внешний угол
    "top_lid": 386,     # Верхнее веко (центр)
    "bottom_lid": 374,  # Нижнее веко (центр)
    "top_lid_inner": 387,
    "top_lid_outer": 385,
    "bottom_lid_inner": 381,
    "bottom_lid_outer": 380,
    "iris_center": 473,
    "iris_points": [473, 474, 475, 476, 477]
}

# Landmarks для Head Pose Estimation
# 6 точек: нос, подбородок, левый/правый углы глаз, левый/правый углы рта
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),        # Chin (landmark 199)
    (-225.0, 170.0, -135.0),     # Left eye left corner (landmark 33)
    (225.0, 170.0, -135.0),      # Right eye right corner (landmark 263)
    (-150.0, -150.0, -125.0),    # Left Mouth corner (landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
], dtype=np.float64)

HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]


def get_point_2d(index: int, landmarks, w: int, h: int) -> np.ndarray:
    """Получает 2D координаты точки landmark."""
    return np.array([landmarks[index].x * w, landmarks[index].y * h], dtype=np.float64)


def get_point_3d(index: int, landmarks) -> np.ndarray:
    """Получает 3D координаты точки landmark (нормализованные)."""
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def get_iris_center_2d(eye_dict: dict, landmarks, w: int, h: int) -> Optional[np.ndarray]:
    """Вычисляет 2D центр радужки через минимальную окружность всех iris points."""
    try:
        iris_pts = np.array([get_point_2d(idx, landmarks, w, h) for idx in eye_dict["iris_points"]], dtype=np.float32)
        (cx, cy), radius = cv2.minEnclosingCircle(iris_pts)
        return np.array([cx, cy], dtype=np.float64)
    except:
        return None


def get_eye_bbox_2d(eye_dict: dict, landmarks, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Получает bounding box глаза: (left_x, right_x, top_y, bottom_y).
    Использует несколько landmarks для более точного определения границ.
    """
    try:
        inner = get_point_2d(eye_dict["inner"], landmarks, w, h)
        outer = get_point_2d(eye_dict["outer"], landmarks, w, h)
        top_lid = get_point_2d(eye_dict["top_lid"], landmarks, w, h)
        bottom_lid = get_point_2d(eye_dict["bottom_lid"], landmarks, w, h)
        top_inner = get_point_2d(eye_dict["top_lid_inner"], landmarks, w, h)
        top_outer = get_point_2d(eye_dict["top_lid_outer"], landmarks, w, h)
        bottom_inner = get_point_2d(eye_dict["bottom_lid_inner"], landmarks, w, h)
        bottom_outer = get_point_2d(eye_dict["bottom_lid_outer"], landmarks, w, h)
        
        # Горизонтальные границы
        left_x = min(inner[0], outer[0])
        right_x = max(inner[0], outer[0])
        
        # Вертикальные границы - берём min/max от всех верхних/нижних точек
        top_y = min(top_lid[1], top_inner[1], top_outer[1])
        bottom_y = max(bottom_lid[1], bottom_inner[1], bottom_outer[1])
        
        return (left_x, right_x, top_y, bottom_y)
    except Exception as e:
        log(f"Error in get_eye_bbox_2d: {e}")
        return None


def compute_gaze_vector_3d(eye_dict: dict, landmarks, w: int, h: int) -> Optional[Tuple[float, float]]:
    """
    Вычисляет нормализованную позицию взгляда используя 3D координаты.
    Без клипа - возвращает raw значения для калибровки.
    
    Returns: (gaze_x, gaze_y) - raw нормализованные значения
    """
    try:
        # Получаем 3D координаты ключевых точек глаза
        inner_3d = get_point_3d(eye_dict["inner"], landmarks)
        outer_3d = get_point_3d(eye_dict["outer"], landmarks)
        top_3d = get_point_3d(eye_dict["top_lid"], landmarks)
        bottom_3d = get_point_3d(eye_dict["bottom_lid"], landmarks)
        iris_3d = get_point_3d(eye_dict["iris_center"], landmarks)
        
        # Центр глаза в 3D
        eye_center_3d = (inner_3d + outer_3d + top_3d + bottom_3d) / 4
        
        # Вычисляем направление взгляда
        gaze_dir = iris_3d - eye_center_3d
        
        # Создаём локальную систему координат глаза
        eye_horizontal = outer_3d - inner_3d
        eye_h_norm = np.linalg.norm(eye_horizontal)
        if eye_h_norm < 0.001:
            return None
        eye_horizontal = eye_horizontal / eye_h_norm
        
        eye_vertical = bottom_3d - top_3d
        eye_v_norm = np.linalg.norm(eye_vertical)
        if eye_v_norm < 0.001:
            return None
        eye_vertical = eye_vertical / eye_v_norm
        
        # Размеры глаза
        eye_width = eye_h_norm
        eye_height = eye_v_norm
        
        # Проецируем gaze_dir на оси глаза
        gaze_x = np.dot(gaze_dir, eye_horizontal) / eye_width
        gaze_y = np.dot(gaze_dir, eye_vertical) / eye_height
        
        # Нормализуем с усилением, но БЕЗ клипа
        gaze_x = 0.5 + gaze_x * 5.0
        gaze_y = 0.5 + gaze_y * 5.0
        
        return (float(gaze_x), float(gaze_y))
    except Exception as e:
        log(f"Error in compute_gaze_vector_3d: {e}")
        return None


def get_iris_position_2d(eye_dict: dict, landmarks, w: int, h: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Вычисляет нормализованную позицию радужки в 2D пространстве.
    Простой и надёжный метод.
    
    Returns: (x_norm, y_norm) где 0..1 - это диапазон от левого/верхнего до правого/нижнего края глаза
    """
    try:
        iris_center = get_iris_center_2d(eye_dict, landmarks, w, h)
        if iris_center is None:
            return None, None
        
        bbox = get_eye_bbox_2d(eye_dict, landmarks, w, h)
        if bbox is None:
            return None, None
        
        left_x, right_x, top_y, bottom_y = bbox
        
        eye_width = right_x - left_x
        eye_height = bottom_y - top_y
        
        if eye_width < 3 or eye_height < 2:
            return None, None
        
        # Нормализация позиции радужки относительно bbox глаза
        x_norm = (iris_center[0] - left_x) / eye_width
        y_norm = (iris_center[1] - top_y) / eye_height
        
        return (float(np.clip(x_norm, 0, 1)), float(np.clip(y_norm, 0, 1)))
    except Exception as e:
        log(f"Error in get_iris_position_2d: {e}")
        return None, None


def estimate_head_pose(landmarks, w: int, h: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Оценивает позу головы (yaw, pitch, roll) используя solvePnP.
    Returns: (yaw, pitch, roll) в градусах или (None, None, None)
    """
    try:
        # 2D точки из landmarks
        image_points = np.array([
            get_point_2d(idx, landmarks, w, h) for idx in HEAD_POSE_LANDMARKS
        ], dtype=np.float64)
        
        # Параметры камеры
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
        
        # Rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            roll = 0
        
        return (float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll)))
    except Exception as e:
        return None, None, None


class OneEuroFilter:
    """One Euro Filter для плавного сглаживания с минимальной задержкой."""
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None
    
    def _smoothing_factor(self, te: float, cutoff: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def _exp_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev
    
    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        te = t - self.t_prev
        if te <= 0:
            te = 1e-6
        
        # Derivative
        dx = (x - self.x_prev) / te
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx_hat = self._exp_smoothing(a_d, dx, self.dx_prev)
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = self._exp_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class GazeTracker:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        self.calibrated = False
        
        # Калибровочные данные:
        # [(gaze_x_2d, gaze_y_2d, gaze_y_3d, yaw, pitch), (screen_x_norm, screen_y_norm)]
        self.calib_data: List[Tuple[Tuple[float, float, float, float, float], Tuple[float, float]]] = []
        
        # Калибровочные матрицы
        self.homography: Optional[np.ndarray] = None
        
        # Fallback линейные коэффициенты
        self.slope_x = 1.0
        self.intercept_x = 0.0
        self.slope_y = 1.0
        self.intercept_y = 0.0
        self.invert_x = False
        self.invert_y = False
        
        # Центры для компенсации
        self.center_gaze_x = 0.5
        self.center_gaze_y = 0.5
        self.center_head_yaw = 0.0
        self.center_head_pitch = 0.0
        
        # Коэффициенты компенсации головы (будут вычислены при калибровке)
        self.head_comp_x = 0.0  # Компенсация X от yaw
        self.head_comp_y = 0.0  # Компенсация Y от pitch
        
        # Диапазоны для маппинга
        self.gaze_x_min = 0.0
        self.gaze_x_max = 1.0
        self.gaze_y_min = -0.5
        self.gaze_y_max = 0.5
        self.pitch_min = -10.0
        self.pitch_max = 10.0
        self.y_source = "gaze_3d"
        
        # One Euro Filters для X и Y (сильное сглаживание)
        self.filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.1, d_cutoff=1.0)
        self.filter_y = OneEuroFilter(min_cutoff=0.5, beta=0.1, d_cutoff=1.0)
        
        # История для медианного фильтра (увеличена для стабильности)
        self.history_x = deque(maxlen=15)
        self.history_y = deque(maxlen=15)
        
        # История raw gaze и head для фильтрации выбросов
        self.raw_gaze_history = deque(maxlen=10)
        self.raw_y_history = deque(maxlen=10)
        self.raw_pitch_history = deque(maxlen=10)
    
    def get_gaze(self, landmarks, w: int, h: int, use_3d: bool = False) -> Optional[Tuple[float, float]]:
        """
        Получает позицию взгляда от обоих глаз.
        Возвращает gaze_x (из iris position) и gaze_y (из iris position).
        Head pitch будет использоваться дополнительно при преобразовании.
        """
        if use_3d:
            gaze_r = compute_gaze_vector_3d(RIGHT_EYE, landmarks, w, h)
            gaze_l = compute_gaze_vector_3d(LEFT_EYE, landmarks, w, h)
        else:
            gx_r, gy_r = get_iris_position_2d(RIGHT_EYE, landmarks, w, h)
            gx_l, gy_l = get_iris_position_2d(LEFT_EYE, landmarks, w, h)
            gaze_r = (gx_r, gy_r) if gx_r is not None else None
            gaze_l = (gx_l, gy_l) if gx_l is not None else None
        
        if gaze_r is None and gaze_l is None:
            return None
        
        if gaze_r is not None and gaze_l is not None:
            gaze_x = (gaze_r[0] + gaze_l[0]) / 2
            gaze_y = (gaze_r[1] + gaze_l[1]) / 2
        elif gaze_r is not None:
            gaze_x, gaze_y = gaze_r
        else:
            gaze_x, gaze_y = gaze_l
        
        return (gaze_x, gaze_y)
    
    def get_combined_gaze_y(self, gaze_y: float, head_pitch: float) -> float:
        """
        Комбинирует gaze_y и head_pitch для лучшего определения вертикального направления.
        Head pitch: больше = смотрим вверх экрана, меньше = смотрим вниз.
        """
        # Нормализуем head_pitch к диапазону [0, 1]
        # Типичный диапазон pitch при смотрении на экран: -10...+20 градусов
        # Инвертируем: больший pitch -> верх экрана -> меньший screen_y
        pitch_norm = (head_pitch - (-10)) / 30.0  # -10..+20 -> 0..1
        pitch_norm = 1.0 - pitch_norm  # Инвертируем: высокий pitch -> низкий Y
        pitch_norm = np.clip(pitch_norm, 0.0, 1.0)
        
        # Комбинируем gaze_y и pitch с весами
        # gaze_y имеет малый диапазон, поэтому даём больший вес pitch
        combined = 0.3 * gaze_y + 0.7 * pitch_norm
        
        return float(combined)
    
    def add_calibration_point(
        self,
        gaze_2d: Tuple[float, float],
        gaze_y_3d: Optional[float],
        head_pose: Tuple[float, float],
        screen_pos: Tuple[float, float]
    ):
        """Добавляет калибровочную точку."""
        yaw, pitch = head_pose
        gy3d = float(gaze_y_3d) if gaze_y_3d is not None else float("nan")
        self.calib_data.append(((gaze_2d[0], gaze_2d[1], gy3d, yaw, pitch), screen_pos))
    
    def compute_calibration(self):
        """
        Вычисляет калибровку.
        X: используем gaze_x_2d
        Y: автоматически выбираем лучший сигнал (gaze_y_3d / gaze_y_2d / pitch)
        """
        if len(self.calib_data) < 4:
            log("Not enough calibration points")
            return

        # Извлекаем данные
        gaze_x = np.array([d[0][0] for d in self.calib_data], dtype=np.float64)
        gaze_y_2d = np.array([d[0][1] for d in self.calib_data], dtype=np.float64)
        gaze_y_3d = np.array([d[0][2] for d in self.calib_data], dtype=np.float64)
        head_yaw = np.array([d[0][3] for d in self.calib_data], dtype=np.float64)
        head_pitch = np.array([d[0][4] for d in self.calib_data], dtype=np.float64)
        screen_x = np.array([d[1][0] for d in self.calib_data], dtype=np.float64)
        screen_y = np.array([d[1][1] for d in self.calib_data], dtype=np.float64)

        log(f"Raw gaze_x range: {gaze_x.min():.4f} - {gaze_x.max():.4f}")
        if np.any(np.isfinite(gaze_y_2d)):
            log(f"Raw gaze_y_2d range: {np.nanmin(gaze_y_2d):.4f} - {np.nanmax(gaze_y_2d):.4f}")
        if np.any(np.isfinite(gaze_y_3d)):
            log(f"Raw gaze_y_3d range: {np.nanmin(gaze_y_3d):.4f} - {np.nanmax(gaze_y_3d):.4f}")
        log(f"Head yaw range: {head_yaw.min():.1f} - {head_yaw.max():.1f}")
        log(f"Head pitch range: {head_pitch.min():.1f} - {head_pitch.max():.1f}")

        self.center_head_pitch = float(np.median(head_pitch))
        self.center_head_yaw = float(np.median(head_yaw))

        # === X: определяем по gaze_x ===
        left_mask = screen_x <= 0.25
        right_mask = screen_x >= 0.75
        if np.any(left_mask) and np.any(right_mask):
            left_gaze = np.median(gaze_x[left_mask])
            right_gaze = np.median(gaze_x[right_mask])
            self.invert_x = right_gaze < left_gaze
            log(f"X: left_gaze={left_gaze:.4f}, right_gaze={right_gaze:.4f}, invert={self.invert_x}")

        # === Y: выбираем лучший сигнал по корреляции со screen_y ===
        candidates = []

        def add_candidate(name: str, signal: np.ndarray):
            mask = np.isfinite(signal)
            if np.sum(mask) < 4:
                return
            corr = np.corrcoef(signal[mask], screen_y[mask])[0, 1]
            if np.isnan(corr):
                return
            candidates.append((name, corr, signal))

        add_candidate("gaze_3d", gaze_y_3d)
        add_candidate("gaze_2d", gaze_y_2d)
        add_candidate("pitch", head_pitch)

        if candidates:
            name, corr, y_signal = max(candidates, key=lambda x: abs(x[1]))
            self.y_source = name
            self.invert_y = corr < 0
            log(f"Y source: {name}, corr={corr:.3f}, invert={self.invert_y}")
        else:
            self.y_source = "gaze_2d"
            self.invert_y = False
            y_signal = gaze_y_2d
            log("Y source fallback: gaze_2d")

        # Применяем инверсию к X/Y
        gx = 1.0 - gaze_x if self.invert_x else gaze_x.copy()
        gy = -y_signal if self.invert_y else y_signal

        # Сохраняем диапазоны для gaze_x
        self.gaze_x_min = float(np.percentile(gx, 10))
        self.gaze_x_max = float(np.percentile(gx, 90))
        x_margin = (self.gaze_x_max - self.gaze_x_min) * 0.30
        self.gaze_x_min -= x_margin
        self.gaze_x_max += x_margin

        # Сохраняем диапазоны для выбранного Y сигнала
        self.gaze_y_min = float(np.nanpercentile(gy, 10))
        self.gaze_y_max = float(np.nanpercentile(gy, 90))
        y_margin = (self.gaze_y_max - self.gaze_y_min) * 0.30
        self.gaze_y_min -= y_margin
        self.gaze_y_max += y_margin

        log(f"Gaze X range for mapping: {self.gaze_x_min:.4f} - {self.gaze_x_max:.4f}")
        log(f"Y range for mapping: {self.gaze_y_min:.4f} - {self.gaze_y_max:.4f}")

        self.center_gaze_x = float(np.mean(gx))
        self.center_gaze_y = float(np.nanmean(gy))
        self.head_comp_x = 0.0
        self.head_comp_y = 0.0

        self.calibrated = True
        log(f"=== Calibration complete: points={len(self.calib_data)} ===")
    
    def gaze_to_screen(self, gaze_x: float, gaze_y_signal: float) -> Tuple[float, float]:
        """
        Преобразует gaze в экранные координаты.
        X: используем gaze_x
        Y: используем выбранный сигнал (gaze_y_3d / gaze_y_2d / pitch)
        """

        gx = 1.0 - gaze_x if self.invert_x else gaze_x
        gy = -gaze_y_signal if self.invert_y else gaze_y_signal

        if self.calibrated:
            gaze_x_range = max(self.gaze_x_max - self.gaze_x_min, 0.01)
            gaze_y_range = max(self.gaze_y_max - self.gaze_y_min, 0.01)

            gx_norm = (gx - self.gaze_x_min) / gaze_x_range
            gy_norm = (gy - self.gaze_y_min) / gaze_y_range

            norm_x = 0.05 + gx_norm * 0.9
            norm_y = 0.05 + gy_norm * 0.9
        else:
            norm_x = 0.5 + (gx - 0.5) * 2.0
            norm_y = 0.5 + gy * 2.0

        norm_x = float(np.clip(norm_x, 0.0, 1.0))
        norm_y = float(np.clip(norm_y, 0.0, 1.0))

        return norm_x * self.screen_w, norm_y * self.screen_h
    
    def filter_raw_gaze(self, gaze_x: float, gaze_y: float) -> Tuple[float, float]:
        """Фильтрует выбросы в raw gaze_x до маппинга на экран."""
        # Валидация: gaze_x должен быть в пределах [0, 1] с небольшим запасом
        if gaze_x < -0.2 or gaze_x > 1.2:
            # Явный выброс, используем последнее хорошее значение
            if len(self.raw_gaze_history) > 0:
                return float(self.raw_gaze_history[-1]), gaze_y
            return 0.5, gaze_y
        
        self.raw_gaze_history.append(gaze_x)
        
        if len(self.raw_gaze_history) < 3:
            return gaze_x, gaze_y
        
        # Вычисляем медиану по истории
        xs = list(self.raw_gaze_history)
        med_x = np.median(xs)
        
        # Вычисляем IQR для определения выбросов
        q1_x, q3_x = np.percentile(xs, [25, 75])
        iqr_x = max(q3_x - q1_x, 0.02)  # Минимум 0.02 IQR
        
        # Если текущее значение - выброс, заменяем на медиану
        if abs(gaze_x - med_x) > 2.5 * iqr_x:
            gaze_x = med_x
        
        return float(gaze_x), gaze_y

    def filter_raw_y_signal(self, y_signal: float) -> float:
        """Фильтрует выбросы в выбранном Y сигнале (gaze_y или pitch)."""
        if not np.isfinite(y_signal):
            if len(self.raw_y_history) > 0:
                return float(self.raw_y_history[-1])
            return 0.0

        self.raw_y_history.append(y_signal)

        if len(self.raw_y_history) < 3:
            return float(y_signal)

        ys = list(self.raw_y_history)
        med_y = np.median(ys)
        q1_y, q3_y = np.percentile(ys, [25, 75])
        iqr_y = max(q3_y - q1_y, 0.02)

        if abs(y_signal - med_y) > 2.5 * iqr_y:
            y_signal = med_y

        return float(y_signal)

    def select_y_signal(self, gaze_y_2d: Optional[float], gaze_y_3d: Optional[float], pitch: float) -> float:
        """Возвращает выбранный источник Y сигнала."""
        if self.y_source == "gaze_3d" and gaze_y_3d is not None and np.isfinite(gaze_y_3d):
            return float(gaze_y_3d)
        if self.y_source == "gaze_2d" and gaze_y_2d is not None and np.isfinite(gaze_y_2d):
            return float(gaze_y_2d)
        if self.y_source == "pitch":
            return float(pitch)

        # Fallback порядок
        if gaze_y_3d is not None and np.isfinite(gaze_y_3d):
            return float(gaze_y_3d)
        if gaze_y_2d is not None and np.isfinite(gaze_y_2d):
            return float(gaze_y_2d)
        return float(pitch)
    
    def filter_raw_pitch(self, pitch: float) -> float:
        """Фильтрует выбросы в head_pitch."""
        # Валидация: pitch должен быть в разумных пределах
        if abs(pitch) > 30.0:
            # Явный выброс от head pose estimation, используем последнее хорошее значение
            if len(self.raw_pitch_history) > 0:
                return float(self.raw_pitch_history[-1])
            return 0.0
        
        self.raw_pitch_history.append(pitch)
        
        if len(self.raw_pitch_history) < 3:
            return pitch
        
        # Вычисляем медиану по истории
        pitches = list(self.raw_pitch_history)
        med_pitch = np.median(pitches)
        
        # Вычисляем IQR для определения выбросов
        q1, q3 = np.percentile(pitches, [25, 75])
        iqr = max(q3 - q1, 1.0)  # Минимум 1 градус IQR
        
        # Если текущее значение - выброс, заменяем на медиану
        if abs(pitch - med_pitch) > 2.0 * iqr:
            pitch = med_pitch
        
        return float(pitch)
    
    def smooth(self, x: float, y: float) -> Tuple[float, float]:
        """Сглаживание с One Euro Filter + медианный фильтр."""
        t = time.time()
        
        # Медианный фильтр для удаления выбросов
        self.history_x.append(x)
        self.history_y.append(y)
        
        if len(self.history_x) >= 5:
            x = float(np.median(list(self.history_x)))
            y = float(np.median(list(self.history_y)))
        
        # One Euro Filter
        smooth_x = self.filter_x(x, t)
        smooth_y = self.filter_y(y, t)
        
        return smooth_x, smooth_y
    
    def reset(self):
        self.calibrated = False
        self.calib_data.clear()
        self.homography = None
        self.invert_x = False
        self.invert_y = False
        self.center_gaze_x = 0.5
        self.center_gaze_y = 0.0
        self.center_head_yaw = 0.0
        self.center_head_pitch = 0.0
        self.head_comp_x = 0.0
        self.head_comp_y = 0.0
        self.gaze_x_min = 0.0
        self.gaze_x_max = 1.0
        self.gaze_y_min = -0.5
        self.gaze_y_max = 0.5
        self.pitch_min = -10.0
        self.pitch_max = 10.0
        self.y_source = "gaze_3d"
        self.filter_x.reset()
        self.filter_y.reset()
        self.history_x.clear()
        self.history_y.clear()
        self.raw_gaze_history.clear()
        self.raw_y_history.clear()
        self.raw_pitch_history.clear()


class GazeApp:
    def __init__(self):
        self.screen_w, self.screen_h = self._get_screen_size()
        self.tracking_active = False
        self._cap = None
        self.calibrating = False
        self.calibration_complete = False
        
        self.tracker = GazeTracker(self.screen_w, self.screen_h)
        
        self.face_mesh = None
        
        # Калибровка - 9 точек (3x3)
        self.calib_points = self._create_calibration_points()
        self.current_calib_idx = 0
        self.calib_hold_time = 2.5  # Время удержания на точке
        self.calib_point_start = None
        self.calib_samples: List[Tuple[float, float, float, float, float]] = []
        self.calib_min_samples = 25
        
        self.current_head_pose = (0.0, 0.0, 0.0)

        self.root = tk.Tk()
        self.root.title("Gaze Tracker v2")
        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        
        log(f"=== App started v2 === Screen: {self.screen_w}x{self.screen_h}")

    def _get_screen_size(self) -> Tuple[int, int]:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            w = ctypes.windll.user32.GetSystemMetrics(0)
            h = ctypes.windll.user32.GetSystemMetrics(1)
            return int(w), int(h)
        except Exception:
            tmp = tk.Tk()
            tmp.withdraw()
            w = tmp.winfo_screenwidth()
            h = tmp.winfo_screenheight()
            tmp.destroy()
            return w, h

    def _create_calibration_points(self) -> List[Tuple[int, int]]:
        """9 калибровочных точек (3x3) с отступами от краёв."""
        points = []
        for y_pct in [0.1, 0.5, 0.9]:
            for x_pct in [0.1, 0.5, 0.9]:
                points.append((int(x_pct * self.screen_w), int(y_pct * self.screen_h)))
        return points

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Gaze Tracker v2", font=("Segoe UI", 14, "bold")).pack(pady=(0, 8))
        
        info_text = "1. Sit comfortably, face the camera\n2. Click 'Calibrate' and follow the dots\n3. Keep your head relatively still"
        tk.Label(frame, text=info_text, justify="left", fg="#555").pack(pady=(0, 10))

        tk.Button(frame, text="Install Dependencies", command=self._install_deps).pack(fill="x", pady=4)
        tk.Button(frame, text="Calibrate", command=self._run_calibration, bg="#4CAF50", fg="white").pack(fill="x", pady=4)
        tk.Button(frame, text="Reset Calibration", command=self._reset_calibration).pack(fill="x", pady=4)
        tk.Button(frame, text="Start Tracking", command=self._start_tracking).pack(fill="x", pady=4)
        tk.Button(frame, text="Exit", command=self._quit).pack(fill="x", pady=(12, 0))

        if not MEDIAPIPE_AVAILABLE:
            tk.Label(frame, text="MediaPipe not installed!", fg="red").pack(anchor="w")

        tk.Label(frame, textvariable=self.status_var, fg="#444").pack(pady=(10, 0))

    def _install_deps(self) -> None:
        self.status_var.set("Installing...")
        def worker():
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                self.status_var.set("Installed!")
                messagebox.showinfo("Done", "Dependencies installed")
            except Exception as exc:
                self.status_var.set("Install error")
                messagebox.showerror("Error", f"Install error:\n{exc}")
        threading.Thread(target=worker, daemon=True).start()

    def _run_calibration(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe not installed")
            return
        if self.tracking_active:
            messagebox.showinfo("Calibration", "Stop tracking first")
            return
        
        log("=== CALIBRATION STARTED v2 ===")
        self.calibrating = True
        self.calibration_complete = False
        self.current_calib_idx = 0
        self.calib_point_start = None
        self.calib_samples.clear()
        self.tracker.reset()
        self.status_var.set("Calibrating... Look at red dots")
        self._start_tracking()

    def _start_tracking(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe not installed")
            return

        if self.tracking_active:
            messagebox.showinfo("Tracking", "Already running")
            return

        self.status_var.set("Tracking...")
        self.tracking_active = True
        log("=== TRACKING STARTED ===")

        # MediaPipe FaceMesh с iris landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Включает iris landmarks (478 точек)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        log("MediaPipe FaceMesh initialized (478 landmarks)")
        
        # Camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер для меньшей задержки
        self._cap = cap
        log("Camera opened")

        display_w, display_h = self.screen_w, self.screen_h
        
        cv2.namedWindow("Gaze Output", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 480, 360)
        cv2.moveWindow("Camera", 10, 10)

        frame_count = 0
        warmup_frames = 30
        log_interval = 30

        while self.tracking_active:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            
            # Mirror
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            
            # Warmup
            if frame_count < warmup_frames:
                canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                cv2.putText(canvas, "Camera warming up...",
                    (display_w // 2 - 150, display_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("Gaze Output", canvas)
                cv2.imshow("Camera", frame)
                cv2.waitKey(1)
                continue

            # MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            cam_display = frame.copy()
            gaze = None
            gaze_3d = None
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Head pose
                yaw, pitch, roll = estimate_head_pose(landmarks, img_w, img_h)
                if yaw is not None:
                    self.current_head_pose = (yaw, pitch, roll)
                
                # Draw eye landmarks on camera view
                for eye_dict, color in [(RIGHT_EYE, (0, 255, 0)), (LEFT_EYE, (0, 255, 0))]:
                    # Eye corners
                    for key in ["inner", "outer", "top_lid", "bottom_lid"]:
                        pt = get_point_2d(eye_dict[key], landmarks, img_w, img_h)
                        cv2.circle(cam_display, (int(pt[0]), int(pt[1])), 2, color, -1)
                    
                    # Iris center
                    iris_center = get_iris_center_2d(eye_dict, landmarks, img_w, img_h)
                    if iris_center is not None:
                        cv2.circle(cam_display, (int(iris_center[0]), int(iris_center[1])), 4, (255, 0, 255), -1)
                
                # Get gaze from both 2D and 3D methods
                gaze = self.tracker.get_gaze(landmarks, img_w, img_h, use_3d=False)
                gaze_3d = self.tracker.get_gaze(landmarks, img_w, img_h, use_3d=True)
                
                if gaze:
                    cv2.putText(cam_display, f"Gaze: ({gaze[0]:.3f}, {gaze[1]:.3f})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    if self.current_head_pose[0] is not None:
                        cv2.putText(cam_display, f"Head: Y={self.current_head_pose[0]:.0f} P={self.current_head_pose[1]:.0f}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            else:
                cv2.putText(cam_display, "Face not detected!",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Camera", cam_display)

            # Main output canvas
            canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)

            # Calibration mode
            if self.calibrating and not self.calibration_complete:
                if self.current_calib_idx >= len(self.calib_points):
                    # All points done
                    self.tracker.compute_calibration()
                    self.calibration_complete = True
                    self.calibrating = False
                    log("=== CALIBRATION COMPLETE ===")
                    self.status_var.set("Calibration complete!")
                else:
                    cx, cy = self.calib_points[self.current_calib_idx]
                    
                    now = time.time()
                    if self.calib_point_start is None:
                        self.calib_point_start = now
                        self.calib_samples.clear()
                        log(f"Calibration point {self.current_calib_idx + 1}/{len(self.calib_points)}: ({cx}, {cy})")
                    
                    elapsed = now - self.calib_point_start
                    progress = min(1.0, elapsed / self.calib_hold_time)
                    
                    # Collect samples (only in second half of hold time for stability)
                    if gaze is not None and elapsed > self.calib_hold_time * 0.3:
                        yaw = self.current_head_pose[0] if self.current_head_pose[0] is not None else 0.0
                        pitch = self.current_head_pose[1] if self.current_head_pose[1] is not None else 0.0
                        gaze_y_3d = gaze_3d[1] if gaze_3d is not None else float("nan")
                        self.calib_samples.append((gaze[0], gaze[1], gaze_y_3d, yaw, pitch))
                    
                    # Draw calibration UI
                    pulse = 1.0 + 0.15 * np.sin(now * 6)
                    outer_r = int(45 * pulse)
                    
                    # Outer ring background
                    cv2.circle(canvas, (cx, cy), outer_r + 5, (30, 30, 30), -1)
                    # Progress arc
                    if progress > 0:
                        angle = int(360 * progress)
                        cv2.ellipse(canvas, (cx, cy), (outer_r, outer_r), -90, 0, angle, (0, 255, 0), 6)
                    # White inner circle
                    cv2.circle(canvas, (cx, cy), 25, (255, 255, 255), -1)
                    # Red center dot
                    cv2.circle(canvas, (cx, cy), 8, (0, 0, 255), -1)
                    
                    # Status text
                    cv2.putText(canvas, f"Calibration: {self.current_calib_idx + 1}/{len(self.calib_points)}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(canvas, "LOOK AT THE RED DOT",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                    cv2.putText(canvas, "Slightly turn your head towards the dot",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                    cv2.putText(canvas, f"Samples: {len(self.calib_samples)}/{self.calib_min_samples}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                    
                    # Transition to next point
                    if elapsed >= self.calib_hold_time:
                        if len(self.calib_samples) < self.calib_min_samples:
                            # Extend time if not enough samples
                            pass
                        else:
                            # Process samples
                            xs = np.array([s[0] for s in self.calib_samples])
                            ys_2d = np.array([s[1] for s in self.calib_samples])
                            ys_3d = np.array([s[2] for s in self.calib_samples])
                            yaws = np.array([s[3] for s in self.calib_samples])
                            pitches = np.array([s[4] for s in self.calib_samples])
                            
                            # IQR outlier removal
                            q1x, q3x = np.percentile(xs, [25, 75])
                            q1y, q3y = np.percentile(ys_2d, [25, 75])
                            iqr_x = max(q3x - q1x, 1e-6)
                            iqr_y = max(q3y - q1y, 1e-6)
                            mask = (
                                (xs >= q1x - 1.5 * iqr_x) &
                                (xs <= q3x + 1.5 * iqr_x) &
                                (ys_2d >= q1y - 1.5 * iqr_y) &
                                (ys_2d <= q3y + 1.5 * iqr_y)
                            )
                            
                            xs_f = xs[mask] if np.sum(mask) >= 5 else xs
                            ys2_f = ys_2d[mask] if np.sum(mask) >= 5 else ys_2d
                            ys3_f = ys_3d[mask] if np.sum(mask) >= 5 else ys_3d
                            yaws_f = yaws[mask] if np.sum(mask) >= 5 else yaws
                            pitches_f = pitches[mask] if np.sum(mask) >= 5 else pitches
                            
                            med_x = float(np.median(xs_f))
                            med_y2 = float(np.median(ys2_f))
                            med_y3 = float(np.nanmedian(ys3_f))
                            med_yaw = float(np.median(yaws_f))
                            med_pitch = float(np.median(pitches_f))
                            
                            screen_x_norm = cx / self.screen_w
                            screen_y_norm = cy / self.screen_h
                            
                            self.tracker.add_calibration_point(
                                (med_x, med_y2),
                                med_y3,
                                (med_yaw, med_pitch),
                                (screen_x_norm, screen_y_norm)
                            )
                            log(f"Point {self.current_calib_idx + 1}: gaze2d=({med_x:.4f}, {med_y2:.4f}) gaze3d_y=({med_y3:.4f}) head=({med_yaw:.1f}, {med_pitch:.1f}) -> screen=({screen_x_norm:.2f}, {screen_y_norm:.2f})")
                            
                            self.current_calib_idx += 1
                            self.calib_point_start = None
                            self.calib_samples.clear()
            
            # Normal tracking mode - draw gaze point
            if gaze is not None and self.tracker.calibrated:
                yaw = self.current_head_pose[0] if self.current_head_pose[0] is not None else 0.0
                pitch = self.current_head_pose[1] if self.current_head_pose[1] is not None else 0.0

                # Фильтруем выбросы по X
                filtered_gaze_x, _ = self.tracker.filter_raw_gaze(gaze[0], gaze[1])

                # Выбираем и фильтруем Y сигнал
                gaze_y_3d = gaze_3d[1] if gaze_3d is not None else None
                y_signal = self.tracker.select_y_signal(gaze[1], gaze_y_3d, pitch)
                if self.tracker.y_source == "pitch":
                    y_signal = self.tracker.filter_raw_pitch(y_signal)
                else:
                    y_signal = self.tracker.filter_raw_y_signal(y_signal)

                screen_x, screen_y = self.tracker.gaze_to_screen(filtered_gaze_x, y_signal)

                if frame_count % log_interval == 0:
                    log(
                        f"X: {gaze[0]:.3f}->{filtered_gaze_x:.3f} "
                        f"Y({self.tracker.y_source}): {y_signal:.3f} -> "
                        f"Screen: ({screen_x:.0f}, {screen_y:.0f})"
                    )
                
                screen_x = max(0, min(self.screen_w - 1, screen_x))
                screen_y = max(0, min(self.screen_h - 1, screen_y))
                
                smooth_x, smooth_y = self.tracker.smooth(screen_x, screen_y)
                
                dot_x = int(smooth_x)
                dot_y = int(smooth_y)
                
                # Draw gaze point
                cv2.circle(canvas, (dot_x, dot_y), 20, (0, 0, 255), -1)
                cv2.circle(canvas, (dot_x, dot_y), 22, (255, 255, 255), 3)
                
                if not self.calibrating:
                    # Draw target circles
                    for pt in self.calib_points:
                        cv2.circle(canvas, pt, 15, (50, 50, 50), 2)
                    
                    cv2.putText(canvas, f"Position: ({dot_x}, {dot_y})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(canvas, "Look at the circles to test | Press Q to exit",
                        (10, display_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            cv2.imshow("Gaze Output", canvas)

            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                break
                
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        self.tracking_active = False
        log("=== TRACKING STOPPED ===")
        
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception as exc:
                log(f"Warning: face_mesh.close failed: {exc}")
            finally:
                self.face_mesh = None
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Tracking stopped")

    def _reset_calibration(self) -> None:
        log("Calibration reset")
        self.calibrating = False
        self.calibration_complete = False
        self.current_calib_idx = 0
        self.tracker.reset()
        self.status_var.set("Calibration reset")
        messagebox.showinfo("Done", "Calibration reset")

    def _quit(self) -> None:
        self.tracking_active = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception as exc:
                log(f"Warning: face_mesh.close failed: {exc}")
            finally:
                self.face_mesh = None
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = GazeApp()
    app.run()


if __name__ == "__main__":
    main()
