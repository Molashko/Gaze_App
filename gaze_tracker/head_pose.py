from typing import Tuple

import cv2
import numpy as np

from .config import CAMERA_FOCAL_LENGTH, CAMERA_CENTER, CAMERA_DIST_COEFFS
from .landmarks import FACE_3D_MODEL, HEAD_POSE_LANDMARKS


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    # Из матрицы вращения извлекаем Эйлеровы углы (pitch/yaw/roll).
    # Отдельно обрабатываем сингулярный случай, когда sy -> 0.
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)


def _normalize_angle_deg(angle: float) -> float:
    """Нормализовать угол к стабильному диапазону вокруг 0 для позы головы."""
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def estimate_head_pose(landmarks, img_w: int, img_h: int) -> Tuple[float, float, float]:
    try:
        # 2D-координаты опорных точек лица в пикселях кадра.
        image_points = np.array(
            [
                (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
                for idx in HEAD_POSE_LANDMARKS
            ],
            dtype=np.float64,
        )

        # Внутренние параметры камеры: либо из конфига, либо приближенные.
        focal_length = (
            0.5 * (img_w + img_h)
            if CAMERA_FOCAL_LENGTH is None
            else float(CAMERA_FOCAL_LENGTH)
        )
        center = (
            (img_w / 2, img_h / 2)
            if CAMERA_CENTER is None
            else (float(CAMERA_CENTER[0]), float(CAMERA_CENTER[1]))
        )
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        # Коэффициенты дисторсии: по умолчанию считаем идеальную камеру.
        dist_coeffs = (
            np.zeros((4, 1))
            if CAMERA_DIST_COEFFS is None
            else np.asarray(CAMERA_DIST_COEFFS, dtype=np.float64).reshape(-1, 1)
        )

        # Решаем задачу PnP: оцениваем поворот/сдвиг головы относительно камеры.
        success, rvec, _ = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return (0.0, 0.0, 0.0)

        # Вектор вращения -> матрица -> Эйлеровы углы.
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _rotation_matrix_to_euler_angles(rmat)
        yaw, pitch, roll = (
            _normalize_angle_deg(float(angle)) for angle in (yaw, pitch, roll)
        )

        # На плохих кадрах возможны NaN/inf — отбрасываем такие оценки.
        if not (np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(roll)):
            return (0.0, 0.0, 0.0)

        # Возвращаем в порядке (yaw, pitch, roll), как ожидает остальной пайплайн.
        return float(yaw), float(pitch), float(roll)
    except Exception:
        # Любая ошибка геометрии не должна ломать основной цикл.
        return (0.0, 0.0, 0.0)
