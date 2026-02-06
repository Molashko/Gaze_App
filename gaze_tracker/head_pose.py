from typing import Tuple

import cv2
import numpy as np

from .config import CAMERA_FOCAL_LENGTH, CAMERA_CENTER, CAMERA_DIST_COEFFS
from .landmarks import FACE_3D_MODEL, HEAD_POSE_LANDMARKS
from .logging_utils import log


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
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
    """Normalize angle to a stable range around 0 for head pose."""
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def estimate_head_pose(landmarks, img_w: int, img_h: int) -> Tuple[float, float, float]:
    try:
        image_points = np.array(
            [
                (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
                for idx in HEAD_POSE_LANDMARKS
            ],
            dtype=np.float64,
        )

        if CAMERA_FOCAL_LENGTH is None:
            focal_length = 0.5 * (img_w + img_h)
        else:
            focal_length = float(CAMERA_FOCAL_LENGTH)

        if CAMERA_CENTER is None:
            center = (img_w / 2, img_h / 2)
        else:
            center = (float(CAMERA_CENTER[0]), float(CAMERA_CENTER[1]))
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        if CAMERA_DIST_COEFFS is None:
            dist_coeffs = np.zeros((4, 1))
        else:
            dist_coeffs = np.asarray(CAMERA_DIST_COEFFS, dtype=np.float64).reshape(-1, 1)

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return (0.0, 0.0, 0.0)

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _rotation_matrix_to_euler_angles(rmat)

        yaw = _normalize_angle_deg(float(yaw))
        pitch = _normalize_angle_deg(float(pitch))
        roll = _normalize_angle_deg(float(roll))

        if not (np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(roll)):
            return (0.0, 0.0, 0.0)

        return float(yaw), float(pitch), float(roll)
    except Exception as exc:
        log(f"Head pose error: {exc}")
        return (0.0, 0.0, 0.0)
