from typing import Optional, Tuple, Dict

import cv2
import numpy as np

from .landmarks import RIGHT_EYE, LEFT_EYE
from .logging_utils import log


def get_point_2d(index: int, landmarks, w: int, h: int) -> np.ndarray:
    return np.array([landmarks[index].x * w, landmarks[index].y * h], dtype=np.float64)


def get_point_3d(index: int, landmarks) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def get_iris_center_2d(eye_dict: dict, landmarks, w: int, h: int) -> Optional[np.ndarray]:
    try:
        iris_pts = np.array(
            [get_point_2d(idx, landmarks, w, h) for idx in eye_dict["iris_points"]],
            dtype=np.float32,
        )
        (cx, cy), _ = cv2.minEnclosingCircle(iris_pts)
        return np.array([cx, cy], dtype=np.float64)
    except Exception:
        return None


def get_eye_bbox_2d(
    eye_dict: dict, landmarks, w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    try:
        inner = get_point_2d(eye_dict["inner"], landmarks, w, h)
        outer = get_point_2d(eye_dict["outer"], landmarks, w, h)
        top_lid = get_point_2d(eye_dict["top_lid"], landmarks, w, h)
        bottom_lid = get_point_2d(eye_dict["bottom_lid"], landmarks, w, h)
        top_inner = get_point_2d(eye_dict["top_lid_inner"], landmarks, w, h)
        top_outer = get_point_2d(eye_dict["top_lid_outer"], landmarks, w, h)
        bottom_inner = get_point_2d(eye_dict["bottom_lid_inner"], landmarks, w, h)
        bottom_outer = get_point_2d(eye_dict["bottom_lid_outer"], landmarks, w, h)

        left_x = min(inner[0], outer[0])
        right_x = max(inner[0], outer[0])
        top_y = min(top_lid[1], top_inner[1], top_outer[1])
        bottom_y = max(bottom_lid[1], bottom_inner[1], bottom_outer[1])

        return (left_x, right_x, top_y, bottom_y)
    except Exception as exc:
        log(f"Error in get_eye_bbox_2d: {exc}")
        return None


def get_iris_position_2d(
    eye_dict: dict, landmarks, w: int, h: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    iris_center = get_iris_center_2d(eye_dict, landmarks, w, h)
    if iris_center is None:
        return None, None, None

    bbox = get_eye_bbox_2d(eye_dict, landmarks, w, h)
    if bbox is None:
        return None, None, None

    left_x, right_x, top_y, bottom_y = bbox
    eye_width = right_x - left_x
    eye_height = bottom_y - top_y
    if eye_width < 3 or eye_height < 2:
        return None, None, None

    x_norm = (iris_center[0] - left_x) / eye_width
    y_norm = (iris_center[1] - top_y) / eye_height

    return float(x_norm), float(y_norm), float(eye_height / eye_width)


def compute_gaze_vector_3d(
    eye_dict: dict, landmarks
) -> Optional[Tuple[float, float]]:
    try:
        inner_3d = get_point_3d(eye_dict["inner"], landmarks)
        outer_3d = get_point_3d(eye_dict["outer"], landmarks)
        top_3d = get_point_3d(eye_dict["top_lid"], landmarks)
        bottom_3d = get_point_3d(eye_dict["bottom_lid"], landmarks)
        iris_3d = get_point_3d(eye_dict["iris_center"], landmarks)

        eye_center_3d = (inner_3d + outer_3d + top_3d + bottom_3d) / 4.0
        gaze_dir = iris_3d - eye_center_3d

        eye_horizontal = outer_3d - inner_3d
        eye_h_norm = np.linalg.norm(eye_horizontal)
        if eye_h_norm < 1e-4:
            return None
        eye_horizontal = eye_horizontal / eye_h_norm

        eye_vertical = bottom_3d - top_3d
        eye_v_norm = np.linalg.norm(eye_vertical)
        if eye_v_norm < 1e-4:
            return None
        eye_vertical = eye_vertical / eye_v_norm

        gaze_x = np.dot(gaze_dir, eye_horizontal) / eye_h_norm
        gaze_y = np.dot(gaze_dir, eye_vertical) / eye_v_norm

        return float(gaze_x), float(gaze_y)
    except Exception as exc:
        log(f"Error in compute_gaze_vector_3d: {exc}")
        return None


def extract_gaze_features(landmarks, w: int, h: int) -> Optional[Dict[str, float]]:
    gx_r, gy_r, ar_r = get_iris_position_2d(RIGHT_EYE, landmarks, w, h)
    gx_l, gy_l, ar_l = get_iris_position_2d(LEFT_EYE, landmarks, w, h)

    g3_r = compute_gaze_vector_3d(RIGHT_EYE, landmarks)
    g3_l = compute_gaze_vector_3d(LEFT_EYE, landmarks)

    if gx_r is None and gx_l is None:
        return None

    gx_vals = [v for v in [gx_r, gx_l] if v is not None]
    gy_vals = [v for v in [gy_r, gy_l] if v is not None]
    ar_vals = [v for v in [ar_r, ar_l] if v is not None]

    g3x_vals = [v[0] for v in [g3_r, g3_l] if v is not None]
    g3y_vals = [v[1] for v in [g3_r, g3_l] if v is not None]

    return {
        "gaze_x_2d": float(np.mean(gx_vals)) if gx_vals else 0.5,
        "gaze_y_2d": float(np.mean(gy_vals)) if gy_vals else 0.5,
        "gaze_x_3d": float(np.mean(g3x_vals)) if g3x_vals else 0.0,
        "gaze_y_3d": float(np.mean(g3y_vals)) if g3y_vals else 0.0,
        "eye_aspect": float(np.mean(ar_vals)) if ar_vals else 0.3,
    }
