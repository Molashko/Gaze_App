from typing import Optional, Tuple, Dict

import cv2
import numpy as np

from .landmarks import RIGHT_EYE, LEFT_EYE


def get_point_2d(index: int, landmarks, w: int, h: int) -> np.ndarray:
    return np.array([landmarks[index].x * w, landmarks[index].y * h], dtype=np.float64)


def get_point_3d(index: int, landmarks) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def get_iris_center_2d(eye_dict: dict, landmarks, w: int, h: int) -> Optional[np.ndarray]:
    try:
        iris_pts = np.array(
            [get_point_2d(idx, landmarks, w, h) for idx in eye_dict["iris_points"]],
            dtype=np.float32,
        )
        if iris_pts.shape[0] >= 5:
            try:
                (cx, cy), _, _ = cv2.fitEllipse(iris_pts)
                return np.array([cx, cy], dtype=np.float64)
            except Exception:
                pass
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
    except Exception:
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

        mid_horizontal = (inner_3d + outer_3d) / 2.0
        mid_vertical = (top_3d + bottom_3d) / 2.0
        eye_center_3d = (mid_horizontal + mid_vertical) / 2.0
        gaze_dir = iris_3d - eye_center_3d
        gaze_norm = np.linalg.norm(gaze_dir)
        if gaze_norm < 1e-4:
            return None
        gaze_dir = gaze_dir / gaze_norm

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

        gaze_x = np.dot(gaze_dir, eye_horizontal)
        gaze_y = np.dot(gaze_dir, eye_vertical)

        return float(gaze_x), float(gaze_y)
    except Exception:
        return None


def _eye_quality(y_norm: Optional[float], ar: Optional[float]) -> float:
    if y_norm is None or ar is None:
        return 0.0
    openness = _clip01((ar - 0.12) / 0.18)
    edge_dist = min(y_norm, 1.0 - y_norm)
    y_score = _clip01((edge_dist - 0.05) / 0.15)
    return openness * y_score


def _weighted_mean(values, weights, default: float) -> float:
    num = 0.0
    den = 0.0
    for val, weight in zip(values, weights):
        if val is None or weight <= 0.0:
            continue
        num += float(val) * float(weight)
        den += float(weight)
    if den > 0.0:
        return float(num / den)
    valid = [float(v) for v in values if v is not None]
    return float(np.mean(valid)) if valid else float(default)


def extract_gaze_features(landmarks, w: int, h: int) -> Optional[Dict[str, float]]:
    gx_r, gy_r, ar_r = get_iris_position_2d(RIGHT_EYE, landmarks, w, h)
    gx_l, gy_l, ar_l = get_iris_position_2d(LEFT_EYE, landmarks, w, h)

    g3_r = compute_gaze_vector_3d(RIGHT_EYE, landmarks)
    g3_l = compute_gaze_vector_3d(LEFT_EYE, landmarks)

    if gx_r is None and gx_l is None:
        return None
    q_r = _eye_quality(gy_r, ar_r)
    q_l = _eye_quality(gy_l, ar_l)

    gx = _weighted_mean([gx_r, gx_l], [q_r, q_l], 0.5)
    gy = _weighted_mean([gy_r, gy_l], [q_r, q_l], 0.5)
    ar = _weighted_mean([ar_r, ar_l], [q_r, q_l], 0.3)

    g3x_r = g3_r[0] if g3_r is not None else None
    g3y_r = g3_r[1] if g3_r is not None else None
    g3x_l = g3_l[0] if g3_l is not None else None
    g3y_l = g3_l[1] if g3_l is not None else None

    g3x = _weighted_mean([g3x_r, g3x_l], [q_r, q_l], 0.0)
    g3y = _weighted_mean([g3y_r, g3y_l], [q_r, q_l], 0.0)

    return {
        "gaze_x_2d": float(gx),
        "gaze_y_2d": float(gy),
        "gaze_x_3d": float(g3x),
        "gaze_y_3d": float(g3y),
        "eye_aspect": float(ar),
    }
