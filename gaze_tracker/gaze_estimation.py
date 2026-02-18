from typing import Optional, Tuple, Dict

import cv2
import numpy as np

from .landmarks import RIGHT_EYE, LEFT_EYE

# Набор ключей для построения "рамки" глаза.
# Используется для оценки ширины/высоты и нормализации положения радужки.
_EYE_BBOX_KEYS = (
    "inner",
    "outer",
    "top_lid",
    "bottom_lid",
    "top_lid_inner",
    "top_lid_outer",
    "bottom_lid_inner",
    "bottom_lid_outer",
)


def get_point_2d(index: int, landmarks, w: int, h: int) -> np.ndarray:
    # Конвертируем нормализованные координаты MediaPipe [0..1] в пиксели кадра.
    return np.array([landmarks[index].x * w, landmarks[index].y * h], dtype=np.float64)


def get_point_3d(index: int, landmarks) -> np.ndarray:
    # 3D-координаты в системе MediaPipe (относительные, но полезны для направлений).
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _clip01(value: float) -> float:
    # Часто используем мягкое ограничение оценок к диапазону [0..1].
    return float(np.clip(value, 0.0, 1.0))


def _safe_normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    # Безопасная нормализация вектора: защищаемся от деления на ~0.
    norm = np.linalg.norm(vec)
    if norm < 1e-4:
        return None
    return vec / norm


def get_iris_center_2d(eye_dict: dict, landmarks, w: int, h: int) -> Optional[np.ndarray]:
    try:
        # Берем контур радужки (5 точек), который отдаёт FaceMesh.
        iris_pts = np.array(
            [get_point_2d(idx, landmarks, w, h) for idx in eye_dict["iris_points"]],
            dtype=np.float32,
        )
        if iris_pts.shape[0] >= 5:
            try:
                # Предпочтительно аппроксимируем эллипсом — обычно это точнее центра окружности.
                (cx, cy), _, _ = cv2.fitEllipse(iris_pts)
                return np.array([cx, cy], dtype=np.float64)
            except Exception:
                # Если эллипс не построился (геометрия плохая), падаем на окружность.
                pass
        (cx, cy), _ = cv2.minEnclosingCircle(iris_pts)
        return np.array([cx, cy], dtype=np.float64)
    except Exception:
        # На шумных кадрах геометрия может "развалиться" — возвращаем None без исключений.
        return None


def get_eye_bbox_2d(
    eye_dict: dict, landmarks, w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    try:
        # Собираем опорные точки века и уголков глаза.
        (
            inner,
            outer,
            top_lid,
            bottom_lid,
            top_inner,
            top_outer,
            bottom_inner,
            bottom_outer,
        ) = [get_point_2d(eye_dict[key], landmarks, w, h) for key in _EYE_BBOX_KEYS]

        # Получаем оси "рамки" глаза.
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
    # Центр радужки в пикселях.
    iris_center = get_iris_center_2d(eye_dict, landmarks, w, h)
    if iris_center is None:
        return None, None, None

    # Геометрия глаза в пикселях.
    bbox = get_eye_bbox_2d(eye_dict, landmarks, w, h)
    if bbox is None:
        return None, None, None

    left_x, right_x, top_y, bottom_y = bbox
    eye_width = right_x - left_x
    eye_height = bottom_y - top_y
    # Слишком маленькая рамка обычно означает плохой детект.
    if eye_width < 3 or eye_height < 2:
        return None, None, None

    # Нормализуем положение радужки внутри глаза.
    x_norm = (iris_center[0] - left_x) / eye_width
    y_norm = (iris_center[1] - top_y) / eye_height

    # Третье значение — аспект (высота/ширина) как индикатор открытости глаза.
    return float(x_norm), float(y_norm), float(eye_height / eye_width)


def compute_gaze_vector_3d(
    eye_dict: dict, landmarks
) -> Optional[Tuple[float, float]]:
    try:
        # Берем базовые точки глаза в 3D.
        inner_3d = get_point_3d(eye_dict["inner"], landmarks)
        outer_3d = get_point_3d(eye_dict["outer"], landmarks)
        top_3d = get_point_3d(eye_dict["top_lid"], landmarks)
        bottom_3d = get_point_3d(eye_dict["bottom_lid"], landmarks)
        iris_3d = get_point_3d(eye_dict["iris_center"], landmarks)

        # Центр глаза как среднее по горизонтальной и вертикальной серединам.
        mid_horizontal = (inner_3d + outer_3d) / 2.0
        mid_vertical = (top_3d + bottom_3d) / 2.0
        eye_center_3d = (mid_horizontal + mid_vertical) / 2.0

        # Направление взгляда = от центра глаза к центру радужки.
        gaze_dir = _safe_normalize(iris_3d - eye_center_3d)
        if gaze_dir is None:
            return None

        # Локальные оси глаза.
        eye_horizontal = _safe_normalize(outer_3d - inner_3d)
        if eye_horizontal is None:
            return None

        eye_vertical = _safe_normalize(bottom_3d - top_3d)
        if eye_vertical is None:
            return None

        # Проекция направления взгляда на локальные оси глаза.
        gaze_x = np.dot(gaze_dir, eye_horizontal)
        gaze_y = np.dot(gaze_dir, eye_vertical)

        return float(gaze_x), float(gaze_y)
    except Exception:
        return None


def _eye_quality(y_norm: Optional[float], ar: Optional[float]) -> float:
    # Оцениваем доверие к глазу:
    # - если глаз прикрыт (низкий ar), качество падает;
    # - если радужка слишком близко к краю века, качество также ниже.
    if y_norm is None or ar is None:
        return 0.0
    openness = _clip01((ar - 0.12) / 0.18)
    edge_dist = min(y_norm, 1.0 - y_norm)
    y_score = _clip01((edge_dist - 0.05) / 0.15)
    return openness * y_score


def _weighted_mean(values, weights, default: float) -> float:
    # Усреднение с весами и мягким fallback, если весов недостаточно.
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
    # Обрабатываем оба глаза одинаково.
    eyes = (RIGHT_EYE, LEFT_EYE)
    iris_data = [get_iris_position_2d(eye, landmarks, w, h) for eye in eyes]

    # Если обе оценки недоступны — отдаём None.
    if all(gx is None for gx, _, _ in iris_data):
        return None

    # Вес каждого глаза зависит от текущего качества детекта.
    weights = [_eye_quality(gy, ar) for _, gy, ar in iris_data]

    # Смешиваем признаки двух глаз в единый устойчивый вектор.
    gx = _weighted_mean([gx for gx, _, _ in iris_data], weights, 0.5)
    gy = _weighted_mean([gy for _, gy, _ in iris_data], weights, 0.5)
    ar = _weighted_mean([aspect for _, _, aspect in iris_data], weights, 0.3)

    # 3D-направление так же объединяем через те же веса качества.
    gaze_3d = [compute_gaze_vector_3d(eye, landmarks) for eye in eyes]
    g3x = _weighted_mean([g[0] if g is not None else None for g in gaze_3d], weights, 0.0)
    g3y = _weighted_mean([g[1] if g is not None else None for g in gaze_3d], weights, 0.0)

    # Итоговые признаки подаются в калибратор/трекер.
    return {
        "gaze_x_2d": float(gx),
        "gaze_y_2d": float(gy),
        "gaze_x_3d": float(g3x),
        "gaze_y_3d": float(g3y),
        "eye_aspect": float(ar),
    }
