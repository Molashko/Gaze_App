import numpy as np

# Индексы landmark-точек MediaPipe Face Mesh при refine_landmarks=True:
# 478 точек (468 лицо + 10 радужка)

RIGHT_EYE = {
    "inner": 133,
    "outer": 33,
    "top_lid": 159,
    "bottom_lid": 145,
    "top_lid_inner": 160,
    "top_lid_outer": 158,
    "bottom_lid_inner": 144,
    "bottom_lid_outer": 153,
    "iris_center": 468,
    "iris_points": [468, 469, 470, 471, 472],
}

LEFT_EYE = {
    "inner": 362,
    "outer": 263,
    "top_lid": 386,
    "bottom_lid": 374,
    "top_lid_inner": 387,
    "top_lid_outer": 385,
    "bottom_lid_inner": 381,
    "bottom_lid_outer": 380,
    "iris_center": 473,
    "iris_points": [473, 474, 475, 476, 477],
}

FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),          # Кончик носа (1)
        (0.0, -330.0, -65.0),     # Подбородок (199)
        (-225.0, 170.0, -135.0),  # Левый угол глаза (33)
        (225.0, 170.0, -135.0),   # Правый угол глаза (263)
        (-150.0, -150.0, -125.0), # Левый угол рта (61)
        (150.0, -150.0, -125.0),  # Правый угол рта (291)
    ],
    dtype=np.float64,
)

HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]
