# -*- coding: utf-8 -*-
import ctypes
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

WINDOW_GAZE = "Вывод взгляда"
WINDOW_CAMERA = "Камера"

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_AVAILABLE = False

from ..config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CAMERA_BUFFERSIZE,
    CALIBRATION_GRID,
    CALIBRATION_EXTRA_POINTS,
    CALIBRATION_HOLD_TIME,
    CALIBRATION_MIN_SAMPLES,
    CALIBRATION_WARMUP_RATIO,
    WARMUP_FRAMES,
)
from ..gaze_estimation import extract_gaze_features, get_point_2d, get_iris_center_2d
from ..head_pose import estimate_head_pose
from ..landmarks import RIGHT_EYE, LEFT_EYE
from ..tracker import GazeTracker

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


def _draw_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    if not PIL_AVAILABLE:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return

    font_size = max(12, int(font_scale * 32))
    font_path = r"C:\Windows\Fonts\arial.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = int(org[0]), int(org[1])
    draw.text((x, y), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
    img[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class GazeApp:
    def __init__(self) -> None:
        self.screen_w, self.screen_h = self._get_screen_size()
        self.tracking_active = False
        self._cap = None
        self.calibrating = False
        self.calibration_complete = False
        self.tracker = GazeTracker(self.screen_w, self.screen_h)
        self.face_mesh = None

        self.calib_points = self._create_calibration_points()
        self.current_calib_idx = 0
        self.calib_hold_time = CALIBRATION_HOLD_TIME
        self.calib_point_start = None
        self.calib_samples: List[np.ndarray] = []
        self.calib_min_samples = CALIBRATION_MIN_SAMPLES

        self.current_head_pose = (0.0, 0.0, 0.0)

        self.root = tk.Tk()
        self.root.title("Трекер взгляда v3")
        self.status_var = tk.StringVar(value="Готово")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)


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
        points = []
        for y_pct in CALIBRATION_GRID:
            for x_pct in CALIBRATION_GRID:
                points.append((int(x_pct * self.screen_w), int(y_pct * self.screen_h)))
        for x_pct, y_pct in CALIBRATION_EXTRA_POINTS:
            points.append((int(x_pct * self.screen_w), int(y_pct * self.screen_h)))
        points = list(dict.fromkeys(points))
        return points

    def _add_button(
        self,
        frame: tk.Frame,
        text: str,
        command,
        *,
        pady: Tuple[int, int] | int = 4,
        **kwargs,
    ) -> None:
        tk.Button(frame, text=text, command=command, **kwargs).pack(fill="x", pady=pady)

    def _ensure_mediapipe(self) -> bool:
        if MEDIAPIPE_AVAILABLE:
            return True
        messagebox.showerror("Ошибка", "MediaPipe не установлен")
        return False

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Трекер взгляда v3", font=("Segoe UI", 14, "bold")).pack(pady=(0, 8))
        tk.Label(frame, text="", justify="left", fg="#555").pack(pady=(0, 10))

        self._add_button(frame, "Калибровка", self._run_calibration, bg="#4CAF50", fg="white")
        self._add_button(frame, "Сброс калибровки", self._reset_calibration)
        self._add_button(frame, "Начать отслеживание", self._start_tracking)
        self._add_button(frame, "Выход", self._quit, pady=(12, 0))

        if not MEDIAPIPE_AVAILABLE:
            tk.Label(frame, text="MediaPipe не установлен!", fg="red").pack(anchor="w")

        tk.Label(frame, textvariable=self.status_var, fg="#444").pack(pady=(10, 0))

    def _run_calibration(self) -> None:
        if not self._ensure_mediapipe():
            return
        if self.tracking_active:
            messagebox.showinfo("Калибровка", "Сначала остановите отслеживание")
            return

        self.calibrating = True
        self.calibration_complete = False
        self.current_calib_idx = 0
        self.calib_point_start = None
        self.calib_samples.clear()
        self.tracker.reset()
        self.status_var.set("Калибровка... Смотрите на красные точки")
        self._start_tracking()

    def _start_tracking(self) -> None:
        if not self._ensure_mediapipe():
            return
        if self.tracking_active:
            messagebox.showinfo("Отслеживание", "Уже запущено")
            return

        self.status_var.set("Отслеживание...")
        self.tracking_active = True

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFERSIZE)
        self._cap = cap

        display_w, display_h = self.screen_w, self.screen_h
        cv2.namedWindow(WINDOW_GAZE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_GAZE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_CAMERA, 480, 360)
        cv2.moveWindow(WINDOW_CAMERA, 10, 10)

        frame_count = 0

        while self.tracking_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            img_h, img_w = frame.shape[:2]

            if frame_count < WARMUP_FRAMES:
                canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                _draw_text(
                    canvas,
                    "Камера прогревается...",
                    (display_w // 2 - 150, display_h // 2),
                    font_scale=1.0,
                    color=(255, 255, 255),
                    thickness=2,
                )
                cv2.imshow(WINDOW_GAZE, canvas)
                cv2.imshow(WINDOW_CAMERA, frame)
                cv2.waitKey(1)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            cam_display = frame.copy()
            gaze_features = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                yaw, pitch, roll = estimate_head_pose(landmarks, img_w, img_h)
                self.current_head_pose = (yaw, pitch, roll)

                for eye_dict, color in [(RIGHT_EYE, (0, 255, 0)), (LEFT_EYE, (0, 255, 0))]:
                    for key in ["inner", "outer", "top_lid", "bottom_lid"]:
                        pt = get_point_2d(eye_dict[key], landmarks, img_w, img_h)
                        cv2.circle(cam_display, (int(pt[0]), int(pt[1])), 2, color, -1)

                    iris_center = get_iris_center_2d(eye_dict, landmarks, img_w, img_h)
                    if iris_center is not None:
                        cv2.circle(cam_display, (int(iris_center[0]), int(iris_center[1])), 4, (255, 0, 255), -1)

                gaze_features = extract_gaze_features(landmarks, img_w, img_h)
                if gaze_features:
                    _draw_text(
                        cam_display,
                        f"Взгляд2D: ({gaze_features['gaze_x_2d']:.3f}, {gaze_features['gaze_y_2d']:.3f})",
                        (10, 25),
                        font_scale=0.55,
                        color=(0, 255, 255),
                        thickness=2,
                    )
                    _draw_text(
                        cam_display,
                        f"Голова: Y={yaw:.0f} P={pitch:.0f}",
                        (10, 50),
                        font_scale=0.5,
                        color=(255, 200, 0),
                        thickness=1,
                    )
            else:
                _draw_text(
                    cam_display,
                    "Лицо не обнаружено!",
                    (10, 25),
                    font_scale=0.7,
                    color=(0, 0, 255),
                    thickness=2,
                )

            cv2.imshow(WINDOW_CAMERA, cam_display)
            canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)

            if self.calibrating and not self.calibration_complete:
                if self.current_calib_idx >= len(self.calib_points):
                    self.tracker.finalize_calibration()
                    self.calibration_complete = True
                    self.calibrating = False
                    self.status_var.set("Калибровка завершена!")
                else:
                    cx, cy = self.calib_points[self.current_calib_idx]
                    now = time.time()
                    if self.calib_point_start is None:
                        self.calib_point_start = now
                        self.calib_samples.clear()

                    elapsed = now - self.calib_point_start
                    progress = min(1.0, elapsed / self.calib_hold_time)

                    if gaze_features is not None and elapsed > self.calib_hold_time * CALIBRATION_WARMUP_RATIO:
                        feature_vec = self.tracker.build_feature_vector(gaze_features, self.current_head_pose)
                        self.calib_samples.append(feature_vec)

                    pulse = 1.0 + 0.15 * np.sin(now * 6)
                    outer_r = int(45 * pulse)
                    cv2.circle(canvas, (cx, cy), outer_r + 5, (30, 30, 30), -1)
                    if progress > 0:
                        angle = int(360 * progress)
                        cv2.ellipse(canvas, (cx, cy), (outer_r, outer_r), -90, 0, angle, (0, 255, 0), 6)
                    cv2.circle(canvas, (cx, cy), 25, (255, 255, 255), -1)
                    cv2.circle(canvas, (cx, cy), 8, (0, 0, 255), -1)

                    _draw_text(
                        canvas,
                        f"Калибровка: {self.current_calib_idx + 1}/{len(self.calib_points)}",
                        (10, 40),
                        font_scale=1.0,
                        color=(255, 255, 255),
                        thickness=2,
                    )
                    _draw_text(
                        canvas,
                        "СМОТРИТЕ НА КРАСНУЮ ТОЧКУ",
                        (10, 85),
                        font_scale=0.9,
                        color=(0, 200, 255),
                        thickness=2,
                    )
                    _draw_text(
                        canvas,
                        f"Образцы: {len(self.calib_samples)}/{self.calib_min_samples}",
                        (10, 125),
                        font_scale=0.7,
                        color=(180, 180, 180),
                        thickness=2,
                    )

                    if elapsed >= self.calib_hold_time and len(self.calib_samples) >= self.calib_min_samples:
                        features = np.array(self.calib_samples, dtype=np.float64)
                        median_feature = np.median(features, axis=0)
                        screen_x_norm = cx / self.screen_w
                        screen_y_norm = cy / self.screen_h
                        self.tracker.add_calibration_sample(median_feature, (screen_x_norm, screen_y_norm))
                        self.current_calib_idx += 1
                        self.calib_point_start = None
                        self.calib_samples.clear()

            if gaze_features is not None and self.tracker.calibrated:
                feature_vec = self.tracker.build_feature_vector(gaze_features, self.current_head_pose)
                screen_pos = self.tracker.predict_screen(feature_vec)
                if screen_pos is not None:

                    screen_x, screen_y = screen_pos
                    screen_x = max(0, min(self.screen_w - 1, screen_x))
                    screen_y = max(0, min(self.screen_h - 1, screen_y))
                    smooth_x, smooth_y = self.tracker.smooth(screen_x, screen_y)
                    dot_x = int(smooth_x)
                    dot_y = int(smooth_y)
                    cv2.circle(canvas, (dot_x, dot_y), 20, (0, 0, 255), -1)
                    cv2.circle(canvas, (dot_x, dot_y), 22, (255, 255, 255), 3)

                    if not self.calibrating:
                        for pt in self.calib_points:
                            cv2.circle(canvas, pt, 15, (50, 50, 50), 2)
                        _draw_text(
                            canvas,
                            f"Позиция: ({dot_x}, {dot_y})",
                            (10, 40),
                            font_scale=1.0,
                            color=(255, 255, 255),
                            thickness=2,
                        )
                        _draw_text(
                            canvas,
                            "Смотрите на круги | Нажмите Q для выхода",
                            (10, display_h - 25),
                            font_scale=0.6,
                            color=(100, 100, 100),
                            thickness=1,
                        )

            cv2.imshow(WINDOW_GAZE, canvas)

            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        self.tracking_active = False

        if self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception as exc:
            finally:
                self.face_mesh = None
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Отслеживание остановлено")

    def _reset_calibration(self) -> None:
        self.calibrating = False
        self.calibration_complete = False
        self.current_calib_idx = 0
        self.tracker.reset()
        self.status_var.set("Сброс калибровки")
        messagebox.showinfo("Готово", "Калибровка сброшена")

    def _quit(self) -> None:
        self.tracking_active = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception as exc:
            finally:
                self.face_mesh = None
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = GazeApp()
    app.run()
