import ctypes
import threading
import subprocess
import time
from typing import List, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

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
    LOG_INTERVAL,
)
from ..gaze_estimation import extract_gaze_features, get_point_2d, get_iris_center_2d
from ..head_pose import estimate_head_pose
from ..landmarks import RIGHT_EYE, LEFT_EYE
from ..logging_utils import log
from ..tracker import GazeTracker

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False


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
        self.root.title("Gaze Tracker v3")
        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        log(f"=== App started v3 === Screen: {self.screen_w}x{self.screen_h}")

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

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Gaze Tracker v3", font=("Segoe UI", 14, "bold")).pack(pady=(0, 8))
        info_text = (
            "1. Sit comfortably, face the camera\n"
            "2. Click 'Calibrate' and follow the dots\n"
            "3. Keep your head relatively still"
        )
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

        import sys
        threading.Thread(target=worker, daemon=True).start()

    def _run_calibration(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror("Error", "MediaPipe not installed")
            return
        if self.tracking_active:
            messagebox.showinfo("Calibration", "Stop tracking first")
            return

        log("=== CALIBRATION STARTED v3 ===")
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

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        log("MediaPipe FaceMesh initialized (478 landmarks)")

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFERSIZE)
        self._cap = cap
        log("Camera opened")

        display_w, display_h = self.screen_w, self.screen_h
        cv2.namedWindow("Gaze Output", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 480, 360)
        cv2.moveWindow("Camera", 10, 10)

        frame_count = 0

        while self.tracking_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            img_h, img_w = frame.shape[:2]

            if frame_count < WARMUP_FRAMES:
                canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                cv2.putText(
                    canvas,
                    "Camera warming up...",
                    (display_w // 2 - 150, display_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Gaze Output", canvas)
                cv2.imshow("Camera", frame)
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
                    cv2.putText(
                        cam_display,
                        f"Gaze2D: ({gaze_features['gaze_x_2d']:.3f}, {gaze_features['gaze_y_2d']:.3f})",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        cam_display,
                        f"Head: Y={yaw:.0f} P={pitch:.0f}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 0),
                        1,
                    )
            else:
                cv2.putText(
                    cam_display,
                    "Face not detected!",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Camera", cam_display)
            canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)

            if self.calibrating and not self.calibration_complete:
                if self.current_calib_idx >= len(self.calib_points):
                    self.tracker.finalize_calibration()
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

                    cv2.putText(canvas, f"Calibration: {self.current_calib_idx + 1}/{len(self.calib_points)}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(canvas, "LOOK AT THE RED DOT",
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                    cv2.putText(canvas, f"Samples: {len(self.calib_samples)}/{self.calib_min_samples}",
                                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

                    if elapsed >= self.calib_hold_time and len(self.calib_samples) >= self.calib_min_samples:
                        features = np.array(self.calib_samples, dtype=np.float64)
                        median_feature = np.median(features, axis=0)
                        screen_x_norm = cx / self.screen_w
                        screen_y_norm = cy / self.screen_h
                        self.tracker.add_calibration_sample(median_feature, (screen_x_norm, screen_y_norm))
                        log(
                            f"Point {self.current_calib_idx + 1}: features={median_feature.round(3).tolist()} "
                            f"-> screen=({screen_x_norm:.2f}, {screen_y_norm:.2f})"
                        )
                        self.current_calib_idx += 1
                        self.calib_point_start = None
                        self.calib_samples.clear()

            if gaze_features is not None and self.tracker.calibrated:
                feature_vec = self.tracker.build_feature_vector(gaze_features, self.current_head_pose)
                screen_pos = self.tracker.predict_screen(feature_vec)
                if screen_pos is not None:
                    if frame_count % LOG_INTERVAL == 0:
                        log(f"Prediction: ({screen_pos[0]:.0f}, {screen_pos[1]:.0f})")

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
                        cv2.putText(canvas, f"Position: ({dot_x}, {dot_y})",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        cv2.putText(canvas, "Look at circles | Press Q to exit",
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
