import sys
import time
import threading
import subprocess
from typing import Optional, Tuple

import ctypes

import cv2
import numpy as np

try:
    from eyeGestures import EyeGestures_v3
    from eyeGestures.utils import VideoCapture
    EYE_GESTURES_AVAILABLE = True
except Exception:
    EYE_GESTURES_AVAILABLE = False

import tkinter as tk
from tkinter import messagebox


class GazeApp:
    def __init__(self):
        self.screen_w, self.screen_h = self._get_screen_size()
        self.tracking_active = False
        self._tracking_cap: Optional[cv2.VideoCapture] = None
        self.eye_context_id = 0
        self.calibrating = False
        self.calib_iterator = 0
        self.calib_prev_point = None
        self.calib_n_points = 25
        self.calib_hold_seconds = 1.5
        self.calib_point_start = None

        self.root = tk.Tk()
        self.root.title("Трекер взгляда (прототип)")
        self.status_var = tk.StringVar(value="Готово.")
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

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Трекер взгляда (прототип)", font=("Segoe UI", 14)).pack(pady=(0, 8))

        tk.Button(frame, text="Установить зависимости", command=self._install_deps).pack(fill="x", pady=4)
        tk.Button(frame, text="Калибровка EyeGestures", command=self._run_calibration).pack(fill="x", pady=4)
        tk.Button(frame, text="Сбросить калибровку", command=self._reset_calibration).pack(fill="x", pady=4)
        tk.Button(frame, text="Старт трекинга", command=self._start_tracking).pack(fill="x", pady=4)
        tk.Button(frame, text="Выход", command=self._quit).pack(fill="x", pady=(12, 0))

        if not EYE_GESTURES_AVAILABLE:
            tk.Label(frame, text="EyeGestures не установлен.", fg="#666").pack(anchor="w")

        tk.Label(frame, textvariable=self.status_var, fg="#444").pack(pady=(10, 0))

    def _install_deps(self) -> None:
        self.status_var.set("Установка зависимостей...")

        def worker():
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                self.status_var.set("Зависимости установлены.")
                messagebox.showinfo("Готово", "Зависимости успешно установлены.")
            except Exception as exc:
                self.status_var.set("Ошибка установки.")
                messagebox.showerror("Ошибка", f"Не удалось установить зависимости:\n{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _run_calibration(self) -> None:
        if not EYE_GESTURES_AVAILABLE:
            messagebox.showerror("Ошибка", "EyeGestures не установлен.")
            return
        if self.tracking_active:
            messagebox.showinfo("Калибровка", "Остановите трекинг и попробуйте снова.")
            return
        self.calibrating = True
        self.calib_iterator = 0
        self.calib_prev_point = None
        self.status_var.set("Калибровка EyeGestures началась.")
        self._start_tracking()

    def _start_tracking(self) -> None:
        if not EYE_GESTURES_AVAILABLE:
            messagebox.showerror("Ошибка", "EyeGestures не установлен.")
            return

        if self.tracking_active:
            messagebox.showinfo("Трекинг", "Трекинг уже запущен.")
            return

        self.status_var.set("Трекинг EyeGestures...")
        self.tracking_active = True

        gestures = EyeGesturesV3Patched()
        gestures.setFixation(1.0)
        cap = VideoCapture(0)
        self._tracking_cap = cap

        display_w, display_h = self.screen_w, self.screen_h
        cv2.namedWindow("Gaze Output", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        context_id = f"context_{self.eye_context_id}"
        if self.calibrating:
            calibration_map = self._create_calibration_map()
            self.calib_n_points = min(len(calibration_map), 25)
            np.random.shuffle(calibration_map)
            gestures.uploadCalibrationMap(calibration_map, context=context_id)

        while self.tracking_active:
            ret, frame = cap.read()
            if not ret:
                continue

            calibrate = self.calibrating and (self.calib_iterator <= self.calib_n_points)
            try:
                event, _ = gestures.step(
                    frame,
                    calibrate,
                    self.screen_w,
                    self.screen_h,
                    context=context_id,
                )
            except Exception as exc:
                self.tracking_active = False
                messagebox.showerror("Ошибка трекинга", f"{exc}")
                break

            if event is None:
                continue

            if calibrate:
                now = time.time()
                calibration = gestures.clb[context_id].getCurrentPoint(self.screen_w, self.screen_h)
                if self.calib_prev_point is None or calibration[0] != self.calib_prev_point[0] or calibration[1] != self.calib_prev_point[1]:
                    self.calib_prev_point = calibration
                    self.calib_point_start = now

                if self.calib_point_start is not None and (now - self.calib_point_start) >= self.calib_hold_seconds:
                    self.calib_iterator += 1
                    self.calib_point_start = now

                if self.calib_iterator >= self.calib_n_points:
                    self.calibrating = False
                    self.status_var.set("Калибровка завершена. Идет трекинг.")
                    messagebox.showinfo("Калибровка", "Калибровка завершена.")

            if event:
                x, y = float(event.point[0]), float(event.point[1])
                x = max(0.0, min(self.screen_w - 1, x))
                y = max(0.0, min(self.screen_h - 1, y))

                dot_x = int(x / self.screen_w * display_w)
                dot_y = int(y / self.screen_h * display_h)

                canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                cv2.circle(canvas, (dot_x, dot_y), 6, (0, 0, 255), -1)
                cv2.putText(
                    canvas,
                    f"x={int(x)} y={int(y)}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                if calibrate and self.calib_prev_point is not None:
                    cx = int(self.calib_prev_point[0] / self.screen_w * display_w)
                    cy = int(self.calib_prev_point[1] / self.screen_h * display_h)
                    cv2.circle(canvas, (cx, cy), 10, (255, 0, 0), 2)
                    cv2.putText(
                        canvas,
                        f"{self.calib_iterator}/{self.calib_n_points}",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 200),
                        2,
                    )
            cv2.imshow("Gaze Output", canvas)

            # Allow Tkinter events (so Exit button works)
            self.root.update_idletasks()
            self.root.update()
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        self.tracking_active = False
        if hasattr(cap, "close"):
            cap.close()
        else:
            cap.release()
        cv2.destroyAllWindows()
        self.status_var.set("Трекинг остановлен.")

    def _reset_calibration(self) -> None:
        self.calibrating = False
        self.calib_iterator = 0
        self.calib_prev_point = None
        self.calib_point_start = None
        self.eye_context_id += 1
        self.status_var.set("Калибровка сброшена.")
        messagebox.showinfo("Готово", "Калибровка сброшена.")

    def _quit(self) -> None:
        self.tracking_active = False
        if self._tracking_cap is not None:
            if hasattr(self._tracking_cap, "close"):
                self._tracking_cap.close()
            else:
                self._tracking_cap.release()
            self._tracking_cap = None
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

    def _create_calibration_map(self) -> np.ndarray:
        x = np.arange(0, 1.1, 0.2)
        y = np.arange(0, 1.1, 0.2)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])


class EyeGesturesV3Patched(EyeGestures_v3):
    def getLandmarks(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        self.face.process(
            frame,
            self.finder.find(frame),
        )

        face_landmarks = self.face.getLandmarks()
        l_eye = self.face.getLeftEye()
        r_eye = self.face.getRightEye()
        l_eye_landmarks = l_eye.getLandmarks()
        r_eye_landmarks = r_eye.getLandmarks()
        blink = l_eye.getBlink() and r_eye.getBlink()

        x_offset = np.min(face_landmarks[:, 0])
        y_offset = np.min(face_landmarks[:, 1])
        x_width = np.max(face_landmarks[:, 0]) - x_offset
        y_width = np.max(face_landmarks[:, 1]) - y_offset

        head_offset = np.zeros((1, 2), dtype=float)
        scale_x = 1.0
        scale_y = 1.0
        if np.array_equal(self.starting_head_position, np.zeros((1, 2))):
            self.starting_head_position = np.array([[x_offset, y_offset]], dtype=float)
            self.starting_size = np.array([[x_width, y_width]], dtype=float)
        else:
            head_offset = np.array([[x_offset, y_offset]], dtype=float) - self.starting_head_position
            scale_x = float(self.starting_size[0, 0] / x_width)
            scale_y = float(self.starting_size[0, 1] / y_width)

        key_points = np.concatenate((l_eye_landmarks, r_eye_landmarks, np.array([[scale_x, scale_y]]), head_offset))
        key_points[:, 0] = key_points[:, 0] - head_offset[0, 0]
        key_points[:, 1] = key_points[:, 1] - head_offset[0, 1]
        key_points[:, 0] = key_points[:, 0] * scale_x
        key_points[:, 1] = key_points[:, 1] * scale_y
        key_points[-1, 0] = float(head_offset[0, 0])
        key_points[-1, 1] = float(head_offset[0, 1])

        subframe = frame[int(y_offset):int(y_offset + y_width), int(x_offset):int(x_offset + x_width)]
        return key_points, blink, subframe

def main() -> None:
    app = GazeApp()
    app.run()


if __name__ == "__main__":
    main()
