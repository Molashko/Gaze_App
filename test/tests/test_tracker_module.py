from types import SimpleNamespace

import numpy as np
import pytest

from gaze_tracker.tracker import GazeTracker


def _gaze_features():
    return {
        "gaze_x_2d": 0.2,
        "gaze_y_2d": 0.4,
        "gaze_x_3d": 0.1,
        "gaze_y_3d": -0.1,
        "eye_aspect": 0.3,
    }


@pytest.mark.tracker
@pytest.mark.smoke
def test_build_feature_vector_shape_and_dtype() -> None:
    tracker = GazeTracker(1920, 1080)
    vec = tracker.build_feature_vector(_gaze_features(), (1.0, 2.0, 3.0))
    assert vec.shape == (8,)
    assert vec.dtype == np.float64


@pytest.mark.tracker
@pytest.mark.smoke
def test_finalize_calibration_updates_flag(monkeypatch) -> None:
    tracker = GazeTracker(1920, 1080)
    monkeypatch.setattr(tracker.calibrator, "fit", lambda: True)
    assert tracker.finalize_calibration() is True
    assert tracker.calibrated is True


@pytest.mark.tracker
@pytest.mark.smoke
def test_predict_screen_returns_none_without_calibration() -> None:
    tracker = GazeTracker(1920, 1080)
    assert tracker.predict_screen(np.zeros(8, dtype=np.float64)) is None


@pytest.mark.tracker
def test_predict_screen_clips_and_scales() -> None:
    tracker = GazeTracker(1000, 500)
    tracker.calibrator = SimpleNamespace(predict=lambda _vec: (-1.0, 2.0))
    x, y = tracker.predict_screen(np.zeros(8, dtype=np.float64))
    assert (x, y) == (0.0, 500.0)


@pytest.mark.tracker
def test_smooth_uses_filters(monkeypatch) -> None:
    tracker = GazeTracker(1000, 500)
    monkeypatch.setattr(tracker, "median_filter", lambda x, y: (x + 1.0, y + 1.0))
    monkeypatch.setattr(tracker, "filter_x", lambda x, t: x + 10.0)
    monkeypatch.setattr(tracker, "filter_y", lambda y, t: y + 20.0)
    x, y = tracker.smooth(5.0, 7.0)
    assert x == pytest.approx(16.0)
    assert y == pytest.approx(28.0)
