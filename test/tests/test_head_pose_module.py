from types import SimpleNamespace

import numpy as np
import pytest

from gaze_tracker import head_pose as hp
from gaze_tracker.landmarks import HEAD_POSE_LANDMARKS


def _landmarks_stub():
    max_idx = max(HEAD_POSE_LANDMARKS)
    arr = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(max_idx + 1)]
    return arr


@pytest.mark.head_pose
@pytest.mark.smoke
def test_normalize_angle_deg_stays_within_expected_range() -> None:
    for angle in (0.0, 90.0, -90.0, 180.0, -180.0, 270.0, -270.0):
        out = hp._normalize_angle_deg(angle)
        assert -90.0 <= out <= 90.0


@pytest.mark.head_pose
@pytest.mark.smoke
def test_estimate_head_pose_returns_zero_tuple_when_solvepnp_fails(monkeypatch) -> None:
    monkeypatch.setattr(hp.cv2, "solvePnP", lambda *args, **kwargs: (False, None, None))
    out = hp.estimate_head_pose(_landmarks_stub(), 640, 480)
    assert out == (0.0, 0.0, 0.0)


@pytest.mark.head_pose
def test_estimate_head_pose_happy_path(monkeypatch) -> None:
    monkeypatch.setattr(
        hp.cv2,
        "solvePnP",
        lambda *args, **kwargs: (True, np.array([[0.1], [0.2], [0.3]], dtype=np.float64), np.zeros((3, 1))),
    )
    monkeypatch.setattr(hp.cv2, "Rodrigues", lambda _rvec: (np.eye(3, dtype=np.float64), None))
    monkeypatch.setattr(hp, "_rotation_matrix_to_euler_angles", lambda _r: (10.0, 20.0, 30.0))

    out = hp.estimate_head_pose(_landmarks_stub(), 640, 480)
    assert out == pytest.approx((20.0, 10.0, 30.0))


@pytest.mark.head_pose
def test_estimate_head_pose_returns_zero_tuple_on_exception(monkeypatch) -> None:
    monkeypatch.setattr(hp.cv2, "solvePnP", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    out = hp.estimate_head_pose(_landmarks_stub(), 640, 480)
    assert out == (0.0, 0.0, 0.0)
