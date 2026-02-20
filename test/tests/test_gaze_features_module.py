import pytest

from gaze_tracker import gaze_estimation as ge


@pytest.mark.gaze
@pytest.mark.smoke
def test_weighted_mean_weighted_and_fallback() -> None:
    assert ge._weighted_mean([1.0, 3.0], [1.0, 3.0], 0.0) == pytest.approx(2.5)
    assert ge._weighted_mean([None, 2.0], [0.0, 0.0], 0.7) == pytest.approx(2.0)
    assert ge._weighted_mean([None, None], [0.0, 0.0], 0.7) == pytest.approx(0.7)


@pytest.mark.gaze
@pytest.mark.smoke
def test_eye_quality_is_bounded_and_zero_for_invalid() -> None:
    assert ge._eye_quality(None, 0.2) == 0.0
    assert ge._eye_quality(0.5, None) == 0.0
    quality = ge._eye_quality(0.5, 0.3)
    assert 0.0 <= quality <= 1.0


@pytest.mark.gaze
@pytest.mark.smoke
def test_extract_gaze_features_returns_none_when_both_eyes_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(ge, "get_iris_position_2d", lambda *args, **kwargs: (None, None, None))
    monkeypatch.setattr(ge, "compute_gaze_vector_3d", lambda *args, **kwargs: None)
    assert ge.extract_gaze_features(landmarks=object(), w=640, h=480) is None


@pytest.mark.gaze
def test_extract_gaze_features_uses_weighted_merge(monkeypatch) -> None:
    seq_iris = iter([(0.2, 0.5, 0.3), (0.9, 0.1, 0.12)])
    seq_3d = iter([(0.4, -0.2), (0.0, 0.5)])

    monkeypatch.setattr(ge, "get_iris_position_2d", lambda *args, **kwargs: next(seq_iris))
    monkeypatch.setattr(ge, "compute_gaze_vector_3d", lambda *args, **kwargs: next(seq_3d))

    out = ge.extract_gaze_features(landmarks=object(), w=640, h=480)
    assert out is not None
    assert out["gaze_x_2d"] == pytest.approx(0.2)
    assert out["gaze_y_2d"] == pytest.approx(0.5)
    assert out["eye_aspect"] == pytest.approx(0.3)
    assert out["gaze_x_3d"] == pytest.approx(0.4)
    assert out["gaze_y_3d"] == pytest.approx(-0.2)
