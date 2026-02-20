import numpy as np
import pytest
from sklearn.linear_model import RANSACRegressor, Ridge

from gaze_tracker.calibration import RegressionCalibrator


def _feature(i: int) -> np.ndarray:
    return np.array(
        [i * 0.1, i * 0.2, i * 0.3, i * 0.4, i * 0.5, i * 0.6, i * 0.7, i * 0.8],
        dtype=np.float64,
    )


@pytest.mark.calibration
@pytest.mark.smoke
def test_fit_requires_minimum_samples() -> None:
    calibrator = RegressionCalibrator()
    for i in range(4):
        calibrator.add_sample(_feature(i), (0.1 * i, 0.2 * i))
    assert calibrator.fit() is False
    assert calibrator.result is None


@pytest.mark.calibration
@pytest.mark.smoke
def test_predict_returns_none_before_fit() -> None:
    calibrator = RegressionCalibrator()
    assert calibrator.predict(_feature(1)) is None


@pytest.mark.calibration
def test_fit_and_predict_returns_finite_coordinates() -> None:
    calibrator = RegressionCalibrator()
    for i in range(12):
        calibrator.add_sample(_feature(i), (i / 11.0, 1.0 - (i / 11.0)))

    assert calibrator.fit() is True
    pred = calibrator.predict(_feature(5))
    assert pred is not None
    x, y = pred
    assert np.isfinite(x)
    assert np.isfinite(y)


@pytest.mark.calibration
def test_build_ransac_produces_estimator_instance() -> None:
    ransac = RegressionCalibrator._build_ransac(base=Ridge(alpha=1.0), random_state=42)
    assert isinstance(ransac, RANSACRegressor)
