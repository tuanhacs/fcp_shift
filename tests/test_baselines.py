import numpy as np

from fcp_shift.conformal.baselines import dkw_forward, dkw_inverse, unweighted_p_values


def test_unweighted_p_values_ignore_calibration_weights_by_construction():
    calibration_scores = np.array([1.0, 2.0, 3.0])
    actual = unweighted_p_values(np.array([1.0, 2.5, 4.0]), calibration_scores)
    np.testing.assert_allclose(actual, np.array([1.0, 0.5, 0.25]))


def test_dkw_forward_and_inverse_are_monotone():
    grid = np.linspace(0.0, 1.0, 21)
    forward = dkw_forward(grid, n=100, m=50, delta=0.1)
    inverse = dkw_inverse(grid, n=100, m=50, delta=0.1)
    assert np.all(np.diff(forward) >= 0.0)
    assert np.all(np.diff(inverse) >= 0.0)
    assert np.all((0.0 <= forward) & (forward <= 1.0))
    assert np.all((0.0 <= inverse) & (inverse <= 1.0))
