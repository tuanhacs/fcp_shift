import numpy as np

from fcp_shift.conformal.algorithms import (
    estimate_g_algorithm1,
    estimate_g_inverse_algorithm2,
    select_level_algorithm3,
)
from fcp_shift.conformal.weighted_cp import CalibrationStructure


def test_algorithm_1_and_generalized_inverse_step_function():
    structure = CalibrationStructure.build(
        np.array([1.0, 2.0, 3.0]), np.ones(3), w_infinity=1.0
    )
    np.testing.assert_allclose(
        estimate_g_algorithm1(structure, np.array([0.2, 0.5, 0.8])),
        np.array([0.0, 2.0 / 3.0, 1.0]),
    )
    np.testing.assert_allclose(
        estimate_g_inverse_algorithm2(structure, np.array([0.0, 0.34, 0.8])),
        np.array([0.25, 0.5, 0.75]),
    )


def test_algorithm_3_returns_nonnegative_levels():
    structure = CalibrationStructure.build(np.arange(5.0), np.ones(5))
    levels = select_level_algorithm3(
        structure, np.array([0.01, 0.5]), delta_shift=0.2, epsilon_test=0.1
    )
    assert np.all(levels >= 0.0)
    assert levels[0] == 0.0

