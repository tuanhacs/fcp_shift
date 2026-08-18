import numpy as np

from fcp_shift.conformal.weighted_cp import CalibrationStructure, fcp_at_levels


def test_weighted_p_values_include_infinity_weight_and_ties():
    structure = CalibrationStructure.build(
        np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]), w_infinity=1.0
    )
    actual = structure.p_values(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(actual, np.array([1.0, 6.0 / 7.0, 1.0 / 7.0]))


def test_fcp_at_different_selected_levels():
    actual = fcp_at_levels(np.array([0.1, 0.4, 0.8]), np.array([0.2, 0.5]))
    np.testing.assert_allclose(actual, np.array([1.0 / 3.0, 2.0 / 3.0]))

