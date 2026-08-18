from fcp_shift.conformal.bounds import fixed_constants, uniform_constants


def test_fixed_and_uniform_constants_are_positive():
    fixed = fixed_constants(2.0, n=1000, m=500, delta=0.1)
    uniform = uniform_constants(2.0, n=1000, m=500, delta=0.1)
    assert fixed.delta_shift > 0.0
    assert fixed.epsilon_test > 0.0
    assert uniform.delta_shift >= fixed.delta_shift
    assert uniform.epsilon_test > 0.0
    assert abs(fixed.delta_calibration + fixed.delta_test - 0.1) < 1e-10
    assert abs(uniform.delta_plus + uniform.delta_minus + uniform.delta_test - 0.1) < 1e-10

