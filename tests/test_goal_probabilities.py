import numpy as np

from fcp_shift.experiments.common import calculate_goals, stack_goal_results


def test_goal_indicators_match_fixed_and_uniform_events() -> None:
    alpha = np.linspace(0.0, 1.0, 9)
    beta = np.linspace(0.0, 1.0, 9)
    result = calculate_goals(
        calibration_scores=np.array([0.1, 0.2, 0.4, 0.8]),
        calibration_weights=np.ones(4),
        test_scores=np.array([0.15, 0.3, 0.9]),
        alpha_grid=alpha,
        beta_grid=beta,
        bound=1.0,
        delta=0.1,
        w_infinity=1.0,
        g_mode="covariate_identity",
        optimize_delta=False,
        eta=1e-10,
    )

    np.testing.assert_array_equal(
        result.goal1_pass, result.empirical_fcp <= result.goal1_bound + 1e-12
    )
    assert result.goal2_uniform_pass == bool(
        np.all(result.empirical_fcp <= result.goal2_bound + 1e-12)
    )
    np.testing.assert_array_equal(
        result.goal3_pass, result.goal3_fcp <= beta + 1e-12
    )
    assert result.goal4_uniform_pass == bool(
        np.all(result.goal4_fcp <= beta + 1e-12)
    )

    arrays = stack_goal_results([result, result])
    assert arrays["goal1_pass"].shape == (2, len(alpha))
    assert arrays["goal2_uniform_pass"].shape == (2,)
    assert arrays["goal3_pass"].shape == (2, len(beta))
    assert arrays["goal4_uniform_pass"].shape == (2,)
