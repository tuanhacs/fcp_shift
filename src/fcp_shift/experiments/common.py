from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fcp_shift.conformal.algorithms import estimate_g_algorithm1, select_level_algorithm3
from fcp_shift.conformal.bounds import fixed_constants, uniform_constants
from fcp_shift.conformal.weighted_cp import CalibrationStructure, fcp_at_levels, fcp_curve


@dataclass
class GoalResult:
    empirical_fcp: np.ndarray
    goal1_bound: np.ndarray
    goal2_bound: np.ndarray
    goal3_alpha: np.ndarray
    goal4_alpha: np.ndarray
    goal3_fcp: np.ndarray
    goal4_fcp: np.ndarray
    fixed: dict[str, Any]
    uniform: dict[str, Any]


def grid(spec: dict[str, Any]) -> np.ndarray:
    if "values" in spec:
        return np.asarray(spec["values"], dtype=float)
    return np.linspace(
        float(spec.get("start", 0.0)),
        float(spec.get("stop", 1.0)),
        int(spec.get("points", 201)),
    )


def calculate_goals(
    calibration_scores: np.ndarray,
    calibration_weights: np.ndarray,
    test_scores: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    bound: float,
    delta: float,
    w_infinity: float,
    g_mode: str,
    optimize_delta: bool,
    eta: float,
) -> GoalResult:
    n = len(calibration_scores)
    m = len(test_scores)
    structure = CalibrationStructure.build(
        calibration_scores, calibration_weights, w_infinity=w_infinity
    )
    p_values = structure.p_values(test_scores)
    empirical = fcp_curve(p_values, alpha_grid)
    fixed = fixed_constants(
        bound, n, m, delta, w_infinity=w_infinity, optimize=optimize_delta
    )
    uniform = uniform_constants(
        bound, n, m, delta, w_infinity=w_infinity, optimize=optimize_delta
    )

    if g_mode == "covariate_identity":
        goal1 = alpha_grid + fixed.delta_shift + fixed.epsilon_test
        goal2 = alpha_grid + uniform.delta_shift + uniform.epsilon_test
        goal3_alpha = np.maximum(
            beta_grid - fixed.epsilon_test - fixed.delta_shift - eta, 0.0
        )
        goal4_alpha = np.maximum(
            beta_grid - uniform.epsilon_test - uniform.delta_shift - eta, 0.0
        )
    elif g_mode == "algorithm_1":
        goal1 = estimate_g_algorithm1(
            structure, alpha_grid + fixed.delta_shift
        ) + fixed.epsilon_test
        goal2 = estimate_g_algorithm1(
            structure, alpha_grid + uniform.delta_shift
        ) + uniform.epsilon_test
        goal3_alpha = select_level_algorithm3(
            structure,
            beta_grid,
            fixed.delta_shift,
            fixed.epsilon_test,
            eta,
        )
        goal4_alpha = select_level_algorithm3(
            structure,
            beta_grid,
            uniform.delta_shift,
            uniform.epsilon_test,
            eta,
        )
    else:
        raise ValueError(f"Unsupported G mode: {g_mode}")

    return GoalResult(
        empirical_fcp=empirical,
        goal1_bound=goal1,
        goal2_bound=goal2,
        goal3_alpha=goal3_alpha,
        goal4_alpha=goal4_alpha,
        goal3_fcp=fcp_at_levels(p_values, goal3_alpha),
        goal4_fcp=fcp_at_levels(p_values, goal4_alpha),
        fixed=fixed.as_dict(),
        uniform=uniform.as_dict(),
    )


def stack_goal_results(results: list[GoalResult]) -> dict[str, np.ndarray]:
    names = [
        "empirical_fcp",
        "goal1_bound",
        "goal2_bound",
        "goal3_alpha",
        "goal4_alpha",
        "goal3_fcp",
        "goal4_fcp",
    ]
    return {name: np.stack([getattr(result, name) for result in results]) for name in names}

