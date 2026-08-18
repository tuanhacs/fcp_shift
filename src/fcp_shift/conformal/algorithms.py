from __future__ import annotations

import numpy as np

from .weighted_cp import CalibrationStructure


def estimate_g_algorithm1(
    structure: CalibrationStructure, beta: np.ndarray | float
) -> np.ndarray:
    """Algorithm 1: estimate G from weighted calibration scores only."""
    beta_array = np.asarray(beta, dtype=float)
    f_values = structure.calibration_cdf_values()
    indicators = f_values[:, None] >= 1.0 - beta_array.reshape(1, -1)
    estimates = np.sum(structure.weights[:, None] * indicators, axis=0)
    estimates = estimates / structure.weight_sum
    return estimates.reshape(beta_array.shape)


def estimate_g_inverse_algorithm2(
    structure: CalibrationStructure, u: np.ndarray | float
) -> np.ndarray:
    """Algorithm 2: exact weighted step-function generalized inverse of Algorithm 1."""
    u_array = np.asarray(u, dtype=float)
    flat_u = u_array.reshape(-1)
    f_values = structure.calibration_cdf_values()
    v_values = 1.0 - f_values
    order = np.argsort(v_values, kind="mergesort")
    sorted_v = v_values[order]
    sorted_weights = structure.weights[order]
    cumulative = np.cumsum(sorted_weights) / structure.weight_sum
    output = np.empty_like(flat_u)
    for index, value in enumerate(flat_u):
        if value < 0.0:
            output[index] = 0.0
        elif value >= 1.0:
            output[index] = 1.0
        else:
            position = int(np.searchsorted(cumulative, value, side="right"))
            position = min(position, len(sorted_v) - 1)
            output[index] = sorted_v[position]
    return output.reshape(u_array.shape)


def select_level_algorithm3(
    structure: CalibrationStructure,
    beta: np.ndarray,
    delta_shift: float,
    epsilon_test: float,
    eta: float = 1e-10,
) -> np.ndarray:
    """Algorithm 3: invert the estimated FCP guarantee to select alpha(beta)."""
    inverse = estimate_g_inverse_algorithm2(structure, np.asarray(beta) - epsilon_test)
    return np.maximum(inverse - delta_shift - eta, 0.0)

