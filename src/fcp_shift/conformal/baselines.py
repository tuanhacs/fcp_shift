from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sanssouci as sa


def unweighted_p_values(test_scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration_scores, dtype=float))
    less = np.searchsorted(calibration, np.asarray(test_scores, dtype=float), side="left")
    return (1.0 + len(calibration) - less) / (len(calibration) + 1.0)


def dkw_lambda(delta: float, n: int, m: int, iterations: int = 8) -> float:
    tau = n * m / (n + m)
    first = np.log(1.0 / delta)
    second = 2.0 * np.sqrt(2.0 * np.pi) * tau / np.sqrt(n + m)
    value = 1.0
    for _ in range(iterations + 1):
        value = min(
            1.0,
            float(np.sqrt((first + np.log(1.0 + second * value)) / (2.0 * tau))),
        )
    return value


def dkw_forward(alpha: np.ndarray, n: int, m: int, delta: float) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    value = np.zeros_like(alpha)
    active = alpha >= 1.0 / (n + 1.0)
    value[active] = np.minimum(1.0, alpha[active] + dkw_lambda(delta, n, m))
    return value


def dkw_inverse(beta: np.ndarray, n: int, m: int, delta: float) -> np.ndarray:
    return np.clip(np.asarray(beta, dtype=float) - dkw_lambda(delta, n, m), 0.0, 1.0)


def simulate_null_ordered_p_values(
    simulations: int, n: int, m: int, rng: np.random.Generator
) -> np.ndarray:
    output = np.empty((simulations, m), dtype=float)
    support = (1.0 + np.arange(n + 1)) / (n + 1.0)
    concentration = np.ones(n + 1)
    for index in range(simulations):
        cdf = np.cumsum(rng.dirichlet(concentration))
        ranks = np.searchsorted(cdf, rng.random(m), side="right")
        output[index] = support[ranks]
    output.sort(axis=1)
    return output


@dataclass(frozen=True)
class CoJER:
    thresholds: np.ndarray

    def forward(self, alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        m = len(self.thresholds)
        positions = np.searchsorted(self.thresholds, alpha, side="left")
        return np.where(positions >= m, 1.0, positions / m)

    def inverse(self, beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=float)
        m = len(self.thresholds)
        positions = np.floor(beta * m).astype(int)
        output = np.zeros_like(beta)
        active = positions >= 1
        output[active] = self.thresholds[np.minimum(positions[active], m - 1)]
        output[beta >= 1.0] = self.thresholds[-1]
        return output


def calibrate_cojer(
    n: int,
    m: int,
    delta: float,
    template_simulations: int,
    calibration_simulations: int,
    seed: int,
    k_max: int | None = None,
) -> CoJER:
    template_raw = simulate_null_ordered_p_values(
        template_simulations, n, m, np.random.default_rng(seed)
    )
    template = np.sort(template_raw, axis=0)
    raw = simulate_null_ordered_p_values(
        calibration_simulations, n, m, np.random.default_rng(seed + 1)
    )
    thresholds = sa.calibrate_jer(
        delta, template, raw, k_max=m if k_max is None else int(k_max)
    )
    return CoJER(np.asarray(thresholds, dtype=float))
