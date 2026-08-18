from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class FixedConstants:
    delta_calibration: float
    delta_test: float
    delta_shift: float
    epsilon_test: float

    @property
    def penalty(self) -> float:
        return self.delta_shift + self.epsilon_test

    def as_dict(self):
        return {**asdict(self), "penalty": self.penalty}


@dataclass(frozen=True)
class UniformConstants:
    delta_plus: float
    delta_minus: float
    delta_test: float
    r_plus: float
    r_minus: float
    delta_shift: float
    epsilon_test: float
    trivial: bool

    @property
    def penalty(self) -> float:
        return self.delta_shift + self.epsilon_test

    def as_dict(self):
        return {**asdict(self), "penalty": self.penalty}


def _fixed_from_allocation(
    bound: float,
    n: int,
    m: int,
    delta_calibration: float,
    delta_test: float,
    mean_weight: float,
    w_infinity: float,
) -> FixedConstants:
    denominator = mean_weight + w_infinity / n
    delta_shift = bound / denominator * np.sqrt(
        np.log(1.0 / delta_calibration) / (2.0 * n)
    )
    epsilon_test = np.sqrt(np.log(1.0 / delta_test) / (2.0 * m))
    return FixedConstants(
        delta_calibration=float(delta_calibration),
        delta_test=float(delta_test),
        delta_shift=float(delta_shift),
        epsilon_test=float(epsilon_test),
    )


def fixed_constants(
    bound: float,
    n: int,
    m: int,
    delta: float,
    mean_weight: float = 1.0,
    w_infinity: float = 1.0,
    optimize: bool = True,
) -> FixedConstants:
    if optimize:
        objective = lambda fraction: _fixed_from_allocation(
            bound,
            n,
            m,
            fraction * delta,
            (1.0 - fraction) * delta,
            mean_weight,
            w_infinity,
        ).penalty
        result = minimize_scalar(objective, bounds=(1e-6, 1.0 - 1e-6), method="bounded")
        fraction = float(result.x)
    else:
        fraction = 0.5
    return _fixed_from_allocation(
        bound,
        n,
        m,
        fraction * delta,
        (1.0 - fraction) * delta,
        mean_weight,
        w_infinity,
    )


def _uniform_from_fraction(
    bound: float,
    n: int,
    m: int,
    delta: float,
    fraction: float,
    mean_weight: float,
    w_infinity: float,
) -> UniformConstants:
    delta_plus = fraction * delta
    delta_minus = fraction * delta
    delta_test = (1.0 - 2.0 * fraction) * delta
    denominator = mean_weight + w_infinity / n
    r_plus = bound / np.sqrt(2.0 * n) * (
        np.sqrt(np.pi / 4.0) + np.sqrt(np.log(1.0 / delta_plus))
    )
    r_minus = bound / np.sqrt(2.0 * n) * np.sqrt(
        np.log(1.0 / delta_minus)
    )
    trivial = bool(r_minus >= denominator)
    if trivial:
        delta_shift = 1.0
    else:
        delta_shift = (
            denominator * r_plus + mean_weight * r_minus
        ) / (denominator * (denominator - r_minus))
    epsilon_test = np.sqrt(np.log(1.0 / delta_test) / (2.0 * m))
    return UniformConstants(
        delta_plus=float(delta_plus),
        delta_minus=float(delta_minus),
        delta_test=float(delta_test),
        r_plus=float(r_plus),
        r_minus=float(r_minus),
        delta_shift=float(delta_shift),
        epsilon_test=float(epsilon_test),
        trivial=trivial,
    )


def uniform_constants(
    bound: float,
    n: int,
    m: int,
    delta: float,
    mean_weight: float = 1.0,
    w_infinity: float = 1.0,
    optimize: bool = True,
) -> UniformConstants:
    if optimize:
        objective = lambda fraction: _uniform_from_fraction(
            bound, n, m, delta, fraction, mean_weight, w_infinity
        ).penalty
        result = minimize_scalar(objective, bounds=(1e-6, 0.5 - 1e-6), method="bounded")
        fraction = float(result.x)
    else:
        fraction = 1.0 / 3.0
    return _uniform_from_fraction(
        bound, n, m, delta, fraction, mean_weight, w_infinity
    )

