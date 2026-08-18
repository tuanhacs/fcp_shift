from __future__ import annotations

import numpy as np


def sample_covariate_shift(
    source_size: int,
    source_weights: np.ndarray,
    n_calibration: int,
    m_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    calibration = rng.choice(source_size, size=n_calibration, replace=True)
    probabilities = np.asarray(source_weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    test = rng.choice(source_size, size=m_test, replace=True, p=probabilities)
    return calibration, test

