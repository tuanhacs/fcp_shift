from __future__ import annotations

import numpy as np


def simulate_heteroscedastic_regression(
    n: int,
    rng: np.random.Generator,
    dimension: int = 4,
    coefficients: list[float] | None = None,
    heteroscedastic_scale: float = 0.75,
    minimum_noise: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients_array = np.asarray(
        coefficients if coefficients is not None else [2.0, 1.0] + [0.0] * (dimension - 2),
        dtype=float,
    )
    if coefficients_array.size != dimension:
        raise ValueError("simulation.coefficients must match simulation.dimension")
    features = rng.normal(size=(n, dimension))
    direction = np.ones(dimension, dtype=float) / np.sqrt(dimension)
    scale = minimum_noise + np.exp(
        heteroscedastic_scale * (features @ direction)
    )
    response = features @ coefficients_array + rng.normal(size=n) * scale
    return features.astype(np.float32), response.astype(float)

