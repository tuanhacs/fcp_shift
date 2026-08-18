from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class FittedWeight:
    name: str
    values: np.ndarray
    bound: float
    metadata: dict[str, Any]


def _standardized_projection(
    x_reference: np.ndarray,
    x_source: np.ndarray,
    scores: np.ndarray,
    ridge: float,
) -> np.ndarray:
    mean = np.mean(x_reference, axis=0)
    scale = np.std(x_reference, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    reference = (x_reference - mean) / scale
    source = (x_source - mean) / scale
    direction = Ridge(alpha=ridge, fit_intercept=True).fit(source, scores).coef_
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-12:
        direction = np.ones(source.shape[1], dtype=float)
        norm = np.linalg.norm(direction)
    direction = direction / norm
    return source @ direction


def fit_weight(
    config: dict[str, Any],
    x_reference: np.ndarray,
    x_source: np.ndarray,
    scores: np.ndarray,
) -> FittedWeight:
    name = config["name"]
    epsilon = float(config.get("epsilon", 1e-8))
    strength = float(config.get("strength", 0.35))
    projection = _standardized_projection(
        x_reference, x_source, scores, float(config.get("ridge", 1e-3))
    )
    projection = (projection - np.mean(projection)) / (np.std(projection) + 1e-12)
    if name == "exponential":
        raw = np.exp(np.clip(strength * projection, -30.0, 30.0))
    elif name == "quadratic":
        raw = epsilon + 1.0 + strength * projection**2
    elif name == "mahalanobis":
        mean = np.mean(x_reference, axis=0)
        scale = np.std(x_reference, axis=0)
        standardized = (x_source - mean) / np.where(scale > 1e-8, scale, 1.0)
        raw = epsilon + 1.0 + strength * np.mean(standardized**2, axis=1)
    else:
        raise ValueError(f"Unsupported weight: {name}")

    clip_quantile = float(config.get("clip_quantile", 0.995))
    if not 0.0 < clip_quantile <= 1.0:
        raise ValueError("weight.clip_quantile must lie in (0, 1]")
    raw = np.clip(raw, epsilon, np.quantile(raw, clip_quantile))
    values = raw / np.mean(raw)
    correlation = float(np.corrcoef(values, scores)[0, 1])
    return FittedWeight(
        name=name,
        values=values,
        bound=float(np.max(values)),
        metadata={
            "name": name,
            "strength": strength,
            "clip_quantile": clip_quantile,
            "bound": float(np.max(values)),
            "mean": float(np.mean(values)),
            "effective_sample_size": float(np.sum(values) ** 2 / np.sum(values**2)),
            "correlation_weight_score": correlation,
        },
    )
