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


@dataclass(frozen=True)
class ScoreProjection:
    """A score-informed direction fitted only on independent auxiliary data."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    direction: np.ndarray
    projection_mean: float
    projection_scale: float

    def transform(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features) - self.feature_mean) / self.feature_scale
        return (
            standardized @ self.direction - self.projection_mean
        ) / self.projection_scale


def fit_score_projection(
    x_reference: np.ndarray,
    x_auxiliary: np.ndarray,
    auxiliary_scores: np.ndarray,
    ridge: float = 1e-3,
    *,
    method: str = "ridge",
    random_seed: int = 2026,
) -> ScoreProjection:
    """Freeze a score-informed Ridge or seeded random direction X -> z(X)."""
    mean = np.mean(x_reference, axis=0)
    scale = np.std(x_reference, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    auxiliary = (x_auxiliary - mean) / scale
    if method == "ridge":
        direction = Ridge(alpha=ridge, fit_intercept=True).fit(
            auxiliary, auxiliary_scores
        ).coef_
    elif method == "random":
        direction = np.random.default_rng(random_seed).normal(size=auxiliary.shape[1])
    else:
        raise ValueError(f"Unsupported weight direction: {method}")
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-12:
        direction = np.ones(auxiliary.shape[1], dtype=float)
        norm = np.linalg.norm(direction)
    direction = direction / norm
    projected = auxiliary @ direction
    return ScoreProjection(
        feature_mean=mean,
        feature_scale=scale,
        direction=direction,
        projection_mean=float(np.mean(projected)),
        projection_scale=float(np.std(projected) + 1e-12),
    )


def _standardized_projection(
    x_reference: np.ndarray,
    x_source: np.ndarray,
    scores: np.ndarray,
    ridge: float,
    method: str,
    random_seed: int,
) -> np.ndarray:
    mean = np.mean(x_reference, axis=0)
    scale = np.std(x_reference, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    reference = (x_reference - mean) / scale
    source = (x_source - mean) / scale
    if method == "ridge":
        direction = Ridge(alpha=ridge, fit_intercept=True).fit(source, scores).coef_
    elif method == "random":
        direction = np.random.default_rng(random_seed).normal(size=source.shape[1])
    else:
        raise ValueError(f"Unsupported weight direction: {method}")
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
    score_projection: ScoreProjection | None = None,
) -> FittedWeight:
    name = config["name"]
    epsilon = float(config.get("epsilon", 1e-8))
    strength = float(config.get("strength", 0.35))
    direction_method = str(config.get("direction", "ridge"))
    direction_seed = int(config.get("direction_seed", 2026))
    if score_projection is None:
        projection = _standardized_projection(
            x_reference, x_source, scores, float(config.get("ridge", 1e-3)),
            direction_method, direction_seed,
        )
        projection = (projection - np.mean(projection)) / (np.std(projection) + 1e-12)
    else:
        projection = score_projection.transform(x_source)
    if name == "exponential":
        raw = np.exp(np.clip(strength * projection, -30.0, 30.0))
    elif name == "quadratic":
        raw = epsilon + 1.0 + strength * projection**2
    elif name == "logarithmic":
        # log(1 + z^2), evaluated without squaring very large projections.
        raw = epsilon + 1.0 + strength * (2.0 * np.log(np.hypot(1.0, projection)))
    elif name == "linear":
        raw = np.maximum(1.0 + strength * projection, epsilon)
    elif name == "sigmoid":
        scaled = np.clip(strength * projection, -30.0, 30.0)
        raw = epsilon + 2.0 / (1.0 + np.exp(-scaled))
    elif name == "arctangent":
        if strength < 0.0:
            raise ValueError("arctangent weight strength must be nonnegative")
        raw = 1.0 + strength * (0.5 + np.arctan(projection) / np.pi)
    elif name == "power_tilt":
        if strength < 0.0:
            raise ValueError("power_tilt weight strength must be nonnegative")
        log_raw = strength * np.sign(projection) * np.log1p(np.abs(projection))
        raw = np.exp(np.clip(log_raw, -30.0, 30.0))
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
    correlation = (
        float(np.corrcoef(values, scores)[0, 1])
        if np.std(values) > 1e-14 and np.std(scores) > 1e-14
        else 0.0
    )
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
            "projection_source": (
                "auxiliary_normalization" if direction_method == "random" and score_projection is not None
                else "random_direction" if direction_method == "random"
                else "independent_auxiliary" if score_projection is not None
                else "source_scores"
            ),
            "direction": direction_method,
            "direction_seed": direction_seed if direction_method == "random" else None,
        },
    )


def direction_variant(config: dict[str, Any]) -> str | None:
    """Output subdirectory for non-default directions; preserve existing Ridge paths."""
    if config.get("direction", "ridge") == "random":
        return f"direction_random_seed_{int(config.get('direction_seed', 2026))}"
    return None
