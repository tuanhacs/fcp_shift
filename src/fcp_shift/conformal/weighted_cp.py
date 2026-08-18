from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationStructure:
    scores: np.ndarray
    weights: np.ndarray
    cumulative_weights: np.ndarray
    weight_sum: float
    w_infinity: float = 1.0

    @classmethod
    def build(
        cls, scores: np.ndarray, weights: np.ndarray, w_infinity: float = 1.0
    ) -> "CalibrationStructure":
        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if scores.ndim != 1 or scores.shape != weights.shape:
            raise ValueError("Calibration scores and weights must be matching vectors")
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("Calibration weights must be finite and positive")
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_weights = weights[order]
        return cls(
            scores=sorted_scores,
            weights=sorted_weights,
            cumulative_weights=np.cumsum(sorted_weights),
            weight_sum=float(np.sum(sorted_weights)),
            w_infinity=float(w_infinity),
        )

    def empirical_cdf(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        right = np.searchsorted(self.scores, values, side="right")
        cumulative = np.concatenate(([0.0], self.cumulative_weights))
        return cumulative[right] / (self.weight_sum + self.w_infinity)

    def calibration_cdf_values(self) -> np.ndarray:
        right = np.searchsorted(self.scores, self.scores, side="right")
        cumulative = np.concatenate(([0.0], self.cumulative_weights))
        return cumulative[right] / (self.weight_sum + self.w_infinity)

    def p_values(self, test_scores: np.ndarray) -> np.ndarray:
        test_scores = np.asarray(test_scores, dtype=float)
        left = np.searchsorted(self.scores, test_scores, side="left")
        cumulative = np.concatenate(([0.0], self.cumulative_weights))
        tail_weight = self.weight_sum - cumulative[left]
        return (self.w_infinity + tail_weight) / (
            self.weight_sum + self.w_infinity
        )


def fcp_curve(p_values: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
    return np.mean(
        np.asarray(p_values, dtype=float)[:, None]
        <= np.asarray(alpha_grid, dtype=float)[None, :],
        axis=0,
    )


def fcp_at_levels(p_values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.mean(
        np.asarray(p_values, dtype=float)[:, None]
        <= np.asarray(levels, dtype=float)[None, :],
        axis=0,
    )

