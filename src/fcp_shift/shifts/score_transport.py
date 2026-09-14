from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreTransport:
    source_scores: np.ndarray
    strata: np.ndarray
    cutpoints: np.ndarray
    p: np.ndarray
    q: np.ndarray
    permutation: np.ndarray

    def weights(self, rho: float) -> np.ndarray:
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        inverse = np.argsort(self.permutation)
        transported_q = (1.0 - rho) * self.q + rho * self.q[inverse]
        values = transported_q[self.strata] / self.p[self.strata]
        return values / np.mean(values)

    def sample_test(
        self, m: int, rho: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw target X indices, retaining their score unless transport is applied."""
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        target_strata = rng.choice(len(self.p), size=m, replace=True, p=self.q)
        transported = rng.random(m) < rho
        donor_strata = np.where(
            transported, self.permutation[target_strata], target_strata
        )
        target_indices = np.empty(m, dtype=int)
        donor_indices = np.empty(m, dtype=int)
        for stratum in range(len(self.p)):
            candidates = np.flatnonzero(self.strata == stratum)
            target_positions = np.flatnonzero(target_strata == stratum)
            donor_positions = np.flatnonzero(donor_strata == stratum)
            target_indices[target_positions] = rng.choice(
                candidates, size=len(target_positions), replace=True
            )
            donor_indices[donor_positions] = rng.choice(
                candidates, size=len(donor_positions), replace=True
            )
        test_scores = self.source_scores[target_indices].copy()
        test_scores[transported] = self.source_scores[donor_indices[transported]]
        return target_indices, test_scores

    def sample_test_scores(
        self, m: int, rho: float, rng: np.random.Generator
    ) -> np.ndarray:
        return self.sample_test(m, rho, rng)[1]


def build_score_transport(
    source_scores: np.ndarray,
    base_weights: np.ndarray,
    strata_count: int,
    *,
    stratification_values: np.ndarray,
    reference_values: np.ndarray,
) -> ScoreTransport:
    source_scores = np.asarray(source_scores, dtype=float)
    base_weights = np.asarray(base_weights, dtype=float)
    stratification_values = np.asarray(stratification_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    if strata_count < 2:
        raise ValueError("transport.strata must be at least 2")
    if (
        source_scores.ndim != 1
        or source_scores.shape != base_weights.shape
        or source_scores.shape != stratification_values.shape
        or reference_values.ndim != 1
        or len(reference_values) == 0
        or not np.all(np.isfinite(source_scores))
        or not np.all(np.isfinite(base_weights))
        or not np.all(np.isfinite(stratification_values))
        or not np.all(np.isfinite(reference_values))
        or np.any(base_weights <= 0.0)
    ):
        raise ValueError("Scores, positive weights, and feature-based strata must be valid")
    # Thresholds are learned from independent auxiliary X, never from source scores.
    quantiles = np.quantile(reference_values, np.linspace(0.0, 1.0, strata_count + 1))
    internal = np.unique(quantiles[1:-1])
    if len(internal) != strata_count - 1:
        raise ValueError("Auxiliary projection does not support the requested strata count")
    strata = np.digitize(stratification_values, internal, right=True)
    p = np.bincount(strata, minlength=strata_count).astype(float)
    if np.any(p == 0.0):
        raise ValueError("At least one feature-defined stratum is empty in the source pool")
    p /= p.sum()
    q = np.bincount(strata, weights=base_weights, minlength=strata_count).astype(float)
    q /= q.sum()
    permutation = np.roll(np.arange(strata_count), -1)
    return ScoreTransport(source_scores, strata, internal, p, q, permutation)
