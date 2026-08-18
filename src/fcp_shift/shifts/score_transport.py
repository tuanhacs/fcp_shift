from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreTransport:
    source_scores: np.ndarray
    strata: np.ndarray
    p: np.ndarray
    q: np.ndarray
    permutation: np.ndarray

    def weights(self, rho: float) -> np.ndarray:
        inverse = np.argsort(self.permutation)
        transported_q = (1.0 - rho) * self.q + rho * self.q[inverse]
        values = transported_q[self.strata] / self.p[self.strata]
        return values / np.mean(values)

    def sample_test_scores(
        self, m: int, rho: float, rng: np.random.Generator
    ) -> np.ndarray:
        target_strata = rng.choice(len(self.p), size=m, replace=True, p=self.q)
        transported = rng.random(m) < rho
        donor_strata = np.where(
            transported, self.permutation[target_strata], target_strata
        )
        result = np.empty(m, dtype=float)
        for stratum in range(len(self.p)):
            positions = np.flatnonzero(donor_strata == stratum)
            candidates = np.flatnonzero(self.strata == stratum)
            result[positions] = self.source_scores[
                rng.choice(candidates, size=len(positions), replace=True)
            ]
        return result


def build_score_transport(
    source_scores: np.ndarray, base_weights: np.ndarray, strata_count: int
) -> ScoreTransport:
    source_scores = np.asarray(source_scores, dtype=float)
    base_weights = np.asarray(base_weights, dtype=float)
    if strata_count < 2:
        raise ValueError("transport.strata must be at least 2")
    quantiles = np.quantile(source_scores, np.linspace(0.0, 1.0, strata_count + 1))
    internal = np.unique(quantiles[1:-1])
    strata = np.digitize(source_scores, internal, right=True)
    actual_count = int(np.max(strata)) + 1
    if actual_count < 2:
        raise ValueError("Scores do not support at least two non-empty strata")
    p = np.bincount(strata, minlength=actual_count).astype(float)
    p /= p.sum()
    q = np.bincount(strata, weights=base_weights, minlength=actual_count).astype(float)
    q /= q.sum()
    permutation = np.roll(np.arange(actual_count), -1)
    return ScoreTransport(source_scores, strata, p, q, permutation)

