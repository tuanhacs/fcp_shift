from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fcp_shift.data import PreparedDataset, prepare_dataset
from fcp_shift.models import conformity_scores, fit_model
from fcp_shift.reproducibility import stable_seed


@dataclass(frozen=True)
class ShiftProblem:
    dataset: PreparedDataset
    source_scores: np.ndarray
    auxiliary_features: np.ndarray
    auxiliary_scores: np.ndarray


def prepare_shift_problem(
    dataset_config: dict[str, Any],
    model_config: dict[str, Any],
    auxiliary_fraction: float = 0.2,
) -> ShiftProblem:
    """Hold out auxiliary pairs before forming the calibration/test source pool."""
    if not 0.0 < auxiliary_fraction < 1.0:
        raise ValueError("shift.auxiliary_fraction must lie in (0, 1)")
    model_seed = int(model_config.get("seed", 2026))
    dataset = prepare_dataset(dataset_config, model_seed)
    source_size = len(dataset.x_source)
    auxiliary_size = round(auxiliary_fraction * source_size)
    if not 2 <= auxiliary_size <= source_size - 2:
        raise ValueError("Auxiliary and source pools must each contain at least two points")
    rng = np.random.default_rng(
        stable_seed("shift_auxiliary", dataset.name, model_seed)
    )
    order = rng.permutation(source_size)
    auxiliary_indices = order[:auxiliary_size]
    source_indices = order[auxiliary_size:]
    auxiliary_features = dataset.x_source[auxiliary_indices]
    auxiliary_targets = dataset.y_source[auxiliary_indices]
    dataset = replace(
        dataset,
        x_source=dataset.x_source[source_indices],
        y_source=dataset.y_source[source_indices],
    )
    model = fit_model(
        dataset.task, dataset.x_train, dataset.y_train, model_config, model_seed
    )
    score_name = model_config.get("classification_score")
    auxiliary_scores = conformity_scores(
        model, auxiliary_features, auxiliary_targets, dataset.task, score_name
    )
    source_scores = conformity_scores(
        model, dataset.x_source, dataset.y_source, dataset.task, score_name
    )
    return ShiftProblem(dataset, source_scores, auxiliary_features, auxiliary_scores)
