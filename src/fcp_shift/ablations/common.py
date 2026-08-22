from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np

from fcp_shift.data import PreparedDataset, prepare_dataset
from fcp_shift.models import conformity_scores, fit_model


@dataclass
class ScoredProblem:
    dataset: PreparedDataset
    model: Any
    scores: np.ndarray


def prepare_scored_problem(
    dataset_config: dict[str, Any], model_config: dict[str, Any]
) -> ScoredProblem:
    seed = int(model_config.get("seed", 2026))
    dataset = prepare_dataset(dataset_config, seed)
    model = fit_model(dataset.task, dataset.x_train, dataset.y_train, model_config, seed)
    scores = conformity_scores(
        model,
        dataset.x_source,
        dataset.y_source,
        dataset.task,
        model_config.get("classification_score"),
    )
    return ScoredProblem(dataset, model, scores)


def mean_and_quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    return (
        np.mean(values, axis=0),
        np.quantile(values, 0.1, axis=0),
        np.quantile(values, 0.9, axis=0),
    )


def scoped_ablation_path(
    root: Path, name: str, seed: int, config: dict[str, Any], *parts: str
) -> Path:
    path = root / "ablations" / name
    for part in parts:
        path /= part
    filters = config.get("_filters", {})
    if filters:
        safe = "__".join(
            f"{key}_{str(value).replace('.', 'p')}" for key, value in sorted(filters.items())
        )
        path /= f"scope_{safe}"
    return path / f"seed_{seed}"
