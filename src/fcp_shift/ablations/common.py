from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np

from fcp_shift.data import PreparedDataset, prepare_dataset
from fcp_shift.models import conformity_scores, fit_model
from fcp_shift.reporting.labels import display_dataset_name
from fcp_shift.reporting.style import set_publication_ticks


@dataclass
class ScoredProblem:
    dataset: PreparedDataset
    model: Any
    scores: np.ndarray


def publication_dataset_name(name: str) -> str:
    """Return the paper-facing name used in figure titles and row labels."""
    return display_dataset_name(name)


def publication_model_name(name: str) -> str:
    return {
        "hist_gradient_boosting": "HistGradientBoosting",
        "random_forest": "Random Forest",
        "linear": "Linear / Ridge",
        "logistic": "Logistic Regression",
        "mlp": "MLP",
    }.get(name, name.replace("_", " ").title())


def add_dataset_row_labels(axes: np.ndarray, datasets: list[str]) -> None:
    """Label each dataset row once, outside the first column."""
    for row, dataset in enumerate(datasets):
        axes[row, 0].annotate(
            publication_dataset_name(dataset),
            xy=(-0.23, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )


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
