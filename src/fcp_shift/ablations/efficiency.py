from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from fcp_shift.ablations.common import (
    prepare_scored_problem,
    publication_dataset_name,
    scoped_ablation_path,
    set_publication_ticks,
)
from fcp_shift.conformal import CalibrationStructure, select_level_algorithm3
from fcp_shift.conformal.bounds import fixed_constants
from fcp_shift.experiments.common import grid
from fcp_shift.models import candidate_classification_scores
from fcp_shift.reporting import RunDirectory
from fcp_shift.reporting.style import compact_tick_label, figure_size, font_size
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import sample_covariate_shift
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)

WEIGHT_COLORS = {
    "exponential": "#0072B2",
    "quadratic": "#D55E00",
    "mahalanobis": "#009E73",
}


def regression_interval_lengths(
    structure: CalibrationStructure, alpha: np.ndarray
) -> np.ndarray:
    """Lengths of symmetric weighted conformal intervals at each alpha.

    At extremely small alpha, the formal interval can be unbounded because of
    the test-point mass. For a finite, reproducible efficiency curve we use the
    largest observed calibration residual, matching the usual clipped empirical
    quantile implementation.
    """
    alpha = np.asarray(alpha, dtype=float)
    target = (1.0 - alpha) * (structure.weight_sum + structure.w_infinity)
    indices = np.searchsorted(structure.cumulative_weights, target, side="left")
    indices = np.clip(indices, 0, len(structure.scores) - 1)
    lengths = 2.0 * structure.scores[indices]
    return np.where(alpha >= 1.0, 0.0, lengths)


def classification_set_sizes(
    structure: CalibrationStructure,
    candidate_scores: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    p_values = structure.p_values(candidate_scores.reshape(-1)).reshape(
        candidate_scores.shape
    )
    included = p_values[:, :, None] > np.asarray(alpha)[None, None, :]
    return included.sum(axis=1).mean(axis=0)


def _plot(
    summary: pd.DataFrame,
    dataset_configs: list[dict[str, Any]],
    weights: list[str],
    output: Path,
) -> Path:
    by_task = {
        task: [item["name"] for item in dataset_configs if item["task"] == task]
        for task in ("regression", "classification")
    }
    columns = max(len(by_task["regression"]), len(by_task["classification"]))
    if columns == 0:
        raise ValueError("Efficiency ablation requires at least one dataset")
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=figure_size((5 * columns, 7.5)),
        squeeze=False,
        sharex=True,
    )
    for row, task in enumerate(("regression", "classification")):
        datasets = by_task[task]
        for column in range(columns):
            axis = axes[row, column]
            if column >= len(datasets):
                axis.set_visible(False)
                continue
            dataset = datasets[column]
            axis.set_title(publication_dataset_name(dataset))
            dataset_summary = summary[summary["dataset"] == dataset]
            for index, weight in enumerate(weights):
                curve = dataset_summary[
                    dataset_summary["weight"] == weight
                ].sort_values("beta")
                if curve.empty:
                    raise ValueError(f"Missing efficiency curve for {dataset}/{weight}")
                color = WEIGHT_COLORS.get(weight, plt.get_cmap("tab10")(index))
                # axis.fill_between(
                #     curve.beta,
                #     np.maximum(curve["mean"] - curve["std"].fillna(0.0), 0.0),
                #     curve["mean"] + curve["std"].fillna(0.0),
                #     color=color,
                #     alpha=0.10,
                #     linewidth=0,
                # )
                axis.plot(
                    curve.beta,
                    curve["mean"],
                    color=color,
                    linewidth=2,
                    label=weight,
                )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(bottom=0.0)
            set_publication_ticks(axis)
            axis.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, position: ""
                    if np.isclose(value, 0.0)
                    else compact_tick_label(value, position)
                )
            )
            axis.grid(alpha=0.25)
    # axes[0, 0].set_ylabel("Average Size")
    axes[1, 0].set_ylabel("Average Size")
    axes[1, 0].set_xlabel(r"Target FCP bound $\beta$")
    # figure.supxlabel(r"Target FCP bound $\beta$")
    legend_axis = axes[0, len(by_task["regression"]) - 1]
    legend_axis.legend(fontsize=font_size("legend", 8))
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    destination = output / f"efficiency_vs_fcp_2x{columns}.pdf"
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    return destination


def run_efficiency_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    beta = grid(config["fcp"]["beta_grid"])
    n = int(config["sample_sizes"]["n_calibration"])
    m = int(config["sample_sizes"]["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    delta = float(config["fcp"]["delta"])
    w_infinity = float(config["fcp"].get("w_infinity", 1.0))
    eta = float(config["fcp"].get("eta", 1e-10))
    weight_names = [item["name"] for item in config["weights"]]

    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(scoped_ablation_path(root, "efficiency", seed, config))
        if run.complete and not force:
            LOGGER.info("Skipping completed run %s", run.path)
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_efficiency",
                "seed": seed,
                "shift": "covariate_shift",
                "inverse_guarantee": "fixed_beta",
                "level_selection": "algorithm_3",
                "regression_efficiency": "symmetric_interval_length",
                "classification_efficiency": "prediction_set_size",
                "regression_extreme_quantile": "clipped_to_max_calibration_residual",
            },
        )
        records: list[dict[str, Any]] = []
        for dataset_config in config["datasets"]:
            problem = prepare_scored_problem(dataset_config, config["model"])
            task = dataset_config["task"]
            candidate_scores = None
            if task == "classification":
                candidate_scores = candidate_classification_scores(
                    problem.model,
                    problem.dataset.x_source,
                    config["model"].get("classification_score"),
                )
            for weight_config in config["weights"]:
                weight = fit_weight(
                    weight_config,
                    problem.dataset.x_train,
                    problem.dataset.x_source,
                    problem.scores,
                )
                constants = fixed_constants(
                    weight.bound,
                    n,
                    m,
                    delta,
                    w_infinity=w_infinity,
                    optimize=bool(config["fcp"].get("optimize_delta", True)),
                )
                for repetition in range(repetitions):
                    rng = np.random.default_rng(
                        stable_seed(
                            "efficiency",
                            dataset_config["name"],
                            weight.name,
                            seed,
                            repetition,
                        )
                    )
                    calibration, test = sample_covariate_shift(
                        len(problem.scores), weight.values, n, m, rng
                    )
                    structure = CalibrationStructure.build(
                        problem.scores[calibration],
                        weight.values[calibration],
                        w_infinity,
                    )
                    selected_alpha = select_level_algorithm3(
                        structure,
                        beta,
                        constants.delta_shift,
                        constants.epsilon_test,
                        eta,
                    )
                    if task == "regression":
                        efficiency = regression_interval_lengths(
                            structure, selected_alpha
                        )
                    else:
                        efficiency = classification_set_sizes(
                            structure, candidate_scores[test], selected_alpha
                        )
                    records.extend(
                        {
                            "dataset": dataset_config["name"],
                            "task": task,
                            "weight": weight.name,
                            "repetition": repetition,
                            "beta": float(beta[index]),
                            "selected_alpha": float(selected_alpha[index]),
                            "efficiency": float(efficiency[index]),
                        }
                        for index in range(len(beta))
                    )

        metrics = pd.DataFrame(records)
        summary = (
            metrics.groupby(["dataset", "task", "weight", "beta"], as_index=False)
            .agg(
                mean=("efficiency", "mean"),
                std=("efficiency", "std"),
                selected_alpha_mean=("selected_alpha", "mean"),
                selected_alpha_std=("selected_alpha", "std"),
            )
        )
        run.save_metrics(metrics)
        summary.to_csv(run.path / "efficiency_curves_summary.csv", index=False)
        run.save_summary(
            {
                "rows": len(metrics),
                "datasets": [item["name"] for item in config["datasets"]],
                "weights": weight_names,
            }
        )
        _plot(summary, config["datasets"], weight_names, run.path)
        run.mark_complete()
