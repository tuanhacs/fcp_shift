from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fcp_shift.conformal import (
    CalibrationStructure,
    estimate_g_algorithm1,
    estimate_g_inverse_algorithm2,
)
from fcp_shift.ablations.common import scoped_ablation_path
from fcp_shift.conformal.bounds import fixed_constants, uniform_constants
from fcp_shift.data import prepare_dataset
from fcp_shift.experiments.common import grid
from fcp_shift.models import conformity_scores, fit_model
from fcp_shift.reporting import RunDirectory
from fcp_shift.reporting.style import figure_size, font_size
from fcp_shift.reproducibility import stable_seed
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def _plot(frame: pd.DataFrame, datasets: list[str], models: list[str], path: Path) -> None:
    figure, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=figure_size((5 * len(models), 4 * len(datasets))),
        squeeze=False,
    )
    for row, dataset in enumerate(datasets):
        for column, model in enumerate(models):
            axis = axes[row, column]
            subset = frame[(frame.dataset == dataset) & (frame.model == model)]
            for family, color in [("goals_1_2", "#0072B2"), ("goals_3_4", "#D55E00")]:
                group = subset[subset.family == family]
                summary = group.groupby("n").seconds.agg(["median", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]).reset_index()
                summary.columns = ["n", "median", "q10", "q90"]
                axis.plot(summary.n, summary["median"], marker="o", color=color, label=family)
                axis.fill_between(summary.n, summary.q10, summary.q90, color=color, alpha=0.12)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(f"{dataset} — {model}")
            axis.set_xlabel(r"Calibration size $n$")
            axis.set_ylabel("Wall time (seconds)")
            axis.grid(alpha=0.25)
            if row == 0 and column == len(models) - 1:
                axis.legend(fontsize=font_size("legend", 8))
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run_timing_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    alpha, beta = grid(config["fcp"]["alpha_grid"]), grid(config["fcp"]["beta_grid"])
    n_grid = [int(value) for value in config["ablation"]["n_grid"]]
    m = int(config["sample_sizes"]["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    model_configs = config["models"]
    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(scoped_ablation_path(root, "timing", seed, config))
        if run.complete and not force:
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_timing", "seed": seed,
                "timing_scope": "calibration score computation plus FCP algorithm; model fitting excluded",
            },
        )
        rows = []
        for dataset_config in config["datasets"]:
            dataset = prepare_dataset(dataset_config, int(model_configs[0].get("seed", 2026)))
            models = {}
            full_scores = {}
            for model_config in model_configs:
                name = model_config["name"]
                models[name] = fit_model(
                    dataset.task, dataset.x_train, dataset.y_train, model_config,
                    int(model_config.get("seed", 2026)),
                )
                full_scores[name] = conformity_scores(
                    models[name], dataset.x_source, dataset.y_source, dataset.task,
                    model_config.get("classification_score", "log_margin"),
                )
            reference = full_scores[model_configs[0]["name"]]
            weight = fit_weight(config["weights"][0], dataset.x_train, dataset.x_source, reference)
            for model_config in model_configs:
                model_name = model_config["name"]
                for n in n_grid:
                    for repetition in range(repetitions + 1):
                        rng = np.random.default_rng(
                            stable_seed("timing", dataset.name, model_name, n, seed, repetition)
                        )
                        indices = rng.choice(len(reference), size=n, replace=True)

                        start = time.perf_counter()
                        scores = conformity_scores(
                            models[model_name], dataset.x_source[indices], dataset.y_source[indices],
                            dataset.task, model_config.get("classification_score", "log_margin"),
                        )
                        structure = CalibrationStructure.build(scores, weight.values[indices])
                        fixed = fixed_constants(weight.bound, n, m, float(config["fcp"]["delta"]))
                        uniform = uniform_constants(weight.bound, n, m, float(config["fcp"]["delta"]))
                        estimate_g_algorithm1(structure, alpha + fixed.delta_shift)
                        estimate_g_algorithm1(structure, alpha + uniform.delta_shift)
                        forward_seconds = time.perf_counter() - start

                        start = time.perf_counter()
                        scores = conformity_scores(
                            models[model_name], dataset.x_source[indices], dataset.y_source[indices],
                            dataset.task, model_config.get("classification_score", "log_margin"),
                        )
                        structure = CalibrationStructure.build(scores, weight.values[indices])
                        fixed = fixed_constants(weight.bound, n, m, float(config["fcp"]["delta"]))
                        uniform = uniform_constants(weight.bound, n, m, float(config["fcp"]["delta"]))
                        estimate_g_inverse_algorithm2(structure, beta - fixed.epsilon_test)
                        estimate_g_inverse_algorithm2(structure, beta - uniform.epsilon_test)
                        inverse_seconds = time.perf_counter() - start
                        if repetition > 0:
                            rows.extend(
                                [
                                    {"dataset": dataset.name, "model": model_name, "n": n, "repetition": repetition - 1, "family": "goals_1_2", "seconds": forward_seconds},
                                    {"dataset": dataset.name, "model": model_name, "n": n, "repetition": repetition - 1, "family": "goals_3_4", "seconds": inverse_seconds},
                                ]
                            )
        frame = pd.DataFrame(rows)
        run.save_metrics(frame)
        run.save_summary({"rows": len(frame), "median_seconds": float(frame.seconds.median())})
        _plot(
            frame,
            [item["name"] for item in config["datasets"]],
            [item["name"] for item in model_configs],
            run.path / "inference_time_3x3.pdf",
        )
        run.mark_complete()
