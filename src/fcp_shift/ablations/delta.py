from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fcp_shift.ablations.common import prepare_scored_problem, scoped_ablation_path
from fcp_shift.conformal import CalibrationStructure, estimate_g_algorithm1
from fcp_shift.conformal.bounds import fixed_constants, uniform_constants
from fcp_shift.reporting import RunDirectory
from fcp_shift.reproducibility import stable_seed
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def _plot(
    frame: pd.DataFrame, datasets: list[str], bound_type: str, output_path: Path
) -> None:
    paths = ["n_increases", "m_increases", "n_equals_m"]
    titles = {
        "n_increases": r"$n\to\infty$ ($m$ fixed)",
        "m_increases": r"$m\to\infty$ ($n$ fixed)",
        "n_equals_m": r"$n=m\to\infty$",
    }
    figure, axes = plt.subplots(len(datasets), 3, figsize=(15, 4 * len(datasets)), squeeze=False)
    palette = plt.get_cmap("tab10")
    for row, dataset in enumerate(datasets):
        for column, path_name in enumerate(paths):
            axis = axes[row, column]
            subset = frame[
                (frame.dataset == dataset)
                & (frame.path == path_name)
                & (frame.bound_type == bound_type)
            ]
            labels = list(dict.fromkeys(subset.allocation.tolist()))
            for index, label in enumerate(labels):
                group = subset[subset.allocation == label]
                summary = group.groupby("x").bound.agg(["mean", "std"]).reset_index()
                optimized = label == "optimized"
                color = "black" if optimized else palette(index)
                axis.plot(
                    summary.x,
                    summary["mean"],
                    marker="o",
                    linewidth=2.5 if optimized else 1.8,
                    color=color,
                    label=label,
                )
                axis.fill_between(
                    summary.x,
                    np.maximum(summary["mean"] - summary["std"].fillna(0.0), 0.0),
                    summary["mean"] + summary["std"].fillna(0.0),
                    color=color,
                    alpha=0.10,
                )
            axis.set_xscale("log")
            axis.set_title(f"{dataset}: {titles[path_name]}")
            axis.set_xlabel("Increasing sample size")
            axis.set_ylabel(r"Estimated $G(\alpha+\Delta)+\epsilon$")
            axis.grid(alpha=0.25)
            if row == 0 and column == 2:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def run_delta_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    ablation = config["ablation"]
    alpha = float(ablation.get("alpha", 0.1))
    sizes = [int(value) for value in ablation["grid"]]
    fixed_n, fixed_m = int(ablation["fixed_n"]), int(ablation["fixed_m"])
    repetitions = int(config["experiment"]["repetitions"])
    delta = float(config["fcp"]["delta"])
    w_infinity = float(config["fcp"].get("w_infinity", 1.0))
    fixed_fractions = [float(value) for value in ablation["fixed_calibration_fractions"]]
    uniform_fractions = [float(value) for value in ablation["uniform_side_fractions"]]
    paths = {
        "n_increases": [(value, fixed_m, value) for value in sizes],
        "m_increases": [(fixed_n, value, value) for value in sizes],
        "n_equals_m": [(value, value, value) for value in sizes],
    }
    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(scoped_ablation_path(root, "delta", seed, config))
        if run.complete and not force:
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_delta",
                "seed": seed,
                "g_estimator": "algorithm_1_in_paper",
                "notebook_alias": "ghat_alg2",
            },
        )
        records = []
        for dataset_config in config["datasets"]:
            problem = prepare_scored_problem(dataset_config, config["model"])
            weight = fit_weight(
                config["weights"][0],
                problem.dataset.x_train,
                problem.dataset.x_source,
                problem.scores,
            )
            for path_name, coordinates in paths.items():
                for n, m, x_value in coordinates:
                    for repetition in range(repetitions):
                        rng = np.random.default_rng(
                            stable_seed("delta", dataset_config["name"], path_name, n, m, seed, repetition)
                        )
                        indices = rng.choice(len(problem.scores), size=n, replace=True)
                        structure = CalibrationStructure.build(
                            problem.scores[indices], weight.values[indices], w_infinity
                        )
                        fixed_choices = [("optimized", None)] + [
                            (f"cal={fraction:.2f}", fraction) for fraction in fixed_fractions
                        ]
                        for label, fraction in fixed_choices:
                            constants = fixed_constants(
                                weight.bound,
                                n,
                                m,
                                delta,
                                w_infinity=w_infinity,
                                optimize=fraction is None,
                                calibration_fraction=fraction,
                            )
                            value = float(
                                estimate_g_algorithm1(structure, alpha + constants.delta_shift)
                                + constants.epsilon_test
                            )
                            records.append(
                                {
                                    "dataset": dataset_config["name"], "path": path_name,
                                    "n": n, "m": m, "x": x_value, "repetition": repetition,
                                    "bound_type": "fixed", "allocation": label, "bound": value,
                                }
                            )
                        uniform_choices = [("optimized", None)] + [
                            (f"side={fraction:.2f}", fraction) for fraction in uniform_fractions
                        ]
                        for label, fraction in uniform_choices:
                            constants = uniform_constants(
                                weight.bound,
                                n,
                                m,
                                delta,
                                w_infinity=w_infinity,
                                optimize=fraction is None,
                                side_fraction=fraction,
                            )
                            value = float(
                                estimate_g_algorithm1(structure, alpha + constants.delta_shift)
                                + constants.epsilon_test
                            )
                            records.append(
                                {
                                    "dataset": dataset_config["name"], "path": path_name,
                                    "n": n, "m": m, "x": x_value, "repetition": repetition,
                                    "bound_type": "uniform", "allocation": label, "bound": value,
                                }
                            )
        frame = pd.DataFrame(records)
        run.save_metrics(frame)
        run.save_summary({"rows": len(frame), "alpha": alpha})
        datasets = [item["name"] for item in config["datasets"]]
        _plot(frame, datasets, "fixed", run.path / "delta_fixed_2x3.pdf")
        _plot(frame, datasets, "uniform", run.path / "delta_uniform_2x3.pdf")
        run.mark_complete()
