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
from fcp_shift.reporting import RunDirectory
from fcp_shift.reproducibility import stable_seed
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def _plot(frame: pd.DataFrame, datasets: list[str], alphas: list[float], path: Path) -> None:
    figure, axes = plt.subplots(
        len(datasets), len(alphas), figsize=(5 * len(alphas), 4 * len(datasets)), squeeze=False
    )
    colors = {"exponential": "#0072B2", "quadratic": "#D55E00", "mahalanobis": "#009E73"}
    for row, dataset in enumerate(datasets):
        for column, alpha in enumerate(alphas):
            axis = axes[row, column]
            subset = frame[(frame.dataset == dataset) & np.isclose(frame.alpha, alpha)]
            axis.axhline(alpha, color="black", linewidth=2, label=rf"$\alpha={alpha:g}$")
            for index, (weight, group) in enumerate(subset.groupby("weight", sort=False)):
                summary = group.groupby("n").estimate.agg(["mean", "std"]).reset_index()
                color = colors.get(weight, plt.get_cmap("tab10")(index))
                axis.plot(summary.n, summary["mean"], marker="o", color=color, label=weight)
                axis.fill_between(
                    summary.n,
                    np.maximum(summary["mean"] - summary["std"].fillna(0.0), 0.0),
                    summary["mean"] + summary["std"].fillna(0.0),
                    color=color,
                    alpha=0.12,
                )
            axis.set_xscale("log")
            axis.set_title(f"{dataset}, alpha={alpha:g}")
            axis.set_xlabel(r"Calibration size $n$")
            axis.set_ylabel(r"Estimated $G_{w,n}(\alpha)$")
            axis.grid(alpha=0.25)
            if row == 0 and column == len(alphas) - 1:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run_corollary_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    repetitions = int(config["experiment"]["repetitions"])
    n_grid = [int(value) for value in config["ablation"]["n_grid"]]
    alphas = [float(value) for value in config["ablation"]["alphas"]]
    w_infinity = float(config["fcp"].get("w_infinity", 1.0))
    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(scoped_ablation_path(root, "corollary", seed, config))
        if run.complete and not force:
            LOGGER.info("Skipping completed run %s", run.path)
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_corollary",
                "seed": seed,
                "g_estimator": "algorithm_1_in_paper",
                "notebook_alias": "ghat_alg2",
            },
        )
        records = []
        for dataset_config in config["datasets"]:
            problem = prepare_scored_problem(dataset_config, config["model"])
            for weight_config in config["weights"]:
                weight = fit_weight(
                    weight_config,
                    problem.dataset.x_train,
                    problem.dataset.x_source,
                    problem.scores,
                )
                for n in n_grid:
                    for repetition in range(repetitions):
                        rng = np.random.default_rng(
                            stable_seed("corollary", dataset_config["name"], weight.name, n, seed, repetition)
                        )
                        indices = rng.choice(len(problem.scores), size=n, replace=True)
                        structure = CalibrationStructure.build(
                            problem.scores[indices], weight.values[indices], w_infinity
                        )
                        estimates = estimate_g_algorithm1(structure, np.asarray(alphas))
                        for alpha, estimate in zip(alphas, estimates):
                            records.append(
                                {
                                    "dataset": dataset_config["name"],
                                    "weight": weight.name,
                                    "n": n,
                                    "repetition": repetition,
                                    "alpha": alpha,
                                    "estimate": float(estimate),
                                    "weight_bound": weight.bound,
                                }
                            )
        frame = pd.DataFrame(records)
        run.save_metrics(frame)
        run.save_summary(
            {
                "rows": len(frame),
                "datasets": [item["name"] for item in config["datasets"]],
                "alphas": alphas,
                "maximum_excess_over_alpha": float((frame.estimate - frame.alpha).max()),
            }
        )
        _plot(
            frame,
            [item["name"] for item in config["datasets"]],
            alphas,
            run.path / "corollary_convergence_2x3.pdf",
        )
        run.mark_complete()
