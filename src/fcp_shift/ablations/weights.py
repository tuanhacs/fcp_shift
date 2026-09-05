from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fcp_shift.ablations.common import prepare_scored_problem, scoped_ablation_path
from fcp_shift.experiments.common import calculate_goals, grid
from fcp_shift.reporting import RunDirectory
from fcp_shift.reporting.style import figure_size, font_size
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import sample_covariate_shift
from fcp_shift.weights import fit_weight


def _plot(frame: pd.DataFrame, dataset: str, path: Path) -> None:
    subset = frame[frame.dataset == dataset]
    summary = subset.groupby("gamma").agg(
        goal1_pass=("goal1_pass", "mean"), goal2_pass=("goal2_pass", "mean"),
        goal3_pass=("goal3_pass", "mean"), goal4_pass=("goal4_pass", "mean"),
        bound=("weight_bound", "mean"), ess=("ess", "mean"),
    ).reset_index()
    figure, axes = plt.subplots(1, 2, figsize=figure_size((12, 4.5)))
    for goal, color in zip(range(1, 5), ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]):
        axes[0].plot(summary.gamma, summary[f"goal{goal}_pass"], marker="o", color=color, label=f"Goal {goal}")
    axes[0].axhline(0.9, color="black", linestyle="--", label="Nominal 1-delta")
    axes[0].set(xlabel=r"Misspecification $\gamma$", ylabel="Empirical pass rate", title=f"{dataset}: validity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=font_size("legend", 8))
    axes[1].plot(summary.gamma, summary.bound, marker="o", color="#0072B2", label="Weight bound B")
    second = axes[1].twinx()
    second.plot(summary.gamma, summary.ess, marker="s", color="#D55E00", label="ESS")
    axes[1].set(xlabel=r"Misspecification $\gamma$", ylabel="Weight bound B", title=f"{dataset}: weight diagnostics")
    second.set_ylabel("Effective sample size")
    axes[1].grid(alpha=0.25)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axes[1].legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=font_size("legend", 8),
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run_weight_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    alpha, beta = grid(config["fcp"]["alpha_grid"]), grid(config["fcp"]["beta_grid"])
    n, m = int(config["sample_sizes"]["n_calibration"]), int(config["sample_sizes"]["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    gammas = [float(value) for value in config["ablation"]["gammas"]]
    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(scoped_ablation_path(root, "weights", seed, config))
        if run.complete and not force:
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_weights", "seed": seed,
                "target_distribution_fixed": True,
                "perturbation": "w_gamma=(1-gamma)*w_oracle+gamma*1",
            },
        )
        rows = []
        for dataset_config in config["datasets"]:
            problem = prepare_scored_problem(dataset_config, config["model"])
            oracle = fit_weight(
                config["weights"][0], problem.dataset.x_train, problem.dataset.x_source, problem.scores
            )
            for repetition in range(repetitions):
                rng = np.random.default_rng(
                    stable_seed("weights", dataset_config["name"], seed, repetition)
                )
                calibration, test = sample_covariate_shift(
                    len(problem.scores), oracle.values, n, m, rng
                )
                for gamma in gammas:
                    used = (1.0 - gamma) * oracle.values + gamma * np.ones_like(oracle.values)
                    used /= used.mean()
                    result = calculate_goals(
                        problem.scores[calibration], used[calibration], problem.scores[test],
                        alpha, beta, float(used.max()), float(config["fcp"]["delta"]),
                        float(config["fcp"].get("w_infinity", 1.0)), "algorithm_1",
                        bool(config["fcp"].get("optimize_delta", True)),
                        float(config["fcp"].get("eta", 1e-10)),
                    )
                    rows.append(
                        {
                            "dataset": dataset_config["name"], "repetition": repetition, "gamma": gamma,
                            "goal1_pass": np.mean(result.empirical_fcp <= result.goal1_bound),
                            "goal2_pass": float(np.all(result.empirical_fcp <= result.goal2_bound)),
                            "goal3_pass": np.mean(result.goal3_fcp <= beta),
                            "goal4_pass": float(np.all(result.goal4_fcp <= beta)),
                            "weight_bound": float(used.max()),
                            "ess": float(used.sum() ** 2 / np.sum(used**2)),
                        }
                    )
        frame = pd.DataFrame(rows)
        run.save_metrics(frame)
        run.save_summary({"rows": len(frame), **frame.mean(numeric_only=True).to_dict()})
        for dataset_config in config["datasets"]:
            _plot(frame, dataset_config["name"], run.path / f"weight_misspecification_{dataset_config['name']}.pdf")
        run.mark_complete()
