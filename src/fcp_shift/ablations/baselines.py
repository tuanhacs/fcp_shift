from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fcp_shift.ablations.common import prepare_scored_problem, scoped_ablation_path
from fcp_shift.conformal.baselines import (
    calibrate_cojer,
    dkw_forward,
    dkw_inverse,
    dkw_lambda,
    unweighted_p_values,
)
from fcp_shift.conformal.weighted_cp import fcp_at_levels, fcp_curve
from fcp_shift.experiments.common import grid
from fcp_shift.reporting import RunDirectory
from fcp_shift.reporting.style import figure_size, font_size
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import sample_covariate_shift
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def _plot_dataset(
    dataset: str,
    alpha: np.ndarray,
    beta: np.ndarray,
    weight_curves: dict[str, dict[str, np.ndarray]],
    dkw_bound: np.ndarray,
    cojer_bound: np.ndarray,
    output: Path,
) -> None:
    colors = {"exponential": "#0072B2", "quadratic": "#D55E00", "mahalanobis": "#009E73"}
    figure, axis = plt.subplots(figsize=figure_size((8, 5)))
    for index, (weight, curves) in enumerate(weight_curves.items()):
        color = colors.get(weight, plt.get_cmap("tab10")(index))
        axis.plot(alpha, curves["forward"].mean(axis=0), color=color, linestyle="--", label=f"Empirical FCP — {weight}")
    axis.plot(alpha, dkw_bound, color="#6A3D9A", linewidth=2.3, label="DKW bound")
    axis.plot(alpha, cojer_bound, color="#E31A1C", linewidth=2.3, label="CoJER bound")
    axis.set(xlabel=r"Miscoverage $\alpha$", ylabel="Ordinary empirical FCP / bound", title=f"{dataset}: baselines under shift")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=font_size("legend", 8))
    figure.tight_layout()
    figure.savefig(output / f"baselines_forward_{dataset}.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=figure_size((8, 5)))
    axis.plot(beta, beta, color="black", linestyle="--", linewidth=2, label=r"Target $\beta$")
    for index, (weight, curves) in enumerate(weight_curves.items()):
        color = colors.get(weight, plt.get_cmap("tab10")(index))
        axis.plot(beta, curves["dkw_inverse"].mean(axis=0), color=color, linestyle="-", label=f"DKW — {weight}")
        axis.plot(beta, curves["cojer_inverse"].mean(axis=0), color=color, linestyle=":", linewidth=2.2, label=f"CoJER — {weight}")
    axis.set(xlabel=r"Target FCP $\beta$", ylabel="Ordinary empirical FCP", title=f"{dataset}: inverse baselines under shift")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=font_size("legend", 8), ncol=2)
    figure.tight_layout()
    figure.savefig(output / f"baselines_inverse_{dataset}.pdf", bbox_inches="tight")
    plt.close(figure)


def run_baseline_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    alpha, beta = grid(config["fcp"]["alpha_grid"]), grid(config["fcp"]["beta_grid"])
    n, m = int(config["sample_sizes"]["n_calibration"]), int(config["sample_sizes"]["m_test"])
    delta = float(config["fcp"]["delta"])
    repetitions = int(config["experiment"]["repetitions"])
    baseline_config = config["baselines"]
    dkw_bound = dkw_forward(alpha, n, m, delta)
    dkw_alpha = dkw_inverse(beta, n, m, delta)
    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(scoped_ablation_path(root, "baselines", seed, config))
        if run.complete and not force:
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_baselines", "seed": seed,
                "empirical_reference": "ordinary_unweighted_fcp_under_shift",
                "weighted_fcp_used_for_baseline_checks": False,
            },
        )
        cojer = calibrate_cojer(
            n, m, delta,
            int(baseline_config.get("cojer_template_simulations", 1000)),
            int(baseline_config.get("cojer_calibration_simulations", 2000)),
            int(baseline_config.get("cojer_seed", 271828)),
            baseline_config.get("cojer_k_max"),
        )
        cojer_bound = cojer.forward(alpha)
        cojer_alpha = cojer.inverse(beta)
        metric_rows = []
        curve_rows = []
        for dataset_config in config["datasets"]:
            problem = prepare_scored_problem(dataset_config, config["model"])
            curves: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
                lambda: {"forward": [], "dkw_inverse": [], "cojer_inverse": []}
            )
            for weight_config in config["weights"]:
                weight = fit_weight(
                    weight_config,
                    problem.dataset.x_train,
                    problem.dataset.x_source,
                    problem.scores,
                )
                for repetition in range(repetitions):
                    rng = np.random.default_rng(
                        stable_seed("baselines", dataset_config["name"], weight.name, seed, repetition)
                    )
                    calibration, test = sample_covariate_shift(
                        len(problem.scores), weight.values, n, m, rng
                    )
                    ordinary_p = unweighted_p_values(
                        problem.scores[test], problem.scores[calibration]
                    )
                    empirical = fcp_curve(ordinary_p, alpha)
                    empirical_dkw_inverse = fcp_at_levels(ordinary_p, dkw_alpha)
                    empirical_cojer_inverse = fcp_at_levels(ordinary_p, cojer_alpha)
                    curves[weight.name]["forward"].append(empirical)
                    curves[weight.name]["dkw_inverse"].append(empirical_dkw_inverse)
                    curves[weight.name]["cojer_inverse"].append(empirical_cojer_inverse)
                    row = {
                        "dataset": dataset_config["name"], "weight": weight.name,
                        "repetition": repetition, "weight_bound": weight.bound,
                        "dkw_forward_pass": float(np.all(empirical <= dkw_bound + 1e-12)),
                        "cojer_forward_pass": float(np.all(empirical <= cojer_bound + 1e-12)),
                        "dkw_forward_max_violation": float(np.max(empirical - dkw_bound)),
                        "cojer_forward_max_violation": float(np.max(empirical - cojer_bound)),
                        "dkw_inverse_pass": float(np.all(empirical_dkw_inverse <= beta + 1e-12)),
                        "cojer_inverse_pass": float(np.all(empirical_cojer_inverse <= beta + 1e-12)),
                        "dkw_inverse_max_violation": float(np.max(empirical_dkw_inverse - beta)),
                        "cojer_inverse_max_violation": float(np.max(empirical_cojer_inverse - beta)),
                    }
                    metric_rows.append(row)
            stacked = {
                weight: {name: np.stack(values) for name, values in values_by_name.items()}
                for weight, values_by_name in curves.items()
            }
            for weight_name, values_by_name in stacked.items():
                for curve_name, values in values_by_name.items():
                    x_grid = alpha if curve_name == "forward" else beta
                    mean = values.mean(axis=0)
                    low, high = np.quantile(values, [0.1, 0.9], axis=0)
                    curve_rows.extend(
                        {
                            "dataset": dataset_config["name"], "weight": weight_name,
                            "curve": curve_name, "x": float(x_grid[index]),
                            "mean": float(mean[index]), "q10": float(low[index]),
                            "q90": float(high[index]),
                        }
                        for index in range(len(x_grid))
                    )
            _plot_dataset(
                dataset_config["name"], alpha, beta, stacked, dkw_bound, cojer_bound, run.path
            )
        metrics = pd.DataFrame(metric_rows)
        aggregation = metrics.groupby(["dataset", "weight"], as_index=False).agg(
            n_repetitions=("repetition", "count"), weight_bound=("weight_bound", "mean"),
            dkw_forward_pass_rate=("dkw_forward_pass", "mean"),
            cojer_forward_pass_rate=("cojer_forward_pass", "mean"),
            dkw_forward_mean_max_violation=("dkw_forward_max_violation", lambda x: np.maximum(x, 0).mean()),
            cojer_forward_mean_max_violation=("cojer_forward_max_violation", lambda x: np.maximum(x, 0).mean()),
            dkw_inverse_pass_rate=("dkw_inverse_pass", "mean"),
            cojer_inverse_pass_rate=("cojer_inverse_pass", "mean"),
            dkw_inverse_mean_max_violation=("dkw_inverse_max_violation", lambda x: np.maximum(x, 0).mean()),
            cojer_inverse_mean_max_violation=("cojer_inverse_max_violation", lambda x: np.maximum(x, 0).mean()),
        )
        run.save_metrics(metrics)
        pd.DataFrame(curve_rows).to_csv(
            run.path / "baseline_curves_summary.csv", index=False
        )
        aggregation.to_csv(run.path / "baseline_comparison_table.csv", index=False)
        run.save_arrays(
            alpha=alpha, beta=beta, dkw_bound=dkw_bound, cojer_bound=cojer_bound,
            dkw_alpha=dkw_alpha, cojer_alpha=cojer_alpha,
        )
        run.save_summary(
            {
                "rows": len(metrics), "dkw_lambda": dkw_lambda(delta, n, m),
                "comparison_uses_unweighted_fcp": True,
            }
        )
        run.mark_complete()
