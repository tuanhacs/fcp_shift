from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fcp_shift.ablations.common import (
    prepare_scored_problem,
    publication_dataset_name,
    scoped_ablation_path,
    set_publication_ticks,
)
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
from fcp_shift.reporting.style import (
    figure_size, font_size, set_probability_limits,
)
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import sample_covariate_shift
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def _plot_dataset(
    dataset: str,
    delta_grid: np.ndarray,
    weight_curves: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    display_dataset = publication_dataset_name(dataset)
    colors = {
        "exponential": "#0072B2",
        "quadratic": "#D55E00",
        "mahalanobis": "#009E73",
        "linear": "#CC79A7",
        "sigmoid": "#7B61A8",
        "logarithmic": "#8C6D31",
        "arctangent": "#D1495B",
        "power_tilt": "#4C5B9B",
    }
    figure, axis = plt.subplots(figsize=figure_size((8, 5)))
    required = 1.0 - delta_grid
    axis.plot(delta_grid, required, color="black", linestyle="--", linewidth=2,
              label=r"Required probability $1-\delta$")
    plotted = []
    for index, (weight, curves) in enumerate(weight_curves.items()):
        color = colors.get(weight, plt.get_cmap("tab10")(index))
        dkw = np.asarray(curves["dkw_forward_pass_rate"], dtype=float).reshape(-1)
        cojer = np.asarray(curves["cojer_forward_pass_rate"], dtype=float).reshape(-1)
        axis.plot(
            delta_grid, dkw, color=color,
            linestyle="-", label=f"DKW — {weight}",
        )
        axis.plot(
            delta_grid, cojer, color=color,
            linestyle=":", linewidth=2.2, label=f"CoJER — {weight}",
        )
        plotted.extend((dkw, cojer))
    axis.set(
        xlabel=r"Failure probability $\delta$",
        ylabel="Guarantee probability",
        title=f"{display_dataset}: baselines under shift",
    )
    set_probability_limits(
        axis, delta_grid, required, *plotted, trim_unit_interval=False
    )
    set_publication_ticks(axis)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=font_size("legend", 8))
    figure.tight_layout()
    figure.savefig(output / f"baselines_forward_{dataset}.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=figure_size((8, 5)))
    axis.plot(delta_grid, required, color="black", linestyle="--", linewidth=2,
              label=r"Required probability $1-\delta$")
    plotted = []
    for index, (weight, curves) in enumerate(weight_curves.items()):
        color = colors.get(weight, plt.get_cmap("tab10")(index))
        dkw = np.asarray(curves["dkw_inverse_pass_rate"], dtype=float).reshape(-1)
        cojer = np.asarray(curves["cojer_inverse_pass_rate"], dtype=float).reshape(-1)
        axis.plot(delta_grid, dkw, color=color, linestyle="-", label=f"DKW — {weight}")
        axis.plot(delta_grid, cojer, color=color, linestyle=":", linewidth=2.2, label=f"CoJER — {weight}")
        plotted.extend((dkw, cojer))
    axis.set(
        xlabel=r"Failure probability $\delta$",
        ylabel="Guarantee probability",
        title=f"{display_dataset}: inverse baselines under shift",
    )
    set_probability_limits(
        axis, delta_grid, required, *plotted, trim_unit_interval=False
    )
    set_publication_ticks(axis)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=font_size("legend", 8), ncol=2)
    figure.tight_layout()
    figure.savefig(output / f"baselines_inverse_{dataset}.pdf", bbox_inches="tight")
    plt.close(figure)


def run_baseline_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    alpha, beta = grid(config["fcp"]["alpha_grid"]), grid(config["fcp"]["beta_grid"])
    n, m = int(config["sample_sizes"]["n_calibration"]), int(config["sample_sizes"]["m_test"])
    delta_grid = grid(
        config["fcp"].get(
            "delta_grid", {"values": [0.01, 0.025, 0.05, 0.1, 0.2]}
        )
    )
    if np.any((delta_grid <= 0.0) | (delta_grid >= 1.0)):
        raise ValueError("fcp.delta_grid values must lie in (0, 1)")
    repetitions = int(config["experiment"]["repetitions"])
    baseline_config = config["baselines"]
    dkw_bounds = np.stack([dkw_forward(alpha, n, m, delta) for delta in delta_grid])
    dkw_alphas = np.stack([dkw_inverse(beta, n, m, delta) for delta in delta_grid])
    cojer_objects = [
        calibrate_cojer(
            n, m, float(delta),
            int(baseline_config.get("cojer_template_simulations", 1000)),
            int(baseline_config.get("cojer_calibration_simulations", 2000)),
            stable_seed(
                "cojer_delta", int(baseline_config.get("cojer_seed", 271828)),
                float(delta),
            ),
            baseline_config.get("cojer_k_max"),
        )
        for delta in delta_grid
    ]
    cojer_bounds = np.stack([cojer.forward(alpha) for cojer in cojer_objects])
    cojer_alphas = np.stack([cojer.inverse(beta) for cojer in cojer_objects])
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
        metric_rows = []
        for dataset_config in config["datasets"]:
            problem = prepare_scored_problem(dataset_config, config["model"])
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
                    for delta_index, delta in enumerate(delta_grid):
                        empirical_dkw_inverse = fcp_at_levels(
                            ordinary_p, dkw_alphas[delta_index]
                        )
                        empirical_cojer_inverse = fcp_at_levels(
                            ordinary_p, cojer_alphas[delta_index]
                        )
                        dkw_forward_pass = empirical <= dkw_bounds[delta_index] + 1e-12
                        cojer_forward_pass = empirical <= cojer_bounds[delta_index] + 1e-12
                        dkw_inverse_pass = empirical_dkw_inverse <= beta + 1e-12
                        cojer_inverse_pass = empirical_cojer_inverse <= beta + 1e-12
                        metric_rows.append(
                            {
                                "dataset": dataset_config["name"],
                                "weight": weight.name,
                                "delta": float(delta),
                                "repetition": repetition,
                                "weight_bound": weight.bound,
                                "dkw_forward_pass": float(np.all(dkw_forward_pass)),
                                "cojer_forward_pass": float(np.all(cojer_forward_pass)),
                                "dkw_forward_max_violation": float(
                                    np.max(empirical - dkw_bounds[delta_index])
                                ),
                                "cojer_forward_max_violation": float(
                                    np.max(empirical - cojer_bounds[delta_index])
                                ),
                                "dkw_inverse_pass": float(np.all(dkw_inverse_pass)),
                                "cojer_inverse_pass": float(np.all(cojer_inverse_pass)),
                                "dkw_inverse_max_violation": float(
                                    np.max(empirical_dkw_inverse - beta)
                                ),
                                "cojer_inverse_max_violation": float(
                                    np.max(empirical_cojer_inverse - beta)
                                ),
                            }
                        )
        metrics = pd.DataFrame(metric_rows)
        aggregation = metrics.groupby(["dataset", "weight", "delta"], as_index=False).agg(
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
        curve_rows = []
        curve_columns = (
            "dkw_forward_pass_rate", "cojer_forward_pass_rate",
            "dkw_inverse_pass_rate", "cojer_inverse_pass_rate",
        )
        for row in aggregation.itertuples(index=False):
            curve_rows.extend(
                {
                    "dataset": row.dataset,
                    "weight": row.weight,
                    "curve": curve,
                    "delta": float(row.delta),
                    "pass_rate": float(getattr(row, curve)),
                }
                for curve in curve_columns
            )
        for dataset_config in config["datasets"]:
            dataset_name = dataset_config["name"]
            weight_curves = {}
            for weight_config in config["weights"]:
                weight_name = weight_config["name"]
                subset = aggregation[
                    (aggregation.dataset == dataset_name)
                    & (aggregation.weight == weight_name)
                ].sort_values("delta")
                weight_curves[weight_name] = {
                    curve: subset[curve].to_numpy(dtype=float)
                    for curve in curve_columns
                }
            _plot_dataset(dataset_name, delta_grid, weight_curves, run.path)
        run.save_metrics(metrics)
        pd.DataFrame(curve_rows).to_csv(
            run.path / "baseline_curves_summary.csv", index=False
        )
        aggregation.to_csv(run.path / "baseline_comparison_table.csv", index=False)
        run.save_arrays(
            alpha=alpha, beta=beta, delta=delta_grid,
            dkw_bound=dkw_bounds, cojer_bound=cojer_bounds,
            dkw_alpha=dkw_alphas, cojer_alpha=cojer_alphas,
        )
        run.save_summary(
            {
                "rows": len(metrics),
                "dkw_lambda": [
                    dkw_lambda(float(delta), n, m) for delta in delta_grid
                ],
                "comparison_uses_unweighted_fcp": True,
            }
        )
        run.mark_complete()
