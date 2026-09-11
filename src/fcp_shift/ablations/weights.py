from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fcp_shift.ablations.common import (
    add_dataset_row_labels,
    prepare_scored_problem,
    scoped_ablation_path,
    set_publication_ticks,
)
from fcp_shift.experiments.common import calculate_goals, grid, stack_goal_results
from fcp_shift.reporting import RunDirectory
from fcp_shift.reporting.labels import (
    FIXED_ALPHA_LABEL,
    FIXED_BETA_LABEL,
    UNIFORM_ALPHA_LABEL,
    UNIFORM_BETA_LABEL,
)
from fcp_shift.reporting.style import figure_size, font_size
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import build_score_transport, sample_covariate_shift
from fcp_shift.weights import fit_weight


WEIGHT_COLORS = {
    "exponential": "#0072B2",
    "quadratic": "#D55E00",
    "mahalanobis": "#009E73",
    "linear": "#CC79A7",
    "sigmoid": "#7B61A8",
}


def _plot_family(
    curves: dict[tuple[str, str, str], dict[str, np.ndarray]] | pd.DataFrame,
    datasets: list[str],
    weights: list[str],
    shift: str,
    family: str,
    alpha: np.ndarray,
    beta: np.ndarray,
    output_path: Path,
) -> None:
    if family == "forward":
        x = alpha
        curve_names = ("goal1_bound", "goal2_bound")
        titles = (FIXED_ALPHA_LABEL, UNIFORM_ALPHA_LABEL)
        xlabel = r"Miscoverage $\alpha$"
        ylabel = "FCP bound"
        reference_label = r"Nominal $\alpha$"
    else:
        x = beta
        curve_names = ("goal3_fcp", "goal4_fcp")
        titles = (FIXED_BETA_LABEL, UNIFORM_BETA_LABEL)
        xlabel = r"Target FCP $\beta$"
        ylabel = "Empirical FCP"
        reference_label = r"Target $\beta$"

    figure, axes = plt.subplots(
        len(datasets),
        2,
        figsize=figure_size((12, 4.2 * len(datasets))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, dataset in enumerate(datasets):
        for column, (curve_name, title) in enumerate(zip(curve_names, titles)):
            axis = axes[row, column]
            axis.plot(
                x,
                x,
                color="black",
                linestyle="--",
                linewidth=2,
                label=reference_label,
            )
            for index, weight in enumerate(weights):
                if isinstance(curves, pd.DataFrame):
                    summary = curves[
                        (curves["shift"] == shift)
                        & (curves["dataset"] == dataset)
                        & (curves["weight"] == weight)
                        & (curves["curve"] == curve_name)
                    ].sort_values("x")
                    if summary.empty:
                        raise ValueError(
                            f"Missing saved curve for {shift}/{dataset}/{weight}/{curve_name}"
                        )
                    mean = summary["mean"].to_numpy(dtype=float)
                    std = summary["std"].to_numpy(dtype=float)
                else:
                    values = curves[(shift, dataset, weight)][curve_name]
                    mean = values.mean(axis=0)
                    std = (
                        values.std(axis=0, ddof=1)
                        if values.shape[0] > 1
                        else np.zeros_like(mean)
                    )
                color = WEIGHT_COLORS.get(weight, plt.get_cmap("tab10")(index))
                axis.fill_between(
                    x,
                    np.maximum(mean - std, 0.0),
                    mean + std,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
                axis.plot(x, mean, color=color, linewidth=2, label=weight)
            if row == 0:
                axis.set_title(title)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            set_publication_ticks(axis)
            axis.grid(alpha=0.25)
            if row == 0 and column == 1:
                axis.legend(fontsize=font_size("legend", 8), ncol=2)
    if len(datasets) == 1:
        axes[0, 0].set_xlabel(xlabel)
        axes[0, 0].set_ylabel(ylabel)
        figure.tight_layout()
    else:
        add_dataset_row_labels(axes, datasets)
        figure.supxlabel(xlabel)
        figure.supylabel(ylabel)
        figure.tight_layout(rect=(0.06, 0.05, 1.0, 1.0))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _weight_curve_mean_std(
    curves: dict[tuple[str, str, str], dict[str, np.ndarray]] | pd.DataFrame,
    shift: str,
    dataset: str,
    weight: str,
    curve_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(curves, pd.DataFrame):
        summary = curves[
            (curves["shift"] == shift)
            & (curves["dataset"] == dataset)
            & (curves["weight"] == weight)
            & (curves["curve"] == curve_name)
        ].sort_values("x")
        if summary.empty:
            raise ValueError(
                f"Missing saved curve for {shift}/{dataset}/{weight}/{curve_name}"
            )
        return (
            summary["mean"].to_numpy(dtype=float),
            summary["std"].fillna(0.0).to_numpy(dtype=float),
        )
    values = curves[(shift, dataset, weight)][curve_name]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def _plot_grid(
    curves: dict[tuple[str, str, str], dict[str, np.ndarray]] | pd.DataFrame,
    datasets: list[str],
    weights: list[str],
    alpha: np.ndarray,
    beta: np.ndarray,
    output_directory: Path,
) -> list[Path]:
    definitions = (
        ("goal1_bound", alpha, FIXED_ALPHA_LABEL, r"Nominal $\alpha$"),
        ("goal2_bound", alpha, UNIFORM_ALPHA_LABEL, r"Nominal $\alpha$"),
        ("goal3_fcp", beta, FIXED_BETA_LABEL, r"Target $\beta$"),
        ("goal4_fcp", beta, UNIFORM_BETA_LABEL, r"Target $\beta$"),
    )
    shifts = (
        ("covariate_shift", "Covariate Shift"),
        ("score_transport_shift", "Score-Transport Shift"),
    )
    for obsolete in (
        "weights_covariate_forward_goals_1_2.pdf",
        "weights_covariate_inverse_goals_3_4.pdf",
        "weights_transport_forward_goals_1_2.pdf",
        "weights_transport_inverse_goals_3_4.pdf",
    ):
        (output_directory / obsolete).unlink(missing_ok=True)

    generated: list[Path] = []
    for dataset in datasets:
        figure, axes = plt.subplots(
            2,
            4,
            figsize=figure_size((18, 7)),
            squeeze=False,
            sharex=False,
            sharey=False,
        )
        for row, (shift, row_label) in enumerate(shifts):
            for column, (curve_name, x, title, reference_label) in enumerate(
                definitions
            ):
                axis = axes[row, column]
                axis.plot(
                    x,
                    x,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    label=reference_label,
                )
                for index, weight in enumerate(weights):
                    mean, std = _weight_curve_mean_std(
                        curves, shift, dataset, weight, curve_name
                    )
                    color = WEIGHT_COLORS.get(weight, plt.get_cmap("tab10")(index))
                    axis.fill_between(
                        x,
                        np.clip(mean - std, 0.0, 1.0),
                        np.clip(mean + std, 0.0, 1.0),
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                    )
                    axis.plot(x, mean, color=color, linewidth=2, label=weight)
                if row == 0:
                    axis.set_title(title)
                axis.set_xlim(0.0, 1.0)
                axis.set_ylim(0.0, 1.0)
                set_publication_ticks(axis)
                axis.grid(alpha=0.25)
            axes[row, 0].annotate(
                row_label,
                xy=(-0.28, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="center",
                va="center",
                fontweight="bold",
            )

        axes[1, 0].set_xlabel(r"Miscoverage $\alpha$")
        axes[1, 2].set_xlabel(r"Target FCP $\beta$")
        axes[0, 0].set_ylabel("FCP bound")
        axes[1, 0].set_ylabel("FCP bound")
        axes[0, 2].set_ylabel("Empirical FCP")
        axes[1, 2].set_ylabel("Empirical FCP")
        axes[0, 0].legend(fontsize=font_size("legend", 8), ncol=2)
        axes[0, 2].legend(fontsize=font_size("legend", 8), ncol=2)
        figure.tight_layout(rect=(0.035, 0.02, 1.0, 1.0))
        filename = (
            "weights_2x4.pdf"
            if len(datasets) == 1
            else f"weights_{dataset}_2x4.pdf"
        )
        destination = output_directory / filename
        figure.savefig(destination, bbox_inches="tight")
        plt.close(figure)
        generated.append(destination)
    return generated


def _curve_records(
    arrays: dict[str, np.ndarray],
    shift: str,
    dataset: str,
    weight: str,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    definitions = {
        "empirical_fcp": alpha,
        "goal1_bound": alpha,
        "goal2_bound": alpha,
        "goal3_fcp": beta,
        "goal4_fcp": beta,
    }
    for curve_name, x in definitions.items():
        values = arrays[curve_name]
        mean = values.mean(axis=0)
        std = (
            values.std(axis=0, ddof=1)
            if values.shape[0] > 1
            else np.zeros_like(mean)
        )
        records.extend(
            {
                "shift": shift,
                "dataset": dataset,
                "weight": weight,
                "curve": curve_name,
                "x": float(x[index]),
                "mean": float(mean[index]),
                "std": float(std[index]),
            }
            for index in range(len(x))
        )
    return records


def run_weight_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    alpha = grid(config["fcp"]["alpha_grid"])
    beta = grid(config["fcp"]["beta_grid"])
    n = int(config["sample_sizes"]["n_calibration"])
    m = int(config["sample_sizes"]["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    delta = float(config["fcp"]["delta"])
    w_infinity = float(config["fcp"].get("w_infinity", 1.0))
    optimize_delta = bool(config["fcp"].get("optimize_delta", True))
    eta = float(config["fcp"].get("eta", 1e-10))
    strata = int(config.get("transport", {}).get("strata", 5))
    rho = float(config.get("transport", {}).get("rho", 0.5))
    datasets = [item["name"] for item in config["datasets"]]
    weights = [item["name"] for item in config["weights"]]

    for seed in config["experiment"]["seeds"]:
        # Separate this five-family study from legacy gamma-misspecification outputs.
        run = RunDirectory(scoped_ablation_path(root, "weight_families", seed, config))
        if run.complete and not force:
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_weights",
                "seed": seed,
                "settings": ["covariate_shift", "score_transport_shift"],
                "transport_rho": rho,
                "transport_strata": strata,
                "target_distribution_varies_by_weight_family": True,
            },
        )
        curves: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
        metric_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []

        for dataset_config in config["datasets"]:
            problem = prepare_scored_problem(dataset_config, config["model"])
            dataset = dataset_config["name"]
            for weight_config in config["weights"]:
                base_weight = fit_weight(
                    weight_config,
                    problem.dataset.x_train,
                    problem.dataset.x_source,
                    problem.scores,
                )
                transport = build_score_transport(
                    problem.scores, base_weight.values, strata
                )
                transport_weights = transport.weights(rho)
                results_by_shift = {
                    "covariate_shift": [],
                    "score_transport_shift": [],
                }

                for repetition in range(repetitions):
                    cov_rng = np.random.default_rng(
                        stable_seed(
                            "weights",
                            "covariate",
                            dataset,
                            base_weight.name,
                            seed,
                            repetition,
                        )
                    )
                    calibration, test = sample_covariate_shift(
                        len(problem.scores), base_weight.values, n, m, cov_rng
                    )
                    cov_result = calculate_goals(
                        problem.scores[calibration],
                        base_weight.values[calibration],
                        problem.scores[test],
                        alpha,
                        beta,
                        base_weight.bound,
                        delta,
                        w_infinity,
                        "covariate_identity",
                        optimize_delta,
                        eta,
                    )
                    results_by_shift["covariate_shift"].append(cov_result)

                    sts_rng = np.random.default_rng(
                        stable_seed(
                            "weights",
                            "transport",
                            dataset,
                            base_weight.name,
                            seed,
                            repetition,
                        )
                    )
                    calibration = sts_rng.choice(
                        len(problem.scores), size=n, replace=True
                    )
                    test_scores = transport.sample_test_scores(m, rho, sts_rng)
                    sts_result = calculate_goals(
                        problem.scores[calibration],
                        transport_weights[calibration],
                        test_scores,
                        alpha,
                        beta,
                        float(np.max(transport_weights)),
                        delta,
                        w_infinity,
                        "algorithm_1",
                        optimize_delta,
                        eta,
                    )
                    results_by_shift["score_transport_shift"].append(sts_result)

                for shift, results in results_by_shift.items():
                    arrays = stack_goal_results(results)
                    curves[(shift, dataset, base_weight.name)] = arrays
                    curve_rows.extend(
                        _curve_records(
                            arrays,
                            shift,
                            dataset,
                            base_weight.name,
                            alpha,
                            beta,
                        )
                    )
                    for repetition, result in enumerate(results):
                        metric_rows.append(
                            {
                                "shift": shift,
                                "dataset": dataset,
                                "weight": base_weight.name,
                                "repetition": repetition,
                                "goal1_pass_fraction": float(
                                    np.mean(
                                        result.empirical_fcp <= result.goal1_bound
                                    )
                                ),
                                "goal2_uniform_pass": float(
                                    np.all(
                                        result.empirical_fcp <= result.goal2_bound
                                    )
                                ),
                                "goal3_pass_fraction": float(
                                    np.mean(result.goal3_fcp <= beta)
                                ),
                                "goal4_uniform_pass": float(
                                    np.all(result.goal4_fcp <= beta)
                                ),
                            }
                        )

        metrics = pd.DataFrame(metric_rows)
        run.save_metrics(metrics)
        pd.DataFrame(curve_rows).to_csv(
            run.path / "weight_curves_summary.csv", index=False
        )
        run.save_summary(
            {
                "rows": len(metrics),
                "datasets": datasets,
                "weights": weights,
                "transport_rho": rho,
                **metrics.mean(numeric_only=True).to_dict(),
            }
        )
        _plot_grid(curves, datasets, weights, alpha, beta, run.path)
        run.mark_complete()
