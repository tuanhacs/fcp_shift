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

from fcp_shift.conformal.weighted_cp import fcp_curve
from fcp_shift.ablations.common import scoped_ablation_path
from fcp_shift.data import prepare_dataset
from fcp_shift.experiments.common import calculate_goals, grid
from fcp_shift.models import conformity_scores, fit_model
from fcp_shift.reporting import RunDirectory
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import sample_covariate_shift
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def _summarize_curves(curves: dict[tuple[str, str], list], alpha, beta) -> pd.DataFrame:
    records = []
    for (weight, model), results in curves.items():
        for name, x in [
            ("empirical_fcp", alpha), ("goal1_bound", alpha), ("goal2_bound", alpha),
            ("goal3_fcp", beta), ("goal4_fcp", beta),
        ]:
            values = np.stack([getattr(result, name) for result in results])
            mean = values.mean(axis=0)
            low, high = np.quantile(values, [0.1, 0.9], axis=0)
            records.extend(
                {
                    "weight": weight, "model": model, "curve": name,
                    "x": float(x[index]), "mean": float(mean[index]),
                    "q10": float(low[index]), "q90": float(high[index]),
                }
                for index in range(len(x))
            )
    return pd.DataFrame(records)


def _plot(summary: pd.DataFrame, weights: list[str], models: list[str], output: Path) -> None:
    colors = {model: plt.get_cmap("tab10")(index) for index, model in enumerate(models)}
    for weight in weights:
        subset_weight = summary[summary.weight == weight]
        for goal in (1, 2):
            figure, axis = plt.subplots(figsize=(8, 5))
            bound = subset_weight[
                (subset_weight.model == models[0]) & (subset_weight.curve == f"goal{goal}_bound")
            ]
            axis.plot(bound.x, bound["mean"], color="black", linewidth=2.5, label=f"Goal {goal} bound")
            for model in models:
                curve = subset_weight[
                    (subset_weight.model == model) & (subset_weight.curve == "empirical_fcp")
                ]
                axis.plot(curve.x, curve["mean"], color=colors[model], label=model)
                axis.fill_between(curve.x, curve.q10, curve.q90, color=colors[model], alpha=0.10)
            axis.set(xlabel=r"Miscoverage $\alpha$", ylabel="FCP / bound", title=f"{weight} — Goal {goal}")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(output / f"models_{weight}_forward_goal_{goal}.pdf", bbox_inches="tight")
            plt.close(figure)
        for goal in (3, 4):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=2, label=r"Target $\beta$")
            for model in models:
                curve = subset_weight[
                    (subset_weight.model == model) & (subset_weight.curve == f"goal{goal}_fcp")
                ]
                axis.plot(curve.x, curve["mean"], color=colors[model], label=model)
                axis.fill_between(curve.x, curve.q10, curve.q90, color=colors[model], alpha=0.10)
            axis.set(xlabel=r"Target FCP $\beta$", ylabel="Empirical FCP", title=f"{weight} — Goal {goal}")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(output / f"models_{weight}_inverse_goal_{goal}.pdf", bbox_inches="tight")
            plt.close(figure)


def run_model_ablation(config: dict[str, Any], force: bool = False) -> None:
    root = Path(config.get("output", {}).get("root", "outputs"))
    dataset_config = config["datasets"][0]
    alpha, beta = grid(config["fcp"]["alpha_grid"]), grid(config["fcp"]["beta_grid"])
    n = int(config["sample_sizes"]["n_calibration"])
    m = int(config["sample_sizes"]["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    model_configs = config["models"]
    model_names = [item["name"] for item in model_configs]
    weight_names = [item["name"] for item in config["weights"]]
    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(
            scoped_ablation_path(root, "models", seed, config, dataset_config["name"])
        )
        if run.complete and not force:
            continue
        run.initialize(
            config,
            {
                "experiment": "ablation_models", "dataset": dataset_config["name"],
                "seed": seed, "target_distribution_fixed_across_models": True,
                "reference_model": model_names[0],
            },
        )
        model_seed = int(model_configs[0].get("seed", 2026))
        dataset = prepare_dataset(dataset_config, model_seed)
        fitted_models, model_scores = {}, {}
        for model_config in model_configs:
            name = model_config["name"]
            fitted_models[name] = fit_model(
                dataset.task, dataset.x_train, dataset.y_train, model_config,
                int(model_config.get("seed", model_seed)),
            )
            model_scores[name] = conformity_scores(
                fitted_models[name], dataset.x_source, dataset.y_source, dataset.task,
                model_config.get("classification_score", "log_margin"),
            )
        reference_scores = model_scores[model_names[0]]
        curves = defaultdict(list)
        metric_rows = []
        for weight_config in config["weights"]:
            weight = fit_weight(weight_config, dataset.x_train, dataset.x_source, reference_scores)
            for repetition in range(repetitions):
                rng = np.random.default_rng(
                    stable_seed("models", dataset.name, weight.name, seed, repetition)
                )
                calibration, test = sample_covariate_shift(
                    len(reference_scores), weight.values, n, m, rng
                )
                for model_name in model_names:
                    result = calculate_goals(
                        model_scores[model_name][calibration], weight.values[calibration],
                        model_scores[model_name][test], alpha, beta, weight.bound,
                        float(config["fcp"]["delta"]), float(config["fcp"].get("w_infinity", 1.0)),
                        "covariate_identity", bool(config["fcp"].get("optimize_delta", True)),
                        float(config["fcp"].get("eta", 1e-10)),
                    )
                    curves[(weight.name, model_name)].append(result)
                    metric_rows.append(
                        {
                            "weight": weight.name, "model": model_name, "repetition": repetition,
                            "goal1_pointwise_pass_fraction": np.mean(result.empirical_fcp <= result.goal1_bound),
                            "goal2_uniform_pass": np.all(result.empirical_fcp <= result.goal2_bound),
                            "goal3_pointwise_pass_fraction": np.mean(result.goal3_fcp <= beta),
                            "goal4_uniform_pass": np.all(result.goal4_fcp <= beta),
                        }
                    )
        metrics = pd.DataFrame(metric_rows)
        summary = _summarize_curves(curves, alpha, beta)
        run.save_metrics(metrics)
        summary.to_csv(run.path / "curves_summary.csv", index=False)
        run.save_summary({"rows": len(metrics), **metrics.mean(numeric_only=True).to_dict()})
        _plot(summary, weight_names, model_names, run.path)
        run.mark_complete()
