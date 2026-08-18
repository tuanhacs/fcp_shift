from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fcp_shift.data import prepare_dataset
from fcp_shift.experiments.common import calculate_goals, grid, stack_goal_results
from fcp_shift.models import conformity_scores, fit_model
from fcp_shift.reporting.plots import plot_goal_results
from fcp_shift.reporting.serialization import RunDirectory
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import sample_covariate_shift
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def run_covariate_shift(config: dict[str, Any], force: bool = False) -> None:
    output_root = Path(config.get("output", {}).get("root", "outputs"))
    alpha = grid(config["fcp"]["alpha_grid"])
    beta = grid(config["fcp"]["beta_grid"])
    sample = config["sample_sizes"]
    n, m = int(sample["n_calibration"]), int(sample["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    model_seed = int(config.get("model", {}).get("seed", 2026))

    for dataset_config in config["datasets"]:
        LOGGER.info("Preparing dataset %s", dataset_config["name"])
        dataset = prepare_dataset(dataset_config, model_seed)
        model = fit_model(
            dataset.task, dataset.x_train, dataset.y_train, config.get("model", {}), model_seed
        )
        scores = conformity_scores(
            model,
            dataset.x_source,
            dataset.y_source,
            dataset.task,
            config.get("model", {}).get("classification_score"),
        )
        for weight_config in config["weights"]:
            fitted_weight = fit_weight(
                weight_config, dataset.x_train, dataset.x_source, scores
            )
            for seed in config["experiment"]["seeds"]:
                run = RunDirectory(
                    output_root
                    / "covariate_shift"
                    / dataset.name
                    / fitted_weight.name
                    / f"seed_{seed}"
                )
                if run.complete and not force:
                    LOGGER.info("Skipping completed run %s", run.path)
                    continue
                run.initialize(
                    config,
                    {
                        "experiment": "covariate_shift",
                        "dataset": dataset.name,
                        "weight": fitted_weight.metadata,
                        "seed": seed,
                        "g_mode": "covariate_identity",
                    },
                )
                results = []
                rows = []
                for repetition in range(repetitions):
                    rng = np.random.default_rng(
                        stable_seed("covariate", dataset.name, fitted_weight.name, seed, repetition)
                    )
                    calibration, test = sample_covariate_shift(
                        len(scores), fitted_weight.values, n, m, rng
                    )
                    result = calculate_goals(
                        scores[calibration],
                        fitted_weight.values[calibration],
                        scores[test],
                        alpha,
                        beta,
                        fitted_weight.bound,
                        float(config["fcp"]["delta"]),
                        float(config["fcp"].get("w_infinity", 1.0)),
                        "covariate_identity",
                        bool(config["fcp"].get("optimize_delta", True)),
                        float(config["fcp"].get("eta", 1e-10)),
                    )
                    results.append(result)
                    rows.append(
                        {
                            "repetition": repetition,
                            "goal1_pointwise_pass_fraction": np.mean(
                                result.empirical_fcp <= result.goal1_bound
                            ),
                            "goal2_uniform_pass": np.all(
                                result.empirical_fcp <= result.goal2_bound
                            ),
                            "goal3_pointwise_pass_fraction": np.mean(
                                result.goal3_fcp <= beta
                            ),
                            "goal4_uniform_pass": np.all(result.goal4_fcp <= beta),
                        }
                    )
                arrays = stack_goal_results(results)
                metrics = pd.DataFrame(rows)
                run.save_metrics(metrics)
                run.save_arrays(alpha=alpha, beta=beta, **arrays)
                run.save_summary(
                    {
                        "rows": len(metrics),
                        **metrics.mean(numeric_only=True).to_dict(),
                        "fixed_constants": results[0].fixed,
                        "uniform_constants": results[0].uniform,
                    }
                )
                plot_goal_results(run.path, alpha, beta, arrays, f"{dataset.name} — {fitted_weight.name}")
                run.mark_complete()
                LOGGER.info("Completed %s", run.path)

