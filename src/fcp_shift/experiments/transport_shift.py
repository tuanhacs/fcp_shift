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
from fcp_shift.shifts import build_score_transport
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def run_transport_shift(config: dict[str, Any], force: bool = False) -> None:
    output_root = Path(config.get("output", {}).get("root", "outputs"))
    alpha = grid(config["fcp"]["alpha_grid"])
    beta = grid(config["fcp"]["beta_grid"])
    n = int(config["sample_sizes"]["n_calibration"])
    m = int(config["sample_sizes"]["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    model_seed = int(config.get("model", {}).get("seed", 2026))
    strata_count = int(config["transport"].get("strata", 5))

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
            base_weight = fit_weight(weight_config, dataset.x_train, dataset.x_source, scores)
            transport = build_score_transport(scores, base_weight.values, strata_count)
            for rho in config["transport"]["rhos"]:
                rho = float(rho)
                transport_weights = transport.weights(rho)
                bound = float(np.max(transport_weights))
                for seed in config["experiment"]["seeds"]:
                    run = RunDirectory(
                        output_root
                        / "transport_shift"
                        / dataset.name
                        / base_weight.name
                        / f"rho_{rho:.2f}"
                        / f"seed_{seed}"
                    )
                    if run.complete and not force:
                        LOGGER.info("Skipping completed run %s", run.path)
                        continue
                    run.initialize(
                        config,
                        {
                            "experiment": "transport_shift",
                            "dataset": dataset.name,
                            "base_weight": base_weight.metadata,
                            "transport_weight_bound": bound,
                            "rho": rho,
                            "stratum_probabilities_calibration": transport.p,
                            "stratum_probabilities_test": transport.q,
                            "permutation": transport.permutation,
                            "seed": seed,
                            "g_mode": "algorithm_1",
                        },
                    )
                    results = []
                    rows = []
                    for repetition in range(repetitions):
                        rng = np.random.default_rng(
                            stable_seed("transport", dataset.name, base_weight.name, rho, seed, repetition)
                        )
                        calibration = rng.choice(len(scores), size=n, replace=True)
                        test_scores = transport.sample_test_scores(m, rho, rng)
                        result = calculate_goals(
                            scores[calibration],
                            transport_weights[calibration],
                            test_scores,
                            alpha,
                            beta,
                            bound,
                            float(config["fcp"]["delta"]),
                            float(config["fcp"].get("w_infinity", 1.0)),
                            "algorithm_1",
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
                    plot_goal_results(
                        run.path,
                        alpha,
                        beta,
                        arrays,
                        f"{dataset.name} — {base_weight.name}, rho={rho:.2f}",
                    )
                    run.mark_complete()
                    LOGGER.info("Completed %s", run.path)

