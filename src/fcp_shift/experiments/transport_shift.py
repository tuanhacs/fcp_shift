from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fcp_shift.experiments.common import calculate_goals, grid, stack_goal_results
from fcp_shift.reporting.plots import plot_goal_results
from fcp_shift.reporting.serialization import RunDirectory
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import build_score_transport, prepare_shift_problem
from fcp_shift.weights import direction_variant, fit_score_projection, fit_weight

LOGGER = logging.getLogger(__name__)


def run_transport_shift(config: dict[str, Any], force: bool = False) -> None:
    output_root = Path(config.get("output", {}).get("root", "outputs"))
    alpha = grid(config["fcp"]["alpha_grid"])
    beta = grid(config["fcp"]["beta_grid"])
    n = int(config["sample_sizes"]["n_calibration"])
    m = int(config["sample_sizes"]["m_test"])
    repetitions = int(config["experiment"]["repetitions"])
    strata_count = int(config["transport"].get("strata", 5))
    auxiliary_fraction = float(config.get("shift", {}).get("auxiliary_fraction", 0.2))

    for dataset_config in config["datasets"]:
        LOGGER.info("Preparing dataset %s", dataset_config["name"])
        problem = prepare_shift_problem(
            dataset_config, config.get("model", {}), auxiliary_fraction
        )
        dataset, scores = problem.dataset, problem.source_scores
        projections = {}
        for weight_config in config["weights"]:
            ridge = float(weight_config.get("ridge", 1e-3))
            direction = weight_config.get("direction", "ridge")
            direction_seed = int(weight_config.get("direction_seed", 2026))
            key = (ridge, direction, direction_seed)
            if key not in projections:
                projections[key] = fit_score_projection(
                    dataset.x_train, problem.auxiliary_features,
                    problem.auxiliary_scores, ridge,
                    method=direction, random_seed=direction_seed,
                )
            projection = projections[key]
            base_weight = fit_weight(
                weight_config, dataset.x_train, dataset.x_source, scores,
                score_projection=projection,
            )
            transport = build_score_transport(
                scores, base_weight.values, strata_count,
                stratification_values=projection.transform(dataset.x_source),
                reference_values=projection.transform(problem.auxiliary_features),
            )
            for rho in config["transport"]["rhos"]:
                rho = float(rho)
                transport_weights = transport.weights(rho)
                bound = float(np.max(transport_weights))
                for seed in config["experiment"]["seeds"]:
                    weight_root = output_root / "transport_shift" / dataset.name / base_weight.name
                    variant = direction_variant(weight_config)
                    if variant:
                        weight_root /= variant
                    run = RunDirectory(weight_root / f"rho_{rho:.2f}" / f"seed_{seed}")
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
                            "strata_cutpoints": transport.cutpoints,
                            "stratification": (
                                "auxiliary_score_projection_of_X"
                                if direction == "ridge" else "seeded_random_projection_of_X"
                            ),
                            "auxiliary_fraction": auxiliary_fraction,
                            "auxiliary_size": len(problem.auxiliary_scores),
                            "source_pool_size": len(scores),
                            "rho_zero_weight": "stratum_coarsened_exponential_tilt",
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
                        _test_indices, test_scores = transport.sample_test(m, rho, rng)
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
