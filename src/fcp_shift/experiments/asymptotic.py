from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fcp_shift.data.simulation import simulate_heteroscedastic_regression
from fcp_shift.experiments.common import calculate_goals
from fcp_shift.models import conformity_scores, fit_model
from fcp_shift.reporting.plots import plot_asymptotic
from fcp_shift.reporting.serialization import RunDirectory
from fcp_shift.reproducibility import stable_seed
from fcp_shift.shifts import sample_covariate_shift
from fcp_shift.weights import fit_weight

LOGGER = logging.getLogger(__name__)


def run_asymptotic(config: dict[str, Any], force: bool = False) -> None:
    output_root = Path(config.get("output", {}).get("root", "outputs"))
    simulation = config["simulation"]
    repetitions = int(config["experiment"]["repetitions"])
    alpha_value = float(config["asymptotic"]["alpha"])
    beta_value = float(config["asymptotic"]["beta"])
    grid_values = [int(value) for value in config["asymptotic"]["grid"]]
    fixed_n = int(config["asymptotic"]["fixed_n"])
    fixed_m = int(config["asymptotic"]["fixed_m"])
    model_seed = int(config.get("model", {}).get("seed", 2026))

    for seed in config["experiment"]["seeds"]:
        run = RunDirectory(output_root / "asymptotic" / "exponential" / f"seed_{seed}")
        if run.complete and not force:
            summary_path = run.path / "summary.csv"
            if summary_path.exists():
                plot_asymptotic(pd.read_csv(summary_path), run.path)
                LOGGER.info("Refreshed asymptotic figure from %s", summary_path)
            LOGGER.info("Skipping completed calculations in %s", run.path)
            continue
        rng = np.random.default_rng(stable_seed("simulation", seed))
        x_train, y_train = simulate_heteroscedastic_regression(
            int(simulation["n_train"]),
            rng,
            int(simulation.get("dimension", 4)),
            simulation.get("coefficients"),
            float(simulation.get("heteroscedastic_scale", 0.75)),
            float(simulation.get("minimum_noise", 0.25)),
        )
        x_source, y_source = simulate_heteroscedastic_regression(
            int(simulation["source_pool_size"]),
            rng,
            int(simulation.get("dimension", 4)),
            simulation.get("coefficients"),
            float(simulation.get("heteroscedastic_scale", 0.75)),
            float(simulation.get("minimum_noise", 0.25)),
        )
        model = fit_model("regression", x_train, y_train, config.get("model", {}), model_seed)
        scores = conformity_scores(model, x_source, y_source, "regression")
        weight_config = config["weights"][0]
        if weight_config["name"] != "exponential":
            raise ValueError("The main asymptotic experiment requires exponential weight")
        fitted_weight = fit_weight(weight_config, x_train, x_source, scores)
        run.initialize(
            config,
            {
                "experiment": "asymptotic",
                "weight": fitted_weight.metadata,
                "seed": seed,
                "g_mode": "algorithm_1",
                "clipped_to_unit_interval": False,
            },
        )

        paths = {
            "m_increases": [(fixed_n, value) for value in grid_values],
            "n_increases": [(value, fixed_m) for value in grid_values],
            "n_equals_m": [(value, value) for value in grid_values],
        }
        rows = []
        for path_name, sizes in paths.items():
            for grid_index, (n, m) in enumerate(sizes):
                LOGGER.info("%s: n=%s, m=%s", path_name, n, m)
                for repetition in range(repetitions):
                    rep_rng = np.random.default_rng(
                        stable_seed("asymptotic", seed, path_name, grid_index, repetition)
                    )
                    calibration, test = sample_covariate_shift(
                        len(scores), fitted_weight.values, n, m, rep_rng
                    )
                    result = calculate_goals(
                        scores[calibration],
                        fitted_weight.values[calibration],
                        scores[test],
                        np.asarray([alpha_value]),
                        np.asarray([beta_value]),
                        fitted_weight.bound,
                        float(config["fcp"]["delta"]),
                        float(config["fcp"].get("w_infinity", 1.0)),
                        "algorithm_1",
                        bool(config["fcp"].get("optimize_delta", True)),
                        float(config["fcp"].get("eta", 1e-10)),
                    )
                    rows.append(
                        {
                            "path": path_name,
                            "grid_index": grid_index,
                            "n": n,
                            "m": m,
                            "repetition": repetition,
                            "goal1": float(result.goal1_bound[0]),
                            "goal2": float(result.goal2_bound[0]),
                            "goal3": float(result.goal3_alpha[0]),
                            "goal4": float(result.goal4_alpha[0]),
                        }
                    )
        metrics = pd.DataFrame(rows)
        aggregation = {
            f"goal{goal}_{stat}": (f"goal{goal}", stat)
            for goal in range(1, 5)
            for stat in ["mean", "std"]
        }
        summary = metrics.groupby(["path", "grid_index", "n", "m"], as_index=False).agg(
            **aggregation
        )
        run.save_metrics(metrics)
        summary.to_csv(run.path / "summary.csv", index=False)
        run.save_summary(
            {
                "rows": len(metrics),
                "alpha": alpha_value,
                "beta": beta_value,
                "maximum_calculated_quantity": float(
                    metrics[["goal1", "goal2", "goal3", "goal4"]].max().max()
                ),
            }
        )
        plot_asymptotic(summary, run.path)
        run.mark_complete()
        LOGGER.info("Completed %s", run.path)
