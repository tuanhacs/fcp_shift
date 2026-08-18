from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

WEIGHT_COLORS = {
    "exponential": "#0072B2",
    "quadratic": "#D55E00",
    "mahalanobis": "#009E73",
}

CURVE_NAMES = (
    "empirical_fcp",
    "goal1_bound",
    "goal2_bound",
    "goal3_alpha",
    "goal4_alpha",
    "goal3_fcp",
    "goal4_fcp",
)


def _load_weight_runs(weight_directory: Path) -> dict[str, np.ndarray] | None:
    files = sorted(weight_directory.glob("seed_*/curves.npz"))
    if not files:
        return None
    combined: dict[str, list[np.ndarray]] = {name: [] for name in CURVE_NAMES}
    alpha = None
    beta = None
    for path in files:
        with np.load(path) as data:
            current_alpha = np.asarray(data["alpha"], dtype=float)
            current_beta = np.asarray(data["beta"], dtype=float)
            if alpha is None:
                alpha, beta = current_alpha, current_beta
            elif not np.array_equal(alpha, current_alpha) or not np.array_equal(
                beta, current_beta
            ):
                raise ValueError(f"Incompatible grids in {path}")
            for name in CURVE_NAMES:
                combined[name].append(np.asarray(data[name], dtype=float))
    return {
        "alpha": alpha,
        "beta": beta,
        **{name: np.concatenate(values, axis=0) for name, values in combined.items()},
    }


def _color(weight: str, index: int) -> str:
    if weight in WEIGHT_COLORS:
        return WEIGHT_COLORS[weight]
    return plt.get_cmap("tab10")(index % 10)


def _plot_mean_band(axis, x, values, color, label, linestyle="-"):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0)
    lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
    axis.fill_between(x, lower, upper, color=color, alpha=0.10)
    axis.plot(x, mean, color=color, linewidth=2, linestyle=linestyle, label=label)


def plot_grouped_weights(
    results: dict[str, dict[str, np.ndarray]],
    output_directory: str | Path,
    title: str,
) -> None:
    """Create figures with every available weight shown in the same panels."""
    if not results:
        return
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))
    alpha, beta = first["alpha"], first["beta"]
    for weight, arrays in results.items():
        if not np.array_equal(alpha, arrays["alpha"]) or not np.array_equal(
            beta, arrays["beta"]
        ):
            raise ValueError(f"Incompatible grids for grouped weight {weight}")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for goal, axis in [(1, axes[0]), (2, axes[1])]:
        for index, (weight, arrays) in enumerate(results.items()):
            color = _color(weight, index)
            _plot_mean_band(
                axis,
                alpha,
                arrays[f"goal{goal}_bound"],
                color,
                f"{weight} bound",
            )
            axis.plot(
                alpha,
                np.mean(arrays["empirical_fcp"], axis=0),
                color=color,
                linewidth=1.8,
                linestyle="--",
                label=f"{weight} empirical FCP",
            )
        axis.set_title(f"Goal {goal}")
        axis.set_xlabel(r"Miscoverage level $\alpha$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("FCP / bound")
    axes[1].legend(fontsize=8, ncol=2)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_directory / "grouped_weights_forward_goals_1_2.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for goal, axis in [(3, axes[0]), (4, axes[1])]:
        axis.plot(beta, beta, color="#111111", linestyle="--", linewidth=2, label=r"Target $\beta$")
        for index, (weight, arrays) in enumerate(results.items()):
            _plot_mean_band(
                axis,
                beta,
                arrays[f"goal{goal}_fcp"],
                _color(weight, index),
                weight,
            )
        axis.set_title(f"Goal {goal}")
        axis.set_xlabel(r"Target FCP $\beta$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical FCP")
    axes[1].legend(fontsize=9)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_directory / "grouped_weights_inverse_goals_3_4.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for goal, axis in [(3, axes[0]), (4, axes[1])]:
        for index, (weight, arrays) in enumerate(results.items()):
            _plot_mean_band(
                axis,
                beta,
                arrays[f"goal{goal}_alpha"],
                _color(weight, index),
                weight,
            )
        axis.set_title(f"Goal {goal}")
        axis.set_xlabel(r"Target FCP $\beta$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"Selected miscoverage $\alpha$")
    axes[1].legend(fontsize=9)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_directory / "grouped_weights_selected_alpha_goals_3_4.pdf", bbox_inches="tight")
    plt.close(figure)


def make_grouped_figures(config: dict[str, Any]) -> list[Path]:
    kind = config["experiment"]["kind"]
    if kind not in {"covariate_shift", "transport_shift"}:
        raise ValueError("Grouped weight figures apply only to shift experiments")
    root = Path(config.get("output", {}).get("root", "outputs"))
    figure_root = root / "main_figures" / kind
    generated: list[Path] = []
    weight_names = [item["name"] for item in config["weights"]]

    for dataset in config["datasets"]:
        dataset_name = dataset["name"]
        if kind == "covariate_shift":
            groups = [(None, root / kind / dataset_name)]
        else:
            groups = [
                (float(rho), root / kind / dataset_name)
                for rho in config["transport"]["rhos"]
            ]
        for rho, dataset_directory in groups:
            results = {}
            for weight in weight_names:
                weight_directory = dataset_directory / weight
                if rho is not None:
                    weight_directory = weight_directory / f"rho_{rho:.2f}"
                loaded = _load_weight_runs(weight_directory)
                if loaded is not None:
                    results[weight] = loaded
            if not results:
                LOGGER.warning(
                    "No completed curves found for dataset=%s rho=%s", dataset_name, rho
                )
                continue
            destination = figure_root / dataset_name
            title = dataset_name
            if rho is not None:
                destination = destination / f"rho_{rho:.2f}"
                title = f"{dataset_name} — rho={rho:.2f}"
            plot_grouped_weights(results, destination, title)
            generated.extend(sorted(destination.glob("grouped_weights_*.pdf")))
            LOGGER.info(
                "Grouped %d weights for dataset=%s rho=%s", len(results), dataset_name, rho
            )
    return generated
