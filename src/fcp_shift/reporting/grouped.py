from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .style import figure_size, font_size

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


def load_weight_runs(weight_directory: Path) -> dict[str, np.ndarray] | None:
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
    for obsolete_name in (
        "grouped_weights_forward_goals_1_2.pdf",
        "grouped_weights_inverse_goals_3_4.pdf",
        "grouped_weights_selected_alpha_goals_3_4.pdf",
    ):
        (output_directory / obsolete_name).unlink(missing_ok=True)
    first = next(iter(results.values()))
    alpha, beta = first["alpha"], first["beta"]
    for weight, arrays in results.items():
        if not np.array_equal(alpha, arrays["alpha"]) or not np.array_equal(
            beta, arrays["beta"]
        ):
            raise ValueError(f"Incompatible grids for grouped weight {weight}")

    for goal in (1, 2):
        figure, axis = plt.subplots(figsize=figure_size((8, 5)))
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
        axis.set_title(f"{title} — Goal {goal}")
        axis.set_xlabel(r"Miscoverage level $\alpha$")
        axis.set_ylabel("FCP / bound")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=font_size("legend", 8), ncol=2)
        figure.tight_layout()
        figure.savefig(
            output_directory / f"grouped_weights_forward_goal_{goal}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)

    for goal in (3, 4):
        figure, axis = plt.subplots(figsize=figure_size((8, 5)))
        axis.plot(beta, beta, color="#111111", linestyle="--", linewidth=2, label=r"Target $\beta$")
        for index, (weight, arrays) in enumerate(results.items()):
            _plot_mean_band(
                axis,
                beta,
                arrays[f"goal{goal}_fcp"],
                _color(weight, index),
                weight,
            )
        axis.set_title(f"{title} — Goal {goal}")
        axis.set_xlabel(r"Target FCP $\beta$")
        axis.set_ylabel("Empirical FCP")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=font_size("legend", 9))
        figure.tight_layout()
        figure.savefig(
            output_directory / f"grouped_weights_inverse_goal_{goal}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)

    for goal in (3, 4):
        figure, axis = plt.subplots(figsize=figure_size((8, 5)))
        for index, (weight, arrays) in enumerate(results.items()):
            _plot_mean_band(
                axis,
                beta,
                arrays[f"goal{goal}_alpha"],
                _color(weight, index),
                weight,
            )
        axis.set_title(f"{title} — Goal {goal}")
        axis.set_xlabel(r"Target FCP $\beta$")
        axis.set_ylabel(r"Selected miscoverage $\alpha$")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=font_size("legend", 9))
        figure.tight_layout()
        figure.savefig(
            output_directory / f"grouped_weights_selected_alpha_goal_{goal}.pdf",
            bbox_inches="tight",
        )
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
                loaded = load_weight_runs(weight_directory)
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
