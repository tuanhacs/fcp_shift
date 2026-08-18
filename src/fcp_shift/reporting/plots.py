from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "goal1": "#0072B2",
    "goal2": "#D55E00",
    "goal3": "#009E73",
    "goal4": "#CC79A7",
    "empirical": "#111111",
}


def _mean_band(axis, x, values, label, color):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0)
    lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
    axis.fill_between(x, lower, upper, color=color, alpha=0.15)
    axis.plot(x, mean, color=color, linewidth=2, label=label)


def plot_goal_results(
    output_dir: str | Path,
    alpha: np.ndarray,
    beta: np.ndarray,
    arrays: dict[str, np.ndarray],
    title_suffix: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    _mean_band(axis, alpha, arrays["empirical_fcp"], "Empirical FCP", COLORS["empirical"])
    _mean_band(axis, alpha, arrays["goal1_bound"], "Goal 1", COLORS["goal1"])
    _mean_band(axis, alpha, arrays["goal2_bound"], "Goal 2", COLORS["goal2"])
    axis.set(xlabel=r"Miscoverage level $\alpha$", ylabel="FCP / bound", title=title_suffix)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "forward_goals_1_2.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(beta, beta, color=COLORS["empirical"], linestyle="--", label=r"Target $\beta$")
    _mean_band(axis, beta, arrays["goal3_fcp"], "Goal 3", COLORS["goal3"])
    _mean_band(axis, beta, arrays["goal4_fcp"], "Goal 4", COLORS["goal4"])
    axis.set(xlabel=r"Target FCP $\beta$", ylabel="Empirical FCP", title=title_suffix)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "inverse_goals_3_4.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    _mean_band(axis, beta, arrays["goal3_alpha"], "Goal 3", COLORS["goal3"])
    _mean_band(axis, beta, arrays["goal4_alpha"], "Goal 4", COLORS["goal4"])
    axis.set(
        xlabel=r"Target FCP $\beta$",
        ylabel=r"Selected miscoverage $\alpha$",
        title=title_suffix,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "selected_alpha_goals_3_4.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_asymptotic(summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "m_increases": ("m", r"Test size $m$ ($n$ fixed)"),
        "n_increases": ("n", r"Calibration size $n$ ($m$ fixed)"),
        "n_equals_m": ("n", r"Joint size $n=m$"),
    }
    for path_name, (x_column, xlabel) in definitions.items():
        subset = summary[summary["path"] == path_name].sort_values(x_column)
        figure, axis = plt.subplots(figsize=(8, 5))
        for goal in range(1, 5):
            axis.plot(
                subset[x_column],
                subset[f"goal{goal}_mean"],
                marker="o",
                color=COLORS[f"goal{goal}"],
                label=f"Goal {goal}",
            )
            axis.fill_between(
                subset[x_column],
                np.maximum(subset[f"goal{goal}_mean"] - subset[f"goal{goal}_std"], 0.0),
                subset[f"goal{goal}_mean"] + subset[f"goal{goal}_std"],
                color=COLORS[f"goal{goal}"],
                alpha=0.12,
            )
        axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"Calculated quantity in $\mathbb{R}^{+}$")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"asymptotic_{path_name}.pdf", bbox_inches="tight")
        plt.close(figure)

