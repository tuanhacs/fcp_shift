from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, FormatStrFormatter


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


def _compact_sample_size(value: float, _position: int | None = None) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def _three_observed_ticks(values: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=float))
    if len(values) <= 3:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, 3)).astype(int)
    return values[indices]


def plot_asymptotic(
    summary: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path]:
    """Create one publication-ready 1 x 3 asymptotic figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "m_increases": ("m", r"Test size $m$", r"$m \to \infty$ ($n$ fixed)"),
        "n_increases": ("n", r"Calibration size $n$", r"$n \to \infty$ ($m$ fixed)"),
        "n_equals_m": ("n", r"Joint size $n=m$", r"$m=n \to \infty$"),
    }
    line_styles = {
        1: ("-", "o"),
        2: ("--", "s"),
        3: ("-.", "^"),
        4: (":", "D"),
    }

    maximum = float(
        summary[[f"goal{goal}_mean" for goal in range(1, 5)]].to_numpy().max()
    )
    y_tick_max = max(1.0, np.ceil(maximum * 5.0 - 1e-9) / 5.0)
    y_ticks = (0.0, y_tick_max / 2.0, y_tick_max)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(9.6, 3.8),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for panel, (path_name, (x_column, xlabel, title)) in enumerate(definitions.items()):
        subset = summary[summary["path"] == path_name].sort_values(x_column)
        if subset.empty:
            raise ValueError(f"Missing asymptotic results for path {path_name!r}")
        axis = axes[panel]
        for goal in range(1, 5):
            linestyle, marker = line_styles[goal]
            axis.plot(
                subset[x_column],
                subset[f"goal{goal}_mean"],
                color=COLORS[f"goal{goal}"],
                label=f"Goal {goal}",
                linewidth=2.1,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                markeredgewidth=0.7,
            )
        axis.set_xscale("log")
        axis.set_title(title, fontsize=11.5, pad=5)
        axis.set_xlabel(xlabel, fontsize=10.5, labelpad=4)
        # A tiny visual margin keeps markers at zero visible; all labeled ticks stay in R+.
        axis.set_ylim(-0.02 * y_tick_max, y_tick_max * 1.04)
        axis.xaxis.set_major_locator(
            FixedLocator(_three_observed_ticks(subset[x_column].to_numpy()))
        )
        axis.xaxis.set_major_formatter(FuncFormatter(_compact_sample_size))
        axis.xaxis.set_minor_locator(FixedLocator([]))
        axis.yaxis.set_major_locator(FixedLocator(y_ticks))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.tick_params(axis="both", which="major", labelsize=10.0, length=4)
        axis.grid(alpha=0.20, linewidth=0.65)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
    axes[0].set_ylabel(
        r"Calculated quantity", fontsize=10.5, labelpad=4
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=10.0,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.4,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.18,
        top=0.79,
        wspace=0.18,
    )

    pdf_path = output_dir / "asymptotic_1x3.pdf"
    png_path = output_dir / "asymptotic_1x3.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return pdf_path, png_path
