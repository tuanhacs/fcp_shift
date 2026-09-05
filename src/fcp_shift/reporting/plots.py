from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, FormatStrFormatter

from .style import figure_size, font_size


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

    figure, axis = plt.subplots(figsize=figure_size((8, 4.8)))
    _mean_band(axis, alpha, arrays["empirical_fcp"], "Empirical FCP", COLORS["empirical"])
    _mean_band(axis, alpha, arrays["goal1_bound"], "Goal 1", COLORS["goal1"])
    _mean_band(axis, alpha, arrays["goal2_bound"], "Goal 2", COLORS["goal2"])
    axis.set(xlabel=r"Miscoverage level $\alpha$", ylabel="FCP / bound", title=title_suffix)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "forward_goals_1_2.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=figure_size((8, 4.8)))
    axis.plot(beta, beta, color=COLORS["empirical"], linestyle="--", label=r"Target $\beta$")
    _mean_band(axis, beta, arrays["goal3_fcp"], "Goal 3", COLORS["goal3"])
    _mean_band(axis, beta, arrays["goal4_fcp"], "Goal 4", COLORS["goal4"])
    axis.set(xlabel=r"Target FCP $\beta$", ylabel="Empirical FCP", title=title_suffix)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "inverse_goals_3_4.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=figure_size((8, 4.8)))
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
    summary: pd.DataFrame,
    output_dir: str | Path,
    alpha: float | None = None,
    beta: float | None = None,
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
    reference_values = [value for value in (alpha, beta) if value is not None]
    if reference_values:
        maximum = max(maximum, *reference_values)
    # Keep a zero baseline, but avoid wasting vertical space up to 1 when all
    # calculated quantities are much smaller. Use a clean 0.1 upper tick and
    # leave at least half a tick of headroom above the largest curve.
    y_tick_max = max(0.1, np.ceil(maximum * 10.0 - 1e-9) / 10.0)
    if y_tick_max - maximum < 0.05:
        y_tick_max += 0.1
    y_ticks = (0.0, y_tick_max / 2.0, y_tick_max)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=figure_size((16, 4.5)),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for panel, (path_name, (x_column, xlabel, title)) in enumerate(definitions.items()):
        subset = summary[summary["path"] == path_name].sort_values(x_column)
        if subset.empty:
            raise ValueError(f"Missing asymptotic results for path {path_name!r}")
        axis = axes[panel]
        if alpha is not None:
            axis.axhline(
                alpha,
                color="#4D4D4D",
                linewidth=1.6,
                linestyle=(0, (5, 2)),
                label=rf"$\alpha={alpha:g}$",
                zorder=1,
            )
        if beta is not None:
            axis.axhline(
                beta,
                color="#7A7A7A",
                linewidth=1.6,
                linestyle=(0, (1.5, 2)),
                label=rf"$\beta={beta:g}$",
                zorder=1,
            )
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
                zorder=2,
            )
        axis.set_xscale("log")
        axis.set_title(title, fontsize=font_size("title", 11.5), pad=5)
        axis.set_xlabel(xlabel, fontsize=font_size("label", 10.5), labelpad=4)
        axis.set_ylim(0.0, y_tick_max * 1.02)
        axis.xaxis.set_major_locator(
            FixedLocator(_three_observed_ticks(subset[x_column].to_numpy()))
        )
        axis.xaxis.set_major_formatter(FuncFormatter(_compact_sample_size))
        axis.xaxis.set_minor_locator(FixedLocator([]))
        axis.yaxis.set_major_locator(FixedLocator(y_ticks))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        axis.tick_params(
            axis="both", which="major", labelsize=font_size("tick", 10.0), length=4
        )
        axis.grid(alpha=0.20, linewidth=0.65)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
    axes[0].set_ylabel(
        r"Calculated quantity", fontsize=font_size("label", 10.5), labelpad=4
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend_items = dict(zip(labels, handles))
    ordered_labels = [f"Goal {goal}" for goal in range(1, 5)]
    if alpha is not None:
        ordered_labels.append(rf"$\alpha={alpha:g}$")
    if beta is not None:
        ordered_labels.append(rf"$\beta={beta:g}$")
    axes[0].legend(
        [legend_items[label] for label in ordered_labels],
        ordered_labels,
        loc="center right",
        ncol=2,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#C8C8C8",
        fontsize=font_size("legend", 10.0),
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=2.1,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.18,
        top=0.88,
        wspace=0.18,
    )

    pdf_path = output_dir / "asymptotic_1x3.pdf"
    png_path = output_dir / "asymptotic_1x3.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return pdf_path, png_path
