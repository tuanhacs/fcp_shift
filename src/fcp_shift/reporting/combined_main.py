from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter

from .grouped import load_weight_runs
from fcp_shift.weights import direction_variant
from .labels import (
    display_dataset_name,
    FIXED_ALPHA_COLOR,
    FIXED_ALPHA_PASS_LABEL,
    FIXED_BETA_COLOR,
    FIXED_BETA_PASS_LABEL,
    TARGET_COLOR,
    UNIFORM_ALPHA_COLOR,
    UNIFORM_ALPHA_PASS_LABEL,
    UNIFORM_BETA_COLOR,
    UNIFORM_BETA_PASS_LABEL,
)
from .style import figure_size, font_size, set_probability_limits


_PAPER_FONT_SIZE = 10.0
_PAPER_TITLE_SIZE = 11.0
_PAPER_LINE_WIDTH = 2.1
_AXIS_TICKS = (0.0, 0.5, 1.0)


def _guarantee_line(axis, x, indicators, color, label, linestyle="-") -> np.ndarray:
    values = np.asarray(indicators, dtype=float)
    if values.ndim == 1:
        values = np.repeat(values[:, None], len(x), axis=1)
    probability = values.mean(axis=0)
    axis.plot(
        x, probability, color=color, linewidth=_PAPER_LINE_WIDTH,
        linestyle=linestyle, label=label, zorder=2,
    )
    return probability


def _format_axis(axis, *, forward: bool, y_tick_max: float = 1.0) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, y_tick_max if forward else y_tick_max * 1.05)
    axis.xaxis.set_major_locator(FixedLocator(_AXIS_TICKS))
    axis.yaxis.set_major_locator(FixedLocator((0.0, y_tick_max / 2.0, y_tick_max)))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: "" if np.isclose(value, 0.0) else f"{value:.1f}")
    )
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=font_size("tick", _PAPER_FONT_SIZE),
        length=3,
    )
    axis.grid(True, which="major", color="#A8A8A8", alpha=0.48, linewidth=0.75)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def _plot_forward(
    axis,
    arrays: dict[str, np.ndarray],
    title: str,
    legend: bool,
    show_ylabel: bool,
    show_xlabel: bool,
    delta: float,
) -> None:
    alpha = arrays["alpha"]
    axis.axhline(
        1.0 - delta, color=TARGET_COLOR, linewidth=_PAPER_LINE_WIDTH,
        linestyle="--", label=r"Required $1-\delta$",
    )
    fixed_probability = _guarantee_line(
        axis, alpha, arrays["goal1_pass"], FIXED_ALPHA_COLOR, FIXED_ALPHA_PASS_LABEL
    )
    uniform_probability = _guarantee_line(
        axis, alpha, arrays["goal2_uniform_pass"],
        UNIFORM_ALPHA_COLOR, UNIFORM_ALPHA_PASS_LABEL
    )
    axis.set_title(title, fontsize=font_size("title", _PAPER_TITLE_SIZE), pad=3)
    if show_xlabel:
        axis.set_xlabel(
            r"Miscoverage $\alpha$",
            fontsize=font_size("label", _PAPER_FONT_SIZE),
            labelpad=2,
        )
    if show_ylabel:
        axis.set_ylabel(
            "Guarantee probability", fontsize=font_size("label", _PAPER_FONT_SIZE), labelpad=2
        )
    _format_axis(axis, forward=True, y_tick_max=1.0)
    set_probability_limits(
        axis, alpha, 1.0 - delta, fixed_probability, uniform_probability
    )
    if legend:
        axis.legend(
            fontsize=font_size("legend", 10.0),
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=2.0,
            handletextpad=0.4,
        )


def _plot_inverse(
    axis,
    arrays: dict[str, np.ndarray],
    title: str,
    legend: bool,
    show_ylabel: bool,
    show_xlabel: bool,
    delta: float,
) -> None:
    beta = arrays["beta"]
    axis.axhline(
        1.0 - delta, color=TARGET_COLOR, linewidth=_PAPER_LINE_WIDTH,
        linestyle="--", label=r"Required $1-\delta$",
    )
    fixed_probability = _guarantee_line(
        axis, beta, arrays["goal3_pass"], FIXED_BETA_COLOR, FIXED_BETA_PASS_LABEL
    )
    uniform_probability = _guarantee_line(
        axis, beta, arrays["goal4_uniform_pass"],
        UNIFORM_BETA_COLOR, UNIFORM_BETA_PASS_LABEL
    )
    axis.set_title(title, fontsize=font_size("title", _PAPER_TITLE_SIZE), pad=3)
    if show_xlabel:
        axis.set_xlabel(
            r"Target FCP $\beta$",
            fontsize=font_size("label", _PAPER_FONT_SIZE),
            labelpad=2,
        )
    if show_ylabel:
        axis.set_ylabel(
            "Guarantee probability", fontsize=font_size("label", _PAPER_FONT_SIZE), labelpad=2
        )
    _format_axis(axis, forward=False)
    set_probability_limits(
        axis, beta, 1.0 - delta, fixed_probability, uniform_probability
    )
    if legend:
        axis.legend(
            fontsize=font_size("legend", 10.0),
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=2.0,
            handletextpad=0.4,
        )


def _display_name(dataset: dict[str, Any]) -> str:
    name = str(dataset.get("title", dataset["name"]))
    return display_dataset_name(name)


def make_covariate_transport_figure(
    covariate_config: dict[str, Any],
    transport_config: dict[str, Any],
    weight: str,
    rho: float,
    datasets: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build a 2 x (2D) main figure for D datasets and one weight."""
    covariate_root = Path(covariate_config.get("output", {}).get("root", "outputs"))
    transport_root = Path(transport_config.get("output", {}).get("root", "outputs"))
    covariate_datasets = {item["name"]: item for item in covariate_config["datasets"]}
    transport_names = {item["name"] for item in transport_config["datasets"]}
    covariate_weights = {item["name"]: item for item in covariate_config.get("weights", [])}
    transport_weights = {item["name"]: item for item in transport_config.get("weights", [])}
    cov_variant = direction_variant(covariate_weights.get(weight, {}))
    trans_variant = direction_variant(transport_weights.get(weight, {}))

    def _weight_directories(name: str) -> tuple[Path, Path]:
        cov = covariate_root / "covariate_shift" / name / weight
        trans = transport_root / "transport_shift" / name / weight
        if cov_variant:
            cov /= cov_variant
        if trans_variant:
            trans /= trans_variant
        return cov, trans / f"rho_{rho:.2f}"

    if datasets is None:
        selected = []
        for name in covariate_datasets:
            if name not in transport_names:
                continue
            cov_directory, trans_directory = _weight_directories(name)
            if load_weight_runs(cov_directory) is not None and load_weight_runs(trans_directory) is not None:
                selected.append(name)
    else:
        selected = list(datasets)
    if not selected:
        raise FileNotFoundError(
            "No dataset has both covariate and transport curves for the selected weight/rho"
        )

    curves: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for name in selected:
        if name not in covariate_datasets or name not in transport_names:
            raise ValueError(f"Dataset {name!r} is not present in both configurations")
        cov_directory, trans_directory = _weight_directories(name)
        covariate = load_weight_runs(cov_directory)
        transport = load_weight_runs(trans_directory)
        if covariate is None:
            raise FileNotFoundError(f"Missing covariate curves below {cov_directory}")
        if transport is None:
            raise FileNotFoundError(f"Missing transport curves below {trans_directory}")
        curves[("covariate", name)] = covariate
        curves[("transport", name)] = transport

    dataset_count = len(selected)
    covariate_delta = float(covariate_config.get("fcp", {}).get("delta", 0.1))
    transport_delta = float(transport_config.get("fcp", {}).get("delta", 0.1))
    figure, axes = plt.subplots(
        2,
        2 * dataset_count,
        # A 2x4 figure needs more than a nominal 7-inch canvas; it can be
        # scaled to \textwidth in LaTeX without losing quality because PDF is vector.
        figsize=figure_size((6.8 * dataset_count, 4.8)),
        squeeze=False,
    )
    for column, name in enumerate(selected):
        title = _display_name(covariate_datasets[name])
        _plot_forward(
            axes[0, column],
            curves[("covariate", name)],
            title,
            column == 0,
            column == 0,
            False,
            covariate_delta,
        )
        _plot_forward(
            axes[1, column],
            curves[("transport", name)],
            "",
            False,
            column == 0,
            column == 0,
            transport_delta,
        )
        inverse_column = dataset_count + column
        _plot_inverse(
            axes[0, inverse_column],
            curves[("covariate", name)],
            title,
            column == 0,
            False,
            False,
            covariate_delta,
        )
        _plot_inverse(
            axes[1, inverse_column],
            curves[("transport", name)],
            "",
            False,
            False,
            column == 0,
            transport_delta,
        )

    # A single set of y tick labels is sufficient for the shared probability
    # scale and avoids collisions with the leftmost x tick in columns 2--4.
    for row in range(2):
        for column in range(1, 2 * dataset_count):
            axes[row, column].tick_params(axis="y", labelleft=False)

    axes[0, 0].annotate(
        "Covariate Shift",
        xy=(-0.33, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=font_size("title", _PAPER_TITLE_SIZE),
        fontweight="bold",
    )
    axes[1, 0].annotate(
        f"Transport Shift",
        xy=(-0.33, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=font_size("title", _PAPER_TITLE_SIZE),
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.12,
        top=0.95,
        wspace=0.30,
        hspace=0.44,
    )

    if output_path is None:
        destination = covariate_root / "main_figures" / "combined"
        if cov_variant or trans_variant:
            destination /= f"cov_{cov_variant or 'ridge'}__trans_{trans_variant or 'ridge'}"
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"covariate_transport_{weight}_rho_{rho:.2f}_{dataset_count}datasets"
        pdf_path = destination / f"{stem}.pdf"
    else:
        pdf_path = Path(output_path)
        if pdf_path.suffix.lower() != ".pdf":
            pdf_path = pdf_path.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = pdf_path.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return pdf_path, png_path
