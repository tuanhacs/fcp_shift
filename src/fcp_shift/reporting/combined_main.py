from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter

from .grouped import load_weight_runs


_PAPER_FONT_SIZE = 10.0
_PAPER_TITLE_SIZE = 11.0
_PAPER_LINE_WIDTH = 2.1
_AXIS_TICKS = (0.0, 0.5, 1.0)
_FIXED_ALPHA_LABEL = (
    r"$\widehat{\mathrm{FCP}}_{\mathrm{fix}}^{w,\delta,\alpha}$"
)
_UNIFORM_ALPHA_LABEL = (
    r"$\widehat{\mathrm{FCP}}_{\mathrm{unif}}^{w,\delta,\alpha}$"
)
_FIXED_BETA_LABEL = (
    r"$\widehat{\mathrm{FCP}}_{\mathrm{fix}}^{w,\delta,\beta}$"
)
_UNIFORM_BETA_LABEL = (
    r"$\widehat{\mathrm{FCP}}_{\mathrm{unif}}^{w,\delta,\beta}$"
)


def _mean_line(axis, x, values, color, label, linestyle="-") -> None:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if values.shape[0] > 1:
        std = values.std(axis=0, ddof=1)
        axis.fill_between(
            x,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
    axis.plot(
        x,
        mean,
        color=color,
        linewidth=_PAPER_LINE_WIDTH,
        linestyle=linestyle,
        label=label,
        zorder=2,
    )


def _format_axis(axis, *, forward: bool, y_tick_max: float = 1.0) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, y_tick_max if forward else y_tick_max * 1.05)
    axis.xaxis.set_major_locator(FixedLocator(_AXIS_TICKS))
    axis.yaxis.set_major_locator(FixedLocator((0.0, y_tick_max / 2.0, y_tick_max)))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.tick_params(axis="both", which="major", labelsize=_PAPER_FONT_SIZE, length=3)
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
) -> None:
    alpha = arrays["alpha"]
    _mean_line(axis, alpha, arrays["empirical_fcp"], "#111111", "Empirical FCP")
    _mean_line(
        axis, alpha, arrays["goal1_bound"], "#0072B2", _FIXED_ALPHA_LABEL
    )
    _mean_line(
        axis, alpha, arrays["goal2_bound"], "#D55E00", _UNIFORM_ALPHA_LABEL
    )
    axis.set_title(title, fontsize=_PAPER_TITLE_SIZE, pad=3)
    axis.set_xlabel(r"Miscoverage $\alpha$", fontsize=_PAPER_FONT_SIZE, labelpad=2)
    if show_ylabel:
        axis.set_ylabel("FCP / bound", fontsize=_PAPER_FONT_SIZE, labelpad=2)
    _format_axis(axis, forward=True, y_tick_max=1.0)
    if legend:
        axis.legend(
            fontsize=8.0,
            loc="upper left",
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
) -> None:
    beta = arrays["beta"]
    axis.plot(
        beta,
        beta,
        color="#111111",
        linewidth=_PAPER_LINE_WIDTH,
        linestyle="--",
        label=r"Target $\beta$",
    )
    _mean_line(axis, beta, arrays["goal3_fcp"], "#009E73", _FIXED_BETA_LABEL)
    _mean_line(axis, beta, arrays["goal4_fcp"], "#CC79A7", _UNIFORM_BETA_LABEL)
    axis.set_title(title, fontsize=_PAPER_TITLE_SIZE, pad=3)
    axis.set_xlabel(r"Target FCP $\beta$", fontsize=_PAPER_FONT_SIZE, labelpad=2)
    if show_ylabel:
        axis.set_ylabel("Empirical FCP", fontsize=_PAPER_FONT_SIZE, labelpad=2)
    _format_axis(axis, forward=False)
    if legend:
        axis.legend(
            fontsize=8.0,
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=2.0,
            handletextpad=0.4,
        )


def _display_name(dataset: dict[str, Any]) -> str:
    name = str(dataset.get("title", dataset["name"]))
    canonical_names = {
        "fashion_mnist": "Fashion-MNIST",
        "year": "YearPredictionMSD",
    }
    return canonical_names.get(name, name.replace("_", " ").title())


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

    if datasets is None:
        selected = []
        for name in covariate_datasets:
            if name not in transport_names:
                continue
            cov_directory = covariate_root / "covariate_shift" / name / weight
            trans_directory = (
                transport_root
                / "transport_shift"
                / name
                / weight
                / f"rho_{rho:.2f}"
            )
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
        cov_directory = covariate_root / "covariate_shift" / name / weight
        trans_directory = (
            transport_root
            / "transport_shift"
            / name
            / weight
            / f"rho_{rho:.2f}"
        )
        covariate = load_weight_runs(cov_directory)
        transport = load_weight_runs(trans_directory)
        if covariate is None:
            raise FileNotFoundError(f"Missing covariate curves below {cov_directory}")
        if transport is None:
            raise FileNotFoundError(f"Missing transport curves below {trans_directory}")
        curves[("covariate", name)] = covariate
        curves[("transport", name)] = transport

    dataset_count = len(selected)
    figure, axes = plt.subplots(
        2,
        2 * dataset_count,
        # A 2x4 figure needs more than a nominal 7-inch canvas; it can be
        # scaled to \textwidth in LaTeX without losing quality because PDF is vector.
        figsize=(6.0 * dataset_count, 4.8),
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
        )
        _plot_forward(
            axes[1, column],
            curves[("transport", name)],
            "",
            False,
            column == 0,
        )
        inverse_column = dataset_count + column
        _plot_inverse(
            axes[0, inverse_column],
            curves[("covariate", name)],
            title,
            column == 0,
            column == 0,
        )
        _plot_inverse(
            axes[1, inverse_column],
            curves[("transport", name)],
            "",
            False,
            column == 0,
        )

    axes[0, 0].annotate(
        "Covariate Shift",
        xy=(-0.33, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=_PAPER_TITLE_SIZE,
        fontweight="bold",
    )
    axes[1, 0].annotate(
        f"Transport Shift (rho={rho:g})",
        xy=(-0.33, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=_PAPER_TITLE_SIZE,
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
