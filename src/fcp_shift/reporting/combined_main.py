from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .grouped import load_weight_runs


def _mean_band(axis, x, values, color, label, linestyle="-") -> None:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
    axis.fill_between(x, lower, upper, color=color, alpha=0.12)
    axis.plot(x, mean, color=color, linewidth=2, linestyle=linestyle, label=label)


def _plot_forward(axis, arrays: dict[str, np.ndarray], title: str, legend: bool) -> None:
    alpha = arrays["alpha"]
    _mean_band(axis, alpha, arrays["empirical_fcp"], "#111111", "Empirical FCP")
    _mean_band(axis, alpha, arrays["goal1_bound"], "#0072B2", "Goal 1 bound")
    _mean_band(axis, alpha, arrays["goal2_bound"], "#D55E00", "Goal 2 bound")
    axis.set_title(title)
    axis.set_xlabel(r"Miscoverage $\alpha$")
    axis.set_ylabel("FCP / bound")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    if legend:
        axis.legend(fontsize=7, loc="upper left")


def _plot_inverse(axis, arrays: dict[str, np.ndarray], title: str, legend: bool) -> None:
    beta = arrays["beta"]
    axis.plot(
        beta, beta, color="#111111", linewidth=2, linestyle="--", label=r"Target $\beta$"
    )
    _mean_band(axis, beta, arrays["goal3_fcp"], "#009E73", "Goal 3 FCP")
    _mean_band(axis, beta, arrays["goal4_fcp"], "#CC79A7", "Goal 4 FCP")
    axis.set_title(title)
    axis.set_xlabel(r"Target FCP $\beta$")
    axis.set_ylabel("Empirical FCP")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    if legend:
        axis.legend(fontsize=7, loc="upper left")


def _display_name(dataset: dict[str, Any]) -> str:
    return str(dataset.get("title", dataset["name"])).replace("_", " ").title()


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
        figsize=(4.2 * 2 * dataset_count, 8.0),
        squeeze=False,
    )
    for column, name in enumerate(selected):
        title = _display_name(covariate_datasets[name])
        _plot_forward(
            axes[0, column], curves[("covariate", name)], f"{title}: Goals 1-2", column == 0
        )
        _plot_forward(
            axes[1, column], curves[("transport", name)], f"{title}: Goals 1-2", column == 0
        )
        inverse_column = dataset_count + column
        _plot_inverse(
            axes[0, inverse_column],
            curves[("covariate", name)],
            f"{title}: Goals 3-4",
            column == 0,
        )
        _plot_inverse(
            axes[1, inverse_column],
            curves[("transport", name)],
            f"{title}: Goals 3-4",
            column == 0,
        )

    axes[0, 0].annotate(
        "Covariate Shift",
        xy=(-0.30, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    axes[1, 0].annotate(
        f"Transport Shift (rho={rho:g})",
        xy=(-0.30, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    figure.suptitle(f"FCP guarantees with {weight} weight", fontsize=15)
    figure.tight_layout(rect=(0.025, 0.0, 1.0, 0.97))

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
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path
