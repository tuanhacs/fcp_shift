from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from fcp_shift.ablations.baselines import _plot_dataset
from fcp_shift.ablations.common import scoped_ablation_path
from fcp_shift.ablations.corollary import _plot as _plot_corollary
from fcp_shift.ablations.delta import _plot as _plot_delta
from fcp_shift.ablations.efficiency import _plot as _plot_efficiency
from fcp_shift.ablations.models import _plot_grid as _plot_models
from fcp_shift.ablations.timing import _plot as _plot_timing
from fcp_shift.ablations.weights import _plot_grid as _plot_weight_grid
from fcp_shift.experiments.common import grid


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Saved result not found: {path}. Run the experiment first, using the "
            "same dataset/weight/seed filters as this plot command."
        )
    return path


def _root(config: dict[str, Any]) -> Path:
    return Path(config.get("output", {}).get("root", "outputs"))


def _replot_corollary(config: dict[str, Any]) -> list[Path]:
    generated = []
    datasets = [item["name"] for item in config["datasets"]]
    alphas = [float(value) for value in config["ablation"]["alphas"]]
    for seed in config["experiment"]["seeds"]:
        run = scoped_ablation_path(_root(config), "corollary", seed, config)
        frame = pd.read_csv(_required(run / "metrics.csv"))
        output = run / "corollary_convergence_2x3.pdf"
        _plot_corollary(frame, datasets, alphas, output)
        generated.append(output)
    return generated


def _replot_delta(config: dict[str, Any]) -> list[Path]:
    generated = []
    datasets = [item["name"] for item in config["datasets"]]
    for seed in config["experiment"]["seeds"]:
        run = scoped_ablation_path(_root(config), "delta", seed, config)
        frame = pd.read_csv(_required(run / "metrics.csv"))
        for bound_type in ("fixed", "uniform"):
            output = run / f"delta_{bound_type}_2x3.pdf"
            _plot_delta(frame, datasets, bound_type, output)
            generated.append(output)
    return generated


def _replot_models(config: dict[str, Any]) -> list[Path]:
    generated = []
    models = [item["name"] for item in config["models"]]
    weights = [item["name"] for item in config["weights"]]
    for dataset in (item["name"] for item in config["datasets"]):
        for seed in config["experiment"]["seeds"]:
            run = scoped_ablation_path(_root(config), "models", seed, config, dataset)
            summary = pd.read_csv(_required(run / "curves_summary.csv"))
            generated.append(_plot_models(summary, weights, models, dataset, run))
    return generated


def _replot_efficiency(config: dict[str, Any]) -> list[Path]:
    generated = []
    weights = [item["name"] for item in config["weights"]]
    columns = max(
        sum(item["task"] == "regression" for item in config["datasets"]),
        sum(item["task"] == "classification" for item in config["datasets"]),
    )
    for seed in config["experiment"]["seeds"]:
        run = scoped_ablation_path(_root(config), "efficiency", seed, config)
        summary = pd.read_csv(_required(run / "efficiency_curves_summary.csv"))
        _plot_efficiency(summary, config["datasets"], weights, run)
        generated.append(run / f"efficiency_vs_fcp_2x{columns}.pdf")
    return generated


def _replot_timing(config: dict[str, Any]) -> list[Path]:
    generated = []
    datasets = [item["name"] for item in config["datasets"]]
    models = [item["name"] for item in config["models"]]
    for seed in config["experiment"]["seeds"]:
        run = scoped_ablation_path(_root(config), "timing", seed, config)
        frame = pd.read_csv(_required(run / "metrics.csv"))
        output = run / "inference_time_3x3.pdf"
        _plot_timing(frame, datasets, models, output)
        generated.append(output)
    return generated


def _replot_weights(config: dict[str, Any]) -> list[Path]:
    generated = []
    alpha = grid(config["fcp"]["alpha_grid"])
    beta = grid(config["fcp"]["beta_grid"])
    datasets = [item["name"] for item in config["datasets"]]
    weights = [item["name"] for item in config["weights"]]
    for seed in config["experiment"]["seeds"]:
        run = scoped_ablation_path(_root(config), "weight_families", seed, config)
        summary = pd.read_csv(_required(run / "weight_curves_summary.csv"))
        generated.extend(
            _plot_weight_grid(summary, datasets, weights, alpha, beta, run)
        )
    return generated


def _saved_baseline_curves(
    summary: pd.DataFrame, dataset: str, weights: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for weight in weights:
        result[weight] = {}
        for curve in ("forward", "dkw_inverse", "cojer_inverse"):
            subset = summary[
                (summary["dataset"] == dataset)
                & (summary["weight"] == weight)
                & (summary["curve"] == curve)
            ].sort_values("x")
            if subset.empty:
                raise ValueError(f"Missing saved baseline curve for {dataset}/{weight}/{curve}")
            result[weight][curve] = subset["mean"].to_numpy(dtype=float)[None, :]
    return result


def _replot_baselines(config: dict[str, Any]) -> list[Path]:
    generated = []
    alpha = grid(config["fcp"]["alpha_grid"])
    beta = grid(config["fcp"]["beta_grid"])
    datasets = [item["name"] for item in config["datasets"]]
    weights = [item["name"] for item in config["weights"]]
    for seed in config["experiment"]["seeds"]:
        run = scoped_ablation_path(_root(config), "baselines", seed, config)
        summary = pd.read_csv(_required(run / "baseline_curves_summary.csv"))
        with np.load(_required(run / "curves.npz")) as arrays:
            dkw_bound = np.asarray(arrays["dkw_bound"], dtype=float)
            cojer_bound = np.asarray(arrays["cojer_bound"], dtype=float)
        for dataset in datasets:
            curves = _saved_baseline_curves(summary, dataset, weights)
            _plot_dataset(dataset, alpha, beta, curves, dkw_bound, cojer_bound, run)
            generated.extend(
                [
                    run / f"baselines_forward_{dataset}.pdf",
                    run / f"baselines_inverse_{dataset}.pdf",
                ]
            )
    return generated


REPLOTTERS: dict[str, Callable[[dict[str, Any]], list[Path]]] = {
    "ablation_corollary": _replot_corollary,
    "ablation_delta": _replot_delta,
    "ablation_efficiency": _replot_efficiency,
    "ablation_models": _replot_models,
    "ablation_timing": _replot_timing,
    "ablation_weights": _replot_weights,
    "ablation_baselines": _replot_baselines,
}


def replot_ablation(config: dict[str, Any]) -> list[Path]:
    kind = config["experiment"]["kind"]
    if kind not in REPLOTTERS:
        raise ValueError(f"plot supports ablation configurations only, got {kind!r}")
    return REPLOTTERS[kind](config)
