from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, LinearLocator


@dataclass(frozen=True)
class PlotStyle:
    """CLI-overridable presentation settings shared by all report plots."""

    figsize: tuple[float, float] | None = None
    font_size: float | None = None
    title_font_size: float | None = None
    label_font_size: float | None = None
    tick_font_size: float | None = None
    legend_font_size: float | None = None


_ACTIVE_STYLE = PlotStyle()
_RC_KEYS = (
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
)
_RC_DEFAULTS = {key: mpl.rcParamsDefault[key] for key in _RC_KEYS}


def _compact_tick(value: float, _position: int | None = None) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.3f}".rstrip("0").rstrip(".") + "M"
    if absolute >= 1_000:
        return f"{value / 1_000:.3f}".rstrip("0").rstrip(".") + "k"
    if 0 < absolute < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def configure_plot_style(style: PlotStyle) -> None:
    global _ACTIVE_STYLE
    _ACTIVE_STYLE = style
    base = style.font_size
    updates = dict(_RC_DEFAULTS)
    if base is not None:
        updates["font.size"] = base
    if style.title_font_size is not None or base is not None:
        updates["axes.titlesize"] = style.title_font_size or base
    if style.label_font_size is not None or base is not None:
        updates["axes.labelsize"] = style.label_font_size or base
    if style.tick_font_size is not None or base is not None:
        tick_size = style.tick_font_size or base
        updates["xtick.labelsize"] = tick_size
        updates["ytick.labelsize"] = tick_size
    if style.legend_font_size is not None or base is not None:
        updates["legend.fontsize"] = style.legend_font_size or base
    mpl.rcParams.update(updates)


def figure_size(default: tuple[float, float]) -> tuple[float, float]:
    return _ACTIVE_STYLE.figsize or default


def font_size(kind: str, default: float) -> float:
    specialized = getattr(_ACTIVE_STYLE, f"{kind}_font_size")
    if specialized is not None:
        return specialized
    if _ACTIVE_STYLE.font_size is not None:
        return _ACTIVE_STYLE.font_size
    return default


def set_publication_ticks(
    axis: Axes,
    *,
    x_values: list[float] | np.ndarray | None = None,
    y_values: list[float] | np.ndarray | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
) -> None:
    """Use exactly three uncluttered major ticks on linear or log axes."""
    axis.set_xscale(xscale)
    axis.set_yscale(yscale)
    if x_values is None:
        axis.xaxis.set_major_locator(LinearLocator(3))
    else:
        values = np.unique(np.asarray(x_values, dtype=float))
        if len(values) <= 3:
            ticks = values
        else:
            ticks = values[np.rint(np.linspace(0, len(values) - 1, 3)).astype(int)]
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
    if y_values is None:
        if yscale == "log":
            low, high = axis.get_ylim()
            ticks = np.geomspace(low, high, 3)
            axis.yaxis.set_major_locator(FixedLocator(ticks))
        else:
            axis.yaxis.set_major_locator(LinearLocator(3))
        axis.yaxis.set_major_formatter(FuncFormatter(_compact_tick))
    else:
        values = np.unique(np.asarray(y_values, dtype=float))
        if yscale == "log":
            values = values[values > 0.0]
        if len(values) <= 3:
            ticks = values
        else:
            ticks = values[np.rint(np.linspace(0, len(values) - 1, 3)).astype(int)]
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(_compact_tick))
    axis.minorticks_off()
