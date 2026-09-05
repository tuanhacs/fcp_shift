from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl


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
