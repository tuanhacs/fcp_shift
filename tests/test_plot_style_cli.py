import pytest

from fcp_shift.cli import build_parser, _plot_style_from_args
from fcp_shift.reporting.style import configure_plot_style, figure_size, font_size


def test_main_figure_plot_style_arguments() -> None:
    args = build_parser().parse_args(
        [
            "main-figure",
            "--weight",
            "exponential",
            "--rho",
            "0.5",
            "--figsize",
            "14",
            "5",
            "--font-size",
            "10",
            "--tick-font-size",
            "9",
        ]
    )
    configure_plot_style(_plot_style_from_args(args))
    assert figure_size((8, 5)) == (14.0, 5.0)
    assert font_size("label", 8) == 10.0
    assert font_size("tick", 8) == 9.0
    configure_plot_style(type(_plot_style_from_args(args))())


def test_plot_sizes_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "figures",
                "--config",
                "config.yaml",
                "--figsize",
                "-1",
                "5",
            ]
        )
