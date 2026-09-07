import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fcp_shift.reporting.labels import display_dataset_name
from fcp_shift.reporting.style import set_publication_ticks


def test_publication_dataset_names() -> None:
    assert display_dataset_name("year") == "YearPredictionMSD"
    assert display_dataset_name("adult") == "Adult"
    assert display_dataset_name("fashion_mnist") == "Fashion-MNIST"


def test_publication_ticks_are_linear_and_have_three_major_ticks() -> None:
    figure, axis = plt.subplots()
    axis.plot([250, 500, 1000, 5000], [0.01, 0.02, 0.03, 0.04])
    axis.set_xscale("log")
    axis.set_yscale("log")

    set_publication_ticks(axis, x_values=[250, 500, 1000, 5000])
    figure.canvas.draw()

    assert axis.get_xscale() == "linear"
    assert axis.get_yscale() == "linear"
    assert len(axis.get_xticks()) == 3
    assert len(axis.get_yticks()) == 3
    plt.close(figure)


def test_publication_ticks_support_clean_log_axes() -> None:
    figure, axis = plt.subplots()
    axis.plot([250, 1000, 5000, 50000], [1e-4, 3e-4, 2e-3, 1e-2])

    set_publication_ticks(
        axis,
        x_values=[250, 1000, 5000, 50000],
        y_values=[1e-4, 3e-4, 2e-3, 1e-2],
        xscale="log",
        yscale="log",
    )
    figure.canvas.draw()

    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    assert len(axis.get_xticks()) == 3
    assert len(axis.get_yticks()) == 3
    plt.close(figure)
