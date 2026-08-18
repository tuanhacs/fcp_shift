from pathlib import Path

import numpy as np

from fcp_shift.reporting.grouped import plot_grouped_weights


def test_multiple_weights_are_written_to_shared_figures(tmp_path: Path):
    alpha = np.linspace(0.0, 1.0, 5)
    beta = np.linspace(0.0, 1.0, 5)
    template = {
        "alpha": alpha,
        "beta": beta,
        "empirical_fcp": np.tile(alpha, (2, 1)),
        "goal1_bound": np.tile(alpha + 0.1, (2, 1)),
        "goal2_bound": np.tile(alpha + 0.2, (2, 1)),
        "goal3_alpha": np.tile(np.maximum(beta - 0.1, 0.0), (2, 1)),
        "goal4_alpha": np.tile(np.maximum(beta - 0.2, 0.0), (2, 1)),
        "goal3_fcp": np.tile(np.maximum(beta - 0.05, 0.0), (2, 1)),
        "goal4_fcp": np.tile(np.maximum(beta - 0.1, 0.0), (2, 1)),
    }
    plot_grouped_weights(
        {"exponential": template, "quadratic": template}, tmp_path, "test"
    )
    assert len(list(tmp_path.glob("grouped_weights_*.pdf"))) == 6
