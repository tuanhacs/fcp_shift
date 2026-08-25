from pathlib import Path

import numpy as np

from fcp_shift.reporting.combined_main import make_covariate_transport_figure


def _write_curves(directory: Path) -> None:
    directory.mkdir(parents=True)
    alpha = np.linspace(0.0, 1.0, 5)
    beta = np.linspace(0.0, 1.0, 5)
    repeated_alpha = np.tile(alpha, (2, 1))
    repeated_beta = np.tile(beta, (2, 1))
    np.savez_compressed(
        directory / "curves.npz",
        alpha=alpha,
        beta=beta,
        empirical_fcp=repeated_alpha,
        goal1_bound=repeated_alpha + 0.1,
        goal2_bound=repeated_alpha + 0.2,
        goal3_alpha=np.maximum(repeated_beta - 0.1, 0.0),
        goal4_alpha=np.maximum(repeated_beta - 0.2, 0.0),
        goal3_fcp=np.maximum(repeated_beta - 0.05, 0.0),
        goal4_fcp=np.maximum(repeated_beta - 0.1, 0.0),
    )


def test_two_datasets_create_combined_2_by_4_outputs(tmp_path: Path):
    datasets = [{"name": "data_a"}, {"name": "data_b"}]
    config = {
        "datasets": datasets,
        "output": {"root": str(tmp_path)},
    }
    for dataset in ("data_a", "data_b"):
        _write_curves(
            tmp_path / "covariate_shift" / dataset / "exponential" / "seed_1"
        )
        _write_curves(
            tmp_path
            / "transport_shift"
            / dataset
            / "exponential"
            / "rho_0.50"
            / "seed_1"
        )
    pdf, png = make_covariate_transport_figure(
        config,
        config,
        weight="exponential",
        rho=0.5,
        datasets=["data_a", "data_b"],
        output_path=tmp_path / "combined.pdf",
    )
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
