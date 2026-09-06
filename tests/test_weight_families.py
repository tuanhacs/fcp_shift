import numpy as np
import pytest

from fcp_shift.weights import fit_weight


@pytest.mark.parametrize(
    "name",
    ["exponential", "quadratic", "mahalanobis", "linear", "sigmoid"],
)
def test_weight_families_are_finite_positive_and_normalized(name: str) -> None:
    rng = np.random.default_rng(17)
    x_train = rng.normal(size=(80, 5))
    x_source = rng.normal(size=(120, 5))
    scores = np.abs(rng.normal(size=120))

    fitted = fit_weight(
        {"name": name, "strength": 0.4, "clip_quantile": 0.95},
        x_train,
        x_source,
        scores,
    )

    assert fitted.values.shape == (120,)
    assert np.all(np.isfinite(fitted.values))
    assert np.all(fitted.values > 0.0)
    assert np.mean(fitted.values) == pytest.approx(1.0)
    assert fitted.bound == pytest.approx(np.max(fitted.values))
