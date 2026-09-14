import numpy as np
import pytest

from fcp_shift.weights import direction_variant, fit_score_projection, fit_weight


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


def test_auxiliary_score_projection_is_reusable_on_unseen_features() -> None:
    rng = np.random.default_rng(31)
    x_reference = rng.normal(size=(100, 3))
    x_auxiliary = rng.normal(size=(200, 3))
    scores = 2.0 * x_auxiliary[:, 0] - x_auxiliary[:, 1]
    projection = fit_score_projection(x_reference, x_auxiliary, scores)
    x_source = rng.normal(size=(80, 3))
    predicted_difficulty = projection.transform(x_source)
    true_difficulty = 2.0 * x_source[:, 0] - x_source[:, 1]
    assert np.corrcoef(predicted_difficulty, true_difficulty)[0, 1] > 0.99
    fitted = fit_weight(
        {"name": "exponential", "strength": 0.35},
        x_reference, x_source, true_difficulty,
        score_projection=projection,
    )
    assert fitted.metadata["projection_source"] == "independent_auxiliary"


def test_random_direction_is_seeded_and_ignores_auxiliary_scores() -> None:
    rng = np.random.default_rng(41)
    x_reference = rng.normal(size=(100, 5))
    x_auxiliary = rng.normal(size=(120, 5))
    scores = rng.normal(size=120)
    first = fit_score_projection(
        x_reference, x_auxiliary, scores, method="random", random_seed=7
    )
    repeated = fit_score_projection(
        x_reference, x_auxiliary, scores[::-1], method="random", random_seed=7
    )
    changed = fit_score_projection(
        x_reference, x_auxiliary, scores, method="random", random_seed=8
    )
    np.testing.assert_allclose(first.transform(x_auxiliary), repeated.transform(x_auxiliary))
    assert not np.allclose(first.direction, changed.direction)


def test_random_fallback_weight_ignores_source_scores() -> None:
    rng = np.random.default_rng(42)
    x_reference = rng.normal(size=(60, 4))
    x_source = rng.normal(size=(100, 4))
    scores = rng.normal(size=100)
    config = {"name": "exponential", "direction": "random", "direction_seed": 11}
    first = fit_weight(config, x_reference, x_source, scores)
    second = fit_weight(config, x_reference, x_source, scores[::-1])
    np.testing.assert_allclose(first.values, second.values)
    assert first.metadata["projection_source"] == "random_direction"


def test_direction_variant_keeps_ridge_path_and_separates_random_seed() -> None:
    assert direction_variant({"name": "exponential"}) is None
    assert direction_variant({"direction": "random", "direction_seed": 19}) == (
        "direction_random_seed_19"
    )
