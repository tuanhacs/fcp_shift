import numpy as np
import pytest

from fcp_shift.weights import ScoreProjection, direction_variant, fit_score_projection, fit_weight


@pytest.mark.parametrize(
    "name",
    [
        "exponential", "quadratic", "mahalanobis", "linear", "sigmoid",
        "logarithmic", "arctangent", "power_tilt",
    ],
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


def test_logarithmic_weight_uses_log_one_plus_squared_projection() -> None:
    projection = ScoreProjection(
        feature_mean=np.zeros(1), feature_scale=np.ones(1),
        direction=np.ones(1), projection_mean=0.0, projection_scale=1.0,
    )
    x_source = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])
    fitted = fit_weight(
        {"name": "logarithmic", "strength": 1.0, "clip_quantile": 1.0},
        np.zeros((2, 1)), x_source, np.arange(len(x_source)),
        score_projection=projection,
    )
    expected = 1.0 + 1e-8 + np.log1p(x_source[:, 0] ** 2)
    np.testing.assert_allclose(fitted.values, expected / expected.mean())
    assert fitted.values[0] > fitted.values[1] > fitted.values[2]
    assert fitted.values[0] == pytest.approx(fitted.values[-1])


@pytest.mark.parametrize("name", ["arctangent", "power_tilt"])
def test_new_weights_increase_with_projection_and_strength(name: str) -> None:
    projection = ScoreProjection(
        feature_mean=np.zeros(1), feature_scale=np.ones(1),
        direction=np.ones(1), projection_mean=0.0, projection_scale=1.0,
    )
    z = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    x_source = z[:, None]

    def fitted_at(strength: float):
        return fit_weight(
            {"name": name, "strength": strength, "clip_quantile": 1.0},
            np.zeros((2, 1)), x_source, z, score_projection=projection,
        ).values

    values = fitted_at(1.0)
    if name == "arctangent":
        raw = 1.0 + 0.5 + np.arctan(z) / np.pi
    else:
        raw = (1.0 + np.abs(z)) ** np.sign(z)
    np.testing.assert_allclose(values, raw / raw.mean())
    assert np.all(np.diff(values) > 0.0)
    assert fitted_at(2.0)[-1] / fitted_at(2.0)[0] > values[-1] / values[0]
    np.testing.assert_allclose(fitted_at(0.0), np.ones_like(z))


@pytest.mark.parametrize("name", ["arctangent", "power_tilt"])
def test_new_monotone_weights_reject_negative_strength(name: str) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        fit_weight(
            {"name": name, "strength": -1.0},
            np.zeros((2, 1)), np.arange(5.0)[:, None], np.arange(5.0),
        )
