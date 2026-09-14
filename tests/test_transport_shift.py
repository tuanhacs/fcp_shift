import numpy as np

from fcp_shift.shifts.score_transport import build_score_transport


def test_transport_weights_match_the_target_score_mixture():
    scores = np.linspace(0.0, 1.0, 200)
    base_weights = np.exp(2.0 * scores)
    # Feature-based strata are deliberately reversed relative to score ranks.
    feature_values = np.linspace(1.0, 0.0, len(scores))
    transport = build_score_transport(
        scores, base_weights, strata_count=4,
        stratification_values=feature_values,
        reference_values=np.linspace(0.0, 1.0, 1000),
    )
    assert transport.strata[0] == 3
    shuffled_scores = np.random.default_rng(12).permutation(scores)
    shuffled_transport = build_score_transport(
        shuffled_scores, base_weights, strata_count=4,
        stratification_values=feature_values,
        reference_values=np.linspace(0.0, 1.0, 1000),
    )
    np.testing.assert_array_equal(transport.strata, shuffled_transport.strata)
    rho = 0.6
    weights = transport.weights(rho)
    threshold = 0.55
    weighted_calibration_cdf = np.sum(weights * (scores <= threshold)) / np.sum(weights)

    expected = 0.0
    for target_stratum, probability in enumerate(transport.q):
        own = scores[transport.strata == target_stratum]
        donor = scores[transport.strata == transport.permutation[target_stratum]]
        expected += probability * (
            (1.0 - rho) * np.mean(own <= threshold)
            + rho * np.mean(donor <= threshold)
        )
    assert abs(weighted_calibration_cdf - expected) < 1e-12


def test_rho_zero_preserves_each_selected_pair_and_rho_one_uses_donor_stratum():
    scores = np.arange(40, dtype=float)
    feature_values = np.arange(40, dtype=float)
    transport = build_score_transport(
        scores, np.exp(feature_values / 20), strata_count=2,
        stratification_values=feature_values,
        reference_values=feature_values,
    )
    source_indices, test_scores = transport.sample_test(
        1000, 0.0, np.random.default_rng(3)
    )
    np.testing.assert_array_equal(test_scores, scores[source_indices])
    source_indices, test_scores = transport.sample_test(
        1000, 1.0, np.random.default_rng(4)
    )
    donor_indices = test_scores.astype(int)
    np.testing.assert_array_equal(
        transport.strata[donor_indices],
        transport.permutation[transport.strata[source_indices]],
    )
