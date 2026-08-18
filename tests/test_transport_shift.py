import numpy as np

from fcp_shift.shifts.score_transport import build_score_transport


def test_transport_weights_match_the_target_score_mixture():
    scores = np.linspace(0.0, 1.0, 200)
    base_weights = np.exp(2.0 * scores)
    transport = build_score_transport(scores, base_weights, strata_count=4)
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

