import numpy as np

from fcp_shift.ablations.efficiency import regression_interval_lengths
from fcp_shift.conformal import CalibrationStructure
from fcp_shift.models import candidate_classification_scores, conformity_scores


class _ProbabilityModel:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, features):
        return np.asarray(features, dtype=float)


def test_candidate_scores_match_observed_class_scores() -> None:
    probabilities = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]]
    )
    targets = np.array([0, 2, 1])
    model = _ProbabilityModel()

    candidates = candidate_classification_scores(model, probabilities, "log_margin")
    observed = conformity_scores(
        model, probabilities, targets, "classification", "log_margin"
    )

    assert np.allclose(candidates[np.arange(3), targets], observed)


def test_regression_efficiency_uses_weighted_empirical_radius() -> None:
    structure = CalibrationStructure.build(
        scores=np.array([1.0, 2.0, 3.0]),
        weights=np.ones(3),
        w_infinity=1.0,
    )

    lengths = regression_interval_lengths(structure, np.array([0.0, 0.5, 1.0]))

    assert np.allclose(lengths, [6.0, 4.0, 0.0])
