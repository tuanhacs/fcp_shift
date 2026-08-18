from __future__ import annotations

import numpy as np


def conformity_scores(model, features, target, task: str, score: str | None = None):
    target = np.asarray(target)
    if task == "regression":
        return np.abs(target.astype(float) - np.asarray(model.predict(features), dtype=float))

    probabilities = np.clip(np.asarray(model.predict_proba(features), dtype=float), 1e-12, 1.0)
    classes = np.asarray(model.classes_)
    positions = np.searchsorted(classes, target)
    if np.any(positions >= len(classes)) or np.any(classes[positions] != target):
        raise RuntimeError("A source class is absent from the fitted classifier")
    rows = np.arange(len(target))
    true_probability = probabilities[rows, positions]
    score = score or "log_margin"
    if score == "lac":
        return 1.0 - true_probability
    if score == "neg_log_prob":
        return -np.log(true_probability)
    competitors = probabilities.copy()
    competitors[rows, positions] = -np.inf
    maximum_other = np.clip(np.max(competitors, axis=1), 1e-12, 1.0)
    if score == "log_margin":
        return np.log(maximum_other / true_probability)
    raise ValueError(f"Unsupported classification score: {score}")

