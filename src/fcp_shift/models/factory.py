from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def fit_model(task: str, x_train, y_train, config: dict[str, Any], seed: int):
    common = {
        "learning_rate": float(config.get("learning_rate", 0.08)),
        "max_iter": int(config.get("max_iter", 300)),
        "max_leaf_nodes": int(config.get("max_leaf_nodes", 31)),
        "early_stopping": bool(config.get("early_stopping", False)),
        "random_state": seed,
    }
    if task == "regression":
        model = HistGradientBoostingRegressor(**common)
    elif task == "classification":
        model = HistGradientBoostingClassifier(**common)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return model.fit(x_train, y_train)

