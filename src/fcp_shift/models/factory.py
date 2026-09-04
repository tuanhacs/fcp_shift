from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_model(task: str, x_train, y_train, config: dict[str, Any], seed: int):
    name = config.get("name", "hist_gradient_boosting")
    if name == "random_forest":
        common = {
            "n_estimators": int(config.get("n_estimators", 200)),
            "max_depth": config.get("max_depth"),
            "min_samples_leaf": int(config.get("min_samples_leaf", 1)),
            "n_jobs": int(config.get("n_jobs", -1)),
            "random_state": seed,
        }
        model = RandomForestRegressor(**common) if task == "regression" else RandomForestClassifier(**common)
        return model.fit(x_train, y_train)
    if name in {"linear", "ridge", "logistic"}:
        if task == "regression":
            model = Ridge(alpha=float(config.get("alpha", 1.0)))
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(config.get("C", 1.0)),
                    max_iter=int(config.get("max_iter", 2000)),
                    tol=float(config.get("tol", 1e-4)),
                    n_jobs=int(config.get("n_jobs", -1)),
                    random_state=seed,
                ),
            )
        return model.fit(x_train, y_train)
    if name == "mlp":
        hidden = tuple(int(value) for value in config.get("hidden_layer_sizes", [100, 50]))
        common = {
            "hidden_layer_sizes": hidden,
            "alpha": float(config.get("alpha", 1e-4)),
            "max_iter": int(config.get("max_iter", 300)),
            "early_stopping": bool(config.get("early_stopping", True)),
            "random_state": seed,
        }
        model = MLPRegressor(**common) if task == "regression" else MLPClassifier(**common)
        return model.fit(x_train, y_train)
    if name != "hist_gradient_boosting":
        raise ValueError(f"Unsupported model: {name}")
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
