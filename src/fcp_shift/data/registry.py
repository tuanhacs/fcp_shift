from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .openml import load_openml_frame
from .preprocessing import make_preprocessor


@dataclass
class PreparedDataset:
    name: str
    task: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_source: np.ndarray
    y_source: np.ndarray


def _load_frame(dataset: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    source = dataset.get("source", "openml")
    if source == "openml":
        return load_openml_frame(dataset)
    if source == "csv":
        path = Path(dataset["path"])
        frame = pd.read_csv(path)
        target_column = dataset["target_column"]
        return frame.drop(columns=[target_column]), frame[target_column]
    if source == "synthetic_regression":
        features, target = make_regression(
            n_samples=int(dataset.get("n_samples", 5000)),
            n_features=int(dataset.get("n_features", 8)),
            noise=float(dataset.get("noise", 10.0)),
            random_state=int(dataset.get("seed", 123)),
        )
        return pd.DataFrame(features), pd.Series(target)
    raise ValueError(f"Unsupported dataset source: {source}")


def prepare_dataset(dataset: dict[str, Any], model_seed: int) -> PreparedDataset:
    features, target = _load_frame(dataset)
    task = dataset["task"]
    if task == "classification":
        counts = target.astype(str).value_counts()
        valid = counts[counts >= 2].index
        keep = target.astype(str).isin(valid)
        features = features.loc[keep].reset_index(drop=True)
        target = target.loc[keep].reset_index(drop=True)
        y = LabelEncoder().fit_transform(target.astype(str))
        stratify = y
    else:
        y = pd.to_numeric(target, errors="raise").to_numpy(dtype=float)
        stratify = None

    indices = np.arange(len(features))
    train_indices, source_indices = train_test_split(
        indices,
        train_size=float(dataset.get("train_fraction", 0.4)),
        random_state=model_seed,
        stratify=stratify,
    )
    train_frame = features.iloc[train_indices].reset_index(drop=True)
    source_frame = features.iloc[source_indices].reset_index(drop=True)
    preprocessor = make_preprocessor(train_frame)
    x_train = np.asarray(preprocessor.fit_transform(train_frame), dtype=np.float32)
    x_source = np.asarray(preprocessor.transform(source_frame), dtype=np.float32)
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_source)):
        raise RuntimeError(f"Non-finite transformed values in dataset {dataset['name']}")
    return PreparedDataset(
        name=dataset["name"],
        task=task,
        x_train=x_train,
        y_train=np.asarray(y)[train_indices],
        x_source=x_source,
        y_source=np.asarray(y)[source_indices],
    )
