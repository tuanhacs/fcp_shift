from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml


def load_openml_frame(dataset: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    bunch = fetch_openml(
        data_id=int(dataset["openml_id"]),
        as_frame=True,
        parser="auto",
    )
    features = bunch.data.copy()
    target_column = dataset.get("target_column")
    if target_column and target_column in features.columns:
        target = features.pop(target_column)
    elif bunch.target is not None:
        target = pd.Series(bunch.target)
    else:
        raise ValueError(f"No target found for OpenML dataset {dataset['name']}")
    target = pd.Series(target)
    keep = ~target.isna()
    return (
        features.loc[keep].reset_index(drop=True),
        target.loc[keep].reset_index(drop=True),
    )

