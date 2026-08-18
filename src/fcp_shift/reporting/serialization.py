from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fcp_shift.config import dump_config
from fcp_shift.reproducibility import environment_metadata


class RunDirectory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def complete(self) -> bool:
        return (self.path / "DONE").exists()

    def initialize(self, config: dict[str, Any], metadata: dict[str, Any]) -> None:
        dump_config(config, self.path / "config.resolved.yaml")
        payload = {**metadata, "environment": environment_metadata()}
        with (self.path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)

    def save_metrics(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.path / "metrics.csv", index=False)

    def save_summary(self, summary: dict[str, Any]) -> None:
        with (self.path / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=_json_default)

    def save_arrays(self, **arrays: np.ndarray) -> None:
        np.savez_compressed(self.path / "curves.npz", **arrays)

    def mark_complete(self) -> None:
        (self.path / "DONE").write_text("complete\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")
