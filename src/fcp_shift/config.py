from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping: {path}")
    config = deepcopy(config)
    config["_config_path"] = str(path.resolve())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    kind = experiment.get("kind")
    if kind not in {"covariate_shift", "transport_shift", "asymptotic"}:
        raise ConfigError(
            "experiment.kind must be covariate_shift, transport_shift, or asymptotic"
        )
    if int(experiment.get("repetitions", 0)) <= 0:
        raise ConfigError("experiment.repetitions must be positive")
    seeds = experiment.get("seeds", [])
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ConfigError("experiment.seeds must be a non-empty integer list")

    if kind != "asymptotic":
        datasets = config.get("datasets", [])
        weights = config.get("weights", [])
        if not datasets:
            raise ConfigError("At least one dataset is required")
        if not weights:
            raise ConfigError("At least one weight configuration is required")
        for dataset in datasets:
            if dataset.get("task") not in {"classification", "regression"}:
                raise ConfigError(f"Invalid task for dataset {dataset.get('name')}")

    delta = float(config.get("fcp", {}).get("delta", 0.0))
    if not 0.0 < delta < 1.0:
        raise ConfigError("fcp.delta must lie in (0, 1)")


def filter_config(
    config: dict[str, Any],
    dataset: str | None = None,
    weight: str | None = None,
    rho: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    result = deepcopy(config)
    if dataset is not None:
        result["datasets"] = [
            item for item in result.get("datasets", []) if item["name"] == dataset
        ]
        if not result["datasets"]:
            raise ConfigError(f"Dataset {dataset!r} is not present in the configuration")
    if weight is not None:
        result["weights"] = [
            item for item in result.get("weights", []) if item["name"] == weight
        ]
        if not result["weights"]:
            raise ConfigError(f"Weight {weight!r} is not present in the configuration")
    if rho is not None:
        if result["experiment"]["kind"] != "transport_shift":
            raise ConfigError("--rho is only valid for transport_shift")
        result.setdefault("transport", {})["rhos"] = [float(rho)]
    if seed is not None:
        result["experiment"]["seeds"] = [int(seed)]
    return result


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False)

