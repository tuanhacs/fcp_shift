import pytest

from fcp_shift.config import ConfigError, filter_config, load_config, validate_config
from fcp_shift.cli import _override_direction, build_parser


def test_smoke_config_loads_and_filters():
    config = load_config("configs/smoke/covariate_shift_smoke.yaml")
    filtered = filter_config(config, dataset="synthetic_smoke", weight="exponential", seed=9)
    assert filtered["experiment"]["seeds"] == [9]
    assert len(filtered["datasets"]) == 1
    assert len(filtered["weights"]) == 1
    assert filtered["_filters"] == {
        "dataset": "synthetic_smoke",
        "weight": "exponential",
        "seed": 9,
    }


def test_direction_can_be_overridden_from_cli():
    config = load_config("configs/smoke/covariate_shift_smoke.yaml")
    args = build_parser().parse_args([
        "run", "--config", "configs/smoke/covariate_shift_smoke.yaml",
        "--weight-direction", "random", "--direction-seed", "19",
    ])
    _override_direction(config, args)
    assert all(weight["direction"] == "random" for weight in config["weights"])
    assert all(weight["direction_seed"] == 19 for weight in config["weights"])


def test_invalid_direction_is_rejected():
    config = load_config("configs/smoke/covariate_shift_smoke.yaml")
    config["weights"][0]["direction"] = "unknown"
    with pytest.raises(ConfigError, match="weight.direction"):
        validate_config(config)
