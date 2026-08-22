from fcp_shift.config import filter_config, load_config


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
