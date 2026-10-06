from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_architecture_comparison import (  # noqa: E402
    build_model,
    contrast_pairs,
    initialize_model,
    validate_config,
)


def _config() -> dict:
    return {
        "model": {
            "action_embedding_dim": 8,
            "training_seeds": [11, 22, 33, 44, 55],
        },
        "conditions": [
            {
                "name": "row_65k",
                "encoder_type": "row_sorted_gru",
                "hidden_dim": 54,
                "capacity_tier": "65k",
                "suffix_source": "same_episode",
            },
            {
                "name": "set_65k",
                "encoder_type": "set",
                "hidden_dim": 48,
                "capacity_tier": "65k",
                "suffix_source": "same_episode",
            },
            {
                "name": "static_65k",
                "encoder_type": "static_graph",
                "hidden_dim": 42,
                "capacity_tier": "65k",
                "suffix_source": "same_episode",
            },
            {
                "name": "dynamic_65k",
                "encoder_type": "dynamic_graph",
                "hidden_dim": 42,
                "capacity_tier": "65k",
                "suffix_source": "same_episode",
            },
        ],
    }


def test_valid_capacity_config_is_accepted() -> None:
    validate_config(_config())


def test_duplicate_condition_names_are_rejected() -> None:
    config = _config()
    config["conditions"][1]["name"] = "row_65k"

    with pytest.raises(ValueError, match="unique"):
        validate_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("encoder_type", "attention", "encoder_type"),
        ("suffix_source", "future_leak", "suffix_source"),
        ("capacity_tier", "70k", "capacity_tier"),
    ],
)
def test_unknown_condition_values_are_rejected(
    field: str, value: str, message: str
) -> None:
    config = _config()
    config["conditions"][0][field] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_capacity_deviation_above_one_point_five_percent_is_rejected() -> None:
    config = _config()
    config["conditions"][0]["hidden_dim"] = 32

    with pytest.raises(ValueError, match="1.5%"):
        validate_config(config)


def test_five_unique_training_seeds_are_required() -> None:
    config = _config()
    config["model"]["training_seeds"] = [11, 22]

    with pytest.raises(ValueError, match="five unique"):
        validate_config(config)


@pytest.mark.parametrize(
    ("condition_index", "expected_parameters"),
    [(0, 64734), (1, 64266), (2, 64302), (3, 64302)],
)
def test_condition_driven_model_construction(
    condition_index: int, expected_parameters: int
) -> None:
    config = _config()
    model = build_model(config["conditions"][condition_index], config)

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters


def test_validation_does_not_mutate_config() -> None:
    config = _config()
    original = copy.deepcopy(config)

    validate_config(config)

    assert config == original


def test_frozen_merge_capacity_config_is_valid() -> None:
    config_path = ROOT / "config" / "architecture_capacity_merge.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    validate_config(config)

    assert len(config["conditions"]) == 10
    assert config["model"]["training_seeds"] == [11, 22, 33, 44, 55]


def test_cuda_determinism_workspace_is_configured() -> None:
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_model_initialization_is_reproducible_independent_of_prior_rng_use() -> None:
    import torch

    config = _config()
    condition = config["conditions"][0]
    first = initialize_model(condition, config, seed=11)
    torch.rand(1000)
    second = initialize_model(condition, config, seed=11)

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_frozen_roundabout_capacity_config_is_stable_id_only() -> None:
    config_path = ROOT / "config" / "architecture_capacity_roundabout.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    validate_config(config)

    assert [condition["name"] for condition in config["conditions"]] == [
        "row_65k",
        "set_65k",
        "set_shuffled_65k",
        "row_83k",
        "set_83k",
        "set_shuffled_83k",
    ]
    assert config["model"]["training_seeds"] == [11, 22, 33, 44, 55]
    assert config["model"]["max_epochs"] == 200


def test_set_only_contrasts_include_structure_and_negative_control() -> None:
    config_path = ROOT / "config" / "architecture_capacity_roundabout.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert contrast_pairs(config["conditions"]) == [
        ("row_65k", "set_65k"),
        ("set_shuffled_65k", "set_65k"),
        ("row_83k", "set_83k"),
        ("set_shuffled_83k", "set_83k"),
    ]
