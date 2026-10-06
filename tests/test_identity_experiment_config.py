from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.identity_controls import assert_same_object_multiset, transform_object_order
from ac_twin.model import PairedObjectEffectGRU
from run_identity_continuity_comparison import (
    apply_object_normalizer,
    build_capacity_matched_model,
    fit_object_normalizer,
    prepare_identity_inputs,
    validate_identity_config,
)


def minimal_config():
    return {
        "experiment_name": "identity_unit_test",
        "scene": "merge",
        "smoke_test": True,
        "capacity_tiers": ["65k"],
        "conditions": [
            {"name": "stable_temporal", "order_mode": "stable", "encoder_type": "identity_temporal"},
            {"name": "distance_temporal", "order_mode": "distance", "encoder_type": "identity_temporal"},
            {"name": "permuted_temporal", "order_mode": "permuted", "encoder_type": "identity_temporal"},
            {"name": "set_pool", "order_mode": "stable", "encoder_type": "set"},
        ],
        "factual_suffix_steps": 2,
        "permutation_seed": 20260905,
        "progress_every_epochs": 5,
        "model": {"training_seeds": [11], "action_embedding_dim": 8, "max_epochs": 2,
                  "batch_size": 4, "early_stopping_patience": 2,
                  "learning_rate": 0.001, "weight_decay": 0.0001},
    }


@pytest.fixture
def dataset():
    rng = np.random.default_rng(123)
    history = rng.normal(size=(6, 3, 4, 8)).astype(np.float32)
    factual = rng.normal(size=(6, 4, 4, 8)).astype(np.float32)
    for values in (history, factual):
        values[..., 0] = 1
        values[..., 1] = 0
        values[:, :, 0, 1] = 1
        values[..., 2] *= 100
        values[0, 0, 3] = 0
    return {"object_history": history, "object_factual_future": factual,
            "effect_xy": rng.normal(size=(6, 4, 2)).astype(np.float32),
            "split": np.array([0, 0, 1, 1, 2, 2]),
            "episode_seed": np.arange(100, 106),
            "factual_actions": np.zeros((6, 4), dtype=np.int64),
            "counterfactual_actions": np.ones((6, 4), dtype=np.int64)}


def test_fair_config_requires_exact_condition_set():
    config = minimal_config()
    validate_identity_config(config)
    config["conditions"].pop()
    with pytest.raises(ValueError, match="exactly"):
        validate_identity_config(config)


@pytest.mark.parametrize("mutation", ["duplicate", "encoder", "order", "override"])
def test_conditions_cannot_change_fair_comparison(mutation):
    config = minimal_config()
    if mutation == "duplicate":
        config["conditions"].append(config["conditions"][0].copy())
    elif mutation == "encoder":
        config["conditions"][0]["encoder_type"] = "dynamic_graph"
    elif mutation == "order":
        config["conditions"][3]["order_mode"] = "distance"
    else:
        config["conditions"][0]["feature_dim"] = 9
    with pytest.raises(ValueError):
        validate_identity_config(config)


def test_formal_config_requires_both_tiers_and_five_unique_seeds():
    config = minimal_config()
    config["smoke_test"] = False
    config["frozen_test_data"] = "future-frozen.npz"
    config["statistics"] = {"bootstrap_resamples": 10000, "confidence_level": 0.95}
    config["model"]["training_seeds"] = [11, 22, 33, 44, 55]
    with pytest.raises(ValueError, match="capacity_tiers"):
        validate_identity_config(config)
    config["capacity_tiers"] = ["65k", "83k"]
    validate_identity_config(config)
    config["model"]["training_seeds"][-1] = 11
    with pytest.raises(ValueError, match="five unique"):
        validate_identity_config(config)


@pytest.mark.parametrize("tiers", [["83k"], ["65k", "65k"], ["70k"], []])
def test_invalid_smoke_capacity_tiers_are_rejected(tiers):
    config = minimal_config()
    config["capacity_tiers"] = tiers
    with pytest.raises(ValueError, match="capacity_tiers"):
        validate_identity_config(config)


@pytest.mark.parametrize("target", [65_000, 83_000])
@pytest.mark.parametrize("encoder_type", ["identity_temporal", "set"])
def test_hidden_dimension_search_stays_within_capacity_tolerance(target, encoder_type):
    model, hidden_dim, count = build_capacity_matched_model(
        encoder_type=encoder_type, target_parameters=target,
        feature_dim=8, action_dim=5, action_embedding_dim=8)
    assert 8 <= hidden_dim <= 256
    assert count == sum(p.numel() for p in model.parameters())
    assert abs(count - target) / target <= 0.015
    with torch.device("meta"):
        counts = {h: sum(p.numel() for p in PairedObjectEffectGRU(
            8, 5, h, 8, encoder_type).parameters()) for h in range(8, 257)}
    assert (abs(count - target), hidden_dim) == min(
        (abs(n - target), h) for h, n in counts.items())


def test_impossible_capacity_match_is_rejected():
    with pytest.raises(ValueError, match="1.5%"):
        build_capacity_matched_model(encoder_type="set", target_parameters=1,
                                     feature_dim=8, action_dim=5, action_embedding_dim=8)


def test_prepared_conditions_share_features_visibility_and_normalization(dataset):
    original = copy.deepcopy(dataset)
    prepared = prepare_identity_inputs(minimal_config(), dataset)
    stable = prepared["conditions"]["stable_temporal"]
    for name in ("distance_temporal", "permuted_temporal", "set_pool"):
        candidate = prepared["conditions"][name]
        assert_same_object_multiset(stable["history"], candidate["history"])
        assert_same_object_multiset(stable["suffix"], candidate["suffix"])
        assert np.all(candidate["suffix"][:, 2:] == 0)
    np.testing.assert_array_equal(stable["history"], prepared["conditions"]["set_pool"]["history"])
    assert prepared["feature_names"] == [
        "presence", "is_ego", "x", "y", "vx", "vy", "cos_heading", "sin_heading"]
    for name, values in original.items():
        np.testing.assert_array_equal(dataset[name], values)


def test_distance_sorting_precedes_anisotropic_normalization(dataset):
    raw = dataset["object_history"]
    raw[4, 0, 0, 2:4] = [0, 0]
    raw[4, 0, 1:, 2:4] = [[50, 0], [0, 2], [90, 3]]
    config = minimal_config()
    prepared = prepare_identity_inputs(config, dataset)
    anchors = raw[:, -1, 0, 2:4]
    ordered = transform_object_order(raw, "distance", dataset["episode_seed"], config["permutation_seed"])
    expected = apply_object_normalizer(ordered, anchors, prepared["normalizer"])
    actual = prepared["conditions"]["distance_temporal"]["history"]
    np.testing.assert_array_equal(actual, expected)
    assert actual[4, 0, 1, 2] == expected[4, 0, 1, 2]
    wrong = transform_object_order(prepared["conditions"]["stable_temporal"]["history"],
                                   "distance", dataset["episode_seed"], config["permutation_seed"])
    assert not np.array_equal(actual[4, 0], wrong[4, 0])


def test_permuted_suffix_uses_continuing_absolute_time(dataset):
    config = minimal_config()
    prepared = prepare_identity_inputs(config, dataset)
    ordered = transform_object_order(dataset["object_factual_future"], "permuted",
                                     dataset["episode_seed"], config["permutation_seed"], start_step=3)
    expected = apply_object_normalizer(ordered, dataset["object_history"][:, -1, 0, 2:4],
                                       prepared["normalizer"])
    expected[:, 2:] = 0
    np.testing.assert_array_equal(prepared["conditions"]["permuted_temporal"]["suffix"], expected)


def test_normalizer_fits_training_history_and_factual_only(dataset):
    train = np.array([0, 1])
    fitted = fit_object_normalizer(dataset["object_history"], dataset["object_factual_future"], train)
    changed = copy.deepcopy(dataset)
    changed["object_history"][2:, ..., 2:6] = 1e8
    changed["object_factual_future"][2:, ..., 2:6] = -1e8
    changed["effect_xy"][2:] = 1e8
    prepared = prepare_identity_inputs(minimal_config(), changed)
    original = prepare_identity_inputs(minimal_config(), dataset)
    for key in ("mean", "std"):
        np.testing.assert_array_equal(prepared["normalizer"][key], fitted[key])
    np.testing.assert_array_equal(prepared["target_mean"], original["target_mean"])
    np.testing.assert_array_equal(prepared["target_std"], original["target_std"])
    anchors = dataset["object_history"][:, -1, 0, 2:4]
    centered = np.concatenate([dataset["object_history"][train], dataset["object_factual_future"][train]], axis=1)
    centered[..., 2:4] -= anchors[train, None, None]
    visible = centered[..., 0] > 0
    np.testing.assert_allclose(fitted["mean"][2:6], centered[..., 2:6][visible].mean(0), rtol=1e-6)


def test_development_evaluation_uses_source_test_only(dataset):
    prepared = prepare_identity_inputs(minimal_config(), dataset)
    assert prepared["evaluation_mode"] == "development_test"
    assert prepared["test_index_source"] == "source_data"
    np.testing.assert_array_equal(prepared["train_indices"], [0, 1])
    np.testing.assert_array_equal(prepared["validation_indices"], [2, 3])
    np.testing.assert_array_equal(prepared["test_indices"], [4, 5])
    np.testing.assert_array_equal(prepared["evaluation"]["target"], dataset["effect_xy"][[4, 5]])


def test_frozen_data_is_entirely_test_and_never_used_for_fitting(dataset, tmp_path):
    config = minimal_config()
    source_only = prepare_identity_inputs(config, dataset)
    frozen = copy.deepcopy(dataset)
    frozen["episode_seed"] += 10000
    frozen["split"][:] = 0
    frozen["object_history"][..., 2:6] *= 200
    frozen["object_factual_future"][..., 2:6] *= 300
    frozen["effect_xy"] *= 1000
    frozen_path = tmp_path / "frozen.npz"
    np.savez(frozen_path, **frozen)
    config["frozen_test_data"] = str(frozen_path)
    prepared = prepare_identity_inputs(config, dataset)
    assert prepared["evaluation_mode"] == "frozen_test"
    assert prepared["test_index_source"] == "frozen_test_data"
    np.testing.assert_array_equal(prepared["train_indices"], [0, 1])
    np.testing.assert_array_equal(prepared["validation_indices"], [2, 3])
    np.testing.assert_array_equal(prepared["test_indices"], np.arange(6))
    np.testing.assert_array_equal(prepared["evaluation"]["target"], frozen["effect_xy"])
    for key in ("mean", "std"):
        np.testing.assert_array_equal(prepared["normalizer"][key], source_only["normalizer"][key])
    for key in ("target_mean", "target_std"):
        np.testing.assert_array_equal(prepared[key], source_only[key])


def test_empty_train_or_validation_split_is_rejected(dataset):
    dataset["split"][:] = 2
    with pytest.raises(ValueError, match="train"):
        prepare_identity_inputs(minimal_config(), dataset)


def test_smoke_limits_are_shared_per_split(dataset):
    config = minimal_config()
    config["sample_limits"] = {"train": 1, "validation": 1, "test": 1}
    prepared = prepare_identity_inputs(config, dataset)
    np.testing.assert_array_equal(prepared["train_indices"], [0])
    np.testing.assert_array_equal(prepared["validation_indices"], [2])
    np.testing.assert_array_equal(prepared["test_indices"], [4])


@pytest.mark.parametrize("filename", ["identity_continuity_smoke.json", "identity_continuity_merge.json", "identity_continuity_roundabout.json"])
def test_shipped_configs_have_all_conditions(filename):
    config = json.loads((ROOT / "config" / filename).read_text(encoding="utf-8"))
    validate_identity_config(config)
    assert len(config["conditions"]) == 4
    if config.get("smoke_test"):
        assert config["output_dir"] == "experiments/identity_continuity_stage/smoke"
        assert config["capacity_tiers"] == ["65k"]
        assert config["model"]["training_seeds"] == [11]
        assert config["model"]["max_epochs"] <= 2
        assert all(v <= 128 for v in config["sample_limits"].values())
    else:
        assert set(config["capacity_tiers"]) == {"65k", "83k"}
        assert config["model"]["training_seeds"] == [11, 22, 33, 44, 55]


def _save_experiment_fixture(tmp_path, dataset, frozen=False):
    config = minimal_config()
    source_path = tmp_path / "source.npz"
    np.savez(source_path, **dataset)
    config.update(source_data=str(source_path), output_dir=str(tmp_path / "output"))
    if frozen:
        frozen_path = tmp_path / "frozen.npz"
        new_data = copy.deepcopy(dataset)
        new_data["effect_xy"] += 1000
        new_data["episode_seed"] += 10000
        np.savez(frozen_path, **new_data)
        config["frozen_test_data"] = str(frozen_path)
    return config


def _trainer_without_optimization(model, history, suffix, factual_actions,
                                   counterfactual_actions, targets, train_indices,
                                   validation_indices, test_indices, config, seed,
                                   device, progress_callback=None):
    # Only the optimizer loop is replaced; input routing and artifact IO remain real.
    assert not set(train_indices) & set(test_indices)
    assert not set(validation_indices) & set(test_indices)
    assert len(history) == len(suffix) == len(factual_actions) == len(targets)
    assert np.max(np.abs(targets[validation_indices])) < 100
    log = [{"epoch": e, "train_mse": 0.5 / e, "val_mse": 1.0 / e}
           for e in range(1, config["max_epochs"] + 1)]
    for record in log:
        progress_callback({**record, "max_epochs": config["max_epochs"],
                           "best_val_mse": record["val_mse"], "patience": 0})
    return model, np.zeros_like(targets[test_indices]), log


@pytest.mark.parametrize("frozen", [False, True])
def test_runner_saves_reviewable_artifacts_and_reuses_matching_runs(dataset, tmp_path, monkeypatch, capsys, frozen):
    import run_identity_continuity_comparison as runner

    config = _save_experiment_fixture(tmp_path, dataset, frozen)
    config["model"]["max_epochs"] = 6
    monkeypatch.setattr(runner, "train_one", _trainer_without_optimization)
    report = runner.run_experiment(config, device="cpu")
    assert report["device"] == "cpu"
    n_test = 6 if frozen else 2
    assert report["evaluation_mode"] == ("frozen_test" if frozen else "development_test")
    assert report["test_index_source"] == ("frozen_test_data" if frozen else "source_data")
    assert report["source_data_sha256"] == runner.sha256_file(Path(config["source_data"]))
    assert report["test_dataset_sha256"] == runner.sha256_file(Path(config.get("frozen_test_data", config["source_data"])))
    assert report["n_test"] == n_test
    assert report["training_seeds"] == [11]
    assert report["config_signature"]
    output = Path(config["output_dir"])
    archive = np.load(output / "predictions.npz", allow_pickle=False)
    for condition in config["conditions"]:
        name = condition["name"] + "_65k"
        result = report["results"][name]
        assert result["parameters"] == result["runs"][0]["parameters"]
        assert result["hidden_dim"] >= 8
        assert np.asarray(result["fde_by_seed_episode"]).shape == (1, n_test)
        assert np.asarray(result["ade_by_seed_episode"]).shape == (1, n_test)
        assert archive[name].shape == (1, n_test, 4, 2)
        assert (output / f"{name}_seed11.pt").is_file()
        assert (output / f"{name}_seed11_training.json").is_file()
        assert (output / f"{name}_seed11_prediction.npy").is_file()
        assert (output / f"{name}_seed11_run.json").is_file()
    archive.close()
    text = capsys.readouterr().out
    assert "[scene=merge] [tier=65k] [condition=stable_temporal] [seed=11] epoch=5/6" in text
    assert "elapsed=" in text and "eta=" in text
    assert "epoch=2/6" not in text

    def fail_training(*args, **kwargs):
        pytest.fail("Completed runs must resume without training")
    monkeypatch.setattr(runner, "train_one", fail_training)
    resumed = runner.run_experiment(config, resume=True)
    assert resumed["results"] == report["results"]
    with pytest.raises(FileExistsError):
        runner.run_experiment(config)
    config["permutation_seed"] += 1
    with pytest.raises(ValueError, match="match"):
        runner.run_experiment(config, resume=True)


def test_incomplete_run_signature_is_checked_before_retraining(dataset, tmp_path, monkeypatch):
    import run_identity_continuity_comparison as runner
    config = _save_experiment_fixture(tmp_path, dataset)
    output = Path(config["output_dir"])
    output.mkdir()
    metadata = output / "stable_temporal_65k_seed11_run.json"
    metadata.write_text(json.dumps({"signature": "stale", "summary": {"status": "running"}}))
    def fail_training(*args, **kwargs):
        pytest.fail("Mismatched partial run must fail before training")
    monkeypatch.setattr(runner, "train_one", fail_training)
    with pytest.raises(ValueError, match="match"):
        runner.run_experiment(config, resume=True)


def test_run_signature_covers_seed_config_and_both_data_sources():
    import run_identity_continuity_comparison as runner
    config = minimal_config()
    condition = config["conditions"][0]
    args = dict(condition=condition, config=config, source_sha256="source",
                frozen_sha256="frozen", capacity_tier="65k", seed=11)
    baseline = runner.run_signature(**args)
    for name, value in (("seed", 22), ("source_sha256", "source2"),
                        ("frozen_sha256", "frozen2"), ("capacity_tier", "83k")):
        assert runner.run_signature(**{**args, name: value}) != baseline
    changed = copy.deepcopy(config)
    changed["model"]["learning_rate"] *= 2
    assert runner.run_signature(**{**args, "config": changed}) != baseline


def test_cli_accepts_config_device_and_resume_defaults():
    import run_identity_continuity_comparison as runner
    args = runner.parse_args(["--config", "config/custom.json", "--device", "cpu"])
    assert args.config == "config/custom.json"
    assert args.device == "cpu"
    assert args.resume is True
    assert runner.parse_args(["--no-resume"]).resume is False
