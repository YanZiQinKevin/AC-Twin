from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_stage1_plus_diagnostic import (  # noqa: E402
    validate_stage1_plus_config,
    verify_sealed_inputs,
)

CONFIG_PATH = ROOT / "config" / "stage1_plus_merge.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_frozen_config_is_valid():
    # The frozen on-disk config must pass validation unchanged.
    validate_stage1_plus_config(_load_config())


@pytest.mark.parametrize("field,value", [
    ("scene", "roundabout"),
    ("experiment_name", "something_else"),
    ("analysis_status", "CONFIRMATORY"),
])
def test_invalid_scalar_fields_fail_before_output(field, value):
    config = _load_config()
    config[field] = value
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)


def test_non_merge_scene_rejected():
    config = _load_config()
    config["scene"] = "roundabout"
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)


def test_training_seeds_must_equal_frozen_five():
    config = _load_config()
    config["training_seeds"] = [11, 22, 33, 44]
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)
    config = _load_config()
    config["training_seeds"] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)


def test_missing_condition_rejected():
    config = _load_config()
    config["conditions"] = ["no_suffix", "uniform_suffix"]
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)


def test_bootstrap_resamples_must_be_10000():
    config = _load_config()
    config["statistics"]["bootstrap_resamples"] = 1000
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)


def test_power_replicates_must_be_10000():
    config = _load_config()
    config["statistics"]["power_monte_carlo_replicates"] = 500
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)


def test_output_dir_inside_stage1_merge_dir_rejected():
    config = _load_config()
    # Output inside (or equal to) the Stage 1 merge directory must be rejected.
    config["output_dir"] = "experiments/stage1_evidence_value/merge"
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)
    config = _load_config()
    config["output_dir"] = "experiments/stage1_evidence_value/merge/stage1_plus_merge"
    with pytest.raises(ValueError):
        validate_stage1_plus_config(config)


def test_verify_sealed_inputs_passes_on_real_artifacts():
    hashes = verify_sealed_inputs(_load_config())
    # Must cover source data, stage1 config, response bins, test manifest and
    # the 15 test predictions.
    assert "source_data" in hashes
    assert "stage1_config" in hashes
    assert "response_bins" in hashes
    assert "test_manifest" in hashes
    preds = [k for k in hashes if k.endswith("_test_prediction.npy")]
    assert len(preds) == 15


def test_verify_sealed_inputs_fails_on_hash_mismatch(tmp_path):
    config = _load_config()
    manifest = json.loads(
        (ROOT / config["sealed_hash_manifest"]).read_text(encoding="utf-8"))
    # Corrupt a single expected hash; verification must fail before any output.
    manifest["source_data"]["merge"] = "0" * 64
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps(manifest), encoding="utf-8")
    config["sealed_hash_manifest"] = str(bad)
    with pytest.raises(ValueError):
        verify_sealed_inputs(config)


def test_verify_sealed_inputs_fails_on_prediction_mismatch(tmp_path):
    config = _load_config()
    manifest = json.loads(
        (ROOT / config["sealed_hash_manifest"]).read_text(encoding="utf-8"))
    manifest["merge"]["no_suffix_seed11_test_prediction.npy"] = "f" * 64
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps(manifest), encoding="utf-8")
    config["sealed_hash_manifest"] = str(bad)
    with pytest.raises(ValueError):
        verify_sealed_inputs(config)


# --- Task 5: runner isolation -----------------------------------------------------

import run_stage1_plus_diagnostic as _runner  # noqa: E402


def _reduced_config(tmp_path):
    """In-memory config with small resample counts and a temp output dir."""
    config = _load_config()
    config["statistics"]["bootstrap_resamples"] = 200
    config["statistics"]["power_monte_carlo_replicates"] = 200
    config["statistics"]["candidate_episode_counts"] = [300, 600]
    config["output_dir"] = str(tmp_path / "stage1_plus_merge")
    return config


def test_runner_does_not_import_torch_or_stage1_trainer():
    # The runner must stay isolated from torch and the Stage 1 training module.
    assert not hasattr(_runner, "torch")
    assert "run_stage1_evidence_value" not in vars(_runner)
    # The runner module source must not reference torch, the trainer, training
    # or backprop, and must not load checkpoints.
    source = (ROOT / "code" / "run_stage1_plus_diagnostic.py").read_text(encoding="utf-8")
    assert "run_stage1_evidence_value" not in source
    assert "import torch" not in source
    assert "torch.optim" not in source
    assert ".backward" not in source
    assert "torch.load" not in source


def test_load_analysis_inputs_never_loads_pt(tmp_path, monkeypatch):
    loaded = []
    real_load = np.load

    def tracking_load(file, *args, **kwargs):
        loaded.append(str(file))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", tracking_load)
    config = _reduced_config(tmp_path)
    _runner.load_analysis_inputs(config)
    assert loaded, "expected the runner to load prediction arrays"
    assert all(not p.endswith(".pt") for p in loaded)


def test_runner_completes_with_stage1_trainer_sentinel(tmp_path, monkeypatch):
    # Monkeypatch the Stage 1 training/prediction entry points to raising
    # sentinels; Stage 1 Plus must still complete because it never uses them.
    import run_stage1_evidence_value as s1

    def _sentinel(*args, **kwargs):
        raise AssertionError("Stage 1 training/inference must not run in Stage 1 Plus")

    monkeypatch.setattr(s1, "train_stage1_run", _sentinel, raising=True)
    monkeypatch.setattr(s1, "_predict", _sentinel, raising=True)
    config = _reduced_config(tmp_path)
    report = _runner.execute_and_write(config)
    assert report["training_performed"] is False
    assert report["inference_performed"] is False
    assert report["stage1_gate_a_unchanged"] is True


def test_runner_writes_only_into_plus_dir_and_conserves_hashes(tmp_path):
    config = _reduced_config(tmp_path)
    out_dir = Path(config["output_dir"])
    report = _runner.execute_and_write(config)
    produced = {p.name for p in out_dir.iterdir()}
    assert produced == {"input_hashes.json", "report.json",
                        "STAGE1_PLUS_DIAGNOSTIC_DECISION_2026-09-07.md"}
    # The plus dir is not the Stage 1 merge dir and Stage 1 hashes are conserved.
    assert "stage1_evidence_value/merge" not in str(out_dir).replace("\\", "/")
    assert report["hash_conservation"]["before_after_identical"] is True


def test_runner_repeated_run_is_deterministic(tmp_path):
    config = _reduced_config(tmp_path)
    a = _runner.execute_and_write(config)
    b = _runner.execute_and_write(config)
    assert a["reproduction"] == b["reproduction"]
    assert a["shape_flags"] == b["shape_flags"]
    assert a["shape_recommendation"] == b["shape_recommendation"]
    assert a["power_analysis"] == b["power_analysis"]


def test_runner_reproduces_stage1_point_estimates(tmp_path):
    config = _reduced_config(tmp_path)
    report = _runner.execute_and_write(config)
    assert report["reproduction"]["all_within_tolerance"] is True
