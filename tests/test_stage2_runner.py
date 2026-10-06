"""Task 5 tests: Stage 2 four-phase runner and the single-unblinding discipline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_stage2_confirmatory as runner  # noqa: E402
from ac_twin import stage2_protocol as sp  # noqa: E402

SMOKE_CONFIG = ROOT / "config" / "stage2_holdout_merge_smoke.json"
N_AGENTS, T_H, T_F = 5, 10, 20
E = 24


def _stub_params(config):
    return {
        "beta_action_horizon": np.zeros(3, dtype=np.float64),
        "beta_action_horizon_distance": np.zeros(4, dtype=np.float64),
        "probe_coefficients": np.zeros((65, 6), dtype=np.float64),
        "probe_coefficients_shuffled": np.zeros((65, 6), dtype=np.float64),
        "probe_scene_mean": np.zeros(6, dtype=np.float64),
    }


def _write_holdout(path: Path, n=E, seed=7):
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(n, T_H, N_AGENTS, 8)).astype(np.float32)
    factual = rng.normal(size=(n, T_F, N_AGENTS, 8)).astype(np.float32)
    counterfactual = factual + rng.normal(scale=0.5, size=(n, T_F, N_AGENTS, 8)).astype(np.float32)
    for arr in (history, factual, counterfactual):
        arr[..., 0] = 1.0
        arr[..., 1] = 0.0
        arr[:, :, 0, 1] = 1.0
        arr[..., 6] = 1.0  # cos_heading
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        object_history=history,
        object_factual_future=factual,
        object_counterfactual_future=counterfactual,
        factual_actions=rng.integers(0, 5, size=(n, T_F)).astype(np.int64),
        counterfactual_actions=rng.integers(0, 5, size=(n, T_F)).astype(np.int64),
        cf_action=rng.integers(0, 5, size=n).astype(np.int64),
        cf_duration=rng.choice([1, 3], size=n).astype(np.int64),
        episode_seed=(9000 + np.arange(n)).astype(np.int64),
        t_star=np.full(n, 12, dtype=np.int64),
        split=np.full(n, 2, dtype=np.int8),
    )


def _build(tmp_path, monkeypatch, status):
    monkeypatch.setattr(sp, "validate_stage2_config", lambda cfg: None)
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config["analysis_status"] = status
    out_dir = tmp_path / "out"
    config["output_dir"] = str(out_dir)
    config["n_pairs"] = E
    config["data_output"] = str(tmp_path / "data" / "holdout.npz")
    config["rejection_ledger"] = str(tmp_path / "data" / "ledger.jsonl")
    config["collection_receipt"] = str(tmp_path / "data" / "receipt.json")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    _write_holdout(Path(config["data_output"]))

    # Minimal collection receipt for Gate Q.
    Path(config["collection_receipt"]).write_text(json.dumps({
        "accepted_pairs": E, "attempts": E, "prefix_max_abs_diff": 0.0,
        "identity_max_abs_diff": 0.0, "training_performed": False,
    }), encoding="utf-8")

    lock = sp.write_protocol_lock(cfg_path, out_dir, params_provider=_stub_params)
    return {"config": config, "cfg_path": cfg_path, "out_dir": out_dir, "lock": lock}


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch, "ENGINEERING_SMOKE_ONLY")


@pytest.fixture()
def pipeline_confirm(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch, "CONFIRMATORY_HOLDOUT")


def test_verify_lock_phase(pipeline, capsys):
    runner.cmd_verify_lock(pipeline["cfg_path"])
    out = capsys.readouterr().out
    assert "PROTOCOL_LOCK_VERIFIED" in out


def test_infer_writes_predictions_without_gate_numbers(pipeline, capsys):
    receipt = runner.cmd_infer(pipeline["cfg_path"], "cpu")
    out = capsys.readouterr().out
    assert "V_evt" not in out and "GATE" not in out and "ci_lower" not in out
    n = len(receipt["prediction_sha256"])
    # 15 main + 5 swap + 20 alt + 2 probe-summary files.
    assert n == 15 + 5 + 20 + 2
    assert len(receipt["manifest_sha256"]) == 5
    assert (pipeline["out_dir"] / "prediction_receipt.json").is_file()


def test_infer_existing_receipt_verifies_without_overwrite(pipeline, capsys):
    runner.cmd_infer(pipeline["cfg_path"], "cpu")
    pred = pipeline["out_dir"] / "uniform_suffix_seed11_prediction.npy"
    mtime = pred.stat().st_mtime_ns
    receipt = runner.cmd_infer(pipeline["cfg_path"], "cpu")
    assert receipt.get("verification") == "existing_predictions_verified_no_overwrite"
    assert pred.stat().st_mtime_ns == mtime


def test_infer_fails_without_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "validate_stage2_config", lambda cfg: None)
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path / "empty")
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        runner.cmd_infer(cfg_path, "cpu")


def test_infer_fails_on_lock_drift(pipeline):
    schedule = np.load(pipeline["out_dir"] / "seed_schedule.npy").copy()
    schedule[0] = schedule[-1]
    np.save(pipeline["out_dir"] / "seed_schedule.npy", schedule)
    with pytest.raises((ValueError, RuntimeError)):
        runner.cmd_infer(pipeline["cfg_path"], "cpu")


def test_unlock_and_analyze_writes_signed_report(pipeline_confirm):
    runner.cmd_infer(pipeline_confirm["cfg_path"], "cpu")
    report = runner.cmd_unlock_and_analyze(pipeline_confirm["cfg_path"],
                                           pipeline_confirm["lock"]["lock_sha256"])
    assert (pipeline_confirm["out_dir"] / "report.json").is_file()
    assert (pipeline_confirm["out_dir"] / "STAGE2_CONFIRMATORY_DECISION.md").is_file()
    assert "final_status" in report
    assert "output_signature" in report


def test_unlock_signature_recomputable_from_disk(pipeline_confirm):
    runner.cmd_infer(pipeline_confirm["cfg_path"], "cpu")
    runner.cmd_unlock_and_analyze(pipeline_confirm["cfg_path"],
                                  pipeline_confirm["lock"]["lock_sha256"])
    on_disk = json.loads((pipeline_confirm["out_dir"] / "report.json").read_text(encoding="utf-8"))
    sig = on_disk.pop("output_signature")
    assert runner._signature(on_disk) == sig


def test_unlock_refuses_second_unblinding(pipeline_confirm):
    runner.cmd_infer(pipeline_confirm["cfg_path"], "cpu")
    runner.cmd_unlock_and_analyze(pipeline_confirm["cfg_path"],
                                  pipeline_confirm["lock"]["lock_sha256"])
    with pytest.raises(FileExistsError):
        runner.cmd_unlock_and_analyze(pipeline_confirm["cfg_path"],
                                      pipeline_confirm["lock"]["lock_sha256"])


def test_unlock_rejects_wrong_expected_hash(pipeline_confirm):
    runner.cmd_infer(pipeline_confirm["cfg_path"], "cpu")
    with pytest.raises(ValueError):
        runner.cmd_unlock_and_analyze(pipeline_confirm["cfg_path"], "0" * 64)


def test_smoke_unlock_is_engineering_only(pipeline):
    runner.cmd_infer(pipeline["cfg_path"], "cpu")
    report = runner.cmd_unlock_and_analyze(pipeline["cfg_path"],
                                           pipeline["lock"]["lock_sha256"])
    # The pipeline fixture uses the smoke config -> ENGINEERING_SMOKE_ONLY.
    assert report["analysis_status"] == "ENGINEERING_SMOKE_ONLY"
    assert "final_status" not in report  # no formal gate judgment in smoke
    assert "gate_s_star" not in report
