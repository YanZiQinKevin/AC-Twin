from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_protocol as s3p  # noqa: E402
from ac_twin.stage3_data import FROZEN_THRESHOLDS  # noqa: E402


def _config(tmp_path):
    return {
        "plan_id": s3p.PLAN_ID,
        "development_source": "data/object_centric_stage/merge_2000/object_twinworld_2000.npz",
        "response_bins": dict(FROZEN_THRESHOLDS),
        "formal_holdouts": {
            s3p.GATEC_ID_NAME: {"durations": [1, 3]},
            s3p.GATEC_UNSEEN_NAME: {"durations": [2]},
        },
    }


# --- config validation ----------------------------------------------------------


def test_config_valid() -> None:
    s3p.validate_stage3_config(_config(None))


def test_config_rejects_forbidden_keys() -> None:
    for key in ("retrain", "fine_tune", "fit_response_bins", "sample_limits"):
        config = _config(None)
        config[key] = True
        with pytest.raises(ValueError, match="forbidden"):
            s3p.validate_stage3_config(config)


def test_config_rejects_sealed_holdout_as_source() -> None:
    config = _config(None)
    config["development_source"] = (
        "data/object_centric_stage/stage2_merge_1200/object_twinworld_stage2_merge_1200.npz"
    )
    with pytest.raises(ValueError, match="sealed"):
        s3p.validate_stage3_config(config)


def test_config_rejects_sealed_holdout_by_recorded_hash() -> None:
    from ac_twin.stage3_data import STAGE2_HOLDOUT_FILE_SHA256
    config = _config(None)
    config["development_source_sha256"] = STAGE2_HOLDOUT_FILE_SHA256
    with pytest.raises(ValueError, match="sealed"):
        s3p.validate_stage3_config(config)


def test_config_rejects_sealed_holdout_anywhere() -> None:
    config = _config(None)
    config["some_other_data"] = (
        "data/object_centric_stage/stage2_merge_1200/object_twinworld_stage2_merge_1200.npz"
    )
    with pytest.raises(ValueError, match="sealed"):
        s3p.validate_stage3_config(config)


def test_config_rejects_wrong_thresholds() -> None:
    config = _config(None)
    config["response_bins"] = {"zero_tolerance": 0.001, "t1": 0.5, "t2": 2.0}
    with pytest.raises(ValueError, match="frozen"):
        s3p.validate_stage3_config(config)


def test_config_formal_duration_rules() -> None:
    config = _config(None)
    config["formal_holdouts"][s3p.GATEC_UNSEEN_NAME]["durations"] = [1]
    with pytest.raises(ValueError, match="unseen"):
        s3p.validate_stage3_config(config)
    config = _config(None)
    config["formal_holdouts"][s3p.GATEC_ID_NAME]["durations"] = [1, 2]
    with pytest.raises(ValueError, match="durations"):
        s3p.validate_stage3_config(config)


# --- output signature -------------------------------------------------------------


def test_output_signature_deterministic_and_recomputable(tmp_path) -> None:
    report = {"gate": "C", "c_uniform": 0.1, "nested": {"a": [1, 2, 3]}}
    sig = s3p.output_signature(report)
    assert sig == s3p.output_signature(report)
    path = tmp_path / "report.json"
    path.write_text(json.dumps({**report, "output_signature": sig}), encoding="utf-8")
    assert s3p.output_signature_from_disk(path) == sig


def test_output_signature_detects_tampering(tmp_path) -> None:
    report = {"gate": "C", "c_uniform": 0.1}
    sig = s3p.output_signature(report)
    path = tmp_path / "report.json"
    # store a signature that does not match the (modified) body
    path.write_text(json.dumps({"gate": "C", "c_uniform": 0.2, "output_signature": sig}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        s3p.output_signature_from_disk(path)


# --- single unlock ------------------------------------------------------------------


def test_single_unlock_refused_when_report_exists(tmp_path) -> None:
    path = tmp_path / "report.json"
    s3p.assert_single_unlock(path)  # no report -> ok
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="second unblinding"):
        s3p.assert_single_unlock(path)


# --- joint lock ------------------------------------------------------------------------


def _seeds(offset):
    return np.arange(offset, offset + 50, dtype=np.int64)


def test_joint_lock_binds_both_holdouts(tmp_path) -> None:
    historical = np.arange(10000, 20000, dtype=np.int64)
    schedules = {
        s3p.GATEC_ID_NAME: _seeds(1),
        s3p.GATEC_UNSEEN_NAME: _seeds(1000),
    }
    lock = s3p.write_joint_lock(_config(tmp_path), tmp_path, historical_seeds=historical, seed_schedules=schedules)
    assert set(lock["seed_schedule_sha256"]) == set(s3p.FORMAL_HOLDOUTS)
    assert (tmp_path / "stage3_joint_lock.json").exists()
    verified = s3p.verify_joint_lock(tmp_path)
    assert verified["lock_sha256"] == lock["lock_sha256"]


def test_joint_lock_refuses_overwrite(tmp_path) -> None:
    historical = np.arange(10000, 20000, dtype=np.int64)
    schedules = {s3p.GATEC_ID_NAME: _seeds(1), s3p.GATEC_UNSEEN_NAME: _seeds(1000)}
    s3p.write_joint_lock(_config(tmp_path), tmp_path, historical_seeds=historical, seed_schedules=schedules)
    with pytest.raises(FileExistsError):
        s3p.write_joint_lock(_config(tmp_path), tmp_path, historical_seeds=historical, seed_schedules=schedules)


def test_joint_lock_rejects_overlapping_holdouts(tmp_path) -> None:
    historical = np.arange(10000, 20000, dtype=np.int64)
    schedules = {s3p.GATEC_ID_NAME: _seeds(1), s3p.GATEC_UNSEEN_NAME: _seeds(1)}
    with pytest.raises(ValueError, match="overlap"):
        s3p.write_joint_lock(_config(tmp_path), tmp_path, historical_seeds=historical, seed_schedules=schedules)


def test_joint_lock_rejects_historical_overlap(tmp_path) -> None:
    historical = np.arange(1, 60, dtype=np.int64)  # overlaps _seeds(1)
    schedules = {s3p.GATEC_ID_NAME: _seeds(1), s3p.GATEC_UNSEEN_NAME: _seeds(1000)}
    with pytest.raises(ValueError):
        s3p.write_joint_lock(_config(tmp_path), tmp_path, historical_seeds=historical, seed_schedules=schedules)


def test_verify_joint_lock_detects_tampering(tmp_path) -> None:
    historical = np.arange(10000, 20000, dtype=np.int64)
    schedules = {s3p.GATEC_ID_NAME: _seeds(1), s3p.GATEC_UNSEEN_NAME: _seeds(1000)}
    s3p.write_joint_lock(_config(tmp_path), tmp_path, historical_seeds=historical, seed_schedules=schedules)
    lock_path = tmp_path / "stage3_joint_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["n_historical_seeds"] = 999
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="tampering"):
        s3p.verify_joint_lock(tmp_path)


# --- Gate Q -------------------------------------------------------------------------------


def _gate_q_checks(ok=True):
    return {
        "lock_hash_match": ok,
        "seed_schedule_zero_overlap": ok,
        "holdouts_mutually_exclusive": ok,
        "duration_set_asserted": ok,
        "no_sealed_stage2_leakage": ok,
        "no_oracle_leakage": ok,
        "parameter_budget_ok": ok,
        "data_output_hash_verified": ok,
    }


def test_gate_q_passes_and_fails() -> None:
    assert s3p.evaluate_gate_q(_gate_q_checks())["passed"]
    checks = _gate_q_checks()
    checks["no_sealed_stage2_leakage"] = False
    assert not s3p.evaluate_gate_q(checks)["passed"]


def test_gate_q_requires_all_checks() -> None:
    checks = _gate_q_checks()
    del checks["duration_set_asserted"]
    with pytest.raises(ValueError, match="missing"):
        s3p.evaluate_gate_q(checks)
