"""Stage 3A Plus Task 10: qualification lock, frozen ten-seed inference and the
single Gate R+ adjudication.

Covers: state gating, the single-use lock (binds data + freeze receipt +
hash chain), refusal BEFORE the adjudication is consumed (replaced data /
replaced checkpoint / modified Gate params), the used-lock burned atomically
before the qualification truth is read, second-call refusal, the frozen formal
bootstrap (10,000 resamples, seed 20260929) and the three terminal outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_stage3a_plus as drv  # noqa: E402
from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin import stage3a_plus_statistics as sps  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3_data import (  # noqa: E402
    STAGE3A_PLUS_ALL_SEEDS,
    STAGE3A_PLUS_DEV_SEEDS,
    STAGE3A_PLUS_PLAN_ID,
    STAGE3A_PLUS_STRESS_SEEDS,
)
from ac_twin.stage3_response import Stage3ResponsePredictor  # noqa: E402

from test_stage3a_plus_qualification import _write_fake_npz  # noqa: E402

PURPOSE = "stage3a_plus_qualification"
N_EPISODES = 20


# --------------------------------------------------------------------------- #
# fixture: a run dir at MANUAL_COLLECTION_COMPLETE with real tiny checkpoints
# --------------------------------------------------------------------------- #
def _make_checkpoints(run_dir: Path) -> dict[str, dict[str, str]]:
    out = {}
    for seed in STAGE3A_PLUS_ALL_SEEDS:
        torch.manual_seed(seed)
        model = Stage3ResponsePredictor(hidden_dim=8, response_feature_dim=4)
        path = run_dir / "checkpoints" / f"qual_seed{seed}.pt"
        rec = s3t.save_predictor_checkpoint(path, model.state_dict(),
                                            meta={"seed": int(seed)})
        out[str(seed)] = {"path": str(path), "signature": rec["signature"]}
    return out


def _freeze_payload(run_dir: Path, checkpoints, *, ece_guard=None) -> dict:
    return {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "final_freeze",
        "round": 0,
        "config": {"final_config": {"hidden_dim": 8, "response_feature_dim": 4,
                                    "lambda_cls": 0.1},
                   "training": {"max_epochs": 1}},
        "calibration": {"selected_candidate": "SHARED_T",
                        "selected_artifact": {"cali": "SHARED_T"},
                        "calibration_signature": "calsig",
                        "selection_receipt_sha256": "sel"},
        "checkpoints": checkpoints,
        "baselines": {"strongest": {"continuous": "brier", "brier": "clf"}},
        "gate": {
            "definition": "GATE_R_PLUS = core AND stability AND Q+",
            "core_metrics": list(sps.CORE_METRICS),
            "ece_guard": sps.ECE_GUARD if ece_guard is None else ece_guard,
            "ece_relative_improvement": sps.ECE_RELATIVE_IMPROVEMENT,
            "min_positive_brier_seeds": sps.MIN_POSITIVE_BRIER_SEEDS,
            "worst_seed_noninferiority_frac": sps.WORST_SEED_NONINFERIORITY_FRAC,
            "nondegenerate_std_floor": sps.NONDEGENERATE_STD_FLOOR,
            "dev_seeds": list(STAGE3A_PLUS_DEV_SEEDS),
            "stress_seeds": list(STAGE3A_PLUS_STRESS_SEEDS),
        },
        "code_sha256": {},
        "dev_report_sha256": {"A": "a", "B": "b", "file": "f"},
        "generated_at": "2026-09-09T00:00:00",
    }


def _qual_run_setup(tmp_path: Path, *, state: str = "MANUAL_COLLECTION_COMPLETE",
                    ece_guard=None):
    """Run dir with freeze receipt, collected (fake) qualification data, and the
    collection receipt hash chain; state walked to ``state``."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    cfg = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "dev_seeds": list(STAGE3A_PLUS_DEV_SEEDS),
        "stress_seeds": list(STAGE3A_PLUS_STRESS_SEEDS),
        "development_source": "a", "replacement_source": "b", "prior_run_dir": "c",
        "final_config": {"hidden_dim": 8, "response_feature_dim": 4, "lambda_cls": 0.1},
        "training": {"max_epochs": 1}, "device": "cpu", "run_dir": str(run_dir),
    }
    config_path = run_dir / "dev_config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    checkpoints = _make_checkpoints(run_dir)
    payload = _freeze_payload(run_dir, checkpoints, ece_guard=ece_guard)
    freeze_dir = run_dir / "freeze"
    freeze_dir.mkdir(parents=True)
    (freeze_dir / "final_freeze_receipt.json").write_text(json.dumps({
        "receipt_payload": payload,
        "receipt_sha256": canonical_fingerprint(payload),
    }), encoding="utf-8")

    qual_dir = run_dir / "qualification"
    data_dir = tmp_path / "data" / "object_centric_stage" / PURPOSE
    data_path = data_dir / "object_twinworld_stage3a_plus_qualification.npz"
    seeds = np.arange(20_000, 20_000 + N_EPISODES)
    pairs = [(a, d) for a in range(5) for d in (1, 3)] * 2
    _write_fake_npz(data_path, seeds, [p[0] for p in pairs], [p[1] for p in pairs])
    data_sha = sha256_file(data_path)

    power_dir = run_dir / "power"
    power_dir.mkdir(parents=True)
    (power_dir / "power_report.json").write_text(json.dumps(
        {"status": "POWER_FEASIBLE", "selected_episode_count": N_EPISODES}),
        encoding="utf-8")
    power_sha = sha256_file(power_dir / "power_report.json")
    schedule_doc = {"schedule": [int(s) for s in seeds], "schedule_sha256": "schedsha"}
    qual_dir.mkdir(parents=True, exist_ok=True)
    (qual_dir / "seed_schedule.json").write_text(json.dumps(schedule_doc),
                                                 encoding="utf-8")
    (qual_dir / "config.json").write_text(json.dumps(
        {"data_output": str(data_path), "n_pairs": N_EPISODES}), encoding="utf-8")
    (qual_dir / "collection_receipt.json").write_text(json.dumps({
        "purpose": PURPOSE, "plan_id": STAGE3A_PLUS_PLAN_ID,
        "data_sha256": data_sha, "power_report_sha256": power_sha,
        "seed_schedule_sha256": "schedsha",
        "qa": {"strata_equal_counts": True, "episodes_per_stratum": 2},
    }), encoding="utf-8")

    if state != "STAGE3A_PLUS_DESIGN_FROZEN":
        for target in ("SOFTWARE_TESTED", "FORENSIC_DIAGNOSTIC_COMPLETE",
                       "DEV_ROUND_0_COMPLETE", "STAGE3A_PLUS_DEVELOPMENT_STABLE",
                       "FINAL_MODEL_FROZEN", "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS",
                       "QUALIFICATION_POWER_FROZEN", "MANUAL_COLLECTION_COMPLETE"):
            drv.advance_state(run_dir, target)
            if target == state:
                break
    return run_dir, config_path, data_path, checkpoints


def _args(config_path):
    return argparse.Namespace(config=str(config_path))


def _fake_infer(qual, checkpoints, config):
    E = len(qual["roles"]["eval"])
    return {int(seed): {
        "r_hat": np.full((E, 10, 5), 0.5, dtype=np.float64),
        "regime_logits": np.tile(np.arange(4, dtype=np.float64), (E, 10, 5, 1)),
    } for seed in STAGE3A_PLUS_ALL_SEEDS}


def _gate_runner_passed(**kwargs):
    return {"report": {"gate_q_passed": True, "core": {}, "stability": {}},
            "evaluation": {"passed": True}, "passed": True}


def _run_through_inference(tmp_path):
    run_dir, config_path, data_path, ckpts = _qual_run_setup(tmp_path)
    assert drv.cmd_lock_qualification(_args(config_path)) == 0
    rc = drv.cmd_qualification_infer(_args(config_path), infer_fn=_fake_infer)
    assert rc == 0
    return run_dir, config_path, data_path, ckpts


# --------------------------------------------------------------------------- #
# lock-qualification
# --------------------------------------------------------------------------- #
def test_lock_binds_data_freeze_and_hash_chain(tmp_path):
    run_dir, config_path, data_path, _ = _qual_run_setup(tmp_path)
    rc = drv.cmd_lock_qualification(_args(config_path))
    assert rc == 0
    lock_path = run_dir / "qualification" / "qualification_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["plan_id"] == STAGE3A_PLUS_PLAN_ID
    assert lock["qualification_data_sha256"] == sha256_file(data_path)
    freeze_doc = json.loads(
        (run_dir / "freeze" / "final_freeze_receipt.json").read_text(encoding="utf-8"))
    assert lock["freeze_receipt_sha256"] == freeze_doc["receipt_sha256"]
    assert lock["power_report_sha256"] == sha256_file(run_dir / "power" / "power_report.json")
    # single-shot: a second lock is refused and nothing is rewritten
    assert drv.cmd_lock_qualification(_args(config_path)) == 2
    assert json.loads(lock_path.read_text(encoding="utf-8")) == lock


def test_lock_refuses_wrong_state(tmp_path):
    run_dir, config_path, _, _ = _qual_run_setup(tmp_path, state="FINAL_MODEL_FROZEN")
    assert drv.cmd_lock_qualification(_args(config_path)) == 2
    assert not (run_dir / "qualification" / "qualification_lock.json").exists()


def test_lock_refuses_replaced_data(tmp_path):
    run_dir, config_path, data_path, _ = _qual_run_setup(tmp_path)
    # Replace the data AFTER collection (bytes no longer match the receipt).
    _write_fake_npz(data_path, np.arange(30_000, 30_000 + N_EPISODES),
                    [0] * N_EPISODES, [1] * N_EPISODES)
    assert drv.cmd_lock_qualification(_args(config_path)) == 3
    assert not (run_dir / "qualification" / "qualification_lock.json").exists()


# --------------------------------------------------------------------------- #
# qualification-infer
# --------------------------------------------------------------------------- #
def test_infer_writes_ten_predictions_and_advances(tmp_path):
    run_dir, config_path, data_path, _ = _qual_run_setup(tmp_path)
    assert drv.cmd_lock_qualification(_args(config_path)) == 0
    rc = drv.cmd_qualification_infer(_args(config_path), infer_fn=_fake_infer)
    assert rc == 0
    pred_dir = run_dir / "qualification" / "predictions"
    for seed in STAGE3A_PLUS_ALL_SEEDS:
        assert (pred_dir / f"seed_{seed}.npz").is_file()
    receipt = json.loads((pred_dir / "inference_receipt.json").read_text(encoding="utf-8"))
    assert receipt["plan_id"] == STAGE3A_PLUS_PLAN_ID
    assert receipt["data_sha256"] == sha256_file(data_path)
    assert receipt["n_episodes"] == N_EPISODES
    assert len(receipt["seeds"]) == 10
    assert drv.load_state(run_dir)["state"] == "FROZEN_INFERENCE_COMPLETE"
    # idempotent restart: verifies, never rewrites
    before = {p.name: p.read_bytes() for p in pred_dir.glob("*.npz")}
    assert drv.cmd_qualification_infer(_args(config_path), infer_fn=_fake_infer) == 0
    assert {p.name: p.read_bytes() for p in pred_dir.glob("*.npz")} == before


def test_infer_requires_lock(tmp_path):
    run_dir, config_path, _, _ = _qual_run_setup(tmp_path)
    assert drv.cmd_qualification_infer(_args(config_path), infer_fn=_fake_infer) == 2
    assert not (run_dir / "qualification" / "predictions").exists()


def test_infer_refuses_replaced_data(tmp_path):
    run_dir, config_path, data_path, _ = _qual_run_setup(tmp_path)
    assert drv.cmd_lock_qualification(_args(config_path)) == 0
    _write_fake_npz(data_path, np.arange(30_000, 30_000 + N_EPISODES),
                    [0] * N_EPISODES, [1] * N_EPISODES)
    assert drv.cmd_qualification_infer(_args(config_path), infer_fn=_fake_infer) == 3
    assert not (run_dir / "qualification" / "predictions").exists()


def test_infer_refuses_replaced_checkpoint(tmp_path):
    run_dir, config_path, _, checkpoints = _qual_run_setup(tmp_path)
    assert drv.cmd_lock_qualification(_args(config_path)) == 0
    # Swap one frozen checkpoint for a differently-seeded model.
    victim = checkpoints[str(STAGE3A_PLUS_ALL_SEEDS[0])]["path"]
    torch.manual_seed(999)
    model = Stage3ResponsePredictor(hidden_dim=8, response_feature_dim=4)
    s3t.save_predictor_checkpoint(victim, model.state_dict(), meta={"seed": 999})
    assert drv.cmd_qualification_infer(_args(config_path), infer_fn=_fake_infer) == 3
    assert not (run_dir / "qualification" / "predictions").exists()


# --------------------------------------------------------------------------- #
# gate-r-plus-qualification (the single adjudication)
# --------------------------------------------------------------------------- #
def test_gate_adjudicates_once_and_qualifies(tmp_path):
    run_dir, config_path, _, _ = _run_through_inference(tmp_path)
    rc = drv.cmd_gate_r_plus_qualification(
        _args(config_path), gate_runner=_gate_runner_passed)
    assert rc == 0
    used = run_dir / "qualification" / "qualification_used.lock"
    assert used.is_file()
    report_path = run_dir / "qualification" / "gate_r_plus_qualification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["plan_id"] == STAGE3A_PLUS_PLAN_ID
    assert report["n_resamples"] == 10_000
    assert report["bootstrap_seed"] == 20_260_929
    assert report["outcome"] == "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS"
    assert drv.load_state(run_dir)["state"] == "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS"
    # a second adjudication is refused
    assert drv.cmd_gate_r_plus_qualification(
        _args(config_path), gate_runner=_gate_runner_passed) == 3


def test_gate_scientific_failure_terminal(tmp_path):
    run_dir, config_path, _, _ = _run_through_inference(tmp_path)

    def failed(**kwargs):
        return {"report": {"gate_q_passed": True}, "evaluation": {"passed": False},
                "passed": False}

    rc = drv.cmd_gate_r_plus_qualification(_args(config_path), gate_runner=failed)
    assert rc == 3
    state = drv.load_state(run_dir)["state"]
    assert state == "INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_STAGE3A_PLUS"
    report = json.loads((run_dir / "qualification" /
                         "gate_r_plus_qualification_report.json").read_text(encoding="utf-8"))
    assert report["outcome"] == state


def test_gate_qplus_failure_is_invalid_run(tmp_path):
    run_dir, config_path, _, _ = _run_through_inference(tmp_path)

    def q_failed(**kwargs):
        return {"report": {"gate_q_passed": False}, "evaluation": {"passed": False},
                "passed": False}

    rc = drv.cmd_gate_r_plus_qualification(_args(config_path), gate_runner=q_failed)
    assert rc == 2
    assert drv.load_state(run_dir)["state"] == "INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE"


def test_gate_refuses_modified_gate_params_without_burning_lock(tmp_path):
    run_dir, config_path, _, _ = _qual_run_setup(tmp_path, ece_guard=0.99)
    assert drv.cmd_lock_qualification(_args(config_path)) == 0
    assert drv.cmd_qualification_infer(_args(config_path), infer_fn=_fake_infer) == 0
    # The freeze receipt is internally consistent but its Gate constants no
    # longer match the code's frozen Gate definition.
    assert drv.cmd_gate_r_plus_qualification(
        _args(config_path), gate_runner=_gate_runner_passed) == 3
    assert not (run_dir / "qualification" / "qualification_used.lock").exists()
    assert not (run_dir / "qualification" /
                "gate_r_plus_qualification_report.json").exists()
    assert drv.load_state(run_dir)["state"] == "FROZEN_INFERENCE_COMPLETE"


def test_gate_refuses_replaced_data_without_burning_lock(tmp_path):
    run_dir, config_path, data_path, _ = _run_through_inference(tmp_path)
    _write_fake_npz(data_path, np.arange(30_000, 30_000 + N_EPISODES),
                    [0] * N_EPISODES, [1] * N_EPISODES)
    assert drv.cmd_gate_r_plus_qualification(
        _args(config_path), gate_runner=_gate_runner_passed) == 3
    assert not (run_dir / "qualification" / "qualification_used.lock").exists()
    assert drv.load_state(run_dir)["state"] == "FROZEN_INFERENCE_COMPLETE"


def test_gate_refuses_replaced_prediction_file(tmp_path):
    run_dir, config_path, _, _ = _run_through_inference(tmp_path)
    victim = run_dir / "qualification" / "predictions" / f"seed_{STAGE3A_PLUS_ALL_SEEDS[0]}.npz"
    np.savez(victim, r_hat=np.zeros((1, 1, 1)), regime_logits=np.zeros((1, 1, 1, 4)))
    assert drv.cmd_gate_r_plus_qualification(
        _args(config_path), gate_runner=_gate_runner_passed) == 3
    assert not (run_dir / "qualification" / "qualification_used.lock").exists()
