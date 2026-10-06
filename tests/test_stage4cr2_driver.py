"""Stage 4C-R2 unified runner tests (plan sections 6.2, 13, 14).

Blocking invariants proved here (task-5 brief / plan section 14):
- stage order is frozen: the screen is refused before the validation gate has passed and
  the final selection is frozen;
- completed hash-matching stages are SKIPPED on resume (no re-collection, no
  re-inference, no re-training, no re-adjudication, no power recompute);
- one-shot adjudication stages (validation, screen) may NOT be skipped-then-rerun: an
  existing create-once outcome is verified, never re-executed;
- every scientific hard stop returns its dedicated non-zero exit code and halts the
  pipeline (later stages never run);
- the confirmation power diagnostic never runs unless the screen passed, and an
  UNDERPOWERED outcome is terminal with its own exit code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import collect_stage4cr2_merge_multiaction as collector  # noqa: E402
import drive_stage4cr2_task3 as task3  # noqa: E402
import drive_stage4cr2_task4 as task4  # noqa: E402
import run_stage4cr2 as runner  # noqa: E402
from ac_twin import stage4cr2_inference as s4cr2i  # noqa: E402
from ac_twin import stage4cr2_power as s4cr2pw  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"


def _tmp_config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "stage4c_r2")
    config["data_root"] = str(tmp_path / "stage4cr2_merge_multiaction_dev")
    path = tmp_path / "stage4cr2_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return config, path


def _write_npz(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:  # file handle: numpy does not append ".npz"
        np.savez_compressed(handle, x=np.arange(4))
    return sha256_file(path)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return Path(path)


# --- frozen stage order -------------------------------------------------------------------------


def test_run_all_executes_stages_in_frozen_order(tmp_path, monkeypatch):
    _, config_path = _tmp_config(tmp_path)
    calls = []
    for name in runner.PIPELINE:
        monkeypatch.setattr(
            runner, "stage_" + name.replace("-", "_"),
            lambda config, _name=name: calls.append(_name) or {"stage": _name})
    summary = runner.run_all(config_path)
    assert calls == list(runner.PIPELINE)
    assert summary.exit_code == 0
    assert summary.terminal_state == s4cr2.STATUS_DEV_READY


def test_collect_screen_refused_before_validation_pass(tmp_path):
    config, _ = _tmp_config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    dev.mkdir(parents=True)
    _write_json(dev / "protocol_lock.json", {"plan_id": s4cr2.PLAN_ID})
    with pytest.raises(ValueError, match="final_selection"):
        runner.stage_collect_screen(config)
    with pytest.raises(ValueError, match="final_selection"):
        runner.stage_infer_screen(config)
    with pytest.raises(ValueError, match="final_selection"):
        runner.stage_adjudicate_screen(config)


# --- resume: completed stages skip, executors never run ------------------------------------------


def _fabricate_completed_pipeline(config, *, screen_terminal=s4cr2.STATUS_DEV_READY,
                                  power_status=s4cr2.STATUS_DEV_READY):
    run_dir = Path(config["run_dir"])
    dev = run_dir / "development"
    smoke = run_dir / "smoke"
    data_root = Path(config["data_root"])
    _write_json(dev / "protocol_lock.json", {"plan_id": s4cr2.PLAN_ID})

    # smoke chain (engineering only)
    smoke_data = _write_npz(data_root / "smoke" / "object_twinworld_stage4cr2_smoke.npz")
    _write_json(smoke / "smoke_collection_receipt.json", {
        "accepted_groups": 12, "partial": False, "status": "ENGINEERING_SMOKE_ONLY",
        "data_output": str(data_root / "smoke" / "object_twinworld_stage4cr2_smoke.npz"),
        "data_output_sha256": smoke_data})
    _write_json(smoke / "smoke_inference_receipt.json", {
        "predictions_sha256": _write_npz(smoke / "smoke_predictions.npz"),
        "scores_sha256": _write_npz(smoke / "smoke_scores.npz")})
    _write_json(smoke / "smoke_determinism_receipt.json", {"bitwise_identical": True})

    for prefix, quota in (("train", 5000), ("validation", 1500), ("screen", 8000)):
        data_path = data_root / f"object_twinworld_stage4cr2_{prefix}.npz"
        _write_json(dev / f"{prefix}_collection_receipt.json", {
            "accepted_groups": quota, "partial": False, "status": "COLLECTED",
            "data_output": str(data_path),
            "data_output_sha256": _write_npz(data_path)})
        if prefix != "screen":
            _write_json(dev / f"{prefix}_inference_receipt.json", {
                "predictions_sha256": _write_npz(dev / f"{prefix}_predictions.npz"),
                "scores_sha256": _write_npz(dev / f"{prefix}_scores.npz")})

    _write_json(dev / "train_support_report_r1.json", {"status": "TRAIN_SUPPORT_OK"})

    checkpoints = {}
    for seed in s4cr2.MODEL_SEEDS:
        ckpt = dev / "models" / f"direct_seed{seed}.pt"
        checkpoints[str(seed)] = {"path": str(ckpt), "sha256": _write_npz(ckpt)}
    _write_json(dev / "model_inventory.json", {
        "checkpoint_sha256": {k: v["sha256"] for k, v in checkpoints.items()},
        "checkpoint_paths": {k: v["path"] for k, v in checkpoints.items()}})
    _write_json(dev / "models" / "fit_receipt.json", {"status": "FIVE_SEEDS_FIT"})

    _write_json(dev / "validation_gate_report.json", {
        "passed": True, "status": "VALIDATION_QUALIFIED"})
    _write_json(dev / "final_selection.json", {
        "status": "VALIDATION_QUALIFIED", "checkpoints": checkpoints})

    _write_json(dev / "development_screen_report.json", {
        "plan_id": s4cr2.PLAN_ID, "terminal_state": screen_terminal,
        "passed": screen_terminal == s4cr2.STATUS_DEV_READY})
    _write_json(dev / "confirmation_power_diagnostic.json", {
        "plan_id": s4cr2.PLAN_ID, "status": power_status,
        "selected_n": 3000 if power_status == s4cr2.STATUS_DEV_READY else None,
        "underpowered": power_status == s4cr2.STATUS_UNDERPOWERED})


def _forbid_executors(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a completed stage executor was called during resume")
    monkeypatch.setattr(collector, "collect_role", boom)
    monkeypatch.setattr(s4cr2i, "infer_role", boom)
    monkeypatch.setattr(s4cr2i, "verify_repeated_inference", boom)
    monkeypatch.setattr(task3, "cmd_train", boom)
    monkeypatch.setattr(task3, "cmd_adjudicate_validation", boom)
    for name in ("cmd_collect_screen", "cmd_recheck_screen", "cmd_infer_screen",
                 "cmd_freeze_screen_scores", "cmd_burn_lock", "cmd_adjudicate_screen"):
        monkeypatch.setattr(task4, name, boom)
    monkeypatch.setattr(s4cr2pw, "run_confirmation_power_diagnostic", boom)
    monkeypatch.setattr(s4cr2, "verify_development_lock", lambda path: None)


def test_resume_skips_every_completed_stage(tmp_path, monkeypatch):
    config, config_path = _tmp_config(tmp_path)
    _fabricate_completed_pipeline(config)
    _forbid_executors(monkeypatch)
    summary = runner.run_all(config_path)
    assert summary.exit_code == 0
    assert summary.terminal_state == s4cr2.STATUS_DEV_READY
    assert len(summary.stages) == len(runner.PIPELINE)
    for stage in summary.stages:
        assert stage["action"] in ("executed", "skipped_complete", "verified_terminal"), stage
    by_name = {s["stage"]: s for s in summary.stages}
    assert by_name["collect-train"]["action"] == "skipped_complete"
    assert by_name["train-five-seeds"]["action"] == "skipped_complete"
    assert by_name["adjudicate-validation"]["action"] == "verified_terminal"
    assert by_name["adjudicate-screen"]["action"] == "verified_terminal"
    assert by_name["power-diagnostic"]["action"] == "skipped_complete"


def test_resume_after_underpowered_is_sticky_terminal(tmp_path, monkeypatch):
    config, config_path = _tmp_config(tmp_path)
    _fabricate_completed_pipeline(config, power_status=s4cr2.STATUS_UNDERPOWERED)
    _forbid_executors(monkeypatch)
    summary = runner.run_all(config_path)
    assert summary.exit_code == runner.EXIT_CODES[s4cr2.STATUS_UNDERPOWERED]
    assert summary.exit_code != 0
    assert summary.terminal_state == s4cr2.STATUS_UNDERPOWERED


# --- one-shot adjudications: never skipped-then-rerun ---------------------------------------------


def test_validation_adjudication_verified_never_rerun(tmp_path, monkeypatch):
    config, _ = _tmp_config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    _write_json(dev / "final_selection.json", {"status": "VALIDATION_QUALIFIED"})
    monkeypatch.setattr(task3, "cmd_adjudicate_validation",
                        lambda *a, **k: pytest.fail("one-shot validation re-adjudicated"))
    outcome = runner.stage_adjudicate_validation(config)
    assert outcome["action"] == "verified_terminal"


def test_failed_validation_is_terminal_never_rerun(tmp_path, monkeypatch):
    config, _ = _tmp_config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    _write_json(dev / "validation_gate_report.json", {
        "passed": False, "status": s4cr2.STATUS_MODEL_DEV_INSUFFICIENT})
    monkeypatch.setattr(task3, "cmd_adjudicate_validation",
                        lambda *a, **k: pytest.fail("failed validation re-adjudicated"))
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.stage_adjudicate_validation(config)
    assert excinfo.value.status == s4cr2.STATUS_MODEL_DEV_INSUFFICIENT


def test_screen_adjudication_verified_never_rerun(tmp_path, monkeypatch):
    config, _ = _tmp_config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    _write_json(dev / "final_selection.json", {"status": "VALIDATION_QUALIFIED"})
    _write_json(dev / "development_screen_report.json", {
        "terminal_state": s4cr2.STATUS_DEV_READY, "passed": True})
    monkeypatch.setattr(task4, "cmd_adjudicate_screen",
                        lambda *a, **k: pytest.fail("one-shot screen re-adjudicated"))
    outcome = runner.stage_adjudicate_screen(config)
    assert outcome["action"] == "verified_terminal"
    assert outcome["terminal_state"] == s4cr2.STATUS_DEV_READY


def test_failed_screen_is_terminal_never_rerun(tmp_path, monkeypatch):
    config, _ = _tmp_config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    _write_json(dev / "final_selection.json", {"status": "VALIDATION_QUALIFIED"})
    _write_json(dev / "development_screen_report.json", {
        "terminal_state": s4cr2.STATUS_LINE_CLOSED, "passed": False})
    monkeypatch.setattr(task4, "cmd_adjudicate_screen",
                        lambda *a, **k: pytest.fail("failed screen re-adjudicated"))
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.stage_adjudicate_screen(config)
    assert excinfo.value.status == s4cr2.STATUS_LINE_CLOSED


# --- dedicated exit codes per scientific hard stop --------------------------------------------------


@pytest.mark.parametrize("failed_stage,status", [
    ("check-train-support", s4cr2.STATUS_TRAIN_SUPPORT_INSUFFICIENT),
    ("adjudicate-validation", s4cr2.STATUS_MODEL_DEV_INSUFFICIENT),
    ("adjudicate-validation", s4cr2.STATUS_STRATUM_IMBALANCED),
    ("adjudicate-screen", s4cr2.STATUS_LINE_CLOSED),
    ("adjudicate-screen", s4cr2.STATUS_INVALID),
    ("adjudicate-screen", s4cr2.STATUS_TASK_DEGENERATE),
    ("adjudicate-screen", s4cr2.STATUS_MODEL_INSUFFICIENT),
    ("power-diagnostic", s4cr2.STATUS_UNDERPOWERED),
])
def test_terminal_states_carry_dedicated_exit_codes(
        tmp_path, monkeypatch, failed_stage, status):
    _, config_path = _tmp_config(tmp_path)
    calls = []
    for name in runner.PIPELINE:
        def _stub(config, _name=name):
            calls.append(_name)
            if _name == failed_stage:
                raise runner.TerminalState(status)
            return {"stage": _name}
        monkeypatch.setattr(runner, "stage_" + name.replace("-", "_"), _stub)
    summary = runner.run_all(config_path)
    assert summary.exit_code == runner.EXIT_CODES[status]
    assert summary.exit_code != 0
    assert summary.terminal_state == status
    # the pipeline HALTS at the hard stop: no later stage runs.
    assert calls == list(runner.PIPELINE)[: runner.PIPELINE.index(failed_stage) + 1]
    assert runner.EXIT_CODES[status] not in (0, 1, 2)
    assert len(set(runner.EXIT_CODES.values()) - {0}) == len(runner.EXIT_CODES) - 1


# --- verify ----------------------------------------------------------------------------------------


def test_verify_reports_missing_resources_on_empty_run_dir(tmp_path):
    config, _ = _tmp_config(tmp_path)
    summary = runner.cmd_verify(config)
    assert summary["all_verified"] is False
    assert summary["development_lock"] == "missing"
    assert summary["screen_report"] == "missing"
