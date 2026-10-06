"""Stage 4C-R2 independent-confirmation runner tests (plan sections 8, 9 Task 4, 10).

Blocking invariants proved here (Scheme B — key invariants only):
- command dispatch: every §10 command routes to its stage function; unknown stages never run;
- run-all executes the frozen §10 order (preflight -> tests -> smoke -> readiness ->
  lock -> collection -> inference/recheck -> freeze -> adjudication -> handoff);
- ``verify`` is read-only and NEVER touches the sealed confirmation archive (proved by
  running it against a fully fabricated artifact chain in which the sealed file simply
  does not exist, plus a sha256 spy and poisoned stage/truth-loader functions);
- resume discipline: an adjudicated terminal state makes run-all verify-only (no stage
  function, no adjudicator runs; exit code = the terminal state's code); a completed
  collection is hash-verified and skipped, never re-collected;
- adjudicate-confirm internally orders burn -> assemble -> adjudicate, refuses a second
  adjudication, and records the terminal state into HANDOFF_STATE.md on every outcome;
- distinct non-zero exit codes per §8 terminal state; engineering failures halt the
  pipeline with exit 1 and later stages never run.

All stage functions are mocked where the real ones would do collection/inference/
adjudication; no real science runs here. Test names deliberately avoid the substring
"truth" (pytest tmp dirs derive from test names and the config guard rejects legal-input
paths containing it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import collect_stage4cr2_merge_confirm as collector  # noqa: E402
import run_stage4cr2_confirm as runner  # noqa: E402
from ac_twin import stage4cr2_confirm_inference as cinf  # noqa: E402
from ac_twin import stage4cr2_confirm_protocol as prot  # noqa: E402
from ac_twin import stage4cr2_confirm_statistics as stat  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CONFIG_PATH = prot.CONFIG_PATH


# --- fixtures ------------------------------------------------------------------------------------


def _config(tmp_path):
    """The frozen confirm config with every stage path redirected into tmp."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run" / "stage4cr2_confirm"
    data_root = (tmp_path / "data" / "object_centric_stage"
                 / "stage4cr2_merge_multiaction_confirm")
    config["run_dir"] = str(run_dir)
    config["data_root"] = str(data_root)
    paths = dict(config["paths"])
    for key, rel in (
        ("protocol_dir", "protocol"),
        ("protocol_lock", "protocol/protocol_lock.json"),
        ("source_inventory", "protocol/source_inventory.json"),
        ("code_inventory", "protocol/code_inventory.json"),
        ("historical_seed_inventory", "protocol/historical_seed_inventory.json"),
        ("seed_audit", "protocol/seed_audit.json"),
        ("candidate_seed_schedule", "protocol/candidate_seed_schedule.npy"),
        ("collection_receipt", "collection/collection_receipt.json"),
        ("inference_predictions", "inference/predictions.npz"),
        ("inference_risk_scores", "inference/risk_scores.npz"),
        ("inference_receipt", "inference/inference_receipt.json"),
        ("determinism_receipt", "inference/determinism_receipt.json"),
        ("score_freeze_receipt", "inference/score_freeze_receipt.json"),
        ("confirmation_used_lock", "confirmation_used.lock"),
        ("confirm_group_quantities", "reports/confirm_group_quantities.npz"),
        ("confirm_group_quantities_receipt",
         "reports/confirm_group_quantities_receipt.json"),
        ("confirmatory_report", "reports/stage4cr2_confirmatory_report.json"),
        ("confirmatory_decision", "reports/STAGE4CR2_CONFIRMATORY_DECISION.md"),
    ):
        paths[key] = str(run_dir / rel)
    paths["inference_inputs"] = str(data_root / "inference_inputs.npz")
    paths["confirmation_truth"] = str(data_root / "confirmation_truth.npz")
    paths["rejection_ledger"] = str(data_root / "rejection_ledger.jsonl")
    config["paths"] = paths
    config_path = tmp_path / "stage4cr2_confirm_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config, config_path


def _write_bytes(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path)


def _write_json(path: Path, payload) -> str:
    return _write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def _fabricate_chain(config, *, status=prot.STATUS_CONFIRMED, sealed_file=False):
    """Fabricate a complete, hash-consistent artifact chain (no real science).

    The sealed archive is NOT created unless ``sealed_file=True``: the verify path must
    pass without ever touching it.
    """
    checked = prot.validate_config(config)
    paths = stat.confirm_artifact_paths(checked)
    inputs_sha = _write_bytes(paths["inputs"], b"legal-inputs-fixture")
    sealed_sha = ("ab" * 32)  # recorded-only hash; the sealed file is absent by default
    if sealed_file:
        sealed_sha = _write_bytes(paths["truth"], b"sealed-fixture")
    pred_sha = _write_bytes(paths["predictions"], b"predictions-fixture")
    score_sha = _write_bytes(paths["risk_scores"], b"risk-scores-fixture")
    coll_sha = _write_json(paths["collection_receipt"], {
        "plan_id": prot.PLAN_ID, "role": "confirm", "status": stat.STATUS_COLLECTED,
        "partial": False, "accepted_groups": prot.CONFIRM_ACCEPTED_GROUPS,
        "inference_inputs_sha256": inputs_sha,
        "confirmation_truth_sha256": sealed_sha,
        "truth_fields": list(stat.TRUTH_FIELDS),
        "group_order_binding": {"same_order": True},
    })
    inf_sha = _write_json(paths["inference_receipt"], {
        "plan_id": prot.PLAN_ID, "role": "confirm", "status": cinf.STATUS_INFERENCE,
        "data_sha256": inputs_sha, "predictions_sha256": pred_sha,
        "risk_scores_sha256": score_sha,
    })
    det_sha = _write_json(paths["determinism_receipt"], {
        "plan_id": prot.PLAN_ID, "bitwise_identical": True, "n_groups_rechecked": 32,
        "bindings": {
            "predictions_sha256": pred_sha, "risk_scores_sha256": score_sha,
            "inference_receipt_sha256": inf_sha, "inference_inputs_sha256": inputs_sha,
            "collection_receipt_sha256": coll_sha,
        },
    })
    lock_sha = _write_json(paths["protocol_lock"], {
        "plan_id": prot.PLAN_ID,
        "config_signature": canonical_fingerprint(json.loads(json.dumps(dict(checked)))),
        "code_sha256": prot.protocol_code_hashes(),
    })
    freeze_bindings = {
        "inference_inputs_sha256": inputs_sha, "predictions_sha256": pred_sha,
        "risk_scores_sha256": score_sha, "inference_receipt_sha256": inf_sha,
        "determinism_receipt_sha256": det_sha, "collection_receipt_sha256": coll_sha,
        "expected_confirmation_truth_sha256": sealed_sha,
        "protocol_lock_sha256": lock_sha,
    }
    freeze_sha = _write_json(paths["score_freeze_receipt"],
                             {"plan_id": prot.PLAN_ID, "bindings": freeze_bindings})
    used_sha = _write_json(paths["lock"], {
        "plan_id": prot.PLAN_ID,
        "bindings": {**freeze_bindings, "score_freeze_receipt_sha256": freeze_sha,
                     "statistics_code_sha256": "00" * 32},
    })
    npz_sha = _write_bytes(paths["quantities_npz"], b"quantities-fixture")
    receipt_sha = _write_json(paths["quantities_receipt"], {
        "plan_id": prot.PLAN_ID, "n_groups": prot.CONFIRM_ACCEPTED_GROUPS,
        "confirm_group_quantities_sha256": npz_sha, "lock_sha256": used_sha,
        "bootstrap_entropy": prot.BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": prot.BOOTSTRAP_RESAMPLES,
        "confidence": prot.CONFIDENCE, "primary_baseline": stat.PRIMARY_BASELINE,
        "bindings": {"confirmation_truth_sha256": sealed_sha},
    })
    _write_json(paths["report"], {
        "plan_id": prot.PLAN_ID, "status": status,
        "passed": status == prot.STATUS_CONFIRMED,
        "n_groups": prot.CONFIRM_ACCEPTED_GROUPS,
        "bindings": {"confirm_group_quantities_sha256": npz_sha,
                     "confirm_group_quantities_receipt_sha256": receipt_sha,
                     "lock_sha256": used_sha},
    })
    return paths


def _fake_all_stages(monkeypatch, calls):
    """Replace every pipeline stage with a recorder; the fake adjudication writes a
    minimal terminal report so run-all can read the terminal state back."""
    for name in runner.PIPELINE:
        def fake(config, _name=name):
            calls.append(_name)
            if _name == "adjudicate-confirm":
                checked = prot.validate_config(config)
                report = stat.confirm_artifact_paths(checked)["report"]
                _write_json(report, {"plan_id": prot.PLAN_ID,
                                     "status": prot.STATUS_CONFIRMED, "passed": True})
            return {"stage": _name, "action": "executed"}
        monkeypatch.setattr(runner, "stage_" + name.replace("-", "_"), fake)


# --- command dispatch ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", (
    "preflight", "smoke", "lock-protocol", "collect-confirm", "infer-confirm",
    "freeze-scores", "adjudicate-confirm",
))
def test_dispatch_routes_to_stage_function(tmp_path, monkeypatch, command):
    _, config_path = _config(tmp_path)
    seen = []
    monkeypatch.setattr(runner, "stage_" + command.replace("-", "_"),
                        lambda config: seen.append(command) or {"stage": command})
    assert runner.main([command, "--config", str(config_path)]) == 0
    assert seen == [command]


def test_dispatch_verify_exit_code_reflects_pass_fail(tmp_path, monkeypatch):
    _, config_path = _config(tmp_path)
    monkeypatch.setattr(runner, "cmd_verify",
                        lambda config: {"stage": "verify", "all_verified": True})
    assert runner.main(["verify", "--config", str(config_path)]) == 0
    monkeypatch.setattr(runner, "cmd_verify",
                        lambda config: {"stage": "verify", "all_verified": False})
    assert runner.main(["verify", "--config", str(config_path)]) == 1


def test_dispatch_run_all_uses_run_all(tmp_path, monkeypatch):
    _, config_path = _config(tmp_path)
    seen = []
    monkeypatch.setattr(runner, "run_all",
                        lambda path: seen.append(str(path)) or runner.RunSummary(
                            config_path=str(path), exit_code=0))
    assert runner.main(["run-all", "--config", str(config_path)]) == 0
    assert seen == [str(config_path)]


# --- run-all order and terminal short-circuit ----------------------------------------------------


def test_run_all_executes_stages_in_frozen_order(tmp_path, monkeypatch):
    _, config_path = _config(tmp_path)
    calls = []
    _fake_all_stages(monkeypatch, calls)
    summary = runner.run_all(config_path)
    assert calls == list(runner.PIPELINE)
    assert summary.exit_code == 0
    assert summary.terminal_state == prot.STATUS_CONFIRMED


@pytest.mark.parametrize("status,code", (
    (prot.STATUS_CONFIRMED, 0),
    (prot.STATUS_NOT_CONFIRMED, 8),
    (prot.STATUS_TASK_DEGENERATE, 6),
    (prot.STATUS_MODEL_INSUFFICIENT, 7),
))
def test_run_all_adjudicated_terminal_only_verifies(tmp_path, monkeypatch, status, code):
    config, config_path = _config(tmp_path)
    _fabricate_chain(config, status=status)  # sealed archive deliberately absent
    for name in runner.PIPELINE:
        monkeypatch.setattr(
            runner, "stage_" + name.replace("-", "_"),
            lambda config, _name=name: pytest.fail(f"stage {_name} must not run"))
    monkeypatch.setattr(stat, "adjudicate_confirmation",
                        lambda *a, **k: pytest.fail("re-adjudication is forbidden"))
    summary = runner.run_all(config_path)
    assert summary.terminal_state == status
    assert summary.exit_code == code


def test_run_all_adjudicated_but_drifted_is_invalid(tmp_path, monkeypatch):
    config, config_path = _config(tmp_path)
    paths = _fabricate_chain(config, status=prot.STATUS_CONFIRMED)
    paths["predictions"].write_bytes(b"tampered")  # hash drift after adjudication
    _fake_all_stages(monkeypatch, [])  # must never run anyway
    summary = runner.run_all(config_path)
    assert summary.terminal_state == prot.STATUS_INVALID
    assert summary.exit_code == runner.EXIT_CODES[prot.STATUS_INVALID]


# --- verify: read-only, never touches the sealed archive -----------------------------------------


def test_verify_read_only_and_never_opens_sealed_archive(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    paths = _fabricate_chain(config)  # sealed file NOT created
    assert not paths["truth"].is_file()

    hashed = []
    real_sha256 = sha256_file
    monkeypatch.setattr(runner, "sha256_file",
                        lambda p: hashed.append(str(p)) or real_sha256(p))
    monkeypatch.setattr(collector, "collect_confirmation",
                        lambda *a, **k: pytest.fail("verify must not collect"))
    monkeypatch.setattr(prot, "preflight",
                        lambda *a, **k: pytest.fail("verify must not preflight"))
    monkeypatch.setattr(prot, "verify_protocol",
                        lambda *a, **k: pytest.fail("verify must not rescan history"))
    for name in ("load_confirmation_truth", "assemble_confirm_quantities",
                 "adjudicate_confirmation", "burn_confirmation_lock", "freeze_scores"):
        monkeypatch.setattr(stat, name,
                            lambda *a, **k: pytest.fail(f"verify must not call {name}"))

    summary = runner.cmd_verify(config)
    assert summary["all_verified"] is True
    assert summary["terminal_state"] == prot.STATUS_CONFIRMED
    assert all("confirmation_truth" not in str(p) for p in hashed)
    assert not paths["truth"].is_file()  # verify never creates/touches it either


def test_verify_fails_closed_on_missing_artifacts(tmp_path):
    config, config_path = _config(tmp_path)
    summary = runner.cmd_verify(config)
    assert summary["all_verified"] is False
    assert runner.main(["verify", "--config", str(config_path)]) == 1


def test_verify_detects_score_drift(tmp_path):
    config, _ = _config(tmp_path)
    paths = _fabricate_chain(config)
    paths["risk_scores"].write_bytes(b"drifted")
    summary = runner.cmd_verify(config)
    assert summary["all_verified"] is False
    failed = {c["check"] for c in summary["checks"] if not c["ok"]}
    assert "inference_receipt" in failed


# --- resume: completed collection skipped, never re-collected -------------------------------------


def test_collect_confirm_skips_complete_collection(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    # completed collection: receipt + both published files, hashes consistent
    checked = prot.validate_config(config)
    paths = stat.confirm_artifact_paths(checked)
    inputs_sha = _write_bytes(paths["inputs"], b"legal-inputs-fixture")
    sealed_sha = _write_bytes(paths["truth"], b"sealed-fixture")
    _write_json(paths["collection_receipt"], {
        "plan_id": prot.PLAN_ID, "role": "confirm", "status": stat.STATUS_COLLECTED,
        "partial": False, "accepted_groups": prot.CONFIRM_ACCEPTED_GROUPS,
        "inference_inputs_sha256": inputs_sha,
        "confirmation_truth_sha256": sealed_sha,
    })
    monkeypatch.setattr(collector, "collect_confirmation",
                        lambda *a, **k: pytest.fail("must not re-collect"))
    outcome = runner.stage_collect_confirm(config)
    assert outcome["action"] == "skipped_complete"


def test_collect_confirm_drift_is_invalid(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    checked = prot.validate_config(config)
    paths = stat.confirm_artifact_paths(checked)
    _write_bytes(paths["inputs"], b"legal-inputs-fixture")
    sealed_sha = _write_bytes(paths["truth"], b"sealed-fixture")
    _write_json(paths["collection_receipt"], {
        "plan_id": prot.PLAN_ID, "role": "confirm", "status": stat.STATUS_COLLECTED,
        "partial": False, "accepted_groups": prot.CONFIRM_ACCEPTED_GROUPS,
        "inference_inputs_sha256": "00" * 32,  # drifted binding
        "confirmation_truth_sha256": sealed_sha,
    })
    monkeypatch.setattr(collector, "collect_confirmation",
                        lambda *a, **k: pytest.fail("drift must halt, not re-collect"))
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.stage_collect_confirm(config)
    assert excinfo.value.status == prot.STATUS_INVALID


# --- adjudicate-confirm: internal burn/unseal transaction ------------------------------------------


def test_adjudicate_confirm_orders_burn_assemble_adjudicate(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    order = []
    monkeypatch.setattr(stat, "burn_confirmation_lock",
                        lambda c: order.append("burn") or "lock")
    monkeypatch.setattr(stat, "assemble_confirm_quantities",
                        lambda c: order.append("assemble") or "npz")
    monkeypatch.setattr(stat, "adjudicate_confirmation",
                        lambda c: order.append("adjudicate") or {
                            "status": prot.STATUS_CONFIRMED, "passed": True})
    outcome = runner.stage_adjudicate_confirm(config)
    assert order == ["burn", "assemble", "adjudicate"]
    assert outcome["status"] == prot.STATUS_CONFIRMED
    handoff = Path(config["run_dir"]) / "HANDOFF_STATE.md"
    assert handoff.is_file()
    assert prot.STATUS_CONFIRMED in handoff.read_text(encoding="utf-8")


def test_adjudicate_confirm_failure_writes_terminal_state(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    monkeypatch.setattr(stat, "burn_confirmation_lock", lambda c: "lock")
    monkeypatch.setattr(stat, "assemble_confirm_quantities", lambda c: "npz")
    monkeypatch.setattr(stat, "adjudicate_confirmation", lambda c: {
        "status": prot.STATUS_NOT_CONFIRMED, "passed": False})
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.stage_adjudicate_confirm(config)
    assert excinfo.value.status == prot.STATUS_NOT_CONFIRMED
    handoff = Path(config["run_dir"]) / "HANDOFF_STATE.md"
    assert prot.STATUS_NOT_CONFIRMED in handoff.read_text(encoding="utf-8")


def test_adjudicate_confirm_refuses_second_adjudication(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    _fabricate_chain(config, status=prot.STATUS_CONFIRMED)
    monkeypatch.setattr(stat, "adjudicate_confirmation",
                        lambda *a, **k: pytest.fail("re-adjudication is forbidden"))
    monkeypatch.setattr(stat, "burn_confirmation_lock",
                        lambda *a, **k: pytest.fail("must not re-burn"))
    outcome = runner.stage_adjudicate_confirm(config)
    assert outcome["action"] == "verified_terminal"
    assert outcome["status"] == prot.STATUS_CONFIRMED


def test_adjudicate_confirm_existing_failure_is_terminal(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    _fabricate_chain(config, status=prot.STATUS_NOT_CONFIRMED)
    monkeypatch.setattr(stat, "adjudicate_confirmation",
                        lambda *a, **k: pytest.fail("re-adjudication is forbidden"))
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.stage_adjudicate_confirm(config)
    assert excinfo.value.status == prot.STATUS_NOT_CONFIRMED


# --- exit codes ------------------------------------------------------------------------------------


@pytest.mark.parametrize("status,code", tuple(
    (s, c) for s, c in runner.EXIT_CODES.items() if s != prot.STATUS_CONFIRMED))
def test_terminal_states_map_to_distinct_exit_codes(tmp_path, monkeypatch, status, code):
    _, config_path = _config(tmp_path)
    calls = []
    _fake_all_stages(monkeypatch, calls)

    def stop(config, _status=status):
        raise runner.TerminalState(_status, "fixture halt")

    monkeypatch.setattr(runner, "stage_smoke", stop)
    summary = runner.run_all(config_path)
    assert summary.exit_code == code
    assert code != 0
    assert summary.terminal_state == status
    assert calls == ["preflight", "tests"]  # the pipeline halted at smoke


def test_preflight_error_maps_to_its_status(tmp_path, monkeypatch):
    _, config_path = _config(tmp_path)
    _fake_all_stages(monkeypatch, [])

    def bad_preflight(config):
        raise prot.PreflightError("seed overlap", status=prot.STATUS_INVALID)

    monkeypatch.setattr(runner, "stage_preflight", bad_preflight)
    summary = runner.run_all(config_path)
    assert summary.exit_code == runner.EXIT_CODES[prot.STATUS_INVALID]


def test_collection_incomplete_maps_to_exit_4(tmp_path, monkeypatch):
    _, config_path = _config(tmp_path)
    _fake_all_stages(monkeypatch, [])

    def exhausted(config):
        raise collector.CollectionIncompleteError("candidate domain exhausted")

    monkeypatch.setattr(runner, "stage_collect_confirm", exhausted)
    summary = runner.run_all(config_path)
    assert summary.exit_code == runner.EXIT_CODES[prot.STATUS_COLLECTION_INCOMPLETE] == 4


def test_resource_at_risk_maps_to_exit_5(tmp_path, monkeypatch):
    _, config_path = _config(tmp_path)
    _fake_all_stages(monkeypatch, [])

    def unprovable(config):
        raise stat.ResourceAtRiskError("post-burn drift")

    monkeypatch.setattr(runner, "stage_adjudicate_confirm", unprovable)
    summary = runner.run_all(config_path)
    assert summary.exit_code == runner.EXIT_CODES[prot.STATUS_RESOURCE_AT_RISK] == 5


def test_blocking_test_failure_halts_pipeline_with_engineering_code(tmp_path, monkeypatch,
                                                                     capsys):
    _, config_path = _config(tmp_path)
    calls = []
    real_tests = runner.stage_tests
    _fake_all_stages(monkeypatch, calls)
    monkeypatch.setattr(runner, "stage_tests", real_tests)  # real stage, mocked pytest
    monkeypatch.setattr(runner, "_run_pytest",
                        lambda label, args: (1, ["tests/test_x.py::test_y"], "FAILED"))
    summary = runner.run_all(config_path)
    assert summary.exit_code == 1
    assert calls == ["preflight"]  # halted inside the tests stage; smoke never ran
    assert "FAILURE" in capsys.readouterr().out


# --- pytest subprocess output parsing ---------------------------------------------------------------


def _fake_proc(stdout: str, stderr: str = "", returncode: int = 1):
    class _Proc:
        pass
    proc = _Proc()
    proc.stdout, proc.stderr, proc.returncode = stdout, stderr, returncode
    return proc


def test_run_pytest_parses_ansi_colored_failures(monkeypatch):
    """Regression: real pytest output carries ANSI escapes (``\\x1b[31mFAILED\\x1b[0m``);
    node extraction must still work, and the subprocess must be asked for plain output."""
    colored = (
        "\x1b[31mFAILED\x1b[0m tests/test_stage2_protocol.py::test_lock_a - AssertionError\n"
        "\x1b[31mFAILED\x1b[0m tests/test_stage2b_protocol.py::test_stale_b\n"
        "\x1b[31m\x1b[1m2 failed\x1b[0m, \x1b[32m10 passed\x1b[0m in 5.01s\n"
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _fake_proc(colored)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    rc, failed, tail = runner._run_pytest("tests", ["tests"])
    assert rc == 1
    assert failed == ["tests/test_stage2_protocol.py::test_lock_a",
                      "tests/test_stage2b_protocol.py::test_stale_b"]
    assert "--color=no" in captured["cmd"]
    assert "\x1b" not in tail  # printed tail is plain text


def test_run_pytest_parses_plain_failures(monkeypatch):
    """Plain output keeps parsing as before (no regression for uncolored runs)."""
    plain = "FAILED tests/test_x.py::test_y - RuntimeError\n1 failed in 0.10s\n"
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda cmd, **kwargs: _fake_proc(plain))
    rc, failed, _ = runner._run_pytest("tests", ["tests"])
    assert rc == 1
    assert failed == ["tests/test_x.py::test_y"]


# --- readiness exemptions ---------------------------------------------------------------------------


def test_readiness_requires_checklist(tmp_path):
    config, _ = _config(tmp_path)
    with pytest.raises(RuntimeError, match="READINESS_CHECKLIST"):
        runner.stage_readiness(config)


def test_readiness_full_regression_exemptions(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "READINESS_CHECKLIST.md").write_text("# checklist\n", encoding="utf-8")
    failing = ["tests/test_stage2_protocol.py::test_lock_a",
               "tests/test_stage2b_protocol.py::test_stale_b"]
    monkeypatch.setattr(runner, "_run_pytest",
                        lambda label, args: (1, list(failing), "FAILED summary"))
    # no exemptions registered -> engineering failure
    with pytest.raises(RuntimeError, match="exemption"):
        runner.stage_readiness(config)
    # register exactly these nodes with evidence -> pass
    (run_dir / "readiness_exemptions.json").write_text(json.dumps({
        "exempt_nodes": failing, "evidence": "registered historical failures",
        "registered_by": "controller", "registered_at": "2026-09-20",
    }), encoding="utf-8")
    outcome = runner.stage_readiness(config)
    assert outcome["exempted_nodes"] == sorted(failing)
    # an unregistered failure is never exempted
    monkeypatch.setattr(runner, "_run_pytest",
                        lambda label, args: (1, ["tests/test_new.py::test_new_failure"],
                                             "FAILED summary"))
    with pytest.raises(RuntimeError, match="exemption"):
        runner.stage_readiness(config)


# --- handoff ---------------------------------------------------------------------------------------


def test_handoff_prints_doc_sync_targets(tmp_path, capsys):
    config, _ = _config(tmp_path)
    _fabricate_chain(config, status=prot.STATUS_CONFIRMED)
    outcome = runner.stage_handoff(config)
    printed = capsys.readouterr().out
    for target in runner.DOC_SYNC_TARGETS:
        assert target in printed
        assert target in outcome["doc_sync_targets"]
    handoff = Path(config["run_dir"]) / "HANDOFF_STATE.md"
    assert prot.STATUS_CONFIRMED in handoff.read_text(encoding="utf-8")


def test_handoff_refused_before_adjudication(tmp_path):
    config, _ = _config(tmp_path)
    with pytest.raises(RuntimeError, match="report"):
        runner.stage_handoff(config)


# --- smoke: collection -> inference -> recheck -> engineering statistics (Task 2 supplement) ----


def _fake_smoke_stages(monkeypatch, calls):
    monkeypatch.setattr(collector, "collect_confirmation",
                        lambda c, role: calls.append(("collect", role)) or {
                            "accepted_groups": prot.SMOKE_ACCEPTED_GROUPS,
                            "partial": False})
    monkeypatch.setattr(cinf, "infer_and_score",
                        lambda c, role: calls.append(("infer", role)) or {
                            "status": cinf.STATUS_SMOKE, "groups": prot.SMOKE_ACCEPTED_GROUPS})
    monkeypatch.setattr(stat, "run_smoke_recheck",
                        lambda c: calls.append("recheck") or {
                            "bitwise_identical": True,
                            "n_groups_rechecked": prot.SMOKE_ACCEPTED_GROUPS})
    monkeypatch.setattr(stat, "run_smoke_engineering_statistics",
                        lambda c: calls.append("engineering") or {
                            "status": cinf.STATUS_SMOKE, "report_path": "smoke/report.json"})


def test_smoke_orders_collection_inference_recheck_engineering(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    calls = []
    _fake_smoke_stages(monkeypatch, calls)
    # the smoke stage NEVER burns or creates any lock.
    monkeypatch.setattr(prot, "create_protocol_lock",
                        lambda *a, **k: pytest.fail("smoke must not lock the protocol"))
    monkeypatch.setattr(stat, "burn_confirmation_lock",
                        lambda *a, **k: pytest.fail("smoke must not burn the lock"))
    outcome = runner.stage_smoke(config)
    assert calls == [("collect", "smoke"), ("infer", "smoke"), "recheck", "engineering"]
    assert outcome["accepted_groups"] == prot.SMOKE_ACCEPTED_GROUPS
    assert outcome["action"] == "executed"
    formal = stat.confirm_artifact_paths(prot.validate_config(config))
    assert not formal["protocol_lock"].exists()
    assert not formal["lock"].exists()


def test_smoke_recheck_failure_is_terminal_and_stops_before_statistics(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    calls = []
    _fake_smoke_stages(monkeypatch, calls)
    monkeypatch.setattr(stat, "run_smoke_recheck",
                        lambda c: calls.append("recheck") or {"bitwise_identical": False})
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.stage_smoke(config)
    assert excinfo.value.status == prot.STATUS_INVALID
    assert calls == [("collect", "smoke"), ("infer", "smoke"), "recheck"]  # no engineering


def test_smoke_engineering_statistics_must_keep_the_smoke_label(tmp_path, monkeypatch):
    config, _ = _config(tmp_path)
    calls = []
    _fake_smoke_stages(monkeypatch, calls)
    monkeypatch.setattr(stat, "run_smoke_engineering_statistics",
                        lambda c: calls.append("engineering") or {"status": "WRONG"})
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.stage_smoke(config)
    assert excinfo.value.status == prot.STATUS_INVALID
