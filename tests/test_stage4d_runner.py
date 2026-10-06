"""Blocking tests for the Stage 4D D0/D1 unified runner skeleton (plan §12, §11 item 10).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 3. Pins the frozen
run-all phase order (§12 verbatim up to ``stop``), the rule that unimplemented Task 4/5/6
phases raise "not implemented yet" instead of being silently skipped, the
behavior -> geometry -> joint pilot collection order, and the create-once resume guard
on the D0 profile selection.
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

import run_stage4d_ood as runner  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402


def test_run_all_phase_order_matches_plan_section_12():
    assert tuple(runner.PIPELINE) == (
        "preflight",
        "focused-tests",
        "engineering-smoke",
        "full-regression-readiness",
        "collect-pilots",
        "evaluate-pilots",
        "select-profile",
        "power",
        "lock-d1",
        "locked-smoke-readiness",
        "collect-d1",
        "inference-scoring",
        "score-freeze",
        "burn-confirmation-lock",
        "adjudicate",
        "reports-status-sync",
    )


def test_all_pipeline_phases_implemented_never_skipped():
    # Task 4 implemented power / lock-d1 / locked-smoke-readiness, Task 5 implemented
    # collect-d1 / inference-scoring / score-freeze and Task 6 implemented
    # burn-confirmation-lock / adjudicate / reports-status-sync: EVERY §12 phase is now
    # registered (none left unimplemented, none silently skipped).
    for phase in runner.PIPELINE:
        assert phase in runner._PHASE_FUNCTIONS, f"phase {phase!r} has no implementation"
        assert callable(runner._PHASE_FUNCTIONS[phase])


def test_pilot_collection_order_is_behavior_geometry_joint():
    assert tuple(runner.PILOT_COLLECTION_ORDER) == (
        "pilot_behavior", "pilot_geometry", "pilot_joint")


def test_collect_pilots_phase_calls_collector_in_order(monkeypatch):
    calls = []

    def fake_collect_role(config, role, **kwargs):
        calls.append(role)
        return {"role": role, "status": "COLLECTED", "accepted_groups": 300,
                "attempts": 400, "partial": False}

    monkeypatch.setattr(runner.collector, "collect_role", fake_collect_role)
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    runner.run_phase("collect-pilots", config={"run_dir": "x", "data_root": "y"})
    assert calls == ["pilot_behavior", "pilot_geometry", "pilot_joint"]


def test_collect_pilot_cli_profile_binding(monkeypatch):
    calls = []

    def fake_collect_role(config, role, **kwargs):
        calls.append(role)
        return {"role": role, "profile": "JOINT_OOD", "status": "COLLECTED",
                "accepted_groups": 300, "attempts": 400, "partial": False}

    monkeypatch.setattr(runner.collector, "collect_role", fake_collect_role)
    monkeypatch.setattr(runner, "_load_config", lambda path: {"_": path})
    code = runner.main(["collect-pilot", "--profile", "joint",
                        "--config", "dummy.json"])
    assert code == 0
    assert calls == ["pilot_joint"]


def test_collect_pilot_cli_rejects_unknown_profile():
    with pytest.raises(SystemExit):
        runner.main(["collect-pilot", "--profile", "bogus", "--config", "x"])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_select_profile_phase_create_once(tmp_path, monkeypatch):
    run_dir = tmp_path / "run" / "stage4d_ood"
    for short, qualified in (("behavior", True), ("geometry", True), ("joint", False)):
        _write_json(
            run_dir / "pilot" / short / "profile_report.json",
            {"profile": {"behavior": "BEHAVIOR_OOD", "geometry": "GEOMETRY_OOD",
                         "joint": "JOINT_OOD"}[short],
             "qualified": qualified},
        )
    config = {
        "run_dir": str(run_dir),
        "data_root": str(tmp_path / "data"),
        "paths": {"ood_profile_selection":
                  str(run_dir / "pilot" / "ood_profile_selection.json")},
    }
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    out = runner.run_phase("select-profile", config=config)
    assert out["selected_profile"] == "BEHAVIOR_OOD"
    selection_path = run_dir / "pilot" / "ood_profile_selection.json"
    written = json.loads(selection_path.read_text(encoding="utf-8"))
    assert written["selected_profile"] == "BEHAVIOR_OOD"
    # create-once resume: a second run verifies and never rewrites.
    again = runner.run_phase("select-profile", config=config)
    assert again["verification"] == "existing_selection_verified_no_overwrite"


def test_select_profile_phase_hard_stop_when_none_qualified(tmp_path, monkeypatch):
    run_dir = tmp_path / "run" / "stage4d_ood"
    for short in ("behavior", "geometry", "joint"):
        _write_json(run_dir / "pilot" / short / "profile_report.json",
                    {"profile": short.upper(), "qualified": False})
    config = {
        "run_dir": str(run_dir),
        "data_root": str(tmp_path / "data"),
        "paths": {"ood_profile_selection":
                  str(run_dir / "pilot" / "ood_profile_selection.json")},
    }
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.run_phase("select-profile", config=config)
    assert excinfo.value.status == prot.STATUS_TASK_INSUFFICIENT
    # the hard stop still writes the create-once selection record honestly.
    written = json.loads(
        (run_dir / "pilot" / "ood_profile_selection.json").read_text(encoding="utf-8"))
    assert written["selected_profile"] is None
    assert written["status"] == prot.STATUS_TASK_INSUFFICIENT
    # the recorded hard stop is absorbing: resume re-raises it, never proceeds past it.
    with pytest.raises(runner.TerminalState) as excinfo2:
        runner.run_phase("select-profile", config=config)
    assert excinfo2.value.status == prot.STATUS_TASK_INSUFFICIENT
