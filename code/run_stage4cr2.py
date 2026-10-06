"""Stage 4C-R2 unified orchestration runner.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 6.2 and 13.
This runner ONLY orchestrates the frozen stage functions (Task 3 / Task 4 drivers, the
collector, the inference layer, the power module); it never re-implements their science
and never re-runs an already-burned one-shot stage.

Subcommands (plan section 13):
    preflight, lock-development, smoke, collect-train, infer-train, check-train-support,
    train-five-seeds, collect-validation, infer-validation, adjudicate-validation,
    freeze-final-selection, collect-screen, infer-screen, adjudicate-screen,
    power-diagnostic, verify, run-all

``run-all`` executes continuously once the plan is approved; completed hash-matching
resumable stages are SKIPPED (collection resumes from atomic checkpoints; create-once
receipts/locks are verified, never overwritten); one-shot adjudications are verified,
never re-executed. Every scientific hard stop prints its terminal state and returns a
dedicated non-zero exit code; long stages print phase/completed/total/%/elapsed/ETA at
least every 30 s (delegated to the stage drivers); a final success/failure summary is
always printed.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage4cr2_merge_multiaction as collector  # noqa: E402
import drive_stage4cr2_task3 as task3  # noqa: E402
import drive_stage4cr2_task4 as task4  # noqa: E402
from ac_twin import stage4cr2_inference as s4cr2i  # noqa: E402
from ac_twin import stage4cr2_power as s4cr2pw  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_statistics as s4cr2st  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

ROOT = PROJECT_ROOT
CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"

# Frozen stage order (plan section 6.2 collapsed onto the plan-13 subcommand list; the
# screen score-freeze / determinism recheck / lock burn run INSIDE infer-screen).
PIPELINE = (
    "preflight",
    "lock-development",
    "smoke",
    "collect-train",
    "infer-train",
    "check-train-support",
    "train-five-seeds",
    "collect-validation",
    "infer-validation",
    "adjudicate-validation",
    "freeze-final-selection",
    "collect-screen",
    "infer-screen",
    "adjudicate-screen",
    "power-diagnostic",
)

# Dedicated non-zero exit code per scientific terminal state (plan section 11).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CODES = {
    s4cr2.STATUS_DEV_READY: EXIT_OK,
    s4cr2.STATUS_TRAIN_SUPPORT_INSUFFICIENT: 3,
    s4cr2.STATUS_MODEL_DEV_INSUFFICIENT: 4,
    s4cr2.STATUS_LINE_CLOSED: 5,
    s4cr2.STATUS_INVALID: 6,
    s4cr2.STATUS_TASK_DEGENERATE: 7,
    s4cr2.STATUS_MODEL_INSUFFICIENT: 8,
    s4cr2.STATUS_STRATUM_IMBALANCED: 9,
    s4cr2.STATUS_UNDERPOWERED: 10,
}


class TerminalState(RuntimeError):
    """A scientific hard stop: carries the plan-11 terminal state; the pipeline halts."""

    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}" if detail else status)
        self.status = str(status)
        self.detail = detail


@dataclass
class RunSummary:
    """run-all outcome: per-stage actions, the terminal state and the exit code."""
    config_path: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    terminal_state: str = ""
    exit_code: int = EXIT_OK
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "stages": self.stages,
            "terminal_state": self.terminal_state,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# --- helpers --------------------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return s4cr2.validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))


def _dev_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["run_dir"]) / "development"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _log(stage: str, message: str) -> None:
    print(f"[stage4cr2:{stage}] {message}", flush=True)


def _hash_matches(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_file(path) == expected


def _collection_complete(config: Mapping[str, Any], role: str) -> bool:
    """True when the role's create-once receipt exists, is full-quota and hash-matches."""
    data_output, _, receipt_path = collector.role_paths(config, role)
    if not receipt_path.is_file():
        return False
    receipt = _read_json(receipt_path)
    if receipt.get("partial") or int(receipt.get("accepted_groups", -1)) \
            != collector.ROLE_QUOTA[role]:
        return False
    recorded = Path(receipt.get("data_output", data_output))
    return _hash_matches(recorded, str(receipt.get("data_output_sha256", "")))


def _inference_complete(config: Mapping[str, Any], role: str) -> bool:
    pred_path, score_path, receipt_path = s4cr2i.role_inference_paths(config, role)
    if not receipt_path.is_file():
        return False
    receipt = _read_json(receipt_path)
    return (_hash_matches(pred_path, str(receipt.get("predictions_sha256", "")))
            and _hash_matches(score_path, str(receipt.get("scores_sha256", ""))))


def _screen_report(config: Mapping[str, Any]) -> dict[str, Any] | None:
    path = s4cr2st.screen_artifact_paths(config)["report"]
    return _read_json(path) if path.is_file() else None


def _final_selection(config: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _dev_dir(config) / "final_selection.json"
    return _read_json(path) if path.is_file() else None


def _require_validation_qualified(config: Mapping[str, Any]) -> None:
    """The screen exists ONLY because validation passed and the selection is frozen."""
    if not (_dev_dir(config) / "protocol_lock.json").is_file():
        raise ValueError("the development protocol lock is missing; run lock-development")
    final = _final_selection(config)
    if final is None:
        raise ValueError(
            "the screen is refused: final_selection.json is missing — the validation "
            "gate must pass and freeze before the screen is touched")
    if final.get("status") != "VALIDATION_QUALIFIED":
        raise ValueError(
            f"the screen is refused: validation status {final.get('status')!r}; the "
            "multi-action decision-risk line is closed")


# --- stages ----------------------------------------------------------------------------------------


def stage_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    checked = s4cr2.validate_config(config)
    inventory = s4cr2.build_lock_inventory()  # re-hashes every plan-2.1 frozen source
    _log("preflight", f"config valid; {len(inventory)} frozen sources hash-match")
    return {"stage": "preflight", "action": "executed",
            "n_frozen_sources": len(inventory)}


def stage_lock_development(config: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = task3.cmd_lock(config)  # create-once; verifies an existing lock instead
    return {"stage": "lock-development", "action": "executed",
            "lock": lock_path.name}


def stage_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _resolve(config["run_dir"])
    smoke_dir = run_dir / "smoke"
    determinism_path = smoke_dir / "smoke_determinism_receipt.json"
    if _collection_complete(config, "smoke") and _inference_complete(config, "smoke") \
            and determinism_path.is_file():
        determinism = _read_json(determinism_path)
        if not determinism.get("bitwise_identical"):
            raise ValueError("the recorded smoke determinism recheck failed")
        _log("smoke", "existing smoke receipts verified, no rerun")
        return {"stage": "smoke", "action": "skipped_complete"}
    receipt = collector.collect_role(config, "smoke", collector.ROLE_QUOTA["smoke"])
    if receipt.partial or receipt.accepted_groups != s4cr2.SMOKE_GROUPS:
        raise TerminalState(s4cr2.STATUS_INVALID, "smoke collection incomplete")
    s4cr2i.infer_role(config, "smoke")
    if not determinism_path.is_file():
        determinism = s4cr2i.verify_repeated_inference(
            "smoke", np.arange(s4cr2.SMOKE_GROUPS, dtype=np.int64), config=config)
        write_once_json(determinism_path, determinism.to_dict())
    _log("smoke", "ENGINEERING_SMOKE_ONLY complete (12 groups, never a science role)")
    return {"stage": "smoke", "action": "executed"}


def _stage_collect(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    name = f"collect-{collector.ROLE_PREFIX[role]}"
    if _collection_complete(config, role):
        _log(name, f"{role} receipt hash-verified, no re-collection")
        return {"stage": name, "action": "skipped_complete"}
    task3.cmd_collect(config, role)  # resumes from atomic checkpoints; never overwrites
    return {"stage": name, "action": "executed"}


def _stage_infer(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    name = f"infer-{collector.ROLE_PREFIX[role]}"
    if _inference_complete(config, role):
        _log(name, f"{role} inference receipt hash-verified, no rerun")
        return {"stage": name, "action": "skipped_complete"}
    task3.cmd_infer(config, role)
    return {"stage": name, "action": "executed"}


def stage_collect_train(config: Mapping[str, Any]) -> dict[str, Any]:
    return _stage_collect(config, "development_train")


def stage_infer_train(config: Mapping[str, Any]) -> dict[str, Any]:
    return _stage_infer(config, "development_train")


def stage_check_train_support(config: Mapping[str, Any]) -> dict[str, Any]:
    report = task3._latest_train_support_report(_dev_dir(config))
    action = "skipped_complete"
    if report is None:
        report = task3.cmd_check_train_support(config)
        action = "executed"
    status = str(report.get("status", ""))
    _log("check-train-support", f"status={status}")
    if status != "TRAIN_SUPPORT_OK":
        raise TerminalState(
            s4cr2.STATUS_TRAIN_SUPPORT_INSUFFICIENT,
            f"train support gate failed ({status}); terminal, no training")
    return {"stage": "check-train-support", "action": action, "status": status}


def stage_train_five_seeds(config: Mapping[str, Any]) -> dict[str, Any]:
    dev = _dev_dir(config)
    inventory_path = dev / "model_inventory.json"
    fit_receipt = dev / "models" / "fit_receipt.json"
    if inventory_path.is_file() and fit_receipt.is_file():
        inventory = _read_json(inventory_path)
        for seed_text, digest in inventory["checkpoint_sha256"].items():
            path = Path(inventory["checkpoint_paths"][seed_text])
            if not _hash_matches(path, str(digest)):
                raise ValueError(f"frozen Direct checkpoint drifted: {path}")
        _log("train-five-seeds", "five checkpoints hash-verified, no retraining")
        return {"stage": "train-five-seeds", "action": "skipped_complete"}
    task3.cmd_train(config)
    return {"stage": "train-five-seeds", "action": "executed"}


def stage_collect_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    return _stage_collect(config, "development_validation")


def stage_infer_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    return _stage_infer(config, "development_validation")


def stage_adjudicate_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    """ONE-SHOT: an existing outcome is verified, NEVER re-adjudicated (plan 14.11)."""
    final = _final_selection(config)
    if final is not None:
        if final.get("status") != "VALIDATION_QUALIFIED":
            raise ValueError(
                f"final_selection.json status is {final.get('status')!r}, not "
                "VALIDATION_QUALIFIED")
        _log("adjudicate-validation", "frozen final selection verified, no rerun")
        return {"stage": "adjudicate-validation", "action": "verified_terminal",
                "status": "VALIDATION_QUALIFIED"}
    gate_report = _dev_dir(config) / "validation_gate_report.json"
    if gate_report.is_file():
        report = _read_json(gate_report)
        status = str(report.get("status", s4cr2.STATUS_MODEL_DEV_INSUFFICIENT))
        raise TerminalState(
            status, "the one-shot validation adjudication already ran and failed; "
            "the multi-action decision-risk line is permanently closed")
    report = task3.cmd_adjudicate_validation(config)  # freezes on pass; records on fail
    if not report.get("passed"):
        raise TerminalState(str(report.get("status", s4cr2.STATUS_MODEL_DEV_INSUFFICIENT)))
    return {"stage": "adjudicate-validation", "action": "executed",
            "status": "VALIDATION_QUALIFIED"}


def stage_freeze_final_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    """The freeze happens inside the ONE validation adjudication; this stage verifies."""
    final = _final_selection(config)
    if final is None or final.get("status") != "VALIDATION_QUALIFIED":
        raise ValueError(
            "final_selection.json missing or not VALIDATION_QUALIFIED; run "
            "adjudicate-validation (a second freeze is refused by create-once)")
    _log("freeze-final-selection", "final selection frozen and verified")
    return {"stage": "freeze-final-selection", "action": "skipped_complete",
            "status": "VALIDATION_QUALIFIED"}


def stage_collect_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    if _screen_report(config) is not None:
        _log("collect-screen", "screen already adjudicated; collection chain complete")
        return {"stage": "collect-screen", "action": "skipped_complete"}
    _require_validation_qualified(config)
    if _collection_complete(config, "development_screen"):
        _log("collect-screen", "screen receipt hash-verified, no re-collection")
        return {"stage": "collect-screen", "action": "skipped_complete"}
    task4.cmd_collect_screen(config)  # resumes from atomic checkpoints
    return {"stage": "collect-screen", "action": "executed"}


def stage_infer_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    """recheck -> frozen inference -> Direct score freeze -> burn the one-shot lock."""
    if _screen_report(config) is not None:
        _log("infer-screen", "screen already adjudicated; score chain complete")
        return {"stage": "infer-screen", "action": "skipped_complete"}
    _require_validation_qualified(config)
    task4.cmd_recheck_screen(config)       # 32-group bitwise repeated inference
    task4.cmd_infer_screen(config)         # create-once predictions/scores + receipt
    task4.cmd_freeze_screen_scores(config)  # five frozen Direct checkpoints, create-once
    task4.cmd_burn_lock(config)            # development_screen_used.lock, create-once
    return {"stage": "infer-screen", "action": "executed"}


def stage_adjudicate_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    """ONE-SHOT: an existing report is verified, NEVER re-adjudicated (plan 14.12)."""
    report = _screen_report(config)
    if report is not None:
        terminal = str(report.get("terminal_state", ""))
        _log("adjudicate-screen", f"existing one-shot report verified: {terminal}")
        if terminal != s4cr2.STATUS_DEV_READY:
            raise TerminalState(
                terminal, "the one-shot screen already ran and did not pass; any "
                "rerun or re-adjudication is permanently refused")
        return {"stage": "adjudicate-screen", "action": "verified_terminal",
                "terminal_state": terminal}
    _require_validation_qualified(config)
    report = task4.cmd_adjudicate_screen(config)
    terminal = str(report.get("terminal_state", ""))
    if terminal != s4cr2.STATUS_DEV_READY:
        raise TerminalState(terminal)
    return {"stage": "adjudicate-screen", "action": "executed",
            "terminal_state": terminal}


def _power_diagnostic_with_heartbeat(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the plan-10 diagnostic while printing a <=30 s plain-text heartbeat."""
    result: dict[str, Any] = {}
    started = time.perf_counter()

    def work() -> None:
        try:
            result["outcome"] = s4cr2pw.run_confirmation_power_diagnostic(
                config, progress=lambda line: _log("power-diagnostic", line))
        except BaseException as exc:  # propagate, never swallow
            result["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=25.0)
        if thread.is_alive():
            _log("power-diagnostic",
                 f"bootstrap/simulation running elapsed={time.perf_counter() - started:.0f}s")
    if "error" in result:
        raise result["error"]
    return result["outcome"]


def stage_power_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan section 10: conditional power ONLY after a passed screen; never confirm."""
    screen = _screen_report(config)
    if screen is None:
        raise ValueError("power-diagnostic requires the one-shot screen report; run "
                         "adjudicate-screen first")
    if screen.get("terminal_state") != s4cr2.STATUS_DEV_READY:
        raise TerminalState(str(screen.get("terminal_state")),
                            "the screen did not pass; no power diagnostic is legal")
    out_path = s4cr2pw.diagnostic_path(config)
    if out_path.is_file():
        diagnostic = _read_json(out_path)
        _log("power-diagnostic", f"existing diagnostic verified: {diagnostic['status']}")
        if diagnostic.get("status") == s4cr2.STATUS_UNDERPOWERED:
            raise TerminalState(
                s4cr2.STATUS_UNDERPOWERED,
                "n=5,000 at 75% shrink below 0.80 joint power; no confirm plan may be "
                "created (sticky terminal on resume)")
        return {"stage": "power-diagnostic", "action": "skipped_complete",
                "selected_n": diagnostic.get("selected_n")}
    outcome = _power_diagnostic_with_heartbeat(config)
    _log("power-diagnostic",
         f"status={outcome['status']} selected_n={outcome['selected_n']}")
    if outcome["status"] == s4cr2.STATUS_UNDERPOWERED:
        raise TerminalState(
            s4cr2.STATUS_UNDERPOWERED,
            "n=5,000 at 75% shrink below 0.80 joint power; no confirm lock/collection "
            "may be created under this plan")
    return {"stage": "power-diagnostic", "action": "executed",
            "selected_n": outcome["selected_n"]}


# --- verify ----------------------------------------------------------------------------------------


def cmd_verify(config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-check the frozen bindings and report every resource state (read-only)."""
    checked = s4cr2.validate_config(config)
    dev = _dev_dir(checked)
    summary: dict[str, Any] = {"stage": "verify"}
    lock_path = dev / "protocol_lock.json"
    if lock_path.is_file():
        s4cr2.verify_development_lock(lock_path)
        summary["development_lock"] = "verified"
    else:
        summary["development_lock"] = "missing"
    final = _final_selection(checked)
    summary["final_selection"] = (final or {}).get("status", "missing")
    screen = _screen_report(checked)
    summary["screen_report"] = (screen or {}).get("terminal_state", "missing")
    diagnostic_file = s4cr2pw.diagnostic_path(checked)
    if diagnostic_file.is_file():
        diagnostic = _read_json(diagnostic_file)
        summary["power_diagnostic"] = {
            "status": diagnostic.get("status"),
            "selected_n": diagnostic.get("selected_n"),
            "underpowered": diagnostic.get("underpowered"),
        }
    else:
        summary["power_diagnostic"] = "missing"
    summary["all_verified"] = bool(
        summary["development_lock"] == "verified"
        and summary["final_selection"] == "VALIDATION_QUALIFIED"
        and summary["screen_report"] == s4cr2.STATUS_DEV_READY
        and isinstance(summary["power_diagnostic"], dict))
    return summary


# --- run-all ---------------------------------------------------------------------------------------


def _stage_function(name: str):
    return globals()["stage_" + name.replace("-", "_")]  # resolved late (test-friendly)


def run_all(config_path: str | Path = CONFIG_PATH) -> RunSummary:
    """Execute the full pipeline continuously; skip completed stages; halt on hard stops."""
    summary = RunSummary(config_path=str(config_path),
                         started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    config = _load_config(config_path)
    started = time.perf_counter()
    try:
        for name in PIPELINE:
            _log(name, f"starting ({PIPELINE.index(name) + 1}/{len(PIPELINE)})")
            outcome = _stage_function(name)(config)
            outcome = {"stage": name, **outcome}
            summary.stages.append(outcome)
            _log(name, f"{outcome.get('action', 'done')}")
        summary.terminal_state = s4cr2.STATUS_DEV_READY
        summary.exit_code = EXIT_OK
    except TerminalState as stop:
        summary.terminal_state = stop.status
        summary.exit_code = EXIT_CODES.get(stop.status, EXIT_ERROR)
        _log("run-all", f"TERMINAL STATE: {stop.status} "
                        f"(exit {summary.exit_code}) {stop.detail}")
    summary.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    elapsed = time.perf_counter() - started
    verdict = "SUCCESS" if summary.exit_code == EXIT_OK else "HALTED"
    _log("run-all", f"{verdict} summary: terminal_state={summary.terminal_state} "
                    f"exit_code={summary.exit_code} stages={len(summary.stages)}/"
                    f"{len(PIPELINE)} elapsed={elapsed:.1f}s")
    return summary


# --- CLI -------------------------------------------------------------------------------------------


_COMMANDS = tuple(PIPELINE) + ("verify", "run-all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4C-R2 unified runner")
    parser.add_argument("command", choices=_COMMANDS)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)
    config = _load_config(args.config)

    if args.command == "run-all":
        return run_all(args.config).exit_code
    if args.command == "verify":
        summary = cmd_verify(config)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
        return EXIT_OK if summary["all_verified"] else EXIT_ERROR
    try:
        outcome = _stage_function(args.command)(config)
    except TerminalState as stop:
        print(json.dumps({"terminal_state": stop.status, "detail": stop.detail},
                         indent=2, ensure_ascii=False), flush=True)
        return EXIT_CODES.get(stop.status, EXIT_ERROR)
    print(json.dumps(outcome, indent=2, ensure_ascii=False, default=str), flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
