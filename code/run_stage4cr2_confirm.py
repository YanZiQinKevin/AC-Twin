"""Stage 4C-R2 independent-confirmation staged runner (small entry; plan section 10).

Plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`` sections 6.4, 8, 9 (Task 4)
and 10. This runner ONLY orchestrates the frozen stage functions of
``stage4cr2_confirm_protocol`` / ``collect_stage4cr2_merge_confirm`` /
``stage4cr2_confirm_inference`` / ``stage4cr2_confirm_statistics`` — it never
re-implements their science and never re-runs an already-completed create-once
stage (completed stages are hash/state-verified and skipped, never overwritten).

Subcommands (plan section 10, verbatim):
    preflight, smoke, lock-protocol, collect-confirm, infer-confirm,
    freeze-scores, adjudicate-confirm, verify, run-all

``run-all`` executes in the frozen section-10 order:
    preflight -> software tests -> smoke -> readiness/full regression -> lock ->
    collection -> inference/recheck -> score freeze -> one-shot adjudication ->
    handoff.
``infer-confirm`` internally performs the fixed leading-32-group repeated
inference/scoring recheck; ``adjudicate-confirm`` internally performs the
create-once burn + unseal transaction (burn -> assemble -> adjudicate). There is
NO bare truth-reading command. Once a valid terminal state exists, ``run-all``
only verifies and exits with the terminal state's code — it never re-adjudicates,
and success never auto-starts anything beyond this stage.

``verify`` is strictly read-only: hash + status cross-checks over receipts and
score-side artifacts. It NEVER opens — never even hashes —
``confirmation_truth.npz``; the sealed archive's integrity is cross-checked only
through receipt-recorded hashes (collection receipt vs score freeze receipt vs
burned lock vs quantities receipt).

Exit codes: 0 success; 1 engineering/verification failure; each section-8
terminal state maps to a distinct non-zero code (see ``EXIT_CODES``). Long stages
print plain-text progress at least every 30 s: collection self-reports
stage/done/total/percent/elapsed/ETA via ``ac_twin.progress.ProgressReporter``;
inference, the determinism recheck, the bootstrap adjudication and the pytest
regressions run under a <=25 s heartbeat wrapper (stage + elapsed; done/total is
unknown for opaque single-call stages). Every run ends with an unambiguous
SUCCESS/FAILURE line.

Scientific failure stops per section 8: all artifacts are preserved, the terminal
state is written into HANDOFF_STATE.md (paper boundary itself is written by the
statistics layer's decision document), and the doc-sync targets needing
manual/controller updates are printed — this runner never edits them itself.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Mapping  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage4cr2_merge_confirm as collector  # noqa: E402
from ac_twin import stage4cr2_confirm_inference as cinf  # noqa: E402
from ac_twin import stage4cr2_confirm_protocol as prot  # noqa: E402
from ac_twin import stage4cr2_confirm_statistics as stat  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.progress import format_duration  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

ROOT = PROJECT_ROOT
CONFIG_PATH = prot.CONFIG_PATH

#: frozen run-all order (plan section 10, verbatim).
PIPELINE = (
    "preflight",
    "tests",
    "smoke",
    "readiness",
    "lock-protocol",
    "collect-confirm",
    "infer-confirm",
    "freeze-scores",
    "adjudicate-confirm",
    "handoff",
)

#: this stage's blocking test files (plan section 9; never exempted).
STAGE_TEST_FILES = (
    "tests/test_stage4cr2_confirm_protocol.py",
    "tests/test_stage4cr2_confirm_inference.py",
    "tests/test_stage4cr2_confirm_collection.py",
    "tests/test_stage4cr2_confirm_statistics.py",
    "tests/test_stage4cr2_confirm_runner.py",
)

READINESS_CHECKLIST_NAME = "READINESS_CHECKLIST.md"
HANDOFF_STATE_NAME = "HANDOFF_STATE.md"
#: operator-registered non-blocking historical failures (plan section 10: by
#: specific node with verified evidence only; this stage's own blocking tests and
#: any NEW failure are never exempted).
READINESS_EXEMPTIONS_NAME = "readiness_exemptions.json"

#: documents that need manual/controller sync once the terminal state is known
#: (plan Task 4: AGENTS / 00_READ_FIRST / master plan / 01 status index / research
#: log). The runner prints this list at handoff; it never edits these files.
DOC_SYNC_TARGETS = (
    "AGENTS.md",
    "00_READ_FIRST_AC_TWIN.md",
    "01_AC_TWIN_DOCUMENT_STATUS_INDEX.md",
    "plans/2026-09-06_AC-Twin反应式事实证据价值与路由研究计划_v1.0.md (master plan)",
    "logs/ 当日科研日志 (research log)",
)

# Dedicated non-zero exit code per section-8 terminal state (0 = success).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CODES = {
    prot.STATUS_CONFIRMED: EXIT_OK,
    prot.STATUS_INVALID: 3,
    prot.STATUS_COLLECTION_INCOMPLETE: 4,   # matches the collector's own CLI
    prot.STATUS_RESOURCE_AT_RISK: 5,
    prot.STATUS_TASK_DEGENERATE: 6,
    prot.STATUS_MODEL_INSUFFICIENT: 7,
    prot.STATUS_NOT_CONFIRMED: 8,
}

_HEARTBEAT_SECONDS = 25.0  # under the 30 s contract


class TerminalState(RuntimeError):
    """A scientific hard stop: carries the section-8 terminal state; the pipeline halts."""

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
    return prot.validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _log(stage: str, message: str) -> None:
    print(f"[stage4cr2-confirm:{stage}] {message}", flush=True)


def _with_heartbeat(label: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a worker thread; print a plain-text heartbeat every <=25 s.

    Opaque single-call stages (inference, determinism recheck, bootstrap
    adjudication, pytest regressions) have no done/total channel, so the
    heartbeat carries stage + elapsed (ETA unknown -> ``--:--``); collection
    self-reports the full done/total/percent/elapsed/ETA line via
    ``ac_twin.progress.ProgressReporter``.
    """
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # propagate, never swallow
            result["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    started = time.perf_counter()
    thread.start()
    while thread.is_alive():
        thread.join(timeout=_HEARTBEAT_SECONDS)
        if thread.is_alive():
            _log(label, "running | elapsed="
                 f"{format_duration(time.perf_counter() - started)} | ETA=--:--")
    if "error" in result:
        raise result["error"]
    return result.get("value")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_FAILED_RE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)


def _strip_ansi(text: str) -> str:
    """Remove ANSI color escapes: pytest colorizes its summary when the environment
    advertises a terminal, and captured subprocess output can carry those escapes,
    which would make ``^FAILED`` never match (operational bug, fixed pre-lock)."""
    return _ANSI_RE.sub("", text)


def _failed_nodes(output: str) -> list[str]:
    """Extract FAILED node IDs from pytest output, robust to ANSI-colored text."""
    return _FAILED_RE.findall(_strip_ansi(output))


def _run_pytest(label: str, args: list[str]) -> tuple[int, list[str], str]:
    """Run pytest as a subprocess under the heartbeat; return (rc, failed nodes, tail).

    ``--color=no`` asks pytest for plain output; captured text is additionally
    ANSI-stripped before parsing (defense in depth — coloring behavior varies with
    the terminal environment).
    """
    cmd = [sys.executable, "-m", "pytest", *args, "-q", "-rf", "--color=no"]
    _log(label, f"running: {' '.join(cmd)}")
    proc = _with_heartbeat(
        label, lambda: subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True))
    output = _strip_ansi((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
    tail = "\n".join(output.strip().splitlines()[-15:]) if output.strip() else ""
    if tail:
        print(tail, flush=True)
    return proc.returncode, _failed_nodes(output), tail


def _stage_function(name: str) -> Callable:
    return globals()["stage_" + name.replace("-", "_")]  # resolved late (test-friendly)


# --- stages ----------------------------------------------------------------------------------------


def stage_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Full pre-collection preflight (config, section-2.1 sources, global seed audit)."""
    try:
        report = _with_heartbeat("preflight", lambda: prot.preflight(config))
    except prot.PreflightError:
        raise
    except ValueError as exc:
        raise TerminalState(prot.STATUS_INVALID, f"preflight: {exc}") from exc
    _log("preflight", f"{prot.STATUS_PREFLIGHT_PASSED}; confirm schedule raw sha256 "
                      f"{report['confirm_schedule_raw_sha256'][:16]}…; "
                      f"R2 executed domain {report['r2_executed_domain_size']} seeds")
    return {"stage": "preflight", "action": "executed",
            "status": prot.STATUS_PREFLIGHT_PASSED}


def stage_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    """Software tests: this stage's five blocking test files must pass, zero exemption."""
    missing = [rel for rel in STAGE_TEST_FILES if not (ROOT / rel).is_file()]
    if missing:
        raise RuntimeError(f"blocking test files missing: {missing}")
    rc, failed, _ = _run_pytest("tests", list(STAGE_TEST_FILES))
    if rc != 0:
        raise RuntimeError(
            f"stage blocking tests failed (exit {rc}); failing nodes {failed}; this "
            "stage's own tests are never exempted (plan section 10)")
    _log("tests", f"{len(STAGE_TEST_FILES)} blocking test files green")
    return {"stage": "tests", "action": "executed", "files": list(STAGE_TEST_FILES)}


def stage_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """ENGINEERING_SMOKE_ONLY: 12-group collection + frozen scoring + the smoke
    engineering recheck + the smoke engineering-truth statistics, fully isolated.

    Order: collection -> inference -> smoke recheck (bitwise repeat over ALL 12 groups
    vs the persisted smoke artifacts) -> smoke engineering-truth statistics (REAL
    quantity/gate path as an engineering diagnostic). The smoke stage never touches
    formal artifact paths, never burns any lock and never produces a scientific verdict.
    """
    receipt = collector.collect_confirmation(config, "smoke")  # self-reporting progress
    if receipt.get("partial") or int(receipt.get("accepted_groups", -1)) \
            != prot.SMOKE_ACCEPTED_GROUPS:
        raise TerminalState(prot.STATUS_INVALID, "smoke collection incomplete")
    inference = _with_heartbeat("smoke", lambda: cinf.infer_and_score(config, "smoke"))
    if inference.get("status") != cinf.STATUS_SMOKE:
        raise TerminalState(prot.STATUS_INVALID,
                            f"smoke inference status {inference.get('status')!r}")
    recheck = _with_heartbeat("smoke", lambda: stat.run_smoke_recheck(config))
    if not recheck.get("bitwise_identical"):
        raise TerminalState(prot.STATUS_INVALID,
                            "the smoke repeated inference/scoring was not bitwise "
                            "identical")
    engineering = _with_heartbeat(
        "smoke", lambda: stat.run_smoke_engineering_statistics(config))
    if engineering.get("status") != cinf.STATUS_SMOKE:
        raise TerminalState(prot.STATUS_INVALID,
                            "the smoke engineering statistics lost the "
                            "ENGINEERING_SMOKE_ONLY label")
    action = ("skipped_complete" if receipt.get("verification")
              and inference.get("verification") and recheck.get("verification")
              and engineering.get("verification") else "executed")
    _log("smoke", f"ENGINEERING_SMOKE_ONLY complete ({receipt['accepted_groups']} groups; "
                  f"recheck bitwise over {recheck['n_groups_rechecked']} groups; "
                  "engineering-truth statistics recorded; never a science role, never "
                  "touches the formal lock)")
    return {"stage": "smoke", "action": action,
            "accepted_groups": int(receipt["accepted_groups"]),
            "recheck_groups": int(recheck["n_groups_rechecked"]),
            "engineering_report": str(engineering.get("report_path", ""))}


def _load_readiness_exemptions(run_dir: Path) -> dict[str, Any]:
    path = run_dir / READINESS_EXEMPTIONS_NAME
    if not path.is_file():
        return {}
    payload = _read_json(path)
    nodes = payload.get("exempt_nodes")
    if not isinstance(nodes, list) or not all(isinstance(n, str) for n in nodes):
        raise RuntimeError(f"{READINESS_EXEMPTIONS_NAME} must list exempt_nodes strings")
    if not str(payload.get("evidence", "")):
        raise RuntimeError(f"{READINESS_EXEMPTIONS_NAME} must carry verified evidence")
    return payload


def stage_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    """Readiness / full regression before the formal lock (plan Task 2 + section 10).

    Requires the filled READINESS_CHECKLIST.md, then runs the WHOLE repo regression.
    A non-zero exit passes only when every failing node is registered in
    ``readiness_exemptions.json`` (specific nodes + verified evidence — plan
    section 10's non-blocking historical failures); new failures and this stage's
    own blocking tests are never exempted.
    """
    checked = prot.validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    checklist = run_dir / READINESS_CHECKLIST_NAME
    if not checklist.is_file():
        raise RuntimeError(
            f"READINESS_CHECKLIST.md missing: {checklist}; the lock is refused until "
            "the readiness checklist exists and is filled in the operational phase")
    rc, failed, _ = _run_pytest("readiness", ["tests"])
    if rc == 0:
        _log("readiness", "full regression green (zero failures)")
        return {"stage": "readiness", "action": "executed", "full_regression": "green"}
    exemptions = _load_readiness_exemptions(run_dir)
    exempted = set(exemptions.get("exempt_nodes", []))
    if failed and set(failed) <= exempted:
        _log("readiness", f"full regression: only registered non-blocking historical "
                          f"failures {sorted(failed)}")
        return {"stage": "readiness", "action": "executed",
                "full_regression": "green_with_registered_exemptions",
                "exempted_nodes": sorted(failed)}
    raise RuntimeError(
        f"full regression failed (exit {rc}); failing nodes {failed} are not all "
        f"registered non-blocking exemptions — register specific nodes with verified "
        f"evidence in {READINESS_EXEMPTIONS_NAME} or fix the failure; the lock is "
        "refused")


def stage_lock_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create-once formal protocol lock; an existing lock is re-verified, never remade."""
    checked = prot.validate_config(config)
    lock_path = _resolve(checked["paths"]["protocol_lock"])
    if lock_path.is_file():
        _with_heartbeat("lock-protocol", lambda: prot.verify_protocol(lock_path))
        _log("lock-protocol", "existing protocol lock re-verified, no re-lock")
        return {"stage": "lock-protocol", "action": "skipped_complete"}
    lock = _with_heartbeat("lock-protocol", lambda: prot.create_protocol_lock(config))
    _log("lock-protocol", f"protocol lock created (create-once): {lock}")
    return {"stage": "lock-protocol", "action": "executed", "lock": str(lock)}


def _confirm_collection_complete(checked: Mapping[str, Any]) -> bool:
    """Hash/state check for a completed confirm collection (section 6.4 step 4).

    Hash-only verification of BOTH published files against the create-once receipt —
    the sealed archive's bytes are hashed for integrity but never loaded as data.
    Drift is INVALID, never a silent re-collection.
    """
    paths = stat.confirm_artifact_paths(checked)
    receipt_path = paths["collection_receipt"]
    if not receipt_path.is_file():
        return False
    receipt = _read_json(receipt_path)
    if receipt.get("partial") or receipt.get("status") != stat.STATUS_COLLECTED \
            or int(receipt.get("accepted_groups", -1)) != prot.CONFIRM_ACCEPTED_GROUPS:
        return False
    for key, path in (("inference_inputs_sha256", paths["inputs"]),
                      ("confirmation_truth_sha256", paths["truth"])):
        if not path.is_file() or sha256_file(path) != str(receipt.get(key, "")):
            raise TerminalState(
                prot.STATUS_INVALID,
                f"collected data drifted from its create-once receipt: {path}")
    return True


def stage_collect_confirm(config: Mapping[str, Any]) -> dict[str, Any]:
    """The 2,000-group formal collection (resumes from atomic checkpoints)."""
    checked = prot.validate_config(config)
    if _confirm_collection_complete(checked):
        _log("collect-confirm", "collection receipt hash-verified, no re-collection")
        return {"stage": "collect-confirm", "action": "skipped_complete"}
    receipt = collector.collect_confirmation(config, "confirm")  # self-reporting
    if receipt.get("partial") or int(receipt.get("accepted_groups", -1)) \
            != prot.CONFIRM_ACCEPTED_GROUPS:
        raise TerminalState(prot.STATUS_COLLECTION_INCOMPLETE,
                            "confirm collection incomplete")
    return {"stage": "collect-confirm", "action": "executed",
            "accepted_groups": int(receipt["accepted_groups"])}


def stage_infer_confirm(config: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen inference/scoring + the fixed leading-32-group determinism recheck."""
    inference = _with_heartbeat("infer-confirm",
                                lambda: cinf.infer_and_score(config, "confirm"))
    if inference.get("status") != cinf.STATUS_INFERENCE:
        raise TerminalState(prot.STATUS_INVALID,
                            f"confirm inference status {inference.get('status')!r}")
    recheck = _with_heartbeat("infer-confirm",
                              lambda: stat.run_determinism_recheck(config))
    if not recheck.get("bitwise_identical"):
        raise TerminalState(prot.STATUS_INVALID,
                            "the fixed leading-groups repeated inference/scoring was "
                            "not bitwise identical")
    action = ("skipped_complete" if inference.get("verification")
              and recheck.get("verification") else "executed")
    _log("infer-confirm", f"groups={inference['groups']} rechecked="
                          f"{recheck['n_groups_rechecked']} bitwise_identical=True")
    return {"stage": "infer-confirm", "action": action,
            "groups": int(inference["groups"]),
            "rechecked_groups": int(recheck["n_groups_rechecked"])}


def stage_freeze_scores(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every score-side artifact (create-once receipt; rerun re-verifies)."""
    receipt = _with_heartbeat("freeze-scores", lambda: stat.freeze_scores(config))
    action = "skipped_complete" if receipt.get("verification") else "executed"
    _log("freeze-scores", f"score freeze verified (n_groups={receipt['n_groups']})")
    return {"stage": "freeze-scores", "action": action,
            "n_groups": int(receipt["n_groups"])}


def _verify_terminal_binding(paths: Mapping[str, Path]) -> None:
    """Hash-only terminal-chain check for an existing adjudication (post-burn drift
    is unprovable -> RESOURCE_AT_RISK, plan section 8)."""
    receipt = _read_json(paths["quantities_receipt"])
    report = _read_json(paths["report"])
    if not paths["quantities_npz"].is_file() or sha256_file(paths["quantities_npz"]) \
            != receipt.get("confirm_group_quantities_sha256"):
        raise TerminalState(prot.STATUS_RESOURCE_AT_RISK,
                            "the confirm quantities drifted from their create-once "
                            "receipt after the burn")
    if receipt.get("lock_sha256") != sha256_file(paths["lock"]):
        raise TerminalState(prot.STATUS_RESOURCE_AT_RISK,
                            "the quantities receipt is not bound to the burned lock")
    if report.get("bindings", {}).get("confirm_group_quantities_sha256") \
            != receipt.get("confirm_group_quantities_sha256"):
        raise TerminalState(prot.STATUS_RESOURCE_AT_RISK,
                            "the formal report is not bound to the confirm quantities")


def stage_adjudicate_confirm(config: Mapping[str, Any]) -> dict[str, Any]:
    """ONE-SHOT: burn -> assemble -> adjudicate; an existing report is verified, NEVER
    re-adjudicated. The terminal state is recorded into HANDOFF_STATE.md on every
    outcome; all artifacts are preserved on failure."""
    checked = prot.validate_config(config)
    paths = stat.confirm_artifact_paths(checked)
    if paths["report"].is_file():
        report = _read_json(paths["report"])
        status = str(report.get("status", prot.STATUS_INVALID))
        _verify_terminal_binding(paths)
        _log("adjudicate-confirm",
             f"existing one-shot adjudication verified: {status} (never re-run)")
        if status != prot.STATUS_CONFIRMED:
            raise TerminalState(
                status, "the one-shot confirmation already ran and did not confirm; "
                "any rerun or re-adjudication is permanently refused")
        return {"stage": "adjudicate-confirm", "action": "verified_terminal",
                "status": status}
    stat.burn_confirmation_lock(config)
    _log("adjudicate-confirm", "confirmation_used.lock burned (create-once)")
    _with_heartbeat("adjudicate-confirm",
                    lambda: stat.assemble_confirm_quantities(config))
    _log("adjudicate-confirm", "confirm group quantities assembled (create-once); "
                              "starting the single frozen-bootstrap adjudication")
    result = _with_heartbeat("adjudicate-confirm",
                             lambda: stat.adjudicate_confirmation(config))
    status = str(result.get("status", prot.STATUS_INVALID))
    write_handoff_state(config, terminal_state=status)
    _log("adjudicate-confirm", f"terminal status {status}; HANDOFF_STATE.md written; "
                               "all artifacts preserved")
    if status != prot.STATUS_CONFIRMED:
        raise TerminalState(
            status, "the one-shot confirmation did not confirm; terminal state and "
            "paper boundary are recorded; no resampling, no score tuning, no re-judging")
    return {"stage": "adjudicate-confirm", "action": "executed", "status": status}


def stage_handoff(config: Mapping[str, Any]) -> dict[str, Any]:
    """Write the terminal HANDOFF_STATE.md and print the doc-sync targets.

    This stage never edits AGENTS.md / 00_READ_FIRST / the master plan / the status
    index / the research log itself — those updates happen only after the terminal
    state is known, by the controller/manual step (plan Task 4).
    """
    checked = prot.validate_config(config)
    paths = stat.confirm_artifact_paths(checked)
    if not paths["report"].is_file():
        raise RuntimeError(
            "handoff requires the formal confirmatory report; run adjudicate-confirm "
            "first (success never auto-starts a next stage)")
    state_path = write_handoff_state(config)
    _log("handoff", f"{HANDOFF_STATE_NAME} updated: {state_path}")
    _log("handoff", "terminal state reached; the following documents need "
                    "manual/controller sync (this runner never edits them):")
    for target in DOC_SYNC_TARGETS:
        _log("handoff", f"  doc-sync -> {target}")
    return {"stage": "handoff", "action": "executed",
            "doc_sync_targets": list(DOC_SYNC_TARGETS)}


# --- handoff state writer ---------------------------------------------------------------------------


_HANDOFF_ARTIFACTS = (
    ("protocol lock", "protocol_lock"),
    ("collection receipt", "collection_receipt"),
    ("legal inference inputs", "inputs"),
    ("sealed confirmation archive (score side never reads it)", "truth"),
    ("predictions", "predictions"),
    ("risk scores", "risk_scores"),
    ("inference receipt", "inference_receipt"),
    ("determinism receipt", "determinism_receipt"),
    ("score freeze receipt", "score_freeze_receipt"),
    ("confirmation_used.lock", "lock"),
    ("confirm group quantities", "quantities_npz"),
    ("quantities receipt", "quantities_receipt"),
    ("confirmatory report", "report"),
    ("confirmatory decision", "decision"),
)


def _artifact_line(label: str, path: Path) -> str:
    if not path.is_file():
        return f"| {label} | 缺失 | — |"
    if path.suffix in (".json", ".lock"):
        return f"| {label} | 存在 | sha256 `{sha256_file(path)[:16]}…` |"
    size_mb = path.stat().st_size / 1e6
    return f"| {label} | 存在 | {size_mb:.1f} MB |"


def write_handoff_state(config: Mapping[str, Any], *, terminal_state: str | None = None,
                        note: str = "") -> Path:
    """(Re)write HANDOFF_STATE.md — the recovery anchor for this stage's own docs.

    The terminal state comes from the explicit argument, else from the formal
    report, else ``NOT_ADJUDICATED``. This file lives in this stage's own
    namespace and is rewritten at each handoff; the external documents listed in
    DOC_SYNC_TARGETS are never touched here.
    """
    checked = prot.validate_config(config)
    paths = stat.confirm_artifact_paths(checked)
    report_status = None
    if paths["report"].is_file():
        report_status = str(_read_json(paths["report"]).get("status", ""))
    terminal = terminal_state or report_status or "NOT_ADJUDICATED"
    run_dir = _resolve(checked["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [_artifact_line(label, paths[key]) for label, key in _HANDOFF_ARTIFACTS]
    lines = [
        "# Stage 4C-R2 独立确认 HANDOFF_STATE",
        "",
        f"> 计划：`{prot.PLAN_ID}`",
        "> 本文件是恢复锚点，由 `code/run_stage4cr2_confirm.py` 的 handoff 阶段重写；"
        "骨架由 Task 4 创建。",
        f"> 更新时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "## 当前状态（终态）",
        "",
        f"- 终态：**{terminal}**"
        + ("（§8 合法终态）" if terminal in prot.TERMINAL_STATES else ""),
        "- 含义与论文边界：见 `reports/STAGE4CR2_CONFIRMATORY_DECISION.md`（裁判后由"
        "统计层写入；成功句式仅当终态为 STAGE4CR2_DECISION_RISK_CONFIRMED 时合法）。",
    ]
    if note:
        lines.append(f"- 备注：{note}")
    lines += [
        "",
        "## 一次性/create-once 产物状态",
        "",
        "| 产物 | 状态 | 校验 |",
        "|---|---|---|",
        *rows,
        "",
        "## 文档同步清单（终态确定后由 controller/人工更新；runner 只打印，不自动改这些文件）",
        "",
        *[f"- [ ] {target}" for target in DOC_SYNC_TARGETS],
        "",
        "## 下一动作",
        "",
        ("- 无 runner 内动作；成功也不自动启动下一阶段。文档同步完成后登记授权结束。"
         if terminal in prot.TERMINAL_STATES else
         "- operational：按 §10 顺序执行 run-all（preflight → tests → smoke → readiness "
         "→ lock → collect → infer/recheck → freeze → adjudicate → handoff）。"),
        "",
        "## Blocker",
        "",
        "- 无。" if terminal in prot.TERMINAL_STATES else "- 见上方终态/备注。",
        "",
    ]
    path = run_dir / HANDOFF_STATE_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- verify (read-only; NEVER touches the sealed archive) --------------------------------------------


def cmd_verify(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only hash + status verification of the whole chain.

    Never opens — never even hashes — ``confirmation_truth.npz``: the sealed
    archive's integrity is cross-checked through receipt-recorded hashes alone
    (collection receipt vs score freeze vs burned lock vs quantities receipt).
    The deep re-derivation of the protocol lock (source/seed rescan) belongs to
    the ``lock-protocol`` stage / ``prot.verify_protocol``; here the lock gets a
    cheap binding check (plan id, config signature, this stage's code hashes).
    """
    checked = prot.validate_config(config)
    paths = stat.confirm_artifact_paths(checked)
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        _log("verify", f"{'OK  ' if ok else 'FAIL'} {name}"
                       + (f" — {detail}" if detail else ""))

    lock_path = paths["protocol_lock"]
    if not lock_path.is_file():
        record("protocol_lock", False, "missing")
    else:
        payload = _read_json(lock_path)
        ok = payload.get("plan_id") == prot.PLAN_ID
        signature = canonical_fingerprint(json.loads(json.dumps(dict(checked))))
        ok = ok and payload.get("config_signature") == signature
        # recovery-chain aware (plan section 8): the effective code set is the
        # original lock extended by any immutable recovery locks chained from it.
        try:
            locked_code = prot.resolve_locked_code_hashes(lock_path)
            drift = [rel for rel in prot.PROTOCOL_CODE_FILES
                     if not (ROOT / "code" / rel).is_file()
                     or locked_code.get(rel) != sha256_file(ROOT / "code" / rel)]
            detail = f"code drift: {drift}" if drift else ""
        except (ValueError, FileNotFoundError) as exc:
            drift = [str(exc)]
            detail = f"recovery chain invalid: {exc}"
        record("protocol_lock", ok and not drift, detail)

    collection = None
    inputs_sha = sha256_file(paths["inputs"]) if paths["inputs"].is_file() else None
    receipt_path = paths["collection_receipt"]
    if not receipt_path.is_file():
        record("collection_receipt", False, "missing")
    else:
        collection = _read_json(receipt_path)
        ok = (collection.get("plan_id") == prot.PLAN_ID
              and collection.get("status") == stat.STATUS_COLLECTED
              and not collection.get("partial")
              and int(collection.get("accepted_groups", -1))
              == prot.CONFIRM_ACCEPTED_GROUPS)
        record("collection_receipt", ok,
               f"status={collection.get('status')!r} "
               f"accepted={collection.get('accepted_groups')}")
        record("collection_inputs_hash",
               inputs_sha is not None
               and inputs_sha == str(collection.get("inference_inputs_sha256", "")))
    # recorded-only: the sealed file itself is never opened nor hashed here.
    sealed_sha = str((collection or {}).get("confirmation_truth_sha256", ""))

    inference = None
    if not paths["inference_receipt"].is_file():
        record("inference_receipt", False, "missing")
    else:
        inference = _read_json(paths["inference_receipt"])
        ok = inference.get("status") == cinf.STATUS_INFERENCE
        for key, path in (("predictions_sha256", paths["predictions"]),
                          ("risk_scores_sha256", paths["risk_scores"])):
            ok = ok and path.is_file() \
                and sha256_file(path) == str(inference.get(key, ""))
        ok = ok and inputs_sha is not None \
            and inputs_sha == str(inference.get("data_sha256", ""))
        record("inference_receipt", ok)

    if not paths["determinism_receipt"].is_file():
        record("determinism_receipt", False, "missing")
    else:
        determinism = _read_json(paths["determinism_receipt"])
        bindings = determinism.get("bindings", {})
        ok = bool(determinism.get("bitwise_identical"))
        if inference is not None and collection is not None:
            expected = {
                "predictions_sha256": inference.get("predictions_sha256"),
                "risk_scores_sha256": inference.get("risk_scores_sha256"),
                "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
                "inference_inputs_sha256": inference.get("data_sha256"),
                "collection_receipt_sha256": sha256_file(receipt_path),
            }
            ok = ok and all(bindings.get(key) == value
                            for key, value in expected.items())
        record("determinism_receipt", ok,
               f"n_rechecked={determinism.get('n_groups_rechecked')}")

    freeze_path = paths["score_freeze_receipt"]
    if not freeze_path.is_file():
        record("score_freeze_receipt", False, "missing")
    else:
        freeze = _read_json(freeze_path)
        bindings = freeze.get("bindings", {})
        ok = freeze.get("plan_id") == prot.PLAN_ID
        if inference is not None:
            ok = ok and bindings.get("inference_inputs_sha256") \
                == inference.get("data_sha256")
            ok = ok and bindings.get("predictions_sha256") \
                == inference.get("predictions_sha256")
            ok = ok and bindings.get("risk_scores_sha256") \
                == inference.get("risk_scores_sha256")
            ok = ok and bindings.get("inference_receipt_sha256") \
                == sha256_file(paths["inference_receipt"])
        if paths["determinism_receipt"].is_file():
            ok = ok and bindings.get("determinism_receipt_sha256") \
                == sha256_file(paths["determinism_receipt"])
        if collection is not None:
            ok = ok and bindings.get("collection_receipt_sha256") \
                == sha256_file(receipt_path)
        if lock_path.is_file():
            ok = ok and bindings.get("protocol_lock_sha256") == sha256_file(lock_path)
        ok = ok and bool(sealed_sha) and bindings.get(
            "expected_confirmation_truth_sha256") == sealed_sha
        record("score_freeze_receipt", ok)

    used_lock = paths["lock"]
    if not used_lock.is_file():
        record("confirmation_used_lock", False, "not burned")
    else:
        payload = _read_json(used_lock)
        bindings = payload.get("bindings", {})
        ok = payload.get("plan_id") == prot.PLAN_ID
        if freeze_path.is_file():
            ok = ok and bindings.get("score_freeze_receipt_sha256") \
                == sha256_file(freeze_path)
        ok = ok and bool(sealed_sha) and bindings.get(
            "expected_confirmation_truth_sha256") == sealed_sha
        record("confirmation_used_lock", ok)

    if not paths["quantities_npz"].is_file() or not paths["quantities_receipt"].is_file():
        record("confirm_group_quantities", False, "missing")
    else:
        receipt = _read_json(paths["quantities_receipt"])
        ok = receipt.get("plan_id") == prot.PLAN_ID
        ok = ok and sha256_file(paths["quantities_npz"]) \
            == str(receipt.get("confirm_group_quantities_sha256", ""))
        if used_lock.is_file():
            ok = ok and receipt.get("lock_sha256") == sha256_file(used_lock)
        ok = ok and int(receipt.get("bootstrap_entropy", -1)) == prot.BOOTSTRAP_ENTROPY
        ok = ok and int(receipt.get("bootstrap_resamples", -1)) == prot.BOOTSTRAP_RESAMPLES
        ok = ok and float(receipt.get("confidence", -1.0)) == prot.CONFIDENCE
        ok = ok and receipt.get("primary_baseline") == stat.PRIMARY_BASELINE
        ok = ok and bool(sealed_sha) and receipt.get("bindings", {}).get(
            "confirmation_truth_sha256") == sealed_sha
        record("confirm_group_quantities", ok, f"n_groups={receipt.get('n_groups')}")

    terminal = "not_adjudicated"
    if not paths["report"].is_file():
        record("confirmatory_report", False, "missing")
    else:
        report = _read_json(paths["report"])
        terminal = str(report.get("status", ""))
        ok = report.get("plan_id") == prot.PLAN_ID and terminal in prot.TERMINAL_STATES
        if paths["quantities_receipt"].is_file():
            receipt = _read_json(paths["quantities_receipt"])
            ok = ok and report.get("bindings", {}).get(
                "confirm_group_quantities_sha256") \
                == receipt.get("confirm_group_quantities_sha256")
        record("confirmatory_report", ok, f"terminal={terminal}")

    all_verified = bool(checks) and all(check["ok"] for check in checks)
    _log("verify", f"terminal_state={terminal} all_verified={all_verified}")
    return {"stage": "verify", "checks": checks, "terminal_state": terminal,
            "all_verified": all_verified}


# --- run-all ---------------------------------------------------------------------------------------


def run_all(config_path: str | Path = CONFIG_PATH) -> RunSummary:
    """Execute the frozen section-10 order; completed stages verify-and-skip; an
    adjudicated terminal state makes the whole run verify-only (never re-adjudicate)."""
    summary = RunSummary(config_path=str(config_path),
                         started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    started = time.perf_counter()
    config: dict[str, Any] | None = None
    try:
        config = _load_config(config_path)
    except Exception as exc:
        summary.terminal_state = prot.STATUS_INVALID
        summary.exit_code = EXIT_CODES[prot.STATUS_INVALID]
        _log("run-all", f"FAILURE: config invalid ({exc})")
        summary.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        return summary

    report_path = stat.confirm_artifact_paths(config)["report"]
    if report_path.is_file():
        # adjudicated terminal: verify only, never re-run any stage (plan section 10).
        terminal = str(_read_json(report_path).get("status", prot.STATUS_INVALID))
        _log("run-all", f"adjudicated terminal {terminal} found: verify-only resume; "
                        "no stage re-runs, no re-adjudication")
        verification = cmd_verify(config)
        summary.stages.append(verification)
        if not verification["all_verified"]:
            summary.terminal_state = prot.STATUS_INVALID
            summary.exit_code = EXIT_CODES[prot.STATUS_INVALID]
            _log("run-all", "FAILURE: adjudicated terminal state failed hash/status "
                            "verification (drift or tampering)")
        else:
            summary.terminal_state = terminal
            summary.exit_code = EXIT_CODES.get(terminal, EXIT_ERROR)
        summary.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        verdict = "SUCCESS" if summary.exit_code == EXIT_OK else "FAILURE"
        _log("run-all", f"{verdict} summary: terminal_state={summary.terminal_state} "
                        f"exit_code={summary.exit_code} (verify-only) "
                        f"elapsed={time.perf_counter() - started:.1f}s")
        return summary

    try:
        for index, name in enumerate(PIPELINE, start=1):
            _log(name, f"starting ({index}/{len(PIPELINE)})")
            outcome = {"stage": name, **_stage_function(name)(config)}
            summary.stages.append(outcome)
            _log(name, outcome.get("action", "done"))
        terminal = prot.STATUS_CONFIRMED
        if report_path.is_file():
            terminal = str(_read_json(report_path).get("status", terminal))
        summary.terminal_state = terminal
        summary.exit_code = EXIT_CODES.get(terminal, EXIT_ERROR)
    except TerminalState as stop:
        summary.terminal_state = stop.status
        summary.exit_code = EXIT_CODES.get(stop.status, EXIT_ERROR)
        _log("run-all", f"TERMINAL STATE: {stop.status} (exit {summary.exit_code}) "
                        f"{stop.detail}")
    except prot.PreflightError as exc:
        summary.terminal_state = exc.status
        summary.exit_code = EXIT_CODES.get(exc.status, EXIT_ERROR)
        _log("run-all", f"TERMINAL STATE: {exc.status} (exit {summary.exit_code}) {exc}")
    except collector.CollectionIncompleteError as exc:
        summary.terminal_state = prot.STATUS_COLLECTION_INCOMPLETE
        summary.exit_code = EXIT_CODES[prot.STATUS_COLLECTION_INCOMPLETE]
        _log("run-all", f"TERMINAL STATE: {prot.STATUS_COLLECTION_INCOMPLETE} "
                        f"(exit {summary.exit_code}) {exc}")
    except stat.ResourceAtRiskError as exc:
        summary.terminal_state = prot.STATUS_RESOURCE_AT_RISK
        summary.exit_code = EXIT_CODES[prot.STATUS_RESOURCE_AT_RISK]
        _log("run-all", f"TERMINAL STATE: {prot.STATUS_RESOURCE_AT_RISK} "
                        f"(exit {summary.exit_code}) {exc}")
    except Exception as exc:  # engineering failure: keep artifacts, halt loudly
        summary.terminal_state = summary.terminal_state or "ENGINEERING_HALT"
        summary.exit_code = EXIT_ERROR
        _log("run-all", f"engineering failure: {type(exc).__name__}: {exc}")
    if summary.exit_code != EXIT_OK and config is not None \
            and summary.terminal_state != prot.STATUS_CONFIRMED:
        # section 8: record the terminal/halt state; every artifact stays in place.
        try:
            note = "" if summary.terminal_state in prot.TERMINAL_STATES else \
                f"工程中断于 stage 序列；未裁判。stages_completed={len(summary.stages)}"
            write_handoff_state(
                config,
                terminal_state=summary.terminal_state
                if summary.terminal_state in prot.TERMINAL_STATES else None,
                note=note or f"终态 {summary.terminal_state}；全部产物保留。")
        except Exception as exc:  # never mask the real failure with a doc failure
            _log("run-all", f"handoff-state write failed (non-masking): {exc}")
    summary.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    verdict = "SUCCESS" if summary.exit_code == EXIT_OK else "FAILURE"
    _log("run-all", f"{verdict} summary: terminal_state={summary.terminal_state} "
                    f"exit_code={summary.exit_code} stages={len(summary.stages)}/"
                    f"{len(PIPELINE)} elapsed={time.perf_counter() - started:.1f}s")
    return summary


# --- CLI -------------------------------------------------------------------------------------------


_COMMANDS = ("preflight", "smoke", "lock-protocol", "collect-confirm", "infer-confirm",
             "freeze-scores", "adjudicate-confirm", "verify", "run-all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4C-R2 independent-confirmation staged runner")
    parser.add_argument("command", choices=_COMMANDS)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    if args.command == "run-all":
        return run_all(args.config).exit_code
    try:
        config = _load_config(args.config)
    except Exception as exc:
        _log(args.command, f"config invalid: {exc}")
        return EXIT_CODES[prot.STATUS_INVALID]
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
    except prot.PreflightError as exc:
        _log(args.command, f"TERMINAL STATE: {exc.status}: {exc}")
        return EXIT_CODES.get(exc.status, EXIT_ERROR)
    except collector.CollectionIncompleteError as exc:
        _log(args.command, f"TERMINAL STATE: {prot.STATUS_COLLECTION_INCOMPLETE}: {exc}")
        return EXIT_CODES[prot.STATUS_COLLECTION_INCOMPLETE]
    except stat.ResourceAtRiskError as exc:
        _log(args.command, f"TERMINAL STATE: {prot.STATUS_RESOURCE_AT_RISK}: {exc}")
        return EXIT_CODES[prot.STATUS_RESOURCE_AT_RISK]
    except Exception as exc:  # engineering failure: artifacts stay in place
        _log(args.command, f"FAILURE (engineering): {type(exc).__name__}: {exc}")
        return EXIT_ERROR
    print(json.dumps(outcome, indent=2, ensure_ascii=False, default=str), flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
