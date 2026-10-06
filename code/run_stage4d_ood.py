"""Stage 4D D0/D1 unified runner (plan §12 orchestration; Task 3 scope: D0 pilot phases).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 3 and §12. This runner
ONLY orchestrates the frozen stage functions of ``stage4d_protocol`` /
``collect_stage4d_merge_ood`` / ``stage4d_ood_statistics`` — it never re-implements their
science and never re-runs an already-completed create-once phase (completed phases are
hash/state-verified and skipped, never overwritten).

Subcommands:
    preflight, collect-pilot --profile {behavior|geometry|joint}, evaluate-pilots,
    select-profile, power, lock-d1, locked-smoke-readiness, collect-d1, infer-d1,
    freeze-scores, run-all

``run-all`` follows the frozen §12 order. Task 5 implements through ``score-freeze``;
the Task 6 phases (``burn-confirmation-lock`` onward) are declared in the full §12
phase list and raise "not implemented yet" — they are NEVER silently skipped.

D0 forbidden zone (plan §5.1, Global Constraint 3): no phase in this file loads a Direct
checkpoint or produces a risk score / AURC / risk-coverage / improvement quantity. The
pilot evaluation runs ONLY the frozen decision-predictor ensemble (for ``J_hat``) and
reads the pilot task truth (plan §5.1 allows it); the sealed Stage 4C-R2 confirm truth
and the R2 train CF-future/regret fields are never opened.

Exit codes: 0 success; 1 engineering/verification failure; 2 pipeline boundary (the
remaining Task 4/5/6 phases are not implemented yet); 4 collection incomplete (matches
the collector); terminal states map to dedicated non-zero codes (see ``EXIT_CODES``).
Long phases print a plain-text heartbeat at least every 25 s.
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

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage4d_merge_ood as collector  # noqa: E402
from ac_twin import stage4d_confirm_protocol as confirm_prot  # noqa: E402
from ac_twin import stage4d_ood_inference as d1_inference  # noqa: E402
from ac_twin import stage4d_ood_statistics as d0  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402
from ac_twin.progress import format_duration  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

ROOT = PROJECT_ROOT
CONFIG_PATH = prot.CONFIG_PATH

#: frozen run-all order (plan §12, verbatim); Task 3 implements through select-profile.
PIPELINE = (
    "preflight",
    "focused-tests",
    "engineering-smoke",
    "full-regression-readiness",
    "collect-pilots",
    "evaluate-pilots",
    "select-profile",
    "power",                    # Task 4
    "lock-d1",                  # Task 4
    "locked-smoke-readiness",   # Task 4
    "collect-d1",               # Task 5
    "inference-scoring",        # Task 5
    "score-freeze",             # Task 5
    "burn-confirmation-lock",   # Task 6
    "adjudicate",               # Task 6
    "reports-status-sync",      # Task 6
)

#: the D0 pilot collection order (plan §10 Task 3): behavior -> geometry -> joint.
PILOT_COLLECTION_ORDER = ("pilot_behavior", "pilot_geometry", "pilot_joint")

#: this stage's blocking test files (never exempted).
STAGE_TEST_FILES = (
    "tests/test_stage4d_protocol.py",
    "tests/test_stage4d_ood.py",
    "tests/test_stage4d_collection.py",
    "tests/test_stage4d_d0_statistics.py",
    "tests/test_stage4d_runner.py",
    "tests/test_stage4d_d1_power.py",
    "tests/test_stage4d_confirm_protocol.py",
    "tests/test_stage4d_inference.py",
    "tests/test_stage4d_adjudication.py",
)

HANDOFF_STATE_NAME = "HANDOFF_STATE.md"
READINESS_EXEMPTIONS_NAME = "readiness_exemptions.json"

#: predictor-stack assets of the frozen R2 chain (read-only; hash-verified inside the
#: reused R2 machinery via the stage3b binding receipt). Used ONLY to recompute the
#: decision predictor's J_hat for the §5.2 task support — never a Direct artifact.
STAGE3B_BINDING = "experiments/stage3_routing/stage3b/input_binding_receipt.json"
STAGE3R_RUN_DIR = "experiments/stage3_routing/stage3r"
TRAJECTORY_SCALER_PATH = "experiments/stage3_routing/stage3r/scaler.json"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PIPELINE_BOUNDARY = 2
EXIT_COLLECTION_INCOMPLETE = 4
EXIT_CODES = {
    prot.STATUS_CONFIRMED: EXIT_OK,
    prot.STATUS_INVALID: 3,
    collector.STATUS_COLLECTION_INCOMPLETE: EXIT_COLLECTION_INCOMPLETE,
    prot.STATUS_UNDERPOWERED: 10,
    prot.STATUS_TASK_DEGENERATE: 11,
    prot.STATUS_MODEL_INSUFFICIENT: 12,
    prot.STATUS_TASK_INSUFFICIENT: 13,
    prot.STATUS_NOT_CONFIRMED: 14,
}

_HEARTBEAT_SECONDS = 25.0  # under the 30 s contract

#: documents needing manual/controller sync at a terminal state (the runner prints
#: them; it edits only this stage's own HANDOFF_STATE.md).
DOC_SYNC_TARGETS = (
    "00_READ_FIRST_AC_TWIN.md",
    "01_AC_TWIN_DOCUMENT_STATUS_INDEX.md",
    "logs/ 当日科研日志 (research log)",
)


class TerminalState(RuntimeError):
    """A scientific hard stop: carries the plan-§7.3 terminal state; the pipeline halts."""

    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}" if detail else status)
        self.status = str(status)
        self.detail = detail


@dataclass
class RunSummary:
    """run-all outcome: per-phase actions, the terminal state and the exit code."""
    config_path: str
    phases: list[dict[str, Any]] = field(default_factory=list)
    terminal_state: str = ""
    exit_code: int = EXIT_OK
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "phases": self.phases,
            "terminal_state": self.terminal_state,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


# --- helpers -----------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return prot.validate_config(
        json.loads(Path(config_path).read_text(encoding="utf-8")))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _log(phase: str, message: str) -> None:
    print(f"[stage4d-ood:{phase}] {message}", flush=True)


def _with_heartbeat(label: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a worker thread; print a heartbeat at least every 25 s.

    Opaque single-call phases (pytest regressions, predictor inference, the bootstrap)
    have no done/total channel, so the heartbeat carries phase + elapsed; collection
    self-reports the full line via ``ac_twin.progress.ProgressReporter`` (30 s).
    """
    result: dict[str, Any] = {}
    started = time.perf_counter()

    def work() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # propagate, never swallow
            result["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=_HEARTBEAT_SECONDS)
        if thread.is_alive():
            elapsed = time.perf_counter() - started
            print(f"[stage4d-ood:{label}] heartbeat | elapsed={format_duration(elapsed)}",
                  flush=True)
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _failed_nodes(pytest_output: str) -> list[str]:
    nodes = []
    for line in pytest_output.splitlines():
        if line.startswith("FAILED "):
            nodes.append(line[len("FAILED "):].split(" - ")[0].strip())
    return nodes


def _run_pytest(label: str, args: list[str]) -> tuple[int, list[str], str]:
    """Run pytest as a subprocess under the heartbeat; return (rc, failed nodes, tail)."""
    cmd = [sys.executable, "-m", "pytest", *args, "-q", "-rf", "--color=no"]
    _log(label, f"running: {' '.join(cmd)}")
    proc = _with_heartbeat(
        label, lambda: subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True))
    output = _strip_ansi((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
    tail = "\n".join(output.strip().splitlines()[-15:]) if output.strip() else ""
    if tail:
        print(tail, flush=True)
    return proc.returncode, _failed_nodes(output), tail


def _sync_handoff_status(run_dir: Path, status: str) -> None:
    """Update this stage's own HANDOFF_STATE.md status line (never other documents)."""
    path = run_dir / HANDOFF_STATE_NAME
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(> \*\*当前状态：\*\* `)[^`]+(`)", rf"\g<1>{status}\g<2>", text, count=1)
    if count:
        path.write_text(updated, encoding="utf-8")


# --- phases (Task 3 scope) -----------------------------------------------------------------


def stage_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Source chain + R2 train binding + global seed audit for the three pilot domains."""
    checked = prot.validate_config(config)
    inventory = _with_heartbeat("preflight", lambda: prot.build_source_inventory())
    if inventory["n_files"] != len(prot.STAGE4D_PINNED_SOURCES):
        raise prot.PreflightError("source inventory does not cover plan §2.1")
    drifted = [rel for rel, entry in inventory["files"].items()
               if entry["status"] != "match"]
    if drifted:
        raise prot.PreflightError(f"frozen source drift: {drifted}")
    roots = [ROOT / rel for rel in checked["historical_scan_roots"]]
    audit = {}
    for role in PILOT_COLLECTION_ORDER:
        schedule = collector.role_schedule(checked, role)
        result = _with_heartbeat(
            "preflight", lambda s=schedule: prot.audit_global_seed_overlap(s, roots))
        prot.assert_seed_audit_clean(result)
        audit[role] = {"n_schedule": result.n_schedule, "zero_overlap": True}
    _sync_handoff_status(_resolve(checked["run_dir"]), prot.STATUS_PREFLIGHT_PASSED)
    _log("preflight", f"{prot.STATUS_PREFLIGHT_PASSED}; §2.1 ten sources match; "
                      "pilot seed domains globally zero-overlap")
    return {"phase": "preflight", "action": "executed",
            "status": prot.STATUS_PREFLIGHT_PASSED, "seed_audit": audit}


def stage_focused_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    """This stage's blocking test files under -W error; never exempted."""
    missing = [rel for rel in STAGE_TEST_FILES if not (ROOT / rel).is_file()]
    if missing:
        raise RuntimeError(f"blocking test files missing: {missing}")
    rc, failed, _ = _run_pytest("focused-tests", [*STAGE_TEST_FILES, "-W", "error"])
    if rc != 0:
        raise RuntimeError(
            f"Stage 4D blocking tests failed (exit {rc}); failing nodes {failed}; "
            "this stage's own tests are never exempted")
    _log("focused-tests", f"{len(STAGE_TEST_FILES)} blocking test files green (-W error)")
    return {"phase": "focused-tests", "action": "executed",
            "files": list(STAGE_TEST_FILES)}


def stage_engineering_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """The 12-group ENGINEERING_SMOKE_ONLY collection (resumes/verifies; never science)."""
    receipt = collector.collect_role(config, "smoke")
    if receipt.get("partial") or int(receipt.get("accepted_groups", -1)) != 12:
        raise TerminalState(prot.STATUS_INVALID, "engineering smoke collection incomplete")
    action = "skipped_complete" if receipt.get("verification") else "executed"
    _log("engineering-smoke", f"ENGINEERING_SMOKE_ONLY verified "
                              f"({receipt['accepted_groups']} groups)")
    return {"phase": "engineering-smoke", "action": action,
            "accepted_groups": int(receipt["accepted_groups"]),
            "status": receipt.get("status")}


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


def stage_full_regression_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    """Full-repo regression; only the controller-registered historical exemptions may
    fail (plan §10; this stage's own tests and any NEW failure are never exempted)."""
    checked = prot.validate_config(config)
    rc, failed, _ = _run_pytest("readiness", ["tests"])
    if rc == 0:
        _log("readiness", "full regression green (zero failures)")
        return {"phase": "full-regression-readiness", "action": "executed",
                "full_regression": "green"}
    exemptions = _load_readiness_exemptions(_resolve(checked["run_dir"]))
    exempted = set(exemptions.get("exempt_nodes", []))
    if failed and set(failed) <= exempted:
        _log("readiness", "full regression: only registered non-blocking historical "
                          f"exemptions failed ({len(failed)} nodes)")
        return {"phase": "full-regression-readiness", "action": "executed",
                "full_regression": "green_with_registered_exemptions",
                "failed_exempted": failed}
    raise RuntimeError(
        f"full regression failures exceed the registered non-blocking exemptions: "
        f"{failed}")


def stage_collect_pilots(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the three D0 pilots in the frozen order behavior -> geometry -> joint.

    300 accepted complete groups each, <= 6,000 candidate seeds each; the collector
    self-reports a 30 s heartbeat and resumes/verifies create-once products. A completed
    pilot is verified against its receipt and never re-collected.
    """
    receipts = {}
    for role in PILOT_COLLECTION_ORDER:
        receipt = collector.collect_role(config, role)
        if receipt.get("partial"):
            raise TerminalState(prot.STATUS_INVALID, f"{role} collection incomplete")
        receipts[role] = {
            "accepted_groups": int(receipt["accepted_groups"]),
            "attempts": int(receipt["attempts"]),
            "status": receipt.get("status"),
            "resumed_verified": bool(receipt.get("verification")),
        }
        _log("collect-pilots", f"{role}: accepted={receipt['accepted_groups']} "
                               f"attempts={receipt['attempts']}")
    _sync_handoff_status(_resolve(prot.validate_config(config)["run_dir"]),
                         prot.STATUS_D0_PILOTS_COLLECTED)
    return {"phase": "collect-pilots", "action": "executed", "receipts": receipts}


# --- pilot evaluation (D0 Gate inputs) ------------------------------------------------------


def _pilot_report_path(checked: Mapping[str, Any], short: str) -> Path:
    return _resolve(checked["run_dir"]) / "pilot" / short / "profile_report.json"


def _pilot_arrays(checked: Mapping[str, Any], short: str) -> tuple[dict, dict, dict]:
    """(receipt, legal arrays, pilot truth arrays); pilot truth is readable (§5.1)."""
    inputs_path, truth_path, _, receipt_path = collector.role_paths(
        checked, f"pilot_{short}")
    receipt = _read_json(receipt_path)
    legal_bundle = collector.load_legal_inputs(inputs_path, checked)
    d0.assert_d0_read_legal(truth_path)  # pilot truth: legal in D0, still screened
    with np.load(truth_path, allow_pickle=False) as archive:
        truth = {key: np.asarray(archive[key]) for key in collector.TRUTH_FIELDS}
    if not np.array_equal(np.asarray(legal_bundle["group_seed"]),
                          np.asarray(truth["group_seed"])):
        raise ValueError(f"{short}: legal/truth group order divergence")
    return receipt, legal_bundle["arrays"], truth


def _run_pilot_e2e_inference(
    checked: Mapping[str, Any],
    legal: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Frozen decision-predictor ensemble e2e effects (5,G,10,20,5,2); NEVER Direct.

    Reuses ``stage4c_inference.run_grouped_frozen_inference`` read-only with the frozen
    stage3r predictors + stage3b binding + train-only trajectory scaler. No risk model,
    no scoring — the output feeds ONLY the §5.2 J_hat recomputation.
    """
    import torch

    from ac_twin import stage4c_inference as s4ci
    from ac_twin.stage3_scaler import FeatureScaler

    history = np.asarray(legal["object_history"], dtype=np.float32)
    factual = np.asarray(legal["object_factual_future"], dtype=np.float32)
    factual_actions = np.asarray(legal["factual_actions"], dtype=np.int64)
    cf_actions = np.asarray(legal["cf_actions"], dtype=np.int64)
    cf_durations = np.asarray(legal["cf_durations"], dtype=np.int64)
    t_star = np.asarray(legal["t_star"], dtype=np.int64).ravel()
    mask = np.asarray(legal["inference_mask"], dtype=bool)
    groups = int(history.shape[0])
    n_actions = prot.N_ACTIONS

    g_idx = np.repeat(np.arange(groups), n_actions)
    hist_b = history[g_idx]
    fact_b = factual[g_idx]
    cf_plans = np.stack([
        prot.materialize_action_plan(factual_actions[g], int(cf_actions[g, a]),
                                     int(cf_durations[g, a]))
        for g in range(groups) for a in range(n_actions)
    ]).astype(np.int64)

    scaler = FeatureScaler.load(_resolve(TRAJECTORY_SCALER_PATH))
    scaled = {
        "history": torch.as_tensor(scaler.transform(hist_b), dtype=torch.float32),
        "factual_suffix": torch.as_tensor(
            scaler.transform(fact_b[:, :prot.FACTUAL_SUFFIX_STEPS]), dtype=torch.float32),
        "factual_actions": torch.as_tensor(factual_actions[g_idx], dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(cf_plans, dtype=torch.long),
        "cf_action": torch.as_tensor(cf_actions.reshape(-1), dtype=torch.long),
        "cf_duration": torch.as_tensor(cf_durations.reshape(-1), dtype=torch.long),
        "t_star": torch.as_tensor(t_star[g_idx], dtype=torch.long),
    }
    bundle = {
        "groups": groups,
        "n_actions": n_actions,
        "branches": groups * n_actions,
        "scaled": scaled,
        "valid": mask[g_idx],
    }
    binding = _read_json(_resolve(STAGE3B_BINDING))["receipt_payload"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = s4ci.run_grouped_frozen_inference(
        bundle, binding=binding, stage3r_dir=_resolve(STAGE3R_RUN_DIR),
        device=device, batch_size=64)
    return np.asarray(predictions["e2e"], dtype=np.float32)


def _evaluate_one_pilot(
    checked: Mapping[str, Any],
    short: str,
    id_features: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate one pilot's D0 Gate and write its create-once profile report."""
    report_path = _pilot_report_path(checked, short)
    profile = d0.PROFILE_OF_PILOT[short]
    if report_path.is_file():
        report = _read_json(report_path)
        inputs_path, truth_path, _, receipt_path = collector.role_paths(
            checked, f"pilot_{short}")
        receipt = _read_json(receipt_path)
        bindings = report.get("bindings", {})
        for key, path in (("inference_inputs_sha256", inputs_path),
                          ("confirmation_truth_sha256", truth_path)):
            if sha256_file(path) != bindings.get(key) \
                    or bindings.get(key) != receipt.get(key):
                raise ValueError(f"{short} report binding drift: {path}")
        return {**report, "verification": "existing_report_verified_no_overwrite"}

    receipt, legal, truth = _pilot_arrays(checked, short)
    _, _, _, receipt_path = collector.role_paths(checked, f"pilot_{short}")
    integrity = d0.evaluate_pilot_integrity(receipt, legal, truth)

    ood_features = d0.pre_intervention_features(legal["object_history"])
    shift = d0.compute_domain_shift(
        id_features, ood_features,
        bootstrap_seed=prot.DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES[short])

    e2e = _run_pilot_e2e_inference(checked, legal)
    quantities = d0.assemble_pilot_task_quantities(legal, truth, e2e)
    task = d0.evaluate_task_support(
        regret=quantities["regret"], a_star=quantities["a_star"],
        J_hat=quantities["J_hat"], J_true=quantities["J_true"],
        idle_regret=quantities["idle_regret"],
        random_regret=quantities["random_regret"])

    gate = d0.evaluate_d0_profile(integrity, shift, task)
    report = {
        "plan_id": prot.PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "profile": profile,
        "pilot": short,
        "collection": {
            "role": receipt.get("role"),
            "status": receipt.get("status"),
            "accepted_groups": int(receipt.get("accepted_groups", -1)),
            "attempts": int(receipt.get("attempts", -1)),
            "seed_schedule_sha256": receipt.get("seed_schedule_sha256"),
            "action_stream_root": receipt.get("action_stream_root"),
        },
        "bindings": {
            "inference_inputs_sha256": receipt.get("inference_inputs_sha256"),
            "confirmation_truth_sha256": receipt.get("confirmation_truth_sha256"),
            "collection_receipt_sha256": sha256_file(receipt_path),
        },
        "integrity": integrity,
        "domain_shift": shift,
        "task_support": task,
        "gate": gate,
        "qualified": bool(gate["qualified"]),
    }
    d0.write_profile_report(report_path, report)
    _log("evaluate-pilots",
         f"{profile}: qualified={report['qualified']} "
         f"shifted={shift['n_shifted']}/{len(d0.PRE_INTERVENTION_QUANTITIES)} "
         f"auc_ci_lb={shift['domain_classifier']['auc_ci_lower']:.4f} "
         f"high_regret={task['high_regret_frac']:.3f} "
         f"spearman={task['spearman_jhat_jtrue']:.4f}")
    return report


def stage_evaluate_pilots(config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the three pilots independently (integrity + shift + task support).

    The ID reference features are computed ONCE from the R2 train pre-intervention
    history (hash-verified; CF-future/regret fields never read). No Direct checkpoint is
    loaded and no risk score / AURC / coverage is produced anywhere in this phase.
    """
    checked = prot.validate_config(config)
    id_features = _with_heartbeat(
        "evaluate-pilots", lambda: d0.load_r2_train_id_features(checked))
    reports = {}
    for short in d0.PILOT_ORDER:
        reports[short] = _with_heartbeat(
            f"evaluate-pilots:{short}",
            lambda s=short: _evaluate_one_pilot(checked, s, id_features))
    return {"phase": "evaluate-pilots", "action": "executed",
            "qualified": {d0.PROFILE_OF_PILOT[s]: bool(r["qualified"])
                          for s, r in reports.items()}}


def stage_select_profile(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §5.3: freeze the first qualified profile; write the create-once selection.

    With no qualified profile the create-once record is written honestly and the
    pre-registered terminal state ``STAGE4D_OOD_TASK_INSUFFICIENT`` is raised.
    """
    checked = prot.validate_config(config)
    selection_path = _resolve(checked["paths"]["ood_profile_selection"])
    if selection_path.is_file():
        selection = _read_json(selection_path)
        selected = selection.get("selected_profile")
        _log("select-profile", f"existing selection verified (create-once): {selected}")
        if selected is None:
            # the recorded hard stop is absorbing: resume re-raises it, never proceeds.
            raise TerminalState(
                prot.STATUS_TASK_INSUFFICIENT,
                "the create-once selection record already records that no D0 profile "
                "qualified; resume never re-runs selection or proceeds past it")
        return {**selection, "verification": "existing_selection_verified_no_overwrite"}

    reports = {}
    for short, profile in d0.PROFILE_OF_PILOT.items():
        report_path = _pilot_report_path(checked, short)
        if not report_path.is_file():
            raise RuntimeError(
                f"profile report missing for {profile}: {report_path}; run "
                "evaluate-pilots first")
        reports[profile] = _read_json(report_path)
    selection = d0.select_ood_profile(reports)
    selection["report_sha256"] = {
        profile: sha256_file(_pilot_report_path(checked, d0.PILOT_OF_PROFILE[profile]))
        for profile in reports}
    d0.write_profile_selection(selection_path, selection)

    run_dir = _resolve(checked["run_dir"])
    if selection["selected_profile"] is None:
        _sync_handoff_status(run_dir, prot.STATUS_TASK_INSUFFICIENT)
        _log("select-profile", f"TERMINAL {prot.STATUS_TASK_INSUFFICIENT}: no qualified "
                               "profile; doc-sync targets: " + ", ".join(DOC_SYNC_TARGETS))
        raise TerminalState(
            prot.STATUS_TASK_INSUFFICIENT,
            "no D0 pilot profile met all plan-§5.2 conditions; the create-once "
            f"selection record is written at {selection_path}")
    _sync_handoff_status(run_dir, prot.STATUS_D0_PROFILE_SELECTED)
    _log("select-profile", f"selected {selection['selected_profile']} "
                           f"(priority {' > '.join(selection['priority'])})")
    return selection


# --- Task 4 phases: D1 power, protocol lock, locked smoke + readiness ----------------------------


def stage_power(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §6.1: D1 joint power; freeze the selected n or hard-stop UNDERPOWERED.

    Reads ONLY the hash-pinned R2 screen quantities NPZ (+ receipt). Writes the create-once
    ``power_report.json``. On ``STAGE4D_OOD_CONFIRMATION_UNDERPOWERED`` the report is written
    honestly and NO confirm resource (no protocol lock, no collection path) is created; the
    recorded hard stop is absorbing (resume re-raises, never proceeds to the lock).
    """
    checked = prot.validate_config(config)
    report_path = _resolve(checked["paths"]["power_report"])
    if report_path.is_file():
        report = _read_json(report_path)
        if report.get("underpowered") or report.get("status") == prot.STATUS_UNDERPOWERED:
            raise TerminalState(
                prot.STATUS_UNDERPOWERED,
                "the create-once power report already records "
                "STAGE4D_OOD_CONFIRMATION_UNDERPOWERED; resume never proceeds to the lock")
        return {**report, "phase": "power", "action": "verified",
                "verification": "existing_power_report_verified_no_overwrite"}

    selection_path = _resolve(checked["paths"]["ood_profile_selection"])
    if not selection_path.is_file():
        raise ValueError(
            f"the D1 power analysis requires the create-once D0 profile selection first: "
            f"{selection_path}")
    selection = _read_json(selection_path)
    if selection.get("selected_profile") is None \
            or selection.get("status") != prot.STATUS_D0_PROFILE_SELECTED:
        raise ValueError(
            f"the D1 power analysis requires a selected D0 profile (status "
            f"{prot.STATUS_D0_PROFILE_SELECTED}); got status={selection.get('status')!r}")

    npz_path = _resolve(checked["r2_development_dir"]) / "screen_group_quantities.npz"
    pinned = checked["r2_pinned_sources"][
        "experiments/stage4_reliability/stage4c_r2/development/screen_group_quantities.npz"]
    if not npz_path.is_file():
        raise FileNotFoundError(f"R2 screen quantities NPZ missing: {npz_path}")
    if sha256_file(npz_path) != pinned:
        raise prot.PreflightError(f"R2 screen quantities hash drift: {npz_path}")

    report = _with_heartbeat("power", lambda: d0.simulate_d1_power(
        npz_path,
        tuple(int(n) for n in checked["candidate_ns"]),
        tuple(float(r) for r in checked["retention_factors"]),
        seed=prot.POWER_ENTROPY,
        n_resamples=prot.BOOTSTRAP_RESAMPLES,
        monte_carlo=prot.POWER_MONTE_CARLO))
    payload = d0._json_safe(report)
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_once_json(report_path, payload)  # create-once: a second write raises

    run_dir = _resolve(checked["run_dir"])
    if report["underpowered"]:
        _sync_handoff_status(run_dir, prot.STATUS_UNDERPOWERED)
        _log("power", f"TERMINAL {prot.STATUS_UNDERPOWERED}: n=4000 at 50% retention still "
                      "falls short of the 0.80 target; no confirm lock/collection created. "
                      "doc-sync targets: " + ", ".join(DOC_SYNC_TARGETS))
        raise TerminalState(
            prot.STATUS_UNDERPOWERED,
            f"power report written honestly at {report_path}; the pre-registered hard stop "
            "creates no confirm resource")
    _sync_handoff_status(run_dir, prot.STATUS_D1_POWER_FROZEN)
    _log("power", f"{prot.STATUS_D1_POWER_FROZEN}: selected n={report['selected_n']} "
                  f"(50%-retention AND-chain joint power >= {prot.POWER_TARGET})")
    return {**report, "phase": "power", "action": "executed"}


def stage_lock_d1(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §6.3: create the write-once formal protocol lock (or verify it on resume).

    The lock re-derives the global seed audit FOR REAL, expands the §2.1 sources to explicit
    path+hash, registers-then-hashes the full D1 code chain, and locks the §7.1-condition-8
    strata. Both the base (``prot.verify_protocol_lock``) and the extended
    (``confirm_prot.verify_confirm_protocol_lock``) verifications must pass.
    """
    checked = prot.validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if lock_path.is_file():
        prot.verify_protocol_lock(run_dir)
        confirm_prot.verify_confirm_protocol_lock(run_dir)
        _log("lock-d1", "existing protocol lock verified (write-once; no overwrite)")
        return {"phase": "lock-d1", "action": "verified", "status": prot.STATUS_D1_LOCKED,
                "verification": "existing_lock_verified_no_overwrite"}
    lock_path_str = _with_heartbeat(
        "lock-d1", lambda: confirm_prot.create_protocol_lock(checked))
    prot.verify_protocol_lock(run_dir)
    confirm_prot.verify_confirm_protocol_lock(run_dir)
    _sync_handoff_status(run_dir, prot.STATUS_D1_LOCKED)
    _log("lock-d1", f"{prot.STATUS_D1_LOCKED}: protocol lock created and verified: "
                    f"{lock_path_str}")
    return {"phase": "lock-d1", "action": "executed", "status": prot.STATUS_D1_LOCKED,
            "protocol_lock": lock_path_str}


def stage_locked_smoke_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    """Post-lock readiness: lock verification + engineering smoke + stage suite + regression.

    Fails closed without a burned lock (``prot.verify_protocol_lock`` raises). The engineering
    smoke re-verifies the existing 12-group product (idempotent; consumes no seeds outside the
    smoke domain). The stage suite runs under ``-W error``; the full regression may fail ONLY
    on the controller-registered historical exemptions.
    """
    checked = prot.validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    prot.verify_protocol_lock(run_dir)
    confirm_prot.verify_confirm_protocol_lock(run_dir)
    _log("locked-smoke-readiness", "protocol lock verified (base + extended)")
    smoke = stage_engineering_smoke(config)
    tests = stage_focused_tests(config)
    regression = stage_full_regression_readiness(config)
    _log("locked-smoke-readiness", "locked smoke + stage suite + full regression green")
    return {"phase": "locked-smoke-readiness", "action": "executed",
            "status": prot.STATUS_D1_LOCKED, "lock": "verified",
            "smoke": smoke, "stage_tests": tests, "regression": regression}


# --- Task 5 phases: D1 collection, frozen inference/scoring, score freeze ---------------------


def _assert_confirmation_unused(checked: Mapping[str, Any]) -> None:
    """The one-shot unseal lock must NOT exist at any score-side phase (Task 6 only)."""
    used_lock = _resolve(checked["paths"]["confirmation_used_lock"])
    if used_lock.exists():
        raise ValueError(
            f"confirmation_used.lock already exists ({used_lock}); the one-shot unseal "
            "is Task 6 only — the score side (collect-d1 / inference-scoring / "
            "score-freeze) must refuse to run")


def stage_collect_d1(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §6.2: REALLY collect the frozen selected-n accepted D1 groups (JOINT_OOD).

    Precondition: the full lock chain (base + extended registry, incl. the §13 recovery
    chain and the live §2.1 source/seed audit) is re-verified BEFORE any collection, and
    ``confirmation_used.lock`` must not exist. The collector reads the frozen
    selected_n/profile from the lock itself (fail-closed), enforces the candidate quota,
    self-reports a 30 s heartbeat, and resumes/verifies create-once products. Candidate
    exhaustion short of the quota raises ``CollectionIncompleteError`` (exit 4) — never a
    silent shortfall.
    """
    checked = prot.validate_config(config)
    _assert_confirmation_unused(checked)
    run_dir = _resolve(checked["run_dir"])
    lock = prot.verify_protocol_lock(run_dir)
    confirm_prot.verify_confirm_protocol_lock(run_dir)
    selected_n = int(lock["selected_n"])
    _log("collect-d1", f"protocol lock verified (base + extended); "
                       f"selected_n={selected_n} profile={lock['ood_profile']}")
    receipt = collector.collect_role(config, d1_inference.D1_ROLE)
    if receipt.get("partial"):
        raise TerminalState(prot.STATUS_INVALID, "D1 collection incomplete")
    accepted = int(receipt["accepted_groups"])
    if accepted != selected_n:
        raise RuntimeError(
            f"D1 accepted {accepted} groups but the lock froze selected_n={selected_n}")
    action = "skipped_complete" if receipt.get("verification") else "executed"
    _sync_handoff_status(run_dir, prot.STATUS_D1_COLLECTED)
    _log("collect-d1", f"{prot.STATUS_D1_COLLECTED}: accepted={accepted} "
                       f"attempts={receipt['attempts']} role={d1_inference.D1_ROLE}")
    return {"phase": "collect-d1", "action": action, "accepted_groups": accepted,
            "attempts": int(receipt["attempts"]),
            "status": prot.STATUS_D1_COLLECTED}


def stage_inference_scoring(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §6.4 step 1: frozen predictors + 5 Direct seeds + baselines over D1 inputs.

    Truth-blind: reads ONLY ``inference_inputs.npz``; the sealed truth is never opened.
    The cheap extended-code-chain drift check (``verify_confirm_code_chain``) guards the
    code set; the full audit already ran at collect-d1. Writes the create-once
    predictions.npz / risk_scores.npz / inference_receipt.json (a rerun verifies, never
    overwrites). No Gate/AURC/improvement quantity is computed (Task 6 only).
    """
    checked = prot.validate_config(config)
    _assert_confirmation_unused(checked)
    run_dir = _resolve(checked["run_dir"])
    confirm_prot.verify_confirm_code_chain(run_dir)
    receipt = _with_heartbeat(
        "inference-scoring", lambda: d1_inference.infer_and_score_d1(checked))
    action = "skipped_complete" if receipt.get("verification") else "executed"
    _log("inference-scoring", f"frozen inference complete: groups={receipt['groups']} "
                              f"status={receipt.get('status', '-')}")
    return {"phase": "inference-scoring", "action": action,
            "groups": int(receipt["groups"]),
            "status": d1_inference.STATUS_INFERENCE}


def stage_score_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §6.4 steps 2-3: determinism recheck + freeze the score-side artifacts.

    The fixed leading >=32 groups are re-inferred twice and must be elementwise
    identical AND equal the formal artifacts' leading slices (row-count-context
    zero-padded to the formal contexts; no tolerance). Then the create-once
    score_freeze_receipt.json binds every score-side artifact hash. The lock/code chain
    is re-verified inside the freeze; ``confirmation_used.lock`` must not exist.
    """
    checked = prot.validate_config(config)
    _assert_confirmation_unused(checked)
    run_dir = _resolve(checked["run_dir"])
    confirm_prot.verify_confirm_code_chain(run_dir)
    determinism = _with_heartbeat(
        "score-freeze:recheck", lambda: d1_inference.run_determinism_recheck_d1(checked))
    if not determinism.get("bitwise_identical"):
        raise RuntimeError("the determinism recheck did not pass bitwise identity")
    freeze = d1_inference.freeze_scores_d1(checked)
    action = "skipped_complete" if freeze.get("verification") else "executed"
    _sync_handoff_status(run_dir, prot.STATUS_D1_SCORES_FROZEN)
    _log("score-freeze", f"{prot.STATUS_D1_SCORES_FROZEN}: n={freeze['n_groups']} "
                         f"rechecked={determinism.get('n_groups_rechecked', '-')} "
                         "groups bitwise")
    return {"phase": "score-freeze", "action": action,
            "n_groups": int(freeze["n_groups"]),
            "status": prot.STATUS_D1_SCORES_FROZEN}


# --- phases (Task 6 scope: the one-shot unseal + adjudication) --------------------------------


def stage_burn_confirmation_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §6.4 step 5: atomically create ``confirmation_used.lock`` (the one-shot unseal).

    Requires the §13 recovery chain (recovery_lock_r2) to cover the current code set; the
    cheap code chain is re-verified before the burn. A pre-existing lock is re-verified
    against the frozen artifacts (never overwritten) and reported as ``skipped_complete``.
    """
    checked = prot.validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    confirm_prot.verify_confirm_code_chain(run_dir)
    used_lock = _resolve(checked["paths"]["confirmation_used_lock"])
    existed = used_lock.exists()
    lock_path = _with_heartbeat(
        "burn-confirmation-lock", lambda: d0.burn_confirmation_lock_d1(checked))
    action = "skipped_complete" if existed else "executed"
    _log("burn-confirmation-lock",
         f"confirmation_used.lock {'verified' if existed else 'burned'}: {lock_path}")
    return {"phase": "burn-confirmation-lock", "action": action, "lock": str(lock_path)}


def stage_adjudicate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §6.4 steps 6-9 + §7: open the sealed truth ONCE; run the single adjudication.

    Resume rule (§13): if the formal report already exists the adjudication already
    happened — verify the report's terminal binding against the frozen per-group quantities
    and return ``verified_terminal`` WITHOUT re-running burn/assemble/adjudicate. Otherwise
    burn the lock (idempotent), assemble the create-once quantities and run the one
    adjudication at the frozen bootstrap constants. Every non-CONFIRMED §7.3 terminal state
    is absorbing: it is raised as a ``TerminalState`` (never silently passed, never
    re-adjudicated), so Gate-TASK qualification failures never claim the risk method is
    falsified and the pipeline halts at the recorded state.
    """
    checked = prot.validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    paths = d0.d1_confirm_artifact_paths(checked)
    if paths["report"].is_file():
        # §13: a completed transaction is only ever re-verified, never re-run.
        report = _read_json(paths["report"])
        status = str(report.get("status", ""))
        receipt = _read_json(paths["quantities_receipt"])
        expected = receipt.get("confirm_group_quantities_sha256")
        if report.get("bindings", {}).get("confirm_group_quantities_sha256") != expected:
            raise RuntimeError(
                "the formal report no longer binds the frozen per-group quantities")
        if not paths["quantities_npz"].is_file() \
                or sha256_file(paths["quantities_npz"]) != expected:
            raise RuntimeError(
                "the frozen per-group quantities drifted from their create-once receipt")
        _sync_handoff_status(run_dir, status)
        _log("adjudicate", f"verified_terminal: {status} (report already exists; the "
                           "one-shot adjudication is never re-run)")
        if status != prot.STATUS_CONFIRMED:
            raise TerminalState(
                status, "resumed adjudication terminal state (absorbing; §7.3)")
        return {"phase": "adjudicate", "action": "verified_terminal", "status": status}

    confirm_prot.verify_confirm_code_chain(run_dir)
    d0.burn_confirmation_lock_d1(checked)
    d0.assemble_d1_quantities(checked)
    result = _with_heartbeat("adjudicate", lambda: d0.adjudicate_d1(checked))
    status = str(result["status"])
    _sync_handoff_status(run_dir, status)
    _log("adjudicate", f"adjudicated: {status} (n={result.get('n_groups')}); doc-sync "
                       "targets: " + ", ".join(DOC_SYNC_TARGETS))
    if status != prot.STATUS_CONFIRMED:
        raise TerminalState(
            status, "the single D1 adjudication reached a non-confirm terminal state "
                    "(§7.3); a qualification failure never claims the risk method is "
                    "falsified")
    return {"phase": "adjudicate", "action": "executed", "status": status,
            "n_groups": int(result.get("n_groups", -1))}


def stage_reports_status_sync(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan §8/§10 Task 6: confirm the §8.1 reports exist and sync the handoff status.

    Verifies the formal report + the §8.1 tables/figures exist, syncs this stage's own
    HANDOFF_STATE.md status line to the terminal state, and prints the manual/controller
    doc-sync targets (the runner edits ONLY this stage's HANDOFF_STATE.md).
    """
    checked = prot.validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    paths = d0.d1_confirm_artifact_paths(checked)
    if not paths["report"].is_file():
        raise ValueError("the confirmatory report is missing; adjudicate must run first")
    report = _read_json(paths["report"])
    status = str(report.get("status", ""))
    missing = [key for key in ("table_ood_1", "table_ood_2", "table_ood_3",
                               "fig_ood_1", "fig_ood_2", "decision")
               if not paths[key].is_file()]
    if missing:
        raise ValueError(f"missing §8.1 report artifacts: {missing}")
    _sync_handoff_status(run_dir, status)
    _log("reports-status-sync", f"terminal status synced: {status}")
    _log("reports-status-sync", "doc-sync targets: " + ", ".join(DOC_SYNC_TARGETS))
    return {"phase": "reports-status-sync", "action": "executed", "status": status,
            "doc_sync_targets": list(DOC_SYNC_TARGETS)}


# --- phase dispatch --------------------------------------------------------------------------


_PHASE_FUNCTIONS = {
    "preflight": stage_preflight,
    "focused-tests": stage_focused_tests,
    "engineering-smoke": stage_engineering_smoke,
    "full-regression-readiness": stage_full_regression_readiness,
    "collect-pilots": stage_collect_pilots,
    "evaluate-pilots": stage_evaluate_pilots,
    "select-profile": stage_select_profile,
    "power": stage_power,
    "lock-d1": stage_lock_d1,
    "locked-smoke-readiness": stage_locked_smoke_readiness,
    "collect-d1": stage_collect_d1,
    "inference-scoring": stage_inference_scoring,
    "score-freeze": stage_score_freeze,
    "burn-confirmation-lock": stage_burn_confirmation_lock,
    "adjudicate": stage_adjudicate,
    "reports-status-sync": stage_reports_status_sync,
}


def run_phase(phase: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one §12 phase. Every phase in PIPELINE is implemented (never skipped)."""
    phase = str(phase)
    if phase not in PIPELINE:
        raise ValueError(f"unknown Stage 4D phase {phase!r}; expected {list(PIPELINE)}")
    fn = _PHASE_FUNCTIONS.get(phase)
    if fn is None:
        raise NotImplementedError(
            f"phase {phase!r} is declared in the §12 phase list but has no "
            "implementation; it must never be silently skipped")
    return fn(config)


# --- CLI -------------------------------------------------------------------------------------


def _cmd_collect_pilot(config: Mapping[str, Any], profile: str) -> int:
    short = str(profile)
    role = f"pilot_{short}"
    if role not in PILOT_COLLECTION_ORDER:
        raise ValueError(
            f"unknown pilot profile {profile!r}; expected {list(d0.PILOT_ORDER)}")
    receipt = collector.collect_role(config, role)
    _log("collect-pilot",
         f"role={receipt['role']} profile={receipt['profile']} "
         f"status={receipt['status']} accepted={receipt['accepted_groups']} "
         f"attempts={receipt['attempts']} partial={receipt['partial']}")
    return EXIT_OK


def _run_all(config_path: str, config: Mapping[str, Any]) -> int:
    summary = RunSummary(
        config_path=str(config_path), started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        for phase in PIPELINE:
            _log("run-all", f"phase {phase} starting")
            try:
                outcome = run_phase(phase, config)
            except NotImplementedError as exc:
                _log("run-all", f"pipeline boundary: {exc}")
                summary.exit_code = EXIT_PIPELINE_BOUNDARY
                break
            summary.phases.append({
                "phase": phase,
                "action": outcome.get("action", "executed"),
                "detail": {k: v for k, v in outcome.items()
                           if k in ("status", "qualified", "selected_profile",
                                    "full_regression", "verification")},
            })
        else:
            summary.exit_code = EXIT_OK
    except TerminalState as exc:
        summary.terminal_state = exc.status
        summary.exit_code = EXIT_CODES.get(exc.status, EXIT_ERROR)
        _log("run-all", f"TERMINAL {exc.status}: {exc.detail}")
        _log("run-all", "doc-sync targets: " + ", ".join(DOC_SYNC_TARGETS))
    except collector.CollectionIncompleteError as exc:
        summary.terminal_state = collector.STATUS_COLLECTION_INCOMPLETE
        summary.exit_code = EXIT_COLLECTION_INCOMPLETE
        _log("run-all", f"TERMINAL {collector.STATUS_COLLECTION_INCOMPLETE}: {exc}")
    except Exception as exc:
        summary.terminal_state = ""
        summary.exit_code = EXIT_ERROR
        summary.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        _log("run-all", f"FAILURE: {type(exc).__name__}: {exc}")
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), flush=True)
        return summary.exit_code
    summary.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    outcome = "SUCCESS" if summary.exit_code == EXIT_OK else (
        "BOUNDARY" if summary.exit_code == EXIT_PIPELINE_BOUNDARY else "TERMINAL/FAILED")
    _log("run-all", f"{outcome} exit={summary.exit_code} "
                    f"terminal={summary.terminal_state or '-'}")
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), flush=True)
    return summary.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4D D0/D1 Merge model-mismatch OOD unified runner")
    parser.add_argument("command", choices=(
        "preflight", "collect-pilot", "evaluate-pilots", "select-profile",
        "power", "lock-d1", "locked-smoke-readiness", "collect-d1", "infer-d1",
        "freeze-scores", "burn-lock", "adjudicate", "reports-sync", "run-all"))
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--profile", default=None,
                        choices=list(d0.PILOT_ORDER),
                        help="pilot profile for collect-pilot")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    try:
        if args.command == "preflight":
            stage_preflight(config)
            return EXIT_OK
        if args.command == "collect-pilot":
            if args.profile is None:
                parser.error("collect-pilot requires --profile {behavior|geometry|joint}")
            return _cmd_collect_pilot(config, args.profile)
        if args.command == "evaluate-pilots":
            stage_evaluate_pilots(config)
            return EXIT_OK
        if args.command == "select-profile":
            stage_select_profile(config)
            return EXIT_OK
        if args.command == "power":
            stage_power(config)
            return EXIT_OK
        if args.command == "lock-d1":
            stage_lock_d1(config)
            return EXIT_OK
        if args.command == "locked-smoke-readiness":
            stage_locked_smoke_readiness(config)
            return EXIT_OK
        if args.command == "collect-d1":
            stage_collect_d1(config)
            return EXIT_OK
        if args.command == "infer-d1":
            stage_inference_scoring(config)
            return EXIT_OK
        if args.command == "freeze-scores":
            stage_score_freeze(config)
            return EXIT_OK
        if args.command == "burn-lock":
            stage_burn_confirmation_lock(config)
            return EXIT_OK
        if args.command == "adjudicate":
            outcome = stage_adjudicate(config)
            return EXIT_CODES.get(outcome.get("status", ""), EXIT_OK)
        if args.command == "reports-sync":
            stage_reports_status_sync(config)
            return EXIT_OK
        if args.command == "run-all":
            return _run_all(args.config, config)
    except TerminalState as exc:
        _log(args.command, f"TERMINAL {exc.status}: {exc.detail}")
        return EXIT_CODES.get(exc.status, EXIT_ERROR)
    except collector.CollectionIncompleteError as exc:
        _log(args.command, f"TERMINAL {collector.STATUS_COLLECTION_INCOMPLETE}: {exc}")
        return EXIT_COLLECTION_INCOMPLETE
    return EXIT_ERROR


__all__ = [
    "DOC_SYNC_TARGETS",
    "EXIT_CODES",
    "PIPELINE",
    "PILOT_COLLECTION_ORDER",
    "STAGE_TEST_FILES",
    "TerminalState",
    "RunSummary",
    "main",
    "run_phase",
    "stage_adjudicate",
    "stage_burn_confirmation_lock",
    "stage_collect_d1",
    "stage_collect_pilots",
    "stage_engineering_smoke",
    "stage_evaluate_pilots",
    "stage_focused_tests",
    "stage_full_regression_readiness",
    "stage_inference_scoring",
    "stage_lock_d1",
    "stage_locked_smoke_readiness",
    "stage_power",
    "stage_preflight",
    "stage_reports_status_sync",
    "stage_score_freeze",
    "stage_select_profile",
]


if __name__ == "__main__":
    raise SystemExit(main())
