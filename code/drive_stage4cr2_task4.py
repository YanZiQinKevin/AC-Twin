"""Stage 4C-R2 Task 4 pipeline driver: the ONE-SHOT 8,000-group development screen.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 6.2, 9, 11.
This script ONLY orchestrates the frozen modules; it never reimplements their science.
Strict order (plan 9.3 + 6.2):

    collect-screen          (8,000 groups, resumable atomic checkpoints; legal ONLY
                             because the validation gate passed and final_selection.json
                             is frozen)
    recheck-screen          (32-group bitwise repeated inference, plan 9.3 step 2)
    infer-screen            (frozen predictor/risk chain -> screen_predictions.npz,
                             screen_scores.npz, inference receipt; create-once)
    freeze-screen-scores    (five frozen Direct checkpoints -> screen_direct_scores.npz
                             + receipt, with the Direct-side 32-group bitwise recheck)
    burn-screen-lock        (development_screen_used.lock, create-once)
    adjudicate-screen       (ONLY NOW read truth -> regret -> ONE 10,000-draw whole-group
                             paired bootstrap (entropy 2026092702, two-sided 80% CI) ->
                             Gate TASK-R2 + Gate RISK-DECISION-R2_dev exactly once ->
                             create-once quantities + receipt + report + terminal state)

After ANY screen failure the terminal state is STAGE4CR2_DECISION_RISK_LINE_CLOSED (or
the plan-11 earlier-priority state) and the line STOPS: no edits, no reruns, no
ablation substitution. A second adjudication is refused at every layer. Long stages
print plain-text progress at least every 30 s.
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
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage4cr2_merge_multiaction as collector  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402  (RISK_CLASSES, read-only)
from ac_twin import stage4cr2_inference as s4cr2i  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_statistics as s4cr2st  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"

EXIT_OK = 0
EXIT_INVALID = 6
EXIT_TASK_DEGENERATE = 7
EXIT_MODEL_INSUFFICIENT = 8
EXIT_LINE_CLOSED = 5

_TERMINAL_EXIT = {
    s4cr2.STATUS_DEV_READY: EXIT_OK,
    s4cr2.STATUS_INVALID: EXIT_INVALID,
    s4cr2.STATUS_TASK_DEGENERATE: EXIT_TASK_DEGENERATE,
    s4cr2.STATUS_MODEL_INSUFFICIENT: EXIT_MODEL_INSUFFICIENT,
    s4cr2.STATUS_LINE_CLOSED: EXIT_LINE_CLOSED,
}


def _load_config() -> dict[str, Any]:
    return s4cr2.validate_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


# --- guards ------------------------------------------------------------------------------------


def require_frozen_selection(config: Mapping[str, Any]) -> None:
    """The screen exists ONLY because validation passed and the selection is frozen."""
    paths = s4cr2st.screen_artifact_paths(config)
    lock = paths["dev_dir"] / "protocol_lock.json"
    if not lock.is_file():
        raise ValueError("the development protocol lock is missing; run lock-development")
    final = paths["final_selection"]
    if not final.is_file():
        raise ValueError(
            "screen collection is refused: final_selection.json is missing — the "
            "validation gate must pass and freeze before the screen is touched")
    status = json.loads(final.read_text(encoding="utf-8")).get("status")
    if status != "VALIDATION_QUALIFIED":
        raise ValueError(
            f"screen collection is refused: validation status {status!r}; the "
            "multi-action decision-risk line is closed")


def guard_screen_not_adjudicated(config: Mapping[str, Any]) -> None:
    """Refuse any rerun/continue once the one-shot screen report exists (plan 14.12)."""
    report = s4cr2st.screen_artifact_paths(config)["report"]
    if report.exists():
        terminal = json.loads(report.read_text(encoding="utf-8")).get(
            "terminal_state", "")
        raise ValueError(
            f"the one-shot screen was already adjudicated (terminal_state={terminal}); "
            "any rerun, edit or re-adjudication is refused — after a failure the "
            "multi-action decision-risk line is permanently closed")


# --- collect / recheck / infer -------------------------------------------------------------------


def cmd_collect_screen(config: Mapping[str, Any]) -> None:
    require_frozen_selection(config)
    guard_screen_not_adjudicated(config)
    started = time.perf_counter()
    receipt = collector.collect_role(
        config, "development_screen", collector.ROLE_QUOTA["development_screen"])
    elapsed = time.perf_counter() - started
    print(f"[stage4cr2-screen] collect status={receipt.status} "
          f"accepted={receipt.accepted_groups} attempts={receipt.attempts} "
          f"partial={receipt.partial} elapsed={elapsed:.1f}s", flush=True)
    if receipt.partial or receipt.accepted_groups != s4cr2.DEV_SCREEN_GROUPS:
        raise SystemExit(
            f"screen collection incomplete ({receipt.accepted_groups}/"
            f"{s4cr2.DEV_SCREEN_GROUPS}); resume by re-running the same command")


def cmd_recheck_screen(config: Mapping[str, Any]) -> None:
    """Plan 9.3 step 2: 32-group repeated inference, elementwise identical required."""
    require_frozen_selection(config)
    guard_screen_not_adjudicated(config)
    paths = s4cr2st.screen_artifact_paths(config)
    receipt_path = paths["determinism_receipt"]
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not receipt.get("bitwise_identical"):
            raise SystemExit("the recorded 32-group recheck failed; halt before the lock")
        print("[stage4cr2-screen] existing determinism receipt verified, no rerun",
              flush=True)
        return
    n = int(config.get("determinism_checks", 32))
    started = time.perf_counter()
    receipt = s4cr2i.verify_repeated_inference(
        "development_screen", np.arange(n, dtype=np.int64), config=config)
    write_once_json(receipt_path, receipt.to_dict())
    print(f"[stage4cr2-screen] determinism recheck bitwise_identical="
          f"{receipt.bitwise_identical} groups={receipt.n_groups} "
          f"elapsed={time.perf_counter() - started:.1f}s", flush=True)


def cmd_infer_screen(config: Mapping[str, Any]) -> None:
    require_frozen_selection(config)
    guard_screen_not_adjudicated(config)
    started = time.perf_counter()
    receipt = s4cr2i.infer_role(config, "development_screen")
    print(f"[stage4cr2-screen] infer status={receipt.status} groups={receipt.groups} "
          f"elapsed={time.perf_counter() - started:.1f}s "
          f"verification={receipt.verification or 'fresh'}", flush=True)


def _build_screen_features(config: Mapping[str, Any]) -> dict[str, Any]:
    """Score-side features from the frozen predictions/scores (NO truth is read)."""
    bundle = s4cr2i.load_role_bundle(config, "development_screen")
    pred_path, score_path, _ = s4cr2i.role_inference_paths(config, "development_screen")
    with np.load(pred_path) as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(score_path) as z:
        scores = {k: z[k] for k in z.files}
    action_risks = {k: scores[k].astype(np.float64) for k in s4cpp.RISK_CLASSES}
    feats = s4cpf.build_relative_tokens(
        predictions["e2e"].astype(np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], action_risks,
        bundle["geometry_group"], bundle["t_star"], bundle["episode_seed"])
    feats["t_star"] = np.asarray(bundle["t_star"], dtype=np.int64)
    return feats


def cmd_freeze_screen_scores(config: Mapping[str, Any]) -> None:
    require_frozen_selection(config)
    guard_screen_not_adjudicated(config)
    started = time.perf_counter()
    features = _build_screen_features(config)
    path = s4cr2st.freeze_screen_direct_scores(
        config, features=features, device=str(config.get("device", "cuda")))
    print(f"[stage4cr2-screen] Direct score freeze -> {path.name} "
          f"elapsed={time.perf_counter() - started:.1f}s", flush=True)


def cmd_burn_lock(config: Mapping[str, Any]) -> None:
    require_frozen_selection(config)
    guard_screen_not_adjudicated(config)
    lock = s4cr2st.burn_screen_lock(config)
    print(f"[stage4cr2-screen] one-shot lock burned (create-once): {lock.name}",
          flush=True)


# --- the single adjudication ---------------------------------------------------------------------


def _bootstrap_with_heartbeat(quantities: Mapping[str, Any]):
    """Run the ONE paired bootstrap while printing a <=30 s heartbeat."""
    result: dict[str, Any] = {}
    started = time.perf_counter()

    def work() -> None:
        try:
            result["bootstrap"] = s4cr2st.paired_group_bootstrap(
                quantities, seed=s4cr2.BOOTSTRAP_ENTROPY,
                n_resamples=s4cr2.BOOTSTRAP_RESAMPLES)
        except BaseException as exc:  # propagate, never swallow
            result["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=25.0)
        if thread.is_alive():
            print(f"[stage4cr2-screen] bootstrap running "
                  f"({s4cr2.BOOTSTRAP_RESAMPLES} whole-group paired resamples, entropy "
                  f"{s4cr2.BOOTSTRAP_ENTROPY}) elapsed={time.perf_counter() - started:.0f}s",
                  flush=True)
    if "error" in result:
        raise result["error"]
    return result["bootstrap"], time.perf_counter() - started


def cmd_adjudicate_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ONE screen adjudication: truth -> bootstrap -> both gates -> report."""
    require_frozen_selection(config)
    guard_screen_not_adjudicated(config)
    paths = s4cr2st.screen_artifact_paths(config)
    started = time.perf_counter()
    try:
        npz_path = s4cr2st.assemble_screen_group_quantities(config)
    except Exception as exc:  # Gate-Q structural failure -> INVALID, recorded once
        report = {
            "plan_id": s4cr2.PLAN_ID,
            "terminal_state": s4cr2.STATUS_INVALID,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_once_json(paths["report"], report)
        print(f"[stage4cr2-screen] TERMINAL {s4cr2.STATUS_INVALID} at assembly: {exc}",
              flush=True)
        return report
    print(f"[stage4cr2-screen] truth assembled create-once -> {npz_path.name} "
          f"({time.perf_counter() - started:.1f}s)", flush=True)

    quantities = s4cr2st.load_screen_group_quantities(
        npz_path, paths["quantities_receipt"])
    bootstrap, boot_elapsed = _bootstrap_with_heartbeat(quantities)
    print(f"[stage4cr2-screen] paired bootstrap done in {boot_elapsed:.1f}s", flush=True)
    decision = s4cr2st.evaluate_screen_gates(quantities, bootstrap)

    receipt = json.loads(paths["quantities_receipt"].read_text(encoding="utf-8"))
    report = {
        **decision.to_dict(),
        "terminal_state": decision.status,
        "bindings": receipt["bindings"],
        "lock_sha256": receipt["lock_sha256"],
        "train_support_binding": receipt["train_support_binding"],
        "screen_group_quantities_sha256": receipt["screen_group_quantities_sha256"],
        "screen_report_semantics": (
            "one-shot development adjudication; the multi-action decision-risk line "
            "closes permanently on any failure; no confirm resource is created here"),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["report"], report)
    task = decision.task
    risk = decision.risk
    print(f"[stage4cr2-screen] Gate TASK-R2 status={task['status']} "
          f"conditions={task['conditions']}", flush=True)
    print(f"[stage4cr2-screen] Gate RISK-DECISION-R2_dev conditions={risk['conditions']}",
          flush=True)
    print(f"[stage4cr2-screen] I_AURC={decision.metrics['point']['i_aurc_regret']:.6f} "
          f"(CI80 lower {decision.metrics['ci_lower']['i_aurc_regret']:.6f}) "
          f"I_R80={decision.metrics['point']['i_regret80']:.6f} "
          f"(CI80 lower {decision.metrics['ci_lower']['i_regret80']:.6f}) "
          f"seeds {risk['seeds_i_aurc_positive']}/5+ {risk['seeds_i_regret80_positive']}/5+",
          flush=True)
    print(f"[stage4cr2-screen] TERMINAL STATE: {decision.status}", flush=True)
    return report


def _exit_code(report: Mapping[str, Any]) -> int:
    return _TERMINAL_EXIT.get(str(report.get("terminal_state", "")), EXIT_INVALID)


# --- run-all -------------------------------------------------------------------------------------


def run_all() -> int:
    config = _load_config()
    report_path = s4cr2st.screen_artifact_paths(config)["report"]
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"[stage4cr2-screen] already adjudicated: terminal_state="
              f"{report.get('terminal_state')}; a rerun is refused", flush=True)
        return _exit_code(report)
    cmd_collect_screen(config)
    cmd_recheck_screen(config)
    cmd_infer_screen(config)
    cmd_freeze_screen_scores(config)
    cmd_burn_lock(config)
    report = cmd_adjudicate_screen(config)
    return _exit_code(report)


_COMMANDS = (
    "collect-screen", "recheck-screen", "infer-screen", "freeze-screen-scores",
    "burn-screen-lock", "adjudicate-screen", "run-all",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4C-R2 Task 4 one-shot screen driver")
    parser.add_argument("command", choices=_COMMANDS)
    args = parser.parse_args(argv)
    config = _load_config()
    if args.command == "collect-screen":
        cmd_collect_screen(config)
    elif args.command == "recheck-screen":
        cmd_recheck_screen(config)
    elif args.command == "infer-screen":
        cmd_infer_screen(config)
    elif args.command == "freeze-screen-scores":
        cmd_freeze_screen_scores(config)
    elif args.command == "burn-screen-lock":
        cmd_burn_lock(config)
    elif args.command == "adjudicate-screen":
        return _exit_code(cmd_adjudicate_screen(config))
    elif args.command == "run-all":
        return run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
