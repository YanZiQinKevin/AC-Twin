"""Stage 4D-D2 Task 3 pipeline driver (plan 12.4): in-platform retraining + validation freeze.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2``. Orchestrates, in order:

    fit-base       train-only scaler -> 5 response predictors -> 10 Full/History experts ->
                   validation calibration -> 5 E2E routers -> train predictions -> 3 risk heads
    infer-train    frozen-stack predictions + 5 risk-class scores + 9x34 Direct tokens (train)
    check-train-support   plan-7.5 train-support hard gate (fail-closed)
    train-five-seeds      the fixed SAME_INFORMATION_DIRECT model (5 seeds, validation early-stop)
    infer-validation      same frozen inference on the 1,500 validation groups
    adjudicate-validation unique baseline selection + plan-7.6 gate + create-once freeze
    run-all        the full sequence; any gate failure is a plan-15 terminal state (no screen)

This driver only orchestrates; all science lives in ``stage4d_external_inference`` (base
stack) and ``stage4d_external_training`` (Direct + gates + freeze). Every stage is create-once
and resumable, and the frozen ``train_model`` loop emits a plain-text heartbeat at least every
30 s (long-running progress contract). The collection itself is a separate long-running entry
(``collect_stage4d_external.py``); this driver requires the train/validation roles present.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin.checkpointing import atomic_save_npz  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
DEV_DIR = RUN_DIR / "development"
BASE_MODEL_DIR = DEV_DIR / "base_models"
ROLES_DIR = RUN_DIR / "roles"

ROLE_QUOTA = {"train": T.DEV_TRAIN_GROUPS, "validation": T.DEV_VALIDATION_GROUPS}

_PREDICTION_KEYS = ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob")


def _role_dir(role: str) -> Path:
    return ROLES_DIR / role


def _check_collected(role: str) -> None:
    npz = _role_dir(role) / "groups.npz"
    meta = _role_dir(role) / "groups_meta.json"
    if not npz.is_file() or not meta.is_file():
        raise SystemExit(f"[stage4d-task3] role {role} not collected: {npz} / {meta}")
    arrays = INF.load_role_arrays(_role_dir(role), role)
    quota = ROLE_QUOTA[role]
    if arrays.groups != quota:
        raise SystemExit(
            f"[stage4d-task3] role {role} has {arrays.groups} groups, expected quota {quota}; "
            "collection must complete before Task 3 uses the role")


def _load_bundle(role: str) -> INF.BranchBundle:
    _check_collected(role)
    return INF.build_branch_bundle(INF.load_role_arrays(_role_dir(role), role))


# --- base-stack retraining -----------------------------------------------------------------------


def _stack_manifest_path() -> Path:
    return BASE_MODEL_DIR / "predictor_stack.json"


def cmd_fit_base(device: str, batch_size: int) -> INF.PredictorStack:
    """Retrain the full base stack on train (+validation early-stop); create-once manifest."""
    manifest = _stack_manifest_path()
    if manifest.is_file():
        stack = INF.PredictorStack.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
        print("[stage4d-task3] base stack manifest present; verified, no retrain", flush=True)
        return stack

    print("[stage4d-task3] loading train + validation roles for base retraining", flush=True)
    train = _load_bundle("train")
    validation = _load_bundle("validation")
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    scaler = INF.fit_scaler(train, BASE_MODEL_DIR / "scaler.json")
    print(f"[stage4d-task3] scaler fitted (train-only) sig={scaler.signature()[:12]}…", flush=True)

    response = INF.fit_response_predictors(train, validation, scaler, BASE_MODEL_DIR,
                                           device=device, progress=_heartbeat("response"))
    experts = INF.fit_experts(train, validation, scaler, BASE_MODEL_DIR,
                              device=device, progress=_heartbeat("expert"))
    calibration = INF.fit_calibration(response, validation, scaler,
                                      BASE_MODEL_DIR / "calibration.json",
                                      device=device, batch_size=batch_size)
    routers = INF.fit_routers(train, validation, scaler, response, experts, calibration,
                              BASE_MODEL_DIR, device=device, batch_size=batch_size,
                              progress=_heartbeat("router"))
    print(f"[stage4d-task3] base predictors trained in {time.perf_counter()-t0:.1f}s",
          flush=True)

    # risk heads fit on the TRAIN predictions (Stage 4A parity: primary frozen on train).
    partial = INF.PredictorStack(
        scaler_path=str(BASE_MODEL_DIR / "scaler.json"), response_paths=response,
        expert_paths=experts, router_paths=routers, calibration=calibration,
        risk_paths={}, primary_predictor="END_TO_END_MIXTURE")
    train_predictions = _infer_predictions(partial, train, "train", device, batch_size)
    risk = INF.fit_risk_heads(train_predictions, train, BASE_MODEL_DIR,
                              progress=_heartbeat("risk"))
    print(f"[stage4d-task3] primary predictor = {risk['primary'].primary_predictor} "
          f"(constant_ade={risk['primary'].train_constant_mean_ade:.6f} "
          f"e2e_ade={risk['primary'].train_e2e_mean_ade:.6f})", flush=True)

    stack = INF.PredictorStack(
        scaler_path=str(BASE_MODEL_DIR / "scaler.json"), response_paths=response,
        expert_paths=experts, router_paths=routers, calibration=calibration,
        risk_paths=risk["paths"], primary_predictor=risk["primary"].primary_predictor)
    payload = stack.to_dict()
    payload["plan_id"] = T.PLAN_ID
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_once_json(manifest, payload)
    print(f"[stage4d-task3] base stack FROZEN -> {manifest}", flush=True)
    return stack


# --- frozen inference (persisted create-once per role) -------------------------------------------


def _infer_predictions(stack: INF.PredictorStack, bundle: INF.BranchBundle, role: str,
                       device: str, batch_size: int) -> dict[str, np.ndarray]:
    pred_path = DEV_DIR / f"{role}_predictions.npz"
    if pred_path.is_file():
        with np.load(pred_path) as z:
            return {k: z[k] for k in z.files}
    t0 = time.perf_counter()
    predictions = INF.run_pair_predictions(stack, bundle, device=device, batch_size=batch_size)
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(pred_path,
                    **{k: np.asarray(predictions[k], dtype=np.float32) for k in _PREDICTION_KEYS},
                    group_seed=bundle.group_seed)
    print(f"[stage4d-task3] {role} predictions ({bundle.groups} groups, "
          f"{bundle.branches} branches) in {time.perf_counter()-t0:.1f}s", flush=True)
    return predictions


def cmd_infer(stack: INF.PredictorStack, role: str, device: str, batch_size: int
              ) -> dict[str, Any]:
    """Frozen-stack predictions + five risk-class scores for one role; create-once npz."""
    score_path = DEV_DIR / f"{role}_scores.npz"
    bundle = _load_bundle(role)
    predictions = _infer_predictions(stack, bundle, role, device, batch_size)
    if score_path.is_file():
        with np.load(score_path) as z:
            action_risks = {k: z[k] for k in z.files}
    else:
        action_risks = INF.score_action_risks(stack, bundle, predictions)
        atomic_save_npz(score_path, group_seed=bundle.group_seed,
                        **{k: np.asarray(v, dtype=np.float64) for k, v in action_risks.items()})
        print(f"[stage4d-task3] {role} risk-class scores frozen -> {score_path}", flush=True)
    return {"bundle": bundle, "predictions": predictions, "action_risks": action_risks}


def assemble_role(stack: INF.PredictorStack, role: str, device: str, batch_size: int
                  ) -> dict[str, Any]:
    """Build the frozen Direct features + truth-side labels for train/validation.

    Truth (J_true, regret, oracle action) is legal for train/validation only — never the
    screen/confirm. Mirrors drive_stage4cr2_task3.assemble_role against the D2 stack.
    """
    if role not in ("train", "validation"):
        raise ValueError(f"Task 3 assembles only train/validation, not {role}")
    out = cmd_infer(stack, role, device, batch_size)
    bundle, predictions, action_risks = out["bundle"], out["predictions"], out["action_risks"]
    direct = INF.build_direct_inputs(bundle, predictions, action_risks)
    labels = T.compute_true_labels(bundle.counterfactual_xy, bundle.history_last_xy,
                                   bundle.inference_mask, direct["j_hat"])
    return {
        "role": role, "groups": bundle.groups, "n_actions": bundle.n_actions,
        "features": direct, "regret": labels["regret"], "y_reg": labels["y_reg"],
        "y_high": labels["y_high"], "oracle_action": labels["oracle_action"],
        "base_mean_regret": labels["base_mean_regret"],
        "idle_mean_regret": labels["idle_mean_regret"],
        "random_mean_regret": labels["random_mean_regret"],
        "group_seed": bundle.group_seed, "data_sha256": bundle.data_sha256,
        "predictions_sha256": sha256_file(DEV_DIR / f"{role}_predictions.npz"),
        "scores_sha256": sha256_file(DEV_DIR / f"{role}_scores.npz"),
    }


# --- gates + Direct training + freeze -------------------------------------------------------------


def _heartbeat(stage: str):
    last = {"at": 0.0}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last["at"] >= 5.0:
            last["at"] = now
            print(f"[stage4d-task3:{stage}] {line}", flush=True)

    return progress


def cmd_check_train_support(stack: INF.PredictorStack, device: str, batch_size: int
                            ) -> dict[str, Any]:
    report_path = DEV_DIR / "train_support_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"[stage4d-task3] train support report present, no re-check: {report['status']}",
              flush=True)
        return report
    assembled = assemble_role(stack, "train", device, batch_size)
    feats = assembled["features"]
    gate_input = {
        "regret": assembled["regret"], "y_reg": assembled["y_reg"],
        "y_high": assembled["y_high"], "tokens": feats["tokens"],
        "global_features": feats["global_features"], "group_seed": assembled["group_seed"],
    }
    report = T.evaluate_train_support_gate(gate_input)
    report["data_sha256"] = assembled["data_sha256"]
    report["predictions_sha256"] = assembled["predictions_sha256"]
    report["scores_sha256"] = assembled["scores_sha256"]
    report["valid_feature_groups"] = int(np.asarray(feats["valid_mask"]).sum())
    report["inference_code_sha256"] = sha256_file(
        PROJECT_ROOT / "code/ac_twin/stage4d_external_inference.py")
    report["training_code_sha256"] = sha256_file(
        PROJECT_ROOT / "code/ac_twin/stage4d_external_training.py")
    report["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_once_json(report_path, report)
    print(f"[stage4d-task3] train-support status={report['status']} "
          f"high_regret={report['values']['high_regret_groups']} "
          f"nonzero={report['values']['nonzero_regret_groups']} "
          f"base={assembled['base_mean_regret']:.6f} idle={assembled['idle_mean_regret']:.6f} "
          f"random={assembled['random_mean_regret']:.6f}", flush=True)
    return report


def cmd_train(stack: INF.PredictorStack, device: str, batch_size: int) -> T.ModelInventory:
    report_path = DEV_DIR / "train_support_report.json"
    if not report_path.is_file():
        raise SystemExit("[stage4d-task3] train support report missing; check-train-support first")
    support = json.loads(report_path.read_text(encoding="utf-8"))
    if support["status"] != T.STATUS_TRAIN_SUPPORT_OK:
        raise SystemExit(f"[stage4d-task3] train support gate failed ({support['status']}); "
                         "terminal, no Direct training")
    assembled = assemble_role(stack, "train", device, batch_size)
    feats = assembled["features"]
    valid = np.asarray(feats["valid_mask"], dtype=bool)
    if not valid.all():
        print(f"[stage4d-task3] WARNING: {int((~valid).sum())} train groups have non-finite "
              "features and are dropped from Direct training (no zero-padding)", flush=True)
    train_bundle = {
        "tokens": feats["tokens"][valid], "global_features": feats["global_features"][valid],
        "y_reg": assembled["y_reg"][valid],
        "y_high": assembled["y_high"][valid].astype(np.float64),
        "group_seed": assembled["group_seed"][valid],
        "interaction_min_distance": feats["interaction_min_distance"][valid],
    }
    # validation role is the early-stopping/checkpoint monitor (plan 7.5/7.6).
    val = assemble_role(stack, "validation", device, batch_size)
    vfeats = val["features"]
    vvalid = np.asarray(vfeats["valid_mask"], dtype=bool)
    validation_bundle = {
        "tokens": vfeats["tokens"][vvalid], "global_features": vfeats["global_features"][vvalid],
        "y_reg": val["y_reg"][vvalid], "y_high": val["y_high"][vvalid].astype(np.float64),
        "group_seed": val["group_seed"][vvalid],
    }
    inventory = T.fit_five_direct_models(train_bundle, validation_bundle, RUN_DIR,
                                         device=device, progress=_heartbeat("direct"))
    print(f"[stage4d-task3] Direct five seeds done param_count={inventory.param_count} "
          f"pos_weight={inventory.pos_weight:.4f} "
          f"interaction_threshold={inventory.interaction_threshold:.6f} "
          f"verification={inventory.verification or 'fresh_fit'}", flush=True)
    return inventory


def cmd_adjudicate_validation(stack: INF.PredictorStack, device: str, batch_size: int
                              ) -> dict[str, Any]:
    gate_report_path = DEV_DIR / "validation_gate_report.json"
    final_path = DEV_DIR / "final_selection.json"
    if gate_report_path.exists() or final_path.exists():
        raise SystemExit("[stage4d-task3] validation adjudication already ran (create-once)")

    inventory = cmd_train(stack, device, batch_size)  # resumes from the fit receipt
    assembled = assemble_role(stack, "validation", device, batch_size)
    feats = assembled["features"]
    valid = np.asarray(feats["valid_mask"], dtype=bool)
    keep = np.where(valid)[0]
    score_bundle = {"tokens": feats["tokens"][keep],
                    "global_features": feats["global_features"][keep],
                    "group_seed": assembled["group_seed"][keep]}
    scores = T.score_models(inventory, score_bundle, device=device)

    baseline_risks = {name: np.asarray(risk, dtype=np.float64)[keep]
                      for name, risk in feats["baselines"].items()}
    validation = {"regret": assembled["regret"][keep], "y_high": assembled["y_high"][keep],
                  "baseline_risks": baseline_risks, "group_seed": assembled["group_seed"][keep]}
    baseline = T.select_strongest_standard_baseline(validation)
    print(f"[stage4d-task3] strongest_standard_baseline={baseline.strongest} "
          f"aurc={ {k: round(v, 6) for k, v in baseline.aurc_regret.items()} }", flush=True)

    support = json.loads((DEV_DIR / "train_support_report.json").read_text(encoding="utf-8"))
    truth = {**validation, "baseline_receipt": baseline,
             "interaction_threshold": inventory.interaction_threshold,
             "bindings": {
                 "train_data_sha256": support["data_sha256"],
                 "train_predictions_sha256": support["predictions_sha256"],
                 "train_scores_sha256": support["scores_sha256"],
                 "validation_data_sha256": assembled["data_sha256"],
                 "validation_predictions_sha256": assembled["predictions_sha256"],
                 "validation_scores_sha256": assembled["scores_sha256"],
             }}
    report = T.evaluate_validation_gate(scores, truth)
    print(f"[stage4d-task3] validation status={report.status} passed={report.passed} "
          f"conditions={report.conditions}", flush=True)
    print(f"[stage4d-task3] i_aurc={report.metrics['i_aurc_regret']:.6f} "
          f"i_r80={report.metrics['i_regret80']:.6f} "
          f"seeds_pos={report.metrics['seeds_i_aurc_positive']}/"
          f"{report.metrics['seeds_i_regret80_positive']}", flush=True)

    if report.passed:
        path = T.freeze_final_selection(report, inventory, DEV_DIR)
        print(f"[stage4d-task3] FROZEN final selection: {path}", flush=True)
    else:
        write_once_json(gate_report_path, report.to_dict())
        print(f"[stage4d-task3] TERMINAL {report.status}; screen is never collected", flush=True)
    return report.to_dict()


def run_all(device: str, batch_size: int) -> int:
    stack = cmd_fit_base(device, batch_size)
    support = cmd_check_train_support(stack, device, batch_size)
    if support["status"] != T.STATUS_TRAIN_SUPPORT_OK:
        print(f"[stage4d-task3] TERMINAL {support['status']}", flush=True)
        return 3
    cmd_train(stack, device, batch_size)
    report = cmd_adjudicate_validation(stack, device, batch_size)
    if not report["passed"]:
        print(f"[stage4d-task3] TERMINAL {report['status']}", flush=True)
        return 4
    print("[stage4d-task3] Task 3 COMPLETE: validation qualified, selection frozen", flush=True)
    return 0


_COMMANDS = ("fit-base", "infer-train", "check-train-support", "train-five-seeds",
             "infer-validation", "adjudicate-validation", "run-all")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4D-D2 Task 3 pipeline driver")
    parser.add_argument("command", choices=_COMMANDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    device = args.device
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    if args.command == "run-all":
        return run_all(device, args.batch_size)
    stack = INF.PredictorStack.from_dict(json.loads(_stack_manifest_path().read_text(
        encoding="utf-8"))) if _stack_manifest_path().is_file() else None
    if args.command == "fit-base":
        cmd_fit_base(device, args.batch_size)
        return 0
    if stack is None:
        raise SystemExit("[stage4d-task3] base stack missing; run fit-base first")
    if args.command == "infer-train":
        cmd_infer(stack, "train", device, args.batch_size)
    elif args.command == "check-train-support":
        cmd_check_train_support(stack, device, args.batch_size)
    elif args.command == "train-five-seeds":
        cmd_train(stack, device, args.batch_size)
    elif args.command == "infer-validation":
        cmd_infer(stack, "validation", device, args.batch_size)
    elif args.command == "adjudicate-validation":
        cmd_adjudicate_validation(stack, device, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
