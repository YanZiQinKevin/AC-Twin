"""Stage 4C-R2 Task 3 pipeline driver: lock -> train -> validation adjudication -> freeze.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 6-8. This
script ONLY orchestrates the frozen modules; it never reimplements their science and it
never touches the screen role (Task 4 scope). The order is fixed:

    lock-development
      -> collect-train        (5,000 groups, resumable atomic checkpoints)
      -> infer-train          (frozen five-pair predictor/risk chain, create-once)
      -> check-train-support  (plan 7.1 hard gate; failure is terminal, STOP)
      -> train-five-seeds     (fixed Direct width 43, five frozen seeds, create-once)
      -> collect-validation   (1,500 groups)
      -> infer-validation
      -> adjudicate-validation (mechanical baseline + the ONE gate evaluation; on pass
                                freeze the five create-once artifacts, else record the
                                terminal state and STOP; screen is never collected here)

The validation adjudication runs AT MOST ONCE: if ``validation_gate_report.json`` (or
``final_selection.json``) already exists, re-adjudication is refused. Long stages print
plain-text progress at least every 30 s.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

import collect_stage4cr2_merge_multiaction as collector  # noqa: E402
from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402  (RISK_CLASSES, read-only)
from ac_twin import stage4cr2_inference as s4cr2i  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_selection as s4cr2s  # noqa: E402
from ac_twin import stage4cr2_training as s4cr2t  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"


def _load_config() -> dict[str, Any]:
    return s4cr2.validate_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value)


def _dev_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["run_dir"]) / "development"


# --- lock ------------------------------------------------------------------------------------


def cmd_lock(config: Mapping[str, Any]) -> Path:
    lock_path = _dev_dir(config) / "protocol_lock.json"
    if lock_path.exists():
        s4cr2.verify_development_lock(lock_path)
        print(f"[stage4cr2-lock] existing lock verified, no overwrite: {lock_path}")
        return lock_path
    inventory = s4cr2.build_lock_inventory()
    path = s4cr2.create_development_lock(config, inventory)
    print(f"[stage4cr2-lock] development lock created: {path}")
    return path


# --- collect / infer --------------------------------------------------------------------------


def cmd_collect(config: Mapping[str, Any], role: str) -> None:
    quota = collector.ROLE_QUOTA[role]
    started = time.perf_counter()
    receipt = collector.collect_role(config, role, quota)
    elapsed = time.perf_counter() - started
    print(f"[stage4cr2-collect] role={role} status={receipt.status} "
          f"accepted={receipt.accepted_groups} attempts={receipt.attempts} "
          f"partial={receipt.partial} elapsed={elapsed:.1f}s", flush=True)
    if receipt.partial or receipt.accepted_groups != quota:
        raise SystemExit(
            f"collection for {role} incomplete ({receipt.accepted_groups}/{quota}); "
            "resume by re-running the same command")


def cmd_infer(config: Mapping[str, Any], role: str) -> None:
    started = time.perf_counter()
    receipt = s4cr2i.infer_role(config, role)
    elapsed = time.perf_counter() - started
    print(f"[stage4cr2-infer] role={role} status={receipt.status} "
          f"groups={receipt.groups} elapsed={elapsed:.1f}s", flush=True)


# --- feature + truth assembly (train/validation truth is legal; NEVER the screen) ------------


def assemble_role(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Build the frozen 9x34-token + 9-global features and the role's truth-side labels.

    Truth (J_true, regret, oracle action) is legal for train/validation — they are not
    one-shot sets. This function is never called for the screen role in Task 3.
    """
    if role not in ("development_train", "development_validation"):
        raise ValueError(f"Task 3 assembles only train/validation, not {role}")
    bundle = s4cr2i.load_role_bundle(config, role)
    pred_path, score_path, _ = s4cr2i.role_inference_paths(config, role)
    with np.load(pred_path) as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(score_path) as z:
        scores = {k: z[k] for k in z.files}
    action_risks = {k: scores[k].astype(np.float64) for k in s4cpp.RISK_CLASSES}
    feats = s4cpf.build_relative_tokens(
        predictions["e2e"].astype(np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], action_risks,
        bundle["geometry_group"], bundle["t_star"], bundle["episode_seed"])

    groups = int(bundle["groups"])
    n_actions = int(bundle["n_actions"])
    true_xy = np.asarray(bundle["counterfactual_xy"], dtype=np.float64)
    hist_b = np.broadcast_to(bundle["history_last_xy"][:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(bundle["inference_mask"][:, None, :, :],
                             (groups, n_actions) + bundle["inference_mask"].shape[1:])
    comp_true = s4co.trajectory_components(true_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                           s4co.efficiency_cost(comp_true["progress"]))
    j_hat = feats["j_hat"]
    rows = np.arange(groups)
    a_hat = np.argmin(j_hat, axis=-1)
    a_star = np.argmin(j_true, axis=-1)
    j_true_star = j_true[rows, a_star]
    regret = s4co.clamp_regret(j_true[rows, a_hat] - j_true_star)
    idle_regret = s4co.clamp_regret(j_true[:, s4co.IDLE_PLAN_INDEX] - j_true_star)
    random_regret = s4co.clamp_regret(j_true.mean(axis=-1) - j_true_star)

    data_output, _, receipt_path = collector.role_paths(config, role)
    return {
        "role": role,
        "groups": groups,
        "n_actions": n_actions,
        "features": feats,
        "regret": regret,
        "oracle_action": a_star,
        "base_mean_regret": float(regret.mean()),
        "random_mean_regret": float(random_regret.mean()),
        "idle_mean_regret": float(idle_regret.mean()),
        "y_reg": np.log1p(regret / s4cr2.HIGH_REGRET_THRESHOLD),
        "y_high": (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64),
        "group_seed": np.asarray(bundle["episode_seed"], dtype=np.int64),
        "data_sha256": sha256_file(data_output),
        "predictions_sha256": sha256_file(pred_path),
        "scores_sha256": sha256_file(score_path),
        "collection_receipt_path": str(receipt_path),
        "inference_receipt_path": str(s4cr2i.role_inference_paths(config, role)[2]),
    }


# --- train support hard gate (plan 7.1) --------------------------------------------------------


def _write_train_support_report(config: Mapping[str, Any], report_path: Path,
                                recovery_note: dict[str, Any] | None = None) -> dict[str, Any]:
    assembled = assemble_role(config, "development_train")
    feats = assembled["features"]
    gate_input = {
        "groups": assembled["groups"],
        "n_actions": assembled["n_actions"],
        "complete_groups": assembled["groups"],  # collector accepts only complete groups
        "regret": assembled["regret"],
        "oracle_action": assembled["oracle_action"],
        "base_mean_regret": assembled["base_mean_regret"],
        "random_mean_regret": assembled["random_mean_regret"],
        "idle_mean_regret": assembled["idle_mean_regret"],
        "tokens": feats["tokens"],
        "global_features": feats["global_features"],
        "group_seed": assembled["group_seed"],
    }
    report = s4cr2t.evaluate_train_support_gate(gate_input, config)
    report["data_sha256"] = assembled["data_sha256"]
    report["predictions_sha256"] = assembled["predictions_sha256"]
    report["scores_sha256"] = assembled["scores_sha256"]
    report["collection_receipt_path"] = assembled["collection_receipt_path"]
    report["inference_receipt_path"] = assembled["inference_receipt_path"]
    report["valid_feature_groups"] = int(np.asarray(feats["valid_mask"]).sum())
    report["training_code_sha256"] = sha256_file(
        PROJECT_ROOT / "code/ac_twin/stage4cr2_training.py")
    if recovery_note is not None:
        report["recovery"] = recovery_note
    report["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_once_json(report_path, report)
    print(f"[stage4cr2-train-support] status={report['status']} "
          f"high_regret={report['values']['high_regret_groups']} "
          f"nonzero={report['values']['nonzero_regret_groups']} "
          f"oracle_covered={report['values']['oracle_actions_covered_ge_5pct']} "
          f"base={report['values']['base_mean_regret']:.6f} "
          f"random={report['values']['random_mean_regret']:.6f} "
          f"idle={report['values']['idle_mean_regret']:.6f} "
          f"report={report_path.name}", flush=True)
    return report


def _latest_train_support_report(dev: Path) -> dict[str, Any] | None:
    """The effective support report: the newest recovery version when present."""
    for name in ("train_support_report_r2.json", "train_support_report_r1.json",
                 "train_support_report.json"):
        path = dev / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def cmd_check_train_support(config: Mapping[str, Any]) -> dict[str, Any]:
    report = _latest_train_support_report(_dev_dir(config))
    if report is not None:
        print(f"[stage4cr2-train-support] existing report verified, no re-check: "
              f"status={report['status']}")
        return report
    return _write_train_support_report(
        config, _dev_dir(config) / "train_support_report.json")


def cmd_check_train_support_recovery(config: Mapping[str, Any]) -> dict[str, Any]:
    """Pure-engineering recovery (plan 6.1): keep the base report, re-run the gate with
    fixed code, write a create-once recovery version. Never overwrites the base report.
    """
    dev = _dev_dir(config)
    base_path = dev / "train_support_report.json"
    if not base_path.exists():
        raise SystemExit("no base train_support_report.json to recover from")
    r1_path = dev / "train_support_report_r1.json"
    if r1_path.exists():
        return json.loads(r1_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    note = {
        "recovered_from": "train_support_report.json",
        "recovered_from_sha256": sha256_file(base_path),
        "recovered_from_status": base["status"],
        "reason": (
            "implementation bug: the plan-7.1 zero-variance check exempts explicit "
            "constant contracts, but the exemption list shipped empty; "
            "visible_vehicle_count is identically 4 by the frozen capacity-5 object "
            "observation contract (verified constant in the R2 train 5,000 and the "
            "historical Stage 4C selection 500). The fix changes no model math, "
            "feature, label, data role, threshold or gate semantics."),
    }
    return _write_train_support_report(config, r1_path, recovery_note=note)


# --- five-seed training ------------------------------------------------------------------------


def cmd_train(config: Mapping[str, Any]) -> Any:
    support = _latest_train_support_report(_dev_dir(config))
    if support is None:
        raise SystemExit("train support report missing; run check-train-support first")
    if support["status"] != s4cr2t.STATUS_TRAIN_SUPPORT_OK:
        raise SystemExit(
            f"train support gate failed ({support['status']}); terminal, no training")
    assembled = assemble_role(config, "development_train")
    feats = assembled["features"]
    valid = np.asarray(feats["valid_mask"], dtype=bool)
    if not valid.all():
        raise SystemExit(
            f"{int((~valid).sum())} train groups have non-finite features; the support "
            "gate must have failed - refusing to train")
    train_bundle = {
        "tokens": feats["tokens"][valid],
        "global_features": feats["global_features"][valid],
        "y_reg": assembled["y_reg"][valid],
        "y_high": assembled["y_high"][valid].astype(np.float64),
        "group_seed": assembled["group_seed"][valid],
        "interaction_min_distance": feats["interaction_min_distance"][valid],
    }

    last_print = {"at": 0.0}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last_print["at"] >= 5.0:
            last_print["at"] = now
            print(f"[stage4cr2-train] {line}", flush=True)

    started = time.perf_counter()
    inventory = s4cr2t.fit_five_direct_models(train_bundle, config, progress=progress)
    elapsed = time.perf_counter() - started
    print(f"[stage4cr2-train] five seeds done in {elapsed:.1f}s "
          f"param_count={inventory.param_count} pos_weight={inventory.pos_weight:.4f} "
          f"interaction_threshold={inventory.interaction_threshold:.6f} "
          f"verification={inventory.verification or 'fresh_fit'}", flush=True)
    for seed in inventory.seeds:
        print(f"[stage4cr2-train]   seed={seed} "
              f"train_loss={inventory.train_losses[seed]:.6f} "
              f"monitor_loss={inventory.monitor_losses[seed]:.6f}", flush=True)
    return inventory


# --- one-shot validation adjudication ------------------------------------------------------------


def cmd_adjudicate_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    dev = _dev_dir(config)
    gate_report_path = dev / "validation_gate_report.json"
    final_path = dev / "final_selection.json"
    if gate_report_path.exists() or final_path.exists():
        raise SystemExit(
            "validation adjudication already ran (create-once); a second adjudication "
            "is refused")

    inventory = cmd_train(config)  # resumes from the fit receipt; never retrains
    assembled = assemble_role(config, "development_validation")
    feats = assembled["features"]
    valid = np.asarray(feats["valid_mask"], dtype=bool)
    if not valid.all():
        print(f"[stage4cr2-validation] WARNING: {int((~valid).sum())} validation groups "
              "have non-finite features and are dropped from scoring (no zero-padding)",
              flush=True)
    keep = np.where(valid)[0]
    score_bundle = {
        "tokens": feats["tokens"][keep],
        "global_features": feats["global_features"][keep],
        "group_seed": assembled["group_seed"][keep],
    }
    scores = s4cr2t.score_models(inventory, score_bundle,
                                 device=str(config.get("device", "cuda")))

    regret = assembled["regret"][keep]
    y_high = assembled["y_high"][keep]
    baseline_risks = {name: np.asarray(risk, dtype=np.float64)[keep]
                      for name, risk in feats["baselines"].items()}
    validation = {
        "regret": regret,
        "y_high": y_high,
        "baseline_risks": baseline_risks,
        "group_seed": assembled["group_seed"][keep],
    }
    baseline = s4cr2s.select_strongest_standard_baseline(validation)
    print(f"[stage4cr2-validation] strongest_standard_baseline={baseline.strongest} "
          f"aurc={ {k: round(v, 6) for k, v in baseline.aurc_regret.items()} }",
          flush=True)

    support = _latest_train_support_report(dev)
    if support is None:
        raise SystemExit("train support report missing; run check-train-support first")
    truth = {
        **validation,
        "baseline_receipt": baseline,
        "interaction_min_distance": feats["interaction_min_distance"][keep],
        "interaction_threshold": inventory.interaction_threshold,
        "bindings": {
            "train_data_sha256": support["data_sha256"],
            "train_predictions_sha256": support["predictions_sha256"],
            "train_scores_sha256": support["scores_sha256"],
            "validation_data_sha256": assembled["data_sha256"],
            "validation_predictions_sha256": assembled["predictions_sha256"],
            "validation_scores_sha256": assembled["scores_sha256"],
            "train_collection_receipt": support.get("collection_receipt_path", ""),
            "validation_collection_receipt": assembled["collection_receipt_path"],
            "validation_inference_receipt": assembled["inference_receipt_path"],
        },
    }
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    print(f"[stage4cr2-validation] status={report.status} passed={report.passed}",
          flush=True)
    print(f"[stage4cr2-validation] conditions={report.conditions}", flush=True)
    print(f"[stage4cr2-validation] i_aurc={report.metrics['i_aurc_regret']:.6f} "
          f"i_r80={report.metrics['i_regret80']:.6f} "
          f"pr_ens={report.metrics['pr_auc_minus_prevalence_ensemble']:.6f} "
          f"pr_base={report.metrics['pr_auc_minus_prevalence_baseline']:.6f} "
          f"spearman_margin={report.metrics['spearman_ensemble_margin']:.6f} "
          f"strata={report.stratum_fractions}", flush=True)

    if report.passed:
        path = s4cr2s.freeze_final_selection(report, inventory)
        print(f"[stage4cr2-validation] FROZEN final selection: {path}", flush=True)
    else:
        # terminal: record the exact gate report (create-once) and STOP; never screen.
        write_once_json(gate_report_path, report.to_dict())
        print(f"[stage4cr2-validation] TERMINAL {report.status}; screen is never "
              "collected; the multi-action decision-risk line is closed", flush=True)
    return report.to_dict()


# --- run-all -------------------------------------------------------------------------------------


def run_all() -> int:
    config = _load_config()
    cmd_lock(config)
    cmd_collect(config, "development_train")
    cmd_infer(config, "development_train")
    support = cmd_check_train_support(config)
    if support["status"] != s4cr2t.STATUS_TRAIN_SUPPORT_OK:
        print(f"[stage4cr2] TERMINAL {support['status']}", flush=True)
        return 3
    cmd_train(config)
    cmd_collect(config, "development_validation")
    cmd_infer(config, "development_validation")
    report = cmd_adjudicate_validation(config)
    if not report["passed"]:
        print(f"[stage4cr2] TERMINAL {report['status']}", flush=True)
        return 4
    print("[stage4cr2] Task 3 pipeline COMPLETE: validation qualified, selection frozen",
          flush=True)
    return 0


_COMMANDS = (
    "lock-development", "collect-train", "infer-train", "check-train-support",
    "check-train-support-recovery", "train-five-seeds", "collect-validation",
    "infer-validation", "adjudicate-validation", "run-all",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4C-R2 Task 3 pipeline driver")
    parser.add_argument("command", choices=_COMMANDS)
    args = parser.parse_args(argv)
    config = _load_config()
    if args.command == "lock-development":
        cmd_lock(config)
    elif args.command == "collect-train":
        cmd_collect(config, "development_train")
    elif args.command == "infer-train":
        cmd_infer(config, "development_train")
    elif args.command == "check-train-support":
        cmd_check_train_support(config)
    elif args.command == "check-train-support-recovery":
        cmd_check_train_support_recovery(config)
    elif args.command == "train-five-seeds":
        cmd_train(config)
    elif args.command == "collect-validation":
        cmd_collect(config, "development_validation")
    elif args.command == "infer-validation":
        cmd_infer(config, "development_validation")
    elif args.command == "adjudicate-validation":
        report = cmd_adjudicate_validation(config)
        return 0 if report["passed"] else 4
    elif args.command == "run-all":
        return run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
