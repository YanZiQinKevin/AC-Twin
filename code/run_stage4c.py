"""Stage 4C orchestration runner.

Plan ``AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1`` section 13 Task 5.
The runner ONLY orchestrates: it wires the collector, the grouped frozen inference,
the objective, the statistics and the power modules. It never re-implements their
science. One-shot resources (the development screen, the confirm set, the decision
receipt, the locks) are create-once: existing ones are never overwritten, and a second
adjudication is refused.

Subcommands: preflight, lock-development, smoke, collect-development,
infer-development, freeze-decision-rule, adjudicate-development, power, lock-confirm,
collect-confirm, infer-confirm, adjudicate-confirm, verify, run-all.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage4c_merge_multiaction as collector  # noqa: E402
from ac_twin import stage4c_inference as s4ci  # noqa: E402
from ac_twin import stage4c_power as s4cpw  # noqa: E402
from ac_twin import stage4c_protocol as s4cp  # noqa: E402
from ac_twin import stage4c_statistics as s4cs  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

ROOT = PROJECT_ROOT
CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c/stage4c_config.json"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _binding(config: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_resolve(config["stage3b_binding"]).read_text(encoding="utf-8"))["receipt_payload"]


def _run_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["run_dir"])


# --- preflight / locks ------------------------------------------------------------------


def cmd_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify frozen sources, model lock, both seed schedules and zero overlap."""
    checked = s4cp.validate_config(config)
    sources = s4cp.inventory_frozen_sources()
    model_lock = s4cp.inventory_model_lock(checked)
    historical = s4cp.scan_historical_group_seeds(
        [_resolve(checked["historical_data_root"]), ROOT / "experiments"])
    dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    confirm = s4cp.make_group_seed_schedule(
        s4cp.CONFIRM_SEED_ENTROPY, s4cp.CONFIRM_SCHEDULE_SIZE)
    from ac_twin.stage2_protocol import validate_seed_schedule
    validate_seed_schedule(dev, historical)
    validate_seed_schedule(confirm, historical)
    if np.intersect1d(dev, confirm).size:
        raise ValueError("dev and confirm schedules overlap")
    return {
        "status": "PREFLIGHT_OK",
        "n_frozen_sources": len(sources["stage4_sha256"]),
        "n_checkpoints": len(model_lock["checkpoint_sha256"]),
        "n_historical_episode_seeds": int(historical.size),
        "dev_schedule_sha256": hashlib.sha256(dev.tobytes()).hexdigest(),
        "confirm_schedule_sha256": hashlib.sha256(confirm.tobytes()).hexdigest(),
        "device": str(checked["device"]),
    }


def cmd_lock_development(config_path: str | Path) -> dict[str, Any]:
    return s4cp.write_development_lock(config_path)


def cmd_lock_confirm(config_path: str | Path, *, selected_n: int) -> dict[str, Any]:
    return s4cp.write_confirm_lock(config_path, selected_n=selected_n)


# --- smoke ------------------------------------------------------------------------------


def cmd_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """12-group engineering smoke; output is marked ENGINEERING_SMOKE_ONLY."""
    checked = s4cp.validate_config(config)
    smoke_dir = _run_dir(checked) / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    report_path = smoke_dir / "smoke_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    # throwaway smoke seeds far from dev/confirm ranges; engineering only. A generous
    # candidate pool (60) absorbs the ~47% group acceptance so 12 groups are reached.
    smoke_seeds = np.arange(2_000_000_000 - 60, 2_000_000_000, dtype=np.int64)
    smoke_config = dict(checked)
    smoke_config["data_output"] = str(smoke_dir / "smoke_dev.npz")
    smoke_config["rejection_ledger"] = str(smoke_dir / "smoke_ledger.jsonl")
    receipt = collector.collect_role(
        "development_selection", 12, smoke_seeds, smoke_config)
    report = {
        "status": "ENGINEERING_SMOKE_ONLY",
        "accepted_groups": receipt["accepted_groups"],
        "attempts": receipt["attempts"],
        "prefix_max_abs_diff": receipt["prefix_max_abs_diff"],
        "note": "engineering smoke only; not a scientific product",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --- collection ---------------------------------------------------------------------------


def cmd_collect_development(config: Mapping[str, Any]) -> dict[str, Any]:
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    lock = json.loads((run_dir / "development" / "protocol_lock.json").read_text())
    dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    if hashlib.sha256(dev.tobytes()).hexdigest() != lock["dev_schedule_sha256"]:
        raise ValueError("dev schedule drifted since lock")
    seg = lock["selection_segment"]
    sel = collector.collect_role(
        "development_selection", s4cp.DEV_SELECTION_GROUPS,
        dev[seg[0]:seg[1]], checked)
    seg = lock["screen_segment"]
    scr = collector.collect_role(
        "development_screen", s4cp.DEV_SCREEN_GROUPS, dev[seg[0]:seg[1]], checked)
    return {"selection": sel, "screen": scr}


def cmd_collect_confirm(config: Mapping[str, Any]) -> dict[str, Any]:
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    lock = json.loads((run_dir / "confirm" / "protocol_lock.json").read_text())
    schedule = np.load(run_dir / "confirm" / "seed_schedule_confirm.npy").astype(np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != lock["confirm_schedule_sha256"]:
        raise ValueError("confirm schedule drifted since lock")
    return collector.collect_role("confirm", int(lock["selected_n"]), schedule, checked)


# --- inference -----------------------------------------------------------------------------


def _infer_role(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    sub = "confirm" if role == "confirm" else "development"
    pred_path = run_dir / sub / f"predictions_{role}.npz"
    risk_path = run_dir / sub / f"risk_scores_{role}.npz"
    receipt_path = run_dir / sub / f"inference_receipt_{role}.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    bundle = s4ci.load_group_eval(checked, role=role)
    binding = _binding(checked)
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    determinism = s4ci.determinism_recheck(
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=checked["device"], batch_size=int(checked["batch_size"]))
    predictions = s4ci.run_grouped_frozen_inference(
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=checked["device"], batch_size=int(checked["batch_size"]))
    features = s4ci.build_action_risk_features(bundle, predictions)
    frozen = s4ci.load_frozen_models()
    action_scores = s4ci.score_actions(
        features, frozen, groups=bundle["groups"], n_actions=bundle["n_actions"],
        rng_seed=int(checked["collection_seed"]))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **{k: v for k, v in predictions.items()})
    np.savez_compressed(risk_path, **action_scores)
    receipt = {
        "plan_id": s4cp.PLAN_ID, "role": role, "groups": int(bundle["groups"]),
        "data_sha256": bundle["data_signature"],
        "predictions_sha256": sha256_file(pred_path),
        "risk_scores_sha256": sha256_file(risk_path),
        "determinism_recheck": determinism,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return receipt


def cmd_infer_development(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selection": _infer_role(config, "development_selection"),
        "screen": _infer_role(config, "development_screen"),
    }


def cmd_infer_confirm(config: Mapping[str, Any]) -> dict[str, Any]:
    return _infer_role(config, "confirm")


def _load_role_products(config: Mapping[str, Any], role: str):
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    sub = "confirm" if role == "confirm" else "development"
    bundle = s4ci.load_group_eval(checked, role=role)
    with np.load(run_dir / sub / f"predictions_{role}.npz") as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(run_dir / sub / f"risk_scores_{role}.npz") as z:
        action_scores = {k: z[k] for k in z.files}
    return bundle, predictions, action_scores


# --- decision-rule freeze (selection only) -------------------------------------------------


def cmd_freeze_decision_rule(config: Mapping[str, Any]) -> dict[str, Any]:
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    receipt_path = run_dir / "development" / "decision_rule_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("decision_rule_receipt already exists (create-once)")
    bundle, predictions, action_scores = _load_role_products(checked, "development_selection")
    # selection may read truth (it is not a one-shot adjudication set).
    groups, n_actions = bundle["groups"], bundle["n_actions"]
    hist_b = np.broadcast_to(bundle["history_last_xy"][:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(bundle["inference_mask"][:, None, :, :],
                             (groups, n_actions) + bundle["inference_mask"].shape[1:])
    from ac_twin import stage4c_objective as s4co
    ens = s4ci.ensemble_predicted_cost(predictions, bundle)
    J_hat = ens["J_hat"]
    comp_true = s4co.trajectory_components(
        np.asarray(bundle["counterfactual_xy"], dtype=np.float64), hist_b, mask_b)
    J_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))
    rows = np.arange(groups)
    a_hat = np.argmin(J_hat, axis=-1)
    a_star = np.argmin(J_true, axis=-1)
    regret = s4co.clamp_regret(J_true[rows, a_hat] - J_true[rows, a_star])
    group_risks = s4ci.aggregate_group_risk(action_scores, J_hat)
    selection_inputs = {
        "baseline_group_risks": {
            "UNCERTAINTY_TOP2_MAX": group_risks["UNCERTAINTY_ONLY_TOP2_MAX"],
            "GEOMETRY_TOP2_MAX": group_risks["GEOMETRY_ONLY_TOP2_MAX"],
            "MARGIN_ONLY": group_risks["MARGIN_ONLY"],
        },
        "regret": regret,
        "J_hat": J_hat,
        "J_true": J_true,
        "combined_action_risk": action_scores["COMBINED_RISK_HEAD"],
        "density_count": bundle["density_count"],
    }
    receipt = s4cs.freeze_decision_rule(selection_inputs)
    receipt["selection_data_sha256"] = bundle["data_signature"]
    receipt["selection_group_seed_sha256"] = hashlib.sha256(
        bundle["episode_seed"].tobytes()).hexdigest()
    write_once_json(receipt_path, receipt)
    return receipt


# --- adjudication -----------------------------------------------------------------------------


def _assemble_and_save_quantities(config, role, decision_rule, lock_name, out_npz, out_receipt):
    """Burn the one-shot lock, then build + persist the create-once group quantities."""
    run_dir = _run_dir(config)
    sub = "confirm" if role == "confirm" else "development"
    lock_path = run_dir / sub / lock_name
    if lock_path.exists():
        raise FileExistsError(f"{lock_name} already burned (single-use)")
    if out_npz.exists():
        raise FileExistsError(f"{out_npz.name} already exists (create-once)")
    bundle, predictions, action_scores = _load_role_products(config, role)
    # burn the one-shot lock BEFORE constructing truth-side quantities.
    lock_path.write_text(json.dumps({
        "plan_id": s4cp.PLAN_ID, "role": role,
        "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }), encoding="utf-8")
    q = s4cs.assemble_group_quantities(bundle, predictions, action_scores, decision_rule)
    persist = {k: v for k, v in q.items() if k not in ("group_risks", "task_point")}
    for name, arr in q["group_risks"].items():
        persist[f"group_risk__{name}"] = arr
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name(out_npz.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **persist)
    Path(tmp).replace(out_npz)
    receipt = {
        "plan_id": s4cp.PLAN_ID, "role": role, "groups": int(bundle["groups"]),
        "quantities_sha256": sha256_file(out_npz),
        "task_point": q["task_point"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_receipt, receipt)
    return q, bundle, receipt


def _gate_q_common(config, role, bundle) -> None:
    """Mechanical Gate Q spot-checks (hashes, completeness, order, one-shot)."""
    checked = s4cp.validate_config(config)
    lock = json.loads(
        (_run_dir(checked) / ("confirm" if role == "confirm" else "development")
         / "protocol_lock.json").read_text())
    inventory = s4cp.inventory_frozen_sources()  # raises on drift
    _ = inventory
    seeds = bundle["episode_seed"]
    if len(np.unique(seeds)) != seeds.size:
        raise ValueError("Gate Q: duplicate group seeds")
    cf = np.load(_resolve(config["data_output"]) if role == "confirm" else
                 _resolve(str(config["data_output"]).replace(
                     ".npz", "_selection.npz" if role == "development_selection" else "_screen.npz")),
                 allow_pickle=True)["cf_actions"]
    for row in np.asarray(cf):
        if [int(a) for a in row] != [a for a, _ in s4cp.ACTION_PLANS]:
            raise ValueError("Gate Q: candidate order mismatch")


def cmd_adjudicate_development(config: Mapping[str, Any]) -> dict[str, Any]:
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    decision_rule = json.loads(
        (run_dir / "development" / "decision_rule_receipt.json").read_text())
    out_npz = run_dir / "development" / "screen_group_quantities.npz"
    out_receipt = run_dir / "development" / "screen_group_quantities_receipt.json"
    report_path = run_dir / "development" / "development_screen_report.json"
    if report_path.exists():
        raise FileExistsError("development screen already adjudicated (single-use)")
    bundle, _, _ = _load_role_products(checked, "development_screen")
    _gate_q_common(checked, "development_screen", bundle)
    q, bundle, _ = _assemble_and_save_quantities(
        checked, "development_screen", decision_rule,
        "development_screen_used.lock", out_npz, out_receipt)

    boot = s4cs.paired_group_bootstrap(
        q, n_resamples=int(checked["bootstrap_resamples"]),
        seed=int(checked["bootstrap_entropy_screen"]),
        confidence=float(checked["confidence_screen"]))
    point, ci_lower = boot["point"], boot["ci_lower"]
    task_inputs = dict(q["task_point"])
    task_inputs.update({
        "spearman_ci_lower": float(ci_lower[s4cpw.Q_SPEARMAN]),
        "base_vs_random_point": float(point[s4cpw.Q_BASE_VS_RANDOM]),
        "base_vs_random_ci_lower": float(ci_lower[s4cpw.Q_BASE_VS_RANDOM]),
    })
    task = s4cs.evaluate_gate_task(task_inputs)
    baseline_valid = bool(np.isfinite(point[s4cpw.Q_I_AURC]))  # NaN => denom <= 0
    risk = s4cs.evaluate_gate_risk_decision(
        point, ci_lower, qa_passed=True, task_status=task["status"],
        baseline_valid=baseline_valid)
    if task["status"] in (s4cp.STATUS_TASK_DEGENERATE, s4cp.STATUS_MODEL_INSUFFICIENT):
        status = task["status"]
    elif risk["status"] == "RISK_DECISION_PASSED":
        status = s4cp.STATUS_DEV_READY
    else:
        status = s4cp.STATUS_DEV_NOT_READY
    report = {
        "plan_id": s4cp.PLAN_ID, "role": "development_screen",
        "confidence": float(checked["confidence_screen"]),
        "point": point.tolist(), "ci_lower": ci_lower.tolist(),
        "gate_task": task, "gate_risk_decision": risk,
        "status": status,
    }
    write_once_json(report_path, report)
    decision = run_dir / "development" / "STAGE4C_DEV_DECISION.md"
    decision.write_text(
        f"# Stage 4C development screen decision\n\nstatus: `{status}`\n",
        encoding="utf-8")
    return report


def cmd_adjudicate_confirm(config: Mapping[str, Any]) -> dict[str, Any]:
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    decision_rule = json.loads(
        (run_dir / "development" / "decision_rule_receipt.json").read_text())
    report_path = run_dir / "confirm" / "stage4c_confirmatory_report.json"
    if report_path.exists():
        raise FileExistsError("confirm already adjudicated (single-use)")
    out_npz = run_dir / "confirm" / "confirm_group_quantities.npz"
    out_receipt = run_dir / "confirm" / "confirm_group_quantities_receipt.json"
    bundle, _, _ = _load_role_products(checked, "confirm")
    _gate_q_common(checked, "confirm", bundle)
    q, bundle, _ = _assemble_and_save_quantities(
        checked, "confirm", decision_rule, "confirmation_used.lock", out_npz, out_receipt)

    boot = s4cs.paired_group_bootstrap(
        q, n_resamples=int(checked["bootstrap_resamples"]),
        seed=int(checked["bootstrap_entropy_confirm"]),
        confidence=float(checked["confidence_confirm"]))
    point, ci_lower = boot["point"], boot["ci_lower"]
    task_inputs = dict(q["task_point"])
    task_inputs.update({
        "spearman_ci_lower": float(ci_lower[s4cpw.Q_SPEARMAN]),
        "base_vs_random_point": float(point[s4cpw.Q_BASE_VS_RANDOM]),
        "base_vs_random_ci_lower": float(ci_lower[s4cpw.Q_BASE_VS_RANDOM]),
    })
    task = s4cs.evaluate_gate_task(task_inputs)
    baseline_valid = bool(np.isfinite(point[s4cpw.Q_I_AURC]))
    risk = s4cs.evaluate_gate_risk_decision(
        point, ci_lower, qa_passed=True, task_status=task["status"],
        baseline_valid=baseline_valid)
    if not risk["risk_passed"] and risk["status"] == "RISK_DECISION_FAILED":
        status = s4cp.STATUS_NOT_CONFIRMED
    elif risk["status"] in (s4cp.STATUS_INVALID, s4cp.STATUS_TASK_DEGENERATE,
                            s4cp.STATUS_MODEL_INSUFFICIENT):
        status = risk["status"]
    elif risk["status"] == "RISK_DECISION_PASSED":
        status = s4cp.STATUS_CONFIRMED
    else:
        status = s4cp.STATUS_NOT_CONFIRMED
    report = {
        "plan_id": s4cp.PLAN_ID, "role": "confirm",
        "confidence": float(checked["confidence_confirm"]),
        "point": point.tolist(), "ci_lower": ci_lower.tolist(),
        "gate_task": task, "gate_risk_decision": risk,
        "status": status,
    }
    write_once_json(report_path, report)
    (run_dir / "confirm" / "STAGE4C_CONFIRMATORY_DECISION.md").write_text(
        f"# Stage 4C confirmatory decision\n\nstatus: `{status}`\n", encoding="utf-8")
    return report


# --- power ---------------------------------------------------------------------------------


def cmd_power(config: Mapping[str, Any]) -> dict[str, Any]:
    checked = s4cp.validate_config(config)
    run_dir = _run_dir(checked)
    npz = run_dir / "development" / "screen_group_quantities.npz"
    inputs = s4cpw.load_screen_group_quantities(npz)
    boot = s4cpw.joint_bootstrap(
        inputs, n_resamples=int(checked["power_bootstrap_resamples"]),
        seed=int(checked["bootstrap_entropy_screen"]) + 7)  # distinct power stream
    selection = s4cpw.select_confirm_n(
        boot["theta_hat"], boot["covariance"],
        candidate_ns=tuple(int(n) for n in checked["candidate_ns"]),
        monte_carlo=int(checked["power_monte_carlo"]),
        seed=int(checked["power_monte_carlo_entropy"]),
        n_reference=int(inputs["regret"].shape[0]),
        shrink_scenarios=tuple(float(s) for s in checked["power_shrink_scenarios"]),
        shrink_main=float(checked["power_shrink_main"]),
        target=float(checked["power_target"]))
    report = {
        "plan_id": s4cp.PLAN_ID,
        "n_screen_groups": int(inputs["regret"].shape[0]),
        "screen_quantities_sha256": inputs["npz_sha256"],
        "theta_hat": boot["theta_hat"].tolist(),
        "selection": selection,
        "selected_n": selection["selected_n"],
        "status": s4cp.STATUS_UNDERPOWERED if selection["underpowered"] else "POWERED",
    }
    write_once_json(run_dir / "power" / "power_report.json", report)
    return report


# --- verify / run-all ------------------------------------------------------------------------


def cmd_verify(config: Mapping[str, Any]) -> dict[str, Any]:
    return cmd_preflight(config)


def cmd_run_all(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    log: dict[str, Any] = {}
    log["preflight"] = cmd_preflight(config)
    log["lock_development"] = {"status": "locked"}
    try:
        s4cp.write_development_lock(config_path)
    except FileExistsError:
        pass
    log["smoke"] = cmd_smoke(config)
    log["collect_development"] = cmd_collect_development(config)
    log["infer_development"] = cmd_infer_development(config)
    try:
        log["freeze_decision_rule"] = cmd_freeze_decision_rule(config)
    except FileExistsError:
        log["freeze_decision_rule"] = {"status": "already_frozen"}
    log["adjudicate_development"] = cmd_adjudicate_development(config)
    if log["adjudicate_development"]["status"] != s4cp.STATUS_DEV_READY:
        log["halted"] = log["adjudicate_development"]["status"]
        return log
    log["power"] = cmd_power(config)
    if log["power"]["status"] == s4cp.STATUS_UNDERPOWERED:
        log["halted"] = s4cp.STATUS_UNDERPOWERED
        return log
    selected_n = int(log["power"]["selected_n"])
    try:
        s4cp.write_confirm_lock(config_path, selected_n=selected_n)
    except FileExistsError:
        pass
    log["collect_confirm"] = cmd_collect_confirm(config)
    log["infer_confirm"] = cmd_infer_confirm(config)
    log["adjudicate_confirm"] = cmd_adjudicate_confirm(config)
    return log


_COMMANDS = (
    "preflight", "lock-development", "smoke", "collect-development",
    "infer-development", "freeze-decision-rule", "adjudicate-development", "power",
    "lock-confirm", "collect-confirm", "infer-confirm", "adjudicate-confirm",
    "verify", "run-all",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4C multi-action runner")
    parser.add_argument("command", choices=_COMMANDS)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--selected-n", type=int, default=None,
                        help="required for lock-confirm")
    args = parser.parse_args(argv)
    config = _load_config(args.config)

    if args.command == "preflight":
        out = cmd_preflight(config)
    elif args.command == "lock-development":
        out = cmd_lock_development(args.config)
    elif args.command == "smoke":
        out = cmd_smoke(config)
    elif args.command == "collect-development":
        out = cmd_collect_development(config)
    elif args.command == "infer-development":
        out = cmd_infer_development(config)
    elif args.command == "freeze-decision-rule":
        out = cmd_freeze_decision_rule(config)
    elif args.command == "adjudicate-development":
        out = cmd_adjudicate_development(config)
    elif args.command == "power":
        out = cmd_power(config)
    elif args.command == "lock-confirm":
        if args.selected_n is None:
            raise ValueError("lock-confirm requires --selected-n")
        out = cmd_lock_confirm(args.config, selected_n=args.selected_n)
    elif args.command == "collect-confirm":
        out = cmd_collect_confirm(config)
    elif args.command == "infer-confirm":
        out = cmd_infer_confirm(config)
    elif args.command == "adjudicate-confirm":
        out = cmd_adjudicate_confirm(config)
    elif args.command == "verify":
        out = cmd_verify(config)
    elif args.command == "run-all":
        out = cmd_run_all(args.config)
    else:  # pragma: no cover
        raise ValueError(args.command)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
