"""Stage 4D-D2E stage driver: fit-ego phase (plan section 3, bounded candidates).

Plan ``AC-TWIN-RFER-STAGE4D-D2E-EGO-CONSISTENT-EXTERNAL-2026-09-24-V0.1``. This driver trains
the NEW portable ego effect branch on the OLD D2 train role (adaptive-development-only) and
early-stops / checkpoint-selects on the OLD D2 validation role, for the two bounded candidates
x five index-paired seeds. It writes ONLY under ``experiments/stage4_reliability/stage4d_d2e/``;
the old ``stage4d_external/`` data and models are read-only. No new screen/confirm data is
touched here (that chain starts only after the §3.3 validation gate and the §4.2 power screen).

Subcommands:
    fit-ego     train the 2 ego candidates x 5 seeds (GPU), write checkpoints + receipt.

Progress: terminal-safe plain-text heartbeat (phase, done/total, %, elapsed, ETA) at least once
per 30 s, plus a final success/failure summary (plan section 5). Checkpoints + receipt are
create-once; a rerun reloads and verifies them rather than overwriting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2e_ego as ego  # noqa: E402
from ac_twin import stage4d_d2e_pipeline as PIPE  # noqa: E402
from ac_twin import stage4d_d2e_power as POW  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

D2E_DIR = ego.D2E_DIR
D2_ROLES_DIR = INF.ROLES_DIR                    # old D2 roles (read-only)
DEV_DIR = D2E_DIR / "development"
EGO_MODEL_DIR = DEV_DIR / "ego_models"

_HEARTBEAT_S = 30.0


class _Heartbeat:
    """Terminal-safe progress: phase, done/total, %, elapsed, ETA; >=1 line / 30 s."""

    def __init__(self, phase: str, total: int) -> None:
        self.phase = phase
        self.total = max(int(total), 1)
        self.done = 0
        self.t0 = time.time()
        self._last = 0.0

    def _emit(self, extra: str = "") -> None:
        elapsed = time.time() - self.t0
        frac = self.done / self.total
        eta = elapsed * (1.0 - frac) / frac if frac > 0 else float("nan")
        pct = 100.0 * frac
        print(f"[d2e-fit-ego] phase={self.phase} {self.done}/{self.total} ({pct:.1f}%) "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s {extra}", flush=True)

    def tick(self, extra: str = "") -> None:
        now = time.time()
        if now - self._last >= _HEARTBEAT_S:
            self._last = now
            self._emit(extra)

    def step(self, extra: str = "") -> None:
        self.done += 1
        self.tick(extra)

    def force(self, extra: str = "") -> None:
        self._emit(extra)


def _load_bundle(role: str) -> INF.BranchBundle:
    arrays = INF.load_role_arrays(D2_ROLES_DIR / role, role)
    return INF.build_branch_bundle(arrays)


def _save_checkpoint(path: Path, result: ego.EgoTrainResult) -> None:
    if path.exists():
        return  # create-once: a rerun keeps the first checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    payload = {
        "state_dict": result.state_dict,
        "candidate": result.candidate,
        "seed": result.seed,
        "input_dim": ego.INPUT_DIM,
        "hidden": ego.EGO_HIDDEN_WIDTH,
        "output_dim": ego.EGO_OUTPUT_DIM,
        "rank_scale": result.rank_scale,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
    }
    torch.save(payload, path)


def cmd_fit_ego(device: str) -> int:
    receipt_path = EGO_MODEL_DIR / "ego_fit_receipt.json"
    if receipt_path.exists():
        print(f"[d2e-fit-ego] receipt exists (create-once): {receipt_path}; no rerun",
              flush=True)
        return 0

    t_start = time.time()
    print("[d2e-fit-ego] phase=load train+validation bundles (old D2, adaptive-dev-only)",
          flush=True)
    train_bundle = _load_bundle("train")
    val_bundle = _load_bundle("validation")
    train_arrays = INF.load_role_arrays(D2_ROLES_DIR / "train", "train")
    val_arrays = INF.load_role_arrays(D2_ROLES_DIR / "validation", "validation")

    print("[d2e-fit-ego] phase=fit train-only scaler + assemble features/masks/targets",
          flush=True)
    scaler = INF.fit_scaler(train_bundle, EGO_MODEL_DIR / "ego_scaler.json")
    x_tr = ego.assemble_ego_features(train_bundle, scaler)
    x_va = ego.assemble_ego_features(val_bundle, scaler)
    y_tr = ego.ego_targets(train_bundle)
    y_va = ego.ego_targets(val_bundle)
    m_tr = ego.compute_ego_train_mask(train_arrays)
    m_va = ego.compute_ego_train_mask(val_arrays)

    rank_scale = ego.rank_label_scale(y_tr, m_tr)   # train-only label scale
    print(f"[d2e-fit-ego] train E={x_tr.shape[0]} val E={x_va.shape[0]} "
          f"input_dim={x_tr.shape[1]} rank_scale={rank_scale:.6f} "
          f"param_count={ego.count_ego_parameters()}", flush=True)

    total_models = len(ego.CANDIDATES) * ego.N_SEEDS
    hb = _Heartbeat("train", total_models)
    summaries: list[dict] = []
    for candidate in ego.CANDIDATES:
        for seed in ego.EGO_SEEDS:
            result = ego.train_ego_candidate(
                candidate, seed, x_tr, y_tr, m_tr, x_va, y_va, m_va,
                device=device,
                rank_scale=(rank_scale if candidate == ego.CANDIDATE_EGO_TRAJECTORY_PLUS_RANK
                            else None),
                progress=lambda _msg: hb.tick())
            ckpt = EGO_MODEL_DIR / f"ego_{candidate}_seed{seed}.pt"
            _save_checkpoint(ckpt, result)
            summaries.append({
                "candidate": candidate, "seed": seed,
                "best_val_loss": result.best_val_loss, "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run, "train_seconds": result.train_seconds,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(ckpt),
            })
            hb.step(f"{candidate}/seed{seed} best_val={result.best_val_loss:.6f}")
    hb.force("all candidates trained")

    # per-candidate selection summary (the single final head is chosen in the validate phase,
    # on old validation, per plan 3.3; here we only record the training evidence).
    report = {
        "plan_id": ego.PLAN_ID,
        "status": "D2E_EGO_FIT_COMPLETE",
        "device": device,
        "n_train_branches": int(x_tr.shape[0]),
        "n_validation_branches": int(x_va.shape[0]),
        "input_dim": ego.INPUT_DIM,
        "param_count": ego.count_ego_parameters(),
        "rank_scale": rank_scale,
        "candidates": list(ego.CANDIDATES),
        "seeds": list(ego.EGO_SEEDS),
        "models": summaries,
        "elapsed_seconds": time.time() - t_start,
    }
    report_path = EGO_MODEL_DIR / "ego_fit_report.json"
    write_once_json(report_path, report)

    receipt = {
        "plan_id": ego.PLAN_ID,
        "artifact": "ego_fit",
        "create_once": True,
        "scaler_path": str((EGO_MODEL_DIR / "ego_scaler.json").relative_to(ROOT)),
        "scaler_sha256": sha256_file(EGO_MODEL_DIR / "ego_scaler.json"),
        "feature_layout": {
            "input_dim": ego.INPUT_DIM,
            "history": [0, ego.HISTORY_FLAT],
            "factual_suffix": [ego.HISTORY_FLAT, ego.HISTORY_FLAT + ego.SUFFIX_FLAT],
            "factual_plan_onehot": [ego.SLICE_FACT_PLAN.start, ego.SLICE_FACT_PLAN.stop],
            "counterfactual_plan_onehot": [ego.SLICE_CF_PLAN.start, ego.SLICE_CF_PLAN.stop],
            "t_star_norm_index": ego.IDX_T_STAR,
            "duration_norm_index": ego.IDX_DURATION,
            "t_star_norm": "(t_star - 10.0) / 5.0",
            "duration_norm": "(duration - 1.0) / 2.0",
            "scaler_channels": "motion (2,3,4,5) only; binary/unit-circle/one-hot unscaled",
        },
        "architecture": "Linear(1002,128)->LayerNorm->ReLU->Linear(128,128)->LayerNorm->ReLU"
                        "->Linear(128,40)->reshape(20,2)",
        "optimizer": {"name": "AdamW", "lr": ego.EGO_LEARNING_RATE,
                      "weight_decay": ego.EGO_WEIGHT_DECAY},
        "train_batch_groups": ego.EGO_TRAIN_BATCH_GROUPS,
        "max_epochs": ego.EGO_MAX_EPOCHS,
        "early_stop_patience": ego.EGO_EARLY_STOP_PATIENCE,
        "rank_aux_weight": ego.EGO_RANK_AUX_WEIGHT,
        "candidates": list(ego.CANDIDATES),
        "seeds": list(ego.EGO_SEEDS),
        "rank_scale": rank_scale,
        "report_path": str(report_path.relative_to(ROOT)),
        "data_roles": {"train": "adaptive_development_only",
                       "validation": "early_stop_and_checkpoint_only"},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    print(f"[d2e-fit-ego] SUMMARY status=D2E_EGO_FIT_COMPLETE models={total_models} "
          f"elapsed={time.time() - t_start:.0f}s", flush=True)
    return 0


def cmd_fit_risk(device: str, batch_size: int) -> int:
    """Retrain the frozen Direct head on each candidate's repaired predictions (plan 3.2)."""
    t_start = time.time()
    print("[d2e-fit-risk] phase=load bundles + frozen base predictions (read-only)", flush=True)
    val_preds, val_risks = PIPE.load_persisted_base("validation")
    train_preds, train_risks = PIPE.load_persisted_base("train")
    val_bundle = PIPE.load_bundle("validation", val_preds["group_seed"])
    train_bundle = PIPE.load_bundle("train", train_preds["group_seed"])
    scaler = PIPE.load_ego_scaler()

    hb = _Heartbeat("fit-risk", len(ego.CANDIDATES))
    report_models = {}
    for candidate in ego.CANDIDATES:
        models = PIPE.load_ego_models(candidate)
        train_e2e = PIPE.candidate_e2e(train_bundle, train_preds["e2e"], scaler, models,
                                       device=device, batch_size=batch_size)
        val_e2e = PIPE.candidate_e2e(val_bundle, val_preds["e2e"], scaler, models,
                                     device=device, batch_size=batch_size)
        train_inputs = PIPE.build_direct_role_inputs(train_bundle, train_e2e, train_risks)
        val_inputs = PIPE.build_direct_role_inputs(val_bundle, val_e2e, val_risks)
        inventory, run_dir = PIPE.fit_risk_inventory(
            candidate, train_inputs, val_inputs, device=device,
            progress=lambda _m: hb.tick())
        report_models[candidate] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "param_count": inventory.param_count,
            "n_train_groups": inventory.n_train_groups,
            "n_validation_groups": inventory.n_validation_groups,
            "train_high_regret_frac": float((train_inputs["regret"] > 0.01).mean()),
            "train_base_mean_regret": float(train_inputs["base_mean_regret"]),
            "validation_losses": {str(k): float(v)
                                  for k, v in inventory.validation_losses.items()},
        }
        hb.step(f"{candidate} Direct refit (5 seeds) param={inventory.param_count}")
    hb.force("both candidates refit")
    report = {
        "plan_id": ego.PLAN_ID, "status": "D2E_RISK_FIT_COMPLETE",
        "device": device, "candidates": report_models,
        "note": ("Direct head retrained on repaired train predictions; the three standard "
                 "baselines share the identical repaired J_hat; risk features unchanged by the "
                 "slot-0 repair (ego is masked out of the inference_mask risk features)."),
        "elapsed_seconds": time.time() - t_start,
    }
    write_once_json(DEV_DIR / "fit_risk_report.json", report)
    print(f"[d2e-fit-risk] SUMMARY status=D2E_RISK_FIT_COMPLETE "
          f"candidates={len(ego.CANDIDATES)} elapsed={time.time() - t_start:.0f}s", flush=True)
    return 0


def cmd_validate(device: str, batch_size: int) -> int:
    """The §3.3 old-validation development qualification gate (adaptive-dev, pre-collection).

    Evaluates every variant (old D2 / kinematic floor / both candidates / truth oracle) on the
    same table, scores each candidate's retrained Direct head against the three standard
    baselines, applies ALL §3.3 continue conditions, and mechanically selects the single final
    ego head. Any failure -> terminal ``D2E_INTERFACE_REPAIR_NOT_QUALIFIED`` (no new screen).
    """
    from ac_twin import stage4cp_statistics as s4cs

    report_path = DEV_DIR / "validation_report.json"
    if report_path.exists():
        print(f"[d2e-validate] report exists (create-once): {report_path}; no rerun", flush=True)
        return 0
    t_start = time.time()
    print("[d2e-validate] phase=load validation bundle + frozen base predictions", flush=True)
    val_preds, val_risks = PIPE.load_persisted_base("validation")
    val_bundle = PIPE.load_bundle("validation", val_preds["group_seed"])
    val_arrays = INF.load_role_arrays(INF.ROLES_DIR / "validation", "validation")
    ego_mask = ego.compute_ego_train_mask(val_arrays)   # M_ego: ego-ADE diagnostic only
    scaler = PIPE.load_ego_scaler()
    models_by_candidate = {c: PIPE.load_ego_models(c) for c in ego.CANDIDATES}

    variants = [PIPE.VARIANT_OLD_D2, PIPE.VARIANT_KINEMATIC, *ego.CANDIDATES,
                PIPE.VARIANT_ORACLE]
    hb = _Heartbeat("validate-variants", len(variants))
    metrics: dict[str, dict] = {}
    cand_inputs: dict[str, dict] = {}
    for name in variants:
        e2e_v = PIPE.variant_e2e(name, val_bundle, val_preds["e2e"], scaler,
                                 models_by_candidate, device=device, batch_size=batch_size)
        inputs = PIPE.build_direct_role_inputs(val_bundle, e2e_v, val_risks)
        eq = PIPE.ego_quality(val_bundle, e2e_v, ego_mask)
        j_hat = inputs["j_hat"].ravel()
        j_true = inputs["j_true"].ravel()
        branch_spearman = float(np.corrcoef(
            np.argsort(np.argsort(j_hat)), np.argsort(np.argsort(j_true)))[0, 1])
        metrics[name] = {
            "a0_mean_regret": float(inputs["base_mean_regret"]),
            "idle_mean_regret": float(inputs["idle_mean_regret"]),
            "random_mean_regret": float(inputs["random_mean_regret"]),
            "branch_spearman_j_hat_j_true": branch_spearman,
            **{k: float(v) for k, v in eq.items()},
        }
        if name in ego.CANDIDATES:
            cand_inputs[name] = inputs
        hb.step(name)
    hb.force("variants scored")

    print("[d2e-validate] phase=score retrained Direct + apply §3.3 gate per candidate",
          flush=True)
    old = metrics[PIPE.VARIANT_OLD_D2]
    candidate_eval: dict[str, dict] = {}
    for candidate in ego.CANDIDATES:
        inventory = PIPE.load_risk_inventory(candidate)
        inp = cand_inputs[candidate]
        risk = T.score_models(
            inventory, {"tokens": inp["tokens"], "global_features": inp["global_features"],
                        "group_seed": inp["group_seed"]}, device=device)
        baseline_receipt = T.select_strongest_standard_baseline(
            {"regret": inp["regret"], "y_high": inp["y_high"],
             "baseline_risks": inp["baselines"]})
        gate = T.evaluate_validation_gate(
            risk, {"regret": inp["regret"], "y_high": inp["y_high"],
                   "baseline_risks": inp["baselines"], "baseline_receipt": baseline_receipt,
                   "group_seed": inp["group_seed"],
                   "interaction_threshold": inventory.interaction_threshold})
        direct_pr = float(s4cs.high_regret_pr(
            np.asarray(risk.risk_ensemble, dtype=np.float64), inp["y_high"]))
        cand = metrics[candidate]
        conditions = {
            # condition 1: ego final-progress MAE AND progress ranking both beat old D2 (strictly)
            "c1_ego_mae_better": bool(
                old["ego_final_progress_mae"] - cand["ego_final_progress_mae"] > 0.0),
            "c1_ego_rank_better": bool(
                cand["ego_progress_rank_spearman"] - old["ego_progress_rank_spearman"] > 0.0),
            # condition 2: a0 mean regret <= ALWAYS_IDLE
            "c2_a0_le_idle": bool(cand["a0_mean_regret"]
                                  <= cand["idle_mean_regret"] + 1e-12),
            # condition 3: improvement over RANDOM_ACTION strictly positive
            "c3_beats_random": bool(
                cand["random_mean_regret"] - cand["a0_mean_regret"] > 0.0),
            # condition 4: non-ego effect ADE not worse than old D2 by > 5%
            "c4_non_ego_ade": bool(
                cand["non_ego_effect_ade"]
                <= old["non_ego_effect_ade"] * (1.0 + 0.05)),
            # condition 5: Direct vs frozen strongest standard baseline (the D2 validation gate)
            "c5_direct_risk_gate": bool(gate.passed),
        }
        candidate_eval[candidate] = {
            "conditions": conditions,
            "passed": bool(all(conditions.values())),
            "strongest_baseline": baseline_receipt.strongest,
            "gate_metrics": gate.metrics,
            "direct_pr_auc_minus_prevalence": direct_pr,
            "baseline_aurc_regret": {k: float(v)
                                     for k, v in baseline_receipt.aurc_regret.items()},
            "baseline_regret80": {k: float(v)
                                  for k, v in baseline_receipt.regret_at_80.items()},
            "baseline_pr_auc": {k: float(v)
                                for k, v in baseline_receipt.pr_auc_minus_prevalence.items()},
        }
        print(f"[d2e-validate] {candidate}: passed={candidate_eval[candidate]['passed']} "
              f"conditions={conditions}", flush=True)

    # mechanical selection among passing candidates (lowest a0 regret; tie -> EGO_TRAJECTORY)
    passing = [c for c in ego.CANDIDATES if candidate_eval[c]["passed"]]
    selected = None
    if passing:
        selected = passing[0]
        for c in passing[1:]:
            if (metrics[c]["a0_mean_regret"]
                    < metrics[selected]["a0_mean_regret"] - ego.SELECTION_TIE_TOLERANCE):
                selected = c
    qualified = selected is not None
    status = ("D2E_INTERFACE_REPAIR_QUALIFIED" if qualified
              else "D2E_INTERFACE_REPAIR_NOT_QUALIFIED")

    report = {
        "plan_id": ego.PLAN_ID,
        "status": status,
        "adaptive_development_only": True,
        "note": ("§3.3 pre-collection development stop-loss on old validation (NOT independent "
                 "confirmation); validation was adaptively viewed by D2/D2R."),
        "variants": metrics,
        "candidate_gate": candidate_eval,
        "selected_candidate": selected,
        "selection_rule": "lowest validation a0 mean regret among passing candidates; "
                          "tie <=1e-12 -> EGO_TRAJECTORY",
        "tie_tolerance": ego.SELECTION_TIE_TOLERANCE,
        "elapsed_seconds": time.time() - t_start,
    }
    write_once_json(report_path, report)

    print(f"[d2e-validate] SUMMARY status={status} selected={selected} "
          f"elapsed={time.time() - t_start:.0f}s", flush=True)
    for name in variants:
        m = metrics[name]
        print(f"  {name:26s} a0={m['a0_mean_regret']:.5f} idle={m['idle_mean_regret']:.5f} "
              f"rand={m['random_mean_regret']:.5f} egoMAE={m['ego_final_progress_mae']:.4f} "
              f"rankSp={m['ego_progress_rank_spearman']:.4f}", flush=True)

    if not qualified:
        print("[d2e-validate] TERMINAL D2E_INTERFACE_REPAIR_NOT_QUALIFIED: no new screen; "
              "report which condition failed; do NOT switch to the oracle or D2R fallback.",
              flush=True)
        return 5
    # freeze the final selection (create-once)
    sel = {
        "plan_id": ego.PLAN_ID,
        "selected_candidate": selected,
        "ego_seeds": list(ego.EGO_SEEDS),
        "ego_checkpoints": {
            str(s): f"development/ego_models/ego_{selected}_seed{s}.pt" for s in ego.EGO_SEEDS},
        "risk_run_dir": f"riskfit_{selected}",
        "ensemble_rule": "mean over the 5 index-paired seeds; slot-0 replacement only",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(DEV_DIR / "final_selection.json", sel)
    print(f"[d2e-validate] QUALIFIED: selected={selected}; final_selection.json frozen; "
          f"next = §4.2 screen power pre-check", flush=True)
    return 0


def cmd_power_screen(device: str, batch_size: int) -> int:
    """The §4.2 screen power pre-check (cheap Go/No-Go on old validation; NO collection).

    Estimates the joint power of the complete D2 screen AND-chain under the repaired pipeline and
    selects the smallest candidate n whose 75%-shrink joint power reaches 0.80. n=2500 still short
    -> terminal ``D2E_SCREEN_UNDERPOWERED`` (no collection/manifest/lock is created).
    """
    t_start = time.time()

    def progress(msg: str) -> None:
        print(f"[d2e-power-screen] elapsed={time.time() - t_start:.0f}s {msg}", flush=True)

    report = POW.run_power(device=device, batch_size=batch_size, progress=progress)
    print(f"[d2e-power-screen] SUMMARY status={report['status']} "
          f"selected_n={report['selected_n']} underpowered={report['underpowered']} "
          f"elapsed={time.time() - t_start:.0f}s", flush=True)
    for n, entry in sorted((int(k), v) for k, v in report["per_n"].items()):
        print(f"  n={n:5d} joint_power@75%={entry['joint_power_at_selection']:.4f} "
              f"binding={entry['binding_condition']} ({entry['binding_power']:.4f})", flush=True)
    if report["underpowered"]:
        print("[d2e-power-screen] TERMINAL D2E_SCREEN_UNDERPOWERED: n=2500 75%-shrink joint "
              "power < 0.80; NO collection/manifest/lock is created (plan 4.2).", flush=True)
        return 5
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4D-D2E stage driver")
    parser.add_argument("command", choices=["fit-ego", "fit-risk", "validate", "power-screen"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    if args.command == "fit-ego":
        return cmd_fit_ego(args.device)
    if args.command == "fit-risk":
        return cmd_fit_risk(args.device, args.batch_size)
    if args.command == "validate":
        return cmd_validate(args.device, args.batch_size)
    if args.command == "power-screen":
        return cmd_power_screen(args.device, args.batch_size)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
