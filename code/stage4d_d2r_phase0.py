"""Stage 4D-D2R Phase 0: read-only diagnostics on old train/validation + cheap-stop Go/No-Go.

Plan ``AC-TWIN-RFER-STAGE4D-D2R-RISK-GATED-IDLE-2026-09-24-V0.1`` section 4. This driver is
READ-ONLY against the frozen D2 artifacts: it rebuilds ``J_hat`` / ``J_true`` / ``a0`` /
``a_star`` / ``regret(a0)`` / ``regret(IDLE)`` and the five-seed Direct + three standard risk
scores on the OLD train and validation roles, reusing the frozen predictions
(``development/{role}_predictions.npz``), the frozen action-risk scores
(``development/{role}_scores.npz``), the frozen Direct checkpoints and ``stage4c_objective.py``.
It never re-runs the torch predictor stack, never touches the old 4,000-group screen, and never
accepts/rejects a group on any outcome.

The old validation role is adaptively used (D2 early stopping / baseline selection), so every
Phase 0 number is ``ADAPTIVE_DEVELOPMENT_ONLY`` — a development cost screen, never an unbiased
confirmation. The Go/No-Go gate (plan 4) decides whether the single fixed idle-fallback rule has
enough signal to justify a fresh, non-overlapping development screen; it is NOT a method
confirmation. Any gate failure is the terminal ``D2R_POLICY_SIGNAL_NOT_FEASIBLE``.

Outputs (create-once, under ``experiments/stage4_reliability/stage4d_d2r/phase0/``):
``diagnostic.npz`` (per-group arrays), ``report.json`` (diagnostics 1-5 + gate + bindings),
``final_policy.json`` (the frozen thresholds / strongest fallback / per-seed taus the screen
will reuse verbatim).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4d_d2r_policy as P  # noqa: E402
from ac_twin import stage4d_d2r_statistics as STAT  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402
from ac_twin.checkpointing import atomic_save_npz  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

D2_RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
D2_DEV_DIR = D2_RUN_DIR / "development"
D2_ROLES_DIR = D2_RUN_DIR / "roles"
D2R_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_d2r"
PHASE0_DIR = D2R_DIR / "phase0"

BASELINE_KEYS = P.FALLBACK_BASELINES  # (MARGIN_ONLY, SEED_VOTE_ENTROPY, SEED_MARGIN_STD)


def _heartbeat(stage: str):
    last = {"at": 0.0}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - last["at"] >= 5.0:
            last["at"] = now
            print(f"[stage4d-d2r-phase0:{stage}] {line}", flush=True)

    return progress


# --- read-only assembly of one role ----------------------------------------------------------


def _load_stack() -> INF.PredictorStack:
    manifest = D2_DEV_DIR / "base_models" / "predictor_stack.json"
    if not manifest.is_file():
        raise SystemExit(f"[stage4d-d2r-phase0] D2 predictor stack missing: {manifest}")
    return INF.PredictorStack.from_dict(json.loads(manifest.read_text(encoding="utf-8")))


def _load_inventory() -> T.ModelInventory:
    """Reload the frozen five-seed Direct inventory via its create-once fit receipt (read-only)."""
    receipt = D2_DEV_DIR / "models" / T.FIT_RECEIPT_NAME
    if not receipt.is_file():
        raise SystemExit(f"[stage4d-d2r-phase0] D2 Direct fit receipt missing: {receipt}")
    return T.fit_five_direct_models({}, {}, D2_RUN_DIR, device="cpu")


def assemble_role_quantities(stack: INF.PredictorStack, inventory: T.ModelInventory,
                             role: str, *, device: str = "cpu", with_risk: bool = True,
                             decompose: bool = True) -> dict[str, Any]:
    """Rebuild per-group quantities for one OLD role (train/validation), valid-groups only.

    Reuses the frozen predictions (e2e) and frozen action-risk scores; the truth side is legal
    for train/validation only. Returns per-group arrays aligned to the VALID subset (groups with
    finite Direct features). ``with_risk`` adds the five-seed Direct + three standard risk scores
    (needed only where thresholds/policies are frozen — validation); ``decompose`` adds the
    predicted-vs-true interaction/efficiency cost split (plan-4 item 3, validation only). The
    train role is descriptive-only (plan-4 item 1), so it skips both.
    """
    if role not in ("train", "validation"):
        raise ValueError(f"Phase 0 assembles only old train/validation, not {role}")
    role_dir = D2_ROLES_DIR / role
    bundle = INF.build_branch_bundle(INF.load_role_arrays(role_dir, role))

    pred_path = D2_DEV_DIR / f"{role}_predictions.npz"
    score_path = D2_DEV_DIR / f"{role}_scores.npz"
    if not pred_path.is_file() or not score_path.is_file():
        raise SystemExit(f"[stage4d-d2r-phase0] frozen {role} predictions/scores missing")
    with np.load(pred_path, allow_pickle=False) as z:
        e2e = np.asarray(z["e2e"], dtype=np.float32)          # (S,E,T,N,2) predicted effect
        pred_seed = np.asarray(z["group_seed"], dtype=np.int64).ravel()
    with np.load(score_path, allow_pickle=False) as z:
        action_risks = {k: np.asarray(z[k], dtype=np.float64)
                        for k in z.files if k != "group_seed"}
        score_seed = np.asarray(z["group_seed"], dtype=np.int64).ravel()
    if not (np.array_equal(pred_seed, bundle.group_seed)
            and np.array_equal(score_seed, bundle.group_seed)):
        raise SystemExit(f"[stage4d-d2r-phase0] {role} group_seed misaligned across artifacts")

    direct = INF.build_direct_inputs(bundle, {"e2e": e2e}, action_risks)
    labels = T.compute_true_labels(bundle.counterfactual_xy, bundle.history_last_xy,
                                   bundle.inference_mask, direct["j_hat"])

    valid = np.asarray(direct["valid_mask"], dtype=bool)
    keep = np.where(valid)[0]
    groups, n_actions = bundle.groups, bundle.n_actions

    def _v(arr):
        return np.asarray(arr, dtype=np.float64)[keep]

    out: dict[str, Any] = {
        "role": role,
        "n_total": int(groups),
        "n_valid": int(keep.size),
        "scenario_ids": tuple(str(bundle.scenario_ids[i]) for i in keep),
        "group_seed": _v(direct["group_seed"]).astype(np.int64),
        "j_hat": _v(direct["j_hat"]),
        "j_true": _v(labels["j_true"]),
        "a0": _v(labels["a_hat"]).astype(np.int64),
        "a_star": _v(labels["a_star"]).astype(np.int64),
        "regret_a0": _v(labels["regret"]),
        "regret_idle": _v(labels["idle_regret"]),
        "regret_random": _v(labels["random_regret"]),
        "y_high": _v(labels["y_high"]).astype(np.int64),
        "t_star": np.asarray(bundle.t_star_group, dtype=np.int64)[keep],
        "interaction_min_distance": _v(direct["interaction_min_distance"]),
        "data_sha256": bundle.data_sha256,
        "predictions_sha256": sha256_file(pred_path),
        "scores_sha256": sha256_file(score_path),
    }

    if with_risk:
        scores = T.score_models(
            inventory,
            {"tokens": np.asarray(direct["tokens"])[keep],
             "global_features": np.asarray(direct["global_features"])[keep],
             "group_seed": np.asarray(direct["group_seed"])[keep]},
            device=device)
        out["risk_direct_ensemble"] = np.asarray(scores.risk_ensemble, dtype=np.float64)
        out["risk_direct_by_seed"] = np.asarray(scores.risk_by_seed, dtype=np.float64)
        out["baseline_risks"] = {name: _v(direct["baselines"][name]) for name in BASELINE_KEYS}

    if decompose:
        e2e_g = e2e.reshape((e2e.shape[0], groups, n_actions) + e2e.shape[2:])
        y_pred = bundle.factual_xy[:, None] + e2e_g.mean(axis=0)   # (G,A,T,N,2)
        hist_b = np.broadcast_to(bundle.history_last_xy[:, None, :], (groups, n_actions, 2))
        mask_b = np.broadcast_to(bundle.inference_mask[:, None, :, :],
                                 (groups, n_actions) + bundle.inference_mask.shape[1:])
        comp_pred = s4co.trajectory_components(y_pred, hist_b, mask_b)
        comp_true = s4co.trajectory_components(bundle.counterfactual_xy, hist_b, mask_b)
        out["interaction_pred"] = _v(comp_pred["interaction"])
        out["efficiency_pred"] = _v(s4co.efficiency_cost(comp_pred["progress"]))
        out["interaction_true"] = _v(comp_true["interaction"])
        out["efficiency_true"] = _v(s4co.efficiency_cost(comp_true["progress"]))

    return out


# --- diagnostics --------------------------------------------------------------------------------


def _paired_mean_ci(diff: np.ndarray, *, seed: int, n_resamples: int = 2000) -> dict[str, float]:
    """Paired two-sided 80% CI of a per-group mean difference (development only)."""
    diff = np.asarray(diff, dtype=np.float64).ravel()
    boot = STAT.paired_bootstrap(lambda idx: np.asarray([diff[idx].mean()]), diff.size,
                                 seed=seed, n_resamples=n_resamples)
    return {"point": float(boot.point[0]), "ci80_lower": float(boot.ci_lower[0]),
            "ci80_upper": float(boot.ci_upper[0])}


def _action_summary(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.int64).ravel()
    return {"counts": {str(a): int((arr == a).sum()) for a in range(P.N_ACTIONS)},
            "shares": {str(a): float((arr == a).mean()) for a in range(P.N_ACTIONS)}}


def _build_policy_arrays(val: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze tau_D + the standard fallbacks on validation; build per-group policy regrets."""
    j_true = val["j_true"]
    a_star = val["a_star"]
    a0 = val["a0"]
    risk = val["risk_direct_ensemble"]
    ids = val["scenario_ids"]

    tau_d = P.freeze_threshold(risk, ids)
    d2r = P.evaluate_policy(j_true, a_star, a0, risk, tau_d)
    fallback = P.select_strongest_standard_fallback(
        j_true, a_star, a0, val["baseline_risks"], ids)
    strongest = fallback["strongest"]
    strongest_regret = fallback["per_baseline"][strongest]["regret"]

    high = risk > tau_d
    diff = val["regret_a0"] - val["regret_idle"]          # >0 means base worse than idle
    n_high = int(high.sum())
    n_low = int((~high).sum())
    diff_high = float(diff[high].mean()) if n_high else float("nan")
    diff_low = float((-diff)[~high].mean()) if n_low else float("nan")

    return {
        "tau_d2r": float(tau_d),
        "chosen_d2r": d2r["chosen_action"],
        "regret_d2r": d2r["regret"],
        "regret_idle": val["regret_idle"],
        "regret_base": val["regret_a0"],
        "regret_strongest_fallback": strongest_regret,
        "a0": a0,
        "risk_d2r": risk,
        "n_high_zone": n_high,
        "n_low_zone": n_low,
        "diff_high_zone": diff_high,
        "diff_low_zone": diff_low,
        "strongest_fallback": strongest,
        "fallback_detail": fallback,
        "d2r_fractions": {k: d2r[k] for k in
                          ("frac_above_threshold", "frac_changed_action", "frac_final_non_idle")},
    }


def run_phase0(device: str = "cpu") -> dict[str, Any]:
    """Execute plan-4 Phase 0 end to end; write the three create-once artifacts; return the report."""
    PHASE0_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PHASE0_DIR / "report.json"
    if report_path.exists():
        print("[stage4d-d2r-phase0] report.json already exists (create-once); no rerun",
              flush=True)
        return json.loads(report_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    stack = _load_stack()
    inventory = _load_inventory()
    print(f"[stage4d-d2r-phase0] stack + frozen Direct inventory loaded "
          f"(seeds={list(inventory.seeds)}, param_count={inventory.param_count})", flush=True)

    val = assemble_role_quantities(stack, inventory, "validation", device=device,
                                   with_risk=True, decompose=True)
    print(f"[stage4d-d2r-phase0] validation assembled: {val['n_valid']}/{val['n_total']} valid "
          f"groups (elapsed={time.perf_counter()-t0:.1f}s)", flush=True)
    train = assemble_role_quantities(stack, inventory, "train", device=device,
                                     with_risk=False, decompose=False)
    print(f"[stage4d-d2r-phase0] train assembled: {train['n_valid']}/{train['n_total']} valid "
          f"groups (elapsed={time.perf_counter()-t0:.1f}s)", flush=True)

    pol = _build_policy_arrays(val)
    print(f"[stage4d-d2r-phase0] tau_D={pol['tau_d2r']:.6f} "
          f"strongest_fallback={pol['strongest_fallback']} "
          f"H={pol['n_high_zone']} L={pol['n_low_zone']} "
          f"diff_H={pol['diff_high_zone']:.6f} diff_L={pol['diff_low_zone']:.6f}", flush=True)

    # --- plan-4 item 5: fixed-seed 2000x paired bootstrap + Go/No-Go gate ----------------------
    boot = STAT.paired_bootstrap(
        lambda idx: STAT.compute_phase0_quantity_vector(idx, pol), val["n_valid"],
        seed=STAT.PHASE0_BOOTSTRAP_SEED, n_resamples=STAT.PHASE0_RESAMPLES)
    gate = STAT.evaluate_phase0_gate(pol, boot)
    print(f"[stage4d-d2r-phase0] Go/No-Go status={gate['status']} passed={gate['passed']} "
          f"conditions={gate['conditions']} (elapsed={time.perf_counter()-t0:.1f}s)", flush=True)

    # --- plan-4 items 1-4 diagnostics -----------------------------------------------------------
    diag_regret = {
        "train": {"base": float(train["regret_a0"].mean()),
                  "idle": float(train["regret_idle"].mean()),
                  "random": float(train["regret_random"].mean())},
        "validation": {"base": float(val["regret_a0"].mean()),
                       "idle": float(val["regret_idle"].mean()),
                       "random": float(val["regret_random"].mean())},
        "validation_paired_base_minus_idle":
            _paired_mean_ci(val["regret_a0"] - val["regret_idle"],
                            seed=STAT.PHASE0_BOOTSTRAP_SEED),
    }
    # item 2: a0/oracle frequencies + confusion + t_star / distance strata regret diffs.
    tstar_split = 13
    dist_threshold = float(inventory.interaction_threshold)
    tstar = val["t_star"]
    dist = val["interaction_min_distance"]
    base_minus_idle = val["regret_a0"] - val["regret_idle"]
    item2 = {
        "a0_action": _action_summary(val["a0"]),
        "oracle_action": _action_summary(val["a_star"]),
        "a0_equals_oracle_frac": float((val["a0"] == val["a_star"]).mean()),
        "a0_equals_idle_frac": float((val["a0"] == P.IDLE_PLAN_INDEX).mean()),
        "strata": {
            "tstar_low": float(base_minus_idle[tstar < tstar_split].mean()),
            "tstar_high": float(base_minus_idle[tstar >= tstar_split].mean()),
            "distance_low": float(base_minus_idle[dist <= dist_threshold].mean()),
            "distance_high": float(base_minus_idle[dist > dist_threshold].mean()),
        },
        "strata_counts": {
            "tstar_low": int((tstar < tstar_split).sum()),
            "tstar_high": int((tstar >= tstar_split).sum()),
            "distance_low": int((dist <= dist_threshold).sum()),
            "distance_high": int((dist > dist_threshold).sum()),
        },
        "tstar_split": tstar_split,
        "distance_threshold": dist_threshold,
    }
    # item 3: J_hat vs J_true interaction/efficiency errors on a0 and IDLE + the a0-IDLE gap.
    rows = np.arange(val["n_valid"])
    a0 = val["a0"]
    idle_idx = np.full(val["n_valid"], P.IDLE_PLAN_INDEX)

    def _term_err(pred, true):
        err = pred - true
        return {"on_a0": float(err[rows, a0].mean()),
                "on_idle": float(err[rows, idle_idx].mean())}

    item3 = {
        "interaction_error_pred_minus_true": _term_err(val["interaction_pred"],
                                                       val["interaction_true"]),
        "efficiency_error_pred_minus_true": _term_err(val["efficiency_pred"],
                                                      val["efficiency_true"]),
        "total_j_error_pred_minus_true": _term_err(val["j_hat"], val["j_true"]),
        "j_hat_a0": float(val["j_hat"][rows, a0].mean()),
        "j_hat_idle": float(val["j_hat"][rows, idle_idx].mean()),
        "j_true_a0": float(val["j_true"][rows, a0].mean()),
        "j_true_idle": float(val["j_true"][rows, idle_idx].mean()),
        "j_hat_a0_minus_idle": float((val["j_hat"][rows, a0]
                                      - val["j_hat"][rows, idle_idx]).mean()),
        "j_true_a0_minus_idle": float((val["j_true"][rows, a0]
                                       - val["j_true"][rows, idle_idx]).mean()),
        "weights": {"interaction": s4co.WEIGHT_INTERACTION,
                    "efficiency": s4co.WEIGHT_EFFICIENCY},
    }

    report = {
        "plan_id": P.PLAN_ID,
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY",
        "status": gate["status"],
        "passed": gate["passed"],
        "go_no_go": gate,
        "diagnostics": {
            "item1_role_regret": diag_regret,
            "item2_action_and_strata": item2,
            "item3_cost_decomposition": item3,
            "item4_zone_paired_differences": {
                "n_high_zone": pol["n_high_zone"], "n_low_zone": pol["n_low_zone"],
                "diff_high_zone_base_minus_idle": pol["diff_high_zone"],
                "diff_low_zone_idle_minus_base": pol["diff_low_zone"],
            },
            "item5_policy_vs_controls": gate["point"],
        },
        "n_validation_valid": val["n_valid"],
        "n_train_valid": train["n_valid"],
        "bindings": {
            "validation_data_sha256": val["data_sha256"],
            "validation_predictions_sha256": val["predictions_sha256"],
            "validation_scores_sha256": val["scores_sha256"],
            "train_data_sha256": train["data_sha256"],
            "train_predictions_sha256": train["predictions_sha256"],
            "train_scores_sha256": train["scores_sha256"],
            "final_selection_sha256": sha256_file(D2_DEV_DIR / "final_selection.json"),
            "model_inventory_sha256": sha256_file(D2_DEV_DIR / "model_inventory.json"),
            "policy_code_sha256": sha256_file(
                PROJECT_ROOT / "code/ac_twin/stage4d_d2r_policy.py"),
            "statistics_code_sha256": sha256_file(
                PROJECT_ROOT / "code/ac_twin/stage4d_d2r_statistics.py"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # --- per-group diagnostic npz (offline audit; ADAPTIVE_DEVELOPMENT_ONLY) -------------------
    atomic_save_npz(
        PHASE0_DIR / "diagnostic.npz",
        group_seed=val["group_seed"].astype(np.int64),
        scenario_ids_json=np.asarray(json.dumps(list(val["scenario_ids"]))),
        j_hat=val["j_hat"], j_true=val["j_true"],
        a0=val["a0"].astype(np.int64), a_star=val["a_star"].astype(np.int64),
        regret_a0=val["regret_a0"], regret_idle=val["regret_idle"],
        regret_random=val["regret_random"], y_high=val["y_high"].astype(np.int64),
        risk_direct_ensemble=val["risk_direct_ensemble"],
        risk_direct_by_seed=val["risk_direct_by_seed"],
        **{f"baseline__{k}": v for k, v in val["baseline_risks"].items()},
        chosen_d2r=pol["chosen_d2r"].astype(np.int64),
        regret_d2r=pol["regret_d2r"],
        regret_strongest_fallback=pol["regret_strongest_fallback"],
        t_star=val["t_star"].astype(np.int64),
        interaction_min_distance=val["interaction_min_distance"],
    )

    # --- the frozen policy the screen will reuse verbatim ---------------------------------------
    per_baseline_tau = {name: float(pol["fallback_detail"]["per_baseline"][name]["tau"])
                        for name in BASELINE_KEYS}
    per_seed_tau = {}
    for i, seed in enumerate(T.MODEL_SEEDS):
        per_seed_tau[str(seed)] = float(P.freeze_threshold(
            val["risk_direct_by_seed"][i], val["scenario_ids"]))
    final_policy = {
        "plan_id": P.PLAN_ID,
        "scope": "ADAPTIVE_DEVELOPMENT_ONLY",
        "rule": ("a_D2R(g) = a0(g) if r_D(g) <= tau_D else IDLE_PLAN_INDEX(=3); "
                 "score == tau executes a0; new data never re-ranks by scenario id"),
        "coverage": P.COVERAGE,
        "tau_d2r": pol["tau_d2r"],
        "idle_plan_index": P.IDLE_PLAN_INDEX,
        "strongest_standard_fallback": pol["strongest_fallback"],
        "per_baseline_tau": per_baseline_tau,
        "baseline_tie_break_order": list(BASELINE_KEYS),
        "baseline_tie_tolerance": P.BASELINE_TIE_TOLERANCE,
        "per_seed_tau": per_seed_tau,
        "random_fallback_salt": P.RANDOM_FALLBACK_SALT,
        "model_seeds": [int(s) for s in T.MODEL_SEEDS],
        "phase0_status": gate["status"],
        "bindings": report["bindings"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(PHASE0_DIR / "final_policy.json", final_policy)
    write_once_json(report_path, report)
    print(f"[stage4d-d2r-phase0] Phase 0 COMPLETE status={gate['status']} "
          f"total_elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    device = "cpu"
    report = run_phase0(device=device)
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
