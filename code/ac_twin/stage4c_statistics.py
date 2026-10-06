"""Stage 4C decision-rule freeze, group risk curves and the Gate adjudication.

Plan ``AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1`` sections 8 / 9 / 11.
The selection role (500 groups) is the ONLY place rules are frozen: the strongest
standard baseline, the high-regret threshold, the interaction-density threshold, the
z-score statistics and the reranking ``lambda``. The 300-group screen and the confirm
set are single-use: they never re-select anything. The statistical unit is always the
decision group; the one paired bootstrap resamples whole groups.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ac_twin import stage4c_inference as s4ci
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4c_power as s4cpw
from ac_twin import stage4c_protocol as s4cp

#: standard baselines in the frozen tie-break order (plan section 8.1).
STANDARD_BASELINE_ORDER = ("UNCERTAINTY_TOP2_MAX", "GEOMETRY_TOP2_MAX", "MARGIN_ONLY")


# --- decision-rule freeze (selection role only) ----------------------------------------


def freeze_decision_rule(selection_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanically freeze the decision rule from the 500 selection groups.

    Returns the create-once ``decision_rule_receipt`` payload. Pure: the caller persists
    it exactly once. Reads selection truth (``J_true``) — legal here because selection
    is not a one-shot adjudication set.
    """
    regret = np.asarray(selection_inputs["regret"], dtype=np.float64)
    baseline_risks = selection_inputs["baseline_group_risks"]
    J_hat = np.asarray(selection_inputs["J_hat"], dtype=np.float64)
    J_true = np.asarray(selection_inputs["J_true"], dtype=np.float64)
    combined_action_risk = np.asarray(selection_inputs["combined_action_risk"], dtype=np.float64)
    density_count = np.asarray(selection_inputs["density_count"], dtype=np.float64)

    # 1. strongest standard baseline = lowest selection group AURC_REGRET.
    aurc = {}
    for name in STANDARD_BASELINE_ORDER:
        risk = np.asarray(baseline_risks[name], dtype=np.float64)
        aurc[name] = s4cpw.aurc_regret(np.argsort(risk, kind="stable"), regret)
    strongest = STANDARD_BASELINE_ORDER[0]
    for name in STANDARD_BASELINE_ORDER[1:]:
        if aurc[name] < aurc[strongest] - 1e-12:
            strongest = name

    # 2. high-regret threshold (80% quantile of selection regret).
    high_regret_threshold = float(np.quantile(regret, s4cp.HIGH_REGRET_QUANTILE))

    # 3. interaction-density threshold (median non-ego count within 30 m at history last).
    density_threshold = float(np.median(density_count))

    # 4. branch-level z-score statistics for the reranking secondary analysis.
    jhat_mean = float(J_hat.mean())
    jhat_std = float(J_hat.std())
    risk_mean = float(combined_action_risk.mean())
    risk_std = float(combined_action_risk.std())
    degenerate = (jhat_std < 1e-8) or (risk_std < 1e-8)

    # 5. mechanical lambda selection on selection mean J_true regret.
    z_j = (J_hat - jhat_mean) / max(jhat_std, 1e-12)
    z_r = (combined_action_risk - risk_mean) / max(risk_std, 1e-12)
    a_star = np.argmin(J_true, axis=-1)
    rows = np.arange(J_true.shape[0])
    j_true_star = J_true[rows, a_star]
    lambda_regret = {}
    if degenerate:
        chosen_lambda = 0.0
    else:
        for lam in s4cp.LAMBDA_CANDIDATES:
            score = z_j + float(lam) * z_r
            a_hat_l = np.argmin(score, axis=-1)
            reg_l = s4co.clamp_regret(J_true[rows, a_hat_l] - j_true_star)
            lambda_regret[float(lam)] = float(reg_l.mean())
        chosen_lambda = s4cp.LAMBDA_CANDIDATES[0]
        for lam in s4cp.LAMBDA_CANDIDATES[1:]:
            if lambda_regret[float(lam)] < lambda_regret[float(chosen_lambda)] - 1e-12:
                chosen_lambda = lam

    receipt = {
        "plan_id": s4cp.PLAN_ID,
        "strongest_standard_baseline": strongest,
        "selection_aurc_regret": aurc,
        "high_regret_threshold": high_regret_threshold,
        "high_regret_quantile": s4cp.HIGH_REGRET_QUANTILE,
        "density_threshold": density_threshold,
        "jhat_mean": jhat_mean,
        "jhat_std": jhat_std,
        "combined_risk_mean": risk_mean,
        "combined_risk_std": risk_std,
        "reranking_lambda": float(chosen_lambda),
        "lambda_candidates": list(s4cp.LAMBDA_CANDIDATES),
        "lambda_selection_regret": lambda_regret,
        "idle_plan_index": int(s4co.IDLE_PLAN_INDEX),
        "action_plans": [list(p) for p in s4co.ACTION_PLANS],
        "consequence_weights": [s4co.WEIGHT_INTERACTION, s4co.WEIGHT_EFFICIENCY],
        "n_selection_groups": int(regret.size),
    }
    if degenerate:
        receipt["reranking_status"] = s4cp.STATUS_RERANK_DEGENERATE
    return receipt


# --- group risk curves --------------------------------------------------------------------


def decision_risk_curve(group_risk: np.ndarray, regret: np.ndarray) -> dict[str, np.ndarray]:
    """R(k) = mean regret of the k lowest-risk groups; AURC_REGRET; REGRET@80."""
    risk = np.asarray(group_risk, dtype=np.float64)
    reg = np.asarray(regret, dtype=np.float64)
    order = np.argsort(risk, kind="stable")
    sorted_regret = reg[order]
    cumulative = np.cumsum(sorted_regret)
    r_of_k = cumulative / np.arange(1, reg.size + 1)
    return {
        "order": order,
        "R_of_k": r_of_k,
        "AURC_REGRET": float(r_of_k.mean()),
        "REGRET_at_80": float(r_of_k[s4cpw.s4s.coverage_index(s4cp.COVERAGE_MAIN, reg.size) - 1]),
    }


# --- paired group bootstrap ------------------------------------------------------------------


def paired_group_bootstrap(
    inputs: Mapping[str, Any],
    *,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    """One whole-group paired bootstrap -> 16-quantity point + two-sided CI.

    Every quantity is recomputed on the same resampled groups each draw (paired). The
    confidence level separates the 80% development screen from the 95% confirm.
    """
    generator = np.random.default_rng(int(seed))
    groups = inputs["regret"].shape[0]
    identity = np.arange(groups, dtype=np.int64)
    args = s4cpw._qv_args(inputs)
    point = s4cpw.compute_group_quantity_vector(identity, **args)
    draws = np.empty((int(n_resamples), s4cpw.N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = s4cpw.compute_group_quantity_vector(index, **args)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "point": point,
        "ci_lower": np.quantile(draws, alpha, axis=0),
        "ci_upper": np.quantile(draws, 1.0 - alpha, axis=0),
        "draws": draws,
        "n_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
        "confidence_level": float(confidence),
    }


# --- group-quantity assembly (adjudication side; reads truth only post-lock) ----------


def assemble_group_quantities(
    bundle: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    action_scores: Mapping[str, np.ndarray],
    decision_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the per-group statistical inputs from data + predictions + frozen rule.

    This reads the TRUE counterfactual trajectories, so it runs only AFTER the matching
    one-shot adjudication lock is burned. Returns every array the paired group
    bootstrap / gates / power need, plus the Gate TASK point quantities.
    """
    groups = int(bundle["groups"])
    n_actions = int(bundle["n_actions"])
    hist_b = np.broadcast_to(bundle["history_last_xy"][:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(bundle["inference_mask"][:, None, :, :],
                             (groups, n_actions) + bundle["inference_mask"].shape[1:])

    # predicted consequences: ensemble + per-pair.
    ens = s4ci.ensemble_predicted_cost(predictions, bundle)
    J_hat = ens["J_hat"]  # (G,10)
    e2e = np.asarray(predictions["e2e"], dtype=np.float64)  # (5,G,10,T,N,2)
    j_hat_pair = np.empty((s4cpw.N_PAIRS, groups, n_actions), dtype=np.float64)
    for p in range(s4cpw.N_PAIRS):
        y_p = bundle["factual_xy"][:, None] + e2e[p]
        comp = s4co.trajectory_components(y_p, hist_b, mask_b)
        j_hat_pair[p] = s4co.total_cost(comp["interaction"],
                                        s4co.efficiency_cost(comp["progress"]))

    # true consequences (TRUTH — post-lock only).
    true_xy = np.asarray(bundle["counterfactual_xy"], dtype=np.float64)  # (G,10,T,N,2)
    comp_true = s4co.trajectory_components(true_xy, hist_b, mask_b)
    J_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))

    rows = np.arange(groups)
    a_hat = np.argmin(J_hat, axis=-1)
    a_star = np.argmin(J_true, axis=-1)
    j_true_star = J_true[rows, a_star]
    regret = s4co.clamp_regret(J_true[rows, a_hat] - j_true_star)
    regret_by_pair = np.empty((s4cpw.N_PAIRS, groups), dtype=np.float64)
    for p in range(s4cpw.N_PAIRS):
        a_hat_p = np.argmin(j_hat_pair[p], axis=-1)
        regret_by_pair[p] = s4co.clamp_regret(J_true[rows, a_hat_p] - j_true_star)
    random_regret = s4co.clamp_regret(J_true.mean(axis=-1) - j_true_star)
    idle_regret = s4co.clamp_regret(J_true[:, s4co.IDLE_PLAN_INDEX] - j_true_star)

    # group risks via the frozen top-2 aggregation.
    group_risks = s4ci.aggregate_group_risk(action_scores, J_hat)
    combined_group_risk = group_risks["COMBINED_RISK_HEAD_TOP2_MAX"]
    baseline_group_risk = group_risks[decision_rule["strongest_standard_baseline"]]

    high_regret_label = (regret >= float(decision_rule["high_regret_threshold"])).astype(np.int64)
    density_layer = (bundle["density_count"] > float(decision_rule["density_threshold"])).astype(np.int64)
    tstar_layer = (np.asarray(bundle["t_star"]) >= 13).astype(np.int64)  # fixed [10,15] split

    # Gate TASK point quantities.
    nonzero_regret_frac = float((regret > s4cp.GATE_TASK_REGRET_EPS).mean())
    oracle_actions = a_star
    distinct_oracle_count = int(sum(
        (oracle_actions == a).mean() >= s4cp.GATE_TASK_MIN_ORACLE_FRAC
        for a in range(n_actions)))

    return {
        "group_seed": np.asarray(bundle["episode_seed"], dtype=np.int64),
        "regret": regret,
        "regret_by_pair": regret_by_pair,
        "random_regret": random_regret,
        "idle_regret": idle_regret,
        "combined_group_risk": combined_group_risk,
        "baseline_group_risk": baseline_group_risk,
        "group_risks": group_risks,
        "high_regret_label": high_regret_label,
        "tstar_layer": tstar_layer,
        "density_layer": density_layer,
        "J_hat": J_hat,
        "J_true": J_true,
        "a_hat": a_hat,
        "a_star": a_star,
        "task_point": {
            "nonzero_regret_frac": nonzero_regret_frac,
            "distinct_oracle_count": distinct_oracle_count,
            "base_mean_regret": float(regret.mean()),
            "idle_mean_regret": float(idle_regret.mean()),
            "random_mean_regret": float(random_regret.mean()),
        },
    }


# --- Gate TASK (decision-task / base-model qualification) ---------------------------------


def evaluate_gate_task(task_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the six Gate TASK conditions (plan section 9.2).

    ``task_inputs`` carries the point quantities and the bootstrapped CI lowers for the
    two CI-gated conditions (Spearman, base-vs-RANDOM). Returns per-condition verdicts
    and the mechanical status.
    """
    conditions = {
        "nonzero_regret_frac_ge_20pct": bool(
            task_inputs["nonzero_regret_frac"] >= s4cp.GATE_TASK_MIN_NONZERO_REGRET_FRAC),
        "at_least_3_distinct_oracle": bool(
            task_inputs["distinct_oracle_count"] >= s4cp.GATE_TASK_MIN_DISTINCT_ORACLE),
        "spearman_ci_lower_positive": bool(task_inputs["spearman_ci_lower"] > 0.0),
        "base_vs_random_improvement": bool(
            task_inputs["base_vs_random_point"] > 0.0
            and task_inputs["base_vs_random_ci_lower"] > 0.0),
        "base_not_worse_than_idle": bool(
            task_inputs["base_mean_regret"] <= task_inputs["idle_mean_regret"] + 1e-12),
    }
    degenerate = not (conditions["nonzero_regret_frac_ge_20pct"]
                      and conditions["at_least_3_distinct_oracle"])
    insufficient = not (conditions["spearman_ci_lower_positive"]
                        and conditions["base_vs_random_improvement"]
                        and conditions["base_not_worse_than_idle"])
    if degenerate:
        status = s4cp.STATUS_TASK_DEGENERATE
    elif insufficient:
        status = s4cp.STATUS_MODEL_INSUFFICIENT
    else:
        status = "TASK_QUALIFIED"
    return {"conditions": conditions, "passed": not (degenerate or insufficient),
            "status": status}


# --- Gate RISK-DECISION ------------------------------------------------------------------------


def evaluate_gate_risk_decision(
    point: np.ndarray,
    ci_lower: np.ndarray,
    *,
    qa_passed: bool,
    task_status: str,
    baseline_valid: bool,
) -> dict[str, Any]:
    """Apply the Gate RISK-DECISION AND-chain with the frozen terminal-state priority.

    Order (plan section 11.3): Gate Q fail -> INVALID; baseline denominator
    non-positive -> INVALID; Gate TASK degenerate/insufficient; then the six risk
    conditions. ``point``/``ci_lower`` are the 16-quantity vectors.
    """
    point = np.asarray(point, dtype=np.float64)
    ci_lower = np.asarray(ci_lower, dtype=np.float64)
    pair_positive = int((point[s4cpw.PAIR_INDEX] > 0.0).sum())
    strata_min = float(np.nanmin(point[s4cpw.STRATA_INDEX]))
    trims_positive = bool((point[s4cpw.TRIM_INDEX] > 0.0).all())

    risk_conditions = {
        "i_aurc_regret": bool(point[s4cpw.Q_I_AURC] >= s4cp.GATE_I_AURC_REGRET_MIN
                              and ci_lower[s4cpw.Q_I_AURC] > 0.0),
        "i_regret80": bool(point[s4cpw.Q_I_REGRET80] >= s4cp.GATE_I_REGRET80_MIN
                           and ci_lower[s4cpw.Q_I_REGRET80] > 0.0),
        "pr_auc_minus_prevalence": bool(point[s4cpw.Q_PR] > 0.0
                                        and ci_lower[s4cpw.Q_PR] > 0.0),
        "at_least_4_of_5_pairs_positive": bool(pair_positive >= s4cp.GATE_MIN_POSITIVE_PAIRS),
        "no_strata_below_floor": bool(strata_min >= s4cp.GATE_STRATA_I_AURC_FLOOR),
        "trims_positive": trims_positive,
    }

    if not qa_passed or not baseline_valid:
        status = s4cp.STATUS_INVALID
    elif task_status == s4cp.STATUS_TASK_DEGENERATE:
        status = s4cp.STATUS_TASK_DEGENERATE
    elif task_status == s4cp.STATUS_MODEL_INSUFFICIENT:
        status = s4cp.STATUS_MODEL_INSUFFICIENT
    elif all(risk_conditions.values()):
        status = "RISK_DECISION_PASSED"
    else:
        status = "RISK_DECISION_FAILED"
    return {
        "risk_conditions": risk_conditions,
        "positive_pairs": pair_positive,
        "strata_min_i_aurc": strata_min,
        "risk_passed": bool(all(risk_conditions.values())),
        "status": status,
    }


__all__ = [
    "STANDARD_BASELINE_ORDER",
    "assemble_group_quantities",
    "decision_risk_curve",
    "evaluate_gate_risk_decision",
    "evaluate_gate_task",
    "freeze_decision_rule",
    "paired_group_bootstrap",
]
