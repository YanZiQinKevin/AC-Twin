"""Stage 3 statistics: effect-ADE endpoint, H4, Gate R/T/C and power.

The Gate C primary endpoint is the episode-level ``effect-ADE@1:20`` over
non-ego, factual/counterfactual jointly-visible agent-times. Each episode is
first averaged over valid vehicles and times, then averaged over the five
seed-pairs, and finally compared with an episode-level paired bootstrap. The
two main contrasts are an intersection-union (AND) test:

- ``C_uniform = ADE_uniform - ADE_dual``;  ``Rel_uniform = C_uniform / ADE_uniform``
- ``C_direct  = ADE_direct  - ADE_dual``;  ``Rel_direct  = C_direct  / ADE_direct``

``effect-FDE@20`` is a key secondary metric and never substitutes the primary
endpoint. The 5% line is a minimum-substantive-effect screen, not a planning
effect; formal power is planned at the 7.5% main scenario (10% auxiliary).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from ac_twin.stage1_statistics import episode_bootstrap, spearman
from ac_twin.stage3_response import macro_brier, pr_auc

# Formal status priority (section 6.4 of the plan).
STATUS_INVALID_TECHNICAL = "INVALID_TECHNICAL"
STATUS_H4_INSUFFICIENT_ID = "INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_ID"
STATUS_GATE_C_NOT_CONFIRMED = "GATE_C_NOT_CONFIRMED"
STATUS_ORACLE_INVALID = "GATE_C_PRIMARY_COMPARISONS_CONFIRMED_ORACLE_REFERENCE_INVALID"
STATUS_EXTERNAL_INCONCLUSIVE = "GATE_C_CONFIRMED_ID_EXTERNAL_INCONCLUSIVE"
STATUS_EXTERNAL_NOT_CONFIRMED = "GATE_C_NOT_CONFIRMED_EXTERNAL_SETTING"
STATUS_GATE_C_CONFIRMED = "GATE_C_CONFIRMED"
STATUS_DEV_NOT_READY = "STAGE3_DEV_NOT_READY_FOR_CONFIRMATION"
STATUS_UNDERPOWERED = "UNDERPOWERED_FOR_GATE_C"

MIN_SUBSTANTIVE_REL = 0.05
PLANNING_REL_MAIN = 0.075
PLANNING_REL_AUX = 0.10
ECE_GUARD = 0.05
ECE_RELATIVE_IMPROVEMENT = 0.20


# --------------------------------------------------------------------------- #
# Per-episode endpoint vectors
# --------------------------------------------------------------------------- #
def per_timestep_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """L2 position error per (e,t,n) from ``...×2`` effect predictions."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.linalg.norm(pred - target, axis=-1)


def effect_ade_per_episode(pred: np.ndarray, target: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    """Per-episode effect-ADE@1:20: mean error over valid agent-times -> (E,)."""
    err = per_timestep_error(pred, target)
    valid = np.asarray(eval_mask, dtype=bool)
    episodes = err.shape[0]
    out = np.full(episodes, np.nan, dtype=np.float64)
    for e in range(episodes):
        vals = err[e][valid[e]]
        if vals.size:
            out[e] = vals.mean()
    return out


def effect_fde_per_episode(pred: np.ndarray, target: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    """Per-episode effect-FDE@20: error at the final valid time step -> (E,)."""
    err = per_timestep_error(pred, target)
    valid = np.asarray(eval_mask, dtype=bool)
    episodes, steps, _ = err.shape
    out = np.full(episodes, np.nan, dtype=np.float64)
    for e in range(episodes):
        # last valid time per episode across any valid vehicle
        final_errs = []
        for n in range(err.shape[2]):
            valid_times = np.flatnonzero(valid[e, :, n])
            if valid_times.size:
                final_errs.append(err[e, valid_times[-1], n])
        if final_errs:
            out[e] = float(np.mean(final_errs))
    return out


def mae_per_episode(r_hat: np.ndarray, r_true: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    """Per-episode MAE of the continuous response prediction -> (E,)."""
    r_hat = np.asarray(r_hat, dtype=np.float64)
    r_true = np.asarray(r_true, dtype=np.float64)
    valid = np.asarray(eval_mask, dtype=bool)
    abs_err = np.abs(r_hat - r_true)
    episodes = r_hat.shape[0]
    out = np.full(episodes, np.nan, dtype=np.float64)
    for e in range(episodes):
        vals = abs_err[e][valid[e]]
        if vals.size:
            out[e] = vals.mean()
    return out


# --------------------------------------------------------------------------- #
# Bootstrap statistic factories (episode-level paired)
# --------------------------------------------------------------------------- #
def _pool_by_episode(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Concatenate per-episode arrays selected by episode indices."""
    return np.concatenate([values[i] for i in indices])


def make_paired_contrast_statistic(
    reference_per_episode: np.ndarray, candidate_per_episode: np.ndarray
) -> Callable[[np.ndarray], float]:
    """C = mean(reference - candidate) over resampled episodes (paired)."""
    reference = np.asarray(reference_per_episode, dtype=np.float64)
    candidate = np.asarray(candidate_per_episode, dtype=np.float64)

    def stat(indices: np.ndarray) -> float:
        diff = reference[indices] - candidate[indices]
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            return float("nan")
        return float(diff.mean())

    return stat


def make_relative_contrast_statistic(
    reference_per_episode: np.ndarray, candidate_per_episode: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Rel = mean(reference - candidate) / mean(reference) over resampled episodes."""
    reference = np.asarray(reference_per_episode, dtype=np.float64)
    candidate = np.asarray(candidate_per_episode, dtype=np.float64)

    def stat(indices: np.ndarray) -> float:
        ref = reference[indices]
        cand = candidate[indices]
        keep = np.isfinite(ref) & np.isfinite(cand)
        ref, cand = ref[keep], cand[keep]
        denom = ref.mean()
        if ref.size == 0 or not np.isfinite(denom) or abs(denom) < 1e-12:
            return float("nan")
        return float((ref - cand).mean() / denom)

    return stat


def make_pooled_spearman_statistic(
    r_hat: np.ndarray, r_true: np.ndarray, eval_mask: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Episode-clustered Spearman between R_hat and R_true over valid agent-times."""
    r_hat = np.asarray(r_hat, dtype=np.float64)
    r_true = np.asarray(r_true, dtype=np.float64)
    valid = np.asarray(eval_mask, dtype=bool)
    per_episode_hat = [r_hat[e][valid[e]] for e in range(r_hat.shape[0])]
    per_episode_true = [r_true[e][valid[e]] for e in range(r_true.shape[0])]

    def stat(indices: np.ndarray) -> float:
        x = _pool_by_episode(per_episode_hat, indices)
        y = _pool_by_episode(per_episode_true, indices)
        if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan")
        return float(spearman(x, y))

    return stat


def make_pooled_difference_statistic(
    metric_fn: Callable[[list[np.ndarray]], float],
    model_per_episode: list[np.ndarray],
    baseline_per_episode: list[np.ndarray],
    *,
    higher_is_better_model: bool,
) -> Callable[[np.ndarray], float]:
    """Episode-clustered difference of a pooled metric (model vs baseline).

    ``model_per_episode``/``baseline_per_episode`` are per-episode arrays already
    restricted to valid agent-times. ``metric_fn`` maps the pooled (concatenated)
    resampled array to a scalar (e.g. a PR-AUC or Brier variant). The difference
    sign is set so a positive value means the model is better; a degenerate
    resample yields NaN (a non-finite bootstrap replicate).
    """
    def stat(indices: np.ndarray) -> float:
        m_val = metric_fn([model_per_episode[i] for i in indices])
        b_val = metric_fn([baseline_per_episode[i] for i in indices])
        if not (np.isfinite(m_val) and np.isfinite(b_val)):
            return float("nan")
        return float(m_val - b_val) if higher_is_better_model else float(b_val - m_val)

    return stat


def make_pr_auc_difference_statistic(
    model_scores: np.ndarray,
    model_labels: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_labels: np.ndarray,
    valid_mask: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """I_PR = PR_AUC_model - PR_AUC_baseline, episode-clustered (positive better)."""
    valid = np.asarray(valid_mask, dtype=bool)

    def pooled(scores, labels, indices):
        s = np.concatenate([np.asarray(scores)[i][valid[i]] for i in indices])
        l = np.concatenate([np.asarray(labels)[i][valid[i]] for i in indices])
        return s, l

    def stat(indices: np.ndarray) -> float:
        ms, ml = pooled(model_scores, model_labels, indices)
        bs, bl = pooled(baseline_scores, baseline_labels, indices)
        m = pr_auc(ms, ml, np.ones(ms.shape, dtype=bool))
        b = pr_auc(bs, bl, np.ones(bs.shape, dtype=bool))
        if not (np.isfinite(m) and np.isfinite(b)):
            return float("nan")
        return float(m - b)

    return stat


def make_brier_difference_statistic(
    model_probs: np.ndarray,
    baseline_probs: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """I_Brier = Brier_baseline - Brier_model, episode-clustered (positive better)."""
    valid = np.asarray(valid_mask, dtype=bool)

    def stat(indices: np.ndarray) -> float:
        def pooled(probs):
            p = np.concatenate([np.asarray(probs)[i][valid[i]] for i in indices])
            l = np.concatenate([np.asarray(labels)[i][valid[i]] for i in indices])
            v = np.ones(l.shape, dtype=bool)
            return macro_brier(p, l, v)

        m = pooled(model_probs)
        b = pooled(baseline_probs)
        if not (np.isfinite(m) and np.isfinite(b)):
            return float("nan")
        return float(b - m)

    return stat


# --------------------------------------------------------------------------- #
# Gate R_dev (development response qualification) -- section 3.5
# --------------------------------------------------------------------------- #
def evaluate_gate_r_dev(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """AND-evaluate the seven development qualification conditions.

    ``results`` must carry bootstrap dicts (with ``ci_lower``/``point``) keyed:
    ``i_mae``, ``spearman``, ``i_pr_presence``, ``i_pr_high_vs_lowmid``,
    ``i_brier``; plus ``ece`` (float), ``ece_baseline`` (float), and
    ``five_seed_consistent`` / ``non_degenerate`` booleans.
    """
    checks = {
        "i_mae_ci_lower_gt0": float(results["i_mae"]["ci_lower"]) > 0.0,
        "spearman_ci_lower_gt0": float(results["spearman"]["ci_lower"]) > 0.0,
        "i_pr_presence_ci_lower_gt0": float(results["i_pr_presence"]["ci_lower"]) > 0.0,
        "i_pr_high_vs_lowmid_ci_lower_gt0": float(results["i_pr_high_vs_lowmid"]["ci_lower"]) > 0.0,
        "i_brier_ci_lower_gt0": float(results["i_brier"]["ci_lower"]) > 0.0,
        "ece_calibrated": (
            float(results["ece"]) <= ECE_GUARD
            or float(results["ece"]) <= (1.0 - ECE_RELATIVE_IMPROVEMENT) * float(results["ece_baseline"])
        ),
        "five_seed_consistent_nondegenerate": bool(results["five_seed_consistent"]) and bool(results["non_degenerate"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


# --------------------------------------------------------------------------- #
# Gate T_dev (routing feasibility bridge) -- section 5.1
# --------------------------------------------------------------------------- #
def evaluate_gate_t_dev(results: Mapping[str, Any]) -> dict[str, Any]:
    """AND-evaluate the routing-feasibility conditions before any formal unlock.

    ``results`` keys: ``reconstruction_ok``/``ablation_ok``/``no_oracle_leakage``
    (bool); ``oracle_beats_uniform`` (bool); ``rel_direct``/``rel_uniform``
    (point estimates); ``rel_direct_ci80_lower``/``rel_uniform_ci80_lower``
    (float); ``crossfit_direct_not_reversed`` (bool); ``audit_ok`` (bool).
    """
    checks = {
        "technical_ok": bool(results["reconstruction_ok"]) and bool(results["ablation_ok"]) and bool(results["no_oracle_leakage"]),
        "oracle_beats_uniform": bool(results["oracle_beats_uniform"]),
        "dual_vs_direct_point_ge5": float(results["rel_direct"]) >= MIN_SUBSTANTIVE_REL,
        "dual_vs_uniform_point_ge5": float(results["rel_uniform"]) >= MIN_SUBSTANTIVE_REL,
        "ci80_lower_gt0": float(results["rel_direct_ci80_lower"]) > 0.0 and float(results["rel_uniform_ci80_lower"]) > 0.0,
        "crossfit_direct_not_reversed": bool(results["crossfit_direct_not_reversed"]),
        "audit_ok": bool(results["audit_ok"]),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "status": None if passed else STATUS_DEV_NOT_READY,
    }


# --------------------------------------------------------------------------- #
# Formal Gate C state machine (section 6.4 priority order)
# --------------------------------------------------------------------------- #
def judge_formal_gate_c(report: Mapping[str, Any]) -> str:
    """Mechanically adjudicate the single formal status in priority order.

    Required report keys (booleans unless noted):
    ``technical_ok``; ``id_h4_ok``; ``id_c_uniform_ci_lower``/``id_c_direct_ci_lower``
    (float); ``id_rel_uniform``/``id_rel_direct`` (float); ``id_robust_ok``;
    ``oracle_gap`` (float); ``unseen_h4_ok``; ``unseen_c_uniform_point``/
    ``unseen_c_direct_point`` (float); ``unseen_midhigh_reversed`` (bool).
    """
    if not report["technical_ok"]:
        return STATUS_INVALID_TECHNICAL
    if not report["id_h4_ok"]:
        return STATUS_H4_INSUFFICIENT_ID
    id_primary_ok = (
        float(report["id_c_uniform_ci_lower"]) > 0.0
        and float(report["id_c_direct_ci_lower"]) > 0.0
        and float(report["id_rel_uniform"]) >= MIN_SUBSTANTIVE_REL
        and float(report["id_rel_direct"]) >= MIN_SUBSTANTIVE_REL
        and bool(report["id_robust_ok"])
    )
    if not id_primary_ok:
        return STATUS_GATE_C_NOT_CONFIRMED
    if float(report["oracle_gap"]) <= 0.0:
        return STATUS_ORACLE_INVALID
    if not report["unseen_h4_ok"]:
        return STATUS_EXTERNAL_INCONCLUSIVE
    if (
        float(report["unseen_c_uniform_point"]) <= 0.0
        or float(report["unseen_c_direct_point"]) <= 0.0
        or bool(report["unseen_midhigh_reversed"])
    ):
        return STATUS_EXTERNAL_NOT_CONFIRMED
    return STATUS_GATE_C_CONFIRMED


# --------------------------------------------------------------------------- #
# Power estimation (section 6.2) -- Monte Carlo over development paired losses
# --------------------------------------------------------------------------- #
def estimate_gate_c_power(
    c_uniform_per_episode: np.ndarray,
    c_direct_per_episode: np.ndarray,
    ade_uniform_per_episode: np.ndarray,
    ade_direct_per_episode: np.ndarray,
    *,
    candidate_ns: range,
    target_rel: float,
    monte_carlo_replicates: int,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[int, dict[str, float]]:
    """Joint pass probability (CI lower>0 AND point>=5%) per candidate n.

    The development episode-level paired contrast vectors are recentered so both
    relative true improvements equal ``target_rel``, preserving their variance,
    correlation and paired structure. For each candidate n, episodes are
    resampled with replacement and the joint pass probability of both contrasts
    is estimated. This is a development power estimate only.
    """
    c_u = np.asarray(c_uniform_per_episode, dtype=np.float64)
    c_d = np.asarray(c_direct_per_episode, dtype=np.float64)
    a_u = np.asarray(ade_uniform_per_episode, dtype=np.float64)
    a_d = np.asarray(ade_direct_per_episode, dtype=np.float64)
    keep = np.isfinite(c_u) & np.isfinite(c_d) & np.isfinite(a_u) & np.isfinite(a_d)
    c_u, c_d, a_u, a_d = c_u[keep], c_d[keep], a_u[keep], a_d[keep]
    n_dev = c_u.size
    if n_dev < 10:
        raise ValueError("Need at least 10 development episodes for power estimation")

    # Recenter both contrasts to the target relative improvement.
    def recenter(contrast, reference):
        current_rel = contrast.mean() / reference.mean()
        shift = target_rel * reference.mean() - contrast.mean()
        return contrast + shift

    c_u_rc = recenter(c_u, a_u)
    c_d_rc = recenter(c_d, a_d)

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    out: dict[int, dict[str, float]] = {}
    for n in candidate_ns:
        joint_pass = 0
        for _ in range(monte_carlo_replicates):
            idx = rng.integers(0, n_dev, size=n)
            su = c_u_rc[idx]
            sd = c_d_rc[idx]
            au = a_u[idx]
            ad = a_d[idx]
            # bootstrap CI lower for each contrast within this replicate
            def ci_lower_and_rel(contrast, reference):
                boots = np.empty(bootstrap_resamples)
                for b in range(bootstrap_resamples):
                    bidx = rng.integers(0, n, size=n)
                    boots[b] = contrast[bidx].mean()
                ci_lo = np.quantile(boots, alpha / 2)
                rel = contrast.mean() / reference.mean()
                return ci_lo, rel

            cu_lo, ru = ci_lower_and_rel(su, au)
            cd_lo, rd = ci_lower_and_rel(sd, ad)
            if cu_lo > 0 and cd_lo > 0 and ru >= MIN_SUBSTANTIVE_REL and rd >= MIN_SUBSTANTIVE_REL:
                joint_pass += 1
        out[int(n)] = {"joint_power": joint_pass / monte_carlo_replicates}
    return out
