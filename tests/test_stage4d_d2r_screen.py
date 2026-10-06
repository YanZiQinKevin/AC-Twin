"""Tests for the Stage 4D-D2R plan-7 screen gate + plan-5 power AND-chain.

Blocking invariants proven here: the screen quantity vector has the frozen 40-slot layout; the
single adjudication gate's terminal priority is exactly INVALID_ANALYSIS -> DEV_NOT_READY ->
DEV_READY (a degenerate denominator or QA/completeness failure is INVALID, a valid screen with a
failed statistical condition is DEV_NOT_READY); the vectorised power AND-chain mirrors the gate's
statistical conditions; and the power subset's local indices map onto the frozen full layout.
Pure NumPy, no torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from ac_twin import stage4d_d2r_policy as P
from ac_twin import stage4d_d2r_statistics as STAT
from ac_twin import stage4d_d2r_power as PW


def _screen_quantities(groups=1000, helpful=True, seed=7):
    """Synthetic plan-7 quantities: a high-risk zone where idle beats base (if helpful)."""
    rng = np.random.default_rng(seed)
    n_act = P.N_ACTIONS
    j_true = rng.uniform(0.10, 0.20, size=(groups, n_act))
    a0 = np.zeros(groups, dtype=np.int64)
    risk = np.linspace(0.0, 1.0, groups)
    ids = [f"g{i}" for i in range(groups)]
    tau = P.freeze_threshold(risk, ids)
    high = risk > tau
    if helpful:
        j_true[high, P.IDLE_PLAN_INDEX] = 0.02          # idle best in the high zone
        j_true[high, 0] = 0.40                          # base (a0=0) worst in the high zone
        j_true[~high, 0] = 0.03                         # base good in the low zone
        low = np.where(~high)[0]
        third = low % 3
        j_true[low[third == 0], 1] = 0.02               # action 1 best for a third of the low zone
        j_true[low[third == 1], 2] = 0.02               # action 2 best for another third
    a_star = np.argmin(j_true, axis=-1)
    j_hat = j_true + rng.uniform(0.0, 0.02, size=j_true.shape)
    d2r = P.evaluate_policy(j_true, a_star, a0, risk, tau)

    regret_by_seed = np.empty((STAT.N_MODEL_SEEDS, groups))
    for i in range(STAT.N_MODEL_SEEDS):
        r_i = risk + rng.normal(0.0, 1e-4, groups)
        chosen_i = P.apply_fallback(a0, r_i, P.freeze_threshold(r_i, ids))
        regret_by_seed[i] = P.action_regret(j_true, chosen_i, a_star)
    regret_by_fallback = np.empty((STAT.N_FALLBACKS, groups))
    for bi in range(STAT.N_FALLBACKS):
        noise = rng.uniform(0.0, 1.0, groups)
        chosen_fb = P.apply_fallback(a0, noise, P.freeze_threshold(noise, ids))
        regret_by_fallback[bi] = P.action_regret(j_true, chosen_fb, a_star)
    idle = np.full(groups, P.IDLE_PLAN_INDEX)
    strong_noise = rng.uniform(0.0, 1.0, groups)
    chosen_strong = P.apply_fallback(a0, strong_noise, P.freeze_threshold(strong_noise, ids))

    return {
        "regret_d2r": d2r["regret"],
        "regret_idle": P.action_regret(j_true, idle, a_star),
        "regret_base": P.action_regret(j_true, a0, a_star),
        "regret_strongest_fallback": P.action_regret(j_true, chosen_strong, a_star),
        "regret_by_seed": regret_by_seed,
        "regret_by_fallback": regret_by_fallback,
        "chosen_d2r": d2r["chosen_action"],
        "a0": a0, "risk_d2r": risk, "tau_d2r": tau,
        "j_hat": j_hat, "j_true": j_true, "a_star": a_star,
        "reactive": rng.random(groups) < 0.5,
        "tstar_layer": (np.arange(groups) % 2).astype(np.int64),
        "interaction_layer": ((np.arange(groups) // 2) % 2).astype(np.int64),
        "strongest_fallback": P.FALLBACK_BASELINES[0],
    }


def _qa(ok=True):
    return {"gate_q_isolation_passed": ok, "new_old_id_overlap_zero": ok,
            "no_outcome_based_rejection": ok, "scores_frozen_before_truth": ok,
            "ensemble_is_arithmetic_mean": ok, "thresholds_match_frozen": ok}


def _gate(quantities, groups, *, qa=None, n_expected=None, proj=True, mask=True, resamples=300):
    q = dict(quantities)
    q["qa"] = _qa() if qa is None else qa
    q["projection_nondegenerate"] = proj
    q["eval_mask_nondegenerate"] = mask
    boot = STAT.paired_bootstrap(
        lambda idx: STAT.compute_screen_quantity_vector(idx, q), groups,
        seed=STAT.SCREEN_BOOTSTRAP_SEED, n_resamples=resamples)
    return STAT.evaluate_screen_gates(q, boot, n_expected=groups if n_expected is None
                                      else n_expected)


def test_screen_quantity_vector_layout():
    q = _screen_quantities(400, helpful=True)
    vec = STAT.compute_screen_quantity_vector(np.arange(400), q)
    assert vec.shape == (STAT.N_SCREEN_QUANTITIES,)
    assert len(STAT.POWER_SLOTS) == STAT.N_POWER_QUANTITIES
    # improvements are positive on the helpful construction; fractions match the frozen rule.
    assert vec[STAT.S_I_IDLE] > 0.0 and vec[STAT.S_I_BASE] > 0.0
    assert vec[STAT.S_FRAC_CHANGED] == pytest.approx(0.2, abs=0.02)
    assert vec[STAT.S_FRAC_NON_IDLE] == pytest.approx(0.8, abs=0.02)
    # oracle shares sum to 1 and at least three actions clear 5%.
    shares = [vec[STAT.S_ORACLE + a] for a in range(P.N_ACTIONS)]
    assert sum(shares) == pytest.approx(1.0)
    assert sum(s >= STAT.MIN_ORACLE_FRAC for s in shares) >= STAT.MIN_DISTINCT_ORACLE


def test_screen_gate_helpful_is_dev_ready():
    groups = 1000
    gate = _gate(_screen_quantities(groups, helpful=True), groups)
    assert gate["status"] == STAT.STATUS_DEV_READY
    assert gate["passed"] is True
    assert all(gate["conditions"].values())


def test_screen_gate_not_helpful_is_dev_not_ready():
    groups = 1000
    gate = _gate(_screen_quantities(groups, helpful=False), groups)
    assert gate["status"] == STAT.STATUS_DEV_NOT_READY
    assert gate["passed"] is False
    # the improvement conditions fail (not an integrity/QA failure).
    assert gate["conditions"]["c1_integrity_qa_and_completeness"] is True
    assert not (gate["conditions"]["c3_i_idle"] and gate["conditions"]["c4_i_base"])


def test_screen_gate_degenerate_denominator_is_invalid_not_method_failure():
    groups = 500
    q = _screen_quantities(groups, helpful=True)
    q["regret_idle"] = np.zeros(groups)                 # degenerate ALWAYS_IDLE denominator
    gate = _gate(q, groups)
    assert gate["status"] == STAT.STATUS_INVALID_ANALYSIS
    assert gate["denominators_valid"] is False


def test_screen_gate_qa_or_completeness_failure_is_invalid():
    groups = 500
    good = _screen_quantities(groups, helpful=True)
    # a QA isolation failure is INVALID even when the statistics all pass.
    assert _gate(good, groups, qa=_qa(ok=False))["status"] == STAT.STATUS_INVALID_ANALYSIS
    # an n/n completeness mismatch is INVALID (incomplete test, not a method failure).
    assert _gate(good, groups, n_expected=groups + 1)["status"] == STAT.STATUS_INVALID_ANALYSIS
    # a degenerate projection / mask is INVALID.
    assert _gate(good, groups, proj=False)["status"] == STAT.STATUS_INVALID_ANALYSIS
    assert _gate(good, groups, mask=False)["status"] == STAT.STATUS_INVALID_ANALYSIS


def test_power_subset_local_indices_map_to_full_layout():
    # the power AND-chain's local indices must read the same quantities as the full 40-vector.
    assert STAT.POWER_SLOTS[STAT.P_I_IDLE] == STAT.S_I_IDLE
    assert STAT.POWER_SLOTS[STAT.P_I_BASE] == STAT.S_I_BASE
    assert STAT.POWER_SLOTS[STAT.P_I_STANDARD] == STAT.S_I_STANDARD
    assert STAT.POWER_SLOTS[STAT.P_FRAC_CHANGED] == STAT.S_FRAC_CHANGED
    assert STAT.POWER_SLOTS[STAT.P_SPEARMAN] == STAT.S_SPEARMAN_BRANCH
    for k in range(STAT.N_MODEL_SEEDS):
        assert STAT.POWER_SLOTS[STAT.P_SEED_I_IDLE[k]] == STAT.S_SEED_I_IDLE + k
        assert STAT.POWER_SLOTS[STAT.P_SEED_I_STANDARD[k]] == STAT.S_SEED_I_STANDARD + k
    for a in range(P.N_ACTIONS):
        assert STAT.POWER_SLOTS[STAT.P_ORACLE[a]] == STAT.S_ORACLE + a


def test_power_draws_chain_mirrors_gate_direction():
    # a helpful population passes the vectorised AND-chain; a non-helpful one fails, matching
    # evaluate_screen_gates on the same construction.
    groups = 800
    boot_help = STAT.paired_bootstrap(
        lambda idx: STAT.compute_screen_quantity_vector(idx, _screen_quantities(groups, True)),
        groups, seed=PW.POWER_BOOTSTRAP_SEED, n_resamples=200)
    subset = list(STAT.POWER_SLOTS)
    theta = boot_help.point[subset]
    cov = PW.symmetrised_psd_covariance(boot_help.draws[:, subset])
    se = np.sqrt(np.diag(cov))
    result = PW.simulate_power(theta, cov, candidate_ns=(groups,), n_reference=groups,
                               monte_carlo=500, seed=PW.POWER_BOOTSTRAP_SEED,
                               shrink_scenarios=(1.0,), selection_scenario=1.0, target=0.5)
    # at the identity effect (shrink 1.0) and n == n_reference the joint power should be high.
    assert result["per_n"][groups]["scenarios"]["1.00"]["joint_power"] > 0.5

    boot_flat = STAT.paired_bootstrap(
        lambda idx: STAT.compute_screen_quantity_vector(idx, _screen_quantities(groups, False)),
        groups, seed=PW.POWER_BOOTSTRAP_SEED, n_resamples=200)
    theta_flat = boot_flat.point[subset]
    cov_flat = PW.symmetrised_psd_covariance(boot_flat.draws[:, subset])
    se_flat = np.sqrt(np.diag(cov_flat))
    draws = np.random.default_rng(0).multivariate_normal(theta_flat, cov_flat, size=500)
    ev = STAT.evaluate_screen_draws(draws, se_flat)
    assert ev["passed"].mean() < 0.2          # the non-helpful construction almost never passes


def test_power_selection_underpowered_when_effect_too_small():
    # a near-zero improvement can never reach the 5% line -> simulate_power reports underpowered.
    groups = 800
    boot = STAT.paired_bootstrap(
        lambda idx: STAT.compute_screen_quantity_vector(idx, _screen_quantities(groups, False)),
        groups, seed=PW.POWER_BOOTSTRAP_SEED, n_resamples=200)
    subset = list(STAT.POWER_SLOTS)
    theta = boot.point[subset]
    cov = PW.symmetrised_psd_covariance(boot.draws[:, subset])
    result = PW.simulate_power(theta, cov, candidate_ns=(1500, 2500), n_reference=groups,
                               monte_carlo=500, seed=PW.POWER_BOOTSTRAP_SEED,
                               shrink_scenarios=(1.0, 0.75), selection_scenario=0.75,
                               target=0.80)
    assert result["underpowered"] is True
    assert result["selected_n"] is None
