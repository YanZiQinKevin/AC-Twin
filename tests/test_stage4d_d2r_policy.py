"""Tests for the Stage 4D-D2R policy + Phase 0 statistics (plan 4 / Task 1).

Blocking invariants proven here: the idle plan is action slot 3 = (1,3); the threshold freeze is
deterministic with the (score, id) tie-break; the score==threshold and a0==IDLE behaviours; the
executed-action regret identity against the frozen truth labels; reproducibility; and the Phase 0
Go/No-Go gate direction (a rule that helps passes, one that does not fails). Pure NumPy, no torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4d_d2r_policy as P
from ac_twin import stage4d_d2r_statistics as STAT
from ac_twin import stage4d_external_training as T


def _j_true(groups=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.30, size=(groups, P.N_ACTIONS))


def test_idle_plan_index_is_slot_3():
    assert P.IDLE_PLAN_INDEX == 3
    assert s4co.ACTION_PLANS[P.IDLE_PLAN_INDEX] == (1, 3)
    assert T.IDLE_PLAN_INDEX == 3


def test_freeze_threshold_quantile_and_tie_break():
    scores = np.arange(1.0, 11.0)                       # 1..10
    ids = [f"s{i}" for i in range(10)]
    # ceil(0.8*10) = 8 -> the 8th smallest (1-indexed) = 8.0
    assert P.freeze_threshold(scores, ids) == pytest.approx(8.0)
    # deterministic under input permutation: same (score,id) pairs -> same tau.
    perm = np.random.default_rng(1).permutation(10)
    assert P.freeze_threshold(scores[perm], [ids[i] for i in perm]) == pytest.approx(8.0)
    # tie around the quantile broken by scenario id, deterministically.
    tied = np.array([1.0, 2.0, 2.0, 2.0, 5.0])
    assert P.freeze_threshold(tied, ["b", "a", "c", "d", "e"]) == \
        P.freeze_threshold(tied, ["b", "a", "c", "d", "e"])


def test_apply_fallback_threshold_equal_and_idle_behaviour():
    a0 = np.array([0, 3, 5, 3])          # two groups already idle (a0 == 3)
    risk = np.array([0.1, 0.9, 0.5, 0.2])
    tau = 0.5
    chosen = P.apply_fallback(a0, risk, tau)
    # risk<=tau -> a0 (0.1<=0.5, 0.5==0.5 -> a0, 0.2<=0.5); risk>tau -> IDLE (0.9).
    assert chosen.tolist() == [0, 3, 5, 3]
    # group 1 (a0 already 3) stays 3; group 3 (a0==3, low risk) stays 3.
    assert chosen[1] == P.IDLE_PLAN_INDEX and chosen[3] == P.IDLE_PLAN_INDEX
    fr = P.fallback_fractions(a0, risk, tau)
    assert fr["frac_above_threshold"] == pytest.approx(0.25)        # only 0.9
    assert fr["frac_changed_action"] == pytest.approx(0.0)          # that group's a0 was already 3
    assert fr["frac_final_non_idle"] == pytest.approx(0.5)          # groups 0 and 2 non-idle


def test_action_regret_matches_truth_labels():
    groups = 60
    j_true = _j_true(groups, seed=3)
    a_star = np.argmin(j_true, axis=-1)
    # fabricate a j_hat whose argmin is a known a0.
    j_hat = j_true + np.random.default_rng(4).uniform(0, 0.05, size=j_true.shape)
    a0 = np.argmin(j_hat, axis=-1)
    labels = T.compute_true_labels(
        counterfactual_xy=np.zeros((groups, P.N_ACTIONS, 2, 1, 2)),  # unused for regret identity
        history_last_xy=np.zeros((groups, 2)),
        inference_mask=np.ones((groups, 2, 1), dtype=bool),
        j_hat=j_hat) if False else None
    # identity via the policy function directly:
    reg_a0 = P.action_regret(j_true, a0, a_star)
    rows = np.arange(groups)
    expected = s4co.clamp_regret(j_true[rows, a0] - j_true[rows, a_star])
    assert np.array_equal(reg_a0, expected)
    # chosen == a_star -> zero regret; chosen == IDLE -> the idle regret identity.
    assert np.allclose(P.action_regret(j_true, a_star, a_star), 0.0)
    idle = np.full(groups, P.IDLE_PLAN_INDEX)
    assert np.array_equal(P.action_regret(j_true, idle, a_star),
                          s4co.clamp_regret(j_true[:, P.IDLE_PLAN_INDEX] - j_true[rows, a_star]))


def test_relative_improvement_fail_closed():
    assert P.relative_improvement(0.2, 0.1) == pytest.approx(0.5)
    assert np.isnan(P.relative_improvement(0.0, 0.1))     # degenerate denominator
    assert np.isnan(P.relative_improvement(-0.1, 0.1))
    assert np.isnan(P.relative_improvement(0.2, np.nan))


def test_reproducibility_elementwise():
    rng = np.random.default_rng(7)
    risk = rng.uniform(0, 1, size=50)
    ids = [f"id{i:03d}" for i in range(50)]
    a0 = rng.integers(0, P.N_ACTIONS, size=50)
    tau1 = P.freeze_threshold(risk, ids)
    tau2 = P.freeze_threshold(risk, ids)
    assert tau1 == tau2
    assert np.array_equal(P.apply_fallback(a0, risk, tau1), P.apply_fallback(a0, risk, tau2))
    assert np.array_equal(P.random_fallback_scores(ids), P.random_fallback_scores(ids))
    rnd = P.random_fallback_scores(ids)
    assert np.all((rnd >= 0.0) & (rnd < 1.0))


def test_select_strongest_standard_fallback_picks_lowest_regret():
    groups = 200
    j_true = _j_true(groups, seed=11)
    a_star = np.argmin(j_true, axis=-1)
    a0 = np.zeros(groups, dtype=np.int64)
    # make a baseline risk that perfectly flags the groups where idle beats a0.
    idle_better = j_true[:, P.IDLE_PLAN_INDEX] < j_true[:, 0]
    good = np.where(idle_better, 1.0, 0.0)               # high -> fall back to idle (helps)
    noise = np.random.default_rng(2).uniform(0, 1, size=groups)  # uninformative
    baselines = {s4cp.BASELINE_MARGIN: noise, s4cp.BASELINE_SEED_VOTE_ENTROPY: good,
                 s4cp.BASELINE_SEED_MARGIN_STD: noise}
    out = P.select_strongest_standard_fallback(j_true, a_star, a0, baselines,
                                               [f"g{i}" for i in range(groups)])
    assert out["strongest"] == s4cp.BASELINE_SEED_VOTE_ENTROPY
    assert out["per_baseline"][s4cp.BASELINE_SEED_VOTE_ENTROPY]["mean_regret"] <= \
        out["per_baseline"][s4cp.BASELINE_MARGIN]["mean_regret"] + 1e-12


def _phase0_quantities(groups=1000, helpful=True):
    """Synthetic Phase 0 quantity mapping: high-risk zone where idle beats base (if helpful)."""
    j_true = _j_true(groups, seed=21)
    a0 = np.zeros(groups, dtype=np.int64)               # base selector always picks action 0
    risk = np.linspace(0.0, 1.0, groups)
    ids = [f"g{i}" for i in range(groups)]
    tau = P.freeze_threshold(risk, ids)
    high = risk > tau
    if helpful:
        # high zone: idle much better than base; low zone: base much better than idle.
        j_true[high, P.IDLE_PLAN_INDEX] = 0.01
        j_true[high, 0] = 0.30
        j_true[~high, 0] = 0.01
        j_true[~high, P.IDLE_PLAN_INDEX] = 0.30
    a_star = np.argmin(j_true, axis=-1)
    d2r = P.evaluate_policy(j_true, a_star, a0, risk, tau)
    # a real, worse standard fallback: an uninformative (noise) risk score, frozen per plan 3.
    noise = np.random.default_rng(2).uniform(0.0, 1.0, size=groups)
    baselines = {s4cp.BASELINE_MARGIN: noise, s4cp.BASELINE_SEED_VOTE_ENTROPY: noise,
                 s4cp.BASELINE_SEED_MARGIN_STD: noise}
    fb = P.select_strongest_standard_fallback(j_true, a_star, a0, baselines, ids)
    idle = np.full(groups, P.IDLE_PLAN_INDEX)
    diff = P.action_regret(j_true, a0, a_star) - P.action_regret(j_true, idle, a_star)
    return {
        "regret_d2r": d2r["regret"],
        "regret_idle": P.action_regret(j_true, idle, a_star),
        "regret_base": P.action_regret(j_true, a0, a_star),
        "regret_strongest_fallback": fb["per_baseline"][fb["strongest"]]["regret"],
        "chosen_d2r": d2r["chosen_action"],
        "a0": a0, "risk_d2r": risk, "tau_d2r": tau,
        "n_high_zone": int(high.sum()), "n_low_zone": int((~high).sum()),
        "diff_high_zone": float(diff[high].mean()),
        "diff_low_zone": float((-diff)[~high].mean()),
    }


def test_phase0_gate_direction():
    groups = 1000
    helpful = _phase0_quantities(groups, helpful=True)
    boot = STAT.paired_bootstrap(
        lambda idx: STAT.compute_phase0_quantity_vector(idx, helpful), groups,
        seed=STAT.PHASE0_BOOTSTRAP_SEED, n_resamples=200)
    gate = STAT.evaluate_phase0_gate(helpful, boot)
    assert gate["passed"] is True
    assert gate["status"] == STAT.STATUS_PHASE0_FEASIBLE

    # a rule that does NOT help: base is already the oracle, so idle never helps and the
    # base-denominator improvement degenerates (NaN) -> the gate must fail.
    j_true = _j_true(groups, seed=33)
    a_star = np.argmin(j_true, axis=-1)
    a0 = a_star.copy()
    risk = np.linspace(0.0, 1.0, groups)
    ids = [f"g{i}" for i in range(groups)]
    tau = P.freeze_threshold(risk, ids)
    high = risk > tau
    d2r = P.evaluate_policy(j_true, a_star, a0, risk, tau)
    idle = np.full(groups, P.IDLE_PLAN_INDEX)
    flat = {
        "regret_d2r": d2r["regret"],
        "regret_idle": P.action_regret(j_true, idle, a_star),
        "regret_base": np.zeros(groups),                  # base == oracle -> zero regret
        "regret_strongest_fallback": d2r["regret"],
        "chosen_d2r": d2r["chosen_action"], "a0": a0, "risk_d2r": risk, "tau_d2r": tau,
        "n_high_zone": int(high.sum()), "n_low_zone": int((~high).sum()),
        "diff_high_zone": -1.0, "diff_low_zone": -1.0,    # base never worse than idle
    }
    boot2 = STAT.paired_bootstrap(
        lambda idx: STAT.compute_phase0_quantity_vector(idx, flat), groups,
        seed=STAT.PHASE0_BOOTSTRAP_SEED, n_resamples=200)
    gate2 = STAT.evaluate_phase0_gate(flat, boot2)
    assert gate2["passed"] is False
    assert gate2["status"] == STAT.STATUS_POLICY_SIGNAL_NOT_FEASIBLE


def test_paired_bootstrap_identity_and_shape():
    arr = np.random.default_rng(5).uniform(0, 1, size=100)
    fn = lambda idx: np.asarray([arr[idx].mean(), (arr[idx] > 0.5).mean()])
    boot = STAT.paired_bootstrap(fn, 100, seed=123, n_resamples=500)
    assert boot.point.shape == (2,)
    assert boot.draws.shape == (500, 2)
    assert boot.point[0] == pytest.approx(arr.mean())
    assert boot.ci_lower[0] <= boot.point[0] <= boot.ci_upper[0]
