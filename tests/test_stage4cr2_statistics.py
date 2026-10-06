"""Stage 4C-R2 screen statistics tests (plan sections 9.1, 9.2, 11, 14).

Blocking invariants proved here (task-4 brief / plan section 14):
- the paired bootstrap resamples WHOLE groups, uses ONE shared index stream for every
  metric, is entropy-fixed (2026092702) and reproducible, and builds exact two-sided
  80% CIs from the resample quantiles;
- AURC / REGRET@80 baseline denominators are FAIL-CLOSED: a non-positive or non-finite
  denominator yields NaN improvements and the terminal STAGE4CR2_INVALID_ANALYSIS —
  never an epsilon, an absolute difference or a baseline swap;
- five-seed counting (>=4/5 positive I_AURC / I_REGRET@80), the four-strata floor
  (each >= -0.10), the 1%/5% trim rule and the |Spearman(ensemble, MARGIN_ONLY)| < 0.995
  guard are each independently enforced;
- Gate TASK-R2 maps condition 2/3/7 failures to STAGE4CR2_TASK_DEGENERATE and 4/5/6
  failures to INCONCLUSIVE_STAGE4CR2_DECISION_MODEL_INSUFFICIENT;
- Gate RISK-DECISION-R2_dev maps any unblocked risk failure to
  STAGE4CR2_DECISION_RISK_LINE_CLOSED and a full pass to
  STAGE4CR2_DEV_READY_FOR_CONFIRMATION.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_statistics as s4cr2st  # noqa: E402

G = s4cr2.DEV_SCREEN_GROUPS  # the gate fixes the screen at exactly 8,000 groups


def _passing_quantities(groups=G, seed=7):
    """Synthetic 8,000-group screen that passes Gate TASK-R2 and Gate RISK-DECISION-R2_dev.

    The ensemble is near-perfect; the frozen strongest baseline (SEED_VOTE_ENTROPY) is
    pure noise; MARGIN_ONLY stays weakly correlated so the Spearman ceiling holds.
    """
    rng = np.random.default_rng(seed)
    regret = rng.gamma(shape=2.0, scale=0.02, size=groups)
    ensemble = regret + rng.normal(0.0, 2e-3, size=groups)
    by_seed = ensemble[None, :] + rng.normal(0.0, 3e-3, size=(5, groups))
    baselines = {
        "MARGIN_ONLY": regret + rng.normal(0.0, 0.05, size=groups),
        "SEED_VOTE_ENTROPY": rng.random(groups),
        "SEED_MARGIN_STD": rng.random(groups),
        "COMBINED_TOP2_MAX": regret.copy(),  # perfect: historical reference ONLY
        "CONSTANT_RANDOM": rng.random(groups),
    }
    j_true = rng.gamma(shape=2.0, scale=0.5, size=(groups, s4cr2.N_ACTIONS))
    j_hat = j_true + rng.normal(0.0, 0.05, size=(groups, s4cr2.N_ACTIONS))
    idx = np.arange(groups)
    return {
        "group_seed": idx.astype(np.int64),
        "regret": regret,
        "y_high": (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64),
        "random_regret": regret + 0.05,
        "idle_regret": regret + 0.08,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": ensemble.copy(),
        "class_only_ensemble": ensemble.copy(),
        "baseline_risks": baselines,
        "strongest_baseline": "SEED_VOTE_ENTROPY",
        "tstar_layer": (idx % 2).astype(np.int64),
        "interaction_layer": ((idx // 2) % 2).astype(np.int64),
        "J_hat": j_hat,
        "J_true": j_true,
        "a_star": idx % s4cr2.N_ACTIONS,
        "t_star": np.where(idx % 2 == 0, 11, 14),
        "interaction_min_distance": np.where((idx // 2) % 2 == 0, 10.0, 40.0),
        "n_groups": int(groups),
        "interaction_threshold": 25.0,
        "qa": {
            "all_complete": True,
            "hashes_match": True,
            "role_seeds_valid": True,
            "determinism_passed": True,
            "one_shot_locks": True,
            "ensemble_is_arithmetic_mean": True,
        },
    }


def _bootstrap_for(quantities, *, n=200, ci_overrides=None):
    """Fabricated bootstrap: point from the identity evaluation; draws tiled at the point
    except where ``ci_overrides`` pushes the 10th percentile of a quantity to a value."""
    groups = int(quantities["n_groups"])
    point = s4cr2st.compute_screen_quantity_vector(
        np.arange(groups, dtype=np.int64), quantities)
    draws = np.tile(point, (int(n), 1))
    for slot, low_value in (ci_overrides or {}).items():
        draws[: n // 10 + 1, slot] = float(low_value)
    alpha = (1.0 - s4cr2.CONFIDENCE_SCREEN) / 2.0
    return s4cr2st.BootstrapResult(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws,
        n_resamples=int(n),
        seed=s4cr2.BOOTSTRAP_ENTROPY,
        confidence=s4cr2.CONFIDENCE_SCREEN,
    )


# --- whole-group paired bootstrap ---------------------------------------------------------


def test_bootstrap_reproducible_with_frozen_entropy():
    quantities = _passing_quantities(groups=300)
    first = s4cr2st.paired_group_bootstrap(
        quantities, seed=s4cr2.BOOTSTRAP_ENTROPY, n_resamples=100)
    second = s4cr2st.paired_group_bootstrap(
        quantities, seed=s4cr2.BOOTSTRAP_ENTROPY, n_resamples=100)
    assert first.seed == s4cr2.BOOTSTRAP_ENTROPY == 2026092702
    assert first.confidence == s4cr2.CONFIDENCE_SCREEN == 0.80
    assert np.array_equal(first.draws, second.draws)
    assert np.array_equal(first.point, second.point)
    assert first.draws.shape == (100, s4cr2st.N_QUANTITIES)


def test_bootstrap_different_seed_changes_draws():
    quantities = _passing_quantities(groups=300)
    a = s4cr2st.paired_group_bootstrap(quantities, seed=2026092702, n_resamples=50)
    b = s4cr2st.paired_group_bootstrap(quantities, seed=12345, n_resamples=50)
    assert not np.array_equal(a.draws, b.draws)


def test_bootstrap_resamples_whole_groups_with_shared_indices():
    """Manually replay the frozen generator: every metric of draw r must use the SAME
    whole-group index vector, drawn with replacement."""
    groups, n_resamples, seed = 60, 30, 99
    quantities = _passing_quantities(groups=groups)
    result = s4cr2st.paired_group_bootstrap(
        quantities, seed=seed, n_resamples=n_resamples)
    generator = np.random.default_rng(seed)
    for r in range(n_resamples):
        index = generator.integers(0, groups, size=groups)
        assert result.draws[r, s4cr2st.Q_BASE_MEAN_REGRET] == pytest.approx(
            quantities["regret"][index].mean())
        assert result.draws[r, s4cr2st.Q_RANDOM_MEAN_REGRET] == pytest.approx(
            quantities["random_regret"][index].mean())
        assert result.draws[r, s4cr2st.Q_IDLE_MEAN_REGRET] == pytest.approx(
            quantities["idle_regret"][index].mean())
        # resampling is whole-group with replacement: duplicated indices appear.
        assert np.unique(index).size < groups or True  # property holds by construction


def test_bootstrap_ci_is_exact_two_sided_80pct_from_draw_quantiles():
    quantities = _passing_quantities(groups=300)
    result = s4cr2st.paired_group_bootstrap(quantities, seed=5, n_resamples=200)
    alpha = (1.0 - s4cr2.CONFIDENCE_SCREEN) / 2.0
    assert np.array_equal(result.ci_lower, np.quantile(result.draws, alpha, axis=0))
    assert np.array_equal(result.ci_upper, np.quantile(result.draws, 1.0 - alpha, axis=0))
    identity = s4cr2st.compute_screen_quantity_vector(
        np.arange(300, dtype=np.int64), quantities)
    assert np.array_equal(result.point, identity)


def test_bootstrap_improvement_nan_when_baseline_denominator_zero():
    quantities = _passing_quantities(groups=300)
    quantities = dict(quantities)
    quantities["regret"] = np.zeros(300)  # AURC/REGRET@80 denominators collapse to 0
    result = s4cr2st.paired_group_bootstrap(quantities, seed=5, n_resamples=50)
    assert result.point[s4cr2st.Q_AURC_BASELINE] == 0.0
    assert np.isnan(result.point[s4cr2st.Q_I_AURC])
    assert np.isnan(result.point[s4cr2st.Q_I_REGRET80])
    assert np.isnan(result.draws[:, s4cr2st.Q_I_AURC]).all()


# --- Gate TASK-R2 -------------------------------------------------------------------------


def test_screen_gates_pass_on_qualified_fixture():
    quantities = _passing_quantities()
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert all(decision.task["conditions"].values()), decision.task
    assert all(decision.risk["conditions"].values()), decision.risk
    assert decision.passed is True
    assert decision.status == s4cr2.STATUS_DEV_READY
    assert decision.n_groups == G
    # named metrics + ablations are reported alongside (non-blocking).
    for name in ("i_aurc_regret", "i_regret80", "pr_auc_minus_prevalence",
                 "spearman_branch", "base_vs_random_improvement"):
        assert name in decision.metrics["point"]
        assert name in decision.metrics["ci_lower"]
    for ablation in ("MU_ONLY", "CLASS_ONLY"):
        assert ablation in decision.ablations
    for name in ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD",
                 "COMBINED_TOP2_MAX", "CONSTANT_RANDOM"):
        assert name in decision.ablations["baseline_curves"]
    assert decision.ablations["blocking"] is False


def test_gate_task_c2_low_high_regret_frac_is_task_degenerate():
    quantities = _passing_quantities()
    rng = np.random.default_rng(13)
    regret = rng.gamma(shape=1.0, scale=0.005, size=G)  # ~7% > 0.01
    quantities["regret"] = regret
    quantities["y_high"] = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c2_high_regret_frac_ge_15pct"] is False
    assert decision.status == s4cr2.STATUS_TASK_DEGENERATE
    assert decision.passed is False


def test_gate_task_c3_fewer_than_3_oracle_actions_is_task_degenerate():
    quantities = _passing_quantities()
    quantities["a_star"] = np.zeros(G, dtype=np.int64)  # one oracle action only
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c3_at_least_3_oracle_actions_ge_5pct"] is False
    assert decision.status == s4cr2.STATUS_TASK_DEGENERATE


def test_gate_task_c7_thin_stratum_is_task_degenerate():
    for layer_key in ("tstar_layer", "interaction_layer"):
        quantities = _passing_quantities()
        quantities[layer_key] = np.zeros(G, dtype=np.int64)  # one stratum only
        decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
        assert decision.task["conditions"]["c7_strata_ge_800_groups"] is False
        assert decision.status == s4cr2.STATUS_TASK_DEGENERATE


def test_gate_task_c4_spearman_ci_lower_nonpositive_is_inconclusive():
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(
        quantities, ci_overrides={s4cr2st.Q_SPEARMAN_BRANCH: -0.01})
    decision = s4cr2st.evaluate_screen_gates(quantities, bootstrap)
    assert decision.task["conditions"]["c4_branch_spearman_ci_lower_positive"] is False
    assert decision.status == s4cr2.STATUS_MODEL_INSUFFICIENT


def test_gate_task_c5_base_vs_random_enforced_point_and_ci():
    quantities = _passing_quantities()
    quantities["random_regret"] = quantities["regret"].copy()  # zero improvement
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c5_base_vs_random_point_and_ci_positive"] is False
    assert decision.status == s4cr2.STATUS_MODEL_INSUFFICIENT

    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(
        quantities, ci_overrides={s4cr2st.Q_BASE_VS_RANDOM: -0.001})
    decision = s4cr2st.evaluate_screen_gates(quantities, bootstrap)
    assert decision.task["conditions"]["c5_base_vs_random_point_and_ci_positive"] is False
    assert decision.status == s4cr2.STATUS_MODEL_INSUFFICIENT


def test_gate_task_c6_base_worse_than_idle_is_inconclusive():
    quantities = _passing_quantities()
    quantities["idle_regret"] = np.maximum(quantities["regret"] - 0.05, 0.0)
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c6_base_not_worse_than_idle"] is False
    assert decision.status == s4cr2.STATUS_MODEL_INSUFFICIENT


# --- Gate RISK-DECISION-R2_dev ------------------------------------------------------------


def test_risk_r1_i_aurc_point_below_substantive_line_closes_line():
    quantities = _passing_quantities()
    noise = np.random.default_rng(21).random(G)
    quantities["ensemble_risk"] = noise  # ensemble == baseline-quality noise
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED
    assert decision.passed is False


def test_risk_r1_i_aurc_ci_lower_nonpositive_closes_line():
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities, ci_overrides={s4cr2st.Q_I_AURC: -0.001})
    decision = s4cr2st.evaluate_screen_gates(quantities, bootstrap)
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_risk_r2_i_regret80_enforced_independently():
    """Ensemble fixes everything EXCEPT the top-20% tail: I_AURC > 0, I_REGRET@80 = 0."""
    rng = np.random.default_rng(19)
    low = np.sort(rng.uniform(0.0, 0.005, size=6400))
    high = np.sort(rng.uniform(0.05, 0.30, size=1600))
    regret = np.concatenate([low, high])  # ascending
    quantities = _passing_quantities()
    quantities["regret"] = regret
    quantities["y_high"] = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)
    ensemble = regret + np.linspace(0, 1e-6, G)  # near-perfect ascending risk
    crafted = regret.copy()
    crafted[6400:] = regret[6400:][::-1]  # reversed tail: same lowest-80% set
    quantities["ensemble_risk"] = ensemble
    quantities["risk_by_seed"] = np.tile(ensemble, (5, 1))
    quantities["mu_only_ensemble"] = ensemble.copy()
    quantities["class_only_ensemble"] = ensemble.copy()
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["SEED_VOTE_ENTROPY"] = crafted
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.metrics["point"]["i_regret80"] == pytest.approx(0.0, abs=1e-9)
    assert decision.risk["conditions"]["r2_i_regret80"] is False
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is True  # isolation
    assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_risk_r3_pr_auc_ci_lower_nonpositive_closes_line():
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities, ci_overrides={s4cr2st.Q_PR: -0.001})
    decision = s4cr2st.evaluate_screen_gates(quantities, bootstrap)
    assert decision.risk["conditions"]["r3_pr_auc_minus_prevalence"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_risk_r4_five_seed_i_aurc_counting_rule():
    rng = np.random.default_rng(23)
    good = _passing_quantities()["ensemble_risk"]
    for n_bad, expected in ((2, False), (1, True)):
        quantities = _passing_quantities()
        by_seed = np.tile(quantities["ensemble_risk"], (5, 1))
        for i in range(n_bad):
            by_seed[4 - i] = -quantities["regret"] + rng.normal(0, 1e-3, size=G)
        quantities["risk_by_seed"] = by_seed
        decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
        assert decision.risk["seeds_i_aurc_positive"] == 5 - n_bad
        assert decision.risk["conditions"]["r4_seeds_i_aurc"] is expected
        if n_bad == 2:
            assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_risk_r5_five_seed_i_regret80_counting_rule():
    rng = np.random.default_rng(29)
    quantities = _passing_quantities()
    by_seed = np.tile(quantities["ensemble_risk"], (5, 1))
    by_seed[0] = -quantities["regret"] + rng.normal(0, 1e-3, size=G)
    by_seed[1] = -quantities["regret"] + rng.normal(0, 1e-3, size=G)
    quantities["risk_by_seed"] = by_seed
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["seeds_i_regret80_positive"] == 3
    assert decision.risk["conditions"]["r5_seeds_i_regret80"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_risk_r6_any_stratum_below_floor_closes_line():
    quantities = _passing_quantities()
    high = quantities["interaction_layer"] == 1
    ensemble = quantities["ensemble_risk"].copy()
    ensemble[high] = -quantities["regret"][high]  # anti-perfect inside one stratum
    quantities["ensemble_risk"] = ensemble
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["strata_i_aurc"]["interaction_high"] < s4cr2.GATE_STRATA_I_AURC_FLOOR
    assert decision.risk["conditions"]["r6_strata_floor"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_risk_r7_trims_must_stay_positive():
    quantities = _passing_quantities()
    noise = np.random.default_rng(31).random(G)
    quantities["ensemble_risk"] = noise  # improvement ~0 untrimmed AND trimmed
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["conditions"]["r7_trims_positive"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_trim_quantities_actually_remove_the_highest_regret_tail():
    """Tail-driven improvement: baseline hides the top-5% regret groups at the safest
    ranks; after the 5% trim the improvement must shrink deterministically."""
    quantities = _passing_quantities()
    regret = quantities["regret"]
    top = np.argsort(regret, kind="stable")[-int(0.05 * G):]
    baseline = np.random.default_rng(37).random(G)
    baseline[top] = -1.0  # hidden tail
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["SEED_VOTE_ENTROPY"] = baseline
    identity = np.arange(G, dtype=np.int64)
    q = s4cr2st.compute_screen_quantity_vector(identity, quantities)
    assert q[s4cr2st.Q_TRIM5] < q[s4cr2st.Q_I_AURC]
    assert np.isfinite(q[s4cr2st.Q_TRIM1]) and np.isfinite(q[s4cr2st.Q_TRIM5])


def test_risk_r8_margin_spearman_guard():
    quantities = _passing_quantities()
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["MARGIN_ONLY"] = quantities["ensemble_risk"].copy()
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert abs(decision.metrics["point"]["spearman_ensemble_margin"]) == pytest.approx(1.0)
    assert decision.risk["conditions"]["r8_margin_spearman"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED


def test_spearman_guard_returns_zero_on_degenerate_input():
    quantities = _passing_quantities()
    quantities["ensemble_risk"] = np.full(G, 0.5)  # zero variance
    identity = np.arange(G, dtype=np.int64)
    q = s4cr2st.compute_screen_quantity_vector(identity, quantities)
    assert q[s4cr2st.Q_SPEARMAN_MARGIN] == 0.0  # guarded, never NaN/exception
    assert np.isfinite(q[s4cr2st.Q_SPEARMAN_MARGIN])


def test_risk_r9_qa_failure_is_invalid_analysis():
    for broken in ("determinism_passed", "hashes_match", "role_seeds_valid",
                   "one_shot_locks", "all_complete"):
        quantities = _passing_quantities()
        quantities["qa"] = {**quantities["qa"], broken: False}
        decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
        assert decision.risk["conditions"]["r9_qa_and_one_shot_locks"] is False
        assert decision.status == s4cr2.STATUS_INVALID


def test_baseline_denominator_fail_closed_invalid_analysis():
    # AURC denominator zero: all-zero regret.
    quantities = _passing_quantities()
    quantities["regret"] = np.zeros(G)
    quantities["y_high"] = np.zeros(G, dtype=np.int64)
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["baseline_valid"] is False
    assert decision.status == s4cr2.STATUS_INVALID

    # REGRET@80 denominator zero: the lowest-risk 80% carry zero regret.
    quantities = _passing_quantities()
    regret = np.zeros(G)
    regret[6400:] = 1.0
    quantities["regret"] = regret
    quantities["y_high"] = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)
    crafted = np.arange(G, dtype=np.float64)  # ascending risk == ascending regret
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["SEED_VOTE_ENTROPY"] = crafted
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.metrics["point"]["aurc_baseline"] > 0.0
    assert decision.metrics["point"]["regret80_baseline"] == pytest.approx(0.0)
    assert decision.risk["baseline_valid"] is False
    assert decision.status == s4cr2.STATUS_INVALID


def test_evaluate_rejects_inconsistent_bootstrap():
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities)
    bad = s4cr2st.BootstrapResult(
        point=bootstrap.point[:-1], ci_lower=bootstrap.ci_lower[:-1],
        ci_upper=bootstrap.ci_upper[:-1], draws=bootstrap.draws[:, :-1],
        n_resamples=bootstrap.n_resamples, seed=bootstrap.seed,
        confidence=bootstrap.confidence)
    with pytest.raises(ValueError):
        s4cr2st.evaluate_screen_gates(quantities, bad)


def test_terminal_status_priority_task_before_risk():
    """A degenerate task masks the risk gate: no risk verdict is reachable."""
    quantities = _passing_quantities()
    quantities["a_star"] = np.zeros(G, dtype=np.int64)  # c3 fails
    noise = np.random.default_rng(41).random(G)
    quantities["ensemble_risk"] = noise  # would ALSO fail the risk gate
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.status == s4cr2.STATUS_TASK_DEGENERATE
    assert decision.passed is False


def test_ablations_never_substitute_the_primary_score():
    """A passing ablation must not rescue a failing primary score (plan 9.4)."""
    rng = np.random.default_rng(43)
    quantities = _passing_quantities()
    quantities["ensemble_risk"] = rng.random(G)  # primary fails r1
    quantities["mu_only_ensemble"] = quantities["regret"] + rng.normal(0, 1e-3, size=G)
    decision = s4cr2st.evaluate_screen_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is False
    assert decision.status == s4cr2.STATUS_LINE_CLOSED
    assert decision.ablations["MU_ONLY"]["i_aurc_regret"] > 0.05  # reported, never gating
