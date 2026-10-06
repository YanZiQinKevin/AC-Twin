"""Stage 4C-R2 independent-confirmation statistics tests (plan sections 6, 7, 8, 9 Task 3).

Blocking invariants proved here:
- hand-computed small-sample checks of AURC / REGRET@80 (coverage rounding via
  ``stage4_statistics.coverage_index``), denominator protection, regret clamping and the
  ten-action efficiency normalization;
- the ONE paired bootstrap is entropy-fixed (2026093202), uses 10,000 whole-group draws by
  default, shares ONE index vector across all metrics/strata/trims/seeds/ablations (incl.
  the MARGIN_ONLY / SEED_MARGIN_STD comparisons) and reads two-sided 95% CIs (2.5%/97.5%)
  off the draw quantiles — never the dev 80% CI;
- ONLY the frozen primary baseline (SEED_VOTE_ENTROPY) enters the gates; the other two
  standard baselines are Table-C2-only and never trigger a baseline swap;
- Gate TASK-CONFIRM / Gate RISK-DECISION-CONFIRM and the section-8 status priority
  (invalid > degenerate 2/3/7 > insufficient 4/5/6 > denominator-invalid > not-confirmed
  > confirmed);
- one-shot ordering discipline: score freeze -> burn lock -> truth load -> assemble ->
  adjudicate; second adjudication refused; legal same-transaction IO resume never
  overwrites completed receipts; changed seed/data/config recovery refused.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4_statistics as s4s  # noqa: E402
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402
from ac_twin import stage4cp_statistics as s4cs  # noqa: E402
from ac_twin import stage4cr2_confirm_protocol as prot  # noqa: E402
from ac_twin import stage4cr2_confirm_inference as cinf  # noqa: E402
from ac_twin import stage4cr2_confirm_statistics as stat  # noqa: E402
from ac_twin import stage4cr2_training as s4cr2t  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

NA = prot.N_ACTIONS
GATE_G = prot.CONFIRM_ACCEPTED_GROUPS  # 2,000
CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4cr2_confirm/stage4cr2_confirm_config.json")

#: mirror of the collector's sealed-truth schema (the collector is never imported here).
TRUTH_FIELDS = (
    "object_counterfactual_futures", "cf_actions", "cf_durations",
    "factual_actions", "t_star", "group_seed", "inference_mask",
)


# --- hand-computed 4-group vector fixture -------------------------------------------------


def _g4_quantities():
    """G=4 fixture with exact hand-computed expectations (see module docstring tests).

    regret = [0.1, 0.2, 0.3, 0.4]; ensemble risk ascending (identity order);
    SEED_VOTE_ENTROPY = [0.3, 0.2, 0.1, 0.0] (reverse order); every label is high-regret
    (degenerate PR -> 0, AP -> prevalence = 1).
    """
    regret = np.array([0.1, 0.2, 0.3, 0.4])
    ensemble = np.array([0.0, 0.1, 0.2, 0.3])
    baselines = {
        "MARGIN_ONLY": np.array([0.0, 0.2, 0.1, 0.3]),
        "SEED_VOTE_ENTROPY": np.array([0.3, 0.2, 0.1, 0.0]),
        "SEED_MARGIN_STD": np.array([0.25, 0.05, 0.15, 0.35]),
        "COMBINED_TOP2_MAX": np.array([0.35, 0.25, 0.05, 0.15]),
        "CONSTANT_RANDOM": np.array([0.12, 0.33, 0.07, 0.29]),
    }
    j_true = (np.arange(40, dtype=np.float64).reshape(4, NA) + 1.0)
    return {
        "group_seed": np.array([11, 22, 33, 44], dtype=np.int64),
        "regret": regret,
        "y_high": np.ones(4, dtype=np.int64),
        "random_regret": regret + 0.05,
        "idle_regret": regret + 0.08,
        "a_hat": np.zeros(4, dtype=np.int64),
        "a_star": np.full(4, 9, dtype=np.int64),
        "ensemble_risk": ensemble,
        "risk_by_seed": np.tile(ensemble, (5, 1)),
        "mu_only_ensemble": ensemble.copy(),
        "class_only_ensemble": ensemble.copy(),
        "baseline_risks": baselines,
        "strongest_baseline": "SEED_VOTE_ENTROPY",
        "tstar_layer": np.array([0, 0, 1, 1], dtype=np.int64),
        "interaction_layer": np.array([0, 1, 0, 1], dtype=np.int64),
        "t_star": np.array([10, 12, 13, 15], dtype=np.int64),
        "interaction_min_distance": np.array([10.0, 40.0, 10.0, 40.0]),
        "J_hat": j_true.copy(),
        "J_true": j_true,
        "trim1_keep": np.ones(4, dtype=bool),
        "trim5_keep": np.ones(4, dtype=bool),
        "n_groups": 4,
        "interaction_threshold": prot.INTERACTION_THRESHOLD,
        "qa": {key: True for key in stat.QA_KEYS},
    }


def _passing_quantities(groups=GATE_G, seed=7):
    """Synthetic G=2,000 confirm set passing Gate TASK-CONFIRM and Gate RISK (95% CI).

    Mirrors the R2 screen fixture: near-perfect ensemble, pure-noise primary baseline,
    weakly correlated MARGIN_ONLY (Spearman ceiling holds).
    """
    rng = np.random.default_rng(seed)
    regret = rng.gamma(shape=2.0, scale=0.02, size=groups)
    ensemble = regret + rng.normal(0.0, 2e-3, size=groups)
    by_seed = ensemble[None, :] + rng.normal(0.0, 3e-3, size=(5, groups))
    baselines = {
        "MARGIN_ONLY": regret + rng.normal(0.0, 0.05, size=groups),
        "SEED_VOTE_ENTROPY": rng.random(groups),
        "SEED_MARGIN_STD": rng.random(groups),
        "COMBINED_TOP2_MAX": regret.copy(),
        "CONSTANT_RANDOM": rng.random(groups),
    }
    j_true = rng.gamma(shape=2.0, scale=0.5, size=(groups, NA))
    j_hat = j_true + rng.normal(0.0, 0.05, size=(groups, NA))
    idx = np.arange(groups)
    return {
        "group_seed": idx.astype(np.int64),
        "regret": regret,
        "y_high": (regret > prot.HIGH_REGRET_THRESHOLD).astype(np.int64),
        "random_regret": regret + 0.05,
        "idle_regret": regret + 0.08,
        "a_hat": ((idx * 3) % NA).astype(np.int64),
        "a_star": (idx % NA).astype(np.int64),
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": ensemble.copy(),
        "class_only_ensemble": ensemble.copy(),
        "baseline_risks": baselines,
        "strongest_baseline": "SEED_VOTE_ENTROPY",
        "tstar_layer": (idx % 2).astype(np.int64),
        "interaction_layer": ((idx // 2) % 2).astype(np.int64),
        "t_star": np.where(idx % 2 == 0, 11, 14),
        "interaction_min_distance": np.where((idx // 2) % 2 == 0, 10.0, 40.0),
        "J_hat": j_hat,
        "J_true": j_true,
        "trim1_keep": s4cs.trim_high_regret(regret, prot.GATE_RISK_TRIM_FRACTIONS[0]),
        "trim5_keep": s4cs.trim_high_regret(regret, prot.GATE_RISK_TRIM_FRACTIONS[1]),
        "n_groups": int(groups),
        "interaction_threshold": prot.INTERACTION_THRESHOLD,
        "qa": {key: True for key in stat.QA_KEYS},
    }


def _bootstrap_for(quantities, *, n=200, ci_overrides=None):
    """Fabricated 95% bootstrap: draws tiled at the point except where ``ci_overrides``
    pushes the 2.5th percentile of a quantity to ``low_value`` (n//40+1 = 3% of draws)."""
    groups = int(quantities["n_groups"])
    point = stat.compute_confirm_quantity_vector(
        np.arange(groups, dtype=np.int64), quantities)
    draws = np.tile(point, (int(n), 1))
    for slot, low_value in (ci_overrides or {}).items():
        draws[: n // 40 + 1, slot] = float(low_value)
    alpha = (1.0 - prot.CONFIDENCE) / 2.0
    return stat.BootstrapResult(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws,
        n_resamples=int(n),
        seed=prot.BOOTSTRAP_ENTROPY,
        confidence=prot.CONFIDENCE,
    )


# --- hand-computed quantity-vector checks ---------------------------------------------------


def test_hand_computed_quantity_vector_g4():
    q = stat.compute_confirm_quantity_vector(
        np.arange(4, dtype=np.int64), _g4_quantities())
    assert q.shape == (stat.N_QUANTITIES,)
    # AURC model: R = [0.1, 0.15, 0.2, 0.25] -> 0.175; baseline (reverse order):
    # R = [0.4, 0.35, 0.3, 0.25] -> 0.325.
    assert q[stat.Q_AURC_MODEL] == pytest.approx(0.175)
    assert q[stat.Q_AURC_BASELINE] == pytest.approx(0.325)
    assert q[stat.Q_I_AURC] == pytest.approx(0.15 / 0.325)
    # coverage_index(0.80, 4) == round(3.2) == 3: R80 model = 0.6/3, baseline = 0.9/3.
    assert s4s.coverage_index(0.80, 4) == 3
    assert q[stat.Q_R80_MODEL] == pytest.approx(0.2)
    assert q[stat.Q_R80_BASELINE] == pytest.approx(0.3)
    assert q[stat.Q_I_REGRET80] == pytest.approx(1.0 / 3.0)
    # base/random/idle reference means.
    assert q[stat.Q_BASE_MEAN_REGRET] == pytest.approx(0.25)
    assert q[stat.Q_RANDOM_MEAN_REGRET] == pytest.approx(0.30)
    assert q[stat.Q_IDLE_MEAN_REGRET] == pytest.approx(0.33)
    assert q[stat.Q_BASE_VS_RANDOM] == pytest.approx(1.0 / 6.0)
    assert q[stat.Q_HIGH_REGRET_FRAC] == pytest.approx(1.0)
    # degenerate all-one labels: PR quantity 0, absolute AP == prevalence == 1.
    assert q[stat.Q_PR] == pytest.approx(0.0)
    for slot in (stat.Q_AP_MODEL, stat.Q_AP_MARGIN, stat.Q_AP_SVE, stat.Q_AP_STD,
                 stat.Q_AP_COMBINED, stat.Q_AP_RANDOM):
        assert q[slot] == pytest.approx(1.0)
    # branch Spearman on identical flattened J: exactly 1.
    assert q[stat.Q_SPEARMAN_BRANCH] == pytest.approx(1.0)
    # Spearman([0,.1,.2,.3], [0,.2,.1,.3]) = 0.8 (hand-computed ranks).
    assert q[stat.Q_SPEARMAN_MARGIN] == pytest.approx(0.8)
    # absolute curves of the other four baselines (Table C2).
    assert q[stat.Q_AURC_MARGIN] == pytest.approx(0.1875)
    assert q[stat.Q_R80_MARGIN] == pytest.approx(0.2)
    assert q[stat.Q_AURC_STD] == pytest.approx(0.225)
    assert q[stat.Q_R80_STD] == pytest.approx(0.2)
    assert q[stat.Q_AURC_COMBINED] == pytest.approx(0.3)
    assert q[stat.Q_R80_COMBINED] == pytest.approx(0.3)
    assert q[stat.Q_AURC_RANDOM] == pytest.approx(61.0 / 240.0)
    assert q[stat.Q_R80_RANDOM] == pytest.approx(4.0 / 15.0)
    # non-primary standard-baseline improvements (Table C2 only, same draws).
    assert q[stat.Q_MARGIN_I_AURC] == pytest.approx(1.0 / 15.0)
    assert q[stat.Q_MARGIN_I_R80] == pytest.approx(0.0)
    assert q[stat.Q_STD_I_AURC] == pytest.approx(2.0 / 9.0)
    assert q[stat.Q_STD_I_R80] == pytest.approx(0.0)
    # marginal strata (two splits, not a 4-cell cross).
    assert q[stat.Q_TSTAR_LOW] == pytest.approx(2.0 / 7.0)
    assert q[stat.Q_TSTAR_HIGH] == pytest.approx(2.0 / 15.0)
    assert q[stat.Q_INTERACT_LOW] == pytest.approx(0.4)
    assert q[stat.Q_INTERACT_HIGH] == pytest.approx(2.0 / 7.0)
    # five seeds / ablations identical to the ensemble in this fixture.
    for i in range(5):
        assert q[stat.Q_SEED_I_AURC + i] == pytest.approx(0.15 / 0.325)
        assert q[stat.Q_SEED_I_R80 + i] == pytest.approx(1.0 / 3.0)
    assert q[stat.Q_MU_I_AURC] == pytest.approx(0.15 / 0.325)
    assert q[stat.Q_CLS_I_R80] == pytest.approx(1.0 / 3.0)
    # floor(0.01*4) == floor(0.05*4) == 0 -> trims keep every group.
    assert q[stat.Q_TRIM1] == pytest.approx(0.15 / 0.325)
    assert q[stat.Q_TRIM5] == pytest.approx(0.15 / 0.325)


def test_coverage_rounding_reuses_stage4_statistics():
    """Coverage rounding is stage4_statistics.coverage_index, never own floor/ceil."""
    assert stat.coverage_index is s4s.coverage_index
    quantities = _g4_quantities()
    order = np.argsort(quantities["ensemble_risk"], kind="stable")
    expected = s4cs.regret_at_coverage(order, quantities["regret"], prot.COVERAGE)
    q = stat.compute_confirm_quantity_vector(np.arange(4, dtype=np.int64), quantities)
    assert q[stat.Q_R80_MODEL] == pytest.approx(expected)


def test_denominator_protection_is_fail_closed_nan():
    quantities = _g4_quantities()
    quantities["regret"] = np.zeros(4)  # AURC/R80 denominators collapse to 0
    quantities["y_high"] = np.zeros(4, dtype=np.int64)
    q = stat.compute_confirm_quantity_vector(np.arange(4, dtype=np.int64), quantities)
    assert q[stat.Q_AURC_BASELINE] == 0.0
    assert q[stat.Q_R80_BASELINE] == 0.0
    for slot in (stat.Q_I_AURC, stat.Q_I_REGRET80, stat.Q_TRIM1, stat.Q_TRIM5,
                 stat.Q_TSTAR_LOW, stat.Q_TSTAR_HIGH, stat.Q_INTERACT_LOW,
                 stat.Q_INTERACT_HIGH, stat.Q_MARGIN_I_AURC, stat.Q_STD_I_AURC):
        assert np.isnan(q[slot]), slot
    # no epsilon, no absolute difference: BASE_VS_RANDOM is still well-defined because
    # the RANDOM_ACTION denominator (0.05 mean) is positive — only the frozen-baseline
    # denominators are fail-closed.
    assert q[stat.Q_BASE_VS_RANDOM] == pytest.approx(1.0)


def test_decision_quantities_ten_action_clamp_and_ties():
    """Ten-action normalization, IDLE=(1,3) index, frozen clamp and tie-break order."""
    assert s4co.ACTION_PLANS[s4co.IDLE_PLAN_INDEX] == (1, 3)
    j_hat = np.zeros((2, NA))
    j_true = np.stack([np.linspace(0.9, 0.0, NA), np.full(NA, 0.5)])
    out = stat._decision_quantities(j_hat, j_true)
    # argmin ties break to the LOWEST ACTION_PLANS index (never regret-assisted).
    assert out["a_hat"].tolist() == [0, 0]
    assert out["a_star"].tolist() == [9, 0]
    assert out["regret"][0] == pytest.approx(0.9)
    assert out["regret"][1] == pytest.approx(0.0)
    # RANDOM_ACTION = mean over ALL TEN actions of the group (never re-selected).
    assert out["random_regret"][0] == pytest.approx(0.45)
    assert out["random_regret"][1] == pytest.approx(0.0)
    # ALWAYS_IDLE is ACTION_PLANS index 3 == (IDLE,3) — never (IDLE,1) at index 2.
    assert out["idle_regret"][0] == pytest.approx(j_true[0, 3])
    assert out["idle_regret"][0] != pytest.approx(j_true[0, 2])
    assert out["y_high"].tolist() == [1, 0]
    # frozen clamp primitive: [-1e-10, 0) -> 0; below -1e-10 raises; positive stays.
    assert s4co.clamp_regret(np.array([-5e-11]))[0] == 0.0
    with pytest.raises(ValueError):
        s4co.clamp_regret(np.array([-2e-10]))
    j_hat_pos = np.zeros((1, NA))
    j_hat_pos[0, 0] = 1.0  # a_hat = 1 (first zero)
    j_true_pos = np.full((1, NA), 0.5)
    j_true_pos[0, 0] = 0.0   # a_star = 0
    j_true_pos[0, 1] = 3e-11
    positive = stat._decision_quantities(j_hat_pos, j_true_pos)
    assert positive["a_hat"][0] == 1 and positive["a_star"][0] == 0
    assert positive["regret"][0] == pytest.approx(3e-11)


# --- the one paired bootstrap ---------------------------------------------------------------


def test_bootstrap_frozen_contract_and_95pct_quantiles():
    signature = inspect.signature(stat.paired_group_bootstrap)
    assert signature.parameters["seed"].default == prot.BOOTSTRAP_ENTROPY == 2026093202
    assert (signature.parameters["n_resamples"].default
            == prot.BOOTSTRAP_RESAMPLES == 10000)
    assert prot.CONFIDENCE == 0.95
    quantities = _passing_quantities(groups=300)
    result = stat.paired_group_bootstrap(quantities, seed=prot.BOOTSTRAP_ENTROPY,
                                         n_resamples=200)
    assert result.confidence == 0.95
    assert result.draws.shape == (200, stat.N_QUANTITIES)
    alpha = (1.0 - prot.CONFIDENCE) / 2.0  # 2.5%/97.5% two-sided 95%
    assert alpha != 0.10  # never the dev 80% CI band
    assert np.array_equal(result.ci_lower, np.quantile(result.draws, alpha, axis=0))
    assert np.array_equal(result.ci_upper, np.quantile(result.draws, 1.0 - alpha, axis=0))
    identity = stat.compute_confirm_quantity_vector(np.arange(300, dtype=np.int64),
                                                    quantities)
    assert np.array_equal(result.point, identity)
    again = stat.paired_group_bootstrap(quantities, seed=prot.BOOTSTRAP_ENTROPY,
                                        n_resamples=200)
    assert np.array_equal(result.draws, again.draws)  # frozen entropy reproduces
    other = stat.paired_group_bootstrap(quantities, seed=12345, n_resamples=200)
    assert not np.array_equal(result.draws, other.draws)


def test_bootstrap_same_index_vector_for_all_baselines_strata_trims_seeds():
    """Replay the frozen generator: EVERY quantity of draw r (three standard baselines,
    strata, trims, five seeds, ablations) is recomputed from the SAME index vector."""
    groups, n_resamples, seed = 80, 25, 99
    quantities = _passing_quantities(groups=groups)
    result = stat.paired_group_bootstrap(quantities, seed=seed, n_resamples=n_resamples)
    generator = np.random.default_rng(seed)
    reg, ens = quantities["regret"], quantities["ensemble_risk"]
    sve = quantities["baseline_risks"]["SEED_VOTE_ENTROPY"]
    margin = quantities["baseline_risks"]["MARGIN_ONLY"]
    std = quantities["baseline_risks"]["SEED_MARGIN_STD"]

    def _aurc(risk, r):
        return s4cs.aurc_regret(np.argsort(risk, kind="stable"), r)

    for r in range(n_resamples):
        index = generator.integers(0, groups, size=groups)
        ri, ei, bi = reg[index], ens[index], sve[index]
        a_base, a_model = _aurc(bi, ri), _aurc(ei, ri)
        expected_i_aurc = s4cs.improvement(a_base, a_model)
        assert result.draws[r, stat.Q_I_AURC] == pytest.approx(expected_i_aurc)
        # same index vector -> the two non-primary standard baselines (Table C2 only).
        assert result.draws[r, stat.Q_MARGIN_I_AURC] == pytest.approx(
            s4cs.improvement(_aurc(margin[index], ri), a_model))
        assert result.draws[r, stat.Q_STD_I_AURC] == pytest.approx(
            s4cs.improvement(_aurc(std[index], ri), a_model))
        assert result.draws[r, stat.Q_BASE_MEAN_REGRET] == pytest.approx(ri.mean())
        assert result.draws[r, stat.Q_SEED_I_AURC] == pytest.approx(
            s4cs.improvement(a_base, _aurc(quantities["risk_by_seed"][0][index], ri)))
        layer = quantities["tstar_layer"][index] == 0
        assert result.draws[r, stat.Q_TSTAR_LOW] == pytest.approx(s4cs.improvement(
            _aurc(bi[layer], ri[layer]), _aurc(ei[layer], ri[layer])))


# --- Gate TASK-CONFIRM / Gate RISK-DECISION-CONFIRM ------------------------------------------


def test_gates_pass_on_qualified_fixture_confirmed():
    quantities = _passing_quantities()
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert all(decision.task["conditions"].values()), decision.task
    assert all(decision.risk["conditions"].values()), decision.risk
    assert decision.passed is True
    assert decision.status == prot.STATUS_CONFIRMED
    assert decision.n_groups == GATE_G
    for name in ("i_aurc_regret", "i_regret80", "pr_auc_minus_prevalence",
                 "spearman_branch", "base_vs_random_improvement",
                 "margin_only_i_aurc", "seed_margin_std_i_aurc"):
        assert name in decision.metrics["point"]
        assert name in decision.metrics["ci_lower"]


def test_task_c1_group_count_and_qa():
    quantities = _passing_quantities(groups=300)  # not the frozen 2,000
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c1_all_complete_gate_q"] is False
    assert decision.status == prot.STATUS_INVALID


def test_task_c2_c3_c7_failures_are_task_degenerate():
    rng = np.random.default_rng(13)
    quantities = _passing_quantities()
    regret = rng.gamma(shape=1.0, scale=0.005, size=GATE_G)  # ~7% > 0.01
    quantities["regret"] = regret
    quantities["y_high"] = (regret > prot.HIGH_REGRET_THRESHOLD).astype(np.int64)
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c2_high_regret_frac_ge_15pct"] is False
    assert decision.status == prot.STATUS_TASK_DEGENERATE

    quantities = _passing_quantities()
    quantities["a_star"] = np.zeros(GATE_G, dtype=np.int64)
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c3_at_least_3_oracle_actions_ge_5pct"] is False
    assert decision.status == prot.STATUS_TASK_DEGENERATE

    for layer_key in ("tstar_layer", "interaction_layer"):
        quantities = _passing_quantities()
        quantities[layer_key] = np.zeros(GATE_G, dtype=np.int64)
        decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
        assert decision.task["conditions"]["c7_strata_ge_200_groups"] is False
        assert decision.status == prot.STATUS_TASK_DEGENERATE


def test_task_c4_c5_c6_failures_are_model_insufficient():
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities,
                               ci_overrides={stat.Q_SPEARMAN_BRANCH: -0.01})
    decision = stat.evaluate_confirmation_gates(quantities, bootstrap)
    assert decision.task["conditions"]["c4_branch_spearman_ci_lower_positive"] is False
    assert decision.status == prot.STATUS_MODEL_INSUFFICIENT

    quantities = _passing_quantities()
    quantities["random_regret"] = quantities["regret"].copy()  # zero improvement point
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c5_base_vs_random_point_and_ci_positive"] is False
    assert decision.status == prot.STATUS_MODEL_INSUFFICIENT

    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities,
                               ci_overrides={stat.Q_BASE_VS_RANDOM: -0.001})
    decision = stat.evaluate_confirmation_gates(quantities, bootstrap)
    assert decision.task["conditions"]["c5_base_vs_random_point_and_ci_positive"] is False
    assert decision.status == prot.STATUS_MODEL_INSUFFICIENT

    quantities = _passing_quantities()
    quantities["idle_regret"] = np.maximum(quantities["regret"] - 0.05, 0.0)
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.task["conditions"]["c6_base_not_worse_than_idle"] is False
    assert decision.status == prot.STATUS_MODEL_INSUFFICIENT


def test_risk_r1_r2_r3_enforced_point_and_ci_lower():
    # R1 point below the frozen 0.05 line.
    quantities = _passing_quantities()
    noise = np.random.default_rng(21).random(GATE_G)
    quantities["ensemble_risk"] = noise  # ensemble no better than the noise baseline
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED
    assert decision.passed is False

    # R1 CI lower <= 0.
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities, ci_overrides={stat.Q_I_AURC: -0.001})
    decision = stat.evaluate_confirmation_gates(quantities, bootstrap)
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED

    # R2 CI lower <= 0 while R1 holds.
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities, ci_overrides={stat.Q_I_REGRET80: -0.001})
    decision = stat.evaluate_confirmation_gates(quantities, bootstrap)
    assert decision.risk["conditions"]["r2_i_regret80"] is False
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is True
    assert decision.status == prot.STATUS_NOT_CONFIRMED

    # R3 PR quantity CI lower <= 0.
    quantities = _passing_quantities()
    bootstrap = _bootstrap_for(quantities, ci_overrides={stat.Q_PR: -0.001})
    decision = stat.evaluate_confirmation_gates(quantities, bootstrap)
    assert decision.risk["conditions"]["r3_pr_auc_minus_prevalence"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED


def test_risk_r4_r5_five_seed_counting():
    rng = np.random.default_rng(23)
    for n_bad, expected in ((2, False), (1, True)):
        quantities = _passing_quantities()
        by_seed = np.tile(quantities["ensemble_risk"], (5, 1))
        for i in range(n_bad):
            by_seed[4 - i] = -quantities["regret"] + rng.normal(0, 1e-3, size=GATE_G)
        quantities["risk_by_seed"] = by_seed
        decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
        assert decision.risk["seeds_i_aurc_positive"] == 5 - n_bad
        assert decision.risk["conditions"]["r4_seeds_i_aurc"] is expected
        if n_bad == 2:
            assert decision.status == prot.STATUS_NOT_CONFIRMED

    quantities = _passing_quantities()
    by_seed = np.tile(quantities["ensemble_risk"], (5, 1))
    by_seed[0] = -quantities["regret"] + rng.normal(0, 1e-3, size=GATE_G)
    by_seed[1] = -quantities["regret"] + rng.normal(0, 1e-3, size=GATE_G)
    quantities["risk_by_seed"] = by_seed
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["seeds_i_regret80_positive"] == 3
    assert decision.risk["conditions"]["r5_seeds_i_regret80"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED


def test_risk_r6_r7_r8_and_trim_tail_mechanics():
    # R6: one marginal stratum below the -0.10 floor.
    quantities = _passing_quantities()
    high = quantities["interaction_layer"] == 1
    ensemble = quantities["ensemble_risk"].copy()
    ensemble[high] = -quantities["regret"][high]
    quantities["ensemble_risk"] = ensemble
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert (decision.risk["strata_i_aurc"]["interaction_high"]
            < prot.GATE_RISK_STRATA_I_AURC_FLOOR)
    assert decision.risk["conditions"]["r6_strata_floor"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED

    # R7: trimmed comparisons must stay positive.
    quantities = _passing_quantities()
    quantities["ensemble_risk"] = np.random.default_rng(31).random(GATE_G)
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["conditions"]["r7_trims_positive"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED

    # trim mechanics: the baseline hides the top-5% regret tail at the safest ranks;
    # after the 5% trim the improvement must shrink deterministically.
    quantities = _passing_quantities()
    regret = quantities["regret"]
    top = np.argsort(regret, kind="stable")[-int(0.05 * GATE_G):]
    baseline = np.random.default_rng(37).random(GATE_G)
    baseline[top] = -1.0
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["SEED_VOTE_ENTROPY"] = baseline
    q = stat.compute_confirm_quantity_vector(np.arange(GATE_G, dtype=np.int64), quantities)
    assert q[stat.Q_TRIM5] < q[stat.Q_I_AURC]
    assert np.isfinite(q[stat.Q_TRIM1]) and np.isfinite(q[stat.Q_TRIM5])

    # R8: |Spearman(ensemble, MARGIN_ONLY)| < 0.995 guard.
    quantities = _passing_quantities()
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["MARGIN_ONLY"] = quantities["ensemble_risk"].copy()
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert abs(decision.metrics["point"]["spearman_ensemble_margin"]) == pytest.approx(1.0)
    assert decision.risk["conditions"]["r8_margin_spearman"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED


def test_r9_qa_failure_is_invalid_analysis():
    for broken in stat.QA_KEYS:
        quantities = _passing_quantities()
        quantities["qa"] = {**quantities["qa"], broken: False}
        decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
        assert decision.risk["conditions"]["r9_qa_and_one_shot_locks"] is False
        assert decision.status == prot.STATUS_INVALID


def test_status_priority_section_8_order():
    """invalid > degenerate(2/3/7) > insufficient(4/5/6) > denominator-invalid >
    not-confirmed > confirmed (plan section 8, literal priority)."""
    # qa failure masks a degenerate task.
    quantities = _passing_quantities()
    quantities["a_star"] = np.zeros(GATE_G, dtype=np.int64)
    quantities["qa"] = {**quantities["qa"], "hashes_match": False}
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.status == prot.STATUS_INVALID

    # degenerate masks insufficient.
    quantities = _passing_quantities()
    quantities["a_star"] = np.zeros(GATE_G, dtype=np.int64)
    bootstrap = _bootstrap_for(quantities,
                               ci_overrides={stat.Q_SPEARMAN_BRANCH: -0.01})
    decision = stat.evaluate_confirmation_gates(quantities, bootstrap)
    assert decision.status == prot.STATUS_TASK_DEGENERATE

    # insufficient masks a non-positive R80 denominator (20% tail carries all regret).
    quantities = _passing_quantities()
    regret = np.zeros(GATE_G)
    regret[int(0.8 * GATE_G):] = 1.0
    quantities["regret"] = regret
    quantities["y_high"] = (regret > prot.HIGH_REGRET_THRESHOLD).astype(np.int64)
    quantities["random_regret"] = regret + 0.05
    quantities["idle_regret"] = regret + 0.08
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["SEED_VOTE_ENTROPY"] = np.arange(GATE_G, dtype=np.float64)
    bootstrap = _bootstrap_for(quantities,
                               ci_overrides={stat.Q_SPEARMAN_BRANCH: -0.01})
    decision = stat.evaluate_confirmation_gates(quantities, bootstrap)
    assert decision.metrics["point"]["aurc_baseline"] > 0.0
    assert decision.metrics["point"]["regret80_baseline"] == pytest.approx(0.0)
    assert decision.risk["baseline_valid"] is False
    assert decision.status == prot.STATUS_MODEL_INSUFFICIENT

    # denominator-invalid alone -> INVALID (never NOT_CONFIRMED, never an epsilon).
    quantities = _passing_quantities()
    regret = np.zeros(GATE_G)
    regret[int(0.8 * GATE_G):] = 1.0
    quantities["regret"] = regret
    quantities["y_high"] = (regret > prot.HIGH_REGRET_THRESHOLD).astype(np.int64)
    quantities["random_regret"] = regret + 0.05
    quantities["idle_regret"] = regret + 0.08
    quantities["baseline_risks"] = dict(quantities["baseline_risks"])
    quantities["baseline_risks"]["SEED_VOTE_ENTROPY"] = np.arange(GATE_G, dtype=np.float64)
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["baseline_valid"] is False
    assert decision.status == prot.STATUS_INVALID


def test_only_primary_baseline_enters_the_gate():
    """A stronger MARGIN_ONLY never swaps the frozen primary baseline out of the Gate."""
    quantities = _passing_quantities()
    baselines = dict(quantities["baseline_risks"])
    # margin ranks regret well (a STRONGER curve than the pure-noise SVE) but stays
    # imperfect, so the R8 Spearman ceiling still holds.
    baselines["MARGIN_ONLY"] = (
        quantities["regret"] + np.random.default_rng(55).normal(0.0, 0.03, GATE_G))
    quantities["baseline_risks"] = baselines
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    denominators = decision.risk["baseline_denominators"]
    order = np.argsort(quantities["baseline_risks"]["SEED_VOTE_ENTROPY"], kind="stable")
    sve_aurc = s4cs.aurc_regret(order, quantities["regret"])
    assert denominators["aurc_regret"] == pytest.approx(sve_aurc)
    margin_order = np.argsort(baselines["MARGIN_ONLY"], kind="stable")
    margin_aurc = s4cs.aurc_regret(margin_order, quantities["regret"])
    assert margin_aurc < sve_aurc  # the swap scenario: a stronger non-primary baseline
    # the margin comparison is reported alongside (Table C2), never gating: the Gate's
    # R1/R2 denominators stay the frozen SEED_VOTE_ENTROPY ones and the verdict stands.
    assert decision.metrics["point"]["margin_only_i_aurc"] > 0.0
    assert "margin_only_i_aurc" in decision.metrics["ci_lower"]
    assert decision.status == prot.STATUS_CONFIRMED
    # a quantities block whose "strongest" baseline drifted from SEED_VOTE_ENTROPY is
    # refused outright: the confirm never re-selects.
    quantities = _passing_quantities()
    quantities["strongest_baseline"] = "MARGIN_ONLY"
    with pytest.raises(ValueError):
        stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))


def test_ablations_never_substitute_the_primary_score():
    rng = np.random.default_rng(43)
    quantities = _passing_quantities()
    quantities["ensemble_risk"] = rng.random(GATE_G)  # primary fails r1
    quantities["mu_only_ensemble"] = quantities["regret"] + rng.normal(0, 1e-3, GATE_G)
    decision = stat.evaluate_confirmation_gates(quantities, _bootstrap_for(quantities))
    assert decision.risk["conditions"]["r1_i_aurc_regret"] is False
    assert decision.status == prot.STATUS_NOT_CONFIRMED
    assert decision.ablations["MU_ONLY"]["i_aurc_regret"] > 0.05  # reported, never gating
    assert decision.ablations["blocking"] is False


# --- IO fixtures: synthetic score side + sealed truth -----------------------------------------


def _config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    data_root = tmp_path / "data" / "object_centric_stage" / prot.PURPOSE_CONFIRM
    config["run_dir"] = str(run_dir)
    config["data_root"] = str(data_root)
    for key, value in config["paths"].items():
        if value.startswith("experiments/stage4_reliability/stage4cr2_confirm/"):
            rel = value.removeprefix("experiments/stage4_reliability/stage4cr2_confirm/")
            config["paths"][key] = str(run_dir / rel)
        elif value.startswith(
                "data/object_centric_stage/stage4cr2_merge_multiaction_confirm/"):
            rel = value.removeprefix(
                "data/object_centric_stage/stage4cr2_merge_multiaction_confirm/")
            config["paths"][key] = str(data_root / rel)
    return config


#: group distances at the last history frame -> interaction strata [0,1,0,1].
_DISTANCES = (10.0, 40.0, 10.0, 40.0)
_T_STARS = (10, 12, 13, 15)


def _datasets(config, groups=4):
    """Crafted legal inputs + sealed truth with an exact analytic truth-side consequence.

    History: ego at the origin; slot1 at (d_g, 0); slots 2-3 far away. Factual: ego at
    x=5 all frames, others at x=1e6 -> J_hat == 0 for every action (a_hat == 0).
    Truth CF: others at x=1e6 (interaction exactly 0); group 0's ego final x == a for
    action a -> J_true[0,a] = 0.2*(9-a)/(9+1e-6) ONLY IF all ten actions enter the
    efficiency normalization; groups 1-3 have identical progress across actions
    (span < 1e-6 -> zero efficiency cost -> zero regret).
    """
    hl, hz, cap, nf = prot.HISTORY_LEN, prot.HORIZON, prot.OBJECT_CAPACITY, 8
    history = np.zeros((groups, hl, cap, nf), dtype=np.float32)
    history[:, :, :4, 0] = 1.0  # slots 0-3 present in every history frame
    factual = np.zeros((groups, hz, cap, nf), dtype=np.float32)
    factual[:, :, :4, 0] = 1.0
    for g in range(groups):
        history[g, :, 1, 2] = _DISTANCES[g]  # slot1 x (min distance to ego)
        history[g, :, 2, 2] = 500.0
        history[g, :, 3, 2] = 600.0
    factual[:, :, 1, 2] = 1e6
    factual[:, :, 2, 2] = 1e6
    factual[:, :, 3, 2] = 1e6
    factual[:, :, 0, 2] = 5.0  # ego x at every factual frame
    truth = np.zeros((groups, NA, hz, cap, nf), dtype=np.float32)
    truth[:, :, :, :4, 0] = 1.0
    truth[:, :, :, 1, 2] = 1e6
    truth[:, :, :, 2, 2] = 1e6
    truth[:, :, :, 3, 2] = 1e6
    ramp = np.linspace(0.0, 1.0, hz, dtype=np.float32)
    for g in range(groups):
        for a in range(NA):
            final_x = float(a) if g == 0 else 5.0
            truth[g, a, :, 0, 2] = ramp * final_x
    factual_actions = np.ones((groups, hz), dtype=np.int64)
    cf_actions = np.tile([a for a, _ in prot.ACTION_PLANS], (groups, 1)).astype(np.int64)
    cf_durations = np.tile([d for _, d in prot.ACTION_PLANS], (groups, 1)).astype(np.int64)
    t_star = np.array(_T_STARS[:groups], dtype=np.int64)
    group_seed = prot.build_confirm_seed_schedule(config)[:groups].astype(np.int64)
    inference_mask = np.zeros((groups, hz, cap), dtype=bool)
    inference_mask[:, :, 1:4] = True

    inputs_path = cinf.role_input_path(config, "confirm")
    truth_path = cinf.role_truth_path(config, "confirm")
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        inputs_path, object_history=history, object_factual_future=factual,
        factual_actions=factual_actions, cf_actions=cf_actions,
        cf_durations=cf_durations, t_star=t_star, group_seed=group_seed,
        inference_mask=inference_mask)
    np.savez_compressed(
        truth_path, object_counterfactual_futures=truth, cf_actions=cf_actions,
        cf_durations=cf_durations, factual_actions=factual_actions, t_star=t_star,
        group_seed=group_seed, inference_mask=inference_mask)
    return inputs_path, truth_path


# --- fake frozen-chain callables (identical to the inference-test fakes) ---------------------


def _fake_predictions(bundle, *, binding, stage3r_dir, device, batch_size):
    groups = int(bundle["groups"])
    shape6 = (5, groups, NA, 20, 5, 2)
    shape5 = (5, groups, NA, 20, 5)
    return {
        "e2e": np.zeros(shape6, dtype=np.float32),
        "full": np.zeros(shape6, dtype=np.float32),
        "history": np.zeros(shape6, dtype=np.float32),
        "r_hat": np.zeros(shape5, dtype=np.float32),
        "presence_prob": np.zeros(shape5, dtype=np.float32),
        "high_prob": np.zeros(shape5, dtype=np.float32),
    }


def _fake_features(bundle, predictions):
    return {"U_seed_mean": np.zeros(int(bundle["branches"]), dtype=np.float64)}


class _FixedModel:
    def predict(self, matrix):
        return np.zeros(matrix.shape[0])


def _fake_frozen_models():
    return {"COMBINED_RISK_HEAD": _FixedModel(), "GEOMETRY_ONLY": _FixedModel(),
            "RESPONSE_ONLY": _FixedModel()}


def _fake_score_actions(features, frozen, *, groups, n_actions, rng_seed):
    scores = {k: np.zeros((groups, n_actions), dtype=np.float64)
              for k in s4cpp.RISK_CLASSES}
    scores["CONSTANT_RANDOM"] = np.full((groups, n_actions), 0.5)
    return scores


def _fake_score_fn(inventory, bundle, *, device):
    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64).ravel()
    base = (group_seed % 97).astype(np.float64) / 97.0
    by_seed = np.stack([base + 0.001 * i for i in range(5)])
    return s4cr2t.RiskScores(
        group_seed=group_seed,
        risk_ensemble=by_seed.mean(axis=0),
        risk_by_seed=by_seed,
        mu_only_ensemble=base * 0.5,
        class_only_ensemble=base * 0.25,
        seeds=tuple(int(s) for s in prot.MODEL_SEEDS),
    )


def _hooks():
    return {
        "run_inference_fn": _fake_predictions,
        "build_features_fn": _fake_features,
        "load_models_fn": _fake_frozen_models,
        "score_actions_fn": _fake_score_actions,
        "score_fn": _fake_score_fn,
    }


def _write_protocol_lock(config, *, code_overrides=None):
    """Stand-in protocol lock binding the running config + schedule + code registry."""
    checked = prot.validate_config(config)
    schedule = prot.build_confirm_seed_schedule(checked)
    code_hashes = {rel: sha256_file(ROOT / "code" / rel)
                   for rel in prot.PROTOCOL_CODE_FILES}
    code_hashes.update(code_overrides or {})
    payload = {
        "plan_id": prot.PLAN_ID,
        "config_signature": canonical_fingerprint(json.loads(json.dumps(checked))),
        "confirm_schedule_raw_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "code_sha256": code_hashes,
    }
    lock_path = Path(checked["paths"]["protocol_lock"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return lock_path


def _write_collection_receipt(config, groups, inputs_path, truth_path):
    with np.load(inputs_path) as z:
        legal_seeds = np.asarray(z["group_seed"], dtype=np.int64)
    with np.load(truth_path) as z:
        truth_seeds = np.asarray(z["group_seed"], dtype=np.int64)
    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "confirm",
        "status": "COLLECTED",
        "accepted_groups": int(groups),
        "inference_inputs_sha256": sha256_file(inputs_path),
        "confirmation_truth_sha256": sha256_file(truth_path),
        "truth_fields": list(TRUTH_FIELDS),
        "group_order_binding": {
            "order": "acceptance_order",
            "n_groups": int(groups),
            "inference_inputs_group_seed_sha256": hashlib.sha256(
                legal_seeds.tobytes()).hexdigest(),
            "confirmation_truth_group_seed_sha256": hashlib.sha256(
                truth_seeds.tobytes()).hexdigest(),
            "same_order": True,
        },
    }
    receipt_path = Path(prot.validate_config(config)["paths"]["collection_receipt"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt_path


def _build_score_side(tmp_path, groups=4):
    """Full synthetic score side: datasets + lock + collection receipt + inference."""
    config = _config(tmp_path)
    inputs_path, truth_path = _datasets(config, groups=groups)
    _write_protocol_lock(config)
    _write_collection_receipt(config, groups, inputs_path, truth_path)
    receipt = cinf.infer_and_score(config, "confirm", **_hooks())
    return {"config": config, "inputs": inputs_path, "truth": truth_path,
            "inference_receipt": receipt}


def _freeze_and_burn(fixture):
    config = fixture["config"]
    stat.run_determinism_recheck(config, **_hooks())
    stat.freeze_scores(config, source_verify_fn=lambda *a, **k: {"ok": True})
    return stat.burn_confirmation_lock(config)


# --- determinism recheck ---------------------------------------------------------------------


def test_determinism_recheck_create_once_and_formal_slice_binding(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    paths = stat.confirm_artifact_paths(config)
    receipt = stat.run_determinism_recheck(config, **_hooks())
    assert receipt["bitwise_identical"] is True
    assert receipt["n_groups_rechecked"] == 4  # min(determinism_checks, G)
    assert receipt["context_rows"] == 4 * NA  # the role's full branch count as context
    assert paths["determinism_receipt"].is_file()
    text = paths["determinism_receipt"].read_text(encoding="utf-8")
    again = stat.run_determinism_recheck(config, **_hooks())
    assert again["verification"] == "existing_determinism_verified_no_overwrite"
    assert paths["determinism_receipt"].read_text(encoding="utf-8") == text

    # a non-deterministic chain is a hard stop, never a tolerance.
    fixture = _build_score_side(tmp_path / "b")
    calls = {"n": 0}

    def noisy(bundle, *, binding, stage3r_dir, device, batch_size):
        calls["n"] += 1
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        if calls["n"] % 2 == 0:
            out["e2e"] = out["e2e"] + 1e-6
        return out

    with pytest.raises(RuntimeError):
        stat.run_determinism_recheck(fixture["config"],
                                     **{**_hooks(), "run_inference_fn": noisy})

    # recheck results must match the FORMAL artifacts' leading slices (not only each
    # other): tamper the formal predictions and the recheck refuses.
    fixture = _build_score_side(tmp_path / "c")
    config = fixture["config"]
    paths = stat.confirm_artifact_paths(config)
    with np.load(paths["predictions"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["e2e"] = payload["e2e"] + 1e-6
    np.savez_compressed(paths["predictions"], **payload)
    with pytest.raises((ValueError, RuntimeError)):
        stat.run_determinism_recheck(config, **_hooks())


# --- Task 7-FIX: the recheck pads Direct scoring to the formal forward context --------


def _context_sensitive_score_fn(inventory, bundle, *, device):
    """Emulates the row-count-context-dependent Direct forward: every row's output shifts
    with the total row count M (as the real unbatched GEMM kernel selection does), so a
    standalone small-M forward diverges from the formal large-M context while a
    zero-padded-to-M forward sliced back reproduces it bitwise."""
    tokens = np.asarray(bundle["tokens"], dtype=np.float64)
    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64).ravel()
    m = tokens.shape[0]
    base = (group_seed % 97).astype(np.float64) / 97.0 + 1e-9 * m
    by_seed = np.stack([base + 0.001 * i for i in range(5)])
    return s4cr2t.RiskScores(
        group_seed=group_seed,
        risk_ensemble=by_seed.mean(axis=0),
        risk_by_seed=by_seed,
        mu_only_ensemble=base * 0.5,
        class_only_ensemble=base * 0.25,
        seeds=tuple(int(s) for s in prot.MODEL_SEEDS),
    )


def test_context_aligned_direct_scores_reproduces_formal_context():
    rng = np.random.default_rng(7)
    rows, context = 4, 16
    tokens = rng.normal(size=(rows, 9, 34))
    glob = rng.normal(size=(rows, 9))
    group_seed = np.array([11, 22, 33, 44], dtype=np.int64)

    aligned = stat._context_aligned_direct_scores(
        _context_sensitive_score_fn, None, tokens, glob, group_seed,
        context_rows=context, device="cpu")
    # a manual formal-context forward, sliced back, is the bitwise reference.
    padded_tokens = np.zeros((context, 9, 34))
    padded_tokens[:rows] = tokens
    padded_glob = np.zeros((context, 9))
    padded_glob[:rows] = glob
    padded_seed = np.zeros(context, dtype=np.int64)
    padded_seed[:rows] = group_seed
    reference = _context_sensitive_score_fn(
        None, {"tokens": padded_tokens, "global_features": padded_glob,
               "group_seed": padded_seed}, device="cpu")
    assert np.array_equal(aligned.risk_by_seed, reference.risk_by_seed[:, :rows])
    assert np.array_equal(aligned.risk_ensemble, reference.risk_ensemble[:rows])
    assert np.array_equal(aligned.mu_only_ensemble, reference.mu_only_ensemble[:rows])
    assert np.array_equal(aligned.class_only_ensemble,
                          reference.class_only_ensemble[:rows])
    assert np.array_equal(aligned.group_seed, group_seed)  # never the padded seeds
    assert tuple(aligned.seeds) == tuple(int(s) for s in prot.MODEL_SEEDS)
    # the standalone small-context forward diverges (the bug this repair fixes).
    standalone = _context_sensitive_score_fn(
        None, {"tokens": tokens, "global_features": glob, "group_seed": group_seed},
        device="cpu")
    assert not np.array_equal(aligned.risk_by_seed, standalone.risk_by_seed)

    # an equal context is an exact passthrough: no padding, no re-wrapping.
    calls = []

    def spy(inventory, bundle, *, device):
        calls.append({k: np.asarray(v).shape for k, v in bundle.items()})
        return _context_sensitive_score_fn(inventory, bundle, device=device)

    same = stat._context_aligned_direct_scores(
        spy, None, tokens, glob, group_seed, context_rows=rows, device="cpu")
    assert calls == [{"tokens": (rows, 9, 34), "global_features": (rows, 9),
                      "group_seed": (rows,)}]
    assert np.array_equal(same.risk_by_seed, standalone.risk_by_seed)

    # a smaller context can never be repaired by padding: fail closed.
    with pytest.raises(RuntimeError):
        stat._context_aligned_direct_scores(
            _context_sensitive_score_fn, None, tokens, glob, group_seed,
            context_rows=rows - 1, device="cpu")


def test_recheck_chain_once_pads_direct_scoring_to_formal_context(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    inputs_path, _truth_path = _datasets(config, groups=4)
    with np.load(inputs_path) as z:
        legal = {k: np.asarray(z[k]) for k in cinf.LEGAL_INPUT_KEYS}
    groups = int(legal["group_seed"].shape[0])
    bundle = cinf._bundle_from_arrays(
        legal, config=checked, data_signature="probe", role="recheck_unit")
    hooks = stat._resolve_hooks(_fake_predictions, _fake_features, _fake_frozen_models,
                                _fake_score_actions, None, _context_sensitive_score_fn)
    direct_context = groups * 4  # the formal path scored more groups than the recheck
    result = stat._recheck_chain_once(
        checked, bundle, context_rows=groups * NA, direct_context_rows=direct_context,
        hooks=hooks, device="cpu", batch_size=32, binding={}, stage3r_dir=Path("."),
        inventory=None)
    tokens = np.asarray(result["tokens"]["tokens"], dtype=np.float64)
    glob = np.asarray(result["tokens"]["global_features"], dtype=np.float64)
    seeds = np.asarray(bundle["group_seed"], dtype=np.int64)
    padded_tokens = np.zeros((direct_context,) + tokens.shape[1:])
    padded_tokens[:groups] = tokens
    padded_glob = np.zeros((direct_context, glob.shape[1]))
    padded_glob[:groups] = glob
    padded_seed = np.zeros(direct_context, dtype=np.int64)
    padded_seed[:groups] = seeds
    reference = _context_sensitive_score_fn(
        None, {"tokens": padded_tokens, "global_features": padded_glob,
               "group_seed": padded_seed}, device="cpu")
    assert np.array_equal(result["scores"].risk_by_seed,
                          reference.risk_by_seed[:, :groups])
    assert np.array_equal(result["scores"].risk_ensemble,
                          reference.risk_ensemble[:groups])
    standalone = _context_sensitive_score_fn(
        None, {"tokens": tokens, "global_features": glob, "group_seed": seeds},
        device="cpu")
    assert not np.array_equal(result["scores"].risk_by_seed, standalone.risk_by_seed)


def test_determinism_recheck_passes_formal_direct_context_rows(tmp_path, monkeypatch):
    fixture = _build_score_side(tmp_path)
    captured = []
    real = stat._recheck_chain_once

    def spy(checked, bundle, **kwargs):
        captured.append(int(kwargs["direct_context_rows"]))
        return real(checked, bundle, **kwargs)

    monkeypatch.setattr(stat, "_recheck_chain_once", spy)
    receipt = stat.run_determinism_recheck(fixture["config"], **_hooks())
    assert receipt["direct_context_rows"] == 4  # the formal risk_by_seed group count
    assert captured == [4, 4]  # BOTH independent runs use the formal forward context


def test_smoke_recheck_passes_smoke_direct_context_rows(tmp_path, monkeypatch):
    fixture = _build_smoke_side(tmp_path)
    captured = []
    real = stat._recheck_chain_once

    def spy(checked, bundle, **kwargs):
        captured.append(int(kwargs["direct_context_rows"]))
        return real(checked, bundle, **kwargs)

    monkeypatch.setattr(stat, "_recheck_chain_once", spy)
    receipt = stat.run_smoke_recheck(fixture["config"], **_hooks())
    assert receipt["direct_context_rows"] == SMOKE_G  # the smoke group count (no-op pad)
    assert captured == [SMOKE_G, SMOKE_G]


# --- Task 7-FIX: recovery-lock chain binding at the score side (plan section 8) --------


def _rewrite_lock_code_hash(config, rel, new_hash):
    """Rewrite the stand-in lock as the PRE-repair lock (binds an older code hash)."""
    checked = prot.validate_config(config)
    lock_path = Path(checked["paths"]["protocol_lock"])
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["code_sha256"][rel] = new_hash
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return lock_path


def _write_recovery_lock(config, changed, *, binds=None, name="recovery_lock_r1.json",
                         kind="pure_engineering_recovery", declared=None):
    checked = prot.validate_config(config)
    lock_path = Path(checked["paths"]["protocol_lock"])
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    code = dict(payload["code_sha256"])
    for rel, record in changed.items():
        code[rel] = record["after_sha256"]
    recovery = {
        "plan_id": prot.PLAN_ID,
        "kind": kind,
        "binds_previous_lock_sha256": binds or sha256_file(lock_path),
        "changed_files": changed,
        "code_sha256": declared if declared is not None else code,
    }
    path = lock_path.parent / name
    path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def test_freeze_accepts_recovery_lock_chain_for_locked_repair(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    stat.run_determinism_recheck(config, **_hooks())
    rel = "ac_twin/stage4cr2_confirm_statistics.py"
    old_hash = "0" * 64
    current = sha256_file(ROOT / "code" / rel)
    _rewrite_lock_code_hash(config, rel, old_hash)
    # the code drift alone is refused, recovery chain or not ...
    with pytest.raises(ValueError):
        stat.freeze_scores(config, source_verify_fn=lambda *a, **k: {"ok": True})
    # ... and the immutable recovery lock binding the repair makes the chain pass.
    _write_recovery_lock(
        config, {rel: {"before_sha256": old_hash, "after_sha256": current}})
    receipt = stat.freeze_scores(config, source_verify_fn=lambda *a, **k: {"ok": True})
    assert receipt["plan_id"] == prot.PLAN_ID


def test_freeze_refuses_invalid_recovery_locks(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    stat.run_determinism_recheck(config, **_hooks())
    rel = "ac_twin/stage4cr2_confirm_statistics.py"
    old_hash = "0" * 64
    current = sha256_file(ROOT / "code" / rel)
    lock_path = _rewrite_lock_code_hash(config, rel, old_hash)
    recovery_path = lock_path.parent / "recovery_lock_r1.json"
    freeze = lambda: stat.freeze_scores(  # noqa: E731
        config, source_verify_fn=lambda *a, **k: {"ok": True})

    # (1) a recovery lock not binding the previous lock file's hash.
    _write_recovery_lock(config, {rel: {"before_sha256": old_hash,
                                        "after_sha256": current}}, binds="f" * 64)
    with pytest.raises(ValueError):
        freeze()
    # (2) a recovery lock whose before state does not match the chained lock.
    _write_recovery_lock(config, {rel: {"before_sha256": "1" * 64,
                                        "after_sha256": current}})
    with pytest.raises(ValueError):
        freeze()
    # (3) a recovery lock whose after state does not match the current code.
    _write_recovery_lock(config, {rel: {"before_sha256": old_hash,
                                        "after_sha256": "2" * 64}})
    with pytest.raises(ValueError):
        freeze()
    # (4) a recovery lock re-binding a frozen prior-stage file can never be legal.
    frozen_rel = prot.REUSED_FROZEN_CODE_FILES[0]
    frozen_current = sha256_file(ROOT / "code" / frozen_rel)
    _rewrite_lock_code_hash(config, frozen_rel, frozen_current)  # register it in the lock
    _write_recovery_lock(config, {frozen_rel: {
        "before_sha256": frozen_current, "after_sha256": "3" * 64}})
    with pytest.raises(ValueError):
        freeze()
    # (5) a broken chain (gap: r2 without r1) fails closed.
    recovery_path.unlink()
    _write_recovery_lock(config, {rel: {"before_sha256": old_hash,
                                        "after_sha256": current}},
                         name="recovery_lock_r2.json")
    with pytest.raises(ValueError):
        freeze()
    # the valid chain is accepted after the invalid ones are removed.
    (lock_path.parent / "recovery_lock_r2.json").unlink()
    _write_recovery_lock(config, {rel: {"before_sha256": old_hash,
                                        "after_sha256": current}})
    receipt = freeze()
    assert receipt["plan_id"] == prot.PLAN_ID


# --- ordering discipline ----------------------------------------------------------------------


def test_ordering_freeze_requires_determinism_receipt(tmp_path):
    fixture = _build_score_side(tmp_path)
    with pytest.raises(ValueError, match="determinism"):
        stat.freeze_scores(fixture["config"],
                           source_verify_fn=lambda *a, **k: {"ok": True})


def test_ordering_burn_requires_score_freeze(tmp_path):
    fixture = _build_score_side(tmp_path)
    stat.run_determinism_recheck(fixture["config"], **_hooks())
    with pytest.raises(ValueError, match="freeze"):
        stat.burn_confirmation_lock(fixture["config"])


def test_ordering_sealed_read_requires_burned_lock(tmp_path):
    fixture = _build_score_side(tmp_path)
    stat.run_determinism_recheck(fixture["config"], **_hooks())
    stat.freeze_scores(fixture["config"], source_verify_fn=lambda *a, **k: {"ok": True})
    with pytest.raises(ValueError, match="lock"):
        stat.load_confirmation_truth(fixture["config"])
    with pytest.raises(ValueError, match="lock"):
        stat.assemble_confirm_quantities(fixture["config"])


def test_missing_receipts_fail_loud(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError):
        stat.freeze_scores(config, source_verify_fn=lambda *a, **k: {"ok": True})
    fixture = _build_score_side(tmp_path / "b")
    paths = stat.confirm_artifact_paths(fixture["config"])
    paths["collection_receipt"].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        stat.run_determinism_recheck(fixture["config"], **_hooks())


def test_freeze_refuses_code_drift_in_lock(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    stat.run_determinism_recheck(config, **_hooks())
    # rewrite the lock with a wrong statistics-code hash (locked code drift).
    _write_protocol_lock(
        config, code_overrides={"ac_twin/stage4cr2_confirm_statistics.py": "0" * 64})
    with pytest.raises(ValueError, match="code"):
        stat.freeze_scores(config, source_verify_fn=lambda *a, **k: {"ok": True})


# --- full synthetic chain: freeze -> burn -> truth -> assemble -> adjudicate ------------------


def test_full_chain_g4_sealed_quantities_exact_and_invalid_status(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    lock_path = _freeze_and_burn(fixture)
    assert Path(lock_path).is_file()

    truth = stat.load_confirmation_truth(config)
    assert truth["n_groups"] == 4
    assert set(truth["arrays"]) == set(TRUTH_FIELDS)

    npz_path = Path(stat.assemble_confirm_quantities(config))
    assert npz_path.is_file()
    paths = stat.confirm_artifact_paths(config)
    quantities = stat.load_confirm_group_quantities(
        npz_path, paths["quantities_receipt"])
    # exact analytic truth-side values: group 0 ego final x == action index a.
    denom = 9.0 + 1e-6
    assert quantities["regret"][0] == pytest.approx(0.2 * 9.0 / denom, rel=1e-9)
    assert quantities["random_regret"][0] == pytest.approx(0.2 * 4.5 / denom, rel=1e-9)
    assert quantities["idle_regret"][0] == pytest.approx(0.2 * 6.0 / denom, rel=1e-9)
    assert quantities["regret"][1:].tolist() == [0.0, 0.0, 0.0]
    assert quantities["a_hat"].tolist() == [0, 0, 0, 0]
    assert quantities["a_star"].tolist() == [9, 0, 0, 0]
    assert quantities["y_high"].tolist() == [1, 0, 0, 0]
    # ten-action normalization proof: if only nine actions had entered the efficiency
    # normalization the oracle would be action 8 and regret 0.2*8/(8+1e-6).
    assert quantities["J_true"][0, 9] == pytest.approx(0.0, abs=1e-12)
    assert quantities["J_true"][0, 0] == pytest.approx(0.2 * 9.0 / denom, rel=1e-9)
    assert quantities["tstar_layer"].tolist() == [0, 0, 1, 1]
    assert quantities["interaction_layer"].tolist() == [0, 1, 0, 1]
    assert quantities["trim1_keep"].all() and quantities["trim5_keep"].all()
    # G=4 is not the frozen 2,000-group confirm task: QA marks it incomplete.
    assert quantities["qa"]["all_complete"] is False
    assert quantities["qa"]["role_seeds_valid"] is True
    assert quantities["strongest_baseline"] == prot.PRIMARY_BASELINE

    decision = stat.adjudicate_confirmation(config)
    assert decision["status"] == prot.STATUS_INVALID
    assert decision["passed"] is False
    # every section-7 artifact exists and the report binds the quantities hash.
    for key in ("table_c1", "table_c2", "table_c3", "table_c4", "table_c5",
                "fig_c1", "secondary_csv"):
        assert paths[key].is_file(), key
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["bindings"]["confirm_group_quantities_sha256"] == sha256_file(npz_path)
    assert report["status"] == prot.STATUS_INVALID
    decision_md = paths["decision"].read_text(encoding="utf-8")
    assert prot.STATUS_INVALID in decision_md
    assert "不得声称十动作获得均衡证据" in decision_md  # action-frequency caveat, always
    # second adjudication / assembly is refused outright.
    with pytest.raises(FileExistsError):
        stat.adjudicate_confirmation(config)
    with pytest.raises(FileExistsError):
        stat.assemble_confirm_quantities(config)


def test_misordered_or_tampered_sealed_data_is_invalid_not_recovered(tmp_path):
    # misordered truth group_seed: the same-order binding fails loudly at assembly.
    fixture = _build_score_side(tmp_path / "a")
    config = fixture["config"]
    _freeze_and_burn(fixture)
    with np.load(fixture["truth"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["group_seed"] = payload["group_seed"][::-1].copy()
    np.savez_compressed(fixture["truth"], **payload)  # same file, tampered content
    with pytest.raises((ValueError, RuntimeError)):
        stat.load_confirmation_truth(config)

    # integrity-copy mismatch (truth factual plan != legal factual plan) -> invalid.
    fixture = _build_score_side(tmp_path / "b")
    config = fixture["config"]
    _freeze_and_burn(fixture)
    with np.load(fixture["truth"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["factual_actions"] = payload["factual_actions"] + 1
    np.savez_compressed(fixture["truth"], **payload)
    with pytest.raises((ValueError, RuntimeError)):
        stat.load_confirmation_truth(config)

    # misordered risk scores vs legal inputs (output order binding) -> invalid.
    fixture = _build_score_side(tmp_path / "c")
    config = fixture["config"]
    paths = stat.confirm_artifact_paths(config)
    _freeze_and_burn(fixture)
    with np.load(paths["risk_scores"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["group_seed"] = payload["group_seed"][::-1].copy()
    np.savez_compressed(paths["risk_scores"], **payload)
    with pytest.raises((ValueError, RuntimeError)):
        stat.assemble_confirm_quantities(config)


# --- one-shot recovery semantics ---------------------------------------------------------------


def test_legal_same_transaction_resume_never_overwrites(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    paths = stat.confirm_artifact_paths(config)
    _freeze_and_burn(fixture)
    stat.assemble_confirm_quantities(config)
    receipts = {
        key: paths[key].read_bytes()
        for key in ("determinism_receipt", "score_freeze_receipt", "lock",
                    "quantities_npz", "quantities_receipt")
    }
    # same-transaction resume: burn/freeze/assemble re-verify and return, byte-identical.
    stat.run_determinism_recheck(config, **_hooks())
    stat.freeze_scores(config, source_verify_fn=lambda *a, **k: {"ok": True})
    stat.burn_confirmation_lock(config)
    again = stat.assemble_confirm_quantities(config)
    assert again == str(paths["quantities_npz"])
    for key, blob in receipts.items():
        assert paths[key].read_bytes() == blob, key
    # crash-after-tables resume: delete report/decision, keep the six output files;
    # the rewrite must be byte-identical for every table/figure.
    decision = stat.adjudicate_confirmation(config)
    assert decision["status"] == prot.STATUS_INVALID
    tables = {key: paths[key].read_bytes()
              for key in ("table_c1", "table_c2", "table_c3", "table_c4", "table_c5",
                          "fig_c1", "secondary_csv")}
    paths["report"].unlink()
    paths["decision"].unlink()
    decision = stat.adjudicate_confirmation(config)
    assert decision["status"] == prot.STATUS_INVALID
    for key, blob in tables.items():
        assert paths[key].read_bytes() == blob, key
    # quantities receipts still untouched.
    assert paths["quantities_npz"].read_bytes() == receipts["quantities_npz"]
    assert paths["quantities_receipt"].read_bytes() == receipts["quantities_receipt"]


def test_recovery_with_changed_seed_data_or_config_refused(tmp_path):
    # changed bootstrap seed/config: validate_config refuses before anything runs.
    fixture = _build_score_side(tmp_path / "a")
    _freeze_and_burn(fixture)
    stat.assemble_confirm_quantities(fixture["config"])
    changed = dict(fixture["config"])
    changed["bootstrap_entropy"] = 99
    with pytest.raises(ValueError):
        stat.adjudicate_confirmation(changed)

    # tampered quantities NPZ after the burn: the transaction is unprovable.
    fixture = _build_score_side(tmp_path / "b")
    config = fixture["config"]
    paths = stat.confirm_artifact_paths(config)
    _freeze_and_burn(fixture)
    stat.assemble_confirm_quantities(config)
    with np.load(paths["quantities_npz"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["regret"] = payload["regret"] + 1.0
    np.savez_compressed(paths["quantities_npz"], **payload)
    with pytest.raises(stat.ResourceAtRiskError) as excinfo:
        stat.adjudicate_confirmation(config)
    assert excinfo.value.status == prot.STATUS_RESOURCE_AT_RISK

    # quantities NPZ without its receipt: unprovable transaction.
    fixture = _build_score_side(tmp_path / "c")
    config = fixture["config"]
    paths = stat.confirm_artifact_paths(config)
    _freeze_and_burn(fixture)
    stat.assemble_confirm_quantities(config)
    paths["quantities_receipt"].unlink()
    with pytest.raises(stat.ResourceAtRiskError):
        stat.assemble_confirm_quantities(config)

    # a pre-existing output table with different bytes: unprovable, never overwritten.
    fixture = _build_score_side(tmp_path / "d")
    config = fixture["config"]
    paths = stat.confirm_artifact_paths(config)
    _freeze_and_burn(fixture)
    stat.assemble_confirm_quantities(config)
    paths["table_c1"].parent.mkdir(parents=True, exist_ok=True)
    paths["table_c1"].write_text("garbage\n", encoding="utf-8")
    with pytest.raises(stat.ResourceAtRiskError):
        stat.adjudicate_confirmation(config)
    assert paths["table_c1"].read_text(encoding="utf-8") == "garbage\n"


def test_burn_never_reads_sealed_bytes_and_missing_sealed_file_fails_loud(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    stat.run_determinism_recheck(config, **_hooks())
    stat.freeze_scores(config, source_verify_fn=lambda *a, **k: {"ok": True})
    fixture["truth"].unlink()  # the truth file is GONE before the burn
    lock_path = stat.burn_confirmation_lock(config)  # burn binds the RECEIPT hash only
    assert Path(lock_path).is_file()
    with pytest.raises(FileNotFoundError):
        stat.load_confirmation_truth(config)


# --- happy path: fabricated 2,000-group quantities -> CONFIRMED ---------------------------------


def _fabricate_quantities_files(config, quantities):
    """Write a quantities NPZ + receipt exactly as assemble would (hash-bound)."""
    paths = stat.confirm_artifact_paths(config)
    lock_payload = {"plan_id": prot.PLAN_ID, "role": "confirm", "bindings": {}}
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["lock"].write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    payload = {}
    for key in stat.REQUIRED_ARRAY_KEYS:
        if key == "baseline_risks":
            continue
        payload[key] = np.asarray(quantities[key])
    for name, risk in quantities["baseline_risks"].items():
        payload[f"baseline_risk__{name}"] = np.asarray(risk)
    payload["n_groups"] = np.int64(quantities["n_groups"])
    payload["interaction_threshold"] = np.float64(quantities["interaction_threshold"])
    npz_path = paths["quantities_npz"]
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **payload)
    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "confirm",
        "n_groups": int(quantities["n_groups"]),
        "confirm_group_quantities_path": str(npz_path),
        "confirm_group_quantities_sha256": sha256_file(npz_path),
        "primary_baseline": prot.PRIMARY_BASELINE,
        "interaction_threshold": prot.INTERACTION_THRESHOLD,
        "tstar_stratum_split": prot.T_STAR_STRATUM_SPLIT,
        "coverage": prot.COVERAGE,
        "bootstrap_entropy": prot.BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": prot.BOOTSTRAP_RESAMPLES,
        "confidence": prot.CONFIDENCE,
        "qa": dict(quantities["qa"]),
        "lock_sha256": sha256_file(paths["lock"]),
        "bindings": {},
    }
    paths["quantities_receipt"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path


def test_adjudicate_confirmed_happy_path(tmp_path):
    config = _config(tmp_path)
    quantities = _passing_quantities()
    npz_path = _fabricate_quantities_files(config, quantities)
    bootstrap_calls = {"n": 0}

    def bootstrap_fn(q):
        bootstrap_calls["n"] += 1
        return stat.paired_group_bootstrap(q, seed=prot.BOOTSTRAP_ENTROPY,
                                           n_resamples=200)

    decision = stat.adjudicate_confirmation(config, bootstrap_fn=bootstrap_fn)
    assert bootstrap_calls["n"] == 1  # exactly ONE adjudication statistic run
    assert decision["status"] == prot.STATUS_CONFIRMED
    assert decision["passed"] is True
    assert decision["bootstrap"]["seed"] == prot.BOOTSTRAP_ENTROPY
    assert decision["bootstrap"]["confidence"] == 0.95
    paths = stat.confirm_artifact_paths(config)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["status"] == prot.STATUS_CONFIRMED
    assert report["bindings"]["confirm_group_quantities_sha256"] == sha256_file(npz_path)
    decision_md = paths["decision"].read_text(encoding="utf-8")
    # the success sentence is legal ONLY on CONFIRMED (plan section 7, verbatim).
    assert ("在所测Merge、回溯事实后缀可用的十动作任务中，直接监督决策风险审计改善了选择性保留组的regret"
            in decision_md)
    assert "不得声称十动作获得均衡证据" in decision_md
    # the main comparison row uses ONLY the frozen primary baseline.
    table_c1 = paths["table_c1"].read_text(encoding="utf-8")
    assert prot.STATUS_CONFIRMED in table_c1
    table_c2 = paths["table_c2"].read_text(encoding="utf-8")
    for name in ("DIRECT_ENSEMBLE", "MARGIN_ONLY", "SEED_VOTE_ENTROPY",
                 "SEED_MARGIN_STD", "COMBINED_TOP2_MAX", "CONSTANT_RANDOM"):
        assert name in table_c2
    with pytest.raises(FileExistsError):
        stat.adjudicate_confirmation(config, bootstrap_fn=bootstrap_fn)


# --- smoke recheck + smoke engineering-truth statistics (Task 2 supplement, pre-lock) ---------

SMOKE_G = prot.SMOKE_ACCEPTED_GROUPS  # 12


def _smoke_datasets(config, groups=SMOKE_G):
    """Synthetic smoke-shaped legal inputs + smoke truth (12 accepted groups).

    Same crafted geometry as ``_datasets`` (J_hat == 0 everywhere; group 0's truth
    ramps with the action index, every other group is action-independent), but bound
    to the SMOKE namespace paths and the frozen smoke seed schedule.
    """
    hl, hz, cap, nf = prot.HISTORY_LEN, prot.HORIZON, prot.OBJECT_CAPACITY, 8
    history = np.zeros((groups, hl, cap, nf), dtype=np.float32)
    history[:, :, :4, 0] = 1.0
    factual = np.zeros((groups, hz, cap, nf), dtype=np.float32)
    factual[:, :, :4, 0] = 1.0
    distances = [10.0, 40.0, 10.0, 40.0]
    t_stars = [10, 12, 13, 15]
    for g in range(groups):
        history[g, :, 1, 2] = distances[g % 4]
        history[g, :, 2, 2] = 500.0
        history[g, :, 3, 2] = 600.0
    factual[:, :, 1, 2] = 1e6
    factual[:, :, 2, 2] = 1e6
    factual[:, :, 3, 2] = 1e6
    factual[:, :, 0, 2] = 5.0
    truth = np.zeros((groups, NA, hz, cap, nf), dtype=np.float32)
    truth[:, :, :, :4, 0] = 1.0
    truth[:, :, :, 1, 2] = 1e6
    truth[:, :, :, 2, 2] = 1e6
    truth[:, :, :, 3, 2] = 1e6
    ramp = np.linspace(0.0, 1.0, hz, dtype=np.float32)
    for g in range(groups):
        for a in range(NA):
            final_x = float(a) if g == 0 else 5.0
            truth[g, a, :, 0, 2] = ramp * final_x
    factual_actions = np.ones((groups, hz), dtype=np.int64)
    cf_actions = np.tile([a for a, _ in prot.ACTION_PLANS], (groups, 1)).astype(np.int64)
    cf_durations = np.tile([d for _, d in prot.ACTION_PLANS], (groups, 1)).astype(np.int64)
    t_star = np.array([t_stars[g % 4] for g in range(groups)], dtype=np.int64)
    group_seed = prot.build_smoke_seed_schedule(config)[:groups].astype(np.int64)
    inference_mask = np.zeros((groups, hz, cap), dtype=bool)
    inference_mask[:, :, 1:4] = True

    inputs_path = cinf.role_input_path(config, "smoke")
    truth_path = cinf.role_truth_path(config, "smoke")
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        inputs_path, object_history=history, object_factual_future=factual,
        factual_actions=factual_actions, cf_actions=cf_actions,
        cf_durations=cf_durations, t_star=t_star, group_seed=group_seed,
        inference_mask=inference_mask)
    np.savez_compressed(
        truth_path, object_counterfactual_futures=truth, cf_actions=cf_actions,
        cf_durations=cf_durations, factual_actions=factual_actions, t_star=t_star,
        group_seed=group_seed, inference_mask=inference_mask)
    return inputs_path, truth_path


def _write_smoke_collection_receipt(config, groups, inputs_path, truth_path):
    """Stand-in smoke collection receipt (ENGINEERING_SMOKE_ONLY, never COLLECTED)."""
    with np.load(inputs_path) as z:
        legal_seeds = np.asarray(z["group_seed"], dtype=np.int64)
    with np.load(truth_path) as z:
        truth_seeds = np.asarray(z["group_seed"], dtype=np.int64)
    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "smoke",
        "status": cinf.STATUS_SMOKE,
        "partial": False,
        "accepted_groups": int(groups),
        "inference_inputs_sha256": sha256_file(inputs_path),
        "confirmation_truth_sha256": sha256_file(truth_path),
        "truth_fields": list(TRUTH_FIELDS),
        "group_order_binding": {
            "order": "acceptance_order",
            "n_groups": int(groups),
            "inference_inputs_group_seed_sha256": hashlib.sha256(
                legal_seeds.tobytes()).hexdigest(),
            "confirmation_truth_group_seed_sha256": hashlib.sha256(
                truth_seeds.tobytes()).hexdigest(),
            "same_order": True,
        },
    }
    receipt_path = Path(config["run_dir"]) / "smoke" / "smoke_collection_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt_path


def _build_smoke_side(tmp_path, groups=SMOKE_G):
    """Full synthetic smoke side: datasets + smoke collection receipt + smoke inference."""
    config = _config(tmp_path)
    inputs_path, truth_path = _smoke_datasets(config, groups=groups)
    _write_smoke_collection_receipt(config, groups, inputs_path, truth_path)
    receipt = cinf.infer_and_score(config, "smoke", **_hooks())
    return {"config": config, "inputs": inputs_path, "truth": truth_path,
            "inference_receipt": receipt}


def test_smoke_paths_stay_inside_the_smoke_namespace(tmp_path):
    config = _config(tmp_path)
    paths = stat.smoke_artifact_paths(config)
    formal = stat.confirm_artifact_paths(config)
    formal_resolved = {Path(p).resolve() for p in formal.values()}
    for key, path in paths.items():
        assert Path(path).parent.name == "smoke", key
        assert Path(path).resolve() not in formal_resolved, key


def test_smoke_recheck_bitwise_identity_create_once(tmp_path):
    fixture = _build_smoke_side(tmp_path)
    config = fixture["config"]
    paths = stat.smoke_artifact_paths(config)
    receipt = stat.run_smoke_recheck(config, **_hooks())
    assert receipt["bitwise_identical"] is True
    assert receipt["role"] == "smoke"
    assert receipt["status"] == cinf.STATUS_SMOKE
    assert receipt["n_groups_rechecked"] == SMOKE_G  # ALL 12 smoke groups, not a slice
    assert receipt["context_rows"] == SMOKE_G * NA   # the smoke role's full branch count
    assert paths["recheck_receipt"].is_file()
    text = paths["recheck_receipt"].read_text(encoding="utf-8")
    again = stat.run_smoke_recheck(config, **_hooks())
    assert again["verification"] == "existing_smoke_recheck_verified_no_overwrite"
    assert paths["recheck_receipt"].read_text(encoding="utf-8") == text


def test_smoke_recheck_refuses_nondeterministic_chain(tmp_path):
    fixture = _build_smoke_side(tmp_path)
    calls = {"n": 0}

    def noisy(bundle, *, binding, stage3r_dir, device, batch_size):
        calls["n"] += 1
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        if calls["n"] % 2 == 0:
            out["e2e"] = out["e2e"] + 1e-6
        return out

    with pytest.raises(RuntimeError):
        stat.run_smoke_recheck(fixture["config"],
                               **{**_hooks(), "run_inference_fn": noisy})


def test_smoke_recheck_refuses_tampered_persisted_artifacts(tmp_path):
    # hash drift vs the smoke inference receipt: fail closed.
    fixture = _build_smoke_side(tmp_path / "a")
    config = fixture["config"]
    paths = stat.smoke_artifact_paths(config)
    with np.load(paths["predictions"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["e2e"] = payload["e2e"] + 1e-6
    np.savez_compressed(paths["predictions"], **payload)
    with pytest.raises((ValueError, RuntimeError)):
        stat.run_smoke_recheck(config, **_hooks())

    # receipt hash tampered along with the file: the FULL-ARRAY comparison itself must
    # still refuse (bitwise vs the persisted smoke artifact, never a tolerance).
    fixture = _build_smoke_side(tmp_path / "b")
    config = fixture["config"]
    paths = stat.smoke_artifact_paths(config)
    with np.load(paths["predictions"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["e2e"] = payload["e2e"] + 1e-6
    np.savez_compressed(paths["predictions"], **payload)
    receipt_path = paths["inference_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["predictions_sha256"] = sha256_file(paths["predictions"])
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    with pytest.raises(RuntimeError, match="smoke"):
        stat.run_smoke_recheck(config, **_hooks())


def test_smoke_engineering_statistics_end_to_end(tmp_path):
    fixture = _build_smoke_side(tmp_path)
    config = fixture["config"]
    stat.run_smoke_recheck(config, **_hooks())
    report = stat.run_smoke_engineering_statistics(config, n_resamples=64)
    paths = stat.smoke_artifact_paths(config)
    assert paths["engineering_report"].is_file()
    assert report["status"] == cinf.STATUS_SMOKE == "ENGINEERING_SMOKE_ONLY"
    assert report["engineering_only"] is True
    assert "no scientific content" in report["no_scientific_content"]
    assert report["n_groups"] == SMOKE_G
    assert report["bootstrap"]["seed"] == prot.BOOTSTRAP_ENTROPY
    assert report["bootstrap"]["n_resamples"] == 64
    assert report["bootstrap"]["reduced_resamples"] is True  # recorded prominently
    diagnostics = report["gate_diagnostics"]
    assert diagnostics["n_groups"] == SMOKE_G
    # 12 groups can never satisfy the formal gate thresholds (c1/c7): recorded as
    # engineering diagnostics, never raised, never a scientific verdict.
    assert diagnostics["gate_task_confirm"]["conditions"]["c1_all_complete_gate_q"] is False
    assert diagnostics["gate_task_confirm"]["conditions"]["c7_strata_ge_200_groups"] is False
    assert diagnostics["status"] == prot.STATUS_INVALID
    assert diagnostics["passed"] is False
    assert report["qa"]["one_shot_locks"] is False  # smoke NEVER burns a lock
    assert report["formal_side_effects"] == {
        "protocol_lock_created": False, "confirmation_used_lock_burned": False,
        "formal_artifacts_written": False, "formal_truth_loader_used": False}
    text = paths["engineering_report"].read_text(encoding="utf-8")
    again = stat.run_smoke_engineering_statistics(config, n_resamples=64)
    assert again["verification"] == "existing_smoke_engineering_report_verified_no_overwrite"
    assert paths["engineering_report"].read_text(encoding="utf-8") == text


def test_smoke_engineering_statistics_requires_the_smoke_recheck_first(tmp_path):
    fixture = _build_smoke_side(tmp_path)
    with pytest.raises(ValueError, match="recheck"):
        stat.run_smoke_engineering_statistics(fixture["config"], n_resamples=16)


def test_smoke_engineering_statistics_raises_on_nonfinite_computation(tmp_path):
    fixture = _build_smoke_side(tmp_path)
    config = fixture["config"]
    stat.run_smoke_recheck(config, **_hooks())
    # non-finite smoke truth: the engineering bug class smoke exists to catch. Rewrite
    # the smoke truth with a NaN and refresh the smoke receipt binding so the failure
    # is the non-finite computation itself, not the hash guard.
    with np.load(fixture["truth"]) as z:
        payload = {k: z[k] for k in z.files}
    futures = payload["object_counterfactual_futures"].copy()
    futures[0, 0, 0, 0, 0] = np.nan
    payload["object_counterfactual_futures"] = futures
    np.savez_compressed(fixture["truth"], **payload)
    receipt_path = Path(config["run_dir"]) / "smoke" / "smoke_collection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["confirmation_truth_sha256"] = sha256_file(fixture["truth"])
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    with pytest.raises(ValueError):
        stat.run_smoke_engineering_statistics(config, n_resamples=16)


def test_smoke_statistics_never_touch_formal_paths_or_the_judge_loader(tmp_path, monkeypatch):
    fixture = _build_smoke_side(tmp_path)
    config = fixture["config"]
    for name in ("load_confirmation_truth", "burn_confirmation_lock", "freeze_scores",
                 "assemble_confirm_quantities", "adjudicate_confirmation",
                 "run_determinism_recheck"):
        monkeypatch.setattr(stat, name,
                            lambda *a, **k: pytest.fail(f"smoke must not call {name}"))
    stat.run_smoke_recheck(config, **_hooks())
    stat.run_smoke_engineering_statistics(config, n_resamples=64)
    formal = stat.confirm_artifact_paths(config)
    for key, path in formal.items():
        assert not Path(path).exists(), key
