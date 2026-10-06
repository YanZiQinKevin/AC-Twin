from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_statistics as s3s  # noqa: E402


# --- per-episode endpoints -------------------------------------------------------


def test_effect_ade_is_episode_level() -> None:
    pred = np.zeros((2, 20, 5, 2))
    target = np.zeros((2, 20, 5, 2))
    valid = np.ones((2, 20, 5), dtype=bool)
    target[0, :, :, 0] = 1.0  # episode 0 error 1 everywhere
    target[1, :, :, 0] = 3.0
    ade = s3s.effect_ade_per_episode(pred, target, valid)
    assert ade.shape == (2,)
    np.testing.assert_allclose(ade, [1.0, 3.0], rtol=1e-6)


def test_effect_ade_respects_mask() -> None:
    pred = np.zeros((1, 20, 5, 2))
    target = np.zeros((1, 20, 5, 2))
    valid = np.ones((1, 20, 5), dtype=bool)
    target[0, :, :, 0] = 1.0
    target[0, 0, 0, 0] = 100.0
    valid[0, 0, 0] = False  # mask out the huge error
    ade = s3s.effect_ade_per_episode(pred, target, valid)
    np.testing.assert_allclose(ade, [1.0], rtol=1e-6)


def test_effect_fde_uses_final_valid_time() -> None:
    pred = np.zeros((1, 20, 5, 2))
    target = np.zeros((1, 20, 5, 2))
    valid = np.ones((1, 20, 5), dtype=bool)
    target[0, -1, :, 0] = 2.0
    fde = s3s.effect_fde_per_episode(pred, target, valid)
    np.testing.assert_allclose(fde, [2.0], rtol=1e-6)


def test_mae_per_episode() -> None:
    r_hat = np.zeros((2, 20, 5))
    r_true = np.ones((2, 20, 5))
    valid = np.ones((2, 20, 5), dtype=bool)
    mae = s3s.mae_per_episode(r_hat, r_true, valid)
    np.testing.assert_allclose(mae, [1.0, 1.0])


# --- contrast statistics ---------------------------------------------------------


def test_paired_contrast_sign() -> None:
    # C = mean(ref - cand); dual better => ref (uniform/direct) error higher
    ref = np.array([2.0, 3.0, 4.0])
    cand = np.array([1.0, 1.5, 2.0])
    stat = s3s.make_paired_contrast_statistic(ref, cand)
    assert stat(np.array([0, 1, 2])) == pytest.approx(1.5)


def test_relative_contrast() -> None:
    ref = np.array([2.0, 4.0])
    cand = np.array([1.0, 2.0])
    stat = s3s.make_relative_contrast_statistic(ref, cand)
    # mean(ref-cand)=1.5, mean(ref)=3 -> 0.5
    assert stat(np.array([0, 1])) == pytest.approx(0.5)


def test_relative_contrast_degenerate_denominator_nan() -> None:
    ref = np.array([0.0, 0.0])
    cand = np.array([0.0, 0.0])
    stat = s3s.make_relative_contrast_statistic(ref, cand)
    assert np.isnan(stat(np.array([0, 1])))


def test_spearman_statistic_episode_clustered() -> None:
    rng = np.random.default_rng(0)
    e, t, n = 8, 5, 4
    r_true = rng.uniform(0, 3, (e, t, n))
    r_hat = r_true + rng.normal(0, 0.05, (e, t, n))  # strongly correlated
    valid = np.ones((e, t, n), dtype=bool)
    stat = s3s.make_pooled_spearman_statistic(r_hat, r_true, valid)
    assert stat(np.arange(e)) > 0.9


def test_i_mae_direction_baseline_minus_model() -> None:
    """I_MAE = MAE_strongest_baseline - MAE_model; positive means model better."""
    baseline_mae = np.array([2.0, 2.5, 3.0])
    model_mae = np.array([1.0, 1.0, 1.5])
    stat = s3s.make_paired_contrast_statistic(baseline_mae, model_mae)
    assert stat(np.array([0, 1, 2])) > 0


def test_pr_auc_difference_positive_when_model_better() -> None:
    e, t, n = 6, 4, 3
    rng = np.random.default_rng(1)
    labels = (rng.uniform(0, 1, (e, t, n)) > 0.5).astype(int)
    valid = np.ones((e, t, n), dtype=bool)
    model_scores = labels + rng.normal(0, 0.05, (e, t, n))  # near-perfect
    baseline_scores = rng.uniform(0, 1, (e, t, n))  # chance
    stat = s3s.make_pr_auc_difference_statistic(
        model_scores, labels, baseline_scores, labels, valid
    )
    assert stat(np.arange(e)) > 0.3


def test_brier_difference_positive_when_model_better() -> None:
    e, t, n, c = 6, 4, 3, 4
    rng = np.random.default_rng(2)
    labels = rng.integers(0, c, (e, t, n))
    valid = np.ones((e, t, n), dtype=bool)
    model_probs = np.full((e, t, n, c), 0.05)
    for i in range(c):
        model_probs[..., i][labels == i] = 0.85  # confident correct
    baseline_probs = np.full((e, t, n, c), 1.0 / c)  # uniform
    stat = s3s.make_brier_difference_statistic(model_probs, baseline_probs, labels, valid)
    assert stat(np.arange(e)) > 0


# --- Gate R_dev --------------------------------------------------------------------


def _bootstrap_ok(point=0.5, lo=0.1):
    return {"point": point, "ci_lower": lo, "ci_upper": point + 0.3}


def test_gate_r_dev_requires_all_seven() -> None:
    results = {
        "i_mae": _bootstrap_ok(),
        "spearman": _bootstrap_ok(),
        "i_pr_presence": _bootstrap_ok(),
        "i_pr_high_vs_lowmid": _bootstrap_ok(),
        "i_brier": _bootstrap_ok(),
        "ece": 0.03,
        "ece_baseline": 0.06,
        "five_seed_consistent": True,
        "non_degenerate": True,
    }
    assert s3s.evaluate_gate_r_dev(results)["passed"]
    # break one
    results["i_pr_high_vs_lowmid"] = _bootstrap_ok(lo=-0.05)
    out = s3s.evaluate_gate_r_dev(results)
    assert not out["passed"]
    assert not out["checks"]["i_pr_high_vs_lowmid_ci_lower_gt0"]


def test_gate_r_dev_ece_guard_accepts_relative_improvement() -> None:
    results = {
        "i_mae": _bootstrap_ok(), "spearman": _bootstrap_ok(),
        "i_pr_presence": _bootstrap_ok(), "i_pr_high_vs_lowmid": _bootstrap_ok(),
        "i_brier": _bootstrap_ok(),
        "ece": 0.055,  # > 0.05 absolute guard
        "ece_baseline": 0.10,  # but 45% relative improvement >= 20%
        "five_seed_consistent": True, "non_degenerate": True,
    }
    assert s3s.evaluate_gate_r_dev(results)["passed"]


# --- Gate T_dev ----------------------------------------------------------------------


def _tdev_ok():
    return {
        "reconstruction_ok": True, "ablation_ok": True, "no_oracle_leakage": True,
        "oracle_beats_uniform": True,
        "rel_direct": 0.06, "rel_uniform": 0.07,
        "rel_direct_ci80_lower": 0.01, "rel_uniform_ci80_lower": 0.01,
        "crossfit_direct_not_reversed": True, "audit_ok": True,
    }


def test_gate_t_dev_passes_when_all_ok() -> None:
    out = s3s.evaluate_gate_t_dev(_tdev_ok())
    assert out["passed"]
    assert out["status"] is None


def test_gate_t_dev_fails_below_5pct_or_ci() -> None:
    bad = _tdev_ok()
    bad["rel_direct"] = 0.03
    assert not s3s.evaluate_gate_t_dev(bad)["passed"]
    bad = _tdev_ok()
    bad["rel_uniform_ci80_lower"] = -0.01
    out = s3s.evaluate_gate_t_dev(bad)
    assert not out["passed"]
    assert out["status"] == s3s.STATUS_DEV_NOT_READY


def test_gate_t_dev_fails_on_crossfit_reversal() -> None:
    bad = _tdev_ok()
    bad["crossfit_direct_not_reversed"] = False
    assert not s3s.evaluate_gate_t_dev(bad)["passed"]


# --- Formal Gate C state machine (priority order) -----------------------------------


def _formal_ok():
    return {
        "technical_ok": True, "id_h4_ok": True,
        "id_c_uniform_ci_lower": 0.02, "id_c_direct_ci_lower": 0.02,
        "id_rel_uniform": 0.08, "id_rel_direct": 0.08,
        "id_robust_ok": True,
        "oracle_gap": 0.05,
        "unseen_h4_ok": True,
        "unseen_c_uniform_point": 0.03, "unseen_c_direct_point": 0.03,
        "unseen_midhigh_reversed": False,
    }


def test_formal_full_pass() -> None:
    assert s3s.judge_formal_gate_c(_formal_ok()) == s3s.STATUS_GATE_C_CONFIRMED


def test_formal_priority_invalid_technical_first() -> None:
    r = _formal_ok()
    r["technical_ok"] = False
    r["id_h4_ok"] = False
    assert s3s.judge_formal_gate_c(r) == s3s.STATUS_INVALID_TECHNICAL


def test_formal_h4_insufficient() -> None:
    r = _formal_ok()
    r["id_h4_ok"] = False
    assert s3s.judge_formal_gate_c(r) == s3s.STATUS_H4_INSUFFICIENT_ID


def test_formal_primary_not_confirmed_below_5pct() -> None:
    r = _formal_ok()
    r["id_rel_direct"] = 0.04
    assert s3s.judge_formal_gate_c(r) == s3s.STATUS_GATE_C_NOT_CONFIRMED


def test_formal_primary_not_confirmed_ci_crosses_zero() -> None:
    r = _formal_ok()
    r["id_c_uniform_ci_lower"] = -0.01
    assert s3s.judge_formal_gate_c(r) == s3s.STATUS_GATE_C_NOT_CONFIRMED


def test_formal_oracle_invalid() -> None:
    r = _formal_ok()
    r["oracle_gap"] = 0.0
    assert s3s.judge_formal_gate_c(r) == s3s.STATUS_ORACLE_INVALID


def test_formal_external_inconclusive_when_unseen_h4_fails() -> None:
    r = _formal_ok()
    r["unseen_h4_ok"] = False
    assert s3s.judge_formal_gate_c(r) == s3s.STATUS_EXTERNAL_INCONCLUSIVE


def test_formal_external_not_confirmed_on_reversal() -> None:
    r = _formal_ok()
    r["unseen_midhigh_reversed"] = True
    assert s3s.judge_formal_gate_c(r) == s3s.STATUS_EXTERNAL_NOT_CONFIRMED


def test_5pct_screen_and_7p5_planning_are_distinct() -> None:
    assert s3s.MIN_SUBSTANTIVE_REL == 0.05
    assert s3s.PLANNING_REL_MAIN == 0.075
    assert s3s.PLANNING_REL_AUX == 0.10


# --- power -----------------------------------------------------------------------------


def test_power_increases_with_n() -> None:
    rng = np.random.default_rng(0)
    n_dev = 60
    # development paired contrasts with positive mean and noise
    ade_u = rng.uniform(2.0, 3.0, n_dev)
    ade_d = rng.uniform(2.0, 3.0, n_dev)
    c_u = 0.075 * ade_u + rng.normal(0, 0.05, n_dev)
    c_d = 0.075 * ade_d + rng.normal(0, 0.05, n_dev)
    out = s3s.estimate_gate_c_power(
        c_u, c_d, ade_u, ade_d,
        candidate_ns=range(200, 801, 300),
        target_rel=0.075,
        monte_carlo_replicates=20,
        confidence_level=0.95,
        bootstrap_resamples=100,
        seed=0,
    )
    ns = sorted(out)
    assert out[ns[-1]]["joint_power"] >= out[ns[0]]["joint_power"]
