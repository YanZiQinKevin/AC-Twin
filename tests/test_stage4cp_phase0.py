"""Blocking tests for the Stage 4C Plus Phase 0 OOF screen (plan sections 5, 14)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_stage4cp as r4cp  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402
from ac_twin import stage4cp_statistics as s4cs  # noqa: E402

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/stage4_reliability/stage4c_plus/stage4cp_config.json")


def _config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["device"] = "cpu"
    return cfg


# --- fold assignment ---------------------------------------------------------------------


def test_fold_assignment_disjoint_balanced_deterministic():
    n, k = 500, int(s4cp.PHASE0_N_FOLDS)
    folds = r4cp._phase0_fold_assignment(n, k, s4cp.PHASE0_FOLD_ENTROPY)
    assert folds.shape == (n,)
    # every group in exactly one fold; folds cover all groups.
    assert sorted(np.unique(folds).tolist()) == list(range(k))
    counts = np.bincount(folds, minlength=k)
    assert counts.max() - counts.min() <= 1  # balanced whole-group split
    # deterministic in the frozen entropy.
    folds2 = r4cp._phase0_fold_assignment(n, k, s4cp.PHASE0_FOLD_ENTROPY)
    np.testing.assert_array_equal(folds, folds2)
    # a group never crosses folds between the two frozen-entropy draws is trivially true;
    # the real invariant is each group maps to exactly one fold id (no re-entry).
    assert folds.dtype == np.int64


# --- quantity vector ----------------------------------------------------------------------


def _synthetic_curves(n=400, seed=7):
    """regret random; candidate PERFECT (score=regret); margin CONSTANT (weak baseline)."""
    rng = np.random.default_rng(seed)
    regret = rng.gamma(shape=1.5, scale=0.01, size=n)          # right-skewed regrets
    cand = regret.copy()                                        # perfect ranking
    margin = np.zeros(n)                                        # constant weak baseline
    high = (regret > s4cp.HIGH_REGRET_THRESHOLD).astype(np.int64)
    return cand, margin, regret, high


def test_phase0_quantity_vector_direction():
    cand, margin, regret, high = _synthetic_curves()
    index = np.arange(cand.size)
    q = r4cp._phase0_quantity_vector(index, cand, margin, regret, high)
    assert q.shape == (4,)
    # perfect candidate vs constant baseline => positive improvements.
    assert q[r4cp._P0_I_AURC] > 0.0
    assert q[r4cp._P0_I_REGRET80] > 0.0
    # perfect candidate ranks high-regret groups top => PR_AUC - prevalence > 0;
    # constant margin baseline => 0.
    assert q[r4cp._P0_PR_CAND] > 0.0
    assert q[r4cp._P0_PR_MARGIN] == pytest.approx(0.0, abs=1e-9)


def test_phase0_quantity_vector_inverted_candidate_worse():
    cand, margin, regret, high = _synthetic_curves()
    index = np.arange(cand.size)
    bad = -regret  # anti-perfect: high risk assigned to low-regret groups
    q = r4cp._phase0_quantity_vector(index, bad, margin, regret, high)
    assert q[r4cp._P0_I_AURC] < 0.0


# --- gate logic ----------------------------------------------------------------------------


def test_evaluate_candidate_pass_and_fail():
    cand, margin, regret, high = _synthetic_curves()
    folds = r4cp._phase0_fold_assignment(cand.size, int(s4cp.PHASE0_N_FOLDS),
                                         s4cp.PHASE0_FOLD_ENTROPY)
    cfg = _config()

    passed = r4cp._phase0_evaluate_candidate(
        s4cp.ARCH_DCR, cand, margin, regret, high, folds, cfg)
    assert passed["passed"] is True
    assert all(passed["conditions"].values())
    assert passed["folds_positive"] >= s4cp.PHASE0_MIN_FOLDS_POSITIVE

    # anti-perfect candidate must FAIL the i_aurc condition (and overall).
    failed = r4cp._phase0_evaluate_candidate(
        s4cp.ARCH_DIRECT, -regret, margin, regret, high, folds, cfg)
    assert failed["conditions"]["i_aurc_regret"] is False
    assert failed["passed"] is False


def test_evaluate_candidate_margin_constant_gives_zero_pr_margin():
    cand, margin, regret, high = _synthetic_curves()
    folds = r4cp._phase0_fold_assignment(cand.size, int(s4cp.PHASE0_N_FOLDS),
                                         s4cp.PHASE0_FOLD_ENTROPY)
    out = r4cp._phase0_evaluate_candidate(
        s4cp.ARCH_DCR, cand, margin, regret, high, folds, _config())
    assert out["point"]["pr_auc_minus_prevalence_margin"] == pytest.approx(0.0, abs=1e-9)
