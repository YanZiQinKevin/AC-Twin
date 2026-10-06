"""Blocking tests for the Stage 5 statistics layer (plan v0.3 section 7.2, Task 4).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Pins: risk-curve
metric parity with the frozen Stage 4 formulas, the secondary-panel definitions
(coverage profile, tail severity/capture, oracle gap closure, retained-tail
stats), native calibration only where semantically valid, TOST equivalence, and
fail-closed improvement denominators.
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

from ac_twin import stage4cp_statistics as s4cs  # noqa: E402
from ac_twin import stage5_statistics as s5st  # noqa: E402


def make_truth(g: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    regret = rng.rand(g) * 0.3
    return {
        "regret": regret,
        "high_regret": (regret > 0.01).astype(np.int64),
        "selected_action": rng.randint(0, 10, g).astype(np.int64),
        "oracle_action": rng.randint(0, 10, g).astype(np.int64),
    }


# --- primary-metric parity with the frozen Stage 4 formulas --------------------------------


def test_evaluate_group_metrics_parity_with_stage4():
    truth = make_truth()
    rng = np.random.RandomState(1)
    ensemble = rng.rand(200)
    by_seed = ensemble[None, :] + rng.rand(5, 200) * 0.01
    m = s5st.evaluate_group_metrics(ensemble, by_seed, truth)
    order = s4cs.order_by_risk(ensemble)
    assert m["aurc_regret"] == pytest.approx(s4cs.aurc_regret(order, truth["regret"]))
    assert m["regret_at_80"] == pytest.approx(
        s4cs.regret_at_coverage(order, truth["regret"], 0.80))
    assert m["regret_at_70"] == pytest.approx(
        s4cs.regret_at_coverage(order, truth["regret"], 0.70))
    assert m["regret_at_90"] == pytest.approx(
        s4cs.regret_at_coverage(order, truth["regret"], 0.90))
    assert m["regret_at_95"] == pytest.approx(
        s4cs.regret_at_coverage(order, truth["regret"], 0.95))
    assert m["pr_auc_minus_prevalence"] == pytest.approx(
        s4cs.high_regret_pr(ensemble, truth["high_regret"]))
    assert m["per_seed_aurc"].shape == (5,)
    assert m["per_seed_regret80"].shape == (5,)


def test_improvement_fail_closed_nonpositive_denominator():
    assert np.isnan(s5st.improvement_fail_closed(0.0, 0.0))
    assert np.isnan(s5st.improvement_fail_closed(-1.0, 0.5))
    assert s5st.improvement_fail_closed(2.0, 1.0) == pytest.approx(0.5)


# --- secondary panel -------------------------------------------------------------------------


def test_coverage_profile_spans_five_to_hundred_percent():
    truth = make_truth()
    risk = np.random.RandomState(2).rand(200)
    coverages, values = s5st.coverage_profile(risk, truth["regret"])
    assert coverages[0] == pytest.approx(0.05)
    assert coverages[-1] == pytest.approx(1.0)
    assert coverages.size == 20 and values.shape == coverages.shape
    assert values[-1] == pytest.approx(float(truth["regret"].mean()))


def test_tail_severity_and_capture():
    regret = np.arange(100, dtype=np.float64)  # risk perfectly tracks regret
    risk = regret.copy()
    # top 20% highest risk = the 20 largest regrets
    assert s5st.tail_severity(risk, regret, fraction=0.2) == pytest.approx(
        regret[80:].mean())
    y_high = (regret >= 80).astype(np.int64)
    assert s5st.tail_capture(risk, y_high, fraction=0.2) == pytest.approx(1.0)
    y_none = np.zeros(100, dtype=np.int64)
    assert s5st.tail_capture(risk, y_none, fraction=0.2) == pytest.approx(0.0)


def test_oracle_gap_closure_perfect_and_degenerate():
    regret = np.random.RandomState(3).rand(200) * 0.3 + 0.01
    perfect = regret.copy()
    assert s5st.oracle_gap_closure(perfect, regret, coverage=0.8) == pytest.approx(1.0)
    # degenerate: all regrets equal -> denominator 0 -> fail-closed NaN
    flat_risk = np.zeros(50)
    flat_regret = np.ones(50) * 0.1
    assert np.isnan(s5st.oracle_gap_closure(flat_risk, flat_regret, coverage=0.8))


def test_retained_tail_stats():
    risk = np.arange(100, dtype=np.float64)
    regret = np.arange(100, dtype=np.float64) * 0.01
    p95, mx = s5st.retained_tail_stats(risk, regret, coverage=0.8)
    retained = regret[:80]
    assert mx == pytest.approx(float(retained.max()))
    assert p95 == pytest.approx(float(np.quantile(retained, 0.95)))


# --- native calibration -----------------------------------------------------------------------


def test_native_calibration_only_where_semantically_valid():
    truth = make_truth()
    for method_id in ("SELECTIVENET_REGRET_ADAPTED", "RWR_NN_KNN_REJECT_ADAPTED"):
        out = s5st.native_calibration(method_id, None, truth)
        assert out["status"] == "NOT_APPLICABLE"
    # P4 conformal: empirical interval coverage from mu + half-width
    raw = {"mu": truth["regret"] - 0.5, "half_width": np.full(200, 1.0)}
    out = s5st.native_calibration("CONFORMAL_REGRESSION_REJECT_ADAPTED", raw, truth)
    assert out["status"] == "OK"
    assert out["empirical_coverage"] == pytest.approx(1.0)
    # P5: gaussian NLL + 95% interval coverage (wildly under-wide sigma -> zero coverage)
    raw5 = {"mu": truth["regret"] + 0.5, "sigma": np.full(200, 1e-3)}
    out5 = s5st.native_calibration("CCTR_DRIVING_CALIBRATION_ADAPTED", raw5, truth)
    assert out5["status"] == "OK"
    assert out5["empirical_coverage_95"] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(out5["nll"])
    # P1: TCP vs selected-is-oracle empirical frequency (10-bin ECE)
    tcp = np.where(truth["selected_action"] == truth["oracle_action"], 0.9, 0.1)
    out1 = s5st.native_calibration("CONFIDNET_TCP_ADAPTED", {"tcp": tcp}, truth)
    assert out1["status"] == "OK"
    assert 0.0 <= out1["ece"] <= 1.0


# --- TOST -------------------------------------------------------------------------------------


def test_tost_equivalence_bounds():
    # 90% CI inside [-0.05, +0.05] -> equivalent
    assert s5st.tost_equivalent(0.01, 0.02, (-0.05, 0.05)) is True
    # CI touches/exceeds the bound -> not equivalent
    assert s5st.tost_equivalent(0.01, 0.06, (-0.05, 0.05)) is False
    assert s5st.tost_equivalent(-0.06, 0.01, (-0.05, 0.05)) is False
    # non-finite CI -> fail closed
    assert s5st.tost_equivalent(float("nan"), 0.02, (-0.05, 0.05)) is False
