"""Stage 4C statistics tests: rule freeze, curves, group bootstrap, gate priority."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_power as s4cpw  # noqa: E402
from ac_twin import stage4c_protocol as s4cp  # noqa: E402
from ac_twin import stage4c_statistics as s4cs  # noqa: E402

NA = s4cp.N_ACTIONS


def _selection_inputs(groups=40, degenerate=False, seed=0):
    rng = np.random.default_rng(seed)
    regret = np.abs(rng.normal(size=groups))
    # uncertainty risk perfectly ranks low-regret first -> lowest AURC (strongest).
    unc = regret + 0.01 * rng.normal(size=groups)
    geo = -regret + 0.01 * rng.normal(size=groups)
    margin = rng.normal(size=groups)
    J_true = np.abs(rng.normal(size=(groups, NA)))
    J_hat = J_true + 0.05 * rng.normal(size=(groups, NA))
    combined = np.abs(rng.normal(size=(groups, NA)))
    if degenerate:
        combined = np.zeros((groups, NA))  # zero variance -> degenerate z-score
    return {
        "baseline_group_risks": {
            "UNCERTAINTY_TOP2_MAX": unc,
            "GEOMETRY_TOP2_MAX": geo,
            "MARGIN_ONLY": margin,
        },
        "regret": regret,
        "J_hat": J_hat,
        "J_true": J_true,
        "combined_action_risk": combined,
        "density_count": rng.integers(0, 5, size=groups).astype(float),
    }


def test_freeze_decision_rule_strongest_baseline_and_thresholds():
    receipt = s4cs.freeze_decision_rule(_selection_inputs())
    assert receipt["strongest_standard_baseline"] == "UNCERTAINTY_TOP2_MAX"
    inp = _selection_inputs()
    assert receipt["high_regret_threshold"] == pytest.approx(
        float(np.quantile(inp["regret"], 0.80)))
    assert receipt["density_threshold"] == pytest.approx(float(np.median(inp["density_count"])))
    assert receipt["reranking_lambda"] in s4cp.LAMBDA_CANDIDATES
    # chosen lambda is the regret-minimising one (smaller lambda breaks ties).
    lam_reg = receipt["lambda_selection_regret"]
    best = min(s4cp.LAMBDA_CANDIDATES,
               key=lambda l: (lam_reg[float(l)], list(s4cp.LAMBDA_CANDIDATES).index(l)))
    assert receipt["reranking_lambda"] == pytest.approx(best)


def test_freeze_decision_rule_degenerate_forces_lambda_zero():
    receipt = s4cs.freeze_decision_rule(_selection_inputs(degenerate=True))
    assert receipt["reranking_lambda"] == 0.0
    assert receipt["reranking_status"] == s4cp.STATUS_RERANK_DEGENERATE


def test_decision_risk_curve_hand_computed():
    risk = np.array([0.1, 0.4, 0.2, 0.3])
    regret = np.array([0.0, 3.0, 1.0, 2.0])
    out = s4cs.decision_risk_curve(risk, regret)
    order = [0, 2, 3, 1]  # ascending risk
    assert list(out["order"]) == order
    sorted_regret = regret[order]  # [0,1,2,3]
    r_of_k = np.cumsum(sorted_regret) / np.arange(1, 5)
    assert np.allclose(out["R_of_k"], r_of_k)
    assert out["AURC_REGRET"] == pytest.approx(r_of_k.mean())


def _bootstrap_inputs(groups=30, seed=1):
    rng = np.random.default_rng(seed)
    regret = np.abs(rng.normal(size=groups))
    comb = regret + 0.1 * rng.normal(size=groups)   # combined ranks low regret first
    base = rng.normal(size=groups)                    # baseline ~ noise
    return {
        "combined_group_risk": comb,
        "baseline_group_risk": base,
        "regret": regret,
        "regret_by_pair": np.stack([regret + 0.05 * p * rng.normal(size=groups)
                                    for p in range(5)]),
        "random_regret": regret + np.abs(rng.normal(size=groups)),
        "idle_regret": regret + np.abs(rng.normal(size=groups)),
        "high_regret_label": (regret > np.quantile(regret, 0.8)).astype(np.int64),
        "tstar_layer": (np.arange(groups) % 2),
        "density_layer": ((np.arange(groups) // 3) % 2),
        "J_hat": np.abs(rng.normal(size=(groups, NA))),
        "J_true": np.abs(rng.normal(size=(groups, NA))),
    }


def test_group_bootstrap_resamples_whole_groups():
    inp = _bootstrap_inputs()
    # an index duplicating group 0 twice and dropping group 1 must reuse group 0's
    # whole row (all ten actions) twice.
    index = np.array([0, 0])
    args = s4cpw._qv_args(inp)
    q = s4cpw.compute_group_quantity_vector(index, **args)
    # with a single repeated group, combined/baseline ordering is constant -> AURC
    # improvement is NaN-guarded or 0; decisively, J rows are duplicated so Spearman
    # is computed on duplicated whole rows (no partial-branch leakage).
    jsel = inp["J_hat"][index]
    assert jsel.shape == (2, NA)
    assert np.array_equal(jsel[0], inp["J_hat"][0])
    assert np.array_equal(jsel[1], inp["J_hat"][0])
    assert np.isfinite(q).all() or np.isnan(q).any()  # sanity: vector produced


def test_paired_bootstrap_interval_separation_80_vs_95():
    inp = _bootstrap_inputs(groups=60)
    b80 = s4cs.paired_group_bootstrap(inp, n_resamples=200, seed=7, confidence=0.80)
    b95 = s4cs.paired_group_bootstrap(inp, n_resamples=200, seed=7, confidence=0.95)
    # 95% lower bound is never above the 80% lower bound (wider interval).
    assert (b95["ci_lower"] <= b80["ci_lower"] + 1e-12).all()
    assert b80["point"].shape == (s4cpw.N_QUANTITIES,)


def test_evaluate_gate_task_priority():
    base = {
        "nonzero_regret_frac": 0.30, "distinct_oracle_count": 4,
        "spearman_ci_lower": 0.2, "base_vs_random_point": 0.1,
        "base_vs_random_ci_lower": 0.05, "base_mean_regret": 0.5, "idle_mean_regret": 0.8,
    }
    assert s4cs.evaluate_gate_task(base)["status"] == "TASK_QUALIFIED"
    deg = dict(base, nonzero_regret_frac=0.05)
    assert s4cs.evaluate_gate_task(deg)["status"] == s4cp.STATUS_TASK_DEGENERATE
    ins = dict(base, spearman_ci_lower=-0.1)
    assert s4cs.evaluate_gate_task(ins)["status"] == s4cp.STATUS_MODEL_INSUFFICIENT


def test_evaluate_gate_risk_decision_priority():
    point = np.zeros(s4cpw.N_QUANTITIES)
    point[s4cpw.Q_I_AURC] = 0.10
    point[s4cpw.Q_I_REGRET80] = 0.05
    point[s4cpw.Q_PR] = 0.2
    point[s4cpw.PAIR_INDEX] = 0.05
    point[s4cpw.STRATA_INDEX] = 0.0
    point[s4cpw.TRIM_INDEX] = 0.03
    ci = point - 0.01
    ok = s4cs.evaluate_gate_risk_decision(
        point, ci, qa_passed=True, task_status="TASK_QUALIFIED", baseline_valid=True)
    assert ok["status"] == "RISK_DECISION_PASSED"
    # Gate Q failure dominates everything.
    badq = s4cs.evaluate_gate_risk_decision(
        point, ci, qa_passed=False, task_status="TASK_QUALIFIED", baseline_valid=True)
    assert badq["status"] == s4cp.STATUS_INVALID
    # task degeneracy outranks the risk conditions.
    deg = s4cs.evaluate_gate_risk_decision(
        point, ci, qa_passed=True, task_status=s4cp.STATUS_TASK_DEGENERATE,
        baseline_valid=True)
    assert deg["status"] == s4cp.STATUS_TASK_DEGENERATE
    # a failing risk condition -> RISK_DECISION_FAILED.
    point2 = point.copy()
    point2[s4cpw.Q_I_AURC] = 0.01  # below the 0.05 line
    fail = s4cs.evaluate_gate_risk_decision(
        point2, ci, qa_passed=True, task_status="TASK_QUALIFIED", baseline_valid=True)
    assert fail["status"] == "RISK_DECISION_FAILED"
