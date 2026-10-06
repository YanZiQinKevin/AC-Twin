"""Stage 4B confirmatory statistics tests: AURC/C80 fixtures, 95% bootstrap, Gate AND."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4_statistics as s4s  # noqa: E402
from ac_twin import stage4b_power as s4bpw  # noqa: E402
from ac_twin import stage4b_protocol as s4bp  # noqa: E402
from ac_twin import stage4b_statistics as s4bs  # noqa: E402


# --- hand-computed AURC / ADE@80 fixture ---------------------------------------------------


def test_aurc_hand_fixture():
    # 4 episodes; risk ranks error perfectly ascending -> lowest-risk kept first.
    error = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([0.1, 0.2, 0.3, 0.4])
    # R(k): 1, 1.5, 2, 2.5 -> AURC = mean = 1.75
    assert s4s.aurc(risk, error) == pytest.approx(1.75)
    # Reverse ranking (worst first): R(k): 4, 3.5, 3, 2.5 -> AURC = 3.25
    assert s4s.aurc(risk[::-1], error) == pytest.approx(3.25)


def test_ade_at_coverage_hand_fixture():
    error = np.array([1.0, 2.0, 3.0, 4.0])
    risk = np.array([0.4, 0.3, 0.2, 0.1])  # ascending risk keeps error 4,3,2,1 order
    # coverage 0.5 -> k=2 lowest-risk = episodes with risk 0.1,0.2 -> error 4,3 -> 3.5
    assert s4s.ade_at_coverage(risk, error, 0.5) == pytest.approx(3.5)


# --- 95% paired bootstrap preserves multiplicity + reproducibility --------------------------


def _inputs(episodes=120, seed=1, spread=0.2):
    rng = np.random.default_rng(seed)
    rt = rng.gamma(2.0, 0.2, size=episodes)
    return {
        "comb_score": rt + rng.normal(0, spread, episodes),
        "geo_score": rt + rng.normal(0, 0.9, episodes),
        "unc_score": rt + rng.normal(0, 0.6, episodes),
        "risk_target": rt,
        "primary_error_by_pair": np.stack([rt + rng.normal(0, 0.05, episodes) for _ in range(5)]),
        "cf_action": np.tile(np.arange(5), episodes // 5 + 1)[:episodes],
        "cf_duration": np.tile(np.repeat([1, 3], 5), episodes // 10 + 1)[:episodes],
        "top_error_threshold": 0.4,
    }


def test_confirm_bootstrap_reproducible_and_95_width():
    inputs = _inputs()
    a = s4bs.confirm_paired_bootstrap(inputs, n_resamples=400, seed=2026091703)
    b = s4bs.confirm_paired_bootstrap(inputs, n_resamples=400, seed=2026091703)
    assert np.array_equal(a["draws"], b["draws"])  # frozen entropy -> identical
    assert np.all(a["ci_lower"] <= a["point"] + 1e-9)
    assert np.all(a["ci_upper"] >= a["point"] - 1e-9)
    # 95% two-sided: lower is the 2.5% quantile.
    assert a["ci_lower"][0] == pytest.approx(np.quantile(a["draws"][:, 0], 0.025))


# --- Gate RISK_confirm AND (intersection-union) --------------------------------------------


def _point_ci(i_aurc_unc=0.09, i_c80_unc=0.05, i_aurc_geo=0.10, i_c80_geo=0.06,
              pr=0.30, pairs=(0.08, 0.10, 0.10, 0.05, 0.09), strata_min=0.05,
              ci_lo=0.01):
    point = np.zeros(s4bpw.N_QUANTITIES)
    ci = np.full(s4bpw.N_QUANTITIES, ci_lo)
    point[s4bpw.Q_I_AURC_UNC] = i_aurc_unc
    point[s4bpw.Q_I_C80_UNC] = i_c80_unc
    point[s4bpw.Q_I_AURC_GEO] = i_aurc_geo
    point[s4bpw.Q_I_C80_GEO] = i_c80_geo
    point[s4bpw.Q_PR] = pr
    for p in range(5):
        point[s4bpw.Q_PAIR_START + p] = pairs[p]
    for slot in list(s4bpw.ACTION_INDEX.values()) + list(s4bpw.DURATION_INDEX.values()):
        point[slot] = strata_min
    return point, ci


def test_gate_passes_when_all_conditions_hold():
    point, ci = _point_ci()
    verdict = s4bs.evaluate_gate_risk_confirm(point, ci, qa_passed=True)
    assert verdict["passed"] and verdict["status"] == s4bp.STATUS_CONFIRMED


def test_gate_fails_if_i_aurc_below_002():
    point, ci = _point_ci(i_aurc_unc=0.015)  # below the 0.02 floor
    verdict = s4bs.evaluate_gate_risk_confirm(point, ci, qa_passed=True)
    assert not verdict["passed"] and verdict["status"] == s4bp.STATUS_NOT_CONFIRMED
    assert not verdict["conditions"]["i_aurc_vs_uncertainty"]


def test_gate_fails_if_ci_lower_crosses_zero():
    point, ci = _point_ci()
    ci[s4bpw.Q_I_C80_UNC] = -0.001  # CI lower not > 0
    verdict = s4bs.evaluate_gate_risk_confirm(point, ci, qa_passed=True)
    assert not verdict["conditions"]["i_c80_vs_uncertainty"]
    assert verdict["status"] == s4bp.STATUS_NOT_CONFIRMED


def test_gate_fails_if_geometry_not_beaten():
    point, ci = _point_ci(i_aurc_geo=-0.02)  # candidate loses to geometry
    verdict = s4bs.evaluate_gate_risk_confirm(point, ci, qa_passed=True)
    assert not verdict["conditions"]["i_aurc_vs_geometry"]


def test_gate_fails_if_fewer_than_4_pairs_positive():
    point, ci = _point_ci(pairs=(0.08, -0.01, -0.02, 0.05, 0.09))  # only 3 positive
    verdict = s4bs.evaluate_gate_risk_confirm(point, ci, qa_passed=True)
    assert not verdict["conditions"]["at_least_4_of_5_pairs_positive"]
    assert verdict["positive_pairs"] == 3


def test_gate_fails_if_a_stratum_below_floor():
    point, ci = _point_ci(strata_min=-0.20)  # below the -0.10 floor
    verdict = s4bs.evaluate_gate_risk_confirm(point, ci, qa_passed=True)
    assert not verdict["conditions"]["no_strata_below_floor"]


def test_gate_q_failure_is_invalid_analysis():
    point, ci = _point_ci()
    verdict = s4bs.evaluate_gate_risk_confirm(point, ci, qa_passed=False)
    assert verdict["status"] == s4bp.STATUS_INVALID
    assert not verdict["passed"]
