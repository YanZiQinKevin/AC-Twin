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


# Hand-computed fixture: 4 episodes, risk already ascending, error ascending.
RISK = np.array([0.0, 1.0, 2.0, 3.0])
ERROR = np.array([10.0, 20.0, 30.0, 40.0])
# order (stable ascending risk) = [0,1,2,3]; sorted_error = [10,20,30,40]
# cumulative = [10,30,60,100]; R(k) = [10,15,20,25]; AURC = 70/4 = 17.5
AURC_GOOD = 17.5
# The reversed ranking (worst-first) gives AURC = 130/4 = 32.5.
AURC_BAD = 32.5


def test_aurc_hand_computed():
    assert s4s.aurc(RISK, ERROR) == pytest.approx(AURC_GOOD)


def test_aurc_lower_is_better():
    # A ranking that keeps the high-error episode LAST (reversed risk) is worse.
    reversed_risk = np.array([3.0, 2.0, 1.0, 0.0])
    assert s4s.aurc(reversed_risk, ERROR) == pytest.approx(AURC_BAD)
    assert s4s.aurc(RISK, ERROR) < s4s.aurc(reversed_risk, ERROR)


def test_coverage_curve_keeps_lowest_risk_first():
    curve = s4s.risk_coverage_curve(RISK, ERROR)
    # The first covered episode is the lowest-risk (risk=0) one.
    assert curve["order"][0] == 0
    # R(1) is the error of the single lowest-risk episode.
    assert curve["selective_error"][0] == pytest.approx(10.0)
    assert np.allclose(curve["coverage"], [0.25, 0.5, 0.75, 1.0])


def test_coverage_index_discrete_rounding():
    assert s4s.coverage_index(0.80, 4) == 3  # round(3.2)
    assert s4s.coverage_index(0.90, 4) == 4  # round(3.6)
    assert s4s.coverage_index(0.10, 4) == 1  # round(0.4) -> clamped to >=1
    assert s4s.coverage_index(1.00, 4) == 4
    with pytest.raises(ValueError):
        s4s.coverage_index(0.0, 4)
    with pytest.raises(ValueError):
        s4s.coverage_index(0.5, 0)


def test_ade_at_coverage_hand_computed():
    assert s4s.ade_at_coverage(RISK, ERROR, 0.80) == pytest.approx(20.0)  # mean 10,20,30
    assert s4s.ade_at_coverage(RISK, ERROR, 0.90) == pytest.approx(25.0)  # mean all four


def test_top_error_threshold_is_train_quantile():
    # quantile(train, 1 - 0.20) = quantile 0.80 of [10,20,30,40] = 34 (linear).
    threshold = s4s.top_error_threshold(ERROR, 0.20)
    assert threshold == pytest.approx(34.0)


def test_pr_auc_minus_prevalence_perfect_ranking():
    threshold = s4s.top_error_threshold(ERROR, 0.20)  # 34 -> label [0,0,0,1]
    result = s4s.pr_auc_minus_prevalence(RISK, ERROR, threshold)
    assert result["prevalence"] == pytest.approx(0.25)
    # risk perfectly ranks the single positive highest -> AP == 1 -> 1 - 0.25.
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["pr_auc_minus_prevalence"] == pytest.approx(0.75)


def test_pr_auc_degenerate_label_is_zero():
    # All episodes below threshold -> all-negative -> PR-AUC == prevalence -> 0.
    result = s4s.pr_auc_minus_prevalence(RISK, ERROR, threshold=1e9)
    assert result["pr_auc_minus_prevalence"] == 0.0


def test_candidate_report_keys_and_freeze():
    threshold = s4s.top_error_threshold(ERROR, 0.20)
    report = s4s.candidate_report(RISK, ERROR, top_error_threshold_value=threshold)
    assert report["aurc"] == pytest.approx(AURC_GOOD)
    assert report["ade@80"] == pytest.approx(20.0)
    assert report["ade@90"] == pytest.approx(25.0)
    assert report["n_episodes"] == 4
    assert "pr_auc_minus_prevalence" in report


def test_paired_bootstrap_is_episode_level_and_deterministic():
    candidate = RISK                      # good ranking
    standard = np.array([3.0, 2.0, 1.0, 0.0])  # bad ranking
    out1 = s4s.paired_improvement_bootstrap(
        candidate, standard, ERROR, n_resamples=200, seed=123, confidence=0.80
    )
    out2 = s4s.paired_improvement_bootstrap(
        candidate, standard, ERROR, n_resamples=200, seed=123, confidence=0.80
    )
    # Same seed -> identical intervals.
    assert out1["i_aurc_ci_lower"] == out2["i_aurc_ci_lower"]
    assert out1["i_c80_ci_lower"] == out2["i_c80_ci_lower"]
    # Point estimates match hand computation.
    assert out1["i_aurc"] == pytest.approx((AURC_BAD - AURC_GOOD) / AURC_BAD)
    # I_C80: standard ade@80=30 (mean 40,30,20), candidate=20 -> (30-20)/30.
    assert out1["i_c80"] == pytest.approx((30.0 - 20.0) / 30.0)
    # A clearly better candidate has a positive interval.
    assert out1["i_aurc_ci_lower"] <= out1["i_aurc_ci_upper"]
    assert out1["i_aurc"] > 0.0


def test_paired_bootstrap_uses_episode_unit_not_cells():
    # Resampling must be over whole episodes; a 4-episode input resamples 4 draws.
    candidate = np.linspace(0.0, 1.0, 4)
    standard = np.linspace(1.0, 0.0, 4)
    error = np.array([1.0, 2.0, 3.0, 4.0])
    out = s4s.paired_improvement_bootstrap(
        candidate, standard, error, n_resamples=50, seed=5, confidence=0.80
    )
    assert out["n_resamples"] == 50
    assert np.isfinite(out["i_aurc_ci_lower"])


def _screen(**overrides):
    base = {
        "i_aurc": 0.05, "i_aurc_ci_lower": 0.01,
        "i_c80": 0.05, "i_c80_ci_lower": 0.01,
        "pr_auc_minus_prevalence": 0.10,
    }
    base.update(overrides)
    return base


_CONFIG = {"gate_i_aurc_min": 0.02, "gate_i_c80_min": 0.02, "gate_min_positive_pairs": 4}
_POSITIVE_PAIRS = np.array([0.1, 0.1, 0.1, 0.1, 0.1])


def test_gate_passes_when_all_conditions_hold():
    verdict = s4s.evaluate_stage4a_gate(
        _screen(), pair_i_aurc=_POSITIVE_PAIRS,
        stratified_reversal=False, qa_passed=True, config=_CONFIG,
    )
    assert verdict["passed"] is True
    assert verdict["status"] == "STAGE4A_RISK_SIGNAL_FEASIBLE"
    assert all(verdict["conditions"].values())


@pytest.mark.parametrize(
    "screen, pairs, reversal, qa",
    [
        (_screen(i_aurc=0.01), _POSITIVE_PAIRS, False, True),   # i_aurc below floor
        (_screen(i_aurc_ci_lower=-0.001), _POSITIVE_PAIRS, False, True),  # CI crosses 0
        (_screen(i_c80=0.0), _POSITIVE_PAIRS, False, True),     # i_c80 below floor
        (_screen(pr_auc_minus_prevalence=0.0), _POSITIVE_PAIRS, False, True),  # no PR skill
        (_screen(), np.array([0.1, 0.1, 0.1, -0.1, -0.1]), False, True),  # only 3/5 pairs
        (_screen(), _POSITIVE_PAIRS, True, True),               # stratified reversal
        (_screen(), _POSITIVE_PAIRS, False, False),             # QA failed
    ],
)
def test_gate_insufficient_when_any_condition_fails(screen, pairs, reversal, qa):
    verdict = s4s.evaluate_stage4a_gate(
        screen, pair_i_aurc=pairs,
        stratified_reversal=reversal, qa_passed=qa, config=_CONFIG,
    )
    assert verdict["passed"] is False
    assert verdict["status"] == "STAGE4A_RISK_SIGNAL_INSUFFICIENT"
    assert not all(verdict["conditions"].values())


def test_statistics_reject_misaligned_or_nonfinite():
    with pytest.raises(ValueError):
        s4s.aurc(np.array([0.0, 1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        s4s.aurc(np.array([0.0, np.nan]), np.array([1.0, 2.0]))
