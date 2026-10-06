"""Stage 4C objective tests: consequence function, tie-break, regret, masks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_objective as s4co  # noqa: E402


# --- action-plan contract -----------------------------------------------------------


def test_action_plans_are_ten_unique_5x2_contracts():
    plans = s4co.ACTION_PLANS
    assert len(plans) == 10
    assert len(set(plans)) == 10  # unique (action, duration) contracts
    actions = sorted({a for a, _ in plans})
    durations = sorted({d for _, d in plans})
    assert actions == [0, 1, 2, 3, 4]
    assert durations == [1, 3]
    # canonical order: action-major, duration-minor (the unique tie-break order).
    assert plans == tuple((a, d) for a in range(5) for d in (1, 3))


# --- interaction severity (hand-computed) --------------------------------------------


def test_trajectory_components_hand_computed():
    # T=1, N=2, ego at origin; non-ego 5 m ahead longitudinally.
    xy = np.zeros((1, 2, 2), dtype=float)  # (T=1? no) -> build (..., T, N, 2)
    xy = np.zeros((1, 1, 2, 2), dtype=float)  # (G=1, T=1, N=2, xy)
    xy[0, 0, 1, 0] = 5.0  # non-ego x = 5, y = 0
    hist_last = np.array([[0.0, 0.0]])
    mask = np.array([[[False, True]]])  # (G, T, N): ego excluded, non-ego legal
    comp = s4co.trajectory_components(xy, hist_last, mask)
    expected_q = np.exp(-0.5 * (5.0 / 5.0) ** 2)
    assert comp["interaction"][0] == pytest.approx(expected_q, rel=1e-9)
    assert comp["progress"][0] == pytest.approx(0.0, abs=1e-12)


def test_interaction_respects_mask():
    # The only close vehicle is masked out -> interaction falls to the far one.
    xy = np.zeros((1, 1, 3, 2), dtype=float)  # N=3
    xy[0, 0, 1, 0] = 1.0   # very close non-ego
    xy[0, 0, 2, 0] = 50.0  # far non-ego
    hist_last = np.array([[0.0, 0.0]])
    mask_close = np.array([[[False, True, True]]])
    comp = s4co.trajectory_components(xy, hist_last, mask_close)
    assert comp["interaction"][0] == pytest.approx(np.exp(-0.5 * (1.0 / 5.0) ** 2))
    mask_far_only = np.array([[[False, False, True]]])
    comp2 = s4co.trajectory_components(xy, hist_last, mask_far_only)
    assert comp2["interaction"][0] == pytest.approx(np.exp(-0.5 * (50.0 / 5.0) ** 2))


# --- efficiency cost (hand-computed) --------------------------------------------------


def test_efficiency_cost_hand_computed():
    progress = np.array([10.0, 5.0, 0.0])
    c = s4co.efficiency_cost(progress)
    denom = 10.0 + s4co.PROGRESS_EPS
    assert c[0] == pytest.approx(0.0, abs=1e-9)
    assert c[1] == pytest.approx(5.0 / denom, rel=1e-6)
    assert c[2] == pytest.approx(10.0 / denom, rel=1e-6)


def test_efficiency_cost_zero_range_is_zero():
    assert np.all(s4co.efficiency_cost(np.array([3.0, 3.0, 3.0])) == 0.0)


# --- decision outputs: hand-computed regret, tie-break, clamp -------------------------


def _interaction_only_case():
    """G=1, T=1, N=2, progress 0 everywhere so only interaction drives J.

    true non-ego distances put the closest vehicle on plan 7 (worst) and the
    farthest on plan 3 (best); predicted flips plan 7 to look best.
    """
    n_actions = len(s4co.ACTION_PLANS)
    d_true = np.array([50, 50, 50, 200, 50, 50, 50, 0, 50, 50], dtype=float)
    d_pred = np.array([50, 50, 50, 50, 50, 50, 50, 200, 50, 50], dtype=float)

    def build(d):
        xy = np.zeros((1, n_actions, 1, 2, 2), dtype=float)  # (G,A,T,N,xy)
        xy[0, :, 0, 1, 0] = d  # non-ego x per action
        return xy

    true_xy = build(d_true)
    pred_xy = build(d_pred)
    hist_last = np.array([[0.0, 0.0]])
    mask = np.array([[[False, True]]])  # (G,T,N)
    return pred_xy, true_xy, hist_last, mask, d_true


def test_decision_outputs_hand_computed_regret():
    pred_xy, true_xy, hist_last, mask, d_true = _interaction_only_case()
    out = s4co.decision_outputs(pred_xy, true_xy, hist_last, mask)
    assert int(out["a_hat"][0]) == 7   # model thinks plan 7 best
    assert int(out["a_star"][0]) == 3  # truly plan 3 best
    c_int7 = np.exp(-0.5 * (d_true[7] / 5.0) ** 2)  # = 1.0
    c_int3 = np.exp(-0.5 * (d_true[3] / 5.0) ** 2)  # ~ 0.0
    expected_regret = 0.80 * (c_int7 - c_int3)  # C_eff == 0
    assert out["regret"][0] == pytest.approx(expected_regret, rel=1e-9)
    assert out["J_true"][0, 7] == pytest.approx(0.80 * c_int7, rel=1e-9)
    assert out["J_true"][0, 3] == pytest.approx(0.80 * c_int3, rel=1e-9)


def test_deterministic_tiebreak_prefers_lowest_plan_index():
    n = len(s4co.ACTION_PLANS)
    # identical cost everywhere -> argmin must pick plan index 0 for both.
    xy = np.zeros((1, n, 1, 2, 2), dtype=float)
    hist_last = np.array([[0.0, 0.0]])
    mask = np.array([[[False, True]]])
    out = s4co.decision_outputs(xy, xy, hist_last, mask)
    assert int(out["a_hat"][0]) == 0
    assert int(out["a_star"][0]) == 0
    assert out["regret"][0] == 0.0


def test_negative_regret_below_tolerance_raises():
    with pytest.raises(ValueError):
        s4co.clamp_regret(np.array([-1e-9]))
    # within tolerance -> clamped to 0
    assert s4co.clamp_regret(np.array([-1e-11]))[0] == 0.0


def test_decision_outputs_rejects_wrong_action_count():
    xy = np.zeros((1, 9, 1, 2, 2), dtype=float)  # 9 != 10 plans
    hist_last = np.array([[0.0, 0.0]])
    mask = np.array([[[False, True]]])
    with pytest.raises(ValueError):
        s4co.decision_outputs(xy, xy, hist_last, mask)
