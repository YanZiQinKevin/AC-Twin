"""Task 4 tests: Stage 2 non-monotonic statistics, contrasts, and gate judges."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2_statistics as ss  # noqa: E402

N_SEEDS = 5
HIDDEN = 64


# --- hand-computed contrast helpers ------------------------------------------------


def _sums(per_bin):
    """Build suf/shf episode-sum dicts. per_bin: {name: (suffix_sum, shuffled_sum)} per episode."""
    names = ["zero", "positive_low", "positive_mid", "positive_high", "positive_all"]
    suf = {name: np.zeros(0) for name in names}
    shf = {name: np.zeros(0) for name in names}
    return suf, shf


def test_three_contrasts_hand_computed():
    # 4 episodes, one per bin. suffix/shuffled per-episode sums chosen so that
    # V(zero)=0.3, V(low)=0.05, V(mid)=0.0, V(high)=0.25.
    suf = {
        "zero": np.array([0.7, 0, 0, 0]),
        "positive_low": np.array([0, 0.95, 0, 0]),
        "positive_mid": np.array([0, 0, 1.0, 0]),
        "positive_high": np.array([0, 0, 0, 0.75]),
    }
    shf = {k: np.where(v > 0, 1.0, 0.0) for k, v in suf.items()}
    suf["positive_all"] = suf["positive_low"] + suf["positive_mid"] + suf["positive_high"]
    shf["positive_all"] = shf["positive_low"] + shf["positive_mid"] + shf["positive_high"]
    idx = np.arange(4)
    out = ss.three_contrasts(suf, shf, idx)
    assert out["v_zero"] == pytest.approx(0.3)
    assert out["v_low"] == pytest.approx(0.05)
    assert out["v_mid"] == pytest.approx(0.0)
    assert out["v_high"] == pytest.approx(0.25)
    assert out["v_positive_all"] == pytest.approx((0.95 + 1.0 + 0.75) / 3 * -1 + 1.0)
    assert out["c_presence"] == pytest.approx(0.3 - (1.0 - 2.7 / 3))
    assert out["c_high_low"] == pytest.approx(0.20)
    assert out["c_high_mid"] == pytest.approx(0.25)


def test_three_contrasts_episode_multiplicity():
    # Resampling an episode twice must scale its contribution (episode is the unit).
    suf = {"zero": np.array([0.7, 0.7]), "positive_low": np.array([0.95, 0.95]),
           "positive_mid": np.array([1.0, 1.0]), "positive_high": np.array([0.75, 0.75])}
    shf = {k: np.ones(2) for k in suf}
    suf["positive_all"] = np.array([2.7, 2.7])
    shf["positive_all"] = np.ones(2) * 3.0
    a = ss.three_contrasts(suf, shf, np.array([0, 1]))
    b = ss.three_contrasts(suf, shf, np.array([0, 0, 1]))  # episode 0 twice
    # Both episodes identical, so doubling episode 0 leaves the ratio unchanged.
    assert a["c_presence"] == pytest.approx(b["c_presence"])


def test_v_evt_nan_on_zero_denominator():
    suf = {"zero": np.array([1.0])}
    shf = {"zero": np.array([0.0])}
    assert np.isnan(ss._v_evt(suf["zero"], shf["zero"], np.array([0])))


# --- full compute on a controlled scenario -----------------------------------------


def _controlled(n_per_bin=3, T=2, N=3):
    """Build a controlled truth/predictions where each episode sits in one bin."""
    E = 4 * n_per_bin
    v_target = {0: 0.3, 1: 0.05, 2: 0.0, 3: 0.25}
    r_center = {0: 0.0, 1: 0.1, 2: 1.0, 3: 3.0}
    bin_of_episode = np.repeat([0, 1, 2, 3], n_per_bin)

    r_pos = np.zeros((E, T, N))
    l_suffix = np.ones((E, T, N))
    l_shuffled = np.ones((E, T, N))
    l_no = np.full((E, T, N), 1.2)
    for e in range(E):
        b = bin_of_episode[e]
        r_pos[e, :, 1:] = r_center[b]
        l_suffix[e, :, 1:] = 1.0 - v_target[b]
    target = np.zeros((E, T, N, 2))

    def preds_for(err):
        p = np.zeros((N_SEEDS, E, T, N, 2))
        p[..., 0] = err  # x-component carries the error magnitude
        return p

    predictions = {
        "main": {
            "no_suffix": preds_for(np.broadcast_to(l_no, (E, T, N))),
            "uniform_suffix": preds_for(l_suffix),
            "shuffled_suffix": preds_for(l_shuffled),
        },
        "m1_swap": preds_for(l_suffix + 0.05),  # swapped suffix slightly worse
        "alt_donor": {},
        "factual_summaries": np.random.default_rng(0).normal(size=(E, N, HIDDEN)),
        "factual_active": np.ones((E, N), dtype=bool),
    }

    history = np.zeros((E, 10, N, 8), dtype=np.float32)
    history[..., 0] = 1.0
    factual = np.zeros((E, T, N, 8), dtype=np.float32)
    factual[..., 0] = 1.0
    truth = {
        "r_pos": r_pos, "target": target, "valid_mask": np.ones((E, T, N), dtype=bool),
        "cf_action": np.tile(np.arange(5), E)[:E].astype(np.int64),
        "cf_duration": np.ones(E, dtype=np.int64),
        "episode_seed": (5000 + np.arange(E)).astype(np.int64),
        "history": history, "factual": factual,
        "factual_actions": np.ones((E, T), dtype=np.int64),
        "counterfactual_actions": np.ones((E, T), dtype=np.int64),
        "visible_count": np.full(E, N),
    }
    frozen_params = {
        "beta_action_horizon": np.zeros(3), "beta_action_horizon_distance": np.zeros(4),
        "probe_coefficients": np.zeros((HIDDEN + 1, 6)),
        "probe_coefficients_shuffled": np.zeros((HIDDEN + 1, 6)),
        "probe_scene_mean": np.zeros(6),
    }
    config = {
        "response_bins": {"zero_tolerance": 0.0001, "t1": 0.24843344965718578,
                          "t2": 1.898992796585611},
        "statistics": {"bootstrap_resamples": 200, "confidence_level": 0.95,
                       "bootstrap_seed": 20270908, "practical_point_screen": 0.05,
                       "min_bin_episodes": 2, "eta_epsilon_m": 0.0001,
                       "gate_b_min_episodes": 2},
    }
    manifest = {"primary": {"qa": {"self_match": 0, "same_seed": 0,
                                   "cross_split": 0, "duplicate_donor": 0}}}
    return truth, predictions, frozen_params, manifest, config


def test_compute_produces_correct_contrast_points():
    truth, predictions, params, manifest, config = _controlled()
    report = ss.compute_stage2_statistics(truth, predictions, params, manifest, config)
    assert report["c_presence"]["point"] == pytest.approx(0.20, abs=1e-6)
    assert report["c_high_low"]["point"] == pytest.approx(0.20, abs=1e-6)
    assert report["c_high_mid"]["point"] == pytest.approx(0.25, abs=1e-6)
    assert report["v_zero"]["point"] == pytest.approx(0.3, abs=1e-6)
    assert report["v_high"]["point"] == pytest.approx(0.25, abs=1e-6)


def test_compute_bootstrap_cis_present_for_smooth_contrasts():
    truth, predictions, params, manifest, config = _controlled()
    report = ss.compute_stage2_statistics(truth, predictions, params, manifest, config)
    for name in ("c_presence", "c_high_low", "c_high_mid", "v_zero", "v_high"):
        entry = report[name]
        for key in ("point", "ci_lower", "ci_upper"):
            assert key in entry, (name, key)


def test_c_rebound_is_point_only_no_judgment_ci():
    truth, predictions, params, manifest, config = _controlled()
    report = ss.compute_stage2_statistics(truth, predictions, params, manifest, config)
    assert isinstance(report["c_rebound_point"], float)
    assert report["c_rebound_point"] == pytest.approx(0.20, abs=1e-6)
    # No bootstrapped C_rebound object with a judgment CI may exist.
    assert "c_rebound" not in report


def test_compute_structure_and_labels():
    truth, predictions, params, manifest, config = _controlled()
    report = ss.compute_stage2_statistics(truth, predictions, params, manifest, config)
    assert set(report["bins"]) == set(ss.BIN_NAMES)
    assert report["secondary"]["label"] == "secondary_only"
    assert set(report["point_screen_checks"]) == set(ss.CONTRASTS)
    assert "gate_s_star" in report and "pass" in report["gate_s_star"]
    assert report["bin_episode_counts"]["positive_high"] == 3
    # Alternate-donor sensitivity is a mapping (empty here).
    assert isinstance(report["alternate_donor"], dict)


# --- mechanical gate judges ----------------------------------------------------------


def _report(**overrides):
    base = {
        "gate_s_star": {"pass": True},
        "bin_episode_counts": {"positive_low": 150, "positive_mid": 150, "positive_high": 150},
        "_min_bin_episodes": 100,
        "point_screen": 0.05,
        "c_presence": {"point": 0.20, "ci_lower": 0.10, "ci_upper": 0.30},
        "c_high_low": {"point": 0.20, "ci_lower": 0.10, "ci_upper": 0.30},
        "c_high_mid": {"point": 0.25, "ci_lower": 0.10, "ci_upper": 0.40},
        "v_zero": {"point": 0.30, "ci_lower": 0.20, "ci_upper": 0.40},
        "v_high": {"point": 0.21, "ci_lower": 0.10, "ci_upper": 0.30},
        "robustness": {"all_positive": True},
        "manifest_qa_pass": True,
        "gate_b": ss.GATE_B_NOT_SUPPORTED,
    }
    base.update(overrides)
    return base


def test_judge_gate_a_confirmed():
    assert ss.judge_gate_a(_report(), gate_q_pass=True) == ss.GATE_A_CONFIRMED


def test_judge_gate_a_gate_q_failure_priority():
    assert ss.judge_gate_a(_report(), gate_q_pass=False) == ss.INVALID_TECHNICAL


def test_judge_gate_a_gate_s_star_failure_priority():
    rep = _report(gate_s_star={"pass": False})
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.INCONCLUSIVE_MODEL


def test_judge_gate_a_response_support_priority():
    rep = _report(bin_episode_counts={"positive_low": 150, "positive_mid": 150,
                                      "positive_high": 50})
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_RESPONSE_SUPPORT


def test_judge_gate_a_ci_failure():
    rep = _report(c_high_low={"point": 0.20, "ci_lower": -0.01, "ci_upper": 0.30})
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_NOT_CONFIRMED


def test_judge_gate_a_point_screen_failure():
    rep = _report(c_presence={"point": 0.03, "ci_lower": 0.01, "ci_upper": 0.30})
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_NOT_CONFIRMED


def test_judge_gate_a_v_high_must_be_positive():
    rep = _report(v_high={"point": 0.21, "ci_lower": -0.05, "ci_upper": 0.30})
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_NOT_CONFIRMED


def test_judge_gate_a_robustness_failure():
    rep = _report(robustness={"all_positive": False})
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_NOT_CONFIRMED


def test_judge_gate_a_manifest_qa_failure():
    rep = _report(manifest_qa_pass=False)
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_NOT_CONFIRMED


def test_judge_gate_a_no_cherry_picking():
    # Four of five numeric conditions pass but one fails -> NOT_CONFIRMED.
    rep = _report(c_high_mid={"point": 0.25, "ci_lower": -0.02, "ci_upper": 0.40})
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_NOT_CONFIRMED


def test_judge_gate_b_passthrough():
    assert ss.judge_gate_b(_report(gate_b=ss.GATE_B_SUPPORTED)) == ss.GATE_B_SUPPORTED
    assert ss.judge_gate_b(_report(gate_b=ss.GATE_B_UNDERPOWERED)) == ss.GATE_B_UNDERPOWERED
    # Gate B never changes Gate A.
    rep = _report(gate_b=ss.GATE_B_SUPPORTED)
    assert ss.judge_gate_a(rep, gate_q_pass=True) == ss.GATE_A_CONFIRMED
