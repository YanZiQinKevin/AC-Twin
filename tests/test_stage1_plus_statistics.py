from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_plus_statistics import (  # noqa: E402
    classify_shape_recommendation,
    compute_shape_diagnostics,
    grouped_v_evt,
    make_group_contrast_statistic,
    make_rebound_statistic,
)

# --- grouped_v_evt ------------------------------------------------------------


def test_grouped_v_evt_ratio_direction():
    # suffix error half of shuffled over the selection -> V_evt = 1 - 0.5 = 0.5.
    l_suffix = np.full((2, 2, 2), 1.0)
    l_shuffled = np.full((2, 2, 2), 2.0)
    selection = np.ones((2, 2, 2), dtype=bool)
    assert grouped_v_evt(l_suffix, l_shuffled, selection) == pytest.approx(0.5)


def test_grouped_v_evt_negative_when_suffix_worse():
    l_suffix = np.full((2, 2, 2), 3.0)
    l_shuffled = np.full((2, 2, 2), 2.0)
    selection = np.ones((2, 2, 2), dtype=bool)
    assert grouped_v_evt(l_suffix, l_shuffled, selection) == pytest.approx(1.0 - 1.5)


def test_grouped_v_evt_partial_selection():
    l_suffix = np.array([[[1.0, 1.0], [1.0, 1.0]]])  # E=1,T=2,N=2
    l_shuffled = np.array([[[2.0, 4.0], [8.0, 16.0]]])
    selection = np.array([[[True, False], [False, False]]])
    # only the (0,0,0) element: 1 - 1/2 = 0.5
    assert grouped_v_evt(l_suffix, l_shuffled, selection) == pytest.approx(0.5)


def test_grouped_v_evt_rejects_shape_mismatch():
    l_suffix = np.ones((2, 2, 2))
    l_shuffled = np.ones((2, 2, 2))
    bad = np.ones((2, 2, 1), dtype=bool)
    with pytest.raises(ValueError):
        grouped_v_evt(l_suffix, l_shuffled, bad)


def test_grouped_v_evt_rejects_empty_selection():
    l_suffix = np.ones((2, 2, 2))
    l_shuffled = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        grouped_v_evt(l_suffix, l_shuffled, np.zeros((2, 2, 2), dtype=bool))


def test_grouped_v_evt_rejects_non_finite():
    l_suffix = np.ones((2, 2, 2))
    l_suffix[0, 0, 0] = np.nan
    l_shuffled = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        grouped_v_evt(l_suffix, l_shuffled, np.ones((2, 2, 2), dtype=bool))


def test_grouped_v_evt_rejects_non_positive_denominator():
    l_suffix = np.ones((2, 2, 2))
    l_shuffled = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        grouped_v_evt(l_suffix, l_shuffled, np.ones((2, 2, 2), dtype=bool))


# --- make_group_contrast_statistic --------------------------------------------


def _toy_errors():
    # E=3, T=1, N=2 (slot 0 ego excluded by valid mask below).
    l_suffix = np.array([
        [[0.0, 1.0]],
        [[0.0, 2.0]],
        [[0.0, 4.0]],
    ])
    l_shuffled = np.array([
        [[0.0, 2.0]],
        [[0.0, 4.0]],
        [[0.0, 8.0]],
    ])
    return l_suffix, l_shuffled


def test_group_contrast_statistic_value_and_episode_clustering():
    l_suffix, l_shuffled = _toy_errors()
    valid = np.zeros((3, 1, 2), dtype=bool)
    valid[:, :, 1] = True  # only slot 1 (non-ego)
    # left = episodes {0,1} via selection, right = episode {2}.
    left = valid & np.isin(np.arange(3)[:, None, None], [0, 1])
    right = valid & np.isin(np.arange(3)[:, None, None], [2])
    stat = make_group_contrast_statistic(l_suffix, l_shuffled, left, right)
    # V_left = 1 - (1+2)/(2+4) = 1 - 0.5 = 0.5 ; V_right = 1 - 4/8 = 0.5
    assert stat(np.array([0, 1, 2])) == pytest.approx(0.0)
    # Resampling episode 2 twice (multiplicity) still uses episode-level sums.
    # left sums over episodes {0,1}: same; right episode 2 counted twice.
    assert stat(np.array([0, 1, 2, 2])) == pytest.approx(0.0)


def test_group_contrast_statistic_episode_is_unit_not_agent_time():
    # Duplicating agent-times inside an episode must not create new independent
    # episodes: the statistic keys on episode indices, not rows.
    l_suffix, l_shuffled = _toy_errors()
    valid = np.zeros((3, 1, 2), dtype=bool)
    valid[:, :, 1] = True
    left = valid & np.isin(np.arange(3)[:, None, None], [0, 1])
    right = valid & np.isin(np.arange(3)[:, None, None], [2])
    stat = make_group_contrast_statistic(l_suffix, l_shuffled, left, right)
    # Leaving out episode 1 changes the left sum deterministically.
    value = stat(np.array([0, 2]))
    # V_left over episode 0 only = 1 - 1/2 = 0.5 ; V_right = 1 - 4/8 = 0.5
    assert value == pytest.approx(0.0)


def test_group_contrast_statistic_nan_on_zero_denominator():
    l_suffix = np.ones((2, 1, 2))
    l_shuffled = np.zeros((2, 1, 2))  # shuffled all zero -> denominator 0
    valid = np.ones((2, 1, 2), dtype=bool)
    stat = make_group_contrast_statistic(l_suffix, l_shuffled, valid, valid)
    assert np.isnan(stat(np.array([0, 1])))


# --- make_rebound_statistic ----------------------------------------------------


def _rebound_errors():
    # E=4 episodes, T=1, N=2 (slot1 non-ego). Bins assigned per episode.
    # Make high bin clearly better (higher V) than low/mid.
    l_suffix = np.array([
        [[0.0, 1.0]],   # ep0 low
        [[0.0, 1.0]],   # ep1 mid
        [[0.0, 1.0]],   # ep2 high
        [[0.0, 1.0]],   # ep3 high
    ])
    l_shuffled = np.array([
        [[0.0, 2.0]],   # low V=0.5
        [[0.0, 2.0]],   # mid V=0.5
        [[0.0, 4.0]],   # high V=0.75
        [[0.0, 4.0]],   # high V=0.75
    ])
    bin_id = np.array([
        [[0, 1]],  # ep0 -> positive_low (bin1)
        [[0, 2]],  # ep1 -> positive_mid (bin2)
        [[0, 3]],  # ep2 -> positive_high (bin3)
        [[0, 3]],  # ep3 -> positive_high (bin3)
    ])
    valid = np.zeros((4, 1, 2), dtype=bool)
    valid[:, :, 1] = True
    return l_suffix, l_shuffled, bin_id, valid


def test_rebound_statistic_value():
    l_suffix, l_shuffled, bin_id, valid = _rebound_errors()
    stat = make_rebound_statistic(l_suffix, l_shuffled, bin_id, valid)
    idx = np.arange(4)
    # V_low = 1 - 1/2 = 0.5 ; V_mid = 0.5 ; V_high = 1 - (1+1)/(4+4) = 0.75
    # C_high_low = 0.25 ; C_high_mid = 0.25 ; C_rebound = 0.25
    assert stat(idx) == pytest.approx(0.25)


def test_rebound_statistic_takes_min_of_two_contrasts():
    l_suffix, l_shuffled, bin_id, valid = _rebound_errors()
    # Make mid bin worse so C_high_mid < C_high_low.
    l_shuffled = l_shuffled.copy()
    l_shuffled[1, 0, 1] = 1.25  # mid V = 1 - 1/1.25 = 0.2
    stat = make_rebound_statistic(l_suffix, l_shuffled, bin_id, valid)
    # V_low=0.5, V_mid=0.2, V_high=0.75 -> C_high_low=0.25, C_high_mid=0.55
    assert stat(np.arange(4)) == pytest.approx(0.25)


# --- classify_shape_recommendation --------------------------------------------


def _flags(rebound, magnitude, presence):
    return {"shape_flags": {
        "ROBUST_HIGH_REBOUND_SUPPORTED": rebound,
        "POSITIVE_MAGNITUDE_DECAY_SUPPORTED": magnitude,
        "PRESENCE_DIFFERENCE_SUPPORTED": presence,
    }}


def test_classify_rebound_first():
    assert classify_shape_recommendation(_flags(True, True, True)) == \
        "REVISE_H2_TO_NONMONOTONIC_BEFORE_STAGE2"


def test_classify_magnitude_second():
    assert classify_shape_recommendation(_flags(False, True, True)) == \
        "KEEP_ORIGINAL_H2_ADD_SHAPE_FALSIFICATION"


def test_classify_presence_third():
    assert classify_shape_recommendation(_flags(False, False, True)) == \
        "SHRINK_CLAIM_TO_RESPONSE_PRESENCE_BEFORE_STAGE2"


def test_classify_no_stable_shape():
    assert classify_shape_recommendation(_flags(False, False, False)) == \
        "NO_STABLE_SHAPE_ON_DEVELOPMENT_DATA"


# --- weighted statistic invariants ----------------------------------------------

from ac_twin.stage1_statistics import (  # noqa: E402
    make_spearman_statistic,
    make_trend_coefficient_statistic,
)
from ac_twin.stage1_plus_statistics import (  # noqa: E402
    estimate_power_curve,
    make_weighted_spearman_statistic,
    make_weighted_trend_statistic,
)


def test_weighted_spearman_equals_unweighted_at_unit_weights():
    rng = np.random.default_rng(0)
    E, T, N = 6, 3, 4
    valid = np.zeros((E, T, N), dtype=bool)
    valid[:, :, 1:] = True
    x = rng.normal(size=(E, T, N))
    y = rng.normal(size=(E, T, N)) * 0.5 + x * 0.3
    ar = np.arange(E)
    weighted = make_weighted_spearman_statistic(x, y, valid)(ar)
    unweighted = make_spearman_statistic(x, y, valid)(ar)
    assert weighted == pytest.approx(unweighted, abs=1e-12)


def _trend_fixture(seed):
    rng = np.random.default_rng(seed)
    E, T, N = 6, 3, 4
    valid = np.zeros((E, T, N), dtype=bool)
    valid[:, :, 1:] = True
    p = 5
    rows = int(valid.sum())
    X = rng.normal(size=(rows, p))
    X[:, 0] = 1.0
    y = X @ np.array([0.1, 0.5, -0.2, 0.3, 0.05]) + rng.normal(scale=0.1, size=rows)
    ei = np.broadcast_to(np.arange(E)[:, None, None], (E, T, N)).reshape(-1)
    row_ep = ei[valid.reshape(-1)]
    return X, y, row_ep, E, p


def test_weighted_trend_equals_ridge_at_unit_weights():
    X, y, row_ep, E, p = _trend_fixture(1)
    ar = np.arange(E)
    weighted = make_weighted_trend_statistic(X, y, 1, row_ep, E)(ar)
    ridge = make_trend_coefficient_statistic(X, y, 1, row_ep, E)(ar)
    assert weighted == pytest.approx(ridge, abs=1e-10)


def test_weighted_trend_multiplicity_matches_row_expansion():
    X, y, row_ep, E, p = _trend_fixture(2)
    stat = make_weighted_trend_statistic(X, y, 1, row_ep, E)
    idx = np.array([0, 0, 1, 2, 3, 4, 5])  # episode 0 counted twice
    mask0 = row_ep == 0
    Xe = np.vstack([X, X[mask0]])
    ye = np.concatenate([y, y[mask0]])
    pen = 1e-6 * np.eye(p)
    pen[0, 0] = 0.0
    expected = np.linalg.solve(Xe.T @ Xe + pen, Xe.T @ ye)[1]
    assert stat(idx) == pytest.approx(expected, abs=1e-10)


# --- synthetic compute_shape_diagnostics scenarios -------------------------------


def _build_specs(specs, n_seeds=5, T=4, N=3, bootstrap_resamples=200, seed=123):
    """Build compute_shape_diagnostics kwargs; specs = list of (bin, v, cf_action)."""
    r_center = {0: 0.0, 1: 0.1, 2: 1.0, 3: 3.0}
    E = len(specs)
    l_sh = np.zeros((E, T, N))
    l_su = np.zeros((E, T, N))
    l_no = np.zeros((E, T, N))
    r_pos = np.zeros((E, T, N))
    bin_id = np.zeros((E, T, N), dtype=int)
    valid = np.zeros((E, T, N), dtype=bool)
    cf_action = np.zeros(E, dtype=int)
    cf_duration = np.zeros(E)
    for e, (b, v, act) in enumerate(specs):
        valid[e, :, 1:] = True
        for t in range(T):
            for n in range(1, N):
                r_pos[e, t, n] = r_center[b]
                bin_id[e, t, n] = b
                l_sh[e, t, n] = 1.0
                l_su[e, t, n] = (1.0 - v)
                l_no[e, t, n] = 1.2
        cf_action[e] = act
        cf_duration[e] = 5 + (e % 3)

    def byseed(a):
        return np.stack([a] * n_seeds)

    horizon = np.broadcast_to(np.arange(T)[None, :, None], (E, T, N)).astype(float)
    covariates = {
        "horizon_index": horizon,
        "cf_action": np.broadcast_to(cf_action[:, None, None], (E, T, N)),
        "cf_duration": np.broadcast_to(cf_duration[:, None, None], (E, T, N)).astype(float),
        "initial_distance": np.broadcast_to((1 + r_pos[:, 0, :])[:, None, :], (E, T, N)),
        "history_visible_ratio": np.ones((E, T, N)),
        "visible_count": np.full((E, T, N), float(N)),
    }
    standardizer = {nm: {"mean": 0.0, "std": 1.0} for nm in
                    ("horizon_index", "cf_duration", "initial_distance",
                     "history_visible_ratio", "visible_count")}
    return dict(l_no_by_seed=byseed(l_no), l_suffix_by_seed=byseed(l_su),
                l_shuffled_by_seed=byseed(l_sh), r_pos=r_pos, bin_id=bin_id,
                valid_mask=valid, cf_action=cf_action, covariates=covariates,
                train_standardizer=standardizer, bootstrap_resamples=bootstrap_resamples,
                confidence_level=0.95, bootstrap_seed=123)


def _uniform_specs(v_per_bin, n_ep_per_bin=(40, 40, 40, 40)):
    return [(b, v_per_bin[b], i % 5)
            for b, ne in enumerate(n_ep_per_bin) for i in range(ne)]


def test_shape_rebound_state():
    specs = _uniform_specs({0: 0.3, 1: 0.05, 2: 0.0, 3: 0.25})
    d = compute_shape_diagnostics(**_build_specs(specs))
    assert d["shape_flags"]["ROBUST_HIGH_REBOUND_SUPPORTED"] is True
    assert d["shape_recommendation"] == "REVISE_H2_TO_NONMONOTONIC_BEFORE_STAGE2"


def test_shape_positive_magnitude_decay_state():
    specs = _uniform_specs({0: 0.3, 1: 0.20, 2: 0.10, 3: 0.02})
    kwargs = _build_specs(specs)
    r_pos = kwargs["r_pos"]
    bin_id = kwargs["bin_id"]
    pos = bin_id >= 1
    l_su = kwargs["l_suffix_by_seed"][0].copy()
    l_su[pos] *= (1.0 + 0.05 * r_pos[pos])  # eta rises with R within positive
    kwargs["l_suffix_by_seed"] = np.stack([l_su] * 5)
    d = compute_shape_diagnostics(**kwargs)
    assert d["shape_flags"]["ROBUST_HIGH_REBOUND_SUPPORTED"] is False
    assert d["shape_flags"]["POSITIVE_MAGNITUDE_DECAY_SUPPORTED"] is True
    assert d["shape_recommendation"] == "KEEP_ORIGINAL_H2_ADD_SHAPE_FALSIFICATION"


def test_shape_presence_only_state():
    specs = _uniform_specs({0: 0.35, 1: 0.05, 2: 0.05, 3: 0.05})
    d = compute_shape_diagnostics(**_build_specs(specs))
    assert d["shape_flags"]["POSITIVE_MAGNITUDE_DECAY_SUPPORTED"] is False
    assert d["shape_flags"]["ROBUST_HIGH_REBOUND_SUPPORTED"] is False
    assert d["shape_flags"]["PRESENCE_DIFFERENCE_SUPPORTED"] is True
    assert d["shape_recommendation"] == "SHRINK_CLAIM_TO_RESPONSE_PRESENCE_BEFORE_STAGE2"


def test_shape_no_stable_state():
    specs = _uniform_specs({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    d = compute_shape_diagnostics(**_build_specs(specs))
    assert d["shape_recommendation"] == "NO_STABLE_SHAPE_ON_DEVELOPMENT_DATA"
    assert d["shape_flags"] == {
        "PRESENCE_DIFFERENCE_SUPPORTED": False,
        "POSITIVE_MAGNITUDE_DECAY_SUPPORTED": False,
        "ROBUST_HIGH_REBOUND_SUPPORTED": False,
    }


def test_rebound_fails_when_positive_bin_too_small():
    specs = ([(0, 0.3, i % 5) for i in range(40)]
             + [(1, 0.05, i % 5) for i in range(40)]
             + [(2, 0.0, i % 5) for i in range(40)]
             + [(3, 0.25, i % 5) for i in range(20)])  # positive_high = 20 < 30
    d = compute_shape_diagnostics(**_build_specs(specs))
    assert d["rebound"]["bin_episode_counts"]["positive_high"] == 20
    assert d["shape_flags"]["ROBUST_HIGH_REBOUND_SUPPORTED"] is False


def test_rebound_fails_on_single_action_reversal():
    specs = ([(0, 0.3, i % 5) for i in range(40)]
             + [(1, 0.05, i % 5) for i in range(40)]
             + [(2, 0.05, i % 5) for i in range(40)]
             + [(3, 0.6, 0) for _ in range(10)]
             + [(3, 0.0, (i % 4) + 1) for i in range(30)])
    d = compute_shape_diagnostics(**_build_specs(specs))
    per_action = d["rebound"]["leave_one_action_type_out"]
    assert per_action[0]["c_rebound"] <= 0
    assert d["shape_flags"]["ROBUST_HIGH_REBOUND_SUPPORTED"] is False


def test_rebound_reports_leave_one_episode_out():
    specs = _uniform_specs({0: 0.3, 1: 0.05, 2: 0.0, 3: 0.25})
    d = compute_shape_diagnostics(**_build_specs(specs))
    assert d["rebound"]["leave_one_episode_out_n_reversed"] == 0
    assert d["rebound"]["leave_one_episode_out_min"] > 0
    assert "drop_top_1pct" in d["rebound"]["high_error_deletion"]
    assert "drop_top_5pct" in d["rebound"]["high_error_deletion"]


# --- power analysis (Task 4) ------------------------------------------------------


def _mean_statistic(values):
    values = np.asarray(values, dtype=np.float64)

    def stat(indices):
        return float(values[np.asarray(indices)].mean())

    stat.pool_size = len(values)
    return stat


def _gate_a_dict(values):
    stat = _mean_statistic(values)
    return {name: stat for name in
            ("v_evt_low", "v_evt_low_minus_high", "spearman_r_eta", "trend_log_r")}


def test_power_reproducible_same_seed():
    values = np.random.default_rng(7).normal(0.5, 1.0, size=20)
    stats = _gate_a_dict(values)
    kwargs = dict(candidate_episode_counts=[5, 10, 20], monte_carlo_replicates=500,
                  confidence_level=0.95, seed=42)
    a = estimate_power_curve(episode_statistics=stats, **kwargs)
    b = estimate_power_curve(episode_statistics=stats, **kwargs)
    assert a["per_candidate"] == b["per_candidate"]


def test_power_joint_is_and_not_product():
    values = np.random.default_rng(7).normal(0.5, 1.0, size=20)
    stats = _gate_a_dict(values)  # all four identical -> joint == single
    out = estimate_power_curve(episode_statistics=stats, candidate_episode_counts=[20],
                               monte_carlo_replicates=1000, confidence_level=0.95, seed=3)
    single = out["per_candidate"][20]["single"]["v_evt_low"]["empirical"]
    joint = out["per_candidate"][20]["joint_numeric"]["empirical"]
    assert joint == pytest.approx(single, abs=1e-9)
    if 0.0 < single < 1.0:
        assert joint != pytest.approx(single ** 4)  # not a product of singles


def test_power_strong_effect_beats_weak():
    strong = _gate_a_dict(np.random.default_rng(0).normal(2.0, 1.0, size=30))
    weak = _gate_a_dict(np.random.default_rng(0).normal(0.2, 1.0, size=30))
    common = dict(candidate_episode_counts=[10, 30], monte_carlo_replicates=500,
                  confidence_level=0.95, seed=11)
    ps = estimate_power_curve(episode_statistics=strong, **common)
    pw = estimate_power_curve(episode_statistics=weak, **common)
    assert (ps["per_candidate"][30]["single"]["v_evt_low"]["empirical"]
            >= pw["per_candidate"][30]["single"]["v_evt_low"]["empirical"])


def test_power_not_reached_returns_sentinel():
    zero = _gate_a_dict(np.zeros(30))  # no effect at all
    out = estimate_power_curve(episode_statistics=zero, candidate_episode_counts=[10, 30],
                               monte_carlo_replicates=300, confidence_level=0.95, seed=5)
    assert out["sample_size_summary"]["empirical"]["smallest_n_joint_numeric_ge_0.90"] == \
        "NOT_REACHED_WITHIN_2400"


def test_power_shrinkage_changes_point_not_se():
    values = np.random.default_rng(7).normal(0.8, 1.0, size=30)
    stats = _gate_a_dict(values)
    out = estimate_power_curve(episode_statistics=stats, candidate_episode_counts=[30],
                               monte_carlo_replicates=800, confidence_level=0.95, seed=9)
    cell = out["per_candidate"][30]["single"]["v_evt_low"]
    assert cell["empirical"] >= cell["75pct"] >= cell["50pct"]


def test_power_requires_gate_a_numeric_keys():
    with pytest.raises(ValueError):
        estimate_power_curve(
            episode_statistics={"v_evt_low": _mean_statistic(np.ones(5))},
            candidate_episode_counts=[5], monte_carlo_replicates=10,
            confidence_level=0.95, seed=1)


def test_power_rejects_non_095_confidence():
    stats = _gate_a_dict(np.ones(10))
    with pytest.raises(ValueError):
        estimate_power_curve(episode_statistics=stats, candidate_episode_counts=[10],
                             monte_carlo_replicates=10, confidence_level=0.90, seed=1)
