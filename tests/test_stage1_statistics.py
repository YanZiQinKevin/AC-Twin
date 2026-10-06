from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_statistics import (
    BIN_NAMES,
    GATE_B_NOT_SUPPORTED,
    GATE_B_SUPPORTED,
    GATE_B_UNDERPOWERED,
    GATE_BLOCKED,
    GATE_S_FAIL,
    GATE_S_PASS,
    OUTCOME_GO,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_NO_GO,
    aggregate_bins,
    assign_response_bin,
    build_trend_design,
    episode_bootstrap,
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_s,
    fit_response_bins,
    make_spearman_statistic,
    make_v_evt_low_high_statistic,
    make_v_evt_statistic,
    ols_coefficient,
    position_errors,
    resolve_stage1_outcome,
    response_magnitudes,
    ridge_fit,
    ridge_predict,
    seed_mean,
    spearman,
    value_metrics,
)


def test_position_errors_and_response_magnitudes():
    pred = np.zeros((1, 2, 1, 2))
    target = np.zeros((1, 2, 1, 2))
    target[0, 0, 0] = [3.0, 4.0]
    errors = position_errors(pred, target)
    assert errors.shape == (1, 2, 1)
    np.testing.assert_allclose(errors[0, 0, 0], 5.0)

    cf = np.zeros((1, 1, 1, 8))
    factual = np.zeros((1, 1, 1, 8))
    cf[0, 0, 0, 2:4] = [0.0, 5.0]
    cf[0, 0, 0, 4:6] = [12.0, 0.0]
    r_pos, r_vel = response_magnitudes(cf, factual)
    np.testing.assert_allclose(r_pos[0, 0, 0], 5.0)
    np.testing.assert_allclose(r_vel[0, 0, 0], 12.0)


def test_value_metric_directions():
    l_suffix = np.array([1.0])
    l_shuffled = np.array([3.0])
    l_no = np.array([4.0])
    metrics = value_metrics(l_suffix, l_shuffled, l_no, epsilon=1e-4)
    assert metrics["D_evt"][0] > 0          # shuffled worse than correct suffix
    assert metrics["Delta"][0] > 0          # no-suffix worse than correct suffix
    assert metrics["eta_evt"][0] < 0        # correct suffix lower log-error


def test_seed_mean_averages_seed_axis():
    errors = np.stack([np.ones((2, 3, 2)), np.full((2, 3, 2), 3.0)])
    np.testing.assert_allclose(seed_mean(errors), np.full((2, 3, 2), 2.0))


def test_response_bins_fit_and_boundaries():
    train_r = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    thresholds = fit_response_bins(train_r, zero_tolerance=1e-4,
                                   positive_quantiles=(1 / 3, 2 / 3))
    t1, t2 = thresholds["t1"], thresholds["t2"]
    assert t1 == pytest.approx(np.quantile([1, 2, 3, 4, 5, 6], 1 / 3))
    assert t2 == pytest.approx(np.quantile([1, 2, 3, 4, 5, 6], 2 / 3))
    bins = assign_response_bin(np.array([0.0, 1e-5, t1 - 1e-3, (t1 + t2) / 2, t2 + 1.0]),
                               thresholds)
    assert bins.tolist() == [0, 0, 1, 2, 3]


def test_response_bins_require_positive_responses():
    with pytest.raises(ValueError, match="positive"):
        fit_response_bins(np.zeros(5), 1e-4, (1 / 3, 2 / 3))


def test_aggregate_bins_exact_values():
    l_suffix = np.array([[[2.0]], [[4.0]], [[6.0]]])
    l_shuffled = np.array([[[4.0]], [[8.0]], [[8.0]]])
    l_no = np.array([[[10.0]], [[10.0]], [[10.0]]])
    bin_id = np.array([[[0]], [[3]], [[3]]])
    valid = np.ones((3, 1, 1), bool)
    out = aggregate_bins(l_no, l_suffix, l_shuffled, bin_id, valid)
    assert out["zero"]["V_evt"] == pytest.approx(1.0 - 2.0 / 4.0)
    assert out["zero"]["n_episodes"] == 1
    assert out["positive_high"]["V_evt"] == pytest.approx(1.0 - 10.0 / 16.0)
    assert out["positive_high"]["rho_no"] == pytest.approx(1.0 - 10.0 / 20.0)
    assert out["positive_high"]["n_episodes"] == 2
    assert out["positive_mid"]["n_agent_time"] == 0


def test_aggregate_bins_zero_denominator_unavailable():
    l_suffix = np.array([[[1.0]]])
    l_shuffled = np.array([[[0.0]]])
    l_no = np.array([[[2.0]]])
    bin_id = np.array([[[0]]])
    valid = np.ones((1, 1, 1), bool)
    out = aggregate_bins(l_no, l_suffix, l_shuffled, bin_id, valid)
    assert out["zero"]["V_evt"] is None          # zero shuffled denominator
    assert out["zero"]["rho_no"] == pytest.approx(1.0 - 1.0 / 2.0)


def test_v_evt_low_high_statistic_point_value():
    l_suffix = np.array([[[2.0]], [[4.0]], [[6.0]]])
    l_shuffled = np.array([[[4.0]], [[8.0]], [[8.0]]])
    bin_id = np.array([[[0]], [[3]], [[3]]])
    valid = np.ones((3, 1, 1), bool)
    statistic = make_v_evt_low_high_statistic(l_suffix, l_shuffled, bin_id, valid,
                                              low_bins=(0,), high_bins=(3,))
    assert statistic(np.arange(3)) == pytest.approx(0.5 - 0.375)


def test_v_evt_statistic_point_value():
    l_suffix = np.array([[[2.0]], [[4.0]]])
    l_shuffled = np.array([[[4.0]], [[8.0]]])
    bin_id = np.array([[[0]], [[0]]])
    valid = np.ones((2, 1, 1), bool)
    statistic = make_v_evt_statistic(l_suffix, l_shuffled, bin_id, valid, bins=(0,))
    assert statistic(np.arange(2)) == pytest.approx(1.0 - 6.0 / 12.0)


def test_episode_bootstrap_resamples_whole_episodes_not_agent_times():
    # Four episodes with very different episode-level means. Replicated
    # agent-times must not narrow the CI: the episode bootstrap stays wide.
    episode_values = np.array([0.0, 0.0, 10.0, 10.0])

    def statistic(indices):
        return episode_values[indices].mean()

    result = episode_bootstrap(4, statistic, n_resamples=2000,
                               confidence_level=0.95, seed=0)
    assert result["point"] == pytest.approx(5.0)
    assert result["ci_upper"] - result["ci_lower"] > 4.0  # wide: episode-level variance


def test_episode_bootstrap_requires_two_episodes():
    with pytest.raises(ValueError, match="at least two"):
        episode_bootstrap(1, lambda i: 0.0, 10, 0.95, 0)


def test_spearman_directions_and_ties():
    assert spearman(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])) == pytest.approx(1.0)
    assert spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])) == pytest.approx(-1.0)
    # Ties get average ranks.
    value = spearman(np.array([1, 1, 2, 2]), np.array([1, 2, 3, 4]))
    assert -1.0 <= value <= 1.0


def test_spearman_episode_statistic():
    x = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])  # E=2,T=2,N=1
    y = np.array([[[1.0], [2.0]], [[2.0], [5.0]]])
    valid = np.ones((2, 2, 1), bool)
    statistic = make_spearman_statistic(x, y, valid)
    assert statistic(np.arange(2)) == pytest.approx(spearman(x.reshape(-1), y.reshape(-1)))


def test_spearman_episode_statistic_preserves_bootstrap_multiplicity():
    """Repeated episode draws must contribute repeatedly to the statistic."""
    x = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])
    y = np.array([[[0.0], [1.0]], [[3.0], [2.0]]])
    valid = np.ones((2, 2, 1), bool)
    statistic = make_spearman_statistic(x, y, valid)
    expected_x = np.concatenate([x[0].reshape(-1), x[0].reshape(-1), x[1].reshape(-1)])
    expected_y = np.concatenate([y[0].reshape(-1), y[0].reshape(-1), y[1].reshape(-1)])
    assert statistic(np.array([0, 0, 1])) == pytest.approx(spearman(expected_x, expected_y))


def test_ols_coefficient_recovers_main_regressor():
    rng = np.random.default_rng(0)
    log_r = rng.uniform(0, 2, 200)
    X = np.column_stack([np.ones(200), log_r])
    y = 0.5 + 2.0 * log_r + rng.normal(0, 0.01, 200)
    assert ols_coefficient(X, y, column=1) == pytest.approx(2.0, abs=1e-2)


def test_build_trend_design_shapes_and_main_column():
    episodes, T, N = 4, 3, 2
    valid = np.ones((episodes, T, N), bool)
    eta = np.random.default_rng(0).normal(size=(episodes, T, N))
    r_pos = np.abs(np.random.default_rng(1).normal(size=(episodes, T, N)))
    horizon = np.broadcast_to(np.arange(T)[None, :, None], (episodes, T, N)).astype(float)
    cf_action = np.broadcast_to(np.arange(episodes)[:, None, None], (episodes, T, N))
    cf_duration = np.ones((episodes, T, N))
    initial_distance = np.ones((episodes, T, N)) * 5
    visible_ratio = np.ones((episodes, T, N)) * 0.5
    visible_count = np.ones((episodes, T, N)) * 3
    standardizer = {
        "horizon_index": {"mean": 1.0, "std": 1.0},
        "cf_duration": {"mean": 1.0, "std": 1.0},
        "initial_distance": {"mean": 5.0, "std": 1.0},
        "history_visible_ratio": {"mean": 0.5, "std": 0.1},
        "visible_count": {"mean": 3.0, "std": 1.0},
    }
    X, y, names, log_col = build_trend_design(
        eta, r_pos, horizon, cf_action, cf_duration, initial_distance,
        visible_ratio, visible_count, valid, standardizer)
    assert X.shape[0] == episodes * T * N
    assert names[log_col] == "log1p_R_pos"
    assert len(y) == episodes * T * N


# --- Gate state machine ----------------------------------------------------

def test_ridge_fit_and_predict_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(200), rng.normal(size=200), rng.normal(size=200)])
    beta_true = np.array([1.0, 2.0, -3.0])
    y = X @ beta_true + rng.normal(0, 0.01, 200)
    beta = ridge_fit(X, y, lam=1e-6)
    np.testing.assert_allclose(beta, beta_true, atol=1e-2)
    np.testing.assert_allclose(ridge_predict(X, beta), y, atol=0.1)


def test_gate_s_requires_all_three():
    assert evaluate_gate_s(True, True, True) == GATE_S_PASS
    assert evaluate_gate_s(True, True, False) == GATE_S_FAIL


def test_gate_a_blocked_until_gate_s_passes():
    assert evaluate_gate_a(GATE_S_FAIL, True) == GATE_BLOCKED
    assert evaluate_gate_a(GATE_S_FAIL, False) == GATE_BLOCKED
    with pytest.raises(ValueError):
        evaluate_gate_a(GATE_S_PASS, None)
    assert evaluate_gate_a(GATE_S_PASS, True) == OUTCOME_GO
    assert evaluate_gate_a(GATE_S_PASS, False) == OUTCOME_NO_GO


def test_gate_b_states():
    assert evaluate_gate_b(GATE_S_FAIL, -1.0, -2.0, 40, 30) == GATE_BLOCKED
    assert evaluate_gate_b(GATE_S_PASS, -1.0, -2.0, 20, 30) == GATE_B_UNDERPOWERED
    assert evaluate_gate_b(GATE_S_PASS, -0.5, -2.0, 40, 30) == GATE_B_SUPPORTED
    assert evaluate_gate_b(GATE_S_PASS, 0.5, -2.0, 40, 30) == GATE_B_NOT_SUPPORTED
    assert evaluate_gate_b(GATE_S_PASS, 0.5, 1.0, 40, 30) == GATE_B_NOT_SUPPORTED


def test_resolve_outcome_never_skips_gate_s():
    assert resolve_stage1_outcome(GATE_S_FAIL, GATE_BLOCKED) == OUTCOME_INCONCLUSIVE
    with pytest.raises(ValueError):
        resolve_stage1_outcome(GATE_S_PASS, GATE_BLOCKED)
    assert resolve_stage1_outcome(GATE_S_PASS, OUTCOME_GO) == OUTCOME_GO
    assert resolve_stage1_outcome(GATE_S_PASS, OUTCOME_NO_GO) == OUTCOME_NO_GO
