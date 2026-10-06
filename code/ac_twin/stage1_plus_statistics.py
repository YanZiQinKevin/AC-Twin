"""Stage 1 Plus read-only shape diagnostics and confirmatory power analysis.

This module is a *development-only* diagnostic layer on top of the sealed Stage 1
merge results. It reuses the frozen Stage 1 error/response/bin/bootstrap
definitions from :mod:`ac_twin.stage1_statistics` and never re-fits bins,
thresholds or models. It answers four narrow questions (D1 presence, D2
positive-only magnitude, D3 high-response rebound, D4 power) and mechanically
maps them to a single ``shape_recommendation``. It never re-judges Stage 1
Gate S/A/B and never emits a confirmatory claim.

Statistic closures returned here carry a ``pool_size`` attribute (the number of
independent episodes they index) so :func:`estimate_power_curve` can resample
episode indices without changing its frozen signature. All power statistics are
multiplicity-aware (a resampled episode is weighted by its draw count), which is
the correct resampling generalization for candidate sizes above the 300
available episodes; with unit weights they reduce to the frozen Stage 1
statistics exactly.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from ac_twin.stage1_statistics import (
    aggregate_bins,
    build_trend_design,
    episode_bootstrap,
    make_spearman_statistic,
    make_trend_coefficient_statistic,
    make_v_evt_low_high_statistic,
    make_v_evt_statistic,
    seed_mean,
)

# Frozen analysis constants (mirror config/stage1_plus_merge.json).
ETA_EPSILON_M = 1e-4
RIDGE_LAMBDA = 1e-6
# z_{0.975} for the 95% CI lower-bound power rule (frozen by the plan).
Z_95 = 1.959963984540054

# The four numeric Stage 1 Gate A conditions whose joint power is reported.
GATE_A_NUMERIC_STATISTICS = (
    "v_evt_low",
    "v_evt_low_minus_high",
    "spearman_r_eta",
    "trend_log_r",
)

# Shape-recommendation vocabulary (exactly one is emitted).
REVISE_NONMONOTONIC = "REVISE_H2_TO_NONMONOTONIC_BEFORE_STAGE2"
KEEP_ORIGINAL = "KEEP_ORIGINAL_H2_ADD_SHAPE_FALSIFICATION"
SHRINK_TO_PRESENCE = "SHRINK_CLAIM_TO_RESPONSE_PRESENCE_BEFORE_STAGE2"
NO_STABLE_SHAPE = "NO_STABLE_SHAPE_ON_DEVELOPMENT_DATA"

# Frozen robustness constants (mirror config/stage1_plus_merge.json statistics).
MINIMUM_BIN_EPISODES = 30
DROP_HIGH_ERROR_FRACTIONS = (0.01, 0.05)

# Bootstrap-seed offsets for the *new* diagnostics (the base seed is frozen in
# the config; reproduction of the original Gate A uses point estimates only).
SEED_BIN_V_EVT = 10
SEED_C_PRESENCE = 20
SEED_POS_SPEARMAN = 30
SEED_POS_TREND = 31
SEED_POS_LOW_MINUS_HIGH = 32
SEED_C_REBOUND = 40
SEED_C_HIGH_LOW = 41
SEED_C_HIGH_MID = 42


# --- per-episode helpers --------------------------------------------------------


def _check_aligned(name: str, array: np.ndarray, reference: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.shape != reference.shape:
        raise ValueError(f"{name} shape {array.shape} != reference {reference.shape}")
    return array


def _episode_group_sums(errors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-episode summed error over that episode's selected agent-times."""
    errors = np.asarray(errors, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    sums = np.zeros(errors.shape[0], dtype=np.float64)
    np.add.at(sums, np.nonzero(mask)[0], errors[mask])
    return sums


def _episode_group_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-episode mean of a masked agent-time array (NaN where empty)."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    sums = np.zeros(values.shape[0], dtype=np.float64)
    counts = np.zeros(values.shape[0], dtype=np.float64)
    np.add.at(sums, np.nonzero(mask)[0], values[mask])
    np.add.at(counts, np.nonzero(mask)[0], 1.0)
    out = np.full(values.shape[0], np.nan)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero]
    return out


# --- grouped V_evt --------------------------------------------------------------


def grouped_v_evt(l_suffix: np.ndarray, l_shuffled: np.ndarray,
                  selection: np.ndarray) -> float:
    """V_evt = 1 - sum(L_suffix)/sum(L_shuffled) over a boolean selection.

    Rejects shape mismatch, empty selection, non-finite errors and a
    non-positive shuffled denominator (no epsilon is ever added).
    """
    l_suffix = np.asarray(l_suffix, dtype=np.float64)
    l_shuffled = np.asarray(l_shuffled, dtype=np.float64)
    selection = np.asarray(selection, dtype=bool)
    if l_suffix.shape != l_shuffled.shape or l_suffix.shape != selection.shape:
        raise ValueError("l_suffix, l_shuffled and selection must share a shape")
    if not selection.any():
        raise ValueError("selection must be non-empty")
    suf = l_suffix[selection]
    shf = l_shuffled[selection]
    if not (np.isfinite(suf).all() and np.isfinite(shf).all()):
        raise ValueError("selected errors must be finite")
    denominator = float(shf.sum())
    if denominator <= 0.0:
        raise ValueError("shuffled denominator must be positive")
    return float(1.0 - suf.sum() / denominator)


def make_group_v_evt_statistic(l_suffix: np.ndarray, l_shuffled: np.ndarray,
                               selection: np.ndarray) -> Callable[[np.ndarray], float]:
    """Multiplicity-aware episode statistic for V_evt over ``selection``."""
    l_suffix = _check_aligned("l_suffix", np.asarray(l_suffix, dtype=np.float64), np.asarray(selection))
    l_shuffled = _check_aligned("l_shuffled", np.asarray(l_shuffled, dtype=np.float64), np.asarray(selection))
    selection = np.asarray(selection, dtype=bool)
    if not selection.any():
        raise ValueError("selection must be non-empty")
    suf = _episode_group_sums(l_suffix, selection)
    shf = _episode_group_sums(l_shuffled, selection)

    def statistic(indices: np.ndarray) -> float:
        denominator = shf[indices].sum()
        if denominator <= 0.0:
            return float("nan")
        return float(1.0 - suf[indices].sum() / denominator)

    statistic.pool_size = int(l_suffix.shape[0])
    return statistic


def make_group_contrast_statistic(l_suffix: np.ndarray, l_shuffled: np.ndarray,
                                  left_selection: np.ndarray,
                                  right_selection: np.ndarray) -> Callable[[np.ndarray], float]:
    """Episode statistic ``V_evt(left) - V_evt(right)`` (positive = left better)."""
    reference = np.asarray(left_selection)
    l_suffix = _check_aligned("l_suffix", np.asarray(l_suffix, dtype=np.float64), reference)
    l_shuffled = _check_aligned("l_shuffled", np.asarray(l_shuffled, dtype=np.float64), reference)
    left_selection = np.asarray(left_selection, dtype=bool)
    right_selection = _check_aligned("right_selection", np.asarray(right_selection, dtype=bool), reference)
    if not left_selection.any() or not right_selection.any():
        raise ValueError("both selections must be non-empty")
    suf_l = _episode_group_sums(l_suffix, left_selection)
    shf_l = _episode_group_sums(l_shuffled, left_selection)
    suf_r = _episode_group_sums(l_suffix, right_selection)
    shf_r = _episode_group_sums(l_shuffled, right_selection)

    def statistic(indices: np.ndarray) -> float:
        sl = suf_l[indices].sum()
        hl = shf_l[indices].sum()
        sr = suf_r[indices].sum()
        hr = shf_r[indices].sum()
        if hl <= 0.0 or hr <= 0.0:
            return float("nan")
        return float((1.0 - sl / hl) - (1.0 - sr / hr))

    statistic.pool_size = int(l_suffix.shape[0])
    return statistic


def make_rebound_statistic(l_suffix: np.ndarray, l_shuffled: np.ndarray,
                           bin_id: np.ndarray,
                           valid_mask: np.ndarray) -> Callable[[np.ndarray], float]:
    """Episode statistic ``C_rebound = min(V_high - V_low, V_high - V_mid)``."""
    reference = np.asarray(valid_mask)
    l_suffix = _check_aligned("l_suffix", np.asarray(l_suffix, dtype=np.float64), reference)
    l_shuffled = _check_aligned("l_shuffled", np.asarray(l_shuffled, dtype=np.float64), reference)
    bin_id = _check_aligned("bin_id", np.asarray(bin_id), reference)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    sums = {}
    for b in (1, 2, 3):
        mask = valid_mask & (bin_id == b)
        sums[b] = (_episode_group_sums(l_suffix, mask),
                   _episode_group_sums(l_shuffled, mask))

    def statistic(indices: np.ndarray) -> float:
        values = {}
        for b in (1, 2, 3):
            suf_b, shf_b = sums[b]
            denom = shf_b[indices].sum()
            if denom <= 0.0:
                return float("nan")
            values[b] = 1.0 - suf_b[indices].sum() / denom
        return float(min(values[3] - values[1], values[3] - values[2]))

    statistic.pool_size = int(l_suffix.shape[0])
    return statistic


# --- multiplicity-aware weighted statistics (for power at n > pool) -------------


def _weighted_midranks(sorted_values: np.ndarray, sorted_weights: np.ndarray) -> np.ndarray:
    """Weighted midranks in sorted order; equals average ranks when weights==1.

    Each tie group receives a single rank ``(W_before + W_end) / 2`` which, with
    unit weights, differs from the ordinary average rank by a constant 0.5 for
    every group; that constant cancels in the Pearson correlation, so the result
    equals the unweighted Spearman exactly.
    """
    n = len(sorted_values)
    cw = np.cumsum(sorted_weights)
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_start = np.flatnonzero(is_start)
    group_sizes = np.diff(np.append(group_start, n))
    start_of = np.repeat(group_start, group_sizes)
    ends = np.append(group_start[1:] - 1, n - 1)
    end_of = np.repeat(ends, group_sizes)
    cw_before = np.zeros(n, dtype=np.float64)
    has_before = start_of > 0
    cw_before[has_before] = cw[start_of[has_before] - 1]
    cw_end = cw[end_of]
    return (cw_before + cw_end) / 2.0


def _weighted_pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    wsum = float(w.sum())
    if wsum <= 0.0:
        return float("nan")
    ca = a - float((w * a).sum()) / wsum
    cb = b - float((w * b).sum()) / wsum
    var_a = float((w * ca * ca).sum())
    var_b = float((w * cb * cb).sum())
    denominator = np.sqrt(var_a * var_b)
    if denominator == 0.0:
        return float("nan")
    return float((w * ca * cb).sum() / denominator)


def make_weighted_spearman_statistic(values_x: np.ndarray, values_y: np.ndarray,
                                     valid_mask: np.ndarray) -> Callable[[np.ndarray], float]:
    """Multiplicity-aware episode statistic for Spearman(x, y).

    Resampled episodes contribute their agent-times weighted by draw
    multiplicity. With unit weights it equals
    :func:`ac_twin.stage1_statistics.spearman` on the valid rows.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n_episodes = valid_mask.shape[0]
    flat_mask = valid_mask.reshape(-1)
    episode_index = np.broadcast_to(
        np.arange(n_episodes)[:, None, None], valid_mask.shape).reshape(-1)
    x = np.asarray(values_x, dtype=np.float64).reshape(-1)[flat_mask]
    y = np.asarray(values_y, dtype=np.float64).reshape(-1)[flat_mask]
    row_episode = episode_index[flat_mask]
    order_x = np.argsort(x, kind="mergesort")
    order_y = np.argsort(y, kind="mergesort")

    def statistic(indices: np.ndarray) -> float:
        weights_ep = np.bincount(np.asarray(indices), minlength=n_episodes).astype(np.float64)
        w_row = weights_ep[row_episode]
        if (w_row > 0).sum() < 2:
            return float("nan")
        rx = np.empty_like(x)
        rx[order_x] = _weighted_midranks(x[order_x], w_row[order_x])
        ry = np.empty_like(y)
        ry[order_y] = _weighted_midranks(y[order_y], w_row[order_y])
        return _weighted_pearson(rx, ry, w_row)

    statistic.pool_size = int(n_episodes)
    return statistic


def make_weighted_trend_statistic(X: np.ndarray, y: np.ndarray, column: int,
                                  row_episode: np.ndarray,
                                  n_episodes: int) -> Callable[[np.ndarray], float]:
    """Multiplicity-aware ridge-trend coefficient via per-episode sufficient stats.

    With unit weights this solves the same penalized normal equations as
    :func:`ac_twin.stage1_statistics.ridge_fit` on all rows.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    row_episode = np.asarray(row_episode, dtype=np.int64)
    n_episodes = int(n_episodes)
    n_columns = X.shape[1]
    sx = np.zeros((n_episodes, n_columns, n_columns), dtype=np.float64)
    sy = np.zeros((n_episodes, n_columns), dtype=np.float64)
    for e in range(n_episodes):
        rows = row_episode == e
        if rows.any():
            xe = X[rows]
            sx[e] = xe.T @ xe
            sy[e] = xe.T @ y[rows]

    def statistic(indices: np.ndarray) -> float:
        weights = np.bincount(np.asarray(indices), minlength=n_episodes).astype(np.float64)
        if weights.sum() <= 0.0:
            return float("nan")
        xtx = np.einsum("e,eij->ij", weights, sx)
        xty = np.einsum("e,ei->i", weights, sy)
        penalty = RIDGE_LAMBDA * np.eye(n_columns)
        penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            return float("nan")
        return float(beta[column])

    statistic.pool_size = int(n_episodes)
    return statistic


# --- shape diagnostics (D1/D2/D3) -----------------------------------------------


def _mean_excluding_seed(errors_by_seed: np.ndarray, drop_seed_index: int) -> np.ndarray:
    """Seed-mean over all seeds except one (seeds are never independent samples)."""
    keep = [s for s in range(errors_by_seed.shape[0]) if s != drop_seed_index]
    return errors_by_seed[keep].mean(axis=0)


def _eta(l_suffix: np.ndarray, l_shuffled: np.ndarray) -> np.ndarray:
    return np.log(l_suffix + ETA_EPSILON_M) - np.log(l_shuffled + ETA_EPSILON_M)


def _rebound_episode_sums(l_suffix: np.ndarray, l_shuffled: np.ndarray,
                          bin_id: np.ndarray, valid_mask: np.ndarray):
    """Per-episode suffix/shuffled sums for the three positive bins."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    suf, shf = {}, {}
    for b in (1, 2, 3):
        mask = valid_mask & (bin_id == b)
        suf[b] = _episode_group_sums(l_suffix, mask)
        shf[b] = _episode_group_sums(l_shuffled, mask)
    return suf, shf


def _rebound_point(suf: Mapping[int, np.ndarray], shf: Mapping[int, np.ndarray],
                   indices: np.ndarray) -> float:
    values = {}
    for b in (1, 2, 3):
        denominator = shf[b][indices].sum()
        if denominator <= 0.0:
            return float("nan")
        values[b] = 1.0 - suf[b][indices].sum() / denominator
    return float(min(values[3] - values[1], values[3] - values[2]))


def compute_shape_diagnostics(*, l_no_by_seed: np.ndarray, l_suffix_by_seed: np.ndarray,
                              l_shuffled_by_seed: np.ndarray, r_pos: np.ndarray,
                              bin_id: np.ndarray, valid_mask: np.ndarray,
                              cf_action: np.ndarray, covariates: Mapping[str, np.ndarray],
                              train_standardizer: Mapping[str, Mapping[str, float]],
                              bootstrap_resamples: int, confidence_level: float,
                              bootstrap_seed: int) -> dict[str, object]:
    """Compute D1/D2/D3 shape diagnostics on the merge development-test split.

    All inputs are non-ego, world-coordinate, aligned ``E×T×N`` arrays on the
    test split; ``*_by_seed`` add a leading seed axis. Seeds are averaged per
    agent-time first; every CI is an episode-cluster bootstrap.
    """
    l_suffix_by_seed = np.asarray(l_suffix_by_seed, dtype=np.float64)
    l_shuffled_by_seed = np.asarray(l_shuffled_by_seed, dtype=np.float64)
    l_no_by_seed = np.asarray(l_no_by_seed, dtype=np.float64)
    if l_suffix_by_seed.shape != l_shuffled_by_seed.shape or \
            l_suffix_by_seed.shape != l_no_by_seed.shape:
        raise ValueError("per-seed error arrays must share a shape")
    if l_suffix_by_seed.ndim != 4:
        raise ValueError("per-seed error arrays must be S×E×T×N")
    n_seeds = l_suffix_by_seed.shape[0]
    reference = np.asarray(valid_mask)
    if l_suffix_by_seed.shape[1:] != reference.shape:
        raise ValueError("valid_mask must match the E×T×N error shape")
    r_pos = _check_aligned("r_pos", np.asarray(r_pos, dtype=np.float64), reference)
    bin_id = _check_aligned("bin_id", np.asarray(bin_id), reference)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    cf_action = np.asarray(cf_action)
    n_episodes = reference.shape[0]
    if cf_action.shape[0] != n_episodes:
        raise ValueError("cf_action must be per-episode with length E")
    arange = np.arange(n_episodes)

    # Seed-mean errors and the stabilized log ratio (frozen epsilon).
    l_suffix = seed_mean(l_suffix_by_seed)
    l_shuffled = seed_mean(l_shuffled_by_seed)
    l_no = seed_mean(l_no_by_seed)
    eta = _eta(l_suffix, l_shuffled)

    base = int(bootstrap_seed)
    R = int(bootstrap_resamples)
    cl = float(confidence_level)

    # --- reproduction of the frozen Stage 1 point estimates (no re-judgement) ---
    bins = aggregate_bins(l_no, l_suffix, l_shuffled, bin_id, valid_mask)
    reproduction = {
        "v_evt_low": make_v_evt_statistic(l_suffix, l_shuffled, bin_id, valid_mask, (0, 1))(arange),
        "v_evt_low_minus_high": make_v_evt_low_high_statistic(
            l_suffix, l_shuffled, bin_id, valid_mask, (0, 1), (3,))(arange),
        "spearman_r_eta": make_spearman_statistic(r_pos, eta, valid_mask)(arange),
    }
    X_all, y_all, feature_names, log_col = build_trend_design(
        eta, r_pos, covariates["horizon_index"], covariates["cf_action"],
        covariates["cf_duration"], covariates["initial_distance"],
        covariates["history_visible_ratio"], covariates["visible_count"],
        valid_mask, train_standardizer)
    row_episode_all = np.nonzero(valid_mask)[0]
    reproduction["trend_log_r"] = make_trend_coefficient_statistic(
        X_all, y_all, log_col, row_episode_all, n_episodes)(arange)

    # Per-bin V_evt CIs (bins 1,2,3 double as the positive bins for D2/D3).
    bin_v_evt_ci = {}
    for offset, b in enumerate((0, 1, 2, 3)):
        stat = make_v_evt_statistic(l_suffix, l_shuffled, bin_id, valid_mask, (b,))
        bin_v_evt_ci[b] = episode_bootstrap(n_episodes, stat, R, cl, base + SEED_BIN_V_EVT + offset)

    # --- D1: zero dominance and response presence -------------------------------
    bin_masks = {b: valid_mask & (bin_id == b) for b in (0, 1, 2, 3)}
    positive_mask = valid_mask & (bin_id >= 1)
    original_low_mask = valid_mask & np.isin(bin_id, [0, 1])
    n_valid = int(valid_mask.sum())
    zero_dominance = {
        "zero_agent_time_share": float(bin_masks[0].sum() / n_valid),
        "zero_share_of_original_low_shuffled_error_sum": float(
            l_shuffled[bin_masks[0]].sum() / l_shuffled[original_low_mask].sum()),
        "zero_share_of_original_low_suffix_error_sum": float(
            l_suffix[bin_masks[0]].sum() / l_suffix[original_low_mask].sum()),
    }
    v_evt_zero = bin_v_evt_ci[0]
    v_evt_positive_all = episode_bootstrap(
        n_episodes, make_v_evt_statistic(l_suffix, l_shuffled, bin_id, valid_mask, (1, 2, 3)),
        R, cl, base + SEED_C_PRESENCE - 1)
    c_presence_stat = make_group_contrast_statistic(
        l_suffix, l_shuffled, bin_masks[0], positive_mask)
    c_presence = episode_bootstrap(n_episodes, c_presence_stat, R, cl, base + SEED_C_PRESENCE)
    loso_c_presence = []
    for s in range(n_seeds):
        ls = _mean_excluding_seed(l_suffix_by_seed, s)
        lh = _mean_excluding_seed(l_shuffled_by_seed, s)
        stat = make_group_contrast_statistic(ls, lh, bin_masks[0], positive_mask)
        loso_c_presence.append(float(stat(arange)))
    presence_supported = bool(
        np.isfinite(v_evt_zero["ci_lower"]) and v_evt_zero["ci_lower"] > 0
        and np.isfinite(c_presence["ci_lower"]) and c_presence["ci_lower"] > 0
        and all(np.isfinite(loso_c_presence)) and all(v > 0 for v in loso_c_presence))
    zero_dominance.update({
        "v_evt_zero": v_evt_zero,
        "v_evt_positive_all": v_evt_positive_all,
        "c_presence": c_presence,
        "leave_one_seed_out_c_presence": loso_c_presence,
    })

    # --- D2: positive-only magnitude trend ---------------------------------------
    def _spearman_point(eta_arr, selection):
        return make_spearman_statistic(r_pos, eta_arr, selection)(arange)

    def _trend_point(eta_arr, selection):
        X, y, _, col = build_trend_design(
            eta_arr, r_pos, covariates["horizon_index"], covariates["cf_action"],
            covariates["cf_duration"], covariates["initial_distance"],
            covariates["history_visible_ratio"], covariates["visible_count"],
            selection, train_standardizer)
        rows = np.nonzero(selection)[0]
        return make_trend_coefficient_statistic(X, y, col, rows, n_episodes)(arange)

    pos_spearman = episode_bootstrap(
        n_episodes, make_spearman_statistic(r_pos, eta, positive_mask),
        R, cl, base + SEED_POS_SPEARMAN)
    X_pos, y_pos, _, pos_log_col = build_trend_design(
        eta, r_pos, covariates["horizon_index"], covariates["cf_action"],
        covariates["cf_duration"], covariates["initial_distance"],
        covariates["history_visible_ratio"], covariates["visible_count"],
        positive_mask, train_standardizer)
    row_episode_pos = np.nonzero(positive_mask)[0]
    pos_trend = episode_bootstrap(
        n_episodes,
        make_trend_coefficient_statistic(X_pos, y_pos, pos_log_col, row_episode_pos, n_episodes),
        R, cl, base + SEED_POS_TREND)
    pos_low_high_stat = make_group_contrast_statistic(
        l_suffix, l_shuffled, bin_masks[1], bin_masks[3])
    pos_low_minus_high = episode_bootstrap(
        n_episodes, pos_low_high_stat, R, cl, base + SEED_POS_LOW_MINUS_HIGH)

    loso_pos_spearman, loso_pos_trend, loso_pos_low_high = [], [], []
    for s in range(n_seeds):
        ls = _mean_excluding_seed(l_suffix_by_seed, s)
        lh = _mean_excluding_seed(l_shuffled_by_seed, s)
        eta_s = _eta(ls, lh)
        loso_pos_spearman.append(float(_spearman_point(eta_s, positive_mask)))
        loso_pos_trend.append(float(_trend_point(eta_s, positive_mask)))
        loso_pos_low_high.append(float(
            make_group_contrast_statistic(ls, lh, bin_masks[1], bin_masks[3])(arange)))
    magnitude_supported = bool(
        np.isfinite(pos_spearman["ci_lower"]) and pos_spearman["ci_lower"] > 0
        and np.isfinite(pos_trend["ci_lower"]) and pos_trend["ci_lower"] > 0
        and np.isfinite(pos_low_minus_high["ci_lower"]) and pos_low_minus_high["ci_lower"] > 0
        and all(v > 0 for v in loso_pos_spearman)
        and all(v > 0 for v in loso_pos_trend)
        and all(v > 0 for v in loso_pos_low_high))
    positive_only = {
        "spearman": pos_spearman,
        "trend_log_r": {**pos_trend, "feature_names": feature_names},
        "v_evt_low_minus_high": pos_low_minus_high,
        "positive_bins": {name: bins[name] for name in ("positive_low", "positive_mid", "positive_high")},
        "positive_bin_v_evt_ci": {str(b): bin_v_evt_ci[b] for b in (1, 2, 3)},
        "leave_one_seed_out": {
            "spearman": loso_pos_spearman,
            "trend_log_r": loso_pos_trend,
            "v_evt_low_minus_high": loso_pos_low_high,
        },
    }

    # --- D3: high-response rebound ------------------------------------------------
    suf_pos, shf_pos = _rebound_episode_sums(l_suffix, l_shuffled, bin_id, valid_mask)
    rebound_full = _rebound_point(suf_pos, shf_pos, arange)
    c_high_low_stat = make_group_contrast_statistic(l_suffix, l_shuffled, bin_masks[3], bin_masks[1])
    c_high_mid_stat = make_group_contrast_statistic(l_suffix, l_shuffled, bin_masks[3], bin_masks[2])
    c_rebound_stat = make_rebound_statistic(l_suffix, l_shuffled, bin_id, valid_mask)
    c_high_low = episode_bootstrap(n_episodes, c_high_low_stat, R, cl, base + SEED_C_HIGH_LOW)
    c_high_mid = episode_bootstrap(n_episodes, c_high_mid_stat, R, cl, base + SEED_C_HIGH_MID)
    c_rebound = episode_bootstrap(n_episodes, c_rebound_stat, R, cl, base + SEED_C_REBOUND)

    loso_rebound = []
    for s in range(n_seeds):
        ls = _mean_excluding_seed(l_suffix_by_seed, s)
        lh = _mean_excluding_seed(l_shuffled_by_seed, s)
        suf_s, shf_s = _rebound_episode_sums(ls, lh, bin_id, valid_mask)
        loso_rebound.append(_rebound_point(suf_s, shf_s, arange))

    action_detail = {}
    for action in sorted(np.unique(cf_action).tolist()):
        keep = np.flatnonzero(cf_action != action)
        action_detail[int(action)] = {
            "c_rebound": _rebound_point(suf_pos, shf_pos, keep),
            "n_episodes": int(len(keep)),
            "n_episodes_removed": int(n_episodes - len(keep)),
        }

    has_valid = valid_mask.any(axis=(1, 2))
    eligible = np.nonzero(has_valid)[0]
    loeo = np.array([
        _rebound_point(suf_pos, shf_pos, eligible[eligible != e]) for e in eligible
    ])
    score = _episode_group_means(l_suffix + l_shuffled, valid_mask)
    deletion = {}
    for frac in DROP_HIGH_ERROR_FRACTIONS:
        k = max(1, int(np.floor(frac * len(eligible))))
        order = eligible[np.argsort(-score[eligible])]
        dropped = order[:k]
        kept = np.setdiff1d(eligible, dropped)
        deletion[f"drop_top_{int(round(frac * 100))}pct"] = {
            "c_rebound": _rebound_point(suf_pos, shf_pos, kept),
            "n_episodes": int(len(kept)),
            "n_episodes_removed": int(len(dropped)),
        }
    bin_episode_counts = {
        "positive_low": int(bins["positive_low"]["n_episodes"]),
        "positive_mid": int(bins["positive_mid"]["n_episodes"]),
        "positive_high": int(bins["positive_high"]["n_episodes"]),
    }
    rebound_supported = bool(
        np.isfinite(c_rebound["ci_lower"]) and c_rebound["ci_lower"] > 0
        and all(np.isfinite(loso_rebound)) and all(v > 0 for v in loso_rebound)
        and all(np.isfinite(v["c_rebound"]) and v["c_rebound"] > 0 for v in action_detail.values())
        and np.isfinite(loeo).all() and float(loeo.min()) > 0 and int((loeo <= 0).sum()) == 0
        and all(np.isfinite(d["c_rebound"]) and d["c_rebound"] > 0 for d in deletion.values())
        and all(count >= MINIMUM_BIN_EPISODES for count in bin_episode_counts.values()))
    rebound = {
        "c_high_low": c_high_low,
        "c_high_mid": c_high_mid,
        "c_rebound": c_rebound,
        "c_rebound_point": float(rebound_full),
        "leave_one_seed_out_c_rebound": loso_rebound,
        "leave_one_action_type_out": action_detail,
        "leave_one_episode_out_min": float(loeo.min()) if len(loeo) else float("nan"),
        "leave_one_episode_out_n_reversed": int((loeo <= 0).sum()) if len(loeo) else 0,
        "high_error_deletion": deletion,
        "bin_episode_counts": bin_episode_counts,
    }

    shape_flags = {
        "PRESENCE_DIFFERENCE_SUPPORTED": presence_supported,
        "POSITIVE_MAGNITUDE_DECAY_SUPPORTED": magnitude_supported,
        "ROBUST_HIGH_REBOUND_SUPPORTED": rebound_supported,
    }
    return {
        "n_episodes": int(n_episodes),
        "n_seeds": int(n_seeds),
        "bins": bins,
        "bin_v_evt_ci": {str(b): bin_v_evt_ci[b] for b in (0, 1, 2, 3)},
        "reproduction": reproduction,
        "zero_dominance": zero_dominance,
        "positive_only": positive_only,
        "rebound": rebound,
        "shape_flags": shape_flags,
        "shape_recommendation": classify_shape_recommendation({"shape_flags": shape_flags}),
    }


def classify_shape_recommendation(diagnostics: Mapping[str, object]) -> str:
    """Mechanically map the three diagnostic flags to one shape recommendation."""
    flags = diagnostics["shape_flags"]
    if flags["ROBUST_HIGH_REBOUND_SUPPORTED"]:
        return REVISE_NONMONOTONIC
    if flags["POSITIVE_MAGNITUDE_DECAY_SUPPORTED"]:
        return KEEP_ORIGINAL
    if flags["PRESENCE_DIFFERENCE_SUPPORTED"]:
        return SHRINK_TO_PRESENCE
    return NO_STABLE_SHAPE


# --- confirmatory power analysis --------------------------------------------------


def _smallest_n_reaching(curve: Mapping[int, float], threshold: float,
                         candidates: list[int]):
    for n in candidates:
        if curve[n] >= threshold:
            return n
    return "NOT_REACHED_WITHIN_2400"


def estimate_power_curve(*, episode_statistics: Mapping[str, Callable[[np.ndarray], float]],
                         candidate_episode_counts: list[int], monte_carlo_replicates: int,
                         confidence_level: float, seed: int) -> dict[str, object]:
    """Monte-Carlo resampling power for the frozen statistics.

    For each candidate size n, episodes are drawn with replacement from the
    development-test pool ``monte_carlo_replicates`` times and every statistic is
    evaluated on the same draw (so the joint pass is a per-draw AND, never a
    product of single powers). ``SE_n`` is the empirical standard deviation of a
    statistic's simulated point estimates. A draw passes when
    ``estimate * multiplier - Z_95 * SE_n > 0``; the 75%/50% scenarios rescale the
    point estimates toward zero while keeping ``SE_n`` fixed.
    """
    if float(confidence_level) != 0.95:
        raise ValueError("estimate_power_curve is frozen to confidence_level = 0.95")
    names = list(episode_statistics.keys())
    missing = [k for k in GATE_A_NUMERIC_STATISTICS if k not in episode_statistics]
    if missing:
        raise ValueError(f"episode_statistics missing Gate A numeric conditions: {missing}")
    pool_sizes = {int(getattr(fn, "pool_size", 0)) for fn in episode_statistics.values()}
    if len(pool_sizes) != 1:
        raise ValueError("all statistics must share a single pool_size")
    pool = pool_sizes.pop()
    if pool < 2:
        raise ValueError("pool_size must be >= 2")

    rng = np.random.default_rng(int(seed))
    R = int(monte_carlo_replicates)
    candidates = [int(n) for n in candidate_episode_counts]
    multipliers = (1.0, 0.75, 0.5)
    scenario_names = {1.0: "empirical", 0.75: "75pct", 0.5: "50pct"}

    full_index = np.arange(pool)
    full_point = {name: float(fn(full_index)) for name, fn in episode_statistics.items()}

    per_n = {}
    for n in candidates:
        import time as _time
        _start = _time.time()
        draws = {name: np.empty(R, dtype=np.float64) for name in names}
        for r in range(R):
            indices = rng.integers(0, pool, size=n)
            for name in names:
                draws[name][r] = episode_statistics[name](indices)
        se = {name: float(np.std(draws[name], ddof=1)) for name in names}
        print(f"[stage1-plus] power n={n} replicate={R}/{R} "
              f"elapsed_seconds={_time.time() - _start:.1f}", flush=True)
        single = {name: {} for name in names}
        for name in names:
            for mult in multipliers:
                passed = (draws[name] * mult - Z_95 * se[name]) > 0
                single[name][scenario_names[mult]] = float(np.mean(passed))
        joint = {}
        for mult in multipliers:
            joint_pass = np.ones(R, dtype=bool)
            for name in GATE_A_NUMERIC_STATISTICS:
                joint_pass &= (draws[name] * mult - Z_95 * se[name]) > 0
            joint[scenario_names[mult]] = float(np.mean(joint_pass))
        per_n[n] = {"single": single, "joint_numeric": joint, "se": se}

    weakest = {}
    for n in candidates:
        powers = {name: per_n[n]["single"][name]["empirical"] for name in GATE_A_NUMERIC_STATISTICS}
        name = min(powers, key=powers.get)
        weakest[n] = {"condition": name, "power": powers[name]}

    summary = {}
    for mult in multipliers:
        scen = scenario_names[mult]
        single_each = {
            name: {n: per_n[n]["single"][name][scen] for n in candidates}
            for name in GATE_A_NUMERIC_STATISTICS
        }
        each_ge_90 = None
        for n in candidates:
            if all(single_each[name][n] >= 0.90 for name in GATE_A_NUMERIC_STATISTICS):
                each_ge_90 = n
                break
        joint_curve = {n: per_n[n]["joint_numeric"][scen] for n in candidates}
        summary[scen] = {
            "smallest_n_single_each_ge_0.90": (
                each_ge_90 if each_ge_90 is not None else "NOT_REACHED_WITHIN_2400"),
            "smallest_n_joint_numeric_ge_0.80": _smallest_n_reaching(joint_curve, 0.80, candidates),
            "smallest_n_joint_numeric_ge_0.90": _smallest_n_reaching(joint_curve, 0.90, candidates),
        }

    return {
        "candidate_episode_counts": candidates,
        "monte_carlo_replicates": R,
        "confidence_level": float(confidence_level),
        "z": Z_95,
        "pool_episodes": pool,
        "full_point_estimates": full_point,
        "gate_a_numeric_statistics": list(GATE_A_NUMERIC_STATISTICS),
        "per_candidate": per_n,
        "weakest_gate_a_condition": weakest,
        "sample_size_summary": summary,
    }
