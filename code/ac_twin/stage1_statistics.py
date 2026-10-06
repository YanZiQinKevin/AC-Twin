"""Stage 1 per-agent metrics, response binning and episode-cluster statistics.

All statistics operate on world-coordinate, de-normalized arrays that keep the
``episode × time × agent`` alignment and a valid mask. The independent unit is
the episode: training seeds are averaged per agent-time first, and every
confidence interval is an episode-cluster bootstrap. Seeds are never counted as
independent samples.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

# Gate status vocabulary (frozen; Gate A is never "CONFIRMED" at this stage).
GATE_S_PASS = "PASS"
GATE_S_FAIL = "FAIL"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE_MODEL_INSUFFICIENT"
OUTCOME_GO = "DEVELOPMENT_GO_TO_NEW_HOLDOUT"
OUTCOME_NO_GO = "DEVELOPMENT_NO_GO"
GATE_BLOCKED = "BLOCKED_AWAITING_GATE_S"
GATE_B_SUPPORTED = "SUPPORTED"
GATE_B_NOT_SUPPORTED = "NOT_SUPPORTED"
GATE_B_UNDERPOWERED = "UNDERPOWERED"

BIN_NAMES = ("zero", "positive_low", "positive_mid", "positive_high")


# --- per-agent base quantities ----------------------------------------------

def position_errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per agent-time L2 error from ``...×2`` arrays; returns ``...`` (E,T,N)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.shape[-1] != 2:
        raise ValueError("pred and target must share a ...×2 shape")
    return np.linalg.norm(pred - target, axis=-1)


def response_magnitudes(counterfactual: np.ndarray, factual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_pos, R_vel) with E×T×N alignment in world coordinates."""
    counterfactual = np.asarray(counterfactual, dtype=np.float64)
    factual = np.asarray(factual, dtype=np.float64)
    if counterfactual.shape != factual.shape or counterfactual.shape[-1] < 6:
        raise ValueError("counterfactual and factual must share a ...×>=6 feature shape")
    r_pos = np.linalg.norm(counterfactual[..., 2:4] - factual[..., 2:4], axis=-1)
    r_vel = np.linalg.norm(counterfactual[..., 4:6] - factual[..., 4:6], axis=-1)
    return r_pos, r_vel


def value_metrics(l_suffix: np.ndarray, l_shuffled: np.ndarray, l_no: np.ndarray,
                  epsilon: float) -> dict[str, np.ndarray]:
    """Absolute event-specific value, net value and stabilized log ratio."""
    l_suffix = np.asarray(l_suffix, dtype=np.float64)
    l_shuffled = np.asarray(l_shuffled, dtype=np.float64)
    l_no = np.asarray(l_no, dtype=np.float64)
    if not (l_suffix.shape == l_shuffled.shape == l_no.shape):
        raise ValueError("Error arrays must share a shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return {
        "D_evt": l_shuffled - l_suffix,
        "Delta": l_no - l_suffix,
        "eta_evt": np.log(l_suffix + epsilon) - np.log(l_shuffled + epsilon),
    }


def seed_mean(errors_by_seed: np.ndarray) -> np.ndarray:
    """Average per-agent-time errors over the seed axis (axis 0)."""
    errors_by_seed = np.asarray(errors_by_seed, dtype=np.float64)
    if errors_by_seed.ndim < 2:
        raise ValueError("errors_by_seed must have a seed axis")
    return errors_by_seed.mean(axis=0)


# --- response binning ---------------------------------------------------------

def fit_response_bins(r_pos_train: np.ndarray, zero_tolerance: float,
                      positive_quantiles: tuple[float, float]) -> dict[str, Any]:
    """Fit zero/low/mid/high thresholds on split-0 valid non-ego R_pos only."""
    r_pos_train = np.asarray(r_pos_train, dtype=np.float64)
    r_pos_train = r_pos_train[np.isfinite(r_pos_train)]
    if len(r_pos_train) == 0:
        raise ValueError("Need non-empty training responses to fit bins")
    if zero_tolerance <= 0:
        raise ValueError("zero_tolerance must be positive")
    q_lo, q_hi = positive_quantiles
    if not (0.0 < q_lo < q_hi < 1.0):
        raise ValueError("positive_quantiles must satisfy 0 < low < high < 1")
    positive = r_pos_train[r_pos_train > zero_tolerance]
    if len(positive) == 0:
        raise ValueError("Need positive training responses to fit quantile bins")
    t1, t2 = np.quantile(positive, [q_lo, q_hi])
    return {
        "zero_tolerance": float(zero_tolerance),
        "positive_quantiles": [float(q_lo), float(q_hi)],
        "t1": float(t1),
        "t2": float(t2),
        "n_train_positive": int(len(positive)),
    }


def assign_response_bin(r_pos: np.ndarray, thresholds: Mapping[str, Any]) -> np.ndarray:
    """Assign 0=zero,1=positive_low,2=positive_mid,3=positive_high per element."""
    r_pos = np.asarray(r_pos, dtype=np.float64)
    zero_tol = float(thresholds["zero_tolerance"])
    t1 = float(thresholds["t1"])
    t2 = float(thresholds["t2"])
    bins = np.zeros(r_pos.shape, dtype=np.int64)
    bins[r_pos > zero_tol] = 1
    bins[r_pos > t1] = 2
    bins[r_pos > t2] = 3
    return bins


# --- bin aggregation ----------------------------------------------------------

def aggregate_bins(l_no: np.ndarray, l_suffix: np.ndarray, l_shuffled: np.ndarray,
                   bin_id: np.ndarray, valid_mask: np.ndarray) -> dict[str, dict]:
    """Per-bin V_evt / rho_no from seed-mean errors, with raw means and counts.

    A bin with a zero denominator is marked unavailable (no epsilon is added).
    """
    for name, array in (("l_no", l_no), ("l_suffix", l_suffix),
                        ("l_shuffled", l_shuffled), ("bin_id", bin_id)):
        if np.asarray(array).shape != np.asarray(valid_mask).shape:
            raise ValueError(f"{name} and valid_mask must share a shape")
    valid_mask = np.asarray(valid_mask, dtype=bool)
    out: dict[str, dict] = {}
    for bin_index, bin_name in enumerate(BIN_NAMES):
        sel = valid_mask & (bin_id == bin_index)
        n_agent_time = int(sel.sum())
        sum_no = float(np.asarray(l_no)[sel].sum())
        sum_suffix = float(np.asarray(l_suffix)[sel].sum())
        sum_shuffled = float(np.asarray(l_shuffled)[sel].sum())
        n_episodes = int(len(np.unique(np.nonzero(sel)[0]))) if n_agent_time else 0
        entry = {
            "n_agent_time": n_agent_time,
            "n_episodes": n_episodes,
            "sum_l_no": sum_no,
            "sum_l_suffix": sum_suffix,
            "sum_l_shuffled": sum_shuffled,
            "mean_l_no": sum_no / n_agent_time if n_agent_time else float("nan"),
            "mean_l_suffix": sum_suffix / n_agent_time if n_agent_time else float("nan"),
            "mean_l_shuffled": sum_shuffled / n_agent_time if n_agent_time else float("nan"),
        }
        entry["V_evt"] = (1.0 - sum_suffix / sum_shuffled) if sum_shuffled > 0 else None
        entry["rho_no"] = (1.0 - sum_suffix / sum_no) if sum_no > 0 else None
        out[bin_name] = entry
    return out


# --- episode-cluster bootstrap -------------------------------------------------

def episode_bootstrap(n_episodes: int, statistic: Callable[[np.ndarray], float],
                      n_resamples: int, confidence_level: float, seed: int) -> dict[str, Any]:
    """Cluster-bootstrap a scalar statistic over episodes.

    ``statistic`` receives an array of resampled episode indices (with
    replacement) and returns a scalar, or NaN when a draw is not computable
    (e.g. a zero denominator). The point estimate uses all episodes in order.
    """
    if n_episodes < 2:
        raise ValueError("Episode bootstrap needs at least two episodes")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_resamples), dtype=np.float64)
    for resample in range(int(n_resamples)):
        indices = rng.integers(0, n_episodes, size=n_episodes)
        draws[resample] = statistic(indices)
    point = float(statistic(np.arange(n_episodes)))
    finite = np.isfinite(draws)
    alpha = 1.0 - float(confidence_level)
    if finite.sum() < 2:
        lower = upper = float("nan")
    else:
        lower, upper = np.quantile(draws[finite], [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "point": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_resamples": int(n_resamples),
        "n_finite_draws": int(finite.sum()),
        "confidence_level": float(confidence_level),
        "bootstrap_seed": int(seed),
        "n_episodes": int(n_episodes),
    }


def _episode_group_sums(errors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-episode summed error over that episode's selected agent-times."""
    errors = np.asarray(errors, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    sums = np.zeros(errors.shape[0], dtype=np.float64)
    np.add.at(sums, np.nonzero(mask)[0], errors[mask])
    return sums


def make_v_evt_low_high_statistic(l_suffix: np.ndarray, l_shuffled: np.ndarray,
                                  bin_id: np.ndarray, valid_mask: np.ndarray,
                                  low_bins: tuple[int, ...], high_bins: tuple[int, ...]):
    """Build the episode statistic V_evt_low - V_evt_high (positive = decay)."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    low_mask = valid_mask & np.isin(bin_id, list(low_bins))
    high_mask = valid_mask & np.isin(bin_id, list(high_bins))
    suf_low = _episode_group_sums(l_suffix, low_mask)
    shuf_low = _episode_group_sums(l_shuffled, low_mask)
    suf_high = _episode_group_sums(l_suffix, high_mask)
    shuf_high = _episode_group_sums(l_shuffled, high_mask)

    def statistic(indices: np.ndarray) -> float:
        s_l = suf_low[indices].sum()
        h_l = shuf_low[indices].sum()
        s_h = suf_high[indices].sum()
        h_h = shuf_high[indices].sum()
        if h_l <= 0 or h_h <= 0:
            return float("nan")
        v_low = 1.0 - s_l / h_l
        v_high = 1.0 - s_h / h_h
        return v_low - v_high

    return statistic


def make_v_evt_statistic(l_suffix: np.ndarray, l_shuffled: np.ndarray,
                         bin_id: np.ndarray, valid_mask: np.ndarray,
                         bins: tuple[int, ...]):
    """Build the episode statistic V_evt over a set of response bins."""
    mask = np.asarray(valid_mask, dtype=bool) & np.isin(bin_id, list(bins))
    suf = _episode_group_sums(l_suffix, mask)
    shuf = _episode_group_sums(l_shuffled, mask)

    def statistic(indices: np.ndarray) -> float:
        denominator = shuf[indices].sum()
        if denominator <= 0:
            return float("nan")
        return 1.0 - suf[indices].sum() / denominator

    return statistic


# --- rank correlation ----------------------------------------------------------

def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    index = 0
    while index < n:
        end = index
        while end + 1 < n and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks_sorted[index:end + 1] = (index + end) / 2.0 + 1.0
        index = end + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("spearman needs paired 1-D arrays of length >= 2")
    rx = _rankdata(x) - _rankdata(x).mean()
    ry = _rankdata(y) - _rankdata(y).mean()
    denominator = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denominator == 0:
        return float("nan")
    return float((rx * ry).sum() / denominator)


def make_spearman_statistic(values_x: np.ndarray, values_y: np.ndarray,
                            valid_mask: np.ndarray):
    """Episode statistic: Spearman(x, y) over valid agent-times of resampled episodes."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if x.shape != valid_mask.shape or y.shape != valid_mask.shape:
        raise ValueError("values and valid_mask must share an E x T x N shape")

    def statistic(indices: np.ndarray) -> float:
        sampled_mask = valid_mask[indices]
        if sampled_mask.sum() < 2:
            return float("nan")
        return spearman(x[indices][sampled_mask], y[indices][sampled_mask])

    return statistic


# --- covariate trend model ------------------------------------------------------

def build_trend_design(eta_evt: np.ndarray, r_pos: np.ndarray, horizon_index: np.ndarray,
                       cf_action: np.ndarray, cf_duration: np.ndarray,
                       initial_distance: np.ndarray, history_visible_ratio: np.ndarray,
                       visible_count: np.ndarray, valid_mask: np.ndarray,
                       standardizer: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Assemble the trend design matrix on valid non-ego agent-times.

    Continuous columns are standardized with the frozen training statistics in
    ``standardizer`` (fitted on split 0 only). ``cf_action`` is one-hot. Returns
    (X, y, feature_names, log_r_column) where ``log_r_column`` is the main
    regressor index for ``log1p(R_pos)``.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    sel = valid_mask
    y = np.asarray(eta_evt, dtype=np.float64)[sel]
    log_r = np.log1p(np.asarray(r_pos, dtype=np.float64)[sel])

    def standardized(name, values):
        values = np.asarray(values, dtype=np.float64)[sel]
        mean = float(standardizer[name]["mean"])
        std = float(standardizer[name]["std"])
        return (values - mean) / std

    horizon = standardized("horizon_index", horizon_index)
    duration = standardized("cf_duration", cf_duration)
    distance = standardized("initial_distance", initial_distance)
    visible_ratio = standardized("history_visible_ratio", history_visible_ratio)
    vcount = standardized("visible_count", visible_count)

    actions = np.asarray(cf_action)[sel].astype(np.int64)
    action_levels = np.unique(actions)
    one_hot = np.column_stack([(actions == level).astype(np.float64)
                               for level in action_levels]) if len(action_levels) else np.zeros((len(y), 0))

    intercept = np.ones((len(y), 1))
    blocks = [intercept, log_r[:, None], horizon[:, None], duration[:, None],
              distance[:, None], visible_ratio[:, None], vcount[:, None], one_hot]
    names = (["intercept", "log1p_R_pos", "horizon_index", "cf_duration",
              "initial_distance", "history_visible_ratio", "visible_count"] +
             [f"cf_action_{int(level)}" for level in action_levels])
    X = np.column_stack(blocks)
    return X, y, names, 1


def ols_coefficient(X: np.ndarray, y: np.ndarray, column: int) -> float:
    """Least-squares coefficient of one column (main regressor)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.shape[0] != y.shape[0] or X.shape[0] < X.shape[1]:
        raise ValueError("Need at least as many rows as columns for OLS")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[column])


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-6,
              penalize_intercept: bool = False) -> np.ndarray:
    """Closed-form ridge regression (NumPy only; no heavy dependency)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be 2-D and row-aligned with y")
    penalty = lam * np.eye(X.shape[1])
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)


def make_trend_coefficient_statistic(X: np.ndarray, y: np.ndarray, column: int,
                                     row_episode: np.ndarray, n_episodes: int):
    """Episode statistic: refit the trend model on resampled episodes' rows.

    Uses the penalized ridge closed form (never singular) so bootstrap resamples
    with rank-deficient covariate subsets cannot destabilize the solver, and a
    non-finite guard returns NaN rather than aborting.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    row_episode = np.asarray(row_episode, dtype=np.int64)
    n_columns = X.shape[1]

    def statistic(indices: np.ndarray) -> float:
        take = np.isin(row_episode, indices)
        if take.sum() < n_columns:
            return float("nan")
        Xs = X[take]
        ys = y[take]
        if not (np.isfinite(Xs).all() and np.isfinite(ys).all()):
            return float("nan")
        try:
            beta = ridge_fit(Xs, ys, lam=1e-6)
        except np.linalg.LinAlgError:
            return float("nan")
        return float(beta[column])

    return statistic


# --- Gate state machine ----------------------------------------------------------

def evaluate_gate_s(s1: bool, s2: bool, s3: bool) -> str:
    """Gate S passes only when optimization, response and usage checks all hold."""
    return GATE_S_PASS if (bool(s1) and bool(s2) and bool(s3)) else GATE_S_FAIL


def evaluate_gate_a(gate_s_status: str, conditions_met: bool | None) -> str:
    """Gate A stays blocked until Gate S passes; it is never CONFIRMED here."""
    if gate_s_status != GATE_S_PASS:
        return GATE_BLOCKED
    if conditions_met is None:
        raise ValueError("Gate A conditions must be provided once Gate S passes")
    return OUTCOME_GO if bool(conditions_met) else OUTCOME_NO_GO


def evaluate_gate_b(gate_s_status: str, delta_ci_upper: float | None,
                    delta_mean: float | None, n_episodes: int | None,
                    min_episodes: int) -> str:
    """Gate B: high-response absolute Delta<0 with CI fully below 0 and enough episodes."""
    if gate_s_status != GATE_S_PASS:
        return GATE_BLOCKED
    if n_episodes is None or delta_mean is None or delta_ci_upper is None:
        return GATE_B_UNDERPOWERED
    if n_episodes < min_episodes:
        return GATE_B_UNDERPOWERED
    if delta_mean < 0 and delta_ci_upper < 0:
        return GATE_B_SUPPORTED
    return GATE_B_NOT_SUPPORTED


def resolve_stage1_outcome(gate_s_status: str, gate_a_status: str) -> str:
    """Final Stage 1 outcome; Gate A cannot be interpreted unless Gate S passed."""
    if gate_s_status != GATE_S_PASS:
        return OUTCOME_INCONCLUSIVE
    if gate_a_status == GATE_BLOCKED:
        raise ValueError("Gate A must be evaluated once Gate S passes")
    return gate_a_status
