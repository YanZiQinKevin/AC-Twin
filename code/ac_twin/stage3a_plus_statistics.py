"""Stage 3A Plus statistics: ten-seed diagnostics, calibration-failure
classification, Gate R+ core/stability and hierarchical bootstrap
(plan `AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1` §3, §5).

This layer is pure numpy so it is unit-testable without torch. Model inference
and calibration fitting live in the driver / calibration module; here we only
turn already-aligned per-seed predictions + frozen baselines into the diagnostic
decomposition and the mechanical Gate R+ adjudication.

Direction convention (positive means the response model beats the
validation-frozen strongest baseline):

- ``I_MAE   = MAE_baseline - MAE_model``
- ``I_PR    = PR_AUC_model - PR_AUC_baseline``
- ``I_Brier = Brier_baseline - Brier_model``
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ac_twin.stage1_statistics import episode_bootstrap, spearman
from ac_twin.stage3_data import (
    STAGE3A_PLUS_ALL_SEEDS,
    STAGE3A_PLUS_DEV_SEEDS,
)
from ac_twin.stage3_response import (
    expected_calibration_error,
    macro_brier,
    pr_auc,
    softmax,
)
from ac_twin.stage3_statistics import (
    mae_per_episode,
    make_brier_difference_statistic,
    make_paired_contrast_statistic,
    make_pooled_spearman_statistic,
    make_pr_auc_difference_statistic,
)

# Binary task definitions (regime ids), shared with stage3a_gate.
_PRESENCE_POSITIVE = (1, 2, 3)
_HIGH = 3
_LOW_MID = (1, 2)

# The five core metrics tracked per seed (Gate R+_stability and direction
# consistency). I_Brier additionally carries an episode bootstrap CI.
CORE_METRICS = ("i_mae", "spearman", "i_pr_presence", "i_pr_high_vs_lowmid", "i_brier")

# Gate R+ frozen thresholds (plan §5).
ECE_GUARD = 0.05
ECE_RELATIVE_IMPROVEMENT = 0.20
MIN_POSITIVE_BRIER_SEEDS = 8          # at least 8/10 seeds with I_Brier > 0
WORST_SEED_NONINFERIORITY_FRAC = 0.05  # delta_B = 5% of strongest-baseline Brier
NONDEGENERATE_STD_FLOOR = 1e-6

# Diagnostic classification statuses (plan §3.2).
STATUS_RUN_INVALID = "RUN_INVALID_ENGINEERING"
STATUS_SHARED_TEMP = "SHARED_TEMPERATURE_MISMATCH"
STATUS_REGIME_LOCALIZED = "REGIME_LOCALIZED_CALIBRATION_FAILURE"
STATUS_BORDERLINE = "BORDERLINE_SEED_VARIATION"
STATUS_SYSTEMATIC = "SYSTEMATIC_SEED_CALIBRATION_FAILURE"
STATUS_MIXED = "MIXED_CALIBRATION_FAILURE"


# --------------------------------------------------------------------------- #
# Small metric helpers
# --------------------------------------------------------------------------- #
def presence_scores(probs: np.ndarray) -> np.ndarray:
    """P(positive response) = 1 - P(zero) from 4-class probs ``...×4``."""
    probs = np.asarray(probs, dtype=np.float64)
    return probs[..., list(_PRESENCE_POSITIVE)].sum(axis=-1)


def high_vs_lowmid_scores(probs: np.ndarray) -> np.ndarray:
    """P(high | positive) for the high-vs-(low+mid) task."""
    probs = np.asarray(probs, dtype=np.float64)
    numer = probs[..., _HIGH]
    denom = probs[..., list(_PRESENCE_POSITIVE)].sum(axis=-1)
    return numer / np.maximum(denom, 1e-12)


def binary_labels(regime_labels: np.ndarray, positive: tuple[int, ...]) -> np.ndarray:
    labels = np.asarray(regime_labels, dtype=np.int64)
    return np.isin(labels, list(positive)).astype(np.int64)


def multiclass_nll(probs: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray) -> float:
    """Mean negative log-likelihood over valid agent-times (lower is better)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    p = probs[..., :]
    flat_p = p[valid]            # (M,4)
    flat_y = labels[valid]       # (M,)
    if flat_y.size == 0:
        return float("nan")
    p_true = np.clip(flat_p[np.arange(flat_y.size), flat_y], 1e-12, 1.0)
    return float(-np.log(p_true).mean())


def reliability_table(
    probs: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    """15-bin equal-mass reliability table (count / mean confidence / accuracy / gap)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    conf = probs.max(axis=-1)[valid]
    pred = probs.argmax(axis=-1)[valid]
    truth = labels[valid]
    n = conf.shape[0]
    if n == 0:
        raise ValueError("reliability table needs valid predictions")
    correct = (pred == truth).astype(np.float64)
    order = np.argsort(conf, kind="stable")
    bins = []
    for idx in np.array_split(order, n_bins):
        if idx.size == 0:
            bins.append({"count": 0, "mean_confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue
        mean_conf = float(conf[idx].mean())
        acc = float(correct[idx].mean())
        bins.append({
            "count": int(idx.size),
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "gap": abs(acc - mean_conf),
        })
    ece = float(sum((b["count"] / n) * b["gap"] for b in bins))
    return {"n_bins": int(n_bins), "bins": bins, "ece": ece, "n_valid": int(n)}


def _per_agent_time_brier(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-(e,t,n) macro Brier contribution: mean over the 4 classes of (p-y)^2."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    onehot = np.eye(probs.shape[-1], dtype=np.float64)[labels]
    return ((probs - onehot) ** 2).mean(axis=-1)


def negative_brier_subgroup_concentration(
    model_probs: np.ndarray,
    baseline_probs: np.ndarray,
    regime_labels: np.ndarray,
    valid_mask: np.ndarray,
    horizons: np.ndarray,
    cf_actions: np.ndarray,
    cf_durations: np.ndarray,
) -> dict[str, Any]:
    """Where the model's excess Brier (vs baseline) concentrates.

    ``excess = brier_model - brier_baseline`` per valid agent-time (positive means
    the model is worse). For each of the four subgroup dimensions (true regime,
    horizon, cf_action, cf_duration) we sum ``|excess|`` per subgroup and report
    each subgroup's share of the total. ``max_subgroup_concentration`` is the
    largest share across all dimensions — the ≥0.80 signal for a
    REGIME_LOCALIZED_CALIBRATION_FAILURE.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    excess = (_per_agent_time_brier(model_probs, regime_labels)
              - _per_agent_time_brier(baseline_probs, regime_labels))
    E, T, N = excess.shape
    dims = {
        "regime": np.asarray(regime_labels, dtype=np.int64),
        "horizon": np.asarray(horizons, dtype=np.int64),
        "action": np.broadcast_to(np.asarray(cf_actions, dtype=np.int64)[:, None, None], (E, T, N)),
        "duration": np.broadcast_to(np.asarray(cf_durations, dtype=np.int64)[:, None, None], (E, T, N)),
    }
    total_abs = float(np.abs(excess[valid]).sum())
    decomposition: dict[str, Any] = {}
    max_conc = 0.0
    for name, key_array in dims.items():
        key_array = np.asarray(key_array)
        subgroup: dict[str, Any] = {}
        for value in sorted(int(v) for v in np.unique(key_array[valid])):
            sel = valid & (key_array == value)
            abs_sum = float(np.abs(excess[sel]).sum())
            share = (abs_sum / total_abs) if total_abs > 0 else 0.0
            subgroup[str(value)] = {
                "abs_excess": abs_sum,
                "net_excess": float(excess[sel].sum()),
                "count": int(sel.sum()),
                "share_of_total_abs": share,
            }
            max_conc = max(max_conc, share)
        decomposition[name] = subgroup
    decomposition["max_subgroup_concentration"] = float(max_conc)
    return decomposition


# --------------------------------------------------------------------------- #
# decompose_seed_metrics (plan §3.1) — one eval set (A or B)
# --------------------------------------------------------------------------- #
def decompose_seed_metrics(
    *,
    r_hat_seeds: np.ndarray,
    logits_seeds: np.ndarray,
    calibrated_probs_seeds: np.ndarray,
    r_true: np.ndarray,
    regime_labels: np.ndarray,
    valid_mask: np.ndarray,
    episode_ids: np.ndarray,
    horizons: np.ndarray,
    cf_actions: np.ndarray,
    cf_durations: np.ndarray,
    baselines: Mapping[str, Any],
    n_resamples: int,
    bootstrap_seed: int,
    expected_episode_ids: np.ndarray | None = None,
    n_production_seeds: int = 5,
) -> dict[str, Any]:
    """Full per-seed + ensemble diagnostic decomposition for one eval set.

    All ten fixed seeds must be present (``S == len(STAGE3A_PLUS_ALL_SEEDS)``) —
    the stability report never drops a seed. Per-seed metrics cover all ten; the
    ensemble-level quantities (ensemble metrics, reliability, ECE, calibration
    semantics, Brier decomposition) use the production-seed probability mean
    (first ``n_production_seeds`` seeds), matching Gate R+_core (plan §5.1).
    ``baselines`` carries the validation-frozen strongest baselines as
    ``{"continuous": {name: E×T×N}, "classification": {name: E×T×N×4},
      "strongest_continuous": name, "strongest_brier": name}``.
    """
    r_hat_seeds = np.asarray(r_hat_seeds, dtype=np.float64)
    logits_seeds = np.asarray(logits_seeds, dtype=np.float64)
    cal_seeds = np.asarray(calibrated_probs_seeds, dtype=np.float64)
    r_true = np.asarray(r_true, dtype=np.float64)
    labels = np.asarray(regime_labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    episode_ids = np.asarray(episode_ids)

    seeds = [str(s) for s in STAGE3A_PLUS_ALL_SEEDS]
    n_seeds = len(seeds)
    if r_hat_seeds.shape[0] != n_seeds or cal_seeds.shape[0] != n_seeds:
        raise ValueError(
            f"decompose_seed_metrics requires all {n_seeds} fixed seeds "
            f"(got r_hat S={r_hat_seeds.shape[0]}, probs S={cal_seeds.shape[0]})"
        )
    if not (np.isfinite(r_hat_seeds).all() and np.isfinite(logits_seeds).all()
            and np.isfinite(cal_seeds).all()):
        raise ValueError("non-finite prediction detected (NaN/Inf)")
    if expected_episode_ids is not None and not np.array_equal(
            episode_ids, np.asarray(expected_episode_ids)):
        raise RuntimeError(
            "prediction episode IDs do not match the requested eval episode IDs; "
            "refusing to diagnose misaligned predictions"
        )

    n_episodes = int(r_true.shape[0])
    cont_base = np.asarray(baselines["continuous"][baselines["strongest_continuous"]], dtype=np.float64)
    clf_base = np.asarray(baselines["classification"][baselines["strongest_brier"]], dtype=np.float64)

    presence_lab = binary_labels(labels, _PRESENCE_POSITIVE)
    pos_mask = valid & (presence_lab == 1)
    high_lab = binary_labels(labels, (_HIGH,))
    baseline_mae_ep = mae_per_episode(cont_base, r_true, valid)
    baseline_presence = presence_scores(clf_base)
    baseline_high = high_vs_lowmid_scores(clf_base)
    base_pr_presence = pr_auc(baseline_presence, presence_lab, valid)
    base_pr_high = pr_auc(baseline_high, high_lab, pos_mask)

    per_seed: dict[str, Any] = {}
    for s_idx, seed in enumerate(seeds):
        r_s = r_hat_seeds[s_idx]
        probs_s = cal_seeds[s_idx]
        model_mae_ep = mae_per_episode(r_s, r_true, valid)
        xs, ys = r_s[valid], r_true[valid]
        sp = float(spearman(xs, ys)) if xs.size > 2 and np.ptp(xs) > 0 and np.ptp(ys) > 0 else float("nan")
        i_brier_boot = episode_bootstrap(
            n_episodes,
            make_brier_difference_statistic(probs_s, clf_base, labels, valid),
            int(n_resamples), 0.95, int(bootstrap_seed) + s_idx,
        )
        per_seed[seed] = {
            "mae": float(np.nanmean(model_mae_ep)),
            "i_mae": float(np.nanmean(baseline_mae_ep) - np.nanmean(model_mae_ep)),
            "spearman": sp,
            "pr_auc_presence": pr_auc(presence_scores(probs_s), presence_lab, valid),
            "pr_auc_high_vs_lowmid": pr_auc(high_vs_lowmid_scores(probs_s), high_lab, pos_mask),
            "i_pr_presence": float(
                pr_auc(presence_scores(probs_s), presence_lab, valid) - base_pr_presence),
            "i_pr_high_vs_lowmid": float(
                pr_auc(high_vs_lowmid_scores(probs_s), high_lab, pos_mask) - base_pr_high),
            "brier": macro_brier(probs_s, labels, valid),
            "nll": multiclass_nll(probs_s, labels, valid),
            "ece": expected_calibration_error(probs_s, labels, valid, n_bins=15, equal_mass=True),
            "i_brier": float(i_brier_boot["point"]),
            "i_brier_ci_lower": float(i_brier_boot["ci_lower"]),
            "i_brier_ci_upper": float(i_brier_boot["ci_upper"]),
            "non_degenerate": bool(
                valid.any() and np.std(r_s[valid]) > NONDEGENERATE_STD_FLOOR),
            "finite": bool(np.isfinite(r_s).all() and np.isfinite(probs_s).all()),
        }

    # --- ensemble: mean of PRODUCTION-seed probabilities (plan §5.1) ----------
    n_prod = int(n_production_seeds)
    cal_ensemble = cal_seeds[:n_prod].mean(axis=0)
    uncal_ensemble = np.stack(
        [softmax(logits_seeds[s], axis=-1) for s in range(n_prod)], axis=0).mean(axis=0)
    r_ensemble = r_hat_seeds[:n_prod].mean(axis=0)

    # Ensemble-level five core metrics with episode bootstrap CIs (Gate R+_core
    # reuses exactly these; §5.1). Continuous uses the frozen strongest
    # continuous baseline; classification the frozen strongest brier baseline.
    model_mae_ep = mae_per_episode(r_ensemble, r_true, valid)
    ens_presence = presence_scores(cal_ensemble)
    ens_high = high_vs_lowmid_scores(cal_ensemble)

    def _boot(stat_fn, offset):
        return episode_bootstrap(n_episodes, stat_fn, int(n_resamples), 0.95,
                                 int(bootstrap_seed) + 1000 + offset)

    ensemble = {
        "mae": float(np.nanmean(model_mae_ep)),
        "spearman": (float(spearman(r_ensemble[valid], r_true[valid]))
                     if np.ptp(r_ensemble[valid]) > 0 else float("nan")),
        "pr_auc_presence": pr_auc(ens_presence, presence_lab, valid),
        "pr_auc_high_vs_lowmid": pr_auc(ens_high, high_lab, pos_mask),
        "brier": macro_brier(cal_ensemble, labels, valid),
        "nll": multiclass_nll(cal_ensemble, labels, valid),
        "ece": expected_calibration_error(cal_ensemble, labels, valid, n_bins=15, equal_mass=True),
        "ece_uncalibrated": expected_calibration_error(uncal_ensemble, labels, valid, n_bins=15, equal_mass=True),
        # baseline-relative core metrics with bootstrap CIs (positive = better):
        "i_mae": _boot(make_paired_contrast_statistic(baseline_mae_ep, model_mae_ep), 0),
        "spearman_boot": _boot(make_pooled_spearman_statistic(r_ensemble, r_true, valid), 1),
        "i_pr_presence": _boot(make_pr_auc_difference_statistic(
            ens_presence, presence_lab, baseline_presence, presence_lab, valid), 2),
        "i_pr_high_vs_lowmid": _boot(make_pr_auc_difference_statistic(
            ens_high, high_lab, baseline_high, high_lab, pos_mask), 3),
        "i_brier": _boot(make_brier_difference_statistic(cal_ensemble, clf_base, labels, valid), 4),
        "r_ensemble_std": float(np.std(r_ensemble[valid])) if valid.any() else 0.0,
        "non_degenerate": bool(valid.any() and np.std(r_ensemble[valid]) > NONDEGENERATE_STD_FLOOR),
    }

    # --- Brier decomposition by subgroup (ensemble excess vs strongest) -------
    brier_decomposition = negative_brier_subgroup_concentration(
        cal_ensemble, clf_base, labels, valid, horizons, cf_actions, cf_durations
    )

    # --- reliability + calibration semantics (item 8) --------------------------
    reliability = reliability_table(cal_ensemble, labels, valid, n_bins=15)
    tv = float(0.5 * np.abs(cal_ensemble - uncal_ensemble).sum(axis=-1)[valid].mean())
    flip = float((cal_ensemble.argmax(-1) != uncal_ensemble.argmax(-1))[valid].mean())

    def _entropy(p):
        pv = np.clip(p[valid], 1e-12, 1.0)
        return float(-(pv * np.log(pv)).sum(axis=-1).mean())

    calibration_semantics = {
        "total_variation": tv,
        "argmax_flip_rate": flip,
        "entropy_uncalibrated": _entropy(uncal_ensemble),
        "entropy_calibrated": _entropy(cal_ensemble),
    }

    return {
        "seeds": seeds,
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "per_seed": per_seed,
        "ensemble": ensemble,
        "brier_decomposition": brier_decomposition,
        "reliability": reliability,
        "calibration_semantics": calibration_semantics,
        "bootstrap": {"n_resamples": int(n_resamples), "seed": int(bootstrap_seed)},
    }


# --------------------------------------------------------------------------- #
# Calibration-failure classification (plan §3.2) — mechanical, exactly one status
# --------------------------------------------------------------------------- #
def classify_calibration_failure(diagnostic: Mapping[str, Any]) -> str:
    """Mechanically classify the ten-seed calibration diagnostic into one status.

    Required keys: ``engineering_ok`` (bool); ``per_seed_cal_improves_303``
    ({"A": bool, "B": bool}); ``ensemble_core_not_degraded_under_per_seed_cal``
    (bool); ``max_subgroup_concentration`` (float, max over A and B); ``seed303``
    ({"A"/"B": {"i_brier_ci_lower","i_brier_ci_upper"}}).
    """
    if not bool(diagnostic.get("engineering_ok", False)):
        return STATUS_RUN_INVALID
    improves = diagnostic.get("per_seed_cal_improves_303", {})
    if (bool(improves.get("A")) and bool(improves.get("B"))
            and bool(diagnostic.get("ensemble_core_not_degraded_under_per_seed_cal", False))):
        return STATUS_SHARED_TEMP
    if float(diagnostic.get("max_subgroup_concentration", 0.0)) >= 0.80:
        return STATUS_REGIME_LOCALIZED
    a = diagnostic["seed303"]["A"]
    b = diagnostic["seed303"]["B"]
    contains_zero = (
        (float(a["i_brier_ci_lower"]) <= 0.0 <= float(a["i_brier_ci_upper"]))
        or (float(b["i_brier_ci_lower"]) <= 0.0 <= float(b["i_brier_ci_upper"]))
    )
    systematic = (float(a["i_brier_ci_upper"]) < 0.0) and (float(b["i_brier_ci_upper"]) < 0.0)
    if contains_zero:
        return STATUS_BORDERLINE
    if systematic:
        return STATUS_SYSTEMATIC
    return STATUS_MIXED


def direction_consistency(
    report_a: Mapping[str, Any], report_b: Mapping[str, Any], *, metric: str
) -> dict[str, Any]:
    """Per-seed sign agreement of ``metric`` between dev sets A and B (item 7)."""
    per_a = report_a["per_seed"]
    per_b = report_b["per_seed"]
    seeds = list(per_a)
    signs = []
    for seed in seeds:
        va = float(per_a[seed][metric])
        vb = float(per_b[seed][metric])
        same = (va > 0 and vb > 0) or (va < 0 and vb < 0) or (va == 0 and vb == 0)
        signs.append({"seed": seed, "a": va, "b": vb, "same_sign": bool(same)})
    fraction = float(np.mean([s["same_sign"] for s in signs])) if signs else 0.0
    return {"metric": metric, "per_seed": signs, "fraction_same_sign": fraction}


# --------------------------------------------------------------------------- #
# Gate R+ (plan §5): core, stability, Q+ and the AND adjudication
# --------------------------------------------------------------------------- #
def hierarchical_seed_episode_bootstrap(
    values: np.ndarray, n_resamples: int, seed: int
) -> dict[str, Any]:
    """Hierarchical bootstrap of mean per-(seed, episode) I_Brier.

    ``values`` is ``S×E`` (per-seed, per-episode I_Brier). Each replicate first
    resamples ``S`` seeds with replacement, then resamples ``E`` episodes with
    replacement within each sampled seed, and averages the pooled S×E values.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("hierarchical bootstrap values must be S×E")
    n_seeds, n_episodes = values.shape
    rng = np.random.default_rng(int(seed))
    point = float(np.nanmean(values))
    draws = np.empty(int(n_resamples), dtype=np.float64)
    for r in range(int(n_resamples)):
        s_idx = rng.integers(0, n_seeds, size=n_seeds)
        pooled = np.concatenate([values[s][rng.integers(0, n_episodes, size=n_episodes)]
                                 for s in s_idx])
        draws[r] = float(np.nanmean(pooled))
    finite = np.isfinite(draws)
    if finite.sum() < 2:
        lower = upper = float("nan")
    else:
        lower, upper = np.quantile(draws[finite], [0.025, 0.975])
    return {
        "point": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_resamples": int(n_resamples),
        "n_finite_draws": int(finite.sum()),
        "seed": int(seed),
        "n_seeds": int(n_seeds),
        "n_episodes": int(n_episodes),
    }


def evaluate_gate_r_plus_core(ensemble: Mapping[str, Any]) -> dict[str, Any]:
    """AND the six Gate R+_core conditions on the production-seed ensemble.

    ``ensemble`` carries the five bootstrap dicts (``i_mae``, ``spearman_boot``,
    ``i_pr_presence``, ``i_pr_high_vs_lowmid``, ``i_brier``) plus ``ece`` and
    ``ece_uncalibrated``.
    """
    checks = {
        "i_mae_ci_lower_gt0": float(ensemble["i_mae"]["ci_lower"]) > 0.0,
        "spearman_ci_lower_gt0": float(ensemble["spearman_boot"]["ci_lower"]) > 0.0,
        "i_pr_presence_ci_lower_gt0": float(ensemble["i_pr_presence"]["ci_lower"]) > 0.0,
        "i_pr_high_vs_lowmid_ci_lower_gt0": float(ensemble["i_pr_high_vs_lowmid"]["ci_lower"]) > 0.0,
        "i_brier_ci_lower_gt0": float(ensemble["i_brier"]["ci_lower"]) > 0.0,
        "ece_calibrated": (
            float(ensemble["ece"]) <= ECE_GUARD
            or float(ensemble["ece"]) <= (1.0 - ECE_RELATIVE_IMPROVEMENT) * float(ensemble["ece_uncalibrated"])
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def evaluate_gate_r_plus_stability(report: Mapping[str, Any]) -> dict[str, Any]:
    """AND the six Gate R+_stability conditions over the ten fixed seeds.

    ``report`` carries ``per_seed`` (per-seed ``i_mae``/``spearman``/
    ``i_pr_presence``/``i_pr_high_vs_lowmid``/``i_brier``/``non_degenerate``/
    ``finite``), ``baseline_brier`` and ``hierarchical_i_brier`` (bootstrap dict).
    Up to two seeds may show a small negative I_Brier within the 5%-of-baseline
    non-inferiority margin, but the population CI, median and worst-seed bound
    must all hold.
    """
    per_seed = report["per_seed"]
    seeds = list(per_seed)
    i_brier = np.array([float(per_seed[s]["i_brier"]) for s in seeds], dtype=np.float64)
    baseline_brier = float(report["baseline_brier"])
    delta_b = WORST_SEED_NONINFERIORITY_FRAC * baseline_brier

    core_positive = all(
        float(per_seed[s]["i_mae"]) > 0.0
        and float(per_seed[s]["spearman"]) > 0.0
        and float(per_seed[s]["i_pr_presence"]) > 0.0
        and float(per_seed[s]["i_pr_high_vs_lowmid"]) > 0.0
        for s in seeds
    )
    checks = {
        "per_seed_core_metrics_positive": bool(core_positive),
        "at_least_8_of_10_positive": bool((i_brier > 0.0).sum() >= MIN_POSITIVE_BRIER_SEEDS),
        "median_i_brier_positive": bool(np.median(i_brier) > 0.0),
        "worst_seed_noninferior": bool(i_brier.min() >= -delta_b),
        "hierarchical_ci_lower_gt0": float(report["hierarchical_i_brier"]["ci_lower"]) > 0.0,
        "all_seeds_nondegenerate": bool(all(
            per_seed[s]["non_degenerate"] and per_seed[s]["finite"] for s in seeds)),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "i_brier_per_seed": {s: float(per_seed[s]["i_brier"]) for s in seeds},
        "n_positive_brier": int((i_brier > 0.0).sum()),
        "median_i_brier": float(np.median(i_brier)),
        "worst_i_brier": float(i_brier.min()),
        "delta_b": float(delta_b),
    }


def evaluate_gate_r_plus(report: Mapping[str, Any]) -> dict[str, Any]:
    """AND Gate R+_core, Gate R+_stability and Gate Q+ (plan §5.1/§5.2/§5.4)."""
    core = evaluate_gate_r_plus_core(report["core"])
    stability = evaluate_gate_r_plus_stability(report["stability"])
    gate_q = bool(report["gate_q_passed"])
    return {
        "passed": bool(core["passed"] and stability["passed"] and gate_q),
        "core": core,
        "stability": stability,
        "gate_q_passed": gate_q,
    }
