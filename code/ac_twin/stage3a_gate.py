"""Stage 3A Gate R_dev evaluation pipeline (section 3.5 of the plan).

This module turns already-inferred five-seed predictions plus the simple
baselines into the seven development-qualification conditions and the AND
adjudication. It is deliberately numpy-only: model inference (torch) lives in
the driver, so this layer is unit-testable without any training.

Direction convention (positive means the response model is better):

- ``I_MAE  = MAE_strongest_baseline - MAE_model``
- ``I_PR   = PR_AUC_model - PR_AUC_strongest_baseline``
- ``I_Brier = Brier_strongest_baseline - Brier_model``

The five-seed mean is the evaluated model; condition 7 separately checks the
five seeds agree in direction and the model is non-degenerate.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ac_twin.stage1_statistics import episode_bootstrap, spearman
from ac_twin.stage3_response import (
    expected_calibration_error,
    macro_brier,
    pr_auc,
    softmax,
)
from ac_twin.stage3_statistics import (
    evaluate_gate_r_dev,
    mae_per_episode,
    make_brier_difference_statistic,
    make_paired_contrast_statistic,
    make_pooled_spearman_statistic,
    make_pr_auc_difference_statistic,
)

# Binary task definitions (regime ids): presence = any positive response;
# high-vs-low/mid is restricted to positive agent-times.
_PRESENCE_POSITIVE = (1, 2, 3)
_HIGH = 3
_LOW_MID = (1, 2)


def _five_seed_mean(stack: np.ndarray) -> np.ndarray:
    return np.asarray(stack, dtype=np.float64).mean(axis=0)


def presence_scores(probs: np.ndarray) -> np.ndarray:
    """P(positive response) = 1 - P(zero), from 4-class probs ``...×4``."""
    probs = np.asarray(probs, dtype=np.float64)
    return probs[..., list(_PRESENCE_POSITIVE)].sum(axis=-1)


def high_vs_lowmid_scores(probs: np.ndarray) -> np.ndarray:
    """P(high | positive) used for the high-vs-(low+mid) task."""
    probs = np.asarray(probs, dtype=np.float64)
    numer = probs[..., _HIGH]
    denom = probs[..., list(_PRESENCE_POSITIVE)].sum(axis=-1)
    return numer / np.maximum(denom, 1e-12)


def binary_labels(regime_labels: np.ndarray, positive: tuple[int, ...]) -> np.ndarray:
    labels = np.asarray(regime_labels, dtype=np.int64)
    return np.isin(labels, list(positive)).astype(np.int64)


def restrict_to_positive(
    scores: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only positive agent-times for the high-vs-low/mid task."""
    positive = labels == 1
    keep = valid & positive
    return scores, labels, keep


# Metrics that must ALL be positive on ALL five seeds for "方向一致".
CONSISTENCY_METRICS = ("i_mae", "spearman", "i_pr_presence", "i_pr_high_vs_lowmid", "i_brier")


def check_five_seed_consistent(per_seed_metrics: Mapping[str, np.ndarray]) -> bool:
    """True when every seed is strictly positive on EVERY consistency metric.

    The plan's "五个 seed 方向一致" is an overall-direction requirement: it is
    not enough for the mean-MAE to be positive on all seeds while the
    classification metrics flip sign on some seeds. ``per_seed_metrics`` must
    contain every metric in :data:`CONSISTENCY_METRICS` (a missing metric is a
    failure, so an MAE-only check can never pass), each a length-5 array.
    """
    if set(per_seed_metrics) != set(CONSISTENCY_METRICS):
        return False
    for name in CONSISTENCY_METRICS:
        arr = np.asarray(per_seed_metrics[name], dtype=np.float64)
        if arr.size != 5 or not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
            return False
    return True


def check_non_degenerate(r_hat_mean: np.ndarray, valid_mask: np.ndarray) -> bool:
    """Reject constant-zero / constant-mean / single-vehicle-template models."""
    r = np.asarray(r_hat_mean, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    vals = r[valid]
    if vals.size == 0:
        return False
    if np.all(vals == 0.0):
        return False  # constant-zero
    if np.ptp(vals) < 1e-9:
        return False  # constant-mean (no variation at all)
    # single-vehicle template: every valid vehicle shares one identical time profile
    episodes, steps, agents = r.shape
    for e in range(episodes):
        profiles = [r[e, :, n] for n in range(agents) if valid[e, :, n].any()]
        if len(profiles) > 1:
            ref = profiles[0]
            if not all(np.allclose(ref, p, atol=1e-9) for p in profiles[1:]):
                return True
    return False  # every episode collapsed to a single shared profile


def select_strongest_baselines(
    *,
    r_true: np.ndarray,
    regime_labels: np.ndarray,
    valid_mask: np.ndarray,
    continuous_baselines: Mapping[str, np.ndarray],
    classification_baselines: Mapping[str, np.ndarray],
) -> dict[str, str]:
    """Freeze the strongest baseline per metric on the VALIDATION split.

    Selecting the strongest baseline on the development-test split would leak
    test information into the gate; it must be frozen on validation beforehand
    (item 3). Returns the winning baseline NAME for each of the four baseline
    comparisons. ``r_true``/``regime_labels``/``valid_mask``/baseline predictions
    are all validation-split arrays here.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    r_true = np.asarray(r_true, dtype=np.float64)
    labels = np.asarray(regime_labels, dtype=np.int64)

    cont_mae = {
        name: float(np.nanmean(mae_per_episode(np.asarray(p, dtype=np.float64), r_true, valid)))
        for name, p in continuous_baselines.items()
    }
    strongest_continuous = min(cont_mae, key=cont_mae.get)

    presence_lab = binary_labels(labels, _PRESENCE_POSITIVE)
    presence_auc = {
        name: pr_auc(presence_scores(p), presence_lab, valid)
        for name, p in classification_baselines.items()
    }
    strongest_presence = max(presence_auc, key=lambda k: (np.nan_to_num(presence_auc[k], nan=-1.0), k))

    pos_mask = valid & (presence_lab == 1)
    high_lab = binary_labels(labels, (_HIGH,))
    high_auc = {
        name: pr_auc(high_vs_lowmid_scores(p), high_lab, pos_mask)
        for name, p in classification_baselines.items()
    }
    strongest_high = max(high_auc, key=lambda k: (np.nan_to_num(high_auc[k], nan=-1.0), k))

    brier = {name: macro_brier(p, labels, valid) for name, p in classification_baselines.items()}
    strongest_brier = min(brier, key=brier.get)

    return {
        "strongest_continuous": strongest_continuous,
        "strongest_presence": strongest_presence,
        "strongest_high_vs_lowmid": strongest_high,
        "strongest_brier": strongest_brier,
    }


def evaluate_h4_gate(
    *,
    model_r_hat_seeds: np.ndarray,
    model_regime_logits_seeds: np.ndarray,
    temperature: float,
    r_true: np.ndarray,
    regime_labels: np.ndarray,
    valid_mask: np.ndarray,
    continuous_baselines: Mapping[str, np.ndarray],
    classification_baselines: Mapping[str, np.ndarray],
    strongest_baselines: Mapping[str, str],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 20260908,
) -> dict[str, Any]:
    """Compute all H4 metrics, episode bootstrap CIs and the 7-condition verdict.

    ``model_r_hat_seeds`` is ``S×E×T×N`` (already ``expm1``-inverted, >=0) and
    ``model_regime_logits_seeds`` is ``S×E×T×N×4`` raw logits; ``temperature``
    was fit on validation only. Baselines map a name to predictions over the same
    ``E×T×N(/×4)`` grid (fit on train only). ``strongest_baselines`` carries the
    baseline NAMES frozen on validation by :func:`select_strongest_baselines` —
    the gate never re-selects a baseline on the development-test split.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    r_true = np.asarray(r_true, dtype=np.float64)
    labels = np.asarray(regime_labels, dtype=np.int64)
    n_episodes = r_true.shape[0]
    n_seeds = model_r_hat_seeds.shape[0]

    # Frozen baseline names (selected on validation, never re-picked here).
    strongest_cont = strongest_baselines["strongest_continuous"]
    strongest_presence = strongest_baselines["strongest_presence"]
    strongest_high = strongest_baselines["strongest_high_vs_lowmid"]
    strongest_brier = strongest_baselines["strongest_brier"]

    # Five-seed ensemble (mean) is the evaluated model.
    model_r = _five_seed_mean(model_r_hat_seeds)  # E,T,N
    raw_logits = _five_seed_mean(model_regime_logits_seeds)  # E,T,N,4
    uncal_probs = softmax(raw_logits, axis=-1)
    cal_probs = softmax(raw_logits / float(temperature), axis=-1)

    # --- continuous -----------------------------------------------------------
    model_mae_ep = mae_per_episode(model_r, r_true, valid)
    baseline_mae = {
        name: mae_per_episode(np.asarray(pred, dtype=np.float64), r_true, valid)
        for name, pred in continuous_baselines.items()
    }
    i_mae_boot = episode_bootstrap(
        n_episodes,
        make_paired_contrast_statistic(baseline_mae[strongest_cont], model_mae_ep),
        n_resamples, confidence_level, bootstrap_seed,
    )
    spearman_boot = episode_bootstrap(
        n_episodes,
        make_pooled_spearman_statistic(model_r, r_true, valid),
        n_resamples, confidence_level, bootstrap_seed,
    )

    # --- classification --------------------------------------------------------
    presence_lab = binary_labels(labels, _PRESENCE_POSITIVE)
    model_presence = presence_scores(cal_probs)
    base_presence = presence_scores(classification_baselines[strongest_presence])
    i_pr_presence_boot = episode_bootstrap(
        n_episodes,
        make_pr_auc_difference_statistic(
            model_presence, presence_lab, base_presence, presence_lab, valid
        ),
        n_resamples, confidence_level, bootstrap_seed,
    )

    pos_mask = valid & (presence_lab == 1)
    high_lab = binary_labels(labels, (_HIGH,))
    model_high = high_vs_lowmid_scores(cal_probs)
    base_high = high_vs_lowmid_scores(classification_baselines[strongest_high])
    i_pr_high_boot = episode_bootstrap(
        n_episodes,
        make_pr_auc_difference_statistic(
            model_high, high_lab, base_high, high_lab, pos_mask
        ),
        n_resamples, confidence_level, bootstrap_seed,
    )

    model_brier = macro_brier(cal_probs, labels, valid)
    base_brier = {n: macro_brier(p, labels, valid) for n, p in classification_baselines.items()}
    i_brier_boot = episode_bootstrap(
        n_episodes,
        make_brier_difference_statistic(
            cal_probs, classification_baselines[strongest_brier], labels, valid
        ),
        n_resamples, confidence_level, bootstrap_seed,
    )

    # --- calibration guard ------------------------------------------------------
    ece_calibrated = expected_calibration_error(cal_probs, labels, valid, n_bins=15, equal_mass=True)
    ece_uncalibrated = expected_calibration_error(uncal_probs, labels, valid, n_bins=15, equal_mass=True)

    # --- per-seed consistency across ALL metrics (item 2) ----------------------
    per_seed = {"i_mae": [], "spearman": [], "i_pr_presence": [], "i_pr_high_vs_lowmid": [], "i_brier": []}
    for s in range(n_seeds):
        r_s = np.asarray(model_r_hat_seeds[s], dtype=np.float64)
        probs_s = softmax(np.asarray(model_regime_logits_seeds[s], dtype=np.float64) / float(temperature), axis=-1)
        per_seed["i_mae"].append(
            float(np.nanmean(baseline_mae[strongest_cont]) - np.nanmean(mae_per_episode(r_s, r_true, valid)))
        )
        xs, ys = r_s[valid], r_true[valid]
        per_seed["spearman"].append(float(spearman(xs, ys)) if xs.size > 2 and np.ptp(xs) > 0 and np.ptp(ys) > 0 else float("nan"))
        per_seed["i_pr_presence"].append(
            pr_auc(presence_scores(probs_s), presence_lab, valid) - pr_auc(base_presence, presence_lab, valid)
        )
        per_seed["i_pr_high_vs_lowmid"].append(
            pr_auc(high_vs_lowmid_scores(probs_s), high_lab, pos_mask) - pr_auc(base_high, high_lab, pos_mask)
        )
        per_seed["i_brier"].append(base_brier[strongest_brier] - macro_brier(probs_s, labels, valid))
    per_seed = {k: np.array(v, dtype=np.float64) for k, v in per_seed.items()}
    five_seed_consistent = check_five_seed_consistent(per_seed)
    non_degenerate = check_non_degenerate(model_r, valid)

    gate_input = {
        "i_mae": i_mae_boot,
        "spearman": spearman_boot,
        "i_pr_presence": i_pr_presence_boot,
        "i_pr_high_vs_lowmid": i_pr_high_boot,
        "i_brier": i_brier_boot,
        "ece": ece_calibrated,
        "ece_baseline": ece_uncalibrated,
        "five_seed_consistent": five_seed_consistent,
        "non_degenerate": non_degenerate,
    }
    verdict = evaluate_gate_r_dev(gate_input)

    return {
        "verdict": verdict,
        "metrics": {
            "i_mae": i_mae_boot,
            "spearman": spearman_boot,
            "i_pr_presence": i_pr_presence_boot,
            "i_pr_high_vs_lowmid": i_pr_high_boot,
            "i_brier": i_brier_boot,
            "ece_calibrated": ece_calibrated,
            "ece_uncalibrated": ece_uncalibrated,
            "model_mae": float(np.nanmean(model_mae_ep)),
            "model_brier": float(model_brier),
        },
        "baselines": {
            "strongest_continuous": strongest_cont,
            "strongest_presence": strongest_presence,
            "strongest_high_vs_lowmid": strongest_high,
            "strongest_brier": strongest_brier,
            "frozen_on": "validation",
            "continuous_mae": {k: float(np.nanmean(v)) for k, v in baseline_mae.items()},
            "classification_brier": {k: float(v) for k, v in base_brier.items()},
        },
        "five_seed": {
            "per_seed": {k: v.tolist() for k, v in per_seed.items()},
            "consistent": five_seed_consistent,
            "non_degenerate": non_degenerate,
        },
        "temperature": float(temperature),
        "bootstrap": {
            "n_resamples": int(n_resamples),
            "confidence_level": float(confidence_level),
            "seed": int(bootstrap_seed),
            "n_episodes": int(n_episodes),
        },
    }
