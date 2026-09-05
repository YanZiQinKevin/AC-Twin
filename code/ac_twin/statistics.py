from __future__ import annotations

from typing import Any

import numpy as np


def trajectory_errors(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-episode ADE and FDE for [N, H, 2] arrays."""
    distances = np.linalg.norm(np.asarray(pred) - np.asarray(target), axis=-1)
    return distances.mean(axis=1), distances[:, -1]


def paired_bootstrap_mean_difference(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    reference_error = np.asarray(reference_error, dtype=np.float64)
    candidate_error = np.asarray(candidate_error, dtype=np.float64)
    if reference_error.shape != candidate_error.shape:
        raise ValueError("Paired errors must have identical shapes.")
    if reference_error.ndim != 1 or len(reference_error) < 2:
        raise ValueError("At least two paired episode errors are required.")

    differences = reference_error - candidate_error
    rng = np.random.default_rng(seed)
    n = len(differences)
    bootstrap_means = np.empty(int(n_resamples), dtype=np.float64)
    chunk_size = 500
    written = 0
    while written < n_resamples:
        take = min(chunk_size, n_resamples - written)
        idx = rng.integers(0, n, size=(take, n))
        bootstrap_means[written : written + take] = differences[idx].mean(axis=1)
        written += take

    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "independent_unit": "test_episode",
        "n": int(n),
        "mean_reference_error": float(reference_error.mean()),
        "mean_candidate_error": float(candidate_error.mean()),
        "mean_paired_difference_reference_minus_candidate": float(differences.mean()),
        "confidence_level": float(confidence_level),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(seed),
    }


def decision_from_bootstrap(
    result: dict[str, Any],
    *,
    minimum_improvement_percent: float,
    strong_improvement_percent: float,
) -> dict[str, Any]:
    ref = float(result["mean_reference_error"])
    cand = float(result["mean_candidate_error"])
    improvement = 100.0 * (ref - cand) / ref if ref > 0 else float("nan")
    lower = float(result["ci_lower"])
    upper = float(result["ci_upper"])

    if upper <= 0:
        verdict = "FAIL"
    elif lower <= 0:
        verdict = "INCONCLUSIVE"
    elif improvement >= strong_improvement_percent:
        verdict = "STRONG_PASS"
    elif improvement >= minimum_improvement_percent:
        verdict = "PASS"
    else:
        verdict = "STATISTICAL_BUT_SMALL"

    return {
        "verdict": verdict,
        "relative_improvement_percent": float(improvement),
        "minimum_practical_improvement_percent": float(minimum_improvement_percent),
        "strong_improvement_percent": float(strong_improvement_percent),
    }

