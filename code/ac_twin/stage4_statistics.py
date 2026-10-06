"""Episode-level risk-coverage statistics and the Stage 4A Gate RISK_dev.

All statistics here are EPISODE-level (plan section 5.1): the statistical unit is
always the episode, never a vehicle or a time point. A risk score is a per-episode
scalar where a HIGHER value means the prediction is more likely to be wrong.

Definitions (plan section 8.1), all computed on a set of episodes with per-episode
true effect error ``episode_error`` and per-episode ``risk_score``:

- selective risk ``R(k)`` = mean error of the ``k`` lowest-risk episodes;
- ``AURC`` = ``(1/E) * sum_k R(k)`` (area under the risk-coverage curve; lower is
  better — a perfect ranking concentrates the largest errors in the last-covered
  episodes);
- ``ADE@coverage`` = ``R(k*)`` at the discrete index ``k* = round(coverage*E)``;
- ``PR-AUC(top-error-20%) - prevalence`` with the top-20% threshold FROZEN on
  risk-train and applied unchanged to risk-validation and the development-screen;
- paired improvements ``I_AURC`` / ``I_C80`` relative to the strongest standard
  baseline, with episode-level paired-bootstrap intervals.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.metrics import average_precision_score


# --- risk-coverage curve --------------------------------------------------------


def _validate_pair(risk_score: np.ndarray, episode_error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    risk = np.asarray(risk_score, dtype=np.float64).ravel()
    error = np.asarray(episode_error, dtype=np.float64).ravel()
    if risk.shape != error.shape or risk.size < 2:
        raise ValueError("risk_score and episode_error must be aligned vectors with >=2 episodes")
    if not np.isfinite(risk).all() or not np.isfinite(error).all():
        raise ValueError("risk_score and episode_error must be finite")
    return risk, error


def risk_coverage_curve(risk_score: np.ndarray, episode_error: np.ndarray) -> dict[str, np.ndarray]:
    """Selective-risk curve keeping the LOWEST-risk episodes first.

    Returns ``coverage`` (k/E for k=1..E), ``selective_error`` (R(k)) and the
    ascending-risk ``order`` (stable, so ties keep the original episode order).
    """
    risk, error = _validate_pair(risk_score, episode_error)
    episodes = risk.size
    order = np.argsort(risk, kind="stable")  # ascending risk: safest first
    sorted_error = error[order]
    cumulative = np.cumsum(sorted_error)
    counts = np.arange(1, episodes + 1, dtype=np.float64)
    return {
        "coverage": counts / episodes,
        "selective_error": cumulative / counts,
        "order": order.astype(np.int64),
    }


def aurc(risk_score: np.ndarray, episode_error: np.ndarray) -> float:
    """Area under the risk-coverage curve: ``(1/E) * sum_k R(k)`` (lower is better)."""
    curve = risk_coverage_curve(risk_score, episode_error)
    return float(curve["selective_error"].mean())


def coverage_index(coverage: float, episodes: int) -> int:
    """Discrete kept-count for a coverage level: ``round(coverage*E)`` in ``[1,E]``."""
    episodes = int(episodes)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not (0.0 < float(coverage) <= 1.0):
        raise ValueError("coverage must lie in (0, 1]")
    index = int(round(float(coverage) * episodes))
    return int(min(max(index, 1), episodes))


def ade_at_coverage(risk_score: np.ndarray, episode_error: np.ndarray, coverage: float) -> float:
    """Selective error (ADE) at a fixed coverage level (lower is better)."""
    risk, error = _validate_pair(risk_score, episode_error)
    k = coverage_index(coverage, risk.size)
    order = np.argsort(risk, kind="stable")
    return float(error[order][:k].mean())


def top_error_threshold(train_error: np.ndarray, fraction: float) -> float:
    """Frozen top-``fraction`` error threshold from RISK-TRAIN errors only."""
    train = np.asarray(train_error, dtype=np.float64).ravel()
    if train.size < 2 or not np.isfinite(train).all():
        raise ValueError("top-error threshold needs a finite risk-train error vector")
    if not (0.0 < float(fraction) < 1.0):
        raise ValueError("top-error fraction must lie in (0,1)")
    return float(np.quantile(train, 1.0 - float(fraction)))


def pr_auc_minus_prevalence(
    risk_score: np.ndarray, episode_error: np.ndarray, threshold: float
) -> dict[str, float]:
    """PR-AUC for the frozen top-error label minus its prevalence (higher is better)."""
    risk, error = _validate_pair(risk_score, episode_error)
    label = (error >= float(threshold)).astype(np.int64)
    prevalence = float(label.mean())
    if label.min() == label.max():
        # Degenerate (all-positive or all-negative) eval set: PR-AUC == prevalence.
        return {"pr_auc": prevalence, "prevalence": prevalence, "pr_auc_minus_prevalence": 0.0}
    pr_auc = float(average_precision_score(label, risk))
    return {
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "pr_auc_minus_prevalence": pr_auc - prevalence,
    }


# --- candidate report -------------------------------------------------------------


def candidate_report(
    risk_score: np.ndarray,
    episode_error: np.ndarray,
    *,
    top_error_threshold_value: float,
    coverage_levels: tuple[float, ...] = (0.80, 0.90),
) -> dict[str, Any]:
    """Full per-candidate risk report on one evaluation set (frozen threshold)."""
    risk, error = _validate_pair(risk_score, episode_error)
    report: dict[str, Any] = {
        "aurc": aurc(risk, error),
        "n_episodes": int(risk.size),
    }
    for level in coverage_levels:
        report[f"ade@{int(round(level * 100))}"] = ade_at_coverage(risk, error, level)
    report.update(
        pr_auc_minus_prevalence(risk, error, top_error_threshold_value)
    )
    return report


# --- paired bootstrap for improvements -------------------------------------------


def _improvement_relative(standard: float, candidate: float) -> float:
    if standard <= 0:
        return float("nan")
    return float((standard - candidate) / standard)


def paired_improvement_bootstrap(
    candidate_score: np.ndarray,
    standard_score: np.ndarray,
    episode_error: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.80,
) -> dict[str, Any]:
    """Episode-level paired bootstrap for I_AURC and I_C80 vs a standard baseline.

    Both candidates are scored on the SAME resampled episodes each draw, so the
    improvement interval accounts for the shared episode sampling.
    """
    cand, error = _validate_pair(candidate_score, episode_error)
    stan, _ = _validate_pair(standard_score, episode_error)
    episodes = error.size
    generator = np.random.default_rng(int(seed))
    i_aurc = np.empty(int(n_resamples), dtype=np.float64)
    i_c80 = np.empty(int(n_resamples), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, episodes, size=episodes)
        e = error[index]
        a_std, a_cnd = aurc(stan[index], e), aurc(cand[index], e)
        c_std = ade_at_coverage(stan[index], e, 0.80)
        c_cnd = ade_at_coverage(cand[index], e, 0.80)
        i_aurc[draw] = _improvement_relative(a_std, a_cnd)
        i_c80[draw] = _improvement_relative(c_std, c_cnd)
    alpha = 1.0 - float(confidence)
    point_aurc = _improvement_relative(aurc(stan, error), aurc(cand, error))
    point_c80 = _improvement_relative(
        ade_at_coverage(stan, error, 0.80), ade_at_coverage(cand, error, 0.80)
    )
    return {
        "i_aurc": point_aurc,
        "i_aurc_ci_lower": float(np.quantile(i_aurc, alpha / 2.0)),
        "i_aurc_ci_upper": float(np.quantile(i_aurc, 1.0 - alpha / 2.0)),
        "i_c80": point_c80,
        "i_c80_ci_lower": float(np.quantile(i_c80, alpha / 2.0)),
        "i_c80_ci_upper": float(np.quantile(i_c80, 1.0 - alpha / 2.0)),
        "n_resamples": int(n_resamples),
        "confidence_level": float(confidence),
        "bootstrap_seed": int(seed),
    }


# --- Gate RISK_dev -----------------------------------------------------------------


def evaluate_stage4a_gate(
    finalist_screen: Mapping[str, Any],
    *,
    pair_i_aurc: np.ndarray,
    stratified_reversal: bool,
    qa_passed: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the six mechanical development-screen continue conditions (plan §9.2).

    ``finalist_screen`` carries the finalist's development-screen ``i_aurc`` /
    ``i_c80`` (+ CI lowers) and ``pr_auc_minus_prevalence``; ``pair_i_aurc`` is the
    five per-pair I_AURC values of the SAME finalist risk score against each pair's
    errors. Returns the verdict and the mechanical terminal state.
    """
    i_aurc_min = float(config.get("gate_i_aurc_min", 0.02))
    i_c80_min = float(config.get("gate_i_c80_min", 0.02))
    min_pairs = int(config.get("gate_min_positive_pairs", 4))

    cond_aurc = (
        float(finalist_screen["i_aurc"]) >= i_aurc_min
        and float(finalist_screen["i_aurc_ci_lower"]) > 0.0
    )
    cond_c80 = (
        float(finalist_screen["i_c80"]) >= i_c80_min
        and float(finalist_screen["i_c80_ci_lower"]) > 0.0
    )
    cond_pr = float(finalist_screen["pr_auc_minus_prevalence"]) > 0.0
    pair_values = np.asarray(pair_i_aurc, dtype=np.float64)
    cond_pairs = int((pair_values > 0).sum()) >= min_pairs
    cond_strata = not bool(stratified_reversal)
    cond_qa = bool(qa_passed)

    conditions = {
        "i_aurc_ge_0.02_and_ci_lower_gt0": bool(cond_aurc),
        "i_c80_ge_0.02_and_ci_lower_gt0": bool(cond_c80),
        "pr_auc_minus_prevalence_gt0": bool(cond_pr),
        "at_least_4_of_5_pairs_positive": bool(cond_pairs),
        "no_stratified_reversal": bool(cond_strata),
        "qa_passed": bool(cond_qa),
    }
    feasible = all(conditions.values())
    return {
        "conditions": conditions,
        "positive_pairs": int((pair_values > 0).sum()),
        "per_pair_i_aurc": pair_values.tolist(),
        "passed": bool(feasible),
        "status": STATUS_FEASIBLE if feasible else STATUS_INSUFFICIENT,
    }


# Imported lazily at the bottom to avoid a hard import cycle in tests that only
# need the pure statistics above.
from ac_twin.stage4_protocol import STATUS_FEASIBLE, STATUS_INSUFFICIENT  # noqa: E402

__all__ = [
    "ade_at_coverage",
    "aurc",
    "candidate_report",
    "coverage_index",
    "evaluate_stage4a_gate",
    "paired_improvement_bootstrap",
    "pr_auc_minus_prevalence",
    "risk_coverage_curve",
    "top_error_threshold",
]
