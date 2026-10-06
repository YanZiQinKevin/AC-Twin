"""Episode-level statistics and development gates for Stage 3R."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import norm, spearmanr
from sklearn.metrics import average_precision_score


def masked_episode_ade(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[:-1] != valid.shape:
        raise ValueError("prediction, target, and valid shapes do not align")
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    mask = valid.to(distance.dtype)
    count = mask.sum(dim=(1, 2))
    if bool((count <= 0).any()):
        raise ValueError("every episode must contain at least one valid cell")
    return (distance * mask).sum(dim=(1, 2)) / count


def find_best_constant_weight(
    full: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    grid: Iterable[float] = tuple(np.arange(0.0, 1.0001, 0.05)),
) -> tuple[float, dict[float, float]]:
    scores: dict[float, float] = {}
    for raw_weight in grid:
        weight = round(float(raw_weight), 10)
        prediction = weight * full + (1.0 - weight) * history
        scores[weight] = float(masked_episode_ade(prediction, target, valid).mean())
    if not scores:
        raise ValueError("constant-weight grid cannot be empty")
    best = min(scores, key=lambda key: (scores[key], key))
    return float(best), scores


def paired_relative_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or candidate.shape != reference.shape or reference.size < 2:
        raise ValueError("paired episode errors must be aligned non-trivial vectors")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("paired episode errors must be finite")
    reference_mean = float(reference.mean())
    if reference_mean <= 0:
        raise ValueError("reference mean must be positive")
    point = 1.0 - float(candidate.mean()) / reference_mean
    generator = np.random.default_rng(int(seed))
    samples = np.empty(int(n_resamples), dtype=np.float64)
    for index in range(int(n_resamples)):
        draw = generator.integers(0, reference.size, size=reference.size)
        denominator = float(reference[draw].mean())
        samples[index] = 1.0 - float(candidate[draw].mean()) / denominator
    return {
        "point_relative": point,
        "ci80_lower": float(np.quantile(samples, 0.10)),
        "ci80_upper": float(np.quantile(samples, 0.90)),
    }


def evaluate_phase0(
    best_constant_errors_by_pair: np.ndarray,
    soft_oracle_errors_by_pair: np.ndarray,
    *,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, object]:
    reference = np.asarray(best_constant_errors_by_pair, dtype=np.float64)
    oracle = np.asarray(soft_oracle_errors_by_pair, dtype=np.float64)
    if reference.ndim != 2 or oracle.shape != reference.shape or reference.shape[0] != 5:
        raise ValueError("Phase 0 requires aligned five-pair episode errors")
    paired = paired_relative_bootstrap(
        reference.mean(axis=0),
        oracle.mean(axis=0),
        seed=seed,
        n_resamples=n_resamples,
    )
    pair_relative = 1.0 - oracle.mean(axis=1) / reference.mean(axis=1)
    positive_pairs = int((pair_relative > 0).sum())

    difficulty = (reference + oracle).mean(axis=0)
    keep = np.ones(reference.shape[1], dtype=bool)
    remove_count = max(1, int(np.ceil(reference.shape[1] * 0.01)))
    keep[np.argsort(difficulty)[-remove_count:]] = False
    deletion_relative = 1.0 - float(oracle[:, keep].mean()) / float(reference[:, keep].mean())

    passed = (
        paired["point_relative"] >= 0.10
        and paired["ci80_lower"] > 0.05
        and positive_pairs >= 4
        and deletion_relative > 0.0
    )
    return {
        **paired,
        "pair_relative": pair_relative.tolist(),
        "positive_pairs": positive_pairs,
        "top_1pct_deletion_relative": deletion_relative,
        "passed": bool(passed),
        "status": (
            "STAGE3R_SOFT_HEADROOM_CONFIRMED"
            if passed
            else "STAGE3R_SOFT_COMPLEMENTARITY_INSUFFICIENT"
        ),
    }


def weighted_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape or weight.shape != prediction.shape:
        raise ValueError("prediction, target, and weight must align")
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=float(beta))
    denominator = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denominator


def pairwise_rank_loss(
    score: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Logistic pairwise ranking loss over agents within each episode."""
    if score.shape != target.shape or valid.shape != score.shape or score.ndim != 2:
        raise ValueError("rank inputs must align as episode by agent")
    losses: list[torch.Tensor] = []
    agents = score.shape[1]
    for left in range(agents):
        for right in range(left + 1, agents):
            usable = valid[:, left] & valid[:, right]
            difference = target[:, left] - target[:, right]
            usable = usable & (difference != 0)
            if bool(usable.any()):
                direction = torch.sign(difference[usable])
                margin = score[usable, left] - score[usable, right]
                losses.append(F.softplus(-direction * margin))
    if not losses:
        return score.sum() * 0.0
    return torch.cat(losses).mean()


def _phase1_metrics(
    predicted: np.ndarray,
    teacher: np.ndarray,
    cea: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float]:
    mask = np.asarray(valid, dtype=bool)
    predicted_flat = np.asarray(predicted)[mask]
    teacher_flat = np.asarray(teacher)[mask]
    label = (np.asarray(cea)[mask] > 0).astype(np.int8)
    correlation = float(spearmanr(predicted_flat, teacher_flat).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    prevalence = float(label.mean())
    if label.min() == label.max():
        pr_auc = prevalence
    else:
        pr_auc = float(average_precision_score(label, predicted_flat))
    return correlation, pr_auc - prevalence, prevalence


def evaluate_phase1(
    predicted_w_by_pair: np.ndarray,
    teacher_cew: np.ndarray,
    teacher_cea: np.ndarray,
    valid: np.ndarray,
    best_constant_errors_by_pair: np.ndarray,
    router_errors_by_pair: np.ndarray,
    *,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, object]:
    predicted = np.asarray(predicted_w_by_pair, dtype=np.float64)
    teacher = np.asarray(teacher_cew, dtype=np.float64)
    cea = np.asarray(teacher_cea, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    reference = np.asarray(best_constant_errors_by_pair, dtype=np.float64)
    router = np.asarray(router_errors_by_pair, dtype=np.float64)
    if predicted.ndim != 4 or predicted.shape[0] != 5:
        raise ValueError("Phase 1 requires five-pair E,T,N weights")
    if predicted.shape[1:] != teacher.shape or teacher.shape != cea.shape or mask.shape != teacher.shape:
        raise ValueError("teacher and valid arrays must align with predicted weights")
    if reference.shape != router.shape or reference.shape != (5, teacher.shape[0]):
        raise ValueError("episode errors must align as five pairs by episode")

    per_pair = [
        _phase1_metrics(predicted[pair], teacher, cea, mask) for pair in range(5)
    ]
    correlations = np.asarray([item[0] for item in per_pair])
    pr_improvements = np.asarray([item[1] for item in per_pair])
    positive_pairs = int(((correlations > 0) & (pr_improvements > 0)).sum())

    generator = np.random.default_rng(int(seed))
    correlation_samples = np.empty(int(n_resamples), dtype=np.float64)
    pr_samples = np.empty(int(n_resamples), dtype=np.float64)
    episode_count = teacher.shape[0]
    for sample_index in range(int(n_resamples)):
        draw = generator.integers(0, episode_count, size=episode_count)
        metrics = [
            _phase1_metrics(
                predicted[pair, draw], teacher[draw], cea[draw], mask[draw]
            )
            for pair in range(5)
        ]
        correlation_samples[sample_index] = np.mean([item[0] for item in metrics])
        pr_samples[sample_index] = np.mean([item[1] for item in metrics])

    relative = 1.0 - router.mean(axis=1) / reference.mean(axis=1)
    router_improvement = float(1.0 - router.mean() / reference.mean())
    passed = (
        float(np.quantile(correlation_samples, 0.10)) > 0
        and float(np.quantile(pr_samples, 0.10)) > 0
        and positive_pairs >= 4
        and router_improvement > 0
    )
    valid_weights = predicted[:, mask]
    valid_teacher = np.broadcast_to(teacher, predicted.shape)[:, mask]
    bins = [0.0, 0.1, 0.4, 0.6, 0.9, 1.0000001]
    counts, _ = np.histogram(valid_weights, bins=bins)
    quantile_edges = np.quantile(valid_weights, np.linspace(0.0, 1.0, 11))
    calibration = []
    for index in range(10):
        lower, upper = quantile_edges[index], quantile_edges[index + 1]
        selected = (valid_weights >= lower) & (
            valid_weights <= upper if index == 9 else valid_weights < upper
        )
        if selected.any():
            calibration.append(
                {
                    "predicted_mean": float(valid_weights[selected].mean()),
                    "teacher_mean": float(valid_teacher[selected].mean()),
                    "fraction": float(selected.mean()),
                }
            )
    return {
        "status": (
            "STAGE3R_VALUE_WEIGHT_PREDICTABLE"
            if passed
            else "STAGE3R_VALUE_WEIGHT_NOT_PREDICTABLE"
        ),
        "passed": bool(passed),
        "spearman_mean": float(correlations.mean()),
        "spearman_ci80_lower": float(np.quantile(correlation_samples, 0.10)),
        "spearman_ci80_upper": float(np.quantile(correlation_samples, 0.90)),
        "pr_auc_minus_prevalence_mean": float(pr_improvements.mean()),
        "pr_auc_minus_prevalence_ci80_lower": float(np.quantile(pr_samples, 0.10)),
        "pr_auc_minus_prevalence_ci80_upper": float(np.quantile(pr_samples, 0.90)),
        "prevalence_mean": float(np.mean([item[2] for item in per_pair])),
        "positive_pairs": positive_pairs,
        "per_pair_spearman": correlations.tolist(),
        "per_pair_pr_auc_minus_prevalence": pr_improvements.tolist(),
        "router_relative_vs_best_constant": router_improvement,
        "per_pair_router_relative": relative.tolist(),
        "weight_mean": float(valid_weights.mean()),
        "weight_std": float(valid_weights.std()),
        "weight_bin_fraction": (counts / counts.sum()).tolist(),
        "cew_mae": float(np.abs(valid_weights - valid_teacher).mean()),
        "calibration_deciles": calibration,
    }


def estimate_joint_power(
    reference_errors: np.ndarray,
    candidate_errors: np.ndarray,
    *,
    n_candidates: Iterable[int],
    mdes: float,
    target_power: float,
    seed: int,
    simulations: int = 1000,
) -> dict[str, object]:
    """Estimate joint clustered power for all registered candidate-vs-baseline contrasts."""
    reference = np.asarray(reference_errors, dtype=np.float64)
    candidate = np.asarray(candidate_errors, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[1] != 5:
        raise ValueError("power inputs must align as contrast by five pairs by episode")
    if not (0 < float(mdes) < 1) or not (0 < float(target_power) < 1):
        raise ValueError("MDES and target power must lie in (0,1)")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("power errors must be finite")
    residual = candidate - reference
    residual = residual - residual.mean(axis=2, keepdims=True)
    shifted_candidate = reference * (1.0 - float(mdes)) + residual
    generator = np.random.default_rng(int(seed))
    z_value = float(norm.ppf(0.90))
    powers: dict[str, float] = {}
    selected: int | None = None
    episode_count = reference.shape[2]
    for raw_n in n_candidates:
        n = int(raw_n)
        successes = 0
        for _ in range(int(simulations)):
            draw = generator.integers(0, episode_count, size=n)
            reference_draw = reference[:, :, draw].mean(axis=1)
            candidate_draw = shifted_candidate[:, :, draw].mean(axis=1)
            difference = reference_draw - candidate_draw
            mean_difference = difference.mean(axis=1)
            standard_error = difference.std(axis=1, ddof=1) / np.sqrt(n)
            lower = mean_difference - z_value * standard_error
            if bool((lower > 0).all()):
                successes += 1
        power = successes / float(simulations)
        powers[str(n)] = power
        if selected is None and power >= float(target_power):
            selected = n
    return {
        "status": (
            "STAGE3R_RSCREEN_POWER_FEASIBLE"
            if selected is not None
            else "STAGE3R_RSCREEN_POWER_INFEASIBLE"
        ),
        "selected_episode_count": selected,
        "joint_power_by_n": powers,
        "mdes": float(mdes),
        "target_power": float(target_power),
        "simulations": int(simulations),
    }


__all__ = [
    "evaluate_phase0",
    "evaluate_phase1",
    "estimate_joint_power",
    "find_best_constant_weight",
    "masked_episode_ade",
    "paired_relative_bootstrap",
    "pairwise_rank_loss",
    "weighted_huber",
]
