"""Train-only teacher targets for Stage 3R evidence routing."""
from __future__ import annotations

import torch


def _validate_predictions(
    full: torch.Tensor, history: torch.Tensor, target: torch.Tensor
) -> None:
    if full.shape != history.shape or full.shape != target.shape:
        raise ValueError("full, history, and target must have identical shapes")
    if full.ndim < 2:
        raise ValueError("predictions must include a final coordinate dimension")


def optimal_convex_weight(
    full: torch.Tensor,
    history: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the squared-error optimal weight on the full expert in [0, 1]."""
    _validate_predictions(full, history, target)
    difference = full - history
    residual = target - history
    numerator = (difference * residual).sum(dim=-1)
    denominator = (difference * difference).sum(dim=-1)
    return (numerator / denominator.clamp_min(float(eps))).clamp(0.0, 1.0)


def counterfactual_evidence_advantage(
    full: torch.Tensor, history: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Positive values mean the full factual-evidence expert has lower loss."""
    _validate_predictions(full, history, target)
    full_loss = ((full - target) ** 2).sum(dim=-1)
    history_loss = ((history - target) ** 2).sum(dim=-1)
    return history_loss - full_loss


def aggregate_episode_vehicle(
    values: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average E x T x N values over valid time cells to E x N."""
    if values.shape != valid.shape or values.ndim != 3:
        raise ValueError("values and valid must both have shape E,T,N")
    mask = valid.to(values.dtype)
    count = mask.sum(dim=1)
    mean = (values * mask).sum(dim=1) / count.clamp_min(1.0)
    agent_valid = count > 0
    return torch.where(agent_valid, mean, torch.zeros_like(mean)), agent_valid


def build_consensus_teacher(
    full_by_seed: torch.Tensor,
    history_by_seed: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build median five-seed CEW/CEA targets and reliability masks."""
    if full_by_seed.shape != history_by_seed.shape or full_by_seed.ndim != 5:
        raise ValueError("seed predictions must have shape S,E,T,N,D")
    if full_by_seed.shape[0] != 5:
        raise ValueError("Stage 3R consensus requires exactly five expert seeds")
    if full_by_seed.shape[1:] != target.shape:
        raise ValueError("target must match one seed prediction shape")
    if target.shape[:-1] != valid.shape:
        raise ValueError("valid must match target without coordinate dimension")
    if not bool(valid.any()):
        raise ValueError("at least one valid cell is required")

    expanded_target = target.unsqueeze(0).expand_as(full_by_seed)
    cew_by_seed = optimal_convex_weight(
        full_by_seed, history_by_seed, expanded_target
    )
    cea_by_seed = counterfactual_evidence_advantage(
        full_by_seed, history_by_seed, expanded_target
    )
    cew = cew_by_seed.median(dim=0).values
    cea = cea_by_seed.median(dim=0).values

    positive = (cea_by_seed > 0).sum(dim=0)
    negative = (cea_by_seed < 0).sum(dim=0)
    agreement = torch.maximum(positive, negative)
    rank_reliable = valid.bool() & (agreement >= 4)

    disagreement_by_seed = torch.linalg.vector_norm(
        full_by_seed - history_by_seed, dim=-1
    )
    disagreement = disagreement_by_seed.median(dim=0).values
    threshold = torch.quantile(disagreement[valid.bool()], 0.05)
    distill_weight = (
        valid.bool() & (disagreement > threshold)
    ).to(cew.dtype)

    valid_cea = cea[valid.bool()]
    lower = torch.quantile(valid_cea, 0.01)
    upper = torch.quantile(valid_cea, 0.99)
    cea_winsorized = cea.clamp(lower, upper)

    return {
        "cew_by_seed": cew_by_seed,
        "cea_by_seed": cea_by_seed,
        "cew": cew,
        "cea": cea,
        "cea_winsorized": cea_winsorized,
        "rank_reliable": rank_reliable,
        "distill_weight": distill_weight,
        "valid_for_effect": valid.bool(),
        "disagreement": disagreement,
        "disagreement_threshold": threshold,
        "winsor_bounds": torch.stack((lower, upper)),
    }


__all__ = [
    "aggregate_episode_vehicle",
    "build_consensus_teacher",
    "counterfactual_evidence_advantage",
    "optimal_convex_weight",
]
