"""Models and deterministic OOF partitioning for Stage 3R."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ac_twin.stage1_data import N_FEATURES
from ac_twin.stage1_model import PerAgentPairedEffectGRU


ExpertCondition = Literal["full", "history"]

NON_ORACLE_CONDITIONS = frozenset(
    {
        "full_expert",
        "history_expert",
        "uniform_mixture",
        "best_constant_w",
        "end_to_end_mixture",
        "cew_distilled_router",
        "same_information_direct",
        "response_only_router",
    }
)


@dataclass(frozen=True)
class OOFFold:
    index: int
    train_indices: np.ndarray
    holdout_indices: np.ndarray


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return int(sum(parameter.numel() for parameter in parameters))


class Stage3RExpert(nn.Module):
    """Matched Full/History evidence experts with an identical module graph."""

    def __init__(
        self,
        condition: ExpertCondition,
        feature_dim: int = N_FEATURES,
        action_dim: int = 5,
        object_embedding_dim: int = 64,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if condition not in {"full", "history"}:
            raise ValueError("condition must be 'full' or 'history'")
        self.condition = condition
        self.predictor = PerAgentPairedEffectGRU(
            feature_dim=feature_dim,
            action_dim=action_dim,
            object_embedding_dim=object_embedding_dim,
            hidden_dim=hidden_dim,
            action_embedding_dim=action_embedding_dim,
        )

    def forward(
        self,
        history: torch.Tensor,
        factual_suffix: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition == "history":
            factual_suffix = torch.zeros_like(factual_suffix)
            suffix_mask = torch.zeros(
                factual_suffix.shape[:-1],
                dtype=torch.bool,
                device=factual_suffix.device,
            )
        return self.predictor(
            history=history,
            factual_suffix=factual_suffix,
            factual_actions=factual_actions,
            counterfactual_actions=counterfactual_actions,
            history_mask=history_mask,
            suffix_mask=suffix_mask,
        )


class CEWRouter(nn.Module):
    """Predict a bounded per-cell weight on the full-evidence expert."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features))


class SameInformationDirect(nn.Module):
    """Predict an unconstrained residual from exactly the router features."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_router_features(
    *,
    history: torch.Tensor,
    factual_suffix: torch.Tensor,
    factual_actions: torch.Tensor,
    counterfactual_actions: torch.Tensor,
    response_features: torch.Tensor,
    regime_probs: torch.Tensor,
    full_prediction: torch.Tensor,
    history_prediction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build legal per-cell Router inputs and a capacity-matched response-only view."""
    if full_prediction.shape != history_prediction.shape or full_prediction.ndim != 4:
        raise ValueError("expert predictions must align as E,T,N,2")
    episodes, horizon, agents, _ = full_prediction.shape
    if response_features.shape[:3] != (episodes, horizon, agents):
        raise ValueError("response features must align with expert cells")
    if regime_probs.shape[:3] != (episodes, horizon, agents):
        raise ValueError("regime probabilities must align with expert cells")
    if history.shape[0] != episodes or history.shape[2] != agents:
        raise ValueError("history must align with expert episodes and agents")
    if factual_suffix.shape[0] != episodes or factual_suffix.shape[2] != agents:
        raise ValueError("factual suffix must align with expert episodes and agents")
    if factual_actions.shape != (episodes, horizon) or counterfactual_actions.shape != (
        episodes,
        horizon,
    ):
        raise ValueError("action sequences must align with expert horizon")

    response_block = torch.cat((response_features, regime_probs), dim=-1)
    response_end = int(response_block.shape[-1])
    history_last = history[:, -1].unsqueeze(1).expand(-1, horizon, -1, -1)
    suffix_mean = factual_suffix.mean(dim=1).unsqueeze(1).expand(-1, horizon, -1, -1)
    factual_one_hot = F.one_hot(factual_actions.long(), num_classes=5).to(
        full_prediction.dtype
    )
    counterfactual_one_hot = F.one_hot(
        counterfactual_actions.long(), num_classes=5
    ).to(full_prediction.dtype)
    action_block = torch.cat((factual_one_hot, counterfactual_one_hot), dim=-1)
    action_block = action_block.unsqueeze(2).expand(-1, -1, agents, -1)
    horizon_value = torch.linspace(
        0.0,
        1.0,
        horizon,
        device=full_prediction.device,
        dtype=full_prediction.dtype,
    ).view(1, horizon, 1, 1).expand(episodes, -1, agents, -1)
    difference = full_prediction - history_prediction
    prediction_block = torch.cat(
        (
            full_prediction,
            history_prediction,
            difference,
            torch.linalg.vector_norm(difference, dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    features = torch.cat(
        (
            response_block,
            history_last,
            suffix_mean,
            action_block,
            horizon_value,
            prediction_block,
        ),
        dim=-1,
    ).float()
    response_only = torch.zeros_like(features)
    response_only[..., :response_end] = features[..., :response_end]
    metadata = {
        "input_dim": int(features.shape[-1]),
        "response_slice_start": 0,
        "response_slice_end": response_end,
    }
    return features, response_only, metadata


def match_direct_hidden_width(input_dim: int, router_hidden_dim: int) -> int:
    """Choose the closest positive Direct width by exact trainable count."""
    router = CEWRouter(input_dim=input_dim, hidden_dim=router_hidden_dim)
    target = parameter_count(router, trainable_only=True)
    candidates = range(1, 4 * int(router_hidden_dim) + 1)
    best = min(
        candidates,
        key=lambda width: (
            abs(
                parameter_count(
                    SameInformationDirect(input_dim=input_dim, hidden_dim=width),
                    trainable_only=True,
                )
                - target
            ),
            width,
        ),
    )
    return int(best)


def stage3r_forward(
    condition: str,
    full_prediction: torch.Tensor,
    history_prediction: torch.Tensor,
    features: torch.Tensor,
    constant_w: float,
    *,
    router: CEWRouter | None = None,
    direct: SameInformationDirect | None = None,
    oracle_weight: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
    true_regime: torch.Tensor | None = None,
    cew: torch.Tensor | None = None,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply one frozen Stage 3R combination rule."""
    if full_prediction.shape != history_prediction.shape:
        raise ValueError("expert predictions must have identical shapes")
    if full_prediction.shape[:-1] != features.shape[:-1]:
        raise ValueError("features must align with expert cells")
    if condition in NON_ORACLE_CONDITIONS and any(
        value is not None for value in (target, true_regime, cew, oracle_weight)
    ):
        raise ValueError("truth inputs forbidden for non-oracle condition")

    base = float(constant_w) * full_prediction + (
        1.0 - float(constant_w)
    ) * history_prediction
    trace: dict[str, torch.Tensor] = {"constant_base": base}

    if condition == "full_expert":
        output = full_prediction
    elif condition == "history_expert":
        output = history_prediction
    elif condition == "uniform_mixture":
        output = 0.5 * full_prediction + 0.5 * history_prediction
    elif condition == "best_constant_w":
        output = base
    elif condition in {
        "end_to_end_mixture",
        "cew_distilled_router",
        "response_only_router",
    }:
        if router is None:
            raise ValueError("router condition requires a CEWRouter")
        weight = router(features)
        output = weight * full_prediction + (1.0 - weight) * history_prediction
        trace["weight"] = weight
    elif condition == "same_information_direct":
        if direct is None:
            raise ValueError("direct condition requires SameInformationDirect")
        residual = direct(features)
        output = base + residual
        trace["direct_residual"] = residual
    elif condition in {"soft_oracle_cew", "hard_oracle_selector"}:
        if oracle_weight is None:
            raise ValueError("oracle condition requires oracle_weight")
        weight = oracle_weight.unsqueeze(-1) if oracle_weight.ndim == 3 else oracle_weight
        output = weight * full_prediction + (1.0 - weight) * history_prediction
        trace["weight"] = weight
    else:
        raise ValueError(f"unknown Stage 3R condition: {condition}")

    if return_trace:
        return output, trace
    return output


def make_oof_folds(
    episode_seed: np.ndarray,
    split: np.ndarray,
    *,
    k: int = 5,
    entropy: int = 20260910,
) -> list[OOFFold]:
    """Partition split-0 episodes into deterministic, disjoint OOF folds."""
    episode_seed = np.asarray(episode_seed, dtype=np.int64)
    split = np.asarray(split)
    if episode_seed.ndim != 1 or split.shape != episode_seed.shape:
        raise ValueError("episode_seed and split must be aligned 1D arrays")
    if np.unique(episode_seed).size != episode_seed.size:
        raise ValueError("episode_seed must be unique")
    train_indices = np.flatnonzero(split == 0).astype(np.int64)
    if int(k) < 2 or train_indices.size < int(k):
        raise ValueError("k must be at least 2 and no larger than train episodes")

    generator = np.random.default_rng(np.random.SeedSequence(int(entropy)))
    shuffled = train_indices.copy()
    generator.shuffle(shuffled)
    held_out_parts = np.array_split(shuffled, int(k))
    folds: list[OOFFold] = []
    for index, holdout in enumerate(held_out_parts):
        train = np.setdiff1d(train_indices, holdout, assume_unique=True)
        folds.append(
            OOFFold(
                index=index,
                train_indices=train.astype(np.int64, copy=False),
                holdout_indices=np.asarray(holdout, dtype=np.int64),
            )
        )
    return folds


__all__ = [
    "CEWRouter",
    "NON_ORACLE_CONDITIONS",
    "OOFFold",
    "SameInformationDirect",
    "Stage3RExpert",
    "build_router_features",
    "make_oof_folds",
    "match_direct_hidden_width",
    "parameter_count",
    "stage3r_forward",
]
