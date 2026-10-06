"""Evidence-value-aligned routing for Stage 3B Plus.

The oracle condition is route-only: response truth can select a state-table
weight but is never concatenated into the decoder input.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ac_twin.stage1_model import PerObjectTemporalEncoder, masked_scene_summary
from ac_twin.stage3_routing import (
    PerObjectTemporalSequenceEncoder,
    RouterDecoder,
    dual_channel_decompose,
)


CONDITIONS = frozenset(
    {"uniform_suffix", "same_information_direct", "learned_router", "route_only_oracle"}
)
ARCHITECTURES = frozenset({"state_full", "state_two_channel"})


class HorizonStateGate(nn.Module):
    """A small unconstrained horizon-by-response-state lookup table."""

    def __init__(self, horizon: int = 20, n_regimes: int = 4, n_channels: int = 1):
        super().__init__()
        self.horizon = int(horizon)
        self.n_regimes = int(n_regimes)
        self.n_channels = int(n_channels)
        self.logits = nn.Parameter(torch.zeros(horizon, n_channels, n_regimes))

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        if regime_probs.ndim != 4 or regime_probs.shape[-1] != self.n_regimes:
            raise ValueError("regime_probs must have shape B,T,N,n_regimes")
        steps = regime_probs.shape[1]
        if steps > self.horizon:
            raise ValueError("regime_probs horizon exceeds frozen state table")
        weights = torch.sigmoid(self.logits[:steps])  # T,C,R
        return torch.einsum("btnr,tcr->btnc", regime_probs, weights)

    def from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 3:
            raise ValueError("oracle labels must have shape B,T,N")
        steps = labels.shape[1]
        if steps > self.horizon:
            raise ValueError("oracle label horizon exceeds frozen state table")
        if bool(((labels < 0) | (labels >= self.n_regimes)).any()):
            raise ValueError("oracle labels outside registered response states")
        weights = torch.sigmoid(self.logits[:steps])  # T,C,R
        expanded = weights.view(1, steps, 1, self.n_channels, self.n_regimes).expand(
            labels.shape[0], steps, labels.shape[2], self.n_channels, self.n_regimes
        )
        index = labels.long().unsqueeze(-1).unsqueeze(-1).expand(
            labels.shape[0], steps, labels.shape[2], self.n_channels, 1
        )
        return torch.gather(expanded, -1, index).squeeze(-1)


def nonmonotonic_shape_loss(
    weights: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    margin: float = 0.10,
    *,
    role: str,
) -> torch.Tensor:
    """Train-only ranking loss matching Stage 2's registered state pattern."""
    if role != "train":
        raise ValueError("nonmonotonic shape loss may only use train labels")
    if weights.shape[:-1] != labels.shape or labels.shape != valid.shape:
        raise ValueError("weights, labels and valid shapes do not align")
    means: dict[int, torch.Tensor] = {}
    for regime in range(4):
        selected = valid.bool() & (labels == regime)
        if bool(selected.any()):
            means[regime] = weights[selected].mean(dim=0)

    penalties: list[torch.Tensor] = []
    positive = [means[index] for index in (1, 2, 3) if index in means]
    if 0 in means and positive:
        positive_all = torch.stack(positive).mean(dim=0)
        penalties.append(F.relu(float(margin) - (means[0] - positive_all)).mean())
    if 3 in means and 1 in means:
        penalties.append(F.relu(float(margin) - (means[3] - means[1])).mean())
    if 3 in means and 2 in means:
        penalties.append(F.relu(float(margin) - (means[3] - means[2])).mean())
    if not penalties:
        return weights.sum() * 0.0
    return torch.stack(penalties).sum()


class Stage3BPlusRouter(nn.Module):
    """One topology implementing Uniform, Direct, Learned and route-only Oracle."""

    def __init__(
        self,
        condition: str,
        architecture: str = "state_full",
        *,
        feature_dim: int = 8,
        action_dim: int = 5,
        object_embedding_dim: int = 64,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8,
        response_feature_dim: int = 32,
        horizon_embedding_dim: int = 8,
        horizon: int = 20,
    ) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown Stage 3B Plus condition: {condition}")
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown Stage 3B Plus architecture: {architecture}")
        self.condition = condition
        self.architecture = architecture
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.n_gate_channels = 1 if architecture == "state_full" else 2

        self.history_encoder = PerObjectTemporalEncoder(
            feature_dim, object_embedding_dim, hidden_dim
        )
        self.factual_sequence_encoder = PerObjectTemporalSequenceEncoder(
            feature_dim, object_embedding_dim, hidden_dim
        )
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        self.horizon_embedding = nn.Embedding(horizon, horizon_embedding_dim)
        self.state_gates = HorizonStateGate(
            horizon=horizon, n_regimes=4, n_channels=self.n_gate_channels
        )

        q_dim = response_feature_dim + 4 + 2 * action_embedding_dim + horizon_embedding_dim
        self.q_dim = int(q_dim)
        context_dim = 2 * hidden_dim
        decoder_input_dim = hidden_dim + q_dim + self.n_gate_channels + context_dim
        self.decoder = RouterDecoder(decoder_input_dim, hidden_dim, context_dim)

    def compute_state_weights(
        self, regime_probs: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        if labels is not None:
            return self.state_gates.from_labels(labels)
        return self.state_gates(regime_probs)

    def _build_q(
        self,
        response_features: torch.Tensor,
        regime_probs: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, agents, _ = response_features.shape
        factual = self.action_embedding(factual_actions.long()).unsqueeze(2).expand(
            batch, steps, agents, -1
        )
        counterfactual = self.action_embedding(counterfactual_actions.long()).unsqueeze(2).expand(
            batch, steps, agents, -1
        )
        horizon_index = torch.arange(steps, device=response_features.device)
        horizon = self.horizon_embedding(horizon_index).view(1, steps, 1, -1).expand(
            batch, steps, agents, -1
        )
        return torch.cat(
            [response_features, regime_probs, factual, counterfactual, horizon], dim=-1
        )

    def forward(
        self,
        history: torch.Tensor,
        factual_suffix: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
        response_features: torch.Tensor,
        regime_probs: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
        oracle_labels: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.condition == "route_only_oracle" and oracle_labels is None:
            raise ValueError("route-only oracle requires labels")
        if self.condition != "route_only_oracle" and oracle_labels is not None:
            raise ValueError("only route-only oracle may read response truth")

        batch, _, agents, _ = history.shape
        steps = factual_actions.shape[1]
        history_summary, active = self.history_encoder(history, history_mask)
        scene = masked_scene_summary(history_summary, active)
        h_seq, valid = self.factual_sequence_encoder(factual_suffix, suffix_mask)
        z_base, z_dyn, h_last = dual_channel_decompose(h_seq, valid)
        z_base_seq = z_base.unsqueeze(1).expand(batch, steps, agents, -1)
        z_dyn_seq = z_dyn.unsqueeze(1).expand(batch, steps, agents, -1)
        h_last_seq = h_last.unsqueeze(1).expand(batch, steps, agents, -1)

        if self.condition in {"uniform_suffix", "route_only_oracle"}:
            decoder_response = torch.zeros_like(response_features)
            decoder_probs = torch.zeros_like(regime_probs)
        else:
            decoder_response = response_features
            decoder_probs = regime_probs

        if self.condition == "uniform_suffix":
            route_gate = torch.ones(
                batch, steps, agents, self.n_gate_channels,
                device=history.device, dtype=history.dtype,
            )
            decoder_gate = route_gate
        elif self.condition == "route_only_oracle":
            route_gate = self.compute_state_weights(regime_probs, oracle_labels)
            decoder_gate = torch.ones_like(route_gate)
        else:
            route_gate = self.compute_state_weights(regime_probs)
            decoder_gate = route_gate

        if self.condition in {"uniform_suffix", "same_information_direct"}:
            z_route = h_last_seq
        elif self.architecture == "state_full":
            z_route = route_gate * h_last_seq
        else:
            z_route = route_gate[..., :1] * z_base_seq + route_gate[..., 1:] * z_dyn_seq

        q = self._build_q(
            decoder_response, decoder_probs, factual_actions, counterfactual_actions
        )
        history_context = torch.cat(
            [history_summary, scene.unsqueeze(1).expand(batch, agents, self.hidden_dim)], dim=-1
        )
        history_context_seq = history_context.unsqueeze(1).expand(
            batch, steps, agents, 2 * self.hidden_dim
        )
        decoder_input = torch.cat(
            [z_route, q, decoder_gate, history_context_seq], dim=-1
        ).permute(0, 2, 1, 3)
        output = self.decoder(decoder_input, history_context)
        output = output * active.unsqueeze(1).unsqueeze(-1).to(output.dtype)
        if not return_trace:
            return output
        trace = {
            "route_gate": route_gate,
            "decoder_gate": decoder_gate,
            "decoder_response_features": decoder_response,
            "decoder_regime_probs": decoder_probs,
            "z_base_seq": z_base_seq,
            "z_dyn_seq": z_dyn_seq,
            "h_last_seq": h_last_seq,
            "z_route": z_route,
            "decoder_input": decoder_input,
        }
        return output, trace


def audit_direct_router_fairness(
    models: Mapping[str, nn.Module], tolerance: float = 0.01
) -> dict[str, Any]:
    required = ("direct", "learned")
    if any(name not in models for name in required):
        raise ValueError("fairness audit requires direct and learned models")
    counts = {
        name: {
            "total": sum(p.numel() for p in models[name].parameters()),
            "trainable": sum(p.numel() for p in models[name].parameters() if p.requires_grad),
        }
        for name in required
    }
    total_values = [counts[name]["total"] for name in required]
    trainable_values = [counts[name]["trainable"] for name in required]
    max_total = max(total_values)
    max_trainable = max(trainable_values)
    relative = max(
        (max(total_values) - min(total_values)) / max(max_total, 1),
        (max(trainable_values) - min(trainable_values)) / max(max_trainable, 1),
    )
    return {
        "counts": counts,
        "max_relative_difference": float(relative),
        "same_state_keys": set(models["direct"].state_dict()) == set(models["learned"].state_dict()),
        "within_tolerance": bool(relative <= tolerance),
    }


def assert_oracle_route_only(trace: Mapping[str, torch.Tensor]) -> None:
    """Assert that oracle truth is absent from all decoder-visible response slots."""
    if not torch.count_nonzero(trace["decoder_response_features"]).item() == 0:
        raise AssertionError("oracle response features leaked into decoder")
    if not torch.count_nonzero(trace["decoder_regime_probs"]).item() == 0:
        raise AssertionError("oracle regime probabilities leaked into decoder")
    if not torch.equal(trace["decoder_gate"], torch.ones_like(trace["decoder_gate"])):
        raise AssertionError("oracle gate value leaked into decoder")


__all__ = [
    "ARCHITECTURES",
    "CONDITIONS",
    "HorizonStateGate",
    "Stage3BPlusRouter",
    "assert_oracle_route_only",
    "audit_direct_router_fairness",
    "nonmonotonic_shape_loss",
]
