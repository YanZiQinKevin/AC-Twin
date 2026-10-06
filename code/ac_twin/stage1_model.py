"""Stage 1 per-agent paired effect model.

``PerAgentPairedEffectGRU`` keeps the vehicle dimension through encoding and
decoding, producing a per-agent, per-time position effect of shape
``B × T_pred × N × 2``. It uses no graph, attention, slot-specific embedding or
per-slot head: every vehicle shares the same object MLP, temporal GRU, decoder
GRU and MLP head. The history and factual encoders share structure but not
parameters.
"""
from __future__ import annotations

import torch
from torch import nn

from ac_twin.stage1_data import N_FEATURES


class PerObjectTemporalEncoder(nn.Module):
    """Encode each stable entity independently with a shared object MLP + GRU.

    Missing time steps never advance a vehicle's hidden state. Returns the
    per-agent summary ``B×N×h`` and the active mask ``B×N``. A fully masked
    (e.g. M0 empty-suffix) input yields an exactly zero summary.
    """

    def __init__(self, feature_dim: int = N_FEATURES, embed_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.object_mlp = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.temporal = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        batch, steps, agents, _ = x.shape
        if mask is None:
            mask = x[..., 0] > 0
        mask = mask.bool()
        embedded = self.object_mlp(x)  # B,T,N,embed
        embedded = embedded.permute(0, 2, 1, 3).reshape(batch * agents, steps, -1)
        step_mask = mask.permute(0, 2, 1).reshape(batch * agents, steps)
        embedded = torch.where(step_mask.unsqueeze(-1), embedded, torch.zeros_like(embedded))

        state = embedded.new_zeros(1, batch * agents, self.hidden_dim)
        for step in range(steps):
            _, next_state = self.temporal(embedded[:, step : step + 1], state)
            state = torch.where(step_mask[:, step].view(1, -1, 1), next_state, state)

        summary = state[-1].reshape(batch, agents, self.hidden_dim)
        active = step_mask.reshape(batch, agents, steps).any(dim=2)
        summary = summary * active.unsqueeze(-1)
        return summary, active


def masked_scene_summary(summary: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over vehicles; padding never enters the denominator."""
    weight = active.unsqueeze(-1).to(summary.dtype)  # B,N,1
    denominator = weight.sum(dim=1).clamp_min(1.0)  # B,1
    return (summary * weight).sum(dim=1) / denominator  # B,h


class PerAgentPairedEffectGRU(nn.Module):
    """Per-agent counterfactual position-effect predictor.

    Outputs ``B × T_pred × N × 2``. The factual-suffix channel is a real input;
    for the M0 condition the caller passes an all-zero suffix tensor and mask,
    which makes the factual summaries exactly zero without deleting any module.
    """

    def __init__(
        self,
        feature_dim: int = N_FEATURES,
        action_dim: int = 5,
        object_embedding_dim: int = 64,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.history_encoder = PerObjectTemporalEncoder(feature_dim, object_embedding_dim, hidden_dim)
        self.factual_encoder = PerObjectTemporalEncoder(feature_dim, object_embedding_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        context_dim = hidden_dim * 4
        self.context_to_hidden = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.GRU(context_dim + 2 * action_embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode_factual_agents(
        self, factual_suffix: torch.Tensor, suffix_mask: torch.Tensor | None = None
    ):
        """Read-only per-agent factual summary (B,N,h) and active mask (B,N)."""
        return self.factual_encoder(factual_suffix, suffix_mask)

    def forward(
        self,
        history: torch.Tensor,
        factual_suffix: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, _, agents, _ = history.shape
        history_summary, history_active = self.history_encoder(history, history_mask)
        factual_summary, factual_active = self.factual_encoder(factual_suffix, suffix_mask)

        history_scene = masked_scene_summary(history_summary, history_active)
        factual_scene = masked_scene_summary(factual_summary, factual_active)

        hidden = self.hidden_dim
        history_scene_b = history_scene.unsqueeze(1).expand(batch, agents, hidden)
        factual_scene_b = factual_scene.unsqueeze(1).expand(batch, agents, hidden)
        agent_context = torch.cat(
            [history_summary, factual_summary, history_scene_b, factual_scene_b], dim=-1
        )  # B,N,4h

        horizon = factual_actions.shape[1]
        context_flat = agent_context.reshape(batch * agents, -1)
        initial_hidden = self.context_to_hidden(context_flat).unsqueeze(0)  # 1,B*N,h

        factual_emb = self.action_embedding(factual_actions.long())  # B,T,ae
        counterfactual_emb = self.action_embedding(counterfactual_actions.long())
        factual_emb = factual_emb.unsqueeze(1).expand(batch, agents, horizon, -1)
        counterfactual_emb = counterfactual_emb.unsqueeze(1).expand(batch, agents, horizon, -1)
        context_seq = agent_context.unsqueeze(2).expand(batch, agents, horizon, -1)

        decoder_input = torch.cat(
            [
                context_seq.reshape(batch * agents, horizon, -1),
                factual_emb.reshape(batch * agents, horizon, -1),
                counterfactual_emb.reshape(batch * agents, horizon, -1),
            ],
            dim=-1,
        )
        decoded, _ = self.decoder(decoder_input, initial_hidden)  # B*N,T,h
        effect = self.head(decoded).reshape(batch, agents, horizon, 2).permute(0, 2, 1, 3)
        # Zero outputs for history-unknown vehicles; the loss mask is separate.
        effect = effect * history_active.unsqueeze(1).unsqueeze(-1).to(effect.dtype)
        return effect
