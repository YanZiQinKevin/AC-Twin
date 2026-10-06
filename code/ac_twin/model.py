from __future__ import annotations

import torch
from torch import nn


class PairedEffectGRU(nn.Module):
    """Predict an ego counterfactual effect sequence from paired action plans.

    Both experimental arms instantiate the same modules. In the history-only arm,
    the factual suffix context is replaced by zeros, preserving parameter count.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 48,
        action_embedding_dim: int = 8,
        use_factual_suffix: bool = True,
    ) -> None:
        super().__init__()
        self.use_factual_suffix = bool(use_factual_suffix)
        self.hidden_dim = int(hidden_dim)
        self.history_encoder = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.factual_encoder = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        context_dim = hidden_dim * 2
        decoder_input_dim = context_dim + action_embedding_dim * 2
        self.context_to_hidden = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.GRU(decoder_input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        history: torch.Tensor,
        factual_suffix: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
    ) -> torch.Tensor:
        _, h_hist = self.history_encoder(history)
        hist_context = h_hist[-1]

        if self.use_factual_suffix:
            _, h_fact = self.factual_encoder(factual_suffix)
            fact_context = h_fact[-1]
        else:
            fact_context = torch.zeros_like(hist_context)

        context = torch.cat([hist_context, fact_context], dim=-1)
        horizon = factual_actions.shape[1]
        repeated_context = context.unsqueeze(1).expand(-1, horizon, -1)
        factual_action_emb = self.action_embedding(factual_actions)
        counterfactual_action_emb = self.action_embedding(counterfactual_actions)
        decoder_input = torch.cat(
            [repeated_context, factual_action_emb, counterfactual_action_emb], dim=-1
        )
        hidden0 = self.context_to_hidden(context).unsqueeze(0)
        decoded, _ = self.decoder(decoder_input, hidden0)
        return self.head(decoded)


class OneStepDeltaMLP(nn.Module):
    """A factual one-step model used to construct residual innovation proxies."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_one_hot = torch.nn.functional.one_hot(
            action.long(), num_classes=self.action_dim
        ).float()
        return self.net(torch.cat([obs, action_one_hot], dim=-1))


class ObjectTemporalEncoder(nn.Module):
    """Permutation-invariant per-frame object pooling followed by a temporal GRU."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[..., 0:1]
        h = self.node(x) * mask
        pooled = h.sum(2) / mask.sum(2).clamp_min(1.0)
        _, state = self.temporal(pooled)
        return state[-1]


class IdentityTemporalEncoder(nn.Module):
    """Shared per-object temporal encoding before masked object aggregation."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.temporal = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.combine = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[..., 0] > 0
        batch, steps, objects, features = x.shape
        sequences = x.permute(0, 2, 1, 3).reshape(batch * objects, steps, features)
        sequence_mask = mask.permute(0, 2, 1).reshape(batch * objects, steps)
        sequences = torch.where(
            sequence_mask.unsqueeze(-1), sequences, torch.zeros_like(sequences)
        )
        state = sequences.new_zeros(1, batch * objects, self.temporal.hidden_size)
        for step in range(steps):
            _, next_state = self.temporal(sequences[:, step : step + 1], state)
            state = torch.where(
                sequence_mask[:, step].view(1, -1, 1), next_state, state
            )
        encoded = state[-1].reshape(batch, objects, -1)
        active = mask.any(dim=1)
        encoded = encoded * active.unsqueeze(-1)
        ego = encoded[:, 0]
        non_ego_mask = active[:, 1:].unsqueeze(-1)
        non_ego = (encoded[:, 1:] * non_ego_mask).sum(1) / non_ego_mask.sum(1).clamp_min(1)
        return self.combine(torch.cat([ego, non_ego], dim=-1))


class GraphTemporalEncoder(nn.Module):
    """Message passing over stable vehicle identities followed by a temporal GRU."""

    def __init__(self, feature_dim: int, hidden_dim: int, radius: float = 60.0,
                 position_scale_xy: tuple[float, float] = (1.0, 1.0)) -> None:
        super().__init__()
        self.radius = float(radius)
        self.register_buffer(
            "position_scale_xy",
            torch.as_tensor(position_scale_xy, dtype=torch.float32),
        )
        self.node = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.frame = nn.Linear(2 * hidden_dim, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def adjacency(self, mask: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metric_distance(self, relative_xy: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(relative_xy * self.position_scale_xy, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[..., 0] > 0
        h = self.node(x)
        hi = h.unsqueeze(3).expand(-1, -1, -1, h.shape[2], -1)
        hj = h.unsqueeze(2).expand(-1, -1, h.shape[2], -1, -1)
        rel = x.unsqueeze(2)[..., 2:6] - x.unsqueeze(3)[..., 2:6]
        distance = self.metric_distance(rel[..., :2])
        adjacency = self.adjacency(mask, distance)
        messages = self.message(torch.cat([hi, hj, rel], dim=-1)) * adjacency.unsqueeze(-1)
        aggregate = messages.sum(3) / adjacency.sum(3, keepdim=True).clamp_min(1)
        updated = torch.relu(h + aggregate) * mask.unsqueeze(-1)
        pooled = updated.sum(2) / mask.sum(2, keepdim=True).clamp_min(1)
        frame = torch.relu(self.frame(torch.cat([updated[:, :, 0], pooled], dim=-1)))
        _, state = self.temporal(frame)
        return state[-1]


class StaticGraphTemporalEncoder(GraphTemporalEncoder):
    """Fully connected message passing with time-invariant connectivity rules."""

    def adjacency(self, mask: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        del distance
        return mask.unsqueeze(3) & mask.unsqueeze(2)


class DynamicGraphTemporalEncoder(GraphTemporalEncoder):
    """Distance-gated dynamic message passing over stable vehicle identities."""

    def adjacency(self, mask: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        return mask.unsqueeze(3) & mask.unsqueeze(2) & (distance <= self.radius)


class PairedObjectEffectGRU(nn.Module):
    """Counterfactual effect decoder with set, identity, or graph object encoding."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 48,
                 action_embedding_dim: int = 8, encoder_type: str = "set", radius: float = 60.0,
                 position_scale_xy: tuple[float, float] = (1.0, 1.0)) -> None:
        super().__init__()
        encoders = {
            "set": ObjectTemporalEncoder,
            "identity_temporal": IdentityTemporalEncoder,
            "static_graph": StaticGraphTemporalEncoder,
            "dynamic_graph": DynamicGraphTemporalEncoder,
        }
        if encoder_type not in encoders:
            raise ValueError(
                f"Unknown encoder_type {encoder_type!r}; expected one of {sorted(encoders)}"
            )
        encoder = encoders[encoder_type]
        kwargs = {} if encoder_type in {"set", "identity_temporal"} else {
            "radius": radius,
            "position_scale_xy": position_scale_xy,
        }
        self.history_encoder = encoder(feature_dim, hidden_dim, **kwargs)
        self.factual_encoder = encoder(feature_dim, hidden_dim, **kwargs)
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        context_dim = 2 * hidden_dim
        self.context_to_hidden = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Tanh())
        self.decoder = nn.GRU(context_dim + 2 * action_embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, history: torch.Tensor, factual_suffix: torch.Tensor,
                factual_actions: torch.Tensor, counterfactual_actions: torch.Tensor) -> torch.Tensor:
        context = torch.cat([self.history_encoder(history), self.factual_encoder(factual_suffix)], dim=-1)
        horizon = factual_actions.shape[1]
        repeated = context.unsqueeze(1).expand(-1, horizon, -1)
        decoder_input = torch.cat([repeated, self.action_embedding(factual_actions),
                                   self.action_embedding(counterfactual_actions)], dim=-1)
        decoded, _ = self.decoder(decoder_input, self.context_to_hidden(context).unsqueeze(0))
        return self.head(decoded)
