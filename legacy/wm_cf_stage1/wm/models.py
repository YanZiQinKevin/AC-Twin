from __future__ import annotations

import torch
import torch.nn as nn


class MLPDynamics(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_agnostic: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_agnostic = action_agnostic
        in_dim = obs_dim if action_agnostic else obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor | None = None) -> torch.Tensor:
        if self.action_agnostic:
            x = obs
        else:
            if act is None:
                raise ValueError("Action tensor is required for action-conditioned model.")
            a_onehot = torch.nn.functional.one_hot(
                act.long(), num_classes=self.action_dim
            ).float()
            x = torch.cat([obs, a_onehot], dim=-1)
        return self.net(x)

