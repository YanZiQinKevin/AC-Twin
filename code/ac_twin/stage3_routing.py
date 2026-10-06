"""Stage 3B non-monotonic evidence routing and the strong Direct baseline.

The factual suffix is split into an operational two-channel decomposition:

- ``z_base_i = masked_mean_s(h^F_{i,s})``            (persistent mean summary)
- ``z_dyn_i  = h^F_{i,last_valid} - z_base_i``       (terminal residual)

so that ``z_base + z_dyn == h^F_{i,last_valid}`` reconstructs exactly. These are
operational names only: ``z_base`` may still carry time-varying reaction
information and the decomposition is NOT an identified driving-style /
causal-response split.

The Dual router and the Same-information Direct baseline share the same
factual/history/action backbone, the same gate ``G``, the same decoder
topology, the same response features, the same training episodes and the same
batch/epoch/early-stopping rule. The ONLY structural difference under test is
whether the gate ``w`` explicitly multiplies ``z_dyn``:

- Dual:   ``z = z_base + w * z_dyn``
- Direct: ``z = z_base + z_dyn``  (``w`` still reaches the decoder)

The router is free to learn a non-monotonic weight; no ``w = 1 - g`` or any
monotonically decreasing hard constraint is imposed. No learned condition reads
oracle response truth.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ac_twin.stage1_data import N_FEATURES
from ac_twin.stage1_model import masked_scene_summary
from ac_twin.stage3_data import N_REGIMES


# --------------------------------------------------------------------------- #
# Frozen predictor/router seed pairs (one-to-one; no best-seed, no 5x5 cross)
# --------------------------------------------------------------------------- #
PREDICTOR_SEEDS = (101, 202, 303, 404, 505)
ROUTER_SEEDS = (1101, 1202, 1303, 1404, 1505)
SEED_PAIRS = tuple(zip(PREDICTOR_SEEDS, ROUTER_SEEDS))
CROSSFIT_SALT = "AC-TWIN-STAGE3-CROSSFIT"


def validate_seed_pairing(pairs: tuple[tuple[int, int], ...] = SEED_PAIRS) -> None:
    """Enforce the frozen one-to-one predictor<->router pairing."""
    if len(pairs) != len(PREDICTOR_SEEDS):
        raise ValueError("seed pairing must have exactly five pairs")
    predictors = tuple(p for p, _ in pairs)
    routers = tuple(r for _, r in pairs)
    if predictors != PREDICTOR_SEEDS:
        raise ValueError("predictor seeds changed or reordered (best-seed selection is forbidden)")
    if routers != ROUTER_SEEDS:
        raise ValueError("router seeds changed or reordered")
    if len(set(predictors)) != 5 or len(set(routers)) != 5:
        raise ValueError("seed pairing must be one-to-one (a 5x5 cross is forbidden)")


class Condition(str, Enum):
    NO_SUFFIX = "no_suffix"
    UNIFORM_SUFFIX = "uniform_suffix"
    SHUFFLED_SUFFIX = "shuffled_suffix"
    DIRECT = "same_information_direct"
    RESPONSE_HEAD_NO_ROUTING = "response_head_no_routing"
    RANDOM_GATE = "random_gate"
    STATIC_GATE = "static_gate"
    WHOLE_SUFFIX_SOFT_GATE = "whole_suffix_soft_gate"
    ORACLE_REFERENCE = "oracle_response_reference"
    DUAL_ROUTER = "dual_channel_learned_router"


# Conditions whose learned forward must never read oracle response truth.
LEARNED_CONDITIONS = frozenset(
    {
        Condition.NO_SUFFIX,
        Condition.UNIFORM_SUFFIX,
        Condition.SHUFFLED_SUFFIX,
        Condition.DIRECT,
        Condition.RESPONSE_HEAD_NO_ROUTING,
        Condition.RANDOM_GATE,
        Condition.STATIC_GATE,
        Condition.WHOLE_SUFFIX_SOFT_GATE,
        Condition.DUAL_ROUTER,
    }
)
# The three conditions under the +/-1% capacity hard audit.
CAPACITY_AUDIT_CONDITIONS = (
    Condition.UNIFORM_SUFFIX,
    Condition.DIRECT,
    Condition.DUAL_ROUTER,
)


# --------------------------------------------------------------------------- #
# Per-timestep factual sequence encoder
# --------------------------------------------------------------------------- #
class PerObjectTemporalSequenceEncoder(nn.Module):
    """Encode each stable entity, returning the full per-timestep hidden sequence.

    Mirrors :class:`PerObjectTemporalEncoder` but returns ``h^F_{i,s}`` for every
    suffix step ``s`` (shape ``B×T×N×h``) plus the per-step valid mask, which the
    dual-channel decomposition needs. Masked steps carry the previous hidden
    state forward but are excluded from the mean by the valid mask.
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
        embedded = self.object_mlp(x)
        embedded = embedded.permute(0, 2, 1, 3).reshape(batch * agents, steps, -1)
        step_mask = mask.permute(0, 2, 1).reshape(batch * agents, steps)
        embedded = torch.where(step_mask.unsqueeze(-1), embedded, torch.zeros_like(embedded))

        state = embedded.new_zeros(1, batch * agents, self.hidden_dim)
        hidden_steps = []
        for step in range(steps):
            _, next_state = self.temporal(embedded[:, step : step + 1], state)
            state = torch.where(step_mask[:, step].view(1, -1, 1), next_state, state)
            hidden_steps.append(state[-1])
        sequence = torch.stack(hidden_steps, dim=1).reshape(batch, agents, steps, self.hidden_dim)
        sequence = sequence.permute(0, 2, 1, 3)  # B,T,N,h
        valid = step_mask.reshape(batch, agents, steps).permute(0, 2, 1)  # B,T,N
        return sequence, valid


def dual_channel_decompose(
    h_seq: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the factual hidden sequence into (z_base, z_dyn, h_last_valid).

    ``h_seq``: ``B×T×N×h``; ``valid``: ``B×T×N`` bool. Returns three ``B×N×h``
    tensors with ``z_base + z_dyn == h_last_valid`` by construction.
    """
    valid_f = valid.unsqueeze(-1).to(h_seq.dtype)  # B,T,N,1
    count = valid_f.sum(dim=1).clamp_min(1.0)  # B,N,1
    z_base = (h_seq * valid_f).sum(dim=1) / count  # B,N,h

    # last valid hidden state per vehicle
    batch, steps, agents, hidden = h_seq.shape
    index = torch.arange(steps, device=h_seq.device).view(1, steps, 1, 1)
    last_index = (index * valid.unsqueeze(-1)).amax(dim=1)  # B,N,1 (0 when none valid)
    has_valid = valid.any(dim=1)  # B,N
    gather_index = last_index.clamp_min(0).view(batch, 1, agents, 1).expand(batch, 1, agents, hidden)
    h_last = torch.gather(h_seq, 1, gather_index).squeeze(1)  # B,N,h
    h_last = h_last * has_valid.unsqueeze(-1).to(h_seq.dtype)
    z_base = z_base * has_valid.unsqueeze(-1).to(h_seq.dtype)
    z_dyn = h_last - z_base
    return z_base, z_dyn, h_last


# --------------------------------------------------------------------------- #
# Gate and decoder
# --------------------------------------------------------------------------- #
class GateHead(nn.Module):
    """Scalar gate ``w = sigmoid(G(q))``. Free to be non-monotonic."""

    def __init__(self, q_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(q))  # ...,1


class RouterDecoder(nn.Module):
    """Shared per-agent effect decoder. Outputs ``B×T×N×2``."""

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_to_hidden = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, seq_input: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, agents, horizon, _ = seq_input.shape
        initial_hidden = self.context_to_hidden(
            context.reshape(batch * agents, -1)
        ).unsqueeze(0)
        decoded, _ = self.decoder(seq_input.reshape(batch * agents, horizon, -1), initial_hidden)
        effect = self.head(decoded).reshape(batch, agents, horizon, 2).permute(0, 2, 1, 3)
        return effect


class Stage3Router(nn.Module):
    """Unified router whose forward path realises one of the ten conditions.

    All conditions instantiate the identical module set (backbone, gate head and
    decoder), so Uniform/Direct/Dual have exactly equal parameter counts with no
    dummy parameters; the condition selects the ``z`` combination rule and the
    information allowed into the routing input ``q``.
    """

    def __init__(
        self,
        condition: Condition | str,
        feature_dim: int = N_FEATURES,
        action_dim: int = 5,
        object_embedding_dim: int = 64,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8,
        response_feature_dim: int = 32,
        horizon_embedding_dim: int = 8,
        gate_hidden_dim: int = 32,
        horizon: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.condition = Condition(condition)
        self.hidden_dim = int(hidden_dim)
        self.response_feature_dim = int(response_feature_dim)
        self.horizon = int(horizon)
        self._seed = int(seed)

        from ac_twin.stage1_model import PerObjectTemporalEncoder

        self.history_encoder = PerObjectTemporalEncoder(feature_dim, object_embedding_dim, hidden_dim)
        self.factual_sequence_encoder = PerObjectTemporalSequenceEncoder(
            feature_dim, object_embedding_dim, hidden_dim
        )
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        self.horizon_embedding = nn.Embedding(horizon, horizon_embedding_dim)

        q_dim = response_feature_dim + N_REGIMES + 2 * action_embedding_dim + horizon_embedding_dim
        self.q_dim = q_dim
        self.gate = GateHead(q_dim, gate_hidden_dim)

        # Decoder per-time input: z_route(h) + q(q_dim) + w(1) + history_context(2h)
        context_dim = hidden_dim * 2
        decoder_input_dim = hidden_dim + q_dim + 1 + context_dim
        self.decoder = RouterDecoder(decoder_input_dim, hidden_dim, context_dim)

    # -- routing input ------------------------------------------------------
    def build_q(
        self,
        response_features: torch.Tensor,
        regime_probs: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
    ) -> torch.Tensor:
        """q = concat(response_features, regime_probs, action_pair, horizon_emb).

        For conditions that may not read response information, the caller passes
        zeroed ``response_features``/``regime_probs`` (same shapes, same params).
        """
        batch, horizon, agents, _ = response_features.shape
        f_emb = self.action_embedding(factual_actions.long())  # B,T,ae
        cf_emb = self.action_embedding(counterfactual_actions.long())
        f_emb = f_emb.unsqueeze(2).expand(batch, horizon, agents, -1)
        cf_emb = cf_emb.unsqueeze(2).expand(batch, horizon, agents, -1)
        horizon_index = torch.arange(horizon, device=response_features.device)
        h_emb = self.horizon_embedding(horizon_index)  # T,he
        h_emb = h_emb.view(1, horizon, 1, -1).expand(batch, horizon, agents, -1)
        return torch.cat([response_features, regime_probs, f_emb, cf_emb, h_emb], dim=-1)

    # -- gate ----------------------------------------------------------------
    def compute_gate(
        self,
        q: torch.Tensor,
        oracle_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return w (B,T,N,1) per the condition's gate rule."""
        if self.condition == Condition.RANDOM_GATE:
            generator = torch.Generator(device=q.device).manual_seed(self._seed)
            return torch.rand(q.shape[:-1] + (1,), generator=generator, device=q.device)
        if self.condition == Condition.STATIC_GATE:
            # Simple frozen rule: gate decreases with horizon-normalised action magnitude
            # summary encoded in the last q block; deterministic and non-learned.
            return torch.full(q.shape[:-1] + (1,), 0.5, device=q.device)
        if self.condition == Condition.ORACLE_REFERENCE:
            if oracle_gate is None:
                raise ValueError("oracle reference requires the oracle gate input")
            return oracle_gate
        if self.condition in (Condition.UNIFORM_SUFFIX, Condition.NO_SUFFIX, Condition.RESPONSE_HEAD_NO_ROUTING):
            # Uniform evidence application: w pinned to 1 (no response gating).
            return torch.ones(q.shape[:-1] + (1,), device=q.device)
        # DIRECT and DUAL learn w from q via the shared gate head.
        return self.gate(q)

    # -- forward --------------------------------------------------------------
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
        oracle_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the per-agent effect prediction ``B×T×N×2``."""
        if self.condition in LEARNED_CONDITIONS and oracle_gate is not None:
            raise ValueError(f"{self.condition.value} is a learned condition and must not read oracle truth")

        batch, _, agents, _ = history.shape
        horizon = factual_actions.shape[1]
        history_summary, history_active = self.history_encoder(history, history_mask)
        history_scene = masked_scene_summary(history_summary, history_active)

        h_seq, seq_valid = self.factual_sequence_encoder(factual_suffix, suffix_mask)
        z_base, z_dyn, h_last = dual_channel_decompose(h_seq, seq_valid)

        # Conditions that may not read response info receive zeroed features.
        if self.condition in (Condition.UNIFORM_SUFFIX, Condition.NO_SUFFIX):
            response_features = torch.zeros_like(response_features)
            regime_probs = torch.zeros_like(regime_probs)

        q = self.build_q(response_features, regime_probs, factual_actions, counterfactual_actions)
        w = self.compute_gate(q, oracle_gate=oracle_gate)  # B,T,N,1

        # z_route per the condition's combination rule.
        if self.condition == Condition.NO_SUFFIX:
            z_route = torch.zeros(
                batch, horizon, agents, self.hidden_dim, device=history.device
            )
        elif self.condition == Condition.DUAL_ROUTER:
            z_route = z_base.unsqueeze(1) + w * z_dyn.unsqueeze(1)
        elif self.condition == Condition.WHOLE_SUFFIX_SOFT_GATE:
            z_route = w * (z_base.unsqueeze(1) + z_dyn.unsqueeze(1))
        elif self.condition in (
            Condition.RANDOM_GATE,
            Condition.STATIC_GATE,
            Condition.ORACLE_REFERENCE,
        ):
            z_route = z_base.unsqueeze(1) + w * z_dyn.unsqueeze(1)
        else:
            # UNIFORM_SUFFIX, SHUFFLED_SUFFIX, DIRECT, RESPONSE_HEAD_NO_ROUTING:
            # full suffix (z_base + z_dyn); w reaches the decoder but is not
            # multiplied into the evidence.
            z_route = (z_base + z_dyn).unsqueeze(1).expand(batch, horizon, agents, self.hidden_dim)

        history_context = torch.cat(
            [
                history_summary,
                history_scene.unsqueeze(1).expand(batch, agents, self.hidden_dim),
            ],
            dim=-1,
        )  # B,N,2h
        history_context_seq = history_context.unsqueeze(1).expand(
            batch, horizon, agents, 2 * self.hidden_dim
        )

        decoder_input = torch.cat([z_route, q, w, history_context_seq], dim=-1)
        decoder_input = decoder_input.permute(0, 2, 1, 3)  # B,N,T,-1
        effect = self.decoder(decoder_input, history_context)
        effect = effect * history_active.unsqueeze(1).unsqueeze(-1).to(effect.dtype)
        return effect


# --------------------------------------------------------------------------- #
# Capacity / parameter audit
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def audit_capacity(
    models: Mapping[Condition, nn.Module],
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Check Uniform/Direct/Dual parameter counts differ by at most +/-1%.

    All three share the identical module set, so the counts are exactly equal
    (0% difference) and no dummy parameters exist. Returns a report with the
    three counts and the pairwise relative differences.
    """
    counts = {c.value: count_parameters(models[c]) for c in CAPACITY_AUDIT_CONDITIONS}
    totals = [counts[c.value]["total"] for c in CAPACITY_AUDIT_CONDITIONS]
    reference = totals[0]
    max_rel_diff = max(abs(t - reference) / max(reference, 1) for t in totals)
    return {
        "counts": counts,
        "max_relative_difference": float(max_rel_diff),
        "within_tolerance": bool(max_rel_diff <= tolerance),
        "tolerance": float(tolerance),
    }


# --------------------------------------------------------------------------- #
# Cross-fit fold assignment (2-fold, within cf_action x cf_duration strata)
# --------------------------------------------------------------------------- #
def _crossfit_rank(episode_seed: int) -> str:
    return hashlib.sha256(f"{CROSSFIT_SALT}{int(episode_seed)}".encode("utf-8")).hexdigest()


def assign_crossfit_folds(
    cf_action: np.ndarray,
    cf_duration: np.ndarray,
    episode_seed: np.ndarray,
) -> np.ndarray:
    """Assign each episode to fold 0/1 within its cf_action x cf_duration stratum.

    Episodes are ordered by ``SHA256(CROSSFIT_SALT + episode_seed)`` and dealt
    alternately into fold 0/1, giving a fixed, reproducible 2-fold split with no
    episode overlap between folds.
    """
    cf_action = np.asarray(cf_action)
    cf_duration = np.asarray(cf_duration)
    episode_seed = np.asarray(episode_seed)
    n = len(episode_seed)
    folds = np.full(n, -1, dtype=np.int64)
    strata: dict[tuple[int, int], list[int]] = {}
    for index in range(n):
        key = (int(cf_action[index]), int(cf_duration[index]))
        strata.setdefault(key, []).append(index)
    for members in strata.values():
        ordered = sorted(members, key=lambda i: _crossfit_rank(int(episode_seed[i])))
        for rank, index in enumerate(ordered):
            folds[index] = rank % 2
    if (folds < 0).any():
        raise RuntimeError("cross-fit fold assignment left episodes unassigned")
    return folds
