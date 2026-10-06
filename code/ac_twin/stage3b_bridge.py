"""Stage 3B predictor->router bridge: effect target, oracle gate, loss, features.

This module holds the small, pure, individually testable pieces that connect the
frozen Stage 3A Plus production predictors to the Stage 3B evidence routers:

- the per-agent positional ``effect`` supervision target (counterfactual minus
  factual xy over the 20-step horizon);
- the deterministic oracle-response-reference gate ``w = R/(R+t1)`` built only
  from the frozen Stage 1 threshold ``t1`` (no learned mapping);
- the masked effect-MSE training/selection loss;
- the frozen ``CAL_PER_SEED_VECTOR`` application that turns per-seed regime
  logits into calibrated regime probabilities;
- ordered, unshuffled frozen-predictor feature generation for a split.

Nothing here reads counterfactual truth into a *model input*: the counterfactual
branch is used exclusively to build the supervision target and the oracle gate.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ac_twin.stage1_data import POSITION_FEATURES
from ac_twin.stage3_data import FROZEN_T1
from ac_twin.stage3a_plus_calibration import (
    CAL_PER_SEED_VECTOR,
    CalibrationArtifact,
    apply_calibrator,
)

_POS = slice(POSITION_FEATURES[0], POSITION_FEATURES[1] + 1)


# --------------------------------------------------------------------------- #
# Supervision target + oracle gate (never model inputs)
# --------------------------------------------------------------------------- #
def build_effect_target(
    object_factual_future: np.ndarray, object_counterfactual_future: np.ndarray
) -> np.ndarray:
    """Per-agent positional effect ``E×20×N×2 = counterfactual_xy - factual_xy``.

    This is the router's supervision target. It is derived from the paired truth
    and is never a model input.
    """
    factual_xy = np.asarray(object_factual_future, dtype=np.float32)[..., _POS]
    counterfactual_xy = np.asarray(object_counterfactual_future, dtype=np.float32)[..., _POS]
    return (counterfactual_xy - factual_xy).astype(np.float32)


def make_oracle_gate(r_true: np.ndarray, *, t1: float = FROZEN_T1) -> np.ndarray:
    """Deterministic oracle-response-reference gate ``w = R/(R+t1)`` -> ``E×T×N×1``.

    The oracle reference reads the true response state, but the gate itself is a
    FIXED monotone map using only the frozen Stage 1 threshold ``t1`` (R=0 -> 0,
    R=t1 -> 0.5, bounded in ``[0, 1)``). No mapping weight is learned here; the
    consuming decoder is still trained on train and selected on validation, per
    plan v0.2 §4.4. This is an attainability reference, not a fairness claim or
    a theoretical upper bound.
    """
    r = np.asarray(r_true, dtype=np.float64)
    if (r < 0).any():
        raise ValueError("oracle gate requires a non-negative response magnitude")
    gate = r / (r + float(t1))
    return gate.astype(np.float32)[..., np.newaxis]


# --------------------------------------------------------------------------- #
# Masked effect loss (training + validation selection)
# --------------------------------------------------------------------------- #
def router_masked_mse(
    effect: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Masked MSE over valid agent-times and the 2 position dims.

    ``effect``/``target`` are ``B×T×N×2``; ``valid`` is ``B×T×N`` bool. Padding
    never enters the loss.
    """
    valid4 = valid.unsqueeze(-1).to(effect.dtype)
    denom = valid4.sum().clamp_min(1.0) * effect.shape[-1]
    return (((effect - target) ** 2) * valid4).sum() / denom


def router_masked_mse_sums(
    effect: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Unnormalised float64 masked sums for batch-size/permutation-invariant val.

    Returns ``reg_sum`` (squared-error mass over valid agent-times and both
    position dims), ``valid_count`` (number of valid agent-times x 2 dims) and a
    zero ``ce_sum`` so the router reuses the global validation aggregation and
    the ``(reg, ce)`` selection hook of :func:`stage3_training.train_model`.
    """
    valid4 = valid.unsqueeze(-1).to(effect.dtype)
    sq = ((effect - target) ** 2) * valid4  # B,T,N,2
    reg_sum = sq.sum(dtype=torch.float64)
    count = valid4.sum(dtype=torch.float64) * effect.shape[-1]
    return {
        "reg_sum": reg_sum,
        "ce_sum": torch.zeros((), dtype=torch.float64, device=effect.device),
        "valid_count": count,
    }


# --------------------------------------------------------------------------- #
# Frozen per-seed calibration -> regime probabilities
# --------------------------------------------------------------------------- #
def frozen_calibration_artifact(per_seed_params: Sequence[Mapping[str, Any]]) -> CalibrationArtifact:
    """Rebuild the frozen CAL_PER_SEED_VECTOR artifact from bound per-seed params.

    ``per_seed_params`` is the five-element list bound in the Stage 3B input
    binding receipt (production seeds only). No refit happens here.
    """
    params = {
        "per_seed": [
            {"a": list(p["a"]), "b": list(p["b"])} for p in per_seed_params
        ]
    }
    return CalibrationArtifact(method=CAL_PER_SEED_VECTOR, params=params)


def calibrated_regime_probs_single(
    params: Mapping[str, Any], logits: np.ndarray
) -> np.ndarray:
    """Apply ONE seed's frozen vector scaling to ``...×4`` logits -> probs.

    ``params`` is that seed's ``{"a", "b"}`` from the binding receipt. Used when a
    single predictor/router pair is being serviced (the common Stage 3B path).
    """
    artifact = CalibrationArtifact(
        method=CAL_PER_SEED_VECTOR,
        params={"a": list(params["a"]), "b": list(params["b"])},
    )
    return apply_calibrator(artifact, np.asarray(logits, dtype=np.float64)).astype(np.float32)


def calibrated_regime_probs(
    per_seed_params: Sequence[Mapping[str, Any]], logits_seeds: np.ndarray
) -> np.ndarray:
    """Apply the frozen per-seed vector scaling to ``S×...×4`` logits -> probs.

    The leading axis of ``logits_seeds`` is the production-seed axis and must
    match the number of bound per-seed parameter sets (five).
    """
    logits_seeds = np.asarray(logits_seeds, dtype=np.float64)
    if logits_seeds.shape[0] != len(per_seed_params):
        raise ValueError(
            f"per-seed calibrator has {len(per_seed_params)} seeds but logits have "
            f"leading dim {logits_seeds.shape[0]}"
        )
    return np.stack(
        [
            calibrated_regime_probs_single(per_seed_params[i], logits_seeds[i])
            for i in range(len(per_seed_params))
        ],
        axis=0,
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Ordered frozen-predictor feature generation
# --------------------------------------------------------------------------- #
def predictor_features_ordered(
    model: torch.nn.Module,
    tensors: Mapping[str, torch.Tensor],
    indices: np.ndarray,
    device: str | torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run a frozen predictor over ``indices`` in the exact requested order.

    Returns ``episode_indices`` (identical, in value and order, to ``indices``),
    ``response_features`` (``B×20×N×d_r``) and ``regime_logits`` (``B×20×N×4``),
    aligned to that order. Never shuffled — this is frozen inference, not
    training.
    """
    indices = np.asarray(indices)
    model.to(device)
    model.eval()
    feats, logits, seen = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = {key: value[selected].to(device) for key, value in tensors.items()}
            out = model(
                batch["history"], batch["factual_suffix"], batch["factual_actions"],
                batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
                batch["t_star"],
            )
            feats.append(out["response_features"].detach().cpu().numpy())
            logits.append(out["regime_logits"].detach().cpu().numpy())
            seen.append(np.asarray(selected))
    return {
        "episode_indices": np.concatenate(seen, axis=0),
        "response_features": np.concatenate(feats, axis=0).astype(np.float32),
        "regime_logits": np.concatenate(logits, axis=0).astype(np.float32),
    }
