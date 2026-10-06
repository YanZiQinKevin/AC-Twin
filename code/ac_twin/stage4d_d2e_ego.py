"""Stage 4D-D2E ego counterfactual-prediction repair branch.

Plan ``AC-TWIN-RFER-STAGE4D-D2E-EGO-CONSISTENT-EXTERNAL-2026-09-24-V0.1`` section 3. The old
D2 pipeline trained its effect model with ``compute_effect_eval_mask(exclude_ego=True)``
(stage3_data.py:472) AND zeroed the ego slot in the cf-independent ``build_inference_mask``
(stage4c_protocol.py), so the ego slot of the ``e2e`` prediction tensor was never supervised.
The ten-action objective ``J = 0.8*C_int + 0.2*C_eff`` has an efficiency term that depends on
the ego's per-action final progress, which ``trajectory_components`` reads from slot 0
UNCONDITIONALLY (stage4c_objective.py:111-112) — so the base selector ``a0`` saw a near-zero
predicted ego progress span and failed Gate TASK c5. This module adds a NEW portable ego
effect branch that predicts the ego's 20-frame ``(ds, dd)`` displacement per action and
replaces ONLY slot 0 of the ``e2e`` tensor; slots 1-4 stay from the frozen old E2E stack and
``compute_predicted_costs`` / ``stage4c_objective.py`` run UNCHANGED.

Mask discipline (plan 3.1): ``M_non_ego`` keeps excluding the ego for the original
effect/risk diagnostics. ``M_ego`` (this module) is True only where the ego truly exists AND
the action/position is legal — it is the ego slot of the ego-INCLUSIVE
``compute_effect_eval_mask(..., exclude_ego=False)`` and is derived from counterfactual truth,
so it is used ONLY for the training loss/labels and NEVER enters risk-feature construction or
any train/validation/screen/confirm inference path (those use the collector's cf-independent
``inference_mask``). The branch inputs are legal-only (pre-intervention history, factual
suffix, action/duration, normalized ``t_star``); no CF-truth future is read at inference.

Everything here is NEW code under the D2E scope; no frozen D2/D2R module is modified.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin import stage4d_external_inference as INF  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402

# --- frozen identity / geometry (inherited from the D2 contract, read-only) ------------------
PLAN_ID = "AC-TWIN-RFER-STAGE4D-D2E-EGO-CONSISTENT-EXTERNAL-2026-09-24-V0.1"
D2E_DIR = ROOT / "experiments/stage4_reliability/stage4d_d2e"

N_ACTIONS = INF.N_ACTIONS                    # 10
HISTORY_LEN = INF.HISTORY_LEN                # 10
FUTURE_LEN = INF.FUTURE_LEN                  # 20
OBJECT_CAPACITY = INF.OBJECT_CAPACITY        # 5
N_FEATURES = INF.N_FEATURES                  # 8
EGO_SLOT = 0
N_SEEDS = INF.N_SEEDS                        # 5
#: ego branch seeds, index-paired with the predictor pair seeds (plan 3.2 "五 seed 按索引一一配对").
EGO_SEEDS = tuple(int(s) for s in INF.PREDICTOR_PAIR_SEEDS)  # (1101,1202,1303,1404,1505)

# --- frozen shared ego-head architecture (plan 3.2) ------------------------------------------
EGO_HIDDEN_WIDTH = 128
EGO_OUTPUT_DIM = 40                          # 20 frames * 2 coords
EGO_OUTPUT_SHAPE = (FUTURE_LEN, 2)

# --- frozen input field layout (plan 3.2, mechanically computed -> input_dim = 1002) ---------
N_ACTION_CLASSES = 5
HISTORY_FLAT = HISTORY_LEN * OBJECT_CAPACITY * N_FEATURES      # 400
SUFFIX_FLAT = HISTORY_LEN * OBJECT_CAPACITY * N_FEATURES       # 400
PLAN_ONEHOT_FLAT = FUTURE_LEN * N_ACTION_CLASSES               # 100 (per-frame 5-class one-hot)
#: [0:400]=history, [400:800]=factual_suffix, [800:900]=factual plan one-hot,
#: [900:1000]=counterfactual plan one-hot, [1000]=normalized t_star, [1001]=normalized duration.
INPUT_DIM = HISTORY_FLAT + SUFFIX_FLAT + 2 * PLAN_ONEHOT_FLAT + 2   # 1002
SLICE_HISTORY = slice(0, HISTORY_FLAT)
SLICE_SUFFIX = slice(HISTORY_FLAT, HISTORY_FLAT + SUFFIX_FLAT)
SLICE_FACT_PLAN = slice(HISTORY_FLAT + SUFFIX_FLAT,
                        HISTORY_FLAT + SUFFIX_FLAT + PLAN_ONEHOT_FLAT)
SLICE_CF_PLAN = slice(HISTORY_FLAT + SUFFIX_FLAT + PLAN_ONEHOT_FLAT,
                      HISTORY_FLAT + SUFFIX_FLAT + 2 * PLAN_ONEHOT_FLAT)
IDX_T_STAR = HISTORY_FLAT + SUFFIX_FLAT + 2 * PLAN_ONEHOT_FLAT        # 1000
IDX_DURATION = IDX_T_STAR + 1                                          # 1001

# --- frozen normalization (plan 3.2 "归一化 t_star 与 duration") ------------------------------
T_STAR_MIN, T_STAR_MAX = 10.0, 15.0          # t_star_candidates = [10..15]
DURATION_MIN, DURATION_MAX = 1.0, 3.0        # duration levels {1, 3}

# --- frozen optimizer / schedule (plan 3.2) ---------------------------------------------------
EGO_LEARNING_RATE = 1e-3
EGO_WEIGHT_DECAY = 1e-4
EGO_TRAIN_BATCH_GROUPS = 64                  # group batch -> 64*10 = 640 branch units / step
EGO_MAX_EPOCHS = 100
EGO_EARLY_STOP_PATIENCE = 12
EGO_RANK_AUX_WEIGHT = 0.25                   # fixed, no weight search

#: the two bounded development candidates (plan 3.2); no further additions are permitted.
CANDIDATE_EGO_TRAJECTORY = "EGO_TRAJECTORY"
CANDIDATE_EGO_TRAJECTORY_PLUS_RANK = "EGO_TRAJECTORY_PLUS_RANK"
CANDIDATES = (CANDIDATE_EGO_TRAJECTORY, CANDIDATE_EGO_TRAJECTORY_PLUS_RANK)
#: selection tie tolerance: prefer the simpler EGO_TRAJECTORY within this margin (plan 3.2/3.3).
SELECTION_TIE_TOLERANCE = 1e-12


# --- input assembly ----------------------------------------------------------------------------


def _one_hot_actions(actions: np.ndarray) -> np.ndarray:
    """Per-frame 5-class one-hot of an ``(E, T)`` action plan -> ``(E, T, 5)`` float32.

    Matches the frozen codebase convention (``stage3r_models.py`` ``F.one_hot(..., 5)``).
    """
    actions = np.asarray(actions, dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != FUTURE_LEN:
        raise ValueError(f"actions must be (E,{FUTURE_LEN}); got {actions.shape}")
    if (actions < 0).any() or (actions >= N_ACTION_CLASSES).any():
        raise ValueError("action indices outside the 5-class range")
    out = np.zeros(actions.shape + (N_ACTION_CLASSES,), dtype=np.float32)
    np.put_along_axis(out, actions[..., None], 1.0, axis=-1)
    return out


def assemble_ego_features(bundle: INF.BranchBundle, scaler: FeatureScaler) -> np.ndarray:
    """Assemble the frozen 1002-dim ego-branch input matrix ``(E, 1002)`` float32.

    Legal inputs only (plan 3.1): pre-intervention history, factual suffix, the factual and
    counterfactual 20-step action plans (per-frame 5-class one-hot), and normalized
    ``t_star``/duration. The motion channels (2,3,4,5) of history/factual_suffix are
    standardised by the train-only ``scaler``; binary/unit-circle channels, the one-hot plans
    and the pre-normalized scalars pass through unscaled. No counterfactual-truth future,
    ``J_true``, regret or risk score is read.
    """
    branches = bundle.branches
    hist = scaler.transform(bundle.history)               # (E,10,5,8), motion channels scaled
    suffix = scaler.transform(bundle.factual_suffix)      # (E,10,5,8)
    feat = np.empty((branches, INPUT_DIM), dtype=np.float32)
    feat[:, SLICE_HISTORY] = hist.reshape(branches, HISTORY_FLAT)
    feat[:, SLICE_SUFFIX] = suffix.reshape(branches, SUFFIX_FLAT)
    feat[:, SLICE_FACT_PLAN] = _one_hot_actions(bundle.factual_actions).reshape(
        branches, PLAN_ONEHOT_FLAT)
    feat[:, SLICE_CF_PLAN] = _one_hot_actions(bundle.counterfactual_actions).reshape(
        branches, PLAN_ONEHOT_FLAT)
    feat[:, IDX_T_STAR] = (bundle.t_star.astype(np.float32) - T_STAR_MIN) / (
        T_STAR_MAX - T_STAR_MIN)
    feat[:, IDX_DURATION] = (bundle.cf_duration.astype(np.float32) - DURATION_MIN) / (
        DURATION_MAX - DURATION_MIN)
    if not np.isfinite(feat).all():
        raise RuntimeError("non-finite ego-branch input feature")
    return feat


def compute_ego_train_mask(arrays: INF.RoleArrays) -> np.ndarray:
    """``M_ego`` (E, FUTURE_LEN) bool: ego-inclusive legal training mask at slot 0.

    The ego slot of ``compute_effect_eval_mask(..., exclude_ego=False)`` = ego is
    history-known AND factual-present AND counterfactual-present (stage1_data
    ``compute_valid_mask``). This is counterfactual-truth-derived, so it is ONLY a training
    loss/label mask; it never enters risk features or any inference path (plan 3.1).
    """
    groups, n_actions = arrays.groups, arrays.n_actions
    branches = groups * n_actions
    g_idx = np.repeat(np.arange(groups), n_actions)
    history_b = arrays.object_history[g_idx]                       # (E,10,5,8)
    factual_b = arrays.object_factual_future[g_idx]                # (E,20,5,8)
    cf_b = arrays.object_counterfactual_futures.reshape(
        branches, FUTURE_LEN, OBJECT_CAPACITY, N_FEATURES)         # (E,20,5,8)
    valid = s3d.compute_effect_eval_mask(history_b, factual_b, cf_b, exclude_ego=False)
    mask = np.asarray(valid[..., EGO_SLOT], dtype=bool)            # (E,20)
    if int(mask.sum()) == 0:
        raise RuntimeError("M_ego is all-zero: ego training mask degenerate (fail-closed)")
    return mask


def ego_targets(bundle: INF.BranchBundle) -> np.ndarray:
    """The true ego displacement target ``(E, FUTURE_LEN, 2)`` = effect_target slot 0.

    ``build_effect_target`` computes the effect for all five slots including the ego (it only
    returns the ego-excluding mask separately); the ego displacement values are present and
    unmasked. Slot 0 is the ego ``(ds, dd)`` in metres — the same native (x,y)/(s,d) frame as
    the ``e2e`` tensor it will replace.
    """
    return np.asarray(bundle.effect_target[:, :, EGO_SLOT, :], dtype=np.float32)


# --- model --------------------------------------------------------------------------------------


class EgoEffectHead(nn.Module):
    """Frozen shared ego head: Linear(input_dim,128)->LN->ReLU->Linear(128,128)->LN->ReLU
    ->Linear(128,40), output reshaped to (20, 2) (plan 3.2). Identical for both candidates."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = EGO_HIDDEN_WIDTH,
                 output_dim: int = EGO_OUTPUT_DIM) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, FUTURE_LEN, 2)


def count_ego_parameters(input_dim: int = INPUT_DIM, hidden: int = EGO_HIDDEN_WIDTH,
                         output_dim: int = EGO_OUTPUT_DIM) -> int:
    """Trainable parameter count of the frozen ego head (receipt/preflight integrity)."""
    model = EgoEffectHead(input_dim, hidden, output_dim)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# --- losses --------------------------------------------------------------------------------------


def _masked_trajectory_loss(pred: torch.Tensor, target: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    """SmoothL1 ego-trajectory loss over valid frames (plan 3.2 candidate EGO_TRAJECTORY).

    ``pred``/``target`` (B,20,2), ``mask`` (B,20). Per-frame loss = SmoothL1 summed over the
    two coords; the batch loss is the mask-weighted mean over valid frames.
    """
    per_frame = F.smooth_l1_loss(pred, target, reduction="none").sum(-1)   # (B,20)
    denom = mask.sum().clamp(min=1.0)
    return (per_frame * mask).sum() / denom


def _rank_aux_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                   rank_scale: float) -> torch.Tensor:
    """Fixed ten-action progress-difference auxiliary loss (candidate EGO_TRAJECTORY_PLUS_RANK).

    The batch is group-batched (complete groups, group-major), so ``(B,)`` reshapes to
    ``(G_b, 10)``. The per-action final progress is mechanically the final-frame longitudinal
    displacement (the per-group factual/history offset cancels in within-group centering), so
    the centered predicted/true final ``ds`` are matched with SmoothL1 after normalizing by
    the train-only ``rank_scale``. The group mean is taken over valid final frames only.
    """
    batch = pred.shape[0]
    groups_b = batch // N_ACTIONS
    pred_fin = pred[:, -1, 0].reshape(groups_b, N_ACTIONS)      # predicted final ds (G_b,10)
    true_fin = target[:, -1, 0].reshape(groups_b, N_ACTIONS)    # true final ds
    m_fin = mask[:, -1].reshape(groups_b, N_ACTIONS)            # final-frame validity

    def _centre(x: torch.Tensor) -> torch.Tensor:
        denom = m_fin.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * m_fin).sum(dim=1, keepdim=True) / denom
        return (x - mean) * m_fin

    c_pred = _centre(pred_fin) / float(rank_scale)
    c_true = _centre(true_fin) / float(rank_scale)
    per = F.smooth_l1_loss(c_pred, c_true, reduction="none") * m_fin
    return per.sum() / m_fin.sum().clamp(min=1.0)


def rank_label_scale(targets: np.ndarray, mask: np.ndarray) -> float:
    """Train-only scale for the rank auxiliary: std of the within-group-centered true ego
    final displacement over valid (group, action) units. Frozen into the training receipt."""
    targets = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    branches = targets.shape[0]
    groups = branches // N_ACTIONS
    true_fin = targets[:, -1, 0].reshape(groups, N_ACTIONS)
    m_fin = mask[:, -1].reshape(groups, N_ACTIONS)
    denom = np.maximum(m_fin.sum(axis=1, keepdims=True), 1)
    mean = (true_fin * m_fin).sum(axis=1, keepdims=True) / denom
    centred = (true_fin - mean) * m_fin
    vals = centred[m_fin]
    if vals.size == 0:
        raise RuntimeError("no valid final-frame ego labels to fit the rank scale")
    scale = float(vals.std())
    return float(max(scale, 1e-6))


# --- training ------------------------------------------------------------------------------------


@dataclass
class EgoTrainResult:
    """One trained ego candidate x seed."""
    candidate: str
    seed: int
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    n_train_branches: int
    state_dict: dict[str, torch.Tensor]
    rank_scale: float | None
    train_seconds: float


def _group_branch_indices(groups: int, batch_groups: int,
                          rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled whole-group batches -> branch-index arrays (group-major, complete groups)."""
    order = rng.permutation(groups)
    blocks: list[np.ndarray] = []
    for start in range(0, groups, batch_groups):
        gblock = order[start:start + batch_groups]
        branch_idx = (gblock[:, None] * N_ACTIONS
                      + np.arange(N_ACTIONS, dtype=np.int64)[None, :]).ravel()
        blocks.append(branch_idx)
    return blocks


def train_ego_candidate(
    candidate: str,
    seed: int,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_mask: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    val_mask: np.ndarray,
    *,
    device: str,
    rank_scale: float | None,
    max_epochs: int = EGO_MAX_EPOCHS,
    patience: int = EGO_EARLY_STOP_PATIENCE,
    batch_groups: int = EGO_TRAIN_BATCH_GROUPS,
    progress: Callable[[str], None] | None = None,
) -> EgoTrainResult:
    """Train one ego candidate on one seed; early-stop/checkpoint on the validation role.

    The validation role is used ONLY for early stopping / mechanical checkpoint selection
    (lowest validation loss), mirroring the frozen base-predictor rule (plan 7.5 / 3.2). For
    ``EGO_TRAJECTORY_PLUS_RANK`` the fixed rank auxiliary (weight ``EGO_RANK_AUX_WEIGHT``) is
    added to both the train and the monitored validation loss.
    """
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown ego candidate {candidate!r}")
    use_rank = candidate == CANDIDATE_EGO_TRAJECTORY_PLUS_RANK
    if use_rank and (rank_scale is None):
        raise ValueError("EGO_TRAJECTORY_PLUS_RANK requires a train-only rank_scale")

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    dev = torch.device(device)

    x_tr = torch.as_tensor(train_features, dtype=torch.float32, device=dev)
    y_tr = torch.as_tensor(train_targets, dtype=torch.float32, device=dev)
    m_tr = torch.as_tensor(train_mask, dtype=torch.float32, device=dev)
    x_va = torch.as_tensor(val_features, dtype=torch.float32, device=dev)
    y_va = torch.as_tensor(val_targets, dtype=torch.float32, device=dev)
    m_va = torch.as_tensor(val_mask, dtype=torch.float32, device=dev)

    model = EgoEffectHead().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=EGO_LEARNING_RATE,
                            weight_decay=EGO_WEIGHT_DECAY)

    groups = train_features.shape[0] // N_ACTIONS
    t0 = time.time()
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since = 0

    def _loss(xb: torch.Tensor, yb: torch.Tensor, mb: torch.Tensor) -> torch.Tensor:
        pred = model(xb)
        loss = _masked_trajectory_loss(pred, yb, mb)
        if use_rank:
            loss = loss + EGO_RANK_AUX_WEIGHT * _rank_aux_loss(pred, yb, mb, rank_scale)
        return loss

    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        for branch_idx in _group_branch_indices(groups, batch_groups, rng):
            idx = torch.as_tensor(branch_idx, dtype=torch.long, device=dev)
            opt.zero_grad(set_to_none=True)
            loss = _loss(x_tr.index_select(0, idx), y_tr.index_select(0, idx),
                         m_tr.index_select(0, idx))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(_loss(x_va, y_va, m_va).item())
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since = 0
        else:
            epochs_since += 1
        if progress is not None:
            progress(f"[ego-fit] candidate={candidate} seed={seed} epoch={epoch + 1}/"
                     f"{max_epochs} val_loss={val_loss:.6f} best={best_val:.6f}"
                     f"@epoch{best_epoch + 1}")
        if epochs_since >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"ego training produced no checkpoint for {candidate} seed {seed}")
    return EgoTrainResult(
        candidate=candidate, seed=int(seed), best_val_loss=best_val,
        best_epoch=int(best_epoch), epochs_run=int(epochs_run),
        n_train_branches=int(train_features.shape[0]), state_dict=best_state,
        rank_scale=(None if rank_scale is None else float(rank_scale)),
        train_seconds=time.time() - t0)


# --- inference + slot-0 repair --------------------------------------------------------------------


def predict_ego_effect(model: EgoEffectHead, features: np.ndarray, *, device: str,
                       batch_size: int = 4096) -> np.ndarray:
    """Frozen ego-branch inference -> predicted ego displacement ``(E, 20, 2)`` float32.

    Reads only the legal assembled features (no CF truth); used identically for
    train/validation/screen/confirm inference paths.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    feats = torch.as_tensor(features, dtype=torch.float32)
    out = np.empty((features.shape[0], FUTURE_LEN, 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            block = feats[start:start + batch_size].to(dev)
            out[start:start + batch_size] = model(block).cpu().numpy().astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite ego-branch prediction")
    return out


def repair_e2e_ego(e2e: np.ndarray, ego_deltas: np.ndarray) -> np.ndarray:
    """Return a copy of ``e2e`` (S,E,T,N,2) with ONLY slot 0 replaced by the ego branch.

    ``ego_deltas`` is (S,E,T,2): per-seed predicted ego displacement, index-paired with the
    predictor pair seeds. Slots 1-4 are copied unchanged from the frozen old E2E stack, so the
    downstream ``compute_predicted_costs`` / ``build_direct_inputs`` run on a tensor identical
    to the old one except for the repaired ego slot (plan 3.1 "只替换 slot 0").
    """
    e2e = np.asarray(e2e, dtype=np.float32)
    ego_deltas = np.asarray(ego_deltas, dtype=np.float32)
    if e2e.ndim != 5 or e2e.shape[-1] != 2:
        raise ValueError(f"e2e must be (S,E,T,N,2); got {e2e.shape}")
    expected = (e2e.shape[0], e2e.shape[1], e2e.shape[2], 2)
    if ego_deltas.shape != expected:
        raise ValueError(f"ego_deltas must be {expected}; got {ego_deltas.shape}")
    repaired = e2e.copy()
    repaired[:, :, :, EGO_SLOT, :] = ego_deltas
    return repaired


__all__ = [
    "CANDIDATES", "CANDIDATE_EGO_TRAJECTORY", "CANDIDATE_EGO_TRAJECTORY_PLUS_RANK",
    "D2E_DIR", "DURATION_MAX", "DURATION_MIN", "EGO_EARLY_STOP_PATIENCE", "EGO_HIDDEN_WIDTH",
    "EGO_LEARNING_RATE", "EGO_MAX_EPOCHS", "EGO_OUTPUT_DIM", "EGO_OUTPUT_SHAPE",
    "EGO_RANK_AUX_WEIGHT", "EGO_SEEDS", "EGO_SLOT", "EGO_TRAIN_BATCH_GROUPS",
    "EGO_WEIGHT_DECAY", "EgoEffectHead", "EgoTrainResult", "FUTURE_LEN", "HISTORY_FLAT",
    "HISTORY_LEN", "IDX_DURATION", "IDX_T_STAR", "INPUT_DIM", "N_ACTIONS", "N_ACTION_CLASSES",
    "N_FEATURES", "N_SEEDS", "OBJECT_CAPACITY", "PLAN_ID", "PLAN_ONEHOT_FLAT",
    "SELECTION_TIE_TOLERANCE", "SLICE_CF_PLAN", "SLICE_FACT_PLAN", "SLICE_HISTORY",
    "SLICE_SUFFIX", "SUFFIX_FLAT", "T_STAR_MAX", "T_STAR_MIN", "assemble_ego_features",
    "compute_ego_train_mask", "count_ego_parameters", "ego_targets", "predict_ego_effect",
    "rank_label_scale", "repair_e2e_ego", "train_ego_candidate",
]
