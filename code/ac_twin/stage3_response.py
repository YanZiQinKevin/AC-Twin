"""Stage 3A response prediction: model, simple baselines and calibration.

``Stage3ResponsePredictor`` predicts, per vehicle and per time step and WITHOUT
reading counterfactual truth, (i) a continuous response magnitude, (ii) a
four-class response regime and (iii) a response feature vector. It reuses the
Stage 1 per-object temporal encoder / masked scene summary / shared per-agent
decoder pattern; stable IDs only align objects within an episode and are never
a cross-episode embedding. No graph network, Transformer, VAE or external prior
is added at P0.

Allowed inputs: object history, the first-10-step factual suffix, the factual
and counterfactual action sequences, ``cf_action``/``cf_duration``/``t_star``,
presence masks, ``is_ego`` and episode-local alignment. Forbidden inputs:
``object_counterfactual_future``, ``R_true``, true regime bins, future
differences, or any derived-truth field. Training supervision (``R_pos``,
regime labels) is computed from the train split's paired truth only.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ac_twin.stage1_data import N_FEATURES
from ac_twin.stage1_model import PerObjectTemporalEncoder, masked_scene_summary
from ac_twin.stage3_data import (
    FROZEN_THRESHOLDS,
    N_REGIMES,
    compute_regime_labels,
)

DEFAULT_LAMBDA_CLS = 0.25
LAMBDA_CLS_CANDIDATES = (0.10, 0.25)
SMOOTH_L1_BETA = 0.1

# Repair plan v0.1 §2.3: candidates keep their own training lambda_cls, but ALL
# early stopping / checkpoint choice / cross-config ranking uses this FIXED
# selection coefficient so scores are comparable across lambda values.
SELECTION_LAMBDA_CLS = 0.25


def selection_score(reg_loss: float, ce_loss: float) -> float:
    """Fixed-coefficient comparison score: L_select = L_reg + 0.25 * L_CE.

    Never parameterised by a candidate's own ``lambda_cls`` — that would make
    smaller-lambda candidates win mechanically on the combined-loss scale.
    """
    return float(reg_loss) + SELECTION_LAMBDA_CLS * float(ce_loss)


def response_loss_sums(
    outputs: Mapping[str, torch.Tensor],
    r_true: torch.Tensor,
    regime_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Unnormalised masked sums (float64) for global validation aggregation.

    Repair plan v0.1 §2.4: validation metrics must be aggregated as
    sum(numerator)/sum(denominator) over ALL valid agent-times, never as an
    equal-weight mean of per-batch scalars. Returning float64 sums makes the
    result invariant to batch size, tail-batch size and episode permutation.
    """
    valid = valid_mask.to(outputs["log1p_r_hat"].dtype)
    count = valid.sum(dtype=torch.float64)

    log1p_true = torch.log1p(r_true.clamp_min(0.0))
    smooth = torch.nn.functional.smooth_l1_loss(
        outputs["log1p_r_hat"], log1p_true, beta=SMOOTH_L1_BETA, reduction="none"
    )
    reg_sum = (smooth * valid).sum(dtype=torch.float64)

    ce = torch.nn.functional.cross_entropy(
        outputs["regime_logits"].reshape(-1, N_REGIMES),
        regime_labels.reshape(-1).long(),
        weight=class_weights,
        reduction="none",
    ).reshape(valid_mask.shape)
    ce_sum = (ce * valid).sum(dtype=torch.float64)
    return {"reg_sum": reg_sum, "ce_sum": ce_sum, "valid_count": count}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Stage3ResponsePredictor(nn.Module):
    """Per-agent, per-time response magnitude + regime + feature predictor.

    Outputs a dict with:

    - ``log1p_r_hat``: ``B×T×N`` prediction of ``log1p(R)`` (inverse is >= 0);
    - ``regime_logits``: ``B×T×N×4`` over zero/low/mid/high;
    - ``response_features``: ``B×T×N×d_r`` (never caches or encodes truth).
    """

    def __init__(
        self,
        feature_dim: int = N_FEATURES,
        action_dim: int = 5,
        object_embedding_dim: int = 64,
        hidden_dim: int = 64,
        action_embedding_dim: int = 8,
        response_feature_dim: int = 32,
        intervention_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.response_feature_dim = int(response_feature_dim)
        self.history_encoder = PerObjectTemporalEncoder(feature_dim, object_embedding_dim, hidden_dim)
        self.factual_encoder = PerObjectTemporalEncoder(feature_dim, object_embedding_dim, hidden_dim)
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        # Episode-level intervention descriptor: cf_action (5) + cf_duration + t_star.
        self.intervention_mlp = nn.Sequential(
            nn.Linear(action_dim + 2, intervention_embedding_dim),
            nn.ReLU(),
            nn.Linear(intervention_embedding_dim, intervention_embedding_dim),
        )
        context_dim = hidden_dim * 4
        self.context_to_hidden = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
        )
        decoder_input_dim = context_dim + 2 * action_embedding_dim + intervention_embedding_dim
        self.decoder = nn.GRU(decoder_input_dim, hidden_dim, batch_first=True)
        out_dim = 1 + N_REGIMES + response_feature_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _intervention_embedding(
        self,
        cf_action: torch.Tensor,
        cf_duration: torch.Tensor,
        t_star: torch.Tensor,
    ) -> torch.Tensor:
        batch = cf_action.shape[0]
        action_onehot = torch.nn.functional.one_hot(cf_action.long(), num_classes=5).float()
        duration_norm = (cf_duration.float().view(batch, 1) - 1.0) / 2.0
        t_star_norm = (t_star.float().view(batch, 1) - 12.5) / 2.5
        return self.intervention_mlp(torch.cat([action_onehot, duration_norm, t_star_norm], dim=-1))

    def forward(
        self,
        history: torch.Tensor,
        factual_suffix: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
        cf_action: torch.Tensor,
        cf_duration: torch.Tensor,
        t_star: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        suffix_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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
        initial_hidden = self.context_to_hidden(agent_context.reshape(batch * agents, -1)).unsqueeze(0)

        factual_emb = self.action_embedding(factual_actions.long())
        counterfactual_emb = self.action_embedding(counterfactual_actions.long())
        factual_emb = factual_emb.unsqueeze(1).expand(batch, agents, horizon, -1)
        counterfactual_emb = counterfactual_emb.unsqueeze(1).expand(batch, agents, horizon, -1)
        intervention_emb = self._intervention_embedding(cf_action, cf_duration, t_star)
        intervention_emb = intervention_emb.unsqueeze(1).unsqueeze(2).expand(
            batch, agents, horizon, -1
        )
        context_seq = agent_context.unsqueeze(2).expand(batch, agents, horizon, -1)

        decoder_input = torch.cat(
            [
                context_seq.reshape(batch * agents, horizon, -1),
                factual_emb.reshape(batch * agents, horizon, -1),
                counterfactual_emb.reshape(batch * agents, horizon, -1),
                intervention_emb.reshape(batch * agents, horizon, -1),
            ],
            dim=-1,
        )
        decoded, _ = self.decoder(decoder_input, initial_hidden)
        raw = self.head(decoded).reshape(batch, agents, horizon, -1).permute(0, 2, 1, 3)
        # raw: B,T,N,1+4+d_r
        log1p_r_hat = raw[..., 0]
        regime_logits = raw[..., 1 : 1 + N_REGIMES]
        response_features = raw[..., 1 + N_REGIMES :]

        keep = history_active.unsqueeze(1).unsqueeze(-1).to(raw.dtype)  # B,1,N,1
        log1p_r_hat = log1p_r_hat * keep.squeeze(-1)
        regime_logits = regime_logits * keep
        response_features = response_features * keep
        return {
            "log1p_r_hat": log1p_r_hat,
            "regime_logits": regime_logits,
            "response_features": response_features,
        }


def invert_log1p(log1p_r_hat: torch.Tensor) -> torch.Tensor:
    """R_hat = expm1(log1p_r_hat), clamped to be non-negative."""
    return torch.expm1(log1p_r_hat).clamp_min(0.0)


def multiclass_brier_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Multiclass Brier loss over VALID agent-times (Stage 3A Plus §4.2).

    Mean over valid positions of ``mean_c (softmax(logits)_c - onehot_c)^2``.
    No class weight is applied (plan §4.2 禁止对 Brier 项使用 class weight);
    padding never enters.
    """
    valid = mask.to(logits.dtype)
    denom = valid.sum().clamp_min(1.0)
    probs = torch.softmax(logits, dim=-1)
    onehot = torch.nn.functional.one_hot(labels.long(), N_REGIMES).to(logits.dtype)
    per_pos = ((probs - onehot) ** 2).mean(dim=-1)  # E,T,N
    return (per_pos * valid).sum() / denom


def response_masked_loss(
    outputs: Mapping[str, torch.Tensor],
    r_true: torch.Tensor,
    regime_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor,
    lambda_cls: float = DEFAULT_LAMBDA_CLS,
    lambda_brier: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """L_R = SmoothL1(log1p R) + lambda_cls*weightedCE + lambda_brier*multiclassBrier.

    All terms are masked by ``valid_mask`` (E×T×N); padding never enters. The
    ``label_smoothing`` applies to the CE term only. Stage 3A Plus Task 5: the
    defaults ``lambda_brier=0.0, label_smoothing=0.0`` reproduce the original
    behaviour bit-for-bit.
    """
    valid = valid_mask.to(outputs["log1p_r_hat"].dtype)
    denom = valid.sum().clamp_min(1.0)

    log1p_true = torch.log1p(r_true.clamp_min(0.0))
    smooth = torch.nn.functional.smooth_l1_loss(
        outputs["log1p_r_hat"], log1p_true, beta=SMOOTH_L1_BETA, reduction="none"
    )
    reg_term = (smooth * valid).sum() / denom

    ce_kwargs: dict[str, Any] = {}
    if label_smoothing:
        ce_kwargs["label_smoothing"] = float(label_smoothing)
    ce = torch.nn.functional.cross_entropy(
        outputs["regime_logits"].reshape(-1, N_REGIMES),
        regime_labels.reshape(-1).long(),
        weight=class_weights,
        reduction="none",
        **ce_kwargs,
    ).reshape(valid_mask.shape)
    cls_term = (ce * valid).sum() / denom
    total = reg_term + float(lambda_cls) * cls_term
    if lambda_brier:
        total = total + float(lambda_brier) * multiclass_brier_loss(
            outputs["regime_logits"], regime_labels, valid_mask
        )
    return total


# --------------------------------------------------------------------------- #
# Class weights (train-only, persisted)
# --------------------------------------------------------------------------- #
def compute_class_weights(regime_labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Inverse-frequency regime class weights from the train split only.

    Returns a length-4 float32 vector normalised so the mean weight is 1. Classes
    absent from train receive weight 0 (they contribute nothing to the CE).
    """
    labels = np.asarray(regime_labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    counts = np.bincount(labels[valid].reshape(-1), minlength=N_REGIMES).astype(np.float64)
    weights = np.zeros(N_REGIMES, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    if present.any():
        weights = weights / weights[present].mean()
    return weights.astype(np.float32)


# --------------------------------------------------------------------------- #
# Simple baselines (coefficients fit on train only)
# --------------------------------------------------------------------------- #
class ConstantZeroBaseline:
    """Predict R_hat = 0 everywhere."""

    def fit(self, r_true: np.ndarray, features: Mapping[str, np.ndarray], valid_mask: np.ndarray) -> "ConstantZeroBaseline":
        return self

    def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.zeros(tuple(features["eval_shape"]), dtype=np.float64)


class TrainMeanByHorizonBaseline:
    """Predict the per-horizon-step train mean of R for every vehicle."""

    def __init__(self) -> None:
        self.mean_by_horizon: np.ndarray | None = None

    def fit(self, r_true: np.ndarray, features: Mapping[str, np.ndarray], valid_mask: np.ndarray) -> "TrainMeanByHorizonBaseline":
        r = np.asarray(r_true, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        steps = r.shape[1]
        means = np.zeros(steps, dtype=np.float64)
        for t in range(steps):
            vals = r[:, t, :][valid[:, t, :]]
            means[t] = vals.mean() if vals.size else 0.0
        self.mean_by_horizon = means
        return self

    def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        shape = features["eval_shape"]  # (E,T,N)
        out = np.zeros(shape, dtype=np.float64)
        out[:] = self.mean_by_horizon[None, :, None]
        return out


def _design_matrix(features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per (e,t) design rows: cf_action one-hot, cf_duration, horizon.

    The per-vehicle ``initial_distance`` (E,N) is NOT included here; it is added
    per (e,t,n) by the baseline when ``with_distance`` is set, because it varies
    by vehicle and cannot be shared across the row.
    """
    n_eval = int(features["n_eval"])
    horizon = int(features["horizon"])
    cf_action = np.asarray(features["cf_action"], dtype=np.int64)
    cf_duration = np.asarray(features["cf_duration"], dtype=np.float64)
    rows = []
    for e in range(n_eval):
        action_onehot = np.zeros(5)
        action_onehot[int(cf_action[e])] = 1.0
        for t in range(horizon):
            row = list(action_onehot) + [cf_duration[e], float(t)]
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class LinearResponseBaseline:
    """Ridge regression of log1p(R) on action+duration+horizon (+ per-vehicle distance)."""

    def __init__(self, with_distance: bool = False, ridge: float = 1e-4) -> None:
        self.with_distance = bool(with_distance)
        self.ridge = float(ridge)
        self.beta: np.ndarray | None = None

    def _rows(self, features: Mapping[str, np.ndarray]) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Per-(e,t,n) design rows (with intercept); distance column if enabled."""
        shape = tuple(features["eval_shape"])
        episodes, horizon, agents = shape
        base = _design_matrix(features)  # (E*T, 7) shared part
        distance = np.asarray(features.get("initial_distance"), dtype=np.float64)
        rows = []
        for e in range(episodes):
            for t in range(horizon):
                shared = base[e * horizon + t]
                for n in range(agents):
                    row = [1.0] + list(shared)
                    if self.with_distance:
                        row.append(float(distance[e, n]))
                    rows.append(row)
        return np.asarray(rows, dtype=np.float64), shape

    def fit(self, r_true: np.ndarray, features: Mapping[str, np.ndarray], valid_mask: np.ndarray) -> "LinearResponseBaseline":
        r = np.asarray(r_true, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        X, shape = self._rows(features)
        episodes, horizon, agents = shape
        y = np.log1p(r.reshape(-1))
        keep = valid.reshape(-1)
        Xv = X[keep]
        yv = y[keep]
        gram = Xv.T @ Xv + self.ridge * np.eye(Xv.shape[1])
        self.beta = np.linalg.solve(gram, Xv.T @ yv)
        return self

    def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        X, shape = self._rows(features)
        eta = X @ self.beta  # (E*T*N,)
        out = np.expm1(eta).clip(min=0.0)
        return out.reshape(shape)


def make_initial_distance(factual_suffix: np.ndarray) -> np.ndarray:
    """Per-vehicle initial distance of each vehicle to the ego at t=0.

    Returns shape ``(E, N)`` — one distance per vehicle, because the response
    task is per-vehicle and an episode-mean distance would artificially weaken
    any baseline that uses it (inflating the apparent H4 improvement). The ego
    slot and absent vehicles are set to 0.0 (they are excluded by the eval mask).
    """
    factual = np.asarray(factual_suffix, dtype=np.float32)
    xy = factual[:, 0, :, 2:4]  # E,N,2 at first suffix step
    presence = factual[:, 0, :, 0] > 0
    is_ego = factual[:, 0, :, 1] > 0
    ego_xy = xy[:, 0, :]  # slot 0
    dist = np.linalg.norm(xy - ego_xy[:, None, :], axis=-1)  # E,N
    keep = presence & ~is_ego
    return np.where(keep, dist, 0.0).astype(np.float64)


# --------------------------------------------------------------------------- #
# Classification baselines (fit on train only) -> 4-class regime probabilities
# --------------------------------------------------------------------------- #
def _broadcast_episode_probs(episode_probs: np.ndarray, eval_shape: tuple[int, int, int]) -> np.ndarray:
    """Broadcast per-episode 4-class probs (E,4) to (E,T,N,4)."""
    episodes, horizon, agents = eval_shape
    out = np.zeros((episodes, horizon, agents, episode_probs.shape[-1]), dtype=np.float64)
    out[:] = episode_probs[:, None, None, :]
    return out


class TrainPrevalenceClassifier:
    """Predict the train regime prevalence for every vehicle/time."""

    def __init__(self) -> None:
        self.prevalence: np.ndarray | None = None

    def fit(self, regime_labels: np.ndarray, features: Mapping[str, np.ndarray], valid_mask: np.ndarray) -> "TrainPrevalenceClassifier":
        labels = np.asarray(regime_labels, dtype=np.int64)
        valid = np.asarray(valid_mask, dtype=bool)
        counts = np.bincount(labels[valid].reshape(-1), minlength=N_REGIMES).astype(np.float64)
        total = counts.sum()
        self.prevalence = counts / total if total > 0 else np.full(N_REGIMES, 1.0 / N_REGIMES)
        return self

    def predict_proba(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        shape = tuple(features["eval_shape"])
        episodes = shape[0]
        episode_probs = np.broadcast_to(self.prevalence[None, :], (episodes, N_REGIMES)).copy()
        return _broadcast_episode_probs(episode_probs, shape)


class MultinomialRegimeClassifier:
    """Per-(cf_action, cf_duration, horizon-step) train regime distribution.

    A multinomial (histogram) classifier over the action+duration+horizon cells,
    optionally adding an initial-distance binary split. Cells unseen in train
    fall back to the global train prevalence. Coefficients come from train only.
    """

    def __init__(self, with_distance: bool = False, alpha: float = 1.0) -> None:
        self.with_distance = bool(with_distance)
        self.alpha = float(alpha)
        self.global_prevalence: np.ndarray | None = None
        self.cell_probs: dict[tuple, np.ndarray] = {}
        self.distance_median: float = 0.0

    def _cell_key(self, cf_action: int, cf_duration: float, horizon_step: int, distance: float) -> tuple:
        key = (int(cf_action), int(cf_duration), int(horizon_step))
        if self.with_distance:
            key = key + (int(distance > self.distance_median),)
        return key

    def fit(self, regime_labels: np.ndarray, features: Mapping[str, np.ndarray], valid_mask: np.ndarray) -> "MultinomialRegimeClassifier":
        labels = np.asarray(regime_labels, dtype=np.int64)
        valid = np.asarray(valid_mask, dtype=bool)
        episodes, horizon, agents = labels.shape
        cf_action = np.asarray(features["cf_action"], dtype=np.int64)
        cf_duration = np.asarray(features["cf_duration"], dtype=np.float64)
        distance = np.asarray(features.get("initial_distance"), dtype=np.float64)  # (E,N) per-vehicle
        if self.with_distance:
            valid_dists = np.array([distance[e, n] for e in range(episodes) for n in range(agents) if valid[e, :, n].any()])
            self.distance_median = float(np.median(valid_dists)) if valid_dists.size else 0.0
        counts: dict[tuple, np.ndarray] = {}
        global_counts = np.zeros(N_REGIMES, dtype=np.float64)
        for e in range(episodes):
            for t in range(horizon):
                for n in range(agents):
                    if not valid[e, t, n]:
                        continue
                    key = self._cell_key(cf_action[e], cf_duration[e], t, distance[e, n])
                    cell = counts.setdefault(key, np.zeros(N_REGIMES, dtype=np.float64))
                    cell[int(labels[e, t, n])] += 1.0
                    global_counts[int(labels[e, t, n])] += 1.0
        total = global_counts.sum()
        self.global_prevalence = global_counts / total if total > 0 else np.full(N_REGIMES, 1.0 / N_REGIMES)
        self.cell_probs = {}
        for key, cell in counts.items():
            smooth = cell + self.alpha
            self.cell_probs[key] = smooth / smooth.sum()
        return self

    def predict_proba(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        shape = tuple(features["eval_shape"])
        episodes, horizon, agents = shape
        cf_action = np.asarray(features["cf_action"], dtype=np.int64)
        cf_duration = np.asarray(features["cf_duration"], dtype=np.float64)
        distance = np.asarray(features.get("initial_distance"), dtype=np.float64)  # (E,N) per-vehicle
        out = np.zeros((episodes, horizon, agents, N_REGIMES), dtype=np.float64)
        for e in range(episodes):
            for t in range(horizon):
                for n in range(agents):
                    key = self._cell_key(cf_action[e], cf_duration[e], t, distance[e, n])
                    out[e, t, n, :] = self.cell_probs.get(key, self.global_prevalence)
        return out


# --------------------------------------------------------------------------- #
# Calibration and discrimination metrics (float64, episode-maskable)
# --------------------------------------------------------------------------- #
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=axis, keepdims=True)


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return np.asarray(logits, dtype=np.float64) / float(temperature)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> float:
    """Fit a single temperature on the VALIDATION split by minimising NLL.

    A bounded coarse-to-fine grid search keeps this deterministic and free of
    any test-split information.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    flat_logits = logits[valid]
    flat_labels = labels[valid]
    if flat_logits.size == 0:
        raise ValueError("fit_temperature needs valid logits")
    if grid is None:
        grid = np.concatenate(
            [np.linspace(0.25, 4.0, 31), np.linspace(4.25, 10.0, 24)]
        )

    def nll(temp: float) -> float:
        probs = softmax(flat_logits / temp, axis=-1)
        p = np.clip(probs[np.arange(len(flat_labels)), flat_labels], 1e-12, 1.0)
        return float(-np.log(p).mean())

    best = min(grid, key=nll)
    # Refine around the best grid point.
    lo, hi = max(1e-2, best * 0.8), best * 1.2
    fine = np.linspace(lo, hi, 21)
    best = min(fine, key=nll)
    return float(best)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_bins: int = 15,
    equal_mass: bool = True,
) -> float:
    """Top-label ECE. Equal-mass uses confidence quantile bins (equal count)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    conf = probs.max(axis=-1)[valid]
    pred = probs.argmax(axis=-1)[valid]
    truth = labels[valid]
    n = conf.shape[0]
    if n == 0:
        raise ValueError("ECE needs valid predictions")
    correct = (pred == truth).astype(np.float64)
    if equal_mass:
        order = np.argsort(conf, kind="stable")
        bins = np.array_split(order, n_bins)
        ece = 0.0
        for idx in bins:
            if idx.size == 0:
                continue
            ece += (idx.size / n) * abs(correct[idx].mean() - conf[idx].mean())
        return float(ece)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
        if sel.any():
            ece += (sel.sum() / n) * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def macro_brier(probs: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray) -> float:
    """Macro-averaged Brier score across the 4 regime classes (lower is better)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    onehot = np.eye(probs.shape[-1], dtype=np.float64)[labels]  # E,T,N,4
    sq = (probs - onehot) ** 2  # E,T,N,4
    per_class = []
    for c in range(probs.shape[-1]):
        vals = sq[..., c][valid]
        per_class.append(vals.mean() if vals.size else 0.0)
    return float(np.mean(per_class))


def pr_auc(scores: np.ndarray, binary_labels: np.ndarray, valid_mask: np.ndarray) -> float:
    """Area under the precision-recall curve (average-precision form).

    Returns NaN when there are no positives in the valid set (a degenerate draw
    the bootstrap layer treats as a non-finite replicate).
    """
    scores = np.asarray(scores, dtype=np.float64)[valid_mask]
    labels = np.asarray(binary_labels, dtype=np.int64)[valid_mask]
    positives = labels == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # Average precision: sum over recall steps of precision * delta-recall.
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


# --------------------------------------------------------------------------- #
# Truth bundle for a split (supervision, never a model input)
# --------------------------------------------------------------------------- #
def build_supervision(
    r_pos: np.ndarray,
    thresholds: Mapping[str, Any] = FROZEN_THRESHOLDS,
) -> dict[str, np.ndarray]:
    """Return ``R_pos`` and regime labels used ONLY as training supervision."""
    return {
        "r_true": np.asarray(r_pos, dtype=np.float32),
        "regime_labels": compute_regime_labels(r_pos, thresholds),
    }
