"""Stage 4A risk labels, legal features and the frozen candidate risk models.

The risk TARGET is the episode-level true effect error ``risk_target(e) =
mean_s effect_error_pair(e, s)`` (plan section 5.1); the statistical unit is always
the episode. Risk FEATURES are restricted to inference-time-legal signals
(plan section 5.2): cross-seed prediction dispersion ``U_seed``, the Stage 3A Plus
response summary ``R_hat``, the Full/History expert disagreement ``D_expert``, a
pre-registered ``geometry`` block and ``metadata``. No feature may be derived from
the counterfactual truth, the true response, the true error, an oracle weight or
which expert was better.

:func:`build_legal_risk_features` deliberately has NO ``counterfactual_xy`` /
``true_response`` / ``true_error`` / ``cea`` / ``cew`` / ``oracle`` parameter, so a
truth-side leak is a type error, not a silent bug (plan section 6.2).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# Feature groupings (plan section 5.3). The order within each group is frozen.
UNCERTAINTY_FEATURES = ("U_seed_mean", "U_seed_max")
DISAGREEMENT_FEATURES = ("D_expert_mean", "D_expert_max")
RESPONSE_FEATURES = (
    "rhat_mean", "rhat_max", "rhat_q50", "rhat_q90",
    "presence_mean", "presence_max", "presence_q50", "presence_q90",
    "high_mean", "high_max", "high_q50", "high_q90",
)
GEOMETRY_FEATURES = (
    "geo_dist_mean", "geo_dist_min", "geo_dist_max",
    "geo_relspeed_mean", "geo_relspeed_max", "geo_visible_count",
)
METADATA_FEATURES = (
    "action_0", "action_1", "action_2", "action_3", "action_4",
    "duration", "horizon",
)
ALL_FEATURES = (
    UNCERTAINTY_FEATURES + DISAGREEMENT_FEATURES + RESPONSE_FEATURES
    + GEOMETRY_FEATURES + METADATA_FEATURES
)

# Candidate identities (plan section 5.3).
CANDIDATE_UNCERTAINTY = "UNCERTAINTY_ONLY"
CANDIDATE_RESPONSE = "RESPONSE_ONLY"
CANDIDATE_DISAGREEMENT = "DISAGREEMENT_ONLY"
CANDIDATE_COMBINED = "COMBINED_RISK_HEAD"
BASELINE_GEOMETRY = "GEOMETRY_ONLY"
# The two standard baselines whose lower-validation-AURC winner is the
# STRONGEST_STANDARD_BASELINE (tie -> UNCERTAINTY_ONLY).
STANDARD_BASELINES = (CANDIDATE_UNCERTAINTY, BASELINE_GEOMETRY)
# The three developed candidates compared against the standard baseline.
DEVELOPED_CANDIDATES = (CANDIDATE_RESPONSE, CANDIDATE_DISAGREEMENT, CANDIDATE_COMBINED)
# COMBINED sub-implementations, in tie-break (simpler-first) order.
COMBINED_IMPLEMENTATIONS = ("ridge", "histgb", "mlp")

# Design-matrix feature set per candidate / baseline.
CANDIDATE_FEATURES = {
    CANDIDATE_UNCERTAINTY: UNCERTAINTY_FEATURES,
    CANDIDATE_DISAGREEMENT: DISAGREEMENT_FEATURES,
    CANDIDATE_RESPONSE: RESPONSE_FEATURES,
    BASELINE_GEOMETRY: GEOMETRY_FEATURES + METADATA_FEATURES,
    CANDIDATE_COMBINED: ALL_FEATURES,
}
# Candidates whose score is a direct (unfitted) pre-registered aggregation.
DIRECT_SCORE_FEATURE = {
    CANDIDATE_UNCERTAINTY: "U_seed_mean",
    CANDIDATE_DISAGREEMENT: "D_expert_mean",
}


# --- labels ---------------------------------------------------------------------


def build_effect_target(
    factual_xy: np.ndarray, counterfactual_xy: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """True effect ``E×T×N×2 = counterfactual_xy - factual_xy`` plus the eval mask.

    ``factual_xy`` / ``counterfactual_xy`` are the position (x,y) slices of the
    factual / counterfactual futures. The mask is returned unchanged so callers
    keep a single source of truth for valid agent-times.
    """
    factual_xy = np.asarray(factual_xy, dtype=np.float32)
    counterfactual_xy = np.asarray(counterfactual_xy, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if factual_xy.shape != counterfactual_xy.shape or factual_xy.shape[-1] != 2:
        raise ValueError("factual_xy and counterfactual_xy must align as E,T,N,2")
    if valid.shape != factual_xy.shape[:-1]:
        raise ValueError("valid_mask must align with E,T,N")
    effect = (counterfactual_xy - factual_xy).astype(np.float32)
    return effect, valid


def episode_effect_ade(
    pred_effect: np.ndarray, true_effect: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Per-episode effect ADE: mean valid-agent-time L2 error -> ``(E,)``.

    ``pred_effect`` may be ``E×T×N×2`` (single predictor) or ``S×E×T×N×2`` (per
    seed-pair), in which case the result is ``(S, E)``.
    """
    pred = np.asarray(pred_effect, dtype=np.float64)
    true = np.asarray(true_effect, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if pred.shape == true.shape:
        pred = pred[None, ...]  # single predictor -> one row
    if pred.ndim != 5 or pred.shape[1:] != true.shape:
        raise ValueError("pred_effect must be E,T,N,2 or S,E,T,N,2 aligned to true_effect")
    if valid.shape != true.shape[:-1]:
        raise ValueError("valid_mask must align with E,T,N")
    distance = np.linalg.norm(pred - true[None, ...], axis=-1)  # S,E,T,N
    mask = valid[None, ...]
    count = mask.sum(axis=(2, 3))
    if (count <= 0).any():
        raise ValueError("every episode must contain at least one valid cell")
    return (distance * mask).sum(axis=(2, 3)) / count  # S,E


def risk_target_from_pairs(effect_error_pair: np.ndarray) -> np.ndarray:
    """Primary risk label ``risk_target(e) = mean_s effect_error_pair(e, s)``."""
    errors = np.asarray(effect_error_pair, dtype=np.float64)
    if errors.ndim != 2:
        raise ValueError("effect_error_pair must align as S,E")
    if not np.isfinite(errors).all():
        raise ValueError("effect_error_pair must be finite")
    return errors.mean(axis=0)


# --- legal features ----------------------------------------------------------------


def _valid_cells(valid_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 3:
        raise ValueError("valid_mask must be E,T,N")
    return valid


def _cross_seed_dispersion(primary_predictions: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode mean/max cross-seed std of the PRIMARY predictions (E,)."""
    dispersion = np.asarray(primary_predictions, dtype=np.float64).std(axis=0)  # E,T,N,2
    magnitude = np.linalg.norm(dispersion, axis=-1)  # E,T,N
    masked = np.where(valid, magnitude, np.nan)
    mean = np.nanmean(masked, axis=(1, 2))
    maxv = np.nanmax(masked, axis=(1, 2))
    return mean, maxv


def _expert_disagreement(
    full_predictions: np.ndarray, history_predictions: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode mean/max Full-vs-History L2 disagreement across pairs (E,)."""
    diff = np.linalg.norm(
        np.asarray(full_predictions, dtype=np.float64)
        - np.asarray(history_predictions, dtype=np.float64),
        axis=-1,
    )  # S,E,T,N
    masked = np.where(valid[None, ...], diff, np.nan)  # S,E,T,N
    per_pair_mean = np.nanmean(masked, axis=(2, 3))  # S,E
    per_pair_max = np.nanmax(masked, axis=(2, 3))  # S,E
    return per_pair_mean.mean(axis=0), per_pair_max.max(axis=0)


def _signal_summary(values: np.ndarray, valid: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Per-episode mean/max/q50/q90 of a ``S,E,T,N`` signal over valid cells."""
    masked = np.where(valid[None, ...], np.asarray(values, dtype=np.float64), np.nan)
    flat = masked.reshape(masked.shape[0], masked.shape[1], -1)  # S,E,cells
    return {
        f"{prefix}_mean": np.nanmean(np.nanmean(flat, axis=2), axis=0),
        f"{prefix}_max": np.nanmax(np.nanmax(flat, axis=2), axis=0),
        f"{prefix}_q50": np.nanmean(np.nanquantile(flat, 0.50, axis=2), axis=0),
        f"{prefix}_q90": np.nanmean(np.nanquantile(flat, 0.90, axis=2), axis=0),
    }


def build_legal_risk_features(
    primary_predictions: np.ndarray,
    full_predictions: np.ndarray,
    history_predictions: np.ndarray,
    response_predictions: Mapping[str, np.ndarray],
    pre_intervention_geometry: Mapping[str, np.ndarray],
    action: np.ndarray,
    duration: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the frozen per-episode legal risk features (plan section 5.2).

    Every output is a per-episode vector aligned with the episode axis. The
    signature intentionally refuses any truth-side input.
    """
    valid = _valid_cells(valid_mask)
    episodes, _, agents = valid.shape
    primary = np.asarray(primary_predictions, dtype=np.float64)
    full = np.asarray(full_predictions, dtype=np.float64)
    history = np.asarray(history_predictions, dtype=np.float64)
    for name, block in (("primary", primary), ("full", full), ("history", history)):
        if block.ndim != 5 or block.shape[1:] != (episodes, valid.shape[1], agents, 2):
            raise ValueError(f"{name}_predictions must align as S,E,T,N,2")

    features: dict[str, np.ndarray] = {}
    u_mean, u_max = _cross_seed_dispersion(primary, valid)
    features["U_seed_mean"], features["U_seed_max"] = u_mean, u_max
    d_mean, d_max = _expert_disagreement(full, history, valid)
    features["D_expert_mean"], features["D_expert_max"] = d_mean, d_max

    for key, prefix in (
        ("r_hat", "rhat"),
        ("presence_prob", "presence"),
        ("high_prob", "high"),
    ):
        signal = np.asarray(response_predictions[key], dtype=np.float64)
        if signal.shape != (primary.shape[0], episodes, valid.shape[1], agents):
            raise ValueError(f"response_predictions[{key!r}] must align as S,E,T,N")
        features.update(_signal_summary(signal, valid, prefix))

    features.update(_geometry_summary(pre_intervention_geometry, episodes))

    action = np.asarray(action, dtype=np.int64).ravel()
    duration = np.asarray(duration, dtype=np.float64).ravel()
    if action.shape != (episodes,) or duration.shape != (episodes,):
        raise ValueError("action and duration must be (E,)")
    for code in range(5):
        features[f"action_{code}"] = (action == code).astype(np.float64)
    features["duration"] = duration
    features["horizon"] = np.full(episodes, float(valid.shape[1]), dtype=np.float64)

    for name, value in features.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (episodes,) or not np.isfinite(arr).all():
            raise ValueError(f"feature {name!r} must be a finite (E,) vector")
        features[name] = arr
    return features


def _geometry_summary(
    geometry: Mapping[str, np.ndarray], episodes: int
) -> dict[str, np.ndarray]:
    """Aggregate the pre-intervention per-vehicle geometry into per-episode means.

    ``geometry`` carries ``distance_to_ego`` (E,N), ``relative_speed`` (E,N) and
    ``visible_count`` (E,) — all computed BEFORE the intervention from the factual
    prefix / history only. Missing vehicles must be NaN so they drop out.
    """
    distance = np.asarray(geometry["distance_to_ego"], dtype=np.float64)
    relspeed = np.asarray(geometry["relative_speed"], dtype=np.float64)
    visible = np.asarray(geometry["visible_count"], dtype=np.float64).ravel()
    if distance.shape != (episodes, distance.shape[1]) or distance.shape[0] != episodes:
        raise ValueError("distance_to_ego must be E,N")
    if relspeed.shape != distance.shape:
        raise ValueError("relative_speed must align with distance_to_ego as E,N")
    if visible.shape != (episodes,):
        raise ValueError("visible_count must be (E,)")
    return {
        "geo_dist_mean": np.nanmean(distance, axis=1),
        "geo_dist_min": np.nanmin(distance, axis=1),
        "geo_dist_max": np.nanmax(distance, axis=1),
        "geo_relspeed_mean": np.nanmean(relspeed, axis=1),
        "geo_relspeed_max": np.nanmax(relspeed, axis=1),
        "geo_visible_count": visible,
    }


def design_matrix(features: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    """Stack the named per-episode features into an ``E×F`` design matrix."""
    columns = [np.asarray(features[name], dtype=np.float64) for name in names]
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("design matrix contains non-finite values")
    return matrix


# --- candidate risk models ------------------------------------------------------------


class FittedRiskModel:
    """A fitted risk scorer; ``predict`` returns a per-episode risk (higher=riskier).

    Feature standardization and target winsorization/standardization are fit on
    RISK-TRAIN only and frozen; evaluation sets reuse the frozen statistics and
    never refit or re-clip the TRUE evaluation errors.
    """

    def __init__(
        self,
        implementation: str,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: float,
        target_std: float,
        winsor_bounds: tuple[float, float],
        estimator: Any,
    ) -> None:
        self.implementation = implementation
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.winsor_bounds = (float(winsor_bounds[0]), float(winsor_bounds[1]))
        self.estimator = estimator

    def _standardize(self, matrix: np.ndarray) -> np.ndarray:
        return (np.asarray(matrix, dtype=np.float64) - self.feature_mean) / self.feature_std

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        standardized = self._standardize(matrix)
        raw = self.estimator.predict(standardized)
        return np.asarray(raw, dtype=np.float64) * self.target_std + self.target_mean

    # -- persistence (fit-once / reuse; never refit on eval sets) --------------
    def save(self, path: "str | Path") -> None:
        """Serialize the fitted model (frozen feature/target stats + estimator)."""
        from pathlib import Path as _Path

        import torch

        target = _Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "implementation": self.implementation,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "winsor_bounds": self.winsor_bounds,
                "estimator": self.estimator,
            },
            temporary,
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path: "str | Path") -> "FittedRiskModel":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            implementation=payload["implementation"],
            feature_mean=payload["feature_mean"],
            feature_std=payload["feature_std"],
            target_mean=payload["target_mean"],
            target_std=payload["target_std"],
            winsor_bounds=tuple(payload["winsor_bounds"]),
            estimator=payload["estimator"],
        )


def _fit_estimator(
    implementation: str, matrix: np.ndarray, target: np.ndarray, config: Mapping[str, Any]
):
    implementation = str(implementation)
    if implementation == "ridge":
        from sklearn.linear_model import Ridge

        estimator = Ridge(alpha=float(config.get("ridge_alpha", 1.0)))
        estimator.fit(matrix, target)
        return estimator
    if implementation == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = dict(config.get("histgb", {}))
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_iter=int(params.get("max_iter", 200)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            early_stopping=bool(params.get("early_stopping", False)),
        )
        estimator.fit(matrix, target)
        return estimator
    if implementation == "mlp":
        return _fit_mlp(matrix, target, config)
    raise ValueError(f"unknown combined implementation: {implementation}")


class _TorchEstimator:
    """Module-level picklable wrapper holding a trained torch module.

    Defined at module scope (not as a closure) so ``FittedRiskModel.save`` can
    serialize it with ``torch.save``. It holds the trained model explicitly and
    changes no architecture, optimization, feature or scientific behaviour.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.model(torch.as_tensor(matrix, dtype=torch.float32)).view(-1).numpy()


def _fit_mlp(matrix: np.ndarray, target: np.ndarray, config: Mapping[str, Any]):
    """Single-hidden-layer MLP: hidden=32, ReLU, dropout=0, Adam 1e-3, batch=64,
    at most 100 epochs, patience=10, SmoothL1 loss (plan section 5.3)."""
    import torch
    from torch import nn

    params = dict(config.get("mlp", {}))
    hidden = int(params.get("hidden_dim", 32))
    lr = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 64))
    max_epochs = int(params.get("max_epochs", 100))
    patience = int(params.get("patience", 10))

    torch.manual_seed(20260911)
    device = torch.device("cpu")
    x = torch.as_tensor(matrix, dtype=torch.float32)
    y = torch.as_tensor(target, dtype=torch.float32).view(-1, 1)
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    # Internal 90/10 split for early stopping (deterministic; train-only data).
    n = x.shape[0]
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(20260911))
    n_val = max(1, int(round(0.1 * n)))
    val_idx, tr_idx = permutation[:n_val], permutation[n_val:]
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    best_state = None
    best_val = float("inf")
    wait = 0
    for _ in range(max_epochs):
        model.train()
        order = torch.randperm(x_tr.shape[0], generator=torch.Generator().manual_seed(20260911))
        for start in range(0, x_tr.shape[0], batch_size):
            sel = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[sel]), y_tr[sel])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val))
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return _TorchEstimator(model)


def fit_risk_candidate(
    name: str,
    train_features: Mapping[str, np.ndarray],
    train_target: np.ndarray,
    config: Mapping[str, Any],
    *,
    implementation: str | None = None,
) -> FittedRiskModel:
    """Fit one risk candidate on RISK-TRAIN features/target only.

    ``name`` selects the frozen feature set (``CANDIDATE_FEATURES``); direct
    candidates are handled by the caller via :func:`direct_risk_score` and never
    reach this fitter. ``implementation`` picks the COMBINED sub-model
    (``ridge``/``histgb``/``mlp``); the response/geometry baselines always use
    ``ridge``.
    """
    if name in DIRECT_SCORE_FEATURE:
        raise ValueError(f"{name} is a direct (unfitted) candidate; use direct_risk_score")
    if name not in CANDIDATE_FEATURES:
        raise ValueError(f"unknown risk candidate: {name}")
    impl = implementation or ("ridge" if name != CANDIDATE_COMBINED else None)
    if impl is None:
        raise ValueError("COMBINED_RISK_HEAD requires an explicit implementation")
    if name != CANDIDATE_COMBINED and impl != "ridge":
        raise ValueError(f"{name} is frozen to the ridge implementation")

    matrix = design_matrix(train_features, CANDIDATE_FEATURES[name])
    target = np.asarray(train_target, dtype=np.float64).ravel()
    if matrix.shape[0] != target.shape[0] or target.size < 2:
        raise ValueError("train features and target must align with >=2 episodes")
    if not np.isfinite(target).all():
        raise ValueError("train target must be finite")

    feature_mean = matrix.mean(axis=0)
    feature_std = np.maximum(matrix.std(axis=0), 1e-12)
    lower_q = float(config.get("winsor_lower_q", 0.01))
    upper_q = float(config.get("winsor_upper_q", 0.99))
    winsor_low, winsor_high = np.quantile(target, [lower_q, upper_q])
    target_w = np.clip(target, winsor_low, winsor_high)
    target_mean = float(target_w.mean())
    target_std = float(max(target_w.std(), 1e-12))
    standardized_target = (target_w - target_mean) / target_std
    standardized_matrix = (matrix - feature_mean) / feature_std

    estimator = _fit_estimator(impl, standardized_matrix, standardized_target, config)
    return FittedRiskModel(
        implementation=impl,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        winsor_bounds=(float(winsor_low), float(winsor_high)),
        estimator=estimator,
    )


def direct_risk_score(name: str, features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Direct (unfitted) candidate score: the pre-registered mean aggregation."""
    if name not in DIRECT_SCORE_FEATURE:
        raise ValueError(f"{name} is not a direct candidate")
    score = np.asarray(features[DIRECT_SCORE_FEATURE[name]], dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("direct risk score must be finite")
    return score


def candidate_risk_scores(
    name: str,
    features: Mapping[str, np.ndarray],
    *,
    model: FittedRiskModel | None = None,
) -> np.ndarray:
    """Score one candidate on a feature set (direct or fitted)."""
    if name in DIRECT_SCORE_FEATURE:
        return direct_risk_score(name, features)
    if model is None:
        raise ValueError(f"fitted candidate {name} requires a model")
    return model.predict(design_matrix(features, CANDIDATE_FEATURES[name]))


__all__ = [
    "ALL_FEATURES",
    "BASELINE_GEOMETRY",
    "CANDIDATE_COMBINED",
    "CANDIDATE_DISAGREEMENT",
    "CANDIDATE_FEATURES",
    "CANDIDATE_RESPONSE",
    "CANDIDATE_UNCERTAINTY",
    "COMBINED_IMPLEMENTATIONS",
    "DEVELOPED_CANDIDATES",
    "DIRECT_SCORE_FEATURE",
    "DISAGREEMENT_FEATURES",
    "FittedRiskModel",
    "GEOMETRY_FEATURES",
    "METADATA_FEATURES",
    "RESPONSE_FEATURES",
    "STANDARD_BASELINES",
    "UNCERTAINTY_FEATURES",
    "build_effect_target",
    "build_legal_risk_features",
    "candidate_risk_scores",
    "design_matrix",
    "direct_risk_score",
    "episode_effect_ade",
    "fit_risk_candidate",
    "risk_target_from_pairs",
]
