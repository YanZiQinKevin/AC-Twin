"""Stage 4C Plus decision-aware relative-comparison features (plan section 3).

The frozen ensemble predictor selects a reference action ``a0 = argmin_a J_hat``; every
other action becomes a *relative comparison token* against ``a0``. The supervision target
is decision regret, but NOTHING here reads the counterfactual truth, ``J_true``, regret,
the true response, the true error or the oracle action — the public builders deliberately
have no truth-side parameter, so a leak is a type error, not a silent bug (plan 3.1).

Feature layout (frozen; recorded verbatim in ``feature_schema.json``):
  * 9 competitor tokens x 34 dims = 20 continuous + 14 one-hot (action/duration semantics);
  * 9 global dims (all continuous).
Continuous means/stds are fit on the train role ONLY (:class:`FrozenFeatureScaler`);
one-hot blocks are never standardised. A group with any missing/non-finite feature is
rejected outright (no zero-padding).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_protocol as s4cp

# --- frozen feature schema (plan section 3.2) ---------------------------------------------

#: per-token continuous features (20), in fixed order.
TOKEN_CONT_NAMES: tuple[str, ...] = (
    "delta_j_mean", "delta_j_seed_std", "beat_vote", "rank_mean", "rank_std",
) + tuple(
    f"{kind}_risk_{k}"
    for k in s4cp.RISK_CLASSES
    for kind in ("selected", "competitor", "delta")
)

#: per-token categorical one-hot features (14), in fixed order.
TOKEN_CAT_NAMES: tuple[str, ...] = (
    tuple(f"selected_action_{a}" for a in range(5))
    + ("selected_duration_1", "selected_duration_3")
    + tuple(f"competitor_action_{a}" for a in range(5))
    + ("competitor_duration_1", "competitor_duration_3")
)

TOKEN_NAMES: tuple[str, ...] = TOKEN_CONT_NAMES + TOKEN_CAT_NAMES
TOKEN_DIM = len(TOKEN_NAMES)              # 34
N_TOKEN_CONT = len(TOKEN_CONT_NAMES)      # 20
TOKEN_CONT_IDX = np.arange(0, N_TOKEN_CONT)
TOKEN_CAT_IDX = np.arange(N_TOKEN_CONT, TOKEN_DIM)

#: global (per-group) continuous features (9), in fixed order.
GLOBAL_NAMES: tuple[str, ...] = (
    "top1_top2_margin", "top1_top3_margin", "seed_vote_entropy", "seed_top1_agreement",
    "mean_pairwise_rank_spearman", "preintervention_min_distance",
    "preintervention_mean_relative_speed", "visible_vehicle_count", "t_star_normalized",
)
GLOBAL_DIM = len(GLOBAL_NAMES)            # 9

_DURATION_LEVELS = (1, 3)


def feature_schema() -> dict[str, Any]:
    """The frozen feature schema persisted in ``feature_schema.json``."""
    return {
        "plan_id": s4cp.PLAN_ID,
        "n_actions": s4cp.N_ACTIONS,
        "n_competitors": s4cp.N_COMPETITORS,
        "token_dim": TOKEN_DIM,
        "token_continuous_names": list(TOKEN_CONT_NAMES),
        "token_categorical_names": list(TOKEN_CAT_NAMES),
        "global_names": list(GLOBAL_NAMES),
        "global_dim": GLOBAL_DIM,
        "risk_classes": list(s4cp.RISK_CLASSES),
        "duration_levels": list(_DURATION_LEVELS),
        "high_regret_threshold": s4cp.HIGH_REGRET_THRESHOLD,
    }


# --- predicted consequences (ensemble + per-seed) ----------------------------------------


def compute_predicted_costs(
    e2e: np.ndarray,
    factual_xy: np.ndarray,
    history_last_xy: np.ndarray,
    inference_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed ``J_hat_s`` (S,G,A) and ensemble ``J_hat`` (G,A) predicted consequences.

    ``Y_hat_CF_s = Y_F + effect_hat_s`` per seed; the ensemble uses ``mean_s effect_hat``.
    The frozen Stage 4C consequence function is applied identically to every seed and to
    the ensemble, so ``J_hat`` is bit-isomorphic to Stage 4C. No truth is read.
    """
    e2e = np.asarray(e2e, dtype=np.float64)
    if e2e.ndim != 6 or e2e.shape[-1] != 2:
        raise ValueError(f"e2e must be (S,G,A,T,N,2); got {e2e.shape}")
    seeds, groups, n_actions = e2e.shape[0], e2e.shape[1], e2e.shape[2]
    if n_actions != s4cp.N_ACTIONS:
        raise ValueError(f"expected {s4cp.N_ACTIONS} actions, got {n_actions}")
    factual_xy = np.asarray(factual_xy, dtype=np.float64)
    history_last_xy = np.asarray(history_last_xy, dtype=np.float64)
    inference_mask = np.asarray(inference_mask, dtype=bool)

    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(inference_mask[:, None, :, :],
                             (groups, n_actions) + inference_mask.shape[1:])

    j_hat_seeds = np.empty((seeds, groups, n_actions), dtype=np.float64)
    for s in range(seeds):
        y_s = factual_xy[:, None] + e2e[s]
        comp = s4co.trajectory_components(y_s, hist_b, mask_b)
        j_hat_seeds[s] = s4co.total_cost(
            comp["interaction"], s4co.efficiency_cost(comp["progress"]))
    y_ens = factual_xy[:, None] + e2e.mean(axis=0)
    comp = s4co.trajectory_components(y_ens, hist_b, mask_b)
    j_hat = s4co.total_cost(comp["interaction"], s4co.efficiency_cost(comp["progress"]))
    return j_hat_seeds, j_hat


# --- internal helpers ---------------------------------------------------------------------


def _one_hot(index: np.ndarray, levels: Sequence[int]) -> np.ndarray:
    """One-hot encode integer ``index`` against ``levels`` -> (..., len(levels))."""
    index = np.asarray(index, dtype=np.int64)
    out = np.zeros(index.shape + (len(levels),), dtype=np.float64)
    for col, level in enumerate(levels):
        out[..., col] = (index == level).astype(np.float64)
    return out


def _pairwise_rank_spearman(ranks: np.ndarray) -> np.ndarray:
    """Mean pairwise Spearman across the S seed rankings of the A actions -> (G,).

    ``ranks`` is (S,G,A); Spearman on predicted costs equals Pearson on the ranks. Pairs
    with a degenerate (zero-variance) ranking contribute 0.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    seeds = ranks.shape[0]
    groups = ranks.shape[1]
    centred = ranks - ranks.mean(axis=-1, keepdims=True)      # (S,G,A)
    norm = np.linalg.norm(centred, axis=-1)                    # (S,G)
    total = np.zeros(groups, dtype=np.float64)
    count = 0
    for s in range(seeds):
        for t in range(s + 1, seeds):
            denom = norm[s] * norm[t]
            num = (centred[s] * centred[t]).sum(axis=-1)
            corr = np.where(denom > 0, num / np.maximum(denom, 1e-12), 0.0)
            total += corr
            count += 1
    return total / max(count, 1)


# --- relative comparison tokens + global features ----------------------------------------


def build_relative_tokens(
    e2e: np.ndarray,
    factual_xy: np.ndarray,
    history_last_xy: np.ndarray,
    inference_mask: np.ndarray,
    action_risks: Mapping[str, np.ndarray],
    geometry: Mapping[str, np.ndarray],
    t_star: np.ndarray,
    group_seed: np.ndarray,
) -> dict[str, Any]:
    """Build the 9 relative-comparison tokens + 9 global features for every group.

    All inputs are inference-time legal (predictions, action-level risk scores, geometry,
    ``t_star``). Returns the model-facing bundle plus the derived quantities the
    non-learning baselines and statistics need (ensemble ``J_hat``, reference/top actions,
    margins, seed-vote summaries). Groups with any non-finite feature are marked invalid
    in ``valid_mask`` (caller drops them; no zero-padding).
    """
    j_hat_seeds, j_hat = compute_predicted_costs(
        e2e, factual_xy, history_last_xy, inference_mask)
    seeds, groups, n_actions = j_hat_seeds.shape
    rows = np.arange(groups)

    a0 = np.argmin(j_hat, axis=-1)                                # (G,) reference action
    order = np.argsort(j_hat, axis=-1, kind="stable")             # ascending, plan tie-break
    top1, top2, top3 = order[:, 0], order[:, 1], order[:, 2]

    # per-seed action ranks (1..A, average on ties).
    ranks = np.stack([rankdata(j_hat_seeds[s], axis=1) for s in range(seeds)])  # (S,G,A)

    # --- token continuous features ---------------------------------------------------
    j0 = j_hat[rows, a0]                                          # (G,)
    j0_seeds = j_hat_seeds[:, rows, a0]                           # (S,G)
    delta_j_mean = j_hat - j0[:, None]                            # (G,A)
    delta_seeds = j_hat_seeds - j0_seeds[:, :, None]              # (S,G,A)
    delta_j_seed_std = delta_seeds.std(axis=0)                    # (G,A)
    beat_vote = (j_hat_seeds < j0_seeds[:, :, None]).mean(axis=0)  # (G,A)
    rank_mean = ranks.mean(axis=0)                                # (G,A)
    rank_std = ranks.std(axis=0)                                  # (G,A)

    risk_blocks: list[np.ndarray] = []
    for k in s4cp.RISK_CLASSES:
        r = np.asarray(action_risks[k], dtype=np.float64)         # (G,A)
        if r.shape != (groups, n_actions):
            raise ValueError(f"action risk {k!r} must be (G,A)")
        risk_blocks.append(np.broadcast_to(r[rows, a0][:, None], (groups, n_actions)))  # selected
        risk_blocks.append(r)                                     # competitor
        risk_blocks.append(r - r[rows, a0][:, None])              # delta
    cont = [delta_j_mean, delta_j_seed_std, beat_vote, rank_mean, rank_std] + risk_blocks
    token_cont = np.stack(cont, axis=-1)                          # (G,A,20)

    # --- token categorical one-hots ---------------------------------------------------
    actions = np.asarray([p[0] for p in s4cp.ACTION_PLANS], dtype=np.int64)   # (A,)
    durations = np.asarray([p[1] for p in s4cp.ACTION_PLANS], dtype=np.int64)  # (A,)
    sel_action = _one_hot(actions[a0], range(5))                  # (G,5)
    sel_duration = _one_hot(durations[a0], _DURATION_LEVELS)      # (G,2)
    comp_action = _one_hot(actions, range(5))                     # (A,5)
    comp_duration = _one_hot(durations, _DURATION_LEVELS)         # (A,2)
    token_cat = np.concatenate([
        np.broadcast_to(sel_action[:, None, :], (groups, n_actions, 5)),
        np.broadcast_to(sel_duration[:, None, :], (groups, n_actions, 2)),
        np.broadcast_to(comp_action[None, :, :], (groups, n_actions, 5)),
        np.broadcast_to(comp_duration[None, :, :], (groups, n_actions, 2)),
    ], axis=-1)                                                   # (G,A,14)

    token_full = np.concatenate([token_cont, token_cat], axis=-1)  # (G,A,34)

    # --- keep only the 9 competitors (a != a0), in frozen ACTION_PLANS order ----------
    tokens = np.empty((groups, s4cp.N_COMPETITORS, TOKEN_DIM), dtype=np.float64)
    competitor_index = np.empty((groups, s4cp.N_COMPETITORS), dtype=np.int64)
    for g in range(groups):
        keep = [a for a in range(n_actions) if a != a0[g]]
        tokens[g] = token_full[g, keep]
        competitor_index[g] = np.asarray(keep, dtype=np.int64)

    # --- global features ---------------------------------------------------------------
    top1_top2_margin = j_hat[rows, top2] - j_hat[rows, top1]
    top1_top3_margin = j_hat[rows, top3] - j_hat[rows, top1]
    top1_seeds = np.argmin(j_hat_seeds, axis=-1)                  # (S,G)
    vote_entropy = np.empty(groups, dtype=np.float64)
    top1_agreement = np.empty(groups, dtype=np.float64)
    for g in range(groups):
        counts = np.bincount(top1_seeds[:, g], minlength=n_actions).astype(np.float64)
        p = counts[counts > 0] / seeds
        vote_entropy[g] = float(-(p * np.log(p)).sum())
        top1_agreement[g] = float(counts.max() / seeds)
    mean_pairwise_spearman = _pairwise_rank_spearman(ranks)

    distance = np.asarray(geometry["distance_to_ego"], dtype=np.float64)   # (G,N) NaN=absent
    relspeed = np.asarray(geometry["relative_speed"], dtype=np.float64)
    visible = np.asarray(geometry["visible_count"], dtype=np.float64).ravel()
    # Reduce only rows with at least one finite value so an all-NaN group yields NaN
    # (-> invalid) WITHOUT emitting numpy's "All-NaN slice" / "Mean of empty slice"
    # RuntimeWarning (the suite runs with -W error).
    min_distance = np.full(groups, np.nan)
    mean_relspeed = np.full(groups, np.nan)
    finite_d = np.isfinite(distance).any(axis=1)
    finite_r = np.isfinite(relspeed).any(axis=1)
    if finite_d.any():
        min_distance[finite_d] = np.nanmin(distance[finite_d], axis=1)
    if finite_r.any():
        mean_relspeed[finite_r] = np.nanmean(relspeed[finite_r], axis=1)
    t_star = np.asarray(t_star, dtype=np.float64).ravel()
    t_star_norm = (t_star - s4cp.T_STAR_LOW) / float(s4cp.T_STAR_HIGH - s4cp.T_STAR_LOW)

    global_features = np.column_stack([
        top1_top2_margin, top1_top3_margin, vote_entropy, top1_agreement,
        mean_pairwise_spearman, min_distance, mean_relspeed, visible, t_star_norm,
    ])                                                            # (G,9)

    # --- validity ----------------------------------------------------------------------
    valid_mask = (
        np.isfinite(tokens).all(axis=(1, 2))
        & np.isfinite(global_features).all(axis=1)
    )

    # --- non-learning / historical baseline group risks --------------------------------
    margin_s = np.empty((seeds, groups), dtype=np.float64)
    for s in range(seeds):
        order_s = np.argsort(j_hat_seeds[s], axis=-1, kind="stable")
        margin_s[s] = j_hat_seeds[s][rows, order_s[:, 1]] - j_hat_seeds[s][rows, order_s[:, 0]]
    combined = np.asarray(action_risks["COMBINED_RISK_HEAD"], dtype=np.float64)
    baselines = {
        s4cp.BASELINE_MARGIN: -top1_top2_margin,
        s4cp.BASELINE_SEED_VOTE_ENTROPY: vote_entropy,
        s4cp.BASELINE_SEED_MARGIN_STD: margin_s.std(axis=0),
        s4cp.BASELINE_COMBINED_TOP2_MAX: np.maximum(
            combined[rows, top1], combined[rows, top2]),
    }

    return {
        "tokens": tokens,                                   # (G,9,34)
        "global_features": global_features,                 # (G,9)
        "competitor_index": competitor_index,               # (G,9)
        "token_mask": np.ones((groups, s4cp.N_COMPETITORS), dtype=bool),
        "valid_mask": valid_mask,                           # (G,)
        "group_seed": np.asarray(group_seed, dtype=np.int64).ravel(),
        # derived quantities for baselines / statistics (prediction side only):
        "j_hat": j_hat,                                     # (G,A)
        "j_hat_seeds": j_hat_seeds,                         # (S,G,A)
        "reference_action": a0,                             # (G,)
        "baselines": baselines,                             # dict of (G,)
        "interaction_min_distance": min_distance,           # (G,) for strata threshold
    }


# --- feature scaler (fit on train role ONLY) ----------------------------------------------


class FrozenFeatureScaler:
    """Standardises continuous token/global features; leaves one-hots untouched.

    Fit ONCE on the train role (or the Phase 0 train folds) and frozen; evaluation roles
    reuse the frozen statistics and never refit.
    """

    def __init__(
        self,
        token_cont_mean: np.ndarray,
        token_cont_std: np.ndarray,
        global_mean: np.ndarray,
        global_std: np.ndarray,
    ) -> None:
        self.token_cont_mean = np.asarray(token_cont_mean, dtype=np.float64)
        self.token_cont_std = np.asarray(token_cont_std, dtype=np.float64)
        self.global_mean = np.asarray(global_mean, dtype=np.float64)
        self.global_std = np.asarray(global_std, dtype=np.float64)

    @classmethod
    def fit(cls, tokens: np.ndarray, global_features: np.ndarray) -> "FrozenFeatureScaler":
        tokens = np.asarray(tokens, dtype=np.float64)            # (G,9,34)
        global_features = np.asarray(global_features, dtype=np.float64)  # (G,9)
        cont = tokens[:, :, TOKEN_CONT_IDX].reshape(-1, N_TOKEN_CONT)
        token_mean = cont.mean(axis=0)
        token_std = np.maximum(cont.std(axis=0), 1e-12)
        global_mean = global_features.mean(axis=0)
        global_std = np.maximum(global_features.std(axis=0), 1e-12)
        return cls(token_mean, token_std, global_mean, global_std)

    def transform_tokens(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.float64).copy()
        tokens[:, :, TOKEN_CONT_IDX] = (
            (tokens[:, :, TOKEN_CONT_IDX] - self.token_cont_mean) / self.token_cont_std)
        return tokens

    def transform_global(self, global_features: np.ndarray) -> np.ndarray:
        return (np.asarray(global_features, dtype=np.float64) - self.global_mean) / self.global_std

    def save(self, path: "str | Path") -> None:
        from pathlib import Path as _Path
        from ac_twin.checkpointing import atomic_save_npz
        atomic_save_npz(_Path(path),
                        token_cont_mean=self.token_cont_mean,
                        token_cont_std=self.token_cont_std,
                        global_mean=self.global_mean,
                        global_std=self.global_std)

    @classmethod
    def load(cls, path: "str | Path") -> "FrozenFeatureScaler":
        from pathlib import Path as _Path
        with np.load(_Path(path)) as z:
            return cls(z["token_cont_mean"], z["token_cont_std"],
                       z["global_mean"], z["global_std"])


def fit_feature_scaler(train_bundle: Mapping[str, Any]) -> FrozenFeatureScaler:
    """Fit the scaler on a train bundle's VALID groups only (never on eval roles)."""
    valid = np.asarray(train_bundle["valid_mask"], dtype=bool)
    tokens = np.asarray(train_bundle["tokens"], dtype=np.float64)[valid]
    global_features = np.asarray(train_bundle["global_features"], dtype=np.float64)[valid]
    if tokens.shape[0] < 2:
        raise ValueError("need >=2 valid train groups to fit the scaler")
    return FrozenFeatureScaler.fit(tokens, global_features)


__all__ = [
    "GLOBAL_DIM",
    "GLOBAL_NAMES",
    "FrozenFeatureScaler",
    "N_TOKEN_CONT",
    "TOKEN_CAT_IDX",
    "TOKEN_CAT_NAMES",
    "TOKEN_CONT_IDX",
    "TOKEN_CONT_NAMES",
    "TOKEN_DIM",
    "TOKEN_NAMES",
    "build_relative_tokens",
    "compute_predicted_costs",
    "feature_schema",
    "fit_feature_scaler",
]
