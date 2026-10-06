"""Stage 4C grouped frozen inference, action-level risk and group-level aggregation.

Plan ``AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1`` section 6. The ten
counterfactual branches of each decision group are scored EXACTLY like Stage 4A/4B
episodes: the group tensor is flattened to ``E = G*10`` branches (group-major,
action-minor) and run through the frozen five-seed-pair ``run_pair_inference`` so the
expert/router/response forward passes are bit-identical to the confirmed Stage 4B
chain. Risk FEATURES use the inference mask (history-known AND factual-present AND
non-ego) and never the counterfactual presence, the true trajectory, the true
response, regret or the oracle action.

Group-level risk (plan section 6.2) aggregates each action-level score over the
predicted-best two actions ``max(r(a1), r(a2))``; ``MARGIN_ONLY`` is the negated
predicted cost gap. Predicted consequences ``J_hat`` come from the frozen consequence
function applied to ``Y_hat_CF = Y_F + mean_s effect_hat`` (ensemble over the five
seed pairs). True consequences/regret are NEVER touched here — they are built only at
adjudication, after the matching one-shot lock is burned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import run_stage4a as s4a
from ac_twin import stage4_risk as s4r
from ac_twin import stage4b_inference as s4bi
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4c_protocol as s4cp
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3_scaler import FeatureScaler

POSITION_SLICE = s4a.POSITION_SLICE
DETERMINISM_GROUPS = 32
DETERMINISM_PAIR = 0
DETERMINISM_TOL = 1e-7


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (s4cp.ROOT / value)


# --- grouped data loading + branch expansion ------------------------------------------


def load_group_eval(config: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    """Load a collected Stage 4C group npz into scaled branch tensors + geometry.

    The true counterfactual trajectories are loaded ONLY for the (deferred) adjudication
    side; they are never fed into a risk feature. Read-only.
    """
    role = str(role)
    base = str(config["data_output"])
    if role == "confirm":
        resolved = base.replace(s4cp.PURPOSE_DEV, s4cp.PURPOSE_CONFIRM)
    elif role in ("development_selection", "development_screen"):
        suffix = "_selection" if role == "development_selection" else "_screen"
        resolved = base[:-4] + suffix + ".npz" if base.endswith(".npz") else base + suffix
    else:
        raise ValueError(f"unknown Stage 4C role: {role}")
    data_output = _resolve(resolved)
    s4cp.assert_allowed_stage4c_data(data_output)
    with np.load(data_output, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}

    history = np.asarray(data["object_history"], dtype=np.float32)        # (G,10,5,8)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)  # (G,20,5,8)
    cf = np.asarray(data["object_counterfactual_futures"], dtype=np.float32)  # (G,10,20,5,8)
    cf_actions = np.asarray(data["cf_actions"], dtype=np.int64)            # (G,10)
    cf_durations = np.asarray(data["cf_durations"], dtype=np.int64)        # (G,10)
    factual_actions = np.asarray(data["factual_actions"], dtype=np.int64)  # (G,20)
    t_star = np.asarray(data["t_star"], dtype=np.int64)                    # (G,)
    episode_seed = np.asarray(data["episode_seed"], dtype=np.int64)        # (G,)
    inference_mask = np.asarray(data["inference_mask"], dtype=bool)        # (G,20,5)

    groups = history.shape[0]
    n_actions = cf.shape[1]
    if n_actions != s4cp.N_ACTIONS:
        raise ValueError(f"expected {s4cp.N_ACTIONS} actions per group, got {n_actions}")

    # Expand to E = G*10 branches (group-major, action-minor): branch e = g*10 + a.
    g_idx = np.repeat(np.arange(groups), n_actions)
    a_idx = np.tile(np.arange(n_actions), groups)
    branches = groups * n_actions

    hist_b = history[g_idx]                      # (E,10,5,8)
    fact_b = factual[g_idx]                      # (E,20,5,8)
    factual_suffix = fact_b[:, :10]              # (E,10,5,8)
    factual_actions_b = factual_actions[g_idx]   # (E,20)
    cf_actions_b = cf_actions.reshape(branches)  # (E,)
    cf_durations_b = cf_durations.reshape(branches)  # (E,)
    t_star_b = t_star[g_idx]                     # (E,)
    # materialised counterfactual action plan per branch.
    cf_plans = np.stack([
        s4cp.materialize_action_plan(factual_actions[g], int(cf_actions[g, a]),
                                     int(cf_durations[g, a]))
        for g in range(groups) for a in range(n_actions)
    ]).astype(np.int64)                          # (E,20)
    valid_b = inference_mask[g_idx]              # (E,20,5) shared across the 10 actions

    scaler = FeatureScaler.load(_resolve(config["scaler_path"]))
    tensors = {
        "history": torch.as_tensor(hist_b, dtype=torch.float32),
        "factual_suffix": torch.as_tensor(factual_suffix, dtype=torch.float32),
        "factual_actions": torch.as_tensor(factual_actions_b, dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(cf_plans, dtype=torch.long),
        "cf_action": torch.as_tensor(cf_actions_b, dtype=torch.long),
        "cf_duration": torch.as_tensor(cf_durations_b, dtype=torch.long),
        "t_star": torch.as_tensor(t_star_b, dtype=torch.long),
    }
    scaled = dict(tensors)
    scaled["history"] = torch.as_tensor(scaler.transform(hist_b), dtype=torch.float32)
    scaled["factual_suffix"] = torch.as_tensor(
        scaler.transform(factual_suffix), dtype=torch.float32)

    geometry_group = s4a._pre_intervention_geometry(history)  # per-group (G,...)
    geometry = {
        "distance_to_ego": geometry_group["distance_to_ego"][g_idx],
        "relative_speed": geometry_group["relative_speed"][g_idx],
        "visible_count": geometry_group["visible_count"][g_idx],
    }
    # interaction density: non-ego vehicles within 30 m of the ego at history last frame.
    last = history[:, -1]  # (G,5,8)
    ego_xy = last[:, 0, POSITION_SLICE]  # (G,2)
    dist = np.linalg.norm(last[:, :, POSITION_SLICE] - ego_xy[:, None, :], axis=-1)  # (G,5)
    present = last[..., 0] > 0
    present[:, 0] = False  # exclude ego
    density_count = ((dist <= 30.0) & present).sum(axis=1).astype(np.float64)  # (G,)
    return {
        "role": role,
        "groups": groups,
        "n_actions": n_actions,
        "branches": branches,
        "tensors": tensors,
        "scaled": scaled,
        "valid": valid_b,
        "geometry": geometry,
        "episode_seed": episode_seed,
        "t_star": t_star,
        "inference_mask": inference_mask,
        "density_count": density_count,
        "history_last_xy": history[:, -1, 0, POSITION_SLICE].astype(np.float64),  # (G,2)
        "factual_xy": factual[..., POSITION_SLICE].astype(np.float64),           # (G,20,5,2)
        "counterfactual_xy": cf[..., POSITION_SLICE].astype(np.float64),         # (G,10,20,5,2)
        "data_signature": sha256_file(data_output),
    }


# --- frozen grouped inference -----------------------------------------------------------


def run_grouped_frozen_inference(
    bundle: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    stage3r_dir: Path,
    device: str,
    batch_size: int,
    pair_indices: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, np.ndarray]:
    """Run the frozen five seed pairs over all ``G*10`` branches; reshape to group axes.

    Returns per-pair stacked predictions with shapes ``e2e/full/history``:
    ``(S, G, A, T, N, 2)`` and ``r_hat/presence_prob/high_prob``: ``(S, G, A, T, N)``.
    """
    branches = int(bundle["branches"])
    groups = int(bundle["groups"])
    n_actions = int(bundle["n_actions"])
    indices = np.arange(branches, dtype=np.int64)
    per_pair: dict[str, list[np.ndarray]] = {
        key: [] for key in
        ("e2e", "full", "history", "r_hat", "presence_prob", "high_prob")}
    config = {"device": device, "batch_size": batch_size}
    for pair_index in pair_indices:
        out = s4bi.run_pair_inference(
            config, bundle["scaled"], indices,
            pair_index=pair_index, binding=binding, stage3r_dir=stage3r_dir)
        for key in per_pair:
            per_pair[key].append(np.asarray(out[key], dtype=np.float32))

    def reshape(values: list[np.ndarray], trailing: tuple[int, ...]) -> np.ndarray:
        stacked = np.stack(values).astype(np.float32)  # (S, E, *trailing)
        return stacked.reshape((len(values), groups, n_actions) + trailing)

    horizon = bundle["valid"].shape[1]
    agents = bundle["valid"].shape[2]
    return {
        "e2e": reshape(per_pair["e2e"], (horizon, agents, 2)),
        "full": reshape(per_pair["full"], (horizon, agents, 2)),
        "history": reshape(per_pair["history"], (horizon, agents, 2)),
        "r_hat": reshape(per_pair["r_hat"], (horizon, agents)),
        "presence_prob": reshape(per_pair["presence_prob"], (horizon, agents)),
        "high_prob": reshape(per_pair["high_prob"], (horizon, agents)),
    }


def ensemble_predicted_cost(
    predictions: Mapping[str, np.ndarray], bundle: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Ensemble predicted consequence ``J_hat`` (G, A) from the frozen consequence fn.

    ``Y_hat_CF = Y_F + mean_s effect_hat``; the same inference mask is used. This is
    prediction-side only and legal before any adjudication lock.
    """
    ensemble_effect = np.asarray(predictions["e2e"], dtype=np.float64).mean(axis=0)  # (G,A,T,N,2)
    y_hat_cf = bundle["factual_xy"][:, None] + ensemble_effect
    groups, n_actions = y_hat_cf.shape[0], y_hat_cf.shape[1]
    hist_b = np.broadcast_to(bundle["history_last_xy"][:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(bundle["inference_mask"][:, None, :, :],
                             y_hat_cf.shape[:-1])
    comp = s4co.trajectory_components(y_hat_cf, hist_b, mask_b)
    j_hat = s4co.total_cost(comp["interaction"], s4co.efficiency_cost(comp["progress"]))
    return {"Y_hat_CF": y_hat_cf, "J_hat": j_hat}


# --- action-level risk features + scores --------------------------------------------------


def build_action_risk_features(
    bundle: Mapping[str, Any], predictions: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-branch legal risk features (E = G*10 vectors), inference-mask based.

    Truth-side inputs are impossible by signature: no counterfactual truth, true
    response, true error, regret or oracle action is accepted.
    """
    groups = int(bundle["groups"])
    n_actions = int(bundle["n_actions"])
    branches = groups * n_actions

    def flatten(values: np.ndarray, trailing: tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)  # (S, G, A, *trailing)
        seeds = arr.shape[0]
        return arr.reshape((seeds, branches) + trailing)

    horizon = bundle["valid"].shape[1]
    agents = bundle["valid"].shape[2]
    action = np.asarray(bundle["tensors"]["cf_action"].numpy(), dtype=np.int64)
    duration = np.asarray(bundle["tensors"]["cf_duration"].numpy(), dtype=np.float64)
    return s4r.build_legal_risk_features(
        primary_predictions=flatten(predictions["e2e"], (horizon, agents, 2)),
        full_predictions=flatten(predictions["full"], (horizon, agents, 2)),
        history_predictions=flatten(predictions["history"], (horizon, agents, 2)),
        response_predictions={
            "r_hat": flatten(predictions["r_hat"], (horizon, agents)),
            "presence_prob": flatten(predictions["presence_prob"], (horizon, agents)),
            "high_prob": flatten(predictions["high_prob"], (horizon, agents)),
        },
        pre_intervention_geometry=bundle["geometry"],
        action=action,
        duration=duration,
        valid_mask=bundle["valid"],
    )


def load_frozen_models() -> dict[str, s4r.FittedRiskModel]:
    """Load the frozen COMBINED/GEOMETRY/RESPONSE risk models (read-only)."""
    return {
        "COMBINED_RISK_HEAD": s4r.FittedRiskModel.load(_resolve(
            dict((k, v[0]) for k, v in s4cp.STAGE4_FROZEN.items())["finalist_risk_model"])),
        "GEOMETRY_ONLY": s4r.FittedRiskModel.load(_resolve(
            dict((k, v[0]) for k, v in s4cp.STAGE4_FROZEN.items())["geometry_risk_model"])),
        "RESPONSE_ONLY": s4bi._load_response_ridge(),
    }


def score_actions(
    features: Mapping[str, np.ndarray],
    frozen_models: Mapping[str, s4r.FittedRiskModel],
    *,
    groups: int,
    n_actions: int,
    rng_seed: int,
) -> dict[str, np.ndarray]:
    """Action-level risk scores, each ``(G, A)``, all 'higher = riskier'."""
    branches = groups * n_actions

    def to_ga(score: np.ndarray) -> np.ndarray:
        return np.asarray(score, dtype=np.float64).reshape(groups, n_actions)

    scores = {
        "COMBINED_RISK_HEAD": to_ga(frozen_models["COMBINED_RISK_HEAD"].predict(
            s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_COMBINED]))),
        "UNCERTAINTY_ONLY": to_ga(s4r.direct_risk_score(s4r.CANDIDATE_UNCERTAINTY, features)),
        "GEOMETRY_ONLY": to_ga(frozen_models["GEOMETRY_ONLY"].predict(
            s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.BASELINE_GEOMETRY]))),
        "RESPONSE_ONLY": to_ga(frozen_models["RESPONSE_ONLY"].predict(
            s4r.design_matrix(features, s4r.CANDIDATE_FEATURES[s4r.CANDIDATE_RESPONSE]))),
        "DISAGREEMENT_ONLY": to_ga(s4r.direct_risk_score(s4r.CANDIDATE_DISAGREEMENT, features)),
    }
    scores["CONSTANT_RANDOM"] = np.random.default_rng(int(rng_seed)).uniform(
        size=(groups, n_actions))
    for name, value in scores.items():
        if value.shape != (groups, n_actions) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Stage 4C action risk score: {name}")
    return scores


# --- group-level aggregation ---------------------------------------------------------------


def aggregate_group_risk(
    action_scores: Mapping[str, np.ndarray], predicted_cost: np.ndarray
) -> dict[str, np.ndarray]:
    """Group risk via the frozen top-2 max aggregation plus the margin baseline.

    ``a1``/``a2`` are the predicted-best and runner-up actions by ``J_hat`` (plan-order
    tie-break). For every action-level score ``r``: ``G_risk = max(r(a1), r(a2))``.
    ``MARGIN_ONLY(g) = -(J_hat(a2) - J_hat(a1))`` (smaller margin -> higher risk).
    """
    j_hat = np.asarray(predicted_cost, dtype=np.float64)  # (G, A)
    groups, n_actions = j_hat.shape
    order = np.argsort(j_hat, axis=-1, kind="stable")  # plan-order tie-break
    a1 = order[:, 0]
    a2 = order[:, 1]
    rows = np.arange(groups)
    group_risk: dict[str, np.ndarray] = {}
    for name, score in action_scores.items():
        r = np.asarray(score, dtype=np.float64)
        if r.shape != (groups, n_actions):
            raise ValueError(f"action score {name} must be (G, A)")
        group_risk[f"{name}_TOP2_MAX" if not name.endswith("_TOP2_MAX") else name] = np.maximum(
            r[rows, a1], r[rows, a2])
    group_risk["MARGIN_ONLY"] = -(j_hat[rows, a2] - j_hat[rows, a1])
    for name, value in group_risk.items():
        if value.shape != (groups,) or not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Stage 4C group risk: {name}")
    return group_risk


# --- determinism re-check -------------------------------------------------------------------


def determinism_recheck(
    bundle: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    stage3r_dir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Re-run pair 0 on the 32 lowest-seed groups twice; require identical outputs."""
    order = np.argsort(bundle["episode_seed"], kind="stable")
    group_subset = order[:DETERMINISM_GROUPS]
    n_actions = int(bundle["n_actions"])
    # expand the selected groups to their branch indices (group-major, action-minor).
    branch_subset = (group_subset[:, None] * n_actions
                     + np.arange(n_actions)[None, :]).ravel()
    config = {"device": device, "batch_size": batch_size}
    runs = [
        s4bi.run_pair_inference(config, bundle["scaled"], branch_subset,
                                pair_index=DETERMINISM_PAIR, binding=binding,
                                stage3r_dir=stage3r_dir)
        for _ in range(2)
    ]
    max_abs = 0.0
    bitwise = True
    for key in runs[0]:
        first, second = runs[0][key], runs[1][key]
        if first.shape != second.shape:
            raise RuntimeError(f"determinism recheck shape mismatch on {key}")
        diff = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
        max_abs = max(max_abs, diff)
        if not np.array_equal(first, second):
            bitwise = False
    if not bitwise and max_abs > DETERMINISM_TOL:
        raise RuntimeError(
            f"Stage 4C determinism recheck failed: max abs diff {max_abs:.3e} > {DETERMINISM_TOL}")
    return {
        "pair_index": DETERMINISM_PAIR,
        "n_groups": int(group_subset.size),
        "group_seeds": [int(s) for s in bundle["episode_seed"][group_subset]],
        "bitwise_identical": bool(bitwise),
        "max_abs_diff": float(max_abs),
        "tolerance": DETERMINISM_TOL,
        "device": str(device),
        "dtype": "float32",
    }


__all__ = [
    "DETERMINISM_GROUPS",
    "DETERMINISM_PAIR",
    "DETERMINISM_TOL",
    "aggregate_group_risk",
    "build_action_risk_features",
    "determinism_recheck",
    "ensemble_predicted_cost",
    "load_frozen_models",
    "load_group_eval",
    "run_grouped_frozen_inference",
    "score_actions",
]
