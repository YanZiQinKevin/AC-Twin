"""Stage 5 unified legal-input layer: the read-only ``LegalRiskBundle`` (plan 3.1)
and the physically separate truth side.

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`` sections 3.1/4.1, Task 2.

The score side (``build_feature_context`` / ``build_legal_bundle``) builds the legal
bundle from the Stage 4C-R2 development roles using the SAME frozen feature code
path as R2 (``stage4cp_features.build_relative_tokens``); it never opens the
counterfactual-truth array. The truth side (``compute_group_truth``) computes
``J_true``/regret/oracle/high-regret with the frozen Stage 4C consequence function
and is only ever called for training labels and validation/holdout evaluation —
never by a score-side path.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import run_stage4a as s4a  # noqa: E402
from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402

POSITION_SLICE = s4a.POSITION_SLICE

_ROLES = ("train", "validation")

# Score-side npz keys. ``object_counterfactual_futures`` (truth) is NEVER read here.
_SCORE_SIDE_KEYS = (
    "object_history", "object_factual_future", "inference_mask",
    "cf_actions", "cf_durations", "t_star", "episode_seed",
)


@dataclass(frozen=True)
class LegalRiskBundle:
    """Plan 3.1 read-only legal input bundle (one decision group per row)."""
    tokens: np.ndarray            # float32 (G, 9, 34)
    token_mask: np.ndarray        # bool    (G, 9)
    global_features: np.ndarray   # float32 (G, 9)
    j_hat: np.ndarray             # float64 (G, 10)
    j_hat_by_seed: np.ndarray     # float64 (5, G, 10)
    selected_action: np.ndarray   # int64   (G,)
    group_seed: np.ndarray        # int64   (G,)
    scenario_id: np.ndarray       # int64   (G,) — the group seed IS the scenario id
    strata: dict[str, np.ndarray] # inference-legal strata metadata only
    valid_mask: np.ndarray        # bool    (G,)


def _load_score_side(data_path: Path) -> dict[str, np.ndarray]:
    with np.load(data_path, allow_pickle=True) as archive:
        missing = [k for k in _SCORE_SIDE_KEYS if k not in archive.files]
        if missing:
            raise ValueError(f"{data_path} is missing score-side keys: {missing}")
        return {k: archive[k] for k in _SCORE_SIDE_KEYS}


def build_feature_context(
    sources: Mapping[str, Any],
    role: str,
    *,
    data_path: str | Path | None = None,
) -> tuple[LegalRiskBundle, dict[str, Any]]:
    """Build the legal bundle plus the prediction-side extras (baselines, indices).

    ``role`` is ``"train"`` or ``"validation"``; inputs come from the resolved
    source mapping (data npz + frozen predictions + frozen action-risk scores).
    ``data_path`` is a test hook allowing a truth-stripped data copy — the build
    must be identical, which is exactly what pins the score side as truth-free.
    Returns ``(bundle, extras)`` where ``extras`` carries the non-learning baseline
    group risks and competitor indices used by the statistics layer.
    """
    role = str(role)
    if role not in _ROLES:
        raise ValueError(f"unknown Stage 5 feature role: {role!r}")
    resolved_data = Path(data_path) if data_path is not None else Path(sources[f"{role}_data"])
    data = _load_score_side(resolved_data)

    history = np.asarray(data["object_history"], dtype=np.float32)       # (G,10,5,8)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)  # (G,20,5,8)
    inference_mask = np.asarray(data["inference_mask"], dtype=bool)      # (G,20,5)
    t_star = np.asarray(data["t_star"], dtype=np.int64).ravel()
    group_seed = np.asarray(data["episode_seed"], dtype=np.int64).ravel()
    groups = int(history.shape[0])

    with np.load(sources[f"{role}_predictions"], allow_pickle=False) as archive:
        e2e = np.asarray(archive["e2e"], dtype=np.float32)               # (5,G,10,20,5,2)
        pred_seed = np.asarray(archive["group_seed"], dtype=np.int64).ravel()
    if not np.array_equal(pred_seed, group_seed):
        raise ValueError(f"{role} predictions do not align with the data group seeds")
    with np.load(sources[f"{role}_scores"], allow_pickle=False) as archive:
        score_seed = np.asarray(archive["group_seed"], dtype=np.int64).ravel()
        if not np.array_equal(score_seed, group_seed):
            raise ValueError(f"{role} action-risk scores do not align with the data group seeds")
        action_risks = {
            name: np.asarray(archive[name], dtype=np.float64) for name in s4cp.RISK_CLASSES}

    factual_xy = factual[..., POSITION_SLICE].astype(np.float64)         # (G,20,5,2)
    history_last_xy = history[:, -1, 0, POSITION_SLICE].astype(np.float64)  # (G,2)
    geometry = s4a._pre_intervention_geometry(history)

    features = s4cpf.build_relative_tokens(
        e2e, factual_xy, history_last_xy, inference_mask,
        action_risks, geometry, t_star, group_seed)

    # interaction density: non-ego vehicles within 30 m at history last frame
    # (same rule as the frozen Stage 4C loader; inference-legal).
    last = history[:, -1]                                                # (G,5,8)
    ego_xy = last[:, 0, POSITION_SLICE]
    dist = np.linalg.norm(last[:, :, POSITION_SLICE] - ego_xy[:, None, :], axis=-1)
    present = last[..., 0] > 0
    present[:, 0] = False
    density_count = ((dist <= 30.0) & present).sum(axis=1).astype(np.float64)

    bundle = LegalRiskBundle(
        tokens=np.asarray(features["tokens"], dtype=np.float32),
        token_mask=np.asarray(features["token_mask"], dtype=bool),
        global_features=np.asarray(features["global_features"], dtype=np.float32),
        j_hat=np.asarray(features["j_hat"], dtype=np.float64),
        j_hat_by_seed=np.asarray(features["j_hat_seeds"], dtype=np.float64),
        selected_action=np.asarray(features["reference_action"], dtype=np.int64),
        group_seed=group_seed,
        scenario_id=group_seed.copy(),
        strata={
            "t_star": t_star,
            "interaction_min_distance": np.asarray(
                features["interaction_min_distance"], dtype=np.float64),
            "density_count": density_count,
        },
        valid_mask=np.asarray(features["valid_mask"], dtype=bool),
    )
    if bundle.tokens.shape != (groups, s4cp.N_COMPETITORS, s4cpf.TOKEN_DIM):
        raise ValueError(f"{role} token contract drift: {bundle.tokens.shape}")
    if bundle.j_hat.shape != (groups, s4cp.N_ACTIONS) \
            or bundle.j_hat_by_seed.shape != (len(s4cr2.MODEL_SEEDS), groups, s4cp.N_ACTIONS):
        raise ValueError(f"{role} J_hat contract drift")
    extras = {
        "baselines": features["baselines"],
        "competitor_index": np.asarray(features["competitor_index"], dtype=np.int64),
    }
    return bundle, extras


def build_legal_bundle(
    sources: Mapping[str, Any],
    role: str,
    *,
    data_path: str | Path | None = None,
) -> LegalRiskBundle:
    """The plan-3.1 bundle only (drops prediction-side extras)."""
    return build_feature_context(sources, role, data_path=data_path)[0]


def compute_group_truth(sources: Mapping[str, Any], role: str) -> dict[str, np.ndarray]:
    """TRUTH SIDE — physically separate from every score-side path.

    Computes per-action true consequences ``J_true`` with the frozen Stage 4C
    consequence function, the selected-action decision regret, oracle action,
    high-regret label, and the random/idle reference regrets. Legal ONLY for
    training labels and validation/holdout evaluation, after the matching data
    role's binding is established — never at inference/scoring time.
    """
    role = str(role)
    if role not in _ROLES:
        raise ValueError(f"unknown Stage 5 truth role: {role!r}")
    data_path = Path(sources[f"{role}_data"])
    with np.load(data_path, allow_pickle=True) as archive:
        cf = np.asarray(archive["object_counterfactual_futures"], dtype=np.float32)
        history = np.asarray(archive["object_history"], dtype=np.float32)
        factual = np.asarray(archive["object_factual_future"], dtype=np.float32)
        inference_mask = np.asarray(archive["inference_mask"], dtype=bool)
        group_seed = np.asarray(archive["episode_seed"], dtype=np.int64).ravel()
    groups, n_actions = int(cf.shape[0]), int(cf.shape[1])
    if n_actions != s4cp.N_ACTIONS:
        raise ValueError(f"expected {s4cp.N_ACTIONS} actions, got {n_actions}")

    counterfactual_xy = cf[..., POSITION_SLICE].astype(np.float64)       # (G,10,20,5,2)
    history_last_xy = history[:, -1, 0, POSITION_SLICE].astype(np.float64)
    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, n_actions, 2))
    mask_b = np.broadcast_to(inference_mask[:, None, :, :],
                             (groups, n_actions) + inference_mask.shape[1:])
    comp_true = s4co.trajectory_components(counterfactual_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))

    # selected action = argmin ensemble J_hat (same frozen path as the score side)
    with np.load(sources[f"{role}_predictions"], allow_pickle=False) as archive:
        e2e = np.asarray(archive["e2e"], dtype=np.float32)
    factual_xy = factual[..., POSITION_SLICE].astype(np.float64)
    _, j_hat = s4cpf.compute_predicted_costs(e2e, factual_xy, history_last_xy, inference_mask)

    rows = np.arange(groups)
    a_hat = np.argmin(j_hat, axis=-1)
    a_star = np.argmin(j_true, axis=-1)
    j_true_star = j_true[rows, a_star]
    regret = s4co.clamp_regret(j_true[rows, a_hat] - j_true_star)
    random_regret = s4co.clamp_regret(j_true.mean(axis=-1) - j_true_star)
    idle_regret = s4co.clamp_regret(j_true[:, s4co.IDLE_PLAN_INDEX] - j_true_star)
    high_regret = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)
    if not (np.isfinite(j_true).all() and np.isfinite(regret).all()):
        raise ValueError(f"{role} truth contains non-finite values")

    return {
        "group_seed": group_seed,
        "selected_action": a_hat.astype(np.int64),
        "oracle_action": a_star.astype(np.int64),
        "j_true": j_true,
        "j_hat": j_hat,
        "regret": regret,
        "random_regret": random_regret,
        "idle_regret": idle_regret,
        "high_regret": high_regret,
    }


__all__ = [
    "POSITION_SLICE", "LegalRiskBundle",
    "build_feature_context", "build_legal_bundle", "compute_group_truth",
]
