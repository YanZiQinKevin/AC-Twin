"""Stage 1 data contract, normalization, targets and fair derangement.

This module implements only the Stage 1 frozen data contract. It never reads
``object_counterfactual_future`` into a model input; the counterfactual branch
is used exclusively to build the position-effect target, the response truth and
the evaluation mask.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

OBJECT_FEATURES = (
    "presence",
    "is_ego",
    "x",
    "y",
    "vx",
    "vy",
    "cos_heading",
    "sin_heading",
)
N_FEATURES = len(OBJECT_FEATURES)
HISTORY_STEPS = 10
FUTURE_STEPS = 20
# x, y, vx, vy are the only standardized features.
MOTION_FEATURES = (2, 3, 4, 5)
POSITION_FEATURES = (2, 3)

DERANGEMENT_BASE_SEED = 20260907
VALIDATION_DERANGEMENT_SEED = 20261907
TEST_DERANGEMENT_SEED = 20262907

REQUIRED_ARRAY_FIELDS = (
    "object_history",
    "object_factual_future",
    "object_counterfactual_future",
    "factual_actions",
    "counterfactual_actions",
    "cf_action",
    "cf_duration",
    "episode_seed",
    "split",
)


def sha256_file(file_path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_features_json(value: Any) -> list:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    return json.loads(value)


def validate_source_contract(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate the Stage 1 source contract; raise before any output is made.

    Returns a small read-only summary used for the mandatory entry report. No
    output directory is created here; callers must run this first.
    """
    missing = [field for field in REQUIRED_ARRAY_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Source data is missing required fields: {missing}")
    if "object_features_json" not in data:
        raise ValueError("Source data is missing required field: object_features_json")

    features = _parse_features_json(data["object_features_json"])
    if list(features) != list(OBJECT_FEATURES):
        raise ValueError(
            f"object_features_json must equal {list(OBJECT_FEATURES)}, got {features}"
        )

    history = np.asarray(data["object_history"])
    factual = np.asarray(data["object_factual_future"])
    counterfactual = np.asarray(data["object_counterfactual_future"])
    if history.ndim != 4 or history.shape[1] != HISTORY_STEPS or history.shape[3] != N_FEATURES:
        raise ValueError(
            f"object_history must have shape E×{HISTORY_STEPS}×N×{N_FEATURES}, got {history.shape}"
        )
    for name, future in (("object_factual_future", factual),
                         ("object_counterfactual_future", counterfactual)):
        if future.ndim != 4 or future.shape[1] != FUTURE_STEPS or future.shape[3] != N_FEATURES:
            raise ValueError(
                f"{name} must have shape E×{FUTURE_STEPS}×N×{N_FEATURES}, got {future.shape}"
            )
    episodes, _, agents, _ = history.shape
    if factual.shape[0] != episodes or counterfactual.shape[0] != episodes:
        raise ValueError("Episode count must match across history and futures")
    if factual.shape[2] != agents or counterfactual.shape[2] != agents:
        raise ValueError("Agent count must match across history and futures")

    for name in ("factual_actions", "counterfactual_actions"):
        actions = np.asarray(data[name])
        if actions.shape != (episodes, FUTURE_STEPS):
            raise ValueError(f"{name} must have shape E×{FUTURE_STEPS}, got {actions.shape}")
    for name in ("cf_action", "cf_duration", "episode_seed", "split"):
        vector = np.asarray(data[name])
        if vector.shape != (episodes,):
            raise ValueError(f"{name} must have shape (E,), got {vector.shape}")

    split = np.asarray(data["split"])
    if not np.issubdtype(split.dtype, np.integer):
        raise ValueError("split must contain integer labels")
    if not set(np.unique(split).tolist()) <= {0, 1, 2}:
        raise ValueError("split labels must be a subset of {0=train,1=validation,2=development_test}")
    episode_seed = np.asarray(data["episode_seed"])
    if not np.issubdtype(episode_seed.dtype, np.integer):
        raise ValueError("episode_seed must contain one integer seed per episode")
    if len(np.unique(episode_seed)) != episodes:
        raise ValueError("episode_seed must be unique per episode for derangement and clustering")

    for name, tensor in (("object_history", history),
                         ("object_factual_future", factual),
                         ("object_counterfactual_future", counterfactual)):
        if not np.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN/Inf values")

    counts = {int(label): int((split == label).sum()) for label in (0, 1, 2)}
    return {
        "episodes": int(episodes),
        "agents": int(agents),
        "split_counts": counts,
        "features": list(OBJECT_FEATURES),
    }


def compute_anchors(history: np.ndarray) -> np.ndarray:
    """Per-episode ego (slot 0) world x,y at the last history frame; shape E×2."""
    return np.asarray(history)[:, -1, 0, POSITION_FEATURES[0]:POSITION_FEATURES[1] + 1].astype(np.float32)


def compute_history_agent_known(history: np.ndarray) -> np.ndarray:
    """Boolean E×N: agent is known if present at any history time."""
    return (np.asarray(history)[..., 0] > 0).any(axis=1)


def compute_valid_mask(
    history: np.ndarray,
    factual: np.ndarray,
    counterfactual: np.ndarray,
) -> np.ndarray:
    """Boolean E×T×N training/eval mask.

    valid = history_agent_known AND factual_presence AND counterfactual_presence.
    """
    known = compute_history_agent_known(history)
    factual_presence = np.asarray(factual)[..., 0] > 0
    counterfactual_presence = np.asarray(counterfactual)[..., 0] > 0
    return known[:, None, :] & factual_presence & counterfactual_presence


def compute_target_effect(factual: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """Position effect truth E×T×N×2 = cf_xy - factual_xy (world coordinates)."""
    factual_xy = np.asarray(factual)[..., POSITION_FEATURES[0]:POSITION_FEATURES[1] + 1]
    counterfactual_xy = np.asarray(counterfactual)[..., POSITION_FEATURES[0]:POSITION_FEATURES[1] + 1]
    return (counterfactual_xy - factual_xy).astype(np.float32)


def fit_feature_normalizer(
    history: np.ndarray,
    factual: np.ndarray,
    train_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit x,y,vx,vy mean/std on split-0 presence-true history + factual suffix.

    anchor = per-episode last-frame ego world x,y is subtracted before fitting.
    presence/is_ego/cos/sin are left unstandardized (mean 0, std 1).
    """
    train_indices = np.asarray(train_indices)
    if len(train_indices) == 0:
        raise ValueError("Feature normalizer needs non-empty train_indices")
    history = np.asarray(history, dtype=np.float32)
    factual = np.asarray(factual, dtype=np.float32)
    anchors = compute_anchors(history)
    values = np.concatenate([history[train_indices], factual[train_indices]], axis=1).copy()
    values[..., POSITION_FEATURES[0]:POSITION_FEATURES[1] + 1] -= anchors[train_indices][:, None, None]
    mask = values[..., 0] > 0
    if not mask.any():
        raise ValueError("Feature normalizer needs visible training objects")
    observed = values[..., MOTION_FEATURES[0]:MOTION_FEATURES[-1] + 1][mask]
    if not np.isfinite(observed).all():
        raise ValueError("Visible training motion features must be finite")
    mean = np.zeros(N_FEATURES, np.float32)
    std = np.ones(N_FEATURES, np.float32)
    mean[MOTION_FEATURES[0]:MOTION_FEATURES[-1] + 1] = observed.mean(axis=0)
    std[MOTION_FEATURES[0]:MOTION_FEATURES[-1] + 1] = np.maximum(observed.std(axis=0), 1e-6)
    return {"mean": mean, "std": std}


def apply_feature_normalizer(
    values: np.ndarray,
    anchors: np.ndarray,
    normalizer: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Subtract anchor, standardize x,y,vx,vy, keep padding exactly zero."""
    result = np.asarray(values, dtype=np.float32).copy()
    mask = result[..., 0] > 0
    lo, hi = POSITION_FEATURES[0], POSITION_FEATURES[1] + 1
    mlo, mhi = MOTION_FEATURES[0], MOTION_FEATURES[-1] + 1
    result[..., lo:hi] -= np.asarray(anchors, dtype=np.float32)[:, None, None]
    result[..., mlo:mhi] = (result[..., mlo:mhi] - normalizer["mean"][mlo:mhi]) / normalizer["std"][mlo:mhi]
    result[..., mlo:mhi] = np.where(mask[..., None], result[..., mlo:mhi], 0.0)
    return result


def fit_target_normalizer(
    target: np.ndarray,
    valid_mask: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the 2D target mean/std on split-0 valid_mask entries only."""
    train_indices = np.asarray(train_indices)
    if len(train_indices) == 0:
        raise ValueError("Target normalizer needs non-empty train_indices")
    selected = np.asarray(target, dtype=np.float32)[train_indices]
    mask = np.asarray(valid_mask, dtype=bool)[train_indices]
    if not mask.any():
        raise ValueError("Target normalizer needs valid training agent-times")
    observed = selected[mask]
    if not np.isfinite(observed).all():
        raise ValueError("Valid training targets must be finite")
    mean = observed.mean(axis=0).astype(np.float32)
    std = np.maximum(observed.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


def apply_suffix_window(factual_normalized: np.ndarray, steps: int) -> np.ndarray:
    """Keep the first `steps` factual frames; zero all later steps' features."""
    if steps <= 0 or steps > factual_normalized.shape[1]:
        raise ValueError("suffix steps must be within the factual horizon")
    result = np.asarray(factual_normalized, dtype=np.float32).copy()
    result[:, steps:] = 0.0
    return result


def suffix_presence_mask(factual: np.ndarray, steps: int) -> np.ndarray:
    """Boolean E×T×N factual presence restricted to the first `steps` frames."""
    if steps <= 0 or steps > factual.shape[1]:
        raise ValueError("suffix steps must be within the factual horizon")
    mask = (np.asarray(factual)[..., 0] > 0).copy()
    mask[:, steps:] = False
    return mask


def visible_count_signature(history_agent_known: np.ndarray) -> np.ndarray:
    """Per-episode count of history-known agents (the visible-vehicle signature)."""
    return np.asarray(history_agent_known, dtype=bool).sum(axis=1).astype(np.int64)


def _deranged_rotation(members: list[int], rng: np.random.Generator) -> dict[int, int]:
    """Deterministic seeded cyclic derangement of >=2 members.

    The member order is shuffled by ``rng`` and each recipient takes the next
    member in that order (a single cycle), so there is no fixed point and the
    mapping is a bijection whose donor is used exactly once. The shuffle makes
    the mapping reproducible for a fixed seed and varying across seeds.
    """
    order = np.array(sorted(members))
    order = order[rng.permutation(len(order))]
    return {int(order[i]): int(order[(i + 1) % len(order)]) for i in range(len(order))}


def build_derangement(
    cf_action: np.ndarray,
    cf_duration: np.ndarray,
    factual_first_action: np.ndarray,
    visible_count: np.ndarray,
    episode_seed: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (donor_index, tier) for each recipient position of one split.

    Operates on already split-restricted arrays. Stratified with bounded
    fallback: (cf_action,cf_duration,first_factual_action,visible_count) ->
    (cf_action,cf_duration,visible_count) -> (cf_action,cf_duration) -> whole
    split. Never self-matches, never reuses a donor, and raises if impossible.
    """
    n = len(episode_seed)
    if n < 2:
        raise ValueError("Derangement needs at least two episodes in a split")
    tier_keys = [
        [(int(a), int(d), int(f), int(v)) for a, d, f, v in
         zip(cf_action, cf_duration, factual_first_action, visible_count)],
        [(int(a), int(d), int(v)) for a, d, v in zip(cf_action, cf_duration, visible_count)],
        [(int(a), int(d)) for a, d in zip(cf_action, cf_duration)],
        [(0,)] * n,
    ]
    rng = np.random.default_rng(int(seed))
    donor = np.full(n, -1, dtype=np.int64)
    tier_of = np.zeros(n, dtype=np.int64)
    remaining = list(range(n))
    for tier, keys in enumerate(tier_keys, start=1):
        if not remaining:
            break
        groups: dict[Any, list[int]] = {}
        for index in remaining:
            groups.setdefault(keys[index], []).append(index)
        leftover: list[int] = []
        for members in groups.values():
            if len(members) >= 2:
                for recipient, donor_index in _deranged_rotation(members, rng).items():
                    donor[recipient] = donor_index
                    tier_of[recipient] = tier
            else:
                leftover.append(members[0])
        remaining = leftover

    # A single episode can be stranded after the strictest tiers (its keys were
    # unique among the leftover set). The last tier means "anywhere in the same
    # split", so splice the loner into an already-formed derangement cycle:
    # recipient y currently maps to d; re-point y -> loner and loner -> d. This
    # keeps the mapping a bijection with no self-match and no same-seed pair.
    if len(remaining) == 1:
        loner = remaining[0]
        candidates = [i for i in range(n) if i != loner and donor[i] != -1 and donor[i] != loner]
        if not candidates:
            raise RuntimeError(
                "Could not form a valid derangement: no assigned episode to splice into")
        partner = candidates[int(rng.integers(0, len(candidates)))]
        donor[loner] = donor[partner]
        donor[partner] = loner
        tier_of[loner] = len(tier_keys)
        tier_of[partner] = len(tier_keys)
        remaining = []
    if remaining:
        raise RuntimeError(
            f"Could not form a valid derangement for {len(remaining)} episode(s); "
            "stratified fallback exhausted"
        )
    if (donor == np.arange(n)).any():
        raise RuntimeError("Derangement produced a self match")
    if len(np.unique(donor)) != n:
        raise RuntimeError("Derangement reused a donor")
    if (episode_seed[donor] == episode_seed).any():
        raise RuntimeError("Derangement matched identical episode seeds")
    return donor, tier_of


def derangement_manifest(
    donor: np.ndarray,
    tier_of: np.ndarray,
    split_label: int,
    cf_action: np.ndarray,
    cf_duration: np.ndarray,
    visible_count: np.ndarray,
    episode_seed: np.ndarray,
) -> dict[str, Any]:
    """Build the per-split derangement manifest with QA counters.

    ``donor``/``tier_of`` index into the split-restricted arrays; this function
    records recipient/donor local indices, episode seeds, split, matched tier
    and per-field match flags, plus global QA counts that must all be zero.
    """
    donor = np.asarray(donor, dtype=np.int64)
    n = len(donor)
    recipient = np.arange(n, dtype=np.int64)
    records = [
        {
            "recipient_index": int(index),
            "donor_index": int(donor[index]),
            "recipient_episode_seed": int(episode_seed[index]),
            "donor_episode_seed": int(episode_seed[donor[index]]),
            "split": int(split_label),
            "tier": int(tier_of[index]),
            "cf_action_match": bool(cf_action[index] == cf_action[donor[index]]),
            "cf_duration_match": bool(cf_duration[index] == cf_duration[donor[index]]),
            "visible_count_match": bool(visible_count[index] == visible_count[donor[index]]),
        }
        for index in range(n)
    ]
    qa = {
        "episodes": int(n),
        "self_match": int((donor == recipient).sum()),
        "duplicate_donor": int(n - len(np.unique(donor))),
        "cross_split": 0,
        "same_seed": int((episode_seed[donor] == episode_seed).sum()),
        "tier_counts": {int(t): int((tier_of == t).sum()) for t in np.unique(tier_of)},
    }
    return {"split": int(split_label), "records": records, "qa": qa}
