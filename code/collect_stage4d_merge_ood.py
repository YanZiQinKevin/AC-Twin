"""Stage 4D D0/D1 Merge model-mismatch OOD collector: truth-separated two-file publish.

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §3.1, §5.1, §6.2, §6.4, §8.1 and
§10 Task 2. This stage collects eleven-branch (one factual + ten counterfactual) Merge
decision groups under one of the three pre-registered generator-mismatch OOD profiles
(``BEHAVIOR_OOD`` / ``GEOMETRY_OOD`` / ``JOINT_OOD``). It never loads a Direct checkpoint,
never produces a risk score / AURC / coverage number and never reads the sealed
Stage 4C-R2 confirm truth.

Reuse policy (read-only, never modified): the low-level complete-group science is REUSED
from the frozen Stage 4C-R2 collector
(``collect_stage4cr2_merge_multiaction.collect_group`` / ``validate_complete_group``) — one
fresh group seed -> one history, one factual branch, one ``t_star`` -> TEN counterfactual
branches forked from the SAME pre-intervention state; object slot == stable entity id. A
frozen-contract guard (:func:`_assert_reused_low_level_contract`) fails loudly if that
reused contract ever drifts from this stage's protocol constants.

Stage 4D difference (plan §4 / §11 item 2): the injected ``branch_rollout`` is this stage's
own. AFTER ``env.reset(seed=group_seed)`` it stamps episode entity ids
(``assign_episode_entity_ids`` — required so a stamping bug cannot masquerade as branch
divergence, and so ``apply_profile_to_env``'s hard id requirement is met) and THEN applies
``stage4d_ood_env.apply_profile_to_env`` BEFORE any rollout step. Each branch returns its
post-perturbation snapshot; :func:`collect_group` cross-checks all eleven pairwise with
``snapshots_identical``. Any divergence rejects the WHOLE group with the engineering reason
``branch_perturbation_inconsistent`` — never an outcome reason.

Truth separation (plan §6.4, Global Constraint 6): collection publishes TWO files per role
— ``inference_inputs.npz`` (EXACTLY the eight legal fields; never any CF truth) and the
sealed ``confirmation_truth.npz`` (CF futures + CF mechanism/integrity fields + same-order
``group_seed``). D0 pilot truth is pilot-analysis task truth (plan §5.1 allows reading it);
D1 confirm truth is sealed. The collector writes the SAME two-file contract for every role
and lets the role/directory carry the semantic difference; the collection receipt binds the
two files' group ordering (acceptance order) and hashes. The input mask is generated ONLY
from the legal factual/history contract (``build_inference_mask``), proven by independent
recomputation at publish time.

Acceptance (engineering whitelist ONLY): branch missing / per-frame stable-ID present-set
mismatch / prefix divergence / fingerprint mismatch / illegal mask reduction / shape /
capacity / finiteness / duplicate seed / no legal agent-time / post-perturbation snapshot
divergence reject the WHOLE group. Regret, oracle action, collision, response strength and
any model error NEVER reject. Resume restores the full action-RNG state and candidate
cursor from the atomic checkpoint and produces results identical to uninterrupted
collection; only hash-valid complete groups are re-accepted.

The collector exposes a 30-second ProgressReporter heartbeat (phase / completed/total /
percentage / elapsed / ETA). Rollout-injectable so unit tests need no highway_env; the real
12-group ENGINEERING_SMOKE_ONLY run is an operational step.
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage2_holdout as engine  # noqa: E402
import collect_stage4cr2_merge_multiaction as r2collector  # noqa: E402  READ-ONLY reuse
from ac_twin import stage2_protocol as sp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402  frozen R2 contract, read-only
from ac_twin import stage4d_protocol as prot  # noqa: E402
from ac_twin import stage4d_ood_env as ood  # noqa: E402
from ac_twin.checkpointing import (  # noqa: E402
    canonical_fingerprint,
    load_collection_checkpoint,
    save_collection_checkpoint,
)
from ac_twin.object_centric import (  # noqa: E402
    assign_episode_entity_ids,
    extract_object_frame,
)
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

ROLES = ("smoke", "pilot_behavior", "pilot_geometry", "pilot_joint", "d1_confirm")

#: the OOD profile each collection role applies. smoke is pure engineering and (the plan
#: leaves the smoke profile undetermined) uses BEHAVIOR_OOD — the simplest behavior-only
#: profile — as sanctioned by the Task 2 brief. D1 confirm's profile is read from the
#: burned protocol lock (the D0-frozen selection), never hard-coded here.
ROLE_PROFILE = {
    "smoke": "BEHAVIOR_OOD",
    "pilot_behavior": "BEHAVIOR_OOD",
    "pilot_geometry": "GEOMETRY_OOD",
    "pilot_joint": "JOINT_OOD",
    "d1_confirm": None,  # frozen by the D0 selection, read from the protocol lock
}

#: accepted-group quotas. smoke = 12 (engineering smoke); pilots = 300 accepted each
#: (plan §5.1). D1 confirm's quota is the power-frozen selected n, read from the lock.
ROLE_QUOTA = {
    "smoke": 12,
    "pilot_behavior": prot.PILOT_ACCEPTED_GROUPS,
    "pilot_geometry": prot.PILOT_ACCEPTED_GROUPS,
    "pilot_joint": prot.PILOT_ACCEPTED_GROUPS,
    "d1_confirm": None,  # power-frozen selected n from the protocol lock (Task 4/5)
}

#: role -> (candidate schedule entropy, size) within this stage's approved §6.2 domains.
ROLE_SCHEDULE = {
    "smoke": (prot.SMOKE_SEED_ENTROPY, prot.SMOKE_SCHEDULE_SIZE),
    "pilot_behavior": (prot.PILOT_BEHAVIOR_SEED_ENTROPY, prot.PILOT_SCHEDULE_SIZE),
    "pilot_geometry": (prot.PILOT_GEOMETRY_SEED_ENTROPY, prot.PILOT_SCHEDULE_SIZE),
    "pilot_joint": (prot.PILOT_JOINT_SEED_ENTROPY, prot.PILOT_SCHEDULE_SIZE),
    "d1_confirm": (prot.D1_CANDIDATE_SEED_ENTROPY, prot.D1_CANDIDATE_SCHEDULE_SIZE),
}

STATUS_COLLECTED = "COLLECTED"
STATUS_SMOKE = "ENGINEERING_SMOKE_ONLY"
STATUS_COLLECTION_INCOMPLETE = "STAGE4D_OOD_COLLECTION_INCOMPLETE"

#: Whole-group rejection reasons: structural/simulation invariants ONLY (the R2
#: engineering vocabulary) plus this stage's post-perturbation snapshot divergence. No
#: outcome-based reason (regret/oracle/collision/response/error/stratum) may appear here.
REASONS = (
    "prefix_terminated",
    "object_prefix_mismatch",
    "factual_future_short",
    "counterfactual_future_short",
    "non_finite",
    "short_history",
    "initial_fingerprint_mismatch",
    "present_set_mismatch",
    "eval_mask_mismatch",
    "no_legal_agent_time",
    "observation_shape",
    "duplicate_seed",
    "branch_perturbation_inconsistent",
)

#: EXACTLY the eight legal fields of the published inference_inputs.npz (score side; never
#: any counterfactual truth). Mirrors the Stage 4C-R2 confirm two-file contract.
LEGAL_INPUT_FIELDS = (
    "object_history",                # (G,10,5,8) float32
    "object_factual_future",         # (G,20,5,8) float32
    "factual_actions",               # (G,20)   int64
    "cf_actions",                    # (G,10)   int64
    "cf_durations",                  # (G,10)   int64
    "t_star",                        # (G,)     int64
    "group_seed",                    # (G,)     int64
    "inference_mask",                # (G,20,5) bool
)

#: the sealed confirmation_truth.npz: CF futures + CF mechanism fields + integrity fields
#: + same-order group_seed. (For D0 pilots this is pilot-analysis task truth — readable
#: per plan §5.1 — but it is published under the identical contract.)
TRUTH_FIELDS = (
    "object_counterfactual_futures",  # (G,10,20,5,8) float32 — the sealed CF truth
    "cf_actions",                     # (G,10) int64 — CF mechanism
    "cf_durations",                   # (G,10) int64 — CF mechanism
    "factual_actions",                # (G,20) int64 — integrity: factual plan
    "t_star",                         # (G,)   int64 — integrity: decision time
    "group_seed",                     # (G,)   int64 — same-order binding (required)
    "inference_mask",                 # (G,20,5) bool — integrity: legal mask copy
)

#: archive-key tokens that mark truth-side content; the score side never reads such a key.
FORBIDDEN_INPUT_KEY_TOKENS = (
    "counterfactual", "truth", "j_true", "regret", "oracle", "a_star", "true_",
)

_GROUP_ARRAY_FIELDS = (
    "object_history",
    "object_factual_future",
    "object_counterfactual_futures",
    "factual_actions",
    "cf_actions",
    "cf_durations",
    "inference_mask",
)
_GROUP_SCALAR_INT = ("group_seed", "t_star", "n_mask_cells")
_ARRAY_SPEC = {
    "object_history": ((prot.HISTORY_LEN, prot.OBJECT_CAPACITY,
                        prot.N_OBJECT_FEATURES), np.float32),
    "object_factual_future": ((prot.HORIZON, prot.OBJECT_CAPACITY,
                               prot.N_OBJECT_FEATURES), np.float32),
    "object_counterfactual_futures": ((prot.N_ACTIONS, prot.HORIZON,
                                       prot.OBJECT_CAPACITY,
                                       prot.N_OBJECT_FEATURES), np.float32),
    "factual_actions": ((prot.HORIZON,), np.int64),
    "cf_actions": ((prot.N_ACTIONS,), np.int64),
    "cf_durations": ((prot.N_ACTIONS,), np.int64),
    "inference_mask": ((prot.HORIZON, prot.OBJECT_CAPACITY), bool),
}


class CollectionIncompleteError(RuntimeError):
    """Candidate domain exhausted before the frozen quota (hard stop; no new domain)."""

    status = STATUS_COLLECTION_INCOMPLETE


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


# --- reused low-level contract guard -------------------------------------------------------


def _assert_reused_low_level_contract() -> None:
    """Fail loudly if the reused R2 low-level rollout/QA contract drifts from this stage's.

    Read-only comparison: this stage's protocol constants are bitwise the same contract as
    R2's. The reused ``collect_group``/``validate_complete_group`` read R2 module
    constants; any drift would silently change the collected science, so it is a hard stop.
    """
    pairs = (
        ("N_ACTIONS", s4cr2.N_ACTIONS, prot.N_ACTIONS),
        ("HISTORY_LEN", s4cr2.HISTORY_LEN, prot.HISTORY_LEN),
        ("HORIZON", s4cr2.HORIZON, prot.HORIZON),
        ("T_STAR_LOW", s4cr2.T_STAR_LOW, prot.T_STAR_LOW),
        ("T_STAR_HIGH", s4cr2.T_STAR_HIGH, prot.T_STAR_HIGH),
        ("OBJECT_CAPACITY", s4cr2.OBJECT_CAPACITY, prot.OBJECT_CAPACITY),
        ("N_OBJECT_FEATURES", s4cr2.N_OBJECT_FEATURES, prot.N_OBJECT_FEATURES),
        ("EGO_SLOT", s4cr2.EGO_SLOT, prot.EGO_SLOT),
        ("ACTION_PLANS", [tuple(p) for p in s4cr2.ACTION_PLANS],
         [tuple(p) for p in prot.ACTION_PLANS]),
    )
    for name, r2_value, stage_value in pairs:
        if r2_value != stage_value:
            raise RuntimeError(
                f"reused R2 low-level contract drifted on {name}: {r2_value!r} != "
                f"{stage_value!r}")
    if s4cr2.build_inference_mask is not prot.build_inference_mask:
        raise RuntimeError("reused build_inference_mask is not the protocol's function")
    if s4cr2.materialize_action_plan is not prot.materialize_action_plan:
        raise RuntimeError("reused materialize_action_plan is not the protocol's function")
    for name, r2_module_value, stage_value in (
        ("r2collector.N_ACTIONS", r2collector.N_ACTIONS, prot.N_ACTIONS),
        ("r2collector.HISTORY_LEN", r2collector.HISTORY_LEN, prot.HISTORY_LEN),
        ("r2collector.FUTURE_LEN", r2collector.FUTURE_LEN, prot.HORIZON),
        ("r2collector.N_OBJECT_FEATURES", r2collector.N_OBJECT_FEATURES,
         prot.N_OBJECT_FEATURES),
        ("r2collector.OBJECT_CAPACITY_CONST", r2collector.OBJECT_CAPACITY_CONST,
         prot.OBJECT_CAPACITY),
    ):
        if r2_module_value != stage_value:
            raise RuntimeError(
                f"reused R2 collector module constant {name} drifted: "
                f"{r2_module_value!r} != {stage_value!r}")


_assert_reused_low_level_contract()


# --- Stage 4D branch rollout (profile applied post-reset, pre-rollout) ----------------------


def _stage4d_rollout_branch_real(
    env_id: str,
    env_cfg: Mapping[str, Any],
    seed: int,
    prefix_actions: np.ndarray,
    future_actions: np.ndarray,
    capacity: int,
    profile: str,
) -> dict[str, Any]:
    """One OOD branch: reset -> stamp entity ids -> apply the profile -> roll out.

    Entity ids are stamped (``assign_episode_entity_ids``) immediately after reset and
    BEFORE :func:`ood.apply_profile_to_env`, so the profile hook's hard id requirement is
    met and the post-perturbation snapshot is keyed by stable ids (a stamping bug surfaces
    as an exception here, never as silent branch divergence). The ego is never modified.
    highway_env is imported lazily (pygame fails under pytest ``-W error``).
    """
    from collect_pilot import make_env  # lazy: pulls in gymnasium/highway_env

    env = make_env(env_id, dict(env_cfg))
    try:
        env.reset(seed=int(seed))
        fingerprints = assign_episode_entity_ids(env)
        profile_config = ood.build_profile_config(profile, int(seed))
        snapshot = ood.apply_profile_to_env(env, profile_config)

        prefix_frames = [extract_object_frame(env, capacity)[0]]
        terminated = truncated = False
        reason = ""
        for action in prefix_actions:
            _, _, term, trunc, _ = env.step(int(action))
            prefix_frames.append(extract_object_frame(env, capacity)[0])
            if term or trunc:
                terminated, truncated = bool(term), bool(trunc)
                reason = "terminated" if term else "truncated"
                break

        future_frames: list[np.ndarray] = []
        first_contact = -1
        if not (terminated or truncated):
            base = getattr(env, "unwrapped", env)
            for step, action in enumerate(future_actions):
                _, _, term, trunc, _ = env.step(int(action))
                future_frames.append(extract_object_frame(env, capacity)[0])
                if first_contact < 0 and any(
                        bool(getattr(v, "crashed", False)) for v in base.road.vehicles):
                    first_contact = step
                    reason = reason or "collision"
                if term or trunc:
                    terminated, truncated = bool(term), bool(trunc)
                    reason = reason or ("terminated" if term else "truncated")
                    break
    finally:
        env.close()

    return {
        "prefix_frames": np.asarray(prefix_frames, dtype=np.float32),
        "future_frames": np.asarray(future_frames, dtype=np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": reason,
        "first_contact_step": int(first_contact),
        "fingerprints": fingerprints,
        "post_perturbation_snapshot": snapshot,
    }


def make_stage4d_branch_rollout(profile: str) -> Callable:
    """Bind ``profile`` into a branch_rollout with the reused collector's six-arg signature.

    The reused ``collect_group`` calls ``branch_rollout(env_id, env_cfg, seed,
    prefix_actions, plan, capacity)``; the profile is bound here so every branch of one
    group seed gets the same deterministic perturbation.
    """
    profile = str(profile)
    if profile not in ood.PROFILES:
        raise ValueError(f"unknown OOD profile {profile!r}; expected {list(ood.PROFILES)}")

    def rollout(env_id, env_cfg, seed, prefix_actions, future_actions, capacity):
        return _stage4d_rollout_branch_real(
            env_id, env_cfg, seed, prefix_actions, future_actions, capacity, profile)

    return rollout


# --- reused low-level group machinery + post-perturbation cross-check ------------------------


def collect_group(
    seed: int,
    t_star: int,
    prefix_actions: np.ndarray,
    factual_plan: np.ndarray,
    config: Mapping[str, Any],
    branch_rollout: Callable,
) -> dict[str, Any]:
    """Roll out one factual + ten CF branches; cross-check the eleven post-perturbation
    snapshots.

    READ-ONLY reuse of R2's frozen low-level group science for the rollouts, then the
    Stage 4D addition: all eleven post-perturbation snapshots must be pairwise identical
    (``snapshots_identical``). A branch rollout that does not expose a snapshot cannot prove
    consistency and is treated as inconsistent (fail-closed).
    """
    group = r2collector.collect_group(seed, t_star, prefix_actions, factual_plan,
                                      config, branch_rollout)
    branches = [group["factual"], *group["cf"]]
    snapshots = [b.get("post_perturbation_snapshot") for b in branches]
    consistent = True
    if any(s is None for s in snapshots):
        consistent = False
    else:
        reference = snapshots[0]
        consistent = all(ood.snapshots_identical(reference, s) for s in snapshots[1:])
    group["snapshot_consistent"] = bool(consistent)
    return group


def validate_complete_group(group: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the complete-group acceptance rules (engineering whitelist ONLY).

    READ-ONLY reuse of R2's frozen validator, then the Stage 4D snapshot-consistency gate:
    an accepted group whose eleven post-perturbation snapshots diverged is rejected as a
    WHOLE group with the engineering reason ``branch_perturbation_inconsistent``. Acceptance
    reads ONLY structural/simulation invariants — regret, oracle action, collision, response
    strength or model error CANNOT reject here.
    """
    verdict = r2collector.validate_complete_group(group, config)
    if not verdict["accepted"]:
        return verdict
    if not group.get("snapshot_consistent", False):
        return {
            "accepted": False,
            "reason": "branch_perturbation_inconsistent",
            "failed_branch": "group",
            "episode_seed": int(group["episode_seed"]),
            "t_star": int(group["t_star"]),
        }
    return verdict


def record_from_verdict(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one accepted validator verdict into this stage's publish record."""
    if not verdict.get("accepted"):
        raise ValueError("only accepted verdicts become publish records")
    return {
        "object_history": np.asarray(verdict["history"], dtype=np.float32),
        "object_factual_future": np.asarray(verdict["factual_future"], dtype=np.float32),
        "object_counterfactual_futures": np.asarray(verdict["cf_futures"], dtype=np.float32),
        "factual_actions": np.asarray(verdict["factual_plan"], dtype=np.int64),
        "cf_actions": np.asarray(verdict["cf_actions"], dtype=np.int64),
        "cf_durations": np.asarray(verdict["cf_durations"], dtype=np.int64),
        "inference_mask": np.asarray(verdict["inference_mask"], dtype=bool),
        "group_seed": int(verdict["episode_seed"]),
        "t_star": int(verdict["t_star"]),
        "n_mask_cells": int(verdict["n_mask_cells"]),
        "prefix_max_abs_diff": float(verdict["prefix_max_abs_diff"]),
    }


# --- role paths / schedules / action stream ---------------------------------------------------


def role_paths(config: Mapping[str, Any], role: str) -> tuple[Path, Path, Path, Path]:
    """Resolve (inference_inputs, confirmation_truth, rejection_ledger, receipt).

    smoke gets its own ``smoke/`` subdirectories; pilots live under ``pilot/{profile}/``;
    d1_confirm binds the formal config ``paths``. Engineering data can never mix into the
    formal artifact paths.
    """
    checked = prot.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4D OOD collection role: {role!r}")
    data_root = _resolve(checked["data_root"])
    run_dir = _resolve(checked["run_dir"])
    if role == "smoke":
        inputs = data_root / "smoke" / "inference_inputs.npz"
        truth = data_root / "smoke" / "confirmation_truth.npz"
        ledger = data_root / "smoke" / "rejection_ledger.jsonl"
        receipt = run_dir / "smoke" / "smoke_collection_receipt.json"
    elif role.startswith("pilot_"):
        short = role[len("pilot_"):]
        inputs = data_root / "pilot" / short / "inference_inputs.npz"
        truth = data_root / "pilot" / short / "confirmation_truth.npz"
        ledger = data_root / "pilot" / short / "rejection_ledger.jsonl"
        receipt = run_dir / "pilot" / short / "collection_receipt.json"
    else:  # d1_confirm
        paths = checked["paths"]
        inputs = _resolve(paths["inference_inputs"])
        truth = _resolve(paths["confirmation_truth"])
        ledger = inputs.parent / "rejection_ledger.jsonl"
        receipt = _resolve(paths["collection_receipt"])
    return inputs, truth, ledger, receipt


def role_schedule(checked: Mapping[str, Any], role: str) -> np.ndarray:
    """The role's frozen candidate schedule (this stage's approved §6.2 domains ONLY)."""
    role = str(role)
    if role not in ROLE_SCHEDULE:
        raise ValueError(f"unknown Stage 4D OOD collection role: {role!r}")
    entropy, size = ROLE_SCHEDULE[role]
    return prot.make_candidate_seed_schedule(entropy, size)


def _action_rng(checked: Mapping[str, Any], role: str) -> np.random.Generator:
    """The frozen action-sampling stream (plan §6.2, verbatim roots)."""
    root = prot.action_stream_root(checked, role)
    return np.random.default_rng(np.random.SeedSequence([int(root[0]), int(root[1])]))


def _checkpoint_path(inputs_path: Path) -> Path:
    """The atomic resume checkpoint (collection_checkpoint*; never a receipt)."""
    return inputs_path.parent / "collection_checkpoint.partial.npz"


def _resolve_profile(checked: Mapping[str, Any], role: str,
                     profile: str | None) -> tuple[str, int]:
    """Resolve (profile, quota) for ``role``; fail closed for the formal D1 role.

    smoke/pilots use their frozen profile + quota. d1_confirm reads the power-frozen
    selected n and the D0-frozen profile from the burned protocol lock — which must already
    exist (``prot.verify_protocol_lock`` raises ``FileNotFoundError`` before any D1
    collection, fail-closed).
    """
    role = str(role)
    if role == "d1_confirm":
        lock = prot.verify_protocol_lock(_resolve(checked["run_dir"]))
        quota = int(lock.get("selected_n"))
        locked_profile = str(lock.get("ood_profile"))
        if profile is not None and str(profile) != locked_profile:
            raise ValueError(
                f"d1_confirm profile must be the D0-frozen {locked_profile!r}; "
                f"got {profile!r}")
        return locked_profile, quota
    expected = ROLE_PROFILE[role]
    if profile is not None and str(profile) != expected:
        raise ValueError(f"{role} must apply its frozen profile {expected!r}; "
                         f"got {profile!r}")
    return expected, int(ROLE_QUOTA[role])


# --- grouped collection state -----------------------------------------------------------------


def _empty_state() -> dict[str, Any]:
    return {
        "groups": [],
        "accepted_seeds": set(),
        "skip_counts": Counter({reason: 0 for reason in REASONS}),
        "attempts": 0,
        "prefix_max_abs_diff": 0.0,
    }


def _pack_state(state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    groups = state["groups"]
    arrays: dict[str, np.ndarray] = {}
    for field in _GROUP_ARRAY_FIELDS:
        trailing, dtype = _ARRAY_SPEC[field]
        if groups:
            arrays[field] = np.stack([g[field] for g in groups]).astype(dtype)
        else:
            arrays[field] = np.empty((0,) + tuple(trailing), dtype=dtype)
    for field in _GROUP_SCALAR_INT:
        arrays[field] = np.asarray([g[field] for g in groups], dtype=np.int64)
    metadata = {
        "accepted": len(groups),
        "attempts": int(state["attempts"]),
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "skip_counts": {str(k): int(v) for k, v in state["skip_counts"].items()},
    }
    return metadata, arrays


def _restore_state(arrays: Mapping[str, np.ndarray],
                   metadata: Mapping[str, Any]) -> dict[str, Any]:
    accepted = int(metadata["accepted"])
    for field in tuple(_GROUP_ARRAY_FIELDS) + _GROUP_SCALAR_INT:
        if field not in arrays or len(arrays[field]) != accepted:
            raise ValueError(f"checkpoint field {field} inconsistent")
    state = _empty_state()
    for i in range(accepted):
        record: dict[str, Any] = {
            field: arrays[field][i].copy() for field in _GROUP_ARRAY_FIELDS}
        for field in _GROUP_SCALAR_INT:
            record[field] = int(arrays[field][i])
        record["prefix_max_abs_diff"] = 0.0  # running max lives in metadata
        state["groups"].append(record)
    state["accepted_seeds"] = {int(v) for v in arrays["group_seed"]}
    state["skip_counts"].update(
        {str(k): int(v) for k, v in metadata.get("skip_counts", {}).items()})
    state["attempts"] = int(metadata["attempts"])
    state["prefix_max_abs_diff"] = float(metadata["prefix_max_abs_diff"])
    return state


# --- publish adapter: truth-separated two-file write -------------------------------------------


def _assert_group_contract(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Full-contract assertions on accepted records; return the stacked arrays.

    Includes the mask-provenance proof: every record's inference mask must equal an
    independent recomputation from the legal history/factual contract ONLY. A mask reduced
    from counterfactual presence is rejected before anything is written.
    """
    g_count = len(records)
    arrays: dict[str, np.ndarray] = {}
    for field in _GROUP_ARRAY_FIELDS:
        trailing, dtype = _ARRAY_SPEC[field]
        stacked = np.stack([np.asarray(g[field]) for g in records]).astype(dtype)
        if stacked.shape != (g_count,) + tuple(trailing):
            raise ValueError(f"{field} stacked shape {stacked.shape} != "
                             f"{(g_count,) + tuple(trailing)}")
        arrays[field] = stacked
    for field in _GROUP_SCALAR_INT:
        arrays[field] = np.asarray([g[field] for g in records], dtype=np.int64)

    seeds = arrays["group_seed"]
    if np.unique(seeds).size != g_count:
        raise ValueError("duplicate group_seed among accepted groups")
    t_star = arrays["t_star"]
    if int(t_star.min()) < prot.T_STAR_LOW or int(t_star.max()) > prot.T_STAR_HIGH:
        raise ValueError("t_star out of the frozen [10, 15] range")
    factual_actions = arrays["factual_actions"]
    if int(factual_actions.min()) < 0 or int(factual_actions.max()) >= len(prot.CF_ACTIONS):
        raise ValueError("factual action id out of the five-action env range")
    for field in ("object_history", "object_factual_future",
                  "object_counterfactual_futures"):
        if not np.isfinite(arrays[field]).all():
            raise ValueError(f"non-finite accepted group array: {field}")
    expected_actions = np.asarray([a for a, _ in prot.ACTION_PLANS], dtype=np.int64)
    expected_durations = np.asarray([d for _, d in prot.ACTION_PLANS], dtype=np.int64)
    if not np.all(arrays["cf_actions"] == expected_actions[None]) \
            or not np.all(arrays["cf_durations"] == expected_durations[None]):
        raise ValueError("cf action/duration rows drifted from the frozen ten-action table")
    recomputed = prot.build_inference_mask(
        arrays["object_history"], arrays["object_factual_future"])
    if not np.array_equal(recomputed, arrays["inference_mask"]):
        raise ValueError(
            "inference mask was not generated from the legal factual/history "
            "contract (an illegal counterfactual-mask reduction is suspected)")
    return arrays


def publish_group_data(groups: list, config: dict, role: str) -> dict:
    """Publish the two §6.4 files from the accepted group records (create-once).

    Writes ``inference_inputs.npz`` (EXACTLY the eight legal fields) and the sealed
    ``confirmation_truth.npz`` (CF futures + CF mechanism/integrity fields + same-order
    ``group_seed``), each via tmp-write -> re-read full-contract verification -> hash ->
    atomic replace. The group order of both files is the acceptance order; the returned
    binding is what the collection receipt records. Refuses to overwrite existing final
    files, and refuses any group list whose size is not exactly the role's frozen quota.
    """
    _assert_reused_low_level_contract()
    checked = prot.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4D OOD collection role: {role!r}")
    if role == "d1_confirm":
        # the formal D1 quota is the power-frozen selected n, read from the burned lock.
        quota = int(prot.verify_protocol_lock(
            _resolve(checked["run_dir"])).get("selected_n"))
    else:
        quota = int(ROLE_QUOTA[role])
    records = list(groups)
    if len(records) != quota:
        raise ValueError(
            f"{role} publishes exactly the frozen quota of {quota} accepted "
            f"groups; got {len(records)}")
    inputs_path, truth_path, _, _ = role_paths(checked, role)
    for path in (inputs_path, truth_path):
        if path.exists():
            raise FileExistsError(
                f"published {role} data already exists (create-once): {path}")

    arrays = _assert_group_contract(records)
    legal_payload = {field: arrays[field] for field in LEGAL_INPUT_FIELDS}
    truth_payload = {field: arrays[field] for field in TRUTH_FIELDS}
    legal_seeds = legal_payload["group_seed"]
    truth_seeds = truth_payload["group_seed"]
    if not np.array_equal(legal_seeds, truth_seeds):
        raise RuntimeError("truth/legal group_seed order divergence before publish")

    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for path, payload, expected_fields in (
        (inputs_path, legal_payload, LEGAL_INPUT_FIELDS),
        (truth_path, truth_payload, TRUTH_FIELDS),
    ):
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **payload)
        with np.load(tmp) as archive:
            if set(archive.files) != set(expected_fields):
                raise RuntimeError(
                    f"published archive {path.name} key set drifted: {archive.files}")
            g_count = len(records)
            for field in expected_fields:
                value = archive[field]
                if field in _ARRAY_SPEC:
                    trailing, dtype = _ARRAY_SPEC[field]
                    expected_shape = (g_count,) + tuple(trailing)
                else:  # per-group scalar fields (t_star, group_seed)
                    dtype = np.int64
                    expected_shape = (g_count,)
                if value.shape != expected_shape:
                    raise RuntimeError(
                        f"published {path.name} field {field} shape {value.shape}")
                if value.dtype != np.dtype(dtype):
                    raise RuntimeError(
                        f"published {path.name} field {field} dtype {value.dtype}")
        hashes[path.name] = sha256_file(tmp)
        Path(tmp).replace(path)

    return {
        "inference_inputs": str(inputs_path),
        "inference_inputs_sha256": hashes[inputs_path.name],
        "confirmation_truth": str(truth_path),
        "confirmation_truth_sha256": hashes[truth_path.name],
        "n_groups": len(records),
        "legal_fields": list(LEGAL_INPUT_FIELDS),
        "truth_fields": list(TRUTH_FIELDS),
        "group_order_binding": {
            "order": "acceptance_order",
            "n_groups": len(records),
            "inference_inputs_group_seed_sha256": hashlib.sha256(
                legal_seeds.tobytes()).hexdigest(),
            "confirmation_truth_group_seed_sha256": hashlib.sha256(
                truth_seeds.tobytes()).hexdigest(),
            "same_order": True,
        },
    }


# --- truth-isolated legal-input loader (score side) --------------------------------------------


def _assert_not_truth_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/").lower()
    if "confirmation_truth" in normalized or "truth" in Path(path).name.lower():
        raise ValueError(
            f"the score side must never open the sealed truth file: {path}")


def _screen_archive_keys(names: Sequence[str]) -> None:
    for name in names:
        lowered = str(name).lower()
        if any(token in lowered for token in FORBIDDEN_INPUT_KEY_TOKENS):
            raise ValueError(
                f"score-side input archive contains a truth-side key {name!r}; the "
                "legal input file carries no counterfactual truth (plan §6.4)")


def load_legal_inputs(input_path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Load a published legal-input file for the score side; NEVER touches the truth.

    The sealed truth path is refused by name; the archive is screened for truth-side keys
    fail-closed; ONLY the eight legal fields are read. The sealed truth file need not exist
    — physical truth isolation (Global Constraint 6) means the score path completes with
    the truth file moved away.
    """
    prot.validate_config(config)
    input_path = _resolve(input_path)
    _assert_not_truth_path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"legal inference inputs missing: {input_path}")
    with np.load(input_path, allow_pickle=True) as archive:
        _screen_archive_keys(list(archive.files))
        missing = [key for key in LEGAL_INPUT_FIELDS if key not in archive.files]
        if missing:
            raise ValueError(
                f"legal input archive {input_path} missing required keys: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in LEGAL_INPUT_FIELDS}
    group_seed = np.asarray(arrays["group_seed"], dtype=np.int64).ravel()
    groups = int(group_seed.size)
    return {
        "arrays": arrays,
        "groups": groups,
        "n_actions": prot.N_ACTIONS,  # fixed contract, never read from CF truth
        "group_seed": group_seed,
        "data_signature": sha256_file(input_path),
    }


# --- receipt -------------------------------------------------------------------------------------


def _receipt(checked: Mapping[str, Any], role: str, profile: str,
             state: Mapping[str, Any], quota: int, schedule: np.ndarray,
             inputs_path: Path, truth_path: Path, ledger_path: Path, receipt_path: Path,
             *, binding: Mapping[str, Any] | None, partial: bool) -> dict[str, Any]:
    accepted = len(state["groups"])
    binding = dict(binding or {})
    return {
        "plan_id": prot.PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "purpose": prot.PURPOSE,
        "role": role,
        "profile": profile,
        "status": STATUS_SMOKE if role == "smoke" else STATUS_COLLECTED,
        "partial": bool(partial),
        "requested_groups": int(quota),
        "accepted_groups": int(accepted),
        "counterfactual_branches": int(accepted) * prot.N_ACTIONS,
        "attempts": int(state["attempts"]),
        "seed_schedule_sha256": hashlib.sha256(np.asarray(schedule).tobytes()).hexdigest(),
        "action_stream_root": [int(v) for v in prot.action_stream_root(checked, role)],
        "inference_inputs": str(inputs_path),
        "inference_inputs_sha256": str(binding.get("inference_inputs_sha256", "")),
        "confirmation_truth": str(truth_path),
        "confirmation_truth_sha256": str(binding.get("confirmation_truth_sha256", "")),
        "group_order_binding": dict(binding.get("group_order_binding", {})),
        "legal_fields": list(LEGAL_INPUT_FIELDS),
        "truth_fields": list(TRUTH_FIELDS),
        "rejection_ledger": str(ledger_path),
        "receipt_path": str(receipt_path),
        "skip_counts": {str(k): int(v) for k, v in state["skip_counts"].items()},
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "environment": sp._environment_versions(),
        "config_signature": canonical_fingerprint(json.loads(json.dumps(checked))),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _verify_existing_final(receipt: Mapping[str, Any], inputs_path: Path,
                           truth_path: Path) -> None:
    """Verify a completed role's published files against its receipt; never overwrite."""
    for key, path in (("inference_inputs_sha256", inputs_path),
                      ("confirmation_truth_sha256", truth_path)):
        if not path.is_file() or sha256_file(path) != receipt.get(key):
            raise ValueError(f"published data drifted from its collection receipt: {path}")
    binding = receipt.get("group_order_binding", {})
    with np.load(inputs_path) as archive:
        legal_seeds = np.asarray(archive["group_seed"], dtype=np.int64)
    with np.load(truth_path) as archive:
        truth_seeds = np.asarray(archive["group_seed"], dtype=np.int64)
    if hashlib.sha256(legal_seeds.tobytes()).hexdigest() \
            != binding.get("inference_inputs_group_seed_sha256"):
        raise ValueError("legal file group order drifted from the collection receipt")
    if hashlib.sha256(truth_seeds.tobytes()).hexdigest() \
            != binding.get("confirmation_truth_group_seed_sha256"):
        raise ValueError("truth file group order drifted from the collection receipt")
    if not np.array_equal(legal_seeds, truth_seeds):
        raise ValueError("published legal/truth group order divergence")


# --- role collection ------------------------------------------------------------------------------


def collect_role(
    config: Mapping[str, Any],
    role: str,
    *,
    profile: str | None = None,
    branch_rollout: Callable | None = None,
    probe_fn: Callable | None = None,
    schedule: np.ndarray | None = None,
    stop_after_accepted: int | None = None,
) -> dict[str, Any]:
    """Collect the role's frozen quota of complete eleven-branch OOD groups end-to-end.

    The role <-> schedule binding is FIXED: each role consumes exactly its own frozen
    candidate domain (an injected schedule must equal it; foreign schedules are a smoke-only
    test hook). Acceptance follows the engineering whitelist ONLY — the first
    engineering-valid groups fill the quota; no regret/response/collision/oracle screening.
    The Stage 4D ``branch_rollout`` applies the role's frozen OOD profile post-reset and the
    eleven post-perturbation snapshots are cross-checked per group.

    Resume restores the full action-RNG state and the candidate cursor from the atomic
    checkpoint and produces results identical to uninterrupted collection; only hash-valid
    complete groups are re-accepted. A completed role is verified against its create-once
    receipt and never overwritten. If the candidate domain exhausts first, raises
    :class:`CollectionIncompleteError` (no new domain, no lock burned).

    ``branch_rollout``/``probe_fn``/``schedule``/``stop_after_accepted`` are test hooks; the
    operational runs use the real engine via the defaults.
    """
    _assert_reused_low_level_contract()
    checked = prot.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4D OOD collection role: {role!r}")
    profile, quota = _resolve_profile(checked, role, profile)
    derived = role_schedule(checked, role)
    if schedule is None:
        schedule = derived
    else:
        schedule = np.asarray(schedule, dtype=np.int64)
        if role != "smoke" and not np.array_equal(schedule, derived):
            raise ValueError(
                f"{role} must consume its own frozen candidate domain; injected "
                "schedules are a smoke-only test hook")

    inputs_path, truth_path, ledger_path, receipt_path = role_paths(checked, role)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(inputs_path)

    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _verify_existing_final(receipt, inputs_path, truth_path)
        return {**receipt, "verification": "existing_final_dataset_verified_no_overwrite"}
    if inputs_path.exists() or truth_path.exists():
        raise RuntimeError(
            "final datasets exist without their create-once collection receipt: "
            f"{inputs_path}, {truth_path}")

    # engine config: embed the frozen action-stream root and the OOD profile so the
    # checkpoint fingerprint covers the full sampling + perturbation derivation.
    root = prot.action_stream_root(checked, role)
    engine_config = dict(checked)
    engine_config["collection_seed"] = int(root[0])
    engine_config["action_stream_root"] = [int(root[0]), int(root[1])]
    engine_config["ood_profile"] = profile

    branch_rollout = branch_rollout or make_stage4d_branch_rollout(profile)
    probe_fn = probe_fn or engine._real_probe
    action_ids, action_probs, capacity = probe_fn(engine_config)
    action_ids = np.asarray(action_ids, dtype=np.int64)
    action_probs = np.asarray(action_probs, dtype=np.float64)
    if int(capacity) != prot.OBJECT_CAPACITY:
        raise ValueError(
            f"observed capacity {capacity} != frozen object_capacity "
            f"{prot.OBJECT_CAPACITY}")

    horizon = prot.HORIZON
    action_rng = _action_rng(checked, role)
    state = _empty_state()
    resuming = False
    if checkpoint_path.exists():
        rng_state, metadata, arrays = load_collection_checkpoint(
            checkpoint_path, engine_config)
        state = _restore_state(arrays, metadata)
        action_rng.bit_generator.state = rng_state["action_rng"]
        resuming = True

    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_handle = open(ledger_path, "a" if resuming else "w", encoding="utf-8")

    checkpoint_every = int(checked.get("checkpoint_every_accepted", 25))
    attempts = int(state["attempts"])
    accepted = len(state["groups"])
    reporter = ProgressReporter(
        total=quota,
        every=max(1, checkpoint_every) if checkpoint_every > 0 else quota,
        prefix=f"stage4d-collect-{role}",
        heartbeat_seconds=30.0)

    def write_ledger(record: dict[str, Any]) -> None:
        ledger_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        ledger_handle.flush()

    def checkpoint() -> None:
        metadata, arrays = _pack_state(state)
        save_collection_checkpoint(
            checkpoint_path, engine_config, {"action_rng": action_rng.bit_generator.state},
            metadata, arrays)

    try:
        while accepted < quota and attempts < schedule.size:
            seed = int(schedule[attempts])
            attempt_index = attempts  # the candidate cursor recorded in the ledger
            attempts += 1
            state["attempts"] = attempts

            if seed in state["accepted_seeds"]:
                state["skip_counts"]["duplicate_seed"] += 1
                write_ledger({
                    "attempt_index": int(attempt_index), "group_seed": seed,
                    "decision": "rejected", "reason": "duplicate_seed",
                    "failed_branch": None, "t_star": None,
                })
                continue

            # sampling order: t_star -> prefix actions -> factual plan.
            t_star = int(action_rng.integers(int(checked["t_star_low"]),
                                             int(checked["t_star_high"]) + 1))
            prefix_actions = action_rng.choice(
                action_ids, size=t_star, p=action_probs).astype(np.int64)
            factual_plan = action_rng.choice(
                action_ids, size=horizon, p=action_probs).astype(np.int64)

            group = collect_group(seed, t_star, prefix_actions, factual_plan,
                                  engine_config, branch_rollout)
            verdict = validate_complete_group(group, engine_config)
            if not verdict["accepted"]:
                state["skip_counts"][verdict["reason"]] += 1
                write_ledger({
                    "attempt_index": int(attempt_index), "group_seed": seed,
                    "decision": "rejected", "reason": verdict["reason"],
                    "failed_branch": verdict["failed_branch"], "t_star": t_star,
                })
                continue

            state["groups"].append(record_from_verdict(verdict))
            state["accepted_seeds"].add(seed)
            state["prefix_max_abs_diff"] = max(
                state["prefix_max_abs_diff"], float(verdict["prefix_max_abs_diff"]))
            accepted += 1
            write_ledger({
                "attempt_index": int(attempt_index), "group_seed": seed,
                "decision": "accepted", "reason": "", "t_star": t_star,
                "n_mask_cells": int(verdict["n_mask_cells"]),
            })
            reporter.update(accepted, attempted=attempts)
            reporter.heartbeat(accepted, attempted=attempts)

            if stop_after_accepted is not None and accepted >= stop_after_accepted:
                checkpoint()
                ledger_handle.close()
                return _receipt(checked, role, profile, state, quota, schedule,
                                inputs_path, truth_path, ledger_path, receipt_path,
                                binding=None, partial=True)
            if checkpoint_every > 0 and accepted % checkpoint_every == 0:
                checkpoint()
    finally:
        if not ledger_handle.closed:
            ledger_handle.close()

    if accepted < quota:
        receipt = _receipt(checked, role, profile, state, quota, schedule,
                           inputs_path, truth_path, ledger_path, receipt_path,
                           binding=None, partial=True)
        raise CollectionIncompleteError(json.dumps(
            {**receipt, "status": STATUS_COLLECTION_INCOMPLETE},
            indent=2, ensure_ascii=False))

    checkpoint()
    binding = publish_group_data(state["groups"], checked, role)
    checkpoint_path.unlink(missing_ok=True)
    receipt = _receipt(checked, role, profile, state, quota, schedule,
                       inputs_path, truth_path, ledger_path, receipt_path,
                       binding=binding, partial=False)
    write_once_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Stage 4D D0/D1 Merge model-mismatch OOD groups")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--profile", default=None,
                        help="OOD profile override (must equal the role's frozen profile)")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        receipt = collect_role(config, args.role, profile=args.profile)
    except CollectionIncompleteError as exc:
        print(f"[stage4d-collect] {STATUS_COLLECTION_INCOMPLETE}: {exc}", flush=True)
        return 4
    print(
        f"[stage4d-collect] role={receipt['role']} profile={receipt['profile']} "
        f"status={receipt['status']} accepted={receipt['accepted_groups']} "
        f"attempts={receipt['attempts']} partial={receipt['partial']}",
        flush=True,
    )
    return 0


__all__ = [
    "CollectionIncompleteError",
    "FORBIDDEN_INPUT_KEY_TOKENS",
    "LEGAL_INPUT_FIELDS",
    "REASONS",
    "ROLES",
    "ROLE_PROFILE",
    "ROLE_QUOTA",
    "ROLE_SCHEDULE",
    "STATUS_COLLECTED",
    "STATUS_COLLECTION_INCOMPLETE",
    "STATUS_SMOKE",
    "TRUTH_FIELDS",
    "collect_group",
    "collect_role",
    "load_legal_inputs",
    "main",
    "make_stage4d_branch_rollout",
    "publish_group_data",
    "record_from_verdict",
    "role_paths",
    "role_schedule",
    "validate_complete_group",
]


if __name__ == "__main__":
    raise SystemExit(main())
