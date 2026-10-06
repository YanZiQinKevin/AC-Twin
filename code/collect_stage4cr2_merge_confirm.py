"""Stage 4C-R2 independent-confirmation collector: truth-separated two-file publish.

Plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`` sections 3.1, 4.2, 5.1,
5.2, 10 and Task 2. This stage collects 2,000 FRESH Merge ten-action decision
groups for the one independent confirmation of the frozen R2 direct-supervised
decision-risk model. It retrains nothing, refits nothing and re-judges nothing
historical.

Reuse policy (plan section 5.1): the low-level rollout/QA science is REUSED by
calling ``collect_stage4cr2_merge_multiaction.collect_group`` /
``validate_complete_group`` read-only — one fresh group seed -> one history, one
factual branch, one ``t_star`` -> TEN counterfactual branches forked from the
SAME pre-intervention state; object slot == stable entity id. This module never
impersonates an R2 development role (``collect_role``/``role_paths``/R2 receipts
are never called) and never overwrites R2 module constants; a frozen-contract
guard fails loudly if the reused low-level contract ever drifts from this stage's
protocol constants.

Truth separation (plan section 5.1): collection publishes TWO files per role —
``inference_inputs.npz`` (EXACTLY the eight legal fields; never any CF truth) and
the sealed ``confirmation_truth.npz`` (CF futures + CF mechanism/integrity fields
+ same-order ``group_seed``). The input mask is generated ONLY from the legal
factual/history contract (``build_inference_mask``); it is never derived by
reducing a counterfactual mask, and the publish adapter proves it by independent
recomputation. CF present/mask data is used at collection time ONLY inside the
reused validator to verify completeness/entity correspondence — never as a
prediction feature. Both files carry the same-order ``group_seed``; the
collection receipt binds the two files' group ordering (acceptance order) and
hashes.

Acceptance (plan section 5.2): the R2 engineering whitelist only — branch
missing / per-frame stable-ID present-set mismatch / prefix divergence / illegal
mask reduction / shape / capacity / finiteness / duplicate seed / no legal
agent-time reject the WHOLE group. Collision itself, action outcome, response
strength, oracle imbalance, regret, model error and stratum shortfall NEVER
reject. The first engineering-valid groups up to the frozen quota are accepted —
no screening, no stratum backfill. If the candidate domain exhausts first:
``STAGE4CR2_CONFIRM_COLLECTION_INCOMPLETE`` (no new domain, no adjudication lock).

Sampling (plan section 4.2): candidate schedules come ONLY from
``stage4cr2_confirm_protocol``'s hash-pinned generators (confirm entropy
2026093012 / size 12000; smoke entropy 2026093100 / size 64). The action stream is
``default_rng(SeedSequence([2026093012, 0]))`` (confirm) /
``default_rng(SeedSequence([2026093100, 12]))`` (smoke), consumed in R2's original
order per attempt: ``t_star`` -> prefix actions -> factual plan. The atomic
checkpoint stores the complete RNG state and the candidate cursor (``attempts``);
a resumed run produces results identical to uninterrupted collection. The
rejection ledger records every attempt's seed, decision, reason and cursor
(``attempt_index``).

Ordering guard: the formal ``confirm`` role refuses to collect unless the formal
protocol lock already exists and binds the running config signature, the locked
confirm schedule raw hash and this collector's locked code hash (a cheap local
guard; full source/code/seed re-verification belongs to ``verify_protocol`` and
the runner). The ``smoke`` role is pure engineering: it writes ONLY under the
``smoke/`` subdirectories, never touches formal artifact paths and never creates
or modifies the formal protocol lock.

Rollout-injectable so unit tests need no highway_env; the real 12-group
ENGINEERING_SMOKE_ONLY run and the real 2,000-group collection are operational
steps executed later by the controller.
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
from ac_twin import stage4cr2_confirm_protocol as prot  # noqa: E402
from ac_twin.checkpointing import (  # noqa: E402
    canonical_fingerprint,
    load_collection_checkpoint,
    save_collection_checkpoint,
)
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

ROLES = prot.CONFIRM_DATA_ROLES  # ("confirm", "smoke") — no other roles exist here
ROLE_QUOTA = {
    "confirm": prot.CONFIRM_ACCEPTED_GROUPS,   # 2000
    "smoke": prot.SMOKE_ACCEPTED_GROUPS,       # 12
}
STATUS_COLLECTED = "COLLECTED"
STATUS_SMOKE = "ENGINEERING_SMOKE_ONLY"

#: Whole-group rejection reasons (plan section 5.2 whitelist; identical to R2's
#: engineering vocabulary). No outcome-based reason (regret/oracle/collision/
#: response/error/stratum) may ever appear here.
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
)

#: EXACTLY the eight legal fields of the published §5.1 inference_inputs.npz
#: (order mirrors stage4cr2_confirm_inference.LEGAL_INPUT_KEYS). The score-side
#: file NEVER carries counterfactual truth.
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

#: The sealed confirmation_truth.npz: CF futures + CF mechanism fields
#: (cf_actions/cf_durations) + integrity fields (factual plan the CFs forked
#: from, decision time, collection-time legal mask copy) + same-order group_seed.
TRUTH_FIELDS = (
    "object_counterfactual_futures",  # (G,10,20,5,8) float32 — the sealed CF truth
    "cf_actions",                     # (G,10) int64 — CF mechanism
    "cf_durations",                   # (G,10) int64 — CF mechanism
    "factual_actions",                # (G,20) int64 — integrity: factual plan
    "t_star",                         # (G,)   int64 — integrity: decision time
    "group_seed",                     # (G,)   int64 — same-order binding (required)
    "inference_mask",                 # (G,20,5) bool — integrity: legal mask copy
)

#: per-record stacked array fields and their per-group trailing shapes/dtypes.
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

_COLLECTOR_CODE_REL = "collect_stage4cr2_merge_confirm.py"


class CollectionIncompleteError(RuntimeError):
    """Candidate domain exhausted before the frozen quota (plan sections 5.2/8).

    Carries the terminal status ``STAGE4CR2_CONFIRM_COLLECTION_INCOMPLETE``; no
    new domain is substituted and no adjudication lock is burned.
    """

    status = prot.STATUS_COLLECTION_INCOMPLETE


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


# --- reused low-level contract guard -------------------------------------------------------


def _assert_reused_low_level_contract() -> None:
    """Fail loudly if the reused R2 low-level rollout/QA contract ever drifts.

    Read-only comparison: this stage's protocol constants are bitwise the same
    contract as R2's (Task 1 asserts the action table against the frozen Stage 4C
    objective). The reused ``collect_group``/``validate_complete_group`` read R2
    module constants; any drift between the two contracts would silently change
    the collected science, so it is a hard stop here.
    """
    pairs = (
        ("ENV_ID", s4cr2.ENV_ID, prot.ENV_ID),
        ("HISTORY_LEN", s4cr2.HISTORY_LEN, prot.HISTORY_LEN),
        ("HORIZON", s4cr2.HORIZON, prot.HORIZON),
        ("T_STAR_LOW", s4cr2.T_STAR_LOW, prot.T_STAR_LOW),
        ("T_STAR_HIGH", s4cr2.T_STAR_HIGH, prot.T_STAR_HIGH),
        ("OBJECT_CAPACITY", s4cr2.OBJECT_CAPACITY, prot.OBJECT_CAPACITY),
        ("N_OBJECT_FEATURES", s4cr2.N_OBJECT_FEATURES, prot.N_OBJECT_FEATURES),
        ("N_ACTIONS", s4cr2.N_ACTIONS, prot.N_ACTIONS),
        ("EGO_SLOT", s4cr2.EGO_SLOT, prot.EGO_SLOT),
        ("ACTION_PLANS", [tuple(p) for p in s4cr2.ACTION_PLANS],
         [tuple(p) for p in prot.ACTION_PLANS]),
    )
    for name, r2_value, confirm_value in pairs:
        if r2_value != confirm_value:
            raise RuntimeError(
                f"reused R2 low-level contract drifted on {name}: {r2_value!r} != "
                f"{confirm_value!r}")
    if s4cr2.build_inference_mask is not prot.build_inference_mask:
        raise RuntimeError("reused build_inference_mask is not the protocol's function")
    if s4cr2.materialize_action_plan is not prot.materialize_action_plan:
        raise RuntimeError("reused materialize_action_plan is not the protocol's function")
    for name, r2_module_value, confirm_value in (
        ("r2collector.N_ACTIONS", r2collector.N_ACTIONS, prot.N_ACTIONS),
        ("r2collector.HISTORY_LEN", r2collector.HISTORY_LEN, prot.HISTORY_LEN),
        ("r2collector.FUTURE_LEN", r2collector.FUTURE_LEN, prot.HORIZON),
        ("r2collector.N_OBJECT_FEATURES", r2collector.N_OBJECT_FEATURES,
         prot.N_OBJECT_FEATURES),
        ("r2collector.OBJECT_CAPACITY_CONST", r2collector.OBJECT_CAPACITY_CONST,
         prot.OBJECT_CAPACITY),
    ):
        if r2_module_value != confirm_value:
            raise RuntimeError(
                f"reused R2 collector module constant {name} drifted: "
                f"{r2_module_value!r} != {confirm_value!r}")


_assert_reused_low_level_contract()


# --- reused low-level rollout/QA wrappers ----------------------------------------------------


def collect_group(
    seed: int,
    t_star: int,
    prefix_actions: np.ndarray,
    factual_plan: np.ndarray,
    config: Mapping[str, Any],
    branch_rollout: Callable,
) -> dict[str, Any]:
    """Roll out one factual + ten counterfactual branches from the same initial state.

    Thin READ-ONLY reuse of R2's frozen low-level science
    (``collect_stage4cr2_merge_multiaction.collect_group``): the ten CF plans are
    ``materialize_action_plan(factual_plan, action, duration)`` in the frozen
    ACTION_PLANS order; ``(IDLE,1)`` and ``(IDLE,3)`` stay distinct.
    """
    return r2collector.collect_group(seed, t_star, prefix_actions, factual_plan,
                                     config, branch_rollout)


def validate_complete_group(group: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the plan-section-5.2 complete-group acceptance rules (R2 whitelist).

    READ-ONLY reuse of R2's frozen validator. Acceptance reads ONLY
    structural/simulation invariants — the signature deliberately has no
    truth-side channel, so regret, oracle action, collision, response strength or
    model error CANNOT reject here. The inference mask is built WITHOUT any
    counterfactual read (legal history/factual contract only); CF presence is
    used solely to verify completeness/entity correspondence.
    """
    return r2collector.validate_complete_group(group, config)


def record_from_verdict(verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one accepted validator verdict into this stage's publish record.

    Renames R2's ``episode_seed`` to this stage's ``group_seed``; carries the
    per-group arrays under their published field names.
    """
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

    The formal confirm role binds the frozen config paths; smoke gets its own
    ``smoke/`` subdirectories so engineering data can never mix into the formal
    artifacts. Every data path passes the Task 1 namespace guard before any
    read/write.
    """
    checked = prot.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4C-R2 confirm collection role: {role!r}")
    data_root = _resolve(checked["data_root"])
    run_dir = _resolve(checked["run_dir"])
    if role == "smoke":
        inputs = data_root / "smoke" / "inference_inputs.npz"
        truth = data_root / "smoke" / "confirmation_truth.npz"
        ledger = data_root / "smoke" / "rejection_ledger.jsonl"
        receipt = run_dir / "smoke" / "smoke_collection_receipt.json"
    else:
        paths = checked["paths"]
        inputs = _resolve(paths["inference_inputs"])
        truth = _resolve(paths["confirmation_truth"])
        ledger = _resolve(paths["rejection_ledger"])
        receipt = _resolve(paths["collection_receipt"])
    for path in (inputs, truth, ledger):
        prot.assert_allowed_confirm_data(path, role=role)
    return inputs, truth, ledger, receipt


def role_schedule(checked: Mapping[str, Any], role: str) -> np.ndarray:
    """The role's frozen candidate schedule (Task 1 hash-pinned generators ONLY)."""
    role = str(role)
    if role == "confirm":
        return prot.build_confirm_seed_schedule(checked)
    if role == "smoke":
        return prot.build_smoke_seed_schedule(checked)
    raise ValueError(f"unknown Stage 4C-R2 confirm collection role: {role!r}")


def _action_rng(checked: Mapping[str, Any], role: str) -> np.random.Generator:
    """The frozen action-sampling stream (plan section 4.2, verbatim roots)."""
    root = prot.action_stream_root(checked, role)
    return np.random.default_rng(np.random.SeedSequence([int(root[0]), int(root[1])]))


def _checkpoint_path(inputs_path: Path) -> Path:
    """The atomic resume checkpoint (collection_checkpoint*; never a receipt)."""
    return inputs_path.parent / "collection_checkpoint.partial.npz"


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

    Includes the mask-provenance proof: every record's inference mask must equal
    an independent recomputation from the legal history/factual contract ONLY.
    A mask reduced from counterfactual presence (or anything else) is rejected
    before anything is written.
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
    if int(factual_actions.min()) < 0 \
            or int(factual_actions.max()) >= len(prot.CF_ACTIONS):
        raise ValueError("factual action id out of the five-action env range")
    for field in ("object_history", "object_factual_future",
                  "object_counterfactual_futures"):
        if not np.isfinite(arrays[field]).all():
            raise ValueError(f"non-finite accepted group array: {field}")
    # the ten-action order matches the frozen table for EVERY group; (IDLE,1)
    # and (IDLE,3) are never merged.
    expected_actions = np.asarray([a for a, _ in prot.ACTION_PLANS], dtype=np.int64)
    expected_durations = np.asarray([d for _, d in prot.ACTION_PLANS], dtype=np.int64)
    if not np.all(arrays["cf_actions"] == expected_actions[None]) \
            or not np.all(arrays["cf_durations"] == expected_durations[None]):
        raise ValueError("cf action/duration rows drifted from the frozen ten-action table")
    # mask provenance: the input mask is generated ONLY by the legal
    # history/factual contract — never by reducing a counterfactual mask.
    recomputed = prot.build_inference_mask(
        arrays["object_history"], arrays["object_factual_future"])
    if not np.array_equal(recomputed, arrays["inference_mask"]):
        raise ValueError(
            "inference mask was not generated from the legal factual/history "
            "contract (an illegal counterfactual-mask reduction is suspected)")
    return arrays


def publish_group_data(groups: list, config: dict, role: str) -> dict:
    """Publish the two §5.1 files from the accepted group records (create-once).

    Writes ``inference_inputs.npz`` (EXACTLY the eight legal fields) and the
    sealed ``confirmation_truth.npz`` (CF futures + CF mechanism/integrity fields
    + same-order ``group_seed``), each via tmp-write -> re-read full-contract
    verification -> hash -> atomic replace. The group order of both files is the
    acceptance order; the returned binding (paths, hashes, per-file group_seed
    hashes, same-order flag) is what the collection receipt records. Refuses to
    overwrite existing final files, and refuses any group list whose size is not
    exactly the role's frozen quota.
    """
    _assert_reused_low_level_contract()
    checked = prot.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4C-R2 confirm collection role: {role!r}")
    quota = ROLE_QUOTA[role]
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


# --- formal-lock ordering guard (confirm role only) ----------------------------------------------


def _assert_formal_lock(checked: Mapping[str, Any]) -> None:
    """Refuse formal confirm collection unless the protocol lock binds this run.

    Cheap local ordering guard (plan section 5.3: collection runs only after the
    candidate domain is frozen): the lock must exist, name this plan, bind the
    running config signature, the freshly derived confirm schedule's raw hash and
    this collector file's locked code hash. The full source/code/seed
    re-verification belongs to ``prot.verify_protocol`` / the runner's ``verify``.
    """
    lock_path = _resolve(checked["paths"]["protocol_lock"])
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"formal confirm collection requires the burned protocol lock first: "
            f"{lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("plan_id") != prot.PLAN_ID:
        raise ValueError(f"protocol lock plan_id must be {prot.PLAN_ID!r}")
    signature = canonical_fingerprint(json.loads(json.dumps(checked)))
    if payload.get("config_signature") != signature:
        raise ValueError("running config does not match the locked configuration")
    schedule = prot.build_confirm_seed_schedule(checked)
    digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    if payload.get("confirm_schedule_raw_sha256") != digest:
        raise ValueError(
            "the locked confirm schedule raw hash drifted from the freshly derived "
            "candidate domain")
    locked_code = payload.get("code_sha256", {})
    if not isinstance(locked_code, Mapping) \
            or locked_code.get(_COLLECTOR_CODE_REL) != sha256_file(
                PROJECT_ROOT / "code" / _COLLECTOR_CODE_REL):
        raise ValueError(
            "the protocol lock does not bind this collector's current code hash "
            "(code drift or a stale lock)")


# --- receipt -------------------------------------------------------------------------------------


def _receipt(checked: Mapping[str, Any], role: str, state: Mapping[str, Any],
             quota: int, schedule: np.ndarray, inputs_path: Path, truth_path: Path,
             ledger_path: Path, receipt_path: Path, *,
             binding: Mapping[str, Any] | None, partial: bool) -> dict[str, Any]:
    accepted = len(state["groups"])
    binding = dict(binding or {})
    return {
        "plan_id": prot.PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "purpose": prot.PURPOSE_CONFIRM,
        "role": role,
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


def collect_confirmation(
    config: dict,
    role: str,
    *,
    branch_rollout: Callable | None = None,
    probe_fn: Callable | None = None,
    schedule: np.ndarray | None = None,
    stop_after_accepted: int | None = None,
) -> dict:
    """Collect the role's frozen quota of complete ten-action groups end-to-end.

    ``role`` is ``"confirm"`` (2,000 groups over the 12,000-seed frozen candidate
    domain; requires the burned protocol lock) or ``"smoke"`` (12 groups over the
    64-seed frozen smoke domain; engineering only, writes solely under
    ``smoke/``). The role <-> schedule binding is FIXED: confirm consumes exactly
    its frozen candidate domain (an injected schedule must equal it; foreign
    schedules are a smoke-only test hook). Acceptance follows the §5.2
    engineering whitelist ONLY — the first engineering-valid groups fill the
    quota; no regret/response/collision/oracle screening and no stratum backfill.

    Resume restores the full action-RNG state and the candidate cursor
    (``attempts``) from the atomic checkpoint and produces results identical to
    uninterrupted collection. A completed role is verified against its
    create-once receipt and never overwritten. If the candidate domain exhausts
    before the quota fills, raises :class:`CollectionIncompleteError` carrying
    ``STAGE4CR2_CONFIRM_COLLECTION_INCOMPLETE`` (no new domain, no lock burned).

    ``branch_rollout``/``probe_fn``/``stop_after_accepted`` are test hooks; the
    operational runs use the real engine via the defaults.
    """
    _assert_reused_low_level_contract()
    checked = prot.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4C-R2 confirm collection role: {role!r}")
    quota = ROLE_QUOTA[role]
    derived = role_schedule(checked, role)
    if schedule is None:
        schedule = derived
    else:
        schedule = np.asarray(schedule, dtype=np.int64)
        if role != "smoke" and not np.array_equal(schedule, derived):
            raise ValueError(
                "confirm must consume its own frozen candidate domain; injected "
                "schedules are a smoke-only test hook")

    inputs_path, truth_path, ledger_path, receipt_path = role_paths(checked, role)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(inputs_path)

    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _verify_existing_final(receipt, inputs_path, truth_path)
        return {**receipt,
                "verification": "existing_final_dataset_verified_no_overwrite"}
    if inputs_path.exists() or truth_path.exists():
        raise RuntimeError(
            "final datasets exist without their create-once collection receipt: "
            f"{inputs_path}, {truth_path}")

    if role == "confirm":
        _assert_formal_lock(checked)

    # engine config: embed the frozen action-stream root so the checkpoint
    # fingerprint covers the full sampling derivation (probe reset seed included).
    root = prot.action_stream_root(checked, role)
    engine_config = dict(checked)
    engine_config["collection_seed"] = int(root[0])
    engine_config["action_stream_root"] = [int(root[0]), int(root[1])]

    branch_rollout = branch_rollout or engine.rollout_branch_real
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
        prefix=f"stage4cr2-confirm-collect-{role}",
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

            # R2's original sampling order: t_star -> prefix -> factual plan.
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
                return _receipt(checked, role, state, quota, schedule,
                                inputs_path, truth_path, ledger_path, receipt_path,
                                binding=None, partial=True)
            if checkpoint_every > 0 and accepted % checkpoint_every == 0:
                checkpoint()
    finally:
        if not ledger_handle.closed:
            ledger_handle.close()

    if accepted < quota:
        receipt = _receipt(checked, role, state, quota, schedule,
                           inputs_path, truth_path, ledger_path, receipt_path,
                           binding=None, partial=True)
        raise CollectionIncompleteError(json.dumps(
            {**receipt, "status": prot.STATUS_COLLECTION_INCOMPLETE},
            indent=2, ensure_ascii=False))

    checkpoint()
    binding = publish_group_data(state["groups"], checked, role)
    checkpoint_path.unlink(missing_ok=True)
    receipt = _receipt(checked, role, state, quota, schedule,
                       inputs_path, truth_path, ledger_path, receipt_path,
                       binding=binding, partial=False)
    write_once_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Stage 4C-R2 independent-confirmation Merge groups")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        receipt = collect_confirmation(config, args.role)
    except CollectionIncompleteError as exc:
        print(f"[stage4cr2-confirm-collect] {prot.STATUS_COLLECTION_INCOMPLETE}: {exc}",
              flush=True)
        return 4
    print(
        f"[stage4cr2-confirm-collect] role={receipt['role']} status={receipt['status']} "
        f"accepted={receipt['accepted_groups']} attempts={receipt['attempts']} "
        f"partial={receipt['partial']}",
        flush=True,
    )
    return 0


__all__ = [
    "CollectionIncompleteError",
    "LEGAL_INPUT_FIELDS",
    "REASONS",
    "ROLES",
    "ROLE_QUOTA",
    "STATUS_COLLECTED",
    "STATUS_SMOKE",
    "TRUTH_FIELDS",
    "collect_confirmation",
    "collect_group",
    "main",
    "publish_group_data",
    "record_from_verdict",
    "role_paths",
    "role_schedule",
    "validate_complete_group",
]


if __name__ == "__main__":
    raise SystemExit(main())
