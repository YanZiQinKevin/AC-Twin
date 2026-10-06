"""Stage 4C-R2 Merge multi-action decision-group collector.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 3, 5, 6.
The complete-group collection science is REUSED from the frozen Stage 4C collector
(``code/collect_stage4c_merge_multiaction.py``): one fresh group seed -> one history, one
factual branch, one ``t_star`` -> TEN counterfactual branches forked from the SAME
pre-intervention state (the frozen ``ACTION_PLANS``, 5 actions x 2 durations); object
slot == stable entity id; the presence channel IS the per-frame stable-ID present set.

Everything output-side lives in the R2 namespace only: data under
``data/object_centric_stage/stage4cr2_merge_multiaction_dev/`` with per-role
``object_twinworld_stage4cr2_{train,validation,screen}.npz`` +
``rejection_ledger_{role}.jsonl`` (smoke keeps its own ``smoke/`` subdir), receipts under
``experiments/stage4_reliability/stage4c_r2/development/``. The Task 1 protocol guard
``assert_allowed_stage4cr2_data`` is enforced on every data path before any read/write.

Roles bind to FIXED schedule segments (train [0,24000), validation [24000,32000), screen
[32000,60000) of the frozen 60,000-seed schedule, entropy 2026092880; smoke uses entropy
2026092700). A role may consume only its own segment — never reassigned by acceptance
order, regret or response. Rejection uses ONLY pre-intervention-knowable structure and
simulation validity (plan section 5.3); no regret, oracle action, collision, response
strength or model error can reject an attempt.

The action-sampling stream (``t_star``, prefix actions, factual plan) is derived
deterministically from the frozen schedule entropy + the role's segment start (smoke:
smoke entropy + smoke quota) via ``numpy.random.SeedSequence`` — no new frozen constant
is introduced and resume restores the exact stream state from the atomic checkpoint.
Rollout-injectable so tests need no highway_env.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage2_holdout as engine  # noqa: E402
from ac_twin import stage2_protocol as sp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin.checkpointing import (  # noqa: E402
    canonical_fingerprint,
    load_collection_checkpoint,
    save_collection_checkpoint,
)
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

HISTORY_LEN = s4cr2.HISTORY_LEN
FUTURE_LEN = s4cr2.HORIZON
N_OBJECT_FEATURES = s4cr2.N_OBJECT_FEATURES
N_ACTIONS = s4cr2.N_ACTIONS
OBJECT_CAPACITY_CONST = s4cr2.OBJECT_CAPACITY

ROLES = ("development_train", "development_validation", "development_screen", "smoke")
ROLE_PREFIX = {
    "development_train": "train",
    "development_validation": "validation",
    "development_screen": "screen",
    "smoke": "smoke",
}
ROLE_QUOTA = {
    "development_train": s4cr2.DEV_TRAIN_GROUPS,
    "development_validation": s4cr2.DEV_VALIDATION_GROUPS,
    "development_screen": s4cr2.DEV_SCREEN_GROUPS,
    "smoke": s4cr2.SMOKE_GROUPS,
}
STATUS_COLLECTED = "COLLECTED"
STATUS_SMOKE = "ENGINEERING_SMOKE_ONLY"
STATUS_COLLECTION_INCOMPLETE = "STAGE4CR2_COLLECTION_INCOMPLETE"

#: Smoke candidate pool size (engineering only; absorbs the group acceptance rate so the
#: 12-group smoke quota is reachable). Smoke seeds come from the frozen smoke entropy via
#: the frozen Stage 2 schedule generator.
SMOKE_CANDIDATE_POOL = 64

#: Whole-group rejection reasons (plan section 5.3): structural/simulation invariants
#: ONLY. No outcome-based reason (regret/oracle/collision/response/error) may appear here.
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


@dataclass(frozen=True)
class CollectionReceipt:
    """One role's collection receipt (mirrors the create-once JSON on disk)."""
    plan_id: str
    role: str
    status: str
    partial: bool
    requested_groups: int
    accepted_groups: int
    counterfactual_branches: int
    attempts: int
    seed_schedule_sha256: str
    data_output: str
    data_output_sha256: str
    rejection_ledger: str
    receipt_path: str
    skip_counts: dict[str, int]
    prefix_max_abs_diff: float
    environment: dict[str, Any]
    config_signature: str
    generated_at: str
    verification: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, verification: str = "") -> "CollectionReceipt":
        data = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        data.pop("verification", None)
        data.setdefault("data_output_sha256", "")
        data.setdefault("data_output", "")
        data.setdefault("rejection_ledger", "")
        data.setdefault("receipt_path", "")
        data.setdefault("status", STATUS_COLLECTED)
        return cls(verification=verification, **data)


class CollectionIncompleteError(RuntimeError):
    """The role's seed segment exhausted before its group quota filled (hard stop)."""


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


# --- role paths / schedule / streams --------------------------------------------------------


def role_paths(config: Mapping[str, Any], role: str) -> tuple[Path, Path, Path]:
    """Resolve (data_output, rejection_ledger, receipt) for ``role``; guard the data paths.

    Formal roles live directly under the R2 purpose dir; smoke gets its own ``smoke/``
    subdir so engineering data can never mix into the formal role files. Both data paths
    must pass the three-layer Task 1 protocol guard before anything reads or writes them.
    """
    checked = s4cr2.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4C-R2 collection role: {role}")
    prefix = ROLE_PREFIX[role]
    data_root = _resolve(checked["data_root"])
    run_dir = _resolve(checked["run_dir"])
    if role == "smoke":
        data_output = data_root / "smoke" / "object_twinworld_stage4cr2_smoke.npz"
        ledger = data_root / "smoke" / "rejection_ledger_smoke.jsonl"
        receipt = run_dir / "smoke" / "smoke_collection_receipt.json"
    else:
        data_output = data_root / f"object_twinworld_stage4cr2_{prefix}.npz"
        ledger = data_root / f"rejection_ledger_{prefix}.jsonl"
        receipt = run_dir / "development" / f"{prefix}_collection_receipt.json"
    s4cr2.assert_allowed_stage4cr2_data(data_output, role=role)
    s4cr2.assert_allowed_stage4cr2_data(ledger, role=role)
    return data_output, ledger, receipt


def role_schedule(checked: Mapping[str, Any], role: str) -> np.ndarray:
    """The role's frozen seed slice: its fixed schedule segment, or the smoke pool."""
    role = str(role)
    if role == "smoke":
        return sp.make_seed_schedule(int(checked["smoke_seed_entropy"]),
                                     size=SMOKE_CANDIDATE_POOL)
    if role not in ROLE_PREFIX:
        raise ValueError(f"unknown Stage 4C-R2 collection role: {role}")
    schedule = s4cr2.build_seed_schedule(checked)
    lo, hi = s4cr2.role_segment(role)
    return schedule[lo:hi]


def action_stream_root(checked: Mapping[str, Any], role: str) -> tuple[int, int]:
    """Deterministic SeedSequence root for the role's action-sampling stream.

    Derived ONLY from frozen config values: (dev entropy, segment start) for the formal
    roles and (smoke entropy, smoke quota) for smoke. Introduces no new frozen constant.
    """
    if str(role) == "smoke":
        return (int(checked["smoke_seed_entropy"]), int(checked["smoke_groups"]))
    lo, _ = s4cr2.role_segment(str(role))
    return (int(checked["dev_seed_entropy"]), int(lo))


def _action_rng(checked: Mapping[str, Any], role: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(action_stream_root(checked, role)))


# --- single-group collection (science reused from the frozen Stage 4C collector) -----------


def collect_group(
    seed: int,
    t_star: int,
    prefix_actions: np.ndarray,
    factual_plan: np.ndarray,
    config: Mapping[str, Any],
    branch_rollout: Callable,
) -> dict[str, Any]:
    """Roll out one factual + ten counterfactual branches from the same initial state."""
    capacity = int(config["object_capacity"])
    env_cfg = engine._env_config(config)
    factual = branch_rollout(config["env_id"], env_cfg, seed, prefix_actions,
                             factual_plan, capacity)
    cf_branches = []
    for action, duration in s4cr2.ACTION_PLANS:
        plan = s4cr2.materialize_action_plan(factual_plan, action, duration)
        cf_branches.append(
            branch_rollout(config["env_id"], env_cfg, seed, prefix_actions, plan, capacity))
    return {
        "episode_seed": int(seed),
        "t_star": int(t_star),
        "factual_plan": np.asarray(factual_plan, dtype=np.int64),
        "factual": factual,
        "cf": cf_branches,
    }


def _present_set(frames: np.ndarray) -> np.ndarray:
    """Boolean (T, N) present stable-ID set (slot == entity id)."""
    return np.asarray(frames)[..., 0] > 0


def validate_complete_group(group: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the plan-section-5.3 complete-group acceptance rules.

    Returns ``{"accepted": bool, "reason": str, "failed_branch": str|None, ...}`` plus,
    on acceptance, the stacked tensors. Acceptance reads ONLY structural/simulation
    invariants — the signature deliberately has no truth-side channel, so a regret,
    oracle action, collision flag, response strength or model error CANNOT reject here.
    """
    history_len = int(config["history_len"])
    horizon = int(config["horizon"])
    capacity = int(config["object_capacity"])
    t_star = int(group["t_star"])
    factual = group["factual"]
    cf = group["cf"]

    def reject(reason: str, branch: str) -> dict[str, Any]:
        return {"accepted": False, "reason": reason, "failed_branch": branch,
                "episode_seed": int(group["episode_seed"]), "t_star": t_star}

    branches = [("factual", factual)] + [(f"cf_{i}", b) for i, b in enumerate(cf)]
    # 0. ten branches exactly, with observation capacity/shape intact everywhere.
    if len(cf) != N_ACTIONS:
        return reject("counterfactual_future_short", "group")
    for name, b in branches:
        prefix_frames = np.asarray(b["prefix_frames"])
        future_frames = np.asarray(b["future_frames"])
        if (prefix_frames.ndim != 3 or future_frames.ndim != 3
                or prefix_frames.shape[1:] != (capacity, N_OBJECT_FEATURES)
                or future_frames.shape[1:] != (capacity, N_OBJECT_FEATURES)):
            return reject("observation_shape", name)

    # 1. simulation validity + prefix length + finite futures.
    for name, b in branches:
        if b["terminated"] or b["truncated"] or len(b["prefix_frames"]) != t_star + 1:
            return reject("prefix_terminated", name)
        if len(b["future_frames"]) != horizon:
            return reject("factual_future_short" if name == "factual"
                          else "counterfactual_future_short", name)
        if not np.isfinite(b["future_frames"]).all():
            return reject("non_finite", name)

    # 2. shared pre-intervention state: prefix frames elementwise identical across all 11.
    ref_prefix = factual["prefix_frames"]
    prefix_max = 0.0
    for name, b in branches[1:]:
        if b["prefix_frames"].shape != ref_prefix.shape:
            return reject("object_prefix_mismatch", name)
        diff = float(np.max(np.abs(b["prefix_frames"] - ref_prefix)))
        prefix_max = max(prefix_max, diff)
        if diff > 1e-6:
            return reject("object_prefix_mismatch", name)

    # 3. initial stable-entity fingerprints identical across all 11 branches.
    for name, b in branches[1:]:
        if not engine._fingerprints_consistent(factual["fingerprints"], b["fingerprints"]):
            return reject("initial_fingerprint_mismatch", name)

    history = factual["prefix_frames"][-history_len:]
    if history.shape[0] != history_len:
        return reject("short_history", "factual")

    # 4. per-frame present stable-ID sets identical across factual + all 10 cf branches
    #    over the WHOLE horizon (merge has no mid-episode dynamic inflow).
    factual_present = _present_set(factual["future_frames"])  # (T, N)
    for i, b in enumerate(cf):
        if not np.array_equal(_present_set(b["future_frames"]), factual_present):
            return reject("present_set_mismatch", f"cf_{i}")

    # 5. inference mask built WITHOUT counterfactual; the per-branch evaluation mask must
    #    equal it for an accepted group.
    inference_mask = s4cr2.build_inference_mask(
        history[None], factual["future_frames"][None])[0]
    for i, b in enumerate(cf):
        cf_present = _present_set(b["future_frames"])
        eval_mask = inference_mask & cf_present  # cf presence must not shrink it
        if not np.array_equal(eval_mask, inference_mask):
            return reject("eval_mask_mismatch", f"cf_{i}")

    # 6. at least one legal non-ego agent-time.
    n_mask_cells = int(inference_mask.sum())
    if n_mask_cells < 1:
        return reject("no_legal_agent_time", "factual")

    cf_futures = np.stack([b["future_frames"] for b in cf]).astype(np.float32)
    cf_actions = np.asarray([a for a, _ in s4cr2.ACTION_PLANS], dtype=np.int64)
    cf_durations = np.asarray([d for _, d in s4cr2.ACTION_PLANS], dtype=np.int64)
    return {
        "accepted": True,
        "reason": "",
        "failed_branch": None,
        "episode_seed": int(group["episode_seed"]),
        "t_star": t_star,
        "history": history.astype(np.float32),
        "factual_future": factual["future_frames"].astype(np.float32),
        "cf_futures": cf_futures,
        "factual_plan": np.asarray(group["factual_plan"], dtype=np.int64),
        "cf_actions": cf_actions,
        "cf_durations": cf_durations,
        "inference_mask": inference_mask,
        "n_mask_cells": n_mask_cells,
        "prefix_max_abs_diff": prefix_max,
    }


# --- grouped collection state ---------------------------------------------------------------

#: per-group stacked fields and their leading shapes (beyond the group axis).
_GROUP_ARRAY_FIELDS = (
    "object_history",                    # (G,10,5,8)
    "object_factual_future",             # (G,20,5,8)
    "object_counterfactual_futures",     # (G,10,20,5,8)
    "factual_actions",                   # (G,20)
    "cf_actions",                        # (G,10)
    "cf_durations",                      # (G,10)
    "inference_mask",                    # (G,20,5) bool
)
_GROUP_SCALAR_INT = ("episode_seed", "t_star", "n_mask_cells")


def _empty_state() -> dict[str, Any]:
    state: dict[str, Any] = {field: [] for field in _GROUP_ARRAY_FIELDS}
    state.update({field: [] for field in _GROUP_SCALAR_INT})
    state["skip_counts"] = Counter({reason: 0 for reason in REASONS})
    state["attempts"] = 0
    state["prefix_max_abs_diff"] = 0.0
    return state


def _pack_state(state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    accepted = len(state["object_history"])
    arrays: dict[str, np.ndarray] = {}
    for field in _GROUP_ARRAY_FIELDS:
        values = state[field]
        if len(values) != accepted:
            raise ValueError(f"field {field} has inconsistent accepted count")
        if values:
            arrays[field] = np.stack(values)
        else:
            arrays[field] = np.empty((0,), dtype=np.float32)
    for field in _GROUP_SCALAR_INT:
        arrays[field] = np.asarray(state[field], dtype=np.int64)
    metadata = {
        "accepted": accepted,
        "attempts": int(state["attempts"]),
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "skip_counts": {str(k): int(v) for k, v in state["skip_counts"].items()},
    }
    return metadata, arrays


def _restore_state(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> dict[str, Any]:
    accepted = int(metadata["accepted"])
    state = _empty_state()
    for field in _GROUP_ARRAY_FIELDS:
        values = arrays.get(field)
        if values is None or len(values) != accepted:
            raise ValueError(f"checkpoint field {field} inconsistent")
        state[field] = [v.copy() for v in values]
    for field in _GROUP_SCALAR_INT:
        state[field] = [int(v) for v in arrays[field]]
    state["skip_counts"].update({str(k): int(v) for k, v in metadata.get("skip_counts", {}).items()})
    state["attempts"] = int(metadata["attempts"])
    state["prefix_max_abs_diff"] = float(metadata["prefix_max_abs_diff"])
    return state


# --- role collection ------------------------------------------------------------------------


def collect_role(
    config: Mapping[str, Any],
    role: str,
    quota: int,
    *,
    schedule: np.ndarray | None = None,
    branch_rollout: Callable | None = None,
    probe_fn: Callable | None = None,
    stop_after_accepted: int | None = None,
) -> CollectionReceipt:
    """Collect ``quota`` complete 10-action groups for one role from its frozen segment.

    The role <-> schedule-segment binding is FIXED by the Task 1 protocol: a formal role
    consumes its own segment in frozen order (an injected schedule must equal it exactly;
    only smoke accepts a foreign schedule, as a test hook). Roles are never reassigned by
    acceptance order, regret or response. Resume continues the per-group atomic
    ``.partial.npz`` checkpoint without re-accepting a group; the final NPZ + create-once
    receipt are never overwritten. ``stop_after_accepted`` is a test hook.
    """
    checked = s4cr2.validate_config(config)
    role = str(role)
    if role not in ROLES:
        raise ValueError(f"unknown Stage 4C-R2 collection role: {role}")
    quota = int(quota)
    if quota != ROLE_QUOTA[role]:
        raise ValueError(
            f"{role} quota is frozen at {ROLE_QUOTA[role]} groups; got {quota}")
    derived = role_schedule(checked, role)
    if schedule is None:
        schedule = derived
    else:
        schedule = np.asarray(schedule, dtype=np.int64)
        if role != "smoke" and not np.array_equal(schedule, derived):
            raise ValueError(
                f"{role} must consume its own frozen schedule segment; "
                "injected schedules are a smoke-only test hook")

    data_output, ledger_path, receipt_path = role_paths(checked, role)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = data_output.with_suffix(".partial.npz")

    if data_output.exists():
        if not receipt_path.exists():
            raise RuntimeError(
                f"final dataset exists without its create-once receipt: {data_output}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("data_output_sha256") != sha256_file(data_output):
            raise ValueError(f"final dataset drifted from its receipt: {data_output}")
        return CollectionReceipt.from_dict(
            receipt, verification="existing_final_dataset_verified_no_overwrite")

    # engine config: embed the deterministic action-stream root as the probe reset seed so
    # the checkpoint fingerprint covers the full sampling derivation.
    engine_config = dict(checked)
    engine_config["collection_seed"] = int(action_stream_root(checked, role)[0])

    branch_rollout = branch_rollout or engine.rollout_branch_real
    probe_fn = probe_fn or engine._real_probe
    action_ids, action_probs, capacity = probe_fn(engine_config)
    action_ids = np.asarray(action_ids, dtype=np.int64)
    action_probs = np.asarray(action_probs, dtype=np.float64)
    if int(capacity) != int(checked["object_capacity"]):
        raise ValueError(
            f"observed capacity {capacity} != frozen object_capacity "
            f"{checked['object_capacity']}")

    horizon = int(checked["horizon"])

    action_rng = _action_rng(checked, role)
    state = _empty_state()
    resuming = False
    if partial_path.exists():
        rng_state, metadata, arrays = load_collection_checkpoint(partial_path, engine_config)
        state = _restore_state(arrays, metadata)
        action_rng.bit_generator.state = rng_state["action_rng"]
        resuming = True

    data_output.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_handle = open(ledger_path, "a" if resuming else "w", encoding="utf-8")

    checkpoint_every = int(checked.get("checkpoint_every_accepted", 25))
    attempts = int(state["attempts"])
    accepted = len(state["object_history"])
    reporter = ProgressReporter(
        total=quota, every=max(1, checkpoint_every), prefix=f"stage4cr2-collect-{role}",
        heartbeat_seconds=30.0)

    def write_ledger(record: dict[str, Any]) -> None:
        ledger_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        ledger_handle.flush()

    def checkpoint() -> None:
        metadata, arrays = _pack_state(state)
        save_collection_checkpoint(
            partial_path, engine_config, {"action_rng": action_rng.bit_generator.state},
            metadata, arrays)

    try:
        while accepted < quota and attempts < schedule.size:
            seed = int(schedule[attempts])
            attempt_index = attempts
            attempts += 1
            state["attempts"] = attempts

            if seed in set(state["episode_seed"]):
                state["skip_counts"]["duplicate_seed"] += 1
                write_ledger({
                    "attempt_index": int(attempt_index), "episode_seed": seed,
                    "decision": "rejected", "reason": "duplicate_seed",
                    "failed_branch": None, "t_star": None,
                })
                continue

            t_star = int(action_rng.integers(int(checked["t_star_low"]),
                                             int(checked["t_star_high"]) + 1))
            prefix_actions = action_rng.choice(action_ids, size=t_star, p=action_probs).astype(np.int64)
            factual_plan = action_rng.choice(action_ids, size=horizon, p=action_probs).astype(np.int64)

            group = collect_group(seed, t_star, prefix_actions, factual_plan,
                                  engine_config, branch_rollout)
            verdict = validate_complete_group(group, engine_config)
            if not verdict["accepted"]:
                state["skip_counts"][verdict["reason"]] += 1
                write_ledger({
                    "attempt_index": int(attempt_index), "episode_seed": seed,
                    "decision": "rejected", "reason": verdict["reason"],
                    "failed_branch": verdict["failed_branch"], "t_star": t_star,
                })
                continue

            state["object_history"].append(verdict["history"])
            state["object_factual_future"].append(verdict["factual_future"])
            state["object_counterfactual_futures"].append(verdict["cf_futures"])
            state["factual_actions"].append(verdict["factual_plan"])
            state["cf_actions"].append(verdict["cf_actions"])
            state["cf_durations"].append(verdict["cf_durations"])
            state["inference_mask"].append(verdict["inference_mask"])
            state["episode_seed"].append(seed)
            state["t_star"].append(t_star)
            state["n_mask_cells"].append(int(verdict["n_mask_cells"]))
            state["prefix_max_abs_diff"] = max(state["prefix_max_abs_diff"],
                                               float(verdict["prefix_max_abs_diff"]))
            accepted += 1
            write_ledger({
                "attempt_index": int(attempt_index), "episode_seed": seed,
                "decision": "accepted", "reason": "", "t_star": t_star,
                "n_mask_cells": int(verdict["n_mask_cells"]),
            })
            reporter.update(accepted, attempted=attempts)
            reporter.heartbeat(accepted, attempted=attempts)

            if stop_after_accepted is not None and accepted >= stop_after_accepted:
                checkpoint()
                ledger_handle.close()
                return _receipt(checked, role, state, quota, schedule,
                                data_output, ledger_path, receipt_path,
                                data_sha256="", partial=True)
            if checkpoint_every > 0 and accepted % checkpoint_every == 0:
                checkpoint()
    finally:
        if not ledger_handle.closed:
            ledger_handle.close()

    if accepted < quota:
        receipt = _receipt(checked, role, state, quota, schedule,
                           data_output, ledger_path, receipt_path,
                           data_sha256="", partial=True)
        raise CollectionIncompleteError(json.dumps(
            {**receipt.to_dict(), "status": STATUS_COLLECTION_INCOMPLETE},
            indent=2, ensure_ascii=False))

    checkpoint()
    data_sha256 = _publish(checked, role, state, data_output)
    partial_path.unlink(missing_ok=True)
    receipt = _receipt(checked, role, state, quota, schedule,
                       data_output, ledger_path, receipt_path,
                       data_sha256=data_sha256, partial=False)
    write_once_json(receipt_path, receipt.to_dict())
    return receipt


# --- publish + receipt ------------------------------------------------------------------------


def _build_payload(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    accepted = len(state["object_history"])
    payload: dict[str, np.ndarray] = {}
    for field in _GROUP_ARRAY_FIELDS:
        if field == "inference_mask":
            payload[field] = np.stack(state[field]).astype(bool)
        else:
            payload[field] = np.stack(state[field])
    for field in _GROUP_SCALAR_INT:
        payload[field] = np.asarray(state[field], dtype=np.int64)
    payload["object_capacity"] = np.asarray(OBJECT_CAPACITY_CONST, dtype=np.int64)
    payload["object_features_json"] = np.asarray(
        json.dumps(["presence", "is_ego", "x", "y", "vx", "vy",
                    "cos_heading", "sin_heading"]), dtype=object)
    payload["action_plans_json"] = np.asarray(
        json.dumps([list(p) for p in s4cr2.ACTION_PLANS]), dtype=object)
    payload["n_actions"] = np.asarray(N_ACTIONS, dtype=np.int64)
    payload["accepted_groups"] = np.asarray(accepted, dtype=np.int64)
    return payload


def _publish(config, role, state, data_output) -> str:
    """Atomically write the final NPZ (tmp -> replace) after full-contract assertions."""
    accepted = len(state["object_history"])
    payload = _build_payload(state)
    tmp_npz = data_output.with_name(data_output.name + ".tmp")
    with open(tmp_npz, "wb") as handle:
        np.savez_compressed(handle, **payload)
    capacity = OBJECT_CAPACITY_CONST
    with np.load(tmp_npz, allow_pickle=True) as z:
        g = accepted
        assert z["object_history"].shape == (g, HISTORY_LEN, capacity, N_OBJECT_FEATURES)
        assert z["object_factual_future"].shape == (g, FUTURE_LEN, capacity, N_OBJECT_FEATURES)
        assert z["object_counterfactual_futures"].shape == (
            g, N_ACTIONS, FUTURE_LEN, capacity, N_OBJECT_FEATURES)
        assert z["cf_actions"].shape == (g, N_ACTIONS)
        assert z["cf_durations"].shape == (g, N_ACTIONS)
        assert len(np.unique(z["episode_seed"])) == g
        for field in ("object_history", "object_factual_future",
                      "object_counterfactual_futures"):
            if not np.isfinite(z[field]).all():
                raise RuntimeError(f"final NPZ field {field} is non-finite")
        # ten-action order matches the frozen table for every group.
        for row in np.asarray(z["cf_actions"]):
            assert [int(a) for a in row] == [a for a, _ in s4cr2.ACTION_PLANS]
    digest = sha256_file(tmp_npz)
    Path(tmp_npz).replace(data_output)
    return digest


def _receipt(config, role, state, quota, schedule, data_output, ledger_path,
             receipt_path, *, data_sha256: str, partial: bool) -> CollectionReceipt:
    accepted = len(state["object_history"])
    status = STATUS_SMOKE if role == "smoke" else STATUS_COLLECTED
    return CollectionReceipt(
        plan_id=str(config["plan_id"]),
        role=role,
        status=status,
        partial=bool(partial),
        requested_groups=int(quota),
        accepted_groups=int(accepted),
        counterfactual_branches=int(accepted) * N_ACTIONS,
        attempts=int(state["attempts"]),
        seed_schedule_sha256=hashlib.sha256(np.asarray(schedule).tobytes()).hexdigest(),
        data_output=str(data_output),
        data_output_sha256=str(data_sha256),
        rejection_ledger=str(ledger_path),
        receipt_path=str(receipt_path),
        skip_counts={str(k): int(v) for k, v in state["skip_counts"].items()},
        prefix_max_abs_diff=float(state["prefix_max_abs_diff"]),
        environment=sp._environment_versions(),
        config_signature=canonical_fingerprint(json.loads(json.dumps(config))),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Stage 4C-R2 multi-action groups")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=ROLES)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        receipt = collect_role(config, args.role, ROLE_QUOTA[args.role])
    except CollectionIncompleteError as exc:
        print(f"[stage4cr2-collect] {STATUS_COLLECTION_INCOMPLETE}: {exc}", flush=True)
        return 4
    print(
        f"[stage4cr2-collect] role={receipt.role} status={receipt.status} "
        f"accepted={receipt.accepted_groups} attempts={receipt.attempts} "
        f"partial={receipt.partial}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
