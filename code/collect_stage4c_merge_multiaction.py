"""Stage 4C Merge multi-action decision-group collector.

Plan ``AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1`` section 4. Unlike
Stage 4A/4B (one counterfactual per episode), the Stage 4C statistical unit is a
*decision group*: one fresh ``episode_seed`` -> one history, one factual branch, one
``t_star`` -> TEN counterfactual branches forked from the SAME pre-intervention state
(the frozen ``ACTION_PLANS``, 5 actions x 2 durations). The factual branch is rolled
out exactly once per group; all eleven branches share the same ``env.reset(seed)`` and
``prefix_actions`` so the intervention-time initial state is identical.

Object slot == stable entity id (``extract_object_frame`` indexes the frame by entity
id; slot 0 is the ego). The presence channel therefore *is* the per-frame stable-ID
present set, which is what the cross-branch consistency assertion compares.

Accept/reject (plan section 4.3) uses ONLY pre-intervention-knowable structure and
simulation validity — never a model error, risk score, regret, action ranking or any
consequence value. A single failed branch rejects the WHOLE group with the offending
branch/reason recorded. Rollout-injectable so tests need no highway_env.
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
from typing import Any, Callable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage2_holdout as engine  # noqa: E402
from ac_twin import stage2_protocol as sp  # noqa: E402
from ac_twin import stage4c_protocol as s4cp  # noqa: E402
from ac_twin.checkpointing import (  # noqa: E402
    canonical_fingerprint,
    load_collection_checkpoint,
    save_collection_checkpoint,
)
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

HISTORY_LEN = s4cp.HISTORY_LEN
FUTURE_LEN = s4cp.HORIZON
N_OBJECT_FEATURES = s4cp.N_OBJECT_FEATURES
N_ACTIONS = s4cp.N_ACTIONS

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
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


# --- single-group collection ---------------------------------------------------------


def collect_group(
    seed: int,
    t_star: int,
    prefix_actions: np.ndarray,
    factual_plan: np.ndarray,
    config: Mapping[str, Any],
    branch_rollout: Callable,
) -> dict[str, Any]:
    """Roll out one factual + ten counterfactual branches from the same initial state.

    Returns a raw (unvalidated) group dict; :func:`validate_complete_group` decides
    acceptance. ``branch_rollout`` matches ``engine.rollout_branch_real``.
    """
    capacity = int(config["object_capacity"])
    env_cfg = engine._env_config(config)
    factual = branch_rollout(config["env_id"], env_cfg, seed, prefix_actions,
                             factual_plan, capacity)
    cf_branches = []
    for action, duration in s4cp.ACTION_PLANS:
        plan = s4cp.materialize_action_plan(factual_plan, action, duration)
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
    """Apply the plan-section-4.3 complete-group acceptance rules.

    Returns ``{"accepted": bool, "reason": str, "failed_branch": str|None, ...}`` plus,
    on acceptance, the stacked tensors. Acceptance never reads model error, risk,
    regret, action ranking or consequence values.
    """
    history_len = int(config["history_len"])
    horizon = int(config["horizon"])
    t_star = int(group["t_star"])
    factual = group["factual"]
    cf = group["cf"]

    def reject(reason: str, branch: str) -> dict[str, Any]:
        return {"accepted": False, "reason": reason, "failed_branch": branch,
                "episode_seed": int(group["episode_seed"]), "t_star": t_star}

    branches = [("factual", factual)] + [(f"cf_{i}", b) for i, b in enumerate(cf)]
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

    # 3. initial fingerprints identical across all 11 branches.
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

    # 5. inference mask built WITHOUT counterfactual; evaluation mask (per-branch cf
    #    presence) must equal it for an accepted group. build_inference_mask expects a
    #    leading group axis, so add one and squeeze it back off.
    inference_mask = s4cp.build_inference_mask(
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
    cf_actions = np.asarray([a for a, _ in s4cp.ACTION_PLANS], dtype=np.int64)
    cf_durations = np.asarray([d for _, d in s4cp.ACTION_PLANS], dtype=np.int64)
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
        "factual": factual,
        "cf": cf,
    }


# --- grouped collection state -----------------------------------------------------------

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


# --- role collection --------------------------------------------------------------------


def collect_role(
    role: str,
    quota: int,
    schedule: np.ndarray,
    config: Mapping[str, Any],
    *,
    branch_rollout: Callable | None = None,
    probe_fn: Callable | None = None,
    stop_after_accepted: int | None = None,
) -> dict[str, Any]:
    """Collect ``quota`` complete groups for one role from ``schedule`` (a seed slice).

    Roles (``development_selection`` / ``development_screen`` / ``confirm``) are bound
    to disjoint seed segments before collection; this function consumes the given slice
    in frozen order and never reshuffles by outcome. The final NPZ is the completion
    marker; resume continues a ``.partial.npz`` without re-accepting a group.
    ``stop_after_accepted`` is a test hook.
    """
    checked = s4cp.validate_config(config)
    role = str(role)
    if role not in ("development_selection", "development_screen", "confirm"):
        raise ValueError(f"unknown Stage 4C role: {role}")
    quota = int(quota)
    schedule = np.asarray(schedule, dtype=np.int64)

    branch_rollout = branch_rollout or engine.rollout_branch_real
    probe_fn = probe_fn or engine._real_probe
    action_ids, action_probs, capacity = probe_fn(config)
    action_ids = np.asarray(action_ids, dtype=np.int64)
    action_probs = np.asarray(action_probs, dtype=np.float64)

    horizon = int(checked["horizon"])
    history_len = int(checked["history_len"])

    run_dir = _resolve(checked["run_dir"])
    data_output = _resolve(_role_path(checked["data_output"], role))
    ledger_path = _resolve(_role_path(checked["rejection_ledger"], role))
    receipt_path = run_dir / "development" / f"collection_receipt_{role}.json" if role != "confirm" \
        else run_dir / "confirm" / "collection_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = data_output.with_suffix(".partial.npz")

    _, action_rng, _ = sp._spawn_streams(int(checked["collection_seed"]))
    state = _empty_state()
    resuming = False
    if data_output.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["verification"] = "existing_final_dataset_verified_no_overwrite"
        return receipt
    if partial_path.exists():
        rng_state, metadata, arrays = load_collection_checkpoint(partial_path, dict(config))
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
        total=quota, every=checkpoint_every, prefix=f"stage4c-collect-{role}",
        heartbeat_seconds=30.0)

    def write_ledger(record: dict[str, Any]) -> None:
        ledger_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        ledger_handle.flush()

    def checkpoint() -> None:
        metadata, arrays = _pack_state(state)
        save_collection_checkpoint(
            partial_path, dict(config), {"action_rng": action_rng.bit_generator.state},
            metadata, arrays)

    try:
        while accepted < quota and attempts < schedule.size:
            seed = int(schedule[attempts])
            attempt_index = attempts
            attempts += 1
            state["attempts"] = attempts

            t_star = int(action_rng.integers(int(checked["t_star_low"]),
                                             int(checked["t_star_high"]) + 1))
            prefix_actions = action_rng.choice(action_ids, size=t_star, p=action_probs).astype(np.int64)
            factual_plan = action_rng.choice(action_ids, size=horizon, p=action_probs).astype(np.int64)

            group = collect_group(seed, t_star, prefix_actions, factual_plan, checked,
                                  branch_rollout)
            verdict = validate_complete_group(group, checked)
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
                return _receipt(checked, role, state, quota, schedule, partial=True)
            if checkpoint_every > 0 and accepted % checkpoint_every == 0:
                checkpoint()
    finally:
        if not ledger_handle.closed:
            ledger_handle.close()

    if accepted < quota:
        receipt = _receipt(checked, role, state, quota, schedule, partial=True)
        receipt["status"] = s4cp.STATUS_COLLECTION_INCOMPLETE
        raise CollectionIncompleteError(json.dumps(receipt, indent=2, ensure_ascii=False))

    checkpoint()
    _publish(checked, role, state, quota, schedule, data_output, receipt_path)
    partial_path.unlink(missing_ok=True)
    return _receipt(checked, role, state, quota, schedule, partial=False)


class CollectionIncompleteError(RuntimeError):
    """The role's seed segment exhausted before its group quota filled (hard stop)."""


def _role_path(template: str | Path, role: str) -> str:
    """Map a config path template to a per-role path.

    The two development roles (selection / screen) are disjoint group sets, so each
    writes its OWN NPZ/ledger (a ``_<role>`` suffix) — they never share a file. The
    confirm role swaps the dev role token for the confirm role token.
    """
    text = str(template)
    if role == "confirm":
        return text.replace(s4cp.PURPOSE_DEV, s4cp.PURPOSE_CONFIRM)
    if role in ("development_selection", "development_screen"):
        suffix = "_selection" if role == "development_selection" else "_screen"
        if text.endswith(".npz"):
            return text[:-4] + suffix + ".npz"
        if text.endswith(".jsonl"):
            return text[:-6] + suffix + ".jsonl"
        return text + suffix
    return text


# --- publish + receipt ---------------------------------------------------------------------


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
        json.dumps([list(p) for p in s4cp.ACTION_PLANS]), dtype=object)
    payload["n_actions"] = np.asarray(N_ACTIONS, dtype=np.int64)
    payload["accepted_groups"] = np.asarray(accepted, dtype=np.int64)
    return payload


OBJECT_CAPACITY_CONST = s4cp.OBJECT_CAPACITY


def _publish(config, role, state, quota, schedule, data_output, receipt_path) -> None:
    accepted = len(state["object_history"])
    payload = _build_payload(state)
    receipt = _receipt(config, role, state, quota, schedule, partial=False)
    receipt["data_output_sha256_pending"] = True

    tmp_npz = data_output.with_name(data_output.name + ".tmp")
    tmp_receipt = receipt_path.with_name(receipt_path.name + ".tmp")
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
            assert [int(a) for a in row] == [a for a, _ in s4cp.ACTION_PLANS]
    receipt["data_output_sha256"] = sha256_file(tmp_npz)
    receipt.pop("data_output_sha256_pending", None)
    tmp_receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(tmp_receipt).replace(receipt_path)
    Path(tmp_npz).replace(data_output)


def _receipt(config, role, state, quota, schedule, *, partial) -> dict[str, Any]:
    accepted = len(state["object_history"])
    return {
        "experiment_name": config["experiment_name"],
        "plan_id": config["plan_id"],
        "purpose": s4cp.PURPOSE_CONFIRM if role == "confirm" else s4cp.PURPOSE_DEV,
        "role": role,
        "partial": bool(partial),
        "requested_groups": int(quota),
        "accepted_groups": int(accepted),
        "counterfactual_branches": int(accepted) * N_ACTIONS,
        "attempts": int(state["attempts"]),
        "collection_seed": int(config["collection_seed"]),
        "seed_schedule_sha256": hashlib.sha256(np.asarray(schedule).tobytes()).hexdigest(),
        "skip_counts": {str(k): int(v) for k, v in state["skip_counts"].items()},
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "environment": sp._environment_versions(),
        "config_signature": canonical_fingerprint(json.loads(json.dumps(config))),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Stage 4C multi-action groups")
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True,
                        choices=("development_selection", "development_screen", "confirm"))
    parser.add_argument("--quota", type=int, required=True)
    parser.add_argument("--schedule", required=True, help="path to .npy seed slice")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    schedule = np.load(args.schedule).astype(np.int64)
    try:
        receipt = collect_role(args.role, args.quota, schedule, config)
    except CollectionIncompleteError as exc:
        print(f"[stage4c-collect] {s4cp.STATUS_COLLECTION_INCOMPLETE}: {exc}", flush=True)
        return 4
    print(
        f"[stage4c-collect] role={receipt.get('role')} "
        f"accepted={receipt.get('accepted_groups')} attempts={receipt.get('attempts')} "
        f"partial={receipt.get('partial', False)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
