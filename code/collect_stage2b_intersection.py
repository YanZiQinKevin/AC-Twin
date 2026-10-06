"""Stage 2B fixed-participant intersection collector.

Plan: AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-14-V0.3.

Collects paired factual/counterfactual object-centric rollouts of
``intersection-v0`` with dynamic spawning disabled (``spawn_probability=0.0``),
a frozen object capacity of 20 and episode-local stable entity IDs. Three
actions x two durations form six mechanically balanced strata with exact
per-stratum quotas; factual action plans are uniform over the three actions and
the counterfactual action always differs from the factual first action.

Acceptance uses only the preregistered complete-horizon / consistency /
identity / capacity rules; every rejection is written to the ledger. The
collector never reads realized R, response bins, collisions-to-come, future
error or any Gate quantity to accept an episode. (``first_contact_step`` and
termination flags are recorded as diagnostics only.)

Collection requires a verified create-once collection manifest (see
``ac_twin.stage2b_protocol.prepare_collection``). The final NPZ is the
completion marker: a temporary NPZ + receipt are validated first, then
published atomically.

The collector is rollout-injectable so unit tests run without importing
highway_env (which pulls in pygame and fails under pytest's ``-W error``).
"""
from __future__ import annotations

import os
# Set before the first numpy/torch import: avoids the torch/MKL duplicate-OpenMP
# abort (OMP Error #15). Deterministic; does not change numerical results.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage2b_protocol as sp  # noqa: E402
from ac_twin import stage2b_data as sd  # noqa: E402
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
from collect_pilot import action_mapping, make_env  # noqa: E402

N_OBJECT_FEATURES = 8

# Per-episode stacked array fields (leading axis = accepted episode).
_STACKED_FIELDS = (
    "object_history", "object_factual_future", "object_counterfactual_future",
    "factual_actions", "counterfactual_actions",
    "factual_lane_index", "factual_target_lane_index", "factual_cmd_accel",
    "factual_cmd_steer", "factual_crashed",
    "counterfactual_lane_index", "counterfactual_target_lane_index",
    "counterfactual_cmd_accel", "counterfactual_cmd_steer", "counterfactual_crashed",
)
_SCALAR_INT_FIELDS = ("cf_action", "cf_duration", "episode_seed", "t_star",
                      "factual_first_contact_step", "counterfactual_first_contact_step",
                      "initial_identity_count", "max_road_vehicles")
_SCALAR_BOOL_FIELDS = ("factual_terminated", "factual_truncated",
                       "counterfactual_terminated", "counterfactual_truncated",
                       "initial_fingerprint_consistent")
_JSON_FIELDS = ("factual_fingerprints_json", "counterfactual_fingerprints_json",
                "vehicle_metadata_json", "factual_termination_reason",
                "counterfactual_termination_reason")

# Acceptance rejection reasons (preregistered complete-horizon / consistency /
# identity / capacity rules only). ``no_eligible_stratum`` is a mechanical
# balancing skip decided before any rollout, never an outcome-based decision.
_REASONS = (
    "prefix_terminated", "object_prefix_mismatch",
    "factual_future_short", "counterfactual_future_short", "non_finite",
    "short_history", "initial_fingerprint_mismatch",
    "capacity_exceeded", "unregistered_identity", "identity_inconsistency",
    "no_eligible_stratum",
)


class CapacityExceededError(RuntimeError):
    """A rollout produced more registered vehicles than the frozen capacity."""


class UnregisteredIdentityError(RuntimeError):
    """A vehicle without a registered stable ID appeared during a branch."""


class IdentityInconsistencyError(RuntimeError):
    """Present stable IDs do not correspond one-to-one with road vehicles."""


# --- environment configuration + P0 assertions ---------------------------------------


def _env_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The intersection-v0 env config. Note: ``vehicles_count`` is NOT the
    population control here (that merge-era key is never read); the population
    knob is ``initial_vehicle_count`` and dynamic spawn is disabled."""
    obs_cfg = {
        "type": "Kinematics",
        "vehicles_count": int(config["observed_vehicles_count"]),
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "clip": False,
        "order": "sorted",
    }
    return {
        "duration": int(config["duration"]),
        "simulation_frequency": int(config["simulation_frequency"]),
        "policy_frequency": int(config["policy_frequency"]),
        "controlled_vehicles": int(config["controlled_vehicles"]),
        "initial_vehicle_count": int(config["initial_vehicle_count"]),
        "spawn_probability": float(config["spawn_probability"]),
        "observation": obs_cfg,
    }


def assert_env_config(env: Any, config: Mapping[str, Any]) -> None:
    """P0: read back the live env config and assert every frozen key took effect."""
    base = getattr(env, "unwrapped", env)
    uc = base.config
    checks = {
        "policy_frequency": int(config["policy_frequency"]),
        "simulation_frequency": int(config["simulation_frequency"]),
        "duration": int(config["duration"]),
        "controlled_vehicles": int(config["controlled_vehicles"]),
        "initial_vehicle_count": int(config["initial_vehicle_count"]),
    }
    mismatches = []
    for key, expected in checks.items():
        if int(uc.get(key, -1)) != expected:
            mismatches.append(f"{key}: live={uc.get(key)!r} expected={expected!r}")
    if float(uc.get("spawn_probability", float("nan"))) != float(config["spawn_probability"]):
        mismatches.append(
            f"spawn_probability: live={uc.get('spawn_probability')!r} "
            f"expected={config['spawn_probability']!r}")
    obs = uc.get("observation", {})
    if int(obs.get("vehicles_count", -1)) != int(config["observed_vehicles_count"]):
        mismatches.append(
            f"observation.vehicles_count: live={obs.get('vehicles_count')!r} "
            f"expected={config['observed_vehicles_count']!r}")
    if mismatches:
        raise RuntimeError(
            "environment config did not take effect: " + "; ".join(mismatches))


def _real_probe(config: Mapping[str, Any]):
    """Probe the real env: version, frozen action semantics, config read-back,
    capacity. Returns uniform action probabilities (the factual plan samples
    the three actions with equal probability, per plan section 3.1)."""
    from importlib.metadata import version as _pkg_version
    installed = _pkg_version("highway_env")
    if installed != sp.HIGHWAY_ENV_VERSION:
        raise RuntimeError(
            f"highway_env version must be {sp.HIGHWAY_ENV_VERSION}, found {installed}")
    env = make_env(config["env_id"], _env_config(config))
    try:
        env.reset(seed=int(config["collection_seed"]))
        mapping = action_mapping(env)
        if mapping != {name: idx for name, idx in sp.ACTION_SEMANTICS.items()}:
            raise RuntimeError(
                f"action mapping must be exactly {sp.ACTION_SEMANTICS}, got {mapping}")
        assert_env_config(env, config)
        initial_count = len(env.unwrapped.road.vehicles)
        if initial_count > sp.OBJECT_CAPACITY:
            raise RuntimeError(
                f"initial vehicles {initial_count} exceed the frozen capacity "
                f"{sp.OBJECT_CAPACITY}; profile would silently truncate")
    finally:
        env.close()
    action_ids = np.asarray(sp.ACTION_IDS, dtype=np.int64)
    action_probs = np.full(len(sp.ACTION_IDS), 1.0 / len(sp.ACTION_IDS), dtype=np.float64)
    return action_ids, action_probs, sp.OBJECT_CAPACITY


# --- real branch rollout with identity / capacity guards ------------------------------


def _checked_frame(env: Any, capacity: int) -> np.ndarray:
    """One object frame with the paired-validity guards of plan section 3.2."""
    base = getattr(env, "unwrapped", env)
    n_road = len(base.road.vehicles) if base.road is not None else 0
    if n_road > capacity:
        raise CapacityExceededError(
            f"{n_road} road vehicles exceed the frozen capacity {capacity}")
    try:
        frame, _ = extract_object_frame(env, capacity)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "unregistered" in message:
            raise UnregisteredIdentityError(str(exc)) from exc
        if "capacity" in message:
            raise CapacityExceededError(str(exc)) from exc
        raise
    if int((frame[:, 0] > 0).sum()) != n_road:
        raise IdentityInconsistencyError(
            f"present stable IDs ({int((frame[:, 0] > 0).sum())}) do not correspond "
            f"one-to-one with road vehicles ({n_road})")
    return frame


def _extract_step_diag(env: Any, capacity: int) -> dict[str, np.ndarray]:
    """Per-vehicle diagnostic values for one frame; missing values are masked."""
    base = getattr(env, "unwrapped", env)
    lane = np.full(capacity, -1, dtype=np.int32)
    target_lane = np.full(capacity, -1, dtype=np.int32)
    cmd_accel = np.full(capacity, np.nan, dtype=np.float32)
    cmd_steer = np.full(capacity, np.nan, dtype=np.float32)
    crashed = np.zeros(capacity, dtype=bool)
    current = list(base.road.vehicles) if base.road is not None else []
    for vehicle in current:
        eid = int(getattr(vehicle, "_ac_twin_entity_id"))
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is not None and len(lane_index) >= 3:
            lane[eid] = int(lane_index[2])
        target = getattr(vehicle, "target_lane_index", None)
        if target is not None and len(target) >= 3:
            target_lane[eid] = int(target[2])
        action = getattr(vehicle, "action", None)
        if isinstance(action, Mapping):
            if action.get("acceleration") is not None:
                cmd_accel[eid] = float(action["acceleration"])
            if action.get("steering") is not None:
                cmd_steer[eid] = float(action["steering"])
        crashed[eid] = bool(getattr(vehicle, "crashed", False))
    return {"lane_index": lane, "target_lane_index": target_lane,
            "commanded_accel": cmd_accel, "commanded_steering": cmd_steer,
            "crashed": crashed}


def _vehicle_static_metadata(env: Any) -> list[dict[str, Any]]:
    """Best-effort per-vehicle route + behavior parameters at reset."""
    base = getattr(env, "unwrapped", env)
    behavior_attrs = ("ACC_MAX", "COMFORT_ACC_MAX", "COMFORT_ACC_MIN", "T",
                      "DESIRED_SPEED", "a", "b", "delta", "s0", "v0", "tau")
    records: list[dict[str, Any]] = []
    current = list(base.road.vehicles) if base.road is not None else []
    for vehicle in current:
        eid = int(getattr(vehicle, "_ac_twin_entity_id"))
        route = getattr(vehicle, "route", None)
        params: dict[str, float] = {}
        for name in behavior_attrs:
            if hasattr(vehicle, name):
                value = getattr(vehicle, name)
                if np.isscalar(value) and np.isfinite(value):
                    params[name] = float(value)
        records.append({
            "entity_id": eid,
            "class": type(vehicle).__name__,
            "route": ([list(map(str, item)) for item in route]
                      if route is not None else None),
            "behavior_params": params,
        })
    return records


def rollout_branch_real(env_id: str, env_cfg: Mapping[str, Any], seed: int,
                        prefix_actions: np.ndarray, future_actions: np.ndarray,
                        capacity: int) -> dict[str, Any]:
    """Roll out one branch of the object-centric twin world with diagnostics."""
    env = make_env(env_id, dict(env_cfg))
    try:
        env.reset(seed=int(seed))
        fingerprints = assign_episode_entity_ids(env)
        if len(fingerprints) > capacity:
            raise CapacityExceededError(
                f"{len(fingerprints)} initial identities exceed the frozen capacity {capacity}")
        static_metadata = _vehicle_static_metadata(env)

        prefix_frames = [_checked_frame(env, capacity)]
        prefix_diag = [_extract_step_diag(env, capacity)]
        max_road = len(getattr(env, "unwrapped", env).road.vehicles)
        terminated = truncated = False
        reason = ""
        for action in prefix_actions:
            _, _, term, trunc, _ = env.step(int(action))
            prefix_frames.append(_checked_frame(env, capacity))
            prefix_diag.append(_extract_step_diag(env, capacity))
            max_road = max(max_road, len(env.unwrapped.road.vehicles))
            if term or trunc:
                terminated, truncated = bool(term), bool(trunc)
                reason = "terminated" if term else "truncated"
                break

        future_frames: list[np.ndarray] = []
        future_diag: list[dict[str, np.ndarray]] = []
        first_contact = -1
        if not (terminated or truncated):
            for step, action in enumerate(future_actions):
                _, _, term, trunc, _ = env.step(int(action))
                frame = _checked_frame(env, capacity)
                diag = _extract_step_diag(env, capacity)
                max_road = max(max_road, len(env.unwrapped.road.vehicles))
                future_frames.append(frame)
                future_diag.append(diag)
                if first_contact < 0 and bool(diag["crashed"].any()):
                    first_contact = step
                    reason = reason or "collision"
                if term or trunc:
                    terminated, truncated = bool(term), bool(trunc)
                    reason = reason or ("terminated" if term else "truncated")
                    break
    finally:
        env.close()

    def stack_diag(diags):
        if not diags:
            return {k: np.empty((0, capacity), dtype=np.float32)
                    for k in ("lane_index", "target_lane_index",
                              "commanded_accel", "commanded_steering", "crashed")}
        return {k: np.stack([d[k] for d in diags]) for k in diags[0]}

    return {
        "prefix_frames": np.asarray(prefix_frames, dtype=np.float32),
        "future_frames": np.asarray(future_frames, dtype=np.float32),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": reason,
        "first_contact_step": int(first_contact),
        "diag_prefix": stack_diag(prefix_diag),
        "diag_future": stack_diag(future_diag),
        "fingerprints": fingerprints,
        "static_metadata": static_metadata,
        "initial_identity_count": int(len(fingerprints)),
        "max_road_vehicles": int(max_road),
    }


# --- collection state ------------------------------------------------------------------


def _empty_state() -> dict[str, Any]:
    state: dict[str, Any] = {field: [] for field in _STACKED_FIELDS}
    state.update({field: [] for field in _SCALAR_INT_FIELDS})
    state.update({field: [] for field in _SCALAR_BOOL_FIELDS})
    state.update({field: [] for field in _JSON_FIELDS})
    state["split"] = []
    state["accepted_strata"] = Counter()
    state["skip_counts"] = Counter({reason: 0 for reason in _REASONS})
    state["attempts"] = 0
    state["prefix_max_abs_diff"] = 0.0
    state["identity_max_abs_diff"] = 0.0
    state["max_road_vehicles_seen"] = 0
    state["initial_identity_count_max"] = 0
    return state


def _pack_state(state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    accepted = len(state["object_history"])
    arrays: dict[str, np.ndarray] = {}
    for field in _STACKED_FIELDS:
        values = state[field]
        if len(values) != accepted:
            raise ValueError(f"field {field} has inconsistent accepted count")
        arrays[field] = np.stack(values) if values else np.empty((0,), dtype=np.float32)
    for field in _SCALAR_INT_FIELDS:
        arrays[field] = np.asarray(state[field], dtype=np.int64)
    for field in _SCALAR_BOOL_FIELDS:
        arrays[field] = np.asarray(state[field], dtype=bool)
    arrays["split"] = np.asarray(state["split"], dtype=np.int8)
    metadata = {
        "accepted": accepted,
        "attempts": int(state["attempts"]),
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "identity_max_abs_diff": float(state["identity_max_abs_diff"]),
        "max_road_vehicles_seen": int(state["max_road_vehicles_seen"]),
        "initial_identity_count_max": int(state["initial_identity_count_max"]),
        "accepted_strata": [[int(a), int(d), int(c)]
                            for (a, d), c in sorted(state["accepted_strata"].items())],
        "skip_counts": {str(k): int(v) for k, v in state["skip_counts"].items()},
        "json_fields": {field: list(state[field]) for field in _JSON_FIELDS},
    }
    return metadata, arrays


def _restore_state(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> dict[str, Any]:
    accepted = int(metadata["accepted"])
    state = _empty_state()
    for field in _STACKED_FIELDS:
        values = arrays.get(field)
        if values is None or len(values) != accepted:
            raise ValueError(f"checkpoint field {field} inconsistent")
        state[field] = [v.copy() for v in values]
    for field in _SCALAR_INT_FIELDS:
        state[field] = [int(v) for v in arrays[field]]
    for field in _SCALAR_BOOL_FIELDS:
        state[field] = [bool(v) for v in arrays[field]]
    state["split"] = [int(v) for v in arrays["split"]]
    json_fields = metadata.get("json_fields", {})
    for field in _JSON_FIELDS:
        state[field] = list(json_fields.get(field, []))
        if len(state[field]) != accepted:
            raise ValueError(f"checkpoint json field {field} inconsistent")
    state["accepted_strata"] = Counter({(int(a), int(d)): int(c)
                                        for a, d, c in metadata.get("accepted_strata", [])})
    state["skip_counts"].update({str(k): int(v) for k, v in metadata.get("skip_counts", {}).items()})
    state["attempts"] = int(metadata["attempts"])
    state["prefix_max_abs_diff"] = float(metadata["prefix_max_abs_diff"])
    state["identity_max_abs_diff"] = float(metadata["identity_max_abs_diff"])
    state["max_road_vehicles_seen"] = int(metadata.get("max_road_vehicles_seen", 0))
    state["initial_identity_count_max"] = int(metadata.get("initial_identity_count_max", 0))
    return state


# --- git / version metadata -------------------------------------------------------------


def _vcs_status() -> dict[str, Any]:
    """Record git commit if inside a repository; otherwise an explicit null."""
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                       cwd=PROJECT_ROOT, capture_output=True, check=True, timeout=10)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, check=True, timeout=10,
                                text=True).stdout.strip()
        return {"git_commit": commit, "vcs_status": "git_repository"}
    except Exception:
        return {"git_commit": None, "vcs_status": "no_git_repository"}


# --- acceptance --------------------------------------------------------------------------


def _fingerprints_consistent(left: list[dict], right: list[dict], atol: float = 1e-6) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        for key in ("entity_id", "class", "route"):
            if a.get(key) != b.get(key):
                return False
        for key in ("position_xy", "heading", "speed"):
            if not np.allclose(a.get(key), b.get(key), atol=atol, rtol=0.0):
                return False
    return True


def _finite_difference_accel(future: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference acceleration (E-free) from a (T,C,8) future branch."""
    vx = future[..., 4]
    vy = future[..., 5]
    presence = future[..., 0] > 0
    ax = np.gradient(vx, dt, axis=0, edge_order=1)
    ay = np.gradient(vy, dt, axis=0, edge_order=1)
    accel = np.stack([ax, ay], axis=-1).astype(np.float32)
    accel = np.where(presence[..., None], accel, np.nan)
    return accel


# --- main entry -------------------------------------------------------------------------


def collect_stage2b(config: Mapping[str, Any], *, resume: bool = False,
                    branch_rollout: Callable | None = None,
                    probe_fn: Callable | None = None,
                    config_path: str | Path | None = None,
                    stop_after_accepted: int | None = None) -> dict[str, Any]:
    """Collect one Stage 2B role dataset; returns the collection receipt mapping.

    Requires a verified collection manifest in ``config['output_dir']``.
    ``stop_after_accepted`` is a test hook that checkpoints and returns early so
    resume equivalence can be exercised.
    """
    sp.validate_stage2b_config(config)
    output_dir = sp._resolve(config["output_dir"])
    manifest = sp.verify_collection_manifest(output_dir, config)
    if config_path is not None and manifest.get("config_sha256") is not None:
        if sp.sha256_file(config_path) != manifest["config_sha256"]:
            raise RuntimeError("config file hash does not match the manifest")

    schedule = np.load(output_dir / "seed_schedule.npy").astype(np.int64)
    if schedule.shape[0] < int(config["max_attempts"]):
        raise RuntimeError("seed schedule smaller than max_attempts")

    branch_rollout = branch_rollout or rollout_branch_real
    probe_fn = probe_fn or _real_probe
    action_ids, action_probs, capacity = probe_fn(config)
    action_ids = np.asarray(action_ids, dtype=np.int64)
    action_probs = np.asarray(action_probs, dtype=np.float64)

    role = str(config["role"])
    n_pairs = int(config["n_pairs"])
    per_stratum = int(config["per_stratum"])
    horizon = int(config["horizon"])
    history_len = int(config["history_len"])
    durations = [int(d) for d in config["intervention_durations"]]
    max_attempts = int(config["max_attempts"])
    determinism_checks = int(config["determinism_checks"])
    dt = 1.0 / float(config["policy_frequency"])
    is_development = role == "development"
    fixed_split_label = sp.SPLIT_LABELS.get(
        {"smoke": "smoke", "pilot_a": "pilot", "pilot_b": "pilot"}.get(role, ""), None)
    if not is_development and fixed_split_label is None:
        raise RuntimeError(f"role {role!r} has no split-label rule yet")

    data_output = sp._resolve(config["data_output"])
    ledger_path = sp._resolve(config["rejection_ledger"])
    receipt_path = sp._resolve(config["collection_receipt"])
    partial_path = data_output.with_suffix(".partial.npz")

    # RNG streams: action and stratum (schedule stream consumed offline at manifest).
    _, action_rng, stratum_rng = sp._spawn_streams(int(config["collection_seed"]))

    state = _empty_state()
    resuming = False
    if data_output.exists():
        # Already complete: report the sealed receipt without overwriting anything.
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["verification"] = "existing_final_dataset_verified_no_overwrite"
        return receipt
    if partial_path.exists():
        rng_state, metadata, arrays = load_collection_checkpoint(partial_path, dict(config))
        state = _restore_state(arrays, metadata)
        action_rng.bit_generator.state = rng_state["action_rng"]
        stratum_rng.bit_generator.state = rng_state["stratum_rng"]
        resuming = True
    # ``resume=True`` with no checkpoint simply starts fresh.

    data_output.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    # Fresh starts truncate the ledger; resumes append to it.
    ledger_handle = open(ledger_path, "a" if resuming else "w", encoding="utf-8")

    all_strata = [(int(a), int(d)) for a in action_ids for d in durations]
    checkpoint_every = int(config.get("checkpoint_every_accepted", 25))
    attempts = int(state["attempts"])
    accepted = len(state["object_history"])
    progress = ProgressReporter(total=n_pairs, every=max(1, n_pairs // 100),
                                prefix=f"collect-{role}", heartbeat_seconds=30.0)

    def write_ledger(record: dict[str, Any]) -> None:
        ledger_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        ledger_handle.flush()

    def checkpoint() -> None:
        metadata, arrays = _pack_state(state)
        save_collection_checkpoint(
            partial_path, dict(config),
            {"action_rng": action_rng.bit_generator.state,
             "stratum_rng": stratum_rng.bit_generator.state},
            metadata, arrays)

    def ledger_base(attempt_index: int, episode_seed: int, decision: str,
                    reason: str, cf_action: int, cf_duration: int, t_star: int,
                    f: Mapping[str, Any] | None, c: Mapping[str, Any] | None
                    ) -> dict[str, Any]:
        def field(branch, key):
            return None if branch is None else branch[key]
        return {
            "attempt_index": int(attempt_index),
            "episode_seed": int(episode_seed),
            "decision": decision,
            "reason": reason,
            "cf_action": None if cf_action is None else int(cf_action),
            "cf_duration": None if cf_duration is None else int(cf_duration),
            "t_star": int(t_star),
            "factual_terminated": field(f, "terminated"),
            "factual_truncated": field(f, "truncated"),
            "counterfactual_terminated": field(c, "terminated"),
            "counterfactual_truncated": field(c, "truncated"),
            "factual_first_contact_step": field(f, "first_contact_step"),
            "counterfactual_first_contact_step": field(c, "first_contact_step"),
            "initial_identity_count": field(f, "initial_identity_count"),
            "max_road_vehicles": None if f is None else int(max(
                f["max_road_vehicles"], c["max_road_vehicles"])),
        }

    try:
        while accepted < n_pairs and attempts < max_attempts:
            episode_seed = int(schedule[attempts])
            attempt_index = attempts
            attempts += 1
            state["attempts"] = attempts

            t_star = int(action_rng.integers(int(config["t_star_low"]),
                                             int(config["t_star_high"]) + 1))
            prefix_actions = action_rng.choice(action_ids, size=t_star,
                                               p=action_probs).astype(np.int64)
            factual_plan = action_rng.choice(action_ids, size=horizon,
                                             p=action_probs).astype(np.int64)

            # Mechanical stratum balancing with exact per-stratum quotas; the
            # counterfactual action always differs from the factual first action.
            eligible = [key for key in all_strata
                        if key[0] != int(factual_plan[0])
                        and state["accepted_strata"][key] < per_stratum]
            if not eligible:
                state["skip_counts"]["no_eligible_stratum"] += 1
                write_ledger(ledger_base(attempt_index, episode_seed, "skipped",
                                         "no_eligible_stratum", -1, -1, t_star,
                                         None, None))
                continue
            min_count = min(state["accepted_strata"][key] for key in eligible)
            least_used = [key for key in eligible if state["accepted_strata"][key] == min_count]
            cf_action, cf_duration = least_used[int(stratum_rng.integers(0, len(least_used)))]
            cf_duration = min(cf_duration, horizon)
            counterfactual_plan = factual_plan.copy()
            counterfactual_plan[:cf_duration] = int(cf_action)

            try:
                f = branch_rollout(config["env_id"], _env_config(config), episode_seed,
                                   prefix_actions, factual_plan, capacity)
                c = branch_rollout(config["env_id"], _env_config(config), episode_seed,
                                   prefix_actions, counterfactual_plan, capacity)
            except CapacityExceededError:
                state["skip_counts"]["capacity_exceeded"] += 1
                write_ledger(ledger_base(attempt_index, episode_seed, "rejected",
                                         "capacity_exceeded", cf_action, cf_duration,
                                         t_star, None, None))
                continue
            except UnregisteredIdentityError:
                state["skip_counts"]["unregistered_identity"] += 1
                write_ledger(ledger_base(attempt_index, episode_seed, "rejected",
                                         "unregistered_identity", cf_action, cf_duration,
                                         t_star, None, None))
                continue
            except IdentityInconsistencyError:
                state["skip_counts"]["identity_inconsistency"] += 1
                write_ledger(ledger_base(attempt_index, episode_seed, "rejected",
                                         "identity_inconsistency", cf_action, cf_duration,
                                         t_star, None, None))
                continue

            state["max_road_vehicles_seen"] = max(
                state["max_road_vehicles_seen"],
                int(f["max_road_vehicles"]), int(c["max_road_vehicles"]))
            state["initial_identity_count_max"] = max(
                state["initial_identity_count_max"],
                int(f["initial_identity_count"]), int(c["initial_identity_count"]))

            # Acceptance rules (preregistered complete-horizon / consistency /
            # identity / capacity rules only).
            reason = None
            if (f["terminated"] or f["truncated"] or c["terminated"] or c["truncated"]
                    or len(f["prefix_frames"]) != t_star + 1
                    or len(c["prefix_frames"]) != t_star + 1):
                reason = "prefix_terminated"
            else:
                prefix_diff = float(np.max(np.abs(
                    f["prefix_frames"] - c["prefix_frames"])))
                state["prefix_max_abs_diff"] = max(state["prefix_max_abs_diff"], prefix_diff)
                if prefix_diff > 1e-6:
                    reason = "object_prefix_mismatch"
            if reason is None and len(f["future_frames"]) != horizon:
                reason = "factual_future_short"
            if reason is None and len(c["future_frames"]) != horizon:
                reason = "counterfactual_future_short"
            if reason is None and not (
                    np.isfinite(f["future_frames"]).all()
                    and np.isfinite(c["future_frames"]).all()):
                reason = "non_finite"
            if reason is None and not _fingerprints_consistent(
                    f["fingerprints"], c["fingerprints"]):
                reason = "initial_fingerprint_mismatch"
            if reason is None and (int(f["initial_identity_count"]) > capacity
                                   or int(c["initial_identity_count"]) > capacity
                                   or int(f["max_road_vehicles"]) > capacity
                                   or int(c["max_road_vehicles"]) > capacity):
                reason = "capacity_exceeded"
            history = None if reason is not None else f["prefix_frames"][-history_len:]
            if reason is None and history.shape[0] != history_len:
                reason = "short_history"

            if reason is not None:
                state["skip_counts"][reason] += 1
                write_ledger(ledger_base(attempt_index, episode_seed, "rejected",
                                         reason, cf_action, cf_duration, t_star, f, c))
                continue

            # Identity-replay determinism check on the first N accepted episodes.
            if accepted < determinism_checks:
                check = branch_rollout(config["env_id"], _env_config(config),
                                       episode_seed, prefix_actions, factual_plan, capacity)
                if (check["prefix_frames"].shape != f["prefix_frames"].shape
                        or check["future_frames"].shape != f["future_frames"].shape):
                    raise RuntimeError("identity replay changed trajectory shape")
                identity_diff = max(
                    float(np.max(np.abs(check["prefix_frames"] - f["prefix_frames"]))),
                    float(np.max(np.abs(check["future_frames"] - f["future_frames"]))),
                )
                state["identity_max_abs_diff"] = max(state["identity_max_abs_diff"],
                                                     identity_diff)
                if identity_diff > 1e-6:
                    raise RuntimeError(
                        f"identity replay failed determinism check: {identity_diff}")

            # Accept the pair: store core arrays + diagnostics.
            state["object_history"].append(history.astype(np.float32))
            state["object_factual_future"].append(f["future_frames"].astype(np.float32))
            state["object_counterfactual_future"].append(c["future_frames"].astype(np.float32))
            state["factual_actions"].append(factual_plan.astype(np.int64))
            state["counterfactual_actions"].append(counterfactual_plan.astype(np.int64))
            state["cf_action"].append(int(cf_action))
            state["cf_duration"].append(int(cf_duration))
            state["episode_seed"].append(int(episode_seed))
            state["t_star"].append(int(t_star))
            state["initial_identity_count"].append(int(f["initial_identity_count"]))
            state["max_road_vehicles"].append(
                int(max(f["max_road_vehicles"], c["max_road_vehicles"])))

            for prefix, branch in (("factual", f), ("counterfactual", c)):
                hist_diag = {k: v[-history_len:] for k, v in branch["diag_prefix"].items()}
                fut_diag = branch["diag_future"]
                lane = np.concatenate([hist_diag["lane_index"], fut_diag["lane_index"]])
                target_lane = np.concatenate(
                    [hist_diag["target_lane_index"], fut_diag["target_lane_index"]])
                cmd_accel = np.concatenate(
                    [hist_diag["commanded_accel"], fut_diag["commanded_accel"]])
                cmd_steer = np.concatenate(
                    [hist_diag["commanded_steering"], fut_diag["commanded_steering"]])
                crashed = np.concatenate([hist_diag["crashed"], fut_diag["crashed"]])
                state[f"{prefix}_lane_index"].append(lane.astype(np.int32))
                state[f"{prefix}_target_lane_index"].append(target_lane.astype(np.int32))
                state[f"{prefix}_cmd_accel"].append(cmd_accel.astype(np.float32))
                state[f"{prefix}_cmd_steer"].append(cmd_steer.astype(np.float32))
                state[f"{prefix}_crashed"].append(crashed.astype(bool))
                state[f"{prefix}_terminated"].append(bool(branch["terminated"]))
                state[f"{prefix}_truncated"].append(bool(branch["truncated"]))
                state[f"{prefix}_first_contact_step"].append(int(branch["first_contact_step"]))
                state[f"{prefix}_termination_reason"].append(str(branch["termination_reason"]))
                state[f"{prefix}_fingerprints_json"].append(
                    json.dumps(branch["fingerprints"], ensure_ascii=False))

            state["initial_fingerprint_consistent"].append(True)
            state["vehicle_metadata_json"].append(
                json.dumps(f["static_metadata"], ensure_ascii=False))
            if is_development:
                ordinal = int(state["accepted_strata"][(int(cf_action), int(cf_duration))])
                state["split"].append(sd.development_split_label(
                    ordinal, train_quota=int(config["train_per_stratum"]),
                    stratum_quota=per_stratum))
            else:
                state["split"].append(int(fixed_split_label))
            state["accepted_strata"][(int(cf_action), int(cf_duration))] += 1
            accepted += 1
            write_ledger(ledger_base(attempt_index, episode_seed, "accepted",
                                     "", cf_action, cf_duration, t_star, f, c))

            line = progress.update(accepted, attempted=attempts)
            if line is None:
                progress.heartbeat(accepted, attempted=attempts)
            if stop_after_accepted is not None and accepted >= stop_after_accepted:
                checkpoint()
                ledger_handle.close()
                return _receipt(config, state, attempts, accepted, partial=True)
            if checkpoint_every > 0 and accepted % checkpoint_every == 0:
                checkpoint()
    finally:
        if not ledger_handle.closed:
            ledger_handle.close()

    if accepted != n_pairs:
        raise RuntimeError(
            f"collected {accepted} valid pairs after {attempts} attempts; expected {n_pairs}")

    checkpoint()
    _publish(config, state, schedule, manifest, action_ids, capacity, dt)
    partial_path.unlink(missing_ok=True)
    progress.update(accepted, attempted=attempts, force=True)
    return _receipt(config, state, attempts, accepted, partial=False)


# --- publish + receipt --------------------------------------------------------------------


def _derive_accel_fields(state: Mapping[str, Any], dt: float) -> dict[str, np.ndarray]:
    out = {}
    for prefix, key in (("factual", "object_factual_future"),
                        ("counterfactual", "object_counterfactual_future")):
        futures = np.stack(state[key])  # E,20,C,8
        accels = [_finite_difference_accel(futures[e], dt) for e in range(futures.shape[0])]
        out[f"{prefix}_accel_fd"] = np.stack(accels).astype(np.float32)
    return out


def _build_payload(config: Mapping[str, Any], state: Mapping[str, Any],
                   capacity: int, dt: float) -> dict[str, np.ndarray]:
    accepted = len(state["object_history"])
    payload: dict[str, np.ndarray] = {}
    for field in _STACKED_FIELDS:
        payload[field] = np.stack(state[field])
    for field in _SCALAR_INT_FIELDS:
        payload[field] = np.asarray(state[field], dtype=np.int64)
    for field in _SCALAR_BOOL_FIELDS:
        payload[field] = np.asarray(state[field], dtype=bool)
    payload["split"] = np.asarray(state["split"], dtype=np.int8)
    # Explicit missing masks for the commanded-channel diagnostics.
    for prefix in ("factual", "counterfactual"):
        for channel in ("cmd_accel", "cmd_steer"):
            values = payload[f"{prefix}_{channel}"]
            payload[f"{prefix}_{channel}_mask"] = np.isfinite(values)
    payload.update(_derive_accel_fields(state, dt))
    payload["object_capacity"] = np.asarray(capacity, dtype=np.int64)
    payload["object_features_json"] = np.asarray(
        json.dumps(["presence", "is_ego", "x", "y", "vx", "vy",
                    "cos_heading", "sin_heading"]), dtype=object)
    for field in _JSON_FIELDS:
        payload[field] = np.asarray([str(v) for v in state[field]], dtype=object)
    return payload


def _publish(config: Mapping[str, Any], state: Mapping[str, Any],
             schedule: np.ndarray, manifest: Mapping[str, Any],
             action_ids: np.ndarray, capacity: int, dt: float) -> None:
    data_output = sp._resolve(config["data_output"])
    receipt_path = sp._resolve(config["collection_receipt"])
    accepted = len(state["object_history"])
    payload = _build_payload(config, state, capacity, dt)

    receipt = _receipt(config, state, int(state["attempts"]), accepted, partial=False)
    receipt["data_output_sha256_pending"] = True

    tmp_npz = data_output.with_name(data_output.name + ".tmp")
    tmp_receipt = receipt_path.with_name(receipt_path.name + ".tmp")
    with open(tmp_npz, "wb") as handle:
        np.savez_compressed(handle, **payload)
    # Validate the temporary NPZ before publishing.
    with np.load(tmp_npz, allow_pickle=True) as z:
        n = accepted
        assert z["object_history"].shape == (n, sp.HISTORY_LEN, capacity, N_OBJECT_FEATURES)
        assert z["object_factual_future"].shape == (n, sp.FUTURE_LEN, capacity, N_OBJECT_FEATURES)
        assert z["object_counterfactual_future"].shape == (n, sp.FUTURE_LEN, capacity, N_OBJECT_FEATURES)
        assert len(np.unique(z["episode_seed"])) == n
        assert int(z["initial_identity_count"].max()) <= capacity
        assert int(z["max_road_vehicles"].max()) <= capacity
        for field in ("object_history", "object_factual_future", "object_counterfactual_future"):
            if not np.isfinite(z[field]).all():
                raise RuntimeError(f"final NPZ field {field} is non-finite")
    receipt["data_output_sha256"] = sp.sha256_file(tmp_npz)
    receipt.pop("data_output_sha256_pending", None)
    tmp_receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    # Publish receipt first, NPZ last (the NPZ is the completion marker).
    Path(tmp_receipt).replace(receipt_path)
    Path(tmp_npz).replace(data_output)


def _receipt(config: Mapping[str, Any], state: Mapping[str, Any], attempts: int,
             accepted: int, *, partial: bool) -> dict[str, Any]:
    movement_lookup_path = config.get("movement_lookup_json")
    movement_lookup_sha256 = None
    if movement_lookup_path and sp._resolve(movement_lookup_path).is_file():
        movement_lookup_sha256 = sp.sha256_file(sp._resolve(movement_lookup_path))
    return {
        "experiment_name": config["experiment_name"],
        "analysis_status": config["analysis_status"],
        "plan_id": config["plan_id"],
        "role": config["role"],
        "initial_vehicle_count": int(config["initial_vehicle_count"]),
        "partial": bool(partial),
        "requested_pairs": int(config["n_pairs"]),
        "accepted_pairs": int(accepted),
        "attempts": int(attempts),
        "collection_seed": int(config["collection_seed"]),
        "env_config_assertions": True,
        "skip_counts": {str(k): int(v) for k, v in state["skip_counts"].items()},
        "accepted_strata": {f"action_{a}_duration_{d}": int(c)
                            for (a, d), c in sorted(state["accepted_strata"].items())},
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "identity_max_abs_diff": float(state["identity_max_abs_diff"]),
        "identity_replay_checks": int(config["determinism_checks"]),
        "max_road_vehicles_seen": int(state["max_road_vehicles_seen"]),
        "initial_identity_count_max": int(state["initial_identity_count_max"]),
        "movement_lookup_sha256": movement_lookup_sha256,
        "vcs": _vcs_status(),
        "environment": sp._environment_versions(),
        "config_signature": canonical_fingerprint(json.loads(json.dumps(config))),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2B fixed-participant intersection collector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config_path = sp._resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt = collect_stage2b(config, resume=args.resume, config_path=config_path)
    print(f"[collect] role={receipt['role']} accepted={receipt['accepted_pairs']} "
          f"attempts={receipt['attempts']} status={receipt['analysis_status']}", flush=True)
    return receipt


if __name__ == "__main__":
    main()
