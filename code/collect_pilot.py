#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym

from ac_twin.object_centric import assign_episode_entity_ids, extract_object_frame
from ac_twin.checkpointing import load_collection_checkpoint, save_collection_checkpoint
from ac_twin.progress import ProgressReporter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the AC-Twin 2,000-pair pilot dataset.")
    parser.add_argument("--config", default="config/pilot_2000.json")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the partial checkpoint associated with the configured data path.",
    )
    return parser.parse_args(argv)


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


_FINAL_SAMPLE_FIELDS = (
    "history",
    "factual_future",
    "counterfactual_future",
    "factual_actions",
    "counterfactual_actions",
    "cf_action",
    "cf_duration",
    "episode_seed",
    "t_star",
    "prefix_actions",
    "prefix_action_lengths",
    "split",
    "factual_ego_xy",
    "counterfactual_ego_xy",
    "effect_xy",
)
_FINAL_METADATA_FIELDS = (
    "obs_shape",
    "action_mapping_json",
    "env_config_json",
    "collection_stats_json",
    "config_json",
)
_FINAL_OBJECT_FIELDS = (
    "object_history",
    "object_factual_future",
    "object_counterfactual_future",
    "object_capacity",
    "object_features_json",
)


def canonical_config_json(config: dict[str, Any]) -> str:
    """Serialize the complete collection configuration deterministically."""
    return json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _archive_json_value(archive: Any, field: str) -> Any:
    """Read one scalar JSON field from a completed NPZ archive."""
    try:
        value = archive[field].item()
        return json.loads(value)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid embedded {field}") from exc


def validate_completed_collection(
    out_path: Path, stats_path: Path, current_config: dict[str, Any]
) -> None:
    """Check that a final output is a complete, readable collection commit."""
    n_pairs = int(current_config["n_pairs"])
    save_objects = bool(current_config.get("save_object_centric", False))
    if not stats_path.exists():
        raise ValueError(f"missing stats file: {stats_path}")
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable stats JSON: {stats_path}") from exc
    if not isinstance(stats, dict):
        raise ValueError("stats JSON must contain an object")
    if stats.get("requested_pairs") != n_pairs or stats.get("accepted_pairs") != n_pairs:
        raise ValueError("stats pair counts do not match the requested collection")

    required_fields = _FINAL_SAMPLE_FIELDS + _FINAL_METADATA_FIELDS
    if save_objects:
        required_fields += _FINAL_OBJECT_FIELDS
    try:
        archive = np.load(out_path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"unreadable final NPZ: {out_path}") from exc
    try:
        with archive:
            missing = [field for field in required_fields if field not in archive]
            if missing:
                raise ValueError(f"NPZ is missing required fields: {missing}")
            wrong_size = [
                field for field in _FINAL_SAMPLE_FIELDS
                if len(archive[field]) != n_pairs
            ]
            if wrong_size:
                raise ValueError(f"NPZ sample counts do not match: {wrong_size}")
            if save_objects:
                wrong_object_size = [
                    field for field in _FINAL_OBJECT_FIELDS[:3]
                    if len(archive[field]) != n_pairs
                ]
                if wrong_object_size:
                    raise ValueError(
                        f"NPZ object sample counts do not match: {wrong_object_size}"
                    )
            embedded_config = _archive_json_value(archive, "config_json")
            if canonical_config_json(embedded_config) != canonical_config_json(current_config):
                raise ValueError(
                    "embedded collection configuration does not match the current configuration"
                )
            embedded_stats = _archive_json_value(archive, "collection_stats_json")
            if embedded_stats != stats:
                raise ValueError(
                    "sidecar stats do not match embedded collection_stats_json"
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"unreadable final NPZ: {out_path}") from exc


def resolve_collection_start(
    out_path: Path,
    stats_path: Path,
    partial_path: Path,
    resume_requested: bool,
    *,
    current_config: dict[str, Any],
) -> str:
    """Return complete, resume, or new without constructing a simulator."""
    if out_path.exists():
        try:
            validate_completed_collection(out_path, stats_path, current_config)
        except ValueError as exc:
            raise RuntimeError(
                f"Existing final collection is invalid or incomplete: {exc}"
            ) from exc
        return "complete"
    if partial_path.exists():
        return "resume"
    if resume_requested:
        raise FileNotFoundError(
            f"--resume was requested, but no partial checkpoint exists: {partial_path}"
        )
    return "new"


def _write_stats_json(path: Path, stats: dict[str, Any]) -> None:
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def commit_collection_outputs(
    out_path: Path,
    payload: dict[str, np.ndarray],
    stats: dict[str, Any],
    *,
    current_config: dict[str, Any],
    npz_writer: Any = np.savez_compressed,
    stats_writer: Any | None = None,
) -> None:
    """Atomically publish validated stats first and NPZ last as the completion marker."""
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite final dataset: {out_path}")
    stats_path = out_path.with_suffix(".stats.json")
    npz_tmp_path = out_path.with_name(f".{out_path.stem}.tmp{out_path.suffix}")
    stats_tmp_path = stats_path.with_name(
        f".{stats_path.stem}.tmp{stats_path.suffix}"
    )
    writer = _write_stats_json if stats_writer is None else stats_writer
    npz_tmp_path.unlink(missing_ok=True)
    stats_tmp_path.unlink(missing_ok=True)
    try:
        npz_writer(npz_tmp_path, **payload)
        writer(stats_tmp_path, stats)
        validate_completed_collection(npz_tmp_path, stats_tmp_path, current_config)
        os.replace(stats_tmp_path, stats_path)
        os.replace(npz_tmp_path, out_path)
    except Exception:
        npz_tmp_path.unlink(missing_ok=True)
        stats_tmp_path.unlink(missing_ok=True)
        raise


def make_env(env_id: str, cfg: dict[str, Any]) -> gym.Env:
    import gymnasium as gym
    import highway_env  # noqa: F401

    env = gym.make(env_id)
    env.unwrapped.configure(cfg)
    return env


def action_mapping(env: gym.Env) -> dict[str, int]:
    action_type = getattr(env.unwrapped, "action_type", None)
    raw = getattr(action_type, "actions_indexes", {})
    mapping = {str(name).upper(): int(idx) for name, idx in dict(raw).items()}
    if not mapping:
        mapping = {f"ACTION_{i}": i for i in range(int(env.action_space.n))}
    return dict(sorted(mapping.items(), key=lambda item: item[1]))


def action_probabilities(mapping: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(set(mapping.values())), dtype=np.int64)
    name_by_id = {idx: name for name, idx in mapping.items()}
    weights = []
    for idx in ids:
        name = name_by_id[int(idx)]
        if "IDLE" in name or "MAINTAIN" in name:
            weights.append(0.45)
        elif "FASTER" in name or "SLOWER" in name:
            weights.append(0.18)
        else:
            weights.append(0.095)
    probs = np.asarray(weights, dtype=np.float64)
    probs /= probs.sum()
    return ids, probs


def rollout(
    env_id: str,
    env_cfg: dict[str, Any],
    seed: int,
    prefix_actions: np.ndarray,
    future_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    env = make_env(env_id, env_cfg)
    obs, _ = env.reset(seed=int(seed))
    prefix_obs = [np.asarray(obs, dtype=np.float32).reshape(-1)]
    prefix_terminated = False
    for action in prefix_actions:
        obs, _, terminated, truncated, _ = env.step(int(action))
        prefix_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        if terminated or truncated:
            prefix_terminated = True
            break

    future_obs: list[np.ndarray] = []
    if not prefix_terminated:
        for action in future_actions:
            obs, _, terminated, truncated, _ = env.step(int(action))
            future_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            if terminated or truncated:
                break
    env.close()
    return (
        np.asarray(prefix_obs, dtype=np.float32),
        np.asarray(future_obs, dtype=np.float32),
        prefix_terminated,
    )


def rollout_objects(
    env_id: str,
    env_cfg: dict[str, Any],
    seed: int,
    prefix_actions: np.ndarray,
    future_actions: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Replay one branch and return fixed-ID object frames."""
    env = make_env(env_id, env_cfg)
    env.reset(seed=int(seed))
    assign_episode_entity_ids(env)
    prefix_frames = [extract_object_frame(env, capacity)[0]]
    done = False
    for action in prefix_actions:
        _, _, terminated, truncated, _ = env.step(int(action))
        prefix_frames.append(extract_object_frame(env, capacity)[0])
        if terminated or truncated:
            done = True
            break
    future_frames: list[np.ndarray] = []
    if not done:
        for action in future_actions:
            _, _, terminated, truncated, _ = env.step(int(action))
            future_frames.append(extract_object_frame(env, capacity)[0])
            if terminated or truncated:
                done = True
                break
    env.close()
    return np.asarray(prefix_frames), np.asarray(future_frames), done


def stratified_split(
    cf_actions: np.ndarray,
    durations: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.full(len(cf_actions), -1, dtype=np.int8)
    strata = sorted(set(zip(cf_actions.tolist(), durations.tolist())))
    for cf_action, duration in strata:
        idx = np.where((cf_actions == cf_action) & (durations == duration))[0]
        idx = rng.permutation(idx)
        n_train = int(round(len(idx) * train_ratio))
        n_val = int(round(len(idx) * val_ratio))
        n_train = min(n_train, len(idx))
        n_val = min(n_val, len(idx) - n_train)
        split[idx[:n_train]] = 0
        split[idx[n_train : n_train + n_val]] = 1
        split[idx[n_train + n_val :]] = 2
    if np.any(split < 0):
        raise RuntimeError("Stratified split left unassigned samples.")
    return split


_FIXED_COLLECTION_FIELDS = (
    "histories",
    "factual_futures",
    "counterfactual_futures",
    "factual_action_plans",
    "counterfactual_action_plans",
    "cf_action_values",
    "cf_duration_values",
    "episode_seeds",
    "t_stars",
)
_OBJECT_COLLECTION_FIELDS = (
    "object_histories",
    "object_factual_futures",
    "object_counterfactual_futures",
)
_TERMINATION_SKIP_REASONS = (
    "prefix_terminated",
    "factual_future_short",
    "counterfactual_future_short",
)


def empty_skip_counts() -> Counter[str]:
    """Initialize reported termination reasons even when their count is zero."""
    return Counter({reason: 0 for reason in _TERMINATION_SKIP_REASONS})


def record_future_short_counts(
    skip_counts: Counter[str], factual_length: int, counterfactual_length: int, horizon: int
) -> bool:
    """Record short future branches and return whether the pair must be skipped."""
    factual_short = factual_length != horizon
    counterfactual_short = counterfactual_length != horizon
    if factual_short:
        skip_counts["factual_future_short"] += 1
    if counterfactual_short:
        skip_counts["counterfactual_future_short"] += 1
    return factual_short or counterfactual_short


def record_prefix_termination(
    skip_counts: Counter[str],
    *,
    factual_prefix_terminated: bool,
    counterfactual_prefix_terminated: bool,
    factual_prefix_length: int,
    counterfactual_prefix_length: int,
    t_star: int,
) -> bool:
    """Reject a prefix-stage termination before inspecting future lengths."""
    prefix_invalid = (
        factual_prefix_terminated
        or counterfactual_prefix_terminated
        or factual_prefix_length != t_star + 1
        or counterfactual_prefix_length != t_star + 1
    )
    if prefix_invalid:
        skip_counts["prefix_terminated"] += 1
    return prefix_invalid


def empty_collection_state() -> dict[str, Any]:
    return {
        **{field: [] for field in _FIXED_COLLECTION_FIELDS},
        "prefix_action_values": [],
        **{field: [] for field in _OBJECT_COLLECTION_FIELDS},
        "accepted_strata": Counter(),
        "skip_counts": empty_skip_counts(),
        "prefix_max_abs_diff": 0.0,
        "identity_max_abs_diff": 0.0,
        "attempts": 0,
    }


def pack_collection_state(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Convert mutable collection lists and counters into checkpoint-safe values."""
    accepted = len(state["histories"])
    arrays: dict[str, np.ndarray] = {}
    for field in _FIXED_COLLECTION_FIELDS:
        values = state[field]
        if len(values) != accepted:
            raise ValueError(f"Collection field {field} has an inconsistent accepted count")
        arrays[field] = np.stack(values) if values else np.empty((0,), dtype=np.float32)

    prefixes = state["prefix_action_values"]
    if len(prefixes) != accepted:
        raise ValueError("Collection field prefix_action_values has an inconsistent accepted count")
    prefix_lengths = np.asarray([len(value) for value in prefixes], dtype=np.int64)
    max_prefix_len = int(prefix_lengths.max()) if accepted else 0
    prefix_actions = np.full((accepted, max_prefix_len), -1, dtype=np.int64)
    for row, values in enumerate(prefixes):
        prefix_actions[row, : len(values)] = values
    arrays["prefix_actions"] = prefix_actions
    arrays["prefix_action_lengths"] = prefix_lengths

    object_fields_present = any(bool(state[field]) for field in _OBJECT_COLLECTION_FIELDS)
    for field in _OBJECT_COLLECTION_FIELDS:
        values = state[field]
        if object_fields_present and len(values) != accepted:
            raise ValueError(f"Collection field {field} has an inconsistent accepted count")
        if values:
            arrays[field] = np.stack(values)

    metadata = {
        "accepted": accepted,
        "attempts": int(state["attempts"]),
        "prefix_max_abs_diff": float(state["prefix_max_abs_diff"]),
        "identity_max_abs_diff": float(state["identity_max_abs_diff"]),
        "accepted_strata": [
            [int(action), int(duration), int(count)]
            for (action, duration), count in sorted(state["accepted_strata"].items())
        ],
        "skip_counts": {str(key): int(value) for key, value in state["skip_counts"].items()},
    }
    return metadata, arrays


def restore_collection_state(
    arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Restore mutable lists and counters from a validated checkpoint payload."""
    accepted = int(metadata["accepted"])
    state = empty_collection_state()
    for field in _FIXED_COLLECTION_FIELDS:
        values = arrays.get(field)
        if values is None or len(values) != accepted:
            raise ValueError(f"Checkpoint field {field} has an inconsistent accepted count")
        state[field] = [value.copy() if isinstance(value, np.ndarray) else value.item()
                        if isinstance(value, np.generic) else value for value in values]

    prefix_actions = arrays.get("prefix_actions")
    prefix_lengths = arrays.get("prefix_action_lengths")
    if prefix_actions is None or prefix_lengths is None or len(prefix_lengths) != accepted:
        raise ValueError("Checkpoint prefix actions have an inconsistent accepted count")
    state["prefix_action_values"] = [
        prefix_actions[row, : int(length)].copy()
        for row, length in enumerate(prefix_lengths)
    ]

    for field in _OBJECT_COLLECTION_FIELDS:
        if field in arrays:
            if len(arrays[field]) != accepted:
                raise ValueError(f"Checkpoint field {field} has an inconsistent accepted count")
            state[field] = [value.copy() for value in arrays[field]]

    state["accepted_strata"] = Counter({
        (int(action), int(duration)): int(count)
        for action, duration, count in metadata.get("accepted_strata", [])
    })
    state["skip_counts"].update({
        str(key): int(value) for key, value in metadata.get("skip_counts", {}).items()
    })
    state["attempts"] = int(metadata["attempts"])
    state["prefix_max_abs_diff"] = float(metadata["prefix_max_abs_diff"])
    state["identity_max_abs_diff"] = float(metadata["identity_max_abs_diff"])
    return state


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(config["collection_seed"]))
    out_path = project_path(config["paths"]["data"])
    stats_path = out_path.with_suffix(".stats.json")
    partial_path = out_path.with_suffix(".partial.npz")
    collection_start = resolve_collection_start(
        out_path,
        stats_path,
        partial_path,
        args.resume,
        current_config=config,
    )
    if collection_start == "complete":
        print(f"[collect] already complete and verified: {out_path}", flush=True)
        return
    resume_collection = collection_start == "resume"
    if resume_collection and not args.resume:
        print(f"[collect] resuming existing checkpoint={partial_path}", flush=True)

    obs_cfg = {
        "type": "Kinematics",
        "vehicles_count": int(config["observed_vehicles_count"]),
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "clip": False,
        "order": "sorted",
    }
    env_cfg = {
        "vehicles_count": int(config["vehicles_count"]),
        "duration": int(config["duration"]),
        "simulation_frequency": int(config["simulation_frequency"]),
        "policy_frequency": int(config["policy_frequency"]),
        "observation": obs_cfg,
    }

    probe = make_env(config["env_id"], env_cfg)
    probe_obs, _ = probe.reset(seed=int(config["collection_seed"]))
    mapping = action_mapping(probe)
    action_ids, action_probs = action_probabilities(mapping)
    obs_shape = tuple(np.asarray(probe_obs).shape)
    object_capacity = len(probe.unwrapped.road.vehicles)
    probe.close()
    if len(obs_shape) != 2 or obs_shape[1] < 3:
        raise RuntimeError(f"Unexpected kinematics observation shape: {obs_shape}")

    n_pairs = int(config["n_pairs"])
    horizon = int(config["horizon"])
    history_len = int(config["history_len"])
    durations = [int(x) for x in config["intervention_durations"]]
    max_attempts = n_pairs * int(config["max_attempt_multiplier"])
    determinism_checks = int(config["determinism_checks"])

    state = empty_collection_state()
    if resume_collection:
        rng_state, checkpoint_metadata, checkpoint_arrays = load_collection_checkpoint(
            partial_path, config
        )
        rng.bit_generator.state = rng_state
        state = restore_collection_state(checkpoint_arrays, checkpoint_metadata)

    histories = state["histories"]
    factual_futures = state["factual_futures"]
    counterfactual_futures = state["counterfactual_futures"]
    factual_action_plans = state["factual_action_plans"]
    counterfactual_action_plans = state["counterfactual_action_plans"]
    cf_action_values = state["cf_action_values"]
    cf_duration_values = state["cf_duration_values"]
    episode_seeds = state["episode_seeds"]
    t_stars = state["t_stars"]
    prefix_action_values = state["prefix_action_values"]
    object_histories = state["object_histories"]
    object_factual_futures = state["object_factual_futures"]
    object_counterfactual_futures = state["object_counterfactual_futures"]
    save_objects = bool(config.get("save_object_centric", False))

    accepted_strata = state["accepted_strata"]
    skip_counts = state["skip_counts"]
    prefix_max_abs_diff = float(state["prefix_max_abs_diff"])
    identity_max_abs_diff = float(state["identity_max_abs_diff"])
    attempts = int(state["attempts"])
    progress = ProgressReporter(
        total=n_pairs,
        every=int(config.get("progress_every_accepted", 100)),
        prefix="collect",
    )
    if histories:
        progress.update(
            len(histories),
            attempted=attempts,
            details=f"resumed=1 rejected={attempts - len(histories)}",
            force=True,
        )

    all_strata = [(int(action), int(duration)) for action in action_ids for duration in durations]
    while len(histories) < n_pairs and attempts < max_attempts:
        attempts += 1
        episode_seed = int(rng.integers(0, 2**31 - 1))
        t_star = int(rng.integers(int(config["t_star_low"]), int(config["t_star_high"]) + 1))
        prefix_actions = rng.choice(action_ids, size=t_star, p=action_probs).astype(np.int64)
        factual_plan = rng.choice(action_ids, size=horizon, p=action_probs).astype(np.int64)

        eligible = [key for key in all_strata if key[0] != int(factual_plan[0])]
        min_count = min(accepted_strata[key] for key in eligible)
        least_used = [key for key in eligible if accepted_strata[key] == min_count]
        cf_action, cf_duration = least_used[int(rng.integers(0, len(least_used)))]
        cf_duration = min(cf_duration, horizon)
        counterfactual_plan = factual_plan.copy()
        counterfactual_plan[:cf_duration] = int(cf_action)

        f_prefix, f_future, f_prefix_terminated = rollout(
            config["env_id"], env_cfg, episode_seed, prefix_actions, factual_plan
        )
        c_prefix, c_future, c_prefix_terminated = rollout(
            config["env_id"], env_cfg, episode_seed, prefix_actions, counterfactual_plan
        )

        if record_prefix_termination(
            skip_counts,
            factual_prefix_terminated=f_prefix_terminated,
            counterfactual_prefix_terminated=c_prefix_terminated,
            factual_prefix_length=len(f_prefix),
            counterfactual_prefix_length=len(c_prefix),
            t_star=t_star,
        ):
            continue
        current_prefix_diff = float(np.max(np.abs(f_prefix - c_prefix)))
        prefix_max_abs_diff = max(prefix_max_abs_diff, current_prefix_diff)
        if current_prefix_diff > 1e-6:
            skip_counts["prefix_mismatch"] += 1
            continue
        if record_future_short_counts(
            skip_counts, len(f_future), len(c_future), horizon
        ):
            continue
        if not np.isfinite(f_future).all() or not np.isfinite(c_future).all():
            skip_counts["non_finite"] += 1
            continue

        if len(histories) < determinism_checks:
            check_prefix, check_future, check_prefix_terminated = rollout(
                config["env_id"], env_cfg, episode_seed, prefix_actions, factual_plan
            )
            if (
                check_prefix_terminated
                or check_prefix.shape != f_prefix.shape
                or check_future.shape != f_future.shape
            ):
                raise RuntimeError("Identity replay changed trajectory shape.")
            identity_diff = max(
                float(np.max(np.abs(check_prefix - f_prefix))),
                float(np.max(np.abs(check_future - f_future))),
            )
            identity_max_abs_diff = max(identity_max_abs_diff, identity_diff)
            if identity_diff > 1e-6:
                raise RuntimeError(
                    f"Identity replay failed determinism check: max_abs_diff={identity_diff}"
                )

        history = f_prefix[-history_len:]
        if history.shape[0] != history_len:
            skip_counts["short_history"] += 1
            continue

        if save_objects:
            of_prefix, of_future, _ = rollout_objects(
                config["env_id"], env_cfg, episode_seed, prefix_actions, factual_plan,
                object_capacity,
            )
            oc_prefix, oc_future, _ = rollout_objects(
                config["env_id"], env_cfg, episode_seed, prefix_actions,
                counterfactual_plan, object_capacity,
            )
            if (
                len(of_prefix) != t_star + 1
                or len(oc_prefix) != t_star + 1
                or len(of_future) != horizon
                or len(oc_future) != horizon
            ):
                skip_counts["object_replay_shape_mismatch"] += 1
                continue
            if float(np.max(np.abs(of_prefix - oc_prefix))) > 1e-6:
                skip_counts["object_prefix_mismatch"] += 1
                continue
            object_histories.append(of_prefix[-history_len:].astype(np.float32))
            object_factual_futures.append(of_future.astype(np.float32))
            object_counterfactual_futures.append(oc_future.astype(np.float32))

        histories.append(history.astype(np.float32))
        factual_futures.append(f_future.astype(np.float32))
        counterfactual_futures.append(c_future.astype(np.float32))
        factual_action_plans.append(factual_plan)
        counterfactual_action_plans.append(counterfactual_plan)
        cf_action_values.append(int(cf_action))
        cf_duration_values.append(int(cf_duration))
        episode_seeds.append(episode_seed)
        t_stars.append(t_star)
        prefix_action_values.append(prefix_actions)
        accepted_strata[(int(cf_action), int(cf_duration))] += 1

        progress.update(
            len(histories),
            attempted=attempts,
            details=(
                f"rejected={attempts - len(histories)} "
                f"prefix_terminated={skip_counts['prefix_terminated']} "
                f"factual_future_short={skip_counts['factual_future_short']} "
                f"counterfactual_future_short="
                f"{skip_counts['counterfactual_future_short']}"
            ),
        )
        checkpoint_every = int(config.get("checkpoint_every_accepted", 0))
        if checkpoint_every > 0 and len(histories) % checkpoint_every == 0:
            state.update(
                attempts=attempts,
                prefix_max_abs_diff=prefix_max_abs_diff,
                identity_max_abs_diff=identity_max_abs_diff,
            )
            checkpoint_metadata, checkpoint_arrays = pack_collection_state(state)
            save_collection_checkpoint(
                partial_path,
                config,
                rng.bit_generator.state,
                checkpoint_metadata,
                checkpoint_arrays,
            )
            print(f"[collect] checkpoint={partial_path}", flush=True)

    if len(histories) != n_pairs:
        raise RuntimeError(
            f"Collected {len(histories)} valid pairs after {attempts} attempts; expected {n_pairs}."
        )

    history_arr = np.stack(histories)
    factual_future_arr = np.stack(factual_futures)
    counterfactual_future_arr = np.stack(counterfactual_futures)
    factual_action_arr = np.stack(factual_action_plans)
    counterfactual_action_arr = np.stack(counterfactual_action_plans)
    cf_action_arr = np.asarray(cf_action_values, dtype=np.int64)
    cf_duration_arr = np.asarray(cf_duration_values, dtype=np.int64)
    factual_ego_xy = factual_future_arr[:, :, :].reshape(n_pairs, horizon, *obs_shape)[:, :, 0, 1:3]
    counterfactual_ego_xy = counterfactual_future_arr.reshape(n_pairs, horizon, *obs_shape)[:, :, 0, 1:3]
    effect_xy = counterfactual_ego_xy - factual_ego_xy
    effect_fde = np.linalg.norm(effect_xy[:, -1], axis=-1)

    split_cfg = config["split"]
    split_arr = stratified_split(
        cf_action_arr,
        cf_duration_arr,
        float(split_cfg["train"]),
        float(split_cfg["val"]),
        int(config["collection_seed"]) + 1,
    )

    effect_bins = {
        "zero_le_1e-6": int(np.sum(effect_fde <= 1e-6)),
        "small_0_to_1m": int(np.sum((effect_fde > 1e-6) & (effect_fde <= 1.0))),
        "medium_1_to_4m": int(np.sum((effect_fde > 1.0) & (effect_fde <= 4.0))),
        "large_gt_4m": int(np.sum(effect_fde > 4.0)),
    }
    stats = {
        "requested_pairs": n_pairs,
        "accepted_pairs": len(histories),
        "attempts": attempts,
        "acceptance_rate": len(histories) / attempts,
        "skip_counts": dict(skip_counts),
        "prefix_max_abs_diff": prefix_max_abs_diff,
        "identity_replay_checks": determinism_checks,
        "identity_max_abs_diff": identity_max_abs_diff,
        "obs_shape": list(obs_shape),
        "action_mapping": mapping,
        "stratum_counts": {
            f"action_{action}_duration_{duration}": int(count)
            for (action, duration), count in sorted(accepted_strata.items())
        },
        "split_counts": {
            "train": int(np.sum(split_arr == 0)),
            "val": int(np.sum(split_arr == 1)),
            "test": int(np.sum(split_arr == 2)),
        },
        "effect_fde_mean_m": float(effect_fde.mean()),
        "effect_fde_median_m": float(np.median(effect_fde)),
        "effect_fde_bins": effect_bins,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_prefix_len = max(len(actions) for actions in prefix_action_values)
    prefix_action_arr = np.full((n_pairs, max_prefix_len), -1, dtype=np.int64)
    for row, actions in enumerate(prefix_action_values):
        prefix_action_arr[row, : len(actions)] = actions
    payload = dict(
        history=history_arr,
        factual_future=factual_future_arr,
        counterfactual_future=counterfactual_future_arr,
        factual_actions=factual_action_arr,
        counterfactual_actions=counterfactual_action_arr,
        cf_action=cf_action_arr,
        cf_duration=cf_duration_arr,
        episode_seed=np.asarray(episode_seeds, dtype=np.int64),
        t_star=np.asarray(t_stars, dtype=np.int64),
        prefix_actions=prefix_action_arr,
        prefix_action_lengths=np.asarray([len(x) for x in prefix_action_values], dtype=np.int64),
        split=split_arr,
        factual_ego_xy=factual_ego_xy.astype(np.float32),
        counterfactual_ego_xy=counterfactual_ego_xy.astype(np.float32),
        effect_xy=effect_xy.astype(np.float32),
        obs_shape=np.asarray(obs_shape, dtype=np.int64),
        action_mapping_json=np.asarray(json.dumps(mapping), dtype=object),
        env_config_json=np.asarray(json.dumps(env_cfg), dtype=object),
        collection_stats_json=np.asarray(json.dumps(stats), dtype=object),
        config_json=np.asarray(canonical_config_json(config), dtype=object),
    )
    if save_objects:
        payload.update(
            object_history=np.stack(object_histories),
            object_factual_future=np.stack(object_factual_futures),
            object_counterfactual_future=np.stack(object_counterfactual_futures),
            object_capacity=np.asarray(object_capacity, dtype=np.int64),
            object_features_json=np.asarray(
                json.dumps(["presence", "is_ego", "x", "y", "vx", "vy", "cos_heading", "sin_heading"]),
                dtype=object,
            ),
        )
    commit_collection_outputs(
        out_path,
        payload,
        stats,
        current_config=config,
    )
    partial_path.unlink(missing_ok=True)
    progress.update(
        n_pairs,
        attempted=attempts,
        details=f"rejected={attempts - n_pairs}",
        force=True,
    )
    print(f"[collect] saved={out_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
