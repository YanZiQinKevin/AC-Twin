#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ac_twin.object_centric import assign_episode_entity_ids, extract_object_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the AC-Twin 2,000-pair pilot dataset.")
    parser.add_argument("--config", default="config/pilot_2000.json")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_env(env_id: str, cfg: dict[str, Any]) -> gym.Env:
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
    done = False
    for action in prefix_actions:
        obs, _, terminated, truncated, _ = env.step(int(action))
        prefix_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        if terminated or truncated:
            done = True
            break

    future_obs: list[np.ndarray] = []
    if not done:
        for action in future_actions:
            obs, _, terminated, truncated, _ = env.step(int(action))
            future_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            if terminated or truncated:
                done = True
                break
    env.close()
    return (
        np.asarray(prefix_obs, dtype=np.float32),
        np.asarray(future_obs, dtype=np.float32),
        done,
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


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(config["collection_seed"]))

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

    histories: list[np.ndarray] = []
    factual_futures: list[np.ndarray] = []
    counterfactual_futures: list[np.ndarray] = []
    factual_action_plans: list[np.ndarray] = []
    counterfactual_action_plans: list[np.ndarray] = []
    cf_action_values: list[int] = []
    cf_duration_values: list[int] = []
    episode_seeds: list[int] = []
    t_stars: list[int] = []
    prefix_action_values: list[np.ndarray] = []
    object_histories: list[np.ndarray] = []
    object_factual_futures: list[np.ndarray] = []
    object_counterfactual_futures: list[np.ndarray] = []
    save_objects = bool(config.get("save_object_centric", False))

    accepted_strata: Counter[tuple[int, int]] = Counter()
    skip_counts: Counter[str] = Counter()
    prefix_max_abs_diff = 0.0
    identity_max_abs_diff = 0.0
    attempts = 0

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

        f_prefix, f_future, _ = rollout(
            config["env_id"], env_cfg, episode_seed, prefix_actions, factual_plan
        )
        c_prefix, c_future, _ = rollout(
            config["env_id"], env_cfg, episode_seed, prefix_actions, counterfactual_plan
        )

        if len(f_prefix) != t_star + 1 or len(c_prefix) != t_star + 1:
            skip_counts["prefix_terminated"] += 1
            continue
        current_prefix_diff = float(np.max(np.abs(f_prefix - c_prefix)))
        prefix_max_abs_diff = max(prefix_max_abs_diff, current_prefix_diff)
        if current_prefix_diff > 1e-6:
            skip_counts["prefix_mismatch"] += 1
            continue
        if len(f_future) != horizon or len(c_future) != horizon:
            skip_counts["short_future"] += 1
            continue
        if not np.isfinite(f_future).all() or not np.isfinite(c_future).all():
            skip_counts["non_finite"] += 1
            continue

        if len(histories) < determinism_checks:
            check_prefix, check_future, _ = rollout(
                config["env_id"], env_cfg, episode_seed, prefix_actions, factual_plan
            )
            if check_prefix.shape != f_prefix.shape or check_future.shape != f_future.shape:
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

        if len(histories) % 100 == 0:
            print(
                f"[collect] accepted={len(histories)}/{n_pairs}, attempts={attempts}, "
                f"short_future={skip_counts['short_future']}",
                flush=True,
            )

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

    out_path = project_path(config["paths"]["data"])
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
        config_json=np.asarray(json.dumps(config), dtype=object),
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
    np.savez_compressed(out_path, **payload)
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[collect] saved={out_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
