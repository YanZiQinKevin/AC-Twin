"""Audit stable object identities across seeded Twin-World branches."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin.object_centric import (
    OBJECT_FEATURES,
    assert_matching_initial_identities,
    assign_episode_entity_ids,
    distance_rank_by_entity,
    extract_object_frame,
)
from collect_pilot import action_mapping, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-scene", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--output",
        default="experiments/object_centric_stage/identity_audit.json",
    )
    return parser.parse_args()


def env_config() -> dict[str, Any]:
    return {
        "vehicles_count": 25,
        "duration": 100,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
            "clip": False,
            "order": "sorted",
        },
    }


def audit_scene(
    env_id: str,
    episode_seeds: np.ndarray,
    action_rng: np.random.Generator,
) -> dict[str, Any]:
    cfg = env_config()
    probe = make_env(env_id, cfg)
    probe.reset(seed=int(episode_seeds[0]))
    mapping = action_mapping(probe)
    capacity = len(probe.unwrapped.road.vehicles)
    action_ids = np.asarray(sorted(mapping.values()), dtype=np.int64)
    probe.close()

    completed = 0
    early_terminated = 0
    initial_identity_failures = 0
    prefix_state_failures = 0
    unregistered_vehicle_failures = 0
    maximum_prefix_difference = 0.0
    rank_transition_count = 0
    episodes_with_rank_transition = 0
    divergent_effect_episodes = 0

    for episode_seed in episode_seeds:
        left = make_env(env_id, cfg)
        right = make_env(env_id, cfg)
        left.reset(seed=int(episode_seed))
        right.reset(seed=int(episode_seed))
        left_records = assign_episode_entity_ids(left)
        right_records = assign_episode_entity_ids(right)
        try:
            assert_matching_initial_identities(left_records, right_records)
        except RuntimeError:
            initial_identity_failures += 1
            left.close()
            right.close()
            continue

        prefix_actions = action_rng.choice(action_ids, size=12, replace=True)
        previous_rank = distance_rank_by_entity(left)
        episode_rank_changed = False
        terminated_early = False
        for action in prefix_actions:
            _, _, left_terminated, left_truncated, _ = left.step(int(action))
            _, _, right_terminated, right_truncated, _ = right.step(int(action))
            try:
                left_frame, left_ids = extract_object_frame(left, capacity)
                right_frame, right_ids = extract_object_frame(right, capacity)
            except RuntimeError:
                unregistered_vehicle_failures += 1
                terminated_early = True
                break
            if not np.array_equal(left_ids, right_ids):
                prefix_state_failures += 1
            difference = float(np.max(np.abs(left_frame - right_frame)))
            maximum_prefix_difference = max(maximum_prefix_difference, difference)
            if difference > 1e-6:
                prefix_state_failures += 1
            current_rank = distance_rank_by_entity(left)
            if current_rank != previous_rank:
                rank_transition_count += 1
                episode_rank_changed = True
            previous_rank = current_rank
            if left_terminated or left_truncated or right_terminated or right_truncated:
                terminated_early = True
                break

        if terminated_early:
            early_terminated += 1
            left.close()
            right.close()
            continue

        factual_action = int(action_rng.choice(action_ids))
        counterfactual_options = action_ids[action_ids != factual_action]
        counterfactual_action = int(action_rng.choice(counterfactual_options))
        factual_plan = np.full(10, factual_action, dtype=np.int64)
        counterfactual_plan = factual_plan.copy()
        counterfactual_plan[:3] = counterfactual_action
        max_branch_effect = 0.0
        for factual, counterfactual in zip(factual_plan, counterfactual_plan):
            _, _, left_terminated, left_truncated, _ = left.step(int(factual))
            _, _, right_terminated, right_truncated, _ = right.step(int(counterfactual))
            try:
                left_frame, left_ids = extract_object_frame(left, capacity)
                right_frame, right_ids = extract_object_frame(right, capacity)
            except RuntimeError:
                unregistered_vehicle_failures += 1
                break
            max_branch_effect = max(
                max_branch_effect,
                float(np.max(np.abs(left_frame[:, 2:4] - right_frame[:, 2:4]))),
            )
            current_rank = distance_rank_by_entity(left)
            if current_rank != previous_rank:
                rank_transition_count += 1
                episode_rank_changed = True
            previous_rank = current_rank
            if left_terminated or left_truncated or right_terminated or right_truncated:
                break

        if max_branch_effect > 1e-6:
            divergent_effect_episodes += 1
        if episode_rank_changed:
            episodes_with_rank_transition += 1
        completed += 1
        left.close()
        right.close()

    return {
        "env_id": env_id,
        "requested_episodes": int(len(episode_seeds)),
        "completed_episodes": completed,
        "early_terminated_episodes": early_terminated,
        "entity_capacity": capacity,
        "object_features": list(OBJECT_FEATURES),
        "initial_identity_failures": initial_identity_failures,
        "prefix_state_failures": prefix_state_failures,
        "unregistered_vehicle_failures": unregistered_vehicle_failures,
        "maximum_same_prefix_state_difference": maximum_prefix_difference,
        "distance_sorted_rank_transition_count": rank_transition_count,
        "episodes_with_distance_sorted_rank_transition": episodes_with_rank_transition,
        "episodes_with_nonzero_branch_effect": divergent_effect_episodes,
        "pass": bool(
            initial_identity_failures == 0
            and prefix_state_failures == 0
            and unregistered_vehicle_failures == 0
            and maximum_prefix_difference <= 1e-6
        ),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    episode_seeds = rng.integers(
        0, 2**31 - 1, size=args.episodes_per_scene, dtype=np.int64
    )
    results = {
        "experiment": "stable_identity_audit",
        "run_timestamp": dt.datetime.now().astimezone().isoformat(),
        "audit_seed": args.seed,
        "identity_scope": "episode_local",
        "assignment_rule": "ego_first_then_seeded_environment_creation_order",
        "scenes": {},
    }
    action_rng = np.random.default_rng(args.seed + 1)
    for env_id in ("merge-v0", "roundabout-v0"):
        print(f"[audit] {env_id}", flush=True)
        results["scenes"][env_id] = audit_scene(env_id, episode_seeds, action_rng)
    results["overall_pass"] = all(scene["pass"] for scene in results["scenes"].values())

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
