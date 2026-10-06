#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from wm.dataset import save_twinworld_npz
from wm.env_utils import (
    extract_ego_xy,
    get_env_config,
    make_env,
    pretty_action_mapping,
    resolve_action_id,
)
from wm.utils import ensure_dir, flatten_obs, set_global_seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Twin-World factual/counterfactual dataset.")
    p.add_argument("--env_id", type=str, default="merge-v0")
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--H", type=int, default=50, help="Counterfactual horizon")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t_star_low", type=int, default=5)
    p.add_argument("--t_star_high", type=int, default=30)
    p.add_argument("--act_cf", type=str, default="LEFT")
    p.add_argument("--vehicles_count", type=int, default=25)
    p.add_argument("--duration", type=int, default=80)
    p.add_argument("--simulation_frequency", type=int, default=15)
    p.add_argument("--policy_frequency", type=int, default=5)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--out_dir", type=str, default="outputs/data")
    p.add_argument("--out_name", type=str, default="twinworld_merge.npz")
    return p.parse_args()


def rollout_world(
    env_id: str,
    cfg: dict[str, Any],
    seed: int,
    prefix_actions: list[int],
    first_action: int,
    replay_actions: list[int],
    horizon: int,
) -> tuple[list[np.ndarray], list[int], bool]:
    env = make_env(env_id, cfg)
    obs, _ = env.reset(seed=seed)
    done = False

    # Control variables: identical seed and identical prefix actions.
    for a in prefix_actions:
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            done = True
            break
    fut_obs: list[np.ndarray] = []
    executed: list[int] = []

    if not done:
        action_plan = [int(first_action)] + [int(a) for a in replay_actions]
        for a in action_plan[:horizon]:
            obs, _, term, trunc, _ = env.step(a)
            fut_obs.append(flatten_obs(obs))
            executed.append(a)
            if term or trunc:
                done = True
                break
    env.close()
    return fut_obs, executed, done


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    out_dir = ensure_dir(args.out_dir)
    out_path = out_dir / args.out_name
    rng = np.random.default_rng(args.seed)

    cfg = {
        "vehicles_count": args.vehicles_count,
        "duration": args.duration,
        "simulation_frequency": args.simulation_frequency,
        "policy_frequency": args.policy_frequency,
    }
    try:
        probe_env = make_env(args.env_id, cfg)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create env '{args.env_id}'. Is highway-env installed? Error: {e}"
        ) from e

    act_cf = resolve_action_id(probe_env, args.act_cf)
    print(f"[collect_twinworld] env={args.env_id}, action_mapping={pretty_action_mapping(probe_env)}")
    print(f"[collect_twinworld] act_cf={args.act_cf} -> {act_cf}")
    env_cfg = get_env_config(probe_env)
    probe_obs, _ = probe_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    obs_shape = list(np.asarray(probe_obs).shape)
    try:
        _, pos_hint = extract_ego_xy(probe_obs)
    except Exception:
        pos_hint = "proxy_obs_first2"
    probe_env.close()

    episode_ids = []
    episode_seeds = []
    obs_prefix = []
    act_prefix = []
    t_stars = []
    act_factual = []
    act_cf_arr = []
    act_future = []
    fut_factual = []
    fut_cf = []
    fut_len_factual = []
    fut_len_cf = []
    skipped_short = 0
    episode_lengths = []
    divergence_warnings = 0

    for ep in range(args.n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        env = make_env(args.env_id, cfg)
        obs, _ = env.reset(seed=ep_seed)
        episode_obs = [flatten_obs(obs)]
        episode_actions: list[int] = []
        terminated = truncated = False
        while not terminated and not truncated:
            a = int(rng.integers(0, env.action_space.n))
            next_obs, _, terminated, truncated, _ = env.step(a)
            episode_actions.append(a)
            episode_obs.append(flatten_obs(next_obs))
        env.close()

        T = len(episode_actions)
        episode_lengths.append(T)
        tmax = min(args.t_star_high, T - 2)
        if tmax < args.t_star_low:
            skipped_short += 1
            continue
        t_star = int(rng.integers(args.t_star_low, tmax + 1))

        prefix_a = episode_actions[:t_star]
        a_factual = int(episode_actions[t_star])
        replay = [int(a) for a in episode_actions[t_star + 1 : t_star + args.H]]

        f_obs, f_exec, _ = rollout_world(
            env_id=args.env_id,
            cfg=cfg,
            seed=ep_seed,
            prefix_actions=prefix_a,
            first_action=a_factual,
            replay_actions=replay,
            horizon=args.H,
        )
        c_obs, c_exec, _ = rollout_world(
            env_id=args.env_id,
            cfg=cfg,
            seed=ep_seed,
            prefix_actions=prefix_a,
            first_action=act_cf,
            replay_actions=replay,
            horizon=args.H,
        )

        if len(f_exec) > 0 and f_exec[0] != a_factual:
            divergence_warnings += 1

        episode_ids.append(ep)
        episode_seeds.append(ep_seed)
        obs_prefix.append(np.asarray(episode_obs[: t_star + 1], dtype=np.float32))
        act_prefix.append(np.asarray(prefix_a, dtype=np.int64))
        t_stars.append(t_star)
        act_factual.append(a_factual)
        act_cf_arr.append(act_cf)
        act_future.append(np.asarray(replay, dtype=np.int64))
        fut_factual.append(np.asarray(f_obs, dtype=np.float32))
        fut_cf.append(np.asarray(c_obs, dtype=np.float32))
        fut_len_factual.append(len(f_obs))
        fut_len_cf.append(len(c_obs))

    n_samples = len(episode_ids)
    if n_samples == 0:
        raise RuntimeError(
            "No valid Twin-World samples collected. Try smaller t_star_low or larger duration."
        )

    perm = np.random.default_rng(args.seed + 1).permutation(np.asarray(episode_ids))
    n_train = int(n_samples * args.train_ratio)
    n_val = int(n_samples * args.val_ratio)
    split = {
        "train": perm[:n_train].tolist(),
        "val": perm[n_train : n_train + n_val].tolist(),
        "test": perm[n_train + n_val :].tolist(),
    }

    uniq_fut, cnt_fut = np.unique(np.asarray(fut_len_cf), return_counts=True)
    fut_dist = {str(int(k)): int(v) for k, v in zip(uniq_fut, cnt_fut)}
    stats = {
        "num_samples": n_samples,
        "num_requested_episodes": args.n_episodes,
        "num_skipped_short_episodes": skipped_short,
        "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "fut_len_cf_distribution": fut_dist,
        "fut_len_cf_mean": float(np.mean(fut_len_cf)),
        "fut_len_factual_mean": float(np.mean(fut_len_factual)),
        "divergence_warnings": divergence_warnings,
    }

    payload = {
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "episode_seeds": np.asarray(episode_seeds, dtype=np.int64),
        "obs_prefix": np.asarray(obs_prefix, dtype=object),
        "act_prefix": np.asarray(act_prefix, dtype=object),
        "t_star": np.asarray(t_stars, dtype=np.int64),
        "act_factual": np.asarray(act_factual, dtype=np.int64),
        "act_cf": np.asarray(act_cf_arr, dtype=np.int64),
        "act_future": np.asarray(act_future, dtype=object),
        "fut_factual": np.asarray(fut_factual, dtype=object),
        "fut_cf": np.asarray(fut_cf, dtype=object),
        "fut_len_factual": np.asarray(fut_len_factual, dtype=np.int64),
        "fut_len_cf": np.asarray(fut_len_cf, dtype=np.int64),
        "env_config_json": json.dumps(env_cfg),
        "split_json": json.dumps(split),
        "data_stats_json": json.dumps(stats),
        "obs_shape": np.asarray(obs_shape, dtype=np.int64),
        "position_hint": np.asarray(pos_hint, dtype=object),
        "cf_action_name": np.asarray(args.act_cf, dtype=object),
    }
    save_twinworld_npz(str(out_path), payload)

    print(f"[collect_twinworld] Saved: {out_path}")
    print(f"[collect_twinworld] stats={json.dumps(stats, ensure_ascii=False)}")
    print(
        "[collect_twinworld] note=Twin-World controls seed+prefix actions. "
        "Randomness may still diverge after intervention due to simulator branching."
    )


if __name__ == "__main__":
    main()

