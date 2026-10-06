#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from wm.dataset import save_factual_npz
from wm.env_utils import get_env_config, make_env, pretty_action_mapping
from wm.utils import ensure_dir, flatten_obs, set_global_seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect factual one-step transitions.")
    p.add_argument("--env_id", type=str, default="merge-v0")
    p.add_argument("--n_steps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_episodes", type=int, default=0, help="0 means unlimited")
    p.add_argument("--vehicles_count", type=int, default=25)
    p.add_argument("--duration", type=int, default=80)
    p.add_argument("--simulation_frequency", type=int, default=15)
    p.add_argument("--policy_frequency", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="outputs/data")
    p.add_argument(
        "--out_name", type=str, default="factual_transitions.npz", help="Output filename"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    out_dir = ensure_dir(args.out_dir)
    out_path = out_dir / args.out_name

    cfg = {
        "vehicles_count": args.vehicles_count,
        "duration": args.duration,
        "simulation_frequency": args.simulation_frequency,
        "policy_frequency": args.policy_frequency,
    }
    try:
        env = make_env(args.env_id, cfg)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create env '{args.env_id}'. Is highway-env installed? Error: {e}"
        ) from e

    print(f"[collect_factual] env={args.env_id}, action_mapping={pretty_action_mapping(env)}")
    rng = np.random.default_rng(args.seed)

    obs_list, act_list, next_obs_list, done_list = [], [], [], []
    ep_ids, step_ids, ep_seed_list = [], [], []
    episode_lengths = []

    total_steps = 0
    ep = 0
    while total_steps < args.n_steps and (args.n_episodes <= 0 or ep < args.n_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        ep_len = 0
        terminated = truncated = False

        while (
            total_steps < args.n_steps
            and not terminated
            and not truncated
            and (args.n_episodes <= 0 or ep < args.n_episodes)
        ):
            obs_vec = flatten_obs(obs)
            act = int(env.action_space.sample())
            next_obs, _, terminated, truncated, _ = env.step(act)
            next_obs_vec = flatten_obs(next_obs)

            obs_list.append(obs_vec)
            act_list.append(act)
            next_obs_list.append(next_obs_vec)
            done_list.append(bool(terminated or truncated))
            ep_ids.append(ep)
            step_ids.append(ep_len)
            ep_seed_list.append(ep_seed)

            obs = next_obs
            ep_len += 1
            total_steps += 1
        episode_lengths.append(ep_len)
        ep += 1

    if total_steps == 0:
        raise RuntimeError("No transitions collected. Check env config and n_steps.")

    obs_arr = np.stack(obs_list).astype(np.float32)
    act_arr = np.asarray(act_list, dtype=np.int64)
    next_obs_arr = np.stack(next_obs_list).astype(np.float32)
    done_arr = np.asarray(done_list, dtype=bool)
    ep_ids_arr = np.asarray(ep_ids, dtype=np.int64)
    step_ids_arr = np.asarray(step_ids, dtype=np.int64)
    ep_seed_arr = np.asarray(ep_seed_list, dtype=np.int64)
    env_cfg = get_env_config(env)

    save_factual_npz(
        path=str(out_path),
        obs=obs_arr,
        acts=act_arr,
        next_obs=next_obs_arr,
        dones=done_arr,
        episode_ids=ep_ids_arr,
        step_ids=step_ids_arr,
        episode_seeds=ep_seed_arr,
        env_config=env_cfg,
    )
    env.close()

    print(f"[collect_factual] Saved: {out_path}")
    print(
        f"[collect_factual] transitions={len(obs_arr)}, episodes={ep}, "
        f"avg_episode_len={float(np.mean(episode_lengths)):.2f}"
    )


if __name__ == "__main__":
    main()

