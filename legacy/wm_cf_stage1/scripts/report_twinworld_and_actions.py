#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from wm.dataset import load_twinworld_npz
from wm.env_utils import get_action_mapping, make_env, pretty_action_mapping
from wm.utils import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print Twin-World data stats + action meanings (LEFT/RIGHT ids)."
    )
    p.add_argument("--twin_data", type=str, default="outputs/data/twinworld_merge.npz")
    p.add_argument("--env_id", type=str, default="merge-v0")
    p.add_argument("--out_json", type=str, default="outputs/eval/twinworld_action_report.json")
    return p.parse_args()


def count_dist(arr: np.ndarray) -> dict[str, int]:
    u, c = np.unique(arr.astype(np.int64), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u, c)}


def main() -> None:
    args = parse_args()

    data = load_twinworld_npz(args.twin_data)
    t_star = np.asarray(data["t_star"], dtype=np.int64)
    fut_len_cf = np.asarray(data["fut_len_cf"], dtype=np.int64)
    fut_len_factual = np.asarray(data["fut_len_factual"], dtype=np.int64)
    episode_ids = np.asarray(data["episode_ids"], dtype=np.int64)

    env = make_env(args.env_id)
    mapping = get_action_mapping(env)
    mapping_pretty = pretty_action_mapping(env)
    env.close()

    left_id = mapping.get("LEFT", mapping.get("LANE_LEFT", None))
    right_id = mapping.get("RIGHT", mapping.get("LANE_RIGHT", None))

    stats = {
        "num_samples_N": int(len(episode_ids)),
        "num_unique_episodes": int(len(np.unique(episode_ids))),
        "t_star_distribution": count_dist(t_star),
        "fut_len_cf_distribution": count_dist(fut_len_cf),
        "fut_len_factual_distribution": count_dist(fut_len_factual),
        "fut_len_cf_mean": float(np.mean(fut_len_cf)) if len(fut_len_cf) else 0.0,
        "fut_len_factual_mean": float(np.mean(fut_len_factual)) if len(fut_len_factual) else 0.0,
        "action_mapping_pretty": mapping_pretty,
        "action_mapping_dict": mapping,
        "left_action_id": None if left_id is None else int(left_id),
        "right_action_id": None if right_id is None else int(right_id),
    }

    save_json(args.out_json, stats)

    print("[report] Twin-World sample stats:")
    print(f"N={stats['num_samples_N']}, unique_episodes={stats['num_unique_episodes']}")
    print(f"t_star_distribution={json.dumps(stats['t_star_distribution'], ensure_ascii=False)}")
    print(
        f"fut_len_cf_distribution={json.dumps(stats['fut_len_cf_distribution'], ensure_ascii=False)}"
    )
    print("[report] Action meanings:")
    print(stats["action_mapping_pretty"])
    print(f"LEFT/LANE_LEFT id={stats['left_action_id']}, RIGHT/LANE_RIGHT id={stats['right_action_id']}")
    print(f"[report] Saved json: {args.out_json}")


if __name__ == "__main__":
    main()

