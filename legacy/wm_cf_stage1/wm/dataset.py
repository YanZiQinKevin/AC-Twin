from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from wm.utils import apply_struct_mask


def save_factual_npz(
    path: str,
    obs: np.ndarray,
    acts: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    episode_ids: np.ndarray,
    step_ids: np.ndarray,
    episode_seeds: np.ndarray,
    env_config: dict[str, Any],
) -> None:
    np.savez_compressed(
        path,
        obs=obs.astype(np.float32),
        acts=acts.astype(np.int64),
        next_obs=next_obs.astype(np.float32),
        dones=dones.astype(bool),
        episode_ids=episode_ids.astype(np.int64),
        step_ids=step_ids.astype(np.int64),
        episode_seeds=episode_seeds.astype(np.int64),
        env_config_json=np.asarray(json.dumps(env_config), dtype=object),
    )


def load_factual_npz(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "env_config_json" in out:
        out["env_config"] = json.loads(str(out["env_config_json"].item()))
    return out


def save_twinworld_npz(path: str, payload: dict[str, Any]) -> None:
    encoded = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            encoded[k] = np.asarray(json.dumps(v), dtype=object)
        else:
            encoded[k] = v
    np.savez_compressed(path, **encoded)


def load_twinworld_npz(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    for key in ["env_config_json", "split_json", "data_stats_json"]:
        if key in out:
            out[key.replace("_json", "")] = json.loads(str(out[key].item()))
    return out


def episode_level_split(
    episode_ids: np.ndarray, seed: int, train_ratio: float, val_ratio: float
) -> dict[str, np.ndarray]:
    unique_eps = np.unique(episode_ids.astype(np.int64))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_eps)
    n = len(perm)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_eps = perm[:n_train]
    val_eps = perm[n_train : n_train + n_val]
    test_eps = perm[n_train + n_val :]
    return {"train": train_eps, "val": val_eps, "test": test_eps}


class FactualTransitionDataset(Dataset):
    def __init__(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        next_obs: np.ndarray,
        *,
        struct_mode: str = "none",
        radius: float = 30.0,
        topk: int | None = None,
    ) -> None:
        self.obs = np.asarray(obs, dtype=np.float32)
        self.acts = np.asarray(acts, dtype=np.int64)
        self.next_obs = np.asarray(next_obs, dtype=np.float32)
        if len(self.obs) != len(self.acts) or len(self.obs) != len(self.next_obs):
            raise ValueError("obs/acts/next_obs length mismatch in FactualTransitionDataset.")
        self.struct_mode = struct_mode
        self.radius = radius
        self.topk = topk

    def __len__(self) -> int:
        return int(len(self.obs))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_masked = apply_struct_mask(
            self.obs[idx], mode=self.struct_mode, radius=self.radius, topk=self.topk
        ).astype(np.float32)
        act = self.acts[idx]
        nxt = self.next_obs[idx]
        return (
            torch.from_numpy(obs_masked).float(),
            torch.tensor(int(act), dtype=torch.long),
            torch.from_numpy(nxt).float(),
        )


def mask_obs_for_model(
    obs_flat: np.ndarray, struct_mode: str = "none", radius: float = 30.0, topk: int | None = None
) -> np.ndarray:
    return apply_struct_mask(obs_flat, mode=struct_mode, radius=radius, topk=topk).astype(np.float32)
