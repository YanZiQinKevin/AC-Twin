import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Import torch lazily to avoid unnecessary OpenMP runtime loading in data-only scripts.
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Data collection can proceed without torch.
        pass


def to_float32_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32)


def flatten_obs(obs: Any) -> np.ndarray:
    return to_float32_obs(obs).reshape(-1)


def obs_to_mat(obs_flat: Any, n_objects: int = 5, n_features: int = 5) -> np.ndarray:
    arr = to_float32_obs(obs_flat).reshape(-1)
    expected = n_objects * n_features
    if arr.size != expected:
        raise ValueError(
            f"obs_to_mat expects flat size {expected} (={n_objects}x{n_features}), got {arr.size}."
        )
    return arr.reshape(n_objects, n_features)


def mat_to_obs(mat: Any) -> np.ndarray:
    arr = to_float32_obs(mat)
    if arr.ndim != 2:
        raise ValueError(f"mat_to_obs expects 2D matrix, got shape {arr.shape}.")
    return arr.reshape(-1).astype(np.float32)


def apply_struct_mask(
    obs_flat: Any,
    mode: str,
    radius: float = 30.0,
    topk: int | None = None,
    *,
    n_objects: int = 5,
    n_features: int = 5,
    presence_idx: int = 0,
    x_idx: int = 1,
    y_idx: int = 2,
    ego_row: int = 0,
) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "none":
        return to_float32_obs(obs_flat).reshape(-1)

    mat = obs_to_mat(obs_flat, n_objects=n_objects, n_features=n_features).copy()
    if not (0 <= ego_row < n_objects):
        raise ValueError(f"ego_row={ego_row} out of range for n_objects={n_objects}.")
    if not (0 <= presence_idx < n_features and 0 <= x_idx < n_features and 0 <= y_idx < n_features):
        raise ValueError(
            f"indices out of range: presence={presence_idx}, x={x_idx}, y={y_idx}, "
            f"n_features={n_features}"
        )

    ego_xy = mat[ego_row, [x_idx, y_idx]].astype(np.float32)
    rows = [r for r in range(n_objects) if r != ego_row]
    if not rows:
        return mat_to_obs(mat)

    pres = mat[rows, presence_idx].astype(np.float32)
    present_mask = pres > 0.0

    if mode == "presence":
        for i, r in enumerate(rows):
            if not bool(present_mask[i]):
                mat[r, :] = 0.0
        return mat_to_obs(mat)

    dists = np.full(len(rows), np.inf, dtype=np.float32)
    for i, r in enumerate(rows):
        if present_mask[i]:
            xy = mat[r, [x_idx, y_idx]].astype(np.float32)
            dists[i] = float(np.linalg.norm(xy - ego_xy))

    if mode == "radius":
        for i, r in enumerate(rows):
            if (not bool(present_mask[i])) or (dists[i] > float(radius)):
                mat[r, :] = 0.0
        return mat_to_obs(mat)

    if mode == "topk":
        if topk is None:
            raise ValueError("topk mode requires non-None topk.")
        k = max(0, int(topk))
        present_indices = np.where(present_mask)[0]
        keep = set()
        if k > 0 and len(present_indices) > 0:
            order_local = np.argsort(dists[present_indices])
            chosen = present_indices[order_local[:k]]
            keep = set(int(i) for i in chosen.tolist())
        for i, r in enumerate(rows):
            if i not in keep:
                mat[r, :] = 0.0
        return mat_to_obs(mat)

    raise ValueError(f"Unknown struct mask mode='{mode}'. Expected one of: none,presence,radius,topk.")


def save_json(path: str | os.PathLike[str], obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
