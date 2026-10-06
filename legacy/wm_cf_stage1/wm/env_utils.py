from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from wm.utils import to_float32_obs


DEFAULT_ENV_CONFIG = {
    "vehicles_count": 25,
    "duration": 80,
    "simulation_frequency": 15,
    "policy_frequency": 5,
}


def make_env(env_id: str, config_overrides: dict[str, Any] | None = None) -> gym.Env:
    # highway-env registers its env ids on import.
    try:
        import highway_env  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"Failed to import highway_env for env registration: {e}. "
            "Please install with `pip install highway-env`."
        ) from e

    try:
        env = gym.make(env_id)
    except Exception as e:
        known_ids = sorted(
            [spec.id for spec in gym.envs.registry.values() if "merge" in spec.id.lower()]
        )
        raise RuntimeError(
            f"Failed to create env '{env_id}'. Known merge-like env ids: {known_ids}. "
            f"Original error: {e}"
        ) from e

    config = dict(DEFAULT_ENV_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if hasattr(env.unwrapped, "configure"):
        env.unwrapped.configure(config)
    return env


def get_env_config(env: gym.Env) -> dict[str, Any]:
    cfg = dict(DEFAULT_ENV_CONFIG)
    if hasattr(env.unwrapped, "config"):
        cfg.update(dict(env.unwrapped.config))
    return cfg


def get_action_mapping(env: gym.Env) -> dict[str, int]:
    mapping: dict[str, int] = {}
    action_type = getattr(env.unwrapped, "action_type", None)
    if action_type is not None and hasattr(action_type, "actions_indexes"):
        raw = dict(action_type.actions_indexes)
        mapping = {str(k).upper(): int(v) for k, v in raw.items()}
    if not mapping:
        if hasattr(env.action_space, "n"):
            mapping = {f"ACTION_{i}": i for i in range(int(env.action_space.n))}
    return mapping


def resolve_action_id(env: gym.Env, action_name: str) -> int:
    mapping = get_action_mapping(env)
    if not mapping:
        raise RuntimeError("Could not infer action mapping from environment.")
    q = action_name.upper()
    if q in mapping:
        return mapping[q]

    aliases = {
        "LEFT": ["LEFT", "LANE_LEFT"],
        "RIGHT": ["RIGHT", "LANE_RIGHT"],
        "IDLE": ["IDLE", "MAINTAIN"],
        "FASTER": ["FASTER"],
        "SLOWER": ["SLOWER"],
    }
    cands = aliases.get(q, [q])
    for cand in cands:
        if cand in mapping:
            return mapping[cand]
    for key, idx in mapping.items():
        for cand in cands:
            if cand in key:
                return idx
    raise RuntimeError(
        f"Action '{action_name}' not found. Available action mapping: {mapping}"
    )


def pretty_action_mapping(env: gym.Env) -> str:
    mapping = get_action_mapping(env)
    if not mapping:
        return "<unknown>"
    pairs = sorted(mapping.items(), key=lambda x: x[1])
    return ", ".join([f"{k}:{v}" for k, v in pairs])


def extract_ego_xy(obs: Any, mode: str | None = None) -> tuple[np.ndarray, str]:
    arr = to_float32_obs(obs)
    if arr.ndim >= 2 and arr.shape[0] >= 1 and arr.shape[1] >= 3:
        ego = arr[0]
        if mode == "kinematics_presence_xy":
            return np.asarray([ego[1], ego[2]], dtype=np.float32), "kinematics_presence_xy"
        if mode == "kinematics_xy_guess":
            return np.asarray([ego[0], ego[1]], dtype=np.float32), "kinematics_xy_guess"
        # Most highway-env kinematics obs start with [presence, x, y, ...].
        if 0.0 <= float(ego[0]) <= 1.0:
            return np.asarray([ego[1], ego[2]], dtype=np.float32), "kinematics_presence_xy"
        return np.asarray([ego[0], ego[1]], dtype=np.float32), "kinematics_xy_guess"
    flat = arr.reshape(-1)
    if flat.shape[0] < 2:
        raise RuntimeError(f"Observation has fewer than 2 dims: shape={arr.shape}")
    return np.asarray([flat[0], flat[1]], dtype=np.float32), "proxy_obs_first2"
