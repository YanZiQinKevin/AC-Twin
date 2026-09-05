"""Stable episode-local object identities for Highway-env rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np


OBJECT_FEATURES = (
    "presence",
    "is_ego",
    "x",
    "y",
    "vx",
    "vy",
    "cos_heading",
    "sin_heading",
)


def _unwrapped(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def ordered_episode_vehicles(env: Any) -> list[Any]:
    """Return ego first, followed by the environment's initial vehicle order."""
    base = _unwrapped(env)
    if base.road is None or base.vehicle is None:
        raise RuntimeError("Environment must be reset before assigning entity IDs.")
    ego = base.vehicle
    return [ego, *[vehicle for vehicle in base.road.vehicles if vehicle is not ego]]


def assign_episode_entity_ids(env: Any) -> list[dict[str, Any]]:
    """Assign deterministic, episode-local IDs based on seeded creation order."""
    records: list[dict[str, Any]] = []
    for entity_id, vehicle in enumerate(ordered_episode_vehicles(env)):
        setattr(vehicle, "_ac_twin_entity_id", int(entity_id))
        records.append(vehicle_fingerprint(vehicle, entity_id))
    return records


def vehicle_fingerprint(vehicle: Any, entity_id: int | None = None) -> dict[str, Any]:
    """Describe initial identity-defining state for cross-branch validation."""
    route = getattr(vehicle, "route", None)
    normalized_route = None if route is None else [list(item) for item in route]
    position = np.asarray(vehicle.position, dtype=float)
    return {
        "entity_id": (
            int(entity_id)
            if entity_id is not None
            else int(getattr(vehicle, "_ac_twin_entity_id"))
        ),
        "class": type(vehicle).__name__,
        "position_xy": [float(position[0]), float(position[1])],
        "heading": float(vehicle.heading),
        "speed": float(vehicle.speed),
        "route": normalized_route,
    }


def extract_object_frame(
    env: Any,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an ID-indexed object tensor and an aligned entity-ID vector."""
    base = _unwrapped(env)
    frame = np.zeros((capacity, len(OBJECT_FEATURES)), dtype=np.float32)
    entity_ids = np.full(capacity, -1, dtype=np.int64)
    current = list(base.road.vehicles) if base.road is not None else []
    for vehicle in current:
        if not hasattr(vehicle, "_ac_twin_entity_id"):
            raise RuntimeError("Encountered an unregistered vehicle during rollout.")
        entity_id = int(getattr(vehicle, "_ac_twin_entity_id"))
        if entity_id < 0 or entity_id >= capacity:
            raise RuntimeError(
                f"Entity ID {entity_id} is outside configured capacity {capacity}."
            )
        velocity = np.asarray(vehicle.velocity, dtype=float)
        position = np.asarray(vehicle.position, dtype=float)
        frame[entity_id] = np.asarray(
            [
                1.0,
                float(vehicle is base.vehicle),
                position[0],
                position[1],
                velocity[0],
                velocity[1],
                np.cos(float(vehicle.heading)),
                np.sin(float(vehicle.heading)),
            ],
            dtype=np.float32,
        )
        entity_ids[entity_id] = entity_id
    return frame, entity_ids


def assert_matching_initial_identities(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    atol: float = 1e-6,
) -> None:
    """Fail if seeded branches do not create the same entity sequence."""
    if len(left) != len(right):
        raise RuntimeError(f"Vehicle-count mismatch: {len(left)} != {len(right)}")
    for left_record, right_record in zip(left, right):
        for key in ("entity_id", "class", "route"):
            if left_record[key] != right_record[key]:
                raise RuntimeError(
                    f"Identity mismatch for key {key}: "
                    f"{left_record[key]} != {right_record[key]}"
                )
        for key in ("position_xy", "heading", "speed"):
            if not np.allclose(left_record[key], right_record[key], atol=atol, rtol=0.0):
                raise RuntimeError(
                    f"Initial-state mismatch for entity {left_record['entity_id']} key {key}."
                )


def distance_rank_by_entity(env: Any) -> tuple[int, ...]:
    """Return the distance-sorted entity order used to audit row-order instability."""
    base = _unwrapped(env)
    ego = base.vehicle
    others = [vehicle for vehicle in base.road.vehicles if vehicle is not ego]
    others.sort(key=lambda vehicle: float(np.linalg.norm(vehicle.position - ego.position)))
    return (int(getattr(ego, "_ac_twin_entity_id")),) + tuple(
        int(getattr(vehicle, "_ac_twin_entity_id")) for vehicle in others
    )
