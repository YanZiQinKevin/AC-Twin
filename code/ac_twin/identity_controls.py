"""Reproducible controls for ordering object slots in trajectory tensors."""

from __future__ import annotations

import numpy as np


VALID_ORDER_MODES = {"stable", "distance", "permuted"}


def transform_object_order(
    values: np.ndarray,
    mode: str,
    episode_seeds: np.ndarray,
    permutation_seed: int,
    start_step: int = 0,
) -> np.ndarray:
    """Return a copy with non-ego object slots ordered by ``mode``.

    ``values`` has shape ``[N, T, K, F]``.  Slot zero is always kept as the
    ego object; all other slots, including slots with ``presence == 0``, are
    treated as objects and moved together with their complete feature rows.
    """
    values = np.asarray(values)
    episode_seeds = np.asarray(episode_seeds)
    if values.ndim != 4 or values.shape[2] < 1 or values.shape[3] < 4:
        raise ValueError("values must have shape [N,T,K,F]")
    if len(episode_seeds) != len(values):
        raise ValueError("episode_seeds length must equal N")
    if mode not in VALID_ORDER_MODES:
        raise ValueError(f"Unknown order mode: {mode}")

    result = values.copy()
    if mode == "stable":
        return result

    for n in range(values.shape[0]):
        for t in range(values.shape[1]):
            non_ego = values[n, t, 1:]
            if mode == "distance":
                ego_xy = values[n, t, 0, 2:4]
                distance = np.linalg.norm(non_ego[:, 2:4] - ego_xy, axis=-1)
                order = np.argsort(distance, kind="stable")
            else:
                seed = np.random.SeedSequence(
                    [int(permutation_seed), int(episode_seeds[n]), int(start_step + t)]
                )
                order = np.random.default_rng(seed).permutation(len(non_ego))
            result[n, t, 1:] = non_ego[order]
    return result


def _real_scalar_key(value: object) -> tuple[int, int, object]:
    """Return an orderable key with explicit NaN and infinity categories."""
    value_array = np.asarray(value)
    if np.issubdtype(value_array.dtype, np.floating):
        scalar = value_array.item()
        if np.isnan(scalar):
            return (3, 0, 0)
        if np.isneginf(scalar):
            return (1, 0, 0)
        if np.isposinf(scalar):
            return (2, 0, 0)
        return (0, 0, scalar)
    return (0, 0, value_array.item())


def _scalar_key(value: object) -> tuple:
    value_array = np.asarray(value)
    if np.issubdtype(value_array.dtype, np.complexfloating):
        scalar = value_array.item()
        return (0, 1, _real_scalar_key(scalar.real), _real_scalar_key(scalar.imag))
    return _real_scalar_key(value)


def _row_key(row: np.ndarray) -> tuple:
    return tuple(_scalar_key(value) for value in row)


def assert_same_object_multiset(reference: np.ndarray, candidate: np.ndarray) -> None:
    """Assert that each episode/frame retains exactly the same object rows."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise AssertionError("Object tensors have different shapes")
    if not np.issubdtype(reference.dtype, np.number) or not np.issubdtype(
        candidate.dtype, np.number
    ):
        raise ValueError("Object tensors must have numeric dtype")
    for index in np.ndindex(reference.shape[0], reference.shape[1]):
        ref_rows = sorted(_row_key(row) for row in reference[index])
        cand_rows = sorted(_row_key(row) for row in candidate[index])
        if ref_rows != cand_rows:
            raise AssertionError(f"Object multiset differs at episode/frame {index}")
