"""Pure Stage 2B v0.4 power and response-bin support utilities."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


POSITIVE_BIN_NAMES = ("positive_low", "positive_mid", "positive_high")
POSITIVE_BIN_IDS = (1, 2, 3)
CONFIRM_CANDIDATES = (1200, 1800, 2400, 3600)
NUMERIC_TARGET = 0.90
SUPPORT_TARGET = 0.95
MIN_BIN_EPISODES = 100


def positive_episode_flags(bin_id: np.ndarray,
                           valid_mask: np.ndarray) -> np.ndarray:
    """Return E x 3 flags for whether each episode supports each positive bin."""
    bin_id = np.asarray(bin_id, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if bin_id.shape != valid_mask.shape or bin_id.ndim != 3:
        raise ValueError("bin_id and valid_mask must share an E x T x N shape")
    return np.column_stack([
        np.any(valid_mask & (bin_id == value), axis=(1, 2))
        for value in POSITIVE_BIN_IDS
    ])


def estimate_support_power(flags: np.ndarray, candidates: Sequence[int], *,
                           replicates: int, min_count: int, seed: int
                           ) -> dict[int, dict[str, float]]:
    """Bootstrap probability that every positive bin reaches ``min_count``."""
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim != 2 or flags.shape[1] != 3 or flags.shape[0] < 2:
        raise ValueError("flags must have shape pool_episodes x 3")
    rng = np.random.default_rng(int(seed))
    result: dict[int, dict[str, float]] = {}
    for n in candidates:
        passed = np.zeros((int(replicates), 3), dtype=bool)
        for start in range(0, int(replicates), 250):
            size = min(250, int(replicates) - start)
            draw = rng.integers(0, flags.shape[0], size=(size, int(n)))
            counts = flags[draw].sum(axis=1)
            passed[start:start + size] = counts >= int(min_count)
        result[int(n)] = {
            name: float(passed[:, column].mean())
            for column, name in enumerate(POSITIVE_BIN_NAMES)
        }
    return result


def select_candidate_n(numeric_joint: Mapping[int, float],
                       validation_support: Mapping[int, Mapping[str, float]],
                       pilot_support: Mapping[int, Mapping[str, float]], *,
                       numeric_target: float = NUMERIC_TARGET,
                       support_target: float = SUPPORT_TARGET) -> int | None:
    """Select the smallest candidate satisfying the preregistered AND rules."""
    for n in sorted(int(value) for value in numeric_joint):
        if float(numeric_joint[n]) < float(numeric_target):
            continue
        if all(float(validation_support[n][name]) >= float(support_target)
               and float(pilot_support[n][name]) >= float(support_target)
               for name in POSITIVE_BIN_NAMES):
            return n
    return None
