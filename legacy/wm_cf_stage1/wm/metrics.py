from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def compute_ade_fde(pred_xy: np.ndarray, true_xy: np.ndarray) -> tuple[float, float]:
    n = min(len(pred_xy), len(true_xy))
    if n <= 0:
        return float("nan"), float("nan")
    diff = pred_xy[:n] - true_xy[:n]
    d = np.linalg.norm(diff, axis=-1)
    ade = float(np.mean(d))
    fde = float(d[-1])
    return ade, fde


def mean_and_se(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se

