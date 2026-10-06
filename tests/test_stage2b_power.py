"""Blocking tests for Stage 2B v0.4 power sizing and support simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2b_power as pwr  # noqa: E402


def test_positive_episode_flags_use_three_frozen_positive_bins():
    bins = np.array([
        [[1, 0], [0, 0]],
        [[0, 2], [3, 0]],
        [[0, 0], [0, 0]],
    ])
    valid = np.ones_like(bins, dtype=bool)
    flags = pwr.positive_episode_flags(bins, valid)
    assert flags.tolist() == [[True, False, False], [False, True, True],
                              [False, False, False]]


def test_support_power_is_reproducible_and_counts_bootstrap_draws():
    flags = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    a = pwr.estimate_support_power(flags, [6], replicates=1000,
                                   min_count=2, seed=17)
    b = pwr.estimate_support_power(flags, [6], replicates=1000,
                                   min_count=2, seed=17)
    assert a == b
    assert set(a[6]) == {"positive_low", "positive_mid", "positive_high"}
    assert all(0.0 < value < 1.0 for value in a[6].values())


def test_select_candidate_requires_numeric_and_both_support_sources():
    numeric = {1200: 0.89, 1800: 0.91, 2400: 0.99}
    validation = {
        1200: {name: 0.99 for name in pwr.POSITIVE_BIN_NAMES},
        1800: {name: 0.94 for name in pwr.POSITIVE_BIN_NAMES},
        2400: {name: 0.99 for name in pwr.POSITIVE_BIN_NAMES},
    }
    pilot = {n: {name: 0.99 for name in pwr.POSITIVE_BIN_NAMES}
             for n in numeric}
    assert pwr.select_candidate_n(numeric, validation, pilot) == 2400
    pilot[2400]["positive_high"] = 0.1
    assert pwr.select_candidate_n(numeric, validation, pilot) is None
