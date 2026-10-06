from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.identity_controls import (  # noqa: E402
    assert_same_object_multiset,
    transform_object_order,
)


def sample_objects(steps: int = 1, episodes: int = 1) -> np.ndarray:
    """Build trackable objects whose non-ego contents are equal each frame."""
    ego = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    # The x coordinate is also an identity marker.  The last object is absent
    # but remains a real slot that must be carried through every permutation.
    non_ego = np.array(
        [
            [1.0, 0.0, 3.0, 4.0, -1000.0],
            [1.0, 0.0, 1.0, 0.0, 1000.0],
            [1.0, 0.0, 0.0, 2.0, 500.0],
            [1.0, 0.0, 2.0, 2.0, -500.0],
            [1.0, 0.0, -2.0, 1.0, 250.0],
            [0.0, 0.0, 9.0, 9.0, -999.0],
        ]
    )
    frame = np.vstack([ego, non_ego])
    return np.broadcast_to(frame, (episodes, steps, *frame.shape)).copy()


def test_distance_order_keeps_ego_and_sorts_non_ego_by_metric_distance():
    x = sample_objects()
    changed = transform_object_order(
        x,
        mode="distance",
        episode_seeds=np.array([17]),
        permutation_seed=91,
        start_step=0,
    )

    np.testing.assert_array_equal(changed[:, :, 0], x[:, :, 0])
    distances = np.linalg.norm(
        changed[0, 0, 1:, 2:4] - changed[0, 0, 0, 2:4], axis=-1
    )
    assert np.all(distances[:-1] <= distances[1:])
    np.testing.assert_array_equal(
        changed[0, 0, 1:, 2:4],
        np.array([[1.0, 0.0], [0.0, 2.0], [-2.0, 1.0], [2.0, 2.0], [3.0, 4.0], [9.0, 9.0]]),
    )
    assert_same_object_multiset(x, changed)


def test_per_frame_permutation_is_reproducible_and_changes_identity_tracks():
    x = sample_objects(steps=6)
    first = transform_object_order(x, "permuted", np.array([17]), 91, 0)
    second = transform_object_order(x, "permuted", np.array([17]), 91, 0)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, :, 0], x[:, :, 0])
    assert_same_object_multiset(x, first)
    assert any(
        not np.array_equal(first[0, t, 1:], first[0, 0, 1:]) for t in range(1, 6)
    )


def test_stable_order_returns_an_equal_copy_and_preserves_absent_slots():
    x = sample_objects(steps=2)

    changed = transform_object_order(x, "stable", np.array([17]), 91, 0)

    np.testing.assert_array_equal(changed, x)
    assert changed is not x
    assert not np.shares_memory(changed, x)
    assert_same_object_multiset(x, changed)
    changed[0, 0, 0, 2] = 123.0
    assert x[0, 0, 0, 2] == 0.0


def test_invalid_mode_and_shape_raise_value_error():
    x = sample_objects()

    with pytest.raises(ValueError, match="Unknown order mode"):
        transform_object_order(x, "unknown", np.array([17]), 91, 0)
    with pytest.raises(ValueError, match="shape"):
        transform_object_order(x[0], "stable", np.array([17]), 91, 0)


def test_episode_seed_count_must_match_episode_count():
    with pytest.raises(ValueError, match="episode_seeds length"):
        transform_object_order(sample_objects(steps=2), "stable", np.array([]), 91, 0)


@pytest.mark.parametrize(
    "bad_values",
    [np.empty((1, 1, 0, 5)), np.empty((1, 1, 1, 3))],
    ids=["zero-object-slots", "missing-xy-features"],
)
def test_object_tensor_requires_object_slot_and_xy_features(bad_values):
    with pytest.raises(ValueError, match="shape"):
        transform_object_order(bad_values, "stable", np.array([17]), 91, 0)


def test_permutation_seed_changes_reproducible_permutation():
    x = sample_objects(steps=8, episodes=2)
    baseline = transform_object_order(x, "permuted", np.array([17, 23]), 91, 0)
    changed = transform_object_order(x, "permuted", np.array([17, 23]), 92, 0)

    assert not np.array_equal(baseline[:, :, 1:], changed[:, :, 1:])


def test_episode_seed_changes_reproducible_permutation():
    x = sample_objects(steps=8, episodes=2)
    baseline = transform_object_order(x, "permuted", np.array([17, 23]), 91, 0)
    changed = transform_object_order(x, "permuted", np.array([17, 29]), 91, 0)

    assert not np.array_equal(baseline[:, :, 1:], changed[:, :, 1:])


def test_start_step_changes_reproducible_permutation():
    x = sample_objects(steps=8, episodes=2)
    baseline = transform_object_order(x, "permuted", np.array([17, 23]), 91, 0)
    changed = transform_object_order(x, "permuted", np.array([17, 23]), 91, 1)

    assert not np.array_equal(baseline[:, :, 1:], changed[:, :, 1:])


def test_multiset_assertion_handles_nan_inf_and_rejects_actual_change():
    reference = sample_objects()
    reference[0, 0, 1, 4] = np.nan
    reference[0, 0, 2, 4] = np.inf
    reference[0, 0, 3, 4] = -np.inf
    candidate = reference.copy()
    candidate[0, 0, 1:] = reference[0, 0, [3, 1, 6, 2, 4, 5]]

    assert_same_object_multiset(reference, candidate)

    candidate[0, 0, 1, 2] += 0.5

    with pytest.raises(AssertionError, match="multiset differs"):
        assert_same_object_multiset(reference, candidate)


def test_multiset_assertion_rejects_non_numeric_dtype():
    reference = np.full((1, 1, 2, 4), "object", dtype="U6")

    with pytest.raises(ValueError, match="numeric"):
        assert_same_object_multiset(reference, reference.copy())
