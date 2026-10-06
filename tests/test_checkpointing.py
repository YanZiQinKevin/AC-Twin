from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.checkpointing import (  # noqa: E402
    atomic_save_npz,
    canonical_fingerprint,
    load_collection_checkpoint,
    save_collection_checkpoint,
)


def test_fingerprint_is_order_independent() -> None:
    assert canonical_fingerprint({"a": 1, "b": 2}) == canonical_fingerprint(
        {"b": 2, "a": 1}
    )


def test_collection_checkpoint_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "collection.partial.npz"
    config = {"env_id": "roundabout-v0", "n_pairs": 2}
    arrays = {"history": np.zeros((1, 10, 25), np.float32)}

    save_collection_checkpoint(
        target, config, {"state": 7}, {"attempts": 3, "accepted": 1}, arrays
    )
    rng_state, metadata, restored = load_collection_checkpoint(target, config)

    assert rng_state == {"state": 7}
    assert metadata["attempts"] == 3
    np.testing.assert_array_equal(restored["history"], arrays["history"])


def test_checkpoint_rejects_different_config(tmp_path: Path) -> None:
    target = tmp_path / "collection.partial.npz"
    save_collection_checkpoint(target, {"n_pairs": 2}, {}, {}, {"x": np.arange(2)})

    with pytest.raises(ValueError, match="configuration does not match"):
        load_collection_checkpoint(target, {"n_pairs": 3})


def test_atomic_save_does_not_leave_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "state.npz"

    atomic_save_npz(target, x=np.arange(3))

    assert target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    with np.load(target, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["x"], np.arange(3))


def test_checkpoint_rejects_reserved_array_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved keys"):
        save_collection_checkpoint(
            tmp_path / "state.npz",
            {},
            {},
            {},
            {"__metadata_json__": np.arange(1)},
        )
