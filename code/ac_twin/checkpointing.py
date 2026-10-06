"""Small atomic checkpoint helpers for resumable AC-Twin experiments."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


_RESERVED_KEYS = {
    "__config_fingerprint__",
    "__rng_state_json__",
    "__metadata_json__",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def canonical_fingerprint(value: dict) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible mapping."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Atomically replace ``path`` with ``temporary``, retrying transient OS locks.

    On Windows the target can be transiently locked by OneDrive sync or antivirus
    (WinError 5 access denied / WinError 32 sharing violation, both ``PermissionError``).
    ``os.replace`` either succeeds or leaves the target untouched, so a bounded
    exponential-backoff retry is safe and preserves the atomic-write contract. Only
    ``PermissionError`` is retried; any other error propagates immediately.
    """
    delay = 0.5
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ archive completely before atomically replacing its target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_collection_checkpoint(
    path: Path,
    config: dict,
    rng_state: dict,
    metadata: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    """Save arrays and JSON metadata needed to resume collection exactly."""
    conflicts = _RESERVED_KEYS.intersection(arrays)
    if conflicts:
        raise ValueError(f"Checkpoint arrays use reserved keys: {sorted(conflicts)}")
    payload = dict(arrays)
    payload.update(
        {
            "__config_fingerprint__": np.asarray(canonical_fingerprint(config)),
            "__rng_state_json__": np.asarray(_canonical_json(rng_state)),
            "__metadata_json__": np.asarray(_canonical_json(metadata)),
        }
    )
    atomic_save_npz(Path(path), **payload)


def load_collection_checkpoint(
    path: Path, config: dict
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    """Load a checkpoint only when it was created by the same configuration."""
    with np.load(Path(path), allow_pickle=False) as archive:
        fingerprint = str(archive["__config_fingerprint__"].item())
        if fingerprint != canonical_fingerprint(config):
            raise ValueError("Checkpoint configuration does not match the current config")
        rng_state = json.loads(str(archive["__rng_state_json__"].item()))
        metadata = json.loads(str(archive["__metadata_json__"].item()))
        arrays = {
            key: archive[key].copy()
            for key in archive.files
            if key not in _RESERVED_KEYS
        }
    return rng_state, metadata, arrays
