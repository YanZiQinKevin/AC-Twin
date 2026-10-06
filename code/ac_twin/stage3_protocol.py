"""Stage 3 protocol: config validation, joint lock, Gate Q and output signatures.

The two formal Gate C holdouts — ``gatec_merge_id`` (duration in {1,3}) and
``gatec_merge_unseen_duration`` (duration = 2) — are bound by a single joint
lock and unsealed exactly once. They are created only after
``Gate R_dev + Gate T_dev + Gate C_dev + power`` all pass, must be mutually
exclusive, and must have zero overlap with the historical episode-seed universe
(which includes the permanently sealed Stage 2 holdout).

The sealed Stage 2 holdout is rejected here by the same triple rule as the data
layer (normalized path, logical path, known SHA256) so it can never be smuggled
into a Stage 3 config. ``output_signature`` is recomputable from the on-disk
report JSON, and a second unlock is refused once a formal report exists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.stage1_data import sha256_file
from ac_twin.stage2_protocol import (
    make_seed_schedule,
    scan_historical_episode_seeds,
    validate_seed_schedule,
)
from ac_twin.stage3_data import (
    EXPECTED_TRAIN_DURATIONS,
    UNSEEN_DURATION,
    assert_not_sealed_stage2,
    is_sealed_stage2_path,
)

PLAN_ID = "AC-TWIN-RFER-STAGE3-2026-09-08-V0.2"
STATUS_DEVELOPMENT = "DEVELOPMENT"
STATUS_CONFIRMATORY = "CONFIRMATORY_GATE_C"

# Formal Gate C holdout identities.
GATEC_ID_NAME = "gatec_merge_id"
GATEC_UNSEEN_NAME = "gatec_merge_unseen_duration"
FORMAL_HOLDOUTS = (GATEC_ID_NAME, GATEC_UNSEEN_NAME)

# Candidate episode-count range for the formal sample size (section 6.2).
N_CANDIDATE_MIN = 800
N_CANDIDATE_MAX = 2400
N_CANDIDATE_STEP = 100

_FORBIDDEN_KEYS = (
    "retrain",
    "fine_tune",
    "fit_response_bins",
    "sample_limits",
    "refit_normalization",
)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def validate_stage3_config(config: Mapping[str, Any]) -> None:
    """Validate a Stage 3 config; raise on any protocol violation."""
    for key in _FORBIDDEN_KEYS:
        if key in config:
            raise ValueError(f"config.{key} is forbidden in Stage 3")

    # Development source must be the merge_2000 file and must NOT be the sealed
    # Stage 2 holdout (checked by path and, when present, by recorded hash).
    source = str(config.get("development_source", ""))
    if not source:
        raise ValueError("config.development_source is required")
    assert_not_sealed_stage2(source)
    recorded_hash = config.get("development_source_sha256")
    if recorded_hash is not None:
        assert_not_sealed_stage2(None, file_sha256=str(recorded_hash))

    # Every referenced data path must be sealed-holdout-free.
    for key, value in config.items():
        if isinstance(value, str) and value.endswith(".npz"):
            assert_not_sealed_stage2(value)

    # Response bins must be the frozen Stage 1 thresholds.
    bins = config.get("response_bins", {})
    from ac_twin.stage3_data import FROZEN_THRESHOLDS
    for key, expected in FROZEN_THRESHOLDS.items():
        if float(bins.get(key, float("nan"))) != float(expected):
            raise ValueError(f"response_bins.{key} must equal the frozen {expected}")

    # Formal holdout durations: ID replicates {1,3}; unseen is exactly {2}.
    formal = config.get("formal_holdouts", {})
    if formal:
        id_durations = set(formal.get(GATEC_ID_NAME, {}).get("durations", []))
        unseen_durations = set(formal.get(GATEC_UNSEEN_NAME, {}).get("durations", []))
        if id_durations != EXPECTED_TRAIN_DURATIONS:
            raise ValueError(
                f"{GATEC_ID_NAME} durations must be {sorted(EXPECTED_TRAIN_DURATIONS)}"
            )
        if unseen_durations != {UNSEEN_DURATION}:
            raise ValueError(f"{GATEC_UNSEEN_NAME} durations must be {{{UNSEEN_DURATION}}}")


# --------------------------------------------------------------------------- #
# Output signature (recomputable from the on-disk report JSON)
# --------------------------------------------------------------------------- #
def output_signature(report: Mapping[str, Any]) -> str:
    """Canonical SHA256 of a report mapping; stable across processes."""
    return canonical_fingerprint(dict(report))


def output_signature_from_disk(report_path: str | Path) -> str:
    """Recompute the output signature from a saved report JSON."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    stored = report.pop("output_signature", None)
    recomputed = output_signature(report)
    if stored is not None and stored != recomputed:
        raise ValueError(
            "stored output_signature does not match the recomputed signature "
            f"(stored={stored}, recomputed={recomputed})"
        )
    return recomputed


# --------------------------------------------------------------------------- #
# Single-unlock enforcement
# --------------------------------------------------------------------------- #
def assert_single_unlock(report_path: str | Path) -> None:
    """Refuse a second unblinding once a formal report already exists."""
    if Path(report_path).is_file():
        raise FileExistsError(
            f"a formal report already exists at {report_path}; "
            "a second unblinding is forbidden"
        )


# --------------------------------------------------------------------------- #
# Joint lock over the two formal holdouts
# --------------------------------------------------------------------------- #
def write_joint_lock(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    historical_seeds: np.ndarray,
    seed_schedules: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write one joint lock binding both formal holdouts; refuse overwrite."""
    output_dir = Path(output_dir)
    lock_path = output_dir / "stage3_joint_lock.json"
    if lock_path.is_file():
        raise FileExistsError(f"joint lock already exists: {lock_path}")
    validate_stage3_config(config)

    schedule_hashes = {}
    for name in FORMAL_HOLDOUTS:
        if name not in seed_schedules:
            raise ValueError(f"missing seed schedule for {name}")
        schedule = np.asarray(seed_schedules[name])
        validate_seed_schedule(schedule, historical_seeds)
        schedule_hashes[name] = hashlib.sha256(schedule.tobytes()).hexdigest()

    # The two schedules must be mutually exclusive.
    id_set = set(np.asarray(seed_schedules[GATEC_ID_NAME]).tolist())
    unseen_set = set(np.asarray(seed_schedules[GATEC_UNSEEN_NAME]).tolist())
    if id_set & unseen_set:
        raise ValueError("the two formal holdout seed schedules overlap")

    lock = {
        "plan_id": PLAN_ID,
        "formal_holdouts": list(FORMAL_HOLDOUTS),
        "seed_schedule_sha256": schedule_hashes,
        "config_signature": canonical_fingerprint(dict(config)),
        "n_historical_seeds": int(len(historical_seeds)),
    }
    lock["lock_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = lock_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    tmp.replace(lock_path)
    return lock


def verify_joint_lock(output_dir: str | Path) -> dict[str, Any]:
    """Recompute the joint-lock self-hash; raise on tampering. Read-only."""
    lock_path = Path(output_dir) / "stage3_joint_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"joint lock missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    stored = lock.pop("lock_sha256")
    recomputed = hashlib.sha256(
        json.dumps(lock, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if stored != recomputed:
        raise ValueError("joint lock self-hash mismatch (tampering detected)")
    lock["lock_sha256"] = stored
    return lock


# --------------------------------------------------------------------------- #
# Gate Q (technical validity, evaluated before any scientific adjudication)
# --------------------------------------------------------------------------- #
def evaluate_gate_q(checks: Mapping[str, bool]) -> dict[str, Any]:
    """AND the technical-validity checks; any failure is INVALID_TECHNICAL."""
    required = (
        "lock_hash_match",
        "seed_schedule_zero_overlap",
        "holdouts_mutually_exclusive",
        "duration_set_asserted",
        "no_sealed_stage2_leakage",
        "no_oracle_leakage",
        "parameter_budget_ok",
        "data_output_hash_verified",
    )
    missing = [k for k in required if k not in checks]
    if missing:
        raise ValueError(f"Gate Q is missing checks: {missing}")
    passed = all(bool(checks[k]) for k in required)
    return {"passed": passed, "checks": {k: bool(checks[k]) for k in required}}


__all__ = [
    "PLAN_ID",
    "FORMAL_HOLDOUTS",
    "GATEC_ID_NAME",
    "GATEC_UNSEEN_NAME",
    "validate_stage3_config",
    "output_signature",
    "output_signature_from_disk",
    "assert_single_unlock",
    "write_joint_lock",
    "verify_joint_lock",
    "evaluate_gate_q",
    "scan_historical_episode_seeds",
    "make_seed_schedule",
    "is_sealed_stage2_path",
]
