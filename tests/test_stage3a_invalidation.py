"""Task 1: immutable invalidation record for the 2026-09-09 misaligned Gate R_dev.

The record must exist, carry the exact status string, and reference the three
original artifacts by SHA256 that can be mechanically recomputed from the files
on disk. The old report/lock/log themselves must remain untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD_RUN_DIR = ROOT / "experiments" / "stage3_routing" / "stage3a"
RECORD_PATH = OLD_RUN_DIR / "INVALIDATION_RECORD_20260909.json"

EXPECTED_STATUS = "INVALID_ANALYSIS_EPISODE_ALIGNMENT_BUG"
REFERENCED_FILES = {
    "invalid_report_sha256": OLD_RUN_DIR / "gate_r_dev_report_h96_r32_l0.1.json",
    "development_test_lock_sha256": OLD_RUN_DIR / "development_test_used.lock",
    "pipeline_log_sha256": OLD_RUN_DIR.parent / "stage3a_pipeline_20260909_014348.log",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_invalidation_record_exists_with_status() -> None:
    assert RECORD_PATH.is_file(), "invalidation record missing"
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    assert record["status"] == EXPECTED_STATUS
    assert record["invalid_report"] == "gate_r_dev_report_h96_r32_l0.1.json"
    assert record["scientific_use"] == "none; engineering failure evidence only"
    assert record["superseded_by"] == "stage3a_retrain_v1 replacement development test"
    assert "root_cause" in record and record["root_cause"]


@pytest.mark.parametrize("field", sorted(REFERENCED_FILES))
def test_recorded_sha256_matches_file_on_disk(field: str) -> None:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    target = REFERENCED_FILES[field]
    assert target.is_file(), f"referenced artifact missing: {target.name}"
    assert record[field] == _sha256(target), (
        f"{field} does not match recomputed SHA256 of {target.name}"
    )


def test_old_artifacts_still_present_and_unmodified() -> None:
    # The invalidation record must never delete or rewrite the old evidence.
    for target in REFERENCED_FILES.values():
        assert target.is_file(), f"old artifact was removed: {target.name}"
    assert (OLD_RUN_DIR / "final_selection.json").is_file()
