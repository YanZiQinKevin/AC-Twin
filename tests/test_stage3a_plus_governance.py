"""Stage 3A Plus Task 1: governance, frozen seed sets and qualification-source
role rejection (plan `AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1` §Task 1).

The single-use qualification set may only ever be the freshly collected
``stage3a_plus_qualification`` data. Every already-viewed or future source —
the sealed Stage 2 holdout, the contaminated old merge split 2, the consumed
replacement development holdout, any old ``stage3a*`` run directory and any
Gate C path — is refused by path and by known content SHA256.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_data as s3d  # noqa: E402

# Paths that must NEVER serve as the Stage 3A Plus qualification source.
STAGE2_AND_VIEWED_AND_GATEC_PATHS = [
    # Sealed Stage 2 confirmatory holdout.
    "data/object_centric_stage/stage2_merge_1200/object_twinworld_stage2_merge_1200.npz",
    # Old development source (its split 2 is the contaminated diagnostic A).
    "data/object_centric_stage/merge_2000/object_twinworld_2000.npz",
    # Consumed replacement development holdout (diagnostic B, single-use lock burned).
    "data/object_centric_stage/stage3a_replacement_dev/object_twinworld_stage3a_replacement_dev.npz",
    # Old Stage 3A run directories (read-only, viewed development products).
    "experiments/stage3_routing/stage3a/anything.npz",
    "experiments/stage3_routing/stage3a_retrain_v1/anything.npz",
    # Gate C holdouts (present or future).
    "data/object_centric_stage/gatec_merge_id/object_twinworld_gatec_merge_id.npz",
    "data/object_centric_stage/gatec_merge_unseen/object_twinworld_gatec_merge_unseen.npz",
]

# Known content fingerprints of viewed/contaminated sources (rename/relocate
# bypasses must still be refused).
VIEWED_SOURCE_SHA256S = [
    s3d.STAGE2_HOLDOUT_FILE_SHA256,
    s3d.OLD_DEVELOPMENT_SOURCE_FILE_SHA256,
    s3d.REPLACEMENT_DEV_FILE_SHA256,
]


def test_qualification_rejects_every_viewed_or_future_source():
    for path in STAGE2_AND_VIEWED_AND_GATEC_PATHS:
        with pytest.raises(ValueError):
            s3d.assert_stage3a_plus_source_role(path, "qualification")


def test_qualification_rejects_viewed_sources_by_sha256():
    # A renamed/relocated copy of a viewed source is still refused by content hash.
    ok_path = "data/object_centric_stage/stage3a_plus_qualification/x.npz"
    for sha in VIEWED_SOURCE_SHA256S:
        with pytest.raises(ValueError):
            s3d.assert_stage3a_plus_source_role(ok_path, "qualification", file_sha256=sha)


def test_qualification_accepts_only_the_dedicated_directory():
    ok = "data/object_centric_stage/stage3a_plus_qualification/object_twinworld_stage3a_plus_qualification.npz"
    s3d.assert_stage3a_plus_source_role(ok, "qualification")  # must not raise
    # A plausible-looking but wrong directory is refused even with a fresh hash.
    with pytest.raises(ValueError):
        s3d.assert_stage3a_plus_source_role(
            "data/object_centric_stage/some_other_dir/x.npz", "qualification"
        )


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        s3d.assert_stage3a_plus_source_role(
            "data/object_centric_stage/stage3a_plus_qualification/x.npz", "training"
        )


def test_seed_sets_are_exact_and_disjoint():
    assert s3d.STAGE3A_PLUS_DEV_SEEDS == (101, 202, 303, 404, 505)
    assert s3d.STAGE3A_PLUS_STRESS_SEEDS == (606, 707, 808, 909, 1010)
    assert set(s3d.STAGE3A_PLUS_DEV_SEEDS).isdisjoint(s3d.STAGE3A_PLUS_STRESS_SEEDS)
    assert s3d.STAGE3A_PLUS_ALL_SEEDS == (
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010
    )
    # The five production seeds are exactly the Stage 3A retrain predictor seeds.
    assert s3d.STAGE3A_PLUS_DEV_SEEDS == (101, 202, 303, 404, 505)


def test_plan_id_and_frozen_numbers():
    assert s3d.STAGE3A_PLUS_PLAN_ID == "AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1"
    # Frozen qualification power/collection/gate constants (plan §6).
    assert s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS == (400, 600, 800, 1000, 1200, 1600, 2000)
    assert s3d.STAGE3A_PLUS_TARGET_POWER == 0.90
    assert s3d.STAGE3A_PLUS_POWER_SEED == 20260930
    assert s3d.STAGE3A_PLUS_COLLECTION_SEED == 20261009
    assert s3d.STAGE3A_PLUS_GATE_BOOTSTRAP_SEED == 20260929


def test_status_file_exists_tracks_a_valid_state():
    status = ROOT / "experiments" / "stage3_routing" / "stage3a_plus" / "STAGE3A_PLUS_STATUS.md"
    assert status.is_file(), "STAGE3A_PLUS_STATUS.md missing"
    text = status.read_text(encoding="utf-8")
    assert "AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1" in text
    import run_stage3a_plus as driver

    known_states = set(driver.ALLOWED_TRANSITIONS) | set(driver.TERMINAL_STATES)
    assert any(state in text for state in known_states)
