"""Stage 4B protocol tests: seed schedule, zero overlap, frozen-chain, guards."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4b_protocol as s4bp  # noqa: E402


def _base_config():
    import json
    return json.loads((ROOT / "experiments/stage4_reliability/stage4b/stage4b_config.json")
                      .read_text(encoding="utf-8"))


# --- seed schedule ----------------------------------------------------------------------


def test_seed_schedule_matches_frozen_contract():
    schedule = s4bp.make_candidate_seed_schedule()
    assert schedule.size == 12000
    assert len(np.unique(schedule)) == 12000
    assert schedule.min() >= 1 and schedule.max() <= 2147483646
    # spawn(3) first stream, draw without replacement, +1 — reproducible.
    import numpy.random as npr
    expected = npr.default_rng(npr.SeedSequence(2026091701).spawn(3)[0]).choice(
        2147483646, size=12000, replace=False) + 1
    assert np.array_equal(schedule, expected.astype(np.int64))


def test_seed_schedule_entropy_and_size_frozen():
    with pytest.raises(ValueError):
        s4bp.make_candidate_seed_schedule(entropy=123)
    with pytest.raises(ValueError):
        s4bp.make_candidate_seed_schedule(size=100)


def test_zero_overlap_rejects_overlapping_schedule():
    schedule = s4bp.make_candidate_seed_schedule()
    from ac_twin.stage2_protocol import validate_seed_schedule
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_schedule(schedule, np.array([schedule[0]], dtype=np.int64))
    validate_seed_schedule(schedule, np.array([0], dtype=np.int64))  # 0 not in [1,2^31-2] usage


# --- stratum balance ---------------------------------------------------------------------


def test_assert_balanced_strata():
    cf_action = np.tile(np.arange(5), 40)
    cf_duration = np.repeat([1, 3], 100)
    s4bp.assert_balanced_strata(cf_action, cf_duration, 20)  # 200 eps / 10 cells = 20
    with pytest.raises(ValueError):
        s4bp.assert_balanced_strata(cf_action[:-1], cf_duration[:-1], 20)


# --- config validation + selected_n contract ----------------------------------------------


def test_config_validates_frozen_constants():
    checked = s4bp.validate_stage4b_config(_base_config())
    assert checked["plan_id"] == s4bp.PLAN_ID
    assert float(checked["top_error_threshold"]) == s4bp.TOP_ERROR_THRESHOLD


def test_config_rejects_threshold_drift():
    config = _base_config()
    config["top_error_threshold"] = 0.5  # must stay frozen to the Stage 4A value
    with pytest.raises(ValueError, match="top_error_threshold"):
        s4bp.validate_stage4b_config(config)


def test_config_rejects_wrong_role_purpose():
    config = _base_config()
    config["purpose"] = "stage4_merge_risk_dev"  # Stage 4A dev purpose, not confirm
    with pytest.raises(ValueError):
        s4bp.validate_stage4b_config(config)


def test_build_lock_requires_selected_n_in_candidates(tmp_path):
    config = _base_config()
    with pytest.raises(ValueError, match="selected_n"):
        s4bp.build_protocol_lock(config, selected_n=2500)


# --- frozen Stage 4A chain + cross-computer relative resolution ----------------------------


def test_frozen_stage4a_inventory_hash_match():
    inventory = s4bp.inventory_frozen_stage4a()
    assert set(inventory["stage4a_sha256"]) == set(s4bp.STAGE4A_FROZEN)
    for name, (_, expected) in s4bp.STAGE4A_FROZEN.items():
        assert inventory["stage4a_sha256"][name] == expected


def test_frozen_paths_resolve_relative_to_project_root():
    # Identity is by content hash; the path is resolved project-relative so a
    # different absolute drive letter still finds the same artifact.
    for name, (rel, expected) in s4bp.STAGE4A_FROZEN.items():
        resolved = s4bp._resolve(rel)
        assert resolved.is_file(), name
        assert not Path(rel).is_absolute()


# --- forbidden-data guard -------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "data/object_centric_stage/stage4_merge_risk_dev/x.npz",       # Stage 4A dev
    "data/object_centric_stage/stage2b_intersection/x.npz",        # Stage 2B
    "data/object_centric_stage/stage3r_rscreen/x.npz",             # Stage 3R screen
    "data/object_centric_stage/stage2_confirmatory/x.npz",
])
def test_forbidden_sources_rejected(path):
    with pytest.raises(ValueError):
        s4bp.assert_allowed_stage4b_data(path)


def test_confirm_role_accepted():
    s4bp.assert_allowed_stage4b_data(
        "data/object_centric_stage/stage4_merge_risk_confirm/"
        "object_twinworld_stage4_merge_risk_confirm.npz")
