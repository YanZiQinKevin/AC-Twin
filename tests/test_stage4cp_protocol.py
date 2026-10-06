"""Blocking tests for the Stage 4C Plus protocol layer (plan section 14)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_protocol as s4c  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402

CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4c_plus/stage4cp_config.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_real_config_validates():
    checked = s4cp.validate_config(_config())
    assert checked["plan_id"] == s4cp.PLAN_ID
    assert checked["dev_train_groups"] == 5000


@pytest.mark.parametrize("key,value", [
    ("dev_seed_entropy", 123),
    ("confirm_schedule_size", 999),
    ("high_regret_threshold", 0.02),
    ("gate_i_aurc_regret_min", 0.01),
    ("phase0_fold_entropy", 42),
    ("candidate_ns", [1, 2, 3]),
    ("action_plans", [[0, 1]]),
])
def test_config_drift_rejected(key, value):
    cfg = _config()
    cfg[key] = value
    with pytest.raises(ValueError):
        s4cp.validate_config(cfg)


def test_seed_schedule_contracts_and_uniqueness():
    dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    confirm = s4cp.make_group_seed_schedule(s4cp.CONFIRM_SEED_ENTROPY, s4cp.CONFIRM_SCHEDULE_SIZE)
    assert dev.size == 30000 and confirm.size == 25000
    assert np.unique(dev).size == dev.size
    assert np.unique(confirm).size == confirm.size
    assert dev.min() >= s4cp.SEED_LOW and dev.max() <= s4cp.SEED_HIGH_INCLUSIVE
    # dev and confirm schedules are disjoint.
    assert np.intersect1d(dev, confirm).size == 0
    # only the two frozen (entropy, size) contracts are legal.
    with pytest.raises(ValueError):
        s4cp.make_group_seed_schedule(123, 100)


def test_role_segments_disjoint_and_cover_schedule():
    segs = {r: s4cp.role_segment(r) for r in
            ("development_train", "development_validation", "development_screen")}
    assert segs["development_train"] == (0, 18000)
    assert segs["development_validation"] == (18000, 24000)
    assert segs["development_screen"] == (24000, 30000)
    # disjoint and cover the whole schedule.
    covered = set()
    for start, stop in segs.values():
        covered.update(range(start, stop))
    assert covered == set(range(s4cp.DEV_SCHEDULE_SIZE))
    with pytest.raises(ValueError):
        s4cp.role_segment("confirm")


def test_forbidden_data_guards():
    # Stage 4C screen is permanently read-only.
    with pytest.raises(ValueError):
        s4cp.assert_allowed_stage4cp_data(
            "experiments/stage4_reliability/stage4c/development/screen_group_quantities.npz",
            allow_phase0_selection=True)
    with pytest.raises(ValueError):
        s4cp.assert_allowed_stage4cp_data(
            "data/object_centric_stage/stage4c_merge_multiaction_dev/"
            "object_twinworld_stage4c_merge_multiaction_dev_screen.npz",
            allow_phase0_selection=True)
    # Stage 4C selection allowed ONLY for Phase 0.
    sel = ("data/object_centric_stage/stage4c_merge_multiaction_dev/"
           "object_twinworld_stage4c_merge_multiaction_dev_selection.npz")
    s4cp.assert_allowed_stage4cp_data(sel, allow_phase0_selection=True)  # no raise
    with pytest.raises(ValueError):
        s4cp.assert_allowed_stage4cp_data(sel, allow_phase0_selection=False)
    # own data allowed.
    s4cp.assert_allowed_stage4cp_data(
        "data/object_centric_stage/stage4cp_merge_multiaction_dev/x.npz")
    # Stage 4B confirm forbidden.
    with pytest.raises(ValueError):
        s4cp.assert_allowed_stage4cp_data(
            "experiments/stage4_reliability/stage4b/confirm/x.npz")
    # unrelated path rejected.
    with pytest.raises(ValueError):
        s4cp.assert_allowed_stage4cp_data("data/somewhere_else/x.npz")


def test_historical_scan_includes_stage4c_and_excludes_own():
    roots = [s4cp.ROOT / "data", s4cp.ROOT / "experiments"]
    historical = s4cp.scan_historical_group_seeds(roots)
    # Stage 4C selection/screen group seeds are present (reserved, not reusable).
    stage4c_sel = np.load(
        s4cp.ROOT / "data/object_centric_stage/stage4c_merge_multiaction_dev/"
        "object_twinworld_stage4c_merge_multiaction_dev_selection.npz")["episode_seed"]
    assert np.intersect1d(historical, stage4c_sel).size == stage4c_sel.size
    # Stage 4C never-persisted confirm schedule reserved by contract.
    stage4c_confirm = s4c.make_group_seed_schedule(
        s4cp.STAGE4C_CONFIRM_SEED_ENTROPY, s4cp.STAGE4C_CONFIRM_SCHEDULE_SIZE)
    assert np.intersect1d(historical, stage4c_confirm).size == stage4c_confirm.size
    # New dev/confirm schedules are disjoint from the whole inventory.
    dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    confirm = s4cp.make_group_seed_schedule(s4cp.CONFIRM_SEED_ENTROPY, s4cp.CONFIRM_SCHEDULE_SIZE)
    assert np.intersect1d(dev, historical).size == 0
    assert np.intersect1d(confirm, historical).size == 0


def test_frozen_sources_and_model_lock_resolve():
    sources = s4cp.inventory_frozen_sources()
    assert len(sources["stage4_sha256"]) == 7
    lock = s4cp.inventory_model_lock(_config())
    assert len(lock["checkpoint_sha256"]) == 15
