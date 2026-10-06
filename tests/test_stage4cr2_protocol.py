"""Blocking tests for the Stage 4C-R2 protocol layer (plan v0.2 section 14).

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2``. v0.2 changed only
the development seed entropy to ``2026092880`` after the v0.1 entropy failed the mandatory
global zero-overlap audit (plan section 5.2); the audit test below re-asserts zero overlap
for the new frozen schedule.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4c_protocol as s4c  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402

CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# --- (f) config validation ---------------------------------------------------------


def test_real_config_validates():
    checked = s4cr2.validate_config(_config())
    assert checked["plan_id"] == s4cr2.PLAN_ID
    assert checked["plan_id"].endswith("V0.2")
    assert checked["dev_train_groups"] == 5000
    assert checked["dev_validation_groups"] == 1500
    assert checked["dev_screen_groups"] == 8000


@pytest.mark.parametrize("key,value", [
    ("plan_id", "AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.1"),
    ("dev_seed_entropy", 2026092701),  # the v0.1 entropy that failed the audit
    ("dev_schedule_size", 30000),
    ("dev_train_groups", 5001),
    ("dev_screen_groups", 4000),
    ("train_segment", [0, 18000]),
    ("validation_segment", [18000, 24000]),
    ("screen_segment", [24000, 60000]),
    ("smoke_seed_entropy", 2026092701),
    ("bootstrap_entropy", 2026092402),
    ("model_seeds", [7101, 7202, 7303, 7404, 7505]),
    ("direct_hidden_width", 64),
    ("expected_trainable_parameters", 15746),
    ("lambda_cls", 0.5),
    ("dropout", 0.2),
    ("high_regret_threshold", 0.02),
    ("validation_i_aurc_min", 0.04),
    ("gate_margin_spearman_ceiling", 0.99),
    ("baseline_tie_tolerance", 0.005),
    ("internal_monitor_frac", 0.2),
    ("train_min_high_regret_groups", 700),
    ("candidate_ns", [1200, 1800, 2400, 3600]),
    ("action_plans", [[0, 1]]),
])
def test_config_drift_rejected(key, value):
    cfg = _config()
    cfg[key] = value
    with pytest.raises(ValueError):
        s4cr2.validate_config(cfg)


# --- frozen objective / model contract reused bitwise --------------------------------


def test_objective_reused_bitwise_from_stage4c():
    assert s4cr2.ACTION_PLANS is s4co.ACTION_PLANS
    assert list(s4cr2.ACTION_PLANS) == [
        (0, 1), (0, 3), (1, 1), (1, 3), (2, 1),
        (2, 3), (3, 1), (3, 3), (4, 1), (4, 3)]
    assert s4cr2.CONSEQUENCE_WEIGHT_INTERACTION is s4co.WEIGHT_INTERACTION
    assert s4cr2.CONSEQUENCE_WEIGHT_EFFICIENCY is s4co.WEIGHT_EFFICIENCY
    assert s4cr2.INTERACTION_LONGITUDINAL_SCALE is s4co.INTERACTION_LONGITUDINAL_SCALE
    assert s4cr2.INTERACTION_LATERAL_SCALE is s4co.INTERACTION_LATERAL_SCALE
    assert s4cr2.REGRET_CLAMP_TOL is s4co.REGRET_CLAMP_TOL
    assert s4cr2.HIGH_REGRET_THRESHOLD == 0.01
    assert s4cr2.LAMBDA_CLS == 0.25


def test_direct_param_count_exact():
    flat = s4cr2.N_COMPETITORS * s4cr2.TOKEN_DIM + s4cr2.GLOBAL_DIM
    assert flat == 315 == s4cr2.DIRECT_FLAT_DIM
    params = (
        flat * 43 + 43      # Linear(315, 43)
        + 2 * 43            # LayerNorm(43)
        + 43 * 43 + 43      # Linear(43, 43)
        + 2 * 43            # LayerNorm(43)
        + 43 * 2 + 2        # Linear(43, 2)
    )
    assert params == 15740 == s4cr2.EXPECTED_TRAINABLE_PARAMETERS


# --- seed schedule + (c) fixed role segments ------------------------------------------


def test_seed_schedule_contract():
    schedule = s4cr2.build_seed_schedule(_config())
    assert schedule.size == 60000
    assert np.unique(schedule).size == schedule.size
    assert schedule.min() >= s4cr2.SEED_LOW
    assert schedule.max() <= s4cr2.SEED_HIGH_INCLUSIVE
    assert s4cr2.DEV_SEED_ENTROPY == 2026092880
    # only the frozen (entropy, size) contract is legal.
    with pytest.raises(ValueError):
        s4cr2.make_group_seed_schedule(2026092701, 60000)  # v0.1 entropy, rejected
    with pytest.raises(ValueError):
        s4cr2.make_group_seed_schedule(2026092880, 30000)
    with pytest.raises(ValueError):
        s4cr2.make_group_seed_schedule(123, 100)


def test_role_segments_fixed_and_disjoint():
    segs = {r: s4cr2.role_segment(r) for r in
            ("development_train", "development_validation", "development_screen")}
    assert segs["development_train"] == (0, 24000)
    assert segs["development_validation"] == (24000, 32000)
    assert segs["development_screen"] == (32000, 60000)
    covered = set()
    for start, stop in segs.values():
        covered.update(range(start, stop))
    assert covered == set(range(s4cr2.DEV_SCHEDULE_SIZE))
    # roles are fixed schedule slices; there is no reassignment API and unknown
    # roles (including any confirm role) are refused.
    with pytest.raises(ValueError):
        s4cr2.role_segment("confirm")
    with pytest.raises(ValueError):
        s4cr2.role_segment("development_selection")
    cfg = _config()
    assert [int(x) for x in cfg["train_segment"]] == [0, 24000]
    assert [int(x) for x in cfg["screen_segment"]] == [32000, 60000]


# --- (a)/(b) forbidden sources ---------------------------------------------------------


R2_ROLES = ("development_train", "development_validation", "development_screen", "power")


@pytest.mark.parametrize("role", R2_ROLES)
@pytest.mark.parametrize("path", [
    "experiments/stage4_reliability/stage4c/development/predictions_development_screen.npz",
    "experiments/stage4_reliability/stage4c/development/risk_scores_development_screen.npz",
    "experiments/stage4_reliability/stage4c/development/screen_group_quantities.npz",
    "experiments/stage4_reliability/stage4c/development/development_screen_report.json",
    "data/object_centric_stage/stage4c_merge_multiaction_dev/"
    "object_twinworld_stage4c_merge_multiaction_dev_screen.npz",
])
def test_stage4c_screen_refused_for_all_r2_roles(path, role):
    with pytest.raises(ValueError):
        s4cr2.assert_allowed_stage4cr2_data(path, role=role)


@pytest.mark.parametrize("role", R2_ROLES)
@pytest.mark.parametrize("path", [
    "experiments/stage4_reliability/stage4c_plus/phase0/phase0_report.json",
    "experiments/stage4_reliability/stage4c_plus/phase0/oof_predictions.npz",
    "experiments/stage4_reliability/stage4c_plus/phase0/fold_assignment.npy",
    "data/object_centric_stage/stage4cp_merge_multiaction_dev/x.npz",
    # old 500-group selection: historical design evidence only.
    "data/object_centric_stage/stage4c_merge_multiaction_dev/"
    "object_twinworld_stage4c_merge_multiaction_dev_selection.npz",
    "experiments/stage4_reliability/stage4c/development/decision_rule_receipt.json",
])
def test_phase0_and_old_selection_refused_for_r2_roles(path, role):
    with pytest.raises(ValueError):
        s4cr2.assert_allowed_stage4cr2_data(path, role=role)


def test_forbidden_sha256_layer_refuses_renamed_content():
    # A forbidden artifact stays forbidden by content hash even under an R2-looking path.
    phase0 = ROOT / ("experiments/stage4_reliability/stage4c_plus/phase0/"
                     "phase0_report.json")
    digest = hashlib.sha256(phase0.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        s4cr2.assert_allowed_stage4cr2_data(
            "data/object_centric_stage/stage4cr2_merge_multiaction_dev/smuggled.npz",
            role="development_train", file_sha256=digest)
    # Stage 4B confirm truth stays refused.
    with pytest.raises(ValueError):
        s4cr2.assert_allowed_stage4cr2_data(
            "experiments/stage4_reliability/stage4b/confirm/x.npz",
            role="development_train")
    # R2's own purpose dir with clean content is allowed.
    s4cr2.assert_allowed_stage4cr2_data(
        "data/object_centric_stage/stage4cr2_merge_multiaction_dev/"
        "object_twinworld_stage4cr2_train.npz", role="development_train")
    with pytest.raises(ValueError):
        s4cr2.assert_allowed_stage4cr2_data("data/somewhere_else/x.npz")
    with pytest.raises(ValueError):
        s4cr2.assert_allowed_stage4cr2_data(
            "data/object_centric_stage/stage4cr2_merge_multiaction_dev/x.npz",
            role="confirm")


# --- (d) global accepted+reserved seed zero-overlap audit -------------------------------


def test_global_seed_audit_zero_overlap_real_workspace():
    schedule = s4cr2.build_seed_schedule(_config())
    audit = s4cr2.audit_global_seed_overlap(
        schedule, [ROOT / "data", ROOT / "experiments"])
    assert audit.n_schedule == 60000
    assert audit.n_inventory > 200000
    assert audit.zero_overlap, f"overlap detected: {audit.overlap[:10]}"
    s4cr2.assert_seed_audit_clean(audit)
    # explicit disjointness from the contract-reserved prior-stage schedules.
    cp_dev = s4cp.make_group_seed_schedule(s4cp.DEV_SEED_ENTROPY, s4cp.DEV_SCHEDULE_SIZE)
    cp_conf = s4cp.make_group_seed_schedule(
        s4cp.CONFIRM_SEED_ENTROPY, s4cp.CONFIRM_SCHEDULE_SIZE)
    c_conf = s4c.make_group_seed_schedule(
        s4c.CONFIRM_SEED_ENTROPY, s4c.CONFIRM_SCHEDULE_SIZE)
    assert np.intersect1d(schedule, cp_dev).size == 0
    assert np.intersect1d(schedule, cp_conf).size == 0
    assert np.intersect1d(schedule, c_conf).size == 0


def test_global_seed_audit_detects_overlap(tmp_path):
    schedule = s4cr2.build_seed_schedule(_config())
    colliding = int(schedule[123])
    fake = tmp_path / "hist"
    fake.mkdir()
    np.savez(fake / "old.npz", episode_seed=np.array([colliding, 999999]))
    audit = s4cr2.audit_global_seed_overlap(schedule, [fake])
    assert not audit.zero_overlap
    assert colliding in audit.overlap
    with pytest.raises(ValueError):
        s4cr2.assert_seed_audit_clean(audit)


# --- (e) development lock create-once + tamper detection --------------------------------


def _tmp_config(tmp_path: Path) -> dict:
    cfg = _config()
    cfg["run_dir"] = str(tmp_path / "run")
    return cfg


def _tmp_inventory(tmp_path: Path) -> dict:
    source = tmp_path / "frozen_source.bin"
    source.write_bytes(b"frozen-bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"sample_frozen": {"path": str(source), "sha256": digest}}


def test_development_lock_create_once_and_verify(tmp_path):
    cfg = _tmp_config(tmp_path)
    inventory = _tmp_inventory(tmp_path)
    lock_path = s4cr2.create_development_lock(cfg, inventory)
    assert lock_path.is_file()
    assert lock_path.name == "protocol_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["plan_id"] == s4cr2.PLAN_ID
    assert payload["dev_seed_entropy"] == 2026092880
    assert payload["seed_audit"]["zero_overlap"] is True
    assert payload["train_segment"] == [0, 24000]
    assert payload["screen_segment"] == [32000, 60000]
    assert (lock_path.parent / "seed_schedule.npy").is_file()
    assert (lock_path.parent / "global_seed_audit.json").is_file()
    # create-once: a second attempt is refused.
    with pytest.raises(FileExistsError):
        s4cr2.create_development_lock(cfg, inventory)
    # clean verify passes.
    s4cr2.verify_development_lock(lock_path)
    # tampering with a recorded frozen source is detected.
    (tmp_path / "frozen_source.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        s4cr2.verify_development_lock(lock_path)


def test_verify_development_lock_detects_lock_tampering(tmp_path):
    cfg = _tmp_config(tmp_path)
    inventory = _tmp_inventory(tmp_path)
    lock_path = s4cr2.create_development_lock(cfg, inventory)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["dev_seed_entropy"] = 2026092701  # drift back to the rejected v0.1 entropy
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    with pytest.raises(ValueError):
        s4cr2.verify_development_lock(lock_path)
