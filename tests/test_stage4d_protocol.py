"""Blocking tests for the Stage 4D D0/D1 protocol layer (plan §2, §4, §6, §7, §11).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0``. These tests prove the new-stage
protocol rejects source hash drift, Direct parameter drift, profile parameter drift,
foreign seed contracts and any seed overlap, and that the code-inventory / lock
verification primitives fail closed BEFORE anything is locked.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage2_protocol as s2  # noqa: E402
from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_confirm_protocol as confirm  # noqa: E402
from ac_twin import stage4d_ood_env as ood  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CONFIG_PATH = prot.CONFIG_PATH
R2_DEV = ROOT / "experiments/stage4_reliability/stage4c_r2/development"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# --- config validation ---------------------------------------------------------------


def test_real_config_validates():
    checked = prot.validate_config(_config())
    assert checked["plan_id"] == prot.PLAN_ID
    assert checked["stage_namespace"] == "stage4d_ood"
    assert checked["env_id"] == "merge-v0"
    assert checked["n_actions"] == 10
    assert checked["primary_baseline"] == "SEED_VOTE_ENTROPY"


@pytest.mark.parametrize("key,value", [
    ("plan_id", "AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2"),
    ("env_id", "highway-v0"),
    ("n_actions", 9),
    ("horizon", 21),
    ("history_len", 9),
    ("factual_suffix_steps", 11),
    # Direct contract drift (frozen from R2; §3.2).
    ("model_seeds", [8101, 8202, 8303, 8404, 8506]),
    ("expected_trainable_parameters", 15741),
    ("direct_hidden_width", 44),
    ("dropout", 0.2),
    ("n_competitors", 8),
    ("token_dim", 35),
    ("global_dim", 8),
    # baseline / primary score drift.
    ("primary_baseline", "MARGIN_ONLY"),
    ("coverage", 0.75),
    # seed domain drift (§6.2).
    ("d1_candidate_seed_entropy", 2026110995),
    ("d1_candidate_schedule_size", 30001),
    ("pilot_behavior_seed_entropy", 2026101001),
    ("bootstrap_resamples", 5000),
    ("d1_bootstrap_entropy", 2026110994),
    # power contract drift (§6.1).
    ("power_selection_retention", 0.75),
    ("power_target", 0.9),
    ("power_monte_carlo", 5000),
    ("action_plans", [[0, 1]]),
])
def test_config_drift_rejected(key, value):
    cfg = _config()
    cfg[key] = value
    with pytest.raises(ValueError):
        prot.validate_config(cfg)


def test_config_rejects_retrain_keys():
    # D1 trains/refits nothing (Global Constraint 2); any training key is forbidden.
    for key in ("learning_rate", "max_epochs", "optimizer", "retrain",
                "fit_normalization", "early_stop_patience"):
        cfg = _config()
        cfg[key] = 123
        with pytest.raises(ValueError):
            prot.validate_config(cfg)


def test_config_profile_param_drift_rejected():
    # plan §4 profile parameters are frozen; a drifted value must fail validation.
    cfg = _config()
    cfg["ood_profiles"]["behavior"]["aggressive"]["DISTANCE_WANTED"] = 6.0
    with pytest.raises(ValueError):
        prot.validate_config(cfg)
    cfg = _config()
    cfg["ood_profiles"]["behavior"]["cautious"]["TIME_WANTED"] = 2.0
    with pytest.raises(ValueError):
        prot.validate_config(cfg)
    cfg = _config()
    cfg["ood_profiles"]["geometry"]["highway_positions_base"] = [80.0, 58.0, 12.0]
    with pytest.raises(ValueError):
        prot.validate_config(cfg)
    cfg = _config()
    cfg["ood_profiles"]["geometry"]["merging_target_speed"] = 31.0
    with pytest.raises(ValueError):
        prot.validate_config(cfg)
    cfg = _config()
    cfg["ood_profiles"]["selection_priority"] = ["BEHAVIOR_OOD", "JOINT_OOD", "GEOMETRY_OOD"]
    with pytest.raises(ValueError):
        prot.validate_config(cfg)


def test_config_r2_pinned_sources_drift_rejected():
    cfg = _config()
    cfg["r2_pinned_sources"]["protocol_lock.json"] = "0" * 64
    with pytest.raises(ValueError):
        prot.validate_config(cfg)


def test_fixed_task_contract_reused_bitwise():
    assert prot.ACTION_PLANS is s4co.ACTION_PLANS
    assert prot.N_ACTIONS == 10 == len(prot.ACTION_PLANS)
    assert prot.ACTION_PLANS[prot.IDLE_PLAN_INDEX] == (1, 3)
    assert prot.ENV_ID == "merge-v0"
    assert prot.EXPECTED_TRAINABLE_PARAMETERS == s4cr2.EXPECTED_TRAINABLE_PARAMETERS == 15740
    assert list(prot.MODEL_SEEDS) == [8101, 8202, 8303, 8404, 8505]


# --- status / state machine (§7) --------------------------------------------------------


def test_terminal_states_verbatim():
    assert prot.STATUS_TASK_INSUFFICIENT == "STAGE4D_OOD_TASK_INSUFFICIENT"
    assert prot.STATUS_UNDERPOWERED == "STAGE4D_OOD_CONFIRMATION_UNDERPOWERED"
    assert prot.STATUS_TASK_DEGENERATE == "STAGE4D_OOD_TASK_DEGENERATE"
    assert prot.STATUS_MODEL_INSUFFICIENT == (
        "INCONCLUSIVE_STAGE4D_OOD_DECISION_MODEL_INSUFFICIENT")
    assert prot.STATUS_INVALID == "INCONCLUSIVE_STAGE4D_OOD_INVALID"
    assert prot.STATUS_NOT_CONFIRMED == "STAGE4D_OOD_ROBUSTNESS_NOT_CONFIRMED"
    assert prot.STATUS_CONFIRMED == "STAGE4D_OOD_ROBUSTNESS_CONFIRMED"
    assert len(prot.TERMINAL_STATES) == 7


def test_state_machine_transitions():
    assert prot.transition("APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION",
                           "SOFTWARE_IN_PROGRESS") == "SOFTWARE_IN_PROGRESS"
    assert prot.transition("SOFTWARE_IN_PROGRESS", "PREFLIGHT_PASSED") == "PREFLIGHT_PASSED"
    assert prot.transition("D0_PILOTS_COLLECTED", "D0_PROFILE_SELECTED") == "D0_PROFILE_SELECTED"
    # D0 with no qualified profile ends in the insufficient terminal state.
    assert prot.transition("D0_PILOTS_COLLECTED",
                           "STAGE4D_OOD_TASK_INSUFFICIENT") == "STAGE4D_OOD_TASK_INSUFFICIENT"
    # invalid / backwards transitions are refused.
    with pytest.raises(ValueError):
        prot.transition("SOFTWARE_IN_PROGRESS", "STAGE4D_OOD_ROBUSTNESS_CONFIRMED")
    with pytest.raises(ValueError):
        prot.transition("D0_PROFILE_SELECTED", "PREFLIGHT_PASSED")
    # terminal states are absorbing: no outgoing transitions.
    for terminal in prot.TERMINAL_STATES:
        with pytest.raises(ValueError):
            prot.transition(terminal, "SOFTWARE_IN_PROGRESS")


# --- candidate seed schedules (§6.2) ----------------------------------------------------


def test_candidate_schedules_match_approved_hashes():
    expected = {
        (2026100099, 256),
        (2026101000, 6000),
        (2026102001, 6000),
        (2026103002, 6000),
        (2026110994, 30000),
    }
    assert set(prot._APPROVED_SCHEDULES) == expected
    for (entropy, size), raw_sha in prot._APPROVED_SCHEDULES.items():
        schedule = prot.make_candidate_seed_schedule(entropy, size)
        assert schedule.size == size and schedule.dtype == np.int64
        assert np.unique(schedule).size == schedule.size
        assert int(schedule.min()) >= prot.SEED_LOW
        assert int(schedule.max()) <= prot.SEED_HIGH_INCLUSIVE
        assert hashlib.sha256(schedule.tobytes()).hexdigest() == raw_sha
        # generated via the frozen Stage 2 primitive EXACTLY.
        assert np.array_equal(schedule, s2.make_seed_schedule(entropy, size=size))


def test_candidate_schedule_maker_rejects_foreign_contracts():
    for entropy, size in (
        (2026092880, 60000),   # R2 v0.2 executed dev domain
        (2026093012, 12000),   # confirm executed domain
        (2026093100, 64),      # confirm smoke domain
        (2026110994, 29999),   # approved entropy, wrong size
        (2026100099, 30000),   # smoke entropy, wrong size
        (2026104101, 5000),    # classifier bootstrap entropy is NOT a schedule
        (2026110995, 10000),   # D1 bootstrap entropy is NOT a schedule
        (123, 100),
    ):
        with pytest.raises(ValueError):
            prot.make_candidate_seed_schedule(entropy, size)


def test_action_stream_roots_frozen():
    roots = prot.ACTION_STREAM_ROOTS
    assert roots["smoke"] == (2026100099, 0)
    assert roots["pilot_behavior"] == (2026101000, 1)
    assert roots["pilot_geometry"] == (2026102001, 2)
    assert roots["pilot_joint"] == (2026103002, 3)
    assert roots["d1_confirm"] == (2026110994, 4)
    cfg = _config()
    assert prot.action_stream_root(cfg, "d1_confirm") == (2026110994, 4)
    with pytest.raises(ValueError):
        prot.action_stream_root(cfg, "development_train")


def test_reserved_bootstrap_power_entropies_registered():
    # controller ruling 2: the D0 classifier bootstrap, D0 power and D1 bootstrap
    # entropies are reserved (registered) even though they are not collection schedules.
    reserved = prot.RESERVED_ENTROPIES
    assert reserved["domain_classifier_bootstrap"]["behavior"] == 2026104101
    assert reserved["domain_classifier_bootstrap"]["geometry"] == 2026104102
    assert reserved["domain_classifier_bootstrap"]["joint"] == 2026104103
    assert reserved["power"] == 2026104201
    assert reserved["d1_bootstrap"] == 2026110995


# --- global seed audit (§6.2): R2 and the confirm stage are history ---------------------


def test_global_seed_audit_zero_overlap_real_workspace():
    roots = [ROOT / "data", ROOT / "experiments"]
    inventory = prot.scan_historical_group_seeds(roots)
    # the inventory must contain the R2 executed domain and BOTH confirm domains —
    # omitting the confirm namespace is exactly the history-loss failure mode.
    assert np.isin(confirm.r2_reserved_seed_domain(), inventory).all()
    assert np.isin(confirm.make_candidate_seed_schedule(2026093012, 12000), inventory).all()
    assert np.isin(confirm.make_candidate_seed_schedule(2026093100, 64), inventory).all()
    for entropy, size in prot._APPROVED_SCHEDULES:
        schedule = prot.make_candidate_seed_schedule(entropy, size)
        audit = prot.audit_global_seed_overlap(schedule, roots, inventory=inventory)
        assert audit.zero_overlap, f"{(entropy, size)} overlaps: {audit.overlap[:5]}"
        prot.assert_seed_audit_clean(audit)


def test_audit_detects_overlap(tmp_path):
    schedule = prot.make_candidate_seed_schedule(2026101000, 6000)
    colliding = int(schedule[123])
    fake = tmp_path / "hist"
    fake.mkdir()
    np.savez(fake / "old.npz", group_seed=np.array([colliding, 999999]))
    audit = prot.audit_global_seed_overlap(schedule, [tmp_path])
    assert not audit.zero_overlap
    assert colliding in audit.overlap
    with pytest.raises(ValueError):
        prot.assert_seed_audit_clean(audit)


def test_own_namespace_excluded_only_for_resume(tmp_path):
    # this stage's own bound products are excluded so a resume does not self-collide.
    schedule = prot.make_candidate_seed_schedule(2026100099, 256)
    own = tmp_path / "experiments/stage4_reliability/stage4d_ood/protocol"
    own.mkdir(parents=True)
    np.save(own / "candidate_seed_schedule.npy", schedule)  # own namespace: excluded
    # a non-own historical product keeps the scan non-empty and is clean.
    hist = tmp_path / "data/object_centric_stage/some_prior_stage"
    hist.mkdir(parents=True)
    np.savez(hist / "accepted.npz", group_seed=np.array([424242, 555555]))
    audit = prot.audit_global_seed_overlap(schedule, [tmp_path])
    assert audit.zero_overlap  # own persisted schedule must not self-collide
    # the SAME seeds under a NON-own path WOULD be detected (no blanket stage4d pass).
    colliding = int(schedule[7])
    np.savez(hist / "collision.npz", group_seed=np.array([colliding]))
    audit2 = prot.audit_global_seed_overlap(schedule, [tmp_path])
    assert not audit2.zero_overlap and colliding in audit2.overlap


# --- §2.1 source chain verification -----------------------------------------------------


def test_verify_pinned_sources_real_workspace():
    sources = prot.verify_pinned_sources()
    assert len(sources) == 10
    for rel, entry in sources.items():
        assert entry["sha256"] == prot.STAGE4D_PINNED_SOURCES[rel]
        assert entry["status"] == "match"


def _copy_pinned_sources(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for rel in prot.STAGE4D_PINNED_SOURCES:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    return root


def test_verify_pinned_sources_detects_drift(tmp_path):
    root = _copy_pinned_sources(tmp_path)
    assert len(prot.verify_pinned_sources(root)) == 10
    # any single-byte drift in any pinned source fails closed.
    target = root / "experiments/stage4_reliability/stage4c_r2/development/feature_scaler.npz"
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        prot.verify_pinned_sources(root)


def test_verify_pinned_sources_missing_file_fails(tmp_path):
    root = _copy_pinned_sources(tmp_path)
    (root / "experiments/stage4_reliability/stage4cr2_confirm/reports/"
     "STAGE4CR2_CONFIRMATORY_DECISION.md").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        prot.verify_pinned_sources(root)


def test_build_source_inventory_covers_all_ten_with_binding():
    inventory = prot.build_source_inventory()
    assert inventory["plan_id"] == prot.PLAN_ID
    assert inventory["n_files"] == 10
    assert set(inventory["files"]) == set(prot.STAGE4D_PINNED_SOURCES)
    # the R2 train NPZ binding recorded by the pinned receipt must match plan §2.1.
    assert inventory["r2_train_npz_sha256"] == prot.R2_TRAIN_NPZ_SHA256


def test_r2_train_receipt_binding_verified():
    receipt = json.loads(
        (R2_DEV / "train_collection_receipt.json").read_text(encoding="utf-8"))
    assert receipt["data_output_sha256"] == prot.R2_TRAIN_NPZ_SHA256


# --- code inventory: missing member / drift fail closed ---------------------------------


def test_protocol_code_hashes_cover_registered_files():
    hashes = prot.protocol_code_hashes()
    for rel in prot.PROTOCOL_CODE_FILES:
        assert rel in hashes
        assert hashes[rel] == sha256_file(ROOT / "code" / rel)
    # both new stage4d modules are registered.
    assert "ac_twin/stage4d_protocol.py" in prot.PROTOCOL_CODE_FILES
    assert "ac_twin/stage4d_ood_env.py" in prot.PROTOCOL_CODE_FILES


def test_code_inventory_missing_member_fails(monkeypatch):
    # §6.3: no "file exists then hash"; a registered-but-missing member fails closed.
    monkeypatch.setattr(prot, "PROTOCOL_CODE_FILES",
                        tuple(prot.PROTOCOL_CODE_FILES) + ("ac_twin/_missing_probe.py",))
    with pytest.raises(FileNotFoundError):
        prot.protocol_code_hashes()


def test_code_hash_drift_detected(tmp_path, monkeypatch):
    probe = CODE / "_stage4d_drift_probe.py"
    probe.write_text("# drift probe v1\n", encoding="utf-8")
    monkeypatch.setattr(prot, "PROTOCOL_CODE_FILES",
                        tuple(prot.PROTOCOL_CODE_FILES) + ("_stage4d_drift_probe.py",))
    try:
        before = prot.protocol_code_hashes()
        probe.write_text("# drift probe v2 - modified\n", encoding="utf-8")
        after = prot.protocol_code_hashes()
        assert before["_stage4d_drift_probe.py"] != after["_stage4d_drift_probe.py"]
    finally:
        probe.unlink(missing_ok=True)


# --- protocol lock verification primitive (§6.3; the formal lock is Task 4) ---------------


def _make_lock_fixture(tmp_path: Path) -> Path:
    """A minimal self-consistent protocol-lock payload for the verifier primitive."""
    cfg = _config()
    payload = {
        "plan_id": prot.PLAN_ID,
        "stage_namespace": "stage4d_ood",
        "config_signature": prot._config_signature(cfg),
        "frozen_source_inventory": prot.verify_pinned_sources(),
        "code_sha256": prot.protocol_code_hashes(),
        "seed_audit": {"zero_overlap": True, "overlap": []},
        "terminal_states": list(prot.TERMINAL_STATES),
    }
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir(parents=True)
    lock_path = protocol_dir / "protocol_lock.json"
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return tmp_path


def test_verify_protocol_lock_missing_fails(tmp_path):
    # fail-closed before the formal lock exists (this is the real run_dir state now).
    with pytest.raises(FileNotFoundError):
        prot.verify_protocol_lock(tmp_path)


def test_verify_protocol_lock_accepts_valid(tmp_path):
    run_dir = _make_lock_fixture(tmp_path)
    verified = prot.verify_protocol_lock(run_dir)
    assert verified["plan_id"] == prot.PLAN_ID


def test_verify_protocol_lock_detects_source_drift(tmp_path):
    run_dir = _make_lock_fixture(tmp_path)
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    rel = next(iter(payload["frozen_source_inventory"]))
    payload["frozen_source_inventory"][rel]["sha256"] = "0" * 64
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    with pytest.raises(ValueError):
        prot.verify_protocol_lock(run_dir)


def test_verify_protocol_lock_detects_config_drift(tmp_path):
    run_dir = _make_lock_fixture(tmp_path)
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["config_signature"] = "f" * 64
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    with pytest.raises(ValueError):
        prot.verify_protocol_lock(run_dir)


def test_verify_protocol_lock_rejects_dirty_seed_audit(tmp_path):
    run_dir = _make_lock_fixture(tmp_path)
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["seed_audit"] = {"zero_overlap": False, "overlap": [123]}
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    with pytest.raises(ValueError):
        prot.verify_protocol_lock(run_dir)


# --- §7.1 condition-8 threshold derivation rule (controller ruling 1) --------------------


def test_r2_train_distance_threshold_rule_is_deterministic():
    # a fixed, never-changing rule: median of the per-group pre-intervention minimum
    # ego-to-other distance over the R2 train groups. Task 4 supplies the array.
    distances = np.array([10.0, 20.0, 30.0, 40.0])
    assert prot.derive_r2_train_distance_threshold(distances) == 25.0
    assert prot.derive_r2_train_distance_threshold(distances) == \
        prot.derive_r2_train_distance_threshold(distances[::-1])
    with pytest.raises(ValueError):
        prot.derive_r2_train_distance_threshold(np.array([]))
    assert prot.R2_TRAIN_DISTANCE_THRESHOLD_RULE == (
        "median_of_pre_intervention_min_ego_distance_over_r2_train_groups")
