"""Blocking tests for the Stage 4C-R2 independent-confirmation protocol layer.

Plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`` sections 2, 4.2, 5.3, 8, 10.
These tests prove the new-stage protocol rejects source/code drift, R2-history seed
omission, unsafe reads and any residual config drift BEFORE anything is locked.
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
from ac_twin import stage4c_protocol as s4c  # noqa: E402
from ac_twin import stage4cp_protocol as s4cp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_confirm_protocol as prot  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4cr2_confirm/stage4cr2_confirm_config.json")
R2_DEV = ROOT / "experiments/stage4_reliability/stage4c_r2/development"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# --- config validation ---------------------------------------------------------------


def test_real_config_validates():
    checked = prot.validate_config(_config())
    assert checked["plan_id"] == prot.PLAN_ID
    assert checked["confirm_accepted_groups"] == 2000
    assert checked["confirm_seed_entropy"] == 2026093012
    assert checked["confirm_schedule_size"] == 12000
    assert checked["smoke_seed_entropy"] == 2026093100
    assert checked["smoke_schedule_size"] == 64
    assert checked["bootstrap_entropy"] == 2026093202
    assert checked["constant_random_salt"] == 2026093203
    assert checked["bootstrap_resamples"] == 10000
    assert checked["confidence"] == 0.95


@pytest.mark.parametrize("key,value", [
    ("plan_id", "AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2"),
    ("confirm_seed_entropy", 2026092880),   # R2 v0.2 dev entropy: NOT reusable here
    ("confirm_seed_entropy", 2026092701),   # R2 v0.1 superseded entropy
    ("confirm_seed_entropy", 2026093013),
    ("confirm_schedule_size", 11999),
    ("confirm_accepted_groups", 2001),
    ("smoke_seed_entropy", 2026093012),
    ("smoke_schedule_size", 128),
    ("smoke_accepted_groups", 13),
    ("bootstrap_entropy", 2026092702),      # R2 bootstrap entropy: NOT reusable here
    ("constant_random_salt", 17),
    ("bootstrap_resamples", 1000),
    ("confidence", 0.8),
    ("n_actions", 9),
    ("horizon", 21),
    ("history_len", 9),
    ("model_seeds", [8101, 8202, 8303, 8404, 8506]),
    ("expected_trainable_parameters", 15741),
    ("direct_hidden_width", 44),
    ("dropout", 0.2),
    ("primary_baseline", "MARGIN_ONLY"),
    ("coverage", 0.75),
    ("interaction_threshold", 29.0),
    ("idle_plan_index", 2),                 # (IDLE,1) is NOT the IDLE reference
    ("t_star_stratum_split", 12),
    ("action_plans", [[0, 1]]),
    ("response_predictor_seeds", [101, 202, 303, 404, 506]),
    ("expert_router_seeds", [1101, 1202, 1303, 1404, 1506]),
])
def test_config_drift_rejected(key, value):
    cfg = _config()
    cfg[key] = value
    with pytest.raises(ValueError):
        prot.validate_config(cfg)


def test_config_rejects_retrain_keys():
    for key in ("learning_rate", "max_epochs", "optimizer", "retrain",
                "fit_normalization", "early_stop_patience"):
        cfg = _config()
        cfg[key] = 123
        with pytest.raises(ValueError):
            prot.validate_config(cfg)


def test_fixed_task_contract_reused_bitwise():
    assert prot.ACTION_PLANS is s4co.ACTION_PLANS
    assert prot.N_ACTIONS == 10 == len(prot.ACTION_PLANS)
    assert prot.HORIZON == 20 and prot.HISTORY_LEN == 10
    assert prot.ENV_ID == "merge-v0"
    assert s4co.ACTION_PLANS[prot.IDLE_PLAN_INDEX] == (1, 3)
    # the ten plans keep both IDLE durations distinct
    assert list(prot.ACTION_PLANS).count((1, 1)) == 1
    assert list(prot.ACTION_PLANS).count((1, 3)) == 1


def test_status_constants_verbatim():
    assert prot.STATUS_INVALID == "STAGE4CR2_CONFIRM_INVALID_ANALYSIS"
    assert prot.STATUS_COLLECTION_INCOMPLETE == "STAGE4CR2_CONFIRM_COLLECTION_INCOMPLETE"
    assert prot.STATUS_RESOURCE_AT_RISK == "STAGE4CR2_CONFIRM_RESOURCE_AT_RISK"
    assert prot.STATUS_TASK_DEGENERATE == "STAGE4CR2_CONFIRM_TASK_DEGENERATE"
    assert prot.STATUS_MODEL_INSUFFICIENT == (
        "INCONCLUSIVE_STAGE4CR2_CONFIRM_DECISION_MODEL_INSUFFICIENT")
    assert prot.STATUS_NOT_CONFIRMED == "STAGE4CR2_DECISION_RISK_NOT_CONFIRMED"
    assert prot.STATUS_CONFIRMED == "STAGE4CR2_DECISION_RISK_CONFIRMED"


# --- seed schedule contract (§4.2) -----------------------------------------------------


def test_confirm_and_smoke_schedules_match_approved_hashes():
    cfg = _config()
    confirm = prot.build_confirm_seed_schedule(cfg)
    smoke = prot.build_smoke_seed_schedule(cfg)
    assert confirm.size == 12000 and smoke.size == 64
    assert confirm.dtype == np.int64 and smoke.dtype == np.int64
    for schedule in (confirm, smoke):
        assert np.unique(schedule).size == schedule.size
        assert int(schedule.min()) >= prot.SEED_LOW
        assert int(schedule.max()) <= prot.SEED_HIGH_INCLUSIVE
    assert hashlib.sha256(confirm.tobytes()).hexdigest() == prot.CONFIRM_SCHEDULE_RAW_SHA256
    assert hashlib.sha256(smoke.tobytes()).hexdigest() == prot.SMOKE_SCHEDULE_RAW_SHA256
    # generated via the frozen Stage 2 primitive EXACTLY (never the R2/4C wrappers).
    assert np.array_equal(confirm, s2.make_seed_schedule(2026093012, size=12000))
    assert np.array_equal(smoke, s2.make_seed_schedule(2026093100, size=64))
    # the two proposed domains are mutually disjoint.
    assert np.intersect1d(confirm, smoke).size == 0


def test_candidate_schedule_maker_rejects_foreign_contracts():
    # the R2/4C wrappers refuse new entropy; our maker must refuse THEIRS (and any
    # other non-approved (entropy, size) pair) rather than silently generating one.
    for entropy, size in ((2026092880, 60000),      # R2 v0.2 executed dev domain
                          (2026092701, 60000),      # R2 v0.1 superseded domain
                          (2026091801, 5000),       # Stage 4C dev
                          (2026091901, 15000),      # Stage 4C confirm
                          (2026093012, 11999),      # approved entropy, wrong size
                          (2026093100, 12000)):
        with pytest.raises(ValueError):
            prot.make_candidate_seed_schedule(entropy, size)


def test_r2_reserved_domain_resolved_from_executed_lock():
    domain = prot.r2_reserved_seed_domain()
    executed = s2.make_seed_schedule(2026092880, size=60000)  # v0.2 EXECUTED config
    superseded = s2.make_seed_schedule(2026092701, size=60000)  # v0.1, never executed
    assert np.array_equal(domain, executed)
    assert not np.array_equal(domain, superseded)


def test_r2_reserved_domain_fails_loud_on_tampered_lock(tmp_path):
    fake = tmp_path / "r2dev"
    fake.mkdir()
    shutil.copy2(R2_DEV / "protocol_lock.json", fake / "protocol_lock.json")
    shutil.copy2(R2_DEV / "seed_schedule.npy", fake / "seed_schedule.npy")
    # tampered lock content no longer matches the pinned SHA256.
    payload = json.loads((fake / "protocol_lock.json").read_text(encoding="utf-8"))
    payload["dev_seed_entropy"] = 2026092701
    (fake / "protocol_lock.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        prot.r2_reserved_seed_domain(fake)
    # untampered copy resolves fine.
    shutil.copy2(R2_DEV / "protocol_lock.json", fake / "protocol_lock.json")
    domain = prot.r2_reserved_seed_domain(fake)
    assert domain.size == 60000
    # tampered persisted schedule disagrees with the regenerated domain.
    schedule = np.load(fake / "seed_schedule.npy")
    schedule[0] = 1 if schedule[0] != 1 else 2
    np.save(fake / "seed_schedule.npy", schedule)
    with pytest.raises(ValueError):
        prot.r2_reserved_seed_domain(fake)


# --- global seed audit (§4.2): R2 is now history --------------------------------------


def test_global_seed_audit_zero_overlap_real_workspace():
    cfg = _config()
    confirm = prot.build_confirm_seed_schedule(cfg)
    smoke = prot.build_smoke_seed_schedule(cfg)
    roots = [ROOT / "data", ROOT / "experiments"]
    audit_confirm = prot.audit_global_seed_overlap(confirm, roots)
    audit_smoke = prot.audit_global_seed_overlap(smoke, roots)
    assert audit_confirm.zero_overlap and audit_smoke.zero_overlap
    prot.assert_seed_audit_clean(audit_confirm)
    prot.assert_seed_audit_clean(audit_smoke)
    # the inventory must include R2's FULL executed 60,000-seed reserved domain —
    # an omission here is exactly the R2-history failure mode this stage must avoid.
    inventory = prot.scan_historical_group_seeds(roots)
    r2_domain = prot.r2_reserved_seed_domain()
    assert np.isin(r2_domain, inventory).all()
    # and the explicitly registered Stage 2B legacy (stale, never-persisted) domains.
    for domain in prot.legacy_stage2b_candidate_domains().values():
        assert np.isin(domain, inventory).all()
    # pinned at the authoring-time synced composition: 420,762 from the plan precheck
    # plus the 42 R2-namespace accepted/ledger seeds the old R2 scanner excluded and
    # this stage must now count as history. Any drift fails loudly before lock.
    assert inventory.size == 420804


def test_audit_detects_overlap_in_r2_namespaced_history(tmp_path):
    """R2-history omission: a colliding seed under an R2 path MUST be detected."""
    cfg = _config()
    confirm = prot.build_confirm_seed_schedule(cfg)
    colliding = int(confirm[123])
    fake = tmp_path / "experiments/stage4_reliability/stage4c_r2/development"
    fake.mkdir(parents=True)
    np.savez(fake / "screen_predictions.npz", group_seed=np.array([colliding, 999999]))
    audit = prot.audit_global_seed_overlap(confirm, [tmp_path])
    assert not audit.zero_overlap
    assert colliding in audit.overlap
    with pytest.raises(ValueError):
        prot.assert_seed_audit_clean(audit)


def test_audit_detects_overlap_in_legacy_stage4c_path(tmp_path):
    """No blanket 'stage4c' string exclusion: legacy Stage 4C paths are history too."""
    cfg = _config()
    confirm = prot.build_confirm_seed_schedule(cfg)
    colliding = int(confirm[7])
    fake = tmp_path / "experiments/stage4_reliability/stage4c/development"
    fake.mkdir(parents=True)
    np.save(fake / "seed_schedule_development_screen.npy", np.array([colliding]))
    audit = prot.audit_global_seed_overlap(confirm, [tmp_path])
    assert not audit.zero_overlap
    assert colliding in audit.overlap


def test_own_namespace_excluded_only_for_resume(tmp_path):
    """This stage's own bound products are excluded so a resume does not self-collide."""
    cfg = _config()
    confirm = prot.build_confirm_seed_schedule(cfg)
    own = tmp_path / "experiments/stage4_reliability/stage4cr2_confirm/protocol"
    own.mkdir(parents=True)
    np.save(own / "candidate_seed_schedule.npy", confirm)
    audit = prot.audit_global_seed_overlap(confirm, [tmp_path])
    assert audit.zero_overlap  # own schedule must not self-collide


def test_scalar_seed_fields_collected(tmp_path):
    """A scalar int under an exact seed key (e.g. "group_seed": 123) is history too."""
    cfg = _config()
    confirm = prot.build_confirm_seed_schedule(cfg)
    colliding = int(confirm[55])
    fake = tmp_path / "experiments/stage4_reliability/stage4c/development"
    fake.mkdir(parents=True)
    (fake / "receipt.json").write_text(json.dumps({
        "group_seed": colliding, "episode_seed": 424242,
        "nested": {"group_seed": 777777}}), encoding="utf-8")
    found = prot._seeds_in_json_strict(fake / "receipt.json")
    assert {colliding, 424242, 777777} <= found
    audit = prot.audit_global_seed_overlap(confirm, [tmp_path])
    assert not audit.zero_overlap
    assert colliding in audit.overlap


def test_scalar_seed_metadata_not_collected(tmp_path):
    """Count/size metadata under seed-hinted names is NOT a seed value.

    `seed_schedule_size`, `n_historical_episode_seeds`, `unique_episode_seeds` etc.
    are integer metadata; collecting them would pollute the audit with sizes.
    """
    fake = tmp_path / "experiments/stage4_reliability/stage4c/development"
    fake.mkdir(parents=True)
    (fake / "protocol_lock.json").write_text(json.dumps({
        "seed_schedule_size": 60000,
        "n_historical_episode_seeds": 29224,
        "unique_episode_seeds": 5000,
        "seed_entropy": 2026092880,
        "seed_schedule": [111, 222]}), encoding="utf-8")
    found = prot._seeds_in_json_strict(fake / "protocol_lock.json")
    assert found == {111, 222}  # the list schedule only; no scalar metadata


def test_scan_read_failure_fails_loudly(tmp_path):
    """No `except: continue`: an unreadable historical product fails the scan."""
    bad = tmp_path / "data/object_centric_stage/stage4c_merge_multiaction_dev"
    bad.mkdir(parents=True)
    (bad / "corrupt.npz").write_bytes(b"not-an-archive")
    with pytest.raises(ValueError):
        prot.scan_historical_group_seeds([tmp_path])
    bad.joinpath("corrupt.npz").unlink()
    (bad / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        prot.scan_historical_group_seeds([tmp_path])


def test_scan_empty_workspace_fails(tmp_path):
    with pytest.raises(ValueError):
        prot.scan_historical_group_seeds([tmp_path])


# --- §2.1 R2 authoritative source hash verification ------------------------------------


def test_verify_r2_sources_real_workspace():
    sources = prot.verify_r2_sources()
    assert len(sources["pinned_files"]) == 10
    for name, entry in sources["pinned_files"].items():
        assert entry["sha256"] == prot.R2_PINNED_SOURCES[name]
    assert sorted(sources["checkpoints"]) == ["8101", "8202", "8303", "8404", "8505"]
    for seed, entry in sources["checkpoints"].items():
        assert entry["sha256"] == entry["final_selection_sha256"]
        assert entry["sha256"] == entry["model_inventory_sha256"]


def _copy_r2_dev_subset(tmp_path: Path) -> Path:
    """Copy the pinned R2 files + five Direct checkpoints into a tmp dev dir."""
    fake = tmp_path / "r2dev"
    (fake / "models").mkdir(parents=True)
    for name in prot.R2_PINNED_SOURCES:
        shutil.copy2(R2_DEV / name, fake / name)
    for seed in prot.MODEL_SEEDS:
        shutil.copy2(R2_DEV / "models" / f"direct_seed{seed}.pt",
                     fake / "models" / f"direct_seed{seed}.pt")
    return fake


def test_verify_r2_sources_detects_source_drift(tmp_path):
    fake = _copy_r2_dev_subset(tmp_path)
    assert prot.verify_r2_sources(fake)["pinned_files"]["final_selection.json"]
    # any single-byte drift in any pinned source must fail preflight loudly.
    target = fake / "feature_scaler.npz"
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        prot.verify_r2_sources(fake)


def test_verify_r2_sources_detects_checkpoint_drift(tmp_path):
    fake = _copy_r2_dev_subset(tmp_path)
    target = fake / "models" / "direct_seed8303.pt"
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        prot.verify_r2_sources(fake)


def test_verify_r2_sources_missing_file_fails(tmp_path):
    fake = _copy_r2_dev_subset(tmp_path)
    (fake / "validation_gate_report.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        prot.verify_r2_sources(fake)


# --- preflight -------------------------------------------------------------------------


def test_preflight_real_workspace_passes():
    report = prot.preflight(_config())
    assert report["status"] == prot.STATUS_PREFLIGHT_PASSED
    assert report["seed_audit"]["confirm"]["zero_overlap"] is True
    assert report["seed_audit"]["smoke"]["zero_overlap"] is True
    assert report["seed_audit"]["mutual_overlap"] == []
    assert report["confirm_schedule_raw_sha256"] == prot.CONFIRM_SCHEDULE_RAW_SHA256
    assert report["smoke_schedule_raw_sha256"] == prot.SMOKE_SCHEDULE_RAW_SHA256


def test_preflight_fails_on_r2_source_drift(tmp_path):
    cfg = _config()
    fake = _copy_r2_dev_subset(tmp_path)
    target = fake / "screen_group_quantities.npz"
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF
    target.write_bytes(bytes(blob))
    cfg["r2_development_dir"] = str(fake)
    with pytest.raises(ValueError):
        prot.preflight(cfg)


# --- protocol lock (create-once + tamper detection) -------------------------------------


def _tmp_config(tmp_path: Path) -> dict:
    cfg = _config()
    run_dir = tmp_path / "run"
    cfg["run_dir"] = str(run_dir)
    for key, value in cfg["paths"].items():
        if value.startswith("experiments/stage4_reliability/stage4cr2_confirm/"):
            rel = value.removeprefix("experiments/stage4_reliability/stage4cr2_confirm/")
            cfg["paths"][key] = str(run_dir / rel)
    return cfg


def test_protocol_lock_create_once_and_verify(tmp_path):
    cfg = _tmp_config(tmp_path)
    lock_path = Path(prot.create_protocol_lock(cfg))
    assert lock_path.is_file()
    assert lock_path.name == "protocol_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["plan_id"] == prot.PLAN_ID
    assert payload["confirm_seed_entropy"] == 2026093012
    assert payload["confirm_schedule_raw_sha256"] == prot.CONFIRM_SCHEDULE_RAW_SHA256
    assert payload["smoke_schedule_raw_sha256"] == prot.SMOKE_SCHEDULE_RAW_SHA256
    assert payload["seed_audit"]["confirm"]["zero_overlap"] is True
    assert payload["seed_audit"]["smoke"]["zero_overlap"] is True
    assert payload["model_seeds"] == [8101, 8202, 8303, 8404, 8505]
    assert payload["response_predictor_seeds"] == [101, 202, 303, 404, 505]
    assert payload["expert_router_seeds"] == [1101, 1202, 1303, 1404, 1505]
    assert payload["primary_baseline"] == "SEED_VOTE_ENTROPY"
    assert payload["n_confirm_groups"] == 2000
    protocol_dir = lock_path.parent
    assert (protocol_dir / "candidate_seed_schedule.npy").is_file()
    assert (protocol_dir / "seed_audit.json").is_file()
    assert (protocol_dir / "source_inventory.json").is_file()
    assert (protocol_dir / "code_inventory.json").is_file()
    assert (protocol_dir / "historical_seed_inventory.json").is_file()
    # the persisted schedule is exactly the approved confirm domain.
    persisted = np.load(protocol_dir / "candidate_seed_schedule.npy")
    assert np.array_equal(persisted, prot.build_confirm_seed_schedule(cfg))
    # create-once.
    with pytest.raises(FileExistsError):
        prot.create_protocol_lock(cfg)
    # clean verify passes.
    verified = prot.verify_protocol(str(lock_path))
    assert verified["plan_id"] == prot.PLAN_ID


def test_verify_protocol_detects_lock_tampering(tmp_path):
    cfg = _tmp_config(tmp_path)
    lock_path = Path(prot.create_protocol_lock(cfg))
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["confirm_seed_entropy"] = 2026092880  # drift toward R2's domain
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    with pytest.raises(ValueError):
        prot.verify_protocol(str(lock_path))


def test_verify_protocol_detects_code_drift(tmp_path, monkeypatch):
    probe = CODE / "_stage4cr2_confirm_drift_probe.py"
    probe.write_text("# drift probe v1\n", encoding="utf-8")
    monkeypatch.setattr(prot, "PROTOCOL_CODE_FILES",
                        tuple(prot.PROTOCOL_CODE_FILES) + (probe.name,))
    try:
        cfg = _tmp_config(tmp_path)
        lock_path = Path(prot.create_protocol_lock(cfg))
        prot.verify_protocol(str(lock_path))
        probe.write_text("# drift probe v2 - modified after lock\n", encoding="utf-8")
        with pytest.raises(ValueError):
            prot.verify_protocol(str(lock_path))
    finally:
        probe.unlink(missing_ok=True)


def test_verify_protocol_detects_schedule_drift(tmp_path):
    cfg = _tmp_config(tmp_path)
    lock_path = Path(prot.create_protocol_lock(cfg))
    schedule = np.load(lock_path.parent / "candidate_seed_schedule.npy")
    schedule[0] = 1 if schedule[0] != 1 else 2
    np.save(lock_path.parent / "candidate_seed_schedule.npy", schedule)
    with pytest.raises(ValueError):
        prot.verify_protocol(str(lock_path))


# --- recovery-lock chain (plan section 8: keep the old lock, append immutable repairs) --


def _drift_probe_lock(tmp_path, monkeypatch):
    """A tmp protocol lock binding an extra probe code file (repair-chain testbed)."""
    probe = CODE / "_stage4cr2_confirm_recovery_probe.py"
    probe.write_text("# recovery probe v1\n", encoding="utf-8")
    monkeypatch.setattr(prot, "PROTOCOL_CODE_FILES",
                        tuple(prot.PROTOCOL_CODE_FILES) + (probe.name,))
    cfg = _tmp_config(tmp_path)
    lock_path = Path(prot.create_protocol_lock(cfg))
    return cfg, lock_path, probe


def _write_recovery(lock_path, changed, *, binds=None, name="recovery_lock_r1.json",
                    kind="pure_engineering_recovery", declared=None):
    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    code = dict(payload["code_sha256"])
    for rel, record in changed.items():
        code[rel] = record["after_sha256"]
    recovery = {
        "plan_id": prot.PLAN_ID,
        "kind": kind,
        "binds_previous_lock_sha256": binds or sha256_file(lock_path),
        "changed_files": changed,
        "code_sha256": declared if declared is not None else code,
    }
    path = Path(lock_path).parent / name
    path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def test_recovery_chain_accepts_valid_repair(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        prot.verify_protocol(str(lock_path))
        v1 = sha256_file(probe)
        probe.write_text("# recovery probe v2 - repaired after lock\n", encoding="utf-8")
        v2 = sha256_file(probe)
        # code drift without a recovery lock fails closed.
        with pytest.raises(ValueError):
            prot.verify_protocol(str(lock_path))
        # the immutable recovery lock binds the old lock's hash and the repair.
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": v1, "after_sha256": v2}})
        verified = prot.verify_protocol(str(lock_path))
        assert verified["plan_id"] == prot.PLAN_ID
        assert prot.resolve_locked_code_hashes(lock_path)[probe.name] == v2
        # the original lock file is untouched: it still binds the v1 hash.
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["code_sha256"][probe.name] == v1
    finally:
        probe.unlink(missing_ok=True)


def test_recovery_chain_refuses_tampered_binding(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        v1 = sha256_file(probe)
        probe.write_text("# recovery probe v2\n", encoding="utf-8")
        v2 = sha256_file(probe)
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": v1, "after_sha256": v2}},
                        binds="f" * 64)
        with pytest.raises(ValueError):
            prot.resolve_locked_code_hashes(lock_path)
        with pytest.raises(ValueError):
            prot.verify_protocol(str(lock_path))
    finally:
        probe.unlink(missing_ok=True)


def test_recovery_chain_refuses_after_hash_mismatch(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        v1 = sha256_file(probe)
        probe.write_text("# recovery probe v2\n", encoding="utf-8")
        # the recovery lock's after state does not match the current code.
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": v1, "after_sha256": "1" * 64}})
        with pytest.raises(ValueError):
            prot.verify_protocol(str(lock_path))
    finally:
        probe.unlink(missing_ok=True)


def test_recovery_chain_refuses_before_hash_mismatch(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        probe.write_text("# recovery probe v2\n", encoding="utf-8")
        v2 = sha256_file(probe)
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": "1" * 64, "after_sha256": v2}})
        with pytest.raises(ValueError):
            prot.resolve_locked_code_hashes(lock_path)
    finally:
        probe.unlink(missing_ok=True)


def test_recovery_chain_refuses_frozen_file_rebind(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        frozen_rel = prot.REUSED_FROZEN_CODE_FILES[0]
        _write_recovery(lock_path, {frozen_rel: {
            "before_sha256": payload["code_sha256"][frozen_rel],
            "after_sha256": "2" * 64}})
        with pytest.raises(ValueError):
            prot.resolve_locked_code_hashes(lock_path)
    finally:
        probe.unlink(missing_ok=True)


def test_recovery_chain_refuses_unregistered_or_noop_or_wrong_kind(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        v1 = sha256_file(probe)
        # a file the original lock never registered can never be re-bound.
        _write_recovery(lock_path, {"ac_twin/not_in_lock.py":
                                    {"before_sha256": "1" * 64, "after_sha256": "2" * 64}})
        with pytest.raises(ValueError):
            prot.resolve_locked_code_hashes(lock_path)
        # no-op change.
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": v1, "after_sha256": v1}})
        with pytest.raises(ValueError):
            prot.resolve_locked_code_hashes(lock_path)
        # wrong kind.
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": v1, "after_sha256": "3" * 64}},
                        kind="silent_rewrite")
        with pytest.raises(ValueError):
            prot.resolve_locked_code_hashes(lock_path)
    finally:
        probe.unlink(missing_ok=True)


def test_recovery_chain_must_be_contiguous_and_the_root_live(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        v1 = sha256_file(probe)
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": v1, "after_sha256": "4" * 64}},
                        name="recovery_lock_r2.json")  # gap: no r1
        with pytest.raises(ValueError):
            prot.recovery_lock_paths(lock_path)
        # a deleted original lock fails closed.
        with pytest.raises(FileNotFoundError):
            prot.resolve_locked_code_hashes(lock_path.parent / "missing_lock.json")
    finally:
        probe.unlink(missing_ok=True)


def test_recovery_chain_refuses_inconsistent_declared_code_set(tmp_path, monkeypatch):
    _cfg, lock_path, probe = _drift_probe_lock(tmp_path, monkeypatch)
    try:
        v1 = sha256_file(probe)
        _write_recovery(lock_path,
                        {probe.name: {"before_sha256": v1, "after_sha256": "5" * 64}},
                        declared={})  # declared set disagrees with changed_files
        with pytest.raises(ValueError):
            prot.resolve_locked_code_hashes(lock_path)
    finally:
        probe.unlink(missing_ok=True)


def test_resolve_locked_code_hashes_without_recovery_returns_original(tmp_path):
    cfg = _tmp_config(tmp_path)
    lock_path = Path(prot.create_protocol_lock(cfg))
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert prot.recovery_lock_paths(lock_path) == []
    assert prot.resolve_locked_code_hashes(lock_path) == dict(payload["code_sha256"])


# --- data access guard ------------------------------------------------------------------


@pytest.mark.parametrize("role", prot.CONFIRM_DATA_ROLES)
@pytest.mark.parametrize("path", [
    # R2 development data is HISTORY now: never a confirm-stage role input.
    "data/object_centric_stage/stage4cr2_merge_multiaction_dev/"
    "object_twinworld_stage4cr2_train.npz",
    "data/object_centric_stage/stage4cr2_merge_multiaction_dev/smoke/x.npz",
    "experiments/stage4_reliability/stage4c_r2/development/train_predictions.npz",
    # the old Stage 4C screen / 4C Plus phase0 / Stage 4B confirm stay forbidden.
    "experiments/stage4_reliability/stage4c/development/screen_group_quantities.npz",
    "experiments/stage4_reliability/stage4c_plus/phase0/phase0_report.json",
    "experiments/stage4_reliability/stage4b/confirm/x.npz",
    "data/object_centric_stage/stage4c_merge_multiaction_dev/"
    "object_twinworld_stage4c_merge_multiaction_dev_screen.npz",
])
def test_forbidden_sources_refused_for_confirm_roles(path, role):
    with pytest.raises(ValueError):
        prot.assert_allowed_confirm_data(path, role=role)


def test_own_namespace_allowed_and_unknown_role_refused():
    prot.assert_allowed_confirm_data(
        "data/object_centric_stage/stage4cr2_merge_multiaction_confirm/"
        "inference_inputs.npz", role="confirm")
    prot.assert_allowed_confirm_data(
        "data/object_centric_stage/stage4cr2_merge_multiaction_confirm/smoke/"
        "inference_inputs.npz", role="smoke")
    with pytest.raises(ValueError):
        prot.assert_allowed_confirm_data(
            "data/object_centric_stage/stage4cr2_merge_multiaction_confirm/"
            "inference_inputs.npz", role="development_train")
    with pytest.raises(ValueError):
        prot.assert_allowed_confirm_data("data/somewhere_else/x.npz", role="confirm")


def test_action_stream_roots_frozen():
    cfg = _config()
    assert prot.action_stream_root(cfg, "confirm") == (2026093012, 0)
    assert prot.action_stream_root(cfg, "smoke") == (2026093100, 12)
    with pytest.raises(ValueError):
        prot.action_stream_root(cfg, "development_train")
