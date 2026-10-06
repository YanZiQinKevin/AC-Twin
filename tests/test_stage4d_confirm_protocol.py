"""Blocking tests for the Stage 4D D1 confirm protocol lock (plan §6.3, §11 item 5).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 4. These tests pin:

* registry-first code inventory: the collector / OOD statistics / runner / confirm module
  are REGISTERED before any hash is taken (no "hash the files that happen to exist");
* the §2.1 last-paragraph expansion: the five Direct checkpoints, the Full/History/E2E
  predictors, the response predictors, the binding and the scaler are expanded from the
  R2/confirm inventories to EXPLICIT path+SHA256 entries — never guessed from directories
  or "best" filenames — and each is measured (re-hashed) at lock time;
* the carried obligation that the lock-time seed audit is re-derived FOR REAL (the scan
  and the zero-overlap verdict are re-run, not copied from a recorded flag);
* the write-once formal ``protocol_lock.json`` satisfies Task 1's
  ``verify_protocol_lock`` (base registry + recorded clean audit) AND carries the
  collector contract keys (``selected_n``, ``ood_profile``) and the §7.1-condition-8
  strata definitions;
* the extended verifier detects code/selected_n drift and re-runs the seed audit live.
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

from ac_twin import stage4cr2_statistics as s4cr2st  # noqa: E402
from ac_twin import stage4d_confirm_protocol as confirm_prot  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

import run_stage4d_ood as runner  # noqa: E402


def _real_config():
    return json.loads(prot.CONFIG_PATH.read_text(encoding="utf-8"))


# --- registry-first code inventory ---------------------------------------------------------


def test_confirm_code_registry_covers_collector_statistics_runner_protocol():
    """§6.3 / Task-1 ⚠️: collector, OOD env, statistics, protocol and runner are ALL in
    the registered set; registration precedes hashing (the registry is the allow-list)."""
    registry = confirm_prot.CONFIRM_REGISTRY
    for rel in ("collect_stage4d_merge_ood.py",
                "ac_twin/stage4d_ood_statistics.py",
                "run_stage4d_ood.py",
                "ac_twin/stage4d_confirm_protocol.py",
                "ac_twin/stage4d_protocol.py",
                "ac_twin/stage4d_ood_env.py"):
        assert rel in registry, f"{rel} must be registered before the lock"
    # registry is a superset of Task 1's registry (nothing Task 1 bound is dropped).
    for rel in list(prot.PROTOCOL_CODE_FILES) + list(prot.REUSED_FROZEN_CODE_FILES):
        assert rel in registry
    # no duplicates (each file hashed exactly once).
    assert len(registry) == len(set(registry))


def test_confirm_code_hashes_cover_every_registered_file_no_hash_on_existence():
    """Every registered file is hashed (registry-driven); unregistered drift files are
    never silently included (hash-on-existence is forbidden)."""
    hashes = confirm_prot.confirm_code_hashes()
    assert set(hashes) == set(confirm_prot.CONFIRM_REGISTRY)
    assert len(hashes) == len(confirm_prot.CONFIRM_REGISTRY)
    for rel, digest in hashes.items():
        assert digest == sha256_file(ROOT / "code" / rel)
    # the base Task-1 subset matches Task 1's own protocol_code_hashes exactly.
    base = prot.protocol_code_hashes()
    for rel, digest in base.items():
        assert hashes[rel] == digest


def test_code_inventory_binds_base_and_extended_sets():
    inventory = confirm_prot.build_confirm_code_inventory()
    assert inventory["plan_id"] == prot.PLAN_ID
    assert set(inventory["code_sha256"]) == set(confirm_prot.CONFIRM_REGISTRY)
    assert inventory["base_code_sha256"] == prot.protocol_code_hashes()
    assert inventory["inventory_sha256"] == hashlib.sha256(
        json.dumps(inventory["code_sha256"], sort_keys=True)
        .encode("utf-8")).hexdigest()


# --- §2.1 last-paragraph expansion (explicit path+hash, measured) ----------------------------


def test_source_inventory_expands_direct_and_predictor_assets_explicitly():
    """The lock source inventory expands §2.1 to explicit path+SHA256 for the five Direct
    checkpoints, the Full/History/E2E predictors, the five response predictors, the
    binding and the scaler — a strict superset of Task 1's ten pinned sources."""
    inventory = confirm_prot.build_confirm_source_inventory(_real_config())
    files = inventory["files"]
    assert inventory["n_files"] == len(files) > len(prot.STAGE4D_PINNED_SOURCES)
    # Task 1's ten pinned sources are preserved byte-identically (superset consistency).
    existing = json.loads(
        (ROOT / "experiments/stage4_reliability/stage4d_ood/protocol/"
         "source_inventory.json").read_text(encoding="utf-8"))
    for rel, entry in existing["files"].items():
        assert rel in files
        assert files[rel]["sha256"] == entry["sha256"]
        assert Path(files[rel]["path"]) == Path(entry["path"])
    # the five Direct checkpoints, explicitly (not a directory glob or "best" name).
    for seed in prot.MODEL_SEEDS:
        label = f"r2_development/models/direct_seed{seed}.pt"
        assert label in files, label
        assert Path(files[label]["path"]).is_file()
        assert files[label]["sha256"] == sha256_file(files[label]["path"])
    # Full/History experts + E2E routers + response predictors + binding + scaler.
    assert any(label.startswith("stage3r/pair") and "fulltrain_full" in label
               for label in files)
    assert any(label.startswith("stage3r/pair") and "fulltrain_history" in label
               for label in files)
    assert any(label.startswith("stage3r/end_to_end_mixture") for label in files)
    for seed in prot.RESPONSE_PREDICTOR_SEEDS:
        assert f"stage3b/response_predictor_seed{seed}.pt" in files
    assert "stage3b/input_binding_receipt.json" in files
    # every entry is measured: re-hash equals the recorded hash.
    for label, entry in files.items():
        assert sha256_file(entry["path"]) == entry["sha256"], label


def test_source_inventory_fails_closed_on_direct_checkpoint_drift():
    """Any drift in a frozen §2.1 asset aborts the inventory (fail-closed before lock)."""
    from ac_twin import stage4cr2_confirm_protocol as r2confirm
    original = r2confirm.R2_PINNED_SOURCES["final_selection.json"]
    try:
        r2confirm.R2_PINNED_SOURCES["final_selection.json"] = "0" * 64
        with pytest.raises(Exception, match="hash drift|drift|mismatch"):
            confirm_prot.build_confirm_source_inventory(_real_config())
    finally:
        r2confirm.R2_PINNED_SOURCES["final_selection.json"] = original


def test_response_predictor_paths_come_from_binding_not_directories(tmp_path, monkeypatch):
    """Response-predictor entries resolve via the binding receipt's explicit path+hash —
    never by scanning a directory or picking a 'best' filename."""
    from ac_twin import stage4cr2_confirm_protocol as r2confirm
    real_binding = json.loads((ROOT / "experiments/stage3_routing/stage3b/"
                                    "input_binding_receipt.json").read_text("utf-8"))
    payload = real_binding["receipt_payload"]
    seed = str(prot.RESPONSE_PREDICTOR_SEEDS[0])
    real_path = Path(payload["checkpoints"][seed]["path"])
    assert real_path.is_file()
    # point the binding record at a tmp copy under a decoy directory; the recorded hash
    # (of the REAL file) must fail -> proves the path+hash both come from the binding.
    decoy = tmp_path / "best_response_predictor.pt"  # a 'best'-looking decoy name
    decoy.write_bytes(b"not the real predictor")
    tampered = json.loads(json.dumps(real_binding))
    tampered["receipt_payload"]["checkpoints"][seed]["path"] = str(decoy)
    binding_path = tmp_path / "input_binding_receipt.json"
    binding_path.write_text(json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(
        confirm_prot, "STAGE3B_BINDING_RECEIPT", str(binding_path))
    with pytest.raises(Exception, match="drift|hash|mismatch|signature"):
        confirm_prot._expand_response_predictors(binding_path)


# --- the lock-time seed audit is re-derived FOR REAL (carried obligation #1) ------------------


def test_real_seed_audit_reruns_scan_and_zero_overlap():
    """The audit re-runs the full historical scan and a live zero-overlap verdict for all
    five domains; it never copies a recorded flag. Inventory hash + per-domain verdicts."""
    config = _real_config()
    audit, inventory = confirm_prot.run_real_seed_audit(config)
    assert audit["zero_overlap"] is True
    assert audit["overlap"] == []
    assert inventory["n_inventory"] > 50000  # real global historical+reserved pool
    assert inventory["inventory_sha256"] == hashlib.sha256(
        np.asarray(
            prot.scan_historical_group_seeds(
                [ROOT / "data", ROOT / "experiments"]), dtype=np.int64
        ).tobytes()).hexdigest()
    for role in prot.SEED_ROLES:
        assert audit["per_domain"][role]["zero_overlap"] is True
        assert audit["per_domain"][role]["n_schedule"] > 0
    # mutual disjointness of the five candidate domains is proven, not assumed.
    assert audit["mutual_disjoint"] is True


# --- strata (§7.1 condition 8) ------------------------------------------------------------------


def test_strata_definitions_two_layers_with_r2_train_threshold(monkeypatch):
    """§7.1 cond 8: t_star split is the frozen 13; the initial-distance threshold is the
    R2-train-only median via derive_r2_train_distance_threshold (rule recorded)."""
    monkeypatch.setattr(
        confirm_prot.d0, "load_r2_train_id_features",
        lambda config: {"min_ego_distance": np.array([10.0, 20.0, 30.0, 40.0])})
    strata = confirm_prot.derive_strata_definitions(_real_config())
    assert strata["t_star"]["split"] == s4cr2st.TSTAR_SPLIT == 13
    assert strata["t_star"]["low"] == "t_star < 13"
    assert strata["t_star"]["high"] == "t_star >= 13"
    distance = strata["initial_distance"]
    assert distance["threshold"] == 25.0  # median of [10,20,30,40]
    assert distance["rule"] == prot.R2_TRAIN_DISTANCE_THRESHOLD_RULE
    assert distance["r2_train_npz_sha256"] == prot.R2_TRAIN_NPZ_SHA256
    assert "history" in distance["source"].lower()  # pre-intervention history ONLY


# --- write-once lock + base/extended verification -----------------------------------------------


def _tmp_lock_env(tmp_path, monkeypatch):
    """Real frozen config content with run_dir/output paths redirected to tmp.

    The pinned §2.1 sources stay at their real absolute locations (r2_development_dir and
    the scan roots are real); only this stage's own products land under tmp. The lock-time
    seed audit is stubbed with a clean recorded verdict (the real re-run is tested
    separately above and executed by the real lock phase).
    """
    config = _real_config()
    run_dir = tmp_path / "stage4d_ood"
    config["run_dir"] = str(run_dir)
    config["paths"] = dict(config["paths"])
    for key in ("protocol_lock", "source_inventory", "code_inventory", "seed_inventory"):
        config["paths"][key] = str(run_dir / "protocol" / Path(key).name)
        # keep the canonical filename for each artifact
    config["paths"]["protocol_dir"] = str(run_dir / "protocol")
    config["paths"]["protocol_lock"] = str(run_dir / "protocol" / "protocol_lock.json")
    config["paths"]["source_inventory"] = str(run_dir / "protocol" / "source_inventory.json")
    config["paths"]["code_inventory"] = str(run_dir / "protocol" / "code_inventory.json")
    config["paths"]["seed_inventory"] = str(run_dir / "protocol" / "seed_inventory.json")
    config["paths"]["power_report"] = str(run_dir / "power" / "power_report.json")
    config["paths"]["ood_profile_selection"] = str(
        run_dir / "pilot" / "ood_profile_selection.json")
    # a frozen power report (selected n) and the D0 selection.
    power_path = Path(config["paths"]["power_report"])
    power_path.parent.mkdir(parents=True, exist_ok=True)
    power_path.write_text(json.dumps({
        "plan_id": prot.PLAN_ID, "status": prot.STATUS_D1_POWER_FROZEN,
        "selected_n": 2000, "underpowered": False,
        "screen_quantities_sha256": "a" * 64,
    }), encoding="utf-8")
    selection_path = Path(config["paths"]["ood_profile_selection"])
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({
        "plan_id": prot.PLAN_ID, "selected_profile": "JOINT_OOD",
        "status": prot.STATUS_D0_PROFILE_SELECTED,
    }), encoding="utf-8")
    # tmp config file so verify_protocol_lock's config-signature check lines up.
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(prot, "CONFIG_PATH", config_path)
    # stub the expensive real audit with a clean recorded verdict.
    audit = {"zero_overlap": True, "overlap": [], "mutual_disjoint": True,
             "per_domain": {role: {"zero_overlap": True, "n_schedule": 1,
                                   "n_inventory": 1} for role in prot.SEED_ROLES},
             "inventory_sha256": "b" * 64, "n_inventory": 1}
    inventory = {"n_inventory": 1, "inventory_sha256": "b" * 64, "domains": {}}
    monkeypatch.setattr(confirm_prot, "run_real_seed_audit",
                        lambda cfg, roots=None: (audit, inventory))
    # stub the R2-train read for strata (the real threshold is derived in the lock phase).
    monkeypatch.setattr(
        confirm_prot.d0, "load_r2_train_id_features",
        lambda cfg: {"min_ego_distance": np.array([10.0, 20.0, 30.0, 40.0])})
    return config, run_dir


def test_create_lock_write_once_and_base_verification(tmp_path, monkeypatch):
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    lock_path = confirm_prot.create_protocol_lock(config)
    assert Path(lock_path).is_file()
    # Task 1's base verification primitive accepts the lock (fail-closed without it).
    verified = prot.verify_protocol_lock(run_dir)
    assert verified["plan_id"] == prot.PLAN_ID
    # the collector contract keys are bound.
    assert int(verified["selected_n"]) == 2000
    assert verified["ood_profile"] == "JOINT_OOD"
    # base code_sha256 == Task 1 registry exactly (strict dict equality).
    assert verified["code_sha256"] == prot.protocol_code_hashes()
    # extended registry carried separately.
    assert set(verified["extended_code_sha256"]) == set(confirm_prot.CONFIRM_REGISTRY)
    # write-once: a second create is refused.
    with pytest.raises(FileExistsError):
        confirm_prot.create_protocol_lock(config)
    # the sidecar inventories exist.
    assert Path(config["paths"]["source_inventory"]).is_file()
    assert Path(config["paths"]["code_inventory"]).is_file()
    assert Path(config["paths"]["seed_inventory"]).is_file()


def test_lock_binds_strata_profile_selection_and_bootstrap_entropy(tmp_path, monkeypatch):
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    confirm_prot.create_protocol_lock(config)
    lock = json.loads(
        Path(config["paths"]["protocol_lock"]).read_text(encoding="utf-8"))
    # §7.1 condition-8 two-layer strata.
    assert lock["strata"]["t_star"]["split"] == 13
    assert lock["strata"]["initial_distance"]["rule"] == \
        prot.R2_TRAIN_DISTANCE_THRESHOLD_RULE
    assert lock["strata"]["initial_distance"]["threshold"] == 25.0
    # unique OOD profile + selection-file hash.
    assert lock["ood_profile"] == "JOINT_OOD"
    assert lock["ood_profile_selection_sha256"] == sha256_file(
        config["paths"]["ood_profile_selection"])
    # D1 bootstrap entropy 2026110995 + the accepted n.
    assert lock["d1_bootstrap_entropy"] == prot.D1_BOOTSTRAP_ENTROPY == 2026110995
    assert lock["selected_n"] == 2000
    # the recorded seed audit is clean (prot.verify_protocol_lock's channel).
    assert lock["seed_audit"]["zero_overlap"] is True
    assert lock["seed_audit"]["overlap"] == []


def test_lock_refused_without_frozen_power(tmp_path, monkeypatch):
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    Path(config["paths"]["power_report"]).unlink()  # no frozen n -> refuse to lock
    with pytest.raises((ValueError, FileNotFoundError)):
        confirm_prot.create_protocol_lock(config)
    assert not Path(config["paths"]["protocol_lock"]).exists()


def test_lock_refused_on_underpowered_power_report(tmp_path, monkeypatch):
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    Path(config["paths"]["power_report"]).write_text(json.dumps({
        "plan_id": prot.PLAN_ID, "status": prot.STATUS_UNDERPOWERED,
        "selected_n": None, "underpowered": True,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="power|underpowered|selected_n"):
        confirm_prot.create_protocol_lock(config)
    assert not Path(config["paths"]["protocol_lock"]).exists()


def test_extended_verify_detects_code_drift(tmp_path, monkeypatch):
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    confirm_prot.create_protocol_lock(config)
    # drift one registered code file -> the extended verifier fails closed.
    target = ROOT / "code" / "collect_stage4d_merge_ood.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# drift\n")
        with pytest.raises(Exception, match="code|drift|hash|mismatch"):
            confirm_prot.verify_confirm_protocol_lock(run_dir)
    finally:
        target.write_bytes(original)


def test_extended_verify_detects_selected_n_drift(tmp_path, monkeypatch):
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    confirm_prot.create_protocol_lock(config)
    lock_path = Path(config["paths"]["protocol_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["selected_n"] = 999  # tamper the frozen n
    # rewrite (test-local tamper of the tmp copy, not a frozen file)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(Exception, match="selected_n|power|drift|mismatch"):
        confirm_prot.verify_confirm_protocol_lock(run_dir)


def test_extended_verify_reruns_seed_audit_live(tmp_path, monkeypatch):
    """Carried obligation #1 at verify time: the extended verifier re-runs the scan +
    zero-overlap; a non-clean live audit fails even with a clean recorded flag."""
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    confirm_prot.create_protocol_lock(config)
    dirty = ({"zero_overlap": False, "overlap": [123], "mutual_disjoint": False,
              "per_domain": {}, "inventory_sha256": "c" * 64, "n_inventory": 1},
             {"n_inventory": 1, "inventory_sha256": "c" * 64, "domains": {}})
    monkeypatch.setattr(confirm_prot, "run_real_seed_audit",
                        lambda cfg, roots=None: dirty)
    with pytest.raises(Exception, match="seed|overlap|audit"):
        confirm_prot.verify_confirm_protocol_lock(run_dir)


# --- runner lock-d1 phase (mocked heavy work) ---------------------------------------------------


def test_runner_lock_d1_phase_creates_lock(tmp_path, monkeypatch):
    config, run_dir = _tmp_lock_env(tmp_path, monkeypatch)
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    calls = []
    real_create = confirm_prot.create_protocol_lock

    def tracking_create(cfg):
        calls.append("create")
        return real_create(cfg)

    monkeypatch.setattr(runner.confirm_prot, "create_protocol_lock", tracking_create)
    outcome = runner.run_phase("lock-d1", config=config)
    assert calls == ["create"]
    assert outcome["status"] == prot.STATUS_D1_LOCKED
    assert Path(config["paths"]["protocol_lock"]).is_file()
    # resume: verifies, never rewrites.
    again = runner.run_phase("lock-d1", config=config)
    assert again["verification"] == "existing_lock_verified_no_overwrite"
    assert calls == ["create"]  # create not called again
