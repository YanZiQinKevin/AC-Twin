"""Blocking tests for the Stage 5 protocol layer (plan v0.3 sections 3-4, Task 2).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Pins: config freeze,
old-confirm triple rejection (path / logical / registered SHA256), train+validation
source binding with live SHA256 verification against the Stage 4C-R2 frozen receipts,
global seed zero-overlap audit for the future comparison holdout schedule, and
create-once protocol-lock semantics with code/source drift rejection.
"""
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

from ac_twin import stage5_protocol as s5p  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

CONFIG_PATH = ROOT / "config/stage5_sota_comparison_v0.3.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# --- config freeze ----------------------------------------------------------------


def test_real_config_validates():
    checked = s5p.validate_config(_config())
    assert checked["plan_id"] == s5p.PLAN_ID
    assert checked["model_seeds"] == [8101, 8202, 8303, 8404, 8505]
    assert checked["power_candidate_n"] == [2000, 3000, 4000, 6000]
    assert checked["coverage_main"] == 0.8
    assert checked["materiality_shrink"] == 0.75
    assert checked["power_target"] == 0.9


@pytest.mark.parametrize("key,value", [
    ("plan_id", "AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.2"),
    ("model_seeds", [8101, 8202, 8303, 8404, 8606]),
    ("coverage_main", 0.9),
    ("materiality_shrink", 0.8),
    ("materiality_i_aurc_min", 0.04),
    ("materiality_i_regret80_min", 0.04),
    ("power_target", 0.8),
    ("power_candidate_n", [2000, 4000]),
    ("power_bootstrap_draws", 5000),
    ("power_monte_carlo", 5000),
    ("adjudication_bootstrap_draws", 5000),
    ("holdout_seed_entropy", 2026092701),
    ("tost_i_aurc", [-0.04, 0.04]),
    ("tost_i_regret80", [-0.04, 0.04]),
    ("n_actions", 9),
    ("determinism_checks", 16),
])
def test_config_field_drift_rejected(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError):
        s5p.validate_config(config)


# --- old-confirm triple rejection ---------------------------------------------------


@pytest.mark.parametrize("relpath", [
    "experiments/stage4_reliability/stage4cr2_confirm/collection/groups.npz",
    "experiments/stage4_reliability/stage4cr2_confirm/reports/STAGE4CR2_CONFIRMATORY_DECISION.md",
    "experiments/stage4_reliability/stage4d_ood/collection/groups.npz",
    "experiments/stage4_reliability/stage4d_ood/reports/STAGE4D_OOD_DECISION.md",
    "experiments/stage4_reliability/stage4d_d2f/confirm/confirm_report.json",
    "experiments/stage4_reliability/stage4d_d2f/confirm/confirm_group_quantities.npz",
    "experiments/stage4_reliability/stage4b/collection/episodes.npz",
    "experiments/stage4_reliability/stage4b/reports/STAGE4B_CONFIRMATORY_DECISION.md",
    "data/object_centric_stage/stage4cr2_merge_multiaction_dev/object_twinworld_stage4cr2_screen.npz",
    "experiments/stage4_reliability/stage4c_r2/development/screen_group_quantities.npz",
    "experiments/stage4_reliability/stage4c_r2/development/screen_direct_scores.npz",
    "experiments/stage4_reliability/stage4c_r2/development/development_screen_report.json",
])
def test_forbidden_paths_rejected(relpath):
    with pytest.raises(ValueError):
        s5p.assert_allowed_stage5_data(ROOT / relpath)


@pytest.mark.parametrize("relpath", [
    "data/object_centric_stage/stage4cr2_merge_multiaction_dev/object_twinworld_stage4cr2_train.npz",
    "data/object_centric_stage/stage4cr2_merge_multiaction_dev/object_twinworld_stage4cr2_validation.npz",
    "experiments/stage4_reliability/stage4c_r2/development/train_predictions.npz",
    "experiments/stage4_reliability/stage4c_r2/development/validation_scores.npz",
    "experiments/stage4_reliability/stage4c_r2/development/final_selection.json",
    "experiments/stage4_reliability/stage4c_r2/development/models/direct_seed8101.pt",
    "experiments/stage5_sota_comparison/literature/method_registry.json",
    "experiments/stage5_sota_comparison/development/train_bundle.npz",
])
def test_allowed_sources_pass(relpath):
    s5p.assert_allowed_stage5_data(ROOT / relpath)


def test_forbidden_content_sha_rejected_even_under_innocent_name():
    registry = s5p.forbidden_source_sha256()
    assert registry, "no forbidden confirm artifacts registered in this workspace"
    stolen_sha = next(iter(registry.values()))
    innocent = ROOT / "experiments/stage5_sota_comparison/development/innocent.json"
    with pytest.raises(ValueError):
        s5p.assert_allowed_stage5_data(innocent, file_sha256=stolen_sha)


# --- source binding -----------------------------------------------------------------


def test_resolve_sources_live_sha_matches_r2_frozen_receipts():
    sources = s5p.resolve_sources(_config())
    expected_keys = {
        "train_data", "validation_data",
        "train_predictions", "train_scores",
        "validation_predictions", "validation_scores",
        "final_selection", "feature_schema", "feature_scaler", "model_inventory",
        "train_collection_receipt", "validation_collection_receipt",
        "train_inference_receipt", "validation_inference_receipt",
    } | {f"direct_checkpoint_{s}" for s in (8101, 8202, 8303, 8404, 8505)}
    assert expected_keys <= set(sources)
    for path in sources.values():
        assert Path(path).is_file()


def test_resolve_sources_refuses_swapped_role_file():
    config = _config()
    config["train_data"] = config["validation_data"]
    with pytest.raises(ValueError):
        s5p.resolve_sources(config)


# --- global seed audit ----------------------------------------------------------------


def test_seed_audit_detects_planted_overlap(tmp_path):
    config = _config()
    schedule = s5p.build_holdout_seed_schedule(config)
    planted = tmp_path / "planted"
    planted.mkdir()
    np.savez(planted / "old_data.npz", episode_seed=schedule[:5])
    audit = s5p.audit_global_seed_overlap(schedule, [planted])
    assert not audit.zero_overlap
    with pytest.raises(ValueError):
        s5p.assert_seed_audit_clean(audit)


def test_holdout_schedule_is_unique_and_in_range():
    schedule = s5p.build_holdout_seed_schedule(_config())
    assert schedule.size == 40000
    assert np.unique(schedule).size == schedule.size
    assert schedule.min() >= 1 and schedule.max() <= 2147483646


@pytest.mark.slow
def test_global_seed_audit_zero_overlap_real_workspace():
    schedule = s5p.build_holdout_seed_schedule(_config())
    audit = s5p.audit_global_seed_overlap(schedule, [ROOT])
    assert audit.n_inventory > 0
    assert audit.zero_overlap, f"holdout schedule collides with history: {audit.overlap[:10]}"
    s5p.assert_seed_audit_clean(audit)


# --- create-once protocol lock ---------------------------------------------------------


def _lock_payload(tmp_path: Path, config: dict) -> dict:
    bound = tmp_path / "bound.json"
    bound.write_text(json.dumps({"x": 1}), encoding="utf-8")
    return {
        "plan_id": s5p.PLAN_ID,
        "config_signature": s5p.config_signature(config),
        "code_hashes": s5p.stage5_code_hashes(),
        "bound_files": {str(bound): sha256_file(bound)},
    }


def test_lock_create_once_and_drift_rejection(tmp_path):
    config = _config()
    payload = _lock_payload(tmp_path, config)
    lock_path = tmp_path / "protocol_lock.json"
    s5p.create_protocol_lock(lock_path, payload)
    s5p.verify_protocol_lock(lock_path)  # passes unchanged
    with pytest.raises((ValueError, RuntimeError)):
        s5p.create_protocol_lock(lock_path, payload)  # never twice
    bound = tmp_path / "bound.json"
    bound.write_text(json.dumps({"x": 2}), encoding="utf-8")
    with pytest.raises(ValueError):
        s5p.verify_protocol_lock(lock_path)


def test_lock_verify_rejects_code_drift(tmp_path, monkeypatch):
    config = _config()
    payload = _lock_payload(tmp_path, config)
    lock_path = tmp_path / "protocol_lock.json"
    s5p.create_protocol_lock(lock_path, payload)
    drifted = dict(payload["code_hashes"])
    drifted["code/ac_twin/stage5_protocol.py"] = "0" * 64
    monkeypatch.setattr(s5p, "stage5_code_hashes", lambda: drifted)
    with pytest.raises(ValueError):
        s5p.verify_protocol_lock(lock_path)


def test_code_inventory_covers_stage5_modules_and_registry():
    hashes = s5p.stage5_code_hashes()
    names = set(hashes)
    assert "code/ac_twin/stage5_protocol.py" in names
    assert "code/ac_twin/stage5_features.py" in names
    assert "experiments/stage5_sota_comparison/literature/method_registry.json" in names
    assert "experiments/stage5_sota_comparison/literature/compatibility_decision.json" in names
    assert all(len(v) == 64 for v in hashes.values())
