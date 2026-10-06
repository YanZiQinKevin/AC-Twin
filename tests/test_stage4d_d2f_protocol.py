"""Focused blocking-invariant tests for the D2F protocol: source binding + dual-pool audit.

Plan ``AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1`` sections
1, 2.2 and 3.1. The distinct invariants: (1) the nine plan-listed sources are bound verbatim
(path + SHA) and any drift / missing file is fail-closed; (2) the burned-set scan unions every
old role's ledger + accepted IDs with dedup (a replayed role cannot double-count) and refuses
an accepted ID absent from its ledger; (3) the three-way classification preserves the OLD
confirm pool for the D2F confirm ONLY and leaves every other never-attempted scenario as the
sole legal screen source; (4) the admissible n sets are the plan candidates intersected with
the 5xn capacity floor (screen pool minus the <=160 smoke budget); (5) the fresh screen order
is salt-deterministic with zero overlap against every burned and every confirm ID; (6) smoke
consumption beyond 160 candidates halts; (7) audit artifacts are create-once.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402


# --- 1. the plan-listed source table -------------------------------------------------------------------------


def test_plan_listed_sources_bound_verbatim() -> None:
    assert len(PROT.PLAN_LISTED_SOURCES) == 9
    table = {rel: sha for rel, sha, _ in PROT.PLAN_LISTED_SOURCES}
    assert table["stage4d_d2e/development/final_selection.json"] == \
        "c2236a9b7798c702cbe6651c30fcac41d037343b3fd7eb984e75495c25e65259"
    assert table["stage4d_d2e/development/power_validation_quantities.npz"] == \
        "023b83827cd278707e59dfcb8928917679e761a6993a9514378d66186985f21a"
    assert table["stage4d_d2e/power/screen_power_report.json"] == \
        "d2570e290d42068de32037a6e3ecabe8d4ccc0ffda6f84a5c23904d603b1f75e"
    assert table["stage4d_external/development/base_models/predictor_stack.json"] == \
        "44de5543818b3644f7cfabf73040949628e2308c7425c06b13eaed4a99f86d60"
    assert table["stage4d_external/platform/scenario_split_manifest.json"] == \
        "6c243960582a46b3b55ab20adda17e3532cf958d69d17c43b9505b7951c2c874"
    assert table["code/ac_twin/stage4c_objective.py"] == \
        "a20da1805c14b57bc2ad02f363133b7c53c480909190f52fa5a1cc09964ac44e"
    notes = " ".join(note for _, _, note in PROT.PLAN_LISTED_SOURCES)
    assert "NEVER a new power result" in notes  # the old D2E report is bound, never reused


def test_hash_existing_fails_closed_on_missing(tmp_path) -> None:
    with pytest.raises(PROT.SourceDriftError):
        PROT._hash_existing(tmp_path / "missing.bin", "unit-test missing file")


# --- 2. the burned-set scan -------------------------------------------------------------------------------------


def _write_role(root: Path, role: str, ledger_ids: list[str], accepted_ids: list[str]) -> None:
    role_dir = root / role
    role_dir.mkdir(parents=True, exist_ok=True)
    with (role_dir / "rejection_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for scenario_id in ledger_ids:
            handle.write(json.dumps({"outcome": "rejected", "scenario_id": scenario_id}) + "\n")
        for scenario_id in accepted_ids:
            handle.write(json.dumps({"outcome": "accepted", "scenario_id": scenario_id}) + "\n")
    (role_dir / "groups_meta.json").write_text(json.dumps(
        [{"scenario_id": s, "t_star": 10} for s in accepted_ids]), encoding="utf-8")


def test_burned_scan_unions_and_dedups_replayed_roles(tmp_path) -> None:
    roles = tmp_path / "roles"
    _write_role(roles, "pilot", ["a", "b"], ["c"])
    _write_role(roles, "pilot_replay", ["a", "b"], ["c"])   # replay: same IDs, must dedup
    _write_role(roles, "train", ["d"], ["e", "f"])
    scan = PROT.scan_burned_ids(platform_dir=tmp_path)
    assert scan["burned"] == {"a", "b", "c", "d", "e", "f"}
    assert scan["per_role"]["pilot"]["burned_ids"] == 3
    # D2R/D2E never collected; their absence is asserted, not an error.
    assert scan["per_role"]["stage4d_d2e"]["attempted_unique_ids"] == 0


def test_burned_scan_refuses_accepted_id_absent_from_ledger(tmp_path) -> None:
    role_dir = tmp_path / "roles" / "validation"
    role_dir.mkdir(parents=True)
    (role_dir / "rejection_ledger.jsonl").write_text(
        json.dumps({"outcome": "accepted", "scenario_id": "x"}) + "\n", encoding="utf-8")
    (role_dir / "groups_meta.json").write_text(
        json.dumps([{"scenario_id": "x", "t_star": 10},
                    {"scenario_id": "y", "t_star": 10}]),  # "y" accepted but never logged
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="absent from its"):
        PROT.scan_burned_ids(platform_dir=tmp_path)


# --- 3. three-way classification + admissible n ------------------------------------------------------------------


def _rows(spec: dict[str, int]) -> list[dict]:
    rows, rank = [], 0
    for role, count in spec.items():
        for i in range(count):
            rows.append({"global_rank": rank, "pool": "primary" if i == 0 else "reserve",
                         "role": role, "scenario_id": f"{role}-{i:05d}",
                         "source_shard": "training_20s.tfrecord-00000-of-01000"})
            rank += 1
    return rows


def test_classification_preserves_confirm_pool_for_confirm_only() -> None:
    rows = _rows({"train": 4, "confirm": 3, "development_screen": 3})
    burned = {"train-00000", "development_screen-00001"}
    out = PROT.classify_pools(rows, burned)
    assert out["confirm_ids"] == {f"confirm-{i:05d}" for i in range(3)}
    assert out["available_confirm"] == out["confirm_ids"]
    expected_screen = {f"train-{i:05d}" for i in range(1, 4)} \
        | {"development_screen-00000", "development_screen-00002"}
    assert out["screen_available"] == expected_screen, (
        "the screen source is every never-attempted, non-confirm scenario; the confirm pool "
        "is preserved for the D2F confirm only")
    assert out["by_old_role"]["train"] == {"attempted_burned": 1,
                                           "reserved_never_attempted": 3}
    with pytest.raises(ValueError, match="duplicate scenario_id"):
        PROT.classify_pools(rows + [rows[0]], burned)


def test_admissible_ns_capacity_floor_and_smoke_reserve() -> None:
    # 5xn floor; the screen pool additionally reserves the 160-candidate smoke budget.
    assert PROT.admissible_ns(12660, (1000, 1500, 2000, 2500), reserve=160) == \
        [1000, 1500, 2000, 2500]
    assert PROT.admissible_ns(12659, (1000, 1500, 2000, 2500), reserve=160) == \
        [1000, 1500, 2000]
    assert PROT.admissible_ns(15000, (1000, 1500, 2000, 2500, 3000)) == \
        [1000, 1500, 2000, 2500, 3000]
    assert PROT.admissible_ns(14999, (1000, 1500, 2000, 2500, 3000)) == \
        [1000, 1500, 2000, 2500]
    assert PROT.admissible_ns(100, (1000, 1500), reserve=160) == []


# --- 4. fresh screen order + smoke budget ------------------------------------------------------------------------


def test_derive_screen_candidate_ids_deterministic_zero_overlap() -> None:
    rows = _rows({"train": 6, "confirm": 4})
    burned = {"train-00000"}
    order = PROT.derive_screen_candidate_ids(rows, burned)
    assert order == PROT.derive_screen_candidate_ids(rows, burned), "salt order is frozen"
    assert set(order) == {f"train-{i:05d}" for i in range(1, 6)}
    assert not (set(order) & burned)
    assert not (set(order) & {f"confirm-{i:05d}" for i in range(4)})
    keyed = sorted(order, key=lambda sid: hashlib.sha256(
        (PROT.SCREEN_SALT + sid).encode("utf-8")).hexdigest())
    assert order == keyed, "the order is the frozen SHA256 salt ranking"
    other = PROT.derive_screen_candidate_ids(rows, burned, salt="AC-TWIN-OTHER|")
    assert set(other) == set(order) and other != order, "a different salt re-ranks"
    assert PROT.SCREEN_SALT == "AC-TWIN-D2F-V0.1|"


def test_smoke_budget_halts_beyond_160_candidates() -> None:
    PROT.assert_smoke_budget(160)
    with pytest.raises(RuntimeError, match="capacity audit"):
        PROT.assert_smoke_budget(161)


# --- 5. create-once audit on a synthetic workspace -------------------------------------------------------------------


def _synth_workspace(tmp_path: Path) -> Path:
    platform = tmp_path / "d2" / "platform"
    platform.mkdir(parents=True)
    rows = _rows({"train": 300, "confirm": 120, "development_screen": 80})
    with (platform / "scenario_split_pools.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (platform / "scenario_split_manifest.json").write_text(
        json.dumps({"plan_id": "d2", "pools_rows": len(rows)}), encoding="utf-8")
    _write_role(tmp_path / "d2" / "roles", "train", ["train-00000"], ["train-00001"])
    return tmp_path


def test_pool_capacity_audit_create_once_and_terminal(tmp_path, monkeypatch) -> None:
    _synth_workspace(tmp_path)
    out_dir = tmp_path / "d2f"
    monkeypatch.setattr(PROT, "D2_DIR", tmp_path / "d2")
    monkeypatch.setattr(PROT, "D2F_DIR", out_dir)
    monkeypatch.setattr(PROT, "SCREEN_CANDIDATE_NS", (10, 20))
    monkeypatch.setattr(PROT, "CONFIRM_CANDIDATE_NS", (10, 20, 24))
    monkeypatch.setattr(PROT, "SMOKE_CANDIDATE_BUDGET", 5)

    audit = PROT.run_pool_capacity_audit()
    # screen_available = 300+80-2 = 378; reserve 5 -> 10,20 feasible (5*20+5=105<=378);
    # confirm_available = 120 -> 10,20 feasible (100<=120), 24 not (120<120? 5*24=120<=120 yes).
    assert audit["status"] == PROT.STATUS_AUDIT_PASS
    assert audit["capacity"]["screen_admissible_ns"] == [10, 20]
    assert audit["capacity"]["confirm_admissible_ns"] == [10, 20, 24]
    assert audit["classification"]["available_confirm"] == 120
    assert audit["classification"]["screen_available"] == 378
    assert audit["classification"]["attempted_burned_unique"] == 2

    again = PROT.run_pool_capacity_audit()
    assert again["written_at"] == audit["written_at"], "the audit is create-once"

    # a tiny workspace with no feasible n is the preregistered terminal state.
    monkeypatch.setattr(PROT, "SCREEN_CANDIDATE_NS", (10_000,))
    monkeypatch.setattr(PROT, "CONFIRM_CANDIDATE_NS", (10_000,))
    monkeypatch.setattr(PROT, "D2F_DIR", tmp_path / "d2f-small")
    audit2 = PROT.run_pool_capacity_audit()
    assert audit2["status"] == PROT.STATUS_CAPACITY_INSUFFICIENT
    assert audit2["terminal"] is True


# --- 6. Phase S: fresh-ID verification / post-smoke audit / manifest / lock ---------------------------------


def _phase_s_workspace(tmp_path: Path, monkeypatch, *, selected_n: int = 10,
                       smoke_ids: tuple[str, ...] = ()) -> Path:
    """Synthetic workspace past Phase P: audit + power report + optional smoke ledger."""
    _synth_workspace(tmp_path)
    out_dir = tmp_path / "d2f"
    monkeypatch.setattr(PROT, "D2_DIR", tmp_path / "d2")
    monkeypatch.setattr(PROT, "D2F_DIR", out_dir)
    monkeypatch.setattr(PROT, "SCREEN_CANDIDATE_NS", (10, 20))
    monkeypatch.setattr(PROT, "CONFIRM_CANDIDATE_NS", (10, 20, 24))
    monkeypatch.setattr(PROT, "SMOKE_CANDIDATE_BUDGET", 5)
    audit = PROT.run_pool_capacity_audit()
    assert audit["status"] == PROT.STATUS_AUDIT_PASS
    (out_dir / PROT.SOURCE_BINDING_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": PROT.STATUS_AUDIT_PASS,
         "expanded": {"checkpoints": [], "scalers": []}}), encoding="utf-8")
    power_dir = out_dir / "power"
    power_dir.mkdir(parents=True, exist_ok=True)
    (power_dir / "power_correction_report.json").write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": "D2F_POWER_CORRECTION_PASS",
         "selected_n": selected_n}), encoding="utf-8")
    if smoke_ids:
        smoke = out_dir / "smoke"
        smoke.mkdir(parents=True, exist_ok=True)
        with (smoke / "rejection_ledger.jsonl").open("w", encoding="utf-8") as handle:
            for sid in smoke_ids:
                handle.write(json.dumps({"scenario_id": sid, "outcome": "rejected",
                                         "reason": "branch_incomplete"}) + "\n")
    return out_dir


def test_fresh_id_verification_create_once_and_zero_overlap(tmp_path, monkeypatch) -> None:
    _phase_s_workspace(tmp_path, monkeypatch)
    payload = PROT.run_fresh_id_verification()
    assert payload["status"] == PROT.STATUS_AUDIT_PASS
    assert payload["screen_candidate_count"] == 378
    assert payload["zero_overlap_burned"] and payload["zero_overlap_confirm"]
    assert payload["prefrozen_screen_ns"] == [10, 20]
    again = PROT.run_fresh_id_verification()
    assert again["written_at"] == payload["written_at"], "create-once reload"
    # a burned-set change after the verification is fail-closed.
    _write_role(tmp_path / "d2" / "roles", "pilot", ["train-00002"], [])
    with pytest.raises(RuntimeError, match="burned set changed"):
        PROT.run_fresh_id_verification()


def test_fresh_id_verification_requires_audit_pass(tmp_path, monkeypatch) -> None:
    _synth_workspace(tmp_path)
    out_dir = tmp_path / "d2f"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(PROT, "D2_DIR", tmp_path / "d2")
    monkeypatch.setattr(PROT, "D2F_DIR", out_dir)
    (out_dir / PROT.POOL_AUDIT_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": PROT.STATUS_CAPACITY_INSUFFICIENT}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="D2F_POOL_AUDIT_PASS"):
        PROT.run_fresh_id_verification()


def test_post_smoke_audit_appends_smoke_and_checks_capacity(tmp_path, monkeypatch) -> None:
    rows = _rows({"train": 300, "confirm": 120, "development_screen": 80})
    burned = {"train-00000", "train-00001"}
    order = PROT.derive_screen_candidate_ids(rows, burned)
    smoke_ids = tuple(order[:3])                                # legal: fresh screen prefix
    _phase_s_workspace(tmp_path, monkeypatch, selected_n=10, smoke_ids=smoke_ids)
    payload = PROT.run_post_smoke_pool_audit()
    assert payload["status"] == PROT.STATUS_AUDIT_PASS
    assert payload["smoke_burned_ids"] == 3
    assert payload["screen_available_post_smoke"] == 378 - 3
    assert payload["screen_capacity_required"] == 50
    assert payload["confirm_pool_preserved"] == 120
    again = PROT.run_post_smoke_pool_audit()
    assert again["written_at"] == payload["written_at"]


def test_post_smoke_audit_terminal_when_capacity_breaks(tmp_path, monkeypatch) -> None:
    _phase_s_workspace(tmp_path, monkeypatch, selected_n=10)
    # shrink the screen pool below 5 x selected_n by monkeypatching the derivation inputs:
    # burn almost everything through an extra old role ledger.
    rows = PROT.load_pool_rows(tmp_path / "d2" / "platform" / "scenario_split_pools.jsonl")
    scan = PROT.scan_burned_ids(platform_dir=tmp_path / "d2")
    cls = PROT.classify_pools(rows, scan["burned"])
    extra = sorted(cls["screen_available"])[:-49]               # leave 49 < 5*10 = 50
    _write_role(tmp_path / "d2" / "roles", "pilot", extra, [])
    payload = PROT.run_post_smoke_pool_audit()
    assert payload["status"] == PROT.STATUS_SCREEN_POOL_INSUFFICIENT
    assert payload["terminal"] is True
    assert payload["screen_capacity_ok"] is False


def test_post_smoke_audit_refuses_smoke_outside_screen_pool(tmp_path, monkeypatch) -> None:
    _phase_s_workspace(tmp_path, monkeypatch, selected_n=10,
                       smoke_ids=("confirm-00000",))            # confirm encroachment
    with pytest.raises(RuntimeError, match="outside the fresh screen pool|confirm pool"):
        PROT.run_post_smoke_pool_audit()


def test_screen_manifest_primary_reserve_and_smoke_exclusion(tmp_path, monkeypatch) -> None:
    rows = _rows({"train": 300, "confirm": 120, "development_screen": 80})
    burned = {"train-00000", "train-00001"}
    pre_order = PROT.derive_screen_candidate_ids(rows, burned)
    smoke_ids = tuple(pre_order[:2])
    _phase_s_workspace(tmp_path, monkeypatch, selected_n=10, smoke_ids=smoke_ids)
    PROT.run_post_smoke_pool_audit()
    manifest = PROT.build_screen_manifest()
    assert manifest["selected_n"] == 10
    assert len(manifest["primary_ids"]) == 50                    # 5 x selected_n
    assert len(manifest["reserve_ids"]) == 378 - 2 - 50
    assert not (set(manifest["primary_ids"]) & set(smoke_ids)), "smoke IDs are excluded"
    assert not (set(manifest["reserve_ids"]) & set(smoke_ids))
    post_order = PROT.derive_screen_candidate_ids(rows, burned | set(smoke_ids))
    assert manifest["primary_ids"] + manifest["reserve_ids"] == post_order
    assert set(manifest["source_shards"]) == set(manifest["primary_ids"])
    again = PROT.build_screen_manifest()
    assert again["written_at"] == manifest["written_at"]


def test_manifest_refused_on_terminal_post_smoke_audit(tmp_path, monkeypatch) -> None:
    out_dir = _phase_s_workspace(tmp_path, monkeypatch, selected_n=10)
    (out_dir / PROT.POST_SMOKE_AUDIT_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "status": PROT.STATUS_SCREEN_POOL_INSUFFICIENT,
         "terminal": True, "selected_n": 10}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="never created on a terminal"):
        PROT.build_screen_manifest()


def test_code_inventory_registry_hash_and_drift(tmp_path, monkeypatch) -> None:
    out_dir = tmp_path / "d2f"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(PROT, "D2F_DIR", out_dir)
    registry_file = tmp_path / "code" / "registered.py"
    monkeypatch.setattr(PROT, "ROOT", tmp_path)
    monkeypatch.setattr(PROT, "D2F_SCREEN_CODE_REGISTRY", ("code/registered.py",))
    with pytest.raises(PROT.SourceDriftError, match="missing"):
        PROT.screen_code_registry()                              # listed but absent -> closed
    registry_file.parent.mkdir(parents=True)
    registry_file.write_text("x = 1\n", encoding="utf-8")
    payload = PROT.write_code_inventory()
    assert payload["code"][0]["path"] == "code/registered.py"
    assert len(payload["code"][0]["sha256"]) == 64
    again = PROT.write_code_inventory()
    assert again["written_at"] == payload["written_at"]
    registry_file.write_text("x = 2\n", encoding="utf-8")        # drift after the freeze
    with pytest.raises(PROT.SourceDriftError, match="versioned"):
        PROT.write_code_inventory()


def test_screen_lock_frozen_gates_and_bindings(tmp_path, monkeypatch) -> None:
    out_dir = _phase_s_workspace(tmp_path, monkeypatch, selected_n=10)
    PROT.run_fresh_id_verification()
    PROT.run_post_smoke_pool_audit()
    PROT.build_screen_manifest()
    # synthetic inventories (the real ones hash the repo; here only the binding contract matters)
    (out_dir / PROT.CODE_INVENTORY_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "code": []}), encoding="utf-8")
    (out_dir / PROT.MODEL_INVENTORY_NAME).write_text(json.dumps(
        {"plan_id": PROT.PLAN_ID, "checkpoints": []}), encoding="utf-8")
    lock = PROT.freeze_screen_lock()
    assert lock["selected_n"] == 10
    assert lock["bootstrap"]["entropy"] == PROT.D2F_SCREEN_BOOTSTRAP_ENTROPY == 2026122407
    assert lock["bootstrap"]["resamples"] == 10000
    assert lock["bootstrap"]["confidence"] == 0.80
    assert lock["gates"]["frozen_primary_baseline"] == "SEED_MARGIN_STD"
    assert lock["gates"]["task_x_dev"]["c1"] == "groups == 10 and all QA flags"
    assert lock["terminal_states"]["both_pass"] == "D2F_DEV_READY_FOR_CONFIRM_POWER"
    assert lock["terminal_states"]["task_fail"] == "D2F_EXTERNAL_DEV_TASK_NOT_READY"
    assert lock["terminal_states"]["risk_fail"] == "D2F_EXTERNAL_DEV_RISK_NOT_READY"
    assert lock["terminal_states"]["invalid"] == "INCONCLUSIVE_D2F_EXTERNAL_INVALID"
    assert len(lock["bindings"]) == 7
    again = PROT.freeze_screen_lock()
    assert again["written_at"] == lock["written_at"]
