"""Blocking tests for the Stage 4D-D2 Task-1 protocol layer.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 2,
3.2, 4.3, 5.1 and 12.7. These tests pin the frozen source-chain comparison, the
deterministic role-rank rule, the action-plan contract, the resumable raw
inventory and its Merkle root receipt. Live-workspace checks are limited to the
frozen source chain (plan-2.1 hashes must never drift); platform/repo state is
verified by the ``preflight`` command, not by this suite, to avoid brittle
snapshot failures of the kind already exempted on 2026-09-17/2026-09-20.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4d_external_protocol as proto  # noqa: E402

CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4d_external/config.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# --- plan 5.1: action-plan contract -------------------------------------------


def test_idle_plan_index_points_to_idle_3():
    config = _config()
    plans = [tuple(p) for p in config["action_plans"]]
    assert len(plans) == 10
    assert plans[config["idle_plan_index"]] == tuple(config["idle_plan_tuple"]) == (1, 3)
    assert plans == [
        (0, 1), (0, 3), (1, 1), (1, 3), (2, 1),
        (2, 3), (3, 1), (3, 3), (4, 1), (4, 3),
    ]


def test_action_weights_match_frozen_r2_weights():
    weights = _config()["action_weights"]
    assert weights == {
        "IDLE": 0.45,
        "FASTER": 0.18,
        "SLOWER": 0.18,
        "LANE_LEFT": 0.095,
        "LANE_RIGHT": 0.095,
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-12


# --- plan 4.3: deterministic role rank ----------------------------------------


def test_role_rank_key_is_salted_sha256():
    salt = _config()["role_salt"]
    assert salt == "AC-TWIN-D2-V0.1|"
    expected = hashlib.sha256((salt + "scenario-xyz").encode("utf-8")).hexdigest()
    assert proto.role_rank_key("scenario-xyz", salt) == expected


def test_rank_scenario_ids_stable_and_unique():
    salt = _config()["role_salt"]
    ids = [f"sid-{i:04d}" for i in range(100)]
    first = proto.rank_scenario_ids(ids, salt)
    assert first == proto.rank_scenario_ids(list(reversed(ids)), salt)
    assert sorted(first) == ids
    with pytest.raises(ValueError):
        proto.rank_scenario_ids(["a", "a"], salt)


def test_role_segments_are_disjoint_by_construction():
    """Plan 4.3: roles take consecutive segments of one hash ranking -> zero overlap."""
    salt = _config()["role_salt"]
    targets = _config()["role_targets"]
    total = sum(targets.values()) + 50  # reserve tail
    ranked = proto.rank_scenario_ids([f"s-{i:06d}" for i in range(total)], salt)
    cursor = 0
    seen: set[str] = set()
    for role, count in targets.items():
        segment = ranked[cursor : cursor + count]
        assert len(segment) == count
        assert not (set(segment) & seen)
        seen.update(segment)
        cursor += count


# --- plan 2.1: frozen source chain (live, must never drift) --------------------


def test_frozen_source_chain_matches_live_workspace():
    receipt = proto.verify_source_chain(_config())
    assert receipt["all_match"] is True
    modes = {e["path"]: e["mode"] for e in receipt["entries"]}
    assert modes[
        "experiments/stage4_reliability/stage4c_r2/development/feature_schema.json"
    ] == "hash_bind"


def test_source_chain_fails_closed_on_drift(tmp_path, monkeypatch):
    config = _config()
    config["source_chain"] = dict(config["source_chain"])
    key = sorted(config["source_chain"])[0]
    config["source_chain"][key] = "0" * 64
    with pytest.raises(proto.PreflightFailure) as excinfo:
        proto.verify_source_chain(config)
    assert excinfo.value.status == proto.STATUS_SOURCE_INVALID


def test_source_chain_fails_closed_on_missing():
    config = _config()
    config["source_chain"] = {"no/such/file.anywhere": "0" * 64}
    with pytest.raises(proto.PreflightFailure):
        proto.verify_source_chain(config)


# --- plan 3.2: waymo raw acceptance (synthetic) --------------------------------


def _make_shards(base: Path, count: int, prefix: str, suffix: str, size: int = 8):
    for i in range(count):
        (base / f"{prefix}{i:05d}{suffix}").write_bytes(b"x" * size)


def _synthetic_config(tmp_path: Path, count: int = 4, size: int = 8) -> dict:
    config = _config()
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_shards(raw, count, "training_20s.tfrecord-", "-of-01000", size)
    config["waymo_raw_dir"] = str(raw)
    config["waymo_shard_count"] = count
    config["waymo_total_bytes"] = count * size
    return config


def test_verify_waymo_raw_accepts_clean_layout(tmp_path):
    entry = proto.verify_waymo_raw(_synthetic_config(tmp_path))
    assert entry["accepted"] is True
    assert entry["shard_count"] == 4
    assert entry["numbering_gaps"] == 0


@pytest.mark.parametrize(
    "corrupt",
    ["gap", "temp_file", "extra_file", "wrong_bytes", "duplicate_prefix_odd_name"],
)
def test_verify_waymo_raw_rejects_dirty_layout(tmp_path, corrupt):
    config = _synthetic_config(tmp_path)
    raw = Path(config["waymo_raw_dir"])
    if corrupt == "gap":
        (raw / "training_20s.tfrecord-00002-of-01000").unlink()
    elif corrupt == "temp_file":
        (raw / "training_20s.tfrecord-00004-of-01000.gstmp").write_bytes(b"t")
    elif corrupt == "extra_file":
        (raw / "README.txt").write_text("x")
    elif corrupt == "wrong_bytes":
        config["waymo_total_bytes"] += 1
    elif corrupt == "duplicate_prefix_odd_name":
        (raw / "training_20s.tfrecord-abc-of-01000").write_bytes(b"zzzzzzzz")
    with pytest.raises(proto.PreflightFailure):
        proto.verify_waymo_raw(config)


# --- plan 3.2: resumable inventory + Merkle root --------------------------------


def test_merkle_root_deterministic_and_nonempty():
    leaves = [hashlib.sha256(bytes([i])).hexdigest() for i in range(5)]
    assert proto._merkle_root(leaves) == proto._merkle_root(list(leaves))
    assert proto._merkle_root(leaves) != proto._merkle_root(list(reversed(leaves)))
    with pytest.raises(ValueError):
        proto._merkle_root([])


def test_run_raw_inventory_resume_and_create_once(tmp_path, monkeypatch):
    config = _synthetic_config(tmp_path)
    monkeypatch.setattr(proto, "PROJECT_ROOT", tmp_path)
    first = proto.run_raw_inventory(config)
    inventory = tmp_path / config["run_dir"] / "platform" / "raw_dataset_inventory.jsonl"
    lines = inventory.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == config["waymo_shard_count"]
    expected_root = proto._merkle_root(
        [json.loads(line)["sha256"] for line in lines]
    )
    assert first["merkle_root_sha256"] == expected_root
    assert first["total_bytes"] == config["waymo_total_bytes"]
    # Second run: all records kept (resume), but the create-once root receipt
    # must refuse to be overwritten.
    with pytest.raises(FileExistsError):
        proto.run_raw_inventory(config)


def test_run_raw_inventory_recovers_partial(tmp_path, monkeypatch):
    config = _synthetic_config(tmp_path)
    monkeypatch.setattr(proto, "PROJECT_ROOT", tmp_path)
    platform_dir = tmp_path / config["run_dir"] / "platform"
    platform_dir.mkdir(parents=True)
    raw = Path(config["waymo_raw_dir"])
    first_shard = sorted(raw.iterdir(), key=lambda p: p.name)[0]
    partial = {
        "relative_path": first_shard.name,
        "bytes": first_shard.stat().st_size,
        "sha256": proto.sha256_file(first_shard),
    }
    (platform_dir / "raw_dataset_inventory.jsonl").write_text(
        json.dumps(partial) + "\n", encoding="utf-8"
    )
    receipt = proto.run_raw_inventory(config)
    assert receipt["shard_count"] == config["waymo_shard_count"]


# --- plan 3.1: environment lock preconditions ----------------------------------


def test_env_lock_requires_conversion_smoke_first(tmp_path, monkeypatch):
    """Plan 3.1.A: the lock freezes only AFTER the one-shard smoke passes."""
    config = _config()
    monkeypatch.setattr(proto, "PROJECT_ROOT", tmp_path)
    with pytest.raises(proto.PreflightFailure) as excinfo:
        proto.freeze_environment_lock(config)
    assert "conversion_smoke_receipt.json" in excinfo.value.detail


def test_env_lock_python_report_parsing():
    payload = proto._env_python_report(
        [
            "python",
        ]
    )
    assert set(payload) == {"python", "implementation", "platform"}
    assert payload["python"].count(".") == 2
