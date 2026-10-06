"""Stage 4D-D2 (MetaDrive external replication) protocol: Task 1 preflight layer.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 2, 3.2,
4.3 and 12.2. This module implements ONLY read-only verification and resumable
inventory work:

* frozen source-chain verification (plan section 2.1, fail-closed on drift);
* external platform verification (plan section 2.2: MetaDrive/ScenarioNet commits,
  clean trees, version hash; Waymo shard count/bytes/numbering/temp files);
* resumable raw-dataset inventory with a filename-sorted Merkle root
  (plan section 3.2); and
* the deterministic scenario-role rank rule (plan section 4.3), a pure function
  used by later tasks and pinned by blocking tests now.

It never modifies frozen upstream artifacts and never consumes any one-shot role.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ac_twin.progress import ProgressReporter
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external/config.json"
)

# Plan section 15 terminal states reachable from Task 1.
STATUS_SOURCE_INVALID = "INCONCLUSIVE_STAGE4D_EXTERNAL_SOURCE_INVALID"


class PreflightFailure(RuntimeError):
    """A blocking preflight invariant failed; carries the plan-15 terminal state."""

    def __init__(self, status: str, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Plan section 4.3: deterministic scenario-role rank (pure, future-blind).
# ---------------------------------------------------------------------------


def role_rank_key(scenario_id: str, role_salt: str) -> str:
    """SHA256 hex of ``role_salt + scenario_id`` (plan section 4.3 ordering key)."""
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    return hashlib.sha256((role_salt + scenario_id).encode("utf-8")).hexdigest()


def rank_scenario_ids(
    scenario_ids: Sequence[str], role_salt: str
) -> list[str]:
    """Return scenario IDs sorted by the frozen role-rank hash (stable, unique)."""
    ids = list(scenario_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("scenario_ids must be globally unique before role ranking")
    return sorted(ids, key=lambda sid: (role_rank_key(sid, role_salt), sid))


# ---------------------------------------------------------------------------
# Plan section 2.1: frozen source chain (fail-closed on any drift).
# ---------------------------------------------------------------------------


def verify_source_chain(config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every plan-2.1 hash; drift or missing files fail closed."""
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    frozen: Mapping[str, str] = config["source_chain"]
    for rel_path, expected in sorted(frozen.items()):
        path = PROJECT_ROOT / rel_path
        entry: dict[str, Any] = {
            "path": rel_path,
            "expected_sha256": expected,
            "mode": "frozen_compare",
        }
        if not path.is_file():
            entry["status"] = "MISSING"
            failures.append(rel_path)
        else:
            actual = sha256_file(path)
            entry["actual_sha256"] = actual
            entry["status"] = "MATCH" if actual == expected else "DRIFT"
            if actual != expected:
                failures.append(rel_path)
        entries.append(entry)
    for rel_path in sorted(config.get("source_chain_bind_only", [])):
        path = PROJECT_ROOT / rel_path
        entry = {"path": rel_path, "mode": "hash_bind"}
        if not path.is_file():
            entry["status"] = "MISSING"
            failures.append(rel_path)
        else:
            entry["actual_sha256"] = sha256_file(path)
            entry["status"] = "BOUND"
        entries.append(entry)
    if failures:
        raise PreflightFailure(
            STATUS_SOURCE_INVALID,
            "source-chain failure: " + ", ".join(sorted(failures)),
        )
    return {"entries": entries, "all_match": True}


# ---------------------------------------------------------------------------
# Plan section 2.2: external platform verification (read-only).
# ---------------------------------------------------------------------------


def _git(dir_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(dir_path), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _verify_repo(dir_path: Path, expected_commit: str, label: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"label": label, "dir": str(dir_path)}
    if not dir_path.is_dir():
        raise PreflightFailure(STATUS_SOURCE_INVALID, f"{label} dir missing: {dir_path}")
    head = _git(dir_path, "rev-parse", "HEAD")
    dirty = _git(dir_path, "status", "--porcelain")
    entry["head"] = head
    entry["expected_commit"] = expected_commit
    entry["head_match"] = head == expected_commit
    entry["worktree_clean"] = dirty == ""
    if not entry["head_match"] or not entry["worktree_clean"]:
        raise PreflightFailure(
            STATUS_SOURCE_INVALID,
            f"{label} repo state invalid: head={head} dirty={bool(dirty)}",
        )
    return entry


def verify_waymo_raw(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan 3.2 acceptance items 1-3: shard count, numbering, bytes, temp files."""
    raw_dir = Path(config["waymo_raw_dir"])
    prefix = config["waymo_shard_prefix"]
    suffix = config["waymo_shard_suffix"]
    expected_count = int(config["waymo_shard_count"])
    expected_bytes = int(config["waymo_total_bytes"])
    if not raw_dir.is_dir():
        raise PreflightFailure(STATUS_SOURCE_INVALID, f"waymo raw dir missing: {raw_dir}")
    files = [p for p in raw_dir.iterdir() if p.is_file()]
    shard_nums: list[int] = []
    shard_files: list[Path] = []
    temp_files: list[str] = []
    other_files: list[str] = []
    for path in files:
        name = path.name
        if name.startswith(prefix) and name.endswith(suffix):
            middle = name[len(prefix) : len(name) - len(suffix)]
            if middle.isdigit():
                shard_nums.append(int(middle))
                shard_files.append(path)
                continue
        if name.endswith((".gstmp", ".tmp")):
            temp_files.append(name)
        else:
            other_files.append(name)
    total_bytes = sum(p.stat().st_size for p in shard_files)
    unique_nums = sorted(set(shard_nums))
    gaps = [n for n in range(expected_count) if n not in set(shard_nums)]
    entry = {
        "shard_count": len(shard_files),
        "unique_numbers": len(unique_nums),
        "numbering_min": unique_nums[0] if unique_nums else None,
        "numbering_max": unique_nums[-1] if unique_nums else None,
        "numbering_gaps": len(gaps),
        "duplicates": len(shard_nums) - len(unique_nums),
        "total_bytes": total_bytes,
        "temp_files": len(temp_files),
        "other_files": len(other_files),
    }
    ok = (
        len(shard_files) == expected_count
        and len(unique_nums) == expected_count
        and unique_nums == list(range(expected_count))
        and total_bytes == expected_bytes
        and not temp_files
        and not other_files
    )
    entry["accepted"] = ok
    if not ok:
        raise PreflightFailure(STATUS_SOURCE_INVALID, f"waymo raw invalid: {entry}")
    return entry


def verify_platform(config: Mapping[str, Any]) -> dict[str, Any]:
    """Plan 2.2 external sources + 3.2 raw acceptance (items 1-3 only)."""
    metadrive = _verify_repo(
        Path(config["metadrive_dir"]), config["metadrive_commit"], "metadrive"
    )
    version_py = Path(config["metadrive_dir"]) / "metadrive" / "version.py"
    version_hash = sha256_file(version_py)
    metadrive["version_py_sha256"] = version_hash
    metadrive["version_py_match"] = (
        version_hash == config["metadrive_version_py_sha256"]
    )
    if not metadrive["version_py_match"]:
        raise PreflightFailure(
            STATUS_SOURCE_INVALID, "metadrive version.py hash drift"
        )
    scenarionet = _verify_repo(
        Path(config["scenarionet_dir"]), config["scenarionet_commit"], "scenarionet"
    )
    waymo = verify_waymo_raw(config)
    return {"metadrive": metadrive, "scenarionet": scenarionet, "waymo_raw": waymo}


# ---------------------------------------------------------------------------
# Task 1 preflight entry: writes the create-once receipt.
# ---------------------------------------------------------------------------


def run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute Task-1 verification and write the create-once preflight receipt."""
    receipt = {
        "plan_id": config["plan_id"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_chain": verify_source_chain(config),
        "platform": verify_platform(config),
        "status": "PREFLIGHT_PASS",
    }
    out_path = (
        PROJECT_ROOT / config["run_dir"] / "platform" / "preflight_receipt.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_once_json(out_path, receipt)
    return receipt


# ---------------------------------------------------------------------------
# Plan 3.1: environment lock (after the one-shard conversion smoke passes).
# ---------------------------------------------------------------------------


def _run_capture(cmd: Sequence[str]) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout.strip()


def _pip_freeze_sim_env(config: Mapping[str, Any]) -> str:
    return _run_capture(
        [config["sim_env_python_exe"], "-m", "pip", "freeze"]
    )


def _pip_freeze_conversion_env(config: Mapping[str, Any]) -> str:
    return _run_capture(
        [
            "wsl.exe",
            "-d",
            config["conversion_wsl_distribution"],
            "--",
            config["conversion_env_python_exe"],
            "-m",
            "pip",
            "freeze",
        ]
    )


def _env_python_report(cmd_prefix: Sequence[str]) -> dict[str, Any]:
    payload = _run_capture(
        [
            *cmd_prefix,
            "-c",
            (
                "import platform,sys,json;"
                "print(json.dumps({'python':platform.python_version(),"
                "'implementation':platform.python_implementation(),"
                "'platform':platform.platform()}))"
            ),
        ]
    )
    return json.loads(payload.strip().splitlines()[-1])


def freeze_environment_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze both isolated environments into ``platform_environment_lock.json``.

    Plan 3.1: runs only after the one-shard conversion smoke passed. Create-once
    in the project platform dir AND copied to the platform env_locks dir.
    """
    platform_dir = PROJECT_ROOT / config["run_dir"] / "platform"
    smoke_receipt = platform_dir / "conversion_smoke_receipt.json"
    root_receipt = platform_dir / "raw_dataset_root_receipt.json"
    for required in (smoke_receipt, root_receipt):
        if not required.is_file():
            raise PreflightFailure(
                STATUS_SOURCE_INVALID,
                f"environment lock requires {required.name} first",
            )
    sim_report = _env_python_report([config["sim_env_python_exe"]])
    conv_report = _env_python_report(
        [
            "wsl.exe",
            "-d",
            config["conversion_wsl_distribution"],
            "--",
            config["conversion_env_python_exe"],
        ]
    )
    nvidia = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    torch_report = _run_capture(
        [
            config["sim_env_python_exe"],
            "-c",
            (
                "import torch,json;"
                "print(json.dumps({'torch':torch.__version__,"
                "'cuda_build':torch.version.cuda,"
                "'cuda_available':torch.cuda.is_available()}))"
            ),
        ]
    )
    converter_files = {
        "scenarionet_converter_waymo_utils": Path(config["scenarionet_dir"])
        / "scenarionet"
        / "converter"
        / "waymo"
        / "utils.py",
        "scenarionet_waymo_scenario_pb2": Path(config["scenarionet_dir"])
        / "scenarionet"
        / "converter"
        / "waymo"
        / "waymo_protos"
        / "scenario_pb2.py",
        "metadrive_version_py": Path(config["metadrive_dir"])
        / "metadrive"
        / "version.py",
    }
    converter_hashes = {
        label: sha256_file(path) for label, path in sorted(converter_files.items())
    }
    lock = {
        "plan_id": config["plan_id"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sim_env": {
            "name": config["sim_env_name"],
            "python_executable": config["sim_env_python_exe"],
            "python_report": sim_report,
            "pip_freeze": _pip_freeze_sim_env(config).splitlines(),
            "torch_report": json.loads(torch_report.strip().splitlines()[-1]),
        },
        "conversion_env": {
            "name": config["conversion_env_name"],
            "wsl_distro": config["conversion_wsl_distribution"],
            "python_executable": config["conversion_env_python_exe"],
            "python_report": conv_report,
            "pip_freeze": _pip_freeze_conversion_env(config).splitlines(),
        },
        "gpu": nvidia,
        "repos": {
            "metadrive_commit": config["metadrive_commit"],
            "scenarionet_commit": config["scenarionet_commit"],
        },
        "converter_source_hashes": converter_hashes,
        "raw_dataset_root_receipt": json.loads(
            root_receipt.read_text(encoding="utf-8")
        ),
        "conversion_smoke_receipt_sha256": sha256_file(smoke_receipt),
        "status": "ENVIRONMENT_LOCKED",
    }
    out_path = platform_dir / "platform_environment_lock.json"
    write_once_json(out_path, lock)
    env_locks_dir = Path(config["env_locks_dir"])
    env_locks_dir.mkdir(parents=True, exist_ok=True)
    mirror = env_locks_dir / "platform_environment_lock.json"
    mirror.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    return lock


# ---------------------------------------------------------------------------
# Plan 3.2: resumable raw inventory with filename-sorted Merkle root.
# ---------------------------------------------------------------------------


def _merkle_root(hex_digests: Sequence[str]) -> str:
    """SHA256 Merkle root over sorted leaf digests; deterministic, order-free."""
    if not hex_digests:
        raise ValueError("cannot hash an empty inventory")
    level = [bytes.fromhex(d) for d in hex_digests]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def _load_inventory_records(inventory_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not inventory_path.is_file():
        return records
    with open(inventory_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["relative_path"]] = record
    return records


def run_raw_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every shard into ``raw_dataset_inventory.jsonl`` (resumable).

    Resume rule: a shard already recorded with the same byte size is kept
    (its stored SHA256 is trusted; shards are immutable raw inputs). Completed
    inventories are verified, never recomputed.
    """
    raw_dir = Path(config["waymo_raw_dir"])
    prefix = config["waymo_shard_prefix"]
    suffix = config["waymo_shard_suffix"]
    expected_count = int(config["waymo_shard_count"])
    platform_dir = PROJECT_ROOT / config["run_dir"] / "platform"
    platform_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = platform_dir / "raw_dataset_inventory.jsonl"
    root_receipt_path = platform_dir / "raw_dataset_root_receipt.json"

    shards = sorted(
        (
            p
            for p in raw_dir.iterdir()
            if p.is_file()
            and p.name.startswith(prefix)
            and p.name.endswith(suffix)
        ),
        key=lambda p: p.name,
    )
    if len(shards) != expected_count:
        raise PreflightFailure(
            STATUS_SOURCE_INVALID,
            f"shard count {len(shards)} != {expected_count}",
        )

    records = _load_inventory_records(inventory_path)
    remaining = [
        p
        for p in shards
        if not (
            p.name in records and records[p.name]["bytes"] == p.stat().st_size
        )
    ]
    progress = ProgressReporter(
        total=len(shards),
        every=25,
        prefix="raw-inventory",
        heartbeat_seconds=30.0,
    )
    done = len(shards) - len(remaining)
    progress.update(done, details="resumed" if done else "", force=True)
    with open(inventory_path, "a", encoding="utf-8", newline="\n") as handle:
        for path in remaining:
            size = path.stat().st_size
            record = {
                "relative_path": path.name,
                "bytes": size,
                "sha256": sha256_file(path),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records[path.name] = record
            done += 1
            line = progress.heartbeat(done) or progress.update(done)
            if line is None:
                continue

    ordered = [records[p.name] for p in shards]
    total_bytes = sum(r["bytes"] for r in ordered)
    if total_bytes != int(config["waymo_total_bytes"]):
        raise PreflightFailure(
            STATUS_SOURCE_INVALID,
            f"inventory bytes {total_bytes} != {config['waymo_total_bytes']}",
        )
    leaf_hashes = [r["sha256"] for r in ordered]
    root_hash = _merkle_root(leaf_hashes)
    receipt = {
        "plan_id": config["plan_id"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shard_count": len(ordered),
        "total_bytes": total_bytes,
        "leaf_order": "filename_sorted",
        "merkle_root_sha256": root_hash,
        "inventory_file": "raw_dataset_inventory.jsonl",
        "inventory_sha256": sha256_file(inventory_path),
    }
    write_once_json(root_receipt_path, receipt)
    progress.update(len(shards), details="done", force=True)
    return receipt
