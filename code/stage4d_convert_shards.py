"""Stage 4D-D2 Task 2: resumable conversion of role-referenced Waymo shards.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` section 3.3
(steps 4-7). Runs INSIDE the frozen WSL conversion environment via the unique
entry point:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File code\\run_stage4d_env.ps1 `
        convert code/stage4d_convert_shards.py

Contract:
* converts ONLY shards referenced by the frozen scenario split pools
  (``platform/scenario_split_pools.jsonl``); never re-downloads raw data;
* per-batch staging on the WSL ext4 root (``AC_TWIN_D2_CONVERSION_STAGING_ROOT``),
  because DrvFS directory renames are unreliable (plan 3.1);
* a batch is published only after per-file SHA256 verification of the staged
  pkls; publish = copy to the flat DB dir on /mnt/d + re-verify size+hash, then
  staging is removed; per-scenario source shard attribution comes from the pkl
  ``metadata.source_file`` (content-authoritative, not filename);
* resume is per-shard: a shard is done iff a SHARD_COMPLETE inventory row exists
  and every listed file is present in the DB with matching bytes and SHA256
  (plan 3.3 step 6: hash-based resume, completed outputs never overwritten);
* append-only ``conversion_inventory.jsonl`` (in the DB dir) is the resume
  anchor; no atomic rename is used on /mnt/d (plain append / full-file writes);
* DB-level ``dataset_summary.pkl``/``dataset_mapping.pkl`` are derived artifacts
  rebuilt from per-batch summary fragments by ``--finalize`` (also run
  automatically at the end of a complete pass);
* prints phase/completed/total/%/elapsed/ETA at least every 30 s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin.progress import ProgressReporter, format_duration  # noqa: E402

SHARD_PREFIX = "training_20s.tfrecord-"
SHARD_SUFFIX = "-of-01000"
BATCH_SHARDS = 25
NUM_WORKERS = 6
PKL_PREFIX = "sd_waymo_v1.2_"
DB_DIR = Path("/mnt/d/MetaDrive/scenarionet_waymo_training_20s")
RAW_DIR = Path(
    "/mnt/d/MetaDrive/waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s"
)
POOLS_PATH = (
    PROJECT_ROOT
    / "experiments/stage4_reliability/stage4d_external/platform/scenario_split_pools.jsonl"
)
INVENTORY_PATH = DB_DIR / "conversion_inventory.jsonl"
FRAGMENT_DIR = DB_DIR / "_summary_fragments"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_referenced_shards() -> tuple[dict[str, int], set[str]]:
    """(shard -> referenced scenario count, referenced scenario IDs) from pools."""
    counts: dict[str, int] = {}
    ids: set[str] = set()
    with open(POOLS_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            counts[row["source_shard"]] = counts.get(row["source_shard"], 0) + 1
            ids.add(row["scenario_id"])
    if not counts:
        raise RuntimeError(f"no referenced shards in {POOLS_PATH}")
    return dict(sorted(counts.items())), ids


def load_completed_shards() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not INVENTORY_PATH.is_file():
        return done
    with open(INVENTORY_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "SHARD_COMPLETE":
                done[row["shard"]] = row
    return done


def shard_is_done(row: dict) -> bool:
    """A recorded shard counts as done only if every file still matches on disk."""
    for entry in row["files"]:
        target = DB_DIR / entry["file"]
        if not target.is_file() or target.stat().st_size != entry["bytes"]:
            return False
        if sha256_file(target) != entry["sha256"]:
            return False
    return True


def append_inventory(row: dict) -> None:
    with open(INVENTORY_PATH, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def convert_batch(shard_names: list[str], staging_root: Path, batch_tag: str) -> Path:
    """Run the frozen ScenarioNet converter over one batch of shards (WSL/ext4)."""
    import tensorflow as tf
    from scenarionet.converter.utils import write_to_directory
    from scenarionet.converter.waymo.utils import (
        convert_waymo_scenario,
        preprocess_waymo_scenarios,
    )

    tf.config.experimental.set_visible_devices([], "GPU")
    staging = staging_root / batch_tag
    if staging.exists():
        shutil.rmtree(staging)
    shard_paths = [str(RAW_DIR / name) for name in shard_names]
    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=shard_paths,
        output_path=str(staging),
        dataset_version="v1.2",
        dataset_name="waymo",
        overwrite=True,
        num_workers=NUM_WORKERS,
        preprocess=preprocess_waymo_scenarios,
    )
    return staging


def publish_batch(
    staging: Path,
    batch_tag: str,
    batch_shards: set[str],
    referenced_ids: set[str],
) -> list[dict]:
    """Verify staged pkls (SHA256), copy flat into the DB, re-verify on disk.

    Per-scenario shard attribution uses the pkl ``metadata.source_file``.
    Returns one inventory row per source shard actually seen in this batch.
    Aborts BEFORE any staging cleanup on mismatch; re-publishing identical
    content (resume of a crashed publish) is accepted, differing content raises.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
    summary_fragment: dict = {}
    rows: dict[str, dict] = {}
    for pkl in sorted(staging.rglob(f"{PKL_PREFIX}*.pkl")):
        staged_sha = sha256_file(pkl)
        staged_bytes = pkl.stat().st_size
        with open(pkl, "rb") as handle:
            data = pickle.load(handle)
        meta = data.get("metadata", {})
        scenario_id = str(meta.get("scenario_id", pkl.name[len(PKL_PREFIX):-len(".pkl")]))
        source_shard = os.path.basename(str(meta.get("source_file", "")))
        if source_shard not in batch_shards:
            raise RuntimeError(
                f"converted scenario {scenario_id} claims source {source_shard} "
                f"outside batch {sorted(batch_shards)}; stop"
            )
        entry = {
            "scenario_id": scenario_id,
            "file": pkl.name,
            "bytes": staged_bytes,
            "sha256": staged_sha,
            "role_referenced": scenario_id in referenced_ids,
        }
        target = DB_DIR / pkl.name
        if target.exists():
            if target.stat().st_size != staged_bytes or sha256_file(target) != staged_sha:
                raise RuntimeError(
                    f"DB target exists with different content: {target}; stop"
                )
        else:
            shutil.copy2(pkl, target)
            if target.stat().st_size != staged_bytes or sha256_file(target) != staged_sha:
                raise RuntimeError(f"publish verification failed for {target}")
        row = rows.setdefault(
            source_shard,
            {"shard": source_shard, "files": [], "converted": 0,
             "converted_referenced": 0},
        )
        row["files"].append(entry)
        row["converted"] += 1
        row["converted_referenced"] += int(entry["role_referenced"])
        summary_fragment[pkl.name] = {
            "id": scenario_id,
            "scenario_id": scenario_id,
            "coordinate": meta.get("coordinate", "waymo"),
            "source_file": source_shard,
        }
    fragment_path = FRAGMENT_DIR / f"{batch_tag}.pkl"
    with open(fragment_path, "wb") as handle:
        pickle.dump(summary_fragment, handle)
    return [rows[name] for name in sorted(rows)]


def rebuild_db_summary() -> dict:
    """Rebuild DB-level dataset_summary/dataset_mapping from summary fragments.

    Plain full-file writes (no rename on /mnt/d); verified by reload. Derived
    artifacts: always regenerable from the inventory + fragments.
    """
    summary: dict = {}
    mapping: dict = {}
    for fragment in sorted(FRAGMENT_DIR.glob("*.pkl")):
        with open(fragment, "rb") as handle:
            fragment_dict = pickle.load(handle)
        for file_name, entry in fragment_dict.items():
            summary[file_name] = entry
            mapping[file_name] = ""
    if not summary:
        raise RuntimeError("no summary fragments found; nothing to finalize")
    for name, payload in (("dataset_summary.pkl", summary), ("dataset_mapping.pkl", mapping)):
        with open(DB_DIR / name, "wb") as handle:
            pickle.dump(payload, handle)
    with open(DB_DIR / "dataset_summary.pkl", "rb") as handle:
        reloaded = pickle.load(handle)
    if len(reloaded) != len(summary):
        raise RuntimeError("dataset_summary reload count mismatch")
    with open(DB_DIR / "dataset_mapping.pkl", "rb") as handle:
        reloaded_mapping = pickle.load(handle)
    if set(reloaded_mapping) != set(mapping):
        raise RuntimeError("dataset_mapping reload key mismatch")
    return {"scenarios": len(summary)}


def shard_tag(shard_name: str) -> str:
    return shard_name[len(SHARD_PREFIX):-len(SHARD_SUFFIX)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="only rebuild dataset_summary/dataset_mapping from fragments",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="engineering throttle: process at most N batches this run (resume-safe)",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    if args.finalize_only:
        summary = rebuild_db_summary()
        print(
            f"[convert] SUCCESS(finalize) | db_scenarios={summary['scenarios']} | "
            f"elapsed={format_duration(time.perf_counter() - started)}",
            flush=True,
        )
        return 0

    staging_root = Path(
        os.environ.get("AC_TWIN_D2_CONVERSION_STAGING_ROOT",
                       "/root/ac_twin_stage4d_conversion_work")
    )
    staging_root.mkdir(parents=True, exist_ok=True)

    referenced, referenced_ids = load_referenced_shards()
    shard_names = list(referenced.keys())
    print(
        f"[convert] phase=init | referenced_shards={len(shard_names)} | "
        f"batch={BATCH_SHARDS} workers={NUM_WORKERS} | db={DB_DIR}",
        flush=True,
    )

    done = load_completed_shards()
    todo = []
    for name in shard_names:
        row = done.get(name)
        if row is not None and shard_is_done(row):
            continue
        todo.append(name)
    completed = len(shard_names) - len(todo)
    print(f"[convert] phase=resume | done={completed} todo={len(todo)}", flush=True)

    progress = ProgressReporter(
        total=len(shard_names), every=1, prefix="convert", heartbeat_seconds=30.0
    )
    progress.update(completed, details="resumed" if completed else "", force=True)

    batches = [todo[i : i + BATCH_SHARDS] for i in range(0, len(todo), BATCH_SHARDS)]
    if args.max_batches is not None:
        batches = batches[: args.max_batches]
    for batch_idx, batch in enumerate(batches):
        batch_tag = f"batch_{shard_tag(batch[0])}_{shard_tag(batch[-1])}"
        print(
            f"[convert] phase=batch | {batch_idx + 1}/{len(batches)} | "
            f"shards={len(batch)} | {batch[0]}..{batch[-1]}",
            flush=True,
        )
        staging = convert_batch(batch, staging_root, batch_tag)
        rows = publish_batch(staging, batch_tag, set(batch), referenced_ids)
        rows_by_shard = {row["shard"]: row for row in rows}
        for name in batch:
            row = rows_by_shard.get(
                name, {"files": [], "converted": 0, "converted_referenced": 0}
            )
            append_inventory(
                {
                    "shard": name,
                    "referenced_scenarios": referenced.get(name, 0),
                    "converted_scenarios": row["converted"],
                    "converted_referenced": row["converted_referenced"],
                    "files": row["files"],
                    "status": "SHARD_COMPLETE",
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            completed += 1
            line = progress.heartbeat(completed) or progress.update(completed)
            if line is None:
                continue
        shutil.rmtree(staging, ignore_errors=True)

    progress.update(completed, details="batches done this run", force=True)
    if len(todo) > sum(len(b) for b in batches):
        print(
            f"[convert] PAUSED | max-batches throttle; resume with the same command | "
            f"remaining_shards={len(todo) - sum(len(b) for b in batches)} | "
            f"elapsed={format_duration(time.perf_counter() - started)}",
            flush=True,
        )
        return 0
    summary = rebuild_db_summary()
    print(
        f"[convert] SUCCESS | shards={len(shard_names)} | "
        f"db_scenarios={summary['scenarios']} | "
        f"elapsed={format_duration(time.perf_counter() - started)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
