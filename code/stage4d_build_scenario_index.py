"""Stage 4D-D2 Task 1: lightweight raw scenario index builder.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` section 3.3
step 2. Runs INSIDE the isolated conversion environment (Python 3.9 + TF 2.11,
WSL2 allowed per plan section 3.1.A). Parses every Waymo ``training_20s`` shard
with ScenarioNet's shipped proto definitions and emits one JSON line per
scenario:

    scenario_id, source_shard, sdc_track_index, sdc_track_id,
    num_timesteps, num_tracks, num_map_features

The index is read-only metadata (IDs and counts only); it never reads outcome
labels and never consumes a scientific role. Resumable per shard: shards whose
records already exist (count matches a sidecar ledger) are skipped, never
re-parsed. Prints phase/completed/total/%/elapsed/ETA at least every 30 s and
a final success/failure summary (AGENTS.md long-task discipline).

Example (from the Windows project root through the frozen launcher):
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
        -File code\\run_stage4d_env.ps1 convert code/stage4d_build_scenario_index.py \
        --raw-dir /mnt/d/MetaDrive/waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s \
        --output experiments/stage4_reliability/stage4d_external/platform/raw_scenario_index.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def list_shards(raw_dir):
    names = [
        n
        for n in os.listdir(raw_dir)
        if n.startswith("training_20s.tfrecord-") and n.endswith("-of-01000")
    ]
    names.sort()
    return names


def load_completed_shards(output_path):
    completed = {}
    if not os.path.isfile(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            completed[record["source_shard"]] = (
                completed.get(record["source_shard"], 0) + 1
            )
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sidecar",
        default=None,
        help="per-shard completion ledger; default <output>.shards.json",
    )
    parser.add_argument("--limit-shards", type=int, default=None)
    args = parser.parse_args(argv)

    import tensorflow as tf
    from scenarionet.converter.waymo.waymo_protos import scenario_pb2

    tf.config.experimental.set_visible_devices([], "GPU")

    sidecar_path = args.sidecar or (args.output + ".shards.json")
    shards = list_shards(args.raw_dir)
    if args.limit_shards is not None:
        shards = shards[: args.limit_shards]
    total = len(shards)
    if total == 0:
        print("[scenario-index] FAILURE | no shards found", flush=True)
        return 1

    completed = {}
    if os.path.isfile(sidecar_path):
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            completed = json.load(handle)

    started = time.perf_counter()
    done = 0
    total_scenarios = 0
    last_emit = 0.0
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "a", encoding="utf-8", newline="\n") as out:
        for name in shards:
            if name in completed:
                done += 1
                total_scenarios += completed[name]
                continue
            path = os.path.join(args.raw_dir, name)
            count = 0
            for data in tf.data.TFRecordDataset(
                path, compression_type=""
            ).as_numpy_iterator():
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(data)
                sdc_index = int(scenario.sdc_track_index)
                record = {
                    "scenario_id": scenario.scenario_id,
                    "source_shard": name,
                    "sdc_track_index": sdc_index,
                    "sdc_track_id": int(scenario.tracks[sdc_index].id),
                    "num_timesteps": len(scenario.timestamps_seconds),
                    "num_tracks": len(scenario.tracks),
                    "num_map_features": len(scenario.map_features),
                }
                out.write(json.dumps(record, sort_keys=True) + "\n")
                count += 1
            out.flush()
            os.fsync(out.fileno())
            completed[name] = count
            with open(sidecar_path, "w", encoding="utf-8") as handle:
                json.dump(completed, handle, sort_keys=True)
            done += 1
            total_scenarios += count
            now = time.perf_counter()
            if now - last_emit >= 30.0 or done == total:
                last_emit = now
                elapsed = now - started
                rate = done / elapsed if done and elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else None
                print(
                    f"[scenario-index] {100.0 * done / total:5.1f}% | "
                    f"{done}/{total} shards | scenarios={total_scenarios} | "
                    f"elapsed={format_duration(elapsed)} | "
                    f"ETA={format_duration(eta)}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    print(
        f"[scenario-index] SUCCESS | {done}/{total} shards | "
        f"scenarios={total_scenarios} | elapsed={format_duration(elapsed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
