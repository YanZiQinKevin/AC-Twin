"""Stage 4D-D2 Task 1: one-shard ScenarioNet conversion smoke.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections
3.3 (step 1) and 12.2 (item 4). Runs INSIDE the isolated conversion environment
(``ac_twin_waymo_convert_py39``, Python 3.9 + TF 2.11; WSL2 allowed per plan
section 3.1.A). Converts the fixed shard ``00000`` with ScenarioNet's frozen
converter, then verifies the outputs load and records a receipt.

The smoke output directory is engineering-only (``conversion_smoke_shard00000``
under the platform root); it is NOT the scientific converted database and never
consumes a scenario role. The receipt is create-once.

Example (from the Windows project root through the frozen launcher):
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
        -File code\\run_stage4d_env.ps1 convert code/stage4d_conversion_smoke.py \
        --raw-dir /mnt/d/MetaDrive/waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s \
        --output-dir /root/ac_twin_stage4d_conversion_smoke_shard00000 \
        --receipt /root/ac_twin_stage4d_conversion_smoke_receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

SHARD_NAME = "training_20s.tfrecord-00000-of-01000"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_duration(seconds):
    total = int(round(max(0.0, seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    shard_path = os.path.join(args.raw_dir, SHARD_NAME)
    if not os.path.isfile(shard_path):
        print(f"[conversion-smoke] FAILURE | missing shard {shard_path}", flush=True)
        return 1
    if os.path.isfile(args.receipt):
        print(
            f"[conversion-smoke] FAILURE | receipt already exists (create-once): "
            f"{args.receipt}",
            flush=True,
        )
        return 1

    import tensorflow as tf
    from scenarionet.converter.utils import write_to_directory
    from scenarionet.converter.waymo.utils import (
        convert_waymo_scenario,
        preprocess_waymo_scenarios,
    )

    tf.config.experimental.set_visible_devices([], "GPU")

    print(
        f"[conversion-smoke] phase=convert | shard={SHARD_NAME} | "
        f"output={args.output_dir}",
        flush=True,
    )
    write_to_directory(
        convert_func=convert_waymo_scenario,
        scenarios=[shard_path],
        output_path=args.output_dir,
        dataset_version="v1.2",
        dataset_name="waymo",
        overwrite=True,
        num_workers=1,
        preprocess=preprocess_waymo_scenarios,
    )
    print(
        f"[conversion-smoke] phase=verify | elapsed={format_duration(time.perf_counter() - started)}",
        flush=True,
    )

    pkls = []
    for root, _dirs, files in os.walk(args.output_dir):
        for name in files:
            if name.endswith(".pkl"):
                pkls.append(os.path.join(root, name))
    pkls.sort()
    if not pkls:
        print("[conversion-smoke] FAILURE | converter produced no .pkl", flush=True)
        return 1

    with open(pkls[0], "rb") as handle:
        first = pickle.load(handle)
    if not isinstance(first, dict):
        print(
            f"[conversion-smoke] FAILURE | first pkl is {type(first).__name__}, "
            "expected dict scenario description",
            flush=True,
        )
        return 1
    keys = sorted(first.keys())

    receipt = {
        "plan_id": "AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shard": SHARD_NAME,
        "shard_sha256": sha256_file(shard_path),
        "output_dir": args.output_dir,
        "num_pkl_files": len(pkls),
        "first_pkl": os.path.basename(pkls[0]),
        "first_pkl_sha256": sha256_file(pkls[0]),
        "first_pkl_keys": keys,
        "all_pkl_sha256_merkle_leaves_sorted": [sha256_file(p) for p in pkls],
        "status": "CONVERSION_SMOKE_PASS",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.receipt)), exist_ok=True)
    tmp = args.receipt + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, args.receipt)
    print(
        f"[conversion-smoke] SUCCESS | pkls={len(pkls)} | keys={len(keys)} | "
        f"elapsed={format_duration(time.perf_counter() - started)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
