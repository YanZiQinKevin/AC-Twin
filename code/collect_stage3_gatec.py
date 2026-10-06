"""Stage 3 Gate C holdout collector.

Reuses the Stage 2 collection engine (``collect_stage2_holdout.collect_stage2``)
so the twin-world identity, determinism and diagnostics infrastructure is NOT
duplicated. Only Stage 3-specific concerns live here: per-holdout duration
sets, the sealed Stage 2 holdout rejection, zero-overlap seed schedules, and
the Stage 3 joint-lock binding.

This is a manually-started long task. The two formal holdouts are collected only
after ``Gate R_dev + Gate T_dev + Gate C_dev + power`` all pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402

import collect_stage2_holdout as _engine  # noqa: E402
import ac_twin.stage2_protocol as _sp2  # noqa: E402
from ac_twin import stage3_protocol as s3p  # noqa: E402
from ac_twin.stage3_data import (  # noqa: E402
    EXPECTED_TRAIN_DURATIONS,
    UNSEEN_DURATION,
    assert_not_sealed_stage2,
)

HOLDOUT_DURATIONS = {
    s3p.GATEC_ID_NAME: sorted(EXPECTED_TRAIN_DURATIONS),
    s3p.GATEC_UNSEEN_NAME: [UNSEEN_DURATION],
}


def validate_stage3_collection_config(config: Mapping[str, Any], holdout: str) -> None:
    """Validate a Stage 3 collection config for one formal holdout."""
    if holdout not in HOLDOUT_DURATIONS:
        raise ValueError(f"unknown Stage 3 holdout: {holdout}")
    s3p.validate_stage3_config(config)

    expected = HOLDOUT_DURATIONS[holdout]
    durations = sorted(int(d) for d in config.get("intervention_durations", []))
    if durations != expected:
        raise ValueError(
            f"{holdout} intervention_durations must be {expected}, got {durations}"
        )

    # The collection output path must never be the sealed Stage 2 holdout.
    for key in ("data_output", "stage1_source_data"):
        if key in config:
            assert_not_sealed_stage2(str(config[key]))


def _write_holdout_seed_schedule(config: Mapping[str, Any], holdout: str) -> np.ndarray:
    """Materialise seed_schedule.npy for the holdout from the joint lock."""
    output_dir = _engine.sp._resolve(config["output_dir"])
    historical = s3p.scan_historical_episode_seeds(ROOT / "data")
    size = int(config.get("seed_schedule_size", config.get("max_attempts", 18000)))
    schedule = _sp2.make_seed_schedule(int(config["collection_seed"]), size=size)
    _sp2.validate_seed_schedule(schedule, historical)
    np.save(output_dir / "seed_schedule.npy", schedule)
    return schedule


def collect_stage3(
    config: Mapping[str, Any],
    holdout: str,
    *,
    resume: bool = False,
    branch_rollout=None,
    probe_fn=None,
    stop_after_accepted: int | None = None,
) -> dict[str, Any]:
    """Collect one Stage 3 formal holdout via the shared Stage 2 engine.

    The engine's Stage 2 protocol hooks are shimmed to the Stage 3 joint lock
    for the duration of the call, then restored. The running config must match
    the locked configuration exactly.
    """
    validate_stage3_collection_config(config, holdout)
    output_dir = _engine.sp._resolve(config["output_dir"])
    lock = s3p.verify_joint_lock(output_dir)

    _write_holdout_seed_schedule(config, holdout)

    original_validate = _engine.sp.validate_stage2_config
    original_verify = _engine.sp.verify_protocol_lock

    def _stage3_validate(cfg: Mapping[str, Any]) -> None:
        validate_stage3_collection_config(cfg, holdout)

    def _stage3_verify(_output_dir):
        # Adapt the Stage 3 joint lock to the keys the engine reads.
        return {
            "config_signature": lock["config_signature"],
            "config_sha256": config.get("config_sha256", lock["config_signature"]),
        }

    _engine.sp.validate_stage2_config = _stage3_validate
    _engine.sp.verify_protocol_lock = _stage3_verify
    try:
        return _engine.collect_stage2(
            config,
            resume=resume,
            branch_rollout=branch_rollout,
            probe_fn=probe_fn,
            config_path=None,
            stop_after_accepted=stop_after_accepted,
        )
    finally:
        _engine.sp.validate_stage2_config = original_validate
        _engine.sp.verify_protocol_lock = original_verify


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 Gate C holdout collector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--holdout", required=True, choices=sorted(HOLDOUT_DURATIONS))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(_engine.sp._resolve(args.config).read_text(encoding="utf-8"))
    receipt = collect_stage3(config, args.holdout, resume=args.resume)
    print(f"[stage3-collect] holdout={args.holdout} accepted="
          f"{receipt.get('accepted_pairs', '?')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
