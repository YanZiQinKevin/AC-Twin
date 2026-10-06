"""Stage 3R one-shot development R-screen collector."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import collect_stage2_holdout as engine  # noqa: E402
from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin.stage2_protocol import validate_seed_schedule  # noqa: E402
from ac_twin.stage3r_screen import PLAN_ID, sha256_file, validate_rscreen_readiness  # noqa: E402


PURPOSE = "stage3r_rscreen_development_test"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def validate_output_path(config: Mapping[str, Any]) -> None:
    value = str(config.get("data_output", "")).replace("\\", "/").lower()
    if "/stage3r_rscreen/" not in f"/{value.strip('/')}/":
        raise ValueError("data_output must live in the dedicated stage3r_rscreen directory")
    s3d.assert_not_sealed_stage2(value)
    if "gatec" in value or "stage3a_plus" in value or "stage3b" in value:
        raise ValueError("R-screen output cannot target another stage or Gate C")


def validate_collection_config(config: Mapping[str, Any]) -> dict[str, object]:
    if config.get("plan_id") != PLAN_ID or config.get("purpose") != PURPOSE:
        raise ValueError("invalid Stage 3R collection identity")
    if config.get("analysis_status") != PURPOSE:
        raise ValueError("analysis_status must equal R-screen purpose")
    validate_output_path(config)
    readiness = validate_rscreen_readiness(_resolve(config["run_dir"]))
    if int(config["n_pairs"]) != int(readiness["n_pairs"]):
        raise ValueError("R-screen n_pairs differs from frozen readiness")
    if int(config["max_attempts"]) != int(config["n_pairs"]) * 15:
        raise ValueError("R-screen max_attempts must equal n_pairs*15")
    return readiness


def _historical(config: Mapping[str, Any]) -> np.ndarray:
    try:
        return s3d.scan_all_episode_seeds(
            _resolve(config["historical_data_root"]),
            exclude_substring="stage3r_rscreen",
        )
    except ValueError as exc:
        if "no historical episode seeds" in str(exc):
            return np.asarray([], dtype=np.int64)
        raise


def collect_rscreen(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    branch_rollout=None,
    probe_fn=None,
    stop_after_accepted: int | None = None,
) -> dict[str, Any]:
    config = dict(config)
    validate_collection_config(config)
    run_dir = _resolve(config["run_dir"])
    schedule = np.load(run_dir / "rscreen" / "rscreen_seed_schedule.npy").astype(np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != config["seed_schedule_sha256"]:
        raise ValueError("frozen R-screen schedule bytes changed")
    historical = _historical(config)
    validate_seed_schedule(schedule, historical)
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "seed_schedule.npy", schedule)

    original_validate = engine.sp.validate_stage2_config
    original_verify = engine.sp.verify_protocol_lock
    engine.sp.validate_stage2_config = lambda cfg: validate_collection_config(cfg)
    engine.sp.verify_protocol_lock = lambda _path: {
        "config_signature": engine.canonical_fingerprint(json.loads(json.dumps(config))),
        "config_sha256": sha256_file(run_dir / "rscreen" / "rscreen_config.json"),
    }
    try:
        receipt = engine.collect_stage2(
            config,
            resume=resume,
            branch_rollout=branch_rollout,
            probe_fn=probe_fn,
            config_path=None,
            stop_after_accepted=stop_after_accepted,
        )
    finally:
        engine.sp.validate_stage2_config = original_validate
        engine.sp.verify_protocol_lock = original_verify
    if receipt.get("partial"):
        return receipt
    data_path = _resolve(config["data_output"])
    with np.load(data_path, allow_pickle=True) as archive:
        episode_seed = np.asarray(archive["episode_seed"], dtype=np.int64)
        if episode_seed.size != int(config["n_pairs"]):
            raise RuntimeError("R-screen accepted episode count mismatch")
        if set(episode_seed.tolist()) & set(historical.tolist()):
            raise RuntimeError("R-screen data overlaps historical seeds")
        for field in (
            "object_history",
            "object_factual_future",
            "object_counterfactual_future",
        ):
            if not np.isfinite(archive[field]).all():
                raise RuntimeError(f"non-finite R-screen field: {field}")
    receipt_path = _resolve(config["collection_receipt"])
    published = dict(receipt)
    published.update(
        {
            "plan_id": PLAN_ID,
            "purpose": PURPOSE,
            "data_sha256": sha256_file(data_path),
            "historical_overlap_count": 0,
            "power_report_sha256": config["power_report_sha256"],
            "model_lock_sha256": config["model_lock_sha256"],
            "seed_schedule_sha256": config["seed_schedule_sha256"],
        }
    )
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(published, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(receipt_path)
    return published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Stage 3R R-screen")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    receipt = collect_rscreen(config, resume=bool(args.resume))
    print(
        f"[stage3r-rscreen] accepted={receipt.get('accepted', receipt.get('n_pairs'))} "
        f"partial={receipt.get('partial', False)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
