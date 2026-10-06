"""Stage 3A replacement development holdout collector (repair plan v0.1 §3.3).

Reuses the Stage 2 collection engine (``collect_stage2_holdout.collect_stage2``)
so checkpoint/resume, the rejection ledger, QA and atomic publish are NOT
duplicated. Only replacement-specific concerns live here: the frozen merge
generation contract, the power-frozen episode count, the hash chain to the
power report and seed schedule, the zero-overlap seed universe (which — unlike
Stage 2's own scan — INCLUDES the sealed Stage 2 holdout), forbidden output
locations, and the ``stage3a_replacement_development_test`` purpose.

The collector is rollout-injectable so unit tests run without highway_env.
This is a manually-started long task with a 30-second progress heartbeat.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import collect_stage2_holdout as _engine  # noqa: E402
from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage2_protocol import validate_seed_schedule  # noqa: E402

MERGE_GENERATION_CONTRACT = s3d.MERGE_GENERATION_CONTRACT

_REQUIRED_KEYS = {
    "experiment_name", "analysis_status", "purpose", "plan_id", "scene", "env_id",
    "n_pairs", "collection_seed", "seed_schedule_size", "max_attempts",
    "horizon", "history_len", "t_star_low", "t_star_high",
    "intervention_durations", "vehicles_count", "observed_vehicles_count",
    "duration", "simulation_frequency", "policy_frequency", "determinism_checks",
    "action_weights", "factual_suffix_steps", "output_dir", "data_output",
    "rejection_ledger", "collection_receipt", "historical_data_root",
    "power_report_sha256", "seed_schedule_sha256",
}

# Modules whose content hash is recorded in the collection receipt (v0.1 §3.3).
_CODE_FILES = (
    "collect_stage3a_replacement_dev.py",
    "collect_stage2_holdout.py",
    "ac_twin/stage3_data.py",
    "ac_twin/stage2_protocol.py",
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _logical(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


# --- config validation -------------------------------------------------------------


def validate_replacement_dev_config(config: Mapping[str, Any], *, run_dir: str | Path) -> None:
    """Validate the frozen replacement collection config; raise before any output.

    Binds the config to the on-disk power report and seed-schedule document, and
    refuses any generation-contract deviation or forbidden output location.
    """
    missing = _REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"replacement config missing keys: {sorted(missing)}")
    if "split" in config or "sample_limits" in config:
        raise ValueError("the replacement holdout forbids train/validation split keys; "
                         "every episode is an eval episode")
    if config["plan_id"] != s3d.REPAIR_PLAN_ID:
        raise ValueError(f"plan_id must be {s3d.REPAIR_PLAN_ID!r}")
    if (config["analysis_status"] != s3d.REPLACEMENT_DEV_PURPOSE
            or config["purpose"] != s3d.REPLACEMENT_DEV_PURPOSE):
        raise ValueError(
            f"purpose must be {s3d.REPLACEMENT_DEV_PURPOSE!r} — the replacement "
            "holdout never shares a purpose label with Stage 2 or Gate C")

    # The generation contract replicates the old merge data EXACTLY (v0.1 §1.7).
    for key, expected in MERGE_GENERATION_CONTRACT.items():
        if key == "max_attempt_multiplier":
            continue  # enforced via max_attempts below
        actual = config.get(key)
        if isinstance(expected, dict):
            ok = (isinstance(actual, Mapping)
                  and set(actual) == set(expected)
                  and all(float(actual[k]) == float(v) for k, v in expected.items()))
        elif isinstance(expected, list):
            ok = sorted(int(v) for v in (actual or [])) == sorted(int(v) for v in expected)
        elif isinstance(expected, int):
            ok = int(actual) == expected
        else:
            ok = actual == expected
        if not ok:
            raise ValueError(
                f"{key} must equal the frozen merge generation contract value "
                f"{expected!r}, got {actual!r}")

    n_pairs = int(config["n_pairs"])
    if int(config["max_attempts"]) != n_pairs * int(MERGE_GENERATION_CONTRACT["max_attempt_multiplier"]):
        raise ValueError("max_attempts must equal n_pairs * max_attempt_multiplier")
    if int(config["seed_schedule_size"]) != int(config["max_attempts"]):
        raise ValueError("seed_schedule_size must equal max_attempts")

    # --- bind to the frozen power report (tamper-evident) -------------------------
    run_dir = _resolve(run_dir)
    report_path = run_dir / "power" / "power_report.json"
    if not report_path.is_file():
        raise ValueError(f"power_report missing: {report_path}")
    if sha256_file(report_path) != config["power_report_sha256"]:
        raise ValueError("power_report sha256 mismatch — the power report was "
                         "edited after the freeze")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") == s3d.REPLACEMENT_DEV_INFEASIBLE:
        raise ValueError(f"{s3d.REPLACEMENT_DEV_INFEASIBLE}: collection is forbidden")
    if int(report.get("selected_episode_count", -1)) != n_pairs:
        raise ValueError(
            f"n_pairs ({n_pairs}) must equal the power-frozen selected_episode_count "
            f"({report.get('selected_episode_count')})")

    # --- bind to the frozen seed schedule document --------------------------------
    schedule_path = run_dir / "replacement_dev" / "replacement_dev_seed_schedule.json"
    if not schedule_path.is_file():
        raise ValueError(f"replacement seed schedule missing: {schedule_path}")
    schedule_doc = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule_doc.get("power_report_sha256") != config["power_report_sha256"]:
        raise ValueError("seed schedule document binds a different power_report")
    if schedule_doc.get("schedule_sha256") != config["seed_schedule_sha256"]:
        raise ValueError("seed_schedule_sha256 mismatch between config and schedule document")
    if int(schedule_doc.get("collection_seed", -1)) != int(config["collection_seed"]):
        raise ValueError("collection_seed mismatch between config and schedule document")
    if schedule_doc.get("config_fingerprint") != canonical_fingerprint(
            json.loads(json.dumps(config))):
        raise ValueError("config fingerprint drift: the config was edited after the freeze")

    # --- forbidden output locations -------------------------------------------------
    data_output = str(config["data_output"])
    if not data_output.endswith(".npz"):
        raise ValueError("data_output must be an .npz path")
    segments = _logical(data_output).split("/")
    if "stage3a_replacement_dev" not in segments:
        raise ValueError(
            "data_output must live in the dedicated stage3a_replacement_dev "
            f"directory, got {data_output}")
    s3d.assert_not_sealed_stage2(data_output)
    for key in ("data_output", "rejection_ledger", "collection_receipt", "output_dir"):
        value = str(config[key])
        s3d.assert_not_sealed_stage2(value)
        value_segments = _logical(value).split("/")
        if any(seg.startswith("gatec") for seg in value_segments):
            raise ValueError(f"{key} must not point into Gate C paths, got {value}")
        if "merge_2000" in value_segments:
            raise ValueError(f"{key} must not point into the old merge_2000 data, got {value}")
        if "stage3a" in value_segments:
            raise ValueError(f"{key} must not point into the read-only old stage3a "
                             f"run directory, got {value}")

    # --- run-dir consistency ---------------------------------------------------------
    if _resolve(config["collection_receipt"]) != (
            run_dir / "replacement_dev" / "replacement_dev_collection_receipt.json"):
        raise ValueError("collection_receipt must be "
                         "<run-dir>/replacement_dev/replacement_dev_collection_receipt.json")
    if _resolve(config["output_dir"]) != run_dir / "replacement_dev" / "collection":
        raise ValueError("output_dir must be <run-dir>/replacement_dev/collection")


# --- collection ----------------------------------------------------------------------


def _historical_seeds_for_collection(config: Mapping[str, Any]) -> np.ndarray:
    """Full historical seed universe (INCLUDING the sealed Stage 2 holdout),
    excluding only this collector's own (partial) products so a resume does not
    collide with itself."""
    try:
        return s3d.scan_all_episode_seeds(
            _resolve(config["historical_data_root"]),
            exclude_substring="stage3a_replacement_dev",
        )
    except FileNotFoundError:
        return np.asarray([], dtype=np.int64)  # nothing collected anywhere yet
    except ValueError as exc:
        if "no historical episode seeds" in str(exc):
            return np.asarray([], dtype=np.int64)
        raise


def collect_replacement_dev(
    config: Mapping[str, Any],
    *,
    run_dir: str | Path,
    resume: bool = False,
    branch_rollout=None,
    probe_fn=None,
    stop_after_accepted: int | None = None,
) -> dict[str, Any]:
    """Collect the replacement development holdout via the Stage 2 engine.

    The engine's Stage 2 protocol hooks are shimmed to the replacement freeze
    chain for the duration of the call, then restored. The running config must
    be fingerprint-identical to the frozen config; the seed schedule is the
    frozen one and is re-validated against the full historical universe at
    collection time.
    """
    config = dict(config)
    validate_replacement_dev_config(config, run_dir=run_dir)
    run_dir = _resolve(run_dir)

    schedule_doc = json.loads(
        (run_dir / "replacement_dev" / "replacement_dev_seed_schedule.json").read_text(
            encoding="utf-8"))
    schedule = np.asarray(schedule_doc["schedule"], dtype=np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != config["seed_schedule_sha256"]:
        raise ValueError("frozen schedule bytes do not match seed_schedule_sha256")

    historical = _historical_seeds_for_collection(config)
    validate_seed_schedule(schedule, historical)

    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "seed_schedule.npy", schedule)

    fingerprint = canonical_fingerprint(json.loads(json.dumps(config)))
    original_validate = _engine.sp.validate_stage2_config
    original_verify = _engine.sp.verify_protocol_lock

    def _validate(cfg: Mapping[str, Any]) -> None:
        validate_replacement_dev_config(cfg, run_dir=run_dir)

    def _verify(_output_dir):
        # Adapt the freeze chain to the keys the engine reads; a drifted config
        # fails the engine's config_signature comparison.
        return {
            "config_signature": schedule_doc["config_fingerprint"],
            "config_sha256": schedule_doc["config_file_sha256"],
        }

    _engine.sp.validate_stage2_config = _validate
    _engine.sp.verify_protocol_lock = _verify
    try:
        receipt = _engine.collect_stage2(
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

    if receipt.get("partial"):
        return receipt  # checkpoint only; nothing published, nothing to augment
    return _finalize(config, run_dir, schedule, historical, fingerprint)


def _finalize(
    config: Mapping[str, Any],
    run_dir: Path,
    schedule: np.ndarray,
    historical: np.ndarray,
    fingerprint: str,
) -> dict[str, Any]:
    """Post-collection QA + receipt augmentation (v0.1 §3.3 checks)."""
    import run_stage3a as drv  # local: keeps module importable without torch

    data_output = _resolve(config["data_output"])
    n_pairs = int(config["n_pairs"])
    with np.load(data_output, allow_pickle=True) as z:
        episode_seeds = np.asarray(z["episode_seed"], dtype=np.int64)
        durations = set(int(v) for v in np.unique(z["cf_duration"]))
        if int(episode_seeds.size) != n_pairs:
            raise RuntimeError(f"collected {episode_seeds.size} episodes, expected {n_pairs}")
        if durations != {1, 3}:
            raise RuntimeError(f"cf_duration must be exactly {{1,3}}, got {durations}")
        if len(np.unique(episode_seeds)) != episode_seeds.size:
            raise RuntimeError("episode seeds are not unique")
        if not set(episode_seeds.tolist()) <= set(schedule.tolist()):
            raise RuntimeError("episode seeds are not drawn from the frozen schedule")
        overlap = set(episode_seeds.tolist()) & set(historical.tolist())
        if overlap:
            raise RuntimeError(f"collected seeds overlap history: {sorted(overlap)[:5]}")
        for field in ("object_history", "object_factual_future", "object_counterfactual_future"):
            if not np.isfinite(z[field]).all():
                raise RuntimeError(f"final NPZ field {field} is non-finite")

    # Tensor schema / presence / non-ego eval mask legality via the Task 7 loader.
    drv.load_replacement_tensors(data_output)

    qa = {
        "episode_count": n_pairs,
        "duration_set": sorted(durations),
        "historical_seed_overlap": 0,
        "episode_seeds_from_frozen_schedule": True,
        "gate_loader": "ok",
    }
    receipt_path = _resolve(config["collection_receipt"])
    published = json.loads(receipt_path.read_text(encoding="utf-8"))
    published["purpose"] = s3d.REPLACEMENT_DEV_PURPOSE
    published["plan_id"] = s3d.REPAIR_PLAN_ID
    published["data_sha256"] = published.get("data_output_sha256") or sha256_file(data_output)
    published["seed_schedule_sha256"] = config["seed_schedule_sha256"]
    published["power_report_sha256"] = config["power_report_sha256"]
    published["config_fingerprint"] = fingerprint
    published["qa"] = qa
    published["code_sha256"] = {
        name: sha256_file(ROOT / "code" / name) for name in _CODE_FILES
    }
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(published, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(receipt_path)
    return published


# --- progress heartbeat + main (manually started) -------------------------------------


def _count_ledger(ledger_path: Path) -> tuple[int, int]:
    """(accepted, attempts) from the rejection ledger; (0, 0) before it exists."""
    if not ledger_path.is_file():
        return 0, 0
    accepted = attempts = 0
    with open(ledger_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            attempts += 1
            try:
                if json.loads(line).get("decision") == "accepted":
                    accepted += 1
            except json.JSONDecodeError:
                continue
    return accepted, attempts


def _heartbeat_loop(config: Mapping[str, Any], stop: threading.Event) -> None:
    reporter = ProgressReporter(
        total=int(config["n_pairs"]), every=1,
        prefix="3A-replacement-collect", heartbeat_seconds=30.0)
    ledger = _resolve(config["rejection_ledger"])
    while not stop.is_set():
        accepted, attempts = _count_ledger(ledger)
        reporter.heartbeat(accepted, attempted=attempts)
        stop.wait(5.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3A replacement dev collector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default="experiments/stage3_routing/stage3a_retrain_v1")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = _resolve(args.run_dir)

    # File-level drift check: the config file must be the frozen one.
    schedule_doc = json.loads(
        (run_dir / "replacement_dev" / "replacement_dev_seed_schedule.json").read_text(
            encoding="utf-8"))
    if sha256_file(config_path) != schedule_doc.get("config_file_sha256"):
        print("[3A-replacement-collect] REFUSED: config file hash differs from the "
              "frozen config", flush=True)
        return 3

    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(config, stop), daemon=True)
    heartbeat.start()
    try:
        receipt = collect_replacement_dev(config, run_dir=run_dir, resume=args.resume)
    finally:
        stop.set()
        heartbeat.join(timeout=10.0)
    print(f"[3A-replacement-collect] accepted={receipt.get('accepted_pairs', '?')} "
          f"attempts={receipt.get('attempts', '?')} purpose={receipt.get('purpose', '?')}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
