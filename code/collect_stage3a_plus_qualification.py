"""Stage 3A Plus qualification-set collector (plan §6.2, Task 9).

Reuses the Stage 2 collection engine (``collect_stage2_holdout.collect_stage2``)
so checkpoint/resume, the rejection ledger, QA and atomic publish are NOT
duplicated. Only qualification-specific concerns live here: the frozen merge
generation contract (field-by-field), the power-frozen episode count with the
ten action x duration strata at EXACTLY equal counts, the hash chain to the
joint-power report and seed schedule, the zero-overlap seed universe (which
includes the sealed Stage 2 holdout), forbidden output locations, the ledger
audit that rejects may ONLY cite engineering completeness reasons (never a
realized response, model output or gate metric), and the
``stage3a_plus_qualification`` purpose.

The collector is rollout-injectable so unit tests run without highway_env.
A 12-episode engineering smoke produces only an ``ENGINEERING_SMOKE_ONLY``
receipt and never publishes the final dataset. The full collection advances
the run state to ``MANUAL_COLLECTION_COMPLETE``; a 30-second progress
heartbeat runs for the duration.
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
from collections import Counter
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
PLAN_ID = s3d.STAGE3A_PLUS_PLAN_ID

SMOKE_EPISODE_COUNT = 12
SMOKE_STATE = "ENGINEERING_SMOKE_ONLY"

# The only reasons a collection attempt may be rejected: engineering
# completeness rules evaluated by the engine. Selection by the realized
# response, any model output or any gate metric is forbidden (plan §6.2).
ALLOWED_REJECTION_REASONS = frozenset({
    "prefix_terminated", "object_prefix_mismatch", "factual_future_short",
    "counterfactual_future_short", "non_finite", "initial_fingerprint_mismatch",
    "short_history",
})

STRATA = tuple((a, d) for a in range(5) for d in (1, 3))
FROZEN_SCENE = "merge-v0"
FROZEN_ACTIONS = [0, 1, 2, 3, 4]

_REQUIRED_KEYS = {
    "experiment_name", "analysis_status", "purpose", "plan_id", "scene", "env_id",
    "n_pairs", "actions", "intervention_durations", "episodes_per_stratum",
    "stratified_equal_counts", "no_posthoc_selection",
    "collection_seed", "seed_schedule_size", "max_attempts",
    "horizon", "history_len", "t_star_low", "t_star_high",
    "vehicles_count", "observed_vehicles_count",
    "duration", "simulation_frequency", "policy_frequency", "determinism_checks",
    "action_weights", "factual_suffix_steps", "checkpoint_every_accepted",
    "output_dir", "data_output", "rejection_ledger", "collection_receipt",
    "historical_data_root", "power_report_sha256", "seed_schedule_sha256",
}

# Modules whose content hash is recorded in the collection receipt.
_CODE_FILES = (
    "collect_stage3a_plus_qualification.py",
    "collect_stage2_holdout.py",
    "ac_twin/stage3_data.py",
    "ac_twin/stage2_protocol.py",
)

# Output segments that are permanently forbidden for the qualification set:
# the sealed Stage 2 holdout family, the contaminated old merge source, the
# consumed replacement development holdout, the read-only old stage3a* run
# directories and any Gate C path.
_FORBIDDEN_EXACT_SEGMENTS = frozenset({
    "merge_2000", "stage3a", "stage3a_retrain_v1", "stage3a_replacement_dev",
})


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _logical(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


# --- config validation -------------------------------------------------------------


def validate_qualification_config(config: Mapping[str, Any], *, run_dir: str | Path) -> None:
    """Validate the frozen qualification collection config; raise before any output.

    Binds the config to the on-disk joint-power report and seed-schedule
    document, and refuses any generation-contract deviation, any non-frozen
    seed/count, or any forbidden output location.
    """
    missing = _REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"qualification config missing keys: {sorted(missing)}")
    if "split" in config or "sample_limits" in config:
        raise ValueError("the qualification set forbids train/validation split keys; "
                         "every episode is an eval episode")
    if config["plan_id"] != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if (config["analysis_status"] != s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE
            or config["purpose"] != s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE):
        raise ValueError(
            f"purpose must be {s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE!r} — the "
            "qualification set never shares a purpose label with Stage 2, the "
            "development sets or Gate C")

    # --- frozen qualification protocol ------------------------------------------
    if config["scene"] != FROZEN_SCENE:
        raise ValueError(f"scene must be {FROZEN_SCENE!r}, got {config['scene']!r}")
    if [int(a) for a in config["actions"]] != FROZEN_ACTIONS:
        raise ValueError(f"actions must be {FROZEN_ACTIONS}, got {config['actions']!r}")
    if int(config["collection_seed"]) != int(s3d.STAGE3A_PLUS_COLLECTION_SEED):
        raise ValueError(
            f"collection_seed is frozen to {s3d.STAGE3A_PLUS_COLLECTION_SEED}")

    # The generation contract replicates the old merge data EXACTLY (plan §6.2).
    for key, expected in MERGE_GENERATION_CONTRACT.items():
        if key in ("scene", "max_attempt_multiplier"):
            continue  # scene is frozen above; the multiplier via max_attempts below
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
    if n_pairs % len(STRATA) != 0:
        raise ValueError(
            f"n_pairs ({n_pairs}) must be divisible by the ten action x duration strata")
    if int(config["episodes_per_stratum"]) != n_pairs // len(STRATA):
        raise ValueError(
            f"episodes_per_stratum must equal n_pairs / {len(STRATA)} "
            f"({n_pairs // len(STRATA)}), got {config['episodes_per_stratum']!r}")
    if config["stratified_equal_counts"] is not True:
        raise ValueError("stratified_equal_counts must be true — the ten strata are "
                         "collected at exactly equal counts")
    if config["no_posthoc_selection"] is not True:
        raise ValueError(
            "no_posthoc_selection must be true — selection by realized response, "
            "model output or gate metrics is forbidden")
    multiplier = int(MERGE_GENERATION_CONTRACT["max_attempt_multiplier"])
    if int(config["max_attempts"]) != n_pairs * multiplier:
        raise ValueError(
            f"max_attempts must equal n_pairs * max_attempt_multiplier "
            f"({n_pairs * multiplier}), got {config['max_attempts']!r}")
    if int(config["seed_schedule_size"]) != int(config["max_attempts"]):
        raise ValueError("seed_schedule_size must equal max_attempts")

    # --- bind to the frozen joint-power report (tamper-evident) -------------------
    run_dir = _resolve(run_dir)
    report_path = run_dir / "power" / "power_report.json"
    if not report_path.is_file():
        raise ValueError(f"power_report missing: {report_path}")
    if sha256_file(report_path) != config["power_report_sha256"]:
        raise ValueError("power_report sha256 mismatch — the power report was "
                         "edited after the freeze")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") == "STAGE3A_PLUS_QUALIFICATION_POWER_INFEASIBLE":
        raise ValueError("STAGE3A_PLUS_QUALIFICATION_POWER_INFEASIBLE: collection "
                         "is forbidden")
    if int(report.get("selected_episode_count", -1)) != n_pairs:
        raise ValueError(
            f"n_pairs ({n_pairs}) must equal the power-frozen selected_episode_count "
            f"({report.get('selected_episode_count')})")

    # --- bind to the frozen seed schedule document --------------------------------
    schedule_path = run_dir / "qualification" / "seed_schedule.json"
    if not schedule_path.is_file():
        raise ValueError(f"qualification seed schedule missing: {schedule_path}")
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
    if s3d.STAGE3A_PLUS_QUALIFICATION_DIR_SEGMENT not in segments:
        raise ValueError(
            "data_output must live in the dedicated stage3a_plus_qualification "
            f"directory, got {data_output}")
    for key in ("data_output", "rejection_ledger", "collection_receipt", "output_dir"):
        value = str(config[key])
        s3d.assert_not_sealed_stage2(value)
        value_segments = _logical(value).split("/")
        if any(seg.startswith("gatec") for seg in value_segments):
            raise ValueError(f"{key} must not point into Gate C paths, got {value}")
        if any(seg.startswith("stage2") for seg in value_segments):
            raise ValueError(f"{key} must not point into Stage 2 paths, got {value}")
        if any(seg in _FORBIDDEN_EXACT_SEGMENTS for seg in value_segments):
            raise ValueError(
                f"{key} must not point into the old merge source, the consumed "
                f"replacement holdout or a read-only old stage3a directory, got {value}")

    # --- run-dir consistency ---------------------------------------------------------
    if _resolve(config["collection_receipt"]) != (
            run_dir / "qualification" / "collection_receipt.json"):
        raise ValueError("collection_receipt must be "
                         "<run-dir>/qualification/collection_receipt.json")
    if _resolve(config["output_dir"]) != run_dir / "qualification" / "collection":
        raise ValueError("output_dir must be <run-dir>/qualification/collection")


# --- collection ----------------------------------------------------------------------


def _historical_seeds_for_collection(config: Mapping[str, Any]) -> np.ndarray:
    """Full historical seed universe (INCLUDING the sealed Stage 2 holdout and
    every development set), excluding only this collector's own (partial)
    products so a resume does not collide with itself."""
    try:
        return s3d.scan_all_episode_seeds(
            _resolve(config["historical_data_root"]),
            exclude_substring=s3d.STAGE3A_PLUS_QUALIFICATION_DIR_SEGMENT,
        )
    except FileNotFoundError:
        return np.asarray([], dtype=np.int64)  # nothing collected anywhere yet
    except ValueError as exc:
        if "no historical episode seeds" in str(exc):
            return np.asarray([], dtype=np.int64)
        raise


def _run_engine(config: Mapping[str, Any], run_dir: Path, schedule: np.ndarray,
                schedule_doc: Mapping[str, Any], *, resume: bool,
                branch_rollout, probe_fn,
                stop_after_accepted: int | None = None) -> dict[str, Any]:
    """Run the Stage 2 engine with its protocol hooks shimmed to the
    qualification freeze chain for the duration of the call."""
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "seed_schedule.npy", schedule)

    original_validate = _engine.sp.validate_stage2_config
    original_verify = _engine.sp.verify_protocol_lock

    def _validate(cfg: Mapping[str, Any]) -> None:
        validate_qualification_config(cfg, run_dir=run_dir)

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


def collect_qualification(
    config: Mapping[str, Any],
    *,
    run_dir: str | Path,
    resume: bool = False,
    branch_rollout=None,
    probe_fn=None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Collect the single-use qualification set via the Stage 2 engine.

    ``smoke=True`` runs exactly ``SMOKE_EPISODE_COUNT`` episodes, leaves the
    checkpoint for a later resume, publishes nothing, and writes only an
    ``ENGINEERING_SMOKE_ONLY`` receipt; smoke is refused once the final
    dataset exists.
    """
    config = dict(config)
    validate_qualification_config(config, run_dir=run_dir)
    run_dir = _resolve(run_dir)

    schedule_doc = json.loads(
        (run_dir / "qualification" / "seed_schedule.json").read_text(encoding="utf-8"))
    schedule = np.asarray(schedule_doc["schedule"], dtype=np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != config["seed_schedule_sha256"]:
        raise ValueError("frozen schedule bytes do not match seed_schedule_sha256")

    historical = _historical_seeds_for_collection(config)
    validate_seed_schedule(schedule, historical)

    data_output = _resolve(config["data_output"])
    if smoke:
        if data_output.exists():
            raise ValueError(
                "smoke is forbidden once the qualification dataset exists — the "
                "single-use set is never re-entered for engineering checks")
        receipt = _run_engine(
            config, run_dir, schedule, schedule_doc, resume=resume,
            branch_rollout=branch_rollout, probe_fn=probe_fn,
            stop_after_accepted=SMOKE_EPISODE_COUNT)
        smoke_receipt = {
            **receipt,
            "state": SMOKE_STATE,
            "purpose": s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE,
            "plan_id": PLAN_ID,
            "power_report_sha256": config["power_report_sha256"],
            "seed_schedule_sha256": config["seed_schedule_sha256"],
            "note": "engineering smoke only — these episodes never enter any gate "
                    "and are only a resume checkpoint for the real collection",
        }
        smoke_path = run_dir / "qualification" / "smoke_receipt.json"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = smoke_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(smoke_receipt, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
        tmp.replace(smoke_path)
        return smoke_receipt

    receipt = _run_engine(
        config, run_dir, schedule, schedule_doc, resume=resume,
        branch_rollout=branch_rollout, probe_fn=probe_fn)
    if receipt.get("partial"):
        return receipt  # checkpoint only; nothing published, nothing to augment
    if receipt.get("verification") == "existing_final_dataset_verified_no_overwrite":
        return receipt  # already collected and finalized; never rewritten
    fingerprint = canonical_fingerprint(json.loads(json.dumps(config)))
    return _finalize(config, run_dir, schedule, historical, fingerprint)


def _finalize(
    config: Mapping[str, Any],
    run_dir: Path,
    schedule: np.ndarray,
    historical: np.ndarray,
    fingerprint: str,
    *,
    historical_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Post-collection QA + receipt augmentation (plan §6.2 / Task 9 checks)."""
    import run_stage3a_plus as drv  # local: keeps module importable without torch

    data_output = _resolve(config["data_output"])
    n_pairs = int(config["n_pairs"])
    with np.load(data_output, allow_pickle=True) as z:
        episode_seeds = np.asarray(z["episode_seed"], dtype=np.int64)
        durations = set(int(v) for v in np.unique(z["cf_duration"]))
        actions = set(int(v) for v in np.unique(z["cf_action"]))
        if int(episode_seeds.size) != n_pairs:
            raise RuntimeError(f"collected {episode_seeds.size} episodes, expected {n_pairs}")
        if durations != {1, 3}:
            raise RuntimeError(f"cf_duration must be exactly {{1,3}}, got {durations}")
        if not actions <= set(FROZEN_ACTIONS):
            raise RuntimeError(f"cf_action must be a subset of {FROZEN_ACTIONS}, got {actions}")
        if len(np.unique(episode_seeds)) != episode_seeds.size:
            raise RuntimeError("episode seeds are not unique")
        if not set(episode_seeds.tolist()) <= set(schedule.tolist()):
            raise RuntimeError("episode seeds are not drawn from the frozen schedule")
        history = np.asarray(
            historical_override if historical_override is not None else historical,
            dtype=np.int64)
        overlap = set(episode_seeds.tolist()) & set(history.tolist())
        if overlap:
            raise RuntimeError(f"collected seeds overlap history: {sorted(overlap)[:5]}")
        for field in ("object_history", "object_factual_future", "object_counterfactual_future"):
            if not np.isfinite(z[field]).all():
                raise RuntimeError(f"final NPZ field {field} is non-finite")

        # Exact equal counts over the ten action x duration strata (plan §6.2).
        expected = n_pairs // len(STRATA)
        counts = Counter((int(a), int(d))
                         for a, d in zip(z["cf_action"], z["cf_duration"]))
        per_stratum = {key: counts.get(key, 0) for key in STRATA}
        if any(count != expected for count in per_stratum.values()):
            raise RuntimeError(
                f"strata counts are not exactly equal: expected {expected} per "
                f"stratum, got {per_stratum}")

    # Ledger audit: rejections may ONLY cite engineering completeness rules.
    ledger_path = _resolve(config["rejection_ledger"])
    reasons_seen: set[str] = set()
    if ledger_path.is_file():
        with open(ledger_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    raise RuntimeError("rejection ledger contains an unparseable line")
                if record.get("decision") == "rejected":
                    reason = str(record.get("reason"))
                    reasons_seen.add(reason)
                    if reason not in ALLOWED_REJECTION_REASONS:
                        raise RuntimeError(
                            f"rejection ledger contains a non-engineering reason "
                            f"{reason!r}: selection by realized response, model "
                            "output or gate metrics is forbidden")

    # Tensor schema / presence / non-ego eval mask legality via the Task 10 loader.
    drv.load_qualification_tensors(data_output)

    qa = {
        "episode_count": n_pairs,
        "duration_set": sorted(durations),
        "actions": sorted(actions),
        "strata_equal_counts": True,
        "episodes_per_stratum": expected,
        "historical_seed_overlap": 0,
        "episode_seeds_from_frozen_schedule": True,
        "rejection_reasons": sorted(reasons_seen),
        "gate_loader": "ok",
    }
    receipt_path = _resolve(config["collection_receipt"])
    published = json.loads(receipt_path.read_text(encoding="utf-8"))
    published["purpose"] = s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE
    published["plan_id"] = PLAN_ID
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


# --- progress heartbeat + main (autonomous under run-all; state-gated) ---------------


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
        prefix="3A+-qualification-collect", heartbeat_seconds=30.0)
    ledger = _resolve(config["rejection_ledger"])
    while not stop.is_set():
        accepted, attempts = _count_ledger(ledger)
        reporter.heartbeat(accepted, attempted=attempts)
        stop.wait(5.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3A Plus qualification collector")
    parser.add_argument("--config", default=None,
                        help="defaults to <run-dir>/qualification/config.json")
    parser.add_argument("--run-dir", default="experiments/stage3_routing/stage3a_plus")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help=f"engineering smoke: {SMOKE_EPISODE_COUNT} episodes, "
                             "no publish, ENGINEERING_SMOKE_ONLY receipt only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import run_stage3a_plus as drv  # local: torch driver

    run_dir = _resolve(args.run_dir)
    state = drv.load_state(run_dir)["state"]
    if state != "QUALIFICATION_POWER_FROZEN":
        print(f"[3A+-collect] REFUSED: state {state} != QUALIFICATION_POWER_FROZEN",
              flush=True)
        return 2
    config_path = _resolve(args.config) if args.config else (
        run_dir / "qualification" / "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # File-level drift check: the config file must be the frozen one.
    schedule_doc = json.loads(
        (run_dir / "qualification" / "seed_schedule.json").read_text(encoding="utf-8"))
    if sha256_file(config_path) != schedule_doc.get("config_file_sha256"):
        print("[3A+-collect] REFUSED: config file hash differs from the frozen config",
              flush=True)
        return 3

    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(config, stop), daemon=True)
    heartbeat.start()
    try:
        receipt = collect_qualification(
            config, run_dir=run_dir, resume=args.resume, smoke=args.smoke)
    finally:
        stop.set()
        heartbeat.join(timeout=10.0)

    if args.smoke:
        print(f"[3A+-collect] smoke accepted={receipt.get('accepted_pairs', '?')} "
              f"state={receipt.get('state', '?')} (checkpoint kept for resume)",
              flush=True)
        return 0
    if receipt.get("partial"):
        print(f"[3A+-collect] checkpoint accepted={receipt.get('accepted_pairs', '?')} "
              f"attempts={receipt.get('attempts', '?')} — resume to continue", flush=True)
        return 0
    drv.advance_state(run_dir, "MANUAL_COLLECTION_COMPLETE",
                      note=f"n={receipt.get('accepted_pairs', '?')}")
    print(f"[3A+-collect] accepted={receipt.get('accepted_pairs', '?')} "
          f"attempts={receipt.get('attempts', '?')} "
          f"purpose={receipt.get('purpose', '?')} -> MANUAL_COLLECTION_COMPLETE",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
