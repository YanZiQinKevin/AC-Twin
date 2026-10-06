"""Stage 4B protocol: seed schedule, single confirm role, model/data locks, write-once.

Freeze layer for the Stage 4B independent Merge risk confirmation
(plan ``AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1``). It never trains, never
fits on new data, never re-selects a candidate/baseline/coverage/feature and never
reads a confirm label/score before ``confirmation_used.lock`` is burned. Its only
pre-collection numeric product is the 12,000-seed candidate schedule (frozen entropy
``2026091701``) plus the historical seed inventory used to prove zero overlap.

Data role (plan section 4.3): the single new role is ``stage4_merge_risk_confirm``.
ALL accepted episodes enter one confirmation set — there is NO train/validation/test
split. The set is balanced across the ten ``cf_action x cf_duration`` strata
(``selected_n / 10`` per cell). This data is never used for training, selection,
threshold fitting or power estimation, and never becomes Stage 4C development data.

Stage 4B binds Stage 4A's frozen artifacts READ-ONLY (plan section 3.1): the v4
recovery lock, the candidate-selection receipt, the finalist ``COMBINED_RISK_HEAD/MLP``
and the ``GEOMETRY_ONLY/Ridge`` models, the scaler and the development-screen
``risk_dataset.npz``. Identity is by content SHA256, never by drive letter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin import stage3_data as s3d
from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.stage1_data import sha256_file
from ac_twin.stage2_protocol import make_seed_schedule, validate_seed_schedule
from ac_twin.stage3r_protocol import write_once_json
from ac_twin.stage4_protocol import (  # shared frozen layout/seed constants
    CF_ACTIONS,
    CF_DURATIONS,
    EXPERT_ROUTER_SEEDS,
    PRIMARY_CONSTANT_W,
    RESPONSE_PREDICTOR_SEEDS,
    e2e_checkpoint_path,
    expert_checkpoint_path,
)

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = "AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1"
PURPOSE = "stage4_merge_risk_confirm"

# Frozen seed-schedule contract (plan section 4.2).
SEED_ENTROPY = 2026091701
SEED_SCHEDULE_SIZE = 12000
SEED_LOW = 1
SEED_HIGH_INCLUSIVE = 2147483646  # 2**31 - 2

# Frozen power / confirm constants (plan sections 5, 7, 8).
POWER_ENTROPY = 2026091702
BOOTSTRAP_ENTROPY_CONFIRM = 2026091703
POWER_BOOTSTRAP_RESAMPLES = 10000
POWER_MONTE_CARLO = 10000
POWER_TARGET = 0.80
POWER_SHRINK_MAIN = 0.75
POWER_SHRINK_SCENARIOS = (1.0, 0.75, 0.5)
CANDIDATE_NS = (2000, 3000, 4000, 5000)
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES_CONFIRM = 10000

# Frozen Stage 4A carry-over (plan section 3.1 / 7.2): the top-error threshold is
# NOT recomputed; Stage 4B has no train split, so it inherits the Stage 4A value.
TOP_ERROR_THRESHOLD = 0.41601440122246675

# Gate RISK_confirm mechanical thresholds (plan section 8.2).
GATE_I_AURC_MIN = 0.02
GATE_I_C80_MIN = 0.02
GATE_MIN_POSITIVE_PAIRS = 4
GATE_STRATA_I_AURC_FLOOR = -0.10

# Mechanical terminal states (plan section 8.3).
STATUS_CONFIRMED = "STAGE4B_RISK_CONFIRMED"
STATUS_NOT_CONFIRMED = "STAGE4B_RISK_NOT_CONFIRMED"
STATUS_UNDERPOWERED = "STAGE4B_CONFIRMATION_UNDERPOWERED"
STATUS_COLLECTION_INCOMPLETE = "STAGE4B_COLLECTION_INCOMPLETE"
STATUS_INVALID = "STAGE4B_INVALID_ANALYSIS"

# State machine (plan section 10).
STATE_TRANSITIONS = {
    "DRAFT": {"SOFTWARE_READY"},
    "SOFTWARE_READY": {"POWERED"},
    "POWERED": {"PROTOCOL_LOCKED"},
    "PROTOCOL_LOCKED": {"CONFIRM_COLLECTED"},
    "CONFIRM_COLLECTED": {"CONFIRM_INFERENCE_COMPLETE"},
    "CONFIRM_INFERENCE_COMPLETE": {"CONFIRM_ADJUDICATED"},
    "CONFIRM_ADJUDICATED": set(),
}

# New-code files whose SHA256 are enumerated in the protocol lock (relative to code/).
CODE_FILES = (
    "run_stage4b.py",
    "collect_stage4b_merge_confirm.py",
    "ac_twin/stage4b_protocol.py",
    "ac_twin/stage4b_power.py",
    "ac_twin/stage4b_inference.py",
    "ac_twin/stage4b_statistics.py",
)

# Frozen Stage 4A source artifacts (plan section 3.1) with their known SHA256.
# Identity is by content hash; only the project-relative path is resolved.
STAGE4A_FROZEN = {
    "stage4a_recovery_lock": (
        "experiments/stage4_reliability/stage4a_v4/protocol/recovery_lock.json",
        "a774d009f1757bb8cb972090e79e347ce3b9771cae1fc81162f45cfe0de99b1e",
    ),
    "stage4a_candidate_selection": (
        "experiments/stage4_reliability/stage4a_v4/risk/candidate_selection.json",
        "b056593c06d0968f71f3af257e0a28e1d4fa2b6928e9bde27f451bf73ce579b0",
    ),
    "finalist_risk_model": (
        "experiments/stage4_reliability/stage4a_v3/risk/models/combined_risk_head_mlp.pt",
        "4a5108030536a2838bac868e3f7da19cc0295fe5164da154b7ee1e8d87bc8edc",
    ),
    "geometry_risk_model": (
        "experiments/stage4_reliability/stage4a_v3/risk/models/geometry_only_ridge.pt",
        "602d32a6aeb74f998a6c41151b7b7338d354022098bc2e29af60935f996b8bd9",
    ),
    "stage4a_risk_dataset": (
        "experiments/stage4_reliability/stage4a_v4/risk/risk_dataset.npz",
        "423a7aabde61f6604c2a3acd5b2875a30cef766b1a6f95b9170cd926d3658232",
    ),
    "scaler": (
        "experiments/stage3_routing/stage3r/scaler.json",
        "9f546efe3bf6e2a907e7115924f7d1dec55491b11b6a1e09c469b104c5db9181",
    ),
}

# Frozen identities carried from the Stage 4A candidate-selection receipt.
FINALIST = "COMBINED_RISK_HEAD"
COMBINED_IMPLEMENTATION = "mlp"
PRIMARY_PREDICTOR = "END_TO_END_MIXTURE"
STANDARD_BASELINE = "UNCERTAINTY_ONLY"
SECOND_BASELINE = "GEOMETRY_ONLY"

# Forbidden source path tokens (plan sections 3.2 / 4.3). Stage 4B additionally
# forbids every Stage 4A development product and the Stage 2B intersection data.
_FORBIDDEN_DATA_TOKENS = (
    "stage2_merge_1200",
    "stage2_confirmatory",
    "stage3a_plus_qualification",
    "stage3a_replacement_dev",
    "stage3b",
    "stage3r_rscreen",
    "gatec",
    "merge_2000",
    "roundabout",
    "intersection",
    "stage4_merge_risk_dev",
)


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _normalised(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


# --- forbidden-data guards ----------------------------------------------------


def assert_allowed_stage4b_data(path: str | Path, *, file_sha256: str | None = None) -> None:
    """Raise ``ValueError`` if ``path`` is a permanently forbidden Stage 4B input.

    The only legal Stage 4B data role is ``stage4_merge_risk_confirm``; anything
    under a sealed/consumed Stage 2/3/4A location is refused by path segment and,
    where a known fingerprint exists, by content SHA256.
    """
    s3d.assert_not_sealed_stage2(path, file_sha256=file_sha256)
    logical = _normalised(path)
    for token in _FORBIDDEN_DATA_TOKENS:
        for segment in logical.split("/"):
            if segment.startswith(token):
                raise ValueError(
                    f"forbidden Stage 4B data source (segment {segment!r}): {path}"
                )
    if PURPOSE not in logical.split("/") and PURPOSE not in logical:
        raise ValueError(
            f"Stage 4B confirm data must live under the {PURPOSE} role: {path}"
        )


# --- seed schedule ------------------------------------------------------------


def make_candidate_seed_schedule(
    entropy: int = SEED_ENTROPY, size: int = SEED_SCHEDULE_SIZE
) -> np.ndarray:
    """12,000 unique positive int32 candidate seeds in ``[1, 2**31-2]``.

    Reuses the frozen Stage 2 generator (``SeedSequence(entropy).spawn(3)`` first
    stream, draw without replacement, +1 shift) with the Stage 4B entropy so the
    schedule is internally unique and in-range by construction.
    """
    if int(entropy) != SEED_ENTROPY:
        raise ValueError(f"Stage 4B seed entropy is frozen to {SEED_ENTROPY}")
    if int(size) != SEED_SCHEDULE_SIZE:
        raise ValueError(f"Stage 4B seed schedule size is frozen to {SEED_SCHEDULE_SIZE}")
    schedule = make_seed_schedule(int(entropy), size=int(size))
    if schedule.min() < SEED_LOW or schedule.max() > SEED_HIGH_INCLUSIVE:
        raise ValueError("Stage 4B candidate seed schedule out of range")
    return schedule.astype(np.int64)


def scan_historical_seeds(workspace_roots: list[Path]) -> np.ndarray:
    """Union of every historical *accepted* episode seed across NPZ, receipts, manifests.

    Identical scope to the canonical prior-stage scanners: only *realized* episode
    seeds are collected (raw candidate ``*seed_schedule*.npy`` pools are excluded —
    an unaccepted candidate never generated an episode). The Stage 4B role's own
    partial products are excluded so a resume does not collide with itself; the
    Stage 4A dev products ARE scanned so Stage 4B is disjoint from them.
    """
    collected: set[int] = set()
    for root in workspace_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob("*")):
            normalized = _normalised(file_path)
            # Exclude Stage 4B's OWN products: the confirm data dir (PURPOSE) and the
            # stage4b experiment run dir. Those are generated FROM this schedule, so
            # they are not independent prior history; counting them would false-flag
            # a self-overlap at Gate Q re-scan time after receipts exist.
            if PURPOSE in normalized or "stage4_reliability/stage4b" in normalized:
                continue
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".npz":
                    with np.load(file_path, allow_pickle=True) as archive:
                        for key in ("episode_seed", "episode_seeds"):
                            if key in archive.files:
                                values = np.atleast_1d(np.asarray(archive[key])).ravel()
                                collected.update(int(v) for v in values)
                elif suffix == ".json":
                    collected.update(_seeds_in_json(file_path))
            except Exception:
                # An unreadable artifact contributes no seeds; the schedule overlap
                # check at lock time is what protects the protocol.
                continue
    if not collected:
        raise ValueError("no historical episode seeds found in the workspace")
    return np.array(sorted(collected), dtype=np.int64)


def _seeds_in_json(file_path: Path) -> set[int]:
    """Recursively collect ints under keys named like episode/seed in a JSON file."""
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    found: set[int] = set()

    def walk(node: Any, key_hint: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, str(key).lower())
        elif isinstance(node, (list, tuple)):
            if any(hint in key_hint for hint in ("episode_seed", "episode_seeds", "seed_schedule")):
                for item in node:
                    if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                        found.add(int(item))
                    else:
                        walk(item, key_hint)
            else:
                for item in node:
                    walk(item, key_hint)

    walk(payload, "")
    return found


# --- stratum balance ------------------------------------------------------------


def strata_counts(cf_action: np.ndarray, cf_duration: np.ndarray) -> dict[tuple[int, int], int]:
    """Per-``(cf_action, cf_duration)`` accepted-episode counts."""
    cf_action = np.asarray(cf_action, dtype=np.int64)
    cf_duration = np.asarray(cf_duration, dtype=np.int64)
    return {
        (int(a), int(d)): int(((cf_action == a) & (cf_duration == d)).sum())
        for a in CF_ACTIONS for d in CF_DURATIONS
    }


def assert_balanced_strata(
    cf_action: np.ndarray, cf_duration: np.ndarray, per_cell: int
) -> None:
    """Raise unless every one of the ten strata has exactly ``per_cell`` episodes."""
    counts = strata_counts(cf_action, cf_duration)
    for cell, count in counts.items():
        if count != int(per_cell):
            raise ValueError(f"stratum {cell} has {count} episodes, expected {per_cell}")


# --- config validation ---------------------------------------------------------


def validate_stage4b_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 4B config; raise before any artifact is produced."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("purpose") != PURPOSE or checked.get("analysis_status") != PURPOSE:
        raise ValueError(f"purpose/analysis_status must be {PURPOSE!r}")
    if checked.get("scene") != "merge" or checked.get("env_id") != "merge-v0":
        raise ValueError("Stage 4B is merge-v0 only")
    if list(checked.get("cf_actions", [])) != list(CF_ACTIONS):
        raise ValueError(f"cf_actions must be {list(CF_ACTIONS)}")
    if list(checked.get("intervention_durations", [])) != list(CF_DURATIONS):
        raise ValueError(f"intervention_durations must be {list(CF_DURATIONS)}")
    if [int(n) for n in checked.get("candidate_ns", [])] != list(CANDIDATE_NS):
        raise ValueError(f"candidate_ns must be {list(CANDIDATE_NS)}")
    if int(checked.get("seed_entropy", -1)) != SEED_ENTROPY:
        raise ValueError(f"seed_entropy must be {SEED_ENTROPY}")
    if int(checked.get("seed_schedule_size", -1)) != SEED_SCHEDULE_SIZE:
        raise ValueError(f"seed_schedule_size must be {SEED_SCHEDULE_SIZE}")
    if int(checked.get("max_attempts", -1)) != SEED_SCHEDULE_SIZE:
        raise ValueError("max_attempts must equal seed_schedule_size")
    if int(checked.get("power_entropy", -1)) != POWER_ENTROPY:
        raise ValueError(f"power_entropy must be {POWER_ENTROPY}")
    if int(checked.get("bootstrap_entropy_confirm", -1)) != BOOTSTRAP_ENTROPY_CONFIRM:
        raise ValueError(f"bootstrap_entropy_confirm must be {BOOTSTRAP_ENTROPY_CONFIRM}")
    if int(checked.get("power_bootstrap_resamples", -1)) != POWER_BOOTSTRAP_RESAMPLES:
        raise ValueError("power_bootstrap_resamples drifted")
    if int(checked.get("power_monte_carlo", -1)) != POWER_MONTE_CARLO:
        raise ValueError("power_monte_carlo drifted")
    if float(checked.get("power_target", -1.0)) != POWER_TARGET:
        raise ValueError(f"power_target must be {POWER_TARGET}")
    if float(checked.get("power_shrink_main", -1.0)) != POWER_SHRINK_MAIN:
        raise ValueError(f"power_shrink_main must be {POWER_SHRINK_MAIN}")
    if float(checked.get("confidence_level", -1.0)) != CONFIDENCE_LEVEL:
        raise ValueError(f"confidence_level must be {CONFIDENCE_LEVEL}")
    if int(checked.get("bootstrap_resamples_confirm", -1)) != BOOTSTRAP_RESAMPLES_CONFIRM:
        raise ValueError("bootstrap_resamples_confirm drifted")
    if float(checked.get("top_error_threshold", -1.0)) != TOP_ERROR_THRESHOLD:
        raise ValueError(f"top_error_threshold is frozen to {TOP_ERROR_THRESHOLD}")
    if float(checked.get("gate_i_aurc_min", -1.0)) != GATE_I_AURC_MIN:
        raise ValueError("gate_i_aurc_min drifted")
    if float(checked.get("gate_i_c80_min", -1.0)) != GATE_I_C80_MIN:
        raise ValueError("gate_i_c80_min drifted")
    if int(checked.get("gate_min_positive_pairs", -1)) != GATE_MIN_POSITIVE_PAIRS:
        raise ValueError("gate_min_positive_pairs drifted")
    if float(checked.get("gate_strata_i_aurc_floor", 1.0)) != GATE_STRATA_I_AURC_FLOOR:
        raise ValueError("gate_strata_i_aurc_floor drifted")
    if list(checked.get("response_predictor_seeds", [])) != list(RESPONSE_PREDICTOR_SEEDS):
        raise ValueError("response_predictor_seeds drifted from the frozen five")
    if list(checked.get("expert_router_seeds", [])) != list(EXPERT_ROUTER_SEEDS):
        raise ValueError("expert_router_seeds drifted from the frozen five pairs")
    if float(checked.get("primary_constant_w", -1.0)) != PRIMARY_CONSTANT_W:
        raise ValueError(f"primary_constant_w must be {PRIMARY_CONSTANT_W}")
    # The data output must live under the dedicated confirm role directory.
    assert_allowed_stage4b_data(checked["data_output"])
    for key in ("run_dir", "output_dir", "rejection_ledger", "stage3r_run_dir",
                "stage3b_binding", "scaler_path", "historical_data_root"):
        if not checked.get(key):
            raise ValueError(f"missing Stage 4B config field: {key}")
    return checked


# --- frozen-artifact inventory ---------------------------------------------------


def inventory_frozen_stage4a() -> dict[str, Any]:
    """Resolve + hash every frozen Stage 4A source artifact; raise on any drift."""
    resolved: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name, (rel, expected) in STAGE4A_FROZEN.items():
        path = _resolve(rel)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Stage 4A artifact: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"frozen Stage 4A artifact hash drift: {name} ({rel}); "
                f"expected {expected}, got {digest}"
            )
        resolved[name] = str(path)
        hashes[name] = digest
    return {"stage4a_paths": resolved, "stage4a_sha256": hashes}


def inventory_model_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and hash every frozen Stage 4B model/scaler/binding dependency."""
    checked = validate_stage4b_config(config)
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    checkpoints: dict[str, str] = {}
    for pair_index, seed in enumerate(EXPERT_ROUTER_SEEDS):
        for condition in ("full", "history"):
            path = expert_checkpoint_path(stage3r_dir, pair_index, condition, seed)
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen expert checkpoint: {path}")
            checkpoints[f"pair{pair_index}_fulltrain_{condition}_seed{seed}"] = sha256_file(path)
        e2e = e2e_checkpoint_path(stage3r_dir, pair_index, seed)
        if not e2e.is_file():
            raise FileNotFoundError(f"missing frozen end_to_end checkpoint: {e2e}")
        checkpoints[f"end_to_end_mixture_pair{pair_index}_s{seed}"] = sha256_file(e2e)
    scaler_path = _resolve(checked["scaler_path"])
    binding_path = _resolve(checked["stage3b_binding"])
    if not scaler_path.is_file():
        raise FileNotFoundError(f"missing frozen scaler: {scaler_path}")
    if not binding_path.is_file():
        raise FileNotFoundError(f"missing Stage 3A Plus binding: {binding_path}")
    inventory = {
        "checkpoint_sha256": checkpoints,
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "stage3b_binding_path": str(binding_path),
        "stage3b_binding_sha256": sha256_file(binding_path),
    }
    inventory.update(inventory_frozen_stage4a())
    return inventory


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in CODE_FILES:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered Stage 4B code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


# --- protocol lock --------------------------------------------------------------


def build_protocol_lock(
    config: Mapping[str, Any], *, selected_n: int, project_root: Path = ROOT
) -> dict[str, Any]:
    """Compute the (not yet written) Stage 4B protocol lock payload.

    ``selected_n`` is the mechanically chosen confirmation size from the create-once
    power report; it is embedded so the collector fills exactly ``selected_n / 10``
    per stratum. Scans the historical seed inventory, builds + validates the
    12,000-seed candidate schedule and records the SHA256 of every frozen
    model/scaler/binding/Stage 4A/code input. Pure and side-effect free.
    """
    checked = validate_stage4b_config(config)
    selected_n = int(selected_n)
    if selected_n not in CANDIDATE_NS:
        raise ValueError(f"selected_n must be one of {CANDIDATE_NS}, got {selected_n}")
    if selected_n % (len(CF_ACTIONS) * len(CF_DURATIONS)) != 0:
        raise ValueError("selected_n must be divisible across the ten strata")
    historical = scan_historical_seeds(
        [_resolve(checked["historical_data_root"]), project_root / "experiments"]
    )
    schedule = make_candidate_seed_schedule()
    validate_seed_schedule(schedule, historical)
    model_lock = inventory_model_lock(checked)
    schedule_path = _resolve(checked["run_dir"]) / "protocol" / "candidate_seed_schedule.npy"
    return {
        "plan_id": PLAN_ID,
        "purpose": PURPOSE,
        "config_signature": canonical_fingerprint(json.loads(json.dumps(checked))),
        "seed_entropy": SEED_ENTROPY,
        "seed_schedule_size": SEED_SCHEDULE_SIZE,
        "selected_n": selected_n,
        "per_stratum": selected_n // (len(CF_ACTIONS) * len(CF_DURATIONS)),
        "n_historical_episode_seeds": int(historical.size),
        "seed_schedule_zero_overlap": True,
        "cf_actions": list(CF_ACTIONS),
        "cf_durations": list(CF_DURATIONS),
        "code_sha256": _code_hashes(),
        "config_sha256": None,  # filled by write_protocol_lock
        "seed_schedule_path": str(schedule_path),
        "seed_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "historical_seed_inventory": historical,
        "seed_schedule": schedule,
        **model_lock,
    }


def write_protocol_lock(
    config_path: str | Path, *, selected_n: int, project_root: Path = ROOT
) -> dict[str, Any]:
    """Create the Stage 4B protocol lock exactly once (refuses to overwrite)."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_stage4b_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "protocol"
    lock_path = protocol_dir / "protocol_lock.json"
    schedule_npy = protocol_dir / "candidate_seed_schedule.npy"
    if lock_path.exists():
        raise FileExistsError(f"Stage 4B protocol lock already exists: {lock_path}")

    lock = build_protocol_lock(checked, selected_n=selected_n, project_root=project_root)
    schedule = lock.pop("seed_schedule")
    historical = lock.pop("historical_seed_inventory")
    lock["config_sha256"] = sha256_file(config_path)

    protocol_dir.mkdir(parents=True, exist_ok=True)
    tmp_schedule = schedule_npy.with_name(schedule_npy.name + ".tmp")
    with open(tmp_schedule, "wb") as handle:
        np.save(handle, schedule)
    Path(tmp_schedule).replace(schedule_npy)

    write_once_json(
        protocol_dir / "candidate_seed_schedule.json",
        {
            "plan_id": PLAN_ID,
            "entropy": SEED_ENTROPY,
            "size": int(schedule.size),
            "schedule_sha256": lock["seed_schedule_sha256"],
            "historical_seed_count": int(historical.size),
            "historical_overlap_count": 0,
        },
    )
    write_once_json(
        protocol_dir / "historical_seed_inventory.json",
        {
            "plan_id": PLAN_ID,
            "count": int(historical.size),
            "inventory_sha256": hashlib.sha256(historical.tobytes()).hexdigest(),
        },
    )
    write_once_json(lock_path, lock)
    return lock


def verify_protocol_lock(run_dir: str | Path) -> dict[str, Any]:
    """Read-only re-verification that every locked Stage 4B input still matches."""
    run_dir = Path(run_dir)
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage 4B protocol lock missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: dict[str, tuple[Path, str]] = {}
    for rel, expected in lock["code_sha256"].items():
        checks[f"code:{rel}"] = (ROOT / "code" / rel, expected)
    for name, expected in lock["checkpoint_sha256"].items():
        stage3r_dir = Path(lock["scaler_path"]).parent
        if name.startswith("end_to_end_mixture_"):
            pair_index = int(name.split("_pair")[1].split("_s")[0])
            seed = int(name.rsplit("_s", 1)[1])
            path = e2e_checkpoint_path(stage3r_dir, pair_index, seed)
        else:
            pair_index = int(name.split("pair")[1].split("_")[0])
            condition = "full" if "_full_" in name else "history"
            seed = int(name.rsplit("_seed", 1)[1])
            path = expert_checkpoint_path(stage3r_dir, pair_index, condition, seed)
        checks[f"checkpoint:{name}"] = (path, expected)
    checks["scaler"] = (Path(lock["scaler_path"]), lock["scaler_sha256"])
    checks["stage3b_binding"] = (Path(lock["stage3b_binding_path"]), lock["stage3b_binding_sha256"])
    for name, expected in lock["stage4a_sha256"].items():
        checks[f"stage4a:{name}"] = (Path(lock["stage4a_paths"][name]), expected)
    schedule_path = Path(lock["seed_schedule_path"])
    checks["seed_schedule"] = (schedule_path, None)  # bytes hashed separately
    for label, (file_path, expected) in checks.items():
        if not file_path.is_file():
            raise ValueError(f"locked Stage 4B artifact missing: {label} -> {file_path}")
        if expected is not None and sha256_file(file_path) != expected:
            raise ValueError(f"locked Stage 4B artifact hash drift: {label}")
    schedule = np.load(schedule_path).astype(np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != lock["seed_schedule_sha256"]:
        raise ValueError("locked Stage 4B seed schedule bytes changed")
    return lock


__all__ = [
    "BOOTSTRAP_ENTROPY_CONFIRM",
    "BOOTSTRAP_RESAMPLES_CONFIRM",
    "CANDIDATE_NS",
    "CF_ACTIONS",
    "CF_DURATIONS",
    "CODE_FILES",
    "COMBINED_IMPLEMENTATION",
    "CONFIDENCE_LEVEL",
    "EXPERT_ROUTER_SEEDS",
    "FINALIST",
    "GATE_I_AURC_MIN",
    "GATE_I_C80_MIN",
    "GATE_MIN_POSITIVE_PAIRS",
    "GATE_STRATA_I_AURC_FLOOR",
    "PLAN_ID",
    "POWER_BOOTSTRAP_RESAMPLES",
    "POWER_ENTROPY",
    "POWER_MONTE_CARLO",
    "POWER_SHRINK_MAIN",
    "POWER_SHRINK_SCENARIOS",
    "POWER_TARGET",
    "PRIMARY_CONSTANT_W",
    "PRIMARY_PREDICTOR",
    "PURPOSE",
    "RESPONSE_PREDICTOR_SEEDS",
    "SECOND_BASELINE",
    "SEED_ENTROPY",
    "SEED_HIGH_INCLUSIVE",
    "SEED_LOW",
    "SEED_SCHEDULE_SIZE",
    "STAGE4A_FROZEN",
    "STANDARD_BASELINE",
    "STATUS_COLLECTION_INCOMPLETE",
    "STATUS_CONFIRMED",
    "STATUS_INVALID",
    "STATUS_NOT_CONFIRMED",
    "STATUS_UNDERPOWERED",
    "STATE_TRANSITIONS",
    "TOP_ERROR_THRESHOLD",
    "assert_allowed_stage4b_data",
    "assert_balanced_strata",
    "build_protocol_lock",
    "inventory_frozen_stage4a",
    "inventory_model_lock",
    "make_candidate_seed_schedule",
    "scan_historical_seeds",
    "strata_counts",
    "validate_stage4b_config",
    "verify_protocol_lock",
    "write_once_json",
    "write_protocol_lock",
]
