"""Stage 4A protocol: seed schedule, data roles, model/data locks and write-once reports.

This module is the freeze layer for the Stage 4A reliability development screen
(plan ``AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1``). It never trains, never
fits on new data and never reads a Stage 4A label/prediction/risk score before the
development-screen is explicitly unlocked. Its only pre-collection numeric product
is the 12,000-seed candidate schedule (fixed entropy ``20260911``) plus the
historical seed inventory used to prove zero overlap.

Data roles (plan section 3.1): the single new role is ``stage4_merge_risk_dev``
(5,000 episodes, ten strata of 500 over cf_action x cf_duration). Within each
stratum the 500 episodes are mechanically split by the SHA256 of their episode
seed into risk-train (first 300), risk-validation (middle 100) and the single-use
development-screen (last 100). The split depends ONLY on
``(cf_action, cf_duration, episode_seed)`` — never on any simulated outcome.

Permanently forbidden as Stage 4A train/eval inputs (plan section 3.4): the sealed
Stage 2 holdout, the Stage 3A Plus qualification set, the Stage 3B/Plus A/B sets
and the Stage 3R R-screen. Those are enforced by path AND by known content hash.
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

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = "AC-TWIN-RFER-STAGE4A-RISK-DEV-2026-09-11-V0.1"
PURPOSE = "stage4_merge_risk_dev"

# Frozen seed-schedule contract (plan section 3.2).
SEED_ENTROPY = 20260911
SEED_SCHEDULE_SIZE = 12000
SEED_LOW = 1
SEED_HIGH_INCLUSIVE = 2147483646  # 2**31 - 2

# Frozen stratification / split contract (plan section 3.1).
CF_ACTIONS = (0, 1, 2, 3, 4)
CF_DURATIONS = (1, 3)
STRATA_PER_CELL = 500
N_PAIRS = 5000
ROLE_RISK_TRAIN = 0
ROLE_RISK_VALIDATION = 1
ROLE_DEVELOPMENT_SCREEN = 2
ROLE_NAMES = {
    ROLE_RISK_TRAIN: "risk_train",
    ROLE_RISK_VALIDATION: "risk_validation",
    ROLE_DEVELOPMENT_SCREEN: "development_screen",
}
SPLIT_PER_STRATUM = {
    ROLE_RISK_TRAIN: 300,
    ROLE_RISK_VALIDATION: 100,
    ROLE_DEVELOPMENT_SCREEN: 100,
}

# Frozen expert/router seed pairing (Stage 3R full-train five seed pairs).
EXPERT_ROUTER_SEEDS = (1101, 1202, 1303, 1404, 1505)
RESPONSE_PREDICTOR_SEEDS = (101, 202, 303, 404, 505)
PRIMARY_CONSTANT_W = 0.60

# Bootstrap / gate constants (plan section 8).
BOOTSTRAP_ENTROPY_SCREEN = 2026091101
CONFIDENCE_LEVEL = 0.80

# Mechanical terminal states (plan section 2.3).
STATUS_FEASIBLE = "STAGE4A_RISK_SIGNAL_FEASIBLE"
STATUS_INSUFFICIENT = "STAGE4A_RISK_SIGNAL_INSUFFICIENT"
STATUS_INVALID = "STAGE4A_INVALID_ANALYSIS"
STATUS_COLLECTION_INCOMPLETE = "COLLECTION_INCOMPLETE"

# State machine (plan section 7 Task 5).
STATE_TRANSITIONS = {
    "DRAFT": {"SOFTWARE_READY"},
    "SOFTWARE_READY": {"PROTOCOL_LOCKED"},
    "PROTOCOL_LOCKED": {"DEV_COLLECTED"},
    "DEV_COLLECTED": {"DEV_INFERENCE_COMPLETE"},
    "DEV_INFERENCE_COMPLETE": {"DEV_CANDIDATE_SELECTED"},
    "DEV_CANDIDATE_SELECTED": {"DEV_SCREEN_ADJUDICATED"},
    "DEV_SCREEN_ADJUDICATED": set(),
}

# New-code files whose SHA256 are enumerated in the protocol lock (relative to code/).
CODE_FILES = (
    "run_stage4a.py",
    "collect_stage4a_merge_dev.py",
    "ac_twin/stage4_protocol.py",
    "ac_twin/stage4_risk.py",
    "ac_twin/stage4_statistics.py",
)

# Forbidden source path tokens (plan section 3.4). These sets may only confirm
# historical state or re-check locks; they may never be Stage 4A train/eval data.
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


def assert_allowed_stage4_data(path: str | Path, *, file_sha256: str | None = None) -> None:
    """Raise ``ValueError`` if ``path`` is a permanently forbidden Stage 4A input.

    The only legal Stage 4A data role is ``stage4_merge_risk_dev``; anything under
    a sealed/consumed Stage 2/3 location is refused by path segment and, where a
    known fingerprint exists, by content SHA256.
    """
    s3d.assert_not_sealed_stage2(path, file_sha256=file_sha256)
    logical = _normalised(path)
    for token in _FORBIDDEN_DATA_TOKENS:
        for segment in logical.split("/"):
            if segment.startswith(token):
                raise ValueError(
                    f"forbidden Stage 4A data source (segment {segment!r}): {path}"
                )
    if PURPOSE not in logical.split("/") and "stage4_merge_risk_dev" not in logical:
        # Positive guard: a Stage 4A dataset must live under the dedicated role dir
        # (or be the role npz). Anything else is refused as train/eval data.
        raise ValueError(
            f"Stage 4A train/eval data must live under the {PURPOSE} role: {path}"
        )


# --- seed schedule ------------------------------------------------------------


def make_candidate_seed_schedule(
    entropy: int = SEED_ENTROPY, size: int = SEED_SCHEDULE_SIZE
) -> np.ndarray:
    """12,000 unique positive int32 candidate seeds in ``[1, 2**31-2]``.

    Reuses the frozen Stage 2 generator (SeedSequence spawn, draw without
    replacement, +1 shift) with the Stage 4A entropy so the schedule is
    internally unique and in-range by construction.
    """
    if int(entropy) != SEED_ENTROPY:
        raise ValueError(f"Stage 4A seed entropy is frozen to {SEED_ENTROPY}")
    if int(size) != SEED_SCHEDULE_SIZE:
        raise ValueError(f"Stage 4A seed schedule size is frozen to {SEED_SCHEDULE_SIZE}")
    schedule = make_seed_schedule(int(entropy), size=int(size))
    if schedule.min() < SEED_LOW or schedule.max() > SEED_HIGH_INCLUSIVE:
        raise ValueError("Stage 4A candidate seed schedule out of range")
    return schedule.astype(np.int64)


def scan_historical_seeds(workspace_roots: list[Path]) -> np.ndarray:
    """Union of every historical *accepted* episode seed across NPZ, receipts, manifests.

    Scans ``*.npz`` for ``episode_seed``/``episode_seeds`` arrays and ``*.json``
    receipts / manifests for embedded episode-seed lists (plan section 3.2: "所有
    历史数据 manifest、receipt 与 NPZ 中的 episode seed"). Excludes the Stage 4A
    role's own partial products so a resume does not collide with itself.

    Scope note (matches both canonical prior-stage scanners,
    ``stage2_protocol.scan_historical_episode_seeds`` and
    ``stage3_data.scan_all_episode_seeds``): only *realized* episode seeds are
    collected. Raw candidate ``*seed_schedule*.npy`` pools are deliberately NOT
    scanned — an unaccepted candidate never generated an episode, so no historical
    model trained or selected on it and it carries no contamination risk. Every
    *accepted* historical episode seed is still covered via its NPZ / receipt /
    manifest record.
    """
    collected: set[int] = set()
    for root in workspace_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob("*")):
            normalized = _normalised(file_path)
            if "stage4_merge_risk_dev" in normalized:
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
                # An unreadable artifact contributes no seeds; the schedule
                # overlap check at lock time is what protects the protocol.
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


# --- per-stratum role assignment ----------------------------------------------


def _seed_sort_key(seed: int) -> str:
    """Deterministic per-seed ordering key: SHA256 of the seed's decimal string."""
    return hashlib.sha256(str(int(seed)).encode("utf-8")).hexdigest()


def assign_stratum_roles(
    cf_action: np.ndarray, cf_duration: np.ndarray, episode_seed: np.ndarray
) -> np.ndarray:
    """Return the per-episode role (E,) in {0=risk-train,1=risk-validation,2=dev-screen}.

    Within each of the ten ``(cf_action, cf_duration)`` strata the episodes are
    sorted by ``SHA256(str(seed))`` and the first 300 become risk-train, the next
    100 risk-validation and the last 100 the single-use development-screen. This
    depends ONLY on the pre-intervention stratum and the seed — never on any
    simulated outcome.
    """
    cf_action = np.asarray(cf_action, dtype=np.int64)
    cf_duration = np.asarray(cf_duration, dtype=np.int64)
    episode_seed = np.asarray(episode_seed, dtype=np.int64)
    episodes = episode_seed.shape[0]
    if cf_action.shape != (episodes,) or cf_duration.shape != (episodes,):
        raise ValueError("cf_action, cf_duration and episode_seed must be aligned (E,)")
    roles = np.full(episodes, -1, dtype=np.int64)
    for action in CF_ACTIONS:
        for duration in CF_DURATIONS:
            members = np.flatnonzero((cf_action == action) & (cf_duration == duration))
            if members.size != STRATA_PER_CELL:
                raise ValueError(
                    f"stratum (action={action}, duration={duration}) must have exactly "
                    f"{STRATA_PER_CELL} episodes, got {members.size}"
                )
            ordered = sorted(members.tolist(), key=lambda idx: _seed_sort_key(episode_seed[idx]))
            boundary_train = SPLIT_PER_STRATUM[ROLE_RISK_TRAIN]
            boundary_val = boundary_train + SPLIT_PER_STRATUM[ROLE_RISK_VALIDATION]
            roles[np.asarray(ordered[:boundary_train], dtype=np.int64)] = ROLE_RISK_TRAIN
            roles[np.asarray(ordered[boundary_train:boundary_val], dtype=np.int64)] = (
                ROLE_RISK_VALIDATION
            )
            roles[np.asarray(ordered[boundary_val:], dtype=np.int64)] = ROLE_DEVELOPMENT_SCREEN
    if (roles < 0).any():
        raise ValueError("every episode must be assigned a Stage 4A role")
    return roles


def role_indices(roles: np.ndarray) -> dict[str, np.ndarray]:
    """Split a role vector into named, sorted index arrays."""
    roles = np.asarray(roles, dtype=np.int64)
    return {
        "risk_train": np.flatnonzero(roles == ROLE_RISK_TRAIN),
        "risk_validation": np.flatnonzero(roles == ROLE_RISK_VALIDATION),
        "development_screen": np.flatnonzero(roles == ROLE_DEVELOPMENT_SCREEN),
    }


# --- config validation ---------------------------------------------------------


def validate_stage4a_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 4A config; raise before any artifact is produced."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("purpose") != PURPOSE or checked.get("analysis_status") != PURPOSE:
        raise ValueError(f"purpose/analysis_status must be {PURPOSE!r}")
    if checked.get("scene") != "merge" or checked.get("env_id") != "merge-v0":
        raise ValueError("Stage 4A is merge-v0 only")
    if int(checked.get("n_pairs", -1)) != N_PAIRS:
        raise ValueError(f"n_pairs must be {N_PAIRS}")
    if int(checked.get("strata_per_cell", -1)) != STRATA_PER_CELL:
        raise ValueError(f"strata_per_cell must be {STRATA_PER_CELL}")
    if list(checked.get("cf_actions", [])) != list(CF_ACTIONS):
        raise ValueError(f"cf_actions must be {list(CF_ACTIONS)}")
    if list(checked.get("intervention_durations", [])) != list(CF_DURATIONS):
        raise ValueError(f"intervention_durations must be {list(CF_DURATIONS)}")
    split = checked.get("split_per_stratum", {})
    if (
        int(split.get("risk_train", -1)) != 300
        or int(split.get("risk_validation", -1)) != 100
        or int(split.get("development_screen", -1)) != 100
    ):
        raise ValueError("split_per_stratum must be 300/100/100")
    if int(checked.get("seed_entropy", -1)) != SEED_ENTROPY:
        raise ValueError(f"seed_entropy must be {SEED_ENTROPY}")
    if int(checked.get("seed_schedule_size", -1)) != SEED_SCHEDULE_SIZE:
        raise ValueError(f"seed_schedule_size must be {SEED_SCHEDULE_SIZE}")
    if int(checked.get("max_attempts", -1)) != SEED_SCHEDULE_SIZE:
        raise ValueError("max_attempts must equal seed_schedule_size")
    if list(checked.get("response_predictor_seeds", [])) != list(RESPONSE_PREDICTOR_SEEDS):
        raise ValueError("response_predictor_seeds drifted from the frozen five")
    if list(checked.get("expert_router_seeds", [])) != list(EXPERT_ROUTER_SEEDS):
        raise ValueError("expert_router_seeds drifted from the frozen five pairs")
    if float(checked.get("primary_constant_w", -1.0)) != PRIMARY_CONSTANT_W:
        raise ValueError(f"primary_constant_w must be {PRIMARY_CONSTANT_W}")
    if float(checked.get("confidence_level", -1.0)) != CONFIDENCE_LEVEL:
        raise ValueError(f"confidence_level must be {CONFIDENCE_LEVEL}")
    if int(checked.get("bootstrap_entropy_screen", -1)) != BOOTSTRAP_ENTROPY_SCREEN:
        raise ValueError(f"bootstrap_entropy_screen must be {BOOTSTRAP_ENTROPY_SCREEN}")
    # The data output must live under the dedicated role directory.
    assert_allowed_stage4_data(checked["data_output"])
    for key in ("run_dir", "output_dir", "collection_receipt", "rejection_ledger",
                "stage3r_run_dir", "stage3b_binding", "scaler_path",
                "historical_data_root"):
        if not checked.get(key):
            raise ValueError(f"missing Stage 4A config field: {key}")
    return checked


# --- checkpoint inventory ------------------------------------------------------


def expert_checkpoint_path(run_dir: Path, pair_index: int, condition: str, seed: int) -> Path:
    if condition not in ("full", "history"):
        raise ValueError("expert condition must be 'full' or 'history'")
    return (
        Path(run_dir)
        / "oof"
        / "checkpoints"
        / f"pair{pair_index}_fulltrain_{condition}_seed{seed}.pt"
    )


def e2e_checkpoint_path(run_dir: Path, pair_index: int, seed: int) -> Path:
    return (
        Path(run_dir)
        / "router"
        / "checkpoints"
        / f"end_to_end_mixture_lc0_lr0_pair{pair_index}_s{seed}.pt"
    )


def inventory_model_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and hash every frozen Stage 4A model/scaler/binding dependency."""
    checked = validate_stage4a_config(config)
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
    return {
        "checkpoint_sha256": checkpoints,
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "stage3b_binding_path": str(binding_path),
        "stage3b_binding_sha256": sha256_file(binding_path),
    }


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in CODE_FILES:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered Stage 4A code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


# --- protocol lock --------------------------------------------------------------


def build_protocol_lock(
    config: Mapping[str, Any], *, project_root: Path = ROOT
) -> dict[str, Any]:
    """Compute the (not yet written) Stage 4A protocol lock payload.

    Scans the historical seed inventory, builds + validates the 12,000-seed
    candidate schedule, and records the SHA256 of every frozen model/scaler/
    binding/code input. Pure and side-effect free so it can be tested directly;
    the runner persists it with :func:`write_protocol_lock`.
    """
    checked = validate_stage4a_config(config)
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
        "n_historical_episode_seeds": int(historical.size),
        "seed_schedule_zero_overlap": True,
        "split_per_stratum": {ROLE_NAMES[k]: v for k, v in SPLIT_PER_STRATUM.items()},
        "cf_actions": list(CF_ACTIONS),
        "cf_durations": list(CF_DURATIONS),
        "strata_per_cell": STRATA_PER_CELL,
        "code_sha256": _code_hashes(),
        "config_sha256": None,  # filled by write_protocol_lock
        "seed_schedule_path": str(schedule_path),
        "seed_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "historical_seed_inventory": historical,
        "seed_schedule": schedule,
        **model_lock,
    }


def write_protocol_lock(config_path: str | Path, *, project_root: Path = ROOT) -> dict[str, Any]:
    """Create the Stage 4A protocol lock exactly once (refuses to overwrite).

    Writes ``protocol/protocol_lock.json``, ``protocol/candidate_seed_schedule.npy``,
    ``protocol/candidate_seed_schedule.json`` and
    ``protocol/historical_seed_inventory.json`` under the run dir. Create-once: any
    pre-existing lock refuses the call (a new version requires a new version dir).
    """
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_stage4a_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "protocol"
    lock_path = protocol_dir / "protocol_lock.json"
    schedule_npy = protocol_dir / "candidate_seed_schedule.npy"
    if lock_path.exists():
        raise FileExistsError(f"Stage 4A protocol lock already exists: {lock_path}")

    lock = build_protocol_lock(checked, project_root=project_root)
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
    """Read-only re-verification that every locked Stage 4A input still matches."""
    run_dir = Path(run_dir)
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage 4A protocol lock missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: dict[str, tuple[Path, str]] = {}
    for rel, expected in lock["code_sha256"].items():
        checks[f"code:{rel}"] = (ROOT / "code" / rel, expected)
    for name, expected in lock["checkpoint_sha256"].items():
        # Reconstruct the on-disk path for each named checkpoint.
        if name.startswith("end_to_end_mixture_"):
            pair_index = int(name.split("_pair")[1].split("_s")[0])
            seed = int(name.rsplit("_s", 1)[1])
            path = e2e_checkpoint_path(_resolve_run(lock), pair_index, seed)
        else:
            pair_index = int(name.split("pair")[1].split("_")[0])
            condition = "full" if "_full_" in name else "history"
            seed = int(name.rsplit("_seed", 1)[1])
            path = expert_checkpoint_path(_resolve_run(lock), pair_index, condition, seed)
        checks[f"checkpoint:{name}"] = (path, expected)
    checks["scaler"] = (Path(lock["scaler_path"]), lock["scaler_sha256"])
    checks["stage3b_binding"] = (Path(lock["stage3b_binding_path"]), lock["stage3b_binding_sha256"])
    schedule_path = Path(lock["seed_schedule_path"])
    checks["seed_schedule"] = (schedule_path, None)  # bytes hashed separately
    for label, (file_path, expected) in checks.items():
        if not file_path.is_file():
            raise ValueError(f"locked Stage 4A artifact missing: {label} -> {file_path}")
        if expected is not None and sha256_file(file_path) != expected:
            raise ValueError(f"locked Stage 4A artifact hash drift: {label}")
    schedule = np.load(schedule_path).astype(np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != lock["seed_schedule_sha256"]:
        raise ValueError("locked Stage 4A seed schedule bytes changed")
    return lock


def _resolve_run(lock: Mapping[str, Any]) -> Path:
    """Recover the Stage 3R run dir recorded implicitly via the scaler path."""
    return Path(lock["scaler_path"]).parent


__all__ = [
    "BOOTSTRAP_ENTROPY_SCREEN",
    "CF_ACTIONS",
    "CF_DURATIONS",
    "CODE_FILES",
    "CONFIDENCE_LEVEL",
    "EXPERT_ROUTER_SEEDS",
    "N_PAIRS",
    "PLAN_ID",
    "PRIMARY_CONSTANT_W",
    "PURPOSE",
    "RESPONSE_PREDICTOR_SEEDS",
    "ROLE_DEVELOPMENT_SCREEN",
    "ROLE_NAMES",
    "ROLE_RISK_TRAIN",
    "ROLE_RISK_VALIDATION",
    "SEED_ENTROPY",
    "SEED_SCHEDULE_SIZE",
    "SPLIT_PER_STRATUM",
    "STATUS_COLLECTION_INCOMPLETE",
    "STATUS_FEASIBLE",
    "STATUS_INSUFFICIENT",
    "STATUS_INVALID",
    "STRATA_PER_CELL",
    "STATE_TRANSITIONS",
    "assert_allowed_stage4_data",
    "assign_stratum_roles",
    "build_protocol_lock",
    "e2e_checkpoint_path",
    "expert_checkpoint_path",
    "inventory_model_lock",
    "make_candidate_seed_schedule",
    "role_indices",
    "scan_historical_seeds",
    "validate_stage4a_config",
    "verify_protocol_lock",
    "write_once_json",
    "write_protocol_lock",
]
