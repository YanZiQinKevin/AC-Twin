"""Stage 2B intersection-v0 protocol: config validation, per-role seed schedules,
historical/cross-role seed disjointness, pre-collection manifests and the
mechanical pilot profile-selection rule.

Plan: AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-14-V0.3.

This module is the freeze layer for the fixed-participant intersection
replication. It never trains, never fits normalization or response bins, and
never reads realized response magnitudes, model errors or any Gate quantity.
The only numeric products created here are (a) per-role candidate seed
schedules and (b) the create-once collection manifests that bind a role config
to its schedule before any collection starts.

Frozen random streams derive from ``numpy.random.SeedSequence(collection_seed).
spawn(3)`` (same construction as Stage 2):

1. ``seed_schedule_rng`` -> the candidate episode seeds (consumed offline here);
2. ``action_rng`` -> t_star / prefix / factual action plans (collector);
3. ``stratum_rng`` -> counterfactual stratum tie-breaking (collector).

Seeds are drawn without replacement from ``[0, 2147483646)`` and shifted by +1,
guaranteeing internal uniqueness and the closed range ``[1, 2147483646]``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402

# --- frozen identity ------------------------------------------------------------

PLAN_ID = "AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-14-V0.3"
V04_PLAN_ID = "AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-15-V0.4"
SCENE = "intersection"
ENV_ID = "intersection-v0"
HIGHWAY_ENV_VERSION = "1.10.2"

# Frozen action semantics: intersection-v0 DiscreteMetaAction (longitudinal only).
ACTION_SEMANTICS = {"SLOWER": 0, "IDLE": 1, "FASTER": 2}
ACTION_IDS = (0, 1, 2)
INTERVENTION_DURATIONS = (1, 3)
N_STRATA = len(ACTION_IDS) * len(INTERVENTION_DURATIONS)

HISTORY_LEN = 10
FUTURE_LEN = 20
T_STAR_LOW = 10
T_STAR_HIGH = 15
DURATION_S = 20
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5
CONTROLLED_VEHICLES = 1
SPAWN_PROBABILITY = 0.0
OBJECT_CAPACITY = 20
OBSERVED_VEHICLES_COUNT = 20
ZERO_TOLERANCE = 1e-4

SEED_LOW = 1
SEED_HIGH_EXCLUSIVE = 2147483647  # integers are drawn from [1, 2147483646]

# Split labels stored per accepted episode.
SPLIT_LABELS = {"train": 0, "validation": 1, "confirm": 2, "pilot": 3, "smoke": 4}

# Roles and their frozen collection seeds (distinct from all derangement seeds).
ROLES = ("smoke", "pilot_a", "pilot_b", "development", "confirm")
COLLECTION_SEEDS = {
    "smoke": 2026091450,
    "pilot_a": 2026091501,
    "pilot_b": 2026091452,
    "development": 2026091504,
    "confirm": 2026091563,
}
V04_COLLECTION_SEEDS = {
    "development": 2026091609,
}

STATUS_SMOKE = "ENGINEERING_SMOKE_ONLY"
STATUS_PILOT = "PILOT_RESPONSE_SUPPORT"
STATUS_DEVELOPMENT = "DEVELOPMENT"
STATUS_CONFIRM = "CONFIRMATORY"

# Pilot (plan section 5): two preregistered profiles, fixed quotas and cap.
PROFILES = {"A": {"initial_vehicle_count": 10}, "B": {"initial_vehicle_count": 15}}
PILOT_ACCEPTED_PER_PROFILE = 600
PILOT_PER_STRATUM = 100
PILOT_MAX_ATTEMPTS = 12000
PILOT_MIN_SUPPORT_RATE = 0.40
PILOT_MIN_POSITIVE_EPISODES = 300
PILOT_MIN_MEDIAN_PAIRED_VALID = 3
PILOT_MIN_BIN_EPISODES = 100

# Development (plan section 6.1): 6,000 accepted, mechanically pre-split.
DEV_N_PAIRS = 6000
DEV_PER_STRATUM = 1000
DEV_TRAIN_PER_STRATUM = 800
DEV_VALIDATION_PER_STRATUM = 200
DEV_MAX_ATTEMPTS = 60000
DEV_SEED_SCHEDULE_SIZE = 60000

# Smoke (plan section 4): 12 ENGINEERING_SMOKE_ONLY episodes at the denser
# profile-B initial count to exercise near-capacity behaviour.
SMOKE_N_PAIRS = 12
SMOKE_PER_STRATUM = 2
SMOKE_MAX_ATTEMPTS = 512
SMOKE_SEED_SCHEDULE_SIZE = 512
SMOKE_INITIAL_VEHICLE_COUNT = 15

# Confirm candidate sizes (plan section 7.2); the final n is fixed at P3.
CONFIRM_CANDIDATE_N = (1200, 1800, 2400, 3600)

# Model conditions and seeds (plan section 6.2).
TRAINING_SEEDS = (11, 22, 33, 44, 55)
CONDITIONS = ("no_suffix", "uniform_suffix", "shuffled_suffix")

# Frozen training hyperparameters (C0) and the single preregistered
# augmentation C1 (plan section 6.3).
TRAINING_FROZEN = {
    "object_embedding_dim": 64,
    "hidden_dim": 64,
    "action_embedding_dim": 8,
    "batch_size": 128,
    "max_epochs": 200,
    "early_stopping_patience": 25,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 5.0,
}
AUGMENTATION_C1_FROZEN = {
    "object_embedding_dim": 96,
    "hidden_dim": 96,
    "action_embedding_dim": 8,
}

# Derangement seeds frozen by the plan (section 7.3).
DERANGEMENT_SEEDS = {"train": 2026091401, "validation": 2026091402, "confirm": 2026091403}
ALTERNATE_DERANGEMENT_SEEDS = (2026091413, 2026091423, 2026091433, 2026091443)

# Keys that would imply training/refitting; forbidden outside the development
# config's preregistered training blocks.
_FORBIDDEN_RETRAIN_KEYS = (
    "learning_rate", "max_epochs", "early_stopping_patience", "weight_decay",
    "gradient_clip_norm", "optimizer", "training_seeds_to_fit", "fit_normalization",
    "fit_response_bins", "retrain", "fine_tune",
)

_COMMON_REQUIRED = {
    "experiment_name", "analysis_status", "plan_id", "scene", "env_id", "role",
    "n_pairs", "per_stratum", "collection_seed", "seed_schedule_size",
    "max_attempts", "horizon", "history_len", "t_star_low", "t_star_high",
    "intervention_durations", "object_capacity", "observed_vehicles_count",
    "duration", "simulation_frequency", "policy_frequency",
    "controlled_vehicles", "spawn_probability", "determinism_checks",
    "output_dir", "data_output", "rejection_ledger", "collection_receipt",
}


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


# --- configuration validation -----------------------------------------------------


def _validate_common(config: Mapping[str, Any]) -> None:
    missing = _COMMON_REQUIRED - set(config)
    if missing:
        raise ValueError(f"Stage 2B config missing keys: {sorted(missing)}")
    if config["plan_id"] not in (PLAN_ID, V04_PLAN_ID):
        raise ValueError(f"plan_id must be {PLAN_ID!r} or {V04_PLAN_ID!r}")
    if config["scene"] != SCENE or config["env_id"] != ENV_ID:
        raise ValueError(f"Stage 2B is {ENV_ID!r} only")
    frozen = {
        "horizon": FUTURE_LEN, "history_len": HISTORY_LEN,
        "t_star_low": T_STAR_LOW, "t_star_high": T_STAR_HIGH,
        "object_capacity": OBJECT_CAPACITY,
        "observed_vehicles_count": OBSERVED_VEHICLES_COUNT,
        "duration": DURATION_S, "simulation_frequency": SIMULATION_FREQUENCY,
        "policy_frequency": POLICY_FREQUENCY,
        "controlled_vehicles": CONTROLLED_VEHICLES,
    }
    for key, expected in frozen.items():
        if int(config[key]) != expected:
            raise ValueError(f"config.{key} must equal the frozen {expected}")
    if float(config["spawn_probability"]) != SPAWN_PROBABILITY:
        raise ValueError("spawn_probability must be 0.0 (fixed-participant contract)")
    if [int(d) for d in config["intervention_durations"]] != list(INTERVENTION_DURATIONS):
        raise ValueError(f"intervention_durations must be {list(INTERVENTION_DURATIONS)}")
    if int(config["per_stratum"]) * N_STRATA != int(config["n_pairs"]):
        raise ValueError("n_pairs must equal per_stratum * 6 (three actions x two durations)")
    if int(config["max_attempts"]) > int(config["seed_schedule_size"]):
        raise ValueError("seed_schedule_size must cover max_attempts")
    for key in config:
        if key in _FORBIDDEN_RETRAIN_KEYS:
            raise ValueError(f"config.{key} is forbidden at top level")


def validate_stage2b_config(config: Mapping[str, Any]) -> None:
    """Validate one Stage 2B role config; raise ValueError before any output.

    Dispatches on ``config['role']``. Smoke/pilot/development only; the confirm
    config is generated by the P3 freeze command and validated then.
    """
    role = config.get("role")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    _validate_common(config)

    if config["plan_id"] == V04_PLAN_ID:
        if role not in V04_COLLECTION_SEEDS:
            raise ValueError(f"v0.4 role {role!r} is not frozen yet")
        expected_collection_seed = V04_COLLECTION_SEEDS[role]
    else:
        expected_collection_seed = COLLECTION_SEEDS[role]
    if config["collection_seed"] != expected_collection_seed:
        raise ValueError(
            f"collection_seed for role {role!r} must be {expected_collection_seed}")

    if role == "smoke":
        if config["analysis_status"] != STATUS_SMOKE:
            raise ValueError(f"smoke analysis_status must be {STATUS_SMOKE!r}")
        if int(config["n_pairs"]) != SMOKE_N_PAIRS:
            raise ValueError(f"smoke requires n_pairs == {SMOKE_N_PAIRS}")
        if int(config["initial_vehicle_count"]) != SMOKE_INITIAL_VEHICLE_COUNT:
            raise ValueError(
                "smoke runs at the denser profile-B initial_vehicle_count "
                f"{SMOKE_INITIAL_VEHICLE_COUNT} to exercise near-capacity behaviour")
        if int(config["seed_schedule_size"]) != SMOKE_SEED_SCHEDULE_SIZE:
            raise ValueError("smoke seed_schedule_size is frozen")
        if "movement_lookup_json" not in config:
            raise ValueError("smoke config must name movement_lookup_json")
        return

    if role in ("pilot_a", "pilot_b"):
        if config["analysis_status"] != STATUS_PILOT:
            raise ValueError(f"pilot analysis_status must be {STATUS_PILOT!r}")
        profile = role.split("_")[1].upper()
        if int(config["initial_vehicle_count"]) != PROFILES[profile]["initial_vehicle_count"]:
            raise ValueError(f"pilot profile {profile} initial_vehicle_count is frozen")
        if int(config["n_pairs"]) != PILOT_ACCEPTED_PER_PROFILE:
            raise ValueError(f"pilot requires n_pairs == {PILOT_ACCEPTED_PER_PROFILE}")
        if int(config["per_stratum"]) != PILOT_PER_STRATUM:
            raise ValueError(f"pilot requires per_stratum == {PILOT_PER_STRATUM}")
        if int(config["max_attempts"]) != PILOT_MAX_ATTEMPTS:
            raise ValueError(f"pilot max_attempts is capped at {PILOT_MAX_ATTEMPTS}")
        if int(config["seed_schedule_size"]) != PILOT_MAX_ATTEMPTS:
            raise ValueError("pilot seed_schedule_size must equal the attempt cap")
        return

    if role == "development":
        if config["analysis_status"] != STATUS_DEVELOPMENT:
            raise ValueError(f"development analysis_status must be {STATUS_DEVELOPMENT!r}")
        if int(config["n_pairs"]) != DEV_N_PAIRS:
            raise ValueError(f"development requires n_pairs == {DEV_N_PAIRS}")
        if int(config["per_stratum"]) != DEV_PER_STRATUM:
            raise ValueError(f"development requires per_stratum == {DEV_PER_STRATUM}")
        if int(config["train_per_stratum"]) != DEV_TRAIN_PER_STRATUM:
            raise ValueError(f"development train_per_stratum == {DEV_TRAIN_PER_STRATUM}")
        if int(config["validation_per_stratum"]) != DEV_VALIDATION_PER_STRATUM:
            raise ValueError(
                f"development validation_per_stratum == {DEV_VALIDATION_PER_STRATUM}")
        if DEV_TRAIN_PER_STRATUM + DEV_VALIDATION_PER_STRATUM != DEV_PER_STRATUM:
            raise ValueError("development split must partition each stratum")
        if int(config["seed_schedule_size"]) != DEV_SEED_SCHEDULE_SIZE:
            raise ValueError("development seed_schedule_size is frozen")
        if not isinstance(config.get("initial_vehicle_count"), int) or \
                int(config["initial_vehicle_count"]) not in (10, 15):
            raise ValueError(
                "development initial_vehicle_count must come from the frozen "
                "profile selection (10 or 15); run select-profile first")
        if "profile_selection_json" not in config:
            raise ValueError("development config must name profile_selection_json")
        training = config.get("training", {})
        for key, expected in TRAINING_FROZEN.items():
            if key not in training:
                raise ValueError(f"development training block missing {key!r}")
            if float(training[key]) != float(expected):
                raise ValueError(f"training.{key} must equal the frozen {expected}")
        c1 = config.get("augmentation_c1", {})
        for key, expected in AUGMENTATION_C1_FROZEN.items():
            if int(c1.get(key, -1)) != expected:
                raise ValueError(f"augmentation_c1.{key} must equal the frozen {expected}")
        if list(config.get("training_seeds", [])) != list(TRAINING_SEEDS):
            raise ValueError(f"training_seeds must be {list(TRAINING_SEEDS)}")
        if list(config.get("conditions", [])) != list(CONDITIONS):
            raise ValueError(f"conditions must be {list(CONDITIONS)}")
        return

    # role == "confirm": generated at P3; nothing to collect yet.
    raise ValueError(
        "confirm configs are generated by the P3 freeze command, not hand-written")


# --- pilot container config (two profiles in one file) --------------------------------

_PILOT_SHARED_KEYS = (
    "horizon", "history_len", "t_star_low", "t_star_high", "intervention_durations",
    "object_capacity", "observed_vehicles_count", "duration", "simulation_frequency",
    "policy_frequency", "controlled_vehicles", "spawn_probability",
    "determinism_checks",
)


def validate_pilot_container_config(config: Mapping[str, Any]) -> None:
    """Validate the two-profile pilot container file (never collected directly)."""
    if config.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if config.get("scene") != SCENE or config.get("env_id") != ENV_ID:
        raise ValueError(f"Stage 2B is {ENV_ID!r} only")
    if config.get("analysis_status") != STATUS_PILOT:
        raise ValueError(f"pilot analysis_status must be {STATUS_PILOT!r}")
    for key in _PILOT_SHARED_KEYS:
        if key not in config:
            raise ValueError(f"pilot container missing shared key {key!r}")
    if int(config.get("accepted_per_profile", -1)) != PILOT_ACCEPTED_PER_PROFILE:
        raise ValueError(f"pilot accepted_per_profile == {PILOT_ACCEPTED_PER_PROFILE}")
    if int(config.get("per_stratum", -1)) != PILOT_PER_STRATUM:
        raise ValueError(f"pilot per_stratum == {PILOT_PER_STRATUM}")
    if int(config.get("max_attempts_per_profile", -1)) != PILOT_MAX_ATTEMPTS:
        raise ValueError(f"pilot max_attempts_per_profile == {PILOT_MAX_ATTEMPTS}")
    profiles = config.get("profiles", {})
    if set(profiles) != {"A", "B"}:
        raise ValueError("pilot container must define exactly profiles A and B")
    for letter, block in profiles.items():
        for key in ("initial_vehicle_count", "output_dir", "data_output",
                    "rejection_ledger", "collection_receipt"):
            if key not in block:
                raise ValueError(f"pilot profile {letter} missing {key!r}")
        if int(block["initial_vehicle_count"]) != PROFILES[letter]["initial_vehicle_count"]:
            raise ValueError(f"pilot profile {letter} initial_vehicle_count is frozen")
    if "profile_selection_json" not in config:
        raise ValueError("pilot container must name profile_selection_json")


def resolve_pilot_profile_config(container: Mapping[str, Any], profile: str) -> dict[str, Any]:
    """Build the effective single-role config for one pilot profile (A or B)."""
    validate_pilot_container_config(container)
    if profile not in ("A", "B"):
        raise ValueError("profile must be 'A' or 'B'")
    block = container["profiles"][profile]
    effective = {key: container[key] for key in _PILOT_SHARED_KEYS}
    effective.update({
        "experiment_name": f"{container['experiment_name']}_{profile.lower()}",
        "analysis_status": container["analysis_status"],
        "plan_id": container["plan_id"],
        "scene": container["scene"],
        "env_id": container["env_id"],
        "role": f"pilot_{profile.lower()}",
        "initial_vehicle_count": int(block["initial_vehicle_count"]),
        "n_pairs": int(container["accepted_per_profile"]),
        "per_stratum": int(container["per_stratum"]),
        "collection_seed": COLLECTION_SEEDS[f"pilot_{profile.lower()}"],
        "seed_schedule_size": int(container["max_attempts_per_profile"]),
        "max_attempts": int(container["max_attempts_per_profile"]),
        "output_dir": block["output_dir"],
        "data_output": block["data_output"],
        "rejection_ledger": block["rejection_ledger"],
        "collection_receipt": block["collection_receipt"],
        "checkpoint_every_accepted": int(container.get("checkpoint_every_accepted", 25)),
    })
    validate_stage2b_config(effective)
    return effective


def resolve_development_config(config: Mapping[str, Any],
                               profile_selection: Mapping[str, Any]) -> dict[str, Any]:
    """Inject the frozen selected profile into the development config.

    The on-disk development config carries ``initial_vehicle_count: null``; it
    is filled exclusively from the create-once ``profile_selection.json``.
    """
    if profile_selection.get("selected_profile") not in ("A", "B"):
        raise ValueError("profile selection must name a selected profile A or B")
    if int(profile_selection["initial_vehicle_count"]) not in (10, 15):
        raise ValueError("profile selection carries an invalid initial_vehicle_count")
    resolved = dict(config)
    resolved["initial_vehicle_count"] = int(profile_selection["initial_vehicle_count"])
    resolved["selected_profile"] = profile_selection["selected_profile"]
    validate_stage2b_config(resolved)
    return resolved


# --- historical seed scan ----------------------------------------------------------


def scan_historical_episode_seeds(data_root: str | Path,
                                  experiments_root: str | Path | None = None
                                  ) -> np.ndarray:
    """Union of every historical episode seed Stage 2B must avoid.

    Scans (a) every ``episode_seed``/``episode_seeds`` array in npz files under
    ``data_root`` and (b) every ``seed_schedule*.npy`` under ``experiments_root``.
    Stage 2B's own products (paths containing ``stage2b_intersection``) are
    excluded: they derive from Stage 2B's own schedules and would otherwise
    block later roles (e.g. confirm) after earlier roles have landed. All Stage
    2/3/4A merge products remain in scope as consumed history.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    collected: set[int] = set()
    for file_path in sorted(data_root.rglob("*.npz")):
        if "stage2b_intersection" in str(file_path).replace("\\", "/"):
            continue
        try:
            with np.load(file_path, allow_pickle=True) as archive:
                for key in ("episode_seed", "episode_seeds"):
                    if key in archive.files:
                        values = np.atleast_1d(np.asarray(archive[key])).ravel()
                        collected.update(int(v) for v in values)
        except Exception:
            continue
    if experiments_root is not None:
        experiments_root = Path(experiments_root)
        if experiments_root.is_dir():
            for file_path in sorted(experiments_root.rglob("seed_schedule*.npy")):
                if "stage2b_intersection" in str(file_path).replace("\\", "/"):
                    continue
                try:
                    values = np.load(file_path).astype(np.int64).ravel()
                    collected.update(int(v) for v in values)
                except Exception:
                    continue
    if not collected:
        raise ValueError(f"no historical episode seeds found under {data_root}")
    return np.array(sorted(collected), dtype=np.int64)


# --- seed schedule -----------------------------------------------------------------


def _spawn_streams(collection_seed: int) -> tuple[np.random.Generator, ...]:
    spawned = np.random.SeedSequence(int(collection_seed)).spawn(3)
    return tuple(np.random.default_rng(s) for s in spawned)


def make_seed_schedule(collection_seed: int, size: int) -> np.ndarray:
    """Candidate episode seeds from the ``seed_schedule_rng`` stream.

    Same construction as Stage 2: drawn without replacement from
    ``[0, 2147483646)`` and shifted by +1, so internally unique and in the
    closed range ``[1, 2147483646]`` by construction.
    """
    size = int(size)
    if size <= 0 or size > (SEED_HIGH_EXCLUSIVE - 1):
        raise ValueError("seed schedule size out of feasible range")
    seed_schedule_rng = _spawn_streams(collection_seed)[0]
    schedule = seed_schedule_rng.choice(
        SEED_HIGH_EXCLUSIVE - 1, size=size, replace=False) + 1
    return schedule.astype(np.int64)


def validate_seed_schedule(schedule: np.ndarray, historical: np.ndarray,
                           other_schedules: Mapping[str, np.ndarray] | None = None
                           ) -> None:
    """Raise unless in-range, internally unique and disjoint from history and
    from every other Stage 2B role schedule supplied."""
    schedule = np.asarray(schedule, dtype=np.int64)
    historical = np.asarray(historical, dtype=np.int64)
    if schedule.size == 0:
        raise ValueError("seed schedule must be non-empty")
    if int(schedule.min()) < SEED_LOW or int(schedule.max()) > SEED_HIGH_EXCLUSIVE - 1:
        raise ValueError(
            f"seed schedule out of range [{SEED_LOW}, {SEED_HIGH_EXCLUSIVE - 1}]")
    if len(np.unique(schedule)) != schedule.size:
        raise ValueError("seed schedule contains duplicate seeds")
    overlap = np.intersect1d(schedule, historical)
    if overlap.size:
        raise ValueError(
            f"seed schedule overlaps {overlap.size} historical episode seed(s); "
            "change the whole candidate schedule (collection_seed), never single seeds")
    for name, other in (other_schedules or {}).items():
        overlap = np.intersect1d(schedule, np.asarray(other, dtype=np.int64))
        if overlap.size:
            raise ValueError(
                f"seed schedule overlaps {overlap.size} seed(s) with role {name!r}; "
                "replace the entire candidate schedule")


# --- collection manifest (pre-collection binding for smoke/pilot/development) -------


def _environment_versions() -> dict[str, str]:
    """Record runtime versions without importing heavy/optional packages."""
    from importlib.metadata import version as _pkg_version
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": _pkg_version("numpy"),
        "torch_version": _pkg_version("torch"),
        "highway_env_version": _pkg_version("highway_env"),
        "gymnasium_version": _pkg_version("gymnasium"),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_write_json(target: Path, obj: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def _other_role_schedules(stage2b_root: Path, own_output_dir: Path) -> dict[str, np.ndarray]:
    """Load every other Stage 2B role schedule already on disk."""
    others: dict[str, np.ndarray] = {}
    own = own_output_dir.resolve()
    if not stage2b_root.is_dir():
        return others
    for path in sorted(stage2b_root.glob("*/seed_schedule.npy")):
        if path.parent.resolve() == own:
            continue
        try:
            others[path.parent.name] = np.load(path).astype(np.int64)
        except Exception:
            continue
    return others


def prepare_collection(config: Mapping[str, Any], output_dir: str | Path,
                       *, config_path: str | Path | None = None,
                       extra_bindings: Mapping[str, str] | None = None,
                       stage2b_root: str | Path | None = None) -> dict[str, Any]:
    """Create the per-role collection manifest + seed schedule exactly once.

    Validates the config, scans historical seeds, builds the candidate schedule,
    verifies zero overlap with history and with every other Stage 2B role
    schedule already prepared, then atomically writes ``seed_schedule.npy`` and
    ``collection_manifest.json``. Refuses to overwrite either artifact.

    ``extra_bindings`` maps a label to a root-relative (or absolute) file path
    whose SHA256 is sealed into the manifest (e.g. the P0 movement lookup, the
    frozen profile selection).
    """
    validate_stage2b_config(config)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "collection_manifest.json"
    schedule_path = output_dir / "seed_schedule.npy"
    if manifest_path.exists() or schedule_path.exists():
        raise FileExistsError(
            f"collection manifest/schedule already exist and are create-once: {output_dir}")

    historical = scan_historical_episode_seeds(ROOT / "data", ROOT / "experiments")
    schedule = make_seed_schedule(int(config["collection_seed"]),
                                  int(config["seed_schedule_size"]))
    root = Path(stage2b_root) if stage2b_root else (ROOT / "experiments" / "stage2b_intersection")
    others = _other_role_schedules(root, output_dir)
    validate_seed_schedule(schedule, historical, others)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_schedule = schedule_path.with_name(schedule_path.name + ".tmp")
    with open(tmp_schedule, "wb") as handle:
        np.save(handle, schedule)
    Path(tmp_schedule).replace(schedule_path)

    bindings: dict[str, Any] = {}
    for label, path_value in (extra_bindings or {}).items():
        binding_path = _resolve(path_value)
        if not binding_path.is_file():
            raise FileNotFoundError(f"manifest binding missing: {label} -> {binding_path}")
        bindings[label] = {"path": str(binding_path),
                           "sha256": sha256_file(binding_path)}

    manifest: dict[str, Any] = {
        "plan_id": config["plan_id"],
        "role": config["role"],
        "experiment_name": config["experiment_name"],
        "analysis_status": config["analysis_status"],
        "collection_seed": int(config["collection_seed"]),
        "seed_schedule_size": int(config["seed_schedule_size"]),
        "max_attempts": int(config["max_attempts"]),
        "config_signature": canonical_fingerprint(json.loads(json.dumps(config))),
        "config_sha256": sha256_file(config_path) if config_path else None,
        "seed_schedule_sha256": sha256_file(schedule_path),
        "zero_overlap": {
            "historical_episode_seeds": int(len(historical)),
            "other_role_schedules": sorted(others),
            "disjoint": True,
        },
        "bindings": bindings,
        "environment": _environment_versions(),
    }
    body = json.dumps(manifest, sort_keys=True, ensure_ascii=False, default=_json_scalar)
    manifest["manifest_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    _atomic_write_json(manifest_path, manifest)
    return manifest


def verify_collection_manifest(output_dir: str | Path,
                               config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only verification that the manifest, schedule and bindings match.

    Raises on the first mismatch; never writes. Returns the manifest on success.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "collection_manifest.json"
    schedule_path = output_dir / "seed_schedule.npy"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"collection manifest missing: {manifest_path}")
    if not schedule_path.is_file():
        raise FileNotFoundError(f"seed schedule missing: {schedule_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stored = manifest.get("manifest_sha256")
    body = json.dumps({k: v for k, v in manifest.items() if k != "manifest_sha256"},
                      sort_keys=True, ensure_ascii=False, default=_json_scalar)
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != stored:
        raise ValueError("collection manifest self-hash mismatch (file was modified)")
    if manifest["config_signature"] != canonical_fingerprint(
            json.loads(json.dumps(config))):
        raise ValueError("running config does not match the manifested configuration")
    if sha256_file(schedule_path) != manifest["seed_schedule_sha256"]:
        raise ValueError("seed schedule hash drift")
    for label, binding in manifest.get("bindings", {}).items():
        binding_path = Path(binding["path"])
        if not binding_path.is_file():
            raise ValueError(f"manifest binding missing: {label} -> {binding_path}")
        if sha256_file(binding_path) != binding["sha256"]:
            raise ValueError(f"manifest binding hash drift: {label}")
    return manifest


# --- pilot profile selection (plan section 5.2, mechanical) -------------------------


def profile_qualifies(metrics: Mapping[str, Any] | None) -> bool:
    """A profile qualifies only when every preregistered condition holds.

    ``metrics`` must be the per-profile pilot support summary (never V_evt or
    any model error); ``None`` means the profile failed to collect.
    """
    if metrics is None:
        return False
    return bool(
        int(metrics["accepted_pairs"]) == PILOT_ACCEPTED_PER_PROFILE
        and int(metrics["per_stratum_min"]) >= PILOT_PER_STRATUM
        and bool(metrics["prefix_identity_qa_pass"])
        and float(metrics["support_rate"]) >= PILOT_MIN_SUPPORT_RATE
        and int(metrics["positive_episodes"]) >= PILOT_MIN_POSITIVE_EPISODES
        and float(metrics["median_paired_valid_non_ego"]) >= PILOT_MIN_MEDIAN_PAIRED_VALID
        and int(metrics["bin_episode_counts"]["positive_low"]) >= PILOT_MIN_BIN_EPISODES
        and int(metrics["bin_episode_counts"]["positive_mid"]) >= PILOT_MIN_BIN_EPISODES
        and int(metrics["bin_episode_counts"]["positive_high"]) >= PILOT_MIN_BIN_EPISODES
    )


def build_v04_profile_b_recovery(metrics: Mapping[str, Any] | None
                                 ) -> dict[str, Any]:
    """Build the v0.4 recovery decision from the completed v0.3 B pilot.

    The v0.3 absolute ``positive_episodes >= 300`` test is intentionally not
    part of this rule: at the frozen pilot size of 600 it silently superseded
    the registered 40% support floor with 50%.  The downstream-relevant
    positive-bin occupancy tests remain mandatory.
    """
    if metrics is None:
        raise ValueError("v0.3 profile B metrics are missing")
    qualifies = bool(
        int(metrics["accepted_pairs"]) == PILOT_ACCEPTED_PER_PROFILE
        and int(metrics["per_stratum_min"]) >= PILOT_PER_STRATUM
        and bool(metrics["prefix_identity_qa_pass"])
        and float(metrics["support_rate"]) >= PILOT_MIN_SUPPORT_RATE
        and float(metrics["median_paired_valid_non_ego"]) >= PILOT_MIN_MEDIAN_PAIRED_VALID
        and int(metrics["bin_episode_counts"]["positive_low"]) >= PILOT_MIN_BIN_EPISODES
        and int(metrics["bin_episode_counts"]["positive_mid"]) >= PILOT_MIN_BIN_EPISODES
        and int(metrics["bin_episode_counts"]["positive_high"]) >= PILOT_MIN_BIN_EPISODES
    )
    if not qualifies:
        raise ValueError("v0.3 profile B does not meet v0.4 support requirements")
    return {
        "selected_profile": "B",
        "initial_vehicle_count": PROFILES["B"]["initial_vehicle_count"],
        "status": "V04_PROFILE_B_FROZEN",
        "source_plan_id": PLAN_ID,
        "qualification_rule": {
            "accepted_pairs": PILOT_ACCEPTED_PER_PROFILE,
            "per_stratum_min": PILOT_PER_STRATUM,
            "support_rate_min": PILOT_MIN_SUPPORT_RATE,
            "median_paired_valid_non_ego_min": PILOT_MIN_MEDIAN_PAIRED_VALID,
            "positive_bin_episode_min": PILOT_MIN_BIN_EPISODES,
        },
        "metrics": dict(metrics),
    }


def _weakest_bin_support(metrics: Mapping[str, Any]) -> int:
    counts = metrics["bin_episode_counts"]
    return int(min(counts["positive_low"], counts["positive_mid"], counts["positive_high"]))


def select_profile(metrics_by_profile: Mapping[str, Mapping[str, Any] | None]
                   ) -> dict[str, Any]:
    """Mechanical unique-profile choice from pilot support metrics only.

    Among qualifying profiles, maximizes the weakest positive-bin independent
    episode count; ties break to profile A. Never reads V_evt, the three Gate A
    contrasts or any model error. Returns the decision record (not yet written).
    """
    qualified = {name: m for name, m in metrics_by_profile.items()
                 if profile_qualifies(m)}
    if not qualified:
        return {
            "selected_profile": None,
            "status": "INCONCLUSIVE_INTERSECTION_RESPONSE_SUPPORT",
            "qualified": {},
            "metrics": {k: v for k, v in metrics_by_profile.items()},
        }
    ranked = sorted(qualified.items(),
                    key=lambda item: (-_weakest_bin_support(item[1]), item[0]))
    selected, _ = ranked[0]
    return {
        "selected_profile": selected,
        "initial_vehicle_count": PROFILES[selected]["initial_vehicle_count"],
        "status": "PROFILE_SELECTED",
        "qualified": {name: True for name in qualified},
        "weakest_bin_support": {name: _weakest_bin_support(m)
                                for name, m in qualified.items()},
        "metrics": {k: v for k, v in metrics_by_profile.items()},
    }


def write_profile_selection(path: str | Path, decision: Mapping[str, Any],
                            *, plan_id: str = PLAN_ID) -> dict[str, Any]:
    """Write ``profile_selection.json`` exactly once; refuse to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"profile selection already exists and is create-once: {path}")
    record = dict(decision)
    record["plan_id"] = plan_id
    body = json.dumps(record, sort_keys=True, ensure_ascii=False, default=_json_scalar)
    record["selection_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    _atomic_write_json(path, record)
    return record
