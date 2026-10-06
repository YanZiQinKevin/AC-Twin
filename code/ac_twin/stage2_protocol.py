"""Stage 2 confirmatory-holdout protocol: config validation, seed schedule,
historical seed scan, frozen Gate S* parameters, protocol lock and the
pre-collection joint-power verification.

This module is the *freeze* layer. It never trains, never fits normalization or
response bins on new data, and never reads any Stage 2 result. The only numeric
products it creates before collection are (a) the candidate seed schedule and
(b) the Gate S* baseline/probe coefficients fitted exclusively on the Stage 1
*frozen* train split and sealed into the protocol lock.

Frozen random streams derive from ``numpy.random.SeedSequence(collection_seed).
spawn(3)``:

1. ``seed_schedule_rng`` -> the ``seed_schedule_size`` candidate episode seeds;
2. ``action_rng`` -> t_star / prefix / factual action plans (used by collector);
3. ``stratum_rng`` -> counterfactual stratum tie-breaking (used by collector).

The schedule is drawn *without replacement* from ``[0, 2147483646)`` and shifted
by +1, guaranteeing internal uniqueness and the closed range ``[1, 2147483646]``
by construction.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_data import (  # noqa: E402
    compute_history_agent_known,
    compute_target_effect,
    compute_valid_mask,
    sha256_file,
    visible_count_signature,
)
from ac_twin import stage1_statistics as _s1  # noqa: E402
from ac_twin import stage1_plus_statistics as _plus  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402

# --- frozen identity ------------------------------------------------------------

PLAN_ID = "AC-TWIN-RFER-STAGE2-HOLDOUT-2026-09-07-V1"
STATUS_CONFIRM = "CONFIRMATORY_HOLDOUT"
STATUS_SMOKE = "ENGINEERING_SMOKE_ONLY"

FORMAL_N_PAIRS = 1200
SMOKE_N_PAIRS = 12
SEED_SCHEDULE_SIZE = 18000
SEED_LOW = 1
SEED_HIGH_EXCLUSIVE = 2147483647  # integers are drawn from [1, 2147483646]
COLLECTION_SEED_CONFIRM = 20270999

TRAINING_SEEDS = (11, 22, 33, 44, 55)
CONDITIONS = ("no_suffix", "uniform_suffix", "shuffled_suffix")
PRIMARY_DERANGEMENT_SEED = 20272907
ALTERNATE_DONOR_SEEDS = (20272917, 20272927, 20272937, 20272947)

FORMAL_OUTPUT_DIR = "experiments/stage2_confirmatory/merge"
SMOKE_OUTPUT_DIR = "experiments/stage2_confirmatory/smoke"

# Fields that would imply training/refitting; their presence voids a config.
_FORBIDDEN_RETRAIN_KEYS = (
    "learning_rate", "max_epochs", "early_stopping_patience", "weight_decay",
    "gradient_clip_norm", "optimizer", "training_seeds_to_fit", "fit_normalization",
    "fit_response_bins", "retrain", "fine_tune",
)

# Code files whose SHA256 are enumerated in the protocol lock (relative to code/).
CODE_FILES = (
    "run_stage2_confirmatory.py",
    "collect_stage2_holdout.py",
    "ac_twin/stage2_protocol.py",
    "ac_twin/stage2_inference.py",
    "ac_twin/stage2_statistics.py",
    "ac_twin/stage1_model.py",
    "ac_twin/stage1_data.py",
    "ac_twin/stage1_statistics.py",
    "ac_twin/stage1_plus_statistics.py",
    "ac_twin/checkpointing.py",
    "ac_twin/object_centric.py",
    "ac_twin/progress.py",
)

# Frozen Stage 1 response-bin thresholds (must match response_bins.json exactly).
FROZEN_ZERO_TOLERANCE = 0.0001
FROZEN_T1 = 0.24843344965718578
FROZEN_T2 = 1.898992796585611

# Frozen power-analysis constants.
Z_95 = 1.959963984540054
STAGE2_POWER_STATISTICS = ("c_presence", "v_zero", "c_high_low", "c_high_mid", "v_high")
# The three contrasts that carry the 0.05 practical point screen.
POINT_SCREEN_STATISTICS = ("c_presence", "c_high_low", "c_high_mid")
POWER_SEED = 20261070


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


# --- configuration validation -----------------------------------------------------


def validate_stage2_config(config: Mapping[str, Any]) -> None:
    """Validate a Stage 2 config; raise ValueError before any output is made.

    Formal (``CONFIRMATORY_HOLDOUT``) and smoke (``ENGINEERING_SMOKE_ONLY``) modes
    are mutually exclusive and bound to their own n_pairs / output directories.
    """
    required = {"experiment_name", "analysis_status", "plan_id", "scene", "env_id",
                "n_pairs", "collection_seed", "seed_schedule_size", "max_attempts",
                "horizon", "history_len", "t_star_low", "t_star_high",
                "intervention_durations", "vehicles_count", "observed_vehicles_count",
                "duration", "simulation_frequency", "policy_frequency",
                "determinism_checks", "action_weights", "factual_suffix_steps",
                "stage1_output_dir", "stage1_source_data", "normalization_npz",
                "response_bins_json", "stage1_gate_s_report", "output_dir",
                "data_output", "rejection_ledger", "collection_receipt",
                "training_seeds", "conditions", "primary_derangement_seed",
                "alternate_donor_seeds", "response_bins", "statistics", "model"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Stage 2 config missing keys: {sorted(missing)}")
    if config["plan_id"] != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if config["scene"] != "merge" or config["env_id"] != "merge-v0":
        raise ValueError("Stage 2 main confirmation is merge-v0 only")

    status = config["analysis_status"]
    if status not in (STATUS_CONFIRM, STATUS_SMOKE):
        raise ValueError(f"analysis_status must be {STATUS_CONFIRM!r} or {STATUS_SMOKE!r}")
    is_smoke = status == STATUS_SMOKE

    n_pairs = int(config["n_pairs"])
    output_dir = str(config["output_dir"]).replace("\\", "/")
    if is_smoke:
        if n_pairs != SMOKE_N_PAIRS:
            raise ValueError(f"smoke requires n_pairs == {SMOKE_N_PAIRS}")
        if output_dir != SMOKE_OUTPUT_DIR:
            raise ValueError(f"smoke output_dir must be {SMOKE_OUTPUT_DIR!r}")
    else:
        if n_pairs != FORMAL_N_PAIRS:
            raise ValueError(f"formal confirmation requires n_pairs == {FORMAL_N_PAIRS}")
        if output_dir != FORMAL_OUTPUT_DIR:
            raise ValueError(f"formal output_dir must be {FORMAL_OUTPUT_DIR!r}")
        if int(config["seed_schedule_size"]) != SEED_SCHEDULE_SIZE:
            raise ValueError(f"formal seed_schedule_size must be {SEED_SCHEDULE_SIZE}")
        if int(config["collection_seed"]) != COLLECTION_SEED_CONFIRM:
            raise ValueError(f"formal collection_seed must be {COLLECTION_SEED_CONFIRM}")
        if int(config["statistics"]["bootstrap_resamples"]) != 10000:
            raise ValueError("formal statistics.bootstrap_resamples must be 10000")
        if float(config["statistics"]["confidence_level"]) != 0.95:
            raise ValueError("formal statistics.confidence_level must be 0.95")

    # No train/validation subset may be introduced.
    if "split" in config or "sample_limits" in config:
        raise ValueError("Stage 2 forbids train/validation splits or sample_limits")

    # No retraining / refitting anywhere in model or statistics blocks.
    for block_name in ("model", "statistics"):
        block = config.get(block_name, {})
        for key in block:
            if key in _FORBIDDEN_RETRAIN_KEYS:
                raise ValueError(
                    f"config.{block_name}.{key} is forbidden: Stage 2 performs no "
                    "training or refitting")
    for key in config:
        if key in _FORBIDDEN_RETRAIN_KEYS:
            raise ValueError(f"config.{key} is forbidden: no retraining")

    if list(config["training_seeds"]) != list(TRAINING_SEEDS):
        raise ValueError(f"training_seeds must be {list(TRAINING_SEEDS)}")
    if list(config["conditions"]) != list(CONDITIONS):
        raise ValueError(f"conditions must be {list(CONDITIONS)}")
    if int(config["primary_derangement_seed"]) != PRIMARY_DERANGEMENT_SEED:
        raise ValueError(f"primary_derangement_seed must be {PRIMARY_DERANGEMENT_SEED}")
    if list(config["alternate_donor_seeds"]) != list(ALTERNATE_DONOR_SEEDS):
        raise ValueError(f"alternate_donor_seeds must be {list(ALTERNATE_DONOR_SEEDS)}")

    # Frozen response bins must match the sealed Stage 1 response_bins.json.
    bins = config["response_bins"]
    for key, expected in (("zero_tolerance", FROZEN_ZERO_TOLERANCE),
                          ("t1", FROZEN_T1), ("t2", FROZEN_T2)):
        if float(bins.get(key, float("nan"))) != float(expected):
            raise ValueError(f"response_bins.{key} must equal the frozen {expected}")
    bins_path = _resolve(config["response_bins_json"])
    if not bins_path.is_file():
        raise ValueError(f"response_bins_json missing: {bins_path}")
    sealed = json.loads(bins_path.read_text(encoding="utf-8"))
    if (float(sealed["zero_tolerance"]) != FROZEN_ZERO_TOLERANCE
            or float(sealed["t1"]) != FROZEN_T1 or float(sealed["t2"]) != FROZEN_T2):
        raise ValueError("sealed response_bins.json does not match the frozen thresholds")

    # Frozen normalization must exist and must not be refit (read-only reference).
    if not _resolve(config["normalization_npz"]).is_file():
        raise ValueError(f"normalization_npz missing: {config['normalization_npz']}")
    if not _resolve(config["stage1_source_data"]).is_file():
        raise ValueError(f"stage1_source_data missing: {config['stage1_source_data']}")
    if not _resolve(config["stage1_gate_s_report"]).is_file():
        raise ValueError(f"stage1_gate_s_report missing: {config['stage1_gate_s_report']}")

    # The 15 frozen checkpoints must exist.
    stage1_dir = _resolve(config["stage1_output_dir"])
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            if not (stage1_dir / f"{condition}_seed{seed}.pt").is_file():
                raise ValueError(f"missing frozen checkpoint: {condition}_seed{seed}.pt")

    stats = config["statistics"]
    if float(stats["practical_point_screen"]) != 0.05:
        raise ValueError("statistics.practical_point_screen must be 0.05")
    if int(stats["min_bin_episodes"]) != 100:
        raise ValueError("statistics.min_bin_episodes must be 100")
    if int(stats["gate_b_min_episodes"]) != 100:
        raise ValueError("statistics.gate_b_min_episodes must be 100")
    if int(stats["bootstrap_seed"]) != 20270908:
        raise ValueError("statistics.bootstrap_seed must be 20270908")


# --- historical seed scan ----------------------------------------------------------


def scan_historical_episode_seeds(data_root: str | Path) -> np.ndarray:
    """Union of every ``episode_seed``/``episode_seeds`` array under ``data_root``.

    Returns a sorted unique int64 array. Raises if the directory does not exist.

    Stage 2's *own* holdout products (any path containing ``stage2_``) are excluded:
    they are generated from the frozen Stage 2 schedule itself, so they are not
    independent prior data and must not (a) block the formal lock after the smoke
    collection lands in ``data/``, nor (b) cause a false overlap when Gate Q
    re-scans history at unlock time.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    collected: set[int] = set()
    for file_path in sorted(data_root.rglob("*.npz")):
        if "stage2_" in str(file_path).replace("\\", "/"):
            continue
        try:
            with np.load(file_path, allow_pickle=True) as archive:
                for key in ("episode_seed", "episode_seeds"):
                    if key in archive.files:
                        values = np.atleast_1d(np.asarray(archive[key])).ravel()
                        collected.update(int(v) for v in values)
        except Exception:
            # A non-archive or unreadable npz contributes no seeds; the schedule
            # collision check below is what protects the protocol.
            continue
    if not collected:
        raise ValueError(f"no historical episode seeds found under {data_root}")
    return np.array(sorted(collected), dtype=np.int64)


# --- seed schedule -----------------------------------------------------------------


def _spawn_streams(collection_seed: int) -> tuple[np.random.Generator, ...]:
    spawned = np.random.SeedSequence(int(collection_seed)).spawn(3)
    return tuple(np.random.default_rng(s) for s in spawned)


def make_seed_schedule(collection_seed: int, size: int = SEED_SCHEDULE_SIZE) -> np.ndarray:
    """Candidate episode seeds from the ``seed_schedule_rng`` stream.

    Drawn without replacement from ``[0, 2147483646)`` and shifted by +1, so the
    result is internally unique and lies in the closed range ``[1, 2147483646]``.
    """
    size = int(size)
    if size <= 0 or size > (SEED_HIGH_EXCLUSIVE - 1):
        raise ValueError("seed schedule size out of feasible range")
    seed_schedule_rng = _spawn_streams(collection_seed)[0]
    schedule = seed_schedule_rng.choice(
        SEED_HIGH_EXCLUSIVE - 1, size=size, replace=False) + 1
    return schedule.astype(np.int64)


def validate_seed_schedule(schedule: np.ndarray, historical: np.ndarray) -> None:
    """Raise unless the schedule is in-range, internally unique and disjoint."""
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
            "change collection_seed and re-lock")


# --- frozen Gate S* baseline + probe parameters ------------------------------------


def _initial_distance_to_ego(history: np.ndarray) -> np.ndarray:
    presence = history[..., 0] > 0
    ego_last = history[:, -1, 0, 2:4]
    episodes, _, agents, _ = history.shape
    distance = np.full((episodes, agents), np.nan)
    for e in range(episodes):
        for i in range(agents):
            present = np.nonzero(presence[e, :, i])[0]
            if len(present):
                distance[e, i] = np.linalg.norm(history[e, present[-1], i, 2:4] - ego_last[e])
    return distance


def _load_stage1_arrays(config: Mapping[str, Any]) -> dict[str, Any]:
    data = np.load(_resolve(config["stage1_source_data"]), allow_pickle=True)
    normalization = np.load(_resolve(config["normalization_npz"]))
    train_idx = np.asarray(normalization["train_indices"], dtype=np.int64)
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual = np.asarray(data["object_counterfactual_future"], dtype=np.float32)
    valid = compute_valid_mask(history, factual, counterfactual)
    r_pos, _ = _s1.response_magnitudes(counterfactual, factual)
    visible_count = visible_count_signature(compute_history_agent_known(history))
    return {
        "data": data,
        "train_idx": train_idx,
        "history": history,
        "factual": factual,
        "counterfactual": counterfactual,
        "valid_mask": valid,
        "r_pos": r_pos,
        "visible_count": visible_count,
        "factual_actions": np.asarray(data["factual_actions"], dtype=np.int64),
        "counterfactual_actions": np.asarray(data["counterfactual_actions"], dtype=np.int64),
        "n_future": int(factual.shape[1]),
        "n_agents": int(factual.shape[2]),
    }


def _baseline_design(arrays: Mapping[str, Any], indices: np.ndarray,
                     use_distance: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix, response and row mask for the frozen S2* baselines.

    Columns: [1, |factual-counterfactual action|, horizon_index] and, when
    ``use_distance``, + initial_distance. Only valid non-ego agent-times are used.
    """
    n = len(indices)
    T = arrays["n_future"]
    N = arrays["n_agents"]
    history_full = arrays["history"]
    distance = _initial_distance_to_ego(history_full[indices])  # n,N
    horizon = np.broadcast_to(np.arange(T)[None, :, None], (n, T, N)).astype(np.float64)
    dist_b = np.broadcast_to(distance[:, None, :], (n, T, N)).astype(np.float64)
    fa = arrays["factual_actions"][indices]
    ca = arrays["counterfactual_actions"][indices]
    action_diff = np.broadcast_to(np.abs(fa - ca)[:, :, None].astype(np.float64), (n, T, N))
    cols = [np.ones((n, T, N)), action_diff, horizon]
    if use_distance:
        cols.append(dist_b)
    X = np.column_stack([c.reshape(-1) for c in cols])
    mask = arrays["valid_mask"][indices].copy()
    mask[:, :, 0] = False  # exclude ego
    y = arrays["r_pos"][indices]
    return X, y.reshape(-1), mask.reshape(-1)


def fit_frozen_baselines(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Fit the two frozen S2* MAE baselines on the Stage 1 train split only.

    Ridge (lambda=1e-3) on Stage 1 train; coefficients are sealed in the lock and
    applied unchanged to the Stage 2 holdout. Never refit on new data.
    """
    arrays = _load_stage1_arrays(config)
    train_idx = arrays["train_idx"]
    out: dict[str, np.ndarray] = {}
    for name, use_distance in (("beta_action_horizon", False),
                               ("beta_action_horizon_distance", True)):
        X, y, mask = _baseline_design(arrays, train_idx, use_distance)
        beta = _s1.ridge_fit(X[mask], y[mask], lam=1e-3)
        out[name] = np.asarray(beta, dtype=np.float64)
    return out


def motion_summary(factual: np.ndarray) -> np.ndarray:
    """Per-agent 6-D motion summary used as the frozen probe target.

    [displacement dx,dy; mean velocity vx,vy; final velocity vx,vy]; E,N,6.
    """
    factual = np.asarray(factual, dtype=np.float64)
    presence = factual[..., 0] > 0
    count = presence.sum(axis=1)
    disp = factual[:, -1, :, 2:4] - factual[:, 0, :, 2:4]
    vel_sum = (factual[..., 4:6] * presence[..., None]).sum(axis=1)
    mean_vel = np.where(count[..., None] > 0, vel_sum / np.maximum(count, 1)[..., None], 0.0)
    final_vel = factual[:, -1, :, 4:6]
    return np.concatenate([disp, mean_vel, final_vel], axis=-1)


def compute_probe_coefficients(train_summaries: np.ndarray, train_active: np.ndarray,
                               motion_targets: np.ndarray) -> np.ndarray:
    """Ridge coefficients (intercept + hidden) -> 6-D motion summary.

    ``train_summaries`` is the seed-mean factual summary ``E×N×hidden`` on Stage 1
    train; ``train_active`` is ``E×N``; ``motion_targets`` is ``E×N×6``. Returns a
    ``(hidden+1)×6`` coefficient matrix. Pure NumPy; no model is loaded here.
    """
    return _fit_probe(train_summaries, train_active, motion_targets)


def _fit_probe(train_summaries: np.ndarray, train_active: np.ndarray,
               motion_targets: np.ndarray) -> np.ndarray:
    train_summaries = np.asarray(train_summaries, dtype=np.float64)
    train_active = np.asarray(train_active, dtype=bool)
    motion_targets = np.asarray(motion_targets, dtype=np.float64)
    X = train_summaries[train_active]
    Y = motion_targets[train_active]
    if X.shape[0] < X.shape[1]:
        raise ValueError("insufficient active agents to fit the frozen probe")
    design = np.column_stack([np.ones(len(X)), X])
    penalty = 1e-3 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ Y)
    return np.asarray(beta, dtype=np.float64)


# Frozen permutation seed for the probe's shuffled-label baseline (mirrors Stage 1).
PROBE_SHUFFLE_SEED = 20260907 + 31


def compute_probe_parameters(train_summaries: np.ndarray, train_active: np.ndarray,
                             motion_targets: np.ndarray) -> dict[str, np.ndarray]:
    """Frozen S3* representation-probe parameters fitted on Stage 1 train only.

    Returns the main ridge coefficients, the shuffled-label baseline coefficients
    (labels permuted with a frozen seed) and the scene-mean baseline vector. All
    three are sealed in the protocol lock and applied unchanged to the holdout.
    """
    train_active = np.asarray(train_active, dtype=bool)
    motion_targets = np.asarray(motion_targets, dtype=np.float64)
    probe = _fit_probe(train_summaries, train_active, motion_targets)
    rng = np.random.default_rng(PROBE_SHUFFLE_SEED)
    Y = motion_targets[train_active]
    permuted = motion_targets.copy()
    permuted[train_active] = Y[rng.permutation(len(Y))]
    probe_shuffled = _fit_probe(train_summaries, train_active, permuted)
    scene_mean = Y.mean(axis=0)
    return {
        "probe_coefficients": probe,
        "probe_coefficients_shuffled": probe_shuffled,
        "probe_scene_mean": np.asarray(scene_mean, dtype=np.float64),
    }


def _default_params_provider(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compute the real frozen Gate S* parameters (baselines + probe).

    Baselines are pure NumPy; the probe requires running the five frozen M1
    checkpoints' factual encoder over the Stage 1 train split (lazy torch import).
    """
    params = fit_frozen_baselines(config)
    arrays = _load_stage1_arrays(config)
    from ac_twin.stage2_inference import compute_train_factual_summaries
    summaries, active = compute_train_factual_summaries(config)
    targets = motion_summary(arrays["factual"][arrays["train_idx"]])
    params.update(compute_probe_parameters(summaries, active, targets))
    return params


# --- protocol lock ------------------------------------------------------------------


def _environment_versions() -> dict[str, str]:
    """Record runtime versions without importing heavy/optional packages.

    Uses ``importlib.metadata`` so locking never imports highway_env (and its
    pygame dependency), which can fail under pytest's ``-W error``.
    """
    from importlib.metadata import version as _pkg_version
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": _pkg_version("numpy"),
        "torch_version": _pkg_version("torch"),
        "highway_env_version": _pkg_version("highway_env"),
        "gymnasium_version": _pkg_version("gymnasium"),
    }


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in CODE_FILES:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


def _checkpoint_hashes(stage1_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            name = f"{condition}_seed{seed}.pt"
            hashes[name] = sha256_file(stage1_dir / name)
    return hashes


def write_protocol_lock(config_path: str | Path, output_dir: str | Path,
                        *, params_provider: Callable[[Mapping[str, Any]],
                                                   dict[str, np.ndarray]] | None = None,
                        ) -> dict[str, Any]:
    """Create the protocol lock exactly once; refuse to overwrite an existing lock.

    Validates the config, scans historical seeds, builds + validates the seed
    schedule, computes and seals the frozen Gate S* parameters, and records the
    SHA256 of every locked input. Writes ``protocol_lock.json``,
    ``seed_schedule.npy`` and ``frozen_gate_s_star_params.npz`` atomically.
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_stage2_config(config)
    provider = params_provider or _default_params_provider

    lock_path = output_dir / "protocol_lock.json"
    schedule_path = output_dir / "seed_schedule.npy"
    params_path = output_dir / "frozen_gate_s_star_params.npz"
    if lock_path.exists():
        raise FileExistsError(
            f"protocol lock already exists and is create-once: {lock_path}")

    historical = scan_historical_episode_seeds(ROOT / "data")
    schedule = make_seed_schedule(int(config["collection_seed"]),
                                  size=int(config["seed_schedule_size"]))
    validate_seed_schedule(schedule, historical)

    params = provider(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Seal schedule and params first so their hashes enter the lock. Write via a
    # file handle so np.save/npz never append an extra extension to the tmp name.
    tmp_schedule = schedule_path.with_name(schedule_path.name + ".tmp")
    with open(tmp_schedule, "wb") as handle:
        np.save(handle, schedule)
    Path(tmp_schedule).replace(schedule_path)
    tmp_params = params_path.with_name(params_path.name + ".tmp")
    with open(tmp_params, "wb") as handle:
        np.savez_compressed(handle, **{k: np.asarray(v) for k, v in params.items()})
    Path(tmp_params).replace(params_path)

    stage1_dir = _resolve(config["stage1_output_dir"])
    lock: dict[str, Any] = {
        "plan_id": PLAN_ID,
        "experiment_name": config["experiment_name"],
        "analysis_status": config["analysis_status"],
        "scene": config["scene"],
        "collection_seed": int(config["collection_seed"]),
        "seed_schedule_size": int(config["seed_schedule_size"]),
        "n_historical_episode_seeds": int(len(historical)),
        "seed_schedule_zero_overlap": True,
        "donor_seeds": {"primary": PRIMARY_DERANGEMENT_SEED,
                        "alternates": list(ALTERNATE_DONOR_SEEDS)},
        "paths": {
            "config": str(config_path),
            "stage1_output_dir": str(stage1_dir),
            "normalization_npz": str(_resolve(config["normalization_npz"])),
            "response_bins_json": str(_resolve(config["response_bins_json"])),
            "stage1_gate_s_report": str(_resolve(config["stage1_gate_s_report"])),
            "stage1_source_data": str(_resolve(config["stage1_source_data"])),
            "stage1_config": str(_resolve(config["stage1_config"])),
            "seed_schedule": str(schedule_path),
            "frozen_params": str(params_path),
        },
        "code_sha256": _code_hashes(),
        "checkpoint_sha256": _checkpoint_hashes(stage1_dir),
        "config_sha256": sha256_file(config_path),
        "normalization_npz_sha256": sha256_file(_resolve(config["normalization_npz"])),
        "response_bins_sha256": sha256_file(_resolve(config["response_bins_json"])),
        "stage1_gate_s_report_sha256": sha256_file(_resolve(config["stage1_gate_s_report"])),
        "stage1_source_data_sha256": sha256_file(_resolve(config["stage1_source_data"])),
        "stage1_config_sha256": sha256_file(_resolve(config["stage1_config"])),
        "seed_schedule_sha256": sha256_file(schedule_path),
        "frozen_params_sha256": sha256_file(params_path),
        "frozen_params_keys": sorted(params.keys()),
        "environment": _environment_versions(),
        "config_signature": canonical_fingerprint(config),
    }
    # The lock hash covers every field except the hash itself.
    body = json.dumps(lock, sort_keys=True, ensure_ascii=False, default=_json_scalar)
    lock["lock_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    tmp_lock = lock_path.with_name(lock_path.name + ".tmp")
    tmp_lock.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(tmp_lock).replace(lock_path)
    return lock


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def verify_protocol_lock(output_dir: str | Path) -> dict[str, Any]:
    """Read-only verification that every locked artifact still matches its hash.

    Raises on the first mismatch; never writes or modifies anything. Returns the
    lock mapping on success.
    """
    output_dir = Path(output_dir)
    lock_path = output_dir / "protocol_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"protocol lock missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    stored = lock.get("lock_sha256")
    body = json.dumps({k: v for k, v in lock.items() if k != "lock_sha256"},
                      sort_keys=True, ensure_ascii=False, default=_json_scalar)
    if hashlib.sha256(body.encode("utf-8")).hexdigest() != stored:
        raise ValueError("protocol lock self-hash mismatch (lock file was modified)")

    paths = lock["paths"]
    checks: dict[str, tuple[Path, str]] = {}
    for rel, expected in lock["code_sha256"].items():
        checks[f"code:{rel}"] = (ROOT / "code" / rel, expected)
    for name, expected in lock["checkpoint_sha256"].items():
        checks[f"checkpoint:{name}"] = (Path(paths["stage1_output_dir"]) / name, expected)
    for label, path_key, hash_key in (
            ("config", "config", "config_sha256"),
            ("normalization_npz", "normalization_npz", "normalization_npz_sha256"),
            ("response_bins", "response_bins_json", "response_bins_sha256"),
            ("stage1_gate_s_report", "stage1_gate_s_report", "stage1_gate_s_report_sha256"),
            ("stage1_source_data", "stage1_source_data", "stage1_source_data_sha256"),
            ("stage1_config", "stage1_config", "stage1_config_sha256"),
            ("seed_schedule", "seed_schedule", "seed_schedule_sha256"),
            ("frozen_params", "frozen_params", "frozen_params_sha256")):
        checks[label] = (Path(paths[path_key]), lock[hash_key])

    for label, (file_path, expected) in checks.items():
        if not file_path.is_file():
            raise ValueError(f"locked artifact missing: {label} -> {file_path}")
        actual = sha256_file(file_path)
        if actual != expected:
            raise ValueError(
                f"locked artifact hash drift: {label}\n  expected {expected}\n  actual   {actual}")
    return lock


# --- confirmatory joint power (Task 1) ----------------------------------------------


def build_stage2_power_statistics(config: Mapping[str, Any]) -> dict[str, Callable]:
    """Build the five frozen, multiplicity-aware episode statistics on the Stage 1
    merge development-test split (pool = 300 episodes)."""
    arrays = _load_stage1_arrays(config)
    normalization = np.load(_resolve(config["normalization_npz"]))
    test_idx = np.asarray(normalization["test_indices"], dtype=np.int64)
    stage1_dir = _resolve(config["stage1_output_dir"])
    target_mean = normalization["target_mean"].astype(np.float64)
    target_std = normalization["target_std"].astype(np.float64)

    target = compute_target_effect(arrays["factual"], arrays["counterfactual"])[test_idx]
    valid = arrays["valid_mask"][test_idx].copy()
    valid[:, :, 0] = False  # non-ego
    r_pos = arrays["r_pos"][test_idx]
    bins = json.loads(_resolve(config["response_bins_json"]).read_text(encoding="utf-8"))
    thresholds = {"zero_tolerance": bins["zero_tolerance"], "t1": bins["t1"], "t2": bins["t2"]}
    bin_id = _s1.assign_response_bin(r_pos, thresholds)

    def denorm(pred):
        return np.asarray(pred, dtype=np.float64) * target_std + target_mean

    def seed_mean_errors(condition):
        per_seed = np.stack([
            _s1.position_errors(
                denorm(np.load(stage1_dir / f"{condition}_seed{seed}_test_prediction.npy",
                               allow_pickle=False)), target)
            for seed in TRAINING_SEEDS
        ])
        return _s1.seed_mean(per_seed)

    l_suffix = seed_mean_errors("uniform_suffix")
    l_shuffled = seed_mean_errors("shuffled_suffix")

    bin0 = valid & (bin_id == 0)
    bin1 = valid & (bin_id == 1)
    bin2 = valid & (bin_id == 2)
    bin3 = valid & (bin_id == 3)
    positive = valid & (bin_id >= 1)

    return {
        "c_presence": _plus.make_group_contrast_statistic(l_suffix, l_shuffled, bin0, positive),
        "v_zero": _plus.make_group_v_evt_statistic(l_suffix, l_shuffled, bin0),
        "c_high_low": _plus.make_group_contrast_statistic(l_suffix, l_shuffled, bin3, bin1),
        "c_high_mid": _plus.make_group_contrast_statistic(l_suffix, l_shuffled, bin3, bin2),
        "v_high": _plus.make_group_v_evt_statistic(l_suffix, l_shuffled, bin3),
    }


def estimate_stage2_joint_power(*, statistics: Mapping[str, Callable], n: int,
                                monte_carlo_replicates: int, seed: int,
                                confidence_level: float,
                                point_screen: float) -> dict[str, Any]:
    """Monte-Carlo joint power for the five Stage 2 numeric Gate A conditions.

    For each replicate, ``n`` episodes are drawn with replacement from the
    development pool and all five statistics are evaluated on the same draw (the
    joint pass is a per-draw AND). ``SE`` is the empirical per-statistic standard
    deviation at ``n``. The 75%/50% scenarios use ``mult*full + (draw - full)``,
    shrinking the point toward zero while preserving the sampling deviation; the
    0.05 point screen is applied to the shrunken draw for the three contrasts.
    """
    if float(confidence_level) != 0.95:
        raise ValueError("estimate_stage2_joint_power is frozen to confidence_level = 0.95")
    names = list(STAGE2_POWER_STATISTICS)
    missing = [k for k in names if k not in statistics]
    if missing:
        raise ValueError(f"statistics missing Stage 2 power conditions: {missing}")
    pool_sizes = {int(getattr(fn, "pool_size", 0)) for fn in statistics.values()}
    if len(pool_sizes) != 1:
        raise ValueError("all statistics must share a single pool_size")
    pool = pool_sizes.pop()
    if pool < 2:
        raise ValueError("pool_size must be >= 2")

    rng = np.random.default_rng(int(seed))
    R = int(monte_carlo_replicates)
    n = int(n)
    full_index = np.arange(pool)
    full_point = {name: float(statistics[name](full_index)) for name in names}

    draws = {name: np.empty(R, dtype=np.float64) for name in names}
    for r in range(R):
        indices = rng.integers(0, pool, size=n)
        for name in names:
            draws[name][r] = statistics[name](indices)
    se = {name: float(np.std(draws[name], ddof=1)) for name in names}

    multipliers = (1.0, 0.75, 0.5)
    scenario_names = {1.0: "empirical", 0.75: "75pct", 0.5: "50pct"}
    scenarios: dict[str, Any] = {}
    for mult in multipliers:
        shrunken = {name: mult * full_point[name] + (draws[name] - full_point[name])
                    for name in names}
        single: dict[str, float] = {}
        pass_masks: dict[str, np.ndarray] = {}
        for name in names:
            sign_pass = (shrunken[name] - Z_95 * se[name]) > 0
            if name in POINT_SCREEN_STATISTICS:
                sign_pass &= (shrunken[name] >= float(point_screen))
            pass_masks[name] = sign_pass
            single[name] = float(np.mean(sign_pass))
        joint_mask = np.ones(R, dtype=bool)
        for name in names:
            joint_mask &= pass_masks[name]
        scenarios[scenario_names[mult]] = {"single": single,
                                           "joint": float(np.mean(joint_mask))}

    return {
        "n": n,
        "monte_carlo_replicates": R,
        "confidence_level": float(confidence_level),
        "z": Z_95,
        "point_screen": float(point_screen),
        "pool_episodes": pool,
        "full_point_estimates": full_point,
        "se": se,
        "statistic_order": names,
        "scenarios": scenarios,
    }
