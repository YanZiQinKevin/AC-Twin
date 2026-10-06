"""Stage 4C protocol: group seed schedules, decision-task constants, freeze layer.

Freeze layer for the Stage 4C multi-action decision-reliability confirmation
(plan ``AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1``). It never trains,
never fits on new data, never re-selects a baseline/threshold/lambda outside the
selection role and never reads a regret/label before the matching one-shot lock is
burned. Its only pre-collection numeric products are the two group seed schedules
(development entropy ``2026091801`` size 5,000; confirm entropy ``2026091901`` size
15,000) plus the historical seed inventory used to prove zero overlap.

Statistical unit (plan Global Constraints): the multi-action *decision group*, not an
action branch, vehicle or time point. Each accepted group shares one history, one
factual branch and one ``t_star``, and forks into the ten frozen ``ACTION_PLANS``
counterfactual branches. All resampling is whole-group.

Stage 4C binds Stage 4A/4B frozen artifacts READ-ONLY (plan section 3). Identity is
by content SHA256, never by drive letter. The Stage 4B confirm set itself is a
permanently forbidden input.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin import stage3_data as s3d
from ac_twin.stage1_data import sha256_file
from ac_twin.stage2_protocol import make_seed_schedule, validate_seed_schedule
from ac_twin.stage3_data import compute_history_agent_known
from ac_twin.stage3r_protocol import write_once_json
from ac_twin.stage4_protocol import (
    CF_ACTIONS,
    CF_DURATIONS,
    EXPERT_ROUTER_SEEDS,
    PRIMARY_CONSTANT_W,
    RESPONSE_PREDICTOR_SEEDS,
    e2e_checkpoint_path,
    expert_checkpoint_path,
)
from ac_twin.stage4c_objective import ACTION_PLANS, EGO_SLOT

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = "AC-TWIN-RFER-STAGE4C-MULTIACTION-REGRET-2026-09-15-V0.1"
PURPOSE_DEV = "stage4c_merge_multiaction_dev"
PURPOSE_CONFIRM = "stage4c_merge_multiaction_confirm"

# --- environment / tensor contract (plan section 4.2) ----------------------------
ENV_ID = "merge-v0"
HISTORY_LEN = 10
HORIZON = 20
T_STAR_LOW = 10
T_STAR_HIGH = 15
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5
VEHICLES_COUNT = 25
OBSERVED_VEHICLES_COUNT = 5
ENV_DURATION = 100
OBJECT_CAPACITY = 5
N_OBJECT_FEATURES = 8
N_ACTIONS = len(ACTION_PLANS)

# --- development / confirm sizing (plan section 7) --------------------------------
DEV_SELECTION_GROUPS = 500
DEV_SCREEN_GROUPS = 300
DEV_TOTAL_GROUPS = DEV_SELECTION_GROUPS + DEV_SCREEN_GROUPS  # 800
CANDIDATE_NS = (600, 900, 1200, 1800, 2400)

# --- frozen group seed schedules (plan section 7) ---------------------------------
DEV_SEED_ENTROPY = 2026091801
DEV_SCHEDULE_SIZE = 5000
CONFIRM_SEED_ENTROPY = 2026091901
CONFIRM_SCHEDULE_SIZE = 15000
SEED_LOW = 1
SEED_HIGH_INCLUSIVE = 2147483646  # 2**31 - 2

# --- frozen statistical entropies (plan sections 9, 10, 11) -----------------------
BOOTSTRAP_ENTROPY_SCREEN = 2026091802
POWER_MONTE_CARLO_ENTROPY = 2026092001
BOOTSTRAP_ENTROPY_CONFIRM = 2026092002
BOOTSTRAP_RESAMPLES = 10000
POWER_BOOTSTRAP_RESAMPLES = 10000
POWER_MONTE_CARLO = 10000
CONFIDENCE_SCREEN = 0.80
CONFIDENCE_CONFIRM = 0.95
POWER_TARGET = 0.80
POWER_SHRINK_MAIN = 0.75
POWER_SHRINK_SCENARIOS = (1.0, 0.75, 0.5)

# --- frozen risk-aggregation / reranking (plan section 6) -------------------------
LAMBDA_CANDIDATES = (0.00, 0.25, 0.50, 1.00)
COVERAGE_MAIN = 0.80
HIGH_REGRET_QUANTILE = 0.80

# --- Gate TASK thresholds (plan section 9.2) --------------------------------------
GATE_TASK_MIN_NONZERO_REGRET_FRAC = 0.20
GATE_TASK_REGRET_EPS = 0.01
GATE_TASK_MIN_DISTINCT_ORACLE = 3
GATE_TASK_MIN_ORACLE_FRAC = 0.05

# --- Gate RISK-DECISION thresholds (plan sections 9.3 / 11.3) ---------------------
GATE_I_AURC_REGRET_MIN = 0.05
GATE_I_REGRET80_MIN = 0.03
GATE_MIN_POSITIVE_PAIRS = 4
GATE_STRATA_I_AURC_FLOOR = -0.10
GATE_STRATA_MIN_COUNT = 50
GATE_TRIM_FRACTIONS = (0.01, 0.05)

# --- frozen Stage 4A carry-over ----------------------------------------------------
TOP_ERROR_THRESHOLD = 0.41601440122246675  # inherited; NOT recomputed (plan s3)

# --- mechanical terminal states (plan sections 9, 10, 11) -------------------------
STATUS_DEV_READY = "STAGE4C_DEV_READY_FOR_CONFIRMATION"
STATUS_DEV_NOT_READY = "STAGE4C_DEV_NOT_READY_FOR_CONFIRMATION"
STATUS_TASK_DEGENERATE = "STAGE4C_MULTIACTION_TASK_DEGENERATE"
STATUS_MODEL_INSUFFICIENT = "INCONCLUSIVE_STAGE4C_DECISION_MODEL_INSUFFICIENT"
STATUS_UNDERPOWERED = "STAGE4C_CONFIRMATION_UNDERPOWERED"
STATUS_INVALID = "STAGE4C_INVALID_ANALYSIS"
STATUS_CONFIRMED = "STAGE4C_DECISION_RELIABILITY_CONFIRMED"
STATUS_NOT_CONFIRMED = "STAGE4C_DECISION_RELIABILITY_NOT_CONFIRMED"
STATUS_RERANK_SUPPORTED = "RERANKING_SUPPORTED"
STATUS_RERANK_NOT_SUPPORTED = "RERANKING_NOT_SUPPORTED"
STATUS_RERANK_DEGENERATE = "RERANKING_SCORE_DEGENERATE"
STATUS_COLLECTION_INCOMPLETE = "STAGE4C_COLLECTION_INCOMPLETE"

# New-code files whose SHA256 are enumerated in the protocol locks (relative to code/).
CODE_FILES = (
    "run_stage4c.py",
    "collect_stage4c_merge_multiaction.py",
    "ac_twin/stage4c_protocol.py",
    "ac_twin/stage4c_objective.py",
    "ac_twin/stage4c_inference.py",
    "ac_twin/stage4c_statistics.py",
    "ac_twin/stage4c_power.py",
)

# Frozen Stage 4A/4B source artifacts (plan section 3) with their known SHA256.
STAGE4_FROZEN = {
    "stage4b_confirmatory_report": (
        "experiments/stage4_reliability/stage4b/reports/stage4b_confirmatory_report.json",
        "f6ab4012ba31ed58594fd596bc0c4a1f5bef2cfd0b326d585deaf18976e9bba2",
    ),
    "stage4b_decision": (
        "experiments/stage4_reliability/stage4b/reports/STAGE4B_CONFIRMATORY_DECISION.md",
        "2cedbf99be3d1d5463cb8896767a92f85df93696eb9d5033aeac140a850c5f2b",
    ),
    "stage4b_protocol_lock": (
        "experiments/stage4_reliability/stage4b/protocol/protocol_lock.json",
        "8cdb6ff85514ec16e96eae5999f35458829efa9a6280055709e6f8e0e81aa517",
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
    "scaler": (
        "experiments/stage3_routing/stage3r/scaler.json",
        "9f546efe3bf6e2a907e7115924f7d1dec55491b11b6a1e09c469b104c5db9181",
    ),
}

# Forbidden source path tokens (plan section 7.3). Stage 4C additionally forbids
# every Stage 4A/4B development/confirm product and the Stage 2B intersection data.
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
    "stage4_merge_risk_confirm",
    "stage2b",
)


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _normalised(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


# --- forbidden-data guards ---------------------------------------------------------


def assert_allowed_stage4c_data(path: str | Path, *, file_sha256: str | None = None) -> None:
    """Raise ``ValueError`` if ``path`` is a permanently forbidden Stage 4C input.

    The only legal Stage 4C data roles are the multi-action dev/confirm roles;
    anything under a sealed/consumed Stage 2/3/4 location (including the Stage 4B
    confirm set) is refused by path segment and, where a known fingerprint exists,
    by content SHA256.
    """
    s3d.assert_not_sealed_stage2(path, file_sha256=file_sha256)
    logical = _normalised(path)
    for token in _FORBIDDEN_DATA_TOKENS:
        for segment in logical.split("/"):
            if segment.startswith(token):
                raise ValueError(
                    f"forbidden Stage 4C data source (segment {segment!r}): {path}")
    if PURPOSE_DEV not in logical and PURPOSE_CONFIRM not in logical:
        raise ValueError(
            f"Stage 4C data must live under a multi-action role dir: {path}")


# --- group seed schedules -----------------------------------------------------------


def make_group_seed_schedule(entropy: int, size: int) -> np.ndarray:
    """Unique positive int32 group seeds for one frozen (entropy, size) contract.

    Only the two frozen Stage 4C contracts are legal: development
    (``2026091801`` / 5,000) and confirm (``2026091901`` / 15,000). Reuses the frozen
    Stage 2 generator (``SeedSequence(entropy).spawn(3)`` first stream, draw without
    replacement, +1 shift) so the schedule is internally unique and in-range.
    """
    entropy = int(entropy)
    size = int(size)
    legal = {(DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE),
             (CONFIRM_SEED_ENTROPY, CONFIRM_SCHEDULE_SIZE)}
    if (entropy, size) not in legal:
        raise ValueError(
            f"Stage 4C group seed (entropy, size) must be one of {sorted(legal)}; "
            f"got ({entropy}, {size})")
    schedule = make_seed_schedule(entropy, size=size)
    if schedule.min() < SEED_LOW or schedule.max() > SEED_HIGH_INCLUSIVE:
        raise ValueError("Stage 4C group seed schedule out of range")
    return schedule.astype(np.int64)


def scan_historical_group_seeds(workspace_roots: Sequence[Path]) -> np.ndarray:
    """Union of every historical *accepted* episode seed across NPZ / receipts / manifests.

    Same scope as the canonical prior-stage scanners: only *realized* episode seeds
    (raw candidate ``*seed_schedule*.npy`` pools excluded). Stage 4C's own dev and
    confirm products are excluded so a resume does not collide with itself; every
    prior-stage accepted seed IS scanned so Stage 4C stays disjoint from them.
    """
    collected: set[int] = set()
    for root in workspace_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob("*")):
            normalized = _normalised(file_path)
            if PURPOSE_DEV in normalized or PURPOSE_CONFIRM in normalized:
                continue
            if "stage4_reliability/stage4c" in normalized:
                continue
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".npz":
                    with np.load(file_path, allow_pickle=True) as archive:
                        for key in ("episode_seed", "episode_seeds", "group_seed", "group_seeds"):
                            if key in archive.files:
                                values = np.atleast_1d(np.asarray(archive[key])).ravel()
                                collected.update(int(v) for v in values)
                elif suffix == ".json":
                    collected.update(_seeds_in_json(file_path))
            except Exception:
                continue
    if not collected:
        raise ValueError("no historical episode seeds found in the workspace")
    return np.array(sorted(collected), dtype=np.int64)


def _seeds_in_json(file_path: Path) -> set[int]:
    """Recursively collect ints under keys named like episode/group seed in a JSON file."""
    import json
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
            if any(hint in key_hint for hint in (
                    "episode_seed", "episode_seeds", "group_seed", "group_seeds",
                    "seed_schedule")):
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


# --- inference mask ------------------------------------------------------------------


def materialize_action_plan(
    factual_plan: np.ndarray, action: int, duration: int
) -> np.ndarray:
    """Counterfactual action plan: copy the factual plan, override the first steps.

    ``plan[:duration] = action``; the remaining steps keep the factual action. This
    single canonical implementation is shared by the collector and the tests so the
    ``(IDLE, 1)`` vs ``(IDLE, 3)`` distinction cannot drift (plan section 4.1).
    """
    plan = np.asarray(factual_plan, dtype=np.int64).copy()
    horizon = plan.shape[0]
    d = min(int(duration), horizon)
    if d > 0:
        plan[:d] = int(action)
    return plan


def build_inference_mask(history: np.ndarray, factual: np.ndarray) -> np.ndarray:
    """Boolean ``(G, T, N)`` legal-cell mask built WITHOUT any counterfactual read.

    ``legal = history-known AND factual-present AND non-ego`` (plan section 4.3.4).
    The ego is object slot 0 by the object-centric contract. The counterfactual
    branch is never an argument here — truth-side presence must not shape the mask.
    For an accepted group this mask must equal the post-hoc evaluation mask.
    """
    known = compute_history_agent_known(history)  # (G, N)
    factual_present = np.asarray(factual)[..., 0] > 0  # (G, T, N)
    valid = known[:, None, :] & factual_present
    valid = valid.copy()
    valid[..., EGO_SLOT] = False
    return valid


# --- config validation -----------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 4C config; raise before any artifact is produced."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("scene") != "merge" or checked.get("env_id") != ENV_ID:
        raise ValueError("Stage 4C is merge-v0 only")
    if [tuple(p) for p in checked.get("action_plans", [])] != [tuple(p) for p in ACTION_PLANS]:
        raise ValueError(f"action_plans must be {list(ACTION_PLANS)}")
    if list(checked.get("cf_actions", [])) != list(CF_ACTIONS):
        raise ValueError(f"cf_actions must be {list(CF_ACTIONS)}")
    if list(checked.get("intervention_durations", [])) != list(CF_DURATIONS):
        raise ValueError(f"intervention_durations must be {list(CF_DURATIONS)}")
    if int(checked.get("history_len", -1)) != HISTORY_LEN:
        raise ValueError(f"history_len must be {HISTORY_LEN}")
    if int(checked.get("horizon", -1)) != HORIZON:
        raise ValueError(f"horizon must be {HORIZON}")
    if int(checked.get("t_star_low", -1)) != T_STAR_LOW:
        raise ValueError(f"t_star_low must be {T_STAR_LOW}")
    if int(checked.get("t_star_high", -1)) != T_STAR_HIGH:
        raise ValueError(f"t_star_high must be {T_STAR_HIGH}")
    if int(checked.get("simulation_frequency", -1)) != SIMULATION_FREQUENCY:
        raise ValueError("simulation_frequency drifted")
    if int(checked.get("policy_frequency", -1)) != POLICY_FREQUENCY:
        raise ValueError("policy_frequency drifted")
    if int(checked.get("vehicles_count", -1)) != VEHICLES_COUNT:
        raise ValueError("vehicles_count drifted")
    if int(checked.get("duration", -1)) != ENV_DURATION:
        raise ValueError("duration drifted")
    if int(checked.get("observed_vehicles_count", -1)) != OBSERVED_VEHICLES_COUNT:
        raise ValueError("observed_vehicles_count drifted")
    if int(checked.get("object_capacity", -1)) != OBJECT_CAPACITY:
        raise ValueError("object_capacity drifted")
    if int(checked.get("dev_selection_groups", -1)) != DEV_SELECTION_GROUPS:
        raise ValueError("dev_selection_groups drifted")
    if int(checked.get("dev_screen_groups", -1)) != DEV_SCREEN_GROUPS:
        raise ValueError("dev_screen_groups drifted")
    if [int(n) for n in checked.get("candidate_ns", [])] != list(CANDIDATE_NS):
        raise ValueError(f"candidate_ns must be {list(CANDIDATE_NS)}")
    if int(checked.get("dev_seed_entropy", -1)) != DEV_SEED_ENTROPY:
        raise ValueError("dev_seed_entropy drifted")
    if int(checked.get("dev_schedule_size", -1)) != DEV_SCHEDULE_SIZE:
        raise ValueError("dev_schedule_size drifted")
    if int(checked.get("confirm_seed_entropy", -1)) != CONFIRM_SEED_ENTROPY:
        raise ValueError("confirm_seed_entropy drifted")
    if int(checked.get("confirm_schedule_size", -1)) != CONFIRM_SCHEDULE_SIZE:
        raise ValueError("confirm_schedule_size drifted")
    if int(checked.get("bootstrap_entropy_screen", -1)) != BOOTSTRAP_ENTROPY_SCREEN:
        raise ValueError("bootstrap_entropy_screen drifted")
    if int(checked.get("bootstrap_entropy_confirm", -1)) != BOOTSTRAP_ENTROPY_CONFIRM:
        raise ValueError("bootstrap_entropy_confirm drifted")
    if int(checked.get("power_monte_carlo_entropy", -1)) != POWER_MONTE_CARLO_ENTROPY:
        raise ValueError("power_monte_carlo_entropy drifted")
    if float(checked.get("confidence_screen", -1.0)) != CONFIDENCE_SCREEN:
        raise ValueError("confidence_screen drifted")
    if float(checked.get("confidence_confirm", -1.0)) != CONFIDENCE_CONFIRM:
        raise ValueError("confidence_confirm drifted")
    if float(checked.get("power_target", -1.0)) != POWER_TARGET:
        raise ValueError("power_target drifted")
    if float(checked.get("power_shrink_main", -1.0)) != POWER_SHRINK_MAIN:
        raise ValueError("power_shrink_main drifted")
    if float(checked.get("top_error_threshold", -1.0)) != TOP_ERROR_THRESHOLD:
        raise ValueError(f"top_error_threshold is frozen to {TOP_ERROR_THRESHOLD}")
    if float(checked.get("primary_constant_w", -1.0)) != PRIMARY_CONSTANT_W:
        raise ValueError(f"primary_constant_w must be {PRIMARY_CONSTANT_W}")
    if list(checked.get("response_predictor_seeds", [])) != list(RESPONSE_PREDICTOR_SEEDS):
        raise ValueError("response_predictor_seeds drifted from the frozen five")
    if list(checked.get("expert_router_seeds", [])) != list(EXPERT_ROUTER_SEEDS):
        raise ValueError("expert_router_seeds drifted from the frozen five pairs")
    if [float(x) for x in checked.get("lambda_candidates", [])] != list(LAMBDA_CANDIDATES):
        raise ValueError(f"lambda_candidates must be {list(LAMBDA_CANDIDATES)}")
    # Gate thresholds are frozen; surface drift loudly.
    for key, expected in (
        ("gate_i_aurc_regret_min", GATE_I_AURC_REGRET_MIN),
        ("gate_i_regret80_min", GATE_I_REGRET80_MIN),
        ("gate_strata_i_aurc_floor", GATE_STRATA_I_AURC_FLOOR),
        ("gate_task_min_nonzero_regret_frac", GATE_TASK_MIN_NONZERO_REGRET_FRAC),
    ):
        if float(checked.get(key, -9.0)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    if int(checked.get("gate_min_positive_pairs", -1)) != GATE_MIN_POSITIVE_PAIRS:
        raise ValueError("gate_min_positive_pairs drifted")
    return checked


# --- frozen-source inventory ------------------------------------------------------------


def inventory_frozen_sources() -> dict[str, Any]:
    """Resolve + hash every frozen Stage 4A/4B source artifact; raise on any drift."""
    resolved: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name, (rel, expected) in STAGE4_FROZEN.items():
        path = _resolve(rel)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Stage 4 source: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"frozen Stage 4 source hash drift: {name} ({rel}); "
                f"expected {expected}, got {digest}")
        resolved[name] = str(path)
        hashes[name] = digest
    return {"stage4_paths": resolved, "stage4_sha256": hashes}


# --- model + code inventory ---------------------------------------------------------


def inventory_model_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and hash every frozen model/scaler/binding dependency + Stage 4 sources."""
    checked = validate_config(config)
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
    inventory.update(inventory_frozen_sources())
    return inventory


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in CODE_FILES:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered Stage 4C code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


def _historical_inventory(config: Mapping[str, Any]) -> np.ndarray:
    return scan_historical_group_seeds(
        [_resolve(config["historical_data_root"]), ROOT / "experiments"])


def _schedule_split() -> dict[str, slice]:
    """Development roles bind to disjoint seed-segment halves of the dev schedule."""
    half = DEV_SCHEDULE_SIZE // 2
    return {
        "development_selection": slice(0, half),
        "development_screen": slice(half, DEV_SCHEDULE_SIZE),
    }


def build_development_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the (not yet written) development protocol lock payload.

    Builds + validates the dev seed schedule (split into selection/screen segments),
    scans the historical inventory for zero overlap and records every frozen input
    hash. Pure and side-effect free.
    """
    checked = validate_config(config)
    historical = _historical_inventory(checked)
    dev_schedule = make_group_seed_schedule(DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE)
    validate_seed_schedule(dev_schedule, historical)
    model_lock = inventory_model_lock(checked)
    segments = _schedule_split()
    return {
        "plan_id": PLAN_ID,
        "purpose_dev": PURPOSE_DEV,
        "config_signature": _config_signature(checked),
        "dev_seed_entropy": DEV_SEED_ENTROPY,
        "dev_schedule_size": DEV_SCHEDULE_SIZE,
        "dev_schedule_sha256": hashlib.sha256(dev_schedule.tobytes()).hexdigest(),
        "dev_selection_groups": DEV_SELECTION_GROUPS,
        "dev_screen_groups": DEV_SCREEN_GROUPS,
        "selection_segment": [segments["development_selection"].start,
                              segments["development_selection"].stop],
        "screen_segment": [segments["development_screen"].start,
                           segments["development_screen"].stop],
        "n_historical_episode_seeds": int(historical.size),
        "seed_schedule_zero_overlap": True,
        "action_plans": [list(p) for p in ACTION_PLANS],
        "code_sha256": _code_hashes(),
        **model_lock,
    }


def write_development_lock(config_path: str | Path) -> dict[str, Any]:
    """Create the development protocol lock + schedules exactly once."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "development"
    lock_path = protocol_dir / "protocol_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Stage 4C development lock already exists: {lock_path}")
    lock = build_development_lock(checked)
    lock["config_sha256"] = sha256_file(config_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    # Persist the two role seed slices create-once for the collector.
    dev_schedule = make_group_seed_schedule(DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE)
    segments = _schedule_split()
    for role, seg in segments.items():
        slice_path = protocol_dir / f"seed_schedule_{role}.npy"
        tmp = slice_path.with_name(slice_path.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.save(handle, dev_schedule[seg])
        Path(tmp).replace(slice_path)
    write_once_json(lock_path, lock)
    return lock


def build_confirm_lock(config: Mapping[str, Any], *, selected_n: int) -> dict[str, Any]:
    """Compute the confirm protocol lock payload, binding the chosen n + dev hashes."""
    checked = validate_config(config)
    selected_n = int(selected_n)
    if selected_n not in CANDIDATE_NS:
        raise ValueError(f"selected_n must be one of {CANDIDATE_NS}, got {selected_n}")
    historical = _historical_inventory(checked)
    confirm_schedule = make_group_seed_schedule(CONFIRM_SEED_ENTROPY, CONFIRM_SCHEDULE_SIZE)
    validate_seed_schedule(confirm_schedule, historical)
    # confirm schedule must also be disjoint from the development schedule.
    dev_schedule = make_group_seed_schedule(DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE)
    if np.intersect1d(confirm_schedule, dev_schedule).size:
        raise ValueError("confirm schedule overlaps the development schedule")
    model_lock = inventory_model_lock(checked)
    return {
        "plan_id": PLAN_ID,
        "purpose_confirm": PURPOSE_CONFIRM,
        "config_signature": _config_signature(checked),
        "selected_n": selected_n,
        "confirm_seed_entropy": CONFIRM_SEED_ENTROPY,
        "confirm_schedule_size": CONFIRM_SCHEDULE_SIZE,
        "confirm_schedule_sha256": hashlib.sha256(confirm_schedule.tobytes()).hexdigest(),
        "n_historical_episode_seeds": int(historical.size),
        "seed_schedule_zero_overlap": True,
        "action_plans": [list(p) for p in ACTION_PLANS],
        "code_sha256": _code_hashes(),
        **model_lock,
    }


def write_confirm_lock(config_path: str | Path, *, selected_n: int) -> dict[str, Any]:
    """Create the confirm protocol lock + schedule exactly once."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "confirm"
    lock_path = protocol_dir / "protocol_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Stage 4C confirm lock already exists: {lock_path}")
    lock = build_confirm_lock(checked, selected_n=selected_n)
    lock["config_sha256"] = sha256_file(config_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    schedule = make_group_seed_schedule(CONFIRM_SEED_ENTROPY, CONFIRM_SCHEDULE_SIZE)
    schedule_path = protocol_dir / "seed_schedule_confirm.npy"
    tmp = schedule_path.with_name(schedule_path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, schedule)
    Path(tmp).replace(schedule_path)
    write_once_json(lock_path, lock)
    return lock


def _config_signature(config: Mapping[str, Any]) -> str:
    from ac_twin.checkpointing import canonical_fingerprint
    return canonical_fingerprint(json.loads(json.dumps(config)))


__all__ = [
    "ACTION_PLANS",
    "BOOTSTRAP_ENTROPY_CONFIRM",
    "BOOTSTRAP_ENTROPY_SCREEN",
    "BOOTSTRAP_RESAMPLES",
    "CANDIDATE_NS",
    "CF_ACTIONS",
    "CF_DURATIONS",
    "CODE_FILES",
    "CONFIDENCE_CONFIRM",
    "CONFIDENCE_SCREEN",
    "CONFIRM_SCHEDULE_SIZE",
    "CONFIRM_SEED_ENTROPY",
    "COVERAGE_MAIN",
    "DEV_SCHEDULE_SIZE",
    "DEV_SCREEN_GROUPS",
    "DEV_SELECTION_GROUPS",
    "DEV_SEED_ENTROPY",
    "DEV_TOTAL_GROUPS",
    "ENV_DURATION",
    "ENV_ID",
    "EXPERT_ROUTER_SEEDS",
    "GATE_I_AURC_REGRET_MIN",
    "GATE_I_REGRET80_MIN",
    "GATE_MIN_POSITIVE_PAIRS",
    "GATE_STRATA_I_AURC_FLOOR",
    "GATE_STRATA_MIN_COUNT",
    "GATE_TASK_MIN_DISTINCT_ORACLE",
    "GATE_TASK_MIN_NONZERO_REGRET_FRAC",
    "GATE_TASK_MIN_ORACLE_FRAC",
    "GATE_TASK_REGRET_EPS",
    "GATE_TRIM_FRACTIONS",
    "HIGH_REGRET_QUANTILE",
    "HISTORY_LEN",
    "HORIZON",
    "LAMBDA_CANDIDATES",
    "N_ACTIONS",
    "N_OBJECT_FEATURES",
    "OBJECT_CAPACITY",
    "OBSERVED_VEHICLES_COUNT",
    "PLAN_ID",
    "POLICY_FREQUENCY",
    "POWER_BOOTSTRAP_RESAMPLES",
    "POWER_MONTE_CARLO",
    "POWER_MONTE_CARLO_ENTROPY",
    "POWER_SHRINK_MAIN",
    "POWER_SHRINK_SCENARIOS",
    "POWER_TARGET",
    "PRIMARY_CONSTANT_W",
    "PURPOSE_CONFIRM",
    "PURPOSE_DEV",
    "RESPONSE_PREDICTOR_SEEDS",
    "SEED_HIGH_INCLUSIVE",
    "SEED_LOW",
    "SIMULATION_FREQUENCY",
    "STAGE4_FROZEN",
    "STATUS_COLLECTION_INCOMPLETE",
    "STATUS_CONFIRMED",
    "STATUS_DEV_NOT_READY",
    "STATUS_DEV_READY",
    "STATUS_INVALID",
    "STATUS_MODEL_INSUFFICIENT",
    "STATUS_NOT_CONFIRMED",
    "STATUS_RERANK_DEGENERATE",
    "STATUS_RERANK_NOT_SUPPORTED",
    "STATUS_RERANK_SUPPORTED",
    "STATUS_TASK_DEGENERATE",
    "STATUS_UNDERPOWERED",
    "TOP_ERROR_THRESHOLD",
    "T_STAR_HIGH",
    "T_STAR_LOW",
    "VEHICLES_COUNT",
    "assert_allowed_stage4c_data",
    "build_confirm_lock",
    "build_development_lock",
    "build_inference_mask",
    "inventory_frozen_sources",
    "inventory_model_lock",
    "make_group_seed_schedule",
    "materialize_action_plan",
    "scan_historical_group_seeds",
    "validate_config",
    "write_confirm_lock",
    "write_development_lock",
    "write_once_json",
]
