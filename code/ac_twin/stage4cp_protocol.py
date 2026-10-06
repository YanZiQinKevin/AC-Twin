"""Stage 4C Plus protocol: group seed schedules, decision-risk constants, freeze layer.

Freeze layer for the Stage 4C Plus decision-aware comparative-risk confirmation
(plan ``AC-TWIN-RFER-STAGE4C-PLUS-DECISION-AWARE-RISK-2026-09-16-V0.1``). It never
trains on this layer, never refits a scaler/threshold outside the train role, never
re-selects a baseline/finalist outside validation and never reads a regret/label
before the matching one-shot lock is burned. Its only pre-collection numeric products
are the two group seed schedules (development entropy ``2026092401`` size 30,000;
confirm entropy ``2026092601`` size 25,000) plus the historical+reserved seed
inventory used to prove zero overlap.

Statistical unit (plan Global Constraints): the multi-action *decision group*, not an
action branch, vehicle or time point. Each accepted group shares one history, one
factual branch and one ``t_star``, and forks into the ten frozen ``ACTION_PLANS``
counterfactual branches (REUSED bit-identically from Stage 4C). All resampling is
whole-group.

Relationship to Stage 4C v0.1 (plan Global Constraints):
  * the Stage 4C 300-group ``development-screen`` (data, quantities, report, lock) is
    PERMANENTLY read-only and never a Stage 4C Plus input for features/training/power;
  * the Stage 4C 500-group ``development-selection`` may be read ONLY by the Phase 0
    exploratory OOF screen; after Phase 0 it never enters training/validation/screen/
    power/confirm;
  * Stage 4A/4B frozen predictors, experts, response predictor, scaler and risk heads
    are bound READ-ONLY (identity by content SHA256, never by drive letter).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin import stage3_data as s3d
from ac_twin import stage4c_protocol as s4c
from ac_twin.stage1_data import sha256_file
from ac_twin.stage2_protocol import make_seed_schedule, validate_seed_schedule
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

PLAN_ID = "AC-TWIN-RFER-STAGE4C-PLUS-DECISION-AWARE-RISK-2026-09-16-V0.1"
PURPOSE_DEV = "stage4cp_merge_multiaction_dev"
PURPOSE_CONFIRM = "stage4cp_merge_multiaction_confirm"

# --- environment / tensor contract (identical to Stage 4C, plan section 2) ----------
ENV_ID = s4c.ENV_ID
HISTORY_LEN = s4c.HISTORY_LEN
HORIZON = s4c.HORIZON
T_STAR_LOW = s4c.T_STAR_LOW
T_STAR_HIGH = s4c.T_STAR_HIGH
SIMULATION_FREQUENCY = s4c.SIMULATION_FREQUENCY
POLICY_FREQUENCY = s4c.POLICY_FREQUENCY
VEHICLES_COUNT = s4c.VEHICLES_COUNT
OBSERVED_VEHICLES_COUNT = s4c.OBSERVED_VEHICLES_COUNT
ENV_DURATION = s4c.ENV_DURATION
OBJECT_CAPACITY = s4c.OBJECT_CAPACITY
N_OBJECT_FEATURES = s4c.N_OBJECT_FEATURES
N_ACTIONS = len(ACTION_PLANS)
N_COMPETITORS = N_ACTIONS - 1  # nine relative-comparison tokens per group

# --- development / confirm sizing (plan section 6) -----------------------------------
DEV_TRAIN_GROUPS = 5000
DEV_VALIDATION_GROUPS = 1500
DEV_SCREEN_GROUPS = 1000
CANDIDATE_NS = (1200, 1800, 2400, 3600)

# --- frozen group seed schedules (plan section 6.2) ----------------------------------
DEV_SEED_ENTROPY = 2026092401
DEV_SCHEDULE_SIZE = 30000
TRAIN_SEGMENT = (0, 18000)
VALIDATION_SEGMENT = (18000, 24000)
SCREEN_SEGMENT = (24000, 30000)
CONFIRM_SEED_ENTROPY = 2026092601
CONFIRM_SCHEDULE_SIZE = 25000
SEED_LOW = 1
SEED_HIGH_INCLUSIVE = 2147483646  # 2**31 - 2

# Stage 4C's unused confirm schedule (never persisted) must be reserved by contract.
STAGE4C_CONFIRM_SEED_ENTROPY = s4c.CONFIRM_SEED_ENTROPY
STAGE4C_CONFIRM_SCHEDULE_SIZE = s4c.CONFIRM_SCHEDULE_SIZE

# --- Phase 0 (plan section 5) ----------------------------------------------------------
PHASE0_FOLD_ENTROPY = 2026092301
PHASE0_BOOTSTRAP_ENTROPY = 2026092302
PHASE0_N_FOLDS = 5
PHASE0_BOOTSTRAP_RESAMPLES = 5000
PHASE0_INIT_SEEDS = (6101, 6202, 6303)
PHASE0_HIDDEN_DIM = 64
PHASE0_LAMBDA_CLS = 0.25
PHASE0_I_AURC_MIN = 0.05
PHASE0_I_REGRET80_MIN = 0.02
PHASE0_MIN_FOLDS_POSITIVE = 4

# --- bounded training / selection (plan section 7) --------------------------------------
HIDDEN_DIM_CANDIDATES = (64, 128)
LAMBDA_CLS_CANDIDATES = (0.25, 0.50)
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 128
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 12
TRAINING_SEEDS = (7101, 7202, 7303, 7404, 7505)
SELECTION_SEEDS = (7101, 7202, 7303)  # first three used for config selection
PARAM_COUNT_TOLERANCE = 0.05
VALIDATION_I_AURC_MIN = 0.05
VALIDATION_I_REGRET80_MIN = 0.03
VALIDATION_MIN_POSITIVE_SEEDS = 4
FINALIST_AURC_TIE_TOLERANCE = 0.005

# --- frozen statistical entropies (plan sections 8, 9, 10) -----------------------------
BOOTSTRAP_ENTROPY_SCREEN = 2026092402
BOOTSTRAP_ENTROPY_CONFIRM = 2026092602
POWER_MONTE_CARLO_ENTROPY = 2026092501
BOOTSTRAP_RESAMPLES = 10000
POWER_BOOTSTRAP_RESAMPLES = 10000
POWER_MONTE_CARLO = 10000
CONFIDENCE_SCREEN = 0.80
CONFIDENCE_CONFIRM = 0.95
POWER_TARGET = 0.80
POWER_SHRINK_MAIN = 0.75
POWER_SHRINK_SCENARIOS = (1.0, 0.75, 0.5)

# --- frozen risk labels / coverage (plan section 3.3) ----------------------------------
COVERAGE_MAIN = 0.80
HIGH_REGRET_THRESHOLD = 0.01  # FIXED (matches Gate TASK substantive regret), never re-fit

# --- Gate TASK+ thresholds (plan sections 8.2 / 11.2) -----------------------------------
GATE_TASK_MIN_HIGH_REGRET_FRAC = 0.15
GATE_TASK_REGRET_EPS = 0.01
GATE_TASK_MIN_DISTINCT_ORACLE = 3
GATE_TASK_MIN_ORACLE_FRAC = 0.05
GATE_TASK_MIN_STRATUM_COUNT = 200
STRATUM_IMBALANCE_FRAC = 0.20

# --- Gate RISK-DECISION+ thresholds (plan sections 8.3 / 11.3) --------------------------
GATE_I_AURC_REGRET_MIN = 0.05
GATE_I_REGRET80_MIN = 0.03
GATE_MIN_POSITIVE_SEEDS = 4
GATE_STRATA_I_AURC_FLOOR = -0.10
GATE_TRIM_FRACTIONS = (0.01, 0.05)
GATE_MARGIN_SPEARMAN_CEILING = 0.995

# --- frozen Stage 4A carry-over ----------------------------------------------------------
TOP_ERROR_THRESHOLD = s4c.TOP_ERROR_THRESHOLD  # inherited; NOT recomputed

# --- mechanical terminal states (plan sections 5, 6, 7, 8, 9, 11) ------------------------
STATUS_FEASIBILITY_NOT_SUPPORTED = "STAGE4CP_FEASIBILITY_NOT_SUPPORTED"
STATUS_TRAIN_SUPPORT_INSUFFICIENT = "STAGE4CP_TRAIN_SUPPORT_INSUFFICIENT"
STATUS_MODEL_DEV_INSUFFICIENT = "STAGE4CP_MODEL_DEV_INSUFFICIENT"
STATUS_STRATUM_IMBALANCED = "STRATUM_IMBALANCED"
STATUS_TASK_DEGENERATE = "STAGE4CP_TASK_DEGENERATE"
STATUS_MODEL_INSUFFICIENT = "INCONCLUSIVE_STAGE4CP_DECISION_MODEL_INSUFFICIENT"
STATUS_DEV_READY = "STAGE4CP_DEV_READY_FOR_CONFIRMATION"
STATUS_DEV_NOT_READY = "STAGE4CP_DEV_NOT_READY_FOR_CONFIRMATION"
STATUS_UNDERPOWERED = "STAGE4CP_CONFIRMATION_UNDERPOWERED"
STATUS_INVALID = "STAGE4CP_INVALID_ANALYSIS"
STATUS_CONFIRMED = "STAGE4CP_DECISION_RISK_CONFIRMED"
STATUS_NOT_CONFIRMED = "STAGE4CP_DECISION_RISK_NOT_CONFIRMED"
STATUS_COLLECTION_INCOMPLETE = "STAGE4CP_COLLECTION_INCOMPLETE"

# Architecture identifiers (plan section 4).
ARCH_DCR = "DECISION_COMPARATIVE_RISK"
ARCH_DIRECT = "SAME_INFORMATION_DIRECT"

# Non-learning / historical baselines (plan section 4.3). STRONGEST_STANDARD_BASELINE is
# chosen ONLY from the first three by lowest validation AURC_REGRET; COMBINED_TOP2_MAX is
# a historical ablation and CONSTANT_RANDOM a curve reference (neither is eligible).
BASELINE_MARGIN = "MARGIN_ONLY"
BASELINE_SEED_VOTE_ENTROPY = "SEED_VOTE_ENTROPY"
BASELINE_SEED_MARGIN_STD = "SEED_MARGIN_STD"
BASELINE_COMBINED_TOP2_MAX = "COMBINED_TOP2_MAX"
BASELINE_CONSTANT_RANDOM = "CONSTANT_RANDOM"
STRONGEST_BASELINE_CANDIDATES = (
    BASELINE_MARGIN, BASELINE_SEED_VOTE_ENTROPY, BASELINE_SEED_MARGIN_STD)

# Action-level risk classes carried into the relative tokens (plan section 3.2).
RISK_CLASSES = (
    "COMBINED_RISK_HEAD", "UNCERTAINTY_ONLY", "GEOMETRY_ONLY",
    "RESPONSE_ONLY", "DISAGREEMENT_ONLY",
)

# New-code files whose SHA256 are enumerated in the protocol locks (relative to code/).
CODE_FILES = (
    "run_stage4cp.py",
    "collect_stage4cp_merge_multiaction.py",
    "ac_twin/stage4cp_protocol.py",
    "ac_twin/stage4cp_features.py",
    "ac_twin/stage4cp_models.py",
    "ac_twin/stage4cp_training.py",
    "ac_twin/stage4cp_statistics.py",
    "ac_twin/stage4cp_power.py",
)
# Frozen Stage 4C chain modules reused read-only; their hashes bind the exact chain.
REUSED_FROZEN_CODE_FILES = (
    "ac_twin/stage4c_objective.py",
    "ac_twin/stage4c_inference.py",
    "ac_twin/stage4c_protocol.py",
)
# Code files that exist and are bound at the Phase 0 report (Task 1 scope).
PHASE0_CODE_FILES = (
    "run_stage4cp.py",
    "ac_twin/stage4cp_protocol.py",
    "ac_twin/stage4cp_features.py",
    "ac_twin/stage4cp_models.py",
)

# Frozen Stage 4A/4B source artifacts with their known SHA256 (reused from Stage 4C).
STAGE4_FROZEN = s4c.STAGE4_FROZEN

# Forbidden source path tokens (plan section 7.3 of Stage 4C, inherited). Stage 4C Plus
# additionally forbids every Stage 4A/4B development/confirm product, the Stage 2B data
# AND the Stage 4C screen (permanent read-only).
_FORBIDDEN_DATA_TOKENS = s4c._FORBIDDEN_DATA_TOKENS

# Stage 4C screen markers (permanently read-only). The selection role is distinguishable
# by the ``_selection`` suffix and is handled separately below.
_STAGE4C_SCREEN_MARKERS = (
    "stage4c_merge_multiaction_dev_screen",
    "predictions_development_screen",
    "risk_scores_development_screen",
    "screen_group_quantities",
    "development_screen_report",
    "development_screen_used",
)
# Stage 4C selection markers (allowed ONLY for Phase 0).
_STAGE4C_SELECTION_MARKERS = (
    "stage4c_merge_multiaction_dev_selection",
    "predictions_development_selection",
    "risk_scores_development_selection",
    "decision_rule_receipt",
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


def is_stage4c_screen(path: str | Path) -> bool:
    """True if ``path`` is any Stage 4C one-shot screen product (permanent read-only)."""
    logical = _normalised(path)
    return any(marker in logical for marker in _STAGE4C_SCREEN_MARKERS)


def is_stage4c_selection(path: str | Path) -> bool:
    """True if ``path`` is a Stage 4C development-selection product (Phase 0 only)."""
    logical = _normalised(path)
    return any(marker in logical for marker in _STAGE4C_SELECTION_MARKERS)


def assert_allowed_stage4cp_data(path: str | Path, *, allow_phase0_selection: bool = False) -> None:
    """Raise ``ValueError`` if ``path`` is a forbidden Stage 4C Plus input.

    Legal Stage 4C Plus data roles are: (a) this stage's OWN dev/confirm purpose dirs,
    and (b) the Stage 4C development-selection products, only when
    ``allow_phase0_selection`` is set (Phase 0). The Stage 4C screen, every sealed
    Stage 2/3 set, the Stage 4A/4B dev/confirm products and the Stage 2B data are all
    refused by path segment.
    """
    s3d.assert_not_sealed_stage2(path)
    logical = _normalised(path)
    for token in _FORBIDDEN_DATA_TOKENS:
        for segment in logical.split("/"):
            if segment.startswith(token):
                raise ValueError(
                    f"forbidden Stage 4C Plus data source (segment {segment!r}): {path}")
    if is_stage4c_screen(path):
        raise ValueError(f"Stage 4C screen is permanently read-only: {path}")
    if PURPOSE_DEV in logical or PURPOSE_CONFIRM in logical:
        return  # this stage's own collected data
    if is_stage4c_selection(path):
        if allow_phase0_selection:
            return
        raise ValueError(
            f"Stage 4C selection is only a Phase 0 input (not train/val/screen/power): {path}")
    raise ValueError(f"not an allowed Stage 4C Plus data source: {path}")


# --- group seed schedules -----------------------------------------------------------


def make_group_seed_schedule(entropy: int, size: int) -> np.ndarray:
    """Unique positive int32 group seeds for one frozen (entropy, size) contract.

    Only the two frozen Stage 4C Plus contracts are legal: development
    (``2026092401`` / 30,000) and confirm (``2026092601`` / 25,000). Reuses the frozen
    Stage 2 generator so the schedule is internally unique and in-range.
    """
    entropy = int(entropy)
    size = int(size)
    legal = {(DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE),
             (CONFIRM_SEED_ENTROPY, CONFIRM_SCHEDULE_SIZE)}
    if (entropy, size) not in legal:
        raise ValueError(
            f"Stage 4C Plus group seed (entropy, size) must be one of {sorted(legal)}; "
            f"got ({entropy}, {size})")
    schedule = make_seed_schedule(entropy, size=size)
    if schedule.min() < SEED_LOW or schedule.max() > SEED_HIGH_INCLUSIVE:
        raise ValueError("Stage 4C Plus group seed schedule out of range")
    return schedule.astype(np.int64)


def _schedule_segments() -> dict[str, tuple[int, int]]:
    """Development roles bind to disjoint fixed segments of the dev schedule."""
    return {
        "development_train": TRAIN_SEGMENT,
        "development_validation": VALIDATION_SEGMENT,
        "development_screen": SCREEN_SEGMENT,
    }


def role_segment(role: str) -> tuple[int, int]:
    segments = _schedule_segments()
    if role not in segments:
        raise ValueError(f"unknown development role segment: {role}")
    return segments[role]


# --- historical + reserved seed inventory --------------------------------------------


def scan_historical_group_seeds(workspace_roots: Sequence[Path]) -> np.ndarray:
    """Union of every historical *accepted* seed AND every reserved candidate pool.

    Scope (plan section 6.2): accepted episode/group seeds across NPZ / receipts /
    manifests, PLUS every persisted ``*seed_schedule*.npy`` candidate pool, PLUS the
    never-persisted Stage 4C confirm schedule regenerated by contract. This stage's own
    dev/confirm products are excluded so a resume does not collide with itself; every
    prior-stage pool (including the Stage 2 frozen 18,000 and both Stage 4C dev halves)
    IS included so Stage 4C Plus stays disjoint from all of them.
    """
    collected: set[int] = set()
    for root in workspace_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob("*")):
            normalized = _normalised(file_path)
            if PURPOSE_DEV in normalized or PURPOSE_CONFIRM in normalized:
                continue  # this stage's own products
            if "stage4c_plus" in normalized:
                continue
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".npz":
                    with np.load(file_path, allow_pickle=True) as archive:
                        for key in ("episode_seed", "episode_seeds", "group_seed", "group_seeds"):
                            if key in archive.files:
                                values = np.atleast_1d(np.asarray(archive[key])).ravel()
                                collected.update(int(v) for v in values)
                elif suffix == ".npy" and "seed_schedule" in file_path.name.lower():
                    values = np.load(file_path).ravel()
                    collected.update(int(v) for v in values)
                elif suffix == ".json":
                    collected.update(s4c._seeds_in_json(file_path))
            except Exception:
                continue
    # Reserve the never-persisted Stage 4C confirm schedule by contract.
    stage4c_confirm = s4c.make_group_seed_schedule(
        STAGE4C_CONFIRM_SEED_ENTROPY, STAGE4C_CONFIRM_SCHEDULE_SIZE)
    collected.update(int(v) for v in stage4c_confirm)
    if not collected:
        raise ValueError("no historical episode seeds found in the workspace")
    return np.array(sorted(collected), dtype=np.int64)


# --- inference mask (REUSED bit-identically from Stage 4C) -----------------------------

build_inference_mask = s4c.build_inference_mask
materialize_action_plan = s4c.materialize_action_plan


# --- config validation -----------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 4C Plus config; raise before any artifact is produced."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("scene") != "merge" or checked.get("env_id") != ENV_ID:
        raise ValueError("Stage 4C Plus is merge-v0 only")
    if [tuple(p) for p in checked.get("action_plans", [])] != [tuple(p) for p in ACTION_PLANS]:
        raise ValueError(f"action_plans must be {list(ACTION_PLANS)}")
    if list(checked.get("cf_actions", [])) != list(CF_ACTIONS):
        raise ValueError(f"cf_actions must be {list(CF_ACTIONS)}")
    if list(checked.get("intervention_durations", [])) != list(CF_DURATIONS):
        raise ValueError(f"intervention_durations must be {list(CF_DURATIONS)}")
    for key, expected in (
        ("history_len", HISTORY_LEN), ("horizon", HORIZON),
        ("t_star_low", T_STAR_LOW), ("t_star_high", T_STAR_HIGH),
        ("simulation_frequency", SIMULATION_FREQUENCY),
        ("policy_frequency", POLICY_FREQUENCY),
        ("vehicles_count", VEHICLES_COUNT), ("duration", ENV_DURATION),
        ("observed_vehicles_count", OBSERVED_VEHICLES_COUNT),
        ("object_capacity", OBJECT_CAPACITY),
        ("dev_train_groups", DEV_TRAIN_GROUPS),
        ("dev_validation_groups", DEV_VALIDATION_GROUPS),
        ("dev_screen_groups", DEV_SCREEN_GROUPS),
        ("dev_seed_entropy", DEV_SEED_ENTROPY),
        ("dev_schedule_size", DEV_SCHEDULE_SIZE),
        ("confirm_seed_entropy", CONFIRM_SEED_ENTROPY),
        ("confirm_schedule_size", CONFIRM_SCHEDULE_SIZE),
        ("bootstrap_entropy_screen", BOOTSTRAP_ENTROPY_SCREEN),
        ("bootstrap_entropy_confirm", BOOTSTRAP_ENTROPY_CONFIRM),
        ("power_monte_carlo_entropy", POWER_MONTE_CARLO_ENTROPY),
        ("phase0_fold_entropy", PHASE0_FOLD_ENTROPY),
        ("phase0_bootstrap_entropy", PHASE0_BOOTSTRAP_ENTROPY),
        ("phase0_n_folds", PHASE0_N_FOLDS),
        ("phase0_bootstrap_resamples", PHASE0_BOOTSTRAP_RESAMPLES),
        ("phase0_hidden_dim", PHASE0_HIDDEN_DIM),
        ("max_epochs", MAX_EPOCHS), ("early_stop_patience", EARLY_STOP_PATIENCE),
        ("train_batch_size", TRAIN_BATCH_SIZE),
    ):
        if int(checked.get(key, -1)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    for key, expected in (
        ("confidence_screen", CONFIDENCE_SCREEN),
        ("confidence_confirm", CONFIDENCE_CONFIRM),
        ("power_target", POWER_TARGET), ("power_shrink_main", POWER_SHRINK_MAIN),
        ("coverage_main", COVERAGE_MAIN),
        ("high_regret_threshold", HIGH_REGRET_THRESHOLD),
        ("gate_i_aurc_regret_min", GATE_I_AURC_REGRET_MIN),
        ("gate_i_regret80_min", GATE_I_REGRET80_MIN),
        ("gate_strata_i_aurc_floor", GATE_STRATA_I_AURC_FLOOR),
        ("gate_margin_spearman_ceiling", GATE_MARGIN_SPEARMAN_CEILING),
        ("gate_task_min_high_regret_frac", GATE_TASK_MIN_HIGH_REGRET_FRAC),
        ("gate_task_regret_eps", GATE_TASK_REGRET_EPS),
        ("gate_task_min_oracle_frac", GATE_TASK_MIN_ORACLE_FRAC),
        ("phase0_i_aurc_min", PHASE0_I_AURC_MIN),
        ("phase0_i_regret80_min", PHASE0_I_REGRET80_MIN),
        ("phase0_lambda_cls", PHASE0_LAMBDA_CLS),
        ("dropout", DROPOUT), ("learning_rate", LEARNING_RATE),
        ("weight_decay", WEIGHT_DECAY),
        ("validation_i_aurc_min", VALIDATION_I_AURC_MIN),
        ("validation_i_regret80_min", VALIDATION_I_REGRET80_MIN),
        ("finalist_aurc_tie_tolerance", FINALIST_AURC_TIE_TOLERANCE),
        ("param_count_tolerance", PARAM_COUNT_TOLERANCE),
        ("top_error_threshold", TOP_ERROR_THRESHOLD),
        ("primary_constant_w", PRIMARY_CONSTANT_W),
    ):
        if float(checked.get(key, -9.0)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    if [int(n) for n in checked.get("candidate_ns", [])] != list(CANDIDATE_NS):
        raise ValueError(f"candidate_ns must be {list(CANDIDATE_NS)}")
    if [int(s) for s in checked.get("phase0_init_seeds", [])] != list(PHASE0_INIT_SEEDS):
        raise ValueError(f"phase0_init_seeds must be {list(PHASE0_INIT_SEEDS)}")
    if [int(s) for s in checked.get("training_seeds", [])] != list(TRAINING_SEEDS):
        raise ValueError(f"training_seeds must be {list(TRAINING_SEEDS)}")
    if [int(s) for s in checked.get("selection_seeds", [])] != list(SELECTION_SEEDS):
        raise ValueError(f"selection_seeds must be {list(SELECTION_SEEDS)}")
    if [int(h) for h in checked.get("hidden_dim_candidates", [])] != list(HIDDEN_DIM_CANDIDATES):
        raise ValueError(f"hidden_dim_candidates must be {list(HIDDEN_DIM_CANDIDATES)}")
    if [float(x) for x in checked.get("lambda_cls_candidates", [])] != list(LAMBDA_CLS_CANDIDATES):
        raise ValueError(f"lambda_cls_candidates must be {list(LAMBDA_CLS_CANDIDATES)}")
    if [float(x) for x in checked.get("gate_trim_fractions", [])] != list(GATE_TRIM_FRACTIONS):
        raise ValueError(f"gate_trim_fractions must be {list(GATE_TRIM_FRACTIONS)}")
    if [int(x) for x in checked.get("train_segment", [])] != list(TRAIN_SEGMENT):
        raise ValueError(f"train_segment must be {list(TRAIN_SEGMENT)}")
    if [int(x) for x in checked.get("validation_segment", [])] != list(VALIDATION_SEGMENT):
        raise ValueError(f"validation_segment must be {list(VALIDATION_SEGMENT)}")
    if [int(x) for x in checked.get("screen_segment", [])] != list(SCREEN_SEGMENT):
        raise ValueError(f"screen_segment must be {list(SCREEN_SEGMENT)}")
    if int(checked.get("gate_min_positive_seeds", -1)) != GATE_MIN_POSITIVE_SEEDS:
        raise ValueError("gate_min_positive_seeds drifted")
    if int(checked.get("gate_task_min_distinct_oracle", -1)) != GATE_TASK_MIN_DISTINCT_ORACLE:
        raise ValueError("gate_task_min_distinct_oracle drifted")
    if int(checked.get("gate_task_min_stratum_count", -1)) != GATE_TASK_MIN_STRATUM_COUNT:
        raise ValueError("gate_task_min_stratum_count drifted")
    if int(checked.get("validation_min_positive_seeds", -1)) != VALIDATION_MIN_POSITIVE_SEEDS:
        raise ValueError("validation_min_positive_seeds drifted")
    if int(checked.get("phase0_min_folds_positive", -1)) != PHASE0_MIN_FOLDS_POSITIVE:
        raise ValueError("phase0_min_folds_positive drifted")
    if list(checked.get("response_predictor_seeds", [])) != list(RESPONSE_PREDICTOR_SEEDS):
        raise ValueError("response_predictor_seeds drifted from the frozen five")
    if list(checked.get("expert_router_seeds", [])) != list(EXPERT_ROUTER_SEEDS):
        raise ValueError("expert_router_seeds drifted from the frozen five pairs")
    return checked


# --- frozen-source inventory ------------------------------------------------------------


def inventory_frozen_sources() -> dict[str, Any]:
    """Resolve + hash every frozen Stage 4A/4B source artifact; raise on any drift."""
    return s4c.inventory_frozen_sources()


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


def _code_hashes(files: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in files:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered Stage 4C Plus code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


def phase0_code_hashes() -> dict[str, str]:
    return _code_hashes(PHASE0_CODE_FILES)


def full_code_hashes() -> dict[str, str]:
    return _code_hashes(CODE_FILES + REUSED_FROZEN_CODE_FILES)


def _historical_inventory(config: Mapping[str, Any]) -> np.ndarray:
    return scan_historical_group_seeds(
        [_resolve(config["historical_data_root"]), ROOT / "experiments"])


# --- development / confirm protocol locks ----------------------------------------------


def build_development_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the (not yet written) development protocol lock payload.

    Builds + validates the dev seed schedule (split into train/validation/screen
    segments), scans the historical+reserved inventory for zero overlap and records
    every frozen input hash. Pure and side-effect free.
    """
    checked = validate_config(config)
    historical = _historical_inventory(checked)
    dev_schedule = make_group_seed_schedule(DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE)
    validate_seed_schedule(dev_schedule, historical)
    model_lock = inventory_model_lock(checked)
    segments = _schedule_segments()
    return {
        "plan_id": PLAN_ID,
        "purpose_dev": PURPOSE_DEV,
        "config_signature": _config_signature(checked),
        "dev_seed_entropy": DEV_SEED_ENTROPY,
        "dev_schedule_size": DEV_SCHEDULE_SIZE,
        "dev_schedule_sha256": hashlib.sha256(dev_schedule.tobytes()).hexdigest(),
        "dev_train_groups": DEV_TRAIN_GROUPS,
        "dev_validation_groups": DEV_VALIDATION_GROUPS,
        "dev_screen_groups": DEV_SCREEN_GROUPS,
        "train_segment": list(segments["development_train"]),
        "validation_segment": list(segments["development_validation"]),
        "screen_segment": list(segments["development_screen"]),
        "n_historical_episode_seeds": int(historical.size),
        "seed_schedule_zero_overlap": True,
        "action_plans": [list(p) for p in ACTION_PLANS],
        "high_regret_threshold": HIGH_REGRET_THRESHOLD,
        "code_sha256": full_code_hashes(),
        **model_lock,
    }


def write_development_lock(config_path: str | Path) -> dict[str, Any]:
    """Create the development protocol lock + per-role seed slices exactly once."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "development"
    lock_path = protocol_dir / "protocol_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Stage 4C Plus development lock already exists: {lock_path}")
    lock = build_development_lock(checked)
    lock["config_sha256"] = sha256_file(config_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    dev_schedule = make_group_seed_schedule(DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE)
    for role, (start, stop) in _schedule_segments().items():
        slice_path = protocol_dir / f"seed_schedule_{role}.npy"
        tmp = slice_path.with_name(slice_path.name + ".tmp")
        with open(tmp, "wb") as handle:
            np.save(handle, dev_schedule[start:stop])
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
        "high_regret_threshold": HIGH_REGRET_THRESHOLD,
        "code_sha256": full_code_hashes(),
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
        raise FileExistsError(f"Stage 4C Plus confirm lock already exists: {lock_path}")
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
    "ARCH_DCR",
    "ARCH_DIRECT",
    "BASELINE_COMBINED_TOP2_MAX",
    "BASELINE_CONSTANT_RANDOM",
    "BASELINE_MARGIN",
    "BASELINE_SEED_MARGIN_STD",
    "BASELINE_SEED_VOTE_ENTROPY",
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
    "DEV_SEED_ENTROPY",
    "DEV_TRAIN_GROUPS",
    "DEV_VALIDATION_GROUPS",
    "DROPOUT",
    "EARLY_STOP_PATIENCE",
    "EGO_SLOT",
    "ENV_DURATION",
    "ENV_ID",
    "EXPERT_ROUTER_SEEDS",
    "FINALIST_AURC_TIE_TOLERANCE",
    "GATE_I_AURC_REGRET_MIN",
    "GATE_I_REGRET80_MIN",
    "GATE_MARGIN_SPEARMAN_CEILING",
    "GATE_MIN_POSITIVE_SEEDS",
    "GATE_STRATA_I_AURC_FLOOR",
    "GATE_TASK_MIN_DISTINCT_ORACLE",
    "GATE_TASK_MIN_HIGH_REGRET_FRAC",
    "GATE_TASK_MIN_ORACLE_FRAC",
    "GATE_TASK_MIN_STRATUM_COUNT",
    "GATE_TASK_REGRET_EPS",
    "GATE_TRIM_FRACTIONS",
    "HIDDEN_DIM_CANDIDATES",
    "HIGH_REGRET_THRESHOLD",
    "HISTORY_LEN",
    "HORIZON",
    "LAMBDA_CLS_CANDIDATES",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "N_ACTIONS",
    "N_COMPETITORS",
    "N_OBJECT_FEATURES",
    "OBJECT_CAPACITY",
    "OBSERVED_VEHICLES_COUNT",
    "PARAM_COUNT_TOLERANCE",
    "PHASE0_BOOTSTRAP_ENTROPY",
    "PHASE0_BOOTSTRAP_RESAMPLES",
    "PHASE0_CODE_FILES",
    "PHASE0_FOLD_ENTROPY",
    "PHASE0_HIDDEN_DIM",
    "PHASE0_I_AURC_MIN",
    "PHASE0_INIT_SEEDS",
    "PHASE0_I_REGRET80_MIN",
    "PHASE0_LAMBDA_CLS",
    "PHASE0_MIN_FOLDS_POSITIVE",
    "PHASE0_N_FOLDS",
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
    "REUSED_FROZEN_CODE_FILES",
    "RISK_CLASSES",
    "SCREEN_SEGMENT",
    "SEED_HIGH_INCLUSIVE",
    "SEED_LOW",
    "SELECTION_SEEDS",
    "SIMULATION_FREQUENCY",
    "STAGE4_FROZEN",
    "STATUS_COLLECTION_INCOMPLETE",
    "STATUS_CONFIRMED",
    "STATUS_DEV_NOT_READY",
    "STATUS_DEV_READY",
    "STATUS_FEASIBILITY_NOT_SUPPORTED",
    "STATUS_INVALID",
    "STATUS_MODEL_DEV_INSUFFICIENT",
    "STATUS_MODEL_INSUFFICIENT",
    "STATUS_NOT_CONFIRMED",
    "STATUS_STRATUM_IMBALANCED",
    "STATUS_TASK_DEGENERATE",
    "STATUS_TRAIN_SUPPORT_INSUFFICIENT",
    "STATUS_UNDERPOWERED",
    "STRATUM_IMBALANCE_FRAC",
    "STRONGEST_BASELINE_CANDIDATES",
    "TOP_ERROR_THRESHOLD",
    "TRAIN_BATCH_SIZE",
    "TRAIN_SEGMENT",
    "TRAINING_SEEDS",
    "T_STAR_HIGH",
    "T_STAR_LOW",
    "VALIDATION_I_AURC_MIN",
    "VALIDATION_I_REGRET80_MIN",
    "VALIDATION_MIN_POSITIVE_SEEDS",
    "VALIDATION_SEGMENT",
    "VEHICLES_COUNT",
    "WEIGHT_DECAY",
    "assert_allowed_stage4cp_data",
    "build_confirm_lock",
    "build_development_lock",
    "build_inference_mask",
    "full_code_hashes",
    "inventory_frozen_sources",
    "inventory_model_lock",
    "is_stage4c_screen",
    "is_stage4c_selection",
    "make_group_seed_schedule",
    "materialize_action_plan",
    "phase0_code_hashes",
    "role_segment",
    "scan_historical_group_seeds",
    "validate_config",
    "write_confirm_lock",
    "write_development_lock",
    "write_once_json",
]
