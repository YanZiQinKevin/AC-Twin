"""Stage 4C-R2 protocol: frozen constants, seed schedule, zero-overlap audit, dev lock.

Freeze layer for the Stage 4C-R2 decision-risk scaling development validation
(plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2``). R2 trains one
fixed ``SAME_INFORMATION_DIRECT`` model on fresh, strictly separated roles
(5,000 train / 1,500 validation / 8,000 one-shot screen groups). It never re-runs or
re-judges Stage 4C Plus Phase 0, never reads the Stage 4C screen products, and never
reassigns a seed role outside the fixed schedule segments.

v0.2 note: the development seed entropy is ``2026092880``. The v0.1 entropy
(``2026092701``) failed the mandatory global zero-overlap audit (four collisions with
persisted reserved pools and the contract-reserved Stage 4C / 4C Plus confirm
schedules); per plan section 5.2 the plan was re-versioned, never patched in place.

Statistical unit (plan Global Constraints): the multi-action *decision group*. All
resampling is whole-group. The ten action plans, the consequence function, the regret
clamp and the feature contract are REUSED bitwise from Stage 4C / Stage 4C Plus and are
never redefined here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin import stage3_data as s3d
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4c_protocol as s4c
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_protocol as s4cp
from ac_twin.stage1_data import sha256_file
from ac_twin.stage2_protocol import make_seed_schedule
from ac_twin.stage3r_protocol import write_once_json
from ac_twin.stage4_protocol import (
    CF_ACTIONS,
    CF_DURATIONS,
    EXPERT_ROUTER_SEEDS,
    PRIMARY_CONSTANT_W,
    RESPONSE_PREDICTOR_SEEDS,
)

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = "AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2"
PURPOSE_DEV = "stage4cr2_merge_multiaction_dev"

# --- environment / tensor contract (identical to Stage 4C, plan section 3) -----------
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

# --- task / objective contract (REUSED bitwise, never redefined; plan sections 3-4) ---
ACTION_PLANS = s4co.ACTION_PLANS
EGO_SLOT = s4co.EGO_SLOT
N_ACTIONS = len(ACTION_PLANS)
N_COMPETITORS = N_ACTIONS - 1  # nine relative-comparison tokens per group
CONSEQUENCE_WEIGHT_INTERACTION = s4co.WEIGHT_INTERACTION
CONSEQUENCE_WEIGHT_EFFICIENCY = s4co.WEIGHT_EFFICIENCY
INTERACTION_LONGITUDINAL_SCALE = s4co.INTERACTION_LONGITUDINAL_SCALE
INTERACTION_LATERAL_SCALE = s4co.INTERACTION_LATERAL_SCALE
REGRET_CLAMP_TOL = s4co.REGRET_CLAMP_TOL

# --- fixed feature contract (REUSED from Stage 4C Plus, plan section 4.1) --------------
TOKEN_DIM = s4cpf.TOKEN_DIM      # 34 (20 continuous + 14 action-semantics one-hot)
GLOBAL_DIM = s4cpf.GLOBAL_DIM    # 9

# --- unique fixed model (plan section 4.2) ---------------------------------------------
ARCH_DIRECT = "SAME_INFORMATION_DIRECT"
DIRECT_FLAT_DIM = N_COMPETITORS * TOKEN_DIM + GLOBAL_DIM  # 315
DIRECT_HIDDEN_WIDTH = 43
DROPOUT = 0.10
LAMBDA_CLS = 0.25
EXPECTED_TRAINABLE_PARAMETERS = 15740  # EXACT; preflight fails otherwise

# --- development sizing (plan section 5.1) ---------------------------------------------
DEV_TRAIN_GROUPS = 5000
DEV_VALIDATION_GROUPS = 1500
DEV_SCREEN_GROUPS = 8000
SMOKE_GROUPS = 12

# --- frozen group seed schedule (plan section 5.2, v0.2) --------------------------------
DEV_SEED_ENTROPY = 2026092880
DEV_SCHEDULE_SIZE = 60000
TRAIN_SEGMENT = (0, 24000)
VALIDATION_SEGMENT = (24000, 32000)
SCREEN_SEGMENT = (32000, 60000)
SMOKE_SEED_ENTROPY = 2026092700
SEED_LOW = 1
SEED_HIGH_INCLUSIVE = 2147483646  # 2**31 - 2

# Prior-stage schedules reserved by contract even though never persisted/used.
STAGE4C_CONFIRM_SEED_ENTROPY = s4c.CONFIRM_SEED_ENTROPY
STAGE4C_CONFIRM_SCHEDULE_SIZE = s4c.CONFIRM_SCHEDULE_SIZE
STAGE4CP_DEV_SEED_ENTROPY = s4cp.DEV_SEED_ENTROPY
STAGE4CP_DEV_SCHEDULE_SIZE = s4cp.DEV_SCHEDULE_SIZE
STAGE4CP_CONFIRM_SEED_ENTROPY = s4cp.CONFIRM_SEED_ENTROPY
STAGE4CP_CONFIRM_SCHEDULE_SIZE = s4cp.CONFIRM_SCHEDULE_SIZE

# --- frozen training / statistics (plan sections 7, 8, 9) -------------------------------
MODEL_SEEDS = (8101, 8202, 8303, 8404, 8505)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 128
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 12
INTERNAL_MONITOR_FRAC = 0.10
BOOTSTRAP_ENTROPY = 2026092702
BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE_SCREEN = 0.80

# --- frozen labels / gates (plan sections 3.3, 7.1, 8.1, 8.2, 9) -------------------------
HIGH_REGRET_THRESHOLD = 0.01
TRAIN_MIN_HIGH_REGRET_GROUPS = 750
TRAIN_MIN_NONZERO_REGRET_GROUPS = 1250
VALIDATION_I_AURC_MIN = 0.05
VALIDATION_I_REGRET80_MIN = 0.03
VALIDATION_MIN_POSITIVE_SEEDS = 4
GATE_TRIM_FRACTIONS = (0.01, 0.05)
GATE_MARGIN_SPEARMAN_CEILING = 0.995
BASELINE_TIE_TOLERANCE = 1e-12
GATE_STRATA_I_AURC_FLOOR = -0.10
STRATUM_IMBALANCE_FRAC = 0.20
GATE_TASK_MIN_HIGH_REGRET_FRAC = 0.15
GATE_TASK_REGRET_EPS = 0.01
GATE_TASK_MIN_DISTINCT_ORACLE = 3
GATE_TASK_MIN_ORACLE_FRAC = 0.05
GATE_TASK_MIN_STRATUM_COUNT = 800

# --- confirmation power diagnostic (plan section 10; never creates confirm resources) ----
CANDIDATE_NS = (2000, 3000, 4000, 5000)
POWER_SHRINK_SCENARIOS = (1.0, 0.75, 0.5)
POWER_SELECTION_SCENARIO = 0.75
POWER_TARGET = 0.80
POWER_MONTE_CARLO = 10000

# --- frozen Stage 4A carry-over -----------------------------------------------------------
TOP_ERROR_THRESHOLD = s4c.TOP_ERROR_THRESHOLD  # inherited; NOT recomputed

# --- mechanical terminal states (plan section 11, verbatim) --------------------------------
STATUS_INVALID = "STAGE4CR2_INVALID_ANALYSIS"
STATUS_TRAIN_SUPPORT_INSUFFICIENT = "STAGE4CR2_TRAIN_SUPPORT_INSUFFICIENT"
STATUS_MODEL_DEV_INSUFFICIENT = "STAGE4CR2_MODEL_DEV_INSUFFICIENT_AND_LINE_CLOSED"
STATUS_STRATUM_IMBALANCED = "STAGE4CR2_STRATUM_IMBALANCED"
STATUS_TASK_DEGENERATE = "STAGE4CR2_TASK_DEGENERATE"
STATUS_MODEL_INSUFFICIENT = "INCONCLUSIVE_STAGE4CR2_DECISION_MODEL_INSUFFICIENT"
STATUS_LINE_CLOSED = "STAGE4CR2_DECISION_RISK_LINE_CLOSED"
STATUS_UNDERPOWERED = "STAGE4CR2_CONFIRMATION_UNDERPOWERED"
STATUS_DEV_READY = "STAGE4CR2_DEV_READY_FOR_CONFIRMATION"

# Standard baselines (plan section 4.4). STRONGEST_STANDARD_BASELINE is chosen ONLY from
# the first three by lowest validation AURC_REGRET, ties at 1e-12 broken in this order;
# COMBINED_TOP2_MAX is historical-only and CONSTANT_RANDOM curve-reference-only.
BASELINE_MARGIN = "MARGIN_ONLY"
BASELINE_SEED_VOTE_ENTROPY = "SEED_VOTE_ENTROPY"
BASELINE_SEED_MARGIN_STD = "SEED_MARGIN_STD"
BASELINE_COMBINED_TOP2_MAX = "COMBINED_TOP2_MAX"
BASELINE_CONSTANT_RANDOM = "CONSTANT_RANDOM"
STRONGEST_BASELINE_CANDIDATES = (
    BASELINE_MARGIN, BASELINE_SEED_VOTE_ENTROPY, BASELINE_SEED_MARGIN_STD)

# New-code files whose SHA256 bind the full R2 chain at lock time (relative to code/).
# Only files that exist are hashed; later tasks extend the registry before locking.
CODE_FILES = (
    "run_stage4cr2.py",
    "collect_stage4cr2_merge_multiaction.py",
    "ac_twin/stage4cr2_protocol.py",
    "ac_twin/stage4cr2_inference.py",
    "ac_twin/stage4cr2_training.py",
    "ac_twin/stage4cr2_selection.py",
    "ac_twin/stage4cr2_statistics.py",
    "ac_twin/stage4cr2_power.py",
)
# Code files bound at protocol-lock time (Task 1 scope): the R2 protocol module itself.
PROTOCOL_CODE_FILES = (
    "ac_twin/stage4cr2_protocol.py",
)
# Frozen Stage 4C / 4C Plus chain modules reused read-only; their hashes bind the chain.
REUSED_FROZEN_CODE_FILES = (
    "ac_twin/stage4c_objective.py",
    "ac_twin/stage4c_protocol.py",
    "ac_twin/stage4cp_protocol.py",
    "ac_twin/stage4cp_features.py",
    "ac_twin/stage4cp_models.py",
)

# Frozen sources listed in plan section 2.1 (relative to repo root), hashed into the
# lock inventory; entries with a pinned hash in ``s4c.STAGE4_FROZEN`` are verified.
PLAN21_FROZEN_SOURCES = (
    "plans/2026-09-16_AC-Twin_Stage4CPlus决策感知比较风险实施计划_v0.1.md",
    "experiments/stage4_reliability/stage4c_plus/phase0/phase0_report.json",
    "experiments/stage4_reliability/stage4c_plus/phase0/oof_predictions.npz",
    "experiments/stage4_reliability/stage4c_plus/phase0/fold_assignment.npy",
    "experiments/stage4_reliability/stage4c/development/decision_rule_receipt.json",
    "experiments/stage4_reliability/stage4b/protocol/protocol_lock.json",
    "experiments/stage3_routing/stage3r/scaler.json",
    "experiments/stage3_routing/stage3b/input_binding_receipt.json",
    "code/ac_twin/stage4c_objective.py",
    "code/ac_twin/stage4cp_features.py",
    "code/ac_twin/stage4cp_models.py",
)

# Forbidden source path tokens (inherited from Stage 4C, plan section 2.3).
_FORBIDDEN_DATA_TOKENS = s4c._FORBIDDEN_DATA_TOKENS

# Stage 4C Plus Phase 0 products: read-only registration evidence, never R2 roles.
_STAGE4CP_PHASE0_MARKERS = (
    "stage4c_plus",
    "stage4cp_merge_multiaction",
    "phase0_report",
    "oof_predictions",
    "fold_assignment",
)

# R2 data roles; there is no confirm role in this plan.
R2_DATA_ROLES = (
    "development_train", "development_validation", "development_screen",
    "smoke", "power",
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
    if PURPOSE_DEV in logical or "stage4c_r2" in logical:
        return False  # R2's own create-once outputs are never Stage 4C products
    return s4cp.is_stage4c_screen(path)


def is_stage4c_selection(path: str | Path) -> bool:
    """True if ``path`` is an old Stage 4C 500-group selection product (evidence only)."""
    logical = _normalised(path)
    if PURPOSE_DEV in logical or "stage4c_r2" in logical:
        return False
    return s4cp.is_stage4c_selection(path)


def is_stage4cp_phase0(path: str | Path) -> bool:
    """True if ``path`` is any Stage 4C Plus Phase 0 product (registration evidence only)."""
    logical = _normalised(path)
    return any(marker in logical for marker in _STAGE4CP_PHASE0_MARKERS)


def forbidden_source_sha256() -> dict[str, str]:
    """SHA256 of every forbidden artifact present in this workspace (content-level guard)."""
    forbidden: dict[str, str] = {}
    candidates = [
        ROOT / "experiments/stage4_reliability/stage4c/development" / name
        for name in (
            "predictions_development_screen.npz",
            "risk_scores_development_screen.npz",
            "screen_group_quantities.npz",
            "development_screen_report.json",
        )
    ]
    candidates += [
        ROOT / "experiments/stage4_reliability/stage4c_plus/phase0" / name
        for name in ("phase0_report.json", "oof_predictions.npz", "fold_assignment.npy")
    ]
    for file_path in candidates:
        if file_path.is_file():
            forbidden[str(file_path)] = sha256_file(file_path)
    return forbidden


def assert_allowed_stage4cr2_data(
    path: str | Path,
    *,
    role: str | None = None,
    file_sha256: str | None = None,
) -> None:
    """Raise ``ValueError`` if ``path`` is a forbidden Stage 4C-R2 input.

    Refusal is three-layer (plan section 2.3): normalised absolute path, logical role
    and content SHA256. The only legal R2 data lives under this stage's own purpose dir;
    the Stage 4C screen (permanent read-only), the old 500-group selection (historical
    design evidence only), every Stage 4C Plus Phase 0 product, all sealed Stage 2/3
    sets and all Stage 4A/4B dev/confirm products are refused. There is no Phase 0 in
    R2, so no ``allow_phase0_selection`` escape hatch exists here.
    """
    s3d.assert_not_sealed_stage2(path, file_sha256=file_sha256)
    logical = _normalised(path)
    for token in _FORBIDDEN_DATA_TOKENS:
        for segment in logical.split("/"):
            if segment.startswith(token):
                raise ValueError(
                    f"forbidden Stage 4C-R2 data source (segment {segment!r}): {path}")
    if is_stage4c_screen(path):
        raise ValueError(f"Stage 4C screen is permanently read-only: {path}")
    if is_stage4c_selection(path):
        raise ValueError(
            f"old Stage 4C selection is historical design evidence only, "
            f"never an R2 role input: {path}")
    if is_stage4cp_phase0(path):
        raise ValueError(
            f"Stage 4C Plus Phase 0 products never enter R2 roles: {path}")
    if role is not None and role not in R2_DATA_ROLES:
        raise ValueError(f"unknown Stage 4C-R2 data role: {role!r}")
    if file_sha256 is not None and file_sha256 in set(forbidden_source_sha256().values()):
        raise ValueError(
            f"forbidden Stage 4C/4C Plus artifact content (SHA256 match): {path}")
    if PURPOSE_DEV not in logical:
        raise ValueError(f"not an allowed Stage 4C-R2 data source: {path}")


# --- group seed schedule -----------------------------------------------------------


def make_group_seed_schedule(entropy: int, size: int) -> np.ndarray:
    """Unique positive int64 group seeds for the single frozen R2 contract.

    Only (``2026092880`` / 60,000) is legal (plan v0.2 section 5.2). Reuses the frozen
    Stage 2 generator so the schedule is internally unique and in-range.
    """
    entropy = int(entropy)
    size = int(size)
    if (entropy, size) != (DEV_SEED_ENTROPY, DEV_SCHEDULE_SIZE):
        raise ValueError(
            f"Stage 4C-R2 group seed (entropy, size) must be "
            f"({DEV_SEED_ENTROPY}, {DEV_SCHEDULE_SIZE}); got ({entropy}, {size})")
    schedule = make_seed_schedule(entropy, size=size)
    if schedule.min() < SEED_LOW or schedule.max() > SEED_HIGH_INCLUSIVE:
        raise ValueError("Stage 4C-R2 group seed schedule out of range")
    return schedule.astype(np.int64)


def build_seed_schedule(config: Mapping[str, Any]) -> np.ndarray:
    """Validate ``config`` and build the frozen 60,000-seed candidate schedule."""
    checked = validate_config(config)
    return make_group_seed_schedule(
        int(checked["dev_seed_entropy"]), int(checked["dev_schedule_size"]))


def _schedule_segments() -> dict[str, tuple[int, int]]:
    """Development roles bind to disjoint fixed segments of the schedule (plan 5.2)."""
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


# --- global accepted+reserved seed inventory / zero-overlap audit ----------------------


@dataclass(frozen=True)
class SeedAudit:
    """Result of the global accepted+reserved seed overlap audit (plan section 5.2)."""
    n_schedule: int
    n_inventory: int
    overlap: tuple[int, ...]

    @property
    def zero_overlap(self) -> bool:
        return len(self.overlap) == 0


def scan_historical_group_seeds(workspace_roots: Sequence[Path]) -> np.ndarray:
    """Union of every historical *accepted* seed AND every reserved candidate pool.

    Scope (plan section 5.2): accepted episode/group seeds across NPZ / receipts /
    manifests, PLUS every persisted ``*seed_schedule*.npy`` candidate pool, PLUS the
    never-persisted Stage 4C confirm schedule and the reserved-but-unused Stage 4C Plus
    dev/confirm schedules regenerated by contract. This stage's own products are
    excluded so a resume does not collide with itself; Stage 4C Plus's own products are
    also excluded (Phase 0 reused the old 500-group selection seeds, which are already
    counted from the Stage 4C data, and no 4C Plus data was ever collected).
    """
    collected: set[int] = set()
    for root in workspace_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob("*")):
            normalized = _normalised(file_path)
            if PURPOSE_DEV in normalized or "stage4c_r2" in normalized:
                continue  # this stage's own products
            if "stage4c_plus" in normalized or "stage4cp_merge_multiaction" in normalized:
                continue  # 4C Plus collected nothing new; its reserved pools are added below
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
    # Reserve the never-persisted / never-used prior-stage schedules by contract.
    for entropy, size, maker in (
        (STAGE4C_CONFIRM_SEED_ENTROPY, STAGE4C_CONFIRM_SCHEDULE_SIZE,
         s4c.make_group_seed_schedule),
        (STAGE4CP_DEV_SEED_ENTROPY, STAGE4CP_DEV_SCHEDULE_SIZE,
         s4cp.make_group_seed_schedule),
        (STAGE4CP_CONFIRM_SEED_ENTROPY, STAGE4CP_CONFIRM_SCHEDULE_SIZE,
         s4cp.make_group_seed_schedule),
    ):
        collected.update(int(v) for v in maker(entropy, size))
    if not collected:
        raise ValueError("no historical episode seeds found in the workspace")
    return np.array(sorted(collected), dtype=np.int64)


def audit_global_seed_overlap(schedule: np.ndarray, roots: Sequence[Path]) -> SeedAudit:
    """Audit ``schedule`` against the global accepted+reserved seed inventory.

    Returns a :class:`SeedAudit`; does not raise on overlap — call
    :func:`assert_seed_audit_clean` (or let the lock creation fail closed). Any overlap
    is a plan-5.2 hard stop requiring a new plan version, never a seed substitution.
    """
    schedule = np.asarray(schedule, dtype=np.int64)
    if schedule.size == 0:
        raise ValueError("seed schedule must be non-empty")
    if int(schedule.min()) < SEED_LOW or int(schedule.max()) > SEED_HIGH_INCLUSIVE:
        raise ValueError("seed schedule out of range")
    if np.unique(schedule).size != schedule.size:
        raise ValueError("seed schedule contains duplicate seeds")
    inventory = scan_historical_group_seeds(roots)
    overlap = np.intersect1d(schedule, inventory)
    return SeedAudit(
        n_schedule=int(schedule.size),
        n_inventory=int(inventory.size),
        overlap=tuple(int(v) for v in overlap),
    )


def assert_seed_audit_clean(audit: SeedAudit) -> None:
    """Raise unless the audit proves global zero overlap."""
    if not audit.zero_overlap:
        raise ValueError(
            f"seed schedule overlaps {len(audit.overlap)} historical/reserved seed(s) "
            f"{list(audit.overlap[:10])}; plan section 5.2 requires a new plan version")


# --- inference mask (REUSED bit-identically from Stage 4C) ------------------------------

build_inference_mask = s4c.build_inference_mask
materialize_action_plan = s4c.materialize_action_plan


# --- config validation -----------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 4C-R2 config; raise before any artifact is produced."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("scene") != "merge" or checked.get("env_id") != ENV_ID:
        raise ValueError("Stage 4C-R2 is merge-v0 only")
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
        ("smoke_groups", SMOKE_GROUPS),
        ("dev_seed_entropy", DEV_SEED_ENTROPY),
        ("dev_schedule_size", DEV_SCHEDULE_SIZE),
        ("smoke_seed_entropy", SMOKE_SEED_ENTROPY),
        ("bootstrap_entropy", BOOTSTRAP_ENTROPY),
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("n_competitors", N_COMPETITORS), ("token_dim", TOKEN_DIM),
        ("global_dim", GLOBAL_DIM),
        ("direct_hidden_width", DIRECT_HIDDEN_WIDTH),
        ("expected_trainable_parameters", EXPECTED_TRAINABLE_PARAMETERS),
        ("train_batch_size", TRAIN_BATCH_SIZE),
        ("max_epochs", MAX_EPOCHS), ("early_stop_patience", EARLY_STOP_PATIENCE),
        ("gate_task_min_distinct_oracle", GATE_TASK_MIN_DISTINCT_ORACLE),
        ("gate_task_min_stratum_count", GATE_TASK_MIN_STRATUM_COUNT),
        ("train_min_high_regret_groups", TRAIN_MIN_HIGH_REGRET_GROUPS),
        ("train_min_nonzero_regret_groups", TRAIN_MIN_NONZERO_REGRET_GROUPS),
        ("validation_min_positive_seeds", VALIDATION_MIN_POSITIVE_SEEDS),
        ("power_monte_carlo", POWER_MONTE_CARLO),
    ):
        if int(checked.get(key, -1)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    for key, expected in (
        ("dropout", DROPOUT), ("lambda_cls", LAMBDA_CLS),
        ("learning_rate", LEARNING_RATE), ("weight_decay", WEIGHT_DECAY),
        ("internal_monitor_frac", INTERNAL_MONITOR_FRAC),
        ("confidence_screen", CONFIDENCE_SCREEN),
        ("high_regret_threshold", HIGH_REGRET_THRESHOLD),
        ("consequence_weight_interaction", CONSEQUENCE_WEIGHT_INTERACTION),
        ("consequence_weight_efficiency", CONSEQUENCE_WEIGHT_EFFICIENCY),
        ("interaction_longitudinal_scale", INTERACTION_LONGITUDINAL_SCALE),
        ("interaction_lateral_scale", INTERACTION_LATERAL_SCALE),
        ("regret_clamp_tol", REGRET_CLAMP_TOL),
        ("validation_i_aurc_min", VALIDATION_I_AURC_MIN),
        ("validation_i_regret80_min", VALIDATION_I_REGRET80_MIN),
        ("gate_margin_spearman_ceiling", GATE_MARGIN_SPEARMAN_CEILING),
        ("baseline_tie_tolerance", BASELINE_TIE_TOLERANCE),
        ("gate_strata_i_aurc_floor", GATE_STRATA_I_AURC_FLOOR),
        ("stratum_imbalance_frac", STRATUM_IMBALANCE_FRAC),
        ("gate_task_min_high_regret_frac", GATE_TASK_MIN_HIGH_REGRET_FRAC),
        ("gate_task_regret_eps", GATE_TASK_REGRET_EPS),
        ("gate_task_min_oracle_frac", GATE_TASK_MIN_ORACLE_FRAC),
        ("power_target", POWER_TARGET),
        ("power_selection_scenario", POWER_SELECTION_SCENARIO),
        ("top_error_threshold", TOP_ERROR_THRESHOLD),
        ("primary_constant_w", PRIMARY_CONSTANT_W),
    ):
        if float(checked.get(key, -9.0)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    if [int(s) for s in checked.get("model_seeds", [])] != list(MODEL_SEEDS):
        raise ValueError(f"model_seeds must be {list(MODEL_SEEDS)}")
    if [int(x) for x in checked.get("train_segment", [])] != list(TRAIN_SEGMENT):
        raise ValueError(f"train_segment must be {list(TRAIN_SEGMENT)}")
    if [int(x) for x in checked.get("validation_segment", [])] != list(VALIDATION_SEGMENT):
        raise ValueError(f"validation_segment must be {list(VALIDATION_SEGMENT)}")
    if [int(x) for x in checked.get("screen_segment", [])] != list(SCREEN_SEGMENT):
        raise ValueError(f"screen_segment must be {list(SCREEN_SEGMENT)}")
    if [float(x) for x in checked.get("gate_trim_fractions", [])] != list(GATE_TRIM_FRACTIONS):
        raise ValueError(f"gate_trim_fractions must be {list(GATE_TRIM_FRACTIONS)}")
    if [int(n) for n in checked.get("candidate_ns", [])] != list(CANDIDATE_NS):
        raise ValueError(f"candidate_ns must be {list(CANDIDATE_NS)}")
    if [float(x) for x in checked.get("power_shrink_scenarios", [])] != list(POWER_SHRINK_SCENARIOS):
        raise ValueError(f"power_shrink_scenarios must be {list(POWER_SHRINK_SCENARIOS)}")
    if list(checked.get("response_predictor_seeds", [])) != list(RESPONSE_PREDICTOR_SEEDS):
        raise ValueError("response_predictor_seeds drifted from the frozen five")
    if list(checked.get("expert_router_seeds", [])) != list(EXPERT_ROUTER_SEEDS):
        raise ValueError("expert_router_seeds drifted from the frozen five pairs")
    if checked.get("architecture", ARCH_DIRECT) != ARCH_DIRECT:
        raise ValueError(f"Stage 4C-R2 trains only {ARCH_DIRECT}")
    return checked


# --- frozen-source inventory (plan section 2.1) -----------------------------------------


def build_lock_inventory() -> dict[str, dict[str, str]]:
    """Resolve + SHA256 every plan-2.1 frozen source; verify pinned hashes on drift."""
    inventory: dict[str, dict[str, str]] = {}
    stage4_pinned = {rel: sha for _, (rel, sha) in s4c.STAGE4_FROZEN.items()}
    for rel in PLAN21_FROZEN_SOURCES:
        path = _resolve(rel)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen plan-2.1 source: {path}")
        digest = sha256_file(path)
        if rel in stage4_pinned and stage4_pinned[rel] != digest:
            raise ValueError(
                f"frozen source hash drift: {rel}; expected {stage4_pinned[rel]}, got {digest}")
        inventory[rel] = {"path": str(path), "sha256": digest}
    return inventory


def _code_hashes(files: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in files:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered Stage 4C-R2 code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


def protocol_code_hashes() -> dict[str, str]:
    """Hashes bound at protocol-lock time: the R2 protocol module + reused frozen chain."""
    return _code_hashes(tuple(PROTOCOL_CODE_FILES) + tuple(REUSED_FROZEN_CODE_FILES))


def _config_signature(config: Mapping[str, Any]) -> str:
    from ac_twin.checkpointing import canonical_fingerprint
    return canonical_fingerprint(json.loads(json.dumps(config)))


# --- development protocol lock -----------------------------------------------------------


def _verify_inventory_entries(inventory: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Re-hash every ``{name: {path, sha256}}`` entry; raise on any drift."""
    normalised: dict[str, dict[str, str]] = {}
    for name, entry in inventory.items():
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"inventoried frozen source missing: {path}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"inventoried frozen source hash drift: {name} ({path}); "
                f"expected {entry['sha256']}, got {digest}")
        normalised[str(name)] = {"path": str(path), "sha256": digest}
    return normalised


def create_development_lock(
    config: Mapping[str, Any], inventory: Mapping[str, Any]
) -> Path:
    """Create the development protocol lock + schedule + audit report exactly once.

    Fails closed: any config drift, any inventoried-source hash drift or any global
    seed overlap raises before anything is written. On success writes create-once
    ``development/seed_schedule.npy``, ``development/global_seed_audit.json`` and
    ``development/protocol_lock.json`` under the configured run dir, and returns the
    lock path. A second call raises ``FileExistsError``; a later pure-engineering fix
    must keep the old lock and build a recovery version, never overwrite in place.
    """
    checked = validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "development"
    lock_path = protocol_dir / "protocol_lock.json"
    if lock_path.exists():
        raise FileExistsError(
            f"Stage 4C-R2 development lock already exists: {lock_path}")
    frozen_inventory = _verify_inventory_entries(inventory)
    schedule = build_seed_schedule(checked)
    audit = audit_global_seed_overlap(schedule, [ROOT / "data", ROOT / "experiments"])
    assert_seed_audit_clean(audit)
    segments = _schedule_segments()
    payload = {
        "plan_id": PLAN_ID,
        "purpose_dev": PURPOSE_DEV,
        "config_signature": _config_signature(checked),
        "dev_seed_entropy": DEV_SEED_ENTROPY,
        "dev_schedule_size": DEV_SCHEDULE_SIZE,
        "dev_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "dev_train_groups": DEV_TRAIN_GROUPS,
        "dev_validation_groups": DEV_VALIDATION_GROUPS,
        "dev_screen_groups": DEV_SCREEN_GROUPS,
        "train_segment": list(segments["development_train"]),
        "validation_segment": list(segments["development_validation"]),
        "screen_segment": list(segments["development_screen"]),
        "seed_audit": {
            "n_schedule": audit.n_schedule,
            "n_inventory": audit.n_inventory,
            "overlap": list(audit.overlap),
            "zero_overlap": audit.zero_overlap,
        },
        "action_plans": [list(p) for p in ACTION_PLANS],
        "architecture": ARCH_DIRECT,
        "direct_hidden_width": DIRECT_HIDDEN_WIDTH,
        "lambda_cls": LAMBDA_CLS,
        "dropout": DROPOUT,
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "high_regret_threshold": HIGH_REGRET_THRESHOLD,
        "model_seeds": list(MODEL_SEEDS),
        "bootstrap_entropy": BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "confidence_screen": CONFIDENCE_SCREEN,
        "strongest_baseline_candidates": list(STRONGEST_BASELINE_CANDIDATES),
        "baseline_tie_tolerance": BASELINE_TIE_TOLERANCE,
        "terminal_states": [
            STATUS_INVALID, STATUS_TRAIN_SUPPORT_INSUFFICIENT,
            STATUS_MODEL_DEV_INSUFFICIENT, STATUS_STRATUM_IMBALANCED,
            STATUS_TASK_DEGENERATE, STATUS_MODEL_INSUFFICIENT,
            STATUS_LINE_CLOSED, STATUS_UNDERPOWERED, STATUS_DEV_READY,
        ],
        "code_sha256": protocol_code_hashes(),
        "frozen_source_inventory": frozen_inventory,
    }
    protocol_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = protocol_dir / "seed_schedule.npy"
    tmp = schedule_path.with_name(schedule_path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, schedule)
    Path(tmp).replace(schedule_path)
    write_once_json(protocol_dir / "global_seed_audit.json", payload["seed_audit"])
    write_once_json(lock_path, payload)
    return lock_path


def verify_development_lock(lock_path: Path) -> None:
    """Re-derive every lock binding and raise ``ValueError`` on any tampering."""
    lock_path = Path(lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("plan_id") != PLAN_ID:
        raise ValueError(f"lock plan_id must be {PLAN_ID!r}")
    schedule = make_group_seed_schedule(
        int(payload["dev_seed_entropy"]), int(payload["dev_schedule_size"]))
    if hashlib.sha256(schedule.tobytes()).hexdigest() != payload["dev_schedule_sha256"]:
        raise ValueError("lock seed schedule hash mismatch")
    for key, segment in (
        ("train_segment", TRAIN_SEGMENT),
        ("validation_segment", VALIDATION_SEGMENT),
        ("screen_segment", SCREEN_SEGMENT),
    ):
        if [int(x) for x in payload[key]] != list(segment):
            raise ValueError(f"lock {key} drifted from {list(segment)}")
    schedule_path = lock_path.parent / "seed_schedule.npy"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"lock seed schedule missing: {schedule_path}")
    if not np.array_equal(np.load(schedule_path), schedule):
        raise ValueError("persisted seed_schedule.npy does not match the frozen contract")
    if protocol_code_hashes() != dict(payload["code_sha256"]):
        raise ValueError("locked code SHA256 set mismatch (code drift or tampering)")
    _verify_inventory_entries(payload["frozen_source_inventory"])
    audit = audit_global_seed_overlap(schedule, [ROOT / "data", ROOT / "experiments"])
    assert_seed_audit_clean(audit)


__all__ = [
    "ACTION_PLANS",
    "ARCH_DIRECT",
    "BASELINE_COMBINED_TOP2_MAX",
    "BASELINE_CONSTANT_RANDOM",
    "BASELINE_MARGIN",
    "BASELINE_SEED_MARGIN_STD",
    "BASELINE_SEED_VOTE_ENTROPY",
    "BASELINE_TIE_TOLERANCE",
    "BOOTSTRAP_ENTROPY",
    "BOOTSTRAP_RESAMPLES",
    "CANDIDATE_NS",
    "CF_ACTIONS",
    "CF_DURATIONS",
    "CODE_FILES",
    "CONFIDENCE_SCREEN",
    "CONSEQUENCE_WEIGHT_EFFICIENCY",
    "CONSEQUENCE_WEIGHT_INTERACTION",
    "DEV_SCHEDULE_SIZE",
    "DEV_SCREEN_GROUPS",
    "DEV_SEED_ENTROPY",
    "DEV_TRAIN_GROUPS",
    "DEV_VALIDATION_GROUPS",
    "DIRECT_FLAT_DIM",
    "DIRECT_HIDDEN_WIDTH",
    "DROPOUT",
    "EARLY_STOP_PATIENCE",
    "EGO_SLOT",
    "ENV_DURATION",
    "ENV_ID",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "EXPERT_ROUTER_SEEDS",
    "GATE_MARGIN_SPEARMAN_CEILING",
    "GATE_STRATA_I_AURC_FLOOR",
    "GATE_TASK_MIN_DISTINCT_ORACLE",
    "GATE_TASK_MIN_HIGH_REGRET_FRAC",
    "GATE_TASK_MIN_ORACLE_FRAC",
    "GATE_TASK_MIN_STRATUM_COUNT",
    "GATE_TASK_REGRET_EPS",
    "GATE_TRIM_FRACTIONS",
    "GLOBAL_DIM",
    "HIGH_REGRET_THRESHOLD",
    "HISTORY_LEN",
    "HORIZON",
    "INTERACTION_LATERAL_SCALE",
    "INTERACTION_LONGITUDINAL_SCALE",
    "INTERNAL_MONITOR_FRAC",
    "LAMBDA_CLS",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "MODEL_SEEDS",
    "N_ACTIONS",
    "N_COMPETITORS",
    "N_OBJECT_FEATURES",
    "OBJECT_CAPACITY",
    "OBSERVED_VEHICLES_COUNT",
    "PLAN21_FROZEN_SOURCES",
    "PLAN_ID",
    "POLICY_FREQUENCY",
    "POWER_MONTE_CARLO",
    "POWER_SELECTION_SCENARIO",
    "POWER_SHRINK_SCENARIOS",
    "POWER_TARGET",
    "PRIMARY_CONSTANT_W",
    "PROTOCOL_CODE_FILES",
    "PURPOSE_DEV",
    "R2_DATA_ROLES",
    "REGRET_CLAMP_TOL",
    "RESPONSE_PREDICTOR_SEEDS",
    "REUSED_FROZEN_CODE_FILES",
    "SCREEN_SEGMENT",
    "SEED_HIGH_INCLUSIVE",
    "SEED_LOW",
    "SIMULATION_FREQUENCY",
    "SMOKE_GROUPS",
    "SMOKE_SEED_ENTROPY",
    "STATUS_DEV_READY",
    "STATUS_INVALID",
    "STATUS_LINE_CLOSED",
    "STATUS_MODEL_DEV_INSUFFICIENT",
    "STATUS_MODEL_INSUFFICIENT",
    "STATUS_STRATUM_IMBALANCED",
    "STATUS_TASK_DEGENERATE",
    "STATUS_TRAIN_SUPPORT_INSUFFICIENT",
    "STATUS_UNDERPOWERED",
    "STRATUM_IMBALANCE_FRAC",
    "STRONGEST_BASELINE_CANDIDATES",
    "SeedAudit",
    "TOKEN_DIM",
    "TOP_ERROR_THRESHOLD",
    "TRAIN_BATCH_SIZE",
    "TRAIN_MIN_HIGH_REGRET_GROUPS",
    "TRAIN_MIN_NONZERO_REGRET_GROUPS",
    "TRAIN_SEGMENT",
    "T_STAR_HIGH",
    "T_STAR_LOW",
    "VALIDATION_I_AURC_MIN",
    "VALIDATION_I_REGRET80_MIN",
    "VALIDATION_MIN_POSITIVE_SEEDS",
    "VALIDATION_SEGMENT",
    "VEHICLES_COUNT",
    "WEIGHT_DECAY",
    "assert_allowed_stage4cr2_data",
    "assert_seed_audit_clean",
    "audit_global_seed_overlap",
    "build_inference_mask",
    "build_lock_inventory",
    "build_seed_schedule",
    "create_development_lock",
    "forbidden_source_sha256",
    "is_stage4c_screen",
    "is_stage4c_selection",
    "is_stage4cp_phase0",
    "make_group_seed_schedule",
    "materialize_action_plan",
    "protocol_code_hashes",
    "role_segment",
    "scan_historical_group_seeds",
    "validate_config",
    "verify_development_lock",
    "write_once_json",
]
