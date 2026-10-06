"""Stage 4D D0/D1 protocol: config, source chain, seed domains, lock guard, state machine.

Freeze layer for the Merge model-mismatch OOD confirmation (plan
``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0``). D0-H freezes one of three
pre-registered OOD profiles from pilots that never load a Direct checkpoint; D1 scores the
five frozen Stage 4C-R2 Direct checkpoints once on a fresh OOD ten-action-group set. This
module retrains NOTHING, refits NOTHING, re-judges NOTHING historical, and never reads the
sealed confirm truth.

Scientific isolation rules encoded here (plan §2, §4, §6, §7):

* Every Stage 4C-R2 / confirm authoritative source (plan §2.1, ten files) is hash-verified
  against the pinned SHA256 table before any use; the R2 train NPZ binding recorded by the
  pinned train receipt is re-checked against plan §2.1. Any drift fails closed.
* The five candidate seed schedules are generated EXACTLY by the frozen Stage 2 primitive
  ``stage2_protocol.make_seed_schedule`` and verified against the plan-§6.2 raw-bytes
  SHA256; any non-approved (entropy, size) pair is refused. The D0 classifier-bootstrap,
  D0 power and D1 bootstrap entropies are registered as RESERVED (controller ruling 2)
  even though they are not collection schedules.
* The global seed audit treats R2 AND the Stage 4C-R2 confirm stage as HISTORY: both their
  full reserved domains are counted, and the filesystem scan opens every prior-stage
  npz/json/jsonl EXCEPT sealed ``confirmation_truth`` files (whose accepted seeds are a
  subset of the persisted candidate schedules, so zero-overlap evidence stays complete).
  Only this stage's own namespace is excluded (resume safety). Any overlap is a plan-§6.2
  hard stop requiring a whole-domain entropy revision, never single-seed substitution.

Task 1 scope: this module builds the config/source/code/seed/lock/state-machine PRIMITIVES
and guards. The formal ``protocol_lock.json`` is created in Task 4 (after the D1 power
analysis freezes n); :func:`verify_protocol_lock` here is the verification primitive that
lock will be checked against.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin import stage2_protocol as s2
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4c_protocol as s4c
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4cr2_protocol as s4cr2
from ac_twin import stage4cr2_confirm_protocol as confirm  # frozen prior stage, read-only
from ac_twin import stage4d_ood_env as ood
from ac_twin.stage1_data import sha256_file
from ac_twin.stage2_protocol import make_seed_schedule
from ac_twin.stage3r_protocol import write_once_json
from ac_twin.stage4_protocol import (
    CF_ACTIONS,
    CF_DURATIONS,
    EXPERT_ROUTER_SEEDS,
    RESPONSE_PREDICTOR_SEEDS,
)

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = "AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0"
STAGE_NAMESPACE = "stage4d_ood"
PURPOSE = "stage4d_merge_ood"  # data namespace

CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4d_ood/config.json")

# --- environment / tensor contract (REUSED bitwise from Stage 4C; plan §3.1) ----------------
ENV_ID = s4c.ENV_ID
HISTORY_LEN = s4c.HISTORY_LEN
HORIZON = s4c.HORIZON
FACTUAL_SUFFIX_STEPS = 10
T_STAR_LOW = s4c.T_STAR_LOW
T_STAR_HIGH = s4c.T_STAR_HIGH
SIMULATION_FREQUENCY = s4c.SIMULATION_FREQUENCY
POLICY_FREQUENCY = s4c.POLICY_FREQUENCY
VEHICLES_COUNT = s4c.VEHICLES_COUNT
OBSERVED_VEHICLES_COUNT = s4c.OBSERVED_VEHICLES_COUNT
ENV_DURATION = s4c.ENV_DURATION
OBJECT_CAPACITY = s4c.OBJECT_CAPACITY
N_OBJECT_FEATURES = s4c.N_OBJECT_FEATURES

# --- fixed task contract (REUSED bitwise; plan §3.1) -----------------------------------------
ACTION_PLANS = s4co.ACTION_PLANS
N_ACTIONS = 10
if len(ACTION_PLANS) != N_ACTIONS:  # fixed contract guard, not a derivation
    raise RuntimeError("frozen ACTION_PLANS must contain exactly ten plans")
IDLE_PLAN_INDEX = s4co.IDLE_PLAN_INDEX
if ACTION_PLANS[IDLE_PLAN_INDEX] != (1, 3):
    raise RuntimeError("IDLE reference must stay (IDLE,3), never (IDLE,1)")
EGO_SLOT = s4co.EGO_SLOT
#: one factual branch + ten counterfactual action-plan branches per group.
N_BRANCHES_PER_GROUP = 11
ACTION_WEIGHTS = {"idle_maintain": 0.45, "faster": 0.18, "slower": 0.18, "other": 0.095}

# --- frozen Direct model + baseline contract (REUSED bitwise from the R2 chain; §3.2) --------
ARCH_DIRECT = s4cr2.ARCH_DIRECT
MODEL_SEEDS = s4cr2.MODEL_SEEDS
N_COMPETITORS = s4cr2.N_COMPETITORS
TOKEN_DIM = s4cr2.TOKEN_DIM
GLOBAL_DIM = s4cr2.GLOBAL_DIM
DIRECT_HIDDEN_WIDTH = s4cr2.DIRECT_HIDDEN_WIDTH
DROPOUT = s4cr2.DROPOUT
EXPECTED_TRAINABLE_PARAMETERS = s4cr2.EXPECTED_TRAINABLE_PARAMETERS  # EXACTLY 15,740
PRIMARY_SCORE_DEFINITION = confirm.PRIMARY_SCORE_DEFINITION
ENSEMBLE_DEFINITION = confirm.ENSEMBLE_DEFINITION
PRIMARY_BASELINE = confirm.PRIMARY_BASELINE  # SEED_VOTE_ENTROPY (validation-frozen)
STANDARD_BASELINES = confirm.STANDARD_BASELINES
REFERENCE_BASELINES = confirm.REFERENCE_BASELINES
BASELINE_TIE_TOLERANCE = 1e-12
HIGH_REGRET_THRESHOLD = s4cr2.HIGH_REGRET_THRESHOLD  # 0.01
COVERAGE = 0.80

# --- D0/D1 seed domains (plan §6.2, verbatim raw-bytes SHA256) -------------------------------
SEED_LOW = 1
SEED_HIGH_INCLUSIVE = 2147483646  # 2**31 - 2

SMOKE_SEED_ENTROPY = 2026100099
SMOKE_SCHEDULE_SIZE = 256
SMOKE_SCHEDULE_RAW_SHA256 = (
    "5414ba4289044a3af39633c335f69a8c8880fb008b03d459640a4ea47d7f6722")
PILOT_BEHAVIOR_SEED_ENTROPY = 2026101000
PILOT_GEOMETRY_SEED_ENTROPY = 2026102001
PILOT_JOINT_SEED_ENTROPY = 2026103002
PILOT_SCHEDULE_SIZE = 6000
PILOT_BEHAVIOR_SCHEDULE_RAW_SHA256 = (
    "f3605a836186d478d60d652d434f489b771bb312c69242d28f61552bbae11430")
PILOT_GEOMETRY_SCHEDULE_RAW_SHA256 = (
    "835811b5376aff3b1acef60dfc2411f742d4b1088415536424e8b4e0af1d27b7")
PILOT_JOINT_SCHEDULE_RAW_SHA256 = (
    "fe51c90a86c67bd594107def30aa9f4c55c0489b6e9776875c321dc2d5acb0ad")
D1_CANDIDATE_SEED_ENTROPY = 2026110994
D1_CANDIDATE_SCHEDULE_SIZE = 30000
D1_CANDIDATE_SCHEDULE_RAW_SHA256 = (
    "aafbba57ecc19096c66a85e59f2a6d248dac518b7523e3924b370c50bb6a536a")

D1_BOOTSTRAP_ENTROPY = 2026110995
BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE = 0.95

#: the ONLY (entropy, size) collection contracts this stage may generate, pinned to SHA256.
_APPROVED_SCHEDULES = {
    (SMOKE_SEED_ENTROPY, SMOKE_SCHEDULE_SIZE): SMOKE_SCHEDULE_RAW_SHA256,
    (PILOT_BEHAVIOR_SEED_ENTROPY, PILOT_SCHEDULE_SIZE):
        PILOT_BEHAVIOR_SCHEDULE_RAW_SHA256,
    (PILOT_GEOMETRY_SEED_ENTROPY, PILOT_SCHEDULE_SIZE):
        PILOT_GEOMETRY_SCHEDULE_RAW_SHA256,
    (PILOT_JOINT_SEED_ENTROPY, PILOT_SCHEDULE_SIZE): PILOT_JOINT_SCHEDULE_RAW_SHA256,
    (D1_CANDIDATE_SEED_ENTROPY, D1_CANDIDATE_SCHEDULE_SIZE):
        D1_CANDIDATE_SCHEDULE_RAW_SHA256,
}

#: action-sampling RNG roots (plan §6.2, verbatim).
ACTION_STREAM_ROOTS = {
    "smoke": (SMOKE_SEED_ENTROPY, 0),
    "pilot_behavior": (PILOT_BEHAVIOR_SEED_ENTROPY, 1),
    "pilot_geometry": (PILOT_GEOMETRY_SEED_ENTROPY, 2),
    "pilot_joint": (PILOT_JOINT_SEED_ENTROPY, 3),
    "d1_confirm": (D1_CANDIDATE_SEED_ENTROPY, 4),
}
SEED_ROLES = tuple(ACTION_STREAM_ROOTS)

# --- D0 domain-shift / pilot contract (plan §5.2) ---------------------------------------------
DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES = {
    "behavior": 2026104101, "geometry": 2026104102, "joint": 2026104103}
DOMAIN_CLASSIFIER_BOOTSTRAP_RESAMPLES = 5000
DOMAIN_CLASSIFIER_CONFIDENCE = 0.80
DOMAIN_SHIFT_SMD_MIN = 0.50
DOMAIN_SHIFT_MIN_SHIFTED = 2
DOMAIN_CLASSIFIER_AUC_CI_LB_MIN = 0.70
PILOT_ACCEPTED_GROUPS = 300
PILOT_MAX_CANDIDATE_SEEDS = 6000

# --- D1 power contract (plan §6.1) -------------------------------------------------------------
POWER_ENTROPY = 2026104201
POWER_MONTE_CARLO = 10000
POWER_TARGET = 0.80
CANDIDATE_NS = (1000, 2000, 3000, 4000)
RETENTION_FACTORS = (1.0, 0.75, 0.5, 0.25)
POWER_SELECTION_RETENTION = 0.50
POWER_AND_CHAIN = {"i_aurc_min": 0.05, "i_regret80_min": 0.03, "min_positive_seeds": 4}

#: controller ruling 2: these entropies are RESERVED in the seed inventory even though they
#: are RNG-stream entropies, not collection group-seed schedules. Registering them prevents
#: any future stage from reusing the same entropy budget.
RESERVED_ENTROPIES = {
    "collection": {
        "smoke": SMOKE_SEED_ENTROPY,
        "pilot_behavior": PILOT_BEHAVIOR_SEED_ENTROPY,
        "pilot_geometry": PILOT_GEOMETRY_SEED_ENTROPY,
        "pilot_joint": PILOT_JOINT_SEED_ENTROPY,
        "d1_candidate": D1_CANDIDATE_SEED_ENTROPY,
    },
    "action_stream_roots": {role: list(root) for role, root in ACTION_STREAM_ROOTS.items()},
    "domain_classifier_bootstrap": dict(DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES),
    "power": POWER_ENTROPY,
    "d1_bootstrap": D1_BOOTSTRAP_ENTROPY,
    "behavior_assign_stream": ood.BEHAVIOR_ASSIGN_STREAM_ENTROPY,
    "geometry_stream": ood.GEOMETRY_STREAM_ENTROPY,
}

# --- Gate thresholds (plan §7.1, §7.2) ----------------------------------------------------------
GATE_TASK_OOD = {
    "min_high_regret_frac": 0.15,
    "regret_eps": 0.01,
    "min_distinct_oracle": 3,
    "min_oracle_frac": 0.05,
    "min_stratum_frac": 0.10,
    "smd_min": DOMAIN_SHIFT_SMD_MIN,
    "domain_auc_ci_lb_min": DOMAIN_CLASSIFIER_AUC_CI_LB_MIN,
}
GATE_OOD_RISK = {
    "i_aurc_min": 0.05,
    "i_regret80_min": 0.03,
    "min_positive_seed_metrics": 4,
}

# --- §2.1 frozen source chain (ten files, verbatim pinned SHA256) -------------------------------
R2_DEVELOPMENT_DIR = "experiments/stage4_reliability/stage4c_r2/development"
R2_TRAIN_COLLECTION_RECEIPT = (
    "experiments/stage4_reliability/stage4c_r2/development/train_collection_receipt.json")
STAGE4D_PINNED_SOURCES = {
    "experiments/stage4_reliability/stage4c_r2/development/protocol_lock.json":
        "c19c5bdce54fe039a10003b4ab78d1fb22040ac740ba950457c4534fc5e8aa71",
    "experiments/stage4_reliability/stage4c_r2/development/final_selection.json":
        "f135fdab973cca25a5ccb37d9c164a6cd3aaae9d5037d8a8099d33d64534b461",
    "experiments/stage4_reliability/stage4c_r2/development/model_inventory.json":
        "982ec5ea9287f6f82d20e6ea6870f690e60227be3566aea8f17a290d73e77637",
    "experiments/stage4_reliability/stage4c_r2/development/feature_schema.json":
        "4d45bb5556502df06630c20ffaea4fcacc5f818bde28854aec8c434ee9d23033",
    "experiments/stage4_reliability/stage4c_r2/development/feature_scaler.npz":
        "2f21c11ae248f5883d8e9c4c8fa7efb876ad1afb453ee7eec462509a48104194",
    "experiments/stage4_reliability/stage4c_r2/development/screen_group_quantities.npz":
        "d68a2c89d6e6827bd366ade73a37ed7e9050fcc06a444e3d533e66ba6c7ed20b",
    R2_TRAIN_COLLECTION_RECEIPT:
        "e5e7a098c8cdc4a1fe00ac27576a337b8f01bbd749f9ebc21921cdaad4f5f126",
    "experiments/stage4_reliability/stage4cr2_confirm/protocol/recovery_lock_r1.json":
        "24795b601a02d13725c45704c73248a2f560df7caa1f2d4e6adbcb852c3bacd8",
    "experiments/stage4_reliability/stage4cr2_confirm/reports/"
    "stage4cr2_confirmatory_report.json":
        "46798b2d3dac509091d70971ff2b94a556b6b3795c16882dd35290fc8514fe9b",
    "experiments/stage4_reliability/stage4cr2_confirm/reports/"
    "STAGE4CR2_CONFIRMATORY_DECISION.md":
        "4455ef156cabefad814b0b0ec60be3c27af663c2d8cedd5c9b47b77be85fc969",
}
#: the R2 train NPZ hash bound BY the pinned train receipt (plan §2.1 row 7). Recorded here;
#: the (large) NPZ itself is hashed at Task-4 lock time, not in this task.
R2_TRAIN_NPZ_SHA256 = "df240cac7686d885b114b45234a3fa1c9041af2802b68d1c25cfdfe76aac53b8"

# --- §7.1 condition-8 threshold derivation rule (controller ruling 1, fixed forever) -----------
#: The "R2 train-only initial distance threshold" is derived by ONE fixed rule from the R2
#: train NPZ's pre-intervention initial ego-to-other distances. Task 4 extracts the array and
#: evaluates this rule at lock time; the rule itself is written here and must never change.
R2_TRAIN_DISTANCE_THRESHOLD_RULE = (
    "median_of_pre_intervention_min_ego_distance_over_r2_train_groups")

# --- code inventory registry (plan §6.3, §9.1) --------------------------------------------------
#: this stage's own new-code files bound at protocol-lock time (relative to code/). Later
#: tasks (collector/statistics/runner) extend this registry before the formal lock.
PROTOCOL_CODE_FILES = (
    "ac_twin/stage4d_protocol.py",
    "ac_twin/stage4d_ood_env.py",
)
#: frozen prior-stage execution dependencies exercised by this stage's chain (read-only).
REUSED_FROZEN_CODE_FILES = (
    "ac_twin/stage1_data.py",
    "ac_twin/stage2_protocol.py",
    "ac_twin/stage2b_protocol.py",
    "ac_twin/stage3_data.py",
    "ac_twin/stage3r_protocol.py",
    "ac_twin/stage4_protocol.py",
    "ac_twin/stage4c_objective.py",
    "ac_twin/stage4c_protocol.py",
    "ac_twin/stage4cp_protocol.py",
    "ac_twin/stage4cr2_protocol.py",
    "ac_twin/stage4cr2_confirm_protocol.py",
    "ac_twin/checkpointing.py",
    "ac_twin/progress.py",
    "ac_twin/object_centric.py",
    "collect_pilot.py",
    "collect_stage2_holdout.py",
)

#: config keys whose presence would imply training/refitting; forbidden in this stage.
_FORBIDDEN_RETRAIN_KEYS = (
    "learning_rate", "max_epochs", "early_stop_patience", "weight_decay",
    "gradient_clip_norm", "optimizer", "training_seeds_to_fit", "fit_normalization",
    "fit_response_bins", "retrain", "fine_tune",
)

#: this stage's own namespace tokens — the ONLY paths excluded from the historical scan.
_OWN_NAMESPACE_TOKENS = (STAGE_NAMESPACE, PURPOSE)
#: seed-bearing keys collected from npz archives (group/episode seeds only).
_SEED_NPZ_KEYS = ("episode_seed", "episode_seeds", "group_seed", "group_seeds")

# --- mechanical / terminal states (plan §7.3, verbatim) -----------------------------------------
STATUS_INVALID = "INCONCLUSIVE_STAGE4D_OOD_INVALID"
STATUS_TASK_INSUFFICIENT = "STAGE4D_OOD_TASK_INSUFFICIENT"
STATUS_UNDERPOWERED = "STAGE4D_OOD_CONFIRMATION_UNDERPOWERED"
STATUS_TASK_DEGENERATE = "STAGE4D_OOD_TASK_DEGENERATE"
STATUS_MODEL_INSUFFICIENT = "INCONCLUSIVE_STAGE4D_OOD_DECISION_MODEL_INSUFFICIENT"
STATUS_NOT_CONFIRMED = "STAGE4D_OOD_ROBUSTNESS_NOT_CONFIRMED"
STATUS_CONFIRMED = "STAGE4D_OOD_ROBUSTNESS_CONFIRMED"

TERMINAL_STATES = (
    STATUS_TASK_INSUFFICIENT, STATUS_UNDERPOWERED, STATUS_TASK_DEGENERATE,
    STATUS_MODEL_INSUFFICIENT, STATUS_INVALID, STATUS_NOT_CONFIRMED, STATUS_CONFIRMED,
)

#: non-terminal engineering states (§10 / §12 run-all sequence).
STATUS_APPROVED = "APPROVED_FOR_AUTONOMOUS_IMPLEMENTATION"
STATUS_SOFTWARE_IN_PROGRESS = "SOFTWARE_IN_PROGRESS"
STATUS_PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
STATUS_D0_PILOTS_COLLECTED = "D0_PILOTS_COLLECTED"
STATUS_D0_PROFILE_SELECTED = "D0_PROFILE_SELECTED"
STATUS_D1_POWER_FROZEN = "D1_POWER_FROZEN"
STATUS_D1_LOCKED = "D1_LOCKED"
STATUS_D1_COLLECTED = "D1_COLLECTED"
STATUS_D1_SCORES_FROZEN = "D1_SCORES_FROZEN"

TRANSITIONS = {
    STATUS_APPROVED: {STATUS_SOFTWARE_IN_PROGRESS},
    STATUS_SOFTWARE_IN_PROGRESS: {STATUS_PREFLIGHT_PASSED},
    STATUS_PREFLIGHT_PASSED: {STATUS_D0_PILOTS_COLLECTED},
    STATUS_D0_PILOTS_COLLECTED: {STATUS_D0_PROFILE_SELECTED, STATUS_TASK_INSUFFICIENT},
    STATUS_D0_PROFILE_SELECTED: {STATUS_D1_POWER_FROZEN, STATUS_UNDERPOWERED},
    STATUS_D1_POWER_FROZEN: {STATUS_D1_LOCKED},
    STATUS_D1_LOCKED: {STATUS_D1_COLLECTED},
    STATUS_D1_COLLECTED: {STATUS_D1_SCORES_FROZEN},
    STATUS_D1_SCORES_FROZEN: {
        STATUS_TASK_DEGENERATE, STATUS_MODEL_INSUFFICIENT, STATUS_INVALID,
        STATUS_NOT_CONFIRMED, STATUS_CONFIRMED,
    },
}


class PreflightError(ValueError):
    """A preflight/verification failure; carries the plan-§7 terminal status."""

    def __init__(self, message: str, *, status: str = STATUS_INVALID):
        super().__init__(message)
        self.status = status


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _normalised(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config_signature(config: Mapping[str, Any]) -> str:
    from ac_twin.checkpointing import canonical_fingerprint
    return canonical_fingerprint(json.loads(json.dumps(config)))


# --- state machine (plan §7) -------------------------------------------------------------------


def transition(current: str, requested: str) -> str:
    """Advance the D0/D1 state machine; terminal states are absorbing."""
    if requested not in TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid Stage 4D transition: {current} -> {requested}")
    return requested


# --- candidate seed schedules (§6.2; frozen Stage 2 primitive ONLY) ----------------------------


def make_candidate_seed_schedule(entropy: int, size: int) -> np.ndarray:
    """Generate one of this stage's five approved candidate schedules; fail closed.

    Uses the frozen Stage 2 primitive directly. Any non-approved (entropy, size) pair is
    refused; the generated raw bytes must match the plan-§6.2 pinned SHA256.
    """
    entropy = int(entropy)
    size = int(size)
    approved = _APPROVED_SCHEDULES.get((entropy, size))
    if approved is None:
        raise ValueError(
            f"Stage 4D candidate (entropy, size) must be one of "
            f"{sorted(_APPROVED_SCHEDULES)}; got ({entropy}, {size})")
    schedule = make_seed_schedule(entropy, size=size).astype(np.int64)
    if int(schedule.min()) < SEED_LOW or int(schedule.max()) > SEED_HIGH_INCLUSIVE:
        raise ValueError("Stage 4D candidate seed schedule out of range")
    if np.unique(schedule).size != schedule.size:
        raise ValueError("Stage 4D candidate seed schedule contains duplicate seeds")
    digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    if digest != approved:
        raise ValueError(
            f"Stage 4D candidate schedule raw-bytes SHA256 drifted: expected "
            f"{approved}, got {digest}")
    return schedule


def action_stream_root(config: Mapping[str, Any], role: str) -> tuple[int, int]:
    """The frozen action-sampling SeedSequence root for ``role`` (plan §6.2)."""
    checked = validate_config(config)
    role = str(role)
    if role not in ACTION_STREAM_ROOTS:
        raise ValueError(f"unknown Stage 4D seed role: {role!r}")
    root = tuple(int(v) for v in checked["action_stream_roots"][role])
    expected = ACTION_STREAM_ROOTS[role]
    if root != expected:
        raise ValueError(f"action stream root for {role} drifted from {expected}")
    return expected


# --- global historical seed inventory (strict; R2 and confirm are history) ---------------------


def scan_historical_group_seeds(workspace_roots: Sequence[Path]) -> np.ndarray:
    """Union of every historical accepted/attempted/reserved seed (strict, fail-loud).

    Scope (plan §6.2): all historical accepted seeds, every attempted/rejected ledger seed,
    every persisted candidate ``.npy/.json`` schedule, the contract-reserved Stage 2 / 4C /
    4C Plus / R2 / confirm domains, and the Stage 2B legacy stale domains. Only this stage's
    own namespace is excluded (resume safety). Sealed ``confirmation_truth`` files are NEVER
    opened — their accepted seeds are a subset of the persisted candidate schedules, which
    ARE scanned, so the zero-overlap evidence is complete. Any unreadable historical product
    raises; ``except: continue`` is forbidden.
    """
    collected: set[int] = set()
    files_visited = 0
    for root in workspace_roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"historical scan root missing: {root}")
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            normalized = _normalised(file_path)
            if any(token in normalized for token in _OWN_NAMESPACE_TOKENS):
                continue  # this stage's own bound products only
            if "confirmation_truth" in normalized:
                continue  # sealed truth: seeds covered by the persisted schedules
            files_visited += 1
            suffix = file_path.suffix.lower()
            if suffix == ".npz":
                try:
                    with np.load(file_path, allow_pickle=True) as archive:
                        for key in _SEED_NPZ_KEYS:
                            if key in archive.files:
                                values = np.atleast_1d(np.asarray(archive[key])).ravel()
                                collected.update(int(v) for v in values)
                except Exception as exc:
                    raise ValueError(
                        f"historical npz product unreadable (fail-loud): "
                        f"{file_path}: {exc}") from exc
            elif suffix == ".npy" and "seed_schedule" in file_path.name.lower():
                try:
                    values = np.load(file_path).ravel()
                except Exception as exc:
                    raise ValueError(
                        f"historical candidate schedule unreadable (fail-loud): "
                        f"{file_path}: {exc}") from exc
                collected.update(int(v) for v in values)
            elif suffix == ".json":
                collected.update(confirm._seeds_in_json_strict(file_path))
            elif suffix == ".jsonl":
                collected.update(confirm._seeds_in_jsonl_strict(file_path))
    if files_visited == 0:
        raise ValueError(
            "empty workspace: an empty scan cannot evidence zero seed overlap; "
            "fail loudly instead of attesting a clean audit")
    # Contract-reserved domains (R2 AND the confirm stage are both history now).
    reserved = {
        "stage2_confirm_frozen": make_seed_schedule(
            s2.COLLECTION_SEED_CONFIRM, size=s2.SEED_SCHEDULE_SIZE),
        "stage4c_confirm_reserved": make_seed_schedule(
            s4c.CONFIRM_SEED_ENTROPY, size=s4c.CONFIRM_SCHEDULE_SIZE),
        "stage4cp_dev_reserved": make_seed_schedule(
            s4cp.DEV_SEED_ENTROPY, size=s4cp.DEV_SCHEDULE_SIZE),
        "stage4cp_confirm_reserved": make_seed_schedule(
            s4cp.CONFIRM_SEED_ENTROPY, size=s4cp.CONFIRM_SCHEDULE_SIZE),
        "stage4cr2_executed_full_domain": confirm.r2_reserved_seed_domain(),
        "stage4cr2_confirm_executed": confirm.make_candidate_seed_schedule(
            confirm.CONFIRM_SEED_ENTROPY, confirm.CONFIRM_SCHEDULE_SIZE),
        "stage4cr2_confirm_smoke": confirm.make_candidate_seed_schedule(
            confirm.SMOKE_SEED_ENTROPY, confirm.SMOKE_SCHEDULE_SIZE),
    }
    for domain in reserved.values():
        collected.update(int(v) for v in domain)
    for domain in confirm.legacy_stage2b_candidate_domains().values():
        collected.update(int(v) for v in domain)
    if not collected:
        raise ValueError("no historical seeds found in the workspace")
    return np.array(sorted(collected), dtype=np.int64)


@dataclass(frozen=True)
class SeedAudit:
    """Result of the global zero-overlap audit for one candidate schedule."""
    n_schedule: int
    n_inventory: int
    overlap: tuple[int, ...]

    @property
    def zero_overlap(self) -> bool:
        return len(self.overlap) == 0


def audit_global_seed_overlap(
    schedule: np.ndarray,
    roots: Sequence[Path],
    *,
    inventory: np.ndarray | None = None,
) -> SeedAudit:
    """Audit ``schedule`` against the global historical+reserved seed inventory.

    Returns a :class:`SeedAudit`; does not raise on overlap — call
    :func:`assert_seed_audit_clean`. Any overlap is a plan-§6.2 hard stop requiring a
    whole-domain entropy revision, never single-seed substitution.
    """
    schedule = np.asarray(schedule, dtype=np.int64)
    if schedule.size == 0:
        raise ValueError("seed schedule must be non-empty")
    if int(schedule.min()) < SEED_LOW or int(schedule.max()) > SEED_HIGH_INCLUSIVE:
        raise ValueError("seed schedule out of range")
    if np.unique(schedule).size != schedule.size:
        raise ValueError("seed schedule contains duplicate seeds")
    if inventory is None:
        inventory = scan_historical_group_seeds(roots)
    overlap = np.intersect1d(schedule, np.asarray(inventory, dtype=np.int64))
    return SeedAudit(
        n_schedule=int(schedule.size),
        n_inventory=int(inventory.size),
        overlap=tuple(int(v) for v in overlap),
    )


def assert_seed_audit_clean(audit: SeedAudit) -> None:
    """Raise unless the audit proves global zero overlap."""
    if not audit.zero_overlap:
        raise PreflightError(
            f"candidate schedule overlaps {len(audit.overlap)} historical/reserved "
            f"seed(s) {list(audit.overlap[:10])}; plan section 6.2 requires a "
            "whole-domain entropy version revision, never single-seed substitution")


# --- §2.1 source chain verification ------------------------------------------------------------


def verify_pinned_sources(root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Re-hash every plan-§2.1 pinned source and verify it against the pinned SHA256.

    Returns ``{rel_path: {path, sha256, pinned_sha256, status}}``. Fail-closed: any missing
    file or hash drift raises before any scientific use.
    """
    base = _resolve(root) if root is not None else ROOT
    out: dict[str, dict[str, str]] = {}
    for rel, expected in STAGE4D_PINNED_SOURCES.items():
        path = base / rel
        if not path.is_file():
            raise PreflightError(f"missing frozen plan-2.1 source: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise PreflightError(
                f"frozen source hash drift: {rel}; expected {expected}, got {digest}")
        out[rel] = {
            "path": str(path), "sha256": digest,
            "pinned_sha256": expected, "status": "match",
        }
    return out


def build_source_inventory(root: str | Path | None = None) -> dict[str, Any]:
    """Expand the ten plan-§2.1 sources into a per-file path/hash inventory with verdicts.

    Also re-checks the R2 train NPZ binding recorded by the pinned train receipt against
    plan §2.1 (the large NPZ itself is hashed at Task-4 lock time, not here).
    """
    files = verify_pinned_sources(root)
    receipt = _read_json(files[R2_TRAIN_COLLECTION_RECEIPT]["path"])
    bound = str(receipt.get("data_output_sha256", ""))
    if bound != R2_TRAIN_NPZ_SHA256:
        raise PreflightError(
            f"R2 train NPZ binding drifted: receipt records {bound}, plan §2.1 binds "
            f"{R2_TRAIN_NPZ_SHA256}")
    return {
        "plan_id": PLAN_ID,
        "stage_namespace": STAGE_NAMESPACE,
        "n_files": len(files),
        "files": files,
        "r2_train_npz_sha256": R2_TRAIN_NPZ_SHA256,
        "inventory_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest(),
    }


# --- code inventory -----------------------------------------------------------------------------


def _environment_versions() -> dict[str, str]:
    from importlib.metadata import version as _pkg_version
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": _pkg_version("numpy"),
        "torch_version": _pkg_version("torch"),
        "highway_env_version": _pkg_version("highway_env"),
        "gymnasium_version": _pkg_version("gymnasium"),
    }


def _code_hashes(files: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in files:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered Stage 4D code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


def protocol_code_hashes() -> dict[str, str]:
    """Hashes bound at protocol-lock time: this stage's modules + the reused frozen chain."""
    return _code_hashes(tuple(PROTOCOL_CODE_FILES) + tuple(REUSED_FROZEN_CODE_FILES))


def build_code_inventory() -> dict[str, Any]:
    """Reviewable code inventory: per-file hashes + frozen third-party versions."""
    hashes = protocol_code_hashes()
    return {
        "plan_id": PLAN_ID,
        "stage_code_files": list(PROTOCOL_CODE_FILES),
        "reused_frozen_code_files": list(REUSED_FROZEN_CODE_FILES),
        "code_sha256": hashes,
        "environment": _environment_versions(),
        "inventory_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest(),
    }


# --- §7.1 condition-8 threshold derivation rule (controller ruling 1) ---------------------------


def derive_r2_train_distance_threshold(pre_intervention_min_distances) -> float:
    """The fixed R2 train-only initial-distance threshold rule (never changed once written).

    Input: the per-R2-train-group pre-intervention minimum ego-to-other distance array
    (extracted from the R2 train NPZ's pre-``t_star`` history/mask by Task 4 — this function
    never reads the NPZ itself). Output: the median, the frozen stratum threshold.
    """
    arr = np.asarray(pre_intervention_min_distances, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("pre-intervention distance array must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError("pre-intervention distance array must be finite")
    return float(np.median(arr))


# --- config validation ---------------------------------------------------------------------------


def _check_ood_profiles(profiles_cfg: Mapping[str, Any]) -> None:
    """Cross-check the config's profile tables against the frozen env-module constants."""
    if list(profiles_cfg.get("profiles", [])) != list(ood.PROFILES):
        raise ValueError(f"ood_profiles.profiles must be {list(ood.PROFILES)}")
    if list(profiles_cfg.get("selection_priority", [])) != list(ood.PROFILE_PRIORITY):
        raise ValueError(
            f"ood_profiles.selection_priority must be {list(ood.PROFILE_PRIORITY)}")
    behavior = profiles_cfg.get("behavior", {})
    for name, expected in (("aggressive", ood.AGGRESSIVE_PARAMS),
                           ("cautious", ood.CAUTIOUS_PARAMS),
                           ("id_reference", ood.ID_REFERENCE_PARAMS),
                           ("unchanged", ood.UNCHANGED_BEHAVIOR_PARAMS)):
        got = {k: float(v) for k, v in behavior.get(name, {}).items()}
        if got != expected:
            raise ValueError(f"ood_profiles.behavior.{name} drifted from plan §4.1")
    for key, expected in (("n_vehicles", ood.N_BEHAVIOR_VEHICLES),
                          ("n_aggressive", ood.N_AGGRESSIVE),
                          ("n_cautious", ood.N_CAUTIOUS)):
        if int(behavior.get(key, -1)) != expected:
            raise ValueError(f"ood_profiles.behavior.{key} drifted from {expected}")
    geometry = profiles_cfg.get("geometry", {})
    for key, expected in (
            ("highway_positions_jitter", ood.HIGHWAY_POSITIONS_JITTER),
            ("highway_speeds_jitter", ood.HIGHWAY_SPEEDS_JITTER),
            ("merging_x_base", ood.MERGING_X_BASE),
            ("merging_x_jitter", ood.MERGING_X_JITTER),
            ("merging_speed_base", ood.MERGING_SPEED_BASE),
            ("merging_speed_jitter", ood.MERGING_SPEED_JITTER),
            ("merging_target_speed", ood.MERGING_TARGET_SPEED)):
        if float(geometry.get(key, -9.0)) != expected:
            raise ValueError(f"ood_profiles.geometry.{key} drifted from plan §4.2")
    if [float(x) for x in geometry.get("highway_positions_base", [])] != \
            list(ood.HIGHWAY_POSITIONS_BASE):
        raise ValueError("ood_profiles.geometry.highway_positions_base drifted from §4.2")
    if [float(x) for x in geometry.get("highway_speeds_base", [])] != \
            list(ood.HIGHWAY_SPEEDS_BASE):
        raise ValueError("ood_profiles.geometry.highway_speeds_base drifted from §4.2")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 4D D0/D1 config; raise before any artifact is produced."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("scene") != "merge" or checked.get("env_id") != ENV_ID:
        raise ValueError("Stage 4D D0/D1 is merge-v0 only")
    if checked.get("stage_namespace") != STAGE_NAMESPACE:
        raise ValueError(f"stage_namespace must be {STAGE_NAMESPACE!r}")
    if checked.get("purpose") != PURPOSE:
        raise ValueError(f"purpose must be {PURPOSE!r}")
    for key in checked:
        if key in _FORBIDDEN_RETRAIN_KEYS:
            raise ValueError(f"config.{key} is forbidden: no retraining/refitting")
    if [tuple(p) for p in checked.get("action_plans", [])] != [tuple(p) for p in ACTION_PLANS]:
        raise ValueError(f"action_plans must be {list(ACTION_PLANS)}")
    if list(checked.get("cf_actions", [])) != list(CF_ACTIONS):
        raise ValueError(f"cf_actions must be {list(CF_ACTIONS)}")
    if list(checked.get("intervention_durations", [])) != list(CF_DURATIONS):
        raise ValueError(f"intervention_durations must be {list(CF_DURATIONS)}")
    weights = checked.get("action_weights", {})
    for key, expected in ACTION_WEIGHTS.items():
        if float(weights.get(key, -1.0)) != expected:
            raise ValueError(f"action_weights.{key} must be {expected}")
    for key, expected in (
        ("n_actions", N_ACTIONS), ("n_branches_per_group", N_BRANCHES_PER_GROUP),
        ("history_len", HISTORY_LEN), ("horizon", HORIZON),
        ("factual_suffix_steps", FACTUAL_SUFFIX_STEPS),
        ("t_star_low", T_STAR_LOW), ("t_star_high", T_STAR_HIGH),
        ("simulation_frequency", SIMULATION_FREQUENCY),
        ("policy_frequency", POLICY_FREQUENCY),
        ("vehicles_count", VEHICLES_COUNT), ("duration", ENV_DURATION),
        ("observed_vehicles_count", OBSERVED_VEHICLES_COUNT),
        ("object_capacity", OBJECT_CAPACITY), ("n_object_features", N_OBJECT_FEATURES),
        ("n_competitors", N_COMPETITORS), ("token_dim", TOKEN_DIM),
        ("global_dim", GLOBAL_DIM), ("direct_hidden_width", DIRECT_HIDDEN_WIDTH),
        ("expected_trainable_parameters", EXPECTED_TRAINABLE_PARAMETERS),
        ("idle_plan_index", IDLE_PLAN_INDEX),
        ("smoke_seed_entropy", SMOKE_SEED_ENTROPY),
        ("smoke_schedule_size", SMOKE_SCHEDULE_SIZE),
        ("pilot_behavior_seed_entropy", PILOT_BEHAVIOR_SEED_ENTROPY),
        ("pilot_geometry_seed_entropy", PILOT_GEOMETRY_SEED_ENTROPY),
        ("pilot_joint_seed_entropy", PILOT_JOINT_SEED_ENTROPY),
        ("pilot_schedule_size", PILOT_SCHEDULE_SIZE),
        ("d1_candidate_seed_entropy", D1_CANDIDATE_SEED_ENTROPY),
        ("d1_candidate_schedule_size", D1_CANDIDATE_SCHEDULE_SIZE),
        ("d1_bootstrap_entropy", D1_BOOTSTRAP_ENTROPY),
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("domain_classifier_bootstrap_resamples", DOMAIN_CLASSIFIER_BOOTSTRAP_RESAMPLES),
        ("power_monte_carlo", POWER_MONTE_CARLO),
    ):
        if int(checked.get(key, -1)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    for key, expected in (
        ("dropout", DROPOUT), ("high_regret_threshold", HIGH_REGRET_THRESHOLD),
        ("coverage", COVERAGE), ("confidence", CONFIDENCE),
        ("baseline_tie_tolerance", BASELINE_TIE_TOLERANCE),
        ("domain_classifier_confidence", DOMAIN_CLASSIFIER_CONFIDENCE),
        ("domain_shift_smd_min", DOMAIN_SHIFT_SMD_MIN),
        ("domain_classifier_auc_ci_lb_min", DOMAIN_CLASSIFIER_AUC_CI_LB_MIN),
        ("power_target", POWER_TARGET),
        ("power_selection_retention", POWER_SELECTION_RETENTION),
    ):
        if float(checked.get(key, -9.0)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    for key, expected in (
        ("smoke_schedule_raw_sha256", SMOKE_SCHEDULE_RAW_SHA256),
        ("pilot_behavior_schedule_raw_sha256", PILOT_BEHAVIOR_SCHEDULE_RAW_SHA256),
        ("pilot_geometry_schedule_raw_sha256", PILOT_GEOMETRY_SCHEDULE_RAW_SHA256),
        ("pilot_joint_schedule_raw_sha256", PILOT_JOINT_SCHEDULE_RAW_SHA256),
        ("d1_candidate_schedule_raw_sha256", D1_CANDIDATE_SCHEDULE_RAW_SHA256),
    ):
        if checked.get(key) != expected:
            raise ValueError(f"{key} drifted from the approved raw-bytes SHA256")
    if [int(s) for s in checked.get("model_seeds", [])] != list(MODEL_SEEDS):
        raise ValueError(f"model_seeds must be {list(MODEL_SEEDS)}")
    if list(checked.get("response_predictor_seeds", [])) != list(RESPONSE_PREDICTOR_SEEDS):
        raise ValueError("response_predictor_seeds drifted from the frozen five")
    if list(checked.get("expert_router_seeds", [])) != list(EXPERT_ROUTER_SEEDS):
        raise ValueError("expert_router_seeds drifted from the frozen five pairs")
    if checked.get("architecture", ARCH_DIRECT) != ARCH_DIRECT:
        raise ValueError(f"this stage scores only the frozen {ARCH_DIRECT}")
    if checked.get("primary_score_definition") != PRIMARY_SCORE_DEFINITION:
        raise ValueError("primary_score_definition drifted from the frozen definition")
    if checked.get("ensemble_definition") != ENSEMBLE_DEFINITION:
        raise ValueError("ensemble_definition must be the arithmetic mean of five seeds")
    if checked.get("primary_baseline") != PRIMARY_BASELINE:
        raise ValueError(f"primary_baseline must be {PRIMARY_BASELINE!r} (validation-frozen)")
    if list(checked.get("standard_baselines", [])) != list(STANDARD_BASELINES):
        raise ValueError(f"standard_baselines must be {list(STANDARD_BASELINES)}")
    if list(checked.get("reference_baselines", [])) != list(REFERENCE_BASELINES):
        raise ValueError(f"reference_baselines must be {list(REFERENCE_BASELINES)}")
    if [int(n) for n in checked.get("candidate_ns", [])] != list(CANDIDATE_NS):
        raise ValueError(f"candidate_ns must be {list(CANDIDATE_NS)}")
    if [float(x) for x in checked.get("retention_factors", [])] != list(RETENTION_FACTORS):
        raise ValueError(f"retention_factors must be {list(RETENTION_FACTORS)}")
    roots = checked.get("action_stream_roots", {})
    for role, expected in ACTION_STREAM_ROOTS.items():
        if [int(v) for v in roots.get(role, [])] != list(expected):
            raise ValueError(f"action_stream_roots.{role} drifted from {expected}")
    classifier = checked.get("domain_classifier_bootstrap_entropies", {})
    if {str(k): int(v) for k, v in classifier.items()} != \
            dict(DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES):
        raise ValueError("domain_classifier_bootstrap_entropies drifted from plan §5.2/§6.2")
    power_chain = checked.get("power_and_chain", {})
    for key, expected in POWER_AND_CHAIN.items():
        if float(power_chain.get(key, -9.0)) != expected:
            raise ValueError(f"power_and_chain.{key} drifted from {expected}")
    gate_task = checked.get("gate_task_ood", {})
    for key, expected in GATE_TASK_OOD.items():
        if float(gate_task.get(key, -9.0)) != expected:
            raise ValueError(f"gate_task_ood.{key} drifted from {expected}")
    gate_risk = checked.get("gate_ood_risk", {})
    for key, expected in GATE_OOD_RISK.items():
        if float(gate_risk.get(key, -9.0)) != expected:
            raise ValueError(f"gate_ood_risk.{key} drifted from {expected}")
    pinned = checked.get("r2_pinned_sources", {})
    if {str(k): str(v) for k, v in pinned.items()} != dict(STAGE4D_PINNED_SOURCES):
        raise ValueError("r2_pinned_sources drifted from the plan-2.1 pinned SHA256 table")
    _check_ood_profiles(checked.get("ood_profiles", {}))
    for key in ("run_dir", "data_root", "r2_development_dir"):
        if key not in checked:
            raise ValueError(f"config missing required path key: {key}")
    return checked


# --- protocol lock verification primitive (§6.3; the formal lock is created in Task 4) ----------


def verify_protocol_lock(run_dir: str | Path) -> dict[str, Any]:
    """Re-derive every binding of this stage's protocol lock; fail closed on any tampering.

    Read-only. The formal lock does not exist yet (it is created in Task 4 after the D1
    power analysis freezes n); calling this on a run dir without a lock raises
    ``FileNotFoundError`` (fail-closed). When a lock exists, every binding it carries —
    plan id, config signature, frozen source inventory, code hashes and seed-audit
    cleanliness — is re-derived and re-checked.
    """
    run_dir = _resolve(run_dir)
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage 4D protocol lock missing: {lock_path}")
    payload = _read_json(lock_path)
    if payload.get("plan_id") != PLAN_ID:
        raise ValueError(f"lock plan_id must be {PLAN_ID!r}")
    if _config_signature(validate_config(_read_json(CONFIG_PATH))) != \
            payload.get("config_signature"):
        raise ValueError("lock config signature mismatch (config drift or tampering)")
    inventory = payload.get("frozen_source_inventory")
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("lock does not carry a frozen_source_inventory")
    for rel, entry in inventory.items():
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"inventoried frozen source missing: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"inventoried frozen source hash drift: {rel} ({path})")
    code = payload.get("code_sha256")
    if not isinstance(code, Mapping):
        raise ValueError("lock does not carry a code_sha256 mapping")
    if protocol_code_hashes() != dict(code):
        raise ValueError("locked code SHA256 set mismatch (code drift or tampering)")
    audit = payload.get("seed_audit", {})
    if not audit.get("zero_overlap") or audit.get("overlap"):
        raise ValueError("lock records a non-clean seed audit")
    return payload


# Reused frozen helpers re-exported for the collection/inference layers (read-only).
materialize_action_plan = s4c.materialize_action_plan
build_inference_mask = s4c.build_inference_mask


__all__ = [
    "ACTION_PLANS",
    "ACTION_STREAM_ROOTS",
    "ACTION_WEIGHTS",
    "ARCH_DIRECT",
    "BASELINE_TIE_TOLERANCE",
    "BOOTSTRAP_RESAMPLES",
    "CANDIDATE_NS",
    "CF_ACTIONS",
    "CF_DURATIONS",
    "CONFIG_PATH",
    "CONFIDENCE",
    "COVERAGE",
    "D1_BOOTSTRAP_ENTROPY",
    "D1_CANDIDATE_SCHEDULE_RAW_SHA256",
    "D1_CANDIDATE_SCHEDULE_SIZE",
    "D1_CANDIDATE_SEED_ENTROPY",
    "DIRECT_HIDDEN_WIDTH",
    "DOMAIN_CLASSIFIER_AUC_CI_LB_MIN",
    "DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES",
    "DOMAIN_CLASSIFIER_BOOTSTRAP_RESAMPLES",
    "DOMAIN_CLASSIFIER_CONFIDENCE",
    "DOMAIN_SHIFT_MIN_SHIFTED",
    "DOMAIN_SHIFT_SMD_MIN",
    "DROPOUT",
    "EGO_SLOT",
    "ENV_DURATION",
    "ENV_ID",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "EXPERT_ROUTER_SEEDS",
    "FACTUAL_SUFFIX_STEPS",
    "GATE_OOD_RISK",
    "GATE_TASK_OOD",
    "GLOBAL_DIM",
    "HIGH_REGRET_THRESHOLD",
    "HISTORY_LEN",
    "HORIZON",
    "IDLE_PLAN_INDEX",
    "MODEL_SEEDS",
    "N_ACTIONS",
    "N_BRANCHES_PER_GROUP",
    "N_COMPETITORS",
    "N_OBJECT_FEATURES",
    "OBJECT_CAPACITY",
    "OBSERVED_VEHICLES_COUNT",
    "PILOT_ACCEPTED_GROUPS",
    "PILOT_BEHAVIOR_SCHEDULE_RAW_SHA256",
    "PILOT_BEHAVIOR_SEED_ENTROPY",
    "PILOT_GEOMETRY_SCHEDULE_RAW_SHA256",
    "PILOT_GEOMETRY_SEED_ENTROPY",
    "PILOT_JOINT_SCHEDULE_RAW_SHA256",
    "PILOT_JOINT_SEED_ENTROPY",
    "PILOT_MAX_CANDIDATE_SEEDS",
    "PILOT_SCHEDULE_SIZE",
    "PLAN_ID",
    "POLICY_FREQUENCY",
    "POWER_AND_CHAIN",
    "POWER_ENTROPY",
    "POWER_MONTE_CARLO",
    "POWER_SELECTION_RETENTION",
    "POWER_TARGET",
    "PRIMARY_BASELINE",
    "PRIMARY_SCORE_DEFINITION",
    "ENSEMBLE_DEFINITION",
    "PROTOCOL_CODE_FILES",
    "PURPOSE",
    "R2_DEVELOPMENT_DIR",
    "R2_TRAIN_COLLECTION_RECEIPT",
    "R2_TRAIN_DISTANCE_THRESHOLD_RULE",
    "R2_TRAIN_NPZ_SHA256",
    "REFERENCE_BASELINES",
    "RESERVED_ENTROPIES",
    "RESPONSE_PREDICTOR_SEEDS",
    "RETENTION_FACTORS",
    "REUSED_FROZEN_CODE_FILES",
    "ROOT",
    "SEED_HIGH_INCLUSIVE",
    "SEED_LOW",
    "SEED_ROLES",
    "SIMULATION_FREQUENCY",
    "SMOKE_SCHEDULE_RAW_SHA256",
    "SMOKE_SCHEDULE_SIZE",
    "SMOKE_SEED_ENTROPY",
    "STAGE4D_PINNED_SOURCES",
    "STAGE_NAMESPACE",
    "STANDARD_BASELINES",
    "STATUS_APPROVED",
    "STATUS_CONFIRMED",
    "STATUS_D0_PILOTS_COLLECTED",
    "STATUS_D0_PROFILE_SELECTED",
    "STATUS_D1_COLLECTED",
    "STATUS_D1_LOCKED",
    "STATUS_D1_POWER_FROZEN",
    "STATUS_D1_SCORES_FROZEN",
    "STATUS_INVALID",
    "STATUS_MODEL_INSUFFICIENT",
    "STATUS_NOT_CONFIRMED",
    "STATUS_PREFLIGHT_PASSED",
    "STATUS_SOFTWARE_IN_PROGRESS",
    "STATUS_TASK_DEGENERATE",
    "STATUS_TASK_INSUFFICIENT",
    "STATUS_UNDERPOWERED",
    "SeedAudit",
    "TERMINAL_STATES",
    "TOKEN_DIM",
    "TRANSITIONS",
    "T_STAR_HIGH",
    "T_STAR_LOW",
    "VEHICLES_COUNT",
    "PreflightError",
    "action_stream_root",
    "assert_seed_audit_clean",
    "audit_global_seed_overlap",
    "build_code_inventory",
    "build_inference_mask",
    "build_source_inventory",
    "derive_r2_train_distance_threshold",
    "make_candidate_seed_schedule",
    "materialize_action_plan",
    "protocol_code_hashes",
    "scan_historical_group_seeds",
    "transition",
    "validate_config",
    "verify_pinned_sources",
    "verify_protocol_lock",
    "write_once_json",
]
