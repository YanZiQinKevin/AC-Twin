"""Stage 4C-R2 independent-confirmation protocol: config, seeds, audit, lock, statuses.

Freeze layer for the independent confirmation of the frozen R2 direct-supervised
decision-risk model (plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1``). This
stage retrains NOTHING, refits NOTHING and re-judges NOTHING historical: it scores the
five frozen Direct checkpoints on 2,000 fresh Merge ten-action decision groups and runs
the one-shot confirmation adjudication.

Scientific isolation rules encoded here (plan sections 2, 4.2, 5.3, 8):

* The candidate seed schedules are generated EXACTLY by the frozen Stage 2 primitive
  ``stage2_protocol.make_seed_schedule(entropy, size=...)`` (``SeedSequence(entropy)
  .spawn(3)`` first stream, draw without replacement, +1 shift, range
  ``[1, 2147483646]``, int64). The R2/4C ``make_group_seed_schedule`` wrappers are
  NEVER called for this stage's domains (they would reject the new entropies), and the
  old wrappers are never modified to accept them.
* The global seed audit treats R2 as HISTORY: R2's full executed 60,000-seed reserved
  domain is resolved from R2's EXECUTED frozen config (its hash-pinned
  ``development/protocol_lock.json``, v0.2 entropy 2026092880, size 60000), never from
  the superseded v0.1 entropy. The scanner excludes ONLY this stage's own namespace
  (so a resume does not collide with itself); no blanket ``stage4c`` string exclusion
  is applied to history. Read failures of historical scientific products FAIL the
  preflight loudly — there is no ``except: continue``.
* The Stage 2B stale module-constant candidate domains (never persisted, never used
  for collection) are registered explicitly under ``legacy_test_compatibility`` and
  conservatively avoided; they are not mislabelled as actually-used data.
* Every R2 authoritative source (plan section 2.1) is hash-verified against the pinned
  SHA256 table before any use; Direct checkpoints are verified against
  ``final_selection.json`` and cross-checked with ``model_inventory.json`` after path
  relocation.
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
from ac_twin import stage3_data as s3d
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4c_protocol as s4c
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4cr2_protocol as s4cr2  # frozen R2 contract, read-only reference
from ac_twin import stage2b_protocol as s2b  # legacy candidate-domain constants only
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

PLAN_ID = "AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1"
STAGE_NAMESPACE = "stage4cr2_confirm"
PURPOSE_CONFIRM = "stage4cr2_merge_multiaction_confirm"  # data namespace (plan section 10)

CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4cr2_confirm/stage4cr2_confirm_config.json")

# --- environment / tensor contract (REUSED bitwise from Stage 4C; plan section 3.1) ------
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

# --- fixed task contract (REUSED bitwise; the ten-action set is NEVER derived from the
# --- counterfactual-truth array shapes; plan section 3.1) --------------------------------
ACTION_PLANS = s4co.ACTION_PLANS
N_ACTIONS = 10
if len(ACTION_PLANS) != N_ACTIONS:  # fixed contract guard, not a derivation
    raise RuntimeError("frozen ACTION_PLANS must contain exactly ten plans")
IDLE_PLAN_INDEX = s4co.IDLE_PLAN_INDEX
if ACTION_PLANS[IDLE_PLAN_INDEX] != (1, 3):
    raise RuntimeError("IDLE reference must stay (IDLE,3), never (IDLE,1)")
EGO_SLOT = s4co.EGO_SLOT

# --- fixed model contract (plan section 3.3; REUSED bitwise from the frozen R2 chain) ----
ARCH_DIRECT = s4cr2.ARCH_DIRECT
MODEL_SEEDS = s4cr2.MODEL_SEEDS  # (8101, 8202, 8303, 8404, 8505)
N_COMPETITORS = s4cr2.N_COMPETITORS
TOKEN_DIM = s4cr2.TOKEN_DIM
GLOBAL_DIM = s4cr2.GLOBAL_DIM
DIRECT_HIDDEN_WIDTH = s4cr2.DIRECT_HIDDEN_WIDTH
DROPOUT = s4cr2.DROPOUT
EXPECTED_TRAINABLE_PARAMETERS = s4cr2.EXPECTED_TRAINABLE_PARAMETERS  # EXACTLY 15,740
PRIMARY_SCORE_DEFINITION = (
    "risk_score = sigmoid(logit_high) * 0.01 * expm1(softplus(raw_mu)); "
    "ensemble = arithmetic mean of the five model seeds' risk_score")
ENSEMBLE_DEFINITION = "arithmetic_mean_of_five_seeds"

# --- baselines / strata (plan section 3.4; validation-frozen, never reselected here) -----
PRIMARY_BASELINE = "SEED_VOTE_ENTROPY"
STANDARD_BASELINES = ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD")
REFERENCE_BASELINES = ("COMBINED_TOP2_MAX", "CONSTANT_RANDOM")
ABLATIONS = ("MU_ONLY", "CLASS_ONLY")
BASELINE_TIE_TOLERANCE = 1e-12
HIGH_REGRET_THRESHOLD = s4cr2.HIGH_REGRET_THRESHOLD  # 0.01
REGRET_CLAMP_TOL = s4co.REGRET_CLAMP_TOL
COVERAGE = 0.80
INTERACTION_THRESHOLD = 29.128996473819946  # R2 train-only median, frozen by selection
T_STAR_STRATUM_SPLIT = 13

# --- frozen gates (plan sections 6.2, 6.3) -------------------------------------------------
GATE_RISK_I_AURC_MIN = 0.05
GATE_RISK_I_REGRET80_MIN = 0.03
GATE_RISK_MIN_POSITIVE_SEED_METRICS = 4
GATE_RISK_STRATA_I_AURC_FLOOR = -0.10
GATE_RISK_TRIM_FRACTIONS = (0.01, 0.05)
GATE_RISK_SPEARMAN_MARGIN_CEILING = 0.995
GATE_TASK_MIN_HIGH_REGRET_FRAC = 0.15
GATE_TASK_REGRET_EPS = 0.01
GATE_TASK_MIN_DISTINCT_ORACLE = 3
GATE_TASK_MIN_ORACLE_FRAC = 0.05
GATE_TASK_MIN_STRATUM_GROUPS = 200  # fixed 10% support line at n=2000 (plan 6.2.7)

# --- approved frozen seed contract (plan section 4.2, verbatim) ----------------------------
CONFIRM_SEED_ENTROPY = 2026093012
CONFIRM_SCHEDULE_SIZE = 12000
CONFIRM_ACCEPTED_GROUPS = 2000
SMOKE_SEED_ENTROPY = 2026093100
SMOKE_SCHEDULE_SIZE = 64
SMOKE_ACCEPTED_GROUPS = 12
BOOTSTRAP_ENTROPY = 2026093202
CONSTANT_RANDOM_SALT = 2026093203
BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE = 0.95
SEED_LOW = 1
SEED_HIGH_INCLUSIVE = 2147483646  # 2**31 - 2

#: action-sampling RNG roots (plan section 4.2, verbatim): confirm -> SeedSequence(
#: [confirm_entropy, 0]); smoke -> SeedSequence([smoke_entropy, 12]).
ACTION_STREAM_ROOTS = {
    "confirm": (CONFIRM_SEED_ENTROPY, 0),
    "smoke": (SMOKE_SEED_ENTROPY, 12),
}

#: raw-bytes SHA256 of the approved schedules (ndarray bytes, NOT the .npy file hash).
CONFIRM_SCHEDULE_RAW_SHA256 = (
    "4c0958fc5e801dfd62cbab0fbc7fe1d13a91704c649755a23fa7c6f23d5974c8")
SMOKE_SCHEDULE_RAW_SHA256 = (
    "8e35067e0be88e2ac7b20679e07ed2765b0cb5bcb564a3307fd80560bff5d04f")

#: the ONLY (entropy, size) contracts this stage may generate, pinned to their hashes.
_APPROVED_SCHEDULES = {
    (CONFIRM_SEED_ENTROPY, CONFIRM_SCHEDULE_SIZE): CONFIRM_SCHEDULE_RAW_SHA256,
    (SMOKE_SEED_ENTROPY, SMOKE_SCHEDULE_SIZE): SMOKE_SCHEDULE_RAW_SHA256,
}

#: data roles of this stage. There is no train/validation/screen role here.
CONFIRM_DATA_ROLES = ("confirm", "smoke")

# --- mechanical / terminal states (plan section 8, verbatim) -------------------------------
STATUS_INVALID = "STAGE4CR2_CONFIRM_INVALID_ANALYSIS"
STATUS_COLLECTION_INCOMPLETE = "STAGE4CR2_CONFIRM_COLLECTION_INCOMPLETE"
STATUS_RESOURCE_AT_RISK = "STAGE4CR2_CONFIRM_RESOURCE_AT_RISK"
STATUS_TASK_DEGENERATE = "STAGE4CR2_CONFIRM_TASK_DEGENERATE"
STATUS_MODEL_INSUFFICIENT = "INCONCLUSIVE_STAGE4CR2_CONFIRM_DECISION_MODEL_INSUFFICIENT"
STATUS_NOT_CONFIRMED = "STAGE4CR2_DECISION_RISK_NOT_CONFIRMED"
STATUS_CONFIRMED = "STAGE4CR2_DECISION_RISK_CONFIRMED"
#: non-terminal engineering states
STATUS_PREFLIGHT_PASSED = "PREFLIGHT_PASSED"

TERMINAL_STATES = (
    STATUS_INVALID, STATUS_COLLECTION_INCOMPLETE, STATUS_RESOURCE_AT_RISK,
    STATUS_TASK_DEGENERATE, STATUS_MODEL_INSUFFICIENT, STATUS_NOT_CONFIRMED,
    STATUS_CONFIRMED,
)

# --- R2 authoritative sources (plan section 2.1, verbatim pinned SHA256) --------------------
R2_DEVELOPMENT_DIR = "experiments/stage4_reliability/stage4c_r2/development"
R2_PINNED_SOURCES = {
    "protocol_lock.json": "c19c5bdce54fe039a10003b4ab78d1fb22040ac740ba950457c4534fc5e8aa71",
    "final_selection.json": "f135fdab973cca25a5ccb37d9c164a6cd3aaae9d5037d8a8099d33d64534b461",
    "model_inventory.json": "982ec5ea9287f6f82d20e6ea6870f690e60227be3566aea8f17a290d73e77637",
    "feature_schema.json": "4d45bb5556502df06630c20ffaea4fcacc5f818bde28854aec8c434ee9d23033",
    "feature_scaler.npz": "2f21c11ae248f5883d8e9c4c8fa7efb876ad1afb453ee7eec462509a48104194",
    "validation_gate_report.json": "1dce238289b3223aa2607bb1b84f7b6506600a48b5499a5d9b992b58b49c244b",
    "development_screen_report.json": "c54c3f7be98c659f8172b4523eccb937ab79ed21f32a34d65f5c69068bb76f8e",
    "screen_group_quantities.npz": "d68a2c89d6e6827bd366ade73a37ed7e9050fcc06a444e3d533e66ba6c7ed20b",
    "screen_group_quantities_receipt.json": "6abc5b431e9c8cb40ca8f718770bf549de8e52a0a8e9e61bfbbda2e628ce399c",
    "confirmation_power_diagnostic.json": "daa4255aaa43aadb8240e46c6395f4cf620637c156d7491a45da4510d29d047a",
}

#: config keys whose presence would imply training/refitting; forbidden in this stage.
_FORBIDDEN_RETRAIN_KEYS = (
    "learning_rate", "max_epochs", "early_stop_patience", "weight_decay",
    "gradient_clip_norm", "optimizer", "training_seeds_to_fit", "fit_normalization",
    "fit_response_bins", "retrain", "fine_tune",
)

#: this stage's own namespace tokens — the ONLY paths excluded from the historical scan
#: (so a resume does not collide with this stage's own already-bound products). R2's
#: namespace is NOT excluded: R2 is history this stage must avoid.
_OWN_NAMESPACE_TOKENS = (STAGE_NAMESPACE, PURPOSE_CONFIRM)

#: code files of THIS stage bound at protocol-lock time (relative to code/).
#: Extended by Task 2 with the collector; Task 3 (statistics) and Task 4 (runner)
#: extend this registry further before the formal lock is created.
PROTOCOL_CODE_FILES = (
    "ac_twin/stage4cr2_confirm_protocol.py",
    "ac_twin/stage4cr2_confirm_inference.py",
    "collect_stage4cr2_merge_confirm.py",
    "ac_twin/stage4cr2_confirm_statistics.py",
    "run_stage4cr2_confirm.py",
)
#: frozen prior-stage execution dependencies actually exercised by this stage's
#: protocol/inference chain (plan section 5.3: the real executed dependency set, not just
#: R2's protocol_code_hashes()). Read-only; their hashes bind the chain.
REUSED_FROZEN_CODE_FILES = (
    "run_stage4a.py",
    "ac_twin/stage1_data.py",
    "ac_twin/stage2_protocol.py",
    "ac_twin/stage2b_protocol.py",
    "ac_twin/stage3_data.py",
    "ac_twin/stage3_scaler.py",
    "ac_twin/stage3_training.py",
    "ac_twin/stage3_response.py",
    "ac_twin/stage3b_bridge.py",
    "ac_twin/stage3r_models.py",
    "ac_twin/stage3r_protocol.py",
    "ac_twin/checkpointing.py",
    "ac_twin/stage4_protocol.py",
    "ac_twin/stage4_risk.py",
    "ac_twin/stage4b_inference.py",
    "ac_twin/stage4c_protocol.py",
    "ac_twin/stage4c_objective.py",
    "ac_twin/stage4c_inference.py",
    "ac_twin/stage4cp_protocol.py",
    "ac_twin/stage4cp_features.py",
    "ac_twin/stage4cp_models.py",
    "ac_twin/stage4cr2_protocol.py",
    "ac_twin/stage4cr2_training.py",
    # Task 2 collection chain (plan section 5.3: the old rollout engine and its
    # real executed helpers, the reused R2 low-level rollout/QA module, and the
    # progress reporter) — read-only, never modified.
    "collect_stage2_holdout.py",
    "collect_stage4cr2_merge_multiaction.py",
    "collect_pilot.py",
    "ac_twin/object_centric.py",
    "ac_twin/progress.py",
)

#: seed-bearing keys collected from npz archives / JSON payloads / rejection ledgers.
_SEED_NPZ_KEYS = ("episode_seed", "episode_seeds", "group_seed", "group_seeds")
_SEED_JSON_HINTS = ("episode_seed", "episode_seeds", "group_seed", "group_seeds",
                    "seed_schedule")
#: exact keys whose SCALAR int value is itself a seed (e.g. "group_seed": 123).
#: Substring hints must NOT qualify scalars: "seed_schedule_size",
#: "n_historical_episode_seeds", "unique_episode_seeds" etc. are counts/sizes, and
#: collecting them as seeds would pollute the audit with metadata integers.
_SEED_SCALAR_KEYS = ("episode_seed", "group_seed")


class PreflightError(ValueError):
    """A preflight failure; carries the plan-section-8 terminal status."""

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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relocate(path: str | Path) -> Path:
    """Relocate an authored absolute path under the current cloud root if needed.

    Path relocation NEVER re-selects or re-locates models semantically: if the authored
    absolute path is missing, the suffix below the first ``experiments/`` or ``data/``
    anchor is resolved under the current ROOT; the caller hash-verifies the result, so
    a wrong relocation fails loudly.
    """
    candidate = Path(path)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    parts = [p for p in Path(path).parts if p not in ("/", "\\")]
    for anchor in ("experiments", "data"):
        if anchor in [p.lower() for p in parts]:
            tail = parts[[p.lower() for p in parts].index(anchor):]
            relocated = ROOT.joinpath(*tail)
            if relocated.is_file():
                return relocated
    return _resolve(path)


# --- candidate seed schedules (§4.2; frozen Stage 2 primitive ONLY) ------------------------


def make_candidate_seed_schedule(entropy: int, size: int) -> np.ndarray:
    """Generate one of this stage's two approved candidate schedules.

    Uses the frozen Stage 2 primitive directly (``SeedSequence(entropy).spawn(3)`` first
    stream, draw without replacement from ``[0, 2147483646)``, +1 shift -> closed range
    ``[1, 2147483646]``, int64). The R2/4C ``make_group_seed_schedule`` wrappers are
    never called (they would reject these entropies) and never modified. Any
    non-approved (entropy, size) pair is refused; the generated bytes must match the
    approved raw-bytes SHA256.
    """
    entropy = int(entropy)
    size = int(size)
    approved = _APPROVED_SCHEDULES.get((entropy, size))
    if approved is None:
        raise ValueError(
            f"Stage 4C-R2 confirm candidate (entropy, size) must be one of "
            f"{sorted(_APPROVED_SCHEDULES)}; got ({entropy}, {size})")
    schedule = make_seed_schedule(entropy, size=size).astype(np.int64)
    if int(schedule.min()) < SEED_LOW or int(schedule.max()) > SEED_HIGH_INCLUSIVE:
        raise ValueError("confirm candidate seed schedule out of range")
    if np.unique(schedule).size != schedule.size:
        raise ValueError("confirm candidate seed schedule contains duplicate seeds")
    digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    if digest != approved:
        raise ValueError(
            f"confirm candidate schedule raw-bytes SHA256 drifted: expected "
            f"{approved}, got {digest}")
    return schedule


def build_confirm_seed_schedule(config: Mapping[str, Any]) -> np.ndarray:
    checked = validate_config(config)
    return make_candidate_seed_schedule(
        int(checked["confirm_seed_entropy"]), int(checked["confirm_schedule_size"]))


def build_smoke_seed_schedule(config: Mapping[str, Any]) -> np.ndarray:
    checked = validate_config(config)
    return make_candidate_seed_schedule(
        int(checked["smoke_seed_entropy"]), int(checked["smoke_schedule_size"]))


def action_stream_root(config: Mapping[str, Any], role: str) -> tuple[int, int]:
    """The frozen action-sampling SeedSequence root for ``role`` (plan section 4.2)."""
    checked = validate_config(config)
    role = str(role)
    if role not in CONFIRM_DATA_ROLES:
        raise ValueError(f"unknown Stage 4C-R2 confirm role: {role!r}")
    root = tuple(int(v) for v in checked["action_stream_roots"][role])
    expected = ACTION_STREAM_ROOTS[role]
    if root != expected:
        raise ValueError(f"action stream root for {role} drifted from {expected}")
    return expected


# --- R2 reserved 60,000-seed domain (resolved from the EXECUTED frozen config) --------------


def r2_reserved_seed_domain(r2_dev_dir: str | Path | None = None) -> np.ndarray:
    """R2's full executed 60,000-seed reserved domain, from its EXECUTED frozen config.

    Reads R2's hash-pinned ``development/protocol_lock.json`` (pinned SHA256 verified
    FIRST), takes ``dev_seed_entropy``/``dev_schedule_size`` from that executed lock
    (v0.2: 2026092880 / 60000 — never the superseded v0.1 entropy 2026092701), and
    regenerates the domain with the frozen Stage 2 primitive. The persisted
    ``seed_schedule.npy`` next to the lock must match the regenerated domain exactly.
    Any missing/corrupt/tampered source raises (fail-loud; no silent fallback).
    """
    r2_dev = _resolve(r2_dev_dir or R2_DEVELOPMENT_DIR)
    lock_path = r2_dev / "protocol_lock.json"
    if not lock_path.is_file():
        raise PreflightError(f"R2 executed protocol lock missing: {lock_path}")
    digest = sha256_file(lock_path)
    pinned = R2_PINNED_SOURCES["protocol_lock.json"]
    if digest != pinned:
        raise PreflightError(
            f"R2 protocol_lock.json hash drift: expected {pinned}, got {digest}")
    payload = _read_json(lock_path)
    entropy = int(payload["dev_seed_entropy"])
    size = int(payload["dev_schedule_size"])
    domain = make_seed_schedule(entropy, size=size).astype(np.int64)
    schedule_path = r2_dev / "seed_schedule.npy"
    if not schedule_path.is_file():
        raise PreflightError(f"R2 persisted seed schedule missing: {schedule_path}")
    try:
        persisted = np.load(schedule_path)
    except Exception as exc:
        raise PreflightError(f"R2 persisted seed schedule unreadable: {exc}") from exc
    if not np.array_equal(persisted.astype(np.int64), domain):
        raise PreflightError(
            "R2 persisted seed_schedule.npy does not match the domain regenerated from "
            "its executed lock")
    return domain


def legacy_stage2b_candidate_domains() -> dict[str, np.ndarray]:
    """Stage 2B stale module-constant candidate domains (``legacy_test_compatibility``).

    These module constants were never persisted and never used for collection (the v0.3
    development domain was superseded by the executed v0.4 domain; the confirm role was
    never frozen). They are registered explicitly and conservatively avoided — not
    mislabelled as actually-used data. Regenerated with the frozen Stage 2 primitive at
    the module's frozen candidate-pool size.
    """
    return {
        "stage2b_development_stale": make_seed_schedule(
            s2b.COLLECTION_SEEDS["development"], size=s2b.DEV_SEED_SCHEDULE_SIZE),
        "stage2b_confirm_stale": make_seed_schedule(
            s2b.COLLECTION_SEEDS["confirm"], size=s2b.DEV_SEED_SCHEDULE_SIZE),
    }


# --- global historical seed inventory (strict; R2 is history) -------------------------------


def _seeds_in_json_strict(file_path: Path) -> set[int]:
    """Recursively collect ints under seed-like keys in a JSON file; fail on bad reads."""
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"historical JSON product unreadable (fail-loud, no silent skip): "
            f"{file_path}: {exc}") from exc
    found: set[int] = set()

    def walk(node: Any, key_hint: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, str(key).lower())
        elif isinstance(node, (list, tuple)):
            if any(hint in key_hint for hint in _SEED_JSON_HINTS):
                for item in node:
                    if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                        found.add(int(item))
                    else:
                        walk(item, key_hint)
            else:
                for item in node:
                    walk(item, key_hint)
        elif (key_hint in _SEED_SCALAR_KEYS
              and isinstance(node, (int, np.integer)) and not isinstance(node, bool)):
            # scalar seed fields (e.g. "group_seed": 123) count as history too;
            # silently skipping them would undercount the global audit. Only exact
            # seed-value keys qualify — count/size metadata keys never do.
            found.add(int(node))

    walk(payload, "")
    return found


def _seeds_in_jsonl_strict(file_path: Path) -> set[int]:
    """Collect attempt/accepted seeds from a rejection ledger; fail on bad reads."""
    found: set[int] = set()
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"ledger line {lineno} is not an object")
            for key, value in record.items():
                lowered = str(key).lower()
                if any(hint in lowered for hint in ("episode_seed", "group_seed")):
                    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                        raise ValueError(
                            f"ledger seed field {key!r} has non-integer value "
                            f"{value!r} ({file_path} line {lineno})")
                    found.add(int(value))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"historical rejection ledger unreadable (fail-loud): {file_path}: {exc}"
        ) from exc
    return found


def scan_historical_group_seeds(workspace_roots: Sequence[Path]) -> np.ndarray:
    """Union of every historical accepted/attempted/reserved seed (strict, fail-loud).

    Scope (plan section 4.2): all historical accepted seeds, the seeds of every
    attempted/rejected ledger, every persisted candidate ``.npy/.json`` schedule, the
    contract-reserved Stage 2 / Stage 4C / Stage 4C Plus / R2 domains (R2's FULL
    executed 60,000 domain resolved from its executed lock), and the Stage 2B stale
    module-constant candidate domains registered as ``legacy_test_compatibility``.

    Only this stage's own namespace is excluded (resume safety). R2's namespace is
    scanned as history; there is NO blanket ``stage4c`` string exclusion. Any unreadable
    historical product raises — ``except: continue`` is forbidden here.
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
            files_visited += 1
            normalized = _normalised(file_path)
            if any(token in normalized for token in _OWN_NAMESPACE_TOKENS):
                continue  # this stage's own bound products only
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
                collected.update(_seeds_in_json_strict(file_path))
            elif suffix == ".jsonl":
                collected.update(_seeds_in_jsonl_strict(file_path))
    if files_visited == 0:
        raise ValueError(
            "empty workspace: an empty scan cannot evidence zero seed overlap; "
            "fail loudly instead of attesting a clean audit")
    # Contract-reserved domains (never-persisted or persisted; set union dedupes).
    reserved: dict[str, np.ndarray] = {
        "stage2_confirm_frozen": make_seed_schedule(
            s2.COLLECTION_SEED_CONFIRM, size=s2.SEED_SCHEDULE_SIZE),
        "stage4c_confirm_reserved": make_seed_schedule(
            s4c.CONFIRM_SEED_ENTROPY, size=s4c.CONFIRM_SCHEDULE_SIZE),
        "stage4cp_dev_reserved": make_seed_schedule(
            s4cp.DEV_SEED_ENTROPY, size=s4cp.DEV_SCHEDULE_SIZE),
        "stage4cp_confirm_reserved": make_seed_schedule(
            s4cp.CONFIRM_SEED_ENTROPY, size=s4cp.CONFIRM_SCHEDULE_SIZE),
        "stage4cr2_executed_full_domain": r2_reserved_seed_domain(),
    }
    for name, domain in reserved.items():
        collected.update(int(v) for v in domain)
    for name, domain in legacy_stage2b_candidate_domains().items():
        collected.update(int(v) for v in domain)
    if not collected:
        raise ValueError("no historical seeds found in the workspace")
    return np.array(sorted(collected), dtype=np.int64)


def seed_inventory_components(workspace_roots: Sequence[Path]) -> dict[str, Any]:
    """Per-component breakdown of the global inventory (for historical_seed_inventory.json)."""
    components: dict[str, Any] = {}
    inventory = scan_historical_group_seeds(workspace_roots)
    components["total_unique_seeds"] = int(inventory.size)
    components["inventory_sha256"] = hashlib.sha256(inventory.tobytes()).hexdigest()
    components["r2_executed_domain"] = {
        "entropy": 2026092880, "size": 60000,
        "source": str(_resolve(R2_DEVELOPMENT_DIR) / "protocol_lock.json"),
        "source_sha256": R2_PINNED_SOURCES["protocol_lock.json"],
        "fully_contained": bool(np.isin(r2_reserved_seed_domain(), inventory).all()),
    }
    components["legacy_test_compatibility"] = {
        name: {"size": int(domain.size),
               "note": "Stage 2B stale module-constant candidate domain; never "
                       "persisted, never used for collection; conservatively avoided"}
        for name, domain in legacy_stage2b_candidate_domains().items()
    }
    return components


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
    :func:`assert_seed_audit_clean`. Any overlap is a plan-4.2 hard stop requiring a
    whole-domain entropy version revision; single-seed substitution is forbidden.
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
            f"seed(s) {list(audit.overlap[:10])}; plan section 4.2 requires a "
            "whole-domain entropy version revision, never single-seed substitution")


# --- §2.1 R2 authoritative source verification -----------------------------------------------


def verify_r2_sources(r2_dev_dir: str | Path | None = None) -> dict[str, Any]:
    """Verify every §2.1 R2 authoritative source against the pinned SHA256 table.

    The ten pinned files are re-hashed; the five Direct checkpoints are located ONLY
    under ``<r2_dev>/models/`` (paths relocated, never guessed from receipts), hashed,
    verified against ``final_selection.json`` and cross-checked against
    ``model_inventory.json``. Any missing file or hash drift raises (fail-loud).
    """
    r2_dev = _resolve(r2_dev_dir or R2_DEVELOPMENT_DIR)
    pinned: dict[str, dict[str, str]] = {}
    for name, expected in R2_PINNED_SOURCES.items():
        path = r2_dev / name
        if not path.is_file():
            raise PreflightError(f"missing R2 authoritative source: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise PreflightError(
                f"R2 authoritative source hash drift: {name}; expected {expected}, "
                f"got {digest}")
        pinned[name] = {"path": str(path), "sha256": digest}

    final = _read_json(r2_dev / "final_selection.json")
    inventory = _read_json(r2_dev / "model_inventory.json")
    checkpoints: dict[str, dict[str, str]] = {}
    for seed_text, record in final["checkpoints"].items():
        expected_final = str(record["sha256"])
        expected_inventory = str(inventory["checkpoint_sha256"][seed_text])
        if expected_final != expected_inventory:
            raise PreflightError(
                f"final_selection/model_inventory checkpoint hash mismatch for seed "
                f"{seed_text}")
        relocated = r2_dev / "models" / Path(record["path"]).name
        if not relocated.is_file():
            raise PreflightError(f"missing frozen Direct checkpoint: {relocated}")
        digest = sha256_file(relocated)
        if digest != expected_final:
            raise PreflightError(
                f"frozen Direct checkpoint hash drift: {relocated}; expected "
                f"{expected_final}, got {digest}")
        checkpoints[str(seed_text)] = {
            "path": str(relocated),
            "sha256": digest,
            "final_selection_sha256": expected_final,
            "model_inventory_sha256": expected_inventory,
        }
    if sorted(checkpoints) != sorted(str(s) for s in MODEL_SEEDS):
        raise PreflightError("final_selection checkpoints do not cover the frozen five seeds")
    return {"r2_development_dir": str(r2_dev), "pinned_files": pinned,
            "checkpoints": checkpoints}


# --- config validation -----------------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Stage 4C-R2 confirm config; raise before any artifact is produced."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("scene") != "merge" or checked.get("env_id") != ENV_ID:
        raise ValueError("Stage 4C-R2 confirm is merge-v0 only")
    if checked.get("purpose") != PURPOSE_CONFIRM:
        raise ValueError(f"purpose must be {PURPOSE_CONFIRM!r}")
    if checked.get("stage_namespace") != STAGE_NAMESPACE:
        raise ValueError(f"stage_namespace must be {STAGE_NAMESPACE!r}")
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
    for key, expected in (("idle_maintain", 0.45), ("faster", 0.18),
                          ("slower", 0.18), ("other", 0.095)):
        if float(weights.get(key, -1.0)) != expected:
            raise ValueError(f"action_weights.{key} must be {expected}")
    for key, expected in (
        ("n_actions", N_ACTIONS),
        ("history_len", HISTORY_LEN), ("horizon", HORIZON),
        ("factual_suffix_steps", FACTUAL_SUFFIX_STEPS),
        ("t_star_low", T_STAR_LOW), ("t_star_high", T_STAR_HIGH),
        ("simulation_frequency", SIMULATION_FREQUENCY),
        ("policy_frequency", POLICY_FREQUENCY),
        ("vehicles_count", VEHICLES_COUNT), ("duration", ENV_DURATION),
        ("observed_vehicles_count", OBSERVED_VEHICLES_COUNT),
        ("object_capacity", OBJECT_CAPACITY), ("n_object_features", N_OBJECT_FEATURES),
        ("confirm_accepted_groups", CONFIRM_ACCEPTED_GROUPS),
        ("confirm_seed_entropy", CONFIRM_SEED_ENTROPY),
        ("confirm_schedule_size", CONFIRM_SCHEDULE_SIZE),
        ("smoke_accepted_groups", SMOKE_ACCEPTED_GROUPS),
        ("smoke_seed_entropy", SMOKE_SEED_ENTROPY),
        ("smoke_schedule_size", SMOKE_SCHEDULE_SIZE),
        ("bootstrap_entropy", BOOTSTRAP_ENTROPY),
        ("constant_random_salt", CONSTANT_RANDOM_SALT),
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("n_competitors", N_COMPETITORS), ("token_dim", TOKEN_DIM),
        ("global_dim", GLOBAL_DIM),
        ("direct_hidden_width", DIRECT_HIDDEN_WIDTH),
        ("expected_trainable_parameters", EXPECTED_TRAINABLE_PARAMETERS),
        ("idle_plan_index", IDLE_PLAN_INDEX),
        ("t_star_stratum_split", T_STAR_STRATUM_SPLIT),
        ("determinism_checks", 32),
        ("adapter_consistency_groups", 32),
    ):
        if int(checked.get(key, -1)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    for key, expected in (
        ("confidence", CONFIDENCE), ("dropout", DROPOUT),
        ("high_regret_threshold", HIGH_REGRET_THRESHOLD),
        ("regret_clamp_tol", REGRET_CLAMP_TOL),
        ("coverage", COVERAGE),
        ("interaction_threshold", INTERACTION_THRESHOLD),
        ("baseline_tie_tolerance", BASELINE_TIE_TOLERANCE),
    ):
        if float(checked.get(key, -9.0)) != expected:
            raise ValueError(f"{key} drifted from {expected}")
    if checked.get("confirm_schedule_raw_sha256") != CONFIRM_SCHEDULE_RAW_SHA256:
        raise ValueError("confirm_schedule_raw_sha256 drifted from the approved hash")
    if checked.get("smoke_schedule_raw_sha256") != SMOKE_SCHEDULE_RAW_SHA256:
        raise ValueError("smoke_schedule_raw_sha256 drifted from the approved hash")
    if checked.get("architecture", ARCH_DIRECT) != ARCH_DIRECT:
        raise ValueError(f"this stage scores only the frozen {ARCH_DIRECT}")
    if [int(s) for s in checked.get("model_seeds", [])] != list(MODEL_SEEDS):
        raise ValueError(f"model_seeds must be {list(MODEL_SEEDS)}")
    if list(checked.get("response_predictor_seeds", [])) != list(RESPONSE_PREDICTOR_SEEDS):
        raise ValueError("response_predictor_seeds drifted from the frozen five")
    if list(checked.get("expert_router_seeds", [])) != list(EXPERT_ROUTER_SEEDS):
        raise ValueError("expert_router_seeds drifted from the frozen five pairs")
    if checked.get("primary_baseline") != PRIMARY_BASELINE:
        raise ValueError(f"primary_baseline must be {PRIMARY_BASELINE!r} (validation-frozen)")
    if list(checked.get("standard_baselines", [])) != list(STANDARD_BASELINES):
        raise ValueError(f"standard_baselines must be {list(STANDARD_BASELINES)}")
    if list(checked.get("reference_baselines", [])) != list(REFERENCE_BASELINES):
        raise ValueError(f"reference_baselines must be {list(REFERENCE_BASELINES)}")
    if list(checked.get("ablations", [])) != list(ABLATIONS):
        raise ValueError(f"ablations must be {list(ABLATIONS)}")
    if checked.get("primary_score_definition") != PRIMARY_SCORE_DEFINITION:
        raise ValueError("primary_score_definition drifted from the frozen definition")
    if checked.get("ensemble_definition") != ENSEMBLE_DEFINITION:
        raise ValueError("ensemble_definition must be the arithmetic mean of five seeds")
    gate_risk = checked.get("gate_risk", {})
    for key, expected in (
        ("i_aurc_min", GATE_RISK_I_AURC_MIN),
        ("i_regret80_min", GATE_RISK_I_REGRET80_MIN),
        ("strata_i_aurc_floor", GATE_RISK_STRATA_I_AURC_FLOOR),
        ("spearman_margin_ceiling", GATE_RISK_SPEARMAN_MARGIN_CEILING),
    ):
        if float(gate_risk.get(key, -9.0)) != expected:
            raise ValueError(f"gate_risk.{key} drifted from {expected}")
    if int(gate_risk.get("min_positive_seed_metrics", -1)) != GATE_RISK_MIN_POSITIVE_SEED_METRICS:
        raise ValueError("gate_risk.min_positive_seed_metrics drifted")
    if [float(x) for x in gate_risk.get("trim_fractions", [])] != list(GATE_RISK_TRIM_FRACTIONS):
        raise ValueError("gate_risk.trim_fractions drifted")
    gate_task = checked.get("gate_task", {})
    for key, expected in (
        ("min_high_regret_frac", GATE_TASK_MIN_HIGH_REGRET_FRAC),
        ("regret_eps", GATE_TASK_REGRET_EPS), ("min_oracle_frac", GATE_TASK_MIN_ORACLE_FRAC),
    ):
        if float(gate_task.get(key, -9.0)) != expected:
            raise ValueError(f"gate_task.{key} drifted from {expected}")
    for key, expected in (("min_distinct_oracle", GATE_TASK_MIN_DISTINCT_ORACLE),
                          ("min_stratum_groups", GATE_TASK_MIN_STRATUM_GROUPS)):
        if int(gate_task.get(key, -1)) != expected:
            raise ValueError(f"gate_task.{key} drifted from {expected}")
    roots = checked.get("action_stream_roots", {})
    for role in CONFIRM_DATA_ROLES:
        if [int(v) for v in roots.get(role, [])] != list(ACTION_STREAM_ROOTS[role]):
            raise ValueError(f"action_stream_roots.{role} drifted from {ACTION_STREAM_ROOTS[role]}")
    pinned = checked.get("r2_pinned_sources", {})
    if {str(k): str(v) for k, v in pinned.items()} != dict(R2_PINNED_SOURCES):
        raise ValueError("r2_pinned_sources drifted from the plan-2.1 pinned SHA256 table")
    for key in ("run_dir", "data_root", "r2_development_dir", "r2_train_data",
                "stage3r_run_dir", "stage3b_binding", "scaler_path"):
        if key not in checked:
            raise ValueError(f"config missing required path key: {key}")
    paths = checked.get("paths", {})
    for key in ("protocol_dir", "protocol_lock", "source_inventory", "code_inventory",
                "historical_seed_inventory", "seed_audit", "candidate_seed_schedule",
                "inference_inputs", "confirmation_truth", "rejection_ledger",
                "inference_predictions", "inference_risk_scores", "inference_receipt",
                "determinism_receipt", "score_freeze_receipt", "confirmation_used_lock",
                "collection_receipt", "confirm_group_quantities",
                "confirm_group_quantities_receipt", "confirmatory_report",
                "confirmatory_decision"):
        if key not in paths:
            raise ValueError(f"config.paths missing required key: {key}")
    truth = _normalised(paths["confirmation_truth"])
    inputs = _normalised(paths["inference_inputs"])
    if "confirmation_truth" not in truth:
        raise ValueError("paths.confirmation_truth must name the sealed truth file")
    if "confirmation_truth" in inputs or "truth" in inputs:
        raise ValueError("paths.inference_inputs must not reference the truth file")
    if PURPOSE_CONFIRM not in truth or PURPOSE_CONFIRM not in inputs:
        raise ValueError("data paths must live under this stage's data namespace")
    return checked


# --- data access guard ------------------------------------------------------------------------


def assert_allowed_confirm_data(
    path: str | Path,
    *,
    role: str | None = None,
    file_sha256: str | None = None,
) -> None:
    """Raise ``ValueError`` if ``path`` is a forbidden input for this stage.

    The only legal data lives under this stage's own data namespace. R2's namespace is
    now HISTORY (refused as a role input), and every permanently-forbidden prior-stage
    product (Stage 4C screen/selection, Stage 4C Plus Phase 0, Stage 4B confirm, sealed
    Stage 2/3 sets) stays refused by the inherited three-layer guard.
    """
    s3d.assert_not_sealed_stage2(path, file_sha256=file_sha256)
    logical = _normalised(path)
    for token in s4c._FORBIDDEN_DATA_TOKENS:
        for segment in logical.split("/"):
            if segment.startswith(token):
                raise ValueError(
                    f"forbidden confirm-stage data source (segment {segment!r}): {path}")
    if "stage4c_r2" in logical or s4cr2.PURPOSE_DEV in logical:
        raise ValueError(
            f"R2 development products are historical registry evidence only, never a "
            f"confirm-stage role input: {path}")
    if s4c.PURPOSE_DEV in logical or s4c.PURPOSE_CONFIRM in logical:
        raise ValueError(f"Stage 4C multi-action products are permanently read-only: {path}")
    if s4cr2.is_stage4cp_phase0(path):
        raise ValueError(f"Stage 4C Plus Phase 0 products never enter this stage: {path}")
    if role is not None and role not in CONFIRM_DATA_ROLES:
        raise ValueError(f"unknown Stage 4C-R2 confirm data role: {role!r}")
    if PURPOSE_CONFIRM not in logical:
        raise ValueError(f"not an allowed Stage 4C-R2 confirm data source: {path}")


# --- source / code inventories (plan section 5.3) ---------------------------------------------


def build_source_inventory(r2_dev_dir: str | Path | None = None) -> dict[str, Any]:
    """Expand every frozen source into a per-file path/hash listing.

    Covers the ten §2.1 pinned R2 files, the five Direct checkpoints (relocated then
    hash-verified), the underlying five Full/History/end-to-end mixture checkpoints,
    the five response predictors (explicit paths from the Stage 3B binding receipt),
    the branch risk models, and both scalers/binding. No "old chain passed" blanket
    entries.
    """
    from ac_twin import stage4_protocol as s4p
    r2_dev = _resolve(r2_dev_dir or R2_DEVELOPMENT_DIR)
    sources = verify_r2_sources(r2_dev)
    files: dict[str, dict[str, str]] = {}
    for name, entry in sources["pinned_files"].items():
        files[f"r2_development/{name}"] = dict(entry)
    for seed_text, entry in sources["checkpoints"].items():
        files[f"r2_development/models/direct_seed{seed_text}.pt"] = {
            "path": entry["path"], "sha256": entry["sha256"]}

    def _register(label: str, path: Path) -> None:
        if not path.is_file():
            raise PreflightError(f"frozen chain source missing: {path}")
        files[label] = {"path": str(path), "sha256": sha256_file(path)}

    # frozen Stage 4A/4B/3R sources with pinned hashes (verified, not just listed).
    stage4_pinned = {rel: sha for _, (rel, sha) in s4c.STAGE4_FROZEN.items()}
    for rel, expected in stage4_pinned.items():
        path = _resolve(rel)
        _register(rel, path)
        if files[rel]["sha256"] != expected:
            raise PreflightError(f"frozen source hash drift: {rel}")

    # the underlying five pairs: full/history experts + end-to-end mixture routers.
    stage3r_dir = _resolve("experiments/stage3_routing/stage3r")
    for pair_index, router_seed in enumerate(EXPERT_ROUTER_SEEDS):
        for condition in ("full", "history"):
            path = s4p.expert_checkpoint_path(stage3r_dir, pair_index, condition, router_seed)
            _register(f"stage3r/pair{pair_index}_fulltrain_{condition}_seed{router_seed}.pt",
                      path)
        path = s4p.e2e_checkpoint_path(stage3r_dir, pair_index, router_seed)
        _register(f"stage3r/end_to_end_mixture_lc0_lr0_pair{pair_index}_s{router_seed}.pt",
                  path)

    # the five response predictors via the binding receipt's explicit paths (relocated
    # under the current cloud root first, then hash-verified against the binding).
    binding_path = _resolve("experiments/stage3_routing/stage3b/input_binding_receipt.json")
    _register("stage3b/input_binding_receipt.json", binding_path)
    binding = _read_json(binding_path)["receipt_payload"]
    for seed in RESPONSE_PREDICTOR_SEEDS:
        record = binding["checkpoints"][str(seed)]
        path = _relocate(record["path"])
        label = f"stage3b/response_predictor_seed{seed}.pt"
        _register(label, path)
        if files[label]["sha256"] != str(record["file_sha256"]):
            raise PreflightError(
                f"response predictor seed {seed} drifted from its binding file hash")
        if not str(record.get("signature", "")):
            raise PreflightError(
                f"response predictor seed {seed} binding record lost its signature")

    # the frozen branch risk models (combined finalist, geometry-only, response-only).
    _register(
        "stage4a_v3/risk/models/response_only_ridge.pt",
        _resolve("experiments/stage4_reliability/stage4a_v3/risk/models/response_only_ridge.pt"))
    return {
        "plan_id": PLAN_ID,
        "n_files": len(files),
        "files": files,
        "inventory_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest(),
    }


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
            raise FileNotFoundError(f"registered confirm-stage code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


def protocol_code_hashes() -> dict[str, str]:
    """Hashes bound at protocol-lock time: this stage's modules + the reused chain."""
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


# --- preflight ---------------------------------------------------------------------------------


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Full pre-collection preflight: config, §2.1 sources, schedules, global audit.

    Raises (fail-loud) on any config drift, source hash drift, missing historical
    source, unreadable historical product, or any seed overlap. Returns the preflight
    report on success. Produces NO artifacts.
    """
    checked = validate_config(config)
    sources = verify_r2_sources(_resolve(checked["r2_development_dir"]))
    r2_domain = r2_reserved_seed_domain(_resolve(checked["r2_development_dir"]))
    confirm = build_confirm_seed_schedule(checked)
    smoke = build_smoke_seed_schedule(checked)
    mutual = np.intersect1d(confirm, smoke)
    if mutual.size:
        raise PreflightError(
            f"confirm and smoke candidate domains overlap in {mutual.size} seed(s)")
    roots = [_resolve(rel) for rel in checked["historical_scan_roots"]]
    inventory = scan_historical_group_seeds(roots)
    audit_confirm = audit_global_seed_overlap(confirm, roots, inventory=inventory)
    audit_smoke = audit_global_seed_overlap(smoke, roots, inventory=inventory)
    assert_seed_audit_clean(audit_confirm)
    assert_seed_audit_clean(audit_smoke)
    return {
        "plan_id": PLAN_ID,
        "status": STATUS_PREFLIGHT_PASSED,
        "confirm_schedule_raw_sha256": hashlib.sha256(confirm.tobytes()).hexdigest(),
        "smoke_schedule_raw_sha256": hashlib.sha256(smoke.tobytes()).hexdigest(),
        "seed_audit": {
            "confirm": {"n_schedule": audit_confirm.n_schedule,
                        "n_inventory": audit_confirm.n_inventory,
                        "overlap": list(audit_confirm.overlap),
                        "zero_overlap": audit_confirm.zero_overlap},
            "smoke": {"n_schedule": audit_smoke.n_schedule,
                      "n_inventory": audit_smoke.n_inventory,
                      "overlap": list(audit_smoke.overlap),
                      "zero_overlap": audit_smoke.zero_overlap},
            "mutual_overlap": [],
        },
        "r2_executed_domain_size": int(r2_domain.size),
        "r2_sources": sources,
    }


# --- protocol lock ------------------------------------------------------------------------------


def _config_signature(config: Mapping[str, Any]) -> str:
    from ac_twin.checkpointing import canonical_fingerprint
    return canonical_fingerprint(json.loads(json.dumps(config)))


def create_protocol_lock(config: Mapping[str, Any]) -> str:
    """Create this stage's unique protocol lock exactly once; fail closed.

    Runs the full preflight first (any drift/overlap raises before anything is
    written), then writes create-once ``candidate_seed_schedule.npy``,
    ``seed_audit.json``, ``historical_seed_inventory.json``, ``source_inventory.json``,
    ``code_inventory.json`` and ``protocol_lock.json`` under ``paths.protocol_dir``.
    A second call raises ``FileExistsError``; a pure-engineering recovery must keep the
    old lock and append an immutable recovery lock, never overwrite in place.
    """
    checked = validate_config(config)
    paths = checked["paths"]
    protocol_dir = _resolve(paths["protocol_dir"])
    lock_path = _resolve(paths["protocol_lock"])
    if lock_path.exists():
        raise FileExistsError(
            f"Stage 4C-R2 confirm protocol lock already exists: {lock_path}")

    report = preflight(checked)
    confirm = build_confirm_seed_schedule(checked)
    smoke = build_smoke_seed_schedule(checked)
    roots = [_resolve(rel) for rel in checked["historical_scan_roots"]]
    inventory_components = seed_inventory_components(roots)
    source_inventory = build_source_inventory(_resolve(checked["r2_development_dir"]))
    code_inventory = build_code_inventory()

    protocol_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = _resolve(paths["candidate_seed_schedule"])
    tmp = schedule_path.with_name(schedule_path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, confirm)
    Path(tmp).replace(schedule_path)

    write_once_json(_resolve(paths["seed_audit"]), report["seed_audit"])
    write_once_json(_resolve(paths["historical_seed_inventory"]), inventory_components)
    write_once_json(_resolve(paths["source_inventory"]), source_inventory)
    write_once_json(_resolve(paths["code_inventory"]), code_inventory)

    payload = {
        "plan_id": PLAN_ID,
        "stage_namespace": STAGE_NAMESPACE,
        "purpose": PURPOSE_CONFIRM,
        "config_signature": _config_signature(checked),
        "config": json.loads(json.dumps(checked)),
        "n_confirm_groups": CONFIRM_ACCEPTED_GROUPS,
        "confirm_seed_entropy": CONFIRM_SEED_ENTROPY,
        "confirm_schedule_size": CONFIRM_SCHEDULE_SIZE,
        "confirm_schedule_raw_sha256": hashlib.sha256(confirm.tobytes()).hexdigest(),
        "confirm_schedule_file_sha256": sha256_file(schedule_path),
        "smoke_seed_entropy": SMOKE_SEED_ENTROPY,
        "smoke_schedule_size": SMOKE_SCHEDULE_SIZE,
        "smoke_schedule_raw_sha256": hashlib.sha256(smoke.tobytes()).hexdigest(),
        "bootstrap_entropy": BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "confidence": CONFIDENCE,
        "constant_random_salt": CONSTANT_RANDOM_SALT,
        "action_stream_roots": {role: list(root)
                                for role, root in ACTION_STREAM_ROOTS.items()},
        "seed_audit": report["seed_audit"],
        "action_plans": [list(p) for p in ACTION_PLANS],
        "architecture": ARCH_DIRECT,
        "model_seeds": [int(s) for s in MODEL_SEEDS],
        "response_predictor_seeds": list(RESPONSE_PREDICTOR_SEEDS),
        "expert_router_seeds": list(EXPERT_ROUTER_SEEDS),
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "primary_score_definition": PRIMARY_SCORE_DEFINITION,
        "ensemble_definition": ENSEMBLE_DEFINITION,
        "primary_baseline": PRIMARY_BASELINE,
        "standard_baselines": list(STANDARD_BASELINES),
        "reference_baselines": list(REFERENCE_BASELINES),
        "ablations": list(ABLATIONS),
        "high_regret_threshold": HIGH_REGRET_THRESHOLD,
        "coverage": COVERAGE,
        "interaction_threshold": INTERACTION_THRESHOLD,
        "t_star_stratum_split": T_STAR_STRATUM_SPLIT,
        "idle_plan_index": IDLE_PLAN_INDEX,
        "gate_risk": dict(checked["gate_risk"]),
        "gate_task": dict(checked["gate_task"]),
        "determinism": {
            "checks": int(checked["determinism_checks"]),
            "adapter_consistency_groups": int(checked["adapter_consistency_groups"]),
            "device": str(checked.get("device", "cuda")),
            "batch_size": int(checked.get("batch_size", 32)),
            "dtype": "float32",
        },
        "terminal_states": list(TERMINAL_STATES),
        "r2_development_dir": str(_resolve(checked["r2_development_dir"])),
        "r2_pinned_sources": dict(R2_PINNED_SOURCES),
        "candidate_seed_schedule_sha256": sha256_file(schedule_path),
        "seed_audit_sha256": sha256_file(_resolve(paths["seed_audit"])),
        "historical_seed_inventory_sha256": sha256_file(
            _resolve(paths["historical_seed_inventory"])),
        "source_inventory_sha256": sha256_file(_resolve(paths["source_inventory"])),
        "code_inventory_sha256": sha256_file(_resolve(paths["code_inventory"])),
        "code_sha256": code_inventory["code_sha256"],
        "environment": code_inventory["environment"],
    }
    write_once_json(lock_path, payload)
    return str(lock_path)


# --- recovery-lock chain (plan section 8: keep the old lock, append immutable recovery locks) ---


#: recovery locks live next to the protocol lock and chain in numeric order (r1, r2, ...).
RECOVERY_LOCK_GLOB = "recovery_lock_r*.json"
_RECOVERY_LOCK_PREFIX = "recovery_lock_r"


def recovery_lock_paths(lock_path: str | Path) -> list[Path]:
    """The immutable recovery locks chained after ``lock_path``, in chain order.

    Names must be exactly ``recovery_lock_r<N>.json`` with N a positive integer and the
    chain contiguous from r1 — a gap, a duplicate or a malformed name fails closed.
    """
    lock_path = Path(lock_path)

    def _index(path: Path) -> int:
        suffix = path.stem.removeprefix(_RECOVERY_LOCK_PREFIX)
        if not suffix.isdigit() or int(suffix) < 1:
            raise ValueError(f"invalid recovery lock name: {path.name}")
        return int(suffix)

    indexed = sorted(
        ((_index(path), path) for path in lock_path.parent.glob(RECOVERY_LOCK_GLOB)),
        key=lambda item: item[0])
    indices = [index for index, _ in indexed]
    if indices != list(range(1, len(indexed) + 1)):
        raise ValueError(
            f"the recovery lock chain must be contiguous from r1; got {indices}")
    return [path for _, path in indexed]


def _verify_recovery_lock(recovery: Mapping[str, Any], *, recovery_path: Path,
                          previous_path: Path,
                          expected_code: Mapping[str, str]) -> dict[str, str]:
    """Validate ONE recovery lock against its predecessor state; return the new code set.

    A recovery lock is a pure-engineering repair record: it must name this plan, bind
    the previous lock FILE's SHA256 (the original protocol lock for r1, the previous
    recovery lock otherwise), and re-bind only this stage's own registered code files
    (``PROTOCOL_CODE_FILES``) — a prior-stage frozen file
    (``REUSED_FROZEN_CODE_FILES``) or an unregistered file can never be re-bound. Each
    changed file's ``before_sha256`` must equal the chained state it replaces, and the
    recovery lock's declared ``code_sha256`` must be exactly the chained set with the
    declared changes applied. Any inconsistency fails closed.
    """
    if recovery.get("plan_id") != PLAN_ID:
        raise ValueError(f"recovery lock plan_id must be {PLAN_ID!r}: {recovery_path}")
    if recovery.get("kind") != "pure_engineering_recovery":
        raise ValueError(
            f"recovery lock kind must be 'pure_engineering_recovery': {recovery_path}")
    bound = str(recovery.get("binds_previous_lock_sha256", ""))
    if not bound or bound != sha256_file(previous_path):
        raise ValueError(
            f"recovery lock {recovery_path.name} does not bind the previous lock's "
            f"hash ({previous_path.name})")
    changed = recovery.get("changed_files")
    if not isinstance(changed, Mapping) or not changed:
        raise ValueError(f"recovery lock {recovery_path.name} carries no changed_files")
    new_code = dict(expected_code)
    for rel, record in changed.items():
        if rel in REUSED_FROZEN_CODE_FILES:
            raise ValueError(
                f"recovery lock {recovery_path.name} re-binds the frozen prior-stage "
                f"file {rel}; frozen files can never change")
        if rel not in PROTOCOL_CODE_FILES:
            raise ValueError(
                f"recovery lock {recovery_path.name} re-binds unregistered code file "
                f"{rel}")
        if rel not in expected_code:
            raise ValueError(
                f"recovery lock {recovery_path.name} names a file absent from the "
                f"chained code set: {rel}")
        if not isinstance(record, Mapping):
            raise ValueError(
                f"recovery lock {recovery_path.name} changed_files[{rel}] must be a "
                "mapping")
        before = str(record.get("before_sha256", ""))
        after = str(record.get("after_sha256", ""))
        if before != expected_code[rel]:
            raise ValueError(
                f"recovery lock {recovery_path.name} before_sha256 of {rel} does not "
                "match the chained state it replaces")
        if not after or len(after) != 64 or any(c not in "0123456789abcdef"
                                                for c in after):
            raise ValueError(
                f"recovery lock {recovery_path.name} after_sha256 of {rel} is not a "
                "SHA256 hex digest")
        if after == before:
            raise ValueError(
                f"recovery lock {recovery_path.name} records a no-op change of {rel}")
        new_code[rel] = after
    declared = recovery.get("code_sha256")
    if not isinstance(declared, Mapping) or dict(declared) != new_code:
        raise ValueError(
            f"recovery lock {recovery_path.name} declared code_sha256 disagrees with "
            "its changed_files applied to the chained state")
    return new_code


def resolve_locked_code_hashes(lock_path: str | Path) -> dict[str, str]:
    """Effective code-hash set bound by the original lock plus its recovery chain.

    The original protocol lock stays the immutable root; each recovery lock must bind
    the previous lock file's SHA256 and may only re-bind this stage's own registered
    code files. With no recovery locks present the effective set is exactly the
    original lock's ``code_sha256``. Fail closed on a missing original lock, a broken
    chain binding, or an inconsistent recovery lock.
    """
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"protocol lock missing: {lock_path}")
    payload = _read_json(lock_path)
    if payload.get("plan_id") != PLAN_ID:
        raise ValueError(f"lock plan_id must be {PLAN_ID!r}")
    code = payload.get("code_sha256")
    if not isinstance(code, Mapping):
        raise ValueError("the protocol lock does not carry a code_sha256 mapping")
    expected = dict(code)
    previous = lock_path
    for recovery_path in recovery_lock_paths(lock_path):
        recovery = _read_json(recovery_path)
        expected = _verify_recovery_lock(
            recovery, recovery_path=recovery_path, previous_path=previous,
            expected_code=expected)
        previous = recovery_path
    return expected


def verify_protocol(lock_path: str | Path) -> dict[str, Any]:
    """Re-derive every lock binding; raise ``ValueError`` on any tampering. Read-only."""
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"protocol lock missing: {lock_path}")
    payload = _read_json(lock_path)
    if payload.get("plan_id") != PLAN_ID:
        raise ValueError(f"lock plan_id must be {PLAN_ID!r}")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("lock payload does not carry the locked config")
    checked = validate_config(config)
    if _config_signature(checked) != payload["config_signature"]:
        raise ValueError("lock config signature mismatch")
    for key, expected in (
        ("confirm_seed_entropy", CONFIRM_SEED_ENTROPY),
        ("confirm_schedule_size", CONFIRM_SCHEDULE_SIZE),
        ("smoke_seed_entropy", SMOKE_SEED_ENTROPY),
        ("smoke_schedule_size", SMOKE_SCHEDULE_SIZE),
        ("n_confirm_groups", CONFIRM_ACCEPTED_GROUPS),
    ):
        if int(payload[key]) != expected:
            raise ValueError(f"lock {key} drifted from {expected}")
    confirm = build_confirm_seed_schedule(checked)
    if hashlib.sha256(confirm.tobytes()).hexdigest() \
            != payload["confirm_schedule_raw_sha256"]:
        raise ValueError("lock confirm schedule raw hash mismatch")
    smoke = build_smoke_seed_schedule(checked)
    if hashlib.sha256(smoke.tobytes()).hexdigest() \
            != payload["smoke_schedule_raw_sha256"]:
        raise ValueError("lock smoke schedule raw hash mismatch")
    schedule_path = _resolve(checked["paths"]["candidate_seed_schedule"])
    if not schedule_path.is_file():
        raise FileNotFoundError(f"locked candidate schedule missing: {schedule_path}")
    if not np.array_equal(np.load(schedule_path).astype(np.int64), confirm):
        raise ValueError("persisted candidate_seed_schedule.npy drifted from the lock")
    if sha256_file(schedule_path) != payload["candidate_seed_schedule_sha256"]:
        raise ValueError("persisted candidate_seed_schedule.npy hash mismatch")
    if _code_hashes(tuple(PROTOCOL_CODE_FILES) + tuple(REUSED_FROZEN_CODE_FILES)) \
            != resolve_locked_code_hashes(lock_path):
        raise ValueError(
            "locked code SHA256 set mismatch (code drift or tampering; no valid "
            "recovery-lock chain binds the current code)")
    for sidecar_key, hash_key in (
        ("seed_audit", "seed_audit_sha256"),
        ("historical_seed_inventory", "historical_seed_inventory_sha256"),
        ("source_inventory", "source_inventory_sha256"),
        ("code_inventory", "code_inventory_sha256"),
    ):
        sidecar = _resolve(checked["paths"][sidecar_key])
        if not sidecar.is_file():
            raise FileNotFoundError(f"locked inventory missing: {sidecar}")
        if sha256_file(sidecar) != payload[hash_key]:
            raise ValueError(f"locked inventory drifted: {sidecar}")
    source_inventory = _read_json(_resolve(checked["paths"]["source_inventory"]))
    for label, entry in source_inventory["files"].items():
        path = Path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"inventoried frozen source missing: {path}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"inventoried frozen source hash drift: {label} ({path})")
    inventory = scan_historical_group_seeds(
        [_resolve(rel) for rel in checked["historical_scan_roots"]])
    for role, schedule in (("confirm", confirm), ("smoke", smoke)):
        audit = audit_global_seed_overlap(schedule, (), inventory=inventory)
        assert_seed_audit_clean(audit)
        recorded = payload["seed_audit"][role]
        if not recorded["zero_overlap"] or recorded["overlap"]:
            raise ValueError(f"lock records a non-clean {role} seed audit")
    return payload


# Reused frozen helpers re-exported for the inference/collection layers (read-only).
materialize_action_plan = s4c.materialize_action_plan
build_inference_mask = s4c.build_inference_mask


__all__ = [
    "ABLATIONS",
    "ACTION_PLANS",
    "ACTION_STREAM_ROOTS",
    "ARCH_DIRECT",
    "BASELINE_TIE_TOLERANCE",
    "BOOTSTRAP_ENTROPY",
    "BOOTSTRAP_RESAMPLES",
    "CF_ACTIONS",
    "CF_DURATIONS",
    "CONFIG_PATH",
    "CONFIDENCE",
    "CONFIRM_ACCEPTED_GROUPS",
    "CONFIRM_DATA_ROLES",
    "CONFIRM_SCHEDULE_RAW_SHA256",
    "CONFIRM_SCHEDULE_SIZE",
    "CONFIRM_SEED_ENTROPY",
    "CONSTANT_RANDOM_SALT",
    "COVERAGE",
    "DIRECT_HIDDEN_WIDTH",
    "DROPOUT",
    "EGO_SLOT",
    "ENV_DURATION",
    "ENV_ID",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "EXPERT_ROUTER_SEEDS",
    "FACTUAL_SUFFIX_STEPS",
    "GATE_RISK_I_AURC_MIN",
    "GATE_RISK_I_REGRET80_MIN",
    "GATE_RISK_MIN_POSITIVE_SEED_METRICS",
    "GATE_RISK_SPEARMAN_MARGIN_CEILING",
    "GATE_RISK_STRATA_I_AURC_FLOOR",
    "GATE_RISK_TRIM_FRACTIONS",
    "GATE_TASK_MIN_DISTINCT_ORACLE",
    "GATE_TASK_MIN_HIGH_REGRET_FRAC",
    "GATE_TASK_MIN_ORACLE_FRAC",
    "GATE_TASK_MIN_STRATUM_GROUPS",
    "GATE_TASK_REGRET_EPS",
    "GLOBAL_DIM",
    "HIGH_REGRET_THRESHOLD",
    "HISTORY_LEN",
    "HORIZON",
    "IDLE_PLAN_INDEX",
    "INTERACTION_THRESHOLD",
    "MODEL_SEEDS",
    "N_ACTIONS",
    "N_COMPETITORS",
    "N_OBJECT_FEATURES",
    "OBJECT_CAPACITY",
    "OBSERVED_VEHICLES_COUNT",
    "PLAN_ID",
    "POLICY_FREQUENCY",
    "PRIMARY_BASELINE",
    "PRIMARY_SCORE_DEFINITION",
    "ENSEMBLE_DEFINITION",
    "PROTOCOL_CODE_FILES",
    "PURPOSE_CONFIRM",
    "PreflightError",
    "R2_DEVELOPMENT_DIR",
    "R2_PINNED_SOURCES",
    "RECOVERY_LOCK_GLOB",
    "REFERENCE_BASELINES",
    "REGRET_CLAMP_TOL",
    "RESPONSE_PREDICTOR_SEEDS",
    "REUSED_FROZEN_CODE_FILES",
    "ROOT",
    "SEED_HIGH_INCLUSIVE",
    "SEED_LOW",
    "SIMULATION_FREQUENCY",
    "SMOKE_ACCEPTED_GROUPS",
    "SMOKE_SCHEDULE_RAW_SHA256",
    "SMOKE_SCHEDULE_SIZE",
    "SMOKE_SEED_ENTROPY",
    "STAGE_NAMESPACE",
    "STANDARD_BASELINES",
    "STATUS_COLLECTION_INCOMPLETE",
    "STATUS_CONFIRMED",
    "STATUS_INVALID",
    "STATUS_MODEL_INSUFFICIENT",
    "STATUS_NOT_CONFIRMED",
    "STATUS_PREFLIGHT_PASSED",
    "STATUS_RESOURCE_AT_RISK",
    "STATUS_TASK_DEGENERATE",
    "SeedAudit",
    "TERMINAL_STATES",
    "TOKEN_DIM",
    "T_STAR_HIGH",
    "T_STAR_LOW",
    "T_STAR_STRATUM_SPLIT",
    "VEHICLES_COUNT",
    "action_stream_root",
    "assert_allowed_confirm_data",
    "assert_seed_audit_clean",
    "audit_global_seed_overlap",
    "build_code_inventory",
    "build_confirm_seed_schedule",
    "build_inference_mask",
    "build_smoke_seed_schedule",
    "build_source_inventory",
    "create_protocol_lock",
    "legacy_stage2b_candidate_domains",
    "make_candidate_seed_schedule",
    "materialize_action_plan",
    "preflight",
    "protocol_code_hashes",
    "r2_reserved_seed_domain",
    "recovery_lock_paths",
    "resolve_locked_code_hashes",
    "scan_historical_group_seeds",
    "seed_inventory_components",
    "validate_config",
    "verify_protocol",
    "verify_r2_sources",
    "write_once_json",
]
