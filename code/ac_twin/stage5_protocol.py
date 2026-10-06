"""Stage 5 protocol layer: config freeze, old-confirm triple rejection, source
binding with live SHA256 verification, global seed audit and create-once locks.

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`` sections 3-4 and Task 2.

Stage 5 trains only published-method risk/rejection modules on the Stage 4C-R2
development train/validation roles; every old Merge/OOD/MetaDrive confirm per-group
product and the consumed R2 screen are permanently forbidden. Rejection is
three-layer (plan 4.2): normalised path, logical role, and registered content
SHA256. The comparison holdout is created only after the full method/baseline
freeze (Task 5); this module provides the seed-schedule, audit and lock machinery.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin import stage4c_protocol as s4c
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4cr2_protocol as s4cr2
from ac_twin.stage1_data import sha256_file
from ac_twin.stage2_protocol import make_seed_schedule

PLAN_ID = "AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3"
PURPOSE = "stage5_sota_comparison"
ROOT = Path(__file__).resolve().parents[2]

MODEL_SEEDS = (8101, 8202, 8303, 8404, 8505)
N_ACTIONS = 10
COVERAGE_MAIN = 0.80
POWER_CANDIDATE_N = (2000, 3000, 4000, 6000)
SEED_LOW = s4cr2.SEED_LOW
SEED_HIGH_INCLUSIVE = s4cr2.SEED_HIGH_INCLUSIVE

R2_DEVELOPMENT_DIR = "experiments/stage4_reliability/stage4c_r2/development"
DATA_DIR = "data/object_centric_stage/stage4cr2_merge_multiaction_dev"

# Plan 4.2: permanently forbidden one-shot / confirm resources (whole dirs).
_FORBIDDEN_DIR_TOKENS = (
    "stage4cr2_confirm",
    "stage4d_ood",
    "stage4d_d2f",
    "stage4b",
)
# Consumed R2 screen products (permanent read-only, never a Stage 5 input).
_FORBIDDEN_FILE_TOKENS = (
    "object_twinworld_stage4cr2_screen",
    "screen_group_quantities",
    "screen_direct_scores",
    "screen_predictions",
    "screen_scores",
    "development_screen_report",
    "screen_adjudication_report",
)

_ROLES = ("train", "validation")

_CONFIG_EXPECTED: dict[str, Any] = {
    "plan_id": PLAN_ID,
    "model_seeds": list(MODEL_SEEDS),
    "n_actions": N_ACTIONS,
    "coverage_main": COVERAGE_MAIN,
    "high_regret_threshold": 0.01,
    "materiality_shrink": 0.75,
    "materiality_i_aurc_min": 0.05,
    "materiality_i_regret80_min": 0.03,
    "power_target": 0.90,
    "power_candidate_n": list(POWER_CANDIDATE_N),
    "power_bootstrap_draws": 10000,
    "power_monte_carlo": 10000,
    "adjudication_bootstrap_draws": 10000,
    "tost_i_aurc": [-0.05, 0.05],
    "tost_i_regret80": [-0.03, 0.03],
    "determinism_checks": 32,
    "holdout_seed_entropy": 2026093073,
    "holdout_schedule_size": 40000,
    "power_bootstrap_entropy": 2026092721,
    "power_mc_entropy": 2026092722,
    "adjudication_bootstrap_entropy": 2026092723,
}


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _normalised(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- config freeze -----------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen Stage 5 config; any drift from the approved values fails."""
    checked = dict(config)
    for key, expected in _CONFIG_EXPECTED.items():
        if key not in checked:
            raise ValueError(f"Stage 5 config is missing frozen key: {key}")
        if checked[key] != expected:
            raise ValueError(
                f"Stage 5 config drift on {key!r}: expected {expected!r}, "
                f"got {checked[key]!r}")
    for key in ("holdout_seed_entropy", "power_bootstrap_entropy", "power_mc_entropy",
                "adjudication_bootstrap_entropy"):
        if int(checked[key]) <= 0:
            raise ValueError(f"Stage 5 config entropy {key!r} must be positive")
    if int(checked["holdout_schedule_size"]) < 4 * max(POWER_CANDIDATE_N):
        raise ValueError("holdout candidate schedule must comfortably cover max n")
    return checked


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(_read_json(path))


def config_signature(config: Mapping[str, Any]) -> str:
    """Deterministic content signature of the validated config."""
    import hashlib
    checked = validate_config(config)
    blob = json.dumps(checked, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --- old-confirm triple rejection ------------------------------------------------------


def _is_forbidden_path(path: str | Path) -> bool:
    logical = _normalised(path)
    if PURPOSE in logical:
        return False  # this stage's own products are never forbidden to itself
    segments = logical.split("/")
    for token in _FORBIDDEN_DIR_TOKENS:
        if any(segment.startswith(token) for segment in segments):
            return True
    name = segments[-1] if segments else ""
    return any(name.startswith(token) for token in _FORBIDDEN_FILE_TOKENS)


def forbidden_source_sha256() -> dict[str, str]:
    """Registered content SHA256 of canonical forbidden confirm artifacts.

    Small canonical files only (reports, locks, receipts, handoffs): hashing the
    per-group NPZ payloads of every consumed confirmation would be needlessly slow,
    and the registered set is what the content layer of the triple rejection needs.
    """
    registry: dict[str, str] = {}
    bases = [
        ROOT / "experiments/stage4_reliability/stage4cr2_confirm",
        ROOT / "experiments/stage4_reliability/stage4d_ood",
        ROOT / "experiments/stage4_reliability/stage4d_d2f/confirm",
        ROOT / "experiments/stage4_reliability/stage4b",
    ]
    for base in bases:
        if not base.is_dir():
            continue
        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() in (".json", ".lock", ".md"):
                try:
                    registry[str(file_path)] = sha256_file(file_path)
                except OSError:
                    continue
    return registry


def assert_allowed_stage5_data(
    path: str | Path,
    *,
    role: str | None = None,
    file_sha256: str | None = None,
) -> None:
    """Raise ``ValueError`` if ``path`` is a forbidden Stage 5 input (plan 4.2).

    Three-layer refusal: normalised path, logical role, registered content SHA256.
    Legal Stage 5 inputs are the R2 development train/validation roles, this stage's
    own purpose directory, and the stage config. Everything under the old
    Merge/OOD/MetaDrive confirm dirs, Stage 4B, and every consumed R2 screen product
    is refused regardless of name.
    """
    if role is not None and role not in _ROLES + ("comparison_holdout",):
        raise ValueError(f"unknown Stage 5 data role: {role!r}")
    if _is_forbidden_path(path):
        raise ValueError(f"forbidden Stage 5 data source (old confirm/screen): {path}")
    if file_sha256 is not None and file_sha256 in set(forbidden_source_sha256().values()):
        raise ValueError(
            f"forbidden old-confirm artifact content (registered SHA256 match): {path}")
    logical = _normalised(path)
    if PURPOSE in logical:
        return
    if "/config/" in logical and "stage5" in logical:
        return
    allowed_r2 = _normalised(ROOT / R2_DEVELOPMENT_DIR)
    allowed_data = _normalised(ROOT / DATA_DIR)
    if logical.startswith(allowed_r2 + "/") or logical.startswith(allowed_data + "/"):
        return
    raise ValueError(f"not an allowed Stage 5 data source: {path}")


# --- source binding --------------------------------------------------------------------


def _r2_bindings(dev_dir: Path) -> dict[str, str]:
    final = _read_json(dev_dir / "final_selection.json")
    return dict(final.get("bindings", {}))


def resolve_sources(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every reused Stage 4C-R2 artifact by role and verify content live.

    Paths are resolved from the project-relative config (cross-computer absolute
    prefixes may vary); each resolved file must exist, pass the forbidden guard and
    reproduce the SHA256 recorded in the frozen R2 receipts (``final_selection.json``
    bindings and ``model_inventory.json`` checkpoint hashes). Any mismatch is a
    hard stop, never a silent re-resolution.
    """
    checked = validate_config(config)
    dev_dir = _resolve(checked["r2_development_dir"])
    if not dev_dir.is_dir():
        raise ValueError(f"R2 development dir not found: {dev_dir}")
    bindings = _r2_bindings(dev_dir)
    inventory = _read_json(dev_dir / "model_inventory.json")

    sources: dict[str, Path] = {
        "train_data": _resolve(checked["train_data"]),
        "validation_data": _resolve(checked["validation_data"]),
        "train_predictions": dev_dir / "train_predictions.npz",
        "train_scores": dev_dir / "train_scores.npz",
        "validation_predictions": dev_dir / "validation_predictions.npz",
        "validation_scores": dev_dir / "validation_scores.npz",
        "final_selection": dev_dir / "final_selection.json",
        "feature_schema": dev_dir / "feature_schema.json",
        "feature_scaler": dev_dir / "feature_scaler.npz",
        "model_inventory": dev_dir / "model_inventory.json",
        "train_collection_receipt": dev_dir / "train_collection_receipt.json",
        "validation_collection_receipt": dev_dir / "validation_collection_receipt.json",
        "train_inference_receipt": dev_dir / "train_inference_receipt.json",
        "validation_inference_receipt": dev_dir / "validation_inference_receipt.json",
    }
    for seed in MODEL_SEEDS:
        sources[f"direct_checkpoint_{seed}"] = dev_dir / "models" / f"direct_seed{seed}.pt"

    expected_sha: dict[str, str] = {
        "train_data": bindings.get("train_data_sha256", ""),
        "validation_data": bindings.get("validation_data_sha256", ""),
        "train_predictions": bindings.get("train_predictions_sha256", ""),
        "train_scores": bindings.get("train_scores_sha256", ""),
        "validation_predictions": bindings.get("validation_predictions_sha256", ""),
        "validation_scores": bindings.get("validation_scores_sha256", ""),
    }
    for seed in MODEL_SEEDS:
        expected_sha[f"direct_checkpoint_{seed}"] = (
            inventory.get("checkpoint_sha256", {}).get(str(seed), ""))

    for key, path in sources.items():
        path = Path(path)
        assert_allowed_stage5_data(path)
        if not path.is_file():
            raise ValueError(f"Stage 5 source missing for role {key!r}: {path}")
        wanted = expected_sha.get(key)
        if wanted:
            actual = sha256_file(path)
            if actual != wanted:
                raise ValueError(
                    f"Stage 5 source {key!r} content drift: expected SHA256 {wanted}, "
                    f"got {actual} ({path})")
    return sources


# --- holdout seed schedule + global audit -----------------------------------------------


def build_holdout_seed_schedule(config: Mapping[str, Any]) -> np.ndarray:
    """Unique int64 candidate seeds for the future comparison holdout.

    The schedule is pure machinery until Task 5 freezes ``selected_n``; it is built
    from the config entropy and must pass the global zero-overlap audit before any
    collection manifest is created (plan 4.3).
    """
    checked = validate_config(config)
    schedule = make_seed_schedule(
        int(checked["holdout_seed_entropy"]), size=int(checked["holdout_schedule_size"]))
    schedule = np.asarray(schedule, dtype=np.int64)
    if schedule.min() < SEED_LOW or schedule.max() > SEED_HIGH_INCLUSIVE:
        raise ValueError("Stage 5 holdout seed schedule out of range")
    if np.unique(schedule).size != schedule.size:
        raise ValueError("Stage 5 holdout seed schedule is not internally unique")
    return schedule


@dataclass(frozen=True)
class SeedAudit:
    n_schedule: int
    n_inventory: int
    overlap: tuple[int, ...]

    @property
    def zero_overlap(self) -> bool:
        return len(self.overlap) == 0


def scan_historical_group_seeds(workspace_roots: Sequence[Path]) -> np.ndarray:
    """Union of every historical accepted/attempted/reserved seed in the workspace.

    Unlike the R2 scanner (which excluded its own stage), Stage 5 counts ALL prior
    stages' products — including stage4c_r2 and the consumed confirm sets — as
    history; only this stage's own purpose dir is excluded so a resume does not
    collide with itself. Confirm NPZs are opened ONLY for their seed arrays
    (inventory duty under plan 4.3's zero-overlap rule); no per-group score, truth
    or quantity is read. The never-persisted prior-stage reserved schedules are
    added by contract, exactly as in the R2 audit.
    """
    collected: set[int] = set()
    for root in workspace_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for file_path in sorted(root.rglob("*")):
            normalized = _normalised(file_path)
            if PURPOSE in normalized:
                continue  # this stage's own products
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".npz":
                    with np.load(file_path, allow_pickle=True) as archive:
                        for key in ("episode_seed", "episode_seeds",
                                    "group_seed", "group_seeds"):
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
    # Reserved-but-never-persisted prior-stage schedules by contract (R2 precedent).
    for entropy, size, maker in (
        (s4cr2.STAGE4C_CONFIRM_SEED_ENTROPY, s4cr2.STAGE4C_CONFIRM_SCHEDULE_SIZE,
         s4c.make_group_seed_schedule),
        (s4cr2.STAGE4CP_DEV_SEED_ENTROPY, s4cr2.STAGE4CP_DEV_SCHEDULE_SIZE,
         s4cp.make_group_seed_schedule),
        (s4cr2.STAGE4CP_CONFIRM_SEED_ENTROPY, s4cr2.STAGE4CP_CONFIRM_SCHEDULE_SIZE,
         s4cp.make_group_seed_schedule),
    ):
        collected.update(int(v) for v in maker(entropy, size))
    if not collected:
        raise ValueError("no historical episode seeds found in the workspace")
    return np.array(sorted(collected), dtype=np.int64)


def audit_global_seed_overlap(schedule: np.ndarray, roots: Sequence[Path]) -> SeedAudit:
    schedule = np.asarray(schedule, dtype=np.int64)
    if schedule.size == 0:
        raise ValueError("seed schedule must be non-empty")
    inventory = scan_historical_group_seeds(roots)
    overlap = np.intersect1d(schedule, inventory)
    return SeedAudit(
        n_schedule=int(schedule.size), n_inventory=int(inventory.size),
        overlap=tuple(int(v) for v in overlap))


def assert_seed_audit_clean(audit: SeedAudit) -> None:
    if not audit.zero_overlap:
        raise ValueError(
            f"Stage 5 holdout schedule overlaps {len(audit.overlap)} historical "
            f"seed(s) (first: {audit.overlap[:5]}); plan 4.3 hard stop — a new plan "
            "version is required, never a seed substitution")


# --- code inventory + create-once lock ---------------------------------------------------


def stage5_code_files() -> tuple[str, ...]:
    """Project-relative files whose content the Stage 5 locks bind."""
    files = [
        "code/ac_twin/stage5_protocol.py",
        "code/ac_twin/stage5_features.py",
        "experiments/stage5_sota_comparison/literature/method_registry.json",
        "experiments/stage5_sota_comparison/literature/compatibility_decision.json",
    ]
    for optional in (
        "code/ac_twin/stage5_methods.py",
        "code/ac_twin/stage5_training.py",
        "code/ac_twin/stage5_selection.py",
        "code/ac_twin/stage5_statistics.py",
        "code/ac_twin/stage5_power.py",
        "code/collect_stage5_merge.py",
        "code/run_stage5.py",
    ):
        if (ROOT / optional).is_file():
            files.append(optional)
    return tuple(files)


def stage5_code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relpath in stage5_code_files():
        path = ROOT / relpath
        if not path.is_file():
            raise ValueError(f"Stage 5 code-inventory file missing: {relpath}")
        hashes[relpath] = sha256_file(path)
    return hashes


def create_protocol_lock(lock_path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a create-once protocol lock. A second create is refused; the existing
    lock must instead be re-verified with :func:`verify_protocol_lock`."""
    lock_path = Path(lock_path)
    if lock_path.exists():
        raise RuntimeError(
            f"protocol lock already exists (create-once): {lock_path}; "
            "call verify_protocol_lock instead")
    record = dict(payload)
    record["lock_type"] = "STAGE5_PROTOCOL_LOCK"
    record["plan_id"] = PLAN_ID
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return lock_path


def verify_protocol_lock(lock_path: str | Path) -> dict[str, Any]:
    """Re-verify a protocol lock: bound files and live code hashes must be unchanged."""
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise ValueError(f"protocol lock missing: {lock_path}")
    record = _read_json(lock_path)
    if record.get("plan_id") != PLAN_ID:
        raise ValueError(f"protocol lock plan drift: {lock_path}")
    for path_text, wanted in dict(record.get("bound_files", {})).items():
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"lock-bound file missing: {path}")
        actual = sha256_file(path)
        if actual != wanted:
            raise ValueError(
                f"lock-bound file drifted: {path} (expected {wanted}, got {actual})")
    live_code = stage5_code_hashes()
    locked_code = dict(record.get("code_hashes", {}))
    for relpath, wanted in locked_code.items():
        actual = live_code.get(relpath)
        if actual != wanted:
            raise ValueError(
                f"Stage 5 code drift after lock: {relpath} "
                f"(expected {wanted}, got {actual})")
    return record


__all__ = [
    "PLAN_ID", "PURPOSE", "ROOT", "MODEL_SEEDS", "N_ACTIONS", "COVERAGE_MAIN",
    "POWER_CANDIDATE_N", "SEED_LOW", "SEED_HIGH_INCLUSIVE",
    "validate_config", "load_config", "config_signature",
    "assert_allowed_stage5_data", "forbidden_source_sha256",
    "resolve_sources", "build_holdout_seed_schedule",
    "SeedAudit", "scan_historical_group_seeds", "audit_global_seed_overlap",
    "assert_seed_audit_clean", "stage5_code_files", "stage5_code_hashes",
    "create_protocol_lock", "verify_protocol_lock",
]
