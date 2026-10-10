"""M1 supervised-baseline ablation protocol: whitelist, identity contract, locks.

Plan ``AC-TWIN-RFER-M1-SUPERVISED-ABLATION-2026-10-07-V0.2``.

Merge development-only component diagnostic (``ADAPTIVE_DEVELOPMENT_ONLY``). No new
collection, no screen/power/confirm. The ONLY legal data are the frozen Stage 4C-R2
development train/validation roles (data npz + frozen predictions + frozen action-risk
scores) resolved through an explicit six-file whitelist and hash-bound to the frozen
R2 receipts. Old screen/confirm per-group products, every Stage 4D platform artifact
and the Stage 5 comparison pipeline are refused.

Cache identity contract (plan 9.2, decision B2-b): new NPZ/JSON caches NEVER persist
raw group seeds. They store ``source_row_index`` + ``group_id`` bound to the source
relative path, the source file SHA256 and ``source_identity_sha256`` (SHA256 of the
source's full ordered int64 identity array, canonical little-endian bytes). The raw
``group_seed`` is read from the whitelisted source in memory only, for alignment, the
fixed main/monitor split and validation. The four historical scanner keys
(``episode_seed``/``episode_seeds``/``group_seed``/``group_seeds``) are forbidden in
every new NPZ; seed-hint JSON fields never hold integers.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from ac_twin import stage5_protocol as s5prot  # read-only reuse: forbidden guard
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = "AC-TWIN-RFER-M1-SUPERVISED-ABLATION-2026-10-07-V0.2"
OUTPUT_DIR = "experiments/paper_revision/m1_supervised_ablation_v02"
ANALYSIS_STATUS = "ADAPTIVE_DEVELOPMENT_ONLY"

R2_DEVELOPMENT_DIR = "experiments/stage4_reliability/stage4c_r2/development"
TRAIN_DATA = "data/object_centric_stage/stage4cr2_merge_multiaction_dev/object_twinworld_stage4cr2_train.npz"
VALIDATION_DATA = "data/object_centric_stage/stage4cr2_merge_multiaction_dev/object_twinworld_stage4cr2_validation.npz"

#: the three new training arms, in the frozen per-seed fit order (plan section 4).
ARMS = ("FULL_MATCHED", "BASELINE3_MATCHED", "NO_RISK_MATCHED")
MODEL_SEEDS = (8101, 8202, 8303, 8404, 8505)
BASELINE3_ORDER = ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD")

# frozen training contract (plan section 5; identical to the R2 mechanics)
DIRECT_HIDDEN_WIDTH = 43
DROPOUT = 0.10
LAMBDA_CLS = 0.25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRAIN_BATCH_SIZE = 128
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 12
EARLY_STOP_TOLERANCE = 1e-10
EXPECTED_TRAINABLE_PARAMETERS = 15740
HIGH_REGRET_THRESHOLD = 0.01
COVERAGE_MAIN = 0.80

# statistics contract (plan sections 6/7)
BOOTSTRAP_ENTROPY = 2026100701
BOOTSTRAP_DRAWS = 10000
MAIN_FAMILY_SIZE = 5
MAIN_CI_QUANTILES = (0.005, 0.995)   # Bonferroni family-95% -> per-contrast 99%
UNCORRECTED_CI_QUANTILES = (0.025, 0.975)
DENOMINATOR_FLOOR = 1e-12
CONTINUITY_TOLERANCE = 1e-8

#: contrasts C1-C5 (plan section 6): (contrast_id, method, reference).
CONTRASTS = (
    ("C1", "FULL_MATCHED", "BASELINE3_MATCHED"),
    ("C2", "FULL_MATCHED", "NO_RISK_MATCHED"),
    ("C3", "BASELINE3_MATCHED", "SEED_VOTE_ENTROPY"),
    ("C4", "BASELINE3_MATCHED", "MARGIN_ONLY"),
    ("C5", "BASELINE3_MATCHED", "SEED_MARGIN_STD"),
)
HEURISTIC_METHODS = ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD")
ALL_METHODS = ("FULL_MATCHED", "BASELINE3_MATCHED", "NO_RISK_MATCHED",
               "FULL_FROZEN_REFERENCE") + HEURISTIC_METHODS

# --- cache identity contract (plan 9.2) ---------------------------------------------------

#: NPZ keys the historical scanners harvest; forbidden in every new cache.
FORBIDDEN_NPZ_KEYS = ("episode_seed", "episode_seeds", "group_seed", "group_seeds")
#: JSON key substrings under which integer lists/scalars would be harvested.
SEED_JSON_HINTS = ("episode_seed", "episode_seeds", "group_seed", "group_seeds",
                   "seed_schedule")
SEED_SCALAR_KEYS = ("episode_seed", "group_seed")

CODE_FILES = (
    "run_m1_ablation.py",
    "ac_twin/m1_ablation_protocol.py",
    "ac_twin/m1_ablation_features.py",
    "ac_twin/m1_ablation_training.py",
    "ac_twin/m1_ablation_statistics.py",
)

#: output must never live inside / shadow these protected trees (plan 9.1 / P0).
PROTECTED_TREES = (
    "experiments/stage4_reliability",
    "experiments/stage5_sota_comparison",
    "experiments/stage1_evidence_value",
    "experiments/stage2_confirmatory",
    "experiments/stage3_routing",
    "data",
    "paper",
)


def _normalised(path: str | Path) -> str:
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- config --------------------------------------------------------------------------------


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the M1 config against the frozen contract; raise before any artifact."""
    checked = dict(config)
    if checked.get("plan_id") != PLAN_ID:
        raise ValueError(f"plan_id must be {PLAN_ID!r}")
    if checked.get("analysis_status") != ANALYSIS_STATUS:
        raise ValueError(f"analysis_status must be {ANALYSIS_STATUS!r}")
    if _normalised(checked.get("output_dir", "")) != _normalised(OUTPUT_DIR):
        raise ValueError(f"output_dir must be {OUTPUT_DIR!r}")
    if [tuple(p) for p in checked.get("action_plans", [])] != \
            [tuple(p) for p in ((0, 1), (0, 3), (1, 1), (1, 3), (2, 1),
                                (2, 3), (3, 1), (3, 3), (4, 1), (4, 3))]:
        raise ValueError("action_plans drifted from the frozen ten-action order")
    for key, expected in (
        ("dev_train_groups", 5000), ("dev_validation_groups", 1500),
        ("dev_seed_entropy", 2026092880), ("internal_monitor_frac", 0.10),
        ("train_batch_size", TRAIN_BATCH_SIZE), ("max_epochs", MAX_EPOCHS),
        ("early_stop_patience", EARLY_STOP_PATIENCE),
        ("bootstrap_draws", BOOTSTRAP_DRAWS), ("bootstrap_entropy", BOOTSTRAP_ENTROPY),
        ("main_family_size", MAIN_FAMILY_SIZE),
        ("expected_trainable_parameters", EXPECTED_TRAINABLE_PARAMETERS),
    ):
        if checked.get(key) != expected:
            raise ValueError(f"{key} drifted from {expected!r}")
    for key, expected in (
        ("learning_rate", LEARNING_RATE), ("weight_decay", WEIGHT_DECAY),
        ("dropout", DROPOUT), ("lambda_cls", LAMBDA_CLS),
        ("early_stop_tolerance", EARLY_STOP_TOLERANCE),
        ("high_regret_threshold", HIGH_REGRET_THRESHOLD),
        ("coverage_main", COVERAGE_MAIN),
    ):
        if float(checked.get(key, -1.0)) != expected:
            raise ValueError(f"{key} drifted from {expected!r}")
    if tuple(int(s) for s in checked.get("model_seeds", ())) != MODEL_SEEDS:
        raise ValueError(f"model_seeds must be {MODEL_SEEDS}")
    if tuple(checked.get("arms", ())) != ARMS:
        raise ValueError(f"arms must be {ARMS}")
    if tuple(checked.get("baseline3_order", ())) != BASELINE3_ORDER:
        raise ValueError(f"baseline3_order must be {BASELINE3_ORDER}")
    if [tuple(c) for c in checked.get("contrasts", [])] != [tuple(c) for c in CONTRASTS]:
        raise ValueError("contrasts drifted from the frozen C1-C5 definitions")
    return checked


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(_read_json(path))


def config_signature(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / "code" / name) for name in CODE_FILES}


# --- sources (explicit six-file whitelist) -------------------------------------------------


def resolve_sources(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve the six whitelisted roles + binding receipts and verify content live.

    Every data-bearing source passes the Stage 5 forbidden guard (old confirm/screen
    refusal) and must reproduce the SHA256 frozen in the R2 ``final_selection.json``
    bindings / ``model_inventory.json`` checkpoint hashes. Any mismatch is a hard stop.
    """
    checked = validate_config(config)
    dev_dir = _resolve(R2_DEVELOPMENT_DIR)
    if not dev_dir.is_dir():
        raise ValueError(f"R2 development dir not found: {dev_dir}")
    bindings = dict(_read_json(dev_dir / "final_selection.json").get("bindings", {}))
    inventory = _read_json(dev_dir / "model_inventory.json")

    sources: dict[str, Path] = {
        "train_data": _resolve(TRAIN_DATA),
        "validation_data": _resolve(VALIDATION_DATA),
        "train_predictions": dev_dir / "train_predictions.npz",
        "train_scores": dev_dir / "train_scores.npz",
        "validation_predictions": dev_dir / "validation_predictions.npz",
        "validation_scores": dev_dir / "validation_scores.npz",
        "final_selection": dev_dir / "final_selection.json",
        "model_inventory": dev_dir / "model_inventory.json",
        "feature_scaler": dev_dir / "feature_scaler.npz",
        "fit_receipt": dev_dir / "models" / "fit_receipt.json",
        "validation_gate_report": dev_dir / "validation_gate_report.json",
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

    actual: dict[str, str] = {}
    for key, path in sources.items():
        path = Path(path)
        s5prot.assert_allowed_stage5_data(path)  # read-only reuse: old confirm/screen refusal
        if not path.is_file():
            raise ValueError(f"M1 source missing for role {key!r}: {path}")
        digest = sha256_file(path)
        actual[key] = digest
        wanted = expected_sha.get(key)
        if wanted and digest != wanted:
            raise ValueError(
                f"M1 source hash mismatch for role {key!r}: {path}\n"
                f"  expected {wanted}\n  actual   {digest}")
    resolve_sources.last_hashes = actual  # type: ignore[attr-defined]
    return sources


def source_group_seed(sources: Mapping[str, Any], role: str) -> np.ndarray:
    """Read the raw group seed from a whitelisted data source (memory only)."""
    if role not in ("train", "validation"):
        raise ValueError(f"unknown M1 role: {role!r}")
    with np.load(Path(sources[f"{role}_data"]), allow_pickle=True) as archive:
        return np.asarray(archive["episode_seed"], dtype=np.int64).ravel()


# --- identity contract ---------------------------------------------------------------------


def identity_digest(identity: np.ndarray) -> str:
    """SHA256 of the source's full ordered identity array (canonical LE int64 bytes)."""
    array = np.ascontiguousarray(np.asarray(identity, dtype="<i8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def make_group_ids(role: str, row_index: np.ndarray) -> np.ndarray:
    return np.asarray([f"{role}:{int(i)}" for i in np.asarray(row_index).ravel()])


def assert_cache_keys_legal(keys: Sequence[str], *, context: str) -> None:
    bad = [k for k in keys if k in FORBIDDEN_NPZ_KEYS]
    if bad:
        raise ValueError(f"{context}: forbidden scanner-visible NPZ keys: {bad}")


def assert_json_seed_free(payload: Any, *, context: str) -> None:
    """Replicate the historical scanner's JSON walk; our JSON must yield zero seeds."""
    found = scanner_json_seeds(payload)
    if found:
        raise ValueError(f"{context}: JSON carries scanner-visible group seeds: {sorted(found)[:5]}")


def scanner_json_seeds(payload: Any) -> set[int]:
    """Mirror of the frozen historical scanner's JSON extraction rules (test/verify use)."""
    found: set[int] = set()

    def walk(node: Any, key_hint: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                walk(value, str(key).lower())
        elif isinstance(node, (list, tuple)):
            if any(hint in key_hint for hint in SEED_JSON_HINTS):
                for item in node:
                    if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                        found.add(int(item))
                    else:
                        walk(item, key_hint)
            else:
                for item in node:
                    walk(item, key_hint)
        elif (key_hint in SEED_SCALAR_KEYS
              and isinstance(node, (int, np.integer)) and not isinstance(node, bool)):
            found.add(int(node))

    walk(payload, "")
    return found


def scanner_npz_seeds(path: str | Path) -> set[int]:
    """Mirror of the scanner's NPZ extraction: the four identity keys as int arrays."""
    found: set[int] = set()
    with np.load(Path(path), allow_pickle=False) as archive:
        for key in FORBIDDEN_NPZ_KEYS:
            if key in archive.files:
                for value in np.asarray(archive[key]).ravel():
                    found.add(int(value))
    return found


def validate_binding(record: Mapping[str, Any], *, role: str, source_path: Path,
                     expected_rows: int, context: str) -> np.ndarray:
    """Verify a cache's identity binding against its live source; return group seeds.

    Re-checks the source file hash, the recorded identity digest, and that the row
    indices are legal, unique, complete and reconstruct exactly the source's groups.
    """
    role = str(role)
    if str(record.get("role")) != role:
        raise ValueError(f"{context}: role mismatch: {record.get('role')!r} != {role!r}")
    actual_sha = sha256_file(source_path)
    if str(record.get("source_sha256")) != actual_sha:
        raise ValueError(f"{context}: source hash drift for {source_path}")
    group_seed = source_group_seed({f"{role}_data": source_path}, role)
    if identity_digest(group_seed) != str(record.get("source_identity_sha256")):
        raise ValueError(f"{context}: source identity digest mismatch for {source_path}")
    row_index = np.asarray(record.get("source_row_index"), dtype=np.int64).ravel()
    if row_index.size != expected_rows or np.unique(row_index).size != row_index.size:
        raise ValueError(f"{context}: row indices not unique/complete for {role}")
    if row_index.min() < 0 or row_index.max() >= group_seed.size:
        raise ValueError(f"{context}: row index out of range for {role}")
    if not np.array_equal(np.sort(row_index), np.arange(group_seed.size)):
        raise ValueError(f"{context}: cache must cover every source row of {role}")
    group_ids = make_group_ids(role, row_index)
    if list(map(str, record.get("group_id", []))) != group_ids.tolist():
        raise ValueError(f"{context}: group_id does not match role + row index")
    return group_seed[row_index]


# --- output isolation & manuscript protection ----------------------------------------------


def output_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(validate_config(config)["output_dir"])


def assert_output_isolation(config: Mapping[str, Any]) -> Path:
    """The output dir must not be an ancestor/descendant/alias of any protected tree."""
    target = output_dir(config).resolve()
    normal = _normalised(target)
    for tree in PROTECTED_TREES:
        protected = _normalised((ROOT / tree).resolve())
        if normal == protected or normal.startswith(protected + "/") \
                or protected.startswith(normal + "/"):
            raise ValueError(f"output dir {target} collides with protected tree {tree}")
    return target


def manuscript_protected_files() -> dict[str, Path]:
    """Files whose hashes are protected for the whole run (plan P0/P3).

    All Markdown/Word artifacts under ``paper/manuscript/`` (Word lock files excepted)
    plus every figure/table file the current manuscript links outside that directory.
    """
    manuscript = ROOT / "paper" / "manuscript"
    protected: dict[str, Path] = {}
    for pattern in ("*.md", "*.docx"):
        for path in sorted(manuscript.glob(pattern)):
            if path.name.startswith("~$"):
                continue
            protected[_normalised(path.relative_to(ROOT))] = path
    draft = manuscript / "AC-Twin_论文初稿_中文.md"
    if draft.is_file():
        import re
        for match in re.findall(r"\]\(([^)]+)\)", draft.read_text(encoding="utf-8")):
            target = (manuscript / match).resolve()
            if target.is_file() and target.suffix.lower() in {".png", ".md", ".csv"}:
                protected[_normalised(target.relative_to(ROOT))] = target
    return protected


def hash_protected_files() -> dict[str, str]:
    return {rel: sha256_file(path) for rel, path in manuscript_protected_files().items()}


def verify_protected_hashes(record: Mapping[str, str]) -> list[str]:
    """Return the list of protected files whose current hash drifted (empty == clean)."""
    drifted: list[str] = []
    for rel, expected in record.items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != expected:
            drifted.append(rel)
    return drifted


# --- protocol lock --------------------------------------------------------------------------


def create_protocol_lock(lock_path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(lock_path)
    write_once_json(target, dict(payload))
    return target


def verify_protocol_lock(lock_path: str | Path) -> dict[str, Any]:
    lock = _read_json(lock_path)
    for name, expected in lock.get("code_hashes", {}).items():
        actual = sha256_file(ROOT / "code" / name)
        if actual != expected:
            raise ValueError(f"code file drifted from the protocol lock: {name}")
    for rel, expected in lock.get("protected_manuscript_hashes", {}).items():
        path = ROOT / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"protected manuscript file drifted: {rel}")
    return lock


__all__ = [
    "ALL_METHODS", "ANALYSIS_STATUS", "ARMS", "BASELINE3_ORDER", "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_ENTROPY", "CODE_FILES", "CONTRASTS", "CONTINUITY_TOLERANCE",
    "COVERAGE_MAIN", "DENOMINATOR_FLOOR", "DIRECT_HIDDEN_WIDTH", "DROPOUT",
    "EARLY_STOP_PATIENCE", "EARLY_STOP_TOLERANCE", "EXPECTED_TRAINABLE_PARAMETERS",
    "FORBIDDEN_NPZ_KEYS", "HEURISTIC_METHODS", "HIGH_REGRET_THRESHOLD", "LAMBDA_CLS",
    "LEARNING_RATE", "MAIN_CI_QUANTILES", "MAIN_FAMILY_SIZE", "MAX_EPOCHS",
    "MODEL_SEEDS", "OUTPUT_DIR", "PLAN_ID", "R2_DEVELOPMENT_DIR", "ROOT",
    "SEED_JSON_HINTS", "SEED_SCALAR_KEYS", "TRAIN_BATCH_SIZE", "TRAIN_DATA",
    "UNCORRECTED_CI_QUANTILES", "VALIDATION_DATA", "WEIGHT_DECAY",
    "assert_cache_keys_legal", "assert_json_seed_free", "assert_output_isolation",
    "code_hashes", "config_signature", "create_protocol_lock", "hash_protected_files",
    "identity_digest", "load_config", "make_group_ids", "manuscript_protected_files",
    "output_dir", "resolve_sources", "scanner_json_seeds", "scanner_npz_seeds",
    "source_group_seed", "validate_binding", "validate_config",
    "verify_protected_hashes", "verify_protocol_lock",
]
