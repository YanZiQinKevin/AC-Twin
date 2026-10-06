"""Stage 4A v2 recovery lock: read-only reuse of the v1-collected data.

A provable engineering bug (the end-to-end router was never moved to the
configured device in ``run_stage4a.cmd_infer_dev``) was found at ``infer-dev``
AFTER the v1 protocol lock and AFTER the 5,000-episode collection completed.
The collection itself was produced by ``collect_stage4a_merge_dev.py`` — which is
unchanged — so the v1 data is valid. Per the recovery protocol this module
establishes a SEPARATE write-once ``recovery_lock.json`` under a v2 run dir
without touching the v1 protocol lock, re-collection, or the scientific contract.

This is NOT a fresh pre-collection protocol lock:

- it does NOT regenerate or rescan a seed schedule — the candidate schedule is
  reused by reference and verified byte-for-byte against the parent v1 lock;
- the zero-overlap guarantee is inherited, not bypassed: the reused episode seeds
  are verified to be exactly a subset of the parent schedule AND to match the v1
  collection receipt / NPZ, so no new world enters and no historical overlap
  is introduced by reuse;
- every frozen model/scaler/binding and every code file except the fixed runner
  must hash-identically match the parent v1 lock, or the build refuses.

Anything scientific (risk candidates, split, gate thresholds, seeds, model
weights) is unchanged; only the runner's device-placement line differs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin import stage4_protocol as s4p
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

RECOVERY_LOCK_NAME = "recovery_lock.json"
# The single code file the engineering fix is allowed to touch.
FIXED_RUNNER = "run_stage4a.py"


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def is_recovery_run(run_dir: str | Path) -> bool:
    """True when ``run_dir`` holds a recovery lock instead of a protocol lock."""
    return (Path(run_dir) / "protocol" / RECOVERY_LOCK_NAME).is_file()


def _parent_paths(parent_run_dir: Path) -> dict[str, Path]:
    protocol = Path(parent_run_dir) / "protocol"
    return {
        "parent_lock": protocol / "protocol_lock.json",
        "seed_schedule": protocol / "candidate_seed_schedule.npy",
        "collection_receipt": Path(parent_run_dir) / "collection" / "collection_receipt.json",
    }


def build_recovery_lock(
    config: Mapping[str, Any],
    *,
    parent_run_dir: str | Path,
    reason: str,
    code_location: str,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Compute (not yet write) the v2 recovery lock; refuse on any drift.

    Verifies the complete reuse chain against the parent v1 lock and raises
    unless EVERY frozen input other than the fixed runner is byte-identical.
    """
    checked = s4p.validate_stage4a_config(config)
    parent_run_dir = Path(parent_run_dir)
    paths = _parent_paths(parent_run_dir)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"parent v1 artifact missing: {label} -> {path}")
    parent_lock = json.loads(paths["parent_lock"].read_text(encoding="utf-8"))
    parent_lock_sha256 = sha256_file(paths["parent_lock"])

    # 1. Collector + all non-runner code must match the parent lock exactly.
    parent_code = parent_lock["code_sha256"]
    current_code: dict[str, str] = {}
    for rel in s4p.CODE_FILES:
        current_code[rel] = sha256_file(project_root / "code" / rel)
    unexpected = [
        rel for rel in s4p.CODE_FILES
        if rel != FIXED_RUNNER and current_code[rel] != parent_code[rel]
    ]
    if unexpected:
        raise ValueError(f"unexplained code drift beyond {FIXED_RUNNER}: {unexpected}")
    original_runner_sha256 = parent_code[FIXED_RUNNER]
    fixed_runner_sha256 = current_code[FIXED_RUNNER]
    if original_runner_sha256 == fixed_runner_sha256:
        raise ValueError("recovery requested but the runner code is unchanged")

    # 2. Seed schedule reused byte-for-byte (no regeneration, no rescan).
    schedule = np.load(paths["seed_schedule"]).astype(np.int64)
    schedule_sha256 = hashlib.sha256(schedule.tobytes()).hexdigest()
    if schedule_sha256 != parent_lock["seed_schedule_sha256"]:
        raise ValueError("parent seed schedule bytes changed since v1 lock")

    # 3. Frozen models / scaler / binding must all match the parent lock.
    model_lock = s4p.inventory_model_lock(checked)
    if model_lock["checkpoint_sha256"] != parent_lock["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint drift detected during recovery")
    if model_lock["scaler_sha256"] != parent_lock["scaler_sha256"]:
        raise ValueError("frozen scaler drift detected during recovery")
    if model_lock["stage3b_binding_sha256"] != parent_lock["stage3b_binding_sha256"]:
        raise ValueError("Stage 3A Plus binding drift detected during recovery")

    # 4. Collection receipt + NPZ: hashes must agree with each other and parent.
    collection_receipt = json.loads(paths["collection_receipt"].read_text(encoding="utf-8"))
    collection_receipt_sha256 = sha256_file(paths["collection_receipt"])
    data_output = _resolve(checked["data_output"])
    if not data_output.is_file():
        raise FileNotFoundError(f"v1-collected NPZ missing: {data_output}")
    data_npz_sha256 = sha256_file(data_output)
    if collection_receipt.get("data_output_sha256") != data_npz_sha256:
        raise ValueError("collection receipt data hash does not match the NPZ on disk")

    # 5. Re-verify the reused dataset: 5,000 episodes, ten strata of 500,
    #    3,000/1,000/1,000 roles, unique seeds drawn from the parent schedule.
    with np.load(data_output, allow_pickle=False) as z:
        episode_seed = np.asarray(z["episode_seed"], dtype=np.int64)
        cf_action = np.asarray(z["cf_action"], dtype=np.int64)
        cf_duration = np.asarray(z["cf_duration"], dtype=np.int64)
        split = np.asarray(z["split"], dtype=np.int64)
    n_episodes = int(episode_seed.size)
    if n_episodes != s4p.N_PAIRS:
        raise ValueError(f"expected {s4p.N_PAIRS} episodes, found {n_episodes}")
    if len(np.unique(episode_seed)) != n_episodes:
        raise ValueError("episode seeds are not unique in the reused NPZ")
    schedule_set = set(schedule.tolist())
    if not all(int(s) in schedule_set for s in episode_seed):
        raise ValueError("reused episode seeds are not a subset of the parent schedule")
    strata_ok = all(
        int(((cf_action == a) & (cf_duration == d)).sum()) == s4p.STRATA_PER_CELL
        for a in s4p.CF_ACTIONS for d in s4p.CF_DURATIONS
    )
    if not strata_ok:
        raise ValueError("reused NPZ does not have exactly 500 episodes per stratum")
    rederived = s4p.assign_stratum_roles(cf_action, cf_duration, episode_seed)
    if not np.array_equal(rederived, split):
        raise ValueError("reused NPZ split column does not match the mechanical role split")
    role_counts = {r: int((split == r).sum()) for r in (0, 1, 2)}
    if role_counts != {0: 3000, 1: 1000, 2: 1000}:
        raise ValueError(f"reused NPZ role split is not 3000/1000/1000: {role_counts}")

    # 6. The v2 inference dir must not yet exist (nothing inferred pre-recovery).
    v2_run_dir = _resolve(checked["run_dir"])
    inference_dir = v2_run_dir / "inference"
    inference_empty = not inference_dir.exists() or not any(inference_dir.iterdir())

    return {
        "plan_id": s4p.PLAN_ID,
        "purpose": s4p.PURPOSE,
        "lock_type": "recovery_from_v1",
        "config_sha256": None,  # filled by write_recovery_lock
        # --- provenance of the recovery ---
        "parent_run_dir": str(parent_run_dir),
        "parent_protocol_lock_sha256": parent_lock_sha256,
        "original_runner_sha256": original_runner_sha256,
        "fixed_runner_sha256": fixed_runner_sha256,
        "recovery_reason": reason,
        "recovery_code_location": code_location,
        # --- reused-by-reference scientific inputs (verified identical) ---
        "seed_schedule_path": str(paths["seed_schedule"]),
        "seed_schedule_sha256": schedule_sha256,
        "collection_receipt_path": str(paths["collection_receipt"]),
        "collection_receipt_sha256": collection_receipt_sha256,
        "data_output": str(data_output),
        "data_npz_sha256": data_npz_sha256,
        "checkpoint_sha256": model_lock["checkpoint_sha256"],
        "scaler_path": model_lock["scaler_path"],
        "scaler_sha256": model_lock["scaler_sha256"],
        "stage3b_binding_path": model_lock["stage3b_binding_path"],
        "stage3b_binding_sha256": model_lock["stage3b_binding_sha256"],
        "code_sha256": current_code,
        # --- reuse verification record ---
        "reuse_verification": {
            "n_episodes": n_episodes,
            "unique_episode_seeds": int(len(np.unique(episode_seed))),
            "episodes_per_stratum": s4p.STRATA_PER_CELL,
            "strata_count": len(s4p.CF_ACTIONS) * len(s4p.CF_DURATIONS),
            "role_counts": {s4p.ROLE_NAMES[r]: role_counts[r] for r in (0, 1, 2)},
            "seeds_subset_of_parent_schedule": True,
            "split_column_matches_mechanical_roles": True,
            "inference_dir_empty_before_fix": bool(inference_empty),
            "collector_code_unchanged": current_code["collect_stage4a_merge_dev.py"]
            == parent_code["collect_stage4a_merge_dev.py"],
            "checkpoints_match_parent": True,
            "scaler_binding_match_parent": True,
        },
        "note": (
            "Recovery from v1 after a provable engineering device-placement fix to "
            f"{FIXED_RUNNER}. No re-collection, no scientific-contract change. The v1 "
            "protocol lock, seed schedule, collection receipt and NPZ are untouched."
        ),
    }


def write_recovery_lock(
    config_path: str | Path,
    *,
    parent_run_dir: str | Path,
    reason: str,
    code_location: str,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Create the v2 recovery lock exactly once (refuses to overwrite)."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = s4p.validate_stage4a_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "protocol"
    lock_path = protocol_dir / RECOVERY_LOCK_NAME
    if lock_path.exists():
        raise FileExistsError(f"Stage 4A recovery lock already exists: {lock_path}")
    lock = build_recovery_lock(
        checked,
        parent_run_dir=parent_run_dir,
        reason=reason,
        code_location=code_location,
        project_root=project_root,
    )
    lock["config_sha256"] = sha256_file(config_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    write_once_json(lock_path, lock)
    return lock


def verify_recovery_lock(run_dir: str | Path, *, project_root: Path = ROOT) -> dict[str, Any]:
    """Read-only re-verification of whichever recovery lock a run dir holds."""
    run_dir = Path(run_dir)
    lock_path = run_dir / "protocol" / RECOVERY_LOCK_NAME
    if not lock_path.is_file():
        raise FileNotFoundError(f"Stage 4A recovery lock missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("lock_type") == "recovery_from_v3":
        return _verify_recovery_lock_v4(lock, project_root=project_root)
    if lock.get("lock_type") == "recovery_from_v2":
        return _verify_recovery_lock_v3(lock, project_root=project_root)
    return _verify_recovery_lock_v1(lock, project_root=project_root)


def _verify_frozen_models(lock: Mapping[str, Any]) -> None:
    """Re-hash every frozen checkpoint / scaler / binding against the lock."""
    for name, expected in lock["checkpoint_sha256"].items():
        s3r = Path(lock["scaler_path"]).parent
        if name.startswith("end_to_end_mixture_"):
            pair_index = int(name.split("_pair")[1].split("_s")[0])
            seed = int(name.rsplit("_s", 1)[1])
            path = s4p.e2e_checkpoint_path(s3r, pair_index, seed)
        else:
            pair_index = int(name.split("pair")[1].split("_")[0])
            condition = "full" if "_full_" in name else "history"
            seed = int(name.rsplit("_seed", 1)[1])
            path = s4p.expert_checkpoint_path(s3r, pair_index, condition, seed)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"recovery-locked checkpoint drift: {name}")
    if sha256_file(lock["scaler_path"]) != lock["scaler_sha256"]:
        raise ValueError("recovery-locked scaler drift")
    if sha256_file(lock["stage3b_binding_path"]) != lock["stage3b_binding_sha256"]:
        raise ValueError("recovery-locked Stage 3A Plus binding drift")


def _verify_code(lock: Mapping[str, Any], project_root: Path) -> None:
    """Re-hash every locked code file so no FURTHER drift after the lock is possible."""
    for rel, expected in lock["code_sha256"].items():
        file_path = project_root / "code" / rel
        if not file_path.is_file():
            raise ValueError(f"recovery-locked code file missing: {rel}")
        if sha256_file(file_path) != expected:
            raise ValueError(f"recovery-locked code drift: {rel}")


def _verify_recovery_lock_v1(lock: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    """Re-verify a v2 (recovery_from_v1) lock: the whole reuse chain still matches."""
    _verify_code(lock, project_root)

    # Parent lock + seed schedule unchanged.
    parent_lock_path = Path(lock["parent_run_dir"]) / "protocol" / "protocol_lock.json"
    if sha256_file(parent_lock_path) != lock["parent_protocol_lock_sha256"]:
        raise ValueError("parent v1 protocol lock changed after recovery")
    schedule = np.load(lock["seed_schedule_path"]).astype(np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != lock["seed_schedule_sha256"]:
        raise ValueError("recovery seed schedule bytes changed")

    _verify_frozen_models(lock)

    # Collection receipt + NPZ unchanged.
    if sha256_file(lock["collection_receipt_path"]) != lock["collection_receipt_sha256"]:
        raise ValueError("collection receipt changed after recovery")
    if sha256_file(lock["data_output"]) != lock["data_npz_sha256"]:
        raise ValueError("reused NPZ changed after recovery")
    return lock


# --- v3 recovery (recovery_from_v2): reuse v2 inference, retrain risk models ---------


# Code files a v3 recovery is allowed to differ from the v1 lock. run_stage4a.py
# carries the v2 device fix (and v3 fit-risk reuse wiring); stage4_risk.py carries
# the v3 MLP-serialization fix. Nothing else may drift.
_V3_ALLOWED_DRIFT = {FIXED_RUNNER, "ac_twin/stage4_risk.py"}


def build_recovery_lock_v3(
    config: Mapping[str, Any],
    *,
    v2_run_dir: str | Path,
    reason: str,
    code_location: str,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Compute (not yet write) the v3 recovery lock; refuse on any inconsistency.

    Chains to the v2 recovery lock and the v1 protocol lock, reuses the v2 frozen
    inference (predictions.npz) and the v2-frozen PRIMARY_PREDICTOR receipt
    read-only, and re-verifies the v1-collected NPZ. No re-collection, no
    re-inference, no re-selection of the primary predictor, no scientific change.
    """
    checked = s4p.validate_stage4a_config(config)
    v2_run_dir = Path(v2_run_dir)
    v2_lock_path = v2_run_dir / "protocol" / RECOVERY_LOCK_NAME
    if not v2_lock_path.is_file():
        raise FileNotFoundError(f"v2 recovery lock missing: {v2_lock_path}")
    v2_lock = json.loads(v2_lock_path.read_text(encoding="utf-8"))
    v2_lock_sha256 = sha256_file(v2_lock_path)

    # 1. Chain to v1 protocol lock (via v2's recorded parent) and verify unchanged.
    v1_lock_path = Path(v2_lock["parent_run_dir"]) / "protocol" / "protocol_lock.json"
    v1_lock_sha256 = sha256_file(v1_lock_path)
    if v1_lock_sha256 != v2_lock["parent_protocol_lock_sha256"]:
        raise ValueError("v1 protocol lock changed since v2 recovery")
    v1_lock = json.loads(v1_lock_path.read_text(encoding="utf-8"))

    # 2. Code: only the two allowed files may differ from the v1 lock.
    parent_code = v1_lock["code_sha256"]
    current_code = {rel: sha256_file(project_root / "code" / rel) for rel in s4p.CODE_FILES}
    unexpected = [
        rel for rel in s4p.CODE_FILES
        if rel not in _V3_ALLOWED_DRIFT and current_code[rel] != parent_code[rel]
    ]
    if unexpected:
        raise ValueError(f"third unexplained locked-code drift: {unexpected}")
    risk_before = v2_lock["code_sha256"]["ac_twin/stage4_risk.py"]
    risk_after = current_code["ac_twin/stage4_risk.py"]
    if risk_before == risk_after:
        raise ValueError("v3 recovery requested but stage4_risk.py is unchanged")

    # 3. Frozen models / scaler / binding unchanged vs the v1 lock.
    model_lock = s4p.inventory_model_lock(checked)
    if model_lock["checkpoint_sha256"] != v1_lock["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint drift detected during v3 recovery")
    if model_lock["scaler_sha256"] != v1_lock["scaler_sha256"]:
        raise ValueError("frozen scaler drift detected during v3 recovery")
    if model_lock["stage3b_binding_sha256"] != v1_lock["stage3b_binding_sha256"]:
        raise ValueError("Stage 3A Plus binding drift detected during v3 recovery")

    # 4. v1-collected NPZ unchanged (matches the v2-recorded hash).
    data_output = _resolve(checked["data_output"])
    if not data_output.is_file():
        raise FileNotFoundError(f"v1-collected NPZ missing: {data_output}")
    data_npz_sha256 = sha256_file(data_output)
    if data_npz_sha256 != v2_lock["data_npz_sha256"]:
        raise ValueError("v1 NPZ changed since v2 recovery")

    # 5. v2 frozen inference: predictions.npz and its receipt must agree + be complete.
    v2_pred_path = v2_run_dir / "inference" / "predictions.npz"
    v2_pred_receipt_path = v2_run_dir / "inference" / "prediction_receipt.json"
    for path in (v2_pred_path, v2_pred_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"v2 inference artifact missing: {path}")
    predictions_sha256 = sha256_file(v2_pred_path)
    prediction_receipt_sha256 = sha256_file(v2_pred_receipt_path)
    pred_receipt = json.loads(v2_pred_receipt_path.read_text(encoding="utf-8"))
    if pred_receipt.get("predictions_sha256") != predictions_sha256:
        raise ValueError("v2 predictions hash does not match its receipt")
    if int(pred_receipt.get("episode_count", -1)) != s4p.N_PAIRS:
        raise ValueError("v2 predictions do not cover all 5,000 episodes")

    # 6. v2-frozen PRIMARY_PREDICTOR receipt: verify, reference, never re-select.
    v2_pp_path = v2_run_dir / "risk" / "primary_predictor_receipt.json"
    if not v2_pp_path.is_file():
        raise FileNotFoundError(f"v2 primary predictor receipt missing: {v2_pp_path}")
    primary_receipt_sha256 = sha256_file(v2_pp_path)
    primary_receipt = json.loads(v2_pp_path.read_text(encoding="utf-8"))
    if primary_receipt.get("predictions_sha256") != predictions_sha256:
        raise ValueError("v2 primary predictor receipt does not match v2 predictions")
    primary_predictor = primary_receipt["primary_predictor"]

    # 7. Hard-stop guards: the development-screen / selection must be untouched in v2.
    if (v2_run_dir / "development_screen_used.lock").exists():
        raise ValueError("development screen already consumed in v2; cannot recover")
    if (v2_run_dir / "risk" / "candidate_selection.json").exists():
        raise ValueError("candidate selection already ran in v2; cannot recover")
    if (v2_run_dir / "reports" / "stage4a_dev_report.json").exists():
        raise ValueError("development-screen report already exists in v2; cannot recover")

    return {
        "plan_id": s4p.PLAN_ID,
        "purpose": s4p.PURPOSE,
        "lock_type": "recovery_from_v2",
        "config_sha256": None,  # filled by write_recovery_lock_v3
        "chain": {
            "v1_protocol_lock_path": str(v1_lock_path),
            "v1_protocol_lock_sha256": v1_lock_sha256,
            "v2_recovery_lock_path": str(v2_lock_path),
            "v2_recovery_lock_sha256": v2_lock_sha256,
        },
        "stage4_risk_fix": {
            "before_sha256": risk_before,
            "after_sha256": risk_after,
            "reason": reason,
            "code_location": code_location,
        },
        "code_sha256": current_code,
        "reused_inference": {
            "predictions_npz_path": str(v2_pred_path),
            "predictions_npz_sha256": predictions_sha256,
            "prediction_receipt_path": str(v2_pred_receipt_path),
            "prediction_receipt_sha256": prediction_receipt_sha256,
        },
        "primary_predictor": {
            "receipt_path": str(v2_pp_path),
            "receipt_sha256": primary_receipt_sha256,
            "primary_predictor": primary_predictor,
            "constant_w": primary_receipt.get("constant_w"),
        },
        "data_output": str(data_output),
        "data_npz_sha256": data_npz_sha256,
        "seed_schedule_path": v2_lock["seed_schedule_path"],
        "seed_schedule_sha256": v2_lock["seed_schedule_sha256"],
        "checkpoint_sha256": model_lock["checkpoint_sha256"],
        "scaler_path": model_lock["scaler_path"],
        "scaler_sha256": model_lock["scaler_sha256"],
        "stage3b_binding_path": model_lock["stage3b_binding_path"],
        "stage3b_binding_sha256": model_lock["stage3b_binding_sha256"],
        "reuse_verification": {
            "v1_protocol_lock_unchanged": True,
            "v2_recovery_lock_unchanged": True,
            "only_allowed_code_drifted": True,
            "predictions_complete_5000": True,
            "predictions_receipt_consistent": True,
            "primary_predictor": primary_predictor,
            "primary_predictor_reused_not_reselected": True,
            "data_npz_unchanged": True,
            "checkpoints_match_v1": True,
            "development_screen_untouched": True,
            "candidate_selection_not_run": True,
        },
        "note": (
            "Recovery from v2 after a provable MLP-serialization engineering fix to "
            "ac_twin/stage4_risk.py. Reuses v2 frozen inference and the v2-frozen "
            "PRIMARY_PREDICTOR read-only; retrains all risk models in v3. No "
            "re-collection, no re-inference, no re-selection, no scientific change."
        ),
    }


def write_recovery_lock_v3(
    config_path: str | Path,
    *,
    v2_run_dir: str | Path,
    reason: str,
    code_location: str,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Create the v3 recovery lock exactly once (refuses to overwrite)."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = s4p.validate_stage4a_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "protocol"
    lock_path = protocol_dir / RECOVERY_LOCK_NAME
    if lock_path.exists():
        raise FileExistsError(f"Stage 4A v3 recovery lock already exists: {lock_path}")
    lock = build_recovery_lock_v3(
        checked,
        v2_run_dir=v2_run_dir,
        reason=reason,
        code_location=code_location,
        project_root=project_root,
    )
    lock["config_sha256"] = sha256_file(config_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    write_once_json(lock_path, lock)
    return lock


def _verify_recovery_lock_v3(lock: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    """Re-verify a v3 (recovery_from_v2) lock: chain + reused inference + models."""
    _verify_code(lock, project_root)

    chain = lock["chain"]
    if sha256_file(chain["v1_protocol_lock_path"]) != chain["v1_protocol_lock_sha256"]:
        raise ValueError("v1 protocol lock changed after v3 recovery")
    if sha256_file(chain["v2_recovery_lock_path"]) != chain["v2_recovery_lock_sha256"]:
        raise ValueError("v2 recovery lock changed after v3 recovery")

    _verify_frozen_models(lock)

    if sha256_file(lock["data_output"]) != lock["data_npz_sha256"]:
        raise ValueError("reused v1 NPZ changed after v3 recovery")

    reused = lock["reused_inference"]
    if sha256_file(reused["predictions_npz_path"]) != reused["predictions_npz_sha256"]:
        raise ValueError("reused v2 predictions.npz changed after v3 recovery")
    if sha256_file(reused["prediction_receipt_path"]) != reused["prediction_receipt_sha256"]:
        raise ValueError("reused v2 prediction receipt changed after v3 recovery")

    primary = lock["primary_predictor"]
    if sha256_file(primary["receipt_path"]) != primary["receipt_sha256"]:
        raise ValueError("reused v2 PRIMARY_PREDICTOR receipt changed after v3 recovery")
    return lock


# --- v4 recovery (recovery_from_v3): reuse v2 inference + v3 risk fit, replay selection ---


# Code files a v4 recovery is allowed to differ from the v3 lock. Only the runner
# drifts: it carries the cmd_select_candidate reused-inference receipt-path fix plus
# the additive lock-recovery-v4 command. ac_twin/stage4_recovery.py is not in
# s4p.CODE_FILES (so it is not drift-checked against the parent) but IS bound into
# the v4 lock's own code_sha256 so any later change is rejected. Nothing else moves.
_V4_ALLOWED_DRIFT = {FIXED_RUNNER}
_RECOVERY_MODULE = "ac_twin/stage4_recovery.py"


def build_recovery_lock_v4(
    config: Mapping[str, Any],
    *,
    v3_run_dir: str | Path,
    patch_receipt_path: str | Path,
    reason: str,
    code_location: str,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Compute (not yet write) the v4 recovery lock; refuse on any inconsistency.

    Chains to the v3 / v2 / v1 locks, reuses read-only: the v2 frozen inference and
    PRIMARY_PREDICTOR receipt, and the v3 risk fit (risk_dataset, fit_receipt and the
    five fitted models). v4 performs NO re-collection, NO re-inference, NO
    PRIMARY_PREDICTOR re-selection and NO risk-model retraining — it deterministically
    replays ``select-candidate`` then ``adjudicate-dev-screen`` under the same frozen
    algorithm. The only scientific-command-path code change vs v3 is the
    cmd_select_candidate reused-inference receipt-path fix (see ``runner_recovery``).
    """
    checked = s4p.validate_stage4a_config(config)
    v3_run_dir = Path(v3_run_dir)
    v3_lock_path = v3_run_dir / "protocol" / RECOVERY_LOCK_NAME
    if not v3_lock_path.is_file():
        raise FileNotFoundError(f"v3 recovery lock missing: {v3_lock_path}")
    v3_lock = json.loads(v3_lock_path.read_text(encoding="utf-8"))
    v3_lock_sha256 = sha256_file(v3_lock_path)

    # 1. Three-layer parent chain: v1 / v2 hashes come from v3's recorded chain.
    v1_lock_path = Path(v3_lock["chain"]["v1_protocol_lock_path"])
    v2_lock_path = Path(v3_lock["chain"]["v2_recovery_lock_path"])
    v1_lock_sha256 = sha256_file(v1_lock_path)
    v2_lock_sha256 = sha256_file(v2_lock_path)
    if v1_lock_sha256 != v3_lock["chain"]["v1_protocol_lock_sha256"]:
        raise ValueError("v1 protocol lock changed since v3 recovery")
    if v2_lock_sha256 != v3_lock["chain"]["v2_recovery_lock_sha256"]:
        raise ValueError("v2 recovery lock changed since v3 recovery")
    v1_lock = json.loads(v1_lock_path.read_text(encoding="utf-8"))

    # 2. Code: only the runner may differ from the v3 lock. stage4_recovery.py is not
    #    parent-tracked, so bind it explicitly into the v4 code_sha256.
    parent_code = v3_lock["code_sha256"]
    current_code = {rel: sha256_file(project_root / "code" / rel) for rel in s4p.CODE_FILES}
    current_code[_RECOVERY_MODULE] = sha256_file(project_root / "code" / _RECOVERY_MODULE)
    unexpected = [
        rel for rel in s4p.CODE_FILES
        if rel not in _V4_ALLOWED_DRIFT and current_code[rel] != parent_code[rel]
    ]
    if unexpected:
        raise ValueError(f"unexplained locked-code drift beyond {sorted(_V4_ALLOWED_DRIFT)}: {unexpected}")
    runner_before = parent_code[FIXED_RUNNER]
    runner_after = current_code[FIXED_RUNNER]
    if runner_before == runner_after:
        raise ValueError("v4 recovery requested but the runner code is unchanged")

    # 2b. Bind the single-point runner-fix patch receipt and require consistency.
    patch_receipt_path = Path(patch_receipt_path)
    if not patch_receipt_path.is_file():
        raise FileNotFoundError(f"runner-fix patch receipt missing: {patch_receipt_path}")
    patch_receipt = json.loads(patch_receipt_path.read_text(encoding="utf-8"))
    fix = patch_receipt["runner_scientific_fix"]
    if fix["before_sha256"] != runner_before:
        raise ValueError("runner-fix receipt before_hash does not match the v3-locked runner")
    patch_file_path = patch_receipt_path.parent / fix["patch_file"]

    # 3. Frozen models / scaler / binding unchanged vs the v1 lock.
    model_lock = s4p.inventory_model_lock(checked)
    if model_lock["checkpoint_sha256"] != v1_lock["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint drift detected during v4 recovery")
    if model_lock["scaler_sha256"] != v1_lock["scaler_sha256"]:
        raise ValueError("frozen scaler drift detected during v4 recovery")
    if model_lock["stage3b_binding_sha256"] != v1_lock["stage3b_binding_sha256"]:
        raise ValueError("Stage 3A Plus binding drift detected during v4 recovery")

    # 4. v1-collected NPZ unchanged (matches the chain-recorded hash).
    data_output = _resolve(checked["data_output"])
    if not data_output.is_file():
        raise FileNotFoundError(f"v1-collected NPZ missing: {data_output}")
    data_npz_sha256 = sha256_file(data_output)
    if data_npz_sha256 != v3_lock["data_npz_sha256"]:
        raise ValueError("v1 NPZ changed since v3 recovery")

    # 5. v2 frozen inference + PRIMARY_PREDICTOR receipt unchanged (from v3 chain).
    reused_inference = dict(v3_lock["reused_inference"])
    if sha256_file(reused_inference["predictions_npz_path"]) != reused_inference["predictions_npz_sha256"]:
        raise ValueError("reused v2 predictions.npz changed since v3 recovery")
    if sha256_file(reused_inference["prediction_receipt_path"]) != reused_inference["prediction_receipt_sha256"]:
        raise ValueError("reused v2 prediction receipt changed since v3 recovery")
    primary = dict(v3_lock["primary_predictor"])
    if sha256_file(primary["receipt_path"]) != primary["receipt_sha256"]:
        raise ValueError("reused v2 PRIMARY_PREDICTOR receipt changed since v3 recovery")
    primary_predictor = primary["primary_predictor"]

    # 6. v3 risk fit: risk_dataset + fit_receipt consistent, five models hashed.
    v3_risk_dataset_path = v3_run_dir / "risk" / "risk_dataset.npz"
    v3_fit_receipt_path = v3_run_dir / "risk" / "fit_receipt.json"
    for path in (v3_risk_dataset_path, v3_fit_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"v3 risk artifact missing: {path}")
    v3_risk_dataset_sha256 = sha256_file(v3_risk_dataset_path)
    v3_fit_receipt_sha256 = sha256_file(v3_fit_receipt_path)
    fit_receipt = json.loads(v3_fit_receipt_path.read_text(encoding="utf-8"))
    if fit_receipt.get("risk_dataset_sha256") != v3_risk_dataset_sha256:
        raise ValueError("v3 fit_receipt does not match the v3 risk dataset on disk")
    if fit_receipt.get("status") != "RISK_FIT_COMPLETE":
        raise ValueError(f"v3 risk fit is not complete: {fit_receipt.get('status')}")
    if fit_receipt.get("primary_predictor") != primary_predictor:
        raise ValueError("v3 fit_receipt PRIMARY_PREDICTOR disagrees with the frozen chain")
    fitted_models: dict[str, dict[str, str]] = {}
    for key, model_path in fit_receipt["fitted_models"].items():
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"v3 fitted model missing: {key} -> {model_path}")
        fitted_models[key] = {"path": str(model_path), "sha256": sha256_file(model_path)}
    if len(fitted_models) != 5:
        raise ValueError(f"expected five fitted risk models, found {len(fitted_models)}")

    # 7. v4 reuse copies: byte-identical risk_dataset + fit_receipt under v4/risk.
    v4_run_dir = _resolve(checked["run_dir"])
    v4_risk_dataset_path = v4_run_dir / "risk" / "risk_dataset.npz"
    v4_fit_receipt_path = v4_run_dir / "risk" / "fit_receipt.json"
    for path in (v4_risk_dataset_path, v4_fit_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"v4 reuse copy missing: {path}. Create byte-identical copies of the v3 "
                "risk_dataset.npz and fit_receipt.json before locking."
            )
    v4_risk_dataset_sha256 = sha256_file(v4_risk_dataset_path)
    v4_fit_receipt_sha256 = sha256_file(v4_fit_receipt_path)
    if v4_risk_dataset_sha256 != v3_risk_dataset_sha256:
        raise ValueError("v4 risk_dataset copy does not match v3 (SHA256 differs)")
    if v4_fit_receipt_sha256 != v3_fit_receipt_sha256:
        raise ValueError("v4 fit_receipt copy does not match v3 (SHA256 differs)")

    # 8. Hard-stop guards: candidate receipt and development-screen untouched everywhere.
    for d in (v3_run_dir, v4_run_dir):
        if (Path(d) / "development_screen_used.lock").exists():
            raise ValueError(f"development screen already consumed in {d}; cannot recover")
        if (Path(d) / "risk" / "candidate_selection.json").exists():
            raise ValueError(f"candidate selection already ran in {d}; cannot recover")
        if (Path(d) / "reports" / "stage4a_dev_report.json").exists():
            raise ValueError(f"development-screen report already exists in {d}; cannot recover")

    return {
        "plan_id": s4p.PLAN_ID,
        "purpose": s4p.PURPOSE,
        "lock_type": "recovery_from_v3",
        "config_sha256": None,  # filled by write_recovery_lock_v4
        "chain": {
            "v1_protocol_lock_path": str(v1_lock_path),
            "v1_protocol_lock_sha256": v1_lock_sha256,
            "v2_recovery_lock_path": str(v2_lock_path),
            "v2_recovery_lock_sha256": v2_lock_sha256,
            "v3_recovery_lock_path": str(v3_lock_path),
            "v3_recovery_lock_sha256": v3_lock_sha256,
        },
        "runner_recovery": {
            "v3_locked_runner_sha256": runner_before,
            "scientific_fix_runner_sha256": fix["after_sha256"],
            "final_runner_sha256": runner_after,
            "patch_receipt_path": str(patch_receipt_path),
            "patch_receipt_sha256": sha256_file(patch_receipt_path),
            "patch_file_path": str(patch_file_path),
            "patch_file_sha256": sha256_file(patch_file_path) if patch_file_path.is_file() else None,
            "code_location": code_location,
            "reason": reason,
            "note": (
                "scientific_fix_runner_sha256 (a8f57d33...) isolates the ONLY "
                "scientific-command-path change vs v3 (cmd_select_candidate reused-"
                "inference receipt path), proven by the toggle receipt. "
                "final_runner_sha256 additionally carries the additive, recovery-only "
                "lock-recovery-v4 command; it does not touch selection/adjudication."
            ),
        },
        "code_sha256": current_code,
        "reused_inference": reused_inference,
        "primary_predictor": primary,
        "reused_risk_fit": {
            "v3_risk_dataset_path": str(v3_risk_dataset_path),
            "v3_risk_dataset_sha256": v3_risk_dataset_sha256,
            "v3_fit_receipt_path": str(v3_fit_receipt_path),
            "v3_fit_receipt_sha256": v3_fit_receipt_sha256,
            "fitted_models": fitted_models,
        },
        "v4_reuse_copies": {
            "risk_dataset_path": str(v4_risk_dataset_path),
            "risk_dataset_sha256": v4_risk_dataset_sha256,
            "fit_receipt_path": str(v4_fit_receipt_path),
            "fit_receipt_sha256": v4_fit_receipt_sha256,
        },
        "data_output": str(data_output),
        "data_npz_sha256": data_npz_sha256,
        "seed_schedule_path": v3_lock["seed_schedule_path"],
        "seed_schedule_sha256": v3_lock["seed_schedule_sha256"],
        "checkpoint_sha256": model_lock["checkpoint_sha256"],
        "scaler_path": model_lock["scaler_path"],
        "scaler_sha256": model_lock["scaler_sha256"],
        "stage3b_binding_path": model_lock["stage3b_binding_path"],
        "stage3b_binding_sha256": model_lock["stage3b_binding_sha256"],
        "reuse_verification": {
            "v1_protocol_lock_unchanged": True,
            "v2_recovery_lock_unchanged": True,
            "v3_recovery_lock_unchanged": True,
            "only_allowed_code_drifted": True,
            "runner_fix_patch_consistent": True,
            "risk_dataset_copy_matches_v3": True,
            "fit_receipt_copy_matches_v3": True,
            "five_models_hashed": True,
            "primary_predictor": primary_predictor,
            "primary_predictor_consistent": True,
            "data_npz_unchanged": True,
            "checkpoints_match_v1": True,
            "candidate_receipt_not_generated": True,
            "development_screen_untouched": True,
            # risk-validation was read in-memory by the aborted v3 select-candidate,
            # but no selection receipt was persisted; v4 deterministically replays the
            # same frozen algorithm rather than re-selecting or re-tuning.
            "validation_computed_not_persisted": True,
        },
        "note": (
            "Recovery from v3 after a provable reused-inference receipt-path fix to "
            "run_stage4a.py::cmd_select_candidate. Reuses v2 frozen inference, the "
            "v2-frozen PRIMARY_PREDICTOR and the complete v3 risk fit read-only; v4 "
            "replays select-candidate then adjudicate-dev-screen under the same frozen "
            "algorithm. No re-collection, no re-inference, no re-selection, no retraining."
        ),
    }


def write_recovery_lock_v4(
    config_path: str | Path,
    *,
    v3_run_dir: str | Path,
    patch_receipt_path: str | Path,
    reason: str,
    code_location: str,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Create the v4 recovery lock exactly once (refuses to overwrite)."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checked = s4p.validate_stage4a_config(config)
    run_dir = _resolve(checked["run_dir"])
    protocol_dir = run_dir / "protocol"
    lock_path = protocol_dir / RECOVERY_LOCK_NAME
    if lock_path.exists():
        raise FileExistsError(f"Stage 4A v4 recovery lock already exists: {lock_path}")
    lock = build_recovery_lock_v4(
        checked,
        v3_run_dir=v3_run_dir,
        patch_receipt_path=patch_receipt_path,
        reason=reason,
        code_location=code_location,
        project_root=project_root,
    )
    lock["config_sha256"] = sha256_file(config_path)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    write_once_json(lock_path, lock)
    return lock


def _verify_recovery_lock_v4(lock: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    """Re-verify a v4 (recovery_from_v3) lock: chain + reused inference + risk fit."""
    _verify_code(lock, project_root)

    chain = lock["chain"]
    if sha256_file(chain["v1_protocol_lock_path"]) != chain["v1_protocol_lock_sha256"]:
        raise ValueError("v1 protocol lock changed after v4 recovery")
    if sha256_file(chain["v2_recovery_lock_path"]) != chain["v2_recovery_lock_sha256"]:
        raise ValueError("v2 recovery lock changed after v4 recovery")
    if sha256_file(chain["v3_recovery_lock_path"]) != chain["v3_recovery_lock_sha256"]:
        raise ValueError("v3 recovery lock changed after v4 recovery")

    _verify_frozen_models(lock)

    schedule = np.load(lock["seed_schedule_path"]).astype(np.int64)
    if hashlib.sha256(schedule.tobytes()).hexdigest() != lock["seed_schedule_sha256"]:
        raise ValueError("recovery seed schedule bytes changed after v4 recovery")

    if sha256_file(lock["data_output"]) != lock["data_npz_sha256"]:
        raise ValueError("reused v1 NPZ changed after v4 recovery")

    reused = lock["reused_inference"]
    if sha256_file(reused["predictions_npz_path"]) != reused["predictions_npz_sha256"]:
        raise ValueError("reused v2 predictions.npz changed after v4 recovery")
    if sha256_file(reused["prediction_receipt_path"]) != reused["prediction_receipt_sha256"]:
        raise ValueError("reused v2 prediction receipt changed after v4 recovery")

    primary = lock["primary_predictor"]
    if sha256_file(primary["receipt_path"]) != primary["receipt_sha256"]:
        raise ValueError("reused v2 PRIMARY_PREDICTOR receipt changed after v4 recovery")

    risk = lock["reused_risk_fit"]
    if sha256_file(risk["v3_risk_dataset_path"]) != risk["v3_risk_dataset_sha256"]:
        raise ValueError("reused v3 risk_dataset.npz changed after v4 recovery")
    if sha256_file(risk["v3_fit_receipt_path"]) != risk["v3_fit_receipt_sha256"]:
        raise ValueError("reused v3 fit_receipt.json changed after v4 recovery")
    for key, entry in risk["fitted_models"].items():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"reused v3 fitted model changed after v4 recovery: {key}")

    copies = lock["v4_reuse_copies"]
    if sha256_file(copies["risk_dataset_path"]) != copies["risk_dataset_sha256"]:
        raise ValueError("v4 risk_dataset copy changed after v4 recovery")
    if sha256_file(copies["fit_receipt_path"]) != copies["fit_receipt_sha256"]:
        raise ValueError("v4 fit_receipt copy changed after v4 recovery")
    return lock


__all__ = [
    "FIXED_RUNNER",
    "RECOVERY_LOCK_NAME",
    "build_recovery_lock",
    "build_recovery_lock_v3",
    "build_recovery_lock_v4",
    "is_recovery_run",
    "verify_recovery_lock",
    "write_recovery_lock",
    "write_recovery_lock_v3",
    "write_recovery_lock_v4",
]
