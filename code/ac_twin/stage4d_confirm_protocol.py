"""Stage 4D D1 confirm protocol lock: extended registry, inventories, audit, lock.

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §6.3 and §10 Task 4. This module
creates the formal write-once ``protocol_lock.json`` AFTER the D1 power analysis has frozen
the accepted n (``D1_POWER_FROZEN``). It NEVER collects D1, NEVER loads a Direct checkpoint
for OOD inference (the lock records path+hash only), NEVER reads the sealed confirm truth and
NEVER reads the R2 train NPZ's CF-future/regret fields.

Carried obligations discharged here (Task 1/3 review items):

1. The lock-time seed audit is RE-DERIVED FOR REAL: :func:`run_real_seed_audit` re-runs the
   full historical scan (:func:`stage4d_protocol.scan_historical_group_seeds`) and the
   zero-overlap verdict for every candidate domain at lock time AND again at verify time; it
   never trusts a recorded flag. Task 1's ``verify_protocol_lock`` only checks the recorded
   flag, so this module's :func:`verify_confirm_protocol_lock` adds the live re-run.
2. §2.1 last paragraph: the five Direct checkpoints, the Full/History/E2E predictors, the
   five response predictors, the binding and the scaler are expanded from the R2/confirm
   inventories to EXPLICIT path+SHA256 entries, each measured (re-hashed) at lock time; any
   drift fails closed before collection.
3. Registry-first code inventory: ``CONFIRM_REGISTRY`` REGISTERS the collector / OOD
   statistics / runner / this module (plus the reused frozen D1-chain dependencies) BEFORE any
   hash is taken — no "hash the files that happen to exist".

The lock's ``code_sha256`` field is EXACTLY Task 1's ``protocol_code_hashes()`` (strict dict
equality is required by the collector's ``prot.verify_protocol_lock`` call); the extended
registry is carried separately as ``extended_code_sha256`` + ``code_inventory.json`` and is
re-verified by :func:`verify_confirm_protocol_lock`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ac_twin import stage4c_protocol as s4c
from ac_twin import stage4_protocol as s4p
from ac_twin import stage4cr2_confirm_protocol as r2confirm  # frozen prior stage, read-only
from ac_twin import stage4cr2_statistics as s4cr2st
from ac_twin import stage4d_ood_statistics as d0
from ac_twin import stage4d_protocol as prot
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = prot.ROOT

PLAN_ID = prot.PLAN_ID

# --- registry-first code inventory (plan §6.3: register BEFORE hashing) -------------------------
#: this stage's own code files bound at lock time that Task 1's registry does not yet cover
#: (Task 2 collector, Task 3 statistics/runner, this Task 4 module). Registered explicitly.
CONFIRM_CODE_FILES = (
    "collect_stage4d_merge_ood.py",
    "ac_twin/stage4d_ood_statistics.py",
    "run_stage4d_ood.py",
    "ac_twin/stage4d_confirm_protocol.py",
    # Task 5 (§13 recovery chain r1): this stage's thin D1 inference/scoring adapter.
    "ac_twin/stage4d_ood_inference.py",
)
#: additional reused frozen dependencies exercised by the D1 collection / inference / feature /
#: scoring / statistics chain, beyond Task 1's ``REUSED_FROZEN_CODE_FILES`` (mirrors the R2
#: confirm stage's real executed dependency set, plus the R2 screen/power/confirm statistics).
CONFIRM_EXTRA_REUSED_FROZEN_CODE_FILES = (
    "run_stage4a.py",
    "ac_twin/stage3_scaler.py",
    "ac_twin/stage3_training.py",
    "ac_twin/stage3_response.py",
    "ac_twin/stage3b_bridge.py",
    "ac_twin/stage3r_models.py",
    "ac_twin/stage4_risk.py",
    "ac_twin/stage4b_inference.py",
    "ac_twin/stage4c_inference.py",
    "ac_twin/stage4cp_features.py",
    "ac_twin/stage4cp_models.py",
    "ac_twin/stage4cr2_training.py",
    "ac_twin/stage4_statistics.py",
    "ac_twin/stage4c_statistics.py",
    "ac_twin/stage4cp_statistics.py",
    "ac_twin/stage4cr2_statistics.py",
    "ac_twin/stage4cr2_power.py",
    "ac_twin/stage4cr2_confirm_statistics.py",
    "collect_stage4cr2_merge_multiaction.py",
    # Task 5 (§13 recovery chain r1): the frozen R2-confirm scoring module ("cinf") whose
    # input contract matches this stage's inference_inputs.npz; reused READ-ONLY.
    "ac_twin/stage4cr2_confirm_inference.py",
)

#: the full ordered registry: Task 1's base registry FIRST, then this task's own files, then the
#: extra reused frozen dependencies — de-duplicated, order-preserved. This is the allow-list;
#: hashing is driven by it, never by directory contents.
CONFIRM_REGISTRY = tuple(dict.fromkeys(
    tuple(prot.PROTOCOL_CODE_FILES) + tuple(prot.REUSED_FROZEN_CODE_FILES)
    + CONFIRM_CODE_FILES + CONFIRM_EXTRA_REUSED_FROZEN_CODE_FILES))

#: the binding receipt that carries the five response predictors' explicit path+SHA256.
STAGE3B_BINDING_RECEIPT = "experiments/stage3_routing/stage3b/input_binding_receipt.json"
STAGE3R_RUN_DIR = "experiments/stage3_routing/stage3r"
TRAJECTORY_SCALER = "experiments/stage3_routing/stage3r/scaler.json"
BRANCH_RISK_MODEL = (
    "experiments/stage4_reliability/stage4a_v3/risk/models/response_only_ridge.pt")

#: the five candidate seed domains audited at lock time (entropy, size), keyed by §6.2 role.
_SEED_DOMAINS = {
    "smoke": (prot.SMOKE_SEED_ENTROPY, prot.SMOKE_SCHEDULE_SIZE),
    "pilot_behavior": (prot.PILOT_BEHAVIOR_SEED_ENTROPY, prot.PILOT_SCHEDULE_SIZE),
    "pilot_geometry": (prot.PILOT_GEOMETRY_SEED_ENTROPY, prot.PILOT_SCHEDULE_SIZE),
    "pilot_joint": (prot.PILOT_JOINT_SEED_ENTROPY, prot.PILOT_SCHEDULE_SIZE),
    "d1_confirm": (prot.D1_CANDIDATE_SEED_ENTROPY, prot.D1_CANDIDATE_SCHEDULE_SIZE),
}


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- extended code inventory ---------------------------------------------------------------------


def _code_hashes(files: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in files:
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"registered Stage 4D confirm code file missing: {rel}")
        hashes[rel] = sha256_file(file_path)
    return hashes


def confirm_code_hashes() -> dict[str, str]:
    """Hashes of the FULL registered D1 chain (registry-driven, never hash-on-existence)."""
    return _code_hashes(CONFIRM_REGISTRY)


def build_confirm_code_inventory() -> dict[str, Any]:
    """Reviewable extended code inventory: per-file hashes + frozen third-party versions.

    ``code_sha256`` covers the full ``CONFIRM_REGISTRY``; ``base_code_sha256`` reproduces Task
    1's ``protocol_code_hashes()`` so the lock can bind both the base registry (strict equality
    required by ``prot.verify_protocol_lock``) and the extended registry.
    """
    hashes = confirm_code_hashes()
    return {
        "plan_id": PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "confirm_code_files": list(CONFIRM_CODE_FILES),
        "extra_reused_frozen_code_files": list(CONFIRM_EXTRA_REUSED_FROZEN_CODE_FILES),
        "base_stage_code_files": list(prot.PROTOCOL_CODE_FILES),
        "base_reused_frozen_code_files": list(prot.REUSED_FROZEN_CODE_FILES),
        "code_sha256": hashes,
        "base_code_sha256": prot.protocol_code_hashes(),
        "environment": prot._environment_versions(),
        "inventory_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest(),
    }


# --- §2.1 last-paragraph source expansion (explicit path+SHA256, measured) -----------------------


def _expand_response_predictors(binding_path: str | Path | None = None) -> dict[str, Any]:
    """Resolve the five response predictors from the binding receipt's EXPLICIT path+SHA256.

    The path is taken from the binding record (relocated under the current root if the authored
    absolute path moved), then re-hashed and checked against the record's ``file_sha256``; the
    record's non-empty ``signature`` is required. A directory scan or a "best" filename is NEVER
    used. Any drift fails closed.
    """
    if binding_path is None:
        binding_path = STAGE3B_BINDING_RECEIPT
    binding_path = _resolve(binding_path)
    if not binding_path.is_file():
        raise prot.PreflightError(f"Stage 3B binding receipt missing: {binding_path}")
    binding = _read_json(binding_path)["receipt_payload"]
    out: dict[str, Any] = {}
    for seed in prot.RESPONSE_PREDICTOR_SEEDS:
        record = binding["checkpoints"][str(seed)]
        path = r2confirm._relocate(record["path"])
        if not path.is_file():
            raise prot.PreflightError(f"response predictor missing: {path}")
        digest = sha256_file(path)
        if digest != str(record["file_sha256"]):
            raise prot.PreflightError(
                f"response predictor seed {seed} drifted from its binding file hash")
        if not str(record.get("signature", "")):
            raise prot.PreflightError(
                f"response predictor seed {seed} binding record lost its signature")
        out[f"stage3b/response_predictor_seed{seed}.pt"] = {
            "path": str(path), "sha256": digest}
    return out


def _expand_chain_assets(files: dict[str, Any]) -> None:
    """Register the frozen R2-chain assets (predictors, binding, scaler, branch risk model).

    Mirrors the R2 confirm stage's real executed dependency expansion: the Stage 4A/4B frozen
    sources (hash-checked against their pins), the five Full/History expert pairs and end-to-end
    mixture routers (via the frozen Stage 3R path helpers), the five response predictors (via
    the binding receipt's explicit path+hash), the binding receipt itself and the train-only
    trajectory scaler. Every entry is measured (re-hashed); nothing is listed on existence.
    """

    def _register(label: str, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise prot.PreflightError(f"frozen chain source missing: {path}")
        files[label] = {"path": str(path), "sha256": sha256_file(path)}

    stage4_pinned = {rel: sha for _, (rel, sha) in s4c.STAGE4_FROZEN.items()}
    for rel, expected in stage4_pinned.items():
        _register(rel, _resolve(rel))
        if files[rel]["sha256"] != expected:
            raise prot.PreflightError(f"frozen source hash drift: {rel}")

    stage3r_dir = _resolve(STAGE3R_RUN_DIR)
    for pair_index, router_seed in enumerate(prot.EXPERT_ROUTER_SEEDS):
        for condition in ("full", "history"):
            _register(
                f"stage3r/pair{pair_index}_fulltrain_{condition}_seed{router_seed}.pt",
                s4p.expert_checkpoint_path(stage3r_dir, pair_index, condition, router_seed))
        _register(
            f"stage3r/end_to_end_mixture_lc0_lr0_pair{pair_index}_s{router_seed}.pt",
            s4p.e2e_checkpoint_path(stage3r_dir, pair_index, router_seed))

    binding_path = _resolve(STAGE3B_BINDING_RECEIPT)
    _register("stage3b/input_binding_receipt.json", binding_path)
    files.update(_expand_response_predictors(binding_path))

    _register("stage3r/scaler.json", _resolve(TRAJECTORY_SCALER))
    _register("stage4a_v3/risk/models/response_only_ridge.pt", _resolve(BRANCH_RISK_MODEL))


def build_confirm_source_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the §2.1 frozen sources into the lock's per-file path/hash inventory.

    A strict SUPERSET of Task 1's ten pinned sources (each preserved byte-identically), plus
    the §2.1 last-paragraph expansion: the five Direct checkpoints (located under the R2
    ``models/`` dir via the confirm stage's ``verify_r2_sources`` — hash-verified against
    ``final_selection.json`` AND cross-checked with ``model_inventory.json``), the
    Full/History/E2E predictors, the five response predictors, the binding and the scaler.
    Any drift fails closed before the lock is written.
    """
    checked = prot.validate_config(config)
    # Task 1's ten pinned sources (full rel-path keys, matching the existing inventory file).
    files: dict[str, Any] = {
        rel: dict(entry) for rel, entry in prot.verify_pinned_sources().items()}
    # §2.1 last paragraph: the five Direct checkpoints, explicitly (hash + cross-check).
    r2_dev = _resolve(checked["r2_development_dir"])
    sources = r2confirm.verify_r2_sources(r2_dev)
    for seed_text, entry in sources["checkpoints"].items():
        files[f"r2_development/models/direct_seed{seed_text}.pt"] = {
            "path": entry["path"], "sha256": entry["sha256"]}
    # the Full/History/E2E predictors, response predictors, binding, scaler, branch risk model.
    _expand_chain_assets(files)
    return {
        "plan_id": PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "n_files": len(files),
        "files": files,
        "r2_train_npz_sha256": prot.R2_TRAIN_NPZ_SHA256,
        "inventory_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest(),
    }


# --- lock-time global seed audit, re-derived FOR REAL (carried obligation #1) ---------------------


def run_real_seed_audit(
    config: Mapping[str, Any], roots: Sequence[Path] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-run the FULL historical scan + zero-overlap verdict for all five candidate domains.

    This is the real lock-time audit: it never trusts a recorded flag. The historical
    inventory is scanned fresh (``stage4d_protocol.scan_historical_group_seeds``), each
    candidate schedule is audited against it, and the five domains are proven mutually
    disjoint. Returns ``(seed_audit, seed_inventory)``; the audit carries the recorded
    ``zero_overlap``/``overlap`` fields ``prot.verify_protocol_lock`` checks, plus the live
    inventory hash that binds the audit to the scan.
    """
    checked = prot.validate_config(config)
    if roots is None:
        roots = [_resolve(rel) for rel in checked["historical_scan_roots"]]
    roots = [_resolve(r) for r in roots]
    inventory = prot.scan_historical_group_seeds(roots)
    inventory_sha = hashlib.sha256(inventory.tobytes()).hexdigest()

    per_domain: dict[str, Any] = {}
    schedules: dict[str, np.ndarray] = {}
    for role in prot.SEED_ROLES:
        entropy, size = _SEED_DOMAINS[role]
        schedule = prot.make_candidate_seed_schedule(entropy, size)
        audit = prot.audit_global_seed_overlap(schedule, roots, inventory=inventory)
        schedules[role] = schedule
        per_domain[role] = {
            "entropy": int(entropy),
            "size": int(size),
            "schedule_raw_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "n_schedule": audit.n_schedule,
            "n_inventory": audit.n_inventory,
            "overlap": [int(v) for v in audit.overlap],
            "zero_overlap": bool(audit.zero_overlap),
        }
    roles = list(schedules)
    mutual_disjoint = all(
        np.intersect1d(schedules[roles[i]], schedules[roles[j]]).size == 0
        for i in range(len(roles)) for j in range(i + 1, len(roles)))
    overlap_all = sorted({v for d in per_domain.values() for v in d["overlap"]})
    zero_overlap = mutual_disjoint and all(d["zero_overlap"] for d in per_domain.values())

    seed_audit = {
        "zero_overlap": bool(zero_overlap),
        "overlap": overlap_all,
        "mutual_disjoint": bool(mutual_disjoint),
        "per_domain": per_domain,
        "n_inventory": int(inventory.size),
        "inventory_sha256": inventory_sha,
        "derived": ("re-run live at lock/verify time: full historical scan + per-domain "
                    "zero-overlap + mutual disjointness; never a copied flag"),
    }
    seed_inventory = {
        "plan_id": PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "n_inventory": int(inventory.size),
        "inventory_sha256": inventory_sha,
        "scan_roots": [str(r) for r in roots],
        "domains": {
            role: {"entropy": per_domain[role]["entropy"],
                   "size": per_domain[role]["size"],
                   "schedule_raw_sha256": per_domain[role]["schedule_raw_sha256"]}
            for role in prot.SEED_ROLES},
        "action_stream_roots": {role: list(root)
                                for role, root in prot.ACTION_STREAM_ROOTS.items()},
        "reserved_entropies": json.loads(json.dumps(prot.RESERVED_ENTROPIES)),
    }
    return seed_audit, seed_inventory


# --- §7.1 condition-8 strata (two t_star layers + two R2-train-only initial-distance layers) ------


def derive_strata_definitions(config: Mapping[str, Any]) -> dict[str, Any]:
    """The §7.1-condition-8 strata, locked at lock time.

    t_star split is the frozen Stage 4C contract (``stage4cr2_statistics.TSTAR_SPLIT == 13``).
    The initial-distance threshold is the R2-train-only median via
    ``prot.derive_r2_train_distance_threshold`` over the pre-intervention min ego distance,
    reading ONLY the R2 train NPZ's ``object_history`` (hash-verified; CF-future/regret fields
    never touched). The rule is fixed forever (``prot.R2_TRAIN_DISTANCE_THRESHOLD_RULE``).
    """
    id_features = d0.load_r2_train_id_features(config)
    threshold = prot.derive_r2_train_distance_threshold(id_features["min_ego_distance"])
    split = int(s4cr2st.TSTAR_SPLIT)
    return {
        "t_star": {
            "split": split,
            "low": f"t_star < {split}",
            "high": f"t_star >= {split}",
            "source": "stage4cr2_statistics.TSTAR_SPLIT (frozen Stage 4C contract)",
        },
        "initial_distance": {
            "threshold": float(threshold),
            "rule": prot.R2_TRAIN_DISTANCE_THRESHOLD_RULE,
            "low": "min_ego_distance <= threshold",
            "high": "min_ego_distance > threshold",
            "source": ("median of the pre-intervention min ego distance over R2 train "
                       "groups; reads ONLY the R2 train NPZ object_history (hash-verified)"),
            "r2_train_npz_sha256": prot.R2_TRAIN_NPZ_SHA256,
        },
    }


# --- write-once formal protocol lock ---------------------------------------------------------------


def _write_source_inventory_superset(path: str | Path, inventory: Mapping[str, Any]) -> None:
    """Atomically replace Task 1's ``source_inventory.json`` with the verified superset.

    Every pre-existing entry must be reproduced unchanged (path + sha256) before the replace;
    any removal or alteration of a Task 1 entry fails closed. This is the one authorized
    regeneration of a Task 1 product (plan §10 Task 4 brief), and it only ever ADDS entries.
    """
    path = Path(path)
    if path.is_file():
        existing = _read_json(path)
        for rel, entry in existing.get("files", {}).items():
            new = inventory["files"].get(rel)
            if new is None or str(new["sha256"]) != str(entry["sha256"]) \
                    or Path(new["path"]) != Path(entry["path"]):
                raise prot.PreflightError(
                    f"source inventory superset violation at {rel}: a Task 1 entry would "
                    "be removed or altered")
    payload = dict(inventory)
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def create_protocol_lock(config: Mapping[str, Any]) -> str:
    """Create this stage's unique write-once protocol lock; fail closed. A second call raises.

    Requires the frozen D1 power report (``D1_POWER_FROZEN`` with a selected n) and the
    create-once D0 profile selection. Re-derives the seed audit FOR REAL, builds the superset
    source inventory and the extended code inventory, locks the §7.1-condition-8 strata, then
    writes create-once ``source_inventory.json`` (superset-verified replace),
    ``code_inventory.json``, ``seed_inventory.json`` and ``protocol_lock.json`` under
    ``run_dir/protocol``. No D1 collection, no Direct OOD inference, no confirm resource beyond
    the lock artifacts.
    """
    checked = prot.validate_config(config)
    paths = checked["paths"]
    run_dir = _resolve(checked["run_dir"])
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if lock_path.exists():
        raise FileExistsError(f"Stage 4D protocol lock already exists: {lock_path}")

    power_path = _resolve(paths["power_report"])
    if not power_path.is_file():
        raise FileNotFoundError(
            f"D1 power report missing: {power_path}; run the power phase first")
    power = _read_json(power_path)
    if power.get("status") != prot.STATUS_D1_POWER_FROZEN \
            or power.get("selected_n") is None:
        raise ValueError(
            f"the D1 power must be frozen ({prot.STATUS_D1_POWER_FROZEN}) with a selected "
            f"n before locking; got status={power.get('status')!r} "
            f"selected_n={power.get('selected_n')!r}")
    selected_n = int(power["selected_n"])

    selection_path = _resolve(paths["ood_profile_selection"])
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"D0 profile selection missing: {selection_path}; run select-profile first")
    selection = _read_json(selection_path)
    ood_profile = str(selection["selected_profile"])
    selection_sha = sha256_file(selection_path)

    roots = [_resolve(rel) for rel in checked["historical_scan_roots"]]
    seed_audit, seed_inventory = run_real_seed_audit(checked, roots)
    if not seed_audit["zero_overlap"]:
        raise prot.PreflightError(
            "lock-time seed audit is not clean (fail-closed before locking)")

    source_inventory = build_confirm_source_inventory(checked)
    code_inventory = build_confirm_code_inventory()
    strata = derive_strata_definitions(checked)

    protocol_dir = run_dir / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    source_inventory_path = _resolve(paths["source_inventory"])
    code_inventory_path = _resolve(paths["code_inventory"])
    seed_inventory_path = _resolve(paths["seed_inventory"])
    _write_source_inventory_superset(source_inventory_path, source_inventory)
    write_once_json(code_inventory_path, code_inventory)
    write_once_json(seed_inventory_path, seed_inventory)

    d1_schedule = prot.make_candidate_seed_schedule(
        prot.D1_CANDIDATE_SEED_ENTROPY, prot.D1_CANDIDATE_SCHEDULE_SIZE)
    payload = {
        "plan_id": PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "config_signature": prot._config_signature(checked),
        "config": json.loads(json.dumps(checked)),
        "ood_profile": ood_profile,
        "ood_profile_selection_sha256": selection_sha,
        "selected_n": selected_n,
        "candidate_ns": [int(n) for n in prot.CANDIDATE_NS],
        "retention_factors": [float(r) for r in prot.RETENTION_FACTORS],
        "power_report_sha256": sha256_file(power_path),
        "d1_candidate_seed_entropy": prot.D1_CANDIDATE_SEED_ENTROPY,
        "d1_candidate_schedule_size": prot.D1_CANDIDATE_SCHEDULE_SIZE,
        "d1_candidate_schedule_raw_sha256": hashlib.sha256(
            d1_schedule.tobytes()).hexdigest(),
        "d1_bootstrap_entropy": prot.D1_BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": prot.BOOTSTRAP_RESAMPLES,
        "confidence": prot.CONFIDENCE,
        "action_stream_roots": {role: list(root)
                                for role, root in prot.ACTION_STREAM_ROOTS.items()},
        "seed_audit": seed_audit,
        "seed_inventory_sha256": sha256_file(seed_inventory_path),
        "frozen_source_inventory": source_inventory["files"],
        "source_inventory_sha256": sha256_file(source_inventory_path),
        "strata": strata,
        "architecture": prot.ARCH_DIRECT,
        "model_seeds": [int(s) for s in prot.MODEL_SEEDS],
        "response_predictor_seeds": list(prot.RESPONSE_PREDICTOR_SEEDS),
        "expert_router_seeds": list(prot.EXPERT_ROUTER_SEEDS),
        "expected_trainable_parameters": prot.EXPECTED_TRAINABLE_PARAMETERS,
        "primary_score_definition": prot.PRIMARY_SCORE_DEFINITION,
        "ensemble_definition": prot.ENSEMBLE_DEFINITION,
        "primary_baseline": prot.PRIMARY_BASELINE,
        "standard_baselines": list(prot.STANDARD_BASELINES),
        "reference_baselines": list(prot.REFERENCE_BASELINES),
        "high_regret_threshold": prot.HIGH_REGRET_THRESHOLD,
        "coverage": prot.COVERAGE,
        "gate_task_ood": dict(checked["gate_task_ood"]),
        "gate_ood_risk": dict(checked["gate_ood_risk"]),
        "action_plans": [list(p) for p in prot.ACTION_PLANS],
        "environment_params": {
            "env_id": prot.ENV_ID, "history_len": prot.HISTORY_LEN,
            "horizon": prot.HORIZON, "n_actions": prot.N_ACTIONS,
            "n_branches_per_group": prot.N_BRANCHES_PER_GROUP,
            "t_star_low": prot.T_STAR_LOW, "t_star_high": prot.T_STAR_HIGH,
            "simulation_frequency": prot.SIMULATION_FREQUENCY,
            "policy_frequency": prot.POLICY_FREQUENCY,
            "vehicles_count": prot.VEHICLES_COUNT,
            "observed_vehicles_count": prot.OBSERVED_VEHICLES_COUNT,
            "duration": prot.ENV_DURATION, "object_capacity": prot.OBJECT_CAPACITY,
            "n_object_features": prot.N_OBJECT_FEATURES,
        },
        "analysis_manifest": {
            "power_report": str(power_path),
            "source_inventory": str(source_inventory_path),
            "code_inventory": str(code_inventory_path),
            "seed_inventory": str(seed_inventory_path),
            "ood_profile_selection": str(selection_path),
        },
        "terminal_states": list(prot.TERMINAL_STATES),
        "code_sha256": prot.protocol_code_hashes(),
        "extended_code_sha256": code_inventory["code_sha256"],
        "code_inventory_sha256": sha256_file(code_inventory_path),
        "environment": code_inventory["environment"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path, payload)
    return str(lock_path)


# --- §13 post-lock engineering recovery chain (registry extension; the lock is immutable) --------
#
# Mirrors the Stage 4C-R2 confirm recovery chain (``stage4cr2_confirm_protocol`` section 8)
# adapted to this stage's TWO code sets: the write-once lock's ``code_sha256`` (Task 1's base
# registry, strictly verified by ``prot.verify_protocol_lock``) is NEVER touched by a recovery
# lock; the chain governs ONLY the ``extended_code_sha256`` registry. A recovery lock binds the
# previous lock file's SHA256, may re-bind ONLY this stage's own registered files
# (``CONFIRM_CODE_FILES``) as ``changed_files``, and may REGISTER NEW files as ``added_files``
# (a new stage-own module, or a newly-reused frozen prior-stage dependency cross-bound to that
# stage's own lock chain). Frozen dependencies can never be re-bound. The original lock file is
# never modified.

#: recovery locks live next to the protocol lock and chain in numeric order (r1, r2, ...).
RECOVERY_LOCK_GLOB = "recovery_lock_r*.json"
_RECOVERY_LOCK_PREFIX = "recovery_lock_r"

#: the frozen stage4cr2 confirm lock chain that cross-binds reused-frozen additions (cinf).
STAGE4CR2_PROTOCOL_LOCK = (
    "experiments/stage4_reliability/stage4cr2_confirm/protocol/protocol_lock.json")
STAGE4CR2_RECOVERY_LOCK = (
    "experiments/stage4_reliability/stage4cr2_confirm/protocol/recovery_lock_r1.json")

#: roles an added registry file may carry.
_ADDED_FILE_ROLES = ("stage4d_own", "reused_frozen")


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


def _resolve_reused_frozen_binding(rel: str) -> str:
    """The chain-bound frozen hash of a reused prior-stage file newly added to the registry.

    The stage4cr2 recovery lock file is pinned by this stage's §2.1 source table
    (``prot.STAGE4D_PINNED_SOURCES``); resolving the stage4cr2 chain yields the frozen hash
    the addition must equal. Any drift or an unknown file fails closed.
    """
    if rel != "ac_twin/stage4cr2_confirm_inference.py":
        raise ValueError(
            f"no reused-frozen binding is registered for {rel}; only cinf may be added "
            "as a reused frozen dependency")
    recovery_path = _resolve(STAGE4CR2_RECOVERY_LOCK)
    if not recovery_path.is_file():
        raise prot.PreflightError(f"stage4cr2 recovery lock missing: {recovery_path}")
    pinned = prot.STAGE4D_PINNED_SOURCES[STAGE4CR2_RECOVERY_LOCK]
    if sha256_file(recovery_path) != pinned:
        raise prot.PreflightError(
            "stage4cr2 recovery lock drifted from this stage's pinned source hash")
    chain = r2confirm.resolve_locked_code_hashes(_resolve(STAGE4CR2_PROTOCOL_LOCK))
    if rel not in chain:
        raise ValueError(f"the stage4cr2 lock chain does not bind {rel}")
    return chain[rel]


def _check_added_file(rel: str, record: Mapping[str, Any]) -> str:
    """Validate one ``added_files`` entry; return its declared SHA256."""
    if not isinstance(record, Mapping):
        raise ValueError(f"recovery lock added_files[{rel}] must be a mapping")
    digest = str(record.get("sha256", ""))
    if not digest or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"recovery lock added_files[{rel}] sha256 is not a SHA256 hex")
    role = str(record.get("role", ""))
    if role not in _ADDED_FILE_ROLES:
        raise ValueError(
            f"recovery lock added_files[{rel}] role must be one of {_ADDED_FILE_ROLES}")
    if role == "reused_frozen":
        if str(record.get("bound_by", "")) != "stage4cr2_confirm_lock_chain":
            raise ValueError(
                f"recovery lock added_files[{rel}] reused-frozen additions must be "
                "cross-bound by the stage4cr2 confirm lock chain")
        if digest != _resolve_reused_frozen_binding(rel):
            raise ValueError(
                f"recovery lock added_files[{rel}] disagrees with the frozen hash bound "
                "by the stage4cr2 confirm lock chain")
    return digest


def _verify_recovery_lock(recovery: Mapping[str, Any], *, recovery_path: Path,
                          previous_path: Path, expected_code: Mapping[str, str]
                          ) -> dict[str, str]:
    """Validate ONE recovery lock against its predecessor state; return the new code set.

    A recovery lock is a pure-engineering repair/registry-extension record: it must name
    this plan, bind the previous lock FILE's SHA256 (the original protocol lock for r1,
    the previous recovery lock otherwise), and govern ONLY the extended registry.
    ``changed_files`` re-bind this stage's own registered files (``CONFIRM_CODE_FILES``)
    — a frozen reused dependency or an unregistered file can never be re-bound, and the
    base registry (the lock's ``code_sha256`` set) is immutable. ``added_files`` register
    new files (this stage's new modules, or reused frozen dependencies cross-bound to the
    stage4cr2 lock chain). Each changed file's ``before_sha256`` must equal the chained
    state it replaces; the declared ``extended_code_sha256`` must be exactly the chained
    set with the declared changes/additions applied. Any inconsistency fails closed.
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
    added = recovery.get("added_files")
    if not isinstance(changed, Mapping) or not isinstance(added, Mapping):
        raise ValueError(
            f"recovery lock {recovery_path.name} must carry changed_files and "
            "added_files mappings")
    if not changed and not added:
        raise ValueError(
            f"recovery lock {recovery_path.name} carries no changed_files/added_files")
    base_set = set(prot.PROTOCOL_CODE_FILES) | set(prot.REUSED_FROZEN_CODE_FILES)
    new_code = dict(expected_code)
    for rel, record in changed.items():
        if rel in CONFIRM_EXTRA_REUSED_FROZEN_CODE_FILES or rel in base_set:
            raise ValueError(
                f"recovery lock {recovery_path.name} re-binds the frozen dependency "
                f"{rel}; frozen files can never change")
        if rel not in CONFIRM_CODE_FILES:
            raise ValueError(
                f"recovery lock {recovery_path.name} re-binds code file {rel} that is "
                "not one of this stage's own registered files")
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
    for rel, record in added.items():
        if rel in expected_code:
            raise ValueError(
                f"recovery lock {recovery_path.name} adds {rel}, which is already in "
                "the chained code set")
        if rel in base_set:
            raise ValueError(
                f"recovery lock {recovery_path.name} adds {rel}, which belongs to the "
                "immutable base registry")
        new_code[rel] = _check_added_file(rel, record)
    declared = recovery.get("extended_code_sha256")
    if not isinstance(declared, Mapping) or dict(declared) != new_code:
        raise ValueError(
            f"recovery lock {recovery_path.name} declared extended_code_sha256 "
            "disagrees with its changed_files/added_files applied to the chained state")
    return new_code


def resolve_locked_extended_code_hashes(lock_path: str | Path) -> dict[str, str]:
    """Effective EXTENDED code-hash set bound by the original lock plus its recovery chain.

    The original protocol lock stays the immutable root; each recovery lock must bind the
    previous lock file's SHA256. With no recovery locks present the effective set is
    exactly the lock's ``extended_code_sha256``. Fail closed on a missing lock, a broken
    chain binding, or an inconsistent recovery lock.
    """
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"protocol lock missing: {lock_path}")
    payload = _read_json(lock_path)
    if payload.get("plan_id") != PLAN_ID:
        raise ValueError(f"lock plan_id must be {PLAN_ID!r}")
    code = payload.get("extended_code_sha256")
    if not isinstance(code, Mapping):
        raise ValueError("the protocol lock does not carry an extended_code_sha256 mapping")
    expected = dict(code)
    previous = lock_path
    for recovery_path in recovery_lock_paths(lock_path):
        recovery = _read_json(recovery_path)
        expected = _verify_recovery_lock(
            recovery, recovery_path=recovery_path, previous_path=previous,
            expected_code=expected)
        previous = recovery_path
    return expected


def create_recovery_lock(
    config: Mapping[str, Any],
    *,
    changed_files: Mapping[str, str],
    added_files: Mapping[str, Mapping[str, Any]],
    reason: str,
    reused_artifacts: Mapping[str, Any] | None = None,
    base_verify_fn: Any = None,
) -> str:
    """Append an immutable recovery lock binding a pure-engineering registry extension.

    Plan §13 (post-lock, pre-``confirmation_used.lock``): the original write-once lock is
    preserved; the recovery lock binds the previous lock file's SHA256, the before/after
    hashes of this stage's own changed files, the hashes of newly registered files (a
    reused frozen dependency's hash must equal the stage4cr2 chain-bound value), the
    reason, and the unchanged scientific bindings (config signature, source/seed/code
    inventories, frozen power report, D0 selection, selected n, OOD profile). The
    resulting declared extended set must equal the current on-disk registry
    (``confirm_code_hashes()``) — the registry constant and the recovery lock are created
    together or not at all. ``base_verify_fn`` is a test hook for the (expensive) base
    lock verification precondition.
    """
    checked = prot.validate_config(config)
    run_dir = _resolve(checked["run_dir"])
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"the write-once protocol lock must exist before a recovery chain: {lock_path}")
    (base_verify_fn or prot.verify_protocol_lock)(run_dir)  # the root must be green
    resolved = resolve_locked_extended_code_hashes(lock_path)
    chain = recovery_lock_paths(lock_path)
    name = f"recovery_lock_r{len(chain) + 1}.json"
    previous = chain[-1] if chain else lock_path
    lock_payload = _read_json(lock_path)

    base_set = set(prot.PROTOCOL_CODE_FILES) | set(prot.REUSED_FROZEN_CODE_FILES)
    changed: dict[str, Any] = {}
    for rel, summary in (changed_files or {}).items():
        if rel not in CONFIRM_CODE_FILES or rel in base_set:
            raise ValueError(
                f"a recovery lock may re-bind only this stage's own registered files "
                f"(CONFIRM_CODE_FILES); refused: {rel}")
        if rel not in resolved:
            raise ValueError(f"{rel} is absent from the chained extended code set")
        before = resolved[rel]
        after = sha256_file(ROOT / "code" / rel)
        if after == before:
            raise ValueError(f"refused no-op recovery change of {rel}")
        changed[rel] = {"before_sha256": before, "after_sha256": after,
                        "diff_summary": str(summary)}
    added: dict[str, Any] = {}
    for rel, record in (added_files or {}).items():
        if rel in resolved:
            raise ValueError(f"{rel} is already in the chained extended code set")
        if rel in base_set:
            raise ValueError(f"{rel} belongs to the immutable base registry")
        file_path = ROOT / "code" / rel
        if not file_path.is_file():
            raise FileNotFoundError(f"added code file missing on disk: {rel}")
        digest = sha256_file(file_path)
        entry = {"sha256": digest, "role": str(record.get("role", "")),
                 "diff_summary": str(record.get("diff_summary", ""))}
        if entry["role"] == "reused_frozen":
            bound = _resolve_reused_frozen_binding(rel)
            if digest != bound:
                raise ValueError(
                    f"the on-disk hash of reused frozen dependency {rel} does not equal "
                    "the stage4cr2-chain-bound frozen hash")
            entry["bound_by"] = "stage4cr2_confirm_lock_chain"
        elif entry["role"] != "stage4d_own":
            raise ValueError(f"added_files[{rel}] role must be one of {_ADDED_FILE_ROLES}")
        added[rel] = entry
    if not changed and not added:
        raise ValueError("a recovery lock must change or add at least one registry file")

    new_set = dict(resolved)
    for rel, record in changed.items():
        new_set[rel] = record["after_sha256"]
    for rel, record in added.items():
        new_set[rel] = record["sha256"]
    current = confirm_code_hashes()
    if new_set != current:
        raise ValueError(
            "the declared recovery result does not equal the current on-disk extended "
            "registry (CONFIRM_REGISTRY and the recovery lock must be created together)")

    paths = checked["paths"]
    config_signature = prot._config_signature(checked)
    if config_signature != lock_payload.get("config_signature"):
        raise ValueError("the running config drifted from the locked configuration; a "
                         "recovery lock may only bind pure-engineering code changes")
    unchanged = {
        "protocol_lock_sha256": sha256_file(lock_path),
        "config_signature": config_signature,
        "source_inventory_sha256": sha256_file(_resolve(paths["source_inventory"])),
        "seed_inventory_sha256": sha256_file(_resolve(paths["seed_inventory"])),
        "code_inventory_sha256": sha256_file(_resolve(paths["code_inventory"])),
        "power_report_sha256": sha256_file(_resolve(paths["power_report"])),
        "ood_profile_selection_sha256": sha256_file(
            _resolve(paths["ood_profile_selection"])),
        "selected_n": int(lock_payload["selected_n"]),
        "ood_profile": str(lock_payload["ood_profile"]),
        "note": ("科学配置、数据、模型、分数、统计全未改 — pure-engineering registry "
                 "extension; the write-once lock and every scientific binding are "
                 "byte-identical before/after"),
    }
    payload = {
        "plan_id": PLAN_ID,
        "stage_namespace": prot.STAGE_NAMESPACE,
        "kind": "pure_engineering_recovery",
        "recovery_lock": name.removesuffix(".json"),
        "previous_lock": previous.name,
        "binds_previous_lock_sha256": sha256_file(previous),
        "reason": str(reason),
        "changed_files": changed,
        "added_files": added,
        "extended_code_sha256": new_set,
        "unchanged_artifacts": unchanged,
        "reused_artifacts": dict(reused_artifacts or {}),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path.parent / name, payload)
    if resolve_locked_extended_code_hashes(lock_path) != current:
        raise RuntimeError("the freshly written recovery lock failed re-verification")
    return str(lock_path.parent / name)


def verify_confirm_code_chain(run_dir: str | Path) -> dict[str, Any]:
    """Cheap chain-aware lock guard: base verification + extended registry drift check.

    Runs Task 1's ``prot.verify_protocol_lock`` (plan id, config signature, frozen source
    inventory, base code set, recorded audit) and then resolves the EXTENDED code set
    through the recovery chain, comparing it against the current on-disk registry. The
    expensive live seed-audit re-run stays with ``verify_confirm_protocol_lock`` (run
    before collection); this guard is for the score-side phases (mirrors the stage4cr2
    cheap local guard).
    """
    run_dir = _resolve(run_dir)
    payload = prot.verify_protocol_lock(run_dir)
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if confirm_code_hashes() != resolve_locked_extended_code_hashes(lock_path):
        raise ValueError(
            "extended locked code SHA256 set mismatch (code drift or tampering; no "
            "valid recovery-lock chain binds the current code)")
    return payload


# --- extended lock verification (base Task-1 check + extended registry + live audit) ---------------


def verify_confirm_protocol_lock(run_dir: str | Path) -> dict[str, Any]:
    """Re-derive every D1 lock binding; fail closed on any tampering. Read-only.

    Runs Task 1's ``prot.verify_protocol_lock`` first (plan id, config signature, frozen source
    inventory, base code hashes, recorded clean audit), then adds the Task-4 extensions: the
    extended code registry drift check, the ``code_inventory.json`` sidecar integrity, the
    ``selected_n``/``ood_profile`` consistency with the frozen power report and D0 selection,
    the §7.1 strata presence, and a LIVE re-run of the global seed audit (carried obligation #1
    — the recorded flag alone is never trusted).
    """
    run_dir = _resolve(run_dir)
    payload = prot.verify_protocol_lock(run_dir)  # base check (fail-closed without a lock)
    config = prot.validate_config(_read_json(prot.CONFIG_PATH))
    paths = config["paths"]

    extended = payload.get("extended_code_sha256")
    if not isinstance(extended, Mapping):
        raise ValueError("the protocol lock does not carry an extended_code_sha256 mapping")
    lock_path = run_dir / "protocol" / "protocol_lock.json"
    if confirm_code_hashes() != resolve_locked_extended_code_hashes(lock_path):
        raise ValueError(
            "extended locked code SHA256 set mismatch (code drift or tampering; no "
            "valid recovery-lock chain binds the current code)")
    code_inventory_path = _resolve(paths["code_inventory"])
    if not code_inventory_path.is_file():
        raise FileNotFoundError(f"locked code inventory missing: {code_inventory_path}")
    if sha256_file(code_inventory_path) != payload.get("code_inventory_sha256"):
        raise ValueError("locked code_inventory.json drifted")

    power_path = _resolve(paths["power_report"])
    if not power_path.is_file():
        raise FileNotFoundError(f"locked power report missing: {power_path}")
    power = _read_json(power_path)
    if int(payload.get("selected_n", -1)) != int(power.get("selected_n", -2)):
        raise ValueError("lock selected_n drifted from the frozen power report")
    if sha256_file(power_path) != payload.get("power_report_sha256"):
        raise ValueError("frozen power report hash drift")
    selection_path = _resolve(paths["ood_profile_selection"])
    if sha256_file(selection_path) != payload.get("ood_profile_selection_sha256"):
        raise ValueError("D0 profile selection drifted from the lock binding")
    strata = payload.get("strata")
    if not isinstance(strata, Mapping) or "t_star" not in strata \
            or "initial_distance" not in strata:
        raise ValueError("lock does not carry the §7.1-condition-8 strata definitions")

    roots = [_resolve(rel) for rel in config["historical_scan_roots"]]
    audit, _ = run_real_seed_audit(config, roots)
    if not audit["zero_overlap"] or audit["overlap"]:
        raise ValueError(
            f"live lock-time seed audit is not clean (overlap={audit['overlap'][:10]}); "
            "the recorded flag alone is never trusted")
    return payload


__all__ = [
    "BRANCH_RISK_MODEL",
    "CONFIRM_CODE_FILES",
    "CONFIRM_EXTRA_REUSED_FROZEN_CODE_FILES",
    "CONFIRM_REGISTRY",
    "PLAN_ID",
    "RECOVERY_LOCK_GLOB",
    "STAGE3B_BINDING_RECEIPT",
    "STAGE3R_RUN_DIR",
    "STAGE4CR2_PROTOCOL_LOCK",
    "STAGE4CR2_RECOVERY_LOCK",
    "TRAJECTORY_SCALER",
    "build_confirm_code_inventory",
    "build_confirm_source_inventory",
    "confirm_code_hashes",
    "create_protocol_lock",
    "create_recovery_lock",
    "derive_strata_definitions",
    "recovery_lock_paths",
    "resolve_locked_extended_code_hashes",
    "run_real_seed_audit",
    "verify_confirm_code_chain",
    "verify_confirm_protocol_lock",
]
