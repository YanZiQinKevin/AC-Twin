"""Stage 4D-D2F protocol: read-only source binding + dual-pool capacity audit + fresh-ID roles.

Plan ``AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1`` sections
1, 2.2 and 3.1. D2F is a versioned RECOVERY in a NEW namespace: ``stage4d_external/`` (D2),
``stage4d_d2r/`` (D2R) and ``stage4d_d2e/`` (D2E) are strictly READ-ONLY; every D2F artifact
lands under ``experiments/stage4_reliability/stage4d_d2f/``. Nothing here trains, collects,
locks or adjudicates; the Phase-P audit chain is: source binding -> dual-pool capacity audit
-> corrected power (``stage4d_d2f_power``). No new screen/confirm manifest, lock, smoke or
collection may be created by this module.

Source binding (plan section 1): the nine plan-listed SHA256 are recomputed live and any drift
is fail-closed. The binding is EXPANDED to every actual checkpoint (recomputed on disk, never
trusted from a receipt alone), scaler, action/adapter/environment contract, the MetaDrive /
ScenarioNet exact commits, the Waymo inventory root receipt and the executed code hashes.

Dual-pool capacity audit (plan 2.2, read-only): re-scan the frozen
``scenario_split_pools.jsonl`` plus every persisted role's accepted / checkpoint / append-only
ledger attempt IDs and reservations, and classify each prefilter survivor as
``attempted_burned`` (in any ledger / accepted meta, any old role), ``reserved_never_attempted``
(in an old role pool but never attempted) — split into the OLD CONFIRM pool (preserved for the
D2F confirm ONLY; the screen must never encroach on it) and the remaining never-attempted
scenarios, which are the ONLY legal source of the new D2F screen role (plan 3.1 "其余未尝试场景",
new salt ``AC-TWIN-D2F-V0.1|``). One scenario ID maps to exactly one context and one role.
Capacity uses the frozen ``5 x n`` floor; the screen pool additionally reserves at most 160
engineering-smoke candidate IDs. The admissible n sets are the intersection of the plan
candidate sets (screen {1000,1500,2000,2500}; confirm {1000,1500,2000,2500,3000}) with the
capacity-feasible set and are frozen BEFORE the corrected power runs; if either set is empty
the terminal state is ``D2F_FRESH_POOL_CAPACITY_INSUFFICIENT`` and no screen manifest / lock /
collection is created.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments" / "stage4_reliability"
D2F_DIR = EXPERIMENTS / "stage4d_d2f"
D2_DIR = EXPERIMENTS / "stage4d_external"
D2R_DIR = EXPERIMENTS / "stage4d_d2r"
D2E_DIR = EXPERIMENTS / "stage4d_d2e"

PLAN_ID = "AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1"

# --- frozen plan constants (sections 2-3) ----------------------------------------------------------
SCREEN_SALT = "AC-TWIN-D2F-V0.1|"
POWER_SEED = 2026122406                    # fresh frozen power entropy, registered in the report
SMOKE_CANDIDATE_BUDGET = 160               # max engineering-smoke candidate IDs (screen pool)
CAPACITY_FACTOR = 5                        # frozen 5 x n collection-capacity floor
SCREEN_CANDIDATE_NS = (1000, 1500, 2000, 2500)
CONFIRM_CANDIDATE_NS = (1000, 1500, 2000, 2500, 3000)

SOURCE_BINDING_NAME = "source_binding.json"
POOL_AUDIT_NAME = "pool_capacity_audit.json"

STATUS_AUDIT_PASS = "D2F_POOL_AUDIT_PASS"                      # descriptive, NON-terminal
STATUS_CAPACITY_INSUFFICIENT = "D2F_FRESH_POOL_CAPACITY_INSUFFICIENT"  # preregistered terminal
STATUS_SOURCE_INVALID = "D2F_SOURCE_OR_ENGINEERING_INVALID"            # preregistered terminal

# --- Phase S artifact names + terminal states (plan sections 3.1-3.3, 7) ------------------------------
FRESH_ID_VERIFICATION_NAME = "fresh_id_verification.json"
POST_SMOKE_AUDIT_NAME = "pool_capacity_audit_post_smoke.json"
SCREEN_MANIFEST_NAME = "screen_manifest.json"
SCREEN_LOCK_NAME = "screen_protocol_lock.json"
CODE_INVENTORY_NAME = "code_inventory.json"
MODEL_INVENTORY_NAME = "model_inventory.json"
SMOKE_DIR_NAME = "smoke"
SCREEN_ROLE = "development_screen"

STATUS_SCREEN_POOL_INSUFFICIENT = "D2F_FRESH_SCREEN_POOL_INSUFFICIENT"  # Phase S terminal (plan 7)

# --- Phase C (confirm) names, fresh entropies and preregistered states (plan 4) -----------------------
CONFIRM_ROLE = "confirm"
CONFIRM_DIR_NAME = "confirm"
CONFIRM_MANIFEST_NAME = "confirm_manifest.json"
CONFIRM_LOCK_NAME = "confirm_protocol_lock.json"
CONFIRM_CODE_INVENTORY_NAME = "confirm_code_inventory.json"
CONFIRM_POWER_REPORT_REL = Path(CONFIRM_DIR_NAME) / "power_report.json"
D2F_CONFIRM_POWER_SEED = 2026122408        # fresh frozen confirm-power entropy, registered
D2F_CONFIRM_BOOTSTRAP_ENTROPY = 2026122409  # fresh frozen confirm-adjudication entropy, registered
STATUS_CONFIRM_POWER_PASS = "D2F_CONFIRM_POWER_PASS"            # descriptive, NON-terminal
STATUS_CONFIRM_UNDERPOWERED = "D2F_CONFIRM_UNDERPOWERED"        # preregistered terminal (plan 7)
STATUS_CONFIRM_POOL_INSUFFICIENT = "D2F_FRESH_CONFIRM_POOL_INSUFFICIENT"  # terminal (plan 7)
_SCREEN_REPORT_REL = Path("development") / "development_screen_report.json"

#: Fresh frozen bootstrap entropy for the D2F screen adjudication (10,000 whole-group paired
#: resamples, two-sided 80% CI). Distinct from every consumed stream (D2 screen 2026122101,
#: D2E power 2026122405, D2F power 2026122406 and the R2/D1 streams). FLAGGED for review (the
#: plan fixes resamples/CI but names no seed, mirroring the D2 registration).
D2F_SCREEN_BOOTSTRAP_ENTROPY = 2026122407

#: the D2F screen-chain executed-code registry (plan 3.3: dependencies are LISTED in the
#: registry and then hashed one by one; a missing file is fail-closed, never silently skipped).
D2F_SCREEN_CODE_REGISTRY = (
    "code/collect_stage4d_external.py",
    "code/collect_stage4d_d2f.py",
    "code/stage4d_metadrive_adapter.py",
    "code/stage4d_sim_entry.py",
    "code/stage4d_split_screen_truth.py",
    "code/drive_stage4d_external_screen.py",
    "code/drive_stage4d_d2f.py",
    "code/ac_twin/stage4d_external_features.py",
    "code/ac_twin/stage4d_external_split.py",
    "code/ac_twin/stage4d_external_training.py",
    "code/ac_twin/stage4d_external_inference.py",
    "code/ac_twin/stage4d_external_statistics.py",
    "code/ac_twin/stage4d_d2e_ego.py",
    "code/ac_twin/stage4d_d2e_pipeline.py",
    "code/ac_twin/stage4d_d2f_protocol.py",
    "code/ac_twin/stage4d_d2f_power.py",
    "code/ac_twin/stage4d_d2f_inference.py",
    "code/ac_twin/stage4d_d2f_statistics.py",
    "code/ac_twin/stage4c_objective.py",
    "code/ac_twin/stage4cp_features.py",
    "code/ac_twin/stage4cp_statistics.py",
)

#: old roles whose pools / ledgers / accepted metas are scanned (D2R/D2E never collected; their
#: absence of a ``roles/`` directory is asserted and recorded as zero attempts).
OLD_ROLE_DIRS = ("smoke", "pilot", "pilot_replay", "train", "validation", "development_screen",
                 "confirm")
OLD_CONFIRM_ROLE = "confirm"

# --- the nine plan-listed sources (section 1 table; drift is fail-closed) ---------------------------
PLAN_LISTED_SOURCES = (
    ("stage4d_d2e/development/final_selection.json",
     "c2236a9b7798c702cbe6651c30fcac41d037343b3fd7eb984e75495c25e65259",
     "D2E frozen mechanical selection (EGO_TRAJECTORY_PLUS_RANK)"),
    ("stage4d_d2e/development/validation_report.json",
     "ca7890c3035eb564f653f44f58a50dd87c5b0c0b9100c8b0bbcc6f37c2e08787",
     "D2E old-validation development qualification (ADAPTIVE_DEVELOPMENT_ONLY)"),
    ("stage4d_d2e/development/power_validation_quantities.npz",
     "023b83827cd278707e59dfcb8928917679e761a6993a9514378d66186985f21a",
     "D2E repaired-pipeline old-validation per-group quantities (power input)"),
    ("stage4d_d2e/power/screen_power_report.json",
     "d2570e290d42068de32037a6e3ecabe8d4ccc0ffda6f84a5c23904d603b1f75e",
     "OLD failed D2E power report: bound/explained only, NEVER a new power result"),
    ("stage4d_d2e/development/ego_models/ego_fit_receipt.json",
     "c48ea6600c110da68cb3e1bf8b4a0ff77d09fa4cf66651a148bae4d17c309c8e",
     "D2E ego-candidate fit receipt"),
    ("stage4d_d2e/riskfit_EGO_TRAJECTORY_PLUS_RANK/development/models/fit_receipt.json",
     "c0d8d251c3116bc79aadfcb78a7699e637abe460dd6f197a47d3768fbb645145",
     "D2E retrained Direct + standard-baseline fit receipt"),
    ("stage4d_external/development/base_models/predictor_stack.json",
     "44de5543818b3644f7cfabf73040949628e2308c7425c06b13eaed4a99f86d60",
     "D2 frozen Full/History/E2E + response predictor stack"),
    ("stage4d_external/platform/scenario_split_manifest.json",
     "6c243960582a46b3b55ab20adda17e3532cf958d69d17c43b9505b7951c2c874",
     "D2 frozen scenario role split manifest"),
    ("code/ac_twin/stage4c_objective.py",
     "a20da1805c14b57bc2ad02f363133b7c53c480909190f52fa5a1cc09964ac44e",
     "frozen ten-action consequence function (reused bitwise)"),
)

#: platform commits frozen by the D2 plan section 2.2 (verified live, read-only).
METADRIVE_ROOT = Path(r"D:\MetaDrive\metadrive")
SCENARIONET_ROOT = Path(r"D:\MetaDrive\scenarionet")
FROZEN_METADRIVE_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"
FROZEN_SCENARIONET_COMMIT = "d4acdb5f5a844744fc85cb2dc3880d7d4a6eb170"

#: D2F power/audit executed code (hashed into the binding; the screen lock later expands this).
D2F_CODE_FILES = (
    "code/ac_twin/stage4d_d2f_protocol.py",
    "code/ac_twin/stage4d_d2f_power.py",
    "code/drive_stage4d_d2f.py",
)
#: frozen executed code the D2F power path imports (read-only reuse; hashed for the binding).
FROZEN_REUSED_CODE_FILES = (
    "code/ac_twin/stage4d_d2e_power.py",
    "code/ac_twin/stage4d_external_statistics.py",
    "code/ac_twin/stage4d_external_training.py",
    "code/ac_twin/stage4cp_statistics.py",
)


class SourceDriftError(RuntimeError):
    """A plan-listed or expanded source hash / commit drifted from its frozen value."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_head(repo: Path) -> str:
    """Read-only ``git rev-parse HEAD``; empty string when the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=60, check=True)
        return out.stdout.strip()
    except Exception:
        return ""


# --- source binding (plan section 1) -----------------------------------------------------------------


def recompute_plan_listed_sources() -> list[dict[str, Any]]:
    """Recompute the nine plan-listed SHA256 live; fail closed on the first drift."""
    entries: list[dict[str, Any]] = []
    for rel, expected, note in PLAN_LISTED_SOURCES:
        path = ROOT / "experiments" / "stage4_reliability" / rel \
            if not rel.startswith("code/") else ROOT / rel
        if not path.is_file():
            raise SourceDriftError(f"plan-listed source missing: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            raise SourceDriftError(
                f"plan-listed source drifted: {rel} expected {expected} got {actual}")
        entries.append({"path": rel, "sha256": actual, "expected_sha256": expected,
                        "match": True, "note": note})
    return entries


def _hash_existing(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise SourceDriftError(f"expanded binding source missing ({label}): {path}")
    return {"path": str(path), "sha256": sha256_file(path), "label": label}


def expand_source_binding() -> dict[str, Any]:
    """Expand the binding to checkpoints / scalers / contracts / commits / executed code.

    Every checkpoint SHA is recomputed on disk and cross-checked against its frozen receipt
    (a receipt is never trusted without a live recompute). Absolute paths in frozen receipts
    are tolerated (cross-machine move); content hashes must not drift.
    """
    expanded: dict[str, Any] = {"checkpoints": [], "scalers": [], "contracts": [],
                                "platform_commits": {}, "code": []}

    final = _read_json(D2E_DIR / "development" / "final_selection.json")
    selected = str(final["selected_candidate"])
    for seed, rel in sorted(final["ego_checkpoints"].items()):
        expanded["checkpoints"].append(_hash_existing(
            D2E_DIR / "development" / "ego_models" / Path(str(rel)).name,
            f"d2e ego checkpoint seed{seed} ({selected})"))

    ego_receipt = _read_json(D2E_DIR / "development" / "ego_models" / "ego_fit_receipt.json")
    scaler_path = ROOT / str(ego_receipt["scaler_path"])
    scaler_entry = _hash_existing(scaler_path, "d2e ego scaler")
    if scaler_entry["sha256"] != str(ego_receipt["scaler_sha256"]):
        raise SourceDriftError("d2e ego scaler drifted from its fit receipt")
    expanded["scalers"].append(scaler_entry)

    riskfit = _read_json(
        D2E_DIR / str(final["risk_run_dir"]) / "development" / "models" / "fit_receipt.json")
    for seed, path_str in sorted(riskfit["checkpoint_paths"].items()):
        entry = _hash_existing(Path(path_str), f"d2e direct checkpoint seed{seed}")
        expected = str(riskfit.get("checkpoint_sha256", {}).get(seed, ""))
        if expected and entry["sha256"] != expected:
            raise SourceDriftError(
                f"d2e direct checkpoint seed{seed} drifted from its fit receipt")
        expanded["checkpoints"].append(entry)
    scaler = riskfit.get("scaler")
    if isinstance(scaler, Mapping) and scaler.get("path"):
        expanded["scalers"].append(_hash_existing(Path(str(scaler["path"])),
                                                  "d2e direct scaler"))

    stack = _read_json(D2_DIR / "development" / "base_models" / "predictor_stack.json")
    for key in ("expert_paths", "response_paths", "risk_paths", "router_paths"):
        for name, path_str in sorted(stack.get(key, {}).items()):
            expanded["checkpoints"].append(_hash_existing(
                Path(str(path_str)), f"d2 {key} {name}"))
    if stack.get("scaler_path"):
        expanded["scalers"].append(_hash_existing(Path(str(stack["scaler_path"])),
                                                  "d2 platform train-only scaler"))

    for rel, label in (
            ("stage4d_external/platform/action_contract.json", "d2 action contract"),
            ("stage4d_external/platform/adapter_contract.json", "d2 adapter contract"),
            ("stage4d_external/platform/platform_environment_lock.json", "d2 environment lock"),
            ("stage4d_external/platform/scenario_split_pools.jsonl", "d2 frozen pools"),
            ("stage4d_external/platform/raw_dataset_root_receipt.json",
             "waymo raw inventory root receipt"),
            ("stage4d_external/config.json", "d2 config"),
            ("stage4d_d2e/config.json", "d2e config")):
        expanded["contracts"].append(_hash_existing(
            EXPERIMENTS / rel, label))

    metadrive_head = _git_head(METADRIVE_ROOT)
    scenarionet_head = _git_head(SCENARIONET_ROOT)
    commits = {"metadrive": {"root": str(METADRIVE_ROOT), "head": metadrive_head,
                             "frozen": FROZEN_METADRIVE_COMMIT,
                             "match": metadrive_head == FROZEN_METADRIVE_COMMIT},
               "scenarionet": {"root": str(SCENARIONET_ROOT), "head": scenarionet_head,
                               "frozen": FROZEN_SCENARIONET_COMMIT,
                               "match": scenarionet_head == FROZEN_SCENARIONET_COMMIT}}
    if not all(c["match"] for c in commits.values()):
        raise SourceDriftError(f"platform commit drift: {commits}")
    expanded["platform_commits"] = commits

    for rel in FROZEN_REUSED_CODE_FILES + D2F_CODE_FILES:
        expanded["code"].append(_hash_existing(ROOT / rel, "executed code"))
    return expanded


def run_source_binding(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write the create-once ``source_binding.json``; a rerun re-verifies and reloads.

    Fail closed (``D2F_SOURCE_OR_ENGINEERING_INVALID``) on any listed/expanded drift. The
    binding is a Phase-P prerequisite: no capacity audit / power / collection may run on an
    unbound or drifted source chain.
    """
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    binding_path = D2F_DIR / SOURCE_BINDING_NAME
    if binding_path.exists():
        existing = _read_json(binding_path)
        recompute_plan_listed_sources()  # fail closed on drift even when the binding exists
        if existing.get("status") != STATUS_AUDIT_PASS:
            raise SourceDriftError(f"existing source binding has status "
                                   f"{existing.get('status')!r}")
        if progress:
            progress("source_binding.json exists (create-once); listed sources re-verified")
        return existing

    if progress:
        progress("recomputing the nine plan-listed source SHA256 (fail closed on drift)")
    listed = recompute_plan_listed_sources()
    if progress:
        progress("expanding the binding to checkpoints / scalers / contracts / commits / code")
    expanded = expand_source_binding()
    payload = {
        "plan_id": PLAN_ID,
        "status": STATUS_AUDIT_PASS,
        "scope": "READ_ONLY_BINDING",
        "create_once": True,
        "listed_sources": listed,
        "expanded": expanded,
        "frozen_boundaries": {
            "read_only_dirs": [str(D2_DIR), str(D2R_DIR), str(D2E_DIR)],
            "old_d2_screen": ("the old 4,000-group D2 screen and its development_screen_used.lock "
                              "are never used for D2F training / selection / power / per-group "
                              "analysis / gates; only its public terminal state is quoted"),
            "old_d2e_power": ("the D2E screen_power_report.json (D2E_SCREEN_UNDERPOWERED) is "
                              "bound and explained, never overwritten or re-adjudicated"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(binding_path, payload)
    if progress:
        progress(f"source binding COMPLETE: {len(listed)} listed + "
                 f"{len(expanded['checkpoints'])} checkpoints + {len(expanded['scalers'])} "
                 f"scalers + {len(expanded['contracts'])} contracts + "
                 f"{len(expanded['code'])} code files")
    return payload


# --- dual-pool capacity audit (plan 2.2, read-only) -----------------------------------------------------


def _ledger_attempt_ids(role_dir: Path) -> tuple[set[str], int]:
    """Union of scenario IDs appearing in a role's append-only ledger (accepted + rejected)."""
    ids: set[str] = set()
    lines = 0
    ledger = role_dir / "rejection_ledger.jsonl"
    if ledger.is_file():
        with ledger.open("r", encoding="utf-8") as handle:
            for line in handle:
                lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scenario_id = record.get("scenario_id")
                if scenario_id:
                    ids.add(str(scenario_id))
    return ids, lines


def _accepted_ids(role_dir: Path) -> set[str]:
    """Accepted scenario IDs from groups_meta.json (and the checkpoint meta cross-check)."""
    ids: set[str] = set()
    meta = role_dir / "groups_meta.json"
    if meta.is_file():
        for entry in json.loads(meta.read_text(encoding="utf-8")):
            if isinstance(entry, Mapping) and entry.get("scenario_id"):
                ids.add(str(entry["scenario_id"]))
    checkpoint = role_dir / "checkpoint.npz"
    if checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as archive:
            if "__meta_json__" in archive.files:
                for entry in json.loads(str(archive["__meta_json__"])):
                    if isinstance(entry, Mapping) and entry.get("scenario_id"):
                        ids.add(str(entry["scenario_id"]))
    return ids


def scan_burned_ids(platform_dir: Path | None = None,
                    progress: Callable[[str], None] | None = None
                    ) -> dict[str, Any]:
    """Union of every attempted / accepted / persisted scenario ID across all old roles.

    Read-only. Attempted means present in ANY role's append-only ledger (accepted or rejected)
    or accepted meta / checkpoint; such IDs are permanently burned for every new role. D2R and
    D2E never collected (no ``roles/`` directory) and are recorded as zero-attempt sources.
    """
    roles_root = (platform_dir or D2_DIR) / "roles"
    per_role: dict[str, Any] = {}
    burned: set[str] = set()
    ledger_files: list[dict[str, str]] = []
    for role in OLD_ROLE_DIRS:
        role_dir = roles_root / role
        attempts, ledger_lines = _ledger_attempt_ids(role_dir)
        accepted = _accepted_ids(role_dir)
        missing_accepted = accepted - attempts
        if missing_accepted:
            raise RuntimeError(
                f"role {role!r}: {len(missing_accepted)} accepted IDs are absent from its "
                f"append-only ledger (audit cannot prove the burned set; fail closed)")
        role_burned = attempts | accepted
        burned |= role_burned
        ledger = role_dir / "rejection_ledger.jsonl"
        if ledger.is_file():
            ledger_files.append({"role": role, "path": str(ledger),
                                 "sha256": sha256_file(ledger)})
        per_role[role] = {
            "ledger_lines": int(ledger_lines),
            "attempted_unique_ids": int(len(attempts)),
            "accepted_ids": int(len(accepted)),
            "burned_ids": int(len(role_burned)),
        }
    for name, directory in (("stage4d_d2r", D2R_DIR), ("stage4d_d2e", D2E_DIR)):
        roles_dir = directory / "roles"
        per_role[name] = {"roles_dir_present": bool(roles_dir.is_dir()),
                          "attempted_unique_ids": 0,
                          "note": "plan reached a terminal state before any collection"}
        if roles_dir.is_dir():
            raise RuntimeError(
                f"{name} unexpectedly has a roles/ directory; the audit cannot certify the "
                f"fresh pools (fail closed)")
    if progress:
        progress(f"burned-set scan complete: {len(burned)} unique attempted/accepted IDs "
                 f"across {len(OLD_ROLE_DIRS)} old roles (D2R/D2E collected nothing)")
    return {"burned": burned, "per_role": per_role, "ledger_files": ledger_files}


def load_pool_rows(pools_path: Path | None = None) -> list[dict[str, Any]]:
    """Frozen prefilter-survivor pool rows (global_rank, pool, role, scenario_id, shard)."""
    path = pools_path or (D2_DIR / "platform" / "scenario_split_pools.jsonl")
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def classify_pools(rows: Sequence[Mapping[str, Any]], burned: set[str]) -> dict[str, Any]:
    """Three-way classification (plan 2.2): attempted/burned vs reserved-never-attempted.

    The never-attempted IDs split into the OLD CONFIRM pool (preserved for the D2F confirm
    only) and the remaining never-attempted scenarios (the only legal D2F screen source).
    """
    all_ids = {str(row["scenario_id"]) for row in rows}
    confirm_ids = {str(row["scenario_id"]) for row in rows
                   if str(row["role"]) == OLD_CONFIRM_ROLE}
    if len(all_ids) != len(rows):
        raise ValueError("duplicate scenario_id in the frozen pools (one scenario = one role)")
    overlap = confirm_ids & burned
    available_confirm = confirm_ids - burned
    screen_available = all_ids - burned - confirm_ids
    by_old_role: dict[str, dict[str, int]] = {}
    for row in rows:
        role = str(row["role"])
        bucket = by_old_role.setdefault(role, {"attempted_burned": 0,
                                               "reserved_never_attempted": 0})
        key = "attempted_burned" if str(row["scenario_id"]) in burned \
            else "reserved_never_attempted"
        bucket[key] += 1
    return {
        "all_ids": all_ids,
        "confirm_ids": confirm_ids,
        "burned_in_pools": all_ids & burned,
        "confirm_pool_attempted_overlap": overlap,
        "available_confirm": available_confirm,
        "screen_available": screen_available,
        "by_old_role": by_old_role,
    }


def admissible_ns(available: int, candidates: Sequence[int], *, reserve: int = 0,
                  factor: int = CAPACITY_FACTOR) -> list[int]:
    """Candidate n values whose frozen ``factor x n`` floor fits the audited capacity."""
    return [int(n) for n in candidates if int(available) - int(reserve) >= factor * int(n)]


def run_pool_capacity_audit(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Write the create-once ``pool_capacity_audit.json`` (plan 2.2). Read-only.

    Freezes the admissible screen/confirm n sets BEFORE the corrected power runs. Terminal
    ``D2F_FRESH_POOL_CAPACITY_INSUFFICIENT`` when either admissible set is empty; a confirm
    maximum below 3000 is prominently flagged (the plan allows continuing with a non-empty
    pre-frozen set). A rerun reloads the create-once audit after re-verifying the raw files.
    """
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = D2F_DIR / POOL_AUDIT_NAME
    pools_path = D2_DIR / "platform" / "scenario_split_pools.jsonl"
    manifest_path = D2_DIR / "platform" / "scenario_split_manifest.json"
    if audit_path.exists():
        existing = _read_json(audit_path)
        if existing.get("raw_hashes", {}).get("scenario_split_pools") != sha256_file(pools_path):
            raise RuntimeError("the frozen pools drifted after the create-once audit")
        if progress:
            progress("pool_capacity_audit.json exists (create-once); raw pools re-verified")
        return existing

    if progress:
        progress("read-only dual-pool capacity audit: scanning pools + all old-role ledgers")
    rows = load_pool_rows(pools_path)
    scan = scan_burned_ids(progress=progress)
    classification = classify_pools(rows, scan["burned"])
    n_screen = len(classification["screen_available"])
    n_confirm = len(classification["available_confirm"])
    screen_ns = admissible_ns(n_screen, SCREEN_CANDIDATE_NS,
                              reserve=SMOKE_CANDIDATE_BUDGET)
    confirm_ns = admissible_ns(n_confirm, CONFIRM_CANDIDATE_NS)
    terminal = not screen_ns or not confirm_ns
    confirm_max = max(confirm_ns) if confirm_ns else None
    status = STATUS_CAPACITY_INSUFFICIENT if terminal else STATUS_AUDIT_PASS

    payload = {
        "plan_id": PLAN_ID,
        "status": status,
        "terminal": bool(terminal),
        "scope": "READ_ONLY_AUDIT",
        "create_once": True,
        "audit_rule": {
            "attempted_burned": ("any scenario ID in ANY old role's append-only ledger "
                                 "(accepted or rejected) or accepted meta / checkpoint is "
                                 "permanently burned for every new role"),
            "reserved_never_attempted": ("pool IDs never attempted: the OLD CONFIRM pool is "
                                         "preserved for the D2F confirm ONLY (the screen never "
                                         "encroaches on it); every OTHER never-attempted "
                                         "scenario is the only legal D2F screen source "
                                         "(plan 2.2 classification + 3.1)"),
            "one_scenario_one_role": ("each scenario ID maps to exactly one context and one "
                                      "role; old train/validation/pilot/smoke/screen attempts "
                                      "are fully excluded"),
        },
        "per_old_role": scan["per_role"],
        "classification": {
            "prefilter_survivors": int(len(rows)),
            "attempted_burned_unique": int(len(classification["burned_in_pools"])),
            "confirm_pool_total": int(len(classification["confirm_ids"])),
            "confirm_pool_attempted_overlap":
                int(len(classification["confirm_pool_attempted_overlap"])),
            "available_confirm": int(n_confirm),
            "screen_available": int(n_screen),
            "by_old_role": classification["by_old_role"],
        },
        "capacity": {
            "factor": CAPACITY_FACTOR,
            "smoke_candidate_budget": SMOKE_CANDIDATE_BUDGET,
            "screen_candidates": [int(n) for n in SCREEN_CANDIDATE_NS],
            "confirm_candidates": [int(n) for n in CONFIRM_CANDIDATE_NS],
            "screen_admissible_ns": screen_ns,
            "confirm_admissible_ns": confirm_ns,
            "confirm_max_admissible": confirm_max,
            "confirm_below_3000_risk": bool(confirm_ns and confirm_max < 3000),
            "per_n_feasibility": {
                "screen": {str(n): bool(n_screen - SMOKE_CANDIDATE_BUDGET
                                        >= CAPACITY_FACTOR * n)
                           for n in SCREEN_CANDIDATE_NS},
                "confirm": {str(n): bool(n_confirm >= CAPACITY_FACTOR * n)
                            for n in CONFIRM_CANDIDATE_NS},
            },
        },
        "raw_hashes": {
            "scenario_split_pools": sha256_file(pools_path),
            "scenario_split_manifest": sha256_file(manifest_path),
            "ledgers": scan["ledger_files"],
        },
        "note": ("the audit only excludes ex-ante impossible capacities; the post-screen "
                 "confirm power may still be insufficient. If smoke consumes more than the "
                 "160-candidate budget, the collection halts and this read-only audit is "
                 "redone before any lock (plan 2.2)."),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(audit_path, payload)
    if progress:
        progress(f"pool capacity audit COMPLETE status={status} "
                 f"screen_available={n_screen} confirm_available={n_confirm} "
                 f"screen_ns={screen_ns} confirm_ns={confirm_ns}")
    return payload


def assert_smoke_budget(consumed_candidate_ids: int) -> None:
    """Plan 2.2: engineering smoke may consume at most 160 screen-pool candidate IDs.

    If the actual smoke consumption exceeds the budget, collection HALTS and the read-only
    dual-pool audit is redone before any lock; the formal screen quota is never silently
    encroached on.
    """
    if int(consumed_candidate_ids) > SMOKE_CANDIDATE_BUDGET:
        raise RuntimeError(
            f"smoke consumed {consumed_candidate_ids} candidate IDs, exceeding the frozen "
            f"{SMOKE_CANDIDATE_BUDGET}-candidate budget; halt and redo the read-only dual-pool "
            f"capacity audit before any screen lock (plan 2.2)")


# --- fresh-ID role derivation (plan 3.1; used by Phase S, never by Phase P) ---------------------------


def derive_screen_candidate_ids(rows: Sequence[Mapping[str, Any]], burned: set[str],
                                *, salt: str = SCREEN_SALT) -> list[str]:
    """Deterministic fresh screen candidate order: never-attempted, non-confirm IDs, SHA256-salted.

    Legal only in Phase S after the capacity audit passed; Phase P never writes a manifest.
    The returned order is the primary-then-reserve source; zero overlap with every burned and
    every old-confirm ID is asserted (fail closed).
    """
    classification = classify_pools(rows, burned)
    pool = classification["screen_available"]
    if pool & burned:
        raise RuntimeError("fresh screen pool intersects the burned set (fail closed)")
    ordered = sorted(pool, key=lambda sid: hashlib.sha256(
        (salt + sid).encode("utf-8")).hexdigest())
    return ordered


# --- Phase S: fresh-ID verification / smoke accounting / manifest + lock (plan 3.1-3.3) --------------

_STATUS_POWER_PASS = "D2F_POWER_CORRECTION_PASS"   # stage4d_d2f_power.STATUS_POWER_PASS (no cycle)
_POWER_REPORT_REL = Path("power") / "power_correction_report.json"


def load_smoke_burned_ids(smoke_dir: Path | None = None) -> set[str]:
    """Scenario IDs attempted by the D2F engineering smoke (ledger + accepted meta).

    The smoke lives in the D2F namespace, is marked ENGINEERING_SMOKE_ONLY and its IDs are
    appended to the burned set before any screen lock (plan 3.3); they never enter the screen.
    """
    directory = smoke_dir or (D2F_DIR / SMOKE_DIR_NAME)
    attempts, _ = _ledger_attempt_ids(directory)
    return attempts | _accepted_ids(directory)


def load_selected_n() -> int:
    """The frozen ``selected_n`` from the create-once corrected-power report (Phase P GO)."""
    report_path = D2F_DIR / _POWER_REPORT_REL
    if not report_path.is_file():
        raise RuntimeError("the corrected-power report is missing; Phase P must pass first")
    report = _read_json(report_path)
    if report.get("status") != _STATUS_POWER_PASS or report.get("selected_n") is None:
        raise RuntimeError(
            f"corrected-power status is {report.get('status')!r} with selected_n="
            f"{report.get('selected_n')!r}; Phase S requires {_STATUS_POWER_PASS}")
    return int(report["selected_n"])


def run_fresh_id_verification(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Phase S step 1 (plan 3.1): re-verify the fresh screen IDs before any manifest/lock.

    Re-checks the create-once dual-pool audit and pre-frozen n sets, re-scans every old role's
    burned IDs live, re-derives the salt-ranked screen candidate order and proves: zero overlap
    with every burned ID and every old-confirm ID, and an exact count match with the audit.
    Create-once; a rerun re-verifies and reloads.
    """
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    out_path = D2F_DIR / FRESH_ID_VERIFICATION_NAME
    audit = _read_json(D2F_DIR / POOL_AUDIT_NAME)
    if audit.get("status") != STATUS_AUDIT_PASS:
        raise RuntimeError(f"pool capacity audit status {audit.get('status')!r}; fresh-ID "
                           "verification requires D2F_POOL_AUDIT_PASS")
    if out_path.exists():
        existing = _read_json(out_path)
        scan = scan_burned_ids()
        if int(existing["burned_unique_ids"]) != len(scan["burned"]):
            raise RuntimeError("the burned set changed after the create-once fresh-ID "
                               "verification (fail closed)")
        if progress:
            progress("fresh_id_verification.json exists (create-once); burned set re-verified")
        return existing

    if progress:
        progress("fresh-ID re-verification: live burned-set scan + salt-ranked order derivation")
    scan = scan_burned_ids(progress=progress)
    rows = load_pool_rows()
    classification = classify_pools(rows, scan["burned"])
    order = derive_screen_candidate_ids(rows, scan["burned"])
    confirm_ids = classification["confirm_ids"]
    overlap = set(order) & (scan["burned"] | confirm_ids)
    if overlap:
        raise RuntimeError(f"fresh screen order overlaps burned/confirm IDs: {len(overlap)}")
    expected = int(audit["classification"]["screen_available"])
    if len(order) != expected or len(order) != len(classification["screen_available"]):
        raise RuntimeError(
            f"fresh screen candidate count {len(order)} != audited {expected} (fail closed)")
    digest = hashlib.sha256("\n".join(order).encode("utf-8")).hexdigest()
    payload = {
        "plan_id": PLAN_ID,
        "status": STATUS_AUDIT_PASS,
        "scope": "READ_ONLY_VERIFICATION",
        "create_once": True,
        "salt": SCREEN_SALT,
        "screen_candidate_count": int(len(order)),
        "screen_order_sha256": digest,
        "burned_unique_ids": int(len(scan["burned"])),
        "confirm_pool_preserved": int(len(confirm_ids)),
        "zero_overlap_burned": True,
        "zero_overlap_confirm": True,
        "prefrozen_screen_ns": audit["capacity"]["screen_admissible_ns"],
        "prefrozen_confirm_ns": audit["capacity"]["confirm_admissible_ns"],
        "bindings": {
            "pool_capacity_audit_sha256": sha256_file(D2F_DIR / POOL_AUDIT_NAME),
            "source_binding_sha256": sha256_file(D2F_DIR / SOURCE_BINDING_NAME),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)
    if progress:
        progress(f"fresh-ID verification COMPLETE: {len(order)} zero-overlap candidates "
                 f"(order sha256 {digest[:12]}...)")
    return payload


def run_post_smoke_pool_audit(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Phase S step 3 (plan 3.3): read-only dual-pool re-audit AFTER smoke, BEFORE any lock.

    Appends the smoke-consumed IDs to the burned set and re-proves the ``5 x selected_n``
    screen capacity and the pre-frozen confirm candidate set. Fails closed (terminal
    ``D2F_FRESH_SCREEN_POOL_INSUFFICIENT``) if either no longer holds; a smoke overrun past
    the 160-candidate budget halts first (plan 2.2). Create-once.
    """
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    out_path = D2F_DIR / POST_SMOKE_AUDIT_NAME
    if out_path.exists():
        existing = _read_json(out_path)
        if int(existing["smoke_burned_ids"]) != len(load_smoke_burned_ids()):
            raise RuntimeError("the smoke burned set changed after the create-once post-smoke "
                               "audit (fail closed)")
        if progress:
            progress("pool_capacity_audit_post_smoke.json exists (create-once); re-verified")
        return existing

    selected_n = load_selected_n()
    smoke_burned = load_smoke_burned_ids()
    assert_smoke_budget(len(smoke_burned))
    scan = scan_burned_ids(progress=progress)
    rows = load_pool_rows()
    classification = classify_pools(rows, scan["burned"])
    pre_smoke_screen = classification["screen_available"]
    confirm_ids = classification["confirm_ids"]
    if smoke_burned - pre_smoke_screen:
        raise RuntimeError(
            f"{len(smoke_burned - pre_smoke_screen)} smoke IDs are outside the fresh screen "
            "pool (smoke must consume screen-pool candidates only; fail closed)")
    if smoke_burned & confirm_ids:
        raise RuntimeError("smoke encroached on the preserved confirm pool (fail closed)")
    screen_after = pre_smoke_screen - smoke_burned
    capacity_ok = len(screen_after) >= CAPACITY_FACTOR * selected_n
    audit = _read_json(D2F_DIR / POOL_AUDIT_NAME)
    confirm_ns = [int(n) for n in audit["capacity"]["confirm_admissible_ns"]]
    confirm_ok = bool(confirm_ns) and all(
        len(confirm_ids) >= CAPACITY_FACTOR * n for n in confirm_ns)
    terminal = not (capacity_ok and confirm_ok)
    payload = {
        "plan_id": PLAN_ID,
        "status": STATUS_SCREEN_POOL_INSUFFICIENT if terminal else STATUS_AUDIT_PASS,
        "terminal": bool(terminal),
        "scope": "READ_ONLY_AUDIT",
        "create_once": True,
        "selected_n": int(selected_n),
        "smoke_burned_ids": int(len(smoke_burned)),
        "smoke_budget": SMOKE_CANDIDATE_BUDGET,
        "screen_available_pre_smoke": int(len(pre_smoke_screen)),
        "screen_available_post_smoke": int(len(screen_after)),
        "screen_capacity_required": int(CAPACITY_FACTOR * selected_n),
        "screen_capacity_ok": bool(capacity_ok),
        "confirm_pool_preserved": int(len(confirm_ids)),
        "confirm_prefrozen_ns": confirm_ns,
        "confirm_capacity_ok": bool(confirm_ok),
        "bindings": {
            "pool_capacity_audit_sha256": sha256_file(D2F_DIR / POOL_AUDIT_NAME),
            "power_report_sha256": sha256_file(D2F_DIR / _POWER_REPORT_REL),
            "smoke_ledger_sha256": sha256_file(D2F_DIR / SMOKE_DIR_NAME / "rejection_ledger.jsonl")
            if (D2F_DIR / SMOKE_DIR_NAME / "rejection_ledger.jsonl").is_file() else "",
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)
    if progress:
        progress(f"post-smoke pool audit COMPLETE status={payload['status']} "
                 f"screen_after={len(screen_after)} required={CAPACITY_FACTOR * selected_n} "
                 f"confirm_preserved={len(confirm_ids)}")
    return payload


def build_screen_manifest(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Phase S freeze (plan 3.1/3.3): the create-once ``screen_manifest.json``.

    Legal only after the post-smoke audit passed. Primary = the first ``5 x selected_n`` of the
    post-smoke salt-ranked order (the frozen capacity floor), reserve = the remainder; the
    collector attempts them in this order under the frozen acceptance rules until ``selected_n``
    groups are accepted. Binds source shards, the t_star candidate-order rule, the acceptance
    rules and every upstream hash.
    """
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    out_path = D2F_DIR / SCREEN_MANIFEST_NAME
    if out_path.exists():
        if progress:
            progress("screen_manifest.json exists (create-once); reloaded")
        return _read_json(out_path)
    post = _read_json(D2F_DIR / POST_SMOKE_AUDIT_NAME)
    if post.get("status") != STATUS_AUDIT_PASS:
        raise RuntimeError(f"post-smoke audit status {post.get('status')!r}; the screen "
                           "manifest is never created on a terminal capacity state")
    selected_n = int(post["selected_n"])
    smoke_burned = load_smoke_burned_ids()
    scan = scan_burned_ids()
    rows = load_pool_rows()
    order = derive_screen_candidate_ids(rows, scan["burned"] | smoke_burned)
    if len(order) != int(post["screen_available_post_smoke"]):
        raise RuntimeError("post-smoke candidate count drifted from the create-once audit")
    primary = order[: CAPACITY_FACTOR * selected_n]
    reserve = order[CAPACITY_FACTOR * selected_n:]
    shards = {str(r["scenario_id"]): str(r.get("source_shard", "")) for r in rows}
    payload = {
        "plan_id": PLAN_ID,
        "status": STATUS_AUDIT_PASS,
        "create_once": True,
        "role": SCREEN_ROLE,
        "salt": SCREEN_SALT,
        "selected_n": int(selected_n),
        "acceptance_rule": ("frozen collect_stage4d_external eligibility + structural-simulation "
                            "QA reason codes; candidates are attempted in manifest order until "
                            "selected_n groups are accepted; every attempt is appended to the "
                            "append-only ledger (burn-once)"),
        "t_star_order_rule": ("stage4d_external_features.t_star_candidate_order with the D2F "
                              "salt (SHA256-ranked candidates, frozen bitwise from D2)"),
        "capacity": {"factor": CAPACITY_FACTOR,
                     "primary_ids": int(len(primary)),
                     "reserve_ids": int(len(reserve)),
                     "smoke_burned_ids": int(len(smoke_burned))},
        "primary_ids": primary,
        "reserve_ids": reserve,
        "source_shards": {sid: shards.get(sid, "") for sid in primary},
        "bindings": {
            "source_binding_sha256": sha256_file(D2F_DIR / SOURCE_BINDING_NAME),
            "pool_capacity_audit_sha256": sha256_file(D2F_DIR / POOL_AUDIT_NAME),
            "post_smoke_audit_sha256": sha256_file(D2F_DIR / POST_SMOKE_AUDIT_NAME),
            "power_report_sha256": sha256_file(D2F_DIR / _POWER_REPORT_REL),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)
    if progress:
        progress(f"screen manifest FROZEN: selected_n={selected_n} primary={len(primary)} "
                 f"reserve={len(reserve)}")
    return payload


def screen_code_registry() -> list[dict[str, Any]]:
    """Hash every listed screen-chain code dependency (plan 3.3: listed first, then hashed).

    Fail-closed when a registered file is missing — a dependency is never silently skipped.
    """
    entries: list[dict[str, Any]] = []
    for rel in D2F_SCREEN_CODE_REGISTRY:
        path = ROOT / rel
        if not path.is_file():
            raise SourceDriftError(f"registered screen-chain code file missing: {rel}")
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return entries


def write_code_inventory(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Freeze ``code_inventory.json`` (create-once): the full D2F screen-chain code registry."""
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    out_path = D2F_DIR / CODE_INVENTORY_NAME
    if out_path.exists():
        existing = _read_json(out_path)
        current = {e["path"]: e["sha256"] for e in screen_code_registry()}
        recorded = {e["path"]: e["sha256"] for e in existing["code"]}
        if current != recorded:
            raise SourceDriftError(
                "screen-chain code drifted after the create-once code inventory; a versioned "
                "recovery (new code hashes, old lock preserved) is required, never an in-place "
                "rewrite (plan 4)")
        if progress:
            progress("code_inventory.json exists (create-once); registry re-verified")
        return existing
    entries = screen_code_registry()
    payload = {"plan_id": PLAN_ID, "create_once": True, "code": entries,
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_once_json(out_path, payload)
    if progress:
        progress(f"code inventory FROZEN: {len(entries)} registered code files hashed")
    return payload


def write_model_inventory(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Freeze ``model_inventory.json`` (create-once): every model artifact the scorer loads.

    Reuses the create-once source binding's live-recomputed checkpoint/scaler hashes and adds
    the D2E final selection / validation report / riskfit receipt bindings (the selected ego
    candidate and the Direct checkpoints pair one-to-one by hash).
    """
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    out_path = D2F_DIR / MODEL_INVENTORY_NAME
    if out_path.exists():
        if progress:
            progress("model_inventory.json exists (create-once); reloaded")
        return _read_json(out_path)
    binding = _read_json(D2F_DIR / SOURCE_BINDING_NAME)
    expanded = binding["expanded"]
    d2e_dev = D2E_DIR / "development"
    final = _read_json(d2e_dev / "final_selection.json")
    selected = str(final["selected_candidate"])
    payload = {
        "plan_id": PLAN_ID,
        "create_once": True,
        "selected_ego_candidate": selected,
        "checkpoints": expanded["checkpoints"],
        "scalers": expanded["scalers"],
        "bindings": {
            "source_binding_sha256": sha256_file(D2F_DIR / SOURCE_BINDING_NAME),
            "d2e_final_selection_sha256": sha256_file(d2e_dev / "final_selection.json"),
            "d2e_validation_report_sha256": sha256_file(d2e_dev / "validation_report.json"),
            "d2e_ego_fit_receipt_sha256": sha256_file(
                d2e_dev / "ego_models" / "ego_fit_receipt.json"),
            "d2e_riskfit_receipt_sha256": sha256_file(
                D2E_DIR / str(final["risk_run_dir"]) / "development" / "models"
                / "fit_receipt.json"),
            "d2_predictor_stack_sha256": sha256_file(
                D2_DIR / "development" / "base_models" / "predictor_stack.json"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)
    if progress:
        progress(f"model inventory FROZEN: {len(expanded['checkpoints'])} checkpoints + "
                 f"{len(expanded['scalers'])} scalers (selected={selected})")
    return payload


def freeze_screen_lock(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Freeze ``screen_protocol_lock.json`` (create-once) — the last artifact before collection.

    Binds the manifest / code / model inventories, the frozen gates (D2 section 8.2 c1-c8 and
    8.3 r1-r6 verbatim, with the completeness condition evaluated at ``selected_n``), the
    10,000-resample 80% bootstrap and its fresh entropy, and the preregistered terminal states.
    """
    D2F_DIR.mkdir(parents=True, exist_ok=True)
    out_path = D2F_DIR / SCREEN_LOCK_NAME
    if out_path.exists():
        if progress:
            progress("screen_protocol_lock.json exists (create-once); reloaded")
        return _read_json(out_path)
    manifest = _read_json(D2F_DIR / SCREEN_MANIFEST_NAME)
    selected_n = int(manifest["selected_n"])
    from ac_twin import stage4d_external_statistics as D2STAT  # lazy: single threshold source
    payload = {
        "plan_id": PLAN_ID,
        "create_once": True,
        "role": SCREEN_ROLE,
        "salt": SCREEN_SALT,
        "selected_n": int(selected_n),
        "screen_groups": int(selected_n),
        "one_shot": ("split -> score (truth-free) -> >=32-group determinism -> freeze scores -> "
                     "burn development_screen_used.lock -> the SINGLE adjudication opens the "
                     "sealed truth; no rerun / resample / baseline / model / threshold change "
                     "after any failure (plan 3.5)"),
        "bootstrap": {"resamples": int(D2STAT.BOOTSTRAP_RESAMPLES),
                      "confidence": float(D2STAT.CONFIDENCE_SCREEN),
                      "entropy": D2F_SCREEN_BOOTSTRAP_ENTROPY,
                      "unit": "whole-group paired"},
        "gates": {
            "task_x_dev": {
                "c1": f"groups == {selected_n} and all QA flags",
                "c2": f"high-regret frac >= {D2STAT.GATE_TASK_MIN_HIGH_REGRET_FRAC}",
                "c3": (f">= {D2STAT.GATE_TASK_MIN_DISTINCT_ORACLE} oracle actions each >= "
                       f"{D2STAT.GATE_TASK_MIN_ORACLE_FRAC}"),
                "c4": "branch Spearman 80% CI lower > 0",
                "c5": ("base beats random (point + CI lower > 0) and base mean regret <= idle "
                       f"+ {D2STAT.BASELINE_TIE_TOLERANCE}"),
                "c6": f"reactive frac >= {D2STAT.GATE_TASK_MIN_REACTIVE_FRAC}",
                "c7": f"every stratum frac >= {D2STAT.GATE_TASK_MIN_STRATUM_FRAC}",
                "c8": "denominators / projection / evaluation mask non-degenerate",
            },
            "risk_x_dev": {
                "r1": f"I_AURC >= {D2STAT.I_AURC_MIN} and CI lower > 0",
                "r2": f"I_REGRET80 >= {D2STAT.I_REGRET80_MIN} and CI lower > 0",
                "r3": "PR-AUC minus prevalence > 0 (point + CI lower)",
                "r4": f">= {D2STAT.MIN_POSITIVE_SEEDS}/5 seeds both improvements positive",
                "r5": "baseline denominators valid",
                "r6": "no leak / no baseline swap / no model swap (QA)",
            },
            "frozen_primary_baseline": "SEED_MARGIN_STD",
            "baseline_source": ("D2E validation_report candidate_gate[EGO_TRAJECTORY_PLUS_RANK]"
                                ".strongest_baseline (repaired-pipeline old validation, frozen)"),
        },
        "terminal_states": {
            "task_fail": "D2F_EXTERNAL_DEV_TASK_NOT_READY",
            "risk_fail": "D2F_EXTERNAL_DEV_RISK_NOT_READY",
            "invalid": "INCONCLUSIVE_D2F_EXTERNAL_INVALID",
            "both_pass": "D2F_DEV_READY_FOR_CONFIRM_POWER",
        },
        "bindings": {
            "screen_manifest_sha256": sha256_file(D2F_DIR / SCREEN_MANIFEST_NAME),
            "code_inventory_sha256": sha256_file(D2F_DIR / CODE_INVENTORY_NAME),
            "model_inventory_sha256": sha256_file(D2F_DIR / MODEL_INVENTORY_NAME),
            "source_binding_sha256": sha256_file(D2F_DIR / SOURCE_BINDING_NAME),
            "power_report_sha256": sha256_file(D2F_DIR / _POWER_REPORT_REL),
            "fresh_id_verification_sha256": sha256_file(D2F_DIR / FRESH_ID_VERIFICATION_NAME),
            "post_smoke_audit_sha256": sha256_file(D2F_DIR / POST_SMOKE_AUDIT_NAME),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)
    if progress:
        progress(f"screen protocol lock FROZEN: selected_n={selected_n} entropy="
                 f"{D2F_SCREEN_BOOTSTRAP_ENTROPY}; collection may start")
    return payload


# --- Phase C: confirm power gate / manifest / code inventory / protocol lock (plan 4) ----------------


def load_screen_report_status() -> str:
    """The completed D2F screen adjudication status (the confirm chain gates on it)."""
    report_path = D2F_DIR / _SCREEN_REPORT_REL
    if not report_path.is_file():
        raise RuntimeError("the D2F screen adjudication report is missing; the confirm phase "
                           "may only start after BOTH screen gates passed (plan 4)")
    return str(_read_json(report_path).get("status"))


def assert_confirm_phase_open() -> None:
    """Fail closed unless the single screen adjudication passed both gates (plan 4 gate)."""
    status = load_screen_report_status()
    if status != "D2F_DEV_READY_FOR_CONFIRM_POWER":
        raise RuntimeError(f"the D2F screen terminal status is {status!r}, not "
                           "D2F_DEV_READY_FOR_CONFIRM_POWER; no confirm power / manifest / "
                           "lock / collection may be created (plan 4)")


def load_confirm_selected_n() -> int:
    """The confirm selected_n from the create-once confirm power report (must be PASS)."""
    report_path = D2F_DIR / CONFIRM_POWER_REPORT_REL
    if not report_path.is_file():
        raise RuntimeError(f"confirm power report missing: {report_path}; run confirm-power "
                           "before any confirm manifest/lock/collection (plan 4)")
    report = _read_json(report_path)
    if report.get("status") != STATUS_CONFIRM_POWER_PASS:
        raise RuntimeError(f"confirm power status {report.get('status')!r}; the confirm phase "
                           "stops at the preregistered terminal state without a manifest")
    return int(report["selected_n"])


def derive_confirm_candidate_ids(rows: Sequence[Mapping[str, Any]], burned: set[str],
                                 *, salt: str = SCREEN_SALT) -> list[str]:
    """Deterministic confirm candidate order over the PRESERVED old-confirm pool (plan 4).

    Only the never-attempted old-confirm IDs are legal confirm candidates; the screen pool is
    never encroached on and the order is never re-ranked after a shortfall. Zero overlap with
    every burned ID is asserted (fail closed).
    """
    classification = classify_pools(rows, burned)
    pool = classification["available_confirm"]
    if pool & burned:
        raise RuntimeError("the preserved confirm pool intersects the burned set (fail closed)")
    return sorted(pool, key=lambda sid: hashlib.sha256(
        (salt + sid).encode("utf-8")).hexdigest())


def build_confirm_manifest(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Phase C freeze (plan 4): the create-once ``confirm_manifest.json``.

    Legal only after the confirm power report is ``D2F_CONFIRM_POWER_PASS``. Primary = the
    first ``5 x selected_n`` of the salt-ranked preserved-confirm order (the frozen capacity
    floor), reserve = the remainder; if the preserved pool cannot meet the floor the
    preregistered terminal ``D2F_FRESH_CONFIRM_POOL_INSUFFICIENT`` is raised and NO manifest
    is written (the screen pool is never raided and the order never re-ranked).
    """
    assert_confirm_phase_open()
    out_path = D2F_DIR / CONFIRM_MANIFEST_NAME
    if out_path.exists():
        if progress:
            progress("confirm_manifest.json exists (create-once); reloaded")
        return _read_json(out_path)
    selected_n = load_confirm_selected_n()
    smoke_burned = load_smoke_burned_ids()
    scan = scan_burned_ids()
    rows = load_pool_rows()
    order = derive_confirm_candidate_ids(rows, scan["burned"] | smoke_burned)
    required = CAPACITY_FACTOR * selected_n
    if len(order) < required:
        raise RuntimeError(
            f"{STATUS_CONFIRM_POOL_INSUFFICIENT}: the preserved confirm pool has "
            f"{len(order)} never-attempted IDs < {required} (5 x {selected_n}); no confirm "
            "manifest / lock / collection is created (plan 4 terminal)")
    primary = order[:required]
    reserve = order[required:]
    shards = {str(r["scenario_id"]): str(r.get("source_shard", "")) for r in rows}
    payload = {
        "plan_id": PLAN_ID,
        "status": STATUS_AUDIT_PASS,
        "create_once": True,
        "role": CONFIRM_ROLE,
        "salt": SCREEN_SALT,
        "selected_n": int(selected_n),
        "acceptance_rule": ("frozen collect_stage4d_external eligibility + structural-simulation "
                            "QA reason codes; candidates are attempted in manifest order until "
                            "selected_n groups are accepted; reserve fills ONLY engineering "
                            "rejections in attempt order, never outcome replacements (plan 4 / "
                            "D2 10.2); every attempt is appended to the append-only ledger"),
        "t_star_order_rule": ("stage4d_external_features.t_star_candidate_order with the D2F "
                              "salt (SHA256-ranked candidates, frozen bitwise from D2)"),
        "capacity": {"factor": CAPACITY_FACTOR,
                     "primary_ids": int(len(primary)),
                     "reserve_ids": int(len(reserve))},
        "primary_ids": primary,
        "reserve_ids": reserve,
        "source_shards": {sid: shards.get(sid, "") for sid in primary},
        "bindings": {
            "source_binding_sha256": sha256_file(D2F_DIR / SOURCE_BINDING_NAME),
            "pool_capacity_audit_sha256": sha256_file(D2F_DIR / POOL_AUDIT_NAME),
            "post_smoke_audit_sha256": sha256_file(D2F_DIR / POST_SMOKE_AUDIT_NAME),
            "screen_lock_sha256": sha256_file(D2F_DIR / SCREEN_LOCK_NAME),
            "screen_report_sha256": sha256_file(D2F_DIR / _SCREEN_REPORT_REL),
            "confirm_power_report_sha256": sha256_file(D2F_DIR / CONFIRM_POWER_REPORT_REL),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)
    if progress:
        progress(f"confirm manifest FROZEN: selected_n={selected_n} primary={len(primary)} "
                 f"reserve={len(reserve)}")
    return payload


def confirm_code_registry() -> list[dict[str, Any]]:
    """Hash every listed confirm-chain code dependency (plan 4: listed first, then hashed).

    The confirm chain executes the SAME registered files as the screen chain (the confirm
    subcommands live in the same modules); they are re-hashed HERE, at confirm-lock time, into
    an independent confirm inventory — the create-once screen inventory is never rewritten.
    """
    entries: list[dict[str, Any]] = []
    for rel in D2F_SCREEN_CODE_REGISTRY:
        path = ROOT / rel
        if not path.is_file():
            raise SourceDriftError(f"registered confirm-chain code file missing: {rel}")
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return entries


def write_confirm_code_inventory(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Freeze ``confirm_code_inventory.json`` (create-once): the confirm-phase code registry."""
    out_path = D2F_DIR / CONFIRM_CODE_INVENTORY_NAME
    if out_path.exists():
        existing = _read_json(out_path)
        current = {e["path"]: e["sha256"] for e in confirm_code_registry()}
        recorded = {e["path"]: e["sha256"] for e in existing["code"]}
        if current != recorded:
            raise SourceDriftError(
                "confirm-chain code drifted after the create-once confirm code inventory; a "
                "versioned recovery (new code hashes, old lock preserved) is required, never "
                "an in-place rewrite (plan 4)")
        if progress:
            progress("confirm_code_inventory.json exists (create-once); registry re-verified")
        return existing
    entries = confirm_code_registry()
    payload = {"plan_id": PLAN_ID, "create_once": True, "role": CONFIRM_ROLE,
               "note": ("independent confirm-phase inventory: the confirm subcommands extend "
                        "the same registered modules after the screen adjudication completed; "
                        "the screen-phase code_inventory.json is frozen and never rewritten"),
               "code": entries,
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_once_json(out_path, payload)
    if progress:
        progress(f"confirm code inventory FROZEN: {len(entries)} registered code files hashed")
    return payload


def freeze_confirm_lock(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Freeze ``confirm_protocol_lock.json`` (create-once) — the last artifact before collection.

    Binds (plan 4 / D2 10.1): plan ID + both selected n's, the confirm manifest primary/reserve,
    the code/model inventories, the unique frozen primary baseline, the 10,000-resample 95%
    bootstrap with its fresh entropy, the confirm gates (D2 10.3/10.4 verbatim) and the
    preregistered terminal states.
    """
    assert_confirm_phase_open()
    out_path = D2F_DIR / CONFIRM_LOCK_NAME
    if out_path.exists():
        if progress:
            progress("confirm_protocol_lock.json exists (create-once); reloaded")
        return _read_json(out_path)
    manifest = _read_json(D2F_DIR / CONFIRM_MANIFEST_NAME)
    confirm_n = int(manifest["selected_n"])
    screen_n = int(_read_json(D2F_DIR / SCREEN_MANIFEST_NAME)["selected_n"])
    from ac_twin import stage4d_external_statistics as D2STAT  # lazy: single threshold source
    payload = {
        "plan_id": PLAN_ID,
        "create_once": True,
        "role": CONFIRM_ROLE,
        "salt": SCREEN_SALT,
        "selected_n": int(confirm_n),
        "screen_selected_n": int(screen_n),
        "confirm_groups": int(confirm_n),
        "one_shot": ("fresh confirm collection -> split -> score (truth-free) -> >=32-group "
                     "determinism -> freeze scores -> burn confirmation_used.lock -> the SINGLE "
                     "adjudication opens the sealed truth; no rerun / resample / baseline / "
                     "model / threshold change after any failure (plan 4)"),
        "bootstrap": {"resamples": int(D2STAT.BOOTSTRAP_RESAMPLES),
                      "confidence": 0.95,
                      "entropy": D2F_CONFIRM_BOOTSTRAP_ENTROPY,
                      "unit": "whole-group paired"},
        "gates": {
            "task_x_confirm": {
                "source": "D2 plan 10.3: section 8.2 conditions verbatim with every CI-based "
                          "condition at the two-sided 95% CI; completeness at selected_n",
                "c1": f"groups == {confirm_n} and all QA flags",
                "c2": f"high-regret frac >= {D2STAT.GATE_TASK_MIN_HIGH_REGRET_FRAC}",
                "c3": (f">= {D2STAT.GATE_TASK_MIN_DISTINCT_ORACLE} oracle actions each >= "
                       f"{D2STAT.GATE_TASK_MIN_ORACLE_FRAC}"),
                "c4": "branch Spearman 95% CI lower > 0",
                "c5": ("base beats random (point + 95% CI lower > 0) and base mean regret <= "
                       f"idle + {D2STAT.BASELINE_TIE_TOLERANCE}"),
                "c6": f"reactive frac >= {D2STAT.GATE_TASK_MIN_REACTIVE_FRAC}",
                "c7": f"every stratum frac >= {D2STAT.GATE_TASK_MIN_STRATUM_FRAC}",
                "c8": "denominators / projection / evaluation mask non-degenerate",
            },
            "risk_x_confirm": {
                "source": "D2 plan 10.4 verbatim (two-sided 95% CI)",
                "r1": f"I_AURC >= {D2STAT.I_AURC_MIN} and 95% CI lower > 0",
                "r2": f"I_REGRET80 >= {D2STAT.I_REGRET80_MIN} and 95% CI lower > 0",
                "r3": "PR-AUC minus prevalence > 0 (point + 95% CI lower)",
                "r4": f">= {D2STAT.MIN_POSITIVE_SEEDS}/5 seeds both improvements positive",
                "r5": "baseline denominators valid",
                "r6": "no leak / no baseline swap / no model swap (QA)",
            },
            "frozen_primary_baseline": "SEED_MARGIN_STD",
            "baseline_source": ("D2E validation_report candidate_gate[EGO_TRAJECTORY_PLUS_RANK]"
                                ".strongest_baseline (repaired-pipeline old validation, frozen); "
                                "the confirm-strongest standard baseline is disclosed with 95% "
                                "CIs but never substituted (D2 10.4)"),
        },
        "terminal_states": {
            "task_fail_or_invalid": "INCONCLUSIVE_D2F_EXTERNAL_INVALID",
            "risk_fail": "D2F_META_DRIVE_NOT_CONFIRMED",
            "both_pass": "D2F_META_DRIVE_EXTERNAL_CONFIRMED",
        },
        "bindings": {
            "confirm_manifest_sha256": sha256_file(D2F_DIR / CONFIRM_MANIFEST_NAME),
            "confirm_code_inventory_sha256": sha256_file(D2F_DIR / CONFIRM_CODE_INVENTORY_NAME),
            "model_inventory_sha256": sha256_file(D2F_DIR / MODEL_INVENTORY_NAME),
            "source_binding_sha256": sha256_file(D2F_DIR / SOURCE_BINDING_NAME),
            "screen_lock_sha256": sha256_file(D2F_DIR / SCREEN_LOCK_NAME),
            "screen_report_sha256": sha256_file(D2F_DIR / _SCREEN_REPORT_REL),
            "confirm_power_report_sha256": sha256_file(D2F_DIR / CONFIRM_POWER_REPORT_REL),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(out_path, payload)
    if progress:
        progress(f"confirm protocol lock FROZEN: selected_n={confirm_n} entropy="
                 f"{D2F_CONFIRM_BOOTSTRAP_ENTROPY}; confirm collection may start")
    return payload



__all__ = [
    "CAPACITY_FACTOR",
    "CODE_INVENTORY_NAME",
    "CONFIRM_CANDIDATE_NS",
    "CONFIRM_CODE_INVENTORY_NAME",
    "CONFIRM_DIR_NAME",
    "CONFIRM_LOCK_NAME",
    "CONFIRM_MANIFEST_NAME",
    "CONFIRM_POWER_REPORT_REL",
    "CONFIRM_ROLE",
    "D2E_DIR",
    "D2F_CODE_FILES",
    "D2F_CONFIRM_BOOTSTRAP_ENTROPY",
    "D2F_CONFIRM_POWER_SEED",
    "D2F_DIR",
    "D2F_SCREEN_BOOTSTRAP_ENTROPY",
    "D2F_SCREEN_CODE_REGISTRY",
    "D2R_DIR",
    "D2_DIR",
    "EXPERIMENTS",
    "FRESH_ID_VERIFICATION_NAME",
    "FROZEN_METADRIVE_COMMIT",
    "FROZEN_SCENARIONET_COMMIT",
    "FROZEN_REUSED_CODE_FILES",
    "MODEL_INVENTORY_NAME",
    "OLD_CONFIRM_ROLE",
    "OLD_ROLE_DIRS",
    "PLAN_ID",
    "PLAN_LISTED_SOURCES",
    "POOL_AUDIT_NAME",
    "POST_SMOKE_AUDIT_NAME",
    "POWER_SEED",
    "SCREEN_CANDIDATE_NS",
    "SCREEN_LOCK_NAME",
    "SCREEN_MANIFEST_NAME",
    "SCREEN_ROLE",
    "SCREEN_SALT",
    "SMOKE_CANDIDATE_BUDGET",
    "SMOKE_DIR_NAME",
    "SOURCE_BINDING_NAME",
    "STATUS_AUDIT_PASS",
    "STATUS_CAPACITY_INSUFFICIENT",
    "STATUS_CONFIRM_POOL_INSUFFICIENT",
    "STATUS_CONFIRM_POWER_PASS",
    "STATUS_CONFIRM_UNDERPOWERED",
    "STATUS_SCREEN_POOL_INSUFFICIENT",
    "STATUS_SOURCE_INVALID",
    "SourceDriftError",
    "admissible_ns",
    "assert_confirm_phase_open",
    "assert_smoke_budget",
    "build_confirm_manifest",
    "build_screen_manifest",
    "classify_pools",
    "confirm_code_registry",
    "derive_confirm_candidate_ids",
    "derive_screen_candidate_ids",
    "expand_source_binding",
    "freeze_confirm_lock",
    "freeze_screen_lock",
    "load_confirm_selected_n",
    "load_pool_rows",
    "load_screen_report_status",
    "load_selected_n",
    "load_smoke_burned_ids",
    "recompute_plan_listed_sources",
    "run_fresh_id_verification",
    "run_pool_capacity_audit",
    "run_post_smoke_pool_audit",
    "run_source_binding",
    "scan_burned_ids",
    "screen_code_registry",
    "write_code_inventory",
    "write_confirm_code_inventory",
    "write_model_inventory",
]
