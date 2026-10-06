"""Stage 4D-D2 role collector: MetaDrive decision-group collection per frozen role.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 4, 5, 6
and 12.3. Runs INSIDE the frozen Windows sim env via the unique entry point:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File code\\run_stage4d_env.ps1 `
        sim code/collect_stage4d_external.py --role pilot --quota 100

Contract:
* reads the FROZEN role pools (``platform/scenario_split_pools.jsonl``) and processes
  candidates in hash-rank order: the role's primary list first, then its reserve list
  (plan 4.3). Every attempted scenario ID is burned into the append-only rejection
  ledger with a fixed reason code; a burned ID is never retried, even across resume.
* acceptance is structural/pre-outcome ONLY (plan 4.2 eligibility + simulation
  validity). No regret, oracle ranking, collision outcome, or response magnitude is
  ever read to accept/reject.
* one MetaDrive env per collection run (map cache reused within a group's 11 branches,
  cleared between groups to bound memory). The env seed is the scenario's frozen DB
  index, so a full collection re-run is bit-reproducible (fresh-env determinism pinned
  by tests/test_stage4d_external_adapter.py).
* resumable: an npz checkpoint holds accepted arrays + per-group metadata; a config
  fingerprint guards against resuming under a changed contract.
* terminal-safe progress: phase/done/total/%/elapsed/ETA at least every 30 s and a
  final SUCCESS/FAILURE summary.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

from ac_twin import stage4d_external_features as F  # noqa: E402
from ac_twin import stage4d_external_split as split  # noqa: E402
from ac_twin.checkpointing import atomic_save_npz, canonical_fingerprint  # noqa: E402
from ac_twin.progress import ProgressReporter, format_duration  # noqa: E402
import stage4d_metadrive_adapter as A  # noqa: E402

RUN_DIR = PROJECT_ROOT / "experiments/stage4_reliability/stage4d_external"
PLATFORM_DIR = RUN_DIR / "platform"
DB_DIR = Path(r"D:\MetaDrive\scenarionet_waymo_training_20s")
ROLE_SALT = "AC-TWIN-D2-V0.1|"

#: frozen rejection reason codes (plan 4.4 eligibility + structural simulation validity).
ELIGIBILITY_REASONS = (
    F.REASON_NO_SDC,
    F.REASON_SDC_PREFIX_INVALID,
    F.REASON_NO_FUTURE_CAPACITY,
    F.REASON_NO_DRIVABLE_LANE,
    F.REASON_NO_EGO_LANE,
    F.REASON_NO_LEFT_LANE,
    F.REASON_NO_RIGHT_LANE,
    F.REASON_FEW_DYNAMIC_NON_EGO,
    "no_eligible_t_star",
)
SIM_REASONS = (
    "branch_incomplete",
    "non_finite",
    "map_build_failed",
    "missing_eval_entity_future",
)
REASONS = ELIGIBILITY_REASONS + SIM_REASONS

_CHECKPOINT_BASENAME = "checkpoint.npz"
_LEDGER_BASENAME = "rejection_ledger.jsonl"
_OUTPUT_BASENAME = "groups.npz"
_META_BASENAME = "groups_meta.json"
_RECEIPT_BASENAME = "collection_receipt.json"
_DEFAULT_MAX_ATTEMPTS_PER_PROCESS = 100
_SUPERVISOR_HEARTBEAT_SECONDS = 30.0

# npz array fields (R2 contract + D2 t_star). Per-group meta (scenario_id, eval track
# ids, ego lane) is stored in the JSON sidecar, not the npz.
_ARRAY_FIELDS = (
    "object_history",                # (G,10,5,8) float32
    "object_factual_future",         # (G,20,5,8) float32
    "object_counterfactual_futures", # (G,10,20,5,8) float32
    "factual_actions",               # (G,20) int64
    "inference_mask",                # (G,20,5) bool
    "t_star",                        # (G,) int64
)


def load_role_candidates(role: str) -> list[str]:
    """Ordered candidate IDs for a role: primary rows first, then reserve, each in
    frozen global-rank order (plan 4.3)."""
    rows = [r for r in split.load_split_pools(PLATFORM_DIR) if r["role"] == role]
    if not rows:
        raise ValueError(f"role {role} has no pool rows")
    rows.sort(key=lambda r: r["global_rank"])
    primary = [r["scenario_id"] for r in rows if r["pool"] == "primary"]
    reserve = [r["scenario_id"] for r in rows if r["pool"] == "reserve"]
    return primary + reserve


def scenario_index_and_file(db_dir: Path) -> dict[str, tuple[int, str]]:
    """scenario_id -> (seed index, pkl file name) from the frozen dataset summary."""
    with open(db_dir / "dataset_summary.pkl", "rb") as fh:
        summary = pickle.load(fh)
    out: dict[str, tuple[int, str]] = {}
    for index, file_name in enumerate(summary.keys()):
        sid = str(summary[file_name].get(
            "scenario_id", file_name.replace("sd_waymo_v1.2_", "").replace(".pkl", "")))
        out[sid] = (index, file_name)
    return out


def _load_centralized(pkl_path: Path):
    from metadrive.scenario.utils import read_scenario_data

    return read_scenario_data(str(pkl_path), centralize=True)


def _valid_simulation(group: Mapping[str, Any]) -> tuple[bool, str]:
    """Structural (pre-outcome) simulation validity for an accepted group."""
    if not group["branches_completed"]:
        return False, "branch_incomplete"
    for key in ("object_history", "object_factual_future", "object_counterfactual_futures"):
        if not np.isfinite(group[key]).all():
            return False, "non_finite"
    if not (group["object_factual_future"][:, 0, 0] > 0).all():
        return False, "missing_eval_entity_future"
    return True, "accepted"


def _group_meta(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": group["scenario_id"],
        "t_star": int(group["t_star"]),
        "eval_track_ids": list(group["eval_track_ids"]),
        "ego_lane_id": group["ego_lane_id"],
    }


def _stack_arrays(groups: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ("object_history", "object_factual_future", "object_counterfactual_futures"):
        out[key] = np.stack([g[key] for g in groups]).astype(np.float32)
    out["inference_mask"] = np.stack([g["inference_mask"] for g in groups]).astype(bool)
    out["t_star"] = np.asarray([int(g["t_star"]) for g in groups], dtype=np.int64)
    out["factual_actions"] = np.stack([g["factual_actions"] for g in groups]).astype(np.int64)
    # canonical frozen action table, identical for every group (R2 contract)
    out["cf_actions"] = np.asarray([a for a, _ in A.ACTION_PLANS], dtype=np.int64)
    out["cf_durations"] = np.asarray([d for _, d in A.ACTION_PLANS], dtype=np.int64)
    out["n_actions"] = np.asarray(len(A.ACTION_PLANS), dtype=np.int64)
    out["object_features_json"] = np.asarray(json.dumps(list(F.FEATURE_NAMES)))
    out["action_plans_json"] = np.asarray(json.dumps([list(p) for p in A.ACTION_PLANS]))
    return out


def _load_checkpoint_groups(
    checkpoint_path: Path,
    expected_config_fingerprint: str,
) -> list[dict[str, Any]]:
    """Restore accepted groups while inflating each compressed NPZ member once.

    Indexing ``archive[key]`` inside the per-group loop repeatedly decompresses the
    complete member and caused the D2 collector's process commit to grow without a
    useful bound.  Bulk-load each member once, then create cheap per-group views.
    """
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        actual = str(archive["__config_fingerprint__"].item())
        if actual != expected_config_fingerprint:
            raise ValueError("checkpoint contract/config fingerprint mismatch; stop")
        meta = json.loads(str(archive["__meta_json__"].item()))
        bulk = {key: archive[key] for key in _ARRAY_FIELDS}

    n_groups = len(meta)
    for key, value in bulk.items():
        if value.shape[0] != n_groups:
            raise ValueError(
                f"checkpoint field {key} has {value.shape[0]} groups, expected {n_groups}")

    groups: list[dict[str, Any]] = []
    for i, record in enumerate(meta):
        groups.append({
            "object_history": bulk["object_history"][i],
            "object_factual_future": bulk["object_factual_future"][i],
            "object_counterfactual_futures": bulk["object_counterfactual_futures"][i],
            "inference_mask": bulk["inference_mask"][i],
            "t_star": int(bulk["t_star"][i]),
            "factual_actions": bulk["factual_actions"][i],
            **_meta_as_group_fields(record),
        })
    return groups


class _Ledger:
    """Append-only acceptance/rejection ledger (fsync'd); the resume anchor."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict[str, Any]] = []
        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.rows.append(json.loads(line))

    @property
    def burned_ids(self) -> set[str]:
        return {r["scenario_id"] for r in self.rows}

    def append(self, row: Mapping[str, Any]) -> None:
        record = dict(row)
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.rows.append(record)


def collect_role(
    role: str,
    quota: int,
    out_dir: Path,
    *,
    contract: A.ActionContract = A.DEFAULT_CONTRACT,
    max_attempts_per_process: int | None = None,
) -> dict[str, Any]:
    """Collect ``quota`` complete groups for ``role``. Resumable; never re-attempts a
    burned ID. Returns the receipt dict (also written to disk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = _Ledger(out_dir / _LEDGER_BASENAME)
    checkpoint_path = out_dir / _CHECKPOINT_BASENAME
    output_path = out_dir / _OUTPUT_BASENAME

    candidates = load_role_candidates(role)
    index_file = scenario_index_and_file(DB_DIR)
    num_scenarios = len(index_file)

    config_fp = canonical_fingerprint({
        "role": role, "quota": int(quota), "role_salt": ROLE_SALT,
        "contract": dict(contract.__dict__), "db": str(DB_DIR),
    })

    # --- resume: rehydrate accepted groups (arrays + meta) from the checkpoint -------
    accepted_groups: list[dict[str, Any]] = []
    if checkpoint_path.is_file():
        accepted_groups = _load_checkpoint_groups(checkpoint_path, config_fp)
    burned = ledger.burned_ids
    # Hardening (2026-09-23): also exclude scenario_ids already accepted into the
    # rehydrated checkpoint. The ledger is the primary burn-once record, but a
    # kill/respawn can leave the checkpoint ahead of the ledger; never re-collect an
    # already-accepted scenario regardless of which record is ahead. This closes the
    # duplicate-scenario hole without changing any scientific contract.
    accepted_ids = {g["scenario_id"] for g in accepted_groups}
    todo = [sid for sid in candidates if sid not in burned and sid not in accepted_ids]
    if len(accepted_groups) + len(todo) < quota:
        raise RuntimeError(
            f"role {role}: remaining pool ({len(accepted_groups)} accepted + "
            f"{len(todo)} untried) cannot reach quota {quota}; registered stop "
            "(pool exhausted -- one-shot resource risk)")

    env = A.build_env(str(DB_DIR), num_scenarios=num_scenarios)
    progress = ProgressReporter(total=quota, every=25, prefix=f"collect:{role}",
                                heartbeat_seconds=30.0)
    started = time.perf_counter()
    skip_counts: Counter = Counter({r: 0 for r in REASONS})
    for row in ledger.rows:
        if row.get("outcome") != "accepted":
            skip_counts[row.get("reason", "?")] += 1

    def _save_checkpoint():
        payload = _stack_arrays(accepted_groups) if accepted_groups else {}
        payload["__config_fingerprint__"] = np.asarray(config_fp)
        payload["__n_accepted__"] = np.asarray(len(accepted_groups), dtype=np.int64)
        payload["__meta_json__"] = np.asarray(json.dumps(
            [_group_meta(g) for g in accepted_groups]))
        atomic_save_npz(checkpoint_path, **payload)

    progress.update(len(accepted_groups),
                    details="resumed" if accepted_groups else "", force=True)

    attempts_this_process = 0
    paused_for_memory_boundary = False
    try:
        for sid in todo:
            if len(accepted_groups) >= quota:
                break
            if (max_attempts_per_process is not None
                    and attempts_this_process >= int(max_attempts_per_process)):
                paused_for_memory_boundary = True
                break
            attempts_this_process += 1
            if sid not in index_file:
                ledger.append({"scenario_id": sid, "outcome": "rejected",
                               "reason": "no_eligible_t_star", "stage": "index_lookup"})
                skip_counts["no_eligible_t_star"] += 1
                continue
            index, file_name = index_file[sid]
            try:
                scenario = _load_centralized(DB_DIR / file_name)
            except Exception:  # noqa: BLE001 - unreadable pkl is structural
                ledger.append({"scenario_id": sid, "outcome": "rejected",
                               "reason": "map_build_failed", "stage": "load"})
                skip_counts["map_build_failed"] += 1
                continue
            t_star, reason, info, attempts = F.select_t_star(scenario, sid, ROLE_SALT)
            if t_star is None:
                ledger.append({"scenario_id": sid, "outcome": "rejected",
                               "reason": reason, "stage": "eligibility",
                               "attempts": attempts})
                skip_counts[reason] += 1
                continue
            try:
                ctx = A.prepare_group_context(scenario, sid, ROLE_SALT, contract)
                group = A.run_group(env, scenario, index, ctx)
            except Exception as exc:  # noqa: BLE001 - structural sim failure
                env.engine.map_manager.clear_stored_maps()
                ledger.append({"scenario_id": sid, "outcome": "rejected",
                               "reason": "map_build_failed", "stage": "simulation",
                               "error": f"{type(exc).__name__}: {exc}"})
                skip_counts["map_build_failed"] += 1
                continue
            # each group uses a distinct map; clear after its 11 branches (memory bound)
            env.engine.map_manager.clear_stored_maps()
            ok, verdict = _valid_simulation(group)
            if not ok:
                ledger.append({"scenario_id": sid, "outcome": "rejected",
                               "reason": verdict, "stage": "simulation"})
                skip_counts[verdict] += 1
                continue
            group = dict(group)
            group["scenario_id"] = sid
            group["eval_track_ids"] = list(ctx.eval_track_ids)
            group["ego_lane_id"] = ctx.ego_lane_id
            accepted_groups.append(group)
            ledger.append({"scenario_id": sid, "outcome": "accepted", "t_star": t_star})
            _save_checkpoint()
            progress.update(
                len(accepted_groups), attempted=len(ledger.rows),
                details=(f"burned={len(ledger.rows)} "
                         f"worker_attempts={attempts_this_process}"))
    finally:
        env.close()

    if paused_for_memory_boundary:
        receipt = {
            "role": role,
            "quota": int(quota),
            "accepted": len(accepted_groups),
            "burned_total": len(ledger.rows),
            "attempts_this_process": attempts_this_process,
            "status": "COLLECTION_PAUSED",
            "reason": "PROCESS_MEMORY_BOUNDARY",
        }
        print(
            f"[collect:{role}] PAUSED | accepted={len(accepted_groups)}/{quota} | "
            f"burned={len(ledger.rows)} | worker_attempts={attempts_this_process} | "
            "checkpoint=durable | next=automatic fresh-process resume",
            flush=True,
        )
        return receipt

    if len(accepted_groups) < quota:
        summary = (f"[collect:{role}] FAILURE | accepted={len(accepted_groups)}/{quota} "
                   f"pool exhausted | elapsed={format_duration(time.perf_counter()-started)}")
        print(summary, flush=True)
        raise RuntimeError(summary)

    # Fail loud rather than write a non-unique role artifact (the Task 3 loader guards
    # on this; catch it at the source). See the 2026-09-23 duplicate-scenario repair.
    final_ids = [g["scenario_id"] for g in accepted_groups]
    if len(set(final_ids)) != len(final_ids):
        raise RuntimeError(
            f"role {role}: accepted_groups has duplicate scenario_ids; refusing to "
            "finalize a non-unique role artifact")

    atomic_save_npz(output_path, **_stack_arrays(accepted_groups))
    with open(out_dir / _META_BASENAME, "w", encoding="utf-8") as fh:
        json.dump([_group_meta(g) for g in accepted_groups], fh, indent=2)
    receipt = {
        "role": role,
        "quota": int(quota),
        "accepted": len(accepted_groups),
        "burned_total": len(ledger.rows),
        "skip_counts": dict(skip_counts),
        "config_fingerprint": config_fp,
        "output_file": _OUTPUT_BASENAME,
        "meta_file": _META_BASENAME,
        "contract": dict(contract.__dict__),
        "status": "COLLECTION_COMPLETE",
        "elapsed_seconds": time.perf_counter() - started,
    }
    with open(out_dir / _RECEIPT_BASENAME, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2)
    print(f"[collect:{role}] SUCCESS | accepted={len(accepted_groups)}/{quota} | "
          f"burned={len(ledger.rows)} | skips={dict(skip_counts)} | "
          f"elapsed={format_duration(time.perf_counter()-started)}", flush=True)
    return receipt


def _meta_as_group_fields(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": meta["scenario_id"],
        "eval_track_ids": list(meta["eval_track_ids"]),
        "ego_lane_id": meta["ego_lane_id"],
    }


def _checkpoint_accepted(checkpoint_path: Path) -> int:
    if not checkpoint_path.is_file():
        return 0
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        return int(archive["__n_accepted__"])


def _ledger_line_count(ledger_path: Path) -> int:
    if not ledger_path.is_file():
        return 0
    with open(ledger_path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _progress_signature(out_dir: Path) -> tuple[int, int]:
    return (
        _checkpoint_accepted(out_dir / _CHECKPOINT_BASENAME),
        _ledger_line_count(out_dir / _LEDGER_BASENAME),
    )


def _supervisor_progress_line(
    *,
    role: str,
    accepted: int,
    quota: int,
    burned: int,
    child_pid: int,
    elapsed_seconds: float,
    start_accepted: int,
) -> str:
    completed_this_run = max(0, int(accepted) - int(start_accepted))
    rate = completed_this_run / elapsed_seconds if elapsed_seconds > 0 else 0.0
    eta = (quota - accepted) / rate if rate > 0 else None
    return (
        f"[collect-supervisor:{role}] {100.0 * accepted / quota:5.1f}% | "
        f"{accepted}/{quota} | burned={burned} | child_pid={child_pid} | "
        f"elapsed={format_duration(elapsed_seconds)} | ETA={format_duration(eta)}"
    )


def _worker_command(
    role: str,
    quota: int,
    out_dir: Path,
    max_attempts_per_process: int,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "code" / "stage4d_sim_entry.py"),
        str(Path(__file__).resolve()),
        "--role", role,
        "--quota", str(int(quota)),
        "--out-dir", str(out_dir),
        "--max-attempts-per-process", str(int(max_attempts_per_process)),
        "--worker-segment",
    ]


def supervise_collection(
    role: str,
    quota: int,
    out_dir: Path,
    *,
    max_attempts_per_process: int = _DEFAULT_MAX_ATTEMPTS_PER_PROCESS,
) -> dict[str, Any]:
    """Run bounded collector workers until a role is durably complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / _RECEIPT_BASENAME
    temporary_path = out_dir / (_CHECKPOINT_BASENAME + ".tmp")
    start_accepted, _ = _progress_signature(out_dir)
    started = time.perf_counter()

    while True:
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (receipt.get("status") != "COLLECTION_COMPLETE"
                    or int(receipt.get("accepted", -1)) != int(quota)):
                raise ValueError("existing collection receipt is not a complete matching role")
            print(
                f"[collect-supervisor:{role}] SUCCESS | accepted={quota}/{quota} | "
                "existing complete receipt verified",
                flush=True,
            )
            return receipt

        if temporary_path.exists():
            raise RuntimeError(f"unfinished atomic checkpoint detected: {temporary_path}")

        before = _progress_signature(out_dir)
        command = _worker_command(role, quota, out_dir, max_attempts_per_process)
        child = subprocess.Popen(command, cwd=PROJECT_ROOT)
        print(
            f"[collect-supervisor:{role}] worker started | child_pid={child.pid} | "
            f"accepted={before[0]}/{quota} | max_attempts={max_attempts_per_process}",
            flush=True,
        )
        last_heartbeat = 0.0
        try:
            while child.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed - last_heartbeat >= _SUPERVISOR_HEARTBEAT_SECONDS:
                    accepted, burned = _progress_signature(out_dir)
                    print(_supervisor_progress_line(
                        role=role, accepted=accepted, quota=quota, burned=burned,
                        child_pid=child.pid, elapsed_seconds=elapsed,
                        start_accepted=start_accepted), flush=True)
                    last_heartbeat = elapsed
                time.sleep(1.0)
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise

        if child.returncode != 0:
            raise RuntimeError(
                f"collector worker {child.pid} failed with exit code {child.returncode}")
        if temporary_path.exists():
            raise RuntimeError(f"worker left an unfinished checkpoint: {temporary_path}")

        after = _progress_signature(out_dir)
        elapsed = time.perf_counter() - started
        print(_supervisor_progress_line(
            role=role, accepted=after[0], quota=quota, burned=after[1],
            child_pid=child.pid, elapsed_seconds=elapsed,
            start_accepted=start_accepted), flush=True)

        if after[0] < before[0] or after[1] < before[1]:
            raise RuntimeError("collector durable progress regressed")
        if after == before and not receipt_path.is_file():
            raise RuntimeError("collector worker exited without durable progress")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=list(split.ROLE_ORDER))
    parser.add_argument("--quota", type=int, required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--max-attempts-per-process", type=int,
        default=_DEFAULT_MAX_ATTEMPTS_PER_PROCESS)
    parser.add_argument("--worker-segment", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else (RUN_DIR / "roles" / args.role)
    if args.worker_segment:
        receipt = collect_role(
            args.role, args.quota, out_dir,
            max_attempts_per_process=args.max_attempts_per_process)
        return 0 if receipt["status"] in {"COLLECTION_COMPLETE", "COLLECTION_PAUSED"} else 1
    supervise_collection(
        args.role, args.quota, out_dir,
        max_attempts_per_process=args.max_attempts_per_process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
