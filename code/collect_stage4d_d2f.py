"""Stage 4D-D2F dedicated collector entry — mechanical reuse of ``collect_stage4d_external.py``.

Plan ``AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1`` section 3.2:
the old collector's candidate pool and salt are bound to old D2, so it may neither be re-run for
D2F nor modified. This entry mechanically reuses its action/scenario/pairing/QA algorithm
BITWISE and injects only the namespace, in memory:

- ID manifest: the D2F salt-ranked fresh screen candidates (``screen_manifest.json`` for the
  formal screen; the derived fresh-order prefix for the engineering smoke);
- salt: ``AC-TWIN-D2F-V0.1|``;
- output dirs: ``experiments/stage4_reliability/stage4d_d2f/`` (``smoke/`` for the engineering
  smoke, ``roles/development_screen/`` for the formal screen);
- plan ID: injected into the collection config fingerprint.

Modes:
- ``smoke``   — supervised small collection over the fresh-order prefix (<=160 candidates,
  marked ENGINEERING_SMOKE_ONLY; the IDs are appended to the burned set and NEVER enter the
  screen) + the section-3.3 structural verification (10 branches, same scenario/context,
  prefix/identity, visible entities, action mapping, 5Hz/20-step, real reactive support,
  process recovery via bounded worker segments);
- ``screen``  — the formal supervised ``selected_n`` collection from the frozen screen manifest;
- ``confirm`` — the formal supervised confirm collection from the frozen confirm manifest
  (legal only after the confirm protocol lock; writes ``roles/confirm/``);
- ``parity``  — same-input per-group parity: rebuild a few OLD train groups with the OLD salt
  through the D2F-injected code path and compare against the stored old arrays (float arrays
  within the pilot-proven 1e-4 cross-process float-noise bound; int/bool/meta exact), proving
  the reuse is mechanical;
- ``collect-worker`` — internal worker segment spawned by the supervisor (fresh process at the
  memory boundary; resume from the atomic checkpoint + append-only ledger).

Terminal-safe progress (plan section 5): the reused collector's ProgressReporter and the
supervisor heartbeat emit phase / completed / total / percentage / elapsed / ETA at least every
30 seconds, plus a final success/failure summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "code"))

import collect_stage4d_external as base  # noqa: E402
from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3r_protocol import write_once_json  # noqa: E402

PLAN_ID = PROT.PLAN_ID
ROLE_SALT = PROT.SCREEN_SALT
OLD_SALT = str(base.ROLE_SALT)
D2F_DIR = PROT.D2F_DIR
D2_DIR = PROT.D2_DIR
SCREEN_ROLE = PROT.SCREEN_ROLE
CONFIRM_ROLE = PROT.CONFIRM_ROLE

#: pilot-proven cross-process float-noise bound for an equivalent re-simulation (D2 handoff
#: 2026-09-23 duplicate-scenario analysis: same-scenario re-run futures differ by ~5e-5 << 1e-4).
PARITY_TOLERANCE = 1e-4

_SMOKE_DIR = D2F_DIR / PROT.SMOKE_DIR_NAME
_RECEIPT_NAME = "collection_receipt.json"


# --- in-memory namespace injection (the original file is never modified) -----------------------------

_candidate_source: str = "manifest"
_smoke_limit: int = 0


def _d2f_worker_command(role: str, quota: int, out_dir: Path,
                        max_attempts_per_process: int) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "stage4d_sim_entry.py"),
        str(Path(__file__).resolve()),
        "--mode", "collect-worker",
        "--role", role,
        "--quota", str(int(quota)),
        "--out-dir", str(out_dir),
        "--max-attempts-per-process", str(int(max_attempts_per_process)),
        "--candidate-source", _candidate_source,
    ]
    if _candidate_source == "smoke":
        command += ["--smoke-limit", str(int(_smoke_limit))]
    return command


def _install_namespace(candidate_ids: list[str]) -> None:
    """Inject salt / candidate manifest / plan ID into the reused collector, in memory only."""
    base.ROLE_SALT = ROLE_SALT
    ids = [str(s) for s in candidate_ids]
    base.load_role_candidates = lambda role: list(ids)
    real_fingerprint = base.canonical_fingerprint

    def _fingerprint(payload: dict[str, Any]) -> str:
        payload = dict(payload)
        payload["plan_id"] = PLAN_ID
        return real_fingerprint(payload)

    base.canonical_fingerprint = _fingerprint
    base._worker_command = _d2f_worker_command


# --- candidate resolution ----------------------------------------------------------------------------


def _derive_smoke_candidates(limit: int) -> list[str]:
    """The engineering-smoke candidate prefix: the FIRST ``limit`` fresh-order IDs (<=160).

    Deterministic across supervisor/worker processes: the derivation scans only the OLD roles
    (the D2F smoke's own ledger never feeds it), so a respawned worker sees the same order.
    """
    PROT.assert_smoke_budget(limit)
    rows = PROT.load_pool_rows()
    burned = PROT.scan_burned_ids()["burned"]
    order = PROT.derive_screen_candidate_ids(rows, burned)
    return order[: int(limit)]


def _manifest_candidates() -> list[str]:
    manifest_path = D2F_DIR / PROT.SCREEN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(
            f"[d2f-collect] screen manifest missing: {manifest_path} — freeze Phase S first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(s) for s in manifest["primary_ids"]] + \
        [str(s) for s in manifest["reserve_ids"]]


def _confirm_candidates() -> list[str]:
    manifest_path = D2F_DIR / PROT.CONFIRM_MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(
            f"[d2f-collect] confirm manifest missing: {manifest_path} — freeze Phase C first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(s) for s in manifest["primary_ids"]] + \
        [str(s) for s in manifest["reserve_ids"]]


def _resolve_candidates(source: str, smoke_limit: int) -> list[str]:
    if source == "smoke":
        return _derive_smoke_candidates(smoke_limit)
    if source == "manifest":
        return _manifest_candidates()
    if source == "confirm-manifest":
        return _confirm_candidates()
    raise ValueError(f"unknown candidate source {source!r}")


# --- section 3.3 smoke structural verification ---------------------------------------------------------


def verify_smoke_groups(smoke_dir: Path) -> dict[str, Any]:
    """The section-3.3 smoke checks over the collected smoke groups (ENGINEERING_SMOKE_ONLY).

    Verifies: ten counterfactual branches + factual per group; group/context uniqueness;
    finite history/futures (prefix/identity integrity); visible eval entities; the ten action
    branches genuinely differ (action mapping); the 5 Hz / 20-step future geometry; and a real
    reactive response in at least one accepted group. The smoke truth is legal here — the smoke
    is an engineering check, its IDs are burned and never adjudicated.
    """
    from ac_twin import stage4d_external_statistics as D2STAT

    groups_path = Path(smoke_dir) / "groups.npz"
    if not groups_path.is_file():
        raise RuntimeError(f"smoke groups missing: {groups_path}")
    with np.load(groups_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    meta = json.loads((Path(smoke_dir) / "groups_meta.json").read_text(encoding="utf-8"))

    history = np.asarray(data["object_history"])
    factual = np.asarray(data["object_factual_future"])
    cf = np.asarray(data["object_counterfactual_futures"])
    mask = np.asarray(data["inference_mask"], dtype=bool)
    t_star = np.asarray(data["t_star"])
    groups = int(history.shape[0])
    scenario_ids = [str(m["scenario_id"]) for m in meta]

    checks = {
        "ten_branches_plus_factual": bool(
            cf.shape == (groups, 10, 20, 5, 8) and factual.shape == (groups, 20, 5, 8)),
        "same_scenario_context_unique": bool(len(set(scenario_ids)) == groups),
        "prefix_identity_finite": bool(
            np.isfinite(history).all() and np.isfinite(factual).all()
            and np.isfinite(cf).all()),
        "visible_entities": bool(mask.reshape(groups, -1).any(axis=1).all()),
        "action_mapping_distinct": bool(
            (np.abs(cf.astype(np.float64)
                    - factual[:, None].astype(np.float64)).max(axis=(2, 3, 4)) > 0.0
             ).any(axis=0).all()),
        "geometry_5hz_20step": bool(factual.shape[1] == 20 and cf.shape[2] == 20),
    }
    reactive = D2STAT.compute_reactive_response(factual, cf)
    checks["real_reactive_support"] = bool(reactive.any())
    checks["t_star_valid"] = bool((t_star > 0).all())
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "checks": checks,
        "passed": not failed,
        "failed": failed,
        "n_groups": groups,
        "reactive_groups": int(reactive.sum()),
        "groups_sha256": sha256_file(groups_path),
    }


# --- same-input per-group parity (plan 3.2) -------------------------------------------------------------


def run_collector_parity(n_groups: int, *, train_role: str = "train") -> dict[str, Any]:
    """Rebuild old train groups through the D2F-injected path with the OLD salt; compare.

    Proves the D2F collector is a mechanical reuse: with the old salt the injected code path
    must reproduce the stored old groups (float arrays within the pilot-proven 1e-4 noise
    bound; int/bool/meta exactly). Read-only against the old role; writes only the parity
    receipt inside the D2F smoke namespace.
    """
    receipt_path = _SMOKE_DIR / "collector_parity_receipt.json"
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    role_dir = D2_DIR / "roles" / train_role
    with np.load(role_dir / "groups.npz", allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    meta = json.loads((role_dir / "groups_meta.json").read_text(encoding="utf-8"))
    scenario_ids = [str(m["scenario_id"]) for m in meta][: int(n_groups)]

    index_file = base.scenario_index_and_file(base.DB_DIR)
    env = base.A.build_env(str(base.DB_DIR), num_scenarios=len(index_file))
    per_group: list[dict[str, Any]] = []
    try:
        for sid in scenario_ids:
            row = scenario_ids.index(sid)
            index, file_name = index_file[sid]
            scenario = base._load_centralized(base.DB_DIR / file_name)
            t_star, reason, _info, _att = base.F.select_t_star(scenario, sid, OLD_SALT)
            if t_star is None:
                raise RuntimeError(f"parity scenario {sid} lost eligibility: {reason}")
            if int(t_star) != int(stored["t_star"][row]):
                raise RuntimeError(
                    f"parity t_star mismatch for {sid}: {t_star} != stored "
                    f"{int(stored['t_star'][row])}")
            ctx = base.A.prepare_group_context(scenario, sid, OLD_SALT, base.A.DEFAULT_CONTRACT)
            group = base.A.run_group(env, scenario, index, ctx)
            env.engine.map_manager.clear_stored_maps()
            diffs: dict[str, float] = {}
            for key in ("object_history", "object_factual_future",
                        "object_counterfactual_futures"):
                diff = float(np.max(np.abs(
                    np.asarray(group[key], dtype=np.float64)
                    - np.asarray(stored[key][row], dtype=np.float64))))
                diffs[key] = diff
            exact = bool(
                np.array_equal(np.asarray(group["factual_actions"]),
                               np.asarray(stored["factual_actions"][row]))
                and np.array_equal(np.asarray(group["inference_mask"]),
                                   np.asarray(stored["inference_mask"][row]))
                and list(ctx.eval_track_ids) == list(meta[row]["eval_track_ids"])
                and str(ctx.ego_lane_id) == str(meta[row]["ego_lane_id"]))
            per_group.append({
                "scenario_id": sid,
                "max_abs_diff": diffs,
                "within_tolerance": all(d <= PARITY_TOLERANCE for d in diffs.values()),
                "int_bool_meta_exact": exact,
            })
            print(f"[d2f-parity] {sid}: max|diff|={max(diffs.values()):.3e} "
                  f"exact={exact}", flush=True)
    finally:
        env.close()

    passed = bool(per_group) and all(
        g["within_tolerance"] and g["int_bool_meta_exact"] for g in per_group)
    receipt = {
        "plan_id": PLAN_ID,
        "scope": "ENGINEERING_SMOKE_ONLY",
        "mode": "collector_parity",
        "salt_used": OLD_SALT,
        "tolerance": PARITY_TOLERANCE,
        "tolerance_source": ("pilot-proven cross-process float-noise bound (D2 handoff "
                             "2026-09-23 same-scenario re-simulation analysis)"),
        "train_role": train_role,
        "groups": per_group,
        "passed": passed,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not passed:
        raise RuntimeError(f"collector parity FAILED: {per_group}")
    _SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    write_once_json(receipt_path, receipt)
    return receipt


# --- supervised collection modes -------------------------------------------------------------------------


def run_smoke(accepted_quota: int, candidate_limit: int, *,
              max_attempts_per_process: int) -> dict[str, Any]:
    """Engineering smoke over the fresh-order prefix (plan 2.2 budget + 3.3 checks).

    Small bounded worker segments exercise checkpoint/ledger process recovery. The smoke
    receipt is marked ENGINEERING_SMOKE_ONLY; the smoke IDs are appended to the burned set by
    the post-smoke audit and never enter the screen.
    """
    global _candidate_source, _smoke_limit
    PROT.assert_smoke_budget(candidate_limit)
    _candidate_source, _smoke_limit = "smoke", int(candidate_limit)
    candidates = _derive_smoke_candidates(candidate_limit)
    _install_namespace(candidates)
    print(f"[d2f-smoke] start: quota={accepted_quota} accepted over <= {len(candidates)} "
          f"fresh-prefix candidates (budget {PROT.SMOKE_CANDIDATE_BUDGET}) | "
          f"max_attempts/worker={max_attempts_per_process}", flush=True)
    receipt = base.supervise_collection(
        "smoke", accepted_quota, _SMOKE_DIR,
        max_attempts_per_process=max_attempts_per_process)
    verification = verify_smoke_groups(_SMOKE_DIR)
    if not verification["passed"]:
        raise RuntimeError(f"smoke structural verification FAILED: {verification['failed']}")
    summary = {
        "plan_id": PLAN_ID,
        "scope": "ENGINEERING_SMOKE_ONLY",
        "collection_receipt": receipt,
        "verification": verification,
        "smoke_burned_ids": int(len(PROT.load_smoke_burned_ids(_SMOKE_DIR))),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(_SMOKE_DIR / "smoke_receipt.json", summary)
    print(f"[d2f-smoke] SUCCESS: {receipt['accepted']}/{accepted_quota} accepted, "
          f"burned={summary['smoke_burned_ids']} candidate IDs, checks all pass", flush=True)
    return summary


def run_screen_collection(*, max_attempts_per_process: int) -> dict[str, Any]:
    """The formal D2F screen collection from the frozen manifest (one-shot resource)."""
    global _candidate_source
    _candidate_source = "manifest"
    manifest = json.loads((D2F_DIR / PROT.SCREEN_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not (D2F_DIR / PROT.SCREEN_LOCK_NAME).is_file():
        raise SystemExit("[d2f-collect] screen_protocol_lock.json missing; freeze Phase S first")
    quota = int(manifest["selected_n"])
    candidates = _manifest_candidates()
    _install_namespace(candidates)
    out_dir = D2F_DIR / "roles" / SCREEN_ROLE
    print(f"[d2f-collect] start: selected_n={quota} over {len(candidates)} manifest candidates "
          f"| salt={ROLE_SALT!r}", flush=True)
    receipt = base.supervise_collection(
        SCREEN_ROLE, quota, out_dir, max_attempts_per_process=max_attempts_per_process)
    print(f"[d2f-collect] SUCCESS: {receipt['accepted']}/{quota} accepted | "
          f"burned={receipt['burned_total']}", flush=True)
    return receipt


def run_confirm_collection(*, max_attempts_per_process: int) -> dict[str, Any]:
    """The formal D2F confirm collection from the frozen confirm manifest (one-shot resource).

    Legal only after the confirm protocol lock exists (plan 4: the confirm chain runs the SAME
    frozen collector/eligibility/QA code as the screen, from the independent preserved pool).
    """
    global _candidate_source
    _candidate_source = "confirm-manifest"
    manifest = json.loads((D2F_DIR / PROT.CONFIRM_MANIFEST_NAME).read_text(encoding="utf-8"))
    if not (D2F_DIR / PROT.CONFIRM_LOCK_NAME).is_file():
        raise SystemExit(
            "[d2f-collect] confirm_protocol_lock.json missing; freeze Phase C first")
    quota = int(manifest["selected_n"])
    candidates = _confirm_candidates()
    _install_namespace(candidates)
    out_dir = D2F_DIR / "roles" / CONFIRM_ROLE
    print(f"[d2f-collect] start CONFIRM: selected_n={quota} over {len(candidates)} manifest "
          f"candidates | salt={ROLE_SALT!r}", flush=True)
    receipt = base.supervise_collection(
        CONFIRM_ROLE, quota, out_dir, max_attempts_per_process=max_attempts_per_process)
    print(f"[d2f-collect] CONFIRM SUCCESS: {receipt['accepted']}/{quota} accepted | "
          f"burned={receipt['burned_total']}", flush=True)
    return receipt


# --- CLI ----------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("smoke", "screen", "confirm", "parity", "collect-worker"))
    parser.add_argument("--role", default=SCREEN_ROLE)
    parser.add_argument("--quota", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--accepted-quota", type=int, default=12)
    parser.add_argument("--candidate-limit", type=int, default=160)
    parser.add_argument("--parity-groups", type=int, default=2)
    parser.add_argument("--max-attempts-per-process", type=int,
                        default=base._DEFAULT_MAX_ATTEMPTS_PER_PROCESS)
    parser.add_argument("--candidate-source",
                        choices=("manifest", "smoke", "confirm-manifest"), default="manifest",
                        help=argparse.SUPPRESS)
    parser.add_argument("--smoke-limit", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.mode == "parity":
        run_collector_parity(int(args.parity_groups))
        return 0
    if args.mode == "smoke":
        run_smoke(int(args.accepted_quota), int(args.candidate_limit),
                  max_attempts_per_process=int(args.max_attempts_per_process))
        return 0
    if args.mode == "screen":
        run_screen_collection(
            max_attempts_per_process=int(args.max_attempts_per_process))
        return 0
    if args.mode == "confirm":
        run_confirm_collection(
            max_attempts_per_process=int(args.max_attempts_per_process))
        return 0

    # collect-worker: one bounded in-process segment, resumed by the supervisor.
    global _candidate_source, _smoke_limit
    _candidate_source = str(args.candidate_source)
    _smoke_limit = int(args.smoke_limit)
    candidates = _resolve_candidates(_candidate_source, _smoke_limit)
    _install_namespace(candidates)
    out_dir = Path(args.out_dir)
    receipt = base.collect_role(
        args.role, int(args.quota), out_dir,
        max_attempts_per_process=int(args.max_attempts_per_process))
    return 0 if receipt["status"] in {"COLLECTION_COMPLETE", "COLLECTION_PAUSED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
