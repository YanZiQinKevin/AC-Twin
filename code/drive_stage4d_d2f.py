"""Stage 4D-D2F driver (plan AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1).

Phase P entry (NO collection / lock / manifest / training):

    python code/drive_stage4d_d2f.py audit-power

runs, in the frozen plan-7 order: source binding (the nine plan-listed SHA256 recomputed live
+ the expanded checkpoint/scaler/contract/commit/code binding) -> the read-only dual-pool
capacity audit (admissible screen/confirm n sets frozen BEFORE power) -> the corrected power
(effect/support separation, fresh entropy 2026122406, admissible n only) with the mechanical
Go/No-Go. Terminal states end the run without creating any screen artifact:
``D2F_FRESH_POOL_CAPACITY_INSUFFICIENT`` (audit) or ``D2F_SCREEN_UNDERPOWERED`` (power).
Progress is plain text with phase / completed / total / elapsed / ETA at least every 30 s and
a final success/failure summary. Later phases (screen / confirm) are appended as subcommands
by their own tasks; they are never entered from ``audit-power``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2f_power as DPOW  # noqa: E402
from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402


def _make_progress(phase: str) -> "callable":
    state = {"last": 0.0, "t0": time.perf_counter()}

    def progress(line: str) -> None:
        now = time.perf_counter()
        if now - state["last"] >= 5.0:
            state["last"] = now
            elapsed = now - state["t0"]
            print(f"[d2f {phase}] elapsed={elapsed:7.1f}s | {line}", flush=True)

    return progress


def cmd_audit_power() -> int:
    """Phase P: source binding -> dual-pool capacity audit -> corrected power + Go/No-Go."""
    t0 = time.perf_counter()
    print("[d2f audit-power] Phase P start: source binding -> capacity audit -> corrected "
          "power (NO collection / lock / manifest)", flush=True)

    try:
        binding = PROT.run_source_binding(progress=_make_progress("source-binding"))
    except PROT.SourceDriftError as exc:
        print(f"[d2f audit-power] TERMINAL {PROT.STATUS_SOURCE_INVALID}: {exc}", flush=True)
        return 4
    print(f"[d2f audit-power] source binding status={binding['status']}", flush=True)

    audit = PROT.run_pool_capacity_audit(progress=_make_progress("pool-audit"))
    print(f"[d2f audit-power] pool capacity audit status={audit['status']} "
          f"screen_admissible_ns={audit['capacity']['screen_admissible_ns']} "
          f"confirm_admissible_ns={audit['capacity']['confirm_admissible_ns']}", flush=True)
    if audit.get("terminal"):
        print(f"[d2f audit-power] TERMINAL {audit['status']}: no screen manifest / lock / "
              f"collection is created; the D2F authorization ends here "
              f"(elapsed={time.perf_counter()-t0:.1f}s)", flush=True)
        return 5
    if audit["capacity"].get("confirm_below_3000_risk"):
        print(f"[d2f audit-power] RISK: confirm max admissible n="
              f"{audit['capacity']['confirm_max_admissible']} < 3000; continuing only inside "
              f"the pre-frozen admissible set (plan 2.2)", flush=True)

    report = DPOW.run_power_correction(progress=_make_progress("corrected-power"))
    elapsed = time.perf_counter() - t0
    if report.get("underpowered"):
        print(f"[d2f audit-power] TERMINAL {report['status']}: corrected 75% joint power "
              f"<0.80 at every admissible n; no new ID manifest / lock / collection; "
              f"elapsed={elapsed:.1f}s", flush=True)
        return 5
    print(f"[d2f audit-power] SUCCESS {report['status']}: selected_n={report['selected_n']} "
          f"(frozen minimal admissible n at 75% joint power >=0.80); Phase S may proceed; "
          f"elapsed={elapsed:.1f}s", flush=True)
    return 0


# --- Phase S (plan section 3) ---------------------------------------------------------------------------


def cmd_verify_fresh_ids() -> int:
    """Phase S step 1: fresh-ID re-verification (read-only, create-once)."""
    t0 = time.perf_counter()
    payload = PROT.run_fresh_id_verification(progress=_make_progress("verify-fresh-ids"))
    print(f"[d2f verify-fresh-ids] SUCCESS: {payload['screen_candidate_count']} zero-overlap "
          f"fresh candidates (burned={payload['burned_unique_ids']}, confirm preserved="
          f"{payload['confirm_pool_preserved']}); elapsed={time.perf_counter()-t0:.1f}s",
          flush=True)
    return 0


def cmd_collector_parity(groups: int) -> int:
    """Phase S step 2: same-input per-group parity of the D2F collector vs old D2 (sim env)."""
    t0 = time.perf_counter()
    import collect_stage4d_d2f as d2fcollect
    receipt = d2fcollect.run_collector_parity(int(groups))
    print(f"[d2f collector-parity] SUCCESS: {len(receipt['groups'])} groups reproduced within "
          f"{receipt['tolerance']} (old salt, mechanical reuse proven); "
          f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return 0


def cmd_smoke(accepted_quota: int, candidate_limit: int, max_attempts: int) -> int:
    """Phase S step 3: engineering smoke over the fresh-order prefix (sim env)."""
    t0 = time.perf_counter()
    import collect_stage4d_d2f as d2fcollect
    summary = d2fcollect.run_smoke(int(accepted_quota), int(candidate_limit),
                                   max_attempts_per_process=int(max_attempts))
    print(f"[d2f smoke] SUCCESS: {summary['smoke_burned_ids']} candidate IDs burned "
          f"(budget {PROT.SMOKE_CANDIDATE_BUDGET}); all section-3.3 checks pass; "
          f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return 0


def cmd_post_smoke_audit() -> int:
    """Phase S step 3b: read-only post-smoke dual-pool re-audit (create-once)."""
    t0 = time.perf_counter()
    payload = PROT.run_post_smoke_pool_audit(progress=_make_progress("post-smoke-audit"))
    if payload.get("terminal"):
        print(f"[d2f post-smoke-audit] TERMINAL {payload['status']}: 5x selected_n screen "
              f"capacity or the pre-frozen confirm set no longer holds after smoke; no screen "
              f"manifest / lock / collection; elapsed={time.perf_counter()-t0:.1f}s", flush=True)
        return 5
    print(f"[d2f post-smoke-audit] SUCCESS: screen_after={payload['screen_available_post_smoke']}"
          f" >= required={payload['screen_capacity_required']}, confirm preserved="
          f"{payload['confirm_pool_preserved']}; elapsed={time.perf_counter()-t0:.1f}s",
          flush=True)
    return 0


def cmd_scorer_parity(device: str | None, batch_size: int) -> int:
    """Phase S step 4: old-validation scorer parity vs the D2E frozen quantities (sim env)."""
    t0 = time.perf_counter()
    from ac_twin import stage4d_d2f_inference as D2FINF
    dev = _resolve_device(device)
    print(f"[d2f scorer-parity] start: device={dev} batch_size={batch_size} "
          f"tolerance={D2FINF.PARITY_TOLERANCE}", flush=True)
    stack = D2FINF.load_predictor_stack()
    selected, ego_scaler, ego_models = D2FINF.load_selected_ego()
    inventory = D2FINF.load_risk_inventory(selected)
    receipt = D2FINF.run_old_validation_parity(
        stack, ego_scaler, ego_models, inventory,
        device=dev, batch_size=int(batch_size), progress=_make_progress("scorer-parity"))
    print(f"[d2f scorer-parity] SUCCESS: {receipt['n_groups']} groups within "
          f"{receipt['tolerance']} + argmin identical; elapsed={time.perf_counter()-t0:.1f}s",
          flush=True)
    return 0


def cmd_freeze_screen() -> int:
    """Phase S step 5: freeze code/model inventories + screen manifest + screen protocol lock."""
    t0 = time.perf_counter()
    PROT.write_code_inventory(progress=_make_progress("freeze-screen"))
    PROT.write_model_inventory(progress=_make_progress("freeze-screen"))
    manifest = PROT.build_screen_manifest(progress=_make_progress("freeze-screen"))
    lock = PROT.freeze_screen_lock(progress=_make_progress("freeze-screen"))
    print(f"[d2f freeze-screen] SUCCESS: selected_n={manifest['selected_n']} "
          f"primary={manifest['capacity']['primary_ids']} reserve={manifest['capacity']['reserve_ids']}"
          f" lock_entropy={lock['bootstrap']['entropy']}; collection may start; "
          f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return 0


def cmd_screen_collect(max_attempts: int) -> int:
    """Phase S step 6: the formal selected_n screen collection (sim env, long, supervised)."""
    import collect_stage4d_d2f as d2fcollect
    receipt = d2fcollect.run_screen_collection(max_attempts_per_process=int(max_attempts))
    return 0 if receipt.get("status") == "COLLECTION_COMPLETE" else 1


def cmd_screen_run(device: str | None, batch_size: int, n_determinism: int) -> int:
    """Phase S step 7: the one-shot chain split -> score -> determinism -> freeze -> lock -> adjudicate."""
    t0 = time.perf_counter()
    from ac_twin import stage4d_d2f_inference as D2FINF
    from ac_twin import stage4d_d2f_statistics as D2FSTAT
    import stage4d_split_screen_truth as SPLIT

    dev = _resolve_device(device)
    paths = D2FSTAT.d2f_screen_artifact_paths()
    manifest = PROT._read_json(paths["screen_manifest"])
    selected_n = int(manifest["selected_n"])
    total_phases = 6
    phase = {"index": 0}

    def _phase(name: str) -> None:
        phase["index"] += 1
        print(f"[d2f screen-run] phase {phase['index']}/{total_phases} START {name} "
              f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    _phase(f"truth split (physical inference_inputs / sealed_truth separation, n={selected_n})")
    SPLIT.split_screen_truth(paths["role_dir"], paths["split_dir"],
                             role_salt=PROT.SCREEN_SALT, expect_groups=selected_n)

    _phase("hash-verified frozen component load (stack / ego / Direct inventory)")
    stack = D2FINF.load_predictor_stack()
    selected, ego_scaler, ego_models = D2FINF.load_selected_ego()
    inventory = D2FINF.load_risk_inventory(selected)
    if not paths["scorer_parity_receipt"].is_file():
        print("[d2f screen-run] REFUSED: the old-validation scorer parity receipt is missing; "
              "run scorer-parity first (plan 3.4 engineering acceptance)", flush=True)
        return 4

    _phase(f"truth-free scoring (device={dev}, batch_size={batch_size})")
    scores = D2FINF.score_d2f_screen(stack, ego_scaler, ego_models, inventory,
                                     paths["inference_inputs"],
                                     device=dev, batch_size=int(batch_size))

    _phase(f">={n_determinism}-group determinism recheck")
    determinism = D2FINF.determinism_recheck(stack, ego_scaler, ego_models, inventory,
                                             paths["inference_inputs"],
                                             device=dev, batch_size=int(batch_size),
                                             n_groups=int(n_determinism))

    _phase("score freeze + one-shot lock burn")
    D2FSTAT.freeze_d2f_screen_scores(None, scores=scores, determinism=determinism)
    D2FSTAT.burn_d2f_screen_lock(None)

    _phase("the SINGLE adjudication (truth opens once; 10,000 paired bootstrap, 80% CI)")
    decision = D2FSTAT.adjudicate_d2f_screen(None)
    elapsed = time.perf_counter() - t0
    print(f"[d2f screen-run] ADJUDICATED status={decision.status} task_passed="
          f"{decision.task['passed']} risk_passed={decision.risk['passed']} "
          f"elapsed={elapsed:.1f}s", flush=True)
    return 0 if decision.passed else 5


def _resolve_device(device: str | None) -> str:
    dev = str(device or "cuda")
    if dev.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            dev = "cpu"
    return dev


# --- Phase C (plan section 4; legal ONLY after the screen adjudication passed BOTH gates) ---------------


def cmd_confirm_power() -> int:
    """Phase C step 1: the 95%-CI confirm power from the frozen screen quantities (no collection)."""
    t0 = time.perf_counter()
    try:
        PROT.assert_confirm_phase_open()
    except RuntimeError as exc:
        print(f"[d2f confirm-power] REFUSED: {exc}", flush=True)
        return 4
    report = DPOW.run_confirm_power(progress=_make_progress("confirm-power"))
    elapsed = time.perf_counter() - t0
    if report.get("terminal"):
        print(f"[d2f confirm-power] TERMINAL {report['status']}: the 95% confirm AND-chain "
              f"reaches 75% joint power <0.80 at every admissible n; NO confirm manifest / lock "
              f"/ collection is created; the D2F authorization ends here "
              f"(elapsed={elapsed:.1f}s)", flush=True)
        return 5
    print(f"[d2f confirm-power] SUCCESS {report['status']}: selected_n={report['selected_n']} "
          f"(minimal admissible n at 75% joint power >=0.80, 95% CI chain); the confirm freeze "
          f"may proceed; elapsed={elapsed:.1f}s", flush=True)
    return 0


def cmd_freeze_confirm() -> int:
    """Phase C step 2: confirm code inventory + confirm manifest + confirm protocol lock."""
    t0 = time.perf_counter()
    try:
        PROT.assert_confirm_phase_open()
    except RuntimeError as exc:
        print(f"[d2f freeze-confirm] REFUSED: {exc}", flush=True)
        return 4
    PROT.write_confirm_code_inventory(progress=_make_progress("freeze-confirm"))
    try:
        manifest = PROT.build_confirm_manifest(progress=_make_progress("freeze-confirm"))
    except RuntimeError as exc:
        print(f"[d2f freeze-confirm] TERMINAL: {exc} "
              f"(elapsed={time.perf_counter()-t0:.1f}s)", flush=True)
        return 5
    lock = PROT.freeze_confirm_lock(progress=_make_progress("freeze-confirm"))
    print(f"[d2f freeze-confirm] SUCCESS: selected_n={manifest['selected_n']} "
          f"primary={manifest['capacity']['primary_ids']} reserve={manifest['capacity']['reserve_ids']}"
          f" lock_entropy={lock['bootstrap']['entropy']}; confirm collection may start; "
          f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return 0


def cmd_confirm_collect(max_attempts: int) -> int:
    """Phase C step 3: the formal selected_n confirm collection (sim env, long, supervised)."""
    import collect_stage4d_d2f as d2fcollect
    receipt = d2fcollect.run_confirm_collection(max_attempts_per_process=int(max_attempts))
    return 0 if receipt.get("status") == "COLLECTION_COMPLETE" else 1


def cmd_confirm_run(device: str | None, batch_size: int, n_determinism: int) -> int:
    """Phase C step 4: split -> score -> determinism -> freeze -> burn lock -> the SINGLE adjudication."""
    t0 = time.perf_counter()
    from ac_twin import stage4d_d2f_inference as D2FINF
    from ac_twin import stage4d_d2f_statistics as D2FSTAT
    import stage4d_split_screen_truth as SPLIT

    dev = _resolve_device(device)
    paths = D2FSTAT.d2f_confirm_artifact_paths()
    manifest = PROT._read_json(paths["confirm_manifest"])
    selected_n = int(manifest["selected_n"])
    total_phases = 6
    phase = {"index": 0}

    def _phase(name: str) -> None:
        phase["index"] += 1
        print(f"[d2f confirm-run] phase {phase['index']}/{total_phases} START {name} "
              f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    _phase(f"truth split (physical inference_inputs / sealed_truth separation, n={selected_n})")
    SPLIT.split_screen_truth(paths["role_dir"], paths["split_dir"],
                             role_salt=PROT.SCREEN_SALT, expect_groups=selected_n)

    _phase("hash-verified frozen component load (stack / ego / Direct inventory)")
    stack = D2FINF.load_predictor_stack()
    selected, ego_scaler, ego_models = D2FINF.load_selected_ego()
    inventory = D2FINF.load_risk_inventory(selected)
    if not paths["scorer_parity_receipt"].is_file():
        print("[d2f confirm-run] REFUSED: the old-validation scorer parity receipt is missing; "
              "run scorer-parity first (plan 3.4 engineering acceptance)", flush=True)
        return 4

    _phase(f"truth-free scoring (device={dev}, batch_size={batch_size})")
    scores = D2FINF.score_d2f_screen(stack, ego_scaler, ego_models, inventory,
                                     paths["inference_inputs"],
                                     device=dev, batch_size=int(batch_size))

    _phase(f">={n_determinism}-group determinism recheck")
    determinism = D2FINF.determinism_recheck(stack, ego_scaler, ego_models, inventory,
                                             paths["inference_inputs"],
                                             device=dev, batch_size=int(batch_size),
                                             n_groups=int(n_determinism))

    _phase("score freeze + one-shot lock burn (confirmation_used.lock)")
    D2FSTAT.freeze_d2f_confirm_scores(None, scores=scores, determinism=determinism)
    D2FSTAT.burn_d2f_confirm_lock(None)

    _phase("the SINGLE confirm adjudication (truth opens once; 10,000 paired bootstrap, 95% CI)")
    decision = D2FSTAT.adjudicate_d2f_confirm(None)
    elapsed = time.perf_counter() - t0
    print(f"[d2f confirm-run] ADJUDICATED status={decision.status} task_passed="
          f"{decision.task['passed']} risk_passed={decision.risk['passed']} "
          f"elapsed={elapsed:.1f}s", flush=True)
    return 0 if decision.passed else 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4D-D2F driver")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-power", help="Phase P: source binding + capacity audit + corrected "
                                       "power (no collection)")
    sub.add_parser("verify-fresh-ids", help="Phase S step 1: fresh-ID re-verification "
                                            "(read-only, create-once)")
    parity = sub.add_parser("collector-parity", help="Phase S step 2: same-input per-group "
                                                     "collector parity (sim env)")
    parity.add_argument("--groups", type=int, default=2)
    smoke = sub.add_parser("smoke", help="Phase S step 3: engineering smoke over the "
                                         "fresh-order prefix (<=160 candidates)")
    smoke.add_argument("--accepted-quota", type=int, default=12)
    smoke.add_argument("--candidate-limit", type=int, default=160)
    smoke.add_argument("--max-attempts-per-process", type=int, default=25)
    sub.add_parser("post-smoke-audit", help="Phase S step 3b: read-only post-smoke dual-pool "
                                            "re-audit (create-once)")
    scorer = sub.add_parser("scorer-parity", help="Phase S step 4: old-validation scorer "
                                                  "parity vs the D2E frozen quantities (sim env)")
    scorer.add_argument("--device", default=None)
    scorer.add_argument("--batch-size", type=int, default=4096)
    sub.add_parser("freeze-screen", help="Phase S step 5: freeze code/model inventories + "
                                         "screen manifest + screen protocol lock")
    collect = sub.add_parser("screen-collect", help="Phase S step 6: the formal selected_n "
                                                    "screen collection (sim env, long)")
    collect.add_argument("--max-attempts-per-process", type=int, default=100)
    run = sub.add_parser("screen-run", help="Phase S step 7: split -> score -> determinism -> "
                                            "freeze -> burn lock -> the single adjudication")
    run.add_argument("--device", default=None)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--determinism-groups", type=int, default=32)
    sub.add_parser("confirm-power", help="Phase C step 1: 95%%-CI confirm power from the frozen "
                                         "screen quantities (no collection)")
    sub.add_parser("freeze-confirm", help="Phase C step 2: confirm code inventory + confirm "
                                          "manifest + confirm protocol lock")
    ccollect = sub.add_parser("confirm-collect", help="Phase C step 3: the formal selected_n "
                                                      "confirm collection (sim env, long)")
    ccollect.add_argument("--max-attempts-per-process", type=int, default=100)
    crun = sub.add_parser("confirm-run", help="Phase C step 4: split -> score -> determinism -> "
                                              "freeze -> burn lock -> the SINGLE confirm "
                                              "adjudication (95%% CI)")
    crun.add_argument("--device", default=None)
    crun.add_argument("--batch-size", type=int, default=32)
    crun.add_argument("--determinism-groups", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "audit-power":
        return cmd_audit_power()
    if args.command == "verify-fresh-ids":
        return cmd_verify_fresh_ids()
    if args.command == "collector-parity":
        return cmd_collector_parity(args.groups)
    if args.command == "smoke":
        return cmd_smoke(args.accepted_quota, args.candidate_limit,
                         args.max_attempts_per_process)
    if args.command == "post-smoke-audit":
        return cmd_post_smoke_audit()
    if args.command == "scorer-parity":
        return cmd_scorer_parity(args.device, args.batch_size)
    if args.command == "freeze-screen":
        return cmd_freeze_screen()
    if args.command == "screen-collect":
        return cmd_screen_collect(args.max_attempts_per_process)
    if args.command == "screen-run":
        return cmd_screen_run(args.device, args.batch_size, args.determinism_groups)
    if args.command == "confirm-power":
        return cmd_confirm_power()
    if args.command == "freeze-confirm":
        return cmd_freeze_confirm()
    if args.command == "confirm-collect":
        return cmd_confirm_collect(args.max_attempts_per_process)
    if args.command == "confirm-run":
        return cmd_confirm_run(args.device, args.batch_size, args.determinism_groups)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
