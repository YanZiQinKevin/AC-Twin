"""Stage 2B intersection runner — phased small entry points.

Plan: AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-14-V0.3.

Implemented phases (Tasks 1-2):
- ``preflight-smoke``: P0 engineering smoke. Freezes the movement-signature /
  conflict lookup (create-once), prepares the collection manifest, collects 12
  ENGINEERING_SMOKE_ONLY episodes and runs the mechanism-field semantics QA.
  Writes one smoke receipt; the scientific status stays NOT_STARTED.
- ``collect-pilot``: P1 response-support pilot for one profile (A or B).
- ``select-profile``: mechanical unique-profile choice from the closed list of
  pilot support metrics (never V_evt or model errors); writes the create-once
  ``profile_selection.json``.
- ``collect-development``: P2 development collection (6,000 episodes,
  mechanically pre-split train/validation) once a profile selection exists.
- ``freeze-v04-profile``: create-once recovery receipt fixing profile B from
  the completed v0.3 support-only pilot, without touching value/model outputs.

Later phases (fit-models, gate-s, power-and-lock, collect-confirm,
infer-confirm, unlock-and-adjudicate) are registered here so the command
surface is stable; they are implemented in their own tasks and currently
refuse to run.

Long entries print plain-text phase / completed / total / percent / elapsed /
ETA at least every 30 seconds (the collector carries the progress reporter).
"""
from __future__ import annotations

import os
# Avoid the torch/MKL duplicate-OpenMP abort (OMP Error #15); deterministic and
# does not change results. Must be set before the first numpy/torch import.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2b_protocol as sp  # noqa: E402
from ac_twin import stage2b_data as sd  # noqa: E402
import collect_stage2b_intersection as col  # noqa: E402

STAGE2B_ROOT = "experiments/stage2b_intersection"


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(target: Path, obj: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


# --- P0: preflight smoke -----------------------------------------------------------------


def _freeze_movement_lookup(config: dict) -> dict:
    """Build the P0 movement/conflict lookup from the real network, or verify
    the frozen one against a rebuild. Create-once; any drift is a P0 failure."""
    from collect_pilot import make_env
    lookup_path = sp._resolve(config["movement_lookup_json"])
    env = make_env(config["env_id"], col._env_config(config))
    try:
        env.reset(seed=int(config["collection_seed"]))
        polylines = sd.extract_network_polylines(env)
    finally:
        env.close()
    rebuilt = sd.build_movement_lookup(polylines)
    n_crossing = sum(1 for a, row in rebuilt["crossing"].items()
                     for b, hit in row.items() if a != b and hit)
    if n_crossing == 0:
        raise RuntimeError("movement lookup found no crossing conflict pair; "
                           "the intersection geometry classification is broken")
    if lookup_path.exists():
        frozen = _load_json(lookup_path)
        if frozen.get("lookup_sha256") != rebuilt["lookup_sha256"]:
            raise RuntimeError(
                f"frozen movement lookup hash drift: {lookup_path}\n"
                f"  frozen  {frozen.get('lookup_sha256')}\n  rebuilt {rebuilt['lookup_sha256']}")
        return frozen
    _atomic_write_json(lookup_path, rebuilt)
    return rebuilt


def cmd_preflight_smoke(config_path: str | Path) -> dict:
    started = time.time()
    config = _load_json(config_path)
    sp.validate_stage2b_config(config)
    output_dir = sp._resolve(config["output_dir"])
    print("[preflight-smoke] phase=config+lookup config validated; "
          "freezing movement lookup", flush=True)

    lookup = _freeze_movement_lookup(config)
    print(f"[preflight-smoke] movement_lookup_sha256={lookup['lookup_sha256']} "
          f"movements={len(lookup['movements'])}", flush=True)

    manifest_path = output_dir / "collection_manifest.json"
    if manifest_path.exists():
        manifest = sp.verify_collection_manifest(output_dir, config)
        print("[preflight-smoke] existing collection manifest verified", flush=True)
    else:
        manifest = sp.prepare_collection(
            config, output_dir, config_path=config_path,
            extra_bindings={"movement_lookup": config["movement_lookup_json"]})
        print(f"[preflight-smoke] manifest created; "
              f"historical_seeds={manifest['zero_overlap']['historical_episode_seeds']} "
              f"schedule={manifest['seed_schedule_size']}", flush=True)

    print("[preflight-smoke] phase=collect collecting 12 ENGINEERING_SMOKE_ONLY "
          "episodes", flush=True)
    receipt = col.collect_stage2b(config, resume=True, config_path=config_path)
    print(f"[preflight-smoke] collected accepted={receipt['accepted_pairs']} "
          f"attempts={receipt['attempts']} "
          f"prefix_max_abs_diff={receipt['prefix_max_abs_diff']:.3e} "
          f"identity_max_abs_diff={receipt['identity_max_abs_diff']:.3e} "
          f"max_road_vehicles={receipt['max_road_vehicles_seen']}", flush=True)

    print("[preflight-smoke] phase=semantics-qa", flush=True)
    with np.load(sp._resolve(config["data_output"]), allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    qa = sd.smoke_semantic_qa(data)

    qa_pass = bool(qa["pass"])
    smoke_receipt = {
        "experiment_name": config["experiment_name"],
        "analysis_status": sp.STATUS_SMOKE,
        "plan_id": config["plan_id"],
        "p0_status": "P0_SMOKE_PASS" if qa_pass else "P0_SMOKE_FAILED",
        "scientific_status": "NOT_STARTED",
        "note": ("Engineering smoke only: validates pipeline mechanics and "
                 "mechanism-field semantics. No response density was estimated, "
                 "no profile was selected, no normalization/bins were fitted and "
                 "no scientific state was produced."),
        "movement_lookup_sha256": lookup["lookup_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "collection_receipt_sha256": sp.sha256_file(sp._resolve(config["collection_receipt"])),
        "data_output_sha256": sp.sha256_file(sp._resolve(config["data_output"])),
        "collection": {
            "accepted_pairs": receipt["accepted_pairs"],
            "attempts": receipt["attempts"],
            "skip_counts": receipt["skip_counts"],
            "accepted_strata": receipt["accepted_strata"],
            "prefix_max_abs_diff": receipt["prefix_max_abs_diff"],
            "identity_max_abs_diff": receipt["identity_max_abs_diff"],
            "max_road_vehicles_seen": receipt["max_road_vehicles_seen"],
            "initial_identity_count_max": receipt["initial_identity_count_max"],
        },
        "semantic_qa": qa,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _atomic_write_json(output_dir / "smoke_receipt.json", smoke_receipt)
    print(f"[preflight-smoke] P0 status={smoke_receipt['p0_status']} "
          f"signatures={qa['distinct_movement_signatures']} "
          f"qa_pass={qa_pass}", flush=True)
    if not qa_pass:
        raise RuntimeError(f"P0 smoke semantic QA failed: {qa}")
    return smoke_receipt


# --- P1: pilot + mechanical profile selection --------------------------------------------


def cmd_collect_pilot(config_path: str | Path, profile: str, resume: bool) -> dict:
    container = _load_json(config_path)
    effective = sp.resolve_pilot_profile_config(container, profile)
    output_dir = sp._resolve(effective["output_dir"])
    if (output_dir / "collection_manifest.json").exists():
        sp.verify_collection_manifest(output_dir, effective)
        print(f"[collect-pilot] profile={profile} existing manifest verified", flush=True)
    else:
        manifest = sp.prepare_collection(effective, output_dir, config_path=config_path)
        print(f"[collect-pilot] profile={profile} manifest created; "
              f"historical_seeds={manifest['zero_overlap']['historical_episode_seeds']} "
              f"other_roles={manifest['zero_overlap']['other_role_schedules']}",
              flush=True)
    receipt = col.collect_stage2b(effective, resume=resume, config_path=config_path)
    print(f"[collect-pilot] profile={profile} accepted={receipt['accepted_pairs']} "
          f"attempts={receipt['attempts']}", flush=True)
    return receipt


def _profile_metrics(container: dict, profile: str) -> dict | None:
    """Support metrics for one collected pilot profile; None if not collected.

    Combines the closed list of pilot-computable quantities (plan 5.1) with
    collection-receipt QA counters. Never touches model outputs or V_evt.
    """
    effective = sp.resolve_pilot_profile_config(container, profile)
    data_output = sp._resolve(effective["data_output"])
    receipt_path = sp._resolve(effective["collection_receipt"])
    if not data_output.is_file() or not receipt_path.is_file():
        return None
    receipt = _load_json(receipt_path)
    if receipt.get("partial", True) or int(receipt["accepted_pairs"]) != int(
            effective["n_pairs"]):
        return None
    with np.load(data_output, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    metrics = sd.pilot_support_metrics(data)
    strata = receipt["accepted_strata"]
    metrics["per_stratum_min"] = int(min(strata.values())) if strata else 0
    metrics["prefix_identity_qa_pass"] = bool(
        float(receipt["prefix_max_abs_diff"]) <= 1e-6
        and float(receipt["identity_max_abs_diff"]) <= 1e-6)
    return metrics


def cmd_select_profile(config_path: str | Path) -> dict:
    container = _load_json(config_path)
    sp.validate_pilot_container_config(container)
    selection_path = sp._resolve(container["profile_selection_json"])
    metrics = {letter: _profile_metrics(container, letter) for letter in ("A", "B")}
    for letter, m in metrics.items():
        if m is None:
            print(f"[select-profile] profile={letter} not collected or incomplete",
                  flush=True)
        else:
            print(f"[select-profile] profile={letter} support_rate={m['support_rate']:.3f} "
                  f"positive_episodes={m['positive_episodes']} "
                  f"median_paired={m['median_paired_valid_non_ego']:.1f} "
                  f"bins={m['bin_episode_counts']}", flush=True)
    decision = sp.select_profile(metrics)
    record = sp.write_profile_selection(selection_path, decision)
    print(f"[select-profile] status={record['status']} "
          f"selected={record['selected_profile']}", flush=True)
    return record


def cmd_freeze_v04_profile(config_path: str | Path) -> dict:
    """Freeze v0.4 profile B from the completed v0.3 support-only pilot."""
    recovery = _load_json(config_path)
    if recovery.get("plan_id") != sp.V04_PLAN_ID:
        raise ValueError(f"recovery plan_id must be {sp.V04_PLAN_ID}")
    pilot_config_path = sp._resolve(recovery["source_pilot_config"])
    container = _load_json(pilot_config_path)
    metrics = _profile_metrics(container, "B")
    decision = sp.build_v04_profile_b_recovery(metrics)

    source_sha256: dict[str, str] = {}
    for label, value in recovery["source_files"].items():
        path = sp._resolve(value)
        if not path.is_file():
            raise FileNotFoundError(f"v0.4 recovery source missing: {label} -> {path}")
        source_sha256[str(label)] = sp.sha256_file(path)
    decision["source_sha256"] = source_sha256
    decision["source_files"] = dict(recovery["source_files"])
    decision["note"] = (
        "Development-only recovery before any model error, V_evt or Gate A2B "
        "was produced. The v0.3 terminal receipt remains immutable.")
    selection_path = sp._resolve(recovery["profile_selection_json"])
    record = sp.write_profile_selection(
        selection_path, decision, plan_id=sp.V04_PLAN_ID)
    print(f"[freeze-v04-profile] status={record['status']} "
          f"selected={record['selected_profile']}", flush=True)
    return record


# --- P2: development collection (software ready; launched manually) -----------------------


def cmd_collect_development(config_path: str | Path, resume: bool) -> dict:
    config = _load_json(config_path)
    selection_path = sp._resolve(config["profile_selection_json"])
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"profile selection missing; run select-profile first: {selection_path}")
    selection = _load_json(selection_path)
    resolved = sp.resolve_development_config(config, selection)
    output_dir = sp._resolve(resolved["output_dir"])
    if (output_dir / "collection_manifest.json").exists():
        sp.verify_collection_manifest(output_dir, resolved)
        print("[collect-development] existing manifest verified", flush=True)
    else:
        manifest = sp.prepare_collection(
            resolved, output_dir, config_path=config_path,
            extra_bindings={"profile_selection": config["profile_selection_json"]})
        print(f"[collect-development] manifest created; profile={selection['selected_profile']} "
              f"historical_seeds={manifest['zero_overlap']['historical_episode_seeds']} "
              f"other_roles={manifest['zero_overlap']['other_role_schedules']}",
              flush=True)
    receipt = col.collect_stage2b(resolved, resume=resume, config_path=config_path)
    print(f"[collect-development] accepted={receipt['accepted_pairs']} "
          f"attempts={receipt['attempts']}", flush=True)
    return receipt


# --- later phases (registered, not yet implemented) ---------------------------------------


def _not_implemented(phase: str, task: str) -> None:
    raise SystemExit(
        f"[{phase}] registered but implemented in {task}; refusing to run. "
        "See experiments/stage2b_intersection/HANDOFF_STATE.md.")


# --- CLI -----------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2B intersection runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight-smoke")
    p.add_argument("--config", required=True)

    p = sub.add_parser("collect-pilot")
    p.add_argument("--config", required=True)
    p.add_argument("--profile", required=True, choices=("A", "B"))
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("select-profile")
    p.add_argument("--config", required=True)

    p = sub.add_parser("freeze-v04-profile")
    p.add_argument("--config", required=True)

    p = sub.add_parser("collect-development")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")

    for name in ("fit-models", "gate-s", "power-and-lock", "collect-confirm",
                 "infer-confirm", "unlock-and-adjudicate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Any:
    args = parse_args(argv)
    config_path = sp._resolve(args.config) if getattr(args, "config", None) else None
    if args.command == "preflight-smoke":
        return cmd_preflight_smoke(config_path)
    if args.command == "collect-pilot":
        return cmd_collect_pilot(config_path, args.profile, args.resume)
    if args.command == "select-profile":
        return cmd_select_profile(config_path)
    if args.command == "freeze-v04-profile":
        return cmd_freeze_v04_profile(config_path)
    if args.command == "collect-development":
        return cmd_collect_development(config_path, args.resume)
    task = {"fit-models": "Task 3", "gate-s": "Task 3", "power-and-lock": "Task 4",
            "collect-confirm": "Task 5", "infer-confirm": "Task 5",
            "unlock-and-adjudicate": "Task 5"}[args.command]
    return _not_implemented(args.command, task)


if __name__ == "__main__":
    main()
