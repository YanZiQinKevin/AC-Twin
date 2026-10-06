"""Stage 4D-D2F development-screen one-shot chain: freeze -> lock -> open truth -> adjudicate.

Plan ``AC-TWIN-RFER-STAGE4D-D2F-POWER-CORRECTION-EXTERNAL-CONFIRM-2026-09-24-V0.1`` section 3.5.
The statistical kernel (truth computation, the 32-slot quantity vector, the 10,000-draw
whole-group paired bootstrap, the gate thresholds) is reused BITWISE from the frozen D2
``stage4d_external_statistics``; this module adapts only the D2F namespace bindings:

- artifact paths under ``experiments/stage4_reliability/stage4d_d2f/`` (the old D2 screen and
  its lock are never touched);
- the frozen selection is the D2E ``EGO_TRAJECTORY_PLUS_RANK`` chain: final_selection +
  validation_report (status ``D2E_INTERFACE_REPAIR_QUALIFIED``) + riskfit receipt
  (interaction threshold) + the D2E-frozen primary baseline ``SEED_MARGIN_STD``;
- the completeness gate c1 is evaluated at ``selected_n`` (1000) instead of D2's 4000;
- the fresh bootstrap entropy ``2026122407`` (registered in ``stage4d_d2f_protocol``);
- the preregistered terminal states: task gate fails first ->
  ``D2F_EXTERNAL_DEV_TASK_NOT_READY``; task passes but the risk gate fails ->
  ``D2F_EXTERNAL_DEV_RISK_NOT_READY``; degenerate denominators / QA / leak ->
  ``INCONCLUSIVE_D2F_EXTERNAL_INVALID``; both gates pass -> ``D2F_DEV_READY_FOR_CONFIRM_POWER``.

Gate conditions themselves are D2 section 8.2 c1-c8 and 8.3 r1-r6 VERBATIM (two-sided 80% CI,
10,000 resamples); no threshold, baseline, model or seed may move.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin import stage4d_d2f_protocol as PROT
from ac_twin import stage4d_external_statistics as D2STAT
from ac_twin import stage4d_external_training as T
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]
D2F_DIR = PROT.D2F_DIR
D2E_DIR = PROT.D2E_DIR
D2_DIR = PROT.D2_DIR

PLAN_ID = PROT.PLAN_ID
SCREEN_ROLE = PROT.SCREEN_ROLE

N_ACTIONS = D2STAT.N_ACTIONS
N_MODEL_SEEDS = D2STAT.N_MODEL_SEEDS
N_QUANTITIES = D2STAT.N_QUANTITIES
BASELINE_TIE_TOLERANCE = D2STAT.BASELINE_TIE_TOLERANCE
STANDARD_BASELINES = D2STAT.STANDARD_BASELINES
ALL_BASELINE_NAMES = D2STAT.ALL_BASELINE_NAMES
QA_KEYS = D2STAT.QA_KEYS
TSTAR_SPLIT = D2STAT.TSTAR_SPLIT
TRIM_FRACTIONS = D2STAT.TRIM_FRACTIONS
CH_S, CH_D = D2STAT.CH_S, D2STAT.CH_D

# --- preregistered terminal states (plan 3.5 / 7) ---------------------------------------------------
STATUS_DEV_READY = "D2F_DEV_READY_FOR_CONFIRM_POWER"
STATUS_TASK_NOT_READY = "D2F_EXTERNAL_DEV_TASK_NOT_READY"
STATUS_RISK_NOT_READY = "D2F_EXTERNAL_DEV_RISK_NOT_READY"
STATUS_INVALID = "INCONCLUSIVE_D2F_EXTERNAL_INVALID"

_BASELINE_NPZ_PREFIX = "baseline_risk__"


# --- D2F artifact paths + the frozen D2E selection binding ---------------------------------------------


def d2f_screen_artifact_paths(config: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Every one-shot D2F screen path (plan 3.4/3.5); the D2/D2E sources are read-only."""
    config = config or {}
    run_dir = Path(config.get("run_dir", D2F_DIR))
    role_dir = run_dir / "roles" / SCREEN_ROLE
    dev = run_dir / "development"
    split_dir = dev / "screen_split"
    final = json.loads((D2E_DIR / "development" / "final_selection.json")
                       .read_text(encoding="utf-8"))
    return {
        "run_dir": run_dir,
        "dev_dir": dev,
        "role_dir": role_dir,
        "groups_npz": role_dir / "groups.npz",
        "groups_meta": role_dir / "groups_meta.json",
        "collection_receipt": role_dir / "collection_receipt.json",
        "split_dir": split_dir,
        "inference_inputs": split_dir / "inference_inputs.npz",
        "sealed_truth": split_dir / "sealed_truth.npz",
        "split_receipt": split_dir / "split_receipt.json",
        "scores_npz": dev / "screen_scores.npz",
        "scores_receipt": dev / "screen_scores_receipt.json",
        "determinism_receipt": dev / "screen_determinism_receipt.json",
        "lock": dev / "development_screen_used.lock",
        "quantities_npz": dev / "screen_group_quantities.npz",
        "quantities_receipt": dev / "screen_group_quantities_receipt.json",
        "report": dev / "development_screen_report.json",
        "scorer_parity_receipt": dev / "scorer_parity_receipt.json",
        "protocol_lock": run_dir / PROT.SCREEN_LOCK_NAME,
        "screen_manifest": run_dir / PROT.SCREEN_MANIFEST_NAME,
        # read-only frozen sources (never written by D2F):
        "d2e_final_selection": D2E_DIR / "development" / "final_selection.json",
        "d2e_validation_report": D2E_DIR / "development" / "validation_report.json",
        "d2e_riskfit_receipt": D2E_DIR / str(final["risk_run_dir"]) / "development" / "models"
                               / "fit_receipt.json",
        "d2_predictor_stack": D2_DIR / "development" / "base_models" / "predictor_stack.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_frozen_d2e_selection(paths: Mapping[str, Path]) -> dict[str, Any]:
    """The D2F screen exists only because the D2E repaired pipeline QUALIFIED on old validation."""
    final = _read_json(paths["d2e_final_selection"])
    report = _read_json(paths["d2e_validation_report"])
    if report.get("status") != "D2E_INTERFACE_REPAIR_QUALIFIED":
        raise ValueError(
            f"D2F screen refused: the D2E validation status is {report.get('status')!r}, not "
            "D2E_INTERFACE_REPAIR_QUALIFIED; no screen artifact may be produced")
    selected = str(final["selected_candidate"])
    gate = report.get("candidate_gate", {}).get(selected, {})
    strongest = str(gate.get("strongest_baseline", ""))
    if strongest not in STANDARD_BASELINES:
        raise ValueError(f"the D2E-frozen primary baseline {strongest!r} is not an eligible "
                         "standard baseline (fail closed)")
    riskfit = _read_json(paths["d2e_riskfit_receipt"])
    return {
        "selected_candidate": selected,
        "strongest_standard_baseline": strongest,
        "interaction_threshold": float(riskfit["interaction_threshold"]),
        "ego_checkpoints": {str(k): str(v) for k, v in final["ego_checkpoints"].items()},
        "riskfit_checkpoint_sha256": {str(k): str(v)
                                      for k, v in riskfit["checkpoint_sha256"].items()},
        "riskfit_checkpoint_paths": {str(k): str(v)
                                     for k, v in riskfit["checkpoint_paths"].items()},
        "ensemble_rule": str(final.get("ensemble_rule", "arithmetic_mean_of_five_seeds")),
    }


def _verify_checkpoint_pairing(selection: Mapping[str, Any]) -> bool:
    """The selected ego checkpoints and the Direct checkpoints pair one-to-one by content hash."""
    binding = _read_json(D2F_DIR / PROT.SOURCE_BINDING_NAME)
    bound = {str(Path(str(e["path"]))): str(e["sha256"])
             for e in binding["expanded"]["checkpoints"]}
    for seed, rel in selection["ego_checkpoints"].items():
        path = D2E_DIR / "development" / "ego_models" / Path(str(rel)).name
        if not path.is_file() or bound.get(str(path)) != sha256_file(path):
            return False
    for seed, digest in selection["riskfit_checkpoint_sha256"].items():
        path = Path(selection["riskfit_checkpoint_paths"][seed])
        if not path.is_file() or sha256_file(path) != digest:
            return False
    return True


# --- plan 3.4 steps: freeze the screen scores create-once -------------------------------------------------


def freeze_d2f_screen_scores(config: Mapping[str, Any] | None,
                             *,
                             scores: Mapping[str, Any],
                             determinism: Mapping[str, Any]) -> Path:
    """Freeze the D2F screen scores + receipts create-once (mirrors the D2 contract bitwise).

    ``scores`` is computed from ``inference_inputs.npz`` ONLY; ``determinism.bitwise_identical``
    must be True. A rerun verifies hashes and never overwrites. Never touches the sealed truth.
    """
    paths = d2f_screen_artifact_paths(config)
    selection = _load_frozen_d2e_selection(paths)
    npz_path = paths["scores_npz"]
    receipt_path = paths["scores_receipt"]
    det_path = paths["determinism_receipt"]
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if not npz_path.is_file() or sha256_file(npz_path) != receipt["scores_sha256"]:
            raise ValueError(f"D2F screen scores drifted from their create-once receipt: {npz_path}")
        if not det_path.is_file():
            raise ValueError("D2F screen determinism receipt missing after score freeze")
        return npz_path
    if npz_path.exists():
        raise RuntimeError(f"D2F screen scores exist without their create-once receipt: {npz_path}")

    group_seed = np.asarray(scores["group_seed"], dtype=np.int64).ravel()
    t_star = np.asarray(scores["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(scores["interaction_min_distance"], dtype=np.float64).ravel()
    j_hat = np.asarray(scores["j_hat"], dtype=np.float64)
    ensemble = np.asarray(scores["ensemble_risk"], dtype=np.float64).ravel()
    by_seed = np.asarray(scores["risk_by_seed"], dtype=np.float64)
    valid = np.asarray(scores["valid_mask"], dtype=bool).ravel()
    groups = int(group_seed.size)
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} D2F screen groups have non-finite features; the screen "
            "contract rejects zero-padding, so scoring is refused")
    if j_hat.shape != (groups, N_ACTIONS) or ensemble.shape != (groups,) \
            or by_seed.shape != (N_MODEL_SEEDS, groups) \
            or distance.shape != (groups,) or t_star.shape != (groups,):
        raise ValueError("D2F screen score arrays must align on the group axis")
    if not np.isfinite(j_hat).all() or not np.isfinite(ensemble).all() \
            or not np.isfinite(by_seed).all():
        raise ValueError("non-finite D2F screen score")
    if not np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15):
        raise ValueError("the ensemble must be exactly the arithmetic mean of five seeds")
    if not bool(determinism.get("bitwise_identical")):
        raise ValueError("the repeated inference was not elementwise identical; the D2F screen "
                         "halts before the score freeze")

    payload = {
        "group_seed": group_seed,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "j_hat": j_hat,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": np.asarray(scores["mu_only_ensemble"], dtype=np.float64).ravel(),
        "class_only_ensemble": np.asarray(scores["class_only_ensemble"], dtype=np.float64).ravel(),
    }
    baselines = scores["baselines"]
    for name in ALL_BASELINE_NAMES:
        risk = np.asarray(baselines[name], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"D2F screen baseline risk {name!r} must be finite and aligned")
        payload[_BASELINE_NPZ_PREFIX + name] = risk

    atomic_save_npz(npz_path, **payload)

    det_receipt = dict(determinism)
    det_receipt.setdefault("plan_id", PLAN_ID)
    det_receipt["groups_sha256"] = sha256_file(paths["groups_npz"]) \
        if paths["groups_npz"].is_file() else ""
    det_receipt["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_once_json(det_path, det_receipt)

    receipt = {
        "plan_id": PLAN_ID,
        "role": SCREEN_ROLE,
        "n_groups": int(groups),
        "scores_path": str(npz_path),
        "scores_sha256": sha256_file(npz_path),
        "inference_inputs_sha256": sha256_file(paths["inference_inputs"])
        if paths["inference_inputs"].is_file() else "",
        "split_receipt_sha256": sha256_file(paths["split_receipt"])
        if paths["split_receipt"].is_file() else "",
        "d2e_final_selection_sha256": sha256_file(paths["d2e_final_selection"]),
        "d2e_riskfit_receipt_sha256": sha256_file(paths["d2e_riskfit_receipt"]),
        "scorer_parity_receipt_sha256": sha256_file(paths["scorer_parity_receipt"])
        if paths["scorer_parity_receipt"].is_file() else "",
        "strongest_standard_baseline": selection["strongest_standard_baseline"],
        "selected_ego_candidate": selection["selected_candidate"],
        "model_seeds": [int(s) for s in T.MODEL_SEEDS],
        "ensemble_definition": "arithmetic_mean_of_five_seeds",
        "determinism_receipt_sha256": sha256_file(det_path),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return npz_path


def _verify_scores_chain(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Re-hash the frozen scores against their create-once receipt (fail-closed)."""
    receipt = _read_json(paths["scores_receipt"])
    if not paths["scores_npz"].is_file() or sha256_file(paths["scores_npz"]) \
            != receipt["scores_sha256"]:
        raise ValueError("D2F screen scores drifted from their create-once receipt")
    if not paths["determinism_receipt"].is_file():
        raise ValueError("the repeated-inference determinism receipt is missing")
    determinism = _read_json(paths["determinism_receipt"])
    if not determinism.get("bitwise_identical"):
        raise ValueError("the repeated inference was not elementwise identical")
    return receipt


# --- burn the one-shot lock ------------------------------------------------------------------------------


def burn_d2f_screen_lock(config: Mapping[str, Any] | None) -> Path:
    """Create the D2F ``development_screen_used.lock`` exactly once, after the full score freeze.

    Same one-shot semantics as D2: refuses unless the D2E selection qualifies, the scores
    hash-match their receipt and the determinism receipt is elementwise identical. The lock
    NEVER reads the sealed truth bytes.
    """
    paths = d2f_screen_artifact_paths(config)
    selection = _load_frozen_d2e_selection(paths)
    receipt = _verify_scores_chain(paths)
    bindings = {
        "scores_sha256": receipt["scores_sha256"],
        "scores_receipt_sha256": sha256_file(paths["scores_receipt"]),
        "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
        "d2e_final_selection_sha256": sha256_file(paths["d2e_final_selection"]),
        "d2e_riskfit_receipt_sha256": sha256_file(paths["d2e_riskfit_receipt"]),
        "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
    }
    lock_path = paths["lock"]
    if lock_path.exists():
        payload = _read_json(lock_path)
        if payload.get("plan_id") != PLAN_ID or payload.get("bindings") != bindings:
            raise ValueError(
                "the burned D2F screen lock no longer matches the on-disk frozen artifacts "
                "(tampering or drift); the one-shot screen is compromised")
        return lock_path
    payload = {
        "plan_id": PLAN_ID,
        "role": SCREEN_ROLE,
        "semantics": ("one-shot: the screen truth may be read only after this lock exists; all "
                      "screen scores are frozen and hash-bound above"),
        "strongest_standard_baseline": selection["strongest_standard_baseline"],
        "selected_ego_candidate": selection["selected_candidate"],
        "interaction_threshold": float(selection["interaction_threshold"]),
        "bindings": bindings,
        "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path, payload)
    return lock_path


# --- open truth once, assemble, adjudicate -----------------------------------------------------------------


def assert_d2f_screen_adjudication_open(config: Mapping[str, Any] | None) -> None:
    """Fail closed unless the one-shot D2F adjudication may legally start."""
    paths = d2f_screen_artifact_paths(config)
    _load_frozen_d2e_selection(paths)
    if not paths["lock"].is_file():
        raise ValueError(
            "D2F screen truth may not be read: development_screen_used.lock is not burned; "
            "freeze every score artifact and burn the lock first (plan 3.4)")
    for key in ("quantities_npz", "quantities_receipt", "report"):
        if paths[key].exists():
            raise FileExistsError(
                f"a second D2F screen adjudication is refused; already exists: {paths[key]}")


def assemble_d2f_screen_group_quantities(config: Mapping[str, Any] | None,
                                         *,
                                         screen_groups: int) -> Path:
    """Open the sealed truth ONCE, build regret/strata/reactive, write the create-once quantities.

    Mirrors the D2 assembly bitwise except the completeness QA evaluates ``screen_groups``
    (the D2F ``selected_n``) and the frozen bindings are the D2E selection chain.
    """
    paths = d2f_screen_artifact_paths(config)
    assert_d2f_screen_adjudication_open(config)
    selection = _load_frozen_d2e_selection(paths)
    score_receipt = _verify_scores_chain(paths)
    burn_d2f_screen_lock(config)  # re-verify the lock bindings against the current files
    lock_payload = _read_json(paths["lock"])

    with np.load(paths["scores_npz"], allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    if not paths["sealed_truth"].is_file():
        raise ValueError(f"sealed truth missing: {paths['sealed_truth']}")
    with np.load(paths["sealed_truth"], allow_pickle=False) as archive:
        truth = {key: archive[key] for key in archive.files}

    group_seed = np.asarray(scores["group_seed"], dtype=np.int64).ravel()
    truth_seed = np.asarray(truth["group_seed"], dtype=np.int64).ravel()
    groups = int(group_seed.size)

    # --- truth side: legal ONLY here, after the burned lock -----------------------------------
    cf = np.asarray(truth["object_counterfactual_futures"], dtype=np.float64)
    factual = np.asarray(truth["object_factual_future"], dtype=np.float64)
    evaluation_mask = np.asarray(truth["evaluation_mask"], dtype=bool)
    history_last_xy = np.asarray(truth["history_last_xy"], dtype=np.float64)
    j_hat = np.asarray(scores["j_hat"], dtype=np.float64)
    labels = D2STAT.compute_screen_truth(cf, factual, history_last_xy, evaluation_mask, j_hat)
    j_true = labels["j_true"]

    threshold = float(selection["interaction_threshold"])
    t_star = np.asarray(scores["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(scores["interaction_min_distance"], dtype=np.float64).ravel()
    tstar_layer = (t_star >= TSTAR_SPLIT).astype(np.int64)
    interaction_layer = (distance > threshold).astype(np.int64)

    ensemble = np.asarray(scores["ensemble_risk"], dtype=np.float64).ravel()
    by_seed = np.asarray(scores["risk_by_seed"], dtype=np.float64)

    # --- validity / QA flags (evaluated inside the gates) --------------------------------------
    truth_finite = bool(np.isfinite(j_true).all() and np.isfinite(labels["regret"]).all())
    scenario_ids = json.loads(np.asarray(truth["scenario_ids_json"]).item()) \
        if "scenario_ids_json" in truth else [str(s) for s in group_seed]
    role_seeds_valid = bool(
        group_seed.size == groups and np.unique(group_seed).size == groups
        and np.array_equal(group_seed, truth_seed)
        and len(set(scenario_ids)) == len(scenario_ids))
    valid_cells = evaluation_mask                                # (G,20,5)
    cf_xy = cf[..., [CH_S, CH_D]]                               # (G,10,20,5,2)
    if valid_cells.any():
        mask_b = np.broadcast_to(valid_cells[:, None, :, :], cf_xy.shape[:-1])
        pts = cf_xy[mask_b]
        span = float((pts.max(axis=0) - pts.min(axis=0)).max()) if pts.size else 0.0
    else:
        span = 0.0
    projection_nondegenerate = bool(np.isfinite(span) and span > 1e-6)
    per_group_cells = valid_cells.reshape(groups, -1).any(axis=1) \
        if valid_cells.shape[0] == groups else np.zeros(groups, dtype=bool)
    eval_mask_nondegenerate = bool(valid_cells.size > 0 and valid_cells.mean() > 0.0
                                   and per_group_cells.all())

    qa = {
        "all_complete": bool(
            groups == int(screen_groups) and j_true.shape == (groups, N_ACTIONS)
            and j_hat.shape == (groups, N_ACTIONS) and truth_finite
            and int(score_receipt["n_groups"]) == groups),
        "hashes_match": True,  # _verify_scores_chain + lock re-verification passed above
        "role_seeds_valid": role_seeds_valid,
        "determinism_passed": True,  # burn re-verified the determinism receipt
        "one_shot_locks": True,
        "ensemble_is_arithmetic_mean": bool(
            by_seed.shape == (N_MODEL_SEEDS, groups)
            and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)),
        "baseline_matches_frozen": bool(
            selection["strongest_standard_baseline"] in STANDARD_BASELINES),
        "checkpoints_match_frozen": _verify_checkpoint_pairing(selection),
    }

    from ac_twin import stage4cp_statistics as s4cs
    trim_keep = {f"trim_keep_{str(frac).replace('.', '_')}": s4cs.trim_high_regret(
        labels["regret"], frac) for frac in TRIM_FRACTIONS}

    npz_path = paths["quantities_npz"]
    payload = {
        "group_seed": group_seed,
        "regret": labels["regret"],
        "y_high": labels["y_high"],
        "a_hat": labels["a_hat"].astype(np.int64),
        "a_star": labels["a_star"].astype(np.int64),
        "random_regret": labels["random_regret"],
        "idle_regret": labels["idle_regret"],
        "J_hat": j_hat,
        "J_true": j_true,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": np.asarray(scores["mu_only_ensemble"], dtype=np.float64).ravel(),
        "class_only_ensemble": np.asarray(scores["class_only_ensemble"], dtype=np.float64).ravel(),
        "tstar_layer": tstar_layer,
        "interaction_layer": interaction_layer,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "reactive": labels["reactive"].astype(bool),
        "scenario_ids_json": np.asarray(json.dumps([str(s) for s in scenario_ids])),
        "map_family": np.zeros(groups, dtype=np.int64),  # single Waymo training_20s family
        "n_groups": np.int64(groups),
        "interaction_threshold": np.float64(threshold),
        **trim_keep,
    }
    for name in ALL_BASELINE_NAMES:
        payload[_BASELINE_NPZ_PREFIX + name] = np.asarray(
            scores[_BASELINE_NPZ_PREFIX + name], dtype=np.float64).ravel()
    atomic_save_npz(npz_path, **payload)

    receipt = {
        "plan_id": PLAN_ID,
        "role": SCREEN_ROLE,
        "n_groups": int(groups),
        "screen_group_quantities_path": str(npz_path),
        "screen_group_quantities_sha256": sha256_file(npz_path),
        "strongest_standard_baseline": selection["strongest_standard_baseline"],
        "selected_ego_candidate": selection["selected_candidate"],
        "interaction_threshold": float(threshold),
        "interaction_threshold_source": "d2e riskfit fit_receipt.json (frozen)",
        "tstar_stratum_split": TSTAR_SPLIT,
        "trim_fractions": [float(f) for f in TRIM_FRACTIONS],
        "map_family_note": ("the frozen collector records no finer map_id; the screen is one "
                            "Waymo training_20s -> MetaDrive family (map_family all 0)."),
        "bootstrap_entropy": PROT.D2F_SCREEN_BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": D2STAT.BOOTSTRAP_RESAMPLES,
        "confidence_screen": D2STAT.CONFIDENCE_SCREEN,
        "qa": qa,
        "projection_nondegenerate": projection_nondegenerate,
        "eval_mask_nondegenerate": eval_mask_nondegenerate,
        "bindings": {
            "scores_sha256": score_receipt["scores_sha256"],
            "scores_receipt_sha256": sha256_file(paths["scores_receipt"]),
            "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
            "sealed_truth_sha256": sha256_file(paths["sealed_truth"]),
            "inference_inputs_sha256": sha256_file(paths["inference_inputs"])
            if paths["inference_inputs"].is_file() else "",
            "split_receipt_sha256": sha256_file(paths["split_receipt"])
            if paths["split_receipt"].is_file() else "",
            "d2e_final_selection_sha256": sha256_file(paths["d2e_final_selection"]),
            "d2e_validation_report_sha256": sha256_file(paths["d2e_validation_report"]),
            "d2e_riskfit_receipt_sha256": sha256_file(paths["d2e_riskfit_receipt"]),
            "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
            "statistics_code_sha256": sha256_file(
                ROOT / "code/ac_twin/stage4d_d2f_statistics.py"),
        },
        "lock_sha256": sha256_file(paths["lock"]),
        "lock_burned_at": lock_payload.get("burned_at", ""),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["quantities_receipt"], receipt)
    return npz_path


# --- the D2F gates (D2 section 8.2 c1-c8 + 8.3 r1-r6 verbatim, n = selected_n) ---------------------------


def evaluate_d2f_screen_gates(quantities: Mapping[str, Any],
                              bootstrap: D2STAT.BootstrapResult,
                              *,
                              screen_groups: int) -> D2STAT.ScreenDecision:
    """Adjudicate Gate TASK-X-DEV and Gate RISK-X-DEV exactly once (fail-closed, D2F states).

    The condition literals are D2 section 8.2/8.3 verbatim; c1 completeness is evaluated at the
    D2F ``selected_n``. Terminal-state priority (plan 3.5): integrity/denominator/projection/
    mask failure -> ``INCONCLUSIVE_D2F_EXTERNAL_INVALID``; else task failure ->
    ``D2F_EXTERNAL_DEV_TASK_NOT_READY``; else risk failure -> ``D2F_EXTERNAL_DEV_RISK_NOT_READY``;
    both pass -> ``D2F_DEV_READY_FOR_CONFIRM_POWER``.
    """
    groups = D2STAT._validate_quantities(quantities)
    point = np.asarray(bootstrap.point, dtype=np.float64)
    ci_lower = np.asarray(bootstrap.ci_lower, dtype=np.float64)
    ci_upper = np.asarray(bootstrap.ci_upper, dtype=np.float64)
    draws = np.asarray(bootstrap.draws, dtype=np.float64)
    if point.shape != (N_QUANTITIES,) or ci_lower.shape != (N_QUANTITIES,) \
            or ci_upper.shape != (N_QUANTITIES,) \
            or draws.shape != (int(bootstrap.n_resamples), N_QUANTITIES):
        raise ValueError("bootstrap result does not match the frozen quantity vector")
    if float(bootstrap.confidence) != float(D2STAT.CONFIDENCE_SCREEN):
        raise ValueError("the screen CI level is frozen at 0.80")

    qa = {key: bool(quantities["qa"][key]) for key in QA_KEYS}
    qa_passed = all(qa.values())

    Q = D2STAT
    baseline_valid = bool(
        np.isfinite([point[Q.Q_AURC_BASELINE], point[Q.Q_R80_BASELINE]]).all()
        and point[Q.Q_AURC_BASELINE] > 0.0 and point[Q.Q_R80_BASELINE] > 0.0)
    projection_ok = bool(quantities["projection_nondegenerate"])
    mask_ok = bool(quantities["eval_mask_nondegenerate"])

    # --- Gate TASK-X-DEV (D2 plan 8.2 verbatim; c1 at the D2F selected_n) ----------------------
    oracle = np.asarray(quantities["a_star"], dtype=np.int64)
    oracle_shares = [float((oracle == a).mean()) for a in range(N_ACTIONS)]
    n_oracle = int(sum(s >= Q.GATE_TASK_MIN_ORACLE_FRAC for s in oracle_shares))
    reactive_frac = float(np.asarray(quantities["reactive"], dtype=bool).mean())
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)
    stratum_fracs = {
        "tstar_low": float((tstar == 0).mean()), "tstar_high": float((tstar == 1).mean()),
        "interaction_low": float((inter == 0).mean()),
        "interaction_high": float((inter == 1).mean()),
    }
    base_mean = float(point[Q.Q_BASE_MEAN_REGRET])
    idle_mean = float(point[Q.Q_IDLE_MEAN_REGRET])
    task_conditions = {
        "c1_completeness_and_qa": bool(qa_passed and groups == int(screen_groups)),
        "c2_high_regret_frac_ge_15pct": bool(
            point[Q.Q_HIGH_REGRET_FRAC] >= Q.GATE_TASK_MIN_HIGH_REGRET_FRAC),
        "c3_at_least_3_oracle_actions_ge_5pct": bool(
            n_oracle >= Q.GATE_TASK_MIN_DISTINCT_ORACLE),
        "c4_branch_spearman_ci_lower_positive": bool(ci_lower[Q.Q_SPEARMAN_BRANCH] > 0.0),
        "c5_base_beats_random_and_not_worse_than_idle": bool(
            point[Q.Q_BASE_VS_RANDOM] > 0.0 and ci_lower[Q.Q_BASE_VS_RANDOM] > 0.0
            and np.isfinite([base_mean, idle_mean]).all()
            and base_mean <= idle_mean + BASELINE_TIE_TOLERANCE),
        "c6_reactive_response_ge_20pct": bool(reactive_frac >= Q.GATE_TASK_MIN_REACTIVE_FRAC),
        "c7_strata_each_ge_10pct": bool(
            min(stratum_fracs.values()) >= Q.GATE_TASK_MIN_STRATUM_FRAC),
        "c8_denominators_projection_mask_valid": bool(
            baseline_valid and projection_ok and mask_ok),
    }
    task_passed = bool(all(task_conditions.values()))

    # --- Gate RISK-X-DEV (D2 plan 8.3 verbatim) --------------------------------------------------
    seeds_both = int(sum(
        point[Q.SEED_I_AURC_INDEX[i]] > 0.0 and point[Q.SEED_I_R80_INDEX[i]] > 0.0
        for i in range(N_MODEL_SEEDS)))
    risk_conditions = {
        "r1_i_aurc_regret": bool(point[Q.Q_I_AURC] >= Q.I_AURC_MIN
                                 and ci_lower[Q.Q_I_AURC] > 0.0),
        "r2_i_regret80": bool(point[Q.Q_I_REGRET80] >= Q.I_REGRET80_MIN
                              and ci_lower[Q.Q_I_REGRET80] > 0.0),
        "r3_pr_auc_minus_prevalence": bool(point[Q.Q_PR] > 0.0 and ci_lower[Q.Q_PR] > 0.0),
        "r4_seed_consistency_both_positive": bool(seeds_both >= Q.MIN_POSITIVE_SEEDS),
        "r5_baseline_denominators_valid": bool(baseline_valid),
        "r6_no_leak_no_baseline_swap_no_model_swap": bool(qa_passed),
    }
    risk_passed = bool(all(risk_conditions.values()))

    # --- terminal-state priority (plan 3.5) -------------------------------------------------------
    invalid = (not qa_passed) or (not baseline_valid) \
        or (not projection_ok) or (not mask_ok)
    if invalid:
        status = STATUS_INVALID
    elif not task_passed:
        status = STATUS_TASK_NOT_READY
    elif not risk_passed:
        status = STATUS_RISK_NOT_READY
    else:
        status = STATUS_DEV_READY

    # --- three standard baselines reported with point + 80% CI (D2 plan 8.3 note) -----------------
    baseline_curves: dict[str, Any] = {}
    for bi, name in enumerate(STANDARD_BASELINES):
        slot = Q.Q_BASELINE_CURVE + 3 * bi
        baseline_curves[name] = {
            "aurc_regret": float(point[slot]),
            "aurc_regret_ci80": [float(ci_lower[slot]), float(ci_upper[slot])],
            "regret_at_80": float(point[slot + 1]),
            "regret_at_80_ci80": [float(ci_lower[slot + 1]), float(ci_upper[slot + 1])],
            "pr_auc_minus_prevalence": float(point[slot + 2]),
            "pr_auc_minus_prevalence_ci80": [float(ci_lower[slot + 2]),
                                             float(ci_upper[slot + 2])],
            "role": ("strongest_standard_baseline"
                     if name == str(quantities["strongest_baseline"]) else "standard_baseline"),
        }
    screen_aurc = {name: float(point[Q.Q_BASELINE_CURVE + 3 * bi])
                   for bi, name in enumerate(STANDARD_BASELINES)}
    screen_strongest = min(screen_aurc, key=lambda k: screen_aurc[k])
    frozen_baseline = str(quantities["strongest_baseline"])
    baseline_disclosure = {
        "frozen_primary_baseline": frozen_baseline,
        "screen_strongest_standard_baseline": screen_strongest,
        "screen_strongest_differs_from_frozen": bool(screen_strongest != frozen_baseline),
        "note": ("only the validation-frozen baseline is adjudicated; a different screen "
                 "strongest baseline is disclosed but never substituted (D2 plan 8.3)"),
    }

    return D2STAT.ScreenDecision(
        plan_id=PLAN_ID,
        n_groups=int(groups),
        status=status,
        passed=bool(status == STATUS_DEV_READY),
        task={
            "conditions": task_conditions,
            "passed": task_passed,
            "high_regret_frac": float(point[Q.Q_HIGH_REGRET_FRAC]),
            "oracle_action_shares": oracle_shares,
            "oracle_actions_covered_ge_5pct": n_oracle,
            "reactive_frac": reactive_frac,
            "stratum_fracs": stratum_fracs,
            "stratum_counts": {k: int(round(v * groups)) for k, v in stratum_fracs.items()},
            "baseline_valid": baseline_valid,
            "projection_nondegenerate": projection_ok,
            "eval_mask_nondegenerate": mask_ok,
        },
        risk={
            "conditions": risk_conditions,
            "passed": risk_passed,
            "baseline_valid": baseline_valid,
            "baseline_denominators": {
                "aurc_regret": float(point[Q.Q_AURC_BASELINE]),
                "regret_at_80": float(point[Q.Q_R80_BASELINE]),
            },
            "seeds_both_directions_positive": seeds_both,
            "min_positive_seeds": int(Q.MIN_POSITIVE_SEEDS),
        },
        metrics={
            "point": D2STAT._named(point),
            "ci_lower": D2STAT._named(ci_lower),
            "ci_upper": D2STAT._named(ci_upper),
        },
        baselines={"curves": baseline_curves, "disclosure": baseline_disclosure},
        bootstrap={
            "seed": int(bootstrap.seed),
            "n_resamples": int(bootstrap.n_resamples),
            "confidence": float(bootstrap.confidence),
            "design": ("whole-group paired bootstrap; every metric of a draw recomputed from "
                       "the SAME resample index vector; two-sided 80% CI from the resample "
                       "quantiles"),
        },
    )


def adjudicate_d2f_screen(config: Mapping[str, Any] | None = None) -> D2STAT.ScreenDecision:
    """The single D2F screen adjudication (plan 3.5): assemble -> bootstrap -> gates -> report.

    Create-once: a second call with the formal report present is refused. Never reruns the
    screen, never changes the model, never starts the confirm.
    """
    paths = d2f_screen_artifact_paths(config)
    if paths["report"].exists():
        raise FileExistsError(
            f"a second D2F screen adjudication is refused; report exists: {paths['report']}")
    manifest = _read_json(paths["screen_manifest"])
    screen_groups = int(manifest["selected_n"])
    quantities_npz = assemble_d2f_screen_group_quantities(config, screen_groups=screen_groups)
    quantities = D2STAT.load_screen_group_quantities(quantities_npz, paths["quantities_receipt"])
    bootstrap = D2STAT.paired_group_bootstrap(
        quantities, seed=PROT.D2F_SCREEN_BOOTSTRAP_ENTROPY)
    decision = evaluate_d2f_screen_gates(quantities, bootstrap, screen_groups=screen_groups)
    report = decision.to_dict()
    report["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["quantities_receipt_sha256"] = sha256_file(paths["quantities_receipt"])
    write_once_json(paths["report"], report)
    return decision


# --- Phase C: the confirm one-shot chain (plan 4; D2 10.2-10.5 verbatim at 95% CI) --------------------

CONFIDENCE_CONFIRM = 0.95
STATUS_CONFIRMED = "D2F_META_DRIVE_EXTERNAL_CONFIRMED"
STATUS_NOT_CONFIRMED = "D2F_META_DRIVE_NOT_CONFIRMED"
CONFIRM_ROLE = PROT.CONFIRM_ROLE


def d2f_confirm_artifact_paths(config: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Every one-shot D2F confirm path (plan 4); the D2/D2E sources stay read-only."""
    config = config or {}
    run_dir = Path(config.get("run_dir", D2F_DIR))
    role_dir = run_dir / "roles" / CONFIRM_ROLE
    confirm_dir = run_dir / PROT.CONFIRM_DIR_NAME
    split_dir = confirm_dir / "confirm_split"
    final = json.loads((D2E_DIR / "development" / "final_selection.json")
                       .read_text(encoding="utf-8"))
    return {
        "run_dir": run_dir,
        "confirm_dir": confirm_dir,
        "role_dir": role_dir,
        "groups_npz": role_dir / "groups.npz",
        "groups_meta": role_dir / "groups_meta.json",
        "collection_receipt": role_dir / "collection_receipt.json",
        "split_dir": split_dir,
        "inference_inputs": split_dir / "inference_inputs.npz",
        "sealed_truth": split_dir / "sealed_truth.npz",
        "split_receipt": split_dir / "split_receipt.json",
        "scores_npz": confirm_dir / "confirm_scores.npz",
        "scores_receipt": confirm_dir / "confirm_scores_receipt.json",
        "determinism_receipt": confirm_dir / "confirm_determinism_receipt.json",
        "lock": confirm_dir / "confirmation_used.lock",
        "quantities_npz": confirm_dir / "confirm_group_quantities.npz",
        "quantities_receipt": confirm_dir / "confirm_group_quantities_receipt.json",
        "report": confirm_dir / "confirm_report.json",
        "power_report": confirm_dir / "power_report.json",
        "protocol_lock": run_dir / PROT.CONFIRM_LOCK_NAME,
        "confirm_manifest": run_dir / PROT.CONFIRM_MANIFEST_NAME,
        "scorer_parity_receipt": run_dir / "development" / "scorer_parity_receipt.json",
        # read-only frozen sources (never written by D2F):
        "d2e_final_selection": D2E_DIR / "development" / "final_selection.json",
        "d2e_validation_report": D2E_DIR / "development" / "validation_report.json",
        "d2e_riskfit_receipt": D2E_DIR / str(final["risk_run_dir"]) / "development" / "models"
                               / "fit_receipt.json",
        "d2_predictor_stack": D2_DIR / "development" / "base_models" / "predictor_stack.json",
    }


def paired_group_bootstrap_95(quantities: Mapping[str, Any], *,
                              seed: int,
                              n_resamples: int = D2STAT.BOOTSTRAP_RESAMPLES
                              ) -> D2STAT.BootstrapResult:
    """10,000-draw whole-group paired bootstrap with shared indices and two-sided 95% CIs.

    The confirm-side counterpart of the frozen D2 screen bootstrap (which is hardcoded to
    0.80 and never re-parameterised): the SAME vector recomputation from the SAME resample
    index per draw, with the (0.025, 0.975) quantile band (D2 10.3/10.4). The frozen D2 module
    is never modified; this confirm variant lives in the D2F namespace.
    """
    groups = D2STAT._validate_quantities(quantities)
    generator = np.random.default_rng(int(seed))
    identity = np.arange(groups, dtype=np.int64)
    point = D2STAT.compute_screen_quantity_vector(identity, quantities)
    draws = np.empty((int(n_resamples), N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = D2STAT.compute_screen_quantity_vector(index, quantities)
    alpha = (1.0 - CONFIDENCE_CONFIRM) / 2.0
    return D2STAT.BootstrapResult(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws,
        n_resamples=int(n_resamples),
        seed=int(seed),
        confidence=float(CONFIDENCE_CONFIRM),
    )


def evaluate_d2f_confirm_gates(quantities: Mapping[str, Any],
                               bootstrap: D2STAT.BootstrapResult,
                               *,
                               confirm_groups: int) -> D2STAT.ScreenDecision:
    """Adjudicate Gate TASK-X-CONFIRM and Gate RISK-X-CONFIRM exactly once (D2 10.3/10.4).

    The condition literals are D2 8.2/8.3 verbatim with every CI-based condition at the
    two-sided 95% CI (D2 10.3); c1 completeness is evaluated at the confirm ``selected_n``.
    Terminal-state priority (D2 10.3/10.5 + 11.3): integrity/denominator/projection/mask
    failure OR any task-gate failure -> ``INCONCLUSIVE_D2F_EXTERNAL_INVALID`` (a failed task
    gate is never recorded as a method falsification); else risk-gate failure ->
    ``D2F_META_DRIVE_NOT_CONFIRMED``; both pass -> ``D2F_META_DRIVE_EXTERNAL_CONFIRMED``.
    """
    groups = D2STAT._validate_quantities(quantities)
    point = np.asarray(bootstrap.point, dtype=np.float64)
    ci_lower = np.asarray(bootstrap.ci_lower, dtype=np.float64)
    ci_upper = np.asarray(bootstrap.ci_upper, dtype=np.float64)
    draws = np.asarray(bootstrap.draws, dtype=np.float64)
    if point.shape != (N_QUANTITIES,) or ci_lower.shape != (N_QUANTITIES,) \
            or ci_upper.shape != (N_QUANTITIES,) \
            or draws.shape != (int(bootstrap.n_resamples), N_QUANTITIES):
        raise ValueError("bootstrap result does not match the frozen quantity vector")
    if float(bootstrap.confidence) != CONFIDENCE_CONFIRM:
        raise ValueError("the confirm CI level is frozen at 0.95")

    qa = {key: bool(quantities["qa"][key]) for key in QA_KEYS}
    qa_passed = all(qa.values())

    Q = D2STAT
    baseline_valid = bool(
        np.isfinite([point[Q.Q_AURC_BASELINE], point[Q.Q_R80_BASELINE]]).all()
        and point[Q.Q_AURC_BASELINE] > 0.0 and point[Q.Q_R80_BASELINE] > 0.0)
    projection_ok = bool(quantities["projection_nondegenerate"])
    mask_ok = bool(quantities["eval_mask_nondegenerate"])

    # --- Gate TASK-X-CONFIRM (D2 10.3: 8.2 verbatim, CI conditions at 95%, c1 at selected_n) ---
    oracle = np.asarray(quantities["a_star"], dtype=np.int64)
    oracle_shares = [float((oracle == a).mean()) for a in range(N_ACTIONS)]
    n_oracle = int(sum(s >= Q.GATE_TASK_MIN_ORACLE_FRAC for s in oracle_shares))
    reactive_frac = float(np.asarray(quantities["reactive"], dtype=bool).mean())
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)
    stratum_fracs = {
        "tstar_low": float((tstar == 0).mean()), "tstar_high": float((tstar == 1).mean()),
        "interaction_low": float((inter == 0).mean()),
        "interaction_high": float((inter == 1).mean()),
    }
    base_mean = float(point[Q.Q_BASE_MEAN_REGRET])
    idle_mean = float(point[Q.Q_IDLE_MEAN_REGRET])
    task_conditions = {
        "c1_completeness_and_qa": bool(qa_passed and groups == int(confirm_groups)),
        "c2_high_regret_frac_ge_15pct": bool(
            point[Q.Q_HIGH_REGRET_FRAC] >= Q.GATE_TASK_MIN_HIGH_REGRET_FRAC),
        "c3_at_least_3_oracle_actions_ge_5pct": bool(
            n_oracle >= Q.GATE_TASK_MIN_DISTINCT_ORACLE),
        "c4_branch_spearman_ci_lower_positive": bool(ci_lower[Q.Q_SPEARMAN_BRANCH] > 0.0),
        "c5_base_beats_random_and_not_worse_than_idle": bool(
            point[Q.Q_BASE_VS_RANDOM] > 0.0 and ci_lower[Q.Q_BASE_VS_RANDOM] > 0.0
            and np.isfinite([base_mean, idle_mean]).all()
            and base_mean <= idle_mean + BASELINE_TIE_TOLERANCE),
        "c6_reactive_response_ge_20pct": bool(reactive_frac >= Q.GATE_TASK_MIN_REACTIVE_FRAC),
        "c7_strata_each_ge_10pct": bool(
            min(stratum_fracs.values()) >= Q.GATE_TASK_MIN_STRATUM_FRAC),
        "c8_denominators_projection_mask_valid": bool(
            baseline_valid and projection_ok and mask_ok),
    }
    task_passed = bool(all(task_conditions.values()))

    # --- Gate RISK-X-CONFIRM (D2 10.4 verbatim, two-sided 95% CI) -------------------------------
    seeds_both = int(sum(
        point[Q.SEED_I_AURC_INDEX[i]] > 0.0 and point[Q.SEED_I_R80_INDEX[i]] > 0.0
        for i in range(N_MODEL_SEEDS)))
    risk_conditions = {
        "r1_i_aurc_regret": bool(point[Q.Q_I_AURC] >= Q.I_AURC_MIN
                                 and ci_lower[Q.Q_I_AURC] > 0.0),
        "r2_i_regret80": bool(point[Q.Q_I_REGRET80] >= Q.I_REGRET80_MIN
                              and ci_lower[Q.Q_I_REGRET80] > 0.0),
        "r3_pr_auc_minus_prevalence": bool(point[Q.Q_PR] > 0.0 and ci_lower[Q.Q_PR] > 0.0),
        "r4_seed_consistency_both_positive": bool(seeds_both >= Q.MIN_POSITIVE_SEEDS),
        "r5_baseline_denominators_valid": bool(baseline_valid),
        "r6_no_leak_no_baseline_swap_no_model_swap": bool(qa_passed),
    }
    risk_passed = bool(all(risk_conditions.values()))

    # --- terminal-state priority (D2 10.3/10.5/11.3; plan 4) ------------------------------------
    invalid = (not qa_passed) or (not baseline_valid) \
        or (not projection_ok) or (not mask_ok)
    if invalid or not task_passed:
        status = STATUS_INVALID
    elif not risk_passed:
        status = STATUS_NOT_CONFIRMED
    else:
        status = STATUS_CONFIRMED

    # --- three standard baselines reported with point + 95% CI (D2 10.4) ------------------------
    baseline_curves: dict[str, Any] = {}
    for bi, name in enumerate(STANDARD_BASELINES):
        slot = Q.Q_BASELINE_CURVE + 3 * bi
        baseline_curves[name] = {
            "aurc_regret": float(point[slot]),
            "aurc_regret_ci95": [float(ci_lower[slot]), float(ci_upper[slot])],
            "regret_at_80": float(point[slot + 1]),
            "regret_at_80_ci95": [float(ci_lower[slot + 1]), float(ci_upper[slot + 1])],
            "pr_auc_minus_prevalence": float(point[slot + 2]),
            "pr_auc_minus_prevalence_ci95": [float(ci_lower[slot + 2]),
                                             float(ci_upper[slot + 2])],
            "role": ("strongest_standard_baseline"
                     if name == str(quantities["strongest_baseline"]) else "standard_baseline"),
        }
    confirm_aurc = {name: float(point[Q.Q_BASELINE_CURVE + 3 * bi])
                    for bi, name in enumerate(STANDARD_BASELINES)}
    confirm_strongest = min(confirm_aurc, key=lambda k: confirm_aurc[k])
    frozen_baseline = str(quantities["strongest_baseline"])
    model_aurc = float(point[Q.Q_AURC_MODEL])
    beats_confirm_strongest = bool(
        model_aurc < confirm_aurc[confirm_strongest]
        and float(ci_upper[Q.Q_AURC_MODEL]) < confirm_aurc[confirm_strongest])
    baseline_disclosure = {
        "frozen_primary_baseline": frozen_baseline,
        "confirm_strongest_standard_baseline": confirm_strongest,
        "confirm_strongest_differs_from_frozen": bool(confirm_strongest != frozen_baseline),
        "model_beats_confirm_strongest_baseline": beats_confirm_strongest,
        "note": ("only the validation-frozen baseline is adjudicated; 'better than the "
                 "confirm-strongest standard baseline' may be written ONLY when the "
                 "corresponding comparisons hold, otherwise the preregistered validation "
                 "baseline phrasing is used (D2 10.4)"),
    }

    return D2STAT.ScreenDecision(
        plan_id=PLAN_ID,
        n_groups=int(groups),
        status=status,
        passed=bool(status == STATUS_CONFIRMED),
        task={
            "conditions": task_conditions,
            "passed": task_passed,
            "high_regret_frac": float(point[Q.Q_HIGH_REGRET_FRAC]),
            "oracle_action_shares": oracle_shares,
            "oracle_actions_covered_ge_5pct": n_oracle,
            "reactive_frac": reactive_frac,
            "stratum_fracs": stratum_fracs,
            "stratum_counts": {k: int(round(v * groups)) for k, v in stratum_fracs.items()},
            "baseline_valid": baseline_valid,
            "projection_nondegenerate": projection_ok,
            "eval_mask_nondegenerate": mask_ok,
        },
        risk={
            "conditions": risk_conditions,
            "passed": risk_passed,
            "baseline_valid": baseline_valid,
            "baseline_denominators": {
                "aurc_regret": float(point[Q.Q_AURC_BASELINE]),
                "regret_at_80": float(point[Q.Q_R80_BASELINE]),
            },
            "seeds_both_directions_positive": seeds_both,
            "min_positive_seeds": int(Q.MIN_POSITIVE_SEEDS),
        },
        metrics={
            "point": D2STAT._named(point),
            "ci_lower": D2STAT._named(ci_lower),
            "ci_upper": D2STAT._named(ci_upper),
        },
        baselines={"curves": baseline_curves, "disclosure": baseline_disclosure},
        bootstrap={
            "seed": int(bootstrap.seed),
            "n_resamples": int(bootstrap.n_resamples),
            "confidence": float(bootstrap.confidence),
            "design": ("whole-group paired bootstrap; every metric of a draw recomputed from "
                       "the SAME resample index vector; two-sided 95% CI from the resample "
                       "quantiles"),
        },
    )


# --- confirm one-shot chain: freeze scores / burn lock / assemble / adjudicate ------------------------


def freeze_d2f_confirm_scores(config: Mapping[str, Any] | None,
                              *,
                              scores: Mapping[str, Any],
                              determinism: Mapping[str, Any]) -> Path:
    """Freeze the D2F confirm scores + receipts create-once (mirrors the screen contract).

    ``scores`` is computed from ``inference_inputs.npz`` ONLY; ``determinism.bitwise_identical``
    must be True. A rerun verifies hashes and never overwrites. Never touches the sealed truth.
    """
    paths = d2f_confirm_artifact_paths(config)
    selection = _load_frozen_d2e_selection(paths)
    npz_path = paths["scores_npz"]
    receipt_path = paths["scores_receipt"]
    det_path = paths["determinism_receipt"]
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if not npz_path.is_file() or sha256_file(npz_path) != receipt["scores_sha256"]:
            raise ValueError(f"D2F confirm scores drifted from their create-once receipt: "
                             f"{npz_path}")
        if not det_path.is_file():
            raise ValueError("D2F confirm determinism receipt missing after score freeze")
        return npz_path
    if npz_path.exists():
        raise RuntimeError(f"D2F confirm scores exist without their create-once receipt: "
                           f"{npz_path}")

    group_seed = np.asarray(scores["group_seed"], dtype=np.int64).ravel()
    t_star = np.asarray(scores["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(scores["interaction_min_distance"], dtype=np.float64).ravel()
    j_hat = np.asarray(scores["j_hat"], dtype=np.float64)
    ensemble = np.asarray(scores["ensemble_risk"], dtype=np.float64).ravel()
    by_seed = np.asarray(scores["risk_by_seed"], dtype=np.float64)
    valid = np.asarray(scores["valid_mask"], dtype=bool).ravel()
    groups = int(group_seed.size)
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} D2F confirm groups have non-finite features; the confirm "
            "contract rejects zero-padding, so scoring is refused")
    if j_hat.shape != (groups, N_ACTIONS) or ensemble.shape != (groups,) \
            or by_seed.shape != (N_MODEL_SEEDS, groups) \
            or distance.shape != (groups,) or t_star.shape != (groups,):
        raise ValueError("D2F confirm score arrays must align on the group axis")
    if not np.isfinite(j_hat).all() or not np.isfinite(ensemble).all() \
            or not np.isfinite(by_seed).all():
        raise ValueError("non-finite D2F confirm score")
    if not np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15):
        raise ValueError("the ensemble must be exactly the arithmetic mean of five seeds")
    if not bool(determinism.get("bitwise_identical")):
        raise ValueError("the repeated inference was not elementwise identical; the D2F "
                         "confirm halts before the score freeze")

    payload = {
        "group_seed": group_seed,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "j_hat": j_hat,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": np.asarray(scores["mu_only_ensemble"], dtype=np.float64).ravel(),
        "class_only_ensemble": np.asarray(scores["class_only_ensemble"], dtype=np.float64).ravel(),
    }
    baselines = scores["baselines"]
    for name in ALL_BASELINE_NAMES:
        risk = np.asarray(baselines[name], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"D2F confirm baseline risk {name!r} must be finite and aligned")
        payload[_BASELINE_NPZ_PREFIX + name] = risk

    atomic_save_npz(npz_path, **payload)

    det_receipt = dict(determinism)
    det_receipt.setdefault("plan_id", PLAN_ID)
    det_receipt["groups_sha256"] = sha256_file(paths["groups_npz"]) \
        if paths["groups_npz"].is_file() else ""
    det_receipt["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_once_json(det_path, det_receipt)

    receipt = {
        "plan_id": PLAN_ID,
        "role": CONFIRM_ROLE,
        "n_groups": int(groups),
        "scores_path": str(npz_path),
        "scores_sha256": sha256_file(npz_path),
        "inference_inputs_sha256": sha256_file(paths["inference_inputs"])
        if paths["inference_inputs"].is_file() else "",
        "split_receipt_sha256": sha256_file(paths["split_receipt"])
        if paths["split_receipt"].is_file() else "",
        "d2e_final_selection_sha256": sha256_file(paths["d2e_final_selection"]),
        "d2e_riskfit_receipt_sha256": sha256_file(paths["d2e_riskfit_receipt"]),
        "strongest_standard_baseline": selection["strongest_standard_baseline"],
        "selected_ego_candidate": selection["selected_candidate"],
        "model_seeds": [int(s) for s in T.MODEL_SEEDS],
        "ensemble_definition": "arithmetic_mean_of_five_seeds",
        "determinism_receipt_sha256": sha256_file(det_path),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return npz_path


def burn_d2f_confirm_lock(config: Mapping[str, Any] | None) -> Path:
    """Create the D2F ``confirmation_used.lock`` exactly once, after the full score freeze.

    Same one-shot semantics as the screen burn: refuses unless the D2E selection qualifies,
    the scores hash-match their receipt and the determinism receipt is elementwise identical.
    The lock NEVER reads the sealed truth bytes.
    """
    paths = d2f_confirm_artifact_paths(config)
    selection = _load_frozen_d2e_selection(paths)
    receipt = _verify_scores_chain(paths)
    bindings = {
        "scores_sha256": receipt["scores_sha256"],
        "scores_receipt_sha256": sha256_file(paths["scores_receipt"]),
        "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
        "d2e_final_selection_sha256": sha256_file(paths["d2e_final_selection"]),
        "d2e_riskfit_receipt_sha256": sha256_file(paths["d2e_riskfit_receipt"]),
        "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
    }
    lock_path = paths["lock"]
    if lock_path.exists():
        payload = _read_json(lock_path)
        if payload.get("plan_id") != PLAN_ID or payload.get("bindings") != bindings:
            raise ValueError(
                "the burned D2F confirm lock no longer matches the on-disk frozen artifacts "
                "(tampering or drift); the one-shot confirm is compromised")
        return lock_path
    payload = {
        "plan_id": PLAN_ID,
        "role": CONFIRM_ROLE,
        "semantics": ("one-shot: the confirm truth may be read only after this lock exists; "
                      "all confirm scores are frozen and hash-bound above"),
        "strongest_standard_baseline": selection["strongest_standard_baseline"],
        "selected_ego_candidate": selection["selected_candidate"],
        "interaction_threshold": float(selection["interaction_threshold"]),
        "bindings": bindings,
        "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path, payload)
    return lock_path


def assert_d2f_confirm_adjudication_open(config: Mapping[str, Any] | None) -> None:
    """Fail closed unless the one-shot D2F confirm adjudication may legally start."""
    paths = d2f_confirm_artifact_paths(config)
    _load_frozen_d2e_selection(paths)
    if not paths["lock"].is_file():
        raise ValueError(
            "D2F confirm truth may not be read: confirmation_used.lock is not burned; freeze "
            "every score artifact and burn the lock first (plan 4 / D2 10.2)")
    for key in ("quantities_npz", "quantities_receipt", "report"):
        if paths[key].exists():
            raise FileExistsError(
                f"a second D2F confirm adjudication is refused; already exists: {paths[key]}")


def assemble_d2f_confirm_group_quantities(config: Mapping[str, Any] | None,
                                          *,
                                          confirm_groups: int) -> Path:
    """Open the sealed confirm truth ONCE, build regret/strata/reactive, write create-once.

    Mirrors the screen assembly bitwise except the completeness QA evaluates
    ``confirm_groups`` (the confirm ``selected_n``) and the receipt records the 95% confirm
    bootstrap contract.
    """
    paths = d2f_confirm_artifact_paths(config)
    assert_d2f_confirm_adjudication_open(config)
    selection = _load_frozen_d2e_selection(paths)
    score_receipt = _verify_scores_chain(paths)
    burn_d2f_confirm_lock(config)  # re-verify the lock bindings against the current files
    lock_payload = _read_json(paths["lock"])

    with np.load(paths["scores_npz"], allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    if not paths["sealed_truth"].is_file():
        raise ValueError(f"sealed truth missing: {paths['sealed_truth']}")
    with np.load(paths["sealed_truth"], allow_pickle=False) as archive:
        truth = {key: archive[key] for key in archive.files}

    group_seed = np.asarray(scores["group_seed"], dtype=np.int64).ravel()
    truth_seed = np.asarray(truth["group_seed"], dtype=np.int64).ravel()
    groups = int(group_seed.size)

    # --- truth side: legal ONLY here, after the burned lock -----------------------------------
    cf = np.asarray(truth["object_counterfactual_futures"], dtype=np.float64)
    factual = np.asarray(truth["object_factual_future"], dtype=np.float64)
    evaluation_mask = np.asarray(truth["evaluation_mask"], dtype=bool)
    history_last_xy = np.asarray(truth["history_last_xy"], dtype=np.float64)
    j_hat = np.asarray(scores["j_hat"], dtype=np.float64)
    labels = D2STAT.compute_screen_truth(cf, factual, history_last_xy, evaluation_mask, j_hat)
    j_true = labels["j_true"]

    threshold = float(selection["interaction_threshold"])
    t_star = np.asarray(scores["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(scores["interaction_min_distance"], dtype=np.float64).ravel()
    tstar_layer = (t_star >= TSTAR_SPLIT).astype(np.int64)
    interaction_layer = (distance > threshold).astype(np.int64)

    ensemble = np.asarray(scores["ensemble_risk"], dtype=np.float64).ravel()
    by_seed = np.asarray(scores["risk_by_seed"], dtype=np.float64)

    truth_finite = bool(np.isfinite(j_true).all() and np.isfinite(labels["regret"]).all())
    scenario_ids = json.loads(np.asarray(truth["scenario_ids_json"]).item()) \
        if "scenario_ids_json" in truth else [str(s) for s in group_seed]
    role_seeds_valid = bool(
        group_seed.size == groups and np.unique(group_seed).size == groups
        and np.array_equal(group_seed, truth_seed)
        and len(set(scenario_ids)) == len(scenario_ids))
    valid_cells = evaluation_mask                                # (G,20,5)
    cf_xy = cf[..., [CH_S, CH_D]]                               # (G,10,20,5,2)
    if valid_cells.any():
        mask_b = np.broadcast_to(valid_cells[:, None, :, :], cf_xy.shape[:-1])
        pts = cf_xy[mask_b]
        span = float((pts.max(axis=0) - pts.min(axis=0)).max()) if pts.size else 0.0
    else:
        span = 0.0
    projection_nondegenerate = bool(np.isfinite(span) and span > 1e-6)
    per_group_cells = valid_cells.reshape(groups, -1).any(axis=1) \
        if valid_cells.shape[0] == groups else np.zeros(groups, dtype=bool)
    eval_mask_nondegenerate = bool(valid_cells.size > 0 and valid_cells.mean() > 0.0
                                   and per_group_cells.all())

    qa = {
        "all_complete": bool(
            groups == int(confirm_groups) and j_true.shape == (groups, N_ACTIONS)
            and j_hat.shape == (groups, N_ACTIONS) and truth_finite
            and int(score_receipt["n_groups"]) == groups),
        "hashes_match": True,  # _verify_scores_chain + lock re-verification passed above
        "role_seeds_valid": role_seeds_valid,
        "determinism_passed": True,  # burn re-verified the determinism receipt
        "one_shot_locks": True,
        "ensemble_is_arithmetic_mean": bool(
            by_seed.shape == (N_MODEL_SEEDS, groups)
            and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)),
        "baseline_matches_frozen": bool(
            selection["strongest_standard_baseline"] in STANDARD_BASELINES),
        "checkpoints_match_frozen": _verify_checkpoint_pairing(selection),
    }

    from ac_twin import stage4cp_statistics as s4cs
    trim_keep = {f"trim_keep_{str(frac).replace('.', '_')}": s4cs.trim_high_regret(
        labels["regret"], frac) for frac in TRIM_FRACTIONS}

    npz_path = paths["quantities_npz"]
    payload = {
        "group_seed": group_seed,
        "regret": labels["regret"],
        "y_high": labels["y_high"],
        "a_hat": labels["a_hat"].astype(np.int64),
        "a_star": labels["a_star"].astype(np.int64),
        "random_regret": labels["random_regret"],
        "idle_regret": labels["idle_regret"],
        "J_hat": j_hat,
        "J_true": j_true,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": np.asarray(scores["mu_only_ensemble"], dtype=np.float64).ravel(),
        "class_only_ensemble": np.asarray(scores["class_only_ensemble"], dtype=np.float64).ravel(),
        "tstar_layer": tstar_layer,
        "interaction_layer": interaction_layer,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "reactive": labels["reactive"].astype(bool),
        "scenario_ids_json": np.asarray(json.dumps([str(s) for s in scenario_ids])),
        "map_family": np.zeros(groups, dtype=np.int64),  # single Waymo training_20s family
        "n_groups": np.int64(groups),
        "interaction_threshold": np.float64(threshold),
        **trim_keep,
    }
    for name in ALL_BASELINE_NAMES:
        payload[_BASELINE_NPZ_PREFIX + name] = np.asarray(
            scores[_BASELINE_NPZ_PREFIX + name], dtype=np.float64).ravel()
    atomic_save_npz(npz_path, **payload)

    receipt = {
        "plan_id": PLAN_ID,
        "role": CONFIRM_ROLE,
        "n_groups": int(groups),
        "confirm_group_quantities_path": str(npz_path),
        "confirm_group_quantities_sha256": sha256_file(npz_path),
        "strongest_standard_baseline": selection["strongest_standard_baseline"],
        "selected_ego_candidate": selection["selected_candidate"],
        "interaction_threshold": float(threshold),
        "interaction_threshold_source": "d2e riskfit fit_receipt.json (frozen)",
        "tstar_stratum_split": TSTAR_SPLIT,
        "trim_fractions": [float(f) for f in TRIM_FRACTIONS],
        "map_family_note": ("the frozen collector records no finer map_id; the confirm is one "
                            "Waymo training_20s -> MetaDrive family (map_family all 0)."),
        "bootstrap_entropy": PROT.D2F_CONFIRM_BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": D2STAT.BOOTSTRAP_RESAMPLES,
        "confidence_confirm": CONFIDENCE_CONFIRM,
        "qa": qa,
        "projection_nondegenerate": projection_nondegenerate,
        "eval_mask_nondegenerate": eval_mask_nondegenerate,
        "bindings": {
            "scores_sha256": score_receipt["scores_sha256"],
            "scores_receipt_sha256": sha256_file(paths["scores_receipt"]),
            "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
            "sealed_truth_sha256": sha256_file(paths["sealed_truth"]),
            "inference_inputs_sha256": sha256_file(paths["inference_inputs"])
            if paths["inference_inputs"].is_file() else "",
            "split_receipt_sha256": sha256_file(paths["split_receipt"])
            if paths["split_receipt"].is_file() else "",
            "d2e_final_selection_sha256": sha256_file(paths["d2e_final_selection"]),
            "d2e_validation_report_sha256": sha256_file(paths["d2e_validation_report"]),
            "d2e_riskfit_receipt_sha256": sha256_file(paths["d2e_riskfit_receipt"]),
            "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
            "statistics_code_sha256": sha256_file(
                ROOT / "code/ac_twin/stage4d_d2f_statistics.py"),
        },
        "lock_sha256": sha256_file(paths["lock"]),
        "lock_burned_at": lock_payload.get("burned_at", ""),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["quantities_receipt"], receipt)
    return npz_path


def adjudicate_d2f_confirm(config: Mapping[str, Any] | None = None) -> D2STAT.ScreenDecision:
    """The single D2F confirm adjudication (plan 4): assemble -> bootstrap 95% -> gates -> report.

    Create-once: a second call with the formal report present is refused. Never reruns the
    confirm, never changes the model, never re-adjudicates any earlier terminal state.
    """
    paths = d2f_confirm_artifact_paths(config)
    if paths["report"].exists():
        raise FileExistsError(
            f"a second D2F confirm adjudication is refused; report exists: {paths['report']}")
    manifest = _read_json(paths["confirm_manifest"])
    confirm_groups = int(manifest["selected_n"])
    quantities_npz = assemble_d2f_confirm_group_quantities(config, confirm_groups=confirm_groups)
    quantities = D2STAT.load_screen_group_quantities(quantities_npz, paths["quantities_receipt"])
    bootstrap = paired_group_bootstrap_95(
        quantities, seed=PROT.D2F_CONFIRM_BOOTSTRAP_ENTROPY)
    decision = evaluate_d2f_confirm_gates(quantities, bootstrap, confirm_groups=confirm_groups)
    report = decision.to_dict()
    report["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["quantities_receipt_sha256"] = sha256_file(paths["quantities_receipt"])
    write_once_json(paths["report"], report)
    return decision


__all__ = [
    "ALL_BASELINE_NAMES",
    "CONFIDENCE_CONFIRM",
    "CONFIRM_ROLE",
    "PLAN_ID",
    "SCREEN_ROLE",
    "STANDARD_BASELINES",
    "STATUS_CONFIRMED",
    "STATUS_DEV_READY",
    "STATUS_INVALID",
    "STATUS_NOT_CONFIRMED",
    "STATUS_RISK_NOT_READY",
    "STATUS_TASK_NOT_READY",
    "adjudicate_d2f_confirm",
    "adjudicate_d2f_screen",
    "assemble_d2f_confirm_group_quantities",
    "assemble_d2f_screen_group_quantities",
    "assert_d2f_confirm_adjudication_open",
    "assert_d2f_screen_adjudication_open",
    "burn_d2f_confirm_lock",
    "burn_d2f_screen_lock",
    "d2f_confirm_artifact_paths",
    "d2f_screen_artifact_paths",
    "evaluate_d2f_confirm_gates",
    "evaluate_d2f_screen_gates",
    "freeze_d2f_confirm_scores",
    "freeze_d2f_screen_scores",
    "paired_group_bootstrap_95",
]
