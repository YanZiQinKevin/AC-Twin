"""Stage 3A Plus driver: diagnostics, calibration screen, dev rounds, freeze,
qualification lock/inference and the single Gate R+ adjudication
(plan `AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1`).

Subcommands (dev phase; long tasks run under the plan's autonomous-authorisation
exception with a 30s plain-text heartbeat):

- ``diagnose``           ten-seed calibration diagnostic on dev A (old merge
                         split 2) and dev B (replacement holdout); fits the four
                         frozen calibration candidates on validation only and
                         emits the mechanical calibration-failure status.
- ``calibration-screen`` Round 0: select the unique calibration candidate by the
                         frozen ``J_cal`` score (validation only).
- ``evaluate-dev``       Gate R+_dev on dev A and dev B for a given round.
- ``train-round``        Round 1/2 bounded retraining (only when invoked).
- ``freeze-final``       freeze the final model+calibration+gate receipt.
- ``run-all``            the autonomous orchestrator (resumable state machine,
                         dev rounds through the single qualification gate).
- ``lock-qualification`` bind the collected qualification set to the freeze
                         receipt + power hash chain (single-use lock).
- ``qualification-infer`` frozen ten-seed inference on the locked set.
- ``gate-r-plus-qualification`` the SINGLE qualification adjudication (10,000
                         bootstraps, seed 20260929; burns qualification_used.lock
                         before reading the truth).

This module reuses the Stage 3A repair driver's loaders/scaler/batching/
baselines/inference so the data path is identical to the adjudicated pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_stage3a as s3a  # noqa: E402
from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin import stage3a_plus_calibration as cal  # noqa: E402
from ac_twin import stage3a_plus_statistics as sps  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage3_data import (  # noqa: E402
    STAGE3A_PLUS_ALL_SEEDS,
    STAGE3A_PLUS_DEV_SEEDS,
    STAGE3A_PLUS_PLAN_ID,
    STAGE3A_PLUS_QUALIFICATION_PURPOSE,
    STAGE3A_PLUS_STRESS_SEEDS,
    assert_stage3a_plus_source_role,
    compute_effect_eval_mask,
    compute_regime_labels,
    compute_response_pos,
)
from ac_twin.stage3_response import (  # noqa: E402
    SMOOTH_L1_BETA,
    Stage3ResponsePredictor,
    compute_class_weights,
    expected_calibration_error,
    macro_brier,
    make_initial_distance,
    multiclass_brier_loss,
    response_masked_loss,
    softmax,
)
from ac_twin.stage3_statistics import make_brier_difference_statistic  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_dev_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if config.get("plan_id") != STAGE3A_PLUS_PLAN_ID:
        raise ValueError(f"dev config plan_id must be {STAGE3A_PLUS_PLAN_ID!r}")
    if tuple(config.get("dev_seeds", ())) != STAGE3A_PLUS_DEV_SEEDS:
        raise ValueError(f"dev_seeds must be {STAGE3A_PLUS_DEV_SEEDS}")
    if tuple(config.get("stress_seeds", ())) != STAGE3A_PLUS_STRESS_SEEDS:
        raise ValueError(f"stress_seeds must be {STAGE3A_PLUS_STRESS_SEEDS}")
    for key in ("development_source", "replacement_source", "prior_run_dir",
                "final_config", "training", "device"):
        if key not in config:
            raise ValueError(f"dev config missing key: {key}")
    config["_config_path"] = str(_resolve(path))
    return config


# --------------------------------------------------------------------------- #
# Eval-set preparation (dev A = old merge split 2; dev B = replacement holdout)
# --------------------------------------------------------------------------- #
def _train_prevalence(tensors, roles) -> np.ndarray:
    train = roles["train"]
    labels = tensors["regime_labels"][train].numpy()
    valid = tensors["valid"][train].numpy()
    counts = np.bincount(labels[valid].reshape(-1), minlength=4).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else np.full(4, 0.25)


def _strongest_basenames(prior_run_dir: Path) -> dict[str, str]:
    sel = json.loads((prior_run_dir / "selection" / "final_selection.json").read_text(encoding="utf-8"))
    return dict(sel["strongest_baselines"])


def _build_eval_baselines(dev_tensors, dev_raw_suffix, dev_roles, target_tensors,
                          target_raw_suffix, target_indices, strongest) -> dict[str, Any]:
    """Fit baselines on dev train, predict on the target; package for decompose."""
    cont, clf = s3a._build_baselines(
        dev_tensors, dev_raw_suffix, dev_roles["train"],
        target_tensors, target_raw_suffix, target_indices,
    )
    return {
        "continuous": cont,
        "classification": clf,
        "strongest_continuous": strongest["strongest_continuous"],
        "strongest_brier": strongest["strongest_brier"],
    }


def _horizon_grid(valid_mask: np.ndarray) -> np.ndarray:
    E, T, N = valid_mask.shape
    return np.broadcast_to(np.arange(T)[None, :, None], (E, T, N)).copy()


# --------------------------------------------------------------------------- #
# Pure diagnostic core (no torch) — plan §3
# --------------------------------------------------------------------------- #
def run_diagnostic(
    *,
    val_logits_seeds: np.ndarray,
    val_labels: np.ndarray,
    val_mask: np.ndarray,
    train_prevalence: np.ndarray,
    eval_sets: Mapping[str, Mapping[str, Any]],
    engineering_ok: bool,
    n_production_seeds: int,
    n_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Fit the four calibration candidates on validation and decompose each on
    dev sets A and B, then mechanically classify the calibration failure.

    ``eval_sets`` maps "A"/"B" to a dict of per-seed predictions + truth +
    baselines. Returns the full diagnostic report and the §3.2 status.
    """
    candidates: dict[str, Any] = {}
    for cid in cal.CALIBRATION_CANDIDATES:
        artifact = cal.fit_calibration_candidate(
            cid, val_logits_seeds, val_labels, val_mask, train_prevalence,
            n_production_seeds=n_production_seeds,
        )
        per_set = {}
        for name, es in eval_sets.items():
            cal_probs = cal.apply_calibrator(artifact, es["logits_seeds"], train_prevalence)
            per_set[name] = sps.decompose_seed_metrics(
                r_hat_seeds=es["r_hat_seeds"],
                logits_seeds=es["logits_seeds"],
                calibrated_probs_seeds=cal_probs,
                r_true=es["r_true"], regime_labels=es["regime_labels"],
                valid_mask=es["valid_mask"], episode_ids=es["episode_ids"],
                horizons=es["horizons"], cf_actions=es["cf_actions"],
                cf_durations=es["cf_durations"], baselines=es["baselines"],
                n_resamples=n_resamples, bootstrap_seed=bootstrap_seed,
                expected_episode_ids=es.get("expected_episode_ids"),
                n_production_seeds=n_production_seeds,
            )
        candidates[cid] = {
            "artifact": artifact,
            "calibration_signature": cal.calibration_signature(artifact),
            "validation_nll": _ensemble_val_nll(artifact, val_logits_seeds, val_labels, val_mask,
                                                train_prevalence, n_production_seeds),
            "per_set": per_set,
        }

    classification_input = _classification_input(candidates, eval_sets, engineering_ok,
                                                 n_production_seeds)
    status = sps.classify_calibration_failure(classification_input)
    return {
        "candidates": candidates,
        "classification_input": classification_input,
        "calibration_failure_status": status,
    }


def _ensemble_val_nll(artifact, val_logits_seeds, val_labels, val_mask,
                      train_prevalence, n_prod) -> float:
    # Apply to ALL seeds (per-seed calibrators need the full leading axis), then
    # average the production-seed probabilities for the core ensemble (§5.1).
    probs = cal.apply_calibrator(artifact, val_logits_seeds, train_prevalence)
    ensemble = probs[:n_prod].mean(axis=0)
    return sps.multiclass_nll(ensemble, val_labels, val_mask)


def _core_metric_vector(report) -> np.ndarray:
    """The ensemble's five core metric point estimates (positive = better)."""
    ens = report["ensemble"]
    return np.array([
        ens["i_mae"]["point"], ens["spearman_boot"]["point"],
        ens["i_pr_presence"]["point"], ens["i_pr_high_vs_lowmid"]["point"],
        ens["i_brier"]["point"],
    ], dtype=np.float64)


def _classification_input(candidates, eval_sets, engineering_ok, n_prod) -> dict[str, Any]:
    """Assemble the mechanical §3.2 classification inputs.

    The reference (status-quo) calibration is CAL_SHARED_T (the single shared
    temperature used by the original gate); per-seed candidates are
    CAL_PER_SEED_T and CAL_PER_SEED_VECTOR.
    """
    ref = candidates[cal.CAL_SHARED_T]["per_set"]
    seed303 = {
        s: {
            "i_brier_ci_lower": ref[s]["per_seed"]["303"]["i_brier_ci_lower"],
            "i_brier_ci_upper": ref[s]["per_seed"]["303"]["i_brier_ci_upper"],
        }
        for s in ("A", "B")
    }
    improves = {"A": False, "B": False}
    best_per_seed = None
    for cid in (cal.CAL_PER_SEED_T, cal.CAL_PER_SEED_VECTOR):
        cand = candidates[cid]["per_set"]
        imp = {
            s: (cand[s]["per_seed"]["303"]["i_brier"] > ref[s]["per_seed"]["303"]["i_brier"])
            for s in ("A", "B")
        }
        if imp["A"] and imp["B"]:
            best_per_seed = cid
            improves = {"A": True, "B": True}
            break
    ensemble_ok = False
    if best_per_seed is not None:
        cand = candidates[best_per_seed]["per_set"]
        ensemble_ok = all(
            np.all(_core_metric_vector(cand[s]) >= _core_metric_vector(ref[s]) - 1e-9)
            for s in ("A", "B")
        )
    max_conc = 0.0
    idx303 = list(STAGE3A_PLUS_ALL_SEEDS).index(303)
    for s in ("A", "B"):
        es = eval_sets[s]
        ref_artifact = candidates[cal.CAL_SHARED_T]["artifact"]
        cal_probs = cal.apply_calibrator(ref_artifact, es["logits_seeds"], None)
        conc = sps.negative_brier_subgroup_concentration(
            cal_probs[idx303],
            es["baselines"]["classification"][es["baselines"]["strongest_brier"]],
            es["regime_labels"], es["valid_mask"], es["horizons"],
            es["cf_actions"], es["cf_durations"],
        )["max_subgroup_concentration"]
        max_conc = max(max_conc, conc)
    return {
        "engineering_ok": bool(engineering_ok),
        "per_seed_cal_improves_303": improves,
        "ensemble_core_not_degraded_under_per_seed_cal": bool(ensemble_ok),
        "max_subgroup_concentration": float(max_conc),
        "seed303": seed303,
    }


# --------------------------------------------------------------------------- #
# Round 0 calibration screen (plan §4.1) — validation-only J_cal selection
# --------------------------------------------------------------------------- #
JCAL_SD_WEIGHT = 0.5
JCAL_ECE_WEIGHT = 2.0
JCAL_ECE_GUARD = 0.05
JCAL_TIE_TOLERANCE = 1e-8
CALIBRATION_SELECTION_RULE = (
    "min J_cal = mean(Brier_seed) + 0.5*SD(Brier_seed) + "
    "2*max(0, ECE_ensemble - 0.05) over the ten fixed seeds; ties within 1e-8 "
    "broken by complexity SHARED_T -> PER_SEED_T -> T_SHRINK -> VECTOR")


def calibration_j_cal(per_seed_brier, ensemble_ece: float) -> float:
    """The frozen Round 0 selection score (plan §4.1; lower is better)."""
    brier = np.asarray(per_seed_brier, dtype=np.float64)
    if brier.ndim != 1 or brier.size == 0 or not np.isfinite(brier).all():
        raise ValueError("per-seed Brier must be a finite 1-D array")
    ece = float(ensemble_ece)
    if not np.isfinite(ece):
        raise ValueError("ensemble ECE must be finite")
    return float(brier.mean() + JCAL_SD_WEIGHT * brier.std(ddof=0)
                 + JCAL_ECE_WEIGHT * max(0.0, ece - JCAL_ECE_GUARD))


def select_calibration_candidate(j_cals: Mapping[str, float],
                                 *, tie_tolerance: float = JCAL_TIE_TOLERANCE) -> str:
    """Unique lowest J_cal; ties within ``tie_tolerance`` go to the simplest
    candidate in the frozen complexity order (plan §4.1)."""
    best: tuple[str, float] | None = None
    for cid in cal.COMPLEXITY_ORDER:
        if cid not in j_cals:
            raise ValueError(f"missing J_cal for candidate {cid}")
        score = float(j_cals[cid])
        if not np.isfinite(score):
            continue
        if best is None or score < best[1] - tie_tolerance:
            best = (cid, score)
    if best is None:
        raise ValueError("no finite J_cal candidate to select")
    return best[0]


def validation_calibration_screen(
    *,
    val_logits_seeds: np.ndarray,
    val_labels: np.ndarray,
    val_mask: np.ndarray,
    train_prevalence: np.ndarray,
    n_production_seeds: int,
    candidates: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fit the frozen candidates on validation and score each with J_cal.

    Per-seed Brier uses all ten fixed seeds (the stability axis); the ensemble
    ECE uses the production-seed calibrated mean (plan §5.1). Dev sets A/B are
    never touched here. ``candidates`` restricts the fit set — rounds 1/2 refit
    only the Round-0-selected family on the new model's validation logits.
    """
    logits = np.asarray(val_logits_seeds, dtype=np.float64)
    if logits.shape[0] != len(STAGE3A_PLUS_ALL_SEEDS):
        raise ValueError("calibration screen needs all ten fixed seeds")
    if not np.isfinite(logits).all():
        raise ValueError("non-finite validation logits")
    fit_set = tuple(candidates) if candidates is not None else cal.CALIBRATION_CANDIDATES
    rows: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for cid in fit_set:
        artifact = cal.fit_calibration_candidate(
            cid, logits, val_labels, val_mask, train_prevalence,
            n_production_seeds=n_production_seeds)
        probs = cal.apply_calibrator(artifact, logits, train_prevalence)
        per_seed_brier = np.array(
            [macro_brier(probs[s], val_labels, val_mask) for s in range(logits.shape[0])],
            dtype=np.float64)
        ensemble = probs[:n_production_seeds].mean(axis=0)
        ece = expected_calibration_error(ensemble, val_labels, val_mask,
                                         n_bins=15, equal_mass=True)
        rows[cid] = {
            "j_cal": calibration_j_cal(per_seed_brier, ece),
            "mean_brier_seed": float(per_seed_brier.mean()),
            "sd_brier_seed": float(per_seed_brier.std(ddof=0)),
            "ece_ensemble": float(ece),
            "per_seed_brier": {
                str(seed): float(per_seed_brier[i])
                for i, seed in enumerate(STAGE3A_PLUS_ALL_SEEDS)},
            "calibration_signature": cal.calibration_signature(artifact),
            "n_fit": int(artifact.n_fit),
            "params": cal._to_jsonable(artifact.params),
        }
        artifacts[cid] = artifact
    if len(fit_set) == 1:
        selected = fit_set[0]  # family refit: no new selection (plan §4.2/§4.3)
    else:
        selected = select_calibration_candidate(
            {cid: r["j_cal"] for cid, r in rows.items()})
    sel = artifacts[selected]
    return {
        "rows": rows,
        "selected": selected,
        "selected_artifact": {
            "method": sel.method,
            "params": cal._to_jsonable(sel.params),
            "fit_split": sel.fit_split,
            "n_fit": int(sel.n_fit),
            "input_signature": sel.input_signature,
            "output_signature": sel.output_signature,
        },
        "n_production_seeds": int(n_production_seeds),
    }


def build_calibration_selection_receipt(
    screen: Mapping[str, Any],
    *,
    data_signature: str,
    validation_logits_signature: str,
) -> dict[str, Any]:
    """Deterministic, hash-chained selection receipt (single-use artifact)."""
    payload = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "calibration_selection",
        "round": 0,
        "fit_split": "validation",
        "selection_rule": CALIBRATION_SELECTION_RULE,
        "tie_tolerance": JCAL_TIE_TOLERANCE,
        "data_signature": str(data_signature),
        "validation_logits_signature": str(validation_logits_signature),
        "n_production_seeds": int(screen["n_production_seeds"]),
        "candidates": {cid: screen["rows"][cid] for cid in screen["rows"]},
        "selected_candidate": screen["selected"],
        "selected_artifact": screen["selected_artifact"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"receipt_payload": payload,
            "receipt_sha256": canonical_fingerprint(payload)}


def write_calibration_selection(path: Path, receipt: Mapping[str, Any]) -> None:
    """Single-use write: refuses to overwrite an existing selection receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, ensure_ascii=False)


def verify_calibration_selection(
    path: Path,
    screen: Mapping[str, Any],
    *,
    data_signature: str,
    validation_logits_signature: str,
) -> tuple[bool, list[tuple[str, bool]]]:
    """Re-verify a stored selection against a freshly recomputed screen.

    Checks receipt integrity (hash over the stored payload), provenance binding
    (plan id, data signature, validation-logits signature), per-candidate
    signature + J_cal equality, and that the stored selection matches the
    recomputed one. Never writes.
    """
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = stored.get("receipt_payload", {})
    checks: list[tuple[str, bool]] = []
    checks.append(("receipt_integrity",
                   stored.get("receipt_sha256") == canonical_fingerprint(payload)))
    checks.append(("plan_id", payload.get("plan_id") == STAGE3A_PLUS_PLAN_ID))
    checks.append(("artifact_type", payload.get("artifact_type") == "calibration_selection"))
    checks.append(("data_signature",
                   payload.get("data_signature") == str(data_signature)))
    checks.append(("validation_logits_signature",
                   payload.get("validation_logits_signature")
                   == str(validation_logits_signature)))
    stored_rows = payload.get("candidates", {})
    for cid, row in stored_rows.items():
        fresh = screen["rows"][cid]
        checks.append((f"signature:{cid}",
                       row.get("calibration_signature") == fresh["calibration_signature"]))
        checks.append((f"j_cal:{cid}",
                       abs(float(row.get("j_cal", float("nan"))) - fresh["j_cal"])
                       <= JCAL_TIE_TOLERANCE))
    checks.append(("selected_candidate",
                   payload.get("selected_candidate") == screen["selected"]))
    return all(ok for _, ok in checks), checks


def calibrator_from_selection_receipt(receipt: Mapping[str, Any]) -> "cal.CalibrationArtifact":
    """Rebuild the frozen calibrator from a selection receipt (self-checked)."""
    block = receipt["receipt_payload"]["selected_artifact"]
    artifact = cal.CalibrationArtifact(
        method=block["method"], params=block["params"],
        fit_split=block["fit_split"], n_fit=int(block["n_fit"]),
        input_signature=block["input_signature"],
        output_signature=block["output_signature"])
    expected = receipt["receipt_payload"]["candidates"][block["method"]]["calibration_signature"]
    if cal.calibration_signature(artifact) != expected:
        raise ValueError("selection receipt artifact failed signature self-check")
    return artifact


# --------------------------------------------------------------------------- #
# Gate R+ dev report assembly (plan §5) — pure numpy, no torch
# --------------------------------------------------------------------------- #
STABILITY_PER_SEED_KEYS = (
    "i_mae", "spearman", "i_pr_presence", "i_pr_high_vs_lowmid",
    "i_brier", "non_degenerate", "finite")


def per_seed_episode_i_brier(
    cal_probs_seeds: np.ndarray,
    baseline_probs: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """(S,E) per-seed per-episode I_Brier values for the hierarchical
    seed×episode bootstrap, using exactly the episode-clustered statistic
    factory (single-episode draws) so definitions cannot drift."""
    probs = np.asarray(cal_probs_seeds, dtype=np.float64)
    n_seeds, n_episodes = probs.shape[0], labels.shape[0]
    values = np.full((n_seeds, n_episodes), np.nan, dtype=np.float64)
    for s in range(n_seeds):
        stat = make_brier_difference_statistic(probs[s], baseline_probs, labels, valid_mask)
        for e in range(n_episodes):
            values[s, e] = stat(np.array([e]))
    if not np.isfinite(values).all():
        raise ValueError("non-finite per-episode I_Brier values (engineering failure)")
    return values


def assemble_gate_r_plus_report(
    decomp: Mapping[str, Any],
    *,
    cal_probs_seeds: np.ndarray,
    eval_set: Mapping[str, Any],
    n_resamples: int,
    bootstrap_seed: int,
    gate_q_passed: bool,
    gate_q_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Assemble one dev set's full Gate R+ report (core + stability + Q+) from
    a ``decompose_seed_metrics`` report, then evaluate the AND-gate. Every
    check lands in ``checks`` — nothing is skipped silently (plan §5.4)."""
    labels, valid = eval_set["regime_labels"], eval_set["valid_mask"]
    clf_base = np.asarray(
        eval_set["baselines"]["classification"][eval_set["baselines"]["strongest_brier"]],
        dtype=np.float64)
    values = per_seed_episode_i_brier(cal_probs_seeds, clf_base, labels, valid)
    hier = sps.hierarchical_seed_episode_bootstrap(values, int(n_resamples), int(bootstrap_seed))
    per_seed = {
        seed: {k: decomp["per_seed"][seed][k] for k in STABILITY_PER_SEED_KEYS}
        for seed in decomp["per_seed"]
    }
    report = {
        "core": decomp["ensemble"],
        "stability": {
            "per_seed": per_seed,
            "baseline_brier": macro_brier(clf_base, labels, valid),
            "hierarchical_i_brier": hier,
        },
        "gate_q_passed": bool(gate_q_passed),
        "gate_q_checks": {k: bool(v) for k, v in gate_q_checks.items()},
    }
    evaluation = sps.evaluate_gate_r_plus(report)
    return {"report": report, "evaluation": evaluation,
            "passed": bool(evaluation["passed"])}


# --------------------------------------------------------------------------- #
# Development state machine (plan Task 7 step 4)
# --------------------------------------------------------------------------- #
STATE_FILE_NAME = "stage3a_plus_state.json"
STATE_INITIAL = "STAGE3A_PLUS_DESIGN_FROZEN"
STATE_NOT_READY = "STAGE3A_PLUS_DEVELOPMENT_NOT_READY"
TERMINAL_STATES = frozenset({
    STATE_NOT_READY,
    "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS",
    "INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_STAGE3A_PLUS",
    "INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE",
})
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_INITIAL: frozenset({"SOFTWARE_TESTED"}),
    "SOFTWARE_TESTED": frozenset({"FORENSIC_DIAGNOSTIC_COMPLETE"}),
    "FORENSIC_DIAGNOSTIC_COMPLETE": frozenset({"DEV_ROUND_0_COMPLETE"}),
    "DEV_ROUND_0_COMPLETE": frozenset(
        {"STAGE3A_PLUS_DEVELOPMENT_STABLE", "DEV_ROUND_1_COMPLETE"}),
    "DEV_ROUND_1_COMPLETE": frozenset(
        {"STAGE3A_PLUS_DEVELOPMENT_STABLE", "DEV_ROUND_2_COMPLETE"}),
    "DEV_ROUND_2_COMPLETE": frozenset(
        {"STAGE3A_PLUS_DEVELOPMENT_STABLE", STATE_NOT_READY}),
    "STAGE3A_PLUS_DEVELOPMENT_STABLE": frozenset({"FINAL_MODEL_FROZEN"}),
    "FINAL_MODEL_FROZEN": frozenset({"COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS"}),
    "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS": frozenset({"QUALIFICATION_POWER_FROZEN"}),
    "QUALIFICATION_POWER_FROZEN": frozenset({"MANUAL_COLLECTION_COMPLETE"}),
    "MANUAL_COLLECTION_COMPLETE": frozenset({"FROZEN_INFERENCE_COMPLETE"}),
    "FROZEN_INFERENCE_COMPLETE": frozenset({"AUTOMATIC_SINGLE_ADJUDICATION_STARTED"}),
    "AUTOMATIC_SINGLE_ADJUDICATION_STARTED": frozenset({"QUALIFICATION_ADJUDICATED"}),
    "QUALIFICATION_ADJUDICATED": frozenset({
        "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS",
        "INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_STAGE3A_PLUS",
        "INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE",
    }),
}


def load_state(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / STATE_FILE_NAME
    if not path.is_file():
        return {"plan_id": STAGE3A_PLUS_PLAN_ID, "state": STATE_INITIAL, "history": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("plan_id") != STAGE3A_PLUS_PLAN_ID:
        raise ValueError(f"state file plan_id mismatch in {path}")
    if state.get("state") not in ALLOWED_TRANSITIONS and state.get("state") not in TERMINAL_STATES:
        raise ValueError(f"unknown state {state.get('state')!r} in {path}")
    return state


def advance_state(run_dir: Path, target: str, *, note: str = "") -> dict[str, Any]:
    """Validated state transition; appends history and writes atomically."""
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    current = state["state"]
    if target not in ALLOWED_TRANSITIONS.get(current, frozenset()):
        raise ValueError(f"illegal transition {current} -> {target}")
    state = dict(state)
    state["history"] = list(state.get("history", [])) + [{
        "from": current, "to": target, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note}]
    state["state"] = target
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / STATE_FILE_NAME, state)
    return state


# --------------------------------------------------------------------------- #
# Final freeze receipt (plan Task 7)
# --------------------------------------------------------------------------- #
FROZEN_CODE_FILES = (
    "run_stage3a_plus.py",
    "run_stage3a.py",
    "ac_twin/stage3a_plus_statistics.py",
    "ac_twin/stage3a_plus_calibration.py",
    "ac_twin/stage3_data.py",
    "ac_twin/stage3_response.py",
    "ac_twin/stage3_training.py",
    "ac_twin/stage3_statistics.py",
    "ac_twin/stage1_statistics.py",
)


def build_final_freeze_receipt(
    *,
    dev_report_path: Path,
    calibration_selection_path: Path,
    checkpoints: Mapping[int, str],
    checkpoint_signatures: Mapping[int, str],
    config: Mapping[str, Any],
    code_root: Path,
    strongest_baselines: Mapping[str, str],
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    """Freeze receipt covering config, calibration, all ten checkpoints,
    baselines, the Gate definition, the code and both dev reports' SHA256
    (plan Task 7 step 2). Raises on any integrity failure."""
    dev_report_path = Path(dev_report_path)
    cal_path = Path(calibration_selection_path)
    report = json.loads(dev_report_path.read_text(encoding="utf-8"))
    if report.get("plan_id") != STAGE3A_PLUS_PLAN_ID:
        raise ValueError("dev report plan_id mismatch")
    selection = json.loads(cal_path.read_text(encoding="utf-8"))
    sel_payload = selection.get("receipt_payload", {})
    if selection.get("receipt_sha256") != canonical_fingerprint(sel_payload):
        raise ValueError("calibration selection receipt failed integrity check")
    calibrator_from_selection_receipt(selection)  # artifact signature self-check

    seeds_sorted = sorted(int(s) for s in checkpoints)
    if seeds_sorted != sorted(STAGE3A_PLUS_ALL_SEEDS):
        raise ValueError("freeze needs exactly the ten fixed seeds")
    checkpoint_block = {}
    for seed in seeds_sorted:
        path = Path(checkpoints[seed])
        sig = str(checkpoint_signatures[seed])
        if path.is_file():
            actual = s3t.checkpoint_signature(s3t.load_predictor_state(path))
            if actual != sig:
                raise ValueError(f"checkpoint signature drift for seed {seed}")
        checkpoint_block[str(seed)] = {"path": str(path), "signature": sig}

    code_root = Path(code_root)
    code_sha = {rel: sha256_file(code_root / rel) for rel in FROZEN_CODE_FILES}

    payload = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "final_freeze",
        "round": int(report["round"]),
        "config": {"final_config": dict(config["final_config"]),
                   "training": dict(config["training"])},
        "calibration": {
            "selected_candidate": sel_payload["selected_candidate"],
            "selected_artifact": sel_payload["selected_artifact"],
            "calibration_signature": sel_payload["candidates"][
                sel_payload["selected_candidate"]]["calibration_signature"],
            "selection_receipt_sha256": str(selection_receipt_sha256),
        },
        "checkpoints": checkpoint_block,
        "baselines": {"strongest": dict(strongest_baselines)},
        "gate": {
            "definition": "GATE_R_PLUS = core AND stability AND Q+",
            "core_metrics": list(sps.CORE_METRICS),
            "ece_guard": sps.ECE_GUARD,
            "ece_relative_improvement": sps.ECE_RELATIVE_IMPROVEMENT,
            "min_positive_brier_seeds": sps.MIN_POSITIVE_BRIER_SEEDS,
            "worst_seed_noninferiority_frac": sps.WORST_SEED_NONINFERIORITY_FRAC,
            "nondegenerate_std_floor": sps.NONDEGENERATE_STD_FLOOR,
            "dev_seeds": list(STAGE3A_PLUS_DEV_SEEDS),
            "stress_seeds": list(STAGE3A_PLUS_STRESS_SEEDS),
        },
        "code_sha256": code_sha,
        "dev_report_sha256": {
            "A": canonical_fingerprint(report["per_set"]["A"]),
            "B": canonical_fingerprint(report["per_set"]["B"]),
            "file": sha256_file(dev_report_path),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"receipt_payload": payload,
            "receipt_sha256": canonical_fingerprint(payload)}


def freeze_final(*, dev_report_path: Path, calibration_selection_path: Path,
                 checkpoints: Mapping[int, str],
                 checkpoint_signatures: Mapping[int, str],
                 config: Mapping[str, Any], code_root: Path,
                 strongest_baselines: Mapping[str, str],
                 selection_receipt_sha256: str, output_path: Path) -> int:
    """Write the single-use freeze receipt. Refuses (rc=2) unless both dev
    sets passed Gate R+_dev, and never overwrites an existing receipt."""
    output_path = Path(output_path)
    report = json.loads(Path(dev_report_path).read_text(encoding="utf-8"))
    per_set = report.get("per_set", {})
    a_ok = bool(per_set.get("A", {}).get("passed"))
    b_ok = bool(per_set.get("B", {}).get("passed"))
    if not (a_ok and b_ok and report.get("both_passed")):
        print(f"[stage3a-plus-freeze] REFUSED: Gate R+_dev not passed on both dev sets "
              f"(A={a_ok}, B={b_ok})", flush=True)
        return 2
    if output_path.exists():
        print(f"[stage3a-plus-freeze] REFUSED: {output_path} already exists "
              f"(single-use)", flush=True)
        return 2
    receipt = build_final_freeze_receipt(
        dev_report_path=dev_report_path,
        calibration_selection_path=calibration_selection_path,
        checkpoints=checkpoints, checkpoint_signatures=checkpoint_signatures,
        config=config, code_root=code_root,
        strongest_baselines=strongest_baselines,
        selection_receipt_sha256=selection_receipt_sha256)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(output_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, ensure_ascii=False)
    print(f"[stage3a-plus-freeze] FINAL_MODEL_FROZEN -> {output_path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Checkpoints + inference (torch; injectable for tests)
# --------------------------------------------------------------------------- #
def _predictor_checkpoint_dir(config: Mapping[str, Any]) -> Path:
    return _resolve(config["run_dir"]) / "checkpoints"


def _production_checkpoint(prior_run_dir: Path, cfg: Mapping[str, Any], seed: int) -> Path:
    name = f"h{cfg['hidden_dim']}_r{cfg['response_feature_dim']}_l{cfg['lambda_cls']}_seed{seed}"
    return prior_run_dir / "fit_five" / f"{name}.pt"


def _train_stress_seed(config, scaled, roles, scaler, data_signature, run_dir, seed):
    """Train one stress seed with the IDENTICAL frozen config (no architecture,
    loss, optimizer or early-stopping change) into run_dir/checkpoints.

    This only extends the stability observation; it never alters the backbone.
    Deterministic: seed is set before any model/random tensor creation.
    """
    cfg = config["final_config"]
    tr = config["training"]
    ckpt_dir = _predictor_checkpoint_dir(config)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"h{cfg['hidden_dim']}_r{cfg['response_feature_dim']}_l{cfg['lambda_cls']}_seed{seed}"
    pt_path = ckpt_dir / f"{name}.pt"
    state_path = ckpt_dir / f"{name}.state.json"
    if state_path.is_file():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("state") == "COMPLETE":
            s3t.verify_checkpoint_integrity(pt_path, expected_signature=prior["signature"])
            return prior
    _atomic_write_json(state_path, {"run": name, "state": "IN_PROGRESS", "seed": seed})
    try:
        s3t.seed_all(seed)  # BEFORE model construction (repair plan v0.1 §2.2)
        train_idx, val_idx = roles["train"], roles["validation"]
        class_weights = compute_class_weights(
            scaled["regime_labels"][train_idx].numpy(), scaled["valid"][train_idx].numpy())
        weights = torch.as_tensor(class_weights)

        def loss_fn(model, batch):
            out = model(
                batch["history"], batch["factual_suffix"], batch["factual_actions"],
                batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
                batch["t_star"])
            return response_masked_loss(
                out, batch["r_true"], batch["regime_labels"], batch["valid"],
                weights.to(out["regime_logits"].device), lambda_cls=float(cfg["lambda_cls"]))

        sums_fn = s3a._response_sums_fn(weights)
        batches = s3a.make_batches(scaled, train_idx, int(tr["batch_size"]), seed=seed)
        val_batches = [b for _, b in s3a.make_eval_batches(scaled, val_idx, int(tr["batch_size"]))]
        model = Stage3ResponsePredictor(
            hidden_dim=int(cfg["hidden_dim"]), response_feature_dim=int(cfg["response_feature_dim"]))
        result = s3t.train_model(
            model, loss_fn, batches, val_batches, seed=seed,
            max_epochs=int(tr["max_epochs"]),
            early_stopping_patience=int(tr["early_stopping_patience"]),
            learning_rate=float(tr["learning_rate"]), device=config["device"],
            progress_prefix=f"3A+stress:{name}", checkpoint_path=ckpt_dir / f"{name}.ckpt",
            resume=True, val_sums_fn=sums_fn, selection_fn=s3a.selection_score,
        )
        meta = {"run": name, "seed": int(seed), "config": dict(cfg),
                "contract_id": "AC-TWIN-STAGE3A-PLUS-STRESS-2026-09-09"}
        record = s3t.save_predictor_checkpoint(pt_path, result["state_dict"], meta=meta)
        summary = {"run": name, "state": "COMPLETE", "seed": int(seed),
                   "config": dict(cfg), "best_val_loss": result["best_val_loss"],
                   "validation_selection_score": result.get("validation_selection_score"),
                   "epochs_run": result["epochs_run"], "signature": record["signature"],
                   "checkpoint": record["path"]}
        _atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:
        _atomic_write_json(state_path, {"run": name, "state": "FAILED", "seed": seed,
                                        "error": repr(exc)})
        raise


def _ensure_ten_checkpoints(config, scaled, roles, scaler, data_signature, run_dir):
    """Return {seed: checkpoint_path} for all ten fixed seeds.

    The five production checkpoints come read-only from the prior retrain run;
    the five stress checkpoints are trained (identical config) if missing.
    """
    cfg = config["final_config"]
    prior = _resolve(config["prior_run_dir"])
    checkpoints: dict[int, str] = {}
    for seed in STAGE3A_PLUS_DEV_SEEDS:
        path = _production_checkpoint(prior, cfg, seed)
        if not path.is_file():
            raise FileNotFoundError(f"production checkpoint missing: {path}")
        checkpoints[seed] = str(path)
    reporter = ProgressReporter(total=len(STAGE3A_PLUS_STRESS_SEEDS), every=1,
                                prefix="3A+stress-fit", heartbeat_seconds=30.0)
    for i, seed in enumerate(STAGE3A_PLUS_STRESS_SEEDS, start=1):
        summary = _train_stress_seed(config, scaled, roles, scaler, data_signature,
                                     run_dir, seed)
        checkpoints[seed] = summary["checkpoint"]
        reporter.update(i, details=f"seed{seed} sig={summary['signature'][:8]}")
    return checkpoints


def _infer_all_seeds(checkpoints, cfg, scaled_tensors, indices, device):
    """Unshuffled, episode-order-asserted inference for all ten seeds."""
    indices = np.asarray(indices)
    r_hats, logits = [], []
    for seed in STAGE3A_PLUS_ALL_SEEDS:
        state = s3t.load_predictor_state(checkpoints[seed])
        model = Stage3ResponsePredictor(
            hidden_dim=int(cfg["hidden_dim"]), response_feature_dim=int(cfg["response_feature_dim"]))
        model.load_state_dict(state)
        pred = s3a._predict_split(model, scaled_tensors, indices, device)
        s3a._assert_episode_order(pred, indices)
        r_hats.append(pred["r_hat"])
        logits.append(pred["regime_logits"])
    return np.stack(r_hats, axis=0), np.stack(logits, axis=0)


def _make_eval_set(name, r_hat_seeds, logits_seeds, tensors, indices, baselines):
    idx = np.asarray(indices)
    valid = tensors["valid"][idx].numpy()
    return {
        "name": name,
        "r_hat_seeds": r_hat_seeds,
        "logits_seeds": logits_seeds,
        "r_true": tensors["r_true"][idx].numpy(),
        "regime_labels": tensors["regime_labels"][idx].numpy(),
        "valid_mask": valid,
        "episode_ids": idx,
        "expected_episode_ids": idx,
        "horizons": _horizon_grid(valid),
        "cf_actions": tensors["cf_action"][idx].numpy(),
        "cf_durations": tensors["cf_duration"][idx].numpy(),
        "baselines": baselines,
    }


def load_qualification_tensors(npz_path: str | Path) -> dict[str, Any]:
    """Load the single-use qualification set as an ALL-EVAL tensor bundle.

    Mirrors ``run_stage3a.load_replacement_tensors`` (same schema, duration set
    {1,3}, same tensor construction) but guarded by the Stage 3A Plus source
    rules: only the dedicated qualification directory may be loaded this way;
    viewed sources (old merge, replacement), sealed Stage 2 and Gate C paths
    are refused by content hash and by path.
    """
    npz_path = _resolve(npz_path)
    assert_stage3a_plus_source_role(npz_path, "qualification")
    with np.load(npz_path, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    required = (
        "object_history", "object_factual_future", "object_counterfactual_future",
        "factual_actions", "counterfactual_actions", "cf_action", "cf_duration", "t_star",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"qualification set missing required arrays: {missing}")
    durations = set(np.unique(np.asarray(data["cf_duration"])).tolist())
    if durations != {1, 3}:
        raise ValueError(f"qualification cf_duration must be exactly {{1,3}}, got {durations}")
    n_episodes = int(len(data["cf_action"]))
    r_pos = compute_response_pos(data["object_factual_future"], data["object_counterfactual_future"])
    regime = compute_regime_labels(r_pos)
    eval_mask = compute_effect_eval_mask(
        data["object_history"], data["object_factual_future"], data["object_counterfactual_future"])
    tensors = {
        "history": torch.as_tensor(data["object_history"], dtype=torch.float32),
        "factual_suffix": torch.as_tensor(data["object_factual_future"][:, :10], dtype=torch.float32),
        "factual_actions": torch.as_tensor(data["factual_actions"], dtype=torch.long),
        "counterfactual_actions": torch.as_tensor(data["counterfactual_actions"], dtype=torch.long),
        "cf_action": torch.as_tensor(data["cf_action"], dtype=torch.long),
        "cf_duration": torch.as_tensor(data["cf_duration"], dtype=torch.long),
        "t_star": torch.as_tensor(data["t_star"], dtype=torch.long),
        "r_true": torch.as_tensor(r_pos, dtype=torch.float32),
        "regime_labels": torch.as_tensor(regime, dtype=torch.long),
        "valid": torch.as_tensor(eval_mask, dtype=torch.bool),
    }
    return {
        "tensors": tensors,
        "roles": {"eval": np.arange(n_episodes)},
        "raw_factual_suffix": np.asarray(data["object_factual_future"][:, :10], dtype=np.float32),
        "data_signature": sha256_file(npz_path),
    }


# --------------------------------------------------------------------------- #
# train-round (Round 1/2 bounded retraining; plan §4.2/§4.3, Task 5)
# --------------------------------------------------------------------------- #
ROUND1_LAMBDA_BRIER = (0.05, 0.10, 0.20)
ROUND2_WEIGHT_DECAY = (0.0, 1e-5, 1e-4)
ROUND2_LABEL_SMOOTHING = (0.0, 0.02)
JTRAIN_CE_WEIGHT = 0.25
JTRAIN_BRIER_WEIGHT = 0.50
JTRAIN_SD_WEIGHT = 0.50


def round_candidate_space(config: Mapping[str, Any], run_dir: Path,
                          round_no: int) -> list[dict[str, float]]:
    """The frozen candidate space for Round 1/2 — exactly §4.2/§4.3; anything
    else is rejected. Round 2 pins ``lambda_brier`` to the Round-1 finalist."""
    if round_no == 1:
        cands = tuple(float(x) for x in config.get("round1", {}).get(
            "lambda_brier_candidates", ()))
        if cands != ROUND1_LAMBDA_BRIER:
            raise ValueError(
                f"round1 lambda_brier candidates must equal {ROUND1_LAMBDA_BRIER}")
        return [{"lambda_brier": lb, "weight_decay": 0.0, "label_smoothing": 0.0}
                for lb in cands]
    if round_no == 2:
        wds = tuple(float(x) for x in config.get("round2", {}).get("weight_decay", ()))
        lss = tuple(float(x) for x in config.get("round2", {}).get("label_smoothing", ()))
        if wds != ROUND2_WEIGHT_DECAY or lss != ROUND2_LABEL_SMOOTHING:
            raise ValueError(
                f"round2 space must equal weight_decay={ROUND2_WEIGHT_DECAY} x "
                f"label_smoothing={ROUND2_LABEL_SMOOTHING}")
        sel1 = json.loads(_train_selection_path(run_dir, 1).read_text(encoding="utf-8"))
        lambda_b = float(sel1["receipt_payload"]["finalist"]["lambda_brier"])
        return [{"lambda_brier": lambda_b, "weight_decay": wd, "label_smoothing": ls}
                for wd in wds for ls in lss]
    raise ValueError(f"train-round only supports rounds 1 and 2, got {round_no}")


def j_train(per_seed_components: list[Mapping[str, float]]) -> float:
    """J_train = mean_s(L_reg + 0.25 L_CE + 0.50 L_Brier) + 0.50 SD_s(L_Brier)
    over the five production seeds (plan §4.2; lower is better)."""
    scores = np.array([c["l_reg"] + JTRAIN_CE_WEIGHT * c["l_ce"]
                       + JTRAIN_BRIER_WEIGHT * c["l_brier"]
                       for c in per_seed_components], dtype=np.float64)
    briers = np.array([c["l_brier"] for c in per_seed_components], dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(briers).all():
        raise ValueError("non-finite J_train components")
    return float(scores.mean() + JTRAIN_SD_WEIGHT * briers.std(ddof=0))


def select_train_finalist(candidates: list[Mapping[str, float]],
                          scores: list[float]) -> int:
    """Unique lowest J_train; exact ties go to the earliest candidate in the
    frozen candidate order (deterministic)."""
    if len(candidates) != len(scores):
        raise ValueError("candidates/scores length mismatch")
    arr = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite J_train scores")
    return int(np.argmin(arr))  # argmin keeps the first (simplest-listed) tie


def check_train_round_allowed(run_dir: Path, round_no: int) -> tuple[bool, str]:
    """Round N may start only from state DEV_ROUND_{N-1}_COMPLETE with a FAILED
    previous round; a passed round forbids any further search (Task 5 step 7)."""
    run_dir = Path(run_dir)
    prev = round_no - 1
    state = load_state(run_dir)["state"]
    expected = f"DEV_ROUND_{prev}_COMPLETE"
    if state != expected:
        return False, f"state {state} != {expected}"
    report_path = _dev_report_path(run_dir, prev)
    if not report_path.is_file():
        return False, f"previous round dev report missing: {report_path}"
    if _dev_round_both_passed(run_dir, prev):
        return False, (f"round {prev} passed Gate R+_dev; continued search is forbidden")
    return True, "allowed"


def _validation_loss_components(state_dict, cfg, scaled, val_idx, batch_size,
                                device, class_weights) -> dict[str, float]:
    """Validation L_reg / L_CE(weighted) / L_Brier for J_train, computed with
    exactly the loss-term definitions of ``response_masked_loss``."""
    model = Stage3ResponsePredictor(
        hidden_dim=int(cfg["hidden_dim"]),
        response_feature_dim=int(cfg["response_feature_dim"]))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    weights = class_weights.to(device)
    totals = {"l_reg": 0.0, "l_ce": 0.0, "l_brier": 0.0}
    n_total = 0.0
    with torch.no_grad():
        for _, batch in s3a.make_eval_batches(scaled, val_idx, batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["history"], batch["factual_suffix"], batch["factual_actions"],
                        batch["counterfactual_actions"], batch["cf_action"],
                        batch["cf_duration"], batch["t_star"])
            valid = batch["valid"].to(out["log1p_r_hat"].dtype)
            n = float(valid.sum().item())
            denom = valid.sum().clamp_min(1.0)
            smooth = torch.nn.functional.smooth_l1_loss(
                out["log1p_r_hat"], torch.log1p(batch["r_true"].clamp_min(0.0)),
                beta=SMOOTH_L1_BETA, reduction="none")
            ce = torch.nn.functional.cross_entropy(
                out["regime_logits"].reshape(-1, out["regime_logits"].shape[-1]),
                batch["regime_labels"].reshape(-1).long(),
                weight=weights, reduction="none").reshape(valid.shape)
            totals["l_reg"] += float(((smooth * valid).sum() / denom).item()) * n
            totals["l_ce"] += float(((ce * valid).sum() / denom).item()) * n
            totals["l_brier"] += float(multiclass_brier_loss(
                out["regime_logits"], batch["regime_labels"], batch["valid"]).item()) * n
            n_total += n
    return {k: totals[k] / max(n_total, 1.0) for k in totals}


def _train_round_seed(config, scaled, roles, ckpt_dir, seed, *, knobs, tag):
    """Train one seed for Round 1/2. Architecture, data, epochs, early
    stopping and the selection score stay frozen; only the round's
    lambda_brier / weight_decay / label_smoothing knobs differ. ``seed_all``
    runs before ANY model construction; per-seed state files make restarts
    idempotent; signatures are verified on reuse."""
    cfg = config["final_config"]
    tr = config["training"]
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = (f"h{cfg['hidden_dim']}_r{cfg['response_feature_dim']}_l{cfg['lambda_cls']}"
            f"_b{knobs['lambda_brier']}_w{knobs['weight_decay']}"
            f"_s{knobs['label_smoothing']}_seed{seed}")
    pt_path = ckpt_dir / f"{name}.pt"
    state_path = ckpt_dir / f"{name}.state.json"
    if state_path.is_file():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("state") == "COMPLETE":
            s3t.verify_checkpoint_integrity(pt_path, expected_signature=prior["signature"])
            return prior
    _atomic_write_json(state_path, {"run": name, "state": "IN_PROGRESS", "seed": seed})
    try:
        s3t.seed_all(seed)  # BEFORE model construction (repair plan v0.1 §2.2)
        train_idx, val_idx = roles["train"], roles["validation"]
        class_weights = compute_class_weights(
            scaled["regime_labels"][train_idx].numpy(), scaled["valid"][train_idx].numpy())
        weights = torch.as_tensor(class_weights)

        def loss_fn(model, batch):
            out = model(
                batch["history"], batch["factual_suffix"], batch["factual_actions"],
                batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
                batch["t_star"])
            return response_masked_loss(
                out, batch["r_true"], batch["regime_labels"], batch["valid"],
                weights.to(out["regime_logits"].device), lambda_cls=float(cfg["lambda_cls"]),
                lambda_brier=float(knobs["lambda_brier"]),
                label_smoothing=float(knobs["label_smoothing"]))

        sums_fn = s3a._response_sums_fn(weights)
        batches = s3a.make_batches(scaled, train_idx, int(tr["batch_size"]), seed=seed)
        val_batches = [b for _, b in s3a.make_eval_batches(scaled, val_idx, int(tr["batch_size"]))]
        model = Stage3ResponsePredictor(
            hidden_dim=int(cfg["hidden_dim"]), response_feature_dim=int(cfg["response_feature_dim"]))
        result = s3t.train_model(
            model, loss_fn, batches, val_batches, seed=seed,
            max_epochs=int(tr["max_epochs"]),
            early_stopping_patience=int(tr["early_stopping_patience"]),
            learning_rate=float(tr["learning_rate"]), device=config["device"],
            weight_decay=float(knobs["weight_decay"]),
            progress_prefix=f"3A+round:{name}", checkpoint_path=ckpt_dir / f"{name}.ckpt",
            resume=True, val_sums_fn=sums_fn, selection_fn=s3a.selection_score)
        meta = {"run": name, "seed": int(seed), "config": dict(cfg), "round_knobs": dict(knobs),
                "contract_id": "AC-TWIN-STAGE3A-PLUS-ROUNDTRAIN-2026-09-09"}
        record = s3t.save_predictor_checkpoint(pt_path, result["state_dict"], meta=meta)
        comps = _validation_loss_components(
            result["state_dict"], cfg, scaled, val_idx, int(tr["batch_size"]),
            config["device"], weights)
        summary = {"run": name, "state": "COMPLETE", "seed": int(seed),
                   "knobs": dict(knobs), "best_val_loss": result["best_val_loss"],
                   "validation_selection_score": result.get("validation_selection_score"),
                   "epochs_run": result["epochs_run"], "signature": record["signature"],
                   "checkpoint": record["path"], "val_components": comps}
        _atomic_write_json(state_path, summary)
        return summary
    except Exception as exc:
        _atomic_write_json(state_path, {"run": name, "state": "FAILED", "seed": seed,
                                        "error": repr(exc)})
        raise


def cmd_train_round(args: argparse.Namespace, *,
                    train_seed_fn=None, validation_infer_fn=None) -> int:
    """Round 1/2 bounded retraining: train the frozen candidate space on the
    five production seeds, select the unique finalist by J_train, train only
    the finalist's five stress seeds, refit the Round-0 calibration family on
    the finalist's validation logits, and write single-use receipts. A second
    run verifies the receipts instead of retraining."""
    config = load_dev_config(args.config)
    round_no = int(args.round)
    run_dir = _resolve(config["run_dir"])
    allowed, reason = check_train_round_allowed(run_dir, round_no)
    if not allowed:
        print(f"[stage3a-plus-train-round] REFUSED: {reason}", flush=True)
        return 2
    out_train = _train_selection_path(run_dir, round_no)
    if out_train.is_file():
        stored = json.loads(out_train.read_text(encoding="utf-8"))
        ok = stored.get("receipt_sha256") == canonical_fingerprint(
            stored.get("receipt_payload", {}))
        for path in stored.get("receipt_payload", {}).get("checkpoints", {}).values():
            ok = ok and Path(path).is_file()
        print(f"[stage3a-plus-train-round] round {round_no} selection exists; "
              f"verify {'PASSED' if ok else 'FAILED'} (no retrain)", flush=True)
        return 0 if ok else 2

    candidates = round_candidate_space(config, run_dir, round_no)
    train_fn = train_seed_fn or _train_round_seed

    if validation_infer_fn is not None and train_seed_fn is not None:
        # Fully injected (tests): no real data loading.
        scaled, roles = None, None
        data_signature = "injected"
    else:
        bundle = s3a.load_development_tensors(_resolve(config["development_source"]))
        tensors, roles = bundle["tensors"], bundle["roles"]
        scaler = s3a.prepare_scaler(tensors, roles, _resolve(config["prior_run_dir"]))
        scaled = s3a.scale_tensors(tensors, scaler)
        data_signature = bundle["data_signature"]

    round_dir = _round_dir(run_dir, round_no)
    scores: list[float] = []
    per_candidate_components: list[dict[str, Any]] = []
    prod_summaries: list[dict[int, Any]] = []
    reporter = ProgressReporter(total=len(candidates) * len(STAGE3A_PLUS_DEV_SEEDS),
                                every=1, prefix=f"3A+round{round_no}-fit",
                                heartbeat_seconds=30.0)
    done = 0
    for i, knobs in enumerate(candidates):
        ckpt_dir = round_dir / "checkpoints" / f"cand{i}"
        comps = []
        summaries: dict[int, Any] = {}
        for seed in STAGE3A_PLUS_DEV_SEEDS:
            summary = train_fn(config, scaled, roles, ckpt_dir, seed,
                               knobs=knobs, tag=f"round{round_no}-cand{i}-seed{seed}")
            comps.append(summary["val_components"])
            summaries[seed] = summary
            done += 1
            reporter.update(done, details=f"cand{i} seed{seed}")
        per_candidate_components.append({str(s): c for s, c in zip(STAGE3A_PLUS_DEV_SEEDS, comps)})
        prod_summaries.append(summaries)
        scores.append(j_train(comps))
    finalist_idx = select_train_finalist(candidates, scores)
    finalist = candidates[finalist_idx]

    # Stress five: finalist only.
    stress_ckpt_dir = round_dir / "checkpoints" / f"cand{finalist_idx}"
    stress_summaries = {}
    for seed in STAGE3A_PLUS_STRESS_SEEDS:
        stress_summaries[seed] = train_fn(
            config, scaled, roles, stress_ckpt_dir, seed,
            knobs=finalist, tag=f"round{round_no}-finalist-seed{seed}")

    # Collect all ten finalist checkpoints + signatures.
    cfg = config["final_config"]
    finalist_prod = prod_summaries[finalist_idx]
    checkpoints = {s: finalist_prod[s]["checkpoint"] for s in STAGE3A_PLUS_DEV_SEEDS}
    checkpoints.update({s: stress_summaries[s]["checkpoint"] for s in STAGE3A_PLUS_STRESS_SEEDS})
    signatures = {s: finalist_prod[s]["signature"] for s in STAGE3A_PLUS_DEV_SEEDS}
    signatures.update({s: stress_summaries[s]["signature"] for s in STAGE3A_PLUS_STRESS_SEEDS})

    # Refit the Round-0-selected calibration family on the finalist's
    # validation logits (plan §4.2: same family, no new selection).
    family = json.loads(_calibration_selection_path(run_dir, 0)
                        .read_text(encoding="utf-8"))["receipt_payload"]["selected_candidate"]
    if validation_infer_fn is not None:
        val_logits, val_labels, val_mask, train_prev, val_data_sig = validation_infer_fn(config)
        if data_signature == "injected":
            data_signature = val_data_sig
    else:
        _, val_logits = _infer_all_seeds(checkpoints, cfg, scaled, roles["validation"],
                                         config["device"])
        val_idx = roles["validation"]
        val_labels = scaled["regime_labels"][val_idx].numpy()
        val_mask = scaled["valid"][val_idx].numpy()
        train_prev = _train_prevalence(tensors, roles)
    screen = validation_calibration_screen(
        val_logits_seeds=val_logits, val_labels=val_labels, val_mask=val_mask,
        train_prevalence=train_prev,
        n_production_seeds=int(config.get("n_production_seeds", 5)),
        candidates=(family,))
    receipt = build_calibration_selection_receipt(
        screen, data_signature=data_signature,
        validation_logits_signature=cal._array_signature(val_logits))
    write_calibration_selection(_calibration_selection_path(run_dir, round_no), receipt)

    payload = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "train_selection",
        "round": round_no,
        "candidates": [dict(c) for c in candidates],
        "j_train": {str(i): float(s) for i, s in enumerate(scores)},
        "val_components": {str(i): per_candidate_components[i]
                           for i in range(len(candidates))},
        "finalist_index": finalist_idx,
        "finalist": dict(finalist),
        "calibration_family": family,
        "checkpoints": {str(s): checkpoints[s] for s in STAGE3A_PLUS_ALL_SEEDS},
        "checkpoint_signatures": {str(s): signatures[s] for s in STAGE3A_PLUS_ALL_SEEDS},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sel_receipt = {"receipt_payload": payload,
                   "receipt_sha256": canonical_fingerprint(payload)}
    fd = os.open(str(out_train), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(sel_receipt, fh, indent=2, ensure_ascii=False)
    print(f"[stage3a-plus-train-round] round={round_no} finalist={finalist} "
          f"j_train={scores[finalist_idx]:.6f} -> {out_train}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# evaluate-dev (Gate R+_dev on A and B for a given round)
# --------------------------------------------------------------------------- #
def _round_dir(run_dir: Path, round_no: int) -> Path:
    return Path(run_dir) / "development" / f"round{round_no}"


def _calibration_selection_path(run_dir: Path, round_no: int) -> Path:
    return _round_dir(run_dir, round_no) / "calibration_selection.json"


def _dev_report_path(run_dir: Path, round_no: int) -> Path:
    return _round_dir(run_dir, round_no) / "gate_r_plus_dev_report.json"


def _train_selection_path(run_dir: Path, round_no: int) -> Path:
    return _round_dir(run_dir, round_no) / "train_selection.json"


def _load_round_calibration(run_dir: Path, round_no: int):
    """Load + integrity-check the round's frozen calibrator from its receipt."""
    path = _calibration_selection_path(run_dir, round_no)
    if not path.is_file():
        raise FileNotFoundError(f"calibration selection missing: {path}")
    selection = json.loads(path.read_text(encoding="utf-8"))
    payload = selection.get("receipt_payload", {})
    if selection.get("receipt_sha256") != canonical_fingerprint(payload):
        raise ValueError(f"calibration selection receipt integrity failure: {path}")
    artifact = calibrator_from_selection_receipt(selection)
    return artifact, selection


def _round_checkpoint_signatures(config, run_dir: Path, round_no: int,
                                 prior: Path) -> dict[int, str]:
    """The round's authoritative per-seed checkpoint signatures.

    Round 0: production five from the frozen prior selection receipt, stress
    five from their training state files. Rounds 1/2: the round's
    train_selection.json (written by ``train-round``)."""
    if round_no == 0:
        selection = json.loads(
            (prior / "selection" / "final_selection.json").read_text(encoding="utf-8"))
        prod_sigs = list(selection["checkpoint_signatures"])
        sigs = {seed: prod_sigs[i] for i, seed in enumerate(STAGE3A_PLUS_DEV_SEEDS)}
        cfg = config["final_config"]
        for seed in STAGE3A_PLUS_STRESS_SEEDS:
            name = (f"h{cfg['hidden_dim']}_r{cfg['response_feature_dim']}"
                    f"_l{cfg['lambda_cls']}_seed{seed}")
            state = json.loads((_predictor_checkpoint_dir(config)
                                / f"{name}.state.json").read_text(encoding="utf-8"))
            if state.get("state") != "COMPLETE":
                raise ValueError(f"stress seed {seed} checkpoint not COMPLETE")
            sigs[seed] = state["signature"]
        return sigs
    train_sel = json.loads(
        _train_selection_path(run_dir, round_no).read_text(encoding="utf-8"))
    return {int(s): sig for s, sig in train_sel["checkpoint_signatures"].items()}


def _resolve_round_checkpoints(config, run_dir: Path, round_no: int,
                               scaled, roles, scaler, data_signature,
                               prior: Path) -> dict[int, str]:
    if round_no == 0:
        return _ensure_ten_checkpoints(config, scaled, roles, scaler,
                                       data_signature, run_dir)
    train_sel_path = _train_selection_path(run_dir, round_no)
    if not train_sel_path.is_file():
        raise FileNotFoundError(
            f"round {round_no} checkpoints require train-round first: {train_sel_path}")
    train_sel = json.loads(train_sel_path.read_text(encoding="utf-8"))
    checkpoints = {int(s): p for s, p in train_sel["checkpoints"].items()}
    if sorted(checkpoints) != sorted(STAGE3A_PLUS_ALL_SEEDS):
        raise ValueError(f"round {round_no} checkpoints do not cover the ten fixed seeds")
    for seed, path in checkpoints.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"round {round_no} checkpoint missing: {path}")
    return checkpoints


def cmd_evaluate_dev(args: argparse.Namespace, *,
                     ensure_checkpoints_fn=None, infer_fn=None) -> int:
    """Gate R+_dev for one round: apply the round's frozen calibrator to the
    ten-seed predictions on dev A and dev B, assemble the full gate report for
    each, and write the single-use dev report. Exit 0 = ran (pass/fail is in
    the report); 2 = engineering/protocol failure or overwrite refusal."""
    config = load_dev_config(args.config)
    round_no = int(args.round)
    if round_no not in (0, 1, 2):
        print(f"[stage3a-plus-evaluate-dev] invalid round {round_no}", flush=True)
        return 2
    run_dir = _resolve(config["run_dir"])
    prior = _resolve(config["prior_run_dir"])
    cfg = config["final_config"]

    bundle = s3a.load_development_tensors(_resolve(config["development_source"]))
    tensors, roles = bundle["tensors"], bundle["roles"]
    scaler = s3a.prepare_scaler(tensors, roles, prior)
    scaled = s3a.scale_tensors(tensors, scaler)
    repl = s3a.load_replacement_tensors(_resolve(config["replacement_source"]))
    repl_scaled = s3a.scale_tensors(repl["tensors"], scaler)
    strongest = _strongest_basenames(prior)
    train_prev = _train_prevalence(tensors, roles)

    try:
        artifact, selection = _load_round_calibration(run_dir, round_no)
        expected_sigs = _round_checkpoint_signatures(config, run_dir, round_no, prior)
        checkpoints = _resolve_round_checkpoints(
            config, run_dir, round_no, scaled, roles, scaler,
            bundle["data_signature"], prior)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[stage3a-plus-evaluate-dev] setup failure: {exc}", flush=True)
        return 2

    actual_sigs = {s: s3t.checkpoint_signature(s3t.load_predictor_state(checkpoints[s]))
                   for s in STAGE3A_PLUS_ALL_SEEDS}
    signatures_ok = all(actual_sigs[s] == expected_sigs[s] for s in STAGE3A_PLUS_ALL_SEEDS)

    infer = infer_fn or _infer_all_seeds
    a_idx, b_idx = roles["development_test"], repl["roles"]["eval"]
    a_r, a_logits = infer(checkpoints, cfg, scaled, a_idx, config["device"])
    b_r, b_logits = infer(checkpoints, cfg, repl_scaled, b_idx, config["device"])
    finite_ok = all(np.isfinite(x).all() for x in (a_r, a_logits, b_r, b_logits))

    baselines_a = _build_eval_baselines(tensors, bundle["raw_factual_suffix"], roles,
                                        tensors, bundle["raw_factual_suffix"], a_idx, strongest)
    baselines_b = _build_eval_baselines(tensors, bundle["raw_factual_suffix"], roles,
                                        repl["tensors"], repl["raw_factual_suffix"], b_idx, strongest)
    eval_sets = {
        "A": _make_eval_set("A", a_r, a_logits, tensors, a_idx, baselines_a),
        "B": _make_eval_set("B", b_r, b_logits, repl["tensors"], b_idx, baselines_b),
    }
    n_prod = int(config.get("n_production_seeds", 5))
    n_resamples = int(config.get("dev_bootstrap_resamples", 2000))
    boot_seed = int(config.get("dev_bootstrap_seed", 20260929))

    gate_q_checks = {
        "signatures_ok": bool(signatures_ok),
        "finite_ok": bool(finite_ok),
        "seed_set_ok": sorted(checkpoints) == sorted(STAGE3A_PLUS_ALL_SEEDS),
        "calibration_receipt_ok": True,  # _load_round_calibration raised otherwise
        "plan_id_ok": True,
    }
    gate_q_passed = all(gate_q_checks.values())

    per_set: dict[str, Any] = {}
    for name, es in eval_sets.items():
        cal_probs = cal.apply_calibrator(artifact, es["logits_seeds"], train_prev)
        decomp = sps.decompose_seed_metrics(
            r_hat_seeds=es["r_hat_seeds"], logits_seeds=es["logits_seeds"],
            calibrated_probs_seeds=cal_probs, r_true=es["r_true"],
            regime_labels=es["regime_labels"], valid_mask=es["valid_mask"],
            episode_ids=es["episode_ids"], horizons=es["horizons"],
            cf_actions=es["cf_actions"], cf_durations=es["cf_durations"],
            baselines=es["baselines"], n_resamples=n_resamples,
            bootstrap_seed=boot_seed,
            expected_episode_ids=es.get("expected_episode_ids"),
            n_production_seeds=n_prod)
        per_set[name] = assemble_gate_r_plus_report(
            decomp, cal_probs_seeds=cal_probs, eval_set=es,
            n_resamples=n_resamples, bootstrap_seed=boot_seed + 7000,
            gate_q_passed=gate_q_passed, gate_q_checks=gate_q_checks)

    both_passed = bool(per_set["A"]["passed"] and per_set["B"]["passed"])
    report = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "gate_r_plus_dev_report",
        "round": round_no,
        "calibration_signature": selection["receipt_payload"]["candidates"][
            selection["receipt_payload"]["selected_candidate"]]["calibration_signature"],
        "checkpoint_signatures": {str(s): actual_sigs[s] for s in STAGE3A_PLUS_ALL_SEEDS},
        "data_signatures": {"development": bundle["data_signature"],
                            "replacement": repl["data_signature"]},
        "gate_q_checks": gate_q_checks,
        "per_set": per_set,
        "both_passed": both_passed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = _resolve(args.output) if args.output else _dev_report_path(run_dir, round_no)
    if out_path.exists():
        print(f"[stage3a-plus-evaluate-dev] REFUSED: {out_path} exists (single-use)",
              flush=True)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(out_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"[stage3a-plus-evaluate-dev] round={round_no} both_passed={both_passed} "
          f"-> {out_path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# freeze-final
# --------------------------------------------------------------------------- #
def cmd_freeze_final(args: argparse.Namespace) -> int:
    config = load_dev_config(args.config)
    round_no = int(args.round)
    run_dir = _resolve(config["run_dir"])
    prior = _resolve(config["prior_run_dir"])
    dev_report = _dev_report_path(run_dir, round_no)
    cal_path = _calibration_selection_path(run_dir, round_no)
    if not dev_report.is_file() or not cal_path.is_file():
        print(f"[stage3a-plus-freeze] missing dev report or calibration selection "
              f"for round {round_no}", flush=True)
        return 2
    selection = json.loads(cal_path.read_text(encoding="utf-8"))
    try:
        expected_sigs = _round_checkpoint_signatures(config, run_dir, round_no, prior)
        checkpoints = ({s: str(_production_checkpoint(prior, config["final_config"], s))
                        for s in STAGE3A_PLUS_DEV_SEEDS} if round_no == 0 else
                       {int(s): p for s, p in json.loads(
                           _train_selection_path(run_dir, round_no)
                           .read_text(encoding="utf-8"))["checkpoints"].items()})
        if round_no == 0:
            cfg = config["final_config"]
            for seed in STAGE3A_PLUS_STRESS_SEEDS:
                name = (f"h{cfg['hidden_dim']}_r{cfg['response_feature_dim']}"
                        f"_l{cfg['lambda_cls']}_seed{seed}")
                checkpoints[seed] = str(_predictor_checkpoint_dir(config) / f"{name}.pt")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[stage3a-plus-freeze] setup failure: {exc}", flush=True)
        return 2
    return freeze_final(
        dev_report_path=dev_report, calibration_selection_path=cal_path,
        checkpoints=checkpoints, checkpoint_signatures=expected_sigs,
        config=config, code_root=ROOT / "code",
        strongest_baselines=_strongest_basenames(prior),
        selection_receipt_sha256=selection["receipt_sha256"],
        output_path=run_dir / "freeze" / "final_freeze_receipt.json")


# --------------------------------------------------------------------------- #
# run-all: autonomous resumable orchestrator (dev phases; qualification phases
# are registered by Tasks 8-10)
# --------------------------------------------------------------------------- #
def _verify_state_artifacts(run_dir: Path, state: str) -> tuple[bool, str]:
    """Re-verify a completed phase's artifacts without rerunning or
    overwriting anything (plan Task 7 step 6)."""
    run_dir = Path(run_dir)
    if state in ("SOFTWARE_TESTED", "STAGE3A_PLUS_DEVELOPMENT_STABLE"):
        return True, "no artifact"
    if state == "FORENSIC_DIAGNOSTIC_COMPLETE":
        path = run_dir / "diagnostics" / "diagnostics_round0.json"
        if not path.is_file():
            return False, f"missing {path}"
        doc = json.loads(path.read_text(encoding="utf-8"))
        return (doc.get("plan_id") == STAGE3A_PLUS_PLAN_ID, f"plan_id in {path.name}")
    if state.startswith("DEV_ROUND_") and state.endswith("_COMPLETE"):
        round_no = int(state.removeprefix("DEV_ROUND_").removesuffix("_COMPLETE"))
        rd = _round_dir(run_dir, round_no)
        for name in ("calibration_selection.json", "gate_r_plus_dev_report.json"):
            path = rd / name
            if not path.is_file():
                return False, f"missing {path}"
        cal_doc = json.loads((rd / "calibration_selection.json").read_text(encoding="utf-8"))
        if cal_doc.get("receipt_sha256") != canonical_fingerprint(
                cal_doc.get("receipt_payload", {})):
            return False, f"calibration receipt integrity failure in round {round_no}"
        rep = json.loads((rd / "gate_r_plus_dev_report.json").read_text(encoding="utf-8"))
        if rep.get("plan_id") != STAGE3A_PLUS_PLAN_ID or rep.get("round") != round_no:
            return False, f"dev report plan/round mismatch in round {round_no}"
        if round_no > 0 and not (rd / "train_selection.json").is_file():
            return False, f"missing train_selection.json in round {round_no}"
        return True, f"round {round_no} artifacts verified"
    if state == "FINAL_MODEL_FROZEN":
        path = run_dir / "freeze" / "final_freeze_receipt.json"
        if not path.is_file():
            return False, f"missing {path}"
        doc = json.loads(path.read_text(encoding="utf-8"))
        return (doc.get("receipt_sha256") == canonical_fingerprint(
            doc.get("receipt_payload", {})), "freeze receipt integrity")
    if state == "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS":
        path = run_dir / "COMPUTE_BUDGET.md"
        return (path.is_file(), f"{path.name} present")
    if state == "QUALIFICATION_POWER_FROZEN":
        path = run_dir / "power" / "power_report.json"
        if not path.is_file():
            return False, f"missing {path}"
        doc = json.loads(path.read_text(encoding="utf-8"))
        return (doc.get("plan_id") == STAGE3A_PLUS_PLAN_ID
                and doc.get("status") == "POWER_FEASIBLE",
                f"power report plan/status in {path.name}")
    if state == "MANUAL_COLLECTION_COMPLETE":
        qual = run_dir / "qualification"
        for name in ("collection_receipt.json", "qualification_lock.json"):
            if not (qual / name).is_file():
                return False, f"missing {qual / name}"
        return True, "collection receipt + lock present"
    if state == "FROZEN_INFERENCE_COMPLETE":
        return _verify_inference_receipt(run_dir)
    if state == "AUTOMATIC_SINGLE_ADJUDICATION_STARTED":
        used = run_dir / "qualification" / QUALIFICATION_USED_LOCK_NAME
        return (used.is_file(), "used lock present (adjudication consumed)")
    if state == "QUALIFICATION_ADJUDICATED" or state in TERMINAL_STATES:
        if state == STATE_NOT_READY:
            return True, "no artifact"
        path = run_dir / "qualification" / "gate_r_plus_qualification_report.json"
        return (path.is_file(), f"{path.name} present")
    return True, "no verifier"


def _dev_round_both_passed(run_dir: Path, round_no: int) -> bool:
    report = json.loads(_dev_report_path(run_dir, round_no).read_text(encoding="utf-8"))
    return bool(report.get("both_passed"))


def cmd_run_all(args: argparse.Namespace, *, smoke_fn=None, phase_runners=None) -> int:
    """Advance the dev state machine autonomously. On restart, every completed
    phase in the state history is re-verified (never re-run, nothing
    overwritten) before the next phase starts."""
    config = load_dev_config(args.config)
    run_dir = _resolve(config["run_dir"])
    runners = dict(phase_runners or {})

    state = load_state(run_dir)
    for step in state.get("history", []):
        ok, msg = _verify_state_artifacts(run_dir, step["to"])
        if not ok:
            print(f"[stage3a-plus-run-all] RESTART VERIFICATION FAILED at "
                  f"{step['to']}: {msg}", flush=True)
            return 2

    def default_runners():
        diag_args = argparse.Namespace(
            config=args.config,
            output_dir=str(run_dir / "diagnostics"))
        return {
            "smoke": lambda: (smoke_fn or _default_smoke)(),
            "diagnose": lambda: cmd_diagnose(diag_args),
        }

    builtins = default_runners()

    while True:
        current = load_state(run_dir)["state"]
        print(f"[stage3a-plus-run-all] state={current}", flush=True)
        if current in TERMINAL_STATES:
            code = 0 if current == "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS" else 3
            print(f"[stage3a-plus-run-all] TERMINAL: {current}", flush=True)
            return code

        if current == STATE_INITIAL:
            rc = runners.get("smoke", builtins["smoke"])()
            if rc != 0:
                return rc
            advance_state(run_dir, "SOFTWARE_TESTED", note="targeted pytest green")
        elif current == "SOFTWARE_TESTED":
            diag_path = run_dir / "diagnostics" / "diagnostics_round0.json"
            if diag_path.is_file():
                print("[stage3a-plus-run-all] diagnose artifact exists; verified, not rerun",
                      flush=True)
            else:
                rc = runners.get("diagnose", builtins["diagnose"])()
                if rc != 0:
                    return rc
            advance_state(run_dir, "FORENSIC_DIAGNOSTIC_COMPLETE")
        elif current == "FORENSIC_DIAGNOSTIC_COMPLETE":
            rc = runners.get("round0", lambda: _run_dev_round(config, run_dir, 0))()
            if rc != 0:
                return rc
            advance_state(run_dir, "DEV_ROUND_0_COMPLETE")
        elif current == "DEV_ROUND_0_COMPLETE":
            if _dev_round_both_passed(run_dir, 0):
                advance_state(run_dir, "STAGE3A_PLUS_DEVELOPMENT_STABLE",
                              note="Round 0 Gate R+_dev passed on A and B")
            else:
                rc = runners.get("round1", lambda: _run_dev_round(config, run_dir, 1))()
                if rc != 0:
                    return rc
                advance_state(run_dir, "DEV_ROUND_1_COMPLETE")
        elif current == "DEV_ROUND_1_COMPLETE":
            if _dev_round_both_passed(run_dir, 1):
                advance_state(run_dir, "STAGE3A_PLUS_DEVELOPMENT_STABLE",
                              note="Round 1 Gate R+_dev passed on A and B")
            else:
                rc = runners.get("round2", lambda: _run_dev_round(config, run_dir, 2))()
                if rc != 0:
                    return rc
                advance_state(run_dir, "DEV_ROUND_2_COMPLETE")
        elif current == "DEV_ROUND_2_COMPLETE":
            if _dev_round_both_passed(run_dir, 2):
                advance_state(run_dir, "STAGE3A_PLUS_DEVELOPMENT_STABLE")
            else:
                advance_state(run_dir, STATE_NOT_READY,
                              note="Round 2 exhausted without a stable Gate R+_dev")
        elif current == "STAGE3A_PLUS_DEVELOPMENT_STABLE":
            rc = runners.get("freeze", lambda: _run_freeze(config, run_dir))()
            if rc != 0:
                return rc
            advance_state(run_dir, "FINAL_MODEL_FROZEN")
        elif current == "FINAL_MODEL_FROZEN":
            rc = runners.get("budget", lambda: _run_budget(config, run_dir))()
            if rc != 0:
                return rc
            advance_state(run_dir, "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS")
        elif current == "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS":
            rc = runners.get("power", lambda: _run_power(config))()
            if rc != 0:
                return rc  # rc 3 = joint power infeasible (hard stop §9.3 cond. 4)
            # the power phase advances the state itself on success
        elif current == "QUALIFICATION_POWER_FROZEN":
            rc = runners.get("collect", lambda: _run_collect(config, run_dir))()
            if rc != 0:
                return rc
            # the collector advances the state itself on completion
        elif current == "MANUAL_COLLECTION_COMPLETE":
            rc = runners.get("lock_infer", lambda: _run_lock_infer(config))()
            if rc != 0:
                return rc
            # qualification-infer advances the state itself
        elif current == "FROZEN_INFERENCE_COMPLETE":
            # The single adjudication: reaches a terminal state itself; its
            # return code IS the run outcome (0 qualified / 2 invalid / 3
            # inconclusive hard stop).
            return runners.get("gate", lambda: _run_gate(config))()
        elif current == "AUTOMATIC_SINGLE_ADJUDICATION_STARTED":
            print("[stage3a-plus-run-all] HARD STOP: the single adjudication was "
                  "started but never completed — the qualification set is burned; "
                  "report to the user before any further action", flush=True)
            return 2
        else:
            print(f"[stage3a-plus-run-all] no handler for state {current}", flush=True)
            return 2


def _default_smoke() -> int:
    """SOFTWARE_TESTED gate: the Stage 3A Plus targeted suites must be green."""
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_stage3a_plus_governance.py",
         "tests/test_stage3a_plus_statistics.py",
         "tests/test_stage3a_plus_calibration.py",
         "tests/test_stage3a_plus_driver.py",
         "tests/test_stage3a_plus_power.py",
         "tests/test_stage3a_plus_qualification.py",
         "tests/test_stage3a_plus_qualification_gate.py",
         "-W", "error", "-q"],
        cwd=str(ROOT), capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    print(f"[stage3a-plus-smoke] pytest rc={proc.returncode} {tail[0]}", flush=True)
    return 0 if proc.returncode == 0 else 2


def _run_dev_round(config, run_dir: Path, round_no: int) -> int:
    """One dev round: round 0 = calibration-screen + evaluate-dev; rounds 1/2
    = train-round (which also refits the calibration family) + evaluate-dev."""
    config_path = str(config["_config_path"])
    if round_no == 0:
        rc = cmd_calibration_screen(argparse.Namespace(
            config=config_path,
            output=str(_calibration_selection_path(run_dir, round_no)), resume=False))
    else:
        rc = cmd_train_round(argparse.Namespace(config=config_path, round=round_no,
                                                resume=False))
    if rc != 0:
        return rc
    return cmd_evaluate_dev(argparse.Namespace(
        config=config_path, round=round_no,
        output=str(_dev_report_path(run_dir, round_no))))


def _run_freeze(config, run_dir: Path) -> int:
    round_no = _stable_round(run_dir)
    if round_no is None:
        print("[stage3a-plus-freeze] no passed dev round found", flush=True)
        return 2
    return cmd_freeze_final(argparse.Namespace(
        config=str(config["_config_path"]), round=round_no))


def _run_budget(config, run_dir: Path) -> int:
    """COMPUTE_BUDGET_RECORDED gate: the Task 11 budget benchmark must already
    have produced COMPUTE_BUDGET.md (it is written before the pipeline reaches
    this state); this phase only verifies and records it."""
    path = Path(run_dir) / "COMPUTE_BUDGET.md"
    if not path.is_file():
        print(f"[stage3a-plus-run-all] budget artifact missing: {path} — run the "
              "Task 11 budget benchmark first", flush=True)
        return 2
    print(f"[stage3a-plus-run-all] compute budget recorded: {path}", flush=True)
    return 0


def _run_power(config) -> int:
    import prepare_stage3a_plus_power as pp
    return pp.main(["--config", str(config["_config_path"])])


def _run_collect(config, run_dir: Path) -> int:
    import collect_stage3a_plus_qualification as cq
    return cq.main(["--run-dir", str(run_dir), "--resume"])


def _run_lock_infer(config) -> int:
    config_path = str(config["_config_path"])
    rc = cmd_lock_qualification(argparse.Namespace(config=config_path))
    if rc != 0:
        return rc
    return cmd_qualification_infer(argparse.Namespace(config=config_path))


def _run_gate(config) -> int:
    return cmd_gate_r_plus_qualification(
        argparse.Namespace(config=str(config["_config_path"])))


def _stable_round(run_dir: Path) -> int | None:
    for round_no in (0, 1, 2):
        path = _dev_report_path(run_dir, round_no)
        if path.is_file():
            if _dev_round_both_passed(run_dir, round_no):
                return round_no
        else:
            break
    return None
def cmd_diagnose(args: argparse.Namespace, *, ensure_checkpoints_fn=None, infer_fn=None) -> int:
    config = load_dev_config(args.config)
    run_dir = _resolve(config["run_dir"])
    prior = _resolve(config["prior_run_dir"])
    device = config["device"]
    cfg = config["final_config"]

    # Development source (merge_2000): split 0/1 fit state; split 2 = dev A.
    bundle = s3a.load_development_tensors(_resolve(config["development_source"]))
    tensors, roles = bundle["tensors"], bundle["roles"]
    scaler = s3a.prepare_scaler(tensors, roles, prior)
    scaled = s3a.scale_tensors(tensors, scaler)
    # Replacement holdout = dev B (transform-only with the same scaler).
    repl = s3a.load_replacement_tensors(_resolve(config["replacement_source"]))
    repl_scaled = s3a.scale_tensors(repl["tensors"], scaler)

    strongest = _strongest_basenames(prior)
    train_prev = _train_prevalence(tensors, roles)

    ensure = ensure_checkpoints_fn or _ensure_ten_checkpoints
    infer = infer_fn or _infer_all_seeds
    checkpoints = ensure(config, scaled, roles, scaler, bundle["data_signature"], run_dir)

    # Engineering re-verification: production checkpoint signatures must match
    # the frozen selection receipt; all inference is finite + episode-ordered.
    selection = json.loads((prior / "selection" / "final_selection.json").read_text(encoding="utf-8"))
    expected_sigs = list(selection["checkpoint_signatures"])
    prod_sigs = [s3t.checkpoint_signature(s3t.load_predictor_state(checkpoints[s]))
                 for s in STAGE3A_PLUS_DEV_SEEDS]
    signatures_ok = prod_sigs == expected_sigs

    val_idx = roles["validation"]
    a_idx = roles["development_test"]
    b_idx = repl["roles"]["eval"]

    val_r, val_logits = infer(checkpoints, cfg, scaled, val_idx, device)
    a_r, a_logits = infer(checkpoints, cfg, scaled, a_idx, device)
    b_r, b_logits = infer(checkpoints, cfg, repl_scaled, b_idx, device)

    finite_ok = all(np.isfinite(x).all() for x in (val_r, val_logits, a_r, a_logits, b_r, b_logits))
    engineering_ok = bool(signatures_ok and finite_ok)

    baselines_a = _build_eval_baselines(tensors, bundle["raw_factual_suffix"], roles,
                                        tensors, bundle["raw_factual_suffix"], a_idx, strongest)
    baselines_b = _build_eval_baselines(tensors, bundle["raw_factual_suffix"], roles,
                                        repl["tensors"], repl["raw_factual_suffix"], b_idx, strongest)
    baselines_val = _build_eval_baselines(tensors, bundle["raw_factual_suffix"], roles,
                                          tensors, bundle["raw_factual_suffix"], val_idx, strongest)

    eval_sets = {
        "A": _make_eval_set("A", a_r, a_logits, tensors, a_idx, baselines_a),
        "B": _make_eval_set("B", b_r, b_logits, repl["tensors"], b_idx, baselines_b),
    }
    n_prod = int(config.get("n_production_seeds", 5))
    diagnostic = run_diagnostic(
        val_logits_seeds=val_logits,
        val_labels=tensors["regime_labels"][val_idx].numpy(),
        val_mask=tensors["valid"][val_idx].numpy(),
        train_prevalence=train_prev,
        eval_sets=eval_sets,
        engineering_ok=engineering_ok,
        n_production_seeds=n_prod,
        n_resamples=int(config.get("dev_bootstrap_resamples", 2000)),
        bootstrap_seed=int(config.get("dev_bootstrap_seed", 20260929)),
    )

    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Serialise: drop the artifact objects, keep signatures + reports.
    serialisable = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "engineering_ok": engineering_ok,
        "signatures_ok": signatures_ok,
        "finite_ok": finite_ok,
        "calibration_failure_status": diagnostic["calibration_failure_status"],
        "classification_input": diagnostic["classification_input"],
        "candidates": {
            cid: {
                "calibration_signature": c["calibration_signature"],
                "validation_nll": c["validation_nll"],
                "params": cal._to_jsonable(c["artifact"].params),
                "per_set": {s: c["per_set"][s] for s in c["per_set"]},
            }
            for cid, c in diagnostic["candidates"].items()
        },
        "data_signatures": {
            "development": bundle["data_signature"],
            "replacement": repl["data_signature"],
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = out_dir / "diagnostics_round0.json"
    _atomic_write_json(report_path, serialisable)
    print(f"[stage3a-plus-diagnose] status={diagnostic['calibration_failure_status']} "
          f"engineering_ok={engineering_ok} -> {report_path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# calibration-screen (Round 0)
# --------------------------------------------------------------------------- #
def cmd_calibration_screen(args: argparse.Namespace, *,
                           ensure_checkpoints_fn=None, infer_fn=None) -> int:
    """Round 0: select the unique calibration candidate by frozen J_cal on
    validation only; write a single-use, hash-chained receipt. ``--resume``
    re-verifies the stored receipt against a deterministic recomputation and
    never rewrites it."""
    config = load_dev_config(args.config)
    run_dir = _resolve(config["run_dir"])
    prior = _resolve(config["prior_run_dir"])
    cfg = config["final_config"]

    bundle = s3a.load_development_tensors(_resolve(config["development_source"]))
    tensors, roles = bundle["tensors"], bundle["roles"]
    scaler = s3a.prepare_scaler(tensors, roles, prior)
    scaled = s3a.scale_tensors(tensors, scaler)
    train_prev = _train_prevalence(tensors, roles)

    ensure = ensure_checkpoints_fn or _ensure_ten_checkpoints
    infer = infer_fn or _infer_all_seeds
    checkpoints = ensure(config, scaled, roles, scaler, bundle["data_signature"], run_dir)

    val_idx = roles["validation"]
    _, val_logits = infer(checkpoints, cfg, scaled, val_idx, config["device"])
    if not np.isfinite(val_logits).all():
        print("[stage3a-plus-calibration-screen] NON-FINITE validation logits — aborting",
              flush=True)
        return 2

    n_prod = int(config.get("n_production_seeds", 5))
    screen = validation_calibration_screen(
        val_logits_seeds=val_logits,
        val_labels=tensors["regime_labels"][val_idx].numpy(),
        val_mask=tensors["valid"][val_idx].numpy(),
        train_prevalence=train_prev,
        n_production_seeds=n_prod,
    )
    val_sig = cal._array_signature(val_logits)
    out_path = _resolve(args.output)

    if out_path.exists():
        if not args.resume:
            print(f"[stage3a-plus-calibration-screen] REFUSED: {out_path} already exists "
                  f"(single-use; pass --resume to re-verify)", flush=True)
            return 2
        ok, checks = verify_calibration_selection(
            out_path, screen, data_signature=bundle["data_signature"],
            validation_logits_signature=val_sig)
        for name, passed in checks:
            print(f"  verify {name}: {'OK' if passed else 'MISMATCH'}", flush=True)
        print(f"[stage3a-plus-calibration-screen] resume verify "
              f"{'PASSED' if ok else 'FAILED'}: {out_path}", flush=True)
        return 0 if ok else 2

    receipt = build_calibration_selection_receipt(
        screen, data_signature=bundle["data_signature"],
        validation_logits_signature=val_sig)
    try:
        write_calibration_selection(out_path, receipt)
    except FileExistsError:
        print(f"[stage3a-plus-calibration-screen] REFUSED: {out_path} already exists",
              flush=True)
        return 2
    selected = screen["selected"]
    print(f"[stage3a-plus-calibration-screen] selected={selected} "
          f"j_cal={screen['rows'][selected]['j_cal']:.6f} -> {out_path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Task 10: qualification lock, frozen inference, single Gate R+ adjudication
# --------------------------------------------------------------------------- #
QUALIFICATION_ADJUDICATION_RESAMPLES = 10_000
QUALIFICATION_ADJUDICATION_SEED = 20_260_929
QUALIFICATION_HIERARCHICAL_SEED_OFFSET = 7_000  # same offset as the dev gate
QUALIFICATION_USED_LOCK_NAME = "qualification_used.lock"


def _qualification_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "qualification"


def _load_freeze_payload(run_dir: Path) -> dict[str, Any]:
    """Read + integrity-check the final freeze receipt; raise on any drift."""
    path = Path(run_dir) / "freeze" / "final_freeze_receipt.json"
    if not path.is_file():
        raise FileNotFoundError(f"freeze receipt missing: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    payload = doc.get("receipt_payload", {})
    if doc.get("receipt_sha256") != canonical_fingerprint(payload):
        raise ValueError("freeze receipt integrity failure")
    if payload.get("plan_id") != STAGE3A_PLUS_PLAN_ID:
        raise ValueError("freeze receipt plan_id mismatch")
    return payload


def _qualification_data_path(run_dir: Path) -> Path:
    cfg = json.loads(
        (_qualification_dir(run_dir) / "config.json").read_text(encoding="utf-8"))
    return _resolve(cfg["data_output"])


def cmd_lock_qualification(args: argparse.Namespace) -> int:
    """Bind the collected qualification set to the freeze receipt and the power
    hash chain in a single-use lock (plan Task 10). Exit 0 = locked; 2 =
    wrong state/missing artifacts/lock exists; 3 = tampered data or chain."""
    config = load_dev_config(args.config)
    run_dir = _resolve(config["run_dir"])
    if load_state(run_dir)["state"] != "MANUAL_COLLECTION_COMPLETE":
        print("[stage3a-plus-lock] REFUSED: state != MANUAL_COLLECTION_COMPLETE",
              flush=True)
        return 2
    qual_dir = _qualification_dir(run_dir)
    lock_path = qual_dir / "qualification_lock.json"
    if lock_path.exists():
        print(f"[stage3a-plus-lock] REFUSED: {lock_path} exists (single-use)",
              flush=True)
        return 2
    receipt_path = qual_dir / "collection_receipt.json"
    if not receipt_path.is_file():
        print(f"[stage3a-plus-lock] REFUSED: collection receipt missing", flush=True)
        return 2
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("purpose") != STAGE3A_PLUS_QUALIFICATION_PURPOSE
            or receipt.get("plan_id") != STAGE3A_PLUS_PLAN_ID
            or receipt.get("qa", {}).get("strata_equal_counts") is not True):
        print("[stage3a-plus-lock] REFUSED: collection receipt purpose/plan/QA "
              "mismatch", flush=True)
        return 3
    try:
        data_path = _qualification_data_path(run_dir)
    except FileNotFoundError as exc:
        print(f"[stage3a-plus-lock] REFUSED: {exc}", flush=True)
        return 2
    if not data_path.is_file():
        print(f"[stage3a-plus-lock] REFUSED: qualification data missing: {data_path}",
              flush=True)
        return 2
    data_sha = sha256_file(data_path)
    try:
        assert_stage3a_plus_source_role(data_path, "qualification", file_sha256=data_sha)
    except ValueError as exc:
        print(f"[stage3a-plus-lock] REFUSED: {exc}", flush=True)
        return 3
    if receipt.get("data_sha256") != data_sha:
        print("[stage3a-plus-lock] REFUSED: qualification data was replaced after "
              "collection (sha256 mismatch with the collection receipt)", flush=True)
        return 3
    power_path = run_dir / "power" / "power_report.json"
    if receipt.get("power_report_sha256") != sha256_file(power_path):
        print("[stage3a-plus-lock] REFUSED: power report hash chain broken", flush=True)
        return 3
    schedule_doc = json.loads((qual_dir / "seed_schedule.json").read_text(encoding="utf-8"))
    if receipt.get("seed_schedule_sha256") != schedule_doc.get("schedule_sha256"):
        print("[stage3a-plus-lock] REFUSED: seed schedule hash chain broken", flush=True)
        return 3
    try:
        payload = _load_freeze_payload(run_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[stage3a-plus-lock] REFUSED: {exc}", flush=True)
        return 2
    freeze_sha = canonical_fingerprint(payload)
    lock = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "qualification_lock",
        "purpose": STAGE3A_PLUS_QUALIFICATION_PURPOSE,
        "qualification_data_sha256": data_sha,
        "collection_receipt_sha256": sha256_file(receipt_path),
        "freeze_receipt_sha256": freeze_sha,
        "power_report_sha256": receipt["power_report_sha256"],
        "seed_schedule_sha256": receipt["seed_schedule_sha256"],
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2, ensure_ascii=False)
    print(f"[stage3a-plus-lock] qualification locked -> {lock_path}", flush=True)
    return 0


def _verify_frozen_checkpoints(payload: Mapping[str, Any]) -> dict[int, Path]:
    """Re-verify all ten frozen checkpoint signatures; raise on any replacement."""
    checkpoints: dict[int, Path] = {}
    for key, rec in payload["checkpoints"].items():
        seed = int(key)
        path = _resolve(rec["path"])
        s3t.load_predictor_state(path, expected_signature=rec["signature"])
        checkpoints[seed] = path
    if sorted(checkpoints) != sorted(STAGE3A_PLUS_ALL_SEEDS):
        raise ValueError("freeze receipt checkpoint seed set drift")
    return checkpoints


def _prediction_paths(run_dir: Path) -> dict[int, Path]:
    pred_dir = _qualification_dir(run_dir) / "predictions"
    return {int(s): pred_dir / f"seed_{s}.npz" for s in STAGE3A_PLUS_ALL_SEEDS}


def _default_qualification_infer(qual, checkpoints, config):
    """Ten-seed frozen inference on the qualification set (development scaler,
    transform-only)."""
    run_dir = _resolve(config["run_dir"])
    prior = _resolve(config["prior_run_dir"])
    bundle = s3a.load_development_tensors(_resolve(config["development_source"]))
    scaler = s3a.prepare_scaler(bundle["tensors"], bundle["roles"], prior)
    scaled = s3a.scale_tensors(qual["tensors"], scaler)
    idx = qual["roles"]["eval"]
    cfg = config["final_config"]
    out = {}
    for seed in STAGE3A_PLUS_ALL_SEEDS:
        state = s3t.load_predictor_state(checkpoints[int(seed)])
        model = Stage3ResponsePredictor(
            hidden_dim=int(cfg["hidden_dim"]),
            response_feature_dim=int(cfg["response_feature_dim"]))
        model.load_state_dict(state)
        ev = s3a._predict_split(model, scaled, idx, config["device"])
        s3a._assert_episode_order(ev, idx)
        out[int(seed)] = {"r_hat": ev["r_hat"], "regime_logits": ev["regime_logits"]}
        print(f"[stage3a-plus-infer] seed {seed} done", flush=True)
    return out


def _verify_inference_receipt(run_dir: Path) -> tuple[bool, str]:
    pred_dir = _qualification_dir(run_dir) / "predictions"
    receipt_path = pred_dir / "inference_receipt.json"
    if not receipt_path.is_file():
        return False, f"missing {receipt_path}"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("plan_id") != STAGE3A_PLUS_PLAN_ID:
        return False, "inference receipt plan_id mismatch"
    for key, rec in receipt.get("seeds", {}).items():
        if sha256_file(_resolve(rec["path"])) != rec["sha256"]:
            return False, f"prediction file for seed {key} was replaced"
    if sorted(int(k) for k in receipt.get("seeds", {})) != sorted(STAGE3A_PLUS_ALL_SEEDS):
        return False, "inference receipt seed set drift"
    return True, "inference artifacts verified"


def cmd_qualification_infer(args: argparse.Namespace, *, infer_fn=None) -> int:
    """Frozen ten-seed inference on the locked qualification set (plan Task 10).

    Exit 0 = inference complete (or verified on restart); 2 = wrong state /
    missing lock; 3 = replaced data or replaced checkpoint."""
    config = load_dev_config(args.config)
    run_dir = _resolve(config["run_dir"])
    state = load_state(run_dir)["state"]
    if state == "FROZEN_INFERENCE_COMPLETE":
        ok, msg = _verify_inference_receipt(run_dir)
        print(f"[stage3a-plus-infer] restart verification: {msg}", flush=True)
        return 0 if ok else 3
    if state != "MANUAL_COLLECTION_COMPLETE":
        print(f"[stage3a-plus-infer] REFUSED: state {state} != "
              "MANUAL_COLLECTION_COMPLETE", flush=True)
        return 2
    lock_path = _qualification_dir(run_dir) / "qualification_lock.json"
    if not lock_path.is_file():
        print("[stage3a-plus-infer] REFUSED: qualification lock missing; run "
              "lock-qualification first", flush=True)
        return 2
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    data_path = _qualification_data_path(run_dir)
    data_sha = sha256_file(data_path)
    if lock.get("qualification_data_sha256") != data_sha:
        print("[stage3a-plus-infer] REFUSED: qualification data was replaced after "
              "the lock", flush=True)
        return 3
    try:
        payload = _load_freeze_payload(run_dir)
        if lock.get("freeze_receipt_sha256") != canonical_fingerprint(payload):
            raise ValueError("lock binds a different freeze receipt")
        checkpoints = _verify_frozen_checkpoints(payload)
        qual = load_qualification_tensors(data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[stage3a-plus-infer] REFUSED: {exc}", flush=True)
        return 3

    infer = infer_fn or _default_qualification_infer
    predictions = infer(qual, checkpoints, config)
    n_episodes = int(len(qual["roles"]["eval"]))
    pred_dir = _qualification_dir(run_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    seeds_block = {}
    for seed in STAGE3A_PLUS_ALL_SEEDS:
        pred = predictions[int(seed)]
        r_hat = np.asarray(pred["r_hat"], dtype=np.float64)
        logits = np.asarray(pred["regime_logits"], dtype=np.float64)
        if r_hat.shape[0] != n_episodes or logits.shape[0] != n_episodes:
            print(f"[stage3a-plus-infer] REFUSED: seed {seed} prediction episode "
                  "count mismatch", flush=True)
            return 2
        if not (np.isfinite(r_hat).all() and np.isfinite(logits).all()):
            print(f"[stage3a-plus-infer] REFUSED: seed {seed} predictions non-finite",
                  flush=True)
            return 2
        path = pred_dir / f"seed_{seed}.npz"
        if path.exists():
            # Restart mid-loop: the existing file must bind the same checkpoint.
            with np.load(path, allow_pickle=True) as z:
                if str(z["checkpoint_signature"]) != \
                        payload["checkpoints"][str(seed)]["signature"]:
                    print(f"[stage3a-plus-infer] REFUSED: prediction file for seed "
                          f"{seed} does not match the frozen checkpoint", flush=True)
                    return 3
        else:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, r_hat=r_hat, regime_logits=logits,
                         episode_ids=np.arange(n_episodes, dtype=np.int64),
                         seed=np.int64(seed),
                         checkpoint_signature=np.str_(
                             payload["checkpoints"][str(seed)]["signature"]))
        seeds_block[str(seed)] = {
            "path": str(path), "sha256": sha256_file(path),
            "checkpoint_signature": payload["checkpoints"][str(seed)]["signature"],
        }
    receipt = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "qualification_inference",
        "data_sha256": data_sha,
        "freeze_receipt_sha256": canonical_fingerprint(payload),
        "n_episodes": n_episodes,
        "seeds": seeds_block,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    receipt_path = pred_dir / "inference_receipt.json"
    fd = os.open(str(receipt_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, ensure_ascii=False)
    advance_state(run_dir, "FROZEN_INFERENCE_COMPLETE",
                  note=f"n={n_episodes}, ten seeds")
    print(f"[stage3a-plus-infer] FROZEN_INFERENCE_COMPLETE -> {receipt_path}",
          flush=True)
    return 0


def _gate_params_match_freeze(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("gate", {})
    return (
        gate.get("core_metrics") == list(sps.CORE_METRICS)
        and gate.get("ece_guard") == sps.ECE_GUARD
        and gate.get("ece_relative_improvement") == sps.ECE_RELATIVE_IMPROVEMENT
        and gate.get("min_positive_brier_seeds") == sps.MIN_POSITIVE_BRIER_SEEDS
        and gate.get("worst_seed_noninferiority_frac") == sps.WORST_SEED_NONINFERIORITY_FRAC
        and gate.get("nondegenerate_std_floor") == sps.NONDEGENERATE_STD_FLOOR
        and gate.get("dev_seeds") == list(STAGE3A_PLUS_DEV_SEEDS)
        and gate.get("stress_seeds") == list(STAGE3A_PLUS_STRESS_SEEDS)
    )


def _default_qualification_gate_runner(*, config, run_dir, freeze_payload,
                                       n_resamples, bootstrap_seed):
    """The formal Gate R+ evaluation on the qualification predictions."""
    qual_dir = _qualification_dir(run_dir)
    data_path = _qualification_data_path(run_dir)
    qual = load_qualification_tensors(data_path)
    receipt = json.loads(
        (qual_dir / "predictions" / "inference_receipt.json").read_text(encoding="utf-8"))

    r_seeds, logit_seeds = [], []
    for seed in STAGE3A_PLUS_ALL_SEEDS:
        with np.load(receipt["seeds"][str(seed)]["path"], allow_pickle=True) as z:
            r_seeds.append(np.asarray(z["r_hat"], dtype=np.float64))
            logit_seeds.append(np.asarray(z["regime_logits"], dtype=np.float64))
    r_seeds = np.stack(r_seeds, axis=0)
    logit_seeds = np.stack(logit_seeds, axis=0)
    if not (np.isfinite(r_seeds).all() and np.isfinite(logit_seeds).all()):
        raise RuntimeError("qualification predictions are non-finite")

    # Baselines fit on DEVELOPMENT train, predicted on the qualification set.
    bundle = s3a.load_development_tensors(_resolve(config["development_source"]))
    tensors, roles = bundle["tensors"], bundle["roles"]
    prior = _resolve(config["prior_run_dir"])
    strongest = _strongest_basenames(prior)
    idx = qual["roles"]["eval"]
    baselines = _build_eval_baselines(tensors, bundle["raw_factual_suffix"], roles,
                                      qual["tensors"], qual["raw_factual_suffix"],
                                      idx, strongest)
    eval_set = _make_eval_set("Q", r_seeds, logit_seeds, qual["tensors"], idx, baselines)

    selection = json.loads(
        _calibration_selection_path(run_dir, int(freeze_payload["round"]))
        .read_text(encoding="utf-8"))
    artifact = calibrator_from_selection_receipt(selection)  # integrity self-check
    train_prev = _train_prevalence(tensors, roles)
    cal_probs = cal.apply_calibrator(artifact, logit_seeds, train_prev)

    n_prod = int(config.get("n_production_seeds", 5))
    decomp = sps.decompose_seed_metrics(
        r_hat_seeds=r_seeds, logits_seeds=logit_seeds, calibrated_probs_seeds=cal_probs,
        r_true=eval_set["r_true"], regime_labels=eval_set["regime_labels"],
        valid_mask=eval_set["valid_mask"], episode_ids=eval_set["episode_ids"],
        horizons=eval_set["horizons"], cf_actions=eval_set["cf_actions"],
        cf_durations=eval_set["cf_durations"], baselines=eval_set["baselines"],
        n_resamples=n_resamples, bootstrap_seed=bootstrap_seed,
        expected_episode_ids=eval_set.get("expected_episode_ids"),
        n_production_seeds=n_prod)
    gate_q_checks = {
        "signatures_ok": True,   # frozen checkpoints re-verified at inference
        "finite_ok": True,       # raised above otherwise
        "seed_set_ok": True,
        "calibration_receipt_ok": True,
        "plan_id_ok": True,
        "lock_ok": True,
    }
    return assemble_gate_r_plus_report(
        decomp, cal_probs_seeds=cal_probs, eval_set=eval_set,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed + QUALIFICATION_HIERARCHICAL_SEED_OFFSET,
        gate_q_passed=all(gate_q_checks.values()), gate_q_checks=gate_q_checks)


def cmd_gate_r_plus_qualification(args: argparse.Namespace, *, gate_runner=None) -> int:
    """The SINGLE Stage 3A Plus qualification adjudication (plan §6.3, Task 10).

    Every refusal (replaced data/checkpoint/predictions, modified Gate params,
    second call) exits 3 BEFORE the single-use lock is burned; the lock is
    acquired atomically before the qualification truth is read. Exit 0 =
    QUALIFIED; 3 = refused or INCONCLUSIVE (scientific failure, hard stop);
    2 = engineering/protocol failure (INVALID_RUN)."""
    config = load_dev_config(args.config)
    run_dir = _resolve(config["run_dir"])
    state = load_state(run_dir)["state"]
    if state in TERMINAL_STATES or state in (
            "QUALIFICATION_ADJUDICATED", "AUTOMATIC_SINGLE_ADJUDICATION_STARTED"):
        print(f"[stage3a-plus-gate] REFUSED: state {state} — the single "
              "adjudication was already consumed", flush=True)
        return 3
    if state != "FROZEN_INFERENCE_COMPLETE":
        print(f"[stage3a-plus-gate] REFUSED: state {state} != "
              "FROZEN_INFERENCE_COMPLETE", flush=True)
        return 2

    # --- all guards run BEFORE the single-use lock is burned -------------------
    qual_dir = _qualification_dir(run_dir)
    lock_path = qual_dir / "qualification_lock.json"
    used_path = qual_dir / QUALIFICATION_USED_LOCK_NAME
    report_path = qual_dir / "gate_r_plus_qualification_report.json"
    if used_path.exists() or report_path.exists():
        print("[stage3a-plus-gate] REFUSED: the adjudication artifacts already exist",
              flush=True)
        return 3
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = _load_freeze_payload(run_dir)
        if lock.get("freeze_receipt_sha256") != canonical_fingerprint(payload):
            raise ValueError("lock binds a different freeze receipt")
        data_path = _qualification_data_path(run_dir)
        data_sha = sha256_file(data_path)
        if lock.get("qualification_data_sha256") != data_sha:
            raise ValueError("qualification data was replaced after the lock")
        if not _gate_params_match_freeze(payload):
            raise ValueError("the freeze receipt Gate constants differ from the "
                             "code's frozen Gate R+ definition (modified Gate params)")
        ok, msg = _verify_inference_receipt(run_dir)
        if not ok:
            raise ValueError(f"inference artifacts failed verification: {msg}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[stage3a-plus-gate] REFUSED: {exc}", flush=True)
        return 3

    # --- burn the single adjudication, THEN read the truth ----------------------
    advance_state(run_dir, "AUTOMATIC_SINGLE_ADJUDICATION_STARTED",
                  note="single-use qualification adjudication started")
    try:
        fd = os.open(str(used_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("[stage3a-plus-gate] REFUSED: qualification_used.lock exists", flush=True)
        return 3
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"plan_id": STAGE3A_PLUS_PLAN_ID,
                   "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "data_sha256": data_sha}, fh, indent=2)

    runner = gate_runner or _default_qualification_gate_runner
    try:
        result = runner(config=config, run_dir=run_dir, freeze_payload=payload,
                        n_resamples=QUALIFICATION_ADJUDICATION_RESAMPLES,
                        bootstrap_seed=QUALIFICATION_ADJUDICATION_SEED)
    except Exception as exc:  # engineering failure AFTER the lock is burned
        advance_state(run_dir, "QUALIFICATION_ADJUDICATED",
                      note=f"engineering failure: {exc}")
        advance_state(run_dir, "INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE",
                      note=str(exc)[:200])
        print(f"[stage3a-plus-gate] INVALID_RUN: {exc}", flush=True)
        return 2

    gate_q_passed = bool(result["report"].get("gate_q_passed", False))
    if not gate_q_passed:
        outcome = "INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE"
    elif result["passed"]:
        outcome = "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS"
    else:
        outcome = "INCONCLUSIVE_RESPONSE_HEAD_INSUFFICIENT_STAGE3A_PLUS"
    report = {
        "plan_id": STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "gate_r_plus_qualification_report",
        "purpose": STAGE3A_PLUS_QUALIFICATION_PURPOSE,
        "n_resamples": QUALIFICATION_ADJUDICATION_RESAMPLES,
        "bootstrap_seed": QUALIFICATION_ADJUDICATION_SEED,
        "hierarchical_bootstrap_seed":
            QUALIFICATION_ADJUDICATION_SEED + QUALIFICATION_HIERARCHICAL_SEED_OFFSET,
        "freeze_receipt_sha256": canonical_fingerprint(payload),
        "qualification_data_sha256": data_sha,
        "inference_receipt_sha256": sha256_file(
            qual_dir / "predictions" / "inference_receipt.json"),
        "result": result,
        "outcome": outcome,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    fd = os.open(str(report_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    advance_state(run_dir, "QUALIFICATION_ADJUDICATED", note=outcome)
    advance_state(run_dir, outcome)
    print(f"[stage3a-plus-gate] outcome={outcome} -> {report_path}", flush=True)
    if outcome == "RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS":
        return 0
    if outcome == "INVALID_RUN_ENGINEERING_OR_PROTOCOL_FAILURE":
        return 2
    return 3  # INCONCLUSIVE: scientific failure — record fully, then stop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AC-Twin Stage 3A Plus driver")
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser("diagnose")
    diag.add_argument("--config", required=True)
    diag.add_argument("--output-dir",
                      default="experiments/stage3_routing/stage3a_plus/diagnostics")
    diag.set_defaults(func=cmd_diagnose)

    screen = sub.add_parser("calibration-screen")
    screen.add_argument("--config", required=True)
    screen.add_argument(
        "--output",
        default="experiments/stage3_routing/stage3a_plus/development/round0/"
                "calibration_selection.json")
    screen.add_argument("--resume", action="store_true",
                        help="re-verify the stored receipt; never re-select or rewrite")
    screen.set_defaults(func=cmd_calibration_screen)

    train = sub.add_parser("train-round")
    train.add_argument("--config", required=True)
    train.add_argument("--round", type=int, choices=(1, 2), required=True)
    train.set_defaults(func=cmd_train_round)

    ev = sub.add_parser("evaluate-dev")
    ev.add_argument("--config", required=True)
    ev.add_argument("--round", type=int, choices=(0, 1, 2), required=True)
    ev.add_argument("--output", default=None,
                    help="defaults to development/roundN/gate_r_plus_dev_report.json")
    ev.set_defaults(func=cmd_evaluate_dev)

    freeze = sub.add_parser("freeze-final")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--round", type=int, choices=(0, 1, 2), required=True)
    freeze.set_defaults(func=cmd_freeze_final)

    run_all = sub.add_parser("run-all")
    run_all.add_argument("--config", required=True)
    run_all.set_defaults(func=cmd_run_all)

    lock = sub.add_parser("lock-qualification")
    lock.add_argument("--config", required=True)
    lock.set_defaults(func=cmd_lock_qualification)

    infer = sub.add_parser("qualification-infer")
    infer.add_argument("--config", required=True)
    infer.set_defaults(func=cmd_qualification_infer)

    gate = sub.add_parser("gate-r-plus-qualification")
    gate.add_argument("--config", required=True)
    gate.set_defaults(func=cmd_gate_r_plus_qualification)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

