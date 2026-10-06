"""Stage 4C-R2 mechanical baseline selection, one-shot validation gate and freeze.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 4.4 and 8.

* ``STRONGEST_STANDARD_BASELINE`` is the unique lowest validation AURC_REGRET among
  MARGIN_ONLY / SEED_VOTE_ENTROPY / SEED_MARGIN_STD; ties within 1e-12 break in that
  listed order. COMBINED_TOP2_MAX is historical reference only and CONSTANT_RANDOM a
  curve reference only — neither is ever eligible. The screen may never re-select.
* The validation gate (plan 8.1) evaluates ALL eight conditions exactly ONCE on the
  1,500 fresh validation groups. Any failure is the terminal state
  ``STAGE4CR2_MODEL_DEV_INSUFFICIENT_AND_LINE_CLOSED`` and the screen is never
  collected; a validation interaction stratum below 20% is
  ``STAGE4CR2_STRATUM_IMBALANCED``. No threshold edits, no retries.
* On a pass the freeze writes the five create-once artifacts and refuses any second
  write. ``final_selection.json`` binds the five checkpoints, the unique baseline, the
  interaction threshold, the train/validation data hashes, the code hashes, the exact
  parameter count, the primary-score definition and every receipt, plus ALL eligible
  baselines' validation AURC_REGRET / REGRET@80 / PR-AUC-prevalence and the
  deterministic tie-break order so the choice is offline-auditable.

Metric definitions are REUSED from ``stage4cp_statistics`` / ``stage4c_statistics``
(AURC_REGRET, REGRET@80, PR-AUC-prevalence, fail-closed improvement, Spearman); nothing
is redefined here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4cp_statistics as s4cs
from ac_twin import stage4cr2_protocol as s4cr2
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

STATUS_VALIDATION_QUALIFIED = "VALIDATION_QUALIFIED"

#: REGRET@80 coverage (frozen; reused from the Stage 4C Plus contract).
COVERAGE_MAIN = s4cp.COVERAGE_MAIN

PRIMARY_SCORE_DEFINITION = (
    "risk_score = sigmoid(logit_high) * 0.01 * expm1(softplus(raw_mu)); "
    "ensemble = arithmetic mean of the five model seeds' risk_score")

FREEZE_ARTIFACTS = (
    "feature_schema.json",
    "feature_scaler.npz",
    "model_inventory.json",
    "validation_gate_report.json",
    "final_selection.json",
)


@dataclass(frozen=True)
class BaselineReceipt:
    """The mechanical strongest-standard-baseline choice + the full audit trail."""
    plan_id: str
    strongest: str
    n_groups: int
    aurc_regret: dict[str, float]
    regret_at_80: dict[str, float]
    pr_auc_minus_prevalence: dict[str, float]
    tie_break_order: tuple[str, ...]
    tie_tolerance: float
    historical_reference_aurc: dict[str, float]  # COMBINED_TOP2_MAX, never eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "strongest_standard_baseline": self.strongest,
            "n_groups": int(self.n_groups),
            "aurc_regret": {k: float(v) for k, v in self.aurc_regret.items()},
            "regret_at_80": {k: float(v) for k, v in self.regret_at_80.items()},
            "pr_auc_minus_prevalence": {k: float(v)
                                        for k, v in self.pr_auc_minus_prevalence.items()},
            "tie_break_order": list(self.tie_break_order),
            "tie_tolerance": float(self.tie_tolerance),
            "historical_reference_aurc": {k: float(v)
                                          for k, v in self.historical_reference_aurc.items()},
        }


@dataclass(frozen=True)
class ValidationGateReport:
    """The single validation adjudication outcome (plan section 8.1)."""
    plan_id: str
    n_groups: int
    strongest_baseline: str
    baseline: BaselineReceipt
    conditions: dict[str, bool]
    metrics: dict[str, Any]
    strata_balanced: bool
    stratum_fractions: dict[str, float]
    interaction_threshold: float
    bindings: dict[str, Any]
    passed: bool
    status: str
    written_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "n_groups": int(self.n_groups),
            "strongest_standard_baseline": self.strongest_baseline,
            "baseline_receipt": self.baseline.to_dict(),
            "conditions": {k: bool(v) for k, v in self.conditions.items()},
            "metrics": self.metrics,
            "strata_balanced": bool(self.strata_balanced),
            "stratum_fractions": {k: float(v) for k, v in self.stratum_fractions.items()},
            "interaction_threshold": float(self.interaction_threshold),
            "bindings": self.bindings,
            "passed": bool(self.passed),
            "status": self.status,
            "written_at": self.written_at,
        }


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _aurc(risk: np.ndarray, regret: np.ndarray) -> float:
    return s4cs.aurc_regret(s4cs.order_by_risk(risk), regret)


def _regret80(risk: np.ndarray, regret: np.ndarray) -> float:
    return s4cs.regret_at_coverage(s4cs.order_by_risk(risk), regret, COVERAGE_MAIN)


# --- mechanical strongest-baseline selection (plan section 4.4) -------------------------------


def select_strongest_standard_baseline(
    validation: Mapping[str, np.ndarray]
) -> BaselineReceipt:
    """Unique lowest validation AURC_REGRET among the three eligible baselines.

    Ties within 1e-12 break in the frozen listed order (MARGIN_ONLY,
    SEED_VOTE_ENTROPY, SEED_MARGIN_STD). COMBINED_TOP2_MAX is recorded as historical
    reference and is NEVER eligible; the screen may never re-select.
    """
    regret = np.asarray(validation["regret"], dtype=np.float64).ravel()
    y_high = np.asarray(validation["y_high"], dtype=np.int64).ravel()
    risks = validation["baseline_risks"]
    if not np.isfinite(regret).all():
        raise ValueError("validation regret must be finite for baseline selection")

    aurc: dict[str, float] = {}
    r80: dict[str, float] = {}
    pr: dict[str, float] = {}
    for name in s4cr2.STRONGEST_BASELINE_CANDIDATES:
        risk = np.asarray(risks[name], dtype=np.float64).ravel()
        if risk.shape != regret.shape or not np.isfinite(risk).all():
            raise ValueError(f"baseline risk {name!r} must be finite and group-aligned")
        aurc[name] = _aurc(risk, regret)
        r80[name] = _regret80(risk, regret)
        pr[name] = s4cs.high_regret_pr(risk, y_high)

    strongest = s4cr2.STRONGEST_BASELINE_CANDIDATES[0]
    for name in s4cr2.STRONGEST_BASELINE_CANDIDATES[1:]:
        if aurc[name] < aurc[strongest] - s4cr2.BASELINE_TIE_TOLERANCE:
            strongest = name

    historical: dict[str, float] = {}
    if s4cr2.BASELINE_COMBINED_TOP2_MAX in risks:
        combined = np.asarray(risks[s4cr2.BASELINE_COMBINED_TOP2_MAX],
                              dtype=np.float64).ravel()
        historical[s4cr2.BASELINE_COMBINED_TOP2_MAX] = _aurc(combined, regret)

    return BaselineReceipt(
        plan_id=s4cr2.PLAN_ID,
        strongest=str(strongest),
        n_groups=int(regret.size),
        aurc_regret=aurc,
        regret_at_80=r80,
        pr_auc_minus_prevalence=pr,
        tie_break_order=tuple(s4cr2.STRONGEST_BASELINE_CANDIDATES),
        tie_tolerance=s4cr2.BASELINE_TIE_TOLERANCE,
        historical_reference_aurc=historical,
    )


# --- the one-shot validation gate (plan section 8.1) --------------------------------------------


def evaluate_validation_gate(
    scores: "Any", truth: Mapping[str, Any]
) -> ValidationGateReport:
    """Adjudicate ALL eight validation conditions exactly once (fail-closed).

    ``scores`` is the five-seed :class:`~ac_twin.stage4cr2_training.RiskScores` on the
    validation role; ``truth`` carries regret, y_high, the baseline risks, the frozen
    baseline receipt, the interaction distances + the TRAIN-frozen threshold, and the
    group seeds. A non-positive baseline denominator yields NaN improvements, which fail
    their conditions — never an epsilon, never an absolute difference.
    """
    ensemble = np.asarray(scores.risk_ensemble, dtype=np.float64).ravel()
    by_seed = np.asarray(scores.risk_by_seed, dtype=np.float64)
    score_seeds = np.asarray(scores.group_seed, dtype=np.int64).ravel()

    regret = np.asarray(truth["regret"], dtype=np.float64).ravel()
    y_high = np.asarray(truth["y_high"], dtype=np.int64).ravel()
    risks = truth["baseline_risks"]
    receipt = truth["baseline_receipt"]
    if isinstance(receipt, Mapping):
        receipt = BaselineReceipt(
            plan_id=str(receipt["plan_id"]),
            strongest=str(receipt["strongest_standard_baseline"]),
            n_groups=int(receipt["n_groups"]),
            aurc_regret={k: float(v) for k, v in receipt["aurc_regret"].items()},
            regret_at_80={k: float(v) for k, v in receipt["regret_at_80"].items()},
            pr_auc_minus_prevalence={k: float(v)
                                     for k, v in receipt["pr_auc_minus_prevalence"].items()},
            tie_break_order=tuple(receipt["tie_break_order"]),
            tie_tolerance=float(receipt["tie_tolerance"]),
            historical_reference_aurc={
                k: float(v) for k, v in receipt["historical_reference_aurc"].items()})
    truth_seeds = np.asarray(truth["group_seed"], dtype=np.int64).ravel()
    groups = regret.size

    distance = np.asarray(truth["interaction_min_distance"], dtype=np.float64).ravel()
    threshold = float(truth["interaction_threshold"])
    layer = (distance > threshold).astype(np.int64)
    fractions = {
        "low": float((layer == 0).mean()),
        "high": float((layer == 1).mean()),
    }
    strata_balanced = bool(
        np.isfinite(distance).all()
        and min(fractions.values()) >= s4cr2.STRATUM_IMBALANCE_FRAC)

    finite_inputs = bool(
        np.isfinite(ensemble).all() and np.isfinite(by_seed).all()
        and np.isfinite(regret).all()
        and by_seed.shape == (len(s4cr2.MODEL_SEEDS), groups)
        and ensemble.shape == (groups,))
    if not finite_inputs:
        # fail closed: no metric is computable; c8 (and everything downstream of the
        # non-finite values) fails and the adjudication is terminal.
        conditions = {key: False for key in (
            "c1_i_aurc_regret", "c2_i_regret80", "c3_pr_auc", "c4_seeds_i_aurc",
            "c5_seeds_i_regret80", "c6_trims_positive", "c7_margin_spearman",
            "c8_finite_matching")}
        status = (s4cr2.STATUS_STRATUM_IMBALANCED if not strata_balanced
                  else s4cr2.STATUS_MODEL_DEV_INSUFFICIENT)
        nan = float("nan")
        metrics = {
            "aurc_regret_ensemble": nan, "aurc_regret_baseline": nan,
            "i_aurc_regret": nan, "regret80_ensemble": nan, "regret80_baseline": nan,
            "i_regret80": nan, "pr_auc_minus_prevalence_ensemble": nan,
            "pr_auc_minus_prevalence_baseline": nan,
            "seed_i_aurc": [], "seed_i_regret80": [],
            "seeds_i_aurc_positive": 0, "seeds_i_regret80_positive": 0,
            "trim_i_aurc": {}, "spearman_ensemble_margin": nan,
        }
        return ValidationGateReport(
            plan_id=s4cr2.PLAN_ID, n_groups=int(groups),
            strongest_baseline=receipt.strongest, baseline=receipt,
            conditions=conditions, metrics=metrics,
            strata_balanced=strata_balanced, stratum_fractions=fractions,
            interaction_threshold=threshold,
            bindings=dict(truth.get("bindings", {})),
            passed=False, status=status,
            written_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    base_risk = np.asarray(risks[receipt.strongest], dtype=np.float64).ravel()
    margin_risk = np.asarray(risks[s4cr2.BASELINE_MARGIN], dtype=np.float64).ravel()

    aurc_base = _aurc(base_risk, regret)
    aurc_ens = _aurc(ensemble, regret)
    i_aurc = s4cs.improvement(aurc_base, aurc_ens)
    r80_base = _regret80(base_risk, regret)
    r80_ens = _regret80(ensemble, regret)
    i_r80 = s4cs.improvement(r80_base, r80_ens)
    pr_ens = s4cs.high_regret_pr(ensemble, y_high)
    pr_base = s4cs.high_regret_pr(base_risk, y_high)

    seed_i_aurc: list[float] = []
    seed_i_r80: list[float] = []
    if by_seed.shape == (len(s4cr2.MODEL_SEEDS), groups):
        for i in range(by_seed.shape[0]):
            seed_i_aurc.append(s4cs.improvement(aurc_base, _aurc(by_seed[i], regret)))
            seed_i_r80.append(s4cs.improvement(r80_base, _regret80(by_seed[i], regret)))
    n_seed_aurc_pos = int(sum(1 for v in seed_i_aurc if v > 0.0))
    n_seed_r80_pos = int(sum(1 for v in seed_i_r80 if v > 0.0))

    trim_i_aurc: dict[str, float] = {}
    for frac in s4cr2.GATE_TRIM_FRACTIONS:
        keep = s4cs.trim_high_regret(regret, frac)
        a_b = _aurc(base_risk[keep], regret[keep])
        a_e = _aurc(ensemble[keep], regret[keep])
        trim_i_aurc[f"{frac:.2f}"] = s4cs.improvement(a_b, a_e)

    spearman_margin = s4cs.spearman(ensemble, margin_risk)

    finite = bool(
        np.isfinite(base_risk).all() and np.isfinite(margin_risk).all()
        and np.isfinite([aurc_base, aurc_ens, r80_base, r80_ens, pr_ens, pr_base]).all())
    matching = bool(
        score_seeds.shape == truth_seeds.shape
        and np.array_equal(score_seeds, truth_seeds)
        and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)
        and tuple(int(s) for s in scores.seeds) == tuple(int(s) for s in s4cr2.MODEL_SEEDS))

    conditions = {
        "c1_i_aurc_regret": bool(i_aurc >= s4cr2.VALIDATION_I_AURC_MIN),
        "c2_i_regret80": bool(i_r80 >= s4cr2.VALIDATION_I_REGRET80_MIN),
        "c3_pr_auc": bool(pr_ens > 0.0 and pr_ens >= pr_base),
        "c4_seeds_i_aurc": bool(n_seed_aurc_pos >= s4cr2.VALIDATION_MIN_POSITIVE_SEEDS),
        "c5_seeds_i_regret80": bool(n_seed_r80_pos >= s4cr2.VALIDATION_MIN_POSITIVE_SEEDS),
        "c6_trims_positive": bool(all(v > 0.0 for v in trim_i_aurc.values())),
        "c7_margin_spearman": bool(abs(spearman_margin)
                                   < s4cr2.GATE_MARGIN_SPEARMAN_CEILING),
        "c8_finite_matching": bool(finite and matching),
    }

    passed = bool(all(conditions.values())) and strata_balanced
    if not strata_balanced:
        status = s4cr2.STATUS_STRATUM_IMBALANCED
    elif all(conditions.values()):
        status = STATUS_VALIDATION_QUALIFIED
    else:
        status = s4cr2.STATUS_MODEL_DEV_INSUFFICIENT

    metrics = {
        "aurc_regret_ensemble": float(aurc_ens),
        "aurc_regret_baseline": float(aurc_base),
        "i_aurc_regret": float(i_aurc),
        "regret80_ensemble": float(r80_ens),
        "regret80_baseline": float(r80_base),
        "i_regret80": float(i_r80),
        "pr_auc_minus_prevalence_ensemble": float(pr_ens),
        "pr_auc_minus_prevalence_baseline": float(pr_base),
        "seed_i_aurc": [float(v) for v in seed_i_aurc],
        "seed_i_regret80": [float(v) for v in seed_i_r80],
        "seeds_i_aurc_positive": n_seed_aurc_pos,
        "seeds_i_regret80_positive": n_seed_r80_pos,
        "trim_i_aurc": {k: float(v) for k, v in trim_i_aurc.items()},
        "spearman_ensemble_margin": float(spearman_margin),
    }
    return ValidationGateReport(
        plan_id=s4cr2.PLAN_ID,
        n_groups=int(groups),
        strongest_baseline=receipt.strongest,
        baseline=receipt,
        conditions=conditions,
        metrics=metrics,
        strata_balanced=strata_balanced,
        stratum_fractions=fractions,
        interaction_threshold=threshold,
        bindings=dict(truth.get("bindings", {})),
        passed=passed,
        status=status,
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# --- create-once freeze (plan section 8.2) -------------------------------------------------------


def freeze_final_selection(report: ValidationGateReport, inventory: "Any") -> Path:
    """Write the five create-once freeze artifacts exactly once; return the final path.

    Refuses a failed gate and any second write. ``final_selection.json`` binds the five
    checkpoints (path + SHA256), the unique strongest baseline with the full
    offline-auditable metrics trail, the interaction threshold, the train/validation
    data hashes, the code hashes, the exact parameter count, the primary-score
    definition and every receipt.
    """
    if not report.passed:
        raise ValueError(
            f"cannot freeze a failed validation gate (status {report.status}); "
            "the terminal state must be recorded instead and the line stops here")
    run_dir = Path(inventory.run_dir)
    dev_dir = run_dir / "development"
    targets = {name: dev_dir / name for name in FREEZE_ARTIFACTS}
    for name, path in targets.items():
        if path.exists():
            raise FileExistsError(
                f"Stage 4C-R2 freeze artifact already exists (create-once): {path}")

    # hash checks at freeze time: the five checkpoints must match the inventory.
    for seed, path in inventory.checkpoint_paths.items():
        digest = sha256_file(Path(path))
        if digest != inventory.checkpoint_sha256[int(seed)]:
            raise ValueError(f"checkpoint drifted since the fit: {path}")

    schema = dict(s4cpf.feature_schema())
    schema["plan_id"] = s4cr2.PLAN_ID
    schema["reused_feature_module"] = "code/ac_twin/stage4cp_features.py"
    schema["reused_feature_module_sha256"] = sha256_file(
        ROOT / "code/ac_twin/stage4cp_features.py")

    write_once_json(targets["feature_schema.json"], schema)
    inventory.scaler.save(targets["feature_scaler.npz"])
    write_once_json(targets["model_inventory.json"], inventory.to_dict())
    write_once_json(targets["validation_gate_report.json"], report.to_dict())

    bindings = dict(report.bindings)
    final = {
        "plan_id": s4cr2.PLAN_ID,
        "status": report.status,
        "architecture": s4cr2.ARCH_DIRECT,
        "param_count": int(inventory.param_count),
        "model_seeds": [int(s) for s in inventory.seeds],
        "ensemble_definition": "arithmetic_mean_of_five_seeds",
        "primary_score_definition": PRIMARY_SCORE_DEFINITION,
        "checkpoints": {
            str(seed): {"path": str(inventory.checkpoint_paths[int(seed)]),
                        "sha256": str(inventory.checkpoint_sha256[int(seed)])}
            for seed in inventory.seeds
        },
        "strongest_standard_baseline": report.strongest_baseline,
        "baseline_validation_metrics": {
            name: {
                "aurc_regret": float(report.baseline.aurc_regret[name]),
                "regret_at_80": float(report.baseline.regret_at_80[name]),
                "pr_auc_minus_prevalence": float(
                    report.baseline.pr_auc_minus_prevalence[name]),
            }
            for name in s4cr2.STRONGEST_BASELINE_CANDIDATES
        },
        "baseline_tie_break_order": list(s4cr2.STRONGEST_BASELINE_CANDIDATES),
        "baseline_tie_tolerance": s4cr2.BASELINE_TIE_TOLERANCE,
        "historical_reference_aurc": {
            k: float(v) for k, v in report.baseline.historical_reference_aurc.items()},
        "interaction_threshold": float(inventory.interaction_threshold),
        "interaction_threshold_source":
            "development_train preintervention_min_distance median (train-only)",
        "train_data_sha256": bindings.get("train_data_sha256", ""),
        "validation_data_sha256": bindings.get("validation_data_sha256", ""),
        "bindings": bindings,
        "code_sha256": {
            "ac_twin/stage4cr2_training.py": sha256_file(
                ROOT / "code/ac_twin/stage4cr2_training.py"),
            "ac_twin/stage4cr2_selection.py": sha256_file(
                ROOT / "code/ac_twin/stage4cr2_selection.py"),
        },
        "validation_gate_report_sha256": sha256_file(
            targets["validation_gate_report.json"]),
        "model_inventory_sha256": sha256_file(targets["model_inventory.json"]),
        "feature_scaler_sha256": sha256_file(targets["feature_scaler.npz"]),
        "feature_schema_sha256": sha256_file(targets["feature_schema.json"]),
        "n_validation_groups": int(report.n_groups),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(targets["final_selection.json"], final)
    return targets["final_selection.json"]


__all__ = [
    "COVERAGE_MAIN",
    "FREEZE_ARTIFACTS",
    "PRIMARY_SCORE_DEFINITION",
    "STATUS_VALIDATION_QUALIFIED",
    "BaselineReceipt",
    "ValidationGateReport",
    "evaluate_validation_gate",
    "freeze_final_selection",
    "select_strongest_standard_baseline",
]
