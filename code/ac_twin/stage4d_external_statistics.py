"""Stage 4D-D2 unique development-screen statistics and gate adjudication (plan section 8).

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 8.1-8.4 and 15
(the ``dev`` row). The statistical unit is the 10-action decision GROUP. ONE paired bootstrap
runs 10,000 whole-group resamples with entropy ``SCREEN_BOOTSTRAP_ENTROPY``; EVERY metric of a
draw is recomputed from the SAME resample index vector and the two-sided 80% CI is read off the
resample quantiles (the screen uses 80%; the confirm uses 95% and is a separate module/scope).
Metric definitions are REUSED bitwise from ``stage4cp_statistics`` (AURC_REGRET, REGRET@80,
PR-AUC-prevalence, fail-closed relative improvement, Spearman, order_by_risk); nothing is
redefined.

Gate TASK-X-DEV (plan 8.2, 8 conditions) and Gate RISK-X-DEV (plan 8.3, 6 conditions) are
implemented mechanically. The primary baseline is the validation-FROZEN unique standard
baseline read from ``final_selection.json`` (``strongest_standard_baseline``); the screen never
re-selects it (plan 7.6 / 8.3). Denominators are FAIL-CLOSED: a non-positive or non-finite
frozen-baseline AURC_REGRET / REGRET@80 yields NaN improvements and the gate returns
``STAGE4D_EXTERNAL_INVALID_ANALYSIS`` — never an epsilon, an absolute difference or a baseline
swap (plan 8.3 condition 5).

Terminal-state priority (plan 11.3 + 8.4 + 15): an integrity/QA failure or a degenerate
denominator / degenerate Map-Frenet projection / degenerate common evaluation mask makes the
adjudication invalid (``STAGE4D_EXTERNAL_INVALID_ANALYSIS`` — an invalid test is never recorded
as a method failure); a valid adjudication with any Gate TASK-X-DEV or RISK-X-DEV condition
failed is ``STAGE4D_EXTERNAL_DEV_NOT_READY`` (no screen rerun, no model change, no confirm);
both gates passing is ``STAGE4D_EXTERNAL_DEV_READY_FOR_POWER``.

Frozen quantity-vector layout (``N_QUANTITIES = 32``)::

    0   I_AURC_REGRET                 (Direct ensemble vs frozen primary baseline)
    1   I_REGRET@80
    2   PR-AUC - prevalence           (ensemble, high-regret label)
    3   branch-level Spearman(J_hat, J_true)
    4   base-vs-RANDOM_ACTION mean-regret improvement
    5   AURC_REGRET of the frozen primary baseline   (fail-closed denominator)
    6   REGRET@80 of the frozen primary baseline     (fail-closed denominator)
    7   base mean regret
    8   ALWAYS_IDLE mean regret
    9   RANDOM_ACTION mean regret
    10  high-regret fraction (regret > 0.01)
    11..15  per-seed I_AURC_REGRET   (five frozen Direct seeds)
    16..20  per-seed I_REGRET@80
    21..29  the three standard baselines' absolute curves (aurc, regret@80, pr-auc) x3
    30  candidate ensemble AURC_REGRET (absolute, reporting)
    31  candidate ensemble REGRET@80   (absolute, reporting)

This module is torch-free: the score freeze / lock burn / quantity assembly / adjudication are
pure numpy + create-once IO and take the score arrays already computed by the driver
(``drive_stage4d_external_screen.py``), so the one-shot ordering is testable on synthetic data.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_statistics as s4cs
from ac_twin import stage4d_external_training as T
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

PLAN_ID = T.PLAN_ID

# --- frozen screen contract (plan 8.1 / config.json) --------------------------------------------
DEV_SCREEN_GROUPS = T.DEV_SCREEN_GROUPS            # 4000
N_ACTIONS = T.N_ACTIONS                            # 10
N_MODEL_SEEDS = len(T.MODEL_SEEDS)                 # 5
COVERAGE_MAIN = T.COVERAGE_MAIN                    # 0.80
HIGH_REGRET_THRESHOLD = T.HIGH_REGRET_THRESHOLD    # 0.01

CONFIDENCE_SCREEN = 0.80                           # screen uses two-sided 80% CI (never 95%)
BOOTSTRAP_RESAMPLES = 10000
#: Fresh frozen bootstrap entropy for the D2 development screen. Plan 8.1 fixes 10,000
#: resamples + 80% CI but names no seed; this constant is distinct from every consumed stream
#: (R2 screen 2026092702, R2 confirm 2026093202, D1 confirm 2026110995, etc.). FLAGGED for review.
SCREEN_BOOTSTRAP_ENTROPY = 2026122101

# --- frozen gate thresholds (plan 8.2 / 8.3 / config.json) --------------------------------------
I_AURC_MIN = T.VALIDATION_I_AURC_MIN               # 0.05
I_REGRET80_MIN = T.VALIDATION_I_REGRET80_MIN       # 0.03
MIN_POSITIVE_SEEDS = T.VALIDATION_MIN_POSITIVE_SEEDS  # 4 (of 5, both improvements > 0)
BASELINE_TIE_TOLERANCE = T.BASELINE_TIE_TOLERANCE  # 1e-12
GATE_TASK_MIN_HIGH_REGRET_FRAC = 0.15
GATE_TASK_MIN_DISTINCT_ORACLE = 3
GATE_TASK_MIN_ORACLE_FRAC = 0.05
GATE_TASK_MIN_REACTIVE_FRAC = 0.20
GATE_TASK_MIN_STRATUM_FRAC = 0.10

#: fixed [t_star_low, t_star_high] split inherited bitwise from the Stage 4C / R2 / D1 contract
#: (``t_star >= 13`` is the high stratum); never re-fit on any role. The initial-distance
#: threshold is the platform TRAIN-only median frozen in ``final_selection.json``.
TSTAR_SPLIT = 13

#: "beyond numerical tolerance" for a non-ego reactive response (metres) — mirrors the frozen
#: pilot adjudication (``stage4d_pilot_adjudicate.REACTIVE_TOLERANCE_M``).
REACTIVE_TOLERANCE_M = 1e-2
CH_S, CH_D = 2, 3                                  # Map-Frenet (s, d) channels

#: trim keep-marker fractions for the plan-9.1 quantities table (family convention; the D2
#: screen has no trim GATE — these are reporting markers only). FLAGGED as inherited.
TRIM_FRACTIONS = (0.01, 0.05)

# --- mechanical terminal states (plan 8.3/8.4/15) -------------------------------------------------
STATUS_DEV_READY = "STAGE4D_EXTERNAL_DEV_READY_FOR_POWER"
STATUS_DEV_NOT_READY = "STAGE4D_EXTERNAL_DEV_NOT_READY"
STATUS_INVALID_ANALYSIS = "STAGE4D_EXTERNAL_INVALID_ANALYSIS"

# --- baselines (the three eligible standard baselines, frozen order; COMBINED is reference) -----
BASELINE_MARGIN = T.BASELINE_MARGIN
BASELINE_SEED_VOTE_ENTROPY = T.BASELINE_SEED_VOTE_ENTROPY
BASELINE_SEED_MARGIN_STD = T.BASELINE_SEED_MARGIN_STD
BASELINE_COMBINED_TOP2_MAX = T.BASELINE_COMBINED_TOP2_MAX
STANDARD_BASELINES = T.STRONGEST_BASELINE_CANDIDATES  # (MARGIN_ONLY, SEED_VOTE_ENTROPY, SEED_MARGIN_STD)
ALL_BASELINE_NAMES = STANDARD_BASELINES + (BASELINE_COMBINED_TOP2_MAX,)

# --- frozen quantity-vector layout ---------------------------------------------------------------
Q_I_AURC = 0
Q_I_REGRET80 = 1
Q_PR = 2
Q_SPEARMAN_BRANCH = 3
Q_BASE_VS_RANDOM = 4
Q_AURC_BASELINE = 5
Q_R80_BASELINE = 6
Q_BASE_MEAN_REGRET = 7
Q_IDLE_MEAN_REGRET = 8
Q_RANDOM_MEAN_REGRET = 9
Q_HIGH_REGRET_FRAC = 10
Q_SEED_I_AURC = 11          # 11..15
Q_SEED_I_R80 = 16           # 16..20
Q_BASELINE_CURVE = 21       # 21..29: 3 standard baselines x (aurc, r80, pr)
Q_AURC_MODEL = 30
Q_R80_MODEL = 31
N_QUANTITIES = 32

SEED_I_AURC_INDEX = tuple(range(Q_SEED_I_AURC, Q_SEED_I_AURC + N_MODEL_SEEDS))
SEED_I_R80_INDEX = tuple(range(Q_SEED_I_R80, Q_SEED_I_R80 + N_MODEL_SEEDS))

_QUANTITY_NAMES = (
    "i_aurc_regret", "i_regret80", "pr_auc_minus_prevalence", "spearman_branch",
    "base_vs_random_improvement", "aurc_baseline", "regret80_baseline",
    "base_mean_regret", "idle_mean_regret", "random_mean_regret", "high_regret_frac",
) + tuple(f"seed{s}_i_aurc" for s in T.MODEL_SEEDS) \
    + tuple(f"seed{s}_i_regret80" for s in T.MODEL_SEEDS) \
    + tuple(f"{name.lower()}_{metric}"
            for name in STANDARD_BASELINES
            for metric in ("aurc", "regret80", "pr")) \
    + ("aurc_model", "regret80_model")

#: per-group arrays the quantities mapping must carry (aligned on the group axis).
REQUIRED_ARRAY_KEYS = (
    "group_seed", "regret", "y_high", "random_regret", "idle_regret",
    "ensemble_risk", "risk_by_seed", "J_hat", "J_true", "a_star",
    "tstar_layer", "interaction_layer", "t_star", "interaction_min_distance", "reactive",
)

_BASELINE_NPZ_PREFIX = "baseline_risk__"

#: integrity QA flags evaluated inside the gates (plan 8.2 condition 1 / 8.3 condition 6).
QA_KEYS = (
    "all_complete", "hashes_match", "role_seeds_valid", "determinism_passed",
    "one_shot_locks", "ensemble_is_arithmetic_mean", "baseline_matches_frozen",
    "checkpoints_match_frozen",
)


# --- result containers ----------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """One whole-group paired bootstrap: identity point + 80% CI + raw draws."""
    point: np.ndarray        # (N_QUANTITIES,)
    ci_lower: np.ndarray     # (N_QUANTITIES,) two-sided 80%
    ci_upper: np.ndarray
    draws: np.ndarray        # (n_resamples, N_QUANTITIES), shared indices per row
    n_resamples: int
    seed: int
    confidence: float


@dataclass(frozen=True)
class ScreenDecision:
    """The single D2 development-screen adjudication (Gate TASK-X-DEV + Gate RISK-X-DEV)."""
    plan_id: str
    n_groups: int
    status: str
    passed: bool
    task: dict[str, Any]
    risk: dict[str, Any]
    metrics: dict[str, Any]
    baselines: dict[str, Any]
    bootstrap: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "n_groups": int(self.n_groups),
            "status": self.status,
            "passed": bool(self.passed),
            "gate_task_x_dev": self.task,
            "gate_risk_x_dev": self.risk,
            "metrics": self.metrics,
            "baseline_curves": self.baselines,
            "bootstrap": self.bootstrap,
        }


# --- truth-side quantities (legal ONLY after the burned lock) --------------------------------------


def compute_reactive_response(
    object_factual_future: np.ndarray,
    object_counterfactual_futures: np.ndarray,
) -> np.ndarray:
    """Per-group bool: any selected non-ego (slots 1..4) whose (s,d) future differs between
    the factual branch and any counterfactual branch beyond ``REACTIVE_TOLERANCE_M``.

    Mirrors the frozen pilot adjudication (``stage4d_pilot_adjudicate.compute_group_quantities``)
    bitwise; only present (factual-presence) non-ego cells count. Truth-side only.
    """
    factual = np.asarray(object_factual_future, dtype=np.float64)
    cf = np.asarray(object_counterfactual_futures, dtype=np.float64)
    factual_xy = factual[:, :, 1:, [CH_S, CH_D]]             # (G,20,4,2)
    factual_presence = factual[:, :, 1:, 0] > 0              # (G,20,4)
    cf_xy = cf[:, :, :, 1:, [CH_S, CH_D]]                    # (G,10,20,4,2)
    diff = np.abs(cf_xy - factual_xy[:, None]).max(axis=-1)  # (G,10,20,4)
    diff = np.where(factual_presence[:, None], diff, 0.0)
    return (diff.max(axis=(1, 2, 3)) > REACTIVE_TOLERANCE_M)  # (G,)


def compute_screen_truth(
    object_counterfactual_futures: np.ndarray,
    object_factual_future: np.ndarray,
    history_last_xy: np.ndarray,
    evaluation_mask: np.ndarray,
    j_hat: np.ndarray,
) -> dict[str, Any]:
    """True consequences + decision labels + reactive response for the screen (truth-side).

    Reuses the frozen ``stage4d_external_training.compute_true_labels`` (itself the frozen
    ``stage4c_objective`` consequence) for J_true/regret/oracle/idle/random/y_high, and the
    frozen pilot reactive-response computation. No outcome is used to accept/reject groups.
    """
    cf = np.asarray(object_counterfactual_futures, dtype=np.float64)
    counterfactual_xy = cf[..., [CH_S, CH_D]]                # (G,10,20,5,2)
    labels = T.compute_true_labels(
        counterfactual_xy=counterfactual_xy,
        history_last_xy=np.asarray(history_last_xy, dtype=np.float64),
        inference_mask=np.asarray(evaluation_mask, dtype=bool),
        j_hat=np.asarray(j_hat, dtype=np.float64))
    reactive = compute_reactive_response(object_factual_future,
                                         object_counterfactual_futures)
    return {
        "j_true": np.asarray(labels["j_true"], dtype=np.float64),
        "a_hat": np.asarray(labels["a_hat"], dtype=np.int64),
        "a_star": np.asarray(labels["a_star"], dtype=np.int64),
        "regret": np.asarray(labels["regret"], dtype=np.float64),
        "idle_regret": np.asarray(labels["idle_regret"], dtype=np.float64),
        "random_regret": np.asarray(labels["random_regret"], dtype=np.float64),
        "oracle_action": np.asarray(labels["oracle_action"], dtype=np.int64),
        "y_high": np.asarray(labels["y_high"], dtype=np.int64),
        "reactive": reactive,
    }


# --- quantities validation ---------------------------------------------------------------------------


def _validate_quantities(quantities: Mapping[str, Any]) -> int:
    """Fail closed on any missing/misaligned/non-finite per-group quantity."""
    for key in REQUIRED_ARRAY_KEYS:
        if key not in quantities:
            raise ValueError(f"screen quantities missing array {key!r}")
    regret = np.asarray(quantities["regret"], dtype=np.float64).ravel()
    groups = int(regret.size)
    if groups < 2:
        raise ValueError("screen quantities need at least 2 groups")
    if int(quantities.get("n_groups", groups)) != groups:
        raise ValueError("n_groups does not match the regret axis")
    for key in ("group_seed", "y_high", "random_regret", "idle_regret", "ensemble_risk",
                "tstar_layer", "interaction_layer", "a_star", "t_star",
                "interaction_min_distance"):
        arr = np.asarray(quantities[key]).ravel()
        if arr.shape != (groups,):
            raise ValueError(f"quantity {key!r} must align as ({groups},)")
    reactive = np.asarray(quantities["reactive"]).ravel()
    if reactive.shape != (groups,):
        raise ValueError(f"quantity 'reactive' must align as ({groups},)")
    for key in ("regret", "random_regret", "idle_regret", "ensemble_risk",
                "interaction_min_distance"):
        if not np.isfinite(np.asarray(quantities[key], dtype=np.float64)).all():
            raise ValueError(f"quantity {key!r} must be finite")
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)
    if by_seed.shape != (N_MODEL_SEEDS, groups) or not np.isfinite(by_seed).all():
        raise ValueError(f"risk_by_seed must be finite and align as ({N_MODEL_SEEDS}, {groups})")
    j_hat = np.asarray(quantities["J_hat"], dtype=np.float64)
    j_true = np.asarray(quantities["J_true"], dtype=np.float64)
    if j_hat.shape != (groups, N_ACTIONS) or j_true.shape != j_hat.shape:
        raise ValueError("J_hat/J_true must align as (G, 10)")
    if not np.isfinite(j_hat).all() or not np.isfinite(j_true).all():
        raise ValueError("J_hat/J_true must be finite")
    baselines = quantities.get("baseline_risks")
    if not isinstance(baselines, Mapping):
        raise ValueError("screen quantities missing baseline_risks mapping")
    strongest = str(quantities.get("strongest_baseline", ""))
    for name in ALL_BASELINE_NAMES:
        if name not in baselines:
            raise ValueError(f"baseline_risks missing {name!r}")
        risk = np.asarray(baselines[name], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"baseline risk {name!r} must be finite and group-aligned")
    if strongest not in STANDARD_BASELINES:
        raise ValueError(
            f"strongest_baseline {strongest!r} is not a frozen eligible candidate")
    qa = quantities.get("qa")
    if not isinstance(qa, Mapping) or any(key not in qa for key in QA_KEYS):
        raise ValueError(f"screen quantities qa must carry {list(QA_KEYS)}")
    for flag in ("projection_nondegenerate", "eval_mask_nondegenerate"):
        if flag not in quantities:
            raise ValueError(f"screen quantities missing validity flag {flag!r}")
    return groups


# --- the joint quantity vector ------------------------------------------------------------------------


def compute_screen_quantity_vector(
    index: np.ndarray, quantities: Mapping[str, Any]
) -> np.ndarray:
    """Evaluate all 32 quantities on ONE whole-group resample ``index``.

    ``index`` is a with-replacement integer resample into the screen groups, so every metric of
    the draw stays coupled through the same group multiplicities. Improvement quantities are NaN
    when the frozen primary-baseline denominator is non-positive (the gate turns that into
    ``STAGE4D_EXTERNAL_INVALID_ANALYSIS``). The primary baseline is the validation-frozen one;
    it is never re-selected inside a draw.
    """
    index = np.asarray(index, dtype=np.int64)
    reg = np.asarray(quantities["regret"], dtype=np.float64)[index]
    y_high = np.asarray(quantities["y_high"], dtype=np.int64)[index]
    ens = np.asarray(quantities["ensemble_risk"], dtype=np.float64)[index]
    baselines = quantities["baseline_risks"]
    base = np.asarray(baselines[str(quantities["strongest_baseline"])],
                      dtype=np.float64)[index]
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)[:, index]

    order_ens = np.argsort(ens, kind="stable")
    order_base = np.argsort(base, kind="stable")
    a_base = s4cs.aurc_regret(order_base, reg)
    a_ens = s4cs.aurc_regret(order_ens, reg)
    r80_base = s4cs.regret_at_coverage(order_base, reg, COVERAGE_MAIN)
    r80_ens = s4cs.regret_at_coverage(order_ens, reg, COVERAGE_MAIN)

    q = np.full(N_QUANTITIES, np.nan, dtype=np.float64)
    q[Q_I_AURC] = s4cs.improvement(a_base, a_ens)
    q[Q_I_REGRET80] = s4cs.improvement(r80_base, r80_ens)
    q[Q_PR] = s4cs.high_regret_pr(ens, y_high)
    q[Q_SPEARMAN_BRANCH] = s4cs.spearman(
        np.asarray(quantities["J_hat"], dtype=np.float64)[index],
        np.asarray(quantities["J_true"], dtype=np.float64)[index])
    rand = np.asarray(quantities["random_regret"], dtype=np.float64)[index]
    idle = np.asarray(quantities["idle_regret"], dtype=np.float64)[index]
    rand_mean = float(rand.mean())
    base_mean = float(reg.mean())
    q[Q_BASE_VS_RANDOM] = ((rand_mean - base_mean) / rand_mean
                           if rand_mean > 0 else float("nan"))
    q[Q_AURC_BASELINE] = a_base
    q[Q_R80_BASELINE] = r80_base
    q[Q_BASE_MEAN_REGRET] = base_mean
    q[Q_IDLE_MEAN_REGRET] = float(idle.mean())
    q[Q_RANDOM_MEAN_REGRET] = rand_mean
    q[Q_HIGH_REGRET_FRAC] = float((reg > HIGH_REGRET_THRESHOLD).mean())

    for i in range(N_MODEL_SEEDS):
        order_seed = np.argsort(by_seed[i], kind="stable")
        q[Q_SEED_I_AURC + i] = s4cs.improvement(
            a_base, s4cs.aurc_regret(order_seed, reg))
        q[Q_SEED_I_R80 + i] = s4cs.improvement(
            r80_base, s4cs.regret_at_coverage(order_seed, reg, COVERAGE_MAIN))

    # three standard baselines' ABSOLUTE curves (plan 8.3 note: report point + 80% CI each).
    for bi, name in enumerate(STANDARD_BASELINES):
        risk = np.asarray(baselines[name], dtype=np.float64)[index]
        order_b = np.argsort(risk, kind="stable")
        slot = Q_BASELINE_CURVE + 3 * bi
        q[slot] = s4cs.aurc_regret(order_b, reg)
        q[slot + 1] = s4cs.regret_at_coverage(order_b, reg, COVERAGE_MAIN)
        q[slot + 2] = s4cs.high_regret_pr(risk, y_high)

    q[Q_AURC_MODEL] = a_ens
    q[Q_R80_MODEL] = r80_ens
    return q


# --- the one paired bootstrap -------------------------------------------------------------------------


def paired_group_bootstrap(
    quantities: Mapping[str, Any],
    *,
    seed: int = SCREEN_BOOTSTRAP_ENTROPY,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> BootstrapResult:
    """10,000-draw whole-group paired bootstrap with shared indices and two-sided 80% CIs.

    EVERY quantity of a draw is recomputed from the SAME whole-group resample index vector; the
    identity evaluation is the point estimate; the two-sided 80% CI is the (0.10, 0.90) resample
    quantile band. The confidence level is frozen at ``CONFIDENCE_SCREEN`` (0.80) — never the
    confirm 0.95 and never re-parameterised per call site.
    """
    groups = _validate_quantities(quantities)
    generator = np.random.default_rng(int(seed))
    identity = np.arange(groups, dtype=np.int64)
    point = compute_screen_quantity_vector(identity, quantities)
    draws = np.empty((int(n_resamples), N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = compute_screen_quantity_vector(index, quantities)
    alpha = (1.0 - CONFIDENCE_SCREEN) / 2.0
    return BootstrapResult(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws,
        n_resamples=int(n_resamples),
        seed=int(seed),
        confidence=float(CONFIDENCE_SCREEN),
    )


# --- the gates -----------------------------------------------------------------------------------------


def _named(vector: np.ndarray) -> dict[str, float]:
    return {name: float(vector[i]) for i, name in enumerate(_QUANTITY_NAMES)}


def evaluate_screen_gates(
    quantities: Mapping[str, Any], bootstrap: BootstrapResult
) -> ScreenDecision:
    """Adjudicate Gate TASK-X-DEV and Gate RISK-X-DEV exactly once (fail-closed).

    Terminal-state priority (plan 11.3 + 8.4): an integrity/QA failure or a degenerate
    denominator / projection / evaluation mask -> ``STAGE4D_EXTERNAL_INVALID_ANALYSIS``; a valid
    adjudication with any task or risk condition failed -> ``STAGE4D_EXTERNAL_DEV_NOT_READY``;
    both gates pass -> ``STAGE4D_EXTERNAL_DEV_READY_FOR_POWER``. NaN quantities fail their
    conditions by construction; denominators are never repaired.
    """
    groups = _validate_quantities(quantities)
    point = np.asarray(bootstrap.point, dtype=np.float64)
    ci_lower = np.asarray(bootstrap.ci_lower, dtype=np.float64)
    ci_upper = np.asarray(bootstrap.ci_upper, dtype=np.float64)
    draws = np.asarray(bootstrap.draws, dtype=np.float64)
    if point.shape != (N_QUANTITIES,) or ci_lower.shape != (N_QUANTITIES,) \
            or ci_upper.shape != (N_QUANTITIES,) \
            or draws.shape != (int(bootstrap.n_resamples), N_QUANTITIES):
        raise ValueError("bootstrap result does not match the frozen quantity vector")
    if float(bootstrap.confidence) != float(CONFIDENCE_SCREEN):
        raise ValueError("screen CI level is frozen at 0.80")

    qa = {key: bool(quantities["qa"][key]) for key in QA_KEYS}
    qa_passed = all(qa.values())

    baseline_valid = bool(
        np.isfinite([point[Q_AURC_BASELINE], point[Q_R80_BASELINE]]).all()
        and point[Q_AURC_BASELINE] > 0.0 and point[Q_R80_BASELINE] > 0.0)
    projection_ok = bool(quantities["projection_nondegenerate"])
    mask_ok = bool(quantities["eval_mask_nondegenerate"])

    # --- Gate TASK-X-DEV (plan 8.2, 8 conditions) ------------------------------------------
    oracle = np.asarray(quantities["a_star"], dtype=np.int64)
    oracle_shares = [float((oracle == a).mean()) for a in range(N_ACTIONS)]
    n_oracle = int(sum(s >= GATE_TASK_MIN_ORACLE_FRAC for s in oracle_shares))
    reactive_frac = float(np.asarray(quantities["reactive"], dtype=bool).mean())
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)
    stratum_fracs = {
        "tstar_low": float((tstar == 0).mean()), "tstar_high": float((tstar == 1).mean()),
        "interaction_low": float((inter == 0).mean()),
        "interaction_high": float((inter == 1).mean()),
    }
    base_mean = float(point[Q_BASE_MEAN_REGRET])
    idle_mean = float(point[Q_IDLE_MEAN_REGRET])
    task_conditions = {
        "c1_completeness_and_qa": bool(qa_passed and groups == DEV_SCREEN_GROUPS),
        "c2_high_regret_frac_ge_15pct": bool(
            point[Q_HIGH_REGRET_FRAC] >= GATE_TASK_MIN_HIGH_REGRET_FRAC),
        "c3_at_least_3_oracle_actions_ge_5pct": bool(
            n_oracle >= GATE_TASK_MIN_DISTINCT_ORACLE),
        "c4_branch_spearman_ci_lower_positive": bool(ci_lower[Q_SPEARMAN_BRANCH] > 0.0),
        "c5_base_beats_random_and_not_worse_than_idle": bool(
            point[Q_BASE_VS_RANDOM] > 0.0 and ci_lower[Q_BASE_VS_RANDOM] > 0.0
            and np.isfinite([base_mean, idle_mean]).all()
            and base_mean <= idle_mean + BASELINE_TIE_TOLERANCE),
        "c6_reactive_response_ge_20pct": bool(reactive_frac >= GATE_TASK_MIN_REACTIVE_FRAC),
        "c7_strata_each_ge_10pct": bool(
            min(stratum_fracs.values()) >= GATE_TASK_MIN_STRATUM_FRAC),
        "c8_denominators_projection_mask_valid": bool(
            baseline_valid and projection_ok and mask_ok),
    }
    task_passed = bool(all(task_conditions.values()))

    # --- Gate RISK-X-DEV (plan 8.3, 6 conditions) -------------------------------------------
    seeds_both = int(sum(
        point[SEED_I_AURC_INDEX[i]] > 0.0 and point[SEED_I_R80_INDEX[i]] > 0.0
        for i in range(N_MODEL_SEEDS)))
    risk_conditions = {
        "r1_i_aurc_regret": bool(point[Q_I_AURC] >= I_AURC_MIN and ci_lower[Q_I_AURC] > 0.0),
        "r2_i_regret80": bool(point[Q_I_REGRET80] >= I_REGRET80_MIN
                              and ci_lower[Q_I_REGRET80] > 0.0),
        "r3_pr_auc_minus_prevalence": bool(point[Q_PR] > 0.0 and ci_lower[Q_PR] > 0.0),
        "r4_seed_consistency_both_positive": bool(seeds_both >= MIN_POSITIVE_SEEDS),
        "r5_baseline_denominators_valid": bool(baseline_valid),
        "r6_no_leak_no_baseline_swap_no_model_swap": bool(qa_passed),
    }
    risk_passed = bool(all(risk_conditions.values()))

    # --- terminal-state priority (plan 11.3 + 8.4) -------------------------------------------
    invalid = (not qa_passed) or (not baseline_valid) \
        or (not projection_ok) or (not mask_ok)
    if invalid:
        status = STATUS_INVALID_ANALYSIS
    elif not task_passed:
        status = STATUS_DEV_NOT_READY
    elif not risk_passed:
        status = STATUS_DEV_NOT_READY
    else:
        status = STATUS_DEV_READY

    # --- three standard baselines reported with point + 80% CI (plan 8.3 note) --------------
    baseline_curves: dict[str, Any] = {}
    for bi, name in enumerate(STANDARD_BASELINES):
        slot = Q_BASELINE_CURVE + 3 * bi
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
    # explicit disclosure if the screen's strongest standard baseline is not the frozen one.
    screen_aurc = {name: float(point[Q_BASELINE_CURVE + 3 * bi])
                   for bi, name in enumerate(STANDARD_BASELINES)}
    screen_strongest = min(screen_aurc, key=lambda k: screen_aurc[k])
    frozen_baseline = str(quantities["strongest_baseline"])
    baseline_disclosure = {
        "frozen_primary_baseline": frozen_baseline,
        "screen_strongest_standard_baseline": screen_strongest,
        "screen_strongest_differs_from_frozen": bool(screen_strongest != frozen_baseline),
        "note": ("only the validation-frozen baseline is adjudicated; a different screen "
                 "strongest baseline is disclosed but never substituted (plan 8.3)"),
    }

    return ScreenDecision(
        plan_id=PLAN_ID,
        n_groups=int(groups),
        status=status,
        passed=bool(status == STATUS_DEV_READY),
        task={
            "conditions": task_conditions,
            "passed": task_passed,
            "high_regret_frac": float(point[Q_HIGH_REGRET_FRAC]),
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
                "aurc_regret": float(point[Q_AURC_BASELINE]),
                "regret_at_80": float(point[Q_R80_BASELINE]),
            },
            "seeds_both_directions_positive": seeds_both,
            "min_positive_seeds": int(MIN_POSITIVE_SEEDS),
        },
        metrics={
            "point": _named(point),
            "ci_lower": _named(ci_lower),
            "ci_upper": _named(ci_upper),
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


# --- screen artifact paths ----------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def screen_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every one-shot screen path (plan 8.1 + 14)."""
    run_dir = _resolve(config.get("run_dir", "experiments/stage4_reliability/stage4d_external"))
    role = str(config.get("screen_role", "development_screen"))
    dev = run_dir / "development"
    role_dir = run_dir / "roles" / role
    split_dir = dev / "screen_split"
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
        "final_selection": dev / "final_selection.json",
        "direct_fit_receipt": dev / "models" / T.FIT_RECEIPT_NAME,
        "predictor_stack": dev / "base_models" / "predictor_stack.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_frozen_final_selection(paths: Mapping[str, Path]) -> dict[str, Any]:
    """The screen exists only because validation PASSED and the selection is frozen."""
    if not paths["final_selection"].is_file():
        raise ValueError(
            "screen is refused: final_selection.json is missing — the validation gate must "
            "pass and freeze before any screen artifact is produced")
    final = _read_json(paths["final_selection"])
    if final.get("status") != T.STATUS_VALIDATION_QUALIFIED:
        raise ValueError(
            f"screen is refused: validation status is {final.get('status')!r}, not "
            f"{T.STATUS_VALIDATION_QUALIFIED}; the screen is never collected")
    return final


# --- plan 8.1 steps 3-5: freeze the screen scores create-once ----------------------------------------


def freeze_screen_scores(
    config: Mapping[str, Any],
    *,
    scores: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> Path:
    """Freeze the screen scores + receipts create-once (plan 8.1 steps 3-5).

    ``scores`` is the driver's torch-side output (ensemble + per-seed + MU/CLASS + j_hat + the
    baseline group risks + strata inputs), computed from ``inference_inputs.npz`` ONLY. ``scores``
    must carry: group_seed, t_star, interaction_min_distance, j_hat (G,10), ensemble_risk (G,),
    risk_by_seed (5,G), mu_only_ensemble (G,), class_only_ensemble (G,), valid_mask (G,) and a
    ``baselines`` mapping over the four baseline names. ``determinism`` is the >=32-group repeated
    -inference receipt (``bitwise_identical`` must be True). A rerun verifies hashes and never
    overwrites. This function never touches the sealed truth.
    """
    paths = screen_artifact_paths(config)
    final = _load_frozen_final_selection(paths)
    npz_path = paths["scores_npz"]
    receipt_path = paths["scores_receipt"]
    det_path = paths["determinism_receipt"]
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if not npz_path.is_file() or sha256_file(npz_path) != receipt["scores_sha256"]:
            raise ValueError(f"screen scores drifted from their create-once receipt: {npz_path}")
        if not det_path.is_file():
            raise ValueError("screen determinism receipt missing after score freeze")
        return npz_path
    if npz_path.exists():
        raise RuntimeError(f"screen scores exist without their create-once receipt: {npz_path}")

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
            f"{int((~valid).sum())} screen groups have non-finite features; the screen contract "
            "rejects zero-padding, so scoring is refused")
    if j_hat.shape != (groups, N_ACTIONS) or ensemble.shape != (groups,) \
            or by_seed.shape != (N_MODEL_SEEDS, groups) \
            or distance.shape != (groups,) or t_star.shape != (groups,):
        raise ValueError("screen score arrays must align on the group axis")
    if not np.isfinite(j_hat).all() or not np.isfinite(ensemble).all() \
            or not np.isfinite(by_seed).all():
        raise ValueError("non-finite screen score")
    if not np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15):
        raise ValueError("the ensemble must be exactly the arithmetic mean of five seeds")
    if not bool(determinism.get("bitwise_identical")):
        raise ValueError("the repeated inference was not elementwise identical; the screen "
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
            raise ValueError(f"screen baseline risk {name!r} must be finite and aligned")
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
        "role": "development_screen",
        "n_groups": int(groups),
        "scores_path": str(npz_path),
        "scores_sha256": sha256_file(npz_path),
        "inference_inputs_sha256": sha256_file(paths["inference_inputs"])
        if paths["inference_inputs"].is_file() else "",
        "split_receipt_sha256": sha256_file(paths["split_receipt"])
        if paths["split_receipt"].is_file() else "",
        "final_selection_sha256": sha256_file(paths["final_selection"]),
        "direct_fit_receipt_sha256": sha256_file(paths["direct_fit_receipt"])
        if paths["direct_fit_receipt"].is_file() else "",
        "strongest_standard_baseline": str(final["strongest_standard_baseline"]),
        "model_seeds": [int(s) for s in T.MODEL_SEEDS],
        "ensemble_definition": "arithmetic_mean_of_five_seeds",
        "primary_score_definition": final.get("primary_score_definition", ""),
        "determinism_receipt_sha256": sha256_file(det_path),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return npz_path


def _verify_scores_chain(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Re-hash the frozen scores against its create-once receipt (fail-closed)."""
    receipt = _read_json(paths["scores_receipt"])
    if not paths["scores_npz"].is_file() or sha256_file(paths["scores_npz"]) \
            != receipt["scores_sha256"]:
        raise ValueError("screen scores drifted from their create-once receipt")
    if not paths["determinism_receipt"].is_file():
        raise ValueError("the repeated-inference determinism receipt is missing")
    determinism = _read_json(paths["determinism_receipt"])
    if not determinism.get("bitwise_identical"):
        raise ValueError("the repeated inference was not elementwise identical")
    return receipt


# --- plan 8.1 step 6: burn the one-shot lock ----------------------------------------------------------


def burn_screen_lock(config: Mapping[str, Any]) -> Path:
    """Create ``development_screen_used.lock`` exactly once, after the full score freeze.

    Refuses unless: the frozen final selection is VALIDATION_QUALIFIED; the scores hash-match
    their create-once receipt; and the >=32-group repeated-inference receipt is elementwise
    identical. A second call re-verifies the bindings against the current files and never
    overwrites. The lock NEVER reads the sealed truth bytes.
    """
    paths = screen_artifact_paths(config)
    final = _load_frozen_final_selection(paths)
    receipt = _verify_scores_chain(paths)
    bindings = {
        "scores_sha256": receipt["scores_sha256"],
        "scores_receipt_sha256": sha256_file(paths["scores_receipt"]),
        "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
        "final_selection_sha256": sha256_file(paths["final_selection"]),
    }
    lock_path = paths["lock"]
    if lock_path.exists():
        payload = _read_json(lock_path)
        if payload.get("plan_id") != PLAN_ID or payload.get("bindings") != bindings:
            raise ValueError(
                "the burned screen lock no longer matches the on-disk frozen artifacts "
                "(tampering or drift); the one-shot screen is compromised")
        return lock_path
    payload = {
        "plan_id": PLAN_ID,
        "role": "development_screen",
        "semantics": ("one-shot: the screen truth may be read only after this lock exists; all "
                      "screen scores are frozen and hash-bound above"),
        "strongest_standard_baseline": str(final["strongest_standard_baseline"]),
        "interaction_threshold": float(final["interaction_threshold"]),
        "bindings": bindings,
        "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path, payload)
    return lock_path


# --- plan 8.1 steps 7-8: open truth once, assemble, adjudicate ----------------------------------------


def assert_screen_adjudication_open(config: Mapping[str, Any]) -> None:
    """Fail closed unless the one-shot adjudication may legally start.

    Requires the burned ``development_screen_used.lock`` (truth is never read before the scores
    are frozen and the lock exists) and refuses any second adjudication (quantities NPZ /
    receipt / screen report already present).
    """
    paths = screen_artifact_paths(config)
    _load_frozen_final_selection(paths)
    if not paths["lock"].is_file():
        raise ValueError(
            "screen truth may not be read: development_screen_used.lock is not burned; freeze "
            "every score artifact and burn the lock first (plan 8.1)")
    for key in ("quantities_npz", "quantities_receipt", "report"):
        if paths[key].exists():
            raise FileExistsError(
                f"a second screen adjudication is refused; already exists: {paths[key]}")


def assemble_screen_group_quantities(config: Mapping[str, Any]) -> Path:
    """Open the sealed truth ONCE, build regret/strata/reactive, write the create-once quantities.

    Plan 8.1 steps 7 + the plan-9.1 per-group table. Refuses unless the one-shot lock is burned;
    re-verifies the full score hash chain; computes J_true/regret/oracle with the frozen
    consequence function and the reactive response with the frozen pilot computation; writes
    ``screen_group_quantities.npz`` and its receipt exactly once. Returns the NPZ path.
    """
    paths = screen_artifact_paths(config)
    assert_screen_adjudication_open(config)
    final = _load_frozen_final_selection(paths)
    score_receipt = _verify_scores_chain(paths)
    burn_screen_lock(config)  # re-verify the lock bindings against the current files
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
    labels = compute_screen_truth(cf, factual, history_last_xy, evaluation_mask, j_hat)
    j_true = labels["j_true"]

    threshold = float(final["interaction_threshold"])
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
    # Map-Frenet projection non-degenerate: the evaluated (s,d) positions span a nonzero range.
    valid_cells = evaluation_mask                                # (G,20,5)
    cf_xy = cf[..., [CH_S, CH_D]]                               # (G,10,20,5,2)
    if valid_cells.any():
        mask_b = np.broadcast_to(valid_cells[:, None, :, :], cf_xy.shape[:-1])  # (G,10,20,5)
        pts = cf_xy[mask_b]                                     # (K,2) evaluated positions
        span = float((pts.max(axis=0) - pts.min(axis=0)).max()) if pts.size else 0.0
    else:
        span = 0.0
    projection_nondegenerate = bool(np.isfinite(span) and span > 1e-6)
    # common evaluation mask non-degenerate: every group evaluates >=1 cell and the mask is
    # not globally empty.
    per_group_cells = valid_cells.reshape(groups, -1).any(axis=1) \
        if valid_cells.shape[0] == groups else np.zeros(groups, dtype=bool)
    eval_mask_nondegenerate = bool(valid_cells.size > 0 and valid_cells.mean() > 0.0
                                   and per_group_cells.all())

    # structural one-shot / no-swap bindings proven by the receipts above.
    frozen_checkpoints = {str(k): str(v["sha256"])
                          for k, v in final.get("checkpoints", {}).items()}
    fit_receipt = _read_json(paths["direct_fit_receipt"]) \
        if paths["direct_fit_receipt"].is_file() else {}
    fit_ckpt = {str(k): str(v) for k, v in fit_receipt.get("checkpoint_sha256", {}).items()}
    qa = {
        "all_complete": bool(
            groups == DEV_SCREEN_GROUPS and j_true.shape == (groups, N_ACTIONS)
            and j_hat.shape == (groups, N_ACTIONS) and truth_finite
            and int(score_receipt["n_groups"]) == groups),
        "hashes_match": True,  # _verify_scores_chain + lock re-verification passed above
        "role_seeds_valid": role_seeds_valid,
        "determinism_passed": True,  # burn_screen_lock re-verified the determinism receipt
        "one_shot_locks": True,
        "ensemble_is_arithmetic_mean": bool(
            by_seed.shape == (N_MODEL_SEEDS, groups)
            and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)),
        "baseline_matches_frozen": bool(
            str(final["strongest_standard_baseline"]) in STANDARD_BASELINES),
        "checkpoints_match_frozen": bool(
            frozen_checkpoints and fit_ckpt and frozen_checkpoints == fit_ckpt),
    }

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
        "role": "development_screen",
        "n_groups": int(groups),
        "screen_group_quantities_path": str(npz_path),
        "screen_group_quantities_sha256": sha256_file(npz_path),
        "strongest_standard_baseline": str(final["strongest_standard_baseline"]),
        "interaction_threshold": float(threshold),
        "interaction_threshold_source": final.get("interaction_threshold_source", ""),
        "tstar_stratum_split": TSTAR_SPLIT,
        "trim_fractions": [float(f) for f in TRIM_FRACTIONS],
        "map_family_note": ("the frozen collector records no finer map_id; the screen is one "
                            "Waymo training_20s -> MetaDrive family (map_family all 0). "
                            "FLAGGED plan-7.1 lists map_id but the collector meta has none."),
        "bootstrap_entropy": SCREEN_BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "confidence_screen": CONFIDENCE_SCREEN,
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
            "final_selection_sha256": sha256_file(paths["final_selection"]),
            "statistics_code_sha256": sha256_file(
                ROOT / "code/ac_twin/stage4d_external_statistics.py"),
        },
        "lock_sha256": sha256_file(paths["lock"]),
        "lock_burned_at": lock_payload.get("burned_at", ""),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["quantities_receipt"], receipt)
    return npz_path


def load_screen_group_quantities(
    npz_path: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    """Load the create-once quantities NPZ + receipt into the gate-input mapping (plan 9.1)."""
    with np.load(Path(npz_path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    receipt = _read_json(Path(receipt_path))
    baselines = {}
    for key in list(data):
        if key.startswith(_BASELINE_NPZ_PREFIX):
            baselines[key[len(_BASELINE_NPZ_PREFIX):]] = data.pop(key)
    quantities: dict[str, Any] = dict(data)
    quantities["baseline_risks"] = baselines
    quantities["strongest_baseline"] = str(receipt["strongest_standard_baseline"])
    quantities["n_groups"] = int(np.asarray(data["n_groups"]).item())
    quantities["interaction_threshold"] = float(np.asarray(data["interaction_threshold"]).item())
    quantities["qa"] = {key: bool(value) for key, value in receipt["qa"].items()}
    quantities["projection_nondegenerate"] = bool(receipt["projection_nondegenerate"])
    quantities["eval_mask_nondegenerate"] = bool(receipt["eval_mask_nondegenerate"])
    _validate_quantities(quantities)
    return quantities


def adjudicate_screen(config: Mapping[str, Any]) -> ScreenDecision:
    """The single screen adjudication (plan 8.1 steps 7-8): assemble -> bootstrap -> gates -> report.

    Create-once: a second call with the formal report present is refused. Writes
    ``development_screen_report.json`` exactly once. Never reruns the screen, never changes the
    model, never starts the confirm.
    """
    paths = screen_artifact_paths(config)
    if paths["report"].exists():
        raise FileExistsError(
            f"a second screen adjudication is refused; report exists: {paths['report']}")
    quantities_npz = assemble_screen_group_quantities(config)
    quantities = load_screen_group_quantities(quantities_npz, paths["quantities_receipt"])
    bootstrap = paired_group_bootstrap(quantities)
    decision = evaluate_screen_gates(quantities, bootstrap)
    report = decision.to_dict()
    report["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["quantities_receipt_sha256"] = sha256_file(paths["quantities_receipt"])
    write_once_json(paths["report"], report)
    return decision


__all__ = [
    "ALL_BASELINE_NAMES",
    "BASELINE_COMBINED_TOP2_MAX",
    "BASELINE_MARGIN",
    "BASELINE_SEED_MARGIN_STD",
    "BASELINE_SEED_VOTE_ENTROPY",
    "BOOTSTRAP_RESAMPLES",
    "BootstrapResult",
    "CONFIDENCE_SCREEN",
    "COVERAGE_MAIN",
    "DEV_SCREEN_GROUPS",
    "GATE_TASK_MIN_DISTINCT_ORACLE",
    "GATE_TASK_MIN_HIGH_REGRET_FRAC",
    "GATE_TASK_MIN_ORACLE_FRAC",
    "GATE_TASK_MIN_REACTIVE_FRAC",
    "GATE_TASK_MIN_STRATUM_FRAC",
    "HIGH_REGRET_THRESHOLD",
    "I_AURC_MIN",
    "I_REGRET80_MIN",
    "MIN_POSITIVE_SEEDS",
    "N_MODEL_SEEDS",
    "N_QUANTITIES",
    "PLAN_ID",
    "QA_KEYS",
    "Q_AURC_BASELINE",
    "Q_AURC_MODEL",
    "Q_BASELINE_CURVE",
    "Q_BASE_MEAN_REGRET",
    "Q_BASE_VS_RANDOM",
    "Q_HIGH_REGRET_FRAC",
    "Q_IDLE_MEAN_REGRET",
    "Q_I_AURC",
    "Q_I_REGRET80",
    "Q_PR",
    "Q_R80_BASELINE",
    "Q_R80_MODEL",
    "Q_RANDOM_MEAN_REGRET",
    "Q_SEED_I_AURC",
    "Q_SEED_I_R80",
    "Q_SPEARMAN_BRANCH",
    "REACTIVE_TOLERANCE_M",
    "REQUIRED_ARRAY_KEYS",
    "SCREEN_BOOTSTRAP_ENTROPY",
    "SEED_I_AURC_INDEX",
    "SEED_I_R80_INDEX",
    "STANDARD_BASELINES",
    "STATUS_DEV_NOT_READY",
    "STATUS_DEV_READY",
    "STATUS_INVALID_ANALYSIS",
    "ScreenDecision",
    "TRIM_FRACTIONS",
    "TSTAR_SPLIT",
    "adjudicate_screen",
    "assemble_screen_group_quantities",
    "assert_screen_adjudication_open",
    "burn_screen_lock",
    "compute_reactive_response",
    "compute_screen_quantity_vector",
    "compute_screen_truth",
    "evaluate_screen_gates",
    "freeze_screen_scores",
    "load_screen_group_quantities",
    "paired_group_bootstrap",
    "screen_artifact_paths",
]
