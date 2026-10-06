"""Stage 4C-R2 one-shot development-screen statistics and gate adjudication.

Plan ``AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`` sections 9 and 11.

The statistical unit is the 10-action decision GROUP. The ONE paired bootstrap runs
10,000 whole-group resamples with entropy ``2026092702``; EVERY metric of a draw is
recomputed from the SAME resample index vector, and the two-sided 80% CI is read off the
resample quantiles. Metric definitions are REUSED bitwise from
``stage4cp_statistics``/``stage4c_statistics`` (AURC_REGRET, REGRET@80,
PR-AUC-prevalence, fail-closed relative improvement, Spearman); nothing is redefined.

Frozen quantity-vector layout (``N_QUANTITIES = 32``)::

    0   I_AURC_REGRET                 (Direct ensemble vs frozen strongest baseline)
    1   I_REGRET@80
    2   PR-AUC - prevalence           (ensemble, high-regret label)
    3   branch-level Spearman(J_hat, J_true)
    4   base-vs-RANDOM_ACTION mean-regret improvement
    5   trim-1% aggregate I_AURC_REGRET
    6   trim-5% aggregate I_AURC_REGRET
    7   t_star-low stratum I_AURC     (t_star < 13; fixed [10,15] split, Stage 4C contract)
    8   t_star-high stratum I_AURC    (t_star >= 13)
    9   interaction-low stratum I_AURC  (min distance <= train-frozen threshold)
    10  interaction-high stratum I_AURC (min distance >  threshold)
    11  Spearman(ensemble, MARGIN_ONLY)
    12  AURC_REGRET of the frozen baseline   (fail-closed denominator)
    13  REGRET@80 of the frozen baseline     (fail-closed denominator)
    14  base mean regret
    15  ALWAYS_IDLE mean regret
    16  RANDOM_ACTION mean regret
    17  high-regret fraction (regret > 0.01)
    18..22  per-seed I_AURC_REGRET   (five frozen model seeds)
    23..27  per-seed I_REGRET@80
    28  MU_ONLY I_AURC   (non-blocking ablation)
    29  MU_ONLY I_REGRET@80
    30  CLASS_ONLY I_AURC   (non-blocking ablation)
    31  CLASS_ONLY I_REGRET@80

One-shot order (plan 9.3) is enforced by IO, not by convention:
``freeze_screen_direct_scores`` writes/hashes the Direct score freeze create-once;
``burn_screen_lock`` refuses until every score-side artifact and both determinism
receipts exist and hash-match; ``assemble_screen_group_quantities`` REFUSES to read the
screen truth until ``development_screen_used.lock`` exists, and refuses any second
adjudication (quantities NPZ / receipt / report already present). Baseline denominators
are fail-closed: a non-positive or non-finite AURC_REGRET / REGRET@80 of the frozen
strongest baseline yields NaN improvements and the gate returns
``STAGE4CR2_INVALID_ANALYSIS`` — never an epsilon, an absolute difference or a baseline
swap.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_protocol as s4cpp
from ac_twin import stage4cp_statistics as s4cs
from ac_twin import stage4cr2_protocol as s4cr2
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = Path(__file__).resolve().parents[2]

# --- frozen quantity-vector layout ---------------------------------------------------------
Q_I_AURC = 0
Q_I_REGRET80 = 1
Q_PR = 2
Q_SPEARMAN_BRANCH = 3
Q_BASE_VS_RANDOM = 4
Q_TRIM1 = 5
Q_TRIM5 = 6
Q_TSTAR_LOW = 7
Q_TSTAR_HIGH = 8
Q_INTERACT_LOW = 9
Q_INTERACT_HIGH = 10
Q_SPEARMAN_MARGIN = 11
Q_AURC_BASELINE = 12
Q_R80_BASELINE = 13
Q_BASE_MEAN_REGRET = 14
Q_IDLE_MEAN_REGRET = 15
Q_RANDOM_MEAN_REGRET = 16
Q_HIGH_REGRET_FRAC = 17
Q_SEED_I_AURC = 18   # 18..22, in the frozen MODEL_SEEDS order
Q_SEED_I_R80 = 23    # 23..27
Q_MU_I_AURC = 28
Q_MU_I_R80 = 29
Q_CLS_I_AURC = 30
Q_CLS_I_R80 = 31
N_QUANTITIES = 32
N_SEEDS = len(s4cr2.MODEL_SEEDS)

STRATA_INDEX = (Q_TSTAR_LOW, Q_TSTAR_HIGH, Q_INTERACT_LOW, Q_INTERACT_HIGH)
TRIM_INDEX = (Q_TRIM1, Q_TRIM5)
SEED_I_AURC_INDEX = tuple(range(Q_SEED_I_AURC, Q_SEED_I_AURC + N_SEEDS))
SEED_I_R80_INDEX = tuple(range(Q_SEED_I_R80, Q_SEED_I_R80 + N_SEEDS))

_QUANTITY_NAMES = (
    "i_aurc_regret", "i_regret80", "pr_auc_minus_prevalence", "spearman_branch",
    "base_vs_random_improvement", "trim1_i_aurc", "trim5_i_aurc",
    "tstar_low_i_aurc", "tstar_high_i_aurc", "interaction_low_i_aurc",
    "interaction_high_i_aurc", "spearman_ensemble_margin", "aurc_baseline",
    "regret80_baseline", "base_mean_regret", "idle_mean_regret",
    "random_mean_regret", "high_regret_frac",
) + tuple(f"seed{s}_i_aurc" for s in s4cr2.MODEL_SEEDS) \
    + tuple(f"seed{s}_i_regret80" for s in s4cr2.MODEL_SEEDS) \
    + ("mu_only_i_aurc", "mu_only_i_regret80", "class_only_i_aurc", "class_only_i_regret80")

COVERAGE_MAIN = s4cpp.COVERAGE_MAIN  # 0.80, frozen

#: fixed [t_star_low, t_star_high] stratum split inherited bitwise from the Stage 4C
#: screen contract (``t_star >= 13`` is the high stratum); never re-fit on any role.
TSTAR_SPLIT = 13

STATUS_TASK_QUALIFIED = "TASK_R2_QUALIFIED"

#: per-group arrays the quantities mapping must carry (all aligned on the group axis).
REQUIRED_ARRAY_KEYS = (
    "group_seed", "regret", "y_high", "random_regret", "idle_regret",
    "ensemble_risk", "risk_by_seed", "mu_only_ensemble", "class_only_ensemble",
    "tstar_layer", "interaction_layer", "J_hat", "J_true", "a_star",
    "t_star", "interaction_min_distance",
)

#: baseline risk arrays persisted in the screen NPZ (prefix stripped on load).
_BASELINE_NPZ_PREFIX = "baseline_risk__"
ALL_BASELINE_NAMES = (
    s4cr2.BASELINE_MARGIN, s4cr2.BASELINE_SEED_VOTE_ENTROPY,
    s4cr2.BASELINE_SEED_MARGIN_STD, s4cr2.BASELINE_COMBINED_TOP2_MAX,
    s4cr2.BASELINE_CONSTANT_RANDOM,
)

#: QA flags evaluated inside the gates (plan 9.1 condition 1 / plan 9.2 condition 9).
QA_KEYS = (
    "all_complete", "hashes_match", "role_seeds_valid", "determinism_passed",
    "one_shot_locks", "ensemble_is_arithmetic_mean",
)


# --- result containers -----------------------------------------------------------------------


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
    """The single screen adjudication outcome (Gate TASK-R2 + Gate RISK-DECISION-R2_dev)."""
    plan_id: str
    n_groups: int
    status: str
    passed: bool
    task: dict[str, Any]
    risk: dict[str, Any]
    metrics: dict[str, Any]
    ablations: dict[str, Any]
    bootstrap: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "n_groups": int(self.n_groups),
            "status": self.status,
            "passed": bool(self.passed),
            "gate_task_r2": self.task,
            "gate_risk_decision_r2_dev": self.risk,
            "metrics": self.metrics,
            "ablations": self.ablations,
            "bootstrap": self.bootstrap,
        }


# --- quantities validation ---------------------------------------------------------------------


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
                "mu_only_ensemble", "class_only_ensemble", "tstar_layer",
                "interaction_layer", "a_star", "t_star", "interaction_min_distance"):
        arr = np.asarray(quantities[key]).ravel()
        if arr.shape != (groups,):
            raise ValueError(f"quantity {key!r} must align as ({groups},)")
    for key in ("regret", "random_regret", "idle_regret", "ensemble_risk",
                "mu_only_ensemble", "class_only_ensemble", "J_hat", "J_true",
                "interaction_min_distance"):
        if not np.isfinite(np.asarray(quantities[key], dtype=np.float64)).all():
            raise ValueError(f"quantity {key!r} must be finite")
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)
    if by_seed.shape != (N_SEEDS, groups) or not np.isfinite(by_seed).all():
        raise ValueError(f"risk_by_seed must be finite and align as ({N_SEEDS}, {groups})")
    j_hat = np.asarray(quantities["J_hat"], dtype=np.float64)
    j_true = np.asarray(quantities["J_true"], dtype=np.float64)
    if j_hat.shape != (groups, s4cr2.N_ACTIONS) or j_true.shape != j_hat.shape:
        raise ValueError("J_hat/J_true must align as (G, 10)")
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
    if strongest not in s4cr2.STRONGEST_BASELINE_CANDIDATES:
        raise ValueError(
            f"strongest_baseline {strongest!r} is not a frozen eligible candidate")
    qa = quantities.get("qa")
    if not isinstance(qa, Mapping) or any(key not in qa for key in QA_KEYS):
        raise ValueError(f"screen quantities qa must carry {list(QA_KEYS)}")
    return groups


# --- the joint quantity vector ------------------------------------------------------------------


def compute_screen_quantity_vector(
    index: np.ndarray, quantities: Mapping[str, Any]
) -> np.ndarray:
    """Evaluate all 32 quantities on ONE whole-group resample ``index``.

    ``index`` is a with-replacement integer resample into the screen groups, so every
    metric of the draw stays coupled through the same group multiplicities. Improvement
    quantities are NaN when the frozen baseline denominator is non-positive (the gate
    turns that into ``STAGE4CR2_INVALID_ANALYSIS``); a stratum/trim subset with fewer
    than two resampled members yields NaN, which fails its gate condition.
    """
    index = np.asarray(index, dtype=np.int64)
    reg = np.asarray(quantities["regret"], dtype=np.float64)[index]
    y_high = np.asarray(quantities["y_high"], dtype=np.int64)[index]
    ens = np.asarray(quantities["ensemble_risk"], dtype=np.float64)[index]
    baselines = quantities["baseline_risks"]
    base = np.asarray(baselines[str(quantities["strongest_baseline"])],
                      dtype=np.float64)[index]
    margin = np.asarray(baselines[s4cr2.BASELINE_MARGIN], dtype=np.float64)[index]
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)[:, index]
    mu_only = np.asarray(quantities["mu_only_ensemble"], dtype=np.float64)[index]
    class_only = np.asarray(quantities["class_only_ensemble"], dtype=np.float64)[index]

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

    for slot, frac in ((Q_TRIM1, s4cr2.GATE_TRIM_FRACTIONS[0]),
                       (Q_TRIM5, s4cr2.GATE_TRIM_FRACTIONS[1])):
        keep = s4cs.trim_high_regret(reg, frac)
        if keep.sum() >= 2:
            q[slot] = s4cs.improvement(
                s4cs.aurc_regret(np.argsort(base[keep], kind="stable"), reg[keep]),
                s4cs.aurc_regret(np.argsort(ens[keep], kind="stable"), reg[keep]))

    for slot, layer_key in ((Q_TSTAR_LOW, "tstar_layer"),
                            (Q_INTERACT_LOW, "interaction_layer")):
        layer = np.asarray(quantities[layer_key], dtype=np.int64)[index]
        for level in (0, 1):
            mask = layer == level
            if mask.sum() < 2:
                continue  # stays NaN -> its floor condition fails closed
            q[slot + level] = s4cs.improvement(
                s4cs.aurc_regret(np.argsort(base[mask], kind="stable"), reg[mask]),
                s4cs.aurc_regret(np.argsort(ens[mask], kind="stable"), reg[mask]))

    q[Q_SPEARMAN_MARGIN] = s4cs.spearman(ens, margin)
    q[Q_AURC_BASELINE] = a_base
    q[Q_R80_BASELINE] = r80_base
    q[Q_BASE_MEAN_REGRET] = base_mean
    q[Q_IDLE_MEAN_REGRET] = float(idle.mean())
    q[Q_RANDOM_MEAN_REGRET] = rand_mean
    q[Q_HIGH_REGRET_FRAC] = float((reg > s4cr2.HIGH_REGRET_THRESHOLD).mean())

    for i in range(N_SEEDS):
        order_seed = np.argsort(by_seed[i], kind="stable")
        q[Q_SEED_I_AURC + i] = s4cs.improvement(
            a_base, s4cs.aurc_regret(order_seed, reg))
        q[Q_SEED_I_R80 + i] = s4cs.improvement(
            r80_base, s4cs.regret_at_coverage(order_seed, reg, COVERAGE_MAIN))
    q[Q_MU_I_AURC] = s4cs.improvement(
        a_base, s4cs.aurc_regret(np.argsort(mu_only, kind="stable"), reg))
    q[Q_MU_I_R80] = s4cs.improvement(
        r80_base, s4cs.regret_at_coverage(np.argsort(mu_only, kind="stable"), reg,
                                          COVERAGE_MAIN))
    q[Q_CLS_I_AURC] = s4cs.improvement(
        a_base, s4cs.aurc_regret(np.argsort(class_only, kind="stable"), reg))
    q[Q_CLS_I_R80] = s4cs.improvement(
        r80_base, s4cs.regret_at_coverage(np.argsort(class_only, kind="stable"), reg,
                                          COVERAGE_MAIN))
    return q


# --- the one paired bootstrap ----------------------------------------------------------------------


def paired_group_bootstrap(
    quantities: Mapping[str, np.ndarray], *, seed: int, n_resamples: int
) -> BootstrapResult:
    """10,000-draw whole-group paired bootstrap with shared indices and 80% CIs.

    EVERY quantity of a draw is recomputed from the SAME whole-group resample index
    vector; the identity evaluation is the point estimate; the two-sided 80% CI is the
    (0.10, 0.90) resample quantile band. The confidence level is frozen at
    ``CONFIDENCE_SCREEN`` (0.80) — never re-parameterised per call site.
    """
    groups = _validate_quantities(quantities)
    generator = np.random.default_rng(int(seed))
    identity = np.arange(groups, dtype=np.int64)
    point = compute_screen_quantity_vector(identity, quantities)
    draws = np.empty((int(n_resamples), N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = compute_screen_quantity_vector(index, quantities)
    alpha = (1.0 - s4cr2.CONFIDENCE_SCREEN) / 2.0
    return BootstrapResult(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws,
        n_resamples=int(n_resamples),
        seed=int(seed),
        confidence=float(s4cr2.CONFIDENCE_SCREEN),
    )


# --- the gates -------------------------------------------------------------------------------------


def _named(vector: np.ndarray) -> dict[str, float]:
    return {name: float(vector[i]) for i, name in enumerate(_QUANTITY_NAMES)}


def evaluate_screen_gates(
    quantities: Mapping[str, np.ndarray], bootstrap: BootstrapResult
) -> ScreenDecision:
    """Adjudicate Gate TASK-R2 and Gate RISK-DECISION-R2_dev exactly once (fail-closed).

    Terminal-state priority (plan section 11): Gate Q / hash / role / lock failure or a
    non-positive baseline denominator -> ``STAGE4CR2_INVALID_ANALYSIS``; Gate TASK
    condition 2/3/7 failure -> ``STAGE4CR2_TASK_DEGENERATE``; condition 4/5/6 failure ->
    ``INCONCLUSIVE_STAGE4CR2_DECISION_MODEL_INSUFFICIENT``; any unblocked risk condition
    failure -> ``STAGE4CR2_DECISION_RISK_LINE_CLOSED``; all pass ->
    ``STAGE4CR2_DEV_READY_FOR_CONFIRMATION``. Non-positive denominators are never
    repaired (no epsilon, no absolute difference, no baseline swap); NaN quantities fail
    their conditions by construction.
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
    if float(bootstrap.confidence) != float(s4cr2.CONFIDENCE_SCREEN):
        raise ValueError("screen CI level is frozen at 0.80")

    qa = {key: bool(quantities["qa"][key]) for key in QA_KEYS}
    qa_passed = all(qa.values())

    # --- Gate TASK-R2 (plan 9.1) ------------------------------------------------------
    oracle = np.asarray(quantities["a_star"], dtype=np.int64)
    oracle_shares = [float((oracle == a).mean()) for a in range(s4cr2.N_ACTIONS)]
    n_oracle = int(sum(s >= s4cr2.GATE_TASK_MIN_ORACLE_FRAC for s in oracle_shares))
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)
    stratum_counts = {
        "tstar_low": int((tstar == 0).sum()), "tstar_high": int((tstar == 1).sum()),
        "interaction_low": int((inter == 0).sum()),
        "interaction_high": int((inter == 1).sum()),
    }
    task_conditions = {
        "c1_all_complete_gate_q": bool(qa_passed and groups == s4cr2.DEV_SCREEN_GROUPS),
        "c2_high_regret_frac_ge_15pct": bool(
            point[Q_HIGH_REGRET_FRAC] >= s4cr2.GATE_TASK_MIN_HIGH_REGRET_FRAC),
        "c3_at_least_3_oracle_actions_ge_5pct": bool(
            n_oracle >= s4cr2.GATE_TASK_MIN_DISTINCT_ORACLE),
        "c4_branch_spearman_ci_lower_positive": bool(
            ci_lower[Q_SPEARMAN_BRANCH] > 0.0),
        "c5_base_vs_random_point_and_ci_positive": bool(
            point[Q_BASE_VS_RANDOM] > 0.0 and ci_lower[Q_BASE_VS_RANDOM] > 0.0),
        "c6_base_not_worse_than_idle": bool(
            np.isfinite([point[Q_BASE_MEAN_REGRET], point[Q_IDLE_MEAN_REGRET]]).all()
            and point[Q_BASE_MEAN_REGRET] <= point[Q_IDLE_MEAN_REGRET] + 1e-12),
        "c7_strata_ge_800_groups": bool(
            min(stratum_counts.values()) >= s4cr2.GATE_TASK_MIN_STRATUM_COUNT),
    }
    degenerate = not (task_conditions["c2_high_regret_frac_ge_15pct"]
                      and task_conditions["c3_at_least_3_oracle_actions_ge_5pct"]
                      and task_conditions["c7_strata_ge_800_groups"])
    insufficient = not (task_conditions["c4_branch_spearman_ci_lower_positive"]
                        and task_conditions["c5_base_vs_random_point_and_ci_positive"]
                        and task_conditions["c6_base_not_worse_than_idle"])
    if degenerate:
        task_status = s4cr2.STATUS_TASK_DEGENERATE
    elif insufficient:
        task_status = s4cr2.STATUS_MODEL_INSUFFICIENT
    else:
        task_status = STATUS_TASK_QUALIFIED

    # --- Gate RISK-DECISION-R2_dev (plan 9.2) ------------------------------------------
    baseline_valid = bool(
        np.isfinite([point[Q_AURC_BASELINE], point[Q_R80_BASELINE]]).all()
        and point[Q_AURC_BASELINE] > 0.0 and point[Q_R80_BASELINE] > 0.0)
    seeds_i_aurc_positive = int(sum(point[i] > 0.0 for i in SEED_I_AURC_INDEX))
    seeds_i_r80_positive = int(sum(point[i] > 0.0 for i in SEED_I_R80_INDEX))
    strata = {
        "tstar_low": float(point[Q_TSTAR_LOW]),
        "tstar_high": float(point[Q_TSTAR_HIGH]),
        "interaction_low": float(point[Q_INTERACT_LOW]),
        "interaction_high": float(point[Q_INTERACT_HIGH]),
    }
    trims = {"trim_0.01": float(point[Q_TRIM1]), "trim_0.05": float(point[Q_TRIM5])}
    risk_conditions = {
        "r1_i_aurc_regret": bool(point[Q_I_AURC] >= s4cr2.VALIDATION_I_AURC_MIN
                                 and ci_lower[Q_I_AURC] > 0.0),
        "r2_i_regret80": bool(point[Q_I_REGRET80] >= s4cr2.VALIDATION_I_REGRET80_MIN
                              and ci_lower[Q_I_REGRET80] > 0.0),
        "r3_pr_auc_minus_prevalence": bool(point[Q_PR] > 0.0 and ci_lower[Q_PR] > 0.0),
        "r4_seeds_i_aurc": bool(seeds_i_aurc_positive
                                >= s4cr2.VALIDATION_MIN_POSITIVE_SEEDS),
        "r5_seeds_i_regret80": bool(seeds_i_r80_positive
                                    >= s4cr2.VALIDATION_MIN_POSITIVE_SEEDS),
        "r6_strata_floor": bool(
            all(np.isfinite(list(strata.values())))
            and min(strata.values()) >= s4cr2.GATE_STRATA_I_AURC_FLOOR),
        "r7_trims_positive": bool(all(np.isfinite(list(trims.values())))
                                  and min(trims.values()) > 0.0),
        "r8_margin_spearman": bool(
            np.isfinite(point[Q_SPEARMAN_MARGIN])
            and abs(point[Q_SPEARMAN_MARGIN]) < s4cr2.GATE_MARGIN_SPEARMAN_CEILING),
        "r9_qa_and_one_shot_locks": bool(qa_passed),
    }

    # --- terminal-state priority (plan section 11) --------------------------------------
    if not qa_passed or not baseline_valid:
        status = s4cr2.STATUS_INVALID
    elif degenerate:
        status = s4cr2.STATUS_TASK_DEGENERATE
    elif insufficient:
        status = s4cr2.STATUS_MODEL_INSUFFICIENT
    elif all(risk_conditions.values()):
        status = s4cr2.STATUS_DEV_READY
    else:
        status = s4cr2.STATUS_LINE_CLOSED

    # --- non-blocking ablations (plan 9.4; never gate substitutes) -----------------------
    def _curve_points(risk: np.ndarray) -> dict[str, float]:
        order = s4cs.order_by_risk(np.asarray(risk, dtype=np.float64))
        regret = np.asarray(quantities["regret"], dtype=np.float64)
        return {
            "aurc_regret": float(s4cs.aurc_regret(order, regret)),
            "regret_at_80": float(s4cs.regret_at_coverage(order, regret, COVERAGE_MAIN)),
            "pr_auc_minus_prevalence": float(s4cs.high_regret_pr(
                np.asarray(risk, dtype=np.float64),
                np.asarray(quantities["y_high"], dtype=np.int64))),
        }

    baseline_curves: dict[str, Any] = {}
    for name in ALL_BASELINE_NAMES:
        entry: dict[str, Any] = _curve_points(quantities["baseline_risks"][name])
        if name == str(quantities["strongest_baseline"]):
            entry["role"] = "strongest_standard_baseline"
        elif name in s4cr2.STRONGEST_BASELINE_CANDIDATES:
            entry["role"] = "standard_baseline"
        elif name == s4cr2.BASELINE_COMBINED_TOP2_MAX:
            entry["role"] = "historical_reference"
        else:
            entry["role"] = "curve_reference"
        baseline_curves[name] = entry
    ablations = {
        "blocking": False,
        "note": ("non-blocking ablations reported alongside the primary score; they "
                 "never enter the gates and never substitute for the primary score"),
        "MU_ONLY": {
            "i_aurc_regret": float(point[Q_MU_I_AURC]),
            "i_aurc_regret_ci80": [float(ci_lower[Q_MU_I_AURC]),
                                   float(ci_upper[Q_MU_I_AURC])],
            "i_regret80": float(point[Q_MU_I_R80]),
            "i_regret80_ci80": [float(ci_lower[Q_MU_I_R80]),
                                float(ci_upper[Q_MU_I_R80])],
        },
        "CLASS_ONLY": {
            "i_aurc_regret": float(point[Q_CLS_I_AURC]),
            "i_aurc_regret_ci80": [float(ci_lower[Q_CLS_I_AURC]),
                                   float(ci_upper[Q_CLS_I_AURC])],
            "i_regret80": float(point[Q_CLS_I_R80]),
            "i_regret80_ci80": [float(ci_lower[Q_CLS_I_R80]),
                                float(ci_upper[Q_CLS_I_R80])],
        },
        "baseline_curves": baseline_curves,
    }

    return ScreenDecision(
        plan_id=s4cr2.PLAN_ID,
        n_groups=int(groups),
        status=status,
        passed=bool(status == s4cr2.STATUS_DEV_READY),
        task={
            "conditions": task_conditions,
            "status": task_status,
            "high_regret_frac": float(point[Q_HIGH_REGRET_FRAC]),
            "oracle_action_shares": oracle_shares,
            "oracle_actions_covered_ge_5pct": n_oracle,
            "stratum_counts": stratum_counts,
        },
        risk={
            "conditions": risk_conditions,
            "baseline_valid": baseline_valid,
            "baseline_denominators": {
                "aurc_regret": float(point[Q_AURC_BASELINE]),
                "regret_at_80": float(point[Q_R80_BASELINE]),
            },
            "seeds_i_aurc_positive": seeds_i_aurc_positive,
            "seeds_i_regret80_positive": seeds_i_r80_positive,
            "strata_i_aurc": strata,
            "trims_i_aurc": trims,
        },
        metrics={
            "point": _named(point),
            "ci_lower": _named(ci_lower),
            "ci_upper": _named(ci_upper),
        },
        ablations=ablations,
        bootstrap={
            "seed": int(bootstrap.seed),
            "n_resamples": int(bootstrap.n_resamples),
            "confidence": float(bootstrap.confidence),
            "design": ("whole-group paired bootstrap; every metric of a draw recomputed "
                       "from the SAME resample index vector; two-sided 80% CI from the "
                       "resample quantiles"),
        },
    )


# --- screen artifact paths -------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def screen_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every one-shot screen path (plan section 15 + the Direct score freeze)."""
    checked = s4cr2.validate_config(config)
    dev = _resolve(checked["run_dir"]) / "development"
    data_root = _resolve(checked["data_root"])
    return {
        "dev_dir": dev,
        "data_output": data_root / "object_twinworld_stage4cr2_screen.npz",
        "collection_receipt": dev / "screen_collection_receipt.json",
        "predictions": dev / "screen_predictions.npz",
        "scores": dev / "screen_scores.npz",
        "inference_receipt": dev / "screen_inference_receipt.json",
        "determinism_receipt": dev / "screen_determinism_receipt.json",
        "direct_scores_npz": dev / "screen_direct_scores.npz",
        "direct_scores_receipt": dev / "screen_direct_scores_receipt.json",
        "lock": dev / "development_screen_used.lock",
        "quantities_npz": dev / "screen_group_quantities.npz",
        "quantities_receipt": dev / "screen_group_quantities_receipt.json",
        "report": dev / "development_screen_report.json",
        "final_selection": dev / "final_selection.json",
        "model_inventory": dev / "model_inventory.json",
        "train_support_r1": dev / "train_support_report_r1.json",
        "train_support_base": dev / "train_support_report.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_frozen_final_selection(paths: Mapping[str, Path]) -> dict[str, Any]:
    """The screen exists only because validation PASSED and the selection is frozen."""
    if not paths["final_selection"].is_file():
        raise ValueError(
            "screen is refused: final_selection.json is missing — the validation gate "
            "must pass and freeze before any screen artifact is produced")
    final = _read_json(paths["final_selection"])
    if final.get("status") != "VALIDATION_QUALIFIED":
        raise ValueError(
            f"screen is refused: validation status is {final.get('status')!r}, not "
            "VALIDATION_QUALIFIED; the multi-action decision-risk line is closed")
    return final


def _verify_hash_chain(paths: Mapping[str, Path]) -> dict[str, str]:
    """Re-hash every score-side artifact against its create-once receipt (fail-closed).

    Also binds the collection receipt: it must exist, its ``data_output_sha256`` must
    match the collected screen NPZ, and its role must be the screen.
    """
    inference = _read_json(paths["inference_receipt"])
    for key, path in (("predictions_sha256", paths["predictions"]),
                      ("scores_sha256", paths["scores"])):
        if not path.is_file() or sha256_file(path) != inference[key]:
            raise ValueError(f"screen inference artifact drifted from its receipt: {path}")
    if not paths["data_output"].is_file() \
            or sha256_file(paths["data_output"]) != inference["data_sha256"]:
        raise ValueError("screen data drifted from its inference receipt")
    if not paths["collection_receipt"].is_file():
        raise ValueError("the screen collection receipt is missing")
    collection = _read_json(paths["collection_receipt"])
    if collection.get("data_output_sha256") != inference["data_sha256"]:
        raise ValueError("screen collection receipt does not bind the inferred data")
    inference["collection_receipt_sha256"] = sha256_file(paths["collection_receipt"])
    inference["collection_accepted_groups"] = int(collection.get("accepted_groups", -1))
    return inference


# --- plan 9.3 steps 1-4 (Direct side): the score freeze --------------------------------------------


def _constant_random_scores(group_seed: np.ndarray) -> np.ndarray:
    """Curve-reference CONSTANT_RANDOM group scores (plan 4.4; never gate-relevant).

    A pure function of the unique group seed — deterministic across reruns, introduces
    no new frozen constant, and is never used by any gate.
    """
    out = np.empty(group_seed.size, dtype=np.float64)
    for i, seed in enumerate(group_seed):
        out[i] = np.random.SeedSequence([int(seed)]).generate_state(1)[0] / float(2**32)
    return out


def freeze_screen_direct_scores(
    config: Mapping[str, Any],
    *,
    features: Mapping[str, Any],
    score_fn: Callable[[Mapping[str, np.ndarray]], Any] | None = None,
    device: str | None = None,
) -> Path:
    """Score the screen with the five frozen Direct checkpoints; freeze create-once.

    Writes ``screen_direct_scores.npz`` (ensemble + per-seed + MU_ONLY/CLASS_ONLY +
    baseline group risks + strata inputs, all hash-bound) and its receipt. Includes the
    Direct-side 32-group bitwise recheck: the first ``determinism_checks`` groups are
    scored twice and must be elementwise identical. A rerun verifies hashes and never
    overwrites. ``features`` is the score-side output of the frozen
    ``build_relative_tokens`` plus ``t_star``; ``score_fn`` is a test hook (the default
    loads the inventory from ``model_inventory.json`` and runs the frozen
    ``score_models``).
    """
    checked = s4cr2.validate_config(config)
    paths = screen_artifact_paths(checked)
    final = _load_frozen_final_selection(paths)
    npz_path = paths["direct_scores_npz"]
    receipt_path = paths["direct_scores_receipt"]
    inference = _verify_hash_chain(paths)
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if not npz_path.is_file() or sha256_file(npz_path) != receipt["direct_scores_sha256"]:
            raise ValueError(f"screen Direct scores drifted from their receipt: {npz_path}")
        return npz_path
    if npz_path.exists():
        raise RuntimeError(
            f"screen Direct scores exist without their create-once receipt: {npz_path}")

    tokens = np.asarray(features["tokens"], dtype=np.float64)
    global_features = np.asarray(features["global_features"], dtype=np.float64)
    group_seed = np.asarray(features["group_seed"], dtype=np.int64).ravel()
    t_star = np.asarray(features["t_star"], dtype=np.int64).ravel()
    j_hat = np.asarray(features["j_hat"], dtype=np.float64)
    distance = np.asarray(features["interaction_min_distance"], dtype=np.float64).ravel()
    valid = np.asarray(features["valid_mask"], dtype=bool).ravel()
    groups = int(group_seed.size)
    if not valid.all():
        raise ValueError(
            f"{int((~valid).sum())} screen groups have non-finite features; the screen "
            "contract rejects zero-padding, so scoring is refused")
    if j_hat.shape != (groups, s4cr2.N_ACTIONS) or distance.shape != (groups,) \
            or t_star.shape != (groups,):
        raise ValueError("screen feature arrays must align on the group axis")

    if score_fn is None:
        from ac_twin import stage4cr2_training as s4cr2t
        inventory = _load_model_inventory(paths)
        device = str(device or checked.get("device", "cuda"))

        def score_fn(bundle):
            return s4cr2t.score_models(inventory, bundle, device=device)

    bundle = {"tokens": tokens, "global_features": global_features,
              "group_seed": group_seed}
    scores = score_fn(bundle)
    _check_scores(scores, groups, group_seed)

    # Direct-side 32-group bitwise recheck on the fixed leading subset.
    n_recheck = int(checked.get("determinism_checks", 32))
    subset = {"tokens": tokens[:n_recheck], "global_features": global_features[:n_recheck],
              "group_seed": group_seed[:n_recheck]}
    first = score_fn(subset)
    second = score_fn(subset)
    max_abs = 0.0
    for key in ("risk_ensemble", "risk_by_seed", "mu_only_ensemble", "class_only_ensemble"):
        a = np.asarray(getattr(first, key))
        b = np.asarray(getattr(second, key))
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"screen Direct scoring is not elementwise reproducible ({key}); "
                "the one-shot screen halts before the lock is burned")
        max_abs = max(max_abs, float(np.max(np.abs(a - b))) if a.size else 0.0)

    baselines = features["baselines"]
    payload = {
        "group_seed": group_seed,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "j_hat": j_hat,
        "risk_ensemble": np.asarray(scores.risk_ensemble, dtype=np.float64),
        "risk_by_seed": np.asarray(scores.risk_by_seed, dtype=np.float64),
        "mu_only_ensemble": np.asarray(scores.mu_only_ensemble, dtype=np.float64),
        "class_only_ensemble": np.asarray(scores.class_only_ensemble, dtype=np.float64),
        **{_BASELINE_NPZ_PREFIX + s4cr2.BASELINE_CONSTANT_RANDOM:
           _constant_random_scores(group_seed)},
    }
    for name in (s4cr2.BASELINE_MARGIN, s4cr2.BASELINE_SEED_VOTE_ENTROPY,
                 s4cr2.BASELINE_SEED_MARGIN_STD, s4cr2.BASELINE_COMBINED_TOP2_MAX):
        risk = np.asarray(baselines[name], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"screen baseline risk {name!r} must be finite and aligned")
        payload[_BASELINE_NPZ_PREFIX + name] = risk

    atomic_save_npz(npz_path, **payload)
    receipt = {
        "plan_id": s4cr2.PLAN_ID,
        "role": "development_screen",
        "n_groups": int(groups),
        "direct_scores_path": str(npz_path),
        "direct_scores_sha256": sha256_file(npz_path),
        "predictions_sha256": inference["predictions_sha256"],
        "scores_sha256": inference["scores_sha256"],
        "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
        "final_selection_sha256": sha256_file(paths["final_selection"]),
        "model_inventory_sha256": sha256_file(paths["model_inventory"]),
        "checkpoints": final["checkpoints"],
        "model_seeds": [int(s) for s in s4cr2.MODEL_SEEDS],
        "ensemble_definition": "arithmetic_mean_of_five_seeds",
        "primary_score_definition": final.get("primary_score_definition", ""),
        "constant_random_derivation": (
            "per-group SeedSequence([group_seed]) / 2**32; curve reference only"),
        "direct_recheck": {
            "n_groups": int(n_recheck),
            "group_indices": list(range(n_recheck)),
            "bitwise_identical": True,
            "max_abs_diff": float(max_abs),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return npz_path


def _check_scores(scores: Any, groups: int, group_seed: np.ndarray) -> None:
    """Fail closed on any malformed Direct score block; the ensemble is the mean."""
    if tuple(int(s) for s in scores.seeds) != tuple(int(s) for s in s4cr2.MODEL_SEEDS):
        raise ValueError("screen Direct scores must come from the five frozen seeds")
    by_seed = np.asarray(scores.risk_by_seed, dtype=np.float64)
    ensemble = np.asarray(scores.risk_ensemble, dtype=np.float64)
    if by_seed.shape != (N_SEEDS, groups) or ensemble.shape != (groups,):
        raise ValueError("Direct score shapes must align as (5, G) and (G,)")
    if not np.isfinite(by_seed).all() or not np.isfinite(ensemble).all():
        raise ValueError("non-finite Direct screen score")
    if not np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15):
        raise ValueError("the ensemble must be exactly the arithmetic mean of five seeds")
    if not np.array_equal(np.asarray(scores.group_seed, dtype=np.int64), group_seed):
        raise ValueError("Direct score group seeds misaligned with the screen groups")


def _load_model_inventory(paths: Mapping[str, Path]) -> Any:
    """Rebuild the frozen ModelInventory from model_inventory.json; verify checkpoints."""
    from ac_twin import stage4cr2_training as s4cr2t
    payload = _read_json(paths["model_inventory"])
    for seed_text, digest in payload["checkpoint_sha256"].items():
        path = Path(payload["checkpoint_paths"][seed_text])
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"frozen Direct checkpoint drifted: {path}")
    scaler_payload = payload["scaler"]
    from ac_twin import stage4cp_features as s4cpf
    return s4cr2t.ModelInventory(
        plan_id=str(payload["plan_id"]),
        arch=str(payload["arch"]),
        seeds=tuple(int(s) for s in payload["seeds"]),
        param_count=int(payload["param_count"]),
        run_dir=str(payload["run_dir"]),
        checkpoint_paths={int(k): str(v) for k, v in payload["checkpoint_paths"].items()},
        checkpoint_sha256={int(k): str(v) for k, v in payload["checkpoint_sha256"].items()},
        scaler=s4cpf.FrozenFeatureScaler(
            np.asarray(scaler_payload["token_cont_mean"]),
            np.asarray(scaler_payload["token_cont_std"]),
            np.asarray(scaler_payload["global_mean"]),
            np.asarray(scaler_payload["global_std"])),
        pos_weight=float(payload["pos_weight"]),
        interaction_threshold=float(payload["interaction_threshold"]),
        main_indices=tuple(int(i) for i in payload["main_indices"]),
        monitor_indices=tuple(int(i) for i in payload["monitor_indices"]),
        train_losses={int(k): float(v) for k, v in payload["train_losses"].items()},
        monitor_losses={int(k): float(v) for k, v in payload["monitor_losses"].items()},
        n_train_groups=int(payload["n_train_groups"]),
        written_at=str(payload["written_at"]),
    )


# --- plan 9.3 step 5: burn the one-shot lock --------------------------------------------------------


def burn_screen_lock(config: Mapping[str, Any]) -> Path:
    """Create ``development_screen_used.lock`` exactly once, after the full score freeze.

    Refuses unless: the frozen final selection is VALIDATION_QUALIFIED; the predictions
    /scores/data hash-match the inference receipt; the Direct score freeze hash-matches
    its receipt; the 32-group repeated-inference receipt is bitwise identical and bound
    to the same data; and the Direct recheck is bitwise identical. A second call
    re-verifies the bindings against the current files and never overwrites.
    """
    checked = s4cr2.validate_config(config)
    paths = screen_artifact_paths(checked)
    final = _load_frozen_final_selection(paths)
    inference = _verify_hash_chain(paths)
    if not paths["direct_scores_receipt"].is_file():
        raise ValueError(
            "the Direct score freeze is missing; the screen lock may not burn before "
            "every score-side artifact is written and hashed")
    direct_receipt = _read_json(paths["direct_scores_receipt"])
    if not paths["direct_scores_npz"].is_file() or sha256_file(paths["direct_scores_npz"]) \
            != direct_receipt["direct_scores_sha256"]:
        raise ValueError("screen Direct scores drifted from their create-once receipt")
    if not paths["determinism_receipt"].is_file():
        raise ValueError("the 32-group repeated-inference receipt is missing")
    determinism = _read_json(paths["determinism_receipt"])
    if not determinism.get("bitwise_identical"):
        raise ValueError("the 32-group repeated inference was not bitwise identical")
    if determinism.get("data_sha256") != inference["data_sha256"]:
        raise ValueError("determinism receipt is not bound to the screen data")
    if not direct_receipt.get("direct_recheck", {}).get("bitwise_identical"):
        raise ValueError("the Direct 32-group recheck was not bitwise identical")

    bindings = {
        "predictions_sha256": inference["predictions_sha256"],
        "scores_sha256": inference["scores_sha256"],
        "data_sha256": inference["data_sha256"],
        "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
        "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
        "direct_scores_sha256": direct_receipt["direct_scores_sha256"],
        "direct_scores_receipt_sha256": sha256_file(paths["direct_scores_receipt"]),
        "final_selection_sha256": sha256_file(paths["final_selection"]),
    }
    lock_path = paths["lock"]
    if lock_path.exists():
        payload = _read_json(lock_path)
        if payload.get("plan_id") != s4cr2.PLAN_ID \
                or payload.get("bindings") != bindings:
            raise ValueError(
                "the burned screen lock no longer matches the on-disk frozen artifacts "
                "(tampering or drift); the one-shot screen is compromised")
        return lock_path
    payload = {
        "plan_id": s4cr2.PLAN_ID,
        "role": "development_screen",
        "semantics": ("one-shot: the screen truth may be read only after this lock "
                      "exists; all screen scores are frozen and hash-bound above"),
        "strongest_standard_baseline": final["strongest_standard_baseline"],
        "interaction_threshold": float(final["interaction_threshold"]),
        "bindings": bindings,
        "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path, payload)
    return lock_path


# --- plan 9.3 steps 6-7: read truth, assemble, write create-once --------------------------------------


def assert_screen_adjudication_open(config: Mapping[str, Any]) -> None:
    """Fail closed unless the one-shot adjudication may legally start.

    Requires the burned ``development_screen_used.lock`` (truth is never read before the
    scores are frozen and the lock exists) and refuses any second adjudication
    (quantities NPZ / receipt / screen report already present).
    """
    paths = screen_artifact_paths(config)
    _load_frozen_final_selection(paths)
    if not paths["lock"].is_file():
        raise ValueError(
            "screen truth may not be read: development_screen_used.lock is not burned; "
            "freeze every score artifact and burn the lock first (plan 9.3)")
    for key in ("quantities_npz", "quantities_receipt", "report"):
        if paths[key].exists():
            raise FileExistsError(
                f"a second screen adjudication is refused; already exists: {paths[key]}")


def assemble_screen_group_quantities(
    config: Mapping[str, Any], *, bundle: Mapping[str, Any] | None = None
) -> Path:
    """Read the screen truth ONCE, build regret/strata, write the create-once quantities.

    Plan 9.3 steps 6-7. Refuses unless the one-shot lock is burned; re-verifies the full
    hash chain; computes ``J_true``/regret/oracle with the frozen Stage 4C consequence
    function; runs the Gate-Q assembly checks; writes ``screen_group_quantities.npz``
    and its receipt exactly once. The receipt binds the USER-CONFIRMED train-support
    recovery report (``train_support_report_r1.json``, path + SHA256) and names the
    superseded base report, so the recovery is offline-auditable. Returns the NPZ path.
    """
    checked = s4cr2.validate_config(config)
    paths = screen_artifact_paths(checked)
    assert_screen_adjudication_open(checked)
    final = _load_frozen_final_selection(paths)
    inference = _verify_hash_chain(paths)
    burn_screen_lock(checked)  # re-verify the lock bindings against the current files
    lock_payload = _read_json(paths["lock"])
    direct_receipt = _read_json(paths["direct_scores_receipt"])

    with np.load(paths["direct_scores_npz"], allow_pickle=False) as archive:
        direct = {key: archive[key] for key in archive.files}
    if bundle is None:
        from ac_twin import stage4cr2_inference as s4cr2i
        bundle = s4cr2i.load_role_bundle(checked, "development_screen")

    groups = int(bundle["groups"])
    n_actions = int(bundle["n_actions"])
    group_seed = np.asarray(bundle["episode_seed"], dtype=np.int64).ravel()

    # --- truth side: legal ONLY here, after the burned lock ----------------------------
    hist_b = np.broadcast_to(
        np.asarray(bundle["history_last_xy"], dtype=np.float64)[:, None, :],
        (groups, n_actions, 2))
    mask = np.asarray(bundle["inference_mask"], dtype=bool)
    mask_b = np.broadcast_to(mask[:, None, :, :], (groups, n_actions) + mask.shape[1:])
    comp_true = s4co.trajectory_components(
        np.asarray(bundle["counterfactual_xy"], dtype=np.float64), hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))
    j_hat = np.asarray(direct["j_hat"], dtype=np.float64)
    rows = np.arange(groups)
    a_hat = np.argmin(j_hat, axis=-1)
    a_star = np.argmin(j_true, axis=-1)
    j_true_star = j_true[rows, a_star]
    regret = s4co.clamp_regret(j_true[rows, a_hat] - j_true_star)
    random_regret = s4co.clamp_regret(j_true.mean(axis=-1) - j_true_star)
    idle_regret = s4co.clamp_regret(j_true[:, s4co.IDLE_PLAN_INDEX] - j_true_star)
    y_high = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)

    threshold = float(final["interaction_threshold"])
    t_star = np.asarray(direct["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(direct["interaction_min_distance"], dtype=np.float64).ravel()
    tstar_layer = (t_star >= TSTAR_SPLIT).astype(np.int64)
    interaction_layer = (distance > threshold).astype(np.int64)

    # --- Gate-Q assembly checks (recorded, evaluated inside the gates) ------------------
    schedule = s4cr2.build_seed_schedule(checked)
    lo, hi = s4cr2.SCREEN_SEGMENT
    screen_segment = set(int(v) for v in schedule[lo:hi])
    other_segments = set(int(v) for v in schedule[:lo])
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        predictions_seed = np.asarray(archive["group_seed"], dtype=np.int64).ravel()
    direct_seed = np.asarray(direct["group_seed"], dtype=np.int64).ravel()
    role_seeds_valid = bool(
        group_seed.size == groups and np.unique(group_seed).size == groups
        and all(int(s) in screen_segment for s in group_seed)
        and not any(int(s) in other_segments for s in group_seed)
        and np.array_equal(group_seed, predictions_seed)
        and np.array_equal(group_seed, direct_seed))
    finite_truth = bool(np.isfinite(j_true).all() and np.isfinite(regret).all())
    all_complete = bool(
        groups == s4cr2.DEV_SCREEN_GROUPS and n_actions == s4cr2.N_ACTIONS
        and inference["collection_accepted_groups"] == s4cr2.DEV_SCREEN_GROUPS
        and j_true.shape == (groups, s4cr2.N_ACTIONS) and finite_truth
        and j_hat.shape == (groups, s4cr2.N_ACTIONS))
    by_seed = np.asarray(direct["risk_by_seed"], dtype=np.float64)
    ensemble = np.asarray(direct["risk_ensemble"], dtype=np.float64)
    qa = {
        "all_complete": all_complete,
        "hashes_match": True,  # _verify_hash_chain + lock re-verification passed above
        "role_seeds_valid": role_seeds_valid,
        "determinism_passed": True,  # burn_screen_lock re-verified both receipts
        "one_shot_locks": True,
        "ensemble_is_arithmetic_mean": bool(
            by_seed.shape == (N_SEEDS, groups)
            and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)),
    }

    npz_path = paths["quantities_npz"]
    payload = {
        "group_seed": group_seed,
        "regret": regret,
        "y_high": y_high,
        "random_regret": random_regret,
        "idle_regret": idle_regret,
        "a_hat": a_hat.astype(np.int64),
        "a_star": a_star.astype(np.int64),
        "J_hat": j_hat,
        "J_true": j_true,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": np.asarray(direct["mu_only_ensemble"], dtype=np.float64),
        "class_only_ensemble": np.asarray(direct["class_only_ensemble"], dtype=np.float64),
        "tstar_layer": tstar_layer,
        "interaction_layer": interaction_layer,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "n_groups": np.int64(groups),
        "interaction_threshold": np.float64(threshold),
    }
    for name in ALL_BASELINE_NAMES:
        payload[_BASELINE_NPZ_PREFIX + name] = np.asarray(
            direct[_BASELINE_NPZ_PREFIX + name], dtype=np.float64)
    atomic_save_npz(npz_path, **payload)

    # --- receipt: binds the recovery report so the train-support fix is auditable --------
    r1_path = paths["train_support_r1"]
    if r1_path.is_file():
        support_report_path, support_payload = r1_path, _read_json(r1_path)
        superseded_path = paths["train_support_base"]
        superseded = {
            "superseded_base_report_path": str(superseded_path),
            "superseded_base_report_sha256": sha256_file(superseded_path),
            "superseded_base_report_status": _read_json(superseded_path).get("status", ""),
        }
    else:
        support_report_path, support_payload = (
            paths["train_support_base"], _read_json(paths["train_support_base"]))
        superseded = {}
    receipt = {
        "plan_id": s4cr2.PLAN_ID,
        "role": "development_screen",
        "n_groups": int(groups),
        "screen_group_quantities_path": str(npz_path),
        "screen_group_quantities_sha256": sha256_file(npz_path),
        "strongest_standard_baseline": str(final["strongest_standard_baseline"]),
        "interaction_threshold": float(threshold),
        "interaction_threshold_source": final.get("interaction_threshold_source", ""),
        "tstar_stratum_split": TSTAR_SPLIT,
        "bootstrap_entropy": s4cr2.BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": s4cr2.BOOTSTRAP_RESAMPLES,
        "confidence_screen": s4cr2.CONFIDENCE_SCREEN,
        "qa": qa,
        "bindings": {
            "data_sha256": inference["data_sha256"],
            "predictions_sha256": inference["predictions_sha256"],
            "scores_sha256": inference["scores_sha256"],
            "collection_receipt_sha256": inference["collection_receipt_sha256"],
            "direct_scores_sha256": direct_receipt["direct_scores_sha256"],
            "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
            "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
            "final_selection_sha256": sha256_file(paths["final_selection"]),
            "model_inventory_sha256": sha256_file(paths["model_inventory"]),
            "statistics_code_sha256": sha256_file(
                ROOT / "code/ac_twin/stage4cr2_statistics.py"),
        },
        "lock_sha256": sha256_file(paths["lock"]),
        "lock_burned_at": lock_payload.get("burned_at", ""),
        "train_support_binding": {
            "report_path": str(support_report_path),
            "report_sha256": sha256_file(support_report_path),
            "report_status": support_payload.get("status", ""),
            **superseded,
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["quantities_receipt"], receipt)
    return npz_path


def load_screen_group_quantities(
    npz_path: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    """Load the create-once quantities NPZ + receipt into the gate-input mapping."""
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
    quantities["interaction_threshold"] = float(
        np.asarray(data["interaction_threshold"]).item())
    quantities["qa"] = {key: bool(value) for key, value in receipt["qa"].items()}
    _validate_quantities(quantities)
    return quantities


__all__ = [
    "ALL_BASELINE_NAMES",
    "BootstrapResult",
    "COVERAGE_MAIN",
    "N_QUANTITIES",
    "N_SEEDS",
    "QA_KEYS",
    "Q_AURC_BASELINE",
    "Q_BASE_MEAN_REGRET",
    "Q_BASE_VS_RANDOM",
    "Q_CLS_I_AURC",
    "Q_CLS_I_R80",
    "Q_HIGH_REGRET_FRAC",
    "Q_IDLE_MEAN_REGRET",
    "Q_I_AURC",
    "Q_INTERACT_HIGH",
    "Q_INTERACT_LOW",
    "Q_I_REGRET80",
    "Q_MU_I_AURC",
    "Q_MU_I_R80",
    "Q_PR",
    "Q_RANDOM_MEAN_REGRET",
    "Q_R80_BASELINE",
    "Q_SEED_I_AURC",
    "Q_SEED_I_R80",
    "Q_SPEARMAN_BRANCH",
    "Q_SPEARMAN_MARGIN",
    "Q_TRIM1",
    "Q_TRIM5",
    "Q_TSTAR_HIGH",
    "Q_TSTAR_LOW",
    "REQUIRED_ARRAY_KEYS",
    "SEED_I_AURC_INDEX",
    "SEED_I_R80_INDEX",
    "STATUS_TASK_QUALIFIED",
    "STRATA_INDEX",
    "ScreenDecision",
    "TRIM_INDEX",
    "TSTAR_SPLIT",
    "assemble_screen_group_quantities",
    "assert_screen_adjudication_open",
    "burn_screen_lock",
    "compute_screen_quantity_vector",
    "evaluate_screen_gates",
    "freeze_screen_direct_scores",
    "load_screen_group_quantities",
    "paired_group_bootstrap",
    "screen_artifact_paths",
]
