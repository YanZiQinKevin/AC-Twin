"""Stage 4C-R2 independent-confirmation statistics: score freeze + one-shot adjudication.

Plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`` sections 6, 7, 8 and 9 Task 3.

The statistical unit is the 10-action decision GROUP. The ONE paired bootstrap runs
10,000 whole-group resamples with entropy ``2026093202``; EVERY metric of a draw
(primary endpoints, marginal strata, trims, five seeds, ablations and the two
non-primary standard-baseline comparisons) is recomputed from the SAME resample index
vector, and the two-sided 95% CI is read off the (2.5%, 97.5%) resample quantiles —
never the development 80% CI and never an SE back-computed from it. Metric definitions
are REUSED bitwise from ``stage4cp_statistics``/``stage4_statistics`` (AURC_REGRET,
REGRET@80 via ``coverage_index``, PR-AUC-prevalence, fail-closed relative improvement,
Spearman); nothing is redefined and no quantile is refit on confirm data.

Frozen quantity-vector layout (``N_QUANTITIES = 52``)::

    0   I_AURC_REGRET  (Direct ensemble vs frozen primary baseline SEED_VOTE_ENTROPY)
    1   I_REGRET@80    (same primary baseline)
    2   PR-AUC - prevalence           (ensemble, high-regret label)
    3   branch-level Spearman(J_hat, J_true)
    4   base-vs-RANDOM_ACTION mean-regret improvement
    5   trim-1% aggregate I_AURC_REGRET
    6   trim-5% aggregate I_AURC_REGRET
    7   t_star-low stratum I_AURC     (t_star < 13; fixed split, never refit)
    8   t_star-high stratum I_AURC    (t_star >= 13)
    9   interaction-low stratum I_AURC  (min distance <= 29.128996473819946)
    10  interaction-high stratum I_AURC (min distance >  29.128996473819946)
    11  Spearman(ensemble, MARGIN_ONLY)
    12  AURC_REGRET of the primary baseline   (fail-closed denominator)
    13  REGRET@80 of the primary baseline     (fail-closed denominator)
    14  base mean regret
    15  ALWAYS_IDLE mean regret (ACTION_PLANS index 3 == (IDLE,3), asserted)
    16  RANDOM_ACTION mean regret (mean over ALL TEN actions of the group)
    17  high-regret fraction (regret > 0.01)
    18..22  per-seed I_AURC_REGRET   (five frozen model seeds)
    23..27  per-seed I_REGRET@80
    28  MU_ONLY I_AURC   (non-blocking ablation)
    29  MU_ONLY I_REGRET@80
    30  CLASS_ONLY I_AURC   (non-blocking ablation)
    31  CLASS_ONLY I_REGRET@80
    32  MARGIN_ONLY I_AURC    (Table C2 ONLY — never enters the Gate)
    33  MARGIN_ONLY I_REGRET@80
    34  SEED_MARGIN_STD I_AURC    (Table C2 ONLY)
    35  SEED_MARGIN_STD I_REGRET@80
    36  AURC_REGRET of the ensemble           (Table C2 absolute)
    37  REGRET@80 of the ensemble
    38..45  absolute AURC_REGRET / REGRET@80 of MARGIN_ONLY, SEED_MARGIN_STD,
            COMBINED_TOP2_MAX, CONSTANT_RANDOM
    46..51  absolute average precision of ensemble, MARGIN_ONLY, SEED_VOTE_ENTROPY,
            SEED_MARGIN_STD, COMBINED_TOP2_MAX, CONSTANT_RANDOM

One-shot order (plan section 6.4) is enforced by IO, not by convention:
``run_determinism_recheck`` -> ``freeze_scores`` -> ``burn_confirmation_lock``
(create-once; the burn binds the truth hash FROM the collection receipt and NEVER
reads the truth bytes) -> ``load_confirmation_truth`` (judge-only, after the lock) ->
``assemble_confirm_quantities`` (create-once) -> ``adjudicate_confirmation`` (exactly
once). A second adjudication with formal results present is refused. Post-burn recovery
is same-transaction only: identical bootstrap seed/config and hash-verified inputs;
anything unprovable stops at ``STAGE4CR2_CONFIRM_RESOURCE_AT_RISK``. Baseline
denominators are fail-closed: a non-positive or non-finite AURC_REGRET / REGRET@80 of
the frozen primary baseline yields NaN improvements and the gate returns
``STAGE4CR2_CONFIRM_INVALID_ANALYSIS`` — never an epsilon, an absolute difference or a
baseline swap. The main baseline is SEED_VOTE_ENTROPY and ONLY it enters the Gate;
MARGIN_ONLY / SEED_MARGIN_STD comparisons on the SAME paired draws go to Table C2 only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ac_twin import stage4_statistics as s4s
from ac_twin import stage4c_objective as s4co
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_protocol as s4cpp
from ac_twin import stage4cp_statistics as s4cs
from ac_twin import stage4cr2_confirm_inference as cinf
from ac_twin import stage4cr2_confirm_protocol as prot
from ac_twin.checkpointing import atomic_save_npz, canonical_fingerprint
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = prot.ROOT

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
Q_MARGIN_I_AURC = 32   # Table C2 only — NEVER gate-relevant
Q_MARGIN_I_R80 = 33
Q_STD_I_AURC = 34      # Table C2 only — NEVER gate-relevant
Q_STD_I_R80 = 35
Q_AURC_MODEL = 36
Q_R80_MODEL = 37
Q_AURC_MARGIN = 38
Q_R80_MARGIN = 39
Q_AURC_STD = 40
Q_R80_STD = 41
Q_AURC_COMBINED = 42
Q_R80_COMBINED = 43
Q_AURC_RANDOM = 44
Q_R80_RANDOM = 45
Q_AP_MODEL = 46
Q_AP_MARGIN = 47
Q_AP_SVE = 48
Q_AP_STD = 49
Q_AP_COMBINED = 50
Q_AP_RANDOM = 51
N_QUANTITIES = 52
N_SEEDS = len(prot.MODEL_SEEDS)

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
) + tuple(f"seed{s}_i_aurc" for s in prot.MODEL_SEEDS) \
    + tuple(f"seed{s}_i_regret80" for s in prot.MODEL_SEEDS) \
    + ("mu_only_i_aurc", "mu_only_i_regret80", "class_only_i_aurc",
       "class_only_i_regret80", "margin_only_i_aurc", "margin_only_i_regret80",
       "seed_margin_std_i_aurc", "seed_margin_std_i_regret80", "aurc_model",
       "regret80_model", "aurc_margin_only", "regret80_margin_only",
       "aurc_seed_margin_std", "regret80_seed_margin_std", "aurc_combined_top2_max",
       "regret80_combined_top2_max", "aurc_constant_random", "regret80_constant_random",
       "ap_model", "ap_margin_only", "ap_seed_vote_entropy", "ap_seed_margin_std",
       "ap_combined_top2_max", "ap_constant_random")

#: coverage rounding is REUSED from stage4_statistics — never an own floor/ceil.
coverage_index = s4s.coverage_index

COVERAGE = prot.COVERAGE  # 0.80, frozen
TSTAR_SPLIT = prot.T_STAR_STRATUM_SPLIT  # 13, frozen fixed split (never refit)
TRIM_FRACTIONS = prot.GATE_RISK_TRIM_FRACTIONS  # (0.01, 0.05), frozen

STATUS_TASK_QUALIFIED = "TASK_CONFIRM_QUALIFIED"

#: the ONLY baseline that may enter the Gate (validation-frozen; never re-selected).
PRIMARY_BASELINE = prot.PRIMARY_BASELINE
ALL_BASELINE_NAMES = tuple(prot.STANDARD_BASELINES) + tuple(prot.REFERENCE_BASELINES)

#: sealed-truth schema — MIRRORS collect_stage4cr2_merge_confirm.TRUTH_FIELDS (the
#: collector pulls highway_env, so it is never imported here); the collection receipt's
#: ``truth_fields`` is cross-checked against this mirror at read time (fail-loud drift).
TRUTH_FIELDS = (
    "object_counterfactual_futures",  # (G,10,20,5,8) float32 — the sealed CF truth
    "cf_actions",                     # (G,10) int64 — CF mechanism
    "cf_durations",                   # (G,10) int64 — CF mechanism
    "factual_actions",                # (G,20) int64 — integrity copy
    "t_star",                         # (G,)   int64 — integrity copy
    "group_seed",                     # (G,)   int64 — same-order binding
    "inference_mask",                 # (G,20,5) bool — integrity copy
)
#: mirrors the collector's completed status (same no-import rationale).
STATUS_COLLECTED = "COLLECTED"

#: per-group arrays the quantities mapping must carry (all aligned on the group axis).
REQUIRED_ARRAY_KEYS = (
    "group_seed", "regret", "y_high", "random_regret", "idle_regret",
    "ensemble_risk", "risk_by_seed", "mu_only_ensemble", "class_only_ensemble",
    "tstar_layer", "interaction_layer", "J_hat", "J_true", "a_hat", "a_star",
    "t_star", "interaction_min_distance", "trim1_keep", "trim5_keep",
)

#: baseline risk arrays persisted in the quantities NPZ (prefix stripped on load).
_BASELINE_NPZ_PREFIX = "baseline_risk__"

#: QA flags evaluated inside the gates (plan 6.2 condition 1 / 6.3 condition R9).
QA_KEYS = (
    "all_complete", "hashes_match", "role_seeds_valid", "determinism_passed",
    "one_shot_locks", "ensemble_is_arithmetic_mean",
)

#: action-id semantics for the preregistered a_hat secondary analysis (plan section 7).
ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

#: exact section-7 paper-boundary sentences (verbatim; the success sentence is legal
#: ONLY when the terminal status is STAGE4CR2_DECISION_RISK_CONFIRMED).
SUCCESS_SENTENCE = (
    "在所测Merge、回溯事实后缀可用的十动作任务中，直接监督决策风险审计改善了选择性保留组的regret。")
ACTION_FREQUENCY_CAVEAT = (
    "确认集必须原样报告动作频率；即使总体Gate通过，也不得声称十动作获得均衡证据、"
    "每种动作同等可靠或稀有动作已被充分验证。")
PROHIBITED_CLAIMS = (
    "禁止由此声称：复杂DCR结构优势、响应特征的独立因果贡献、仅增加数据导致成功、"
    "跨场景/跨平台泛化、线上预测可用、实际碰撞率降低或闭环策略改善。")

POSITION_SLICE = cinf.POSITION_SLICE


class ResourceAtRiskError(RuntimeError):
    """A post-burn transaction cannot be proven consistent (plan section 8).

    Carries the terminal status ``STAGE4CR2_CONFIRM_RESOURCE_AT_RISK``; raised instead
    of overwriting or re-deriving anything once the confirmation lock is burned.
    """

    status = prot.STATUS_RESOURCE_AT_RISK


# --- result containers -----------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """One whole-group paired bootstrap: identity point + 95% CI + raw draws."""
    point: np.ndarray        # (N_QUANTITIES,)
    ci_lower: np.ndarray     # (N_QUANTITIES,) two-sided 95% (2.5% quantile)
    ci_upper: np.ndarray     # (N_QUANTITIES,) two-sided 95% (97.5% quantile)
    draws: np.ndarray        # (n_resamples, N_QUANTITIES), shared indices per row
    n_resamples: int
    seed: int
    confidence: float


@dataclass(frozen=True)
class ConfirmDecision:
    """The single confirmation adjudication (Gate TASK-CONFIRM + Gate RISK-CONFIRM)."""
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
            "gate_task_confirm": self.task,
            "gate_risk_decision_confirm": self.risk,
            "metrics": self.metrics,
            "ablations": self.ablations,
            "bootstrap": self.bootstrap,
        }


# --- path contract ---------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def confirm_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every one-shot confirm path (plan section 10 + the section-7 files)."""
    checked = prot.validate_config(config)
    paths = checked["paths"]
    report_dir = _resolve(paths["confirmatory_report"]).parent
    return {
        "report_dir": report_dir,
        "protocol_lock": _resolve(paths["protocol_lock"]),
        "collection_receipt": _resolve(paths["collection_receipt"]),
        "inputs": _resolve(paths["inference_inputs"]),
        "truth": _resolve(paths["confirmation_truth"]),
        "predictions": _resolve(paths["inference_predictions"]),
        "risk_scores": _resolve(paths["inference_risk_scores"]),
        "inference_receipt": _resolve(paths["inference_receipt"]),
        "determinism_receipt": _resolve(paths["determinism_receipt"]),
        "score_freeze_receipt": _resolve(paths["score_freeze_receipt"]),
        "lock": _resolve(paths["confirmation_used_lock"]),
        "quantities_npz": _resolve(paths["confirm_group_quantities"]),
        "quantities_receipt": _resolve(paths["confirm_group_quantities_receipt"]),
        "report": _resolve(paths["confirmatory_report"]),
        "decision": _resolve(paths["confirmatory_decision"]),
        "table_c1": report_dir / "Table_C1_gates.csv",
        "table_c2": report_dir / "Table_C2_baselines.csv",
        "fig_c1": report_dir / "Fig_C1_risk_coverage.png",
        "table_c3": report_dir / "Table_C3_seeds_strata_trim.csv",
        "table_c4": report_dir / "Table_C4_ablation.csv",
        "table_c5": report_dir / "Table_C5_task.csv",
        "secondary_csv": report_dir / "secondary_analyses.csv",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config_signature(checked: Mapping[str, Any]) -> str:
    """The same canonical config signature the protocol/collector locks bind."""
    return canonical_fingerprint(json.loads(json.dumps(dict(checked))))


# --- quantities validation ---------------------------------------------------------------------


def _validate_quantities(quantities: Mapping[str, Any]) -> int:
    """Fail closed on any missing/misaligned/non-finite per-group quantity."""
    for key in REQUIRED_ARRAY_KEYS:
        if key not in quantities:
            raise ValueError(f"confirm quantities missing array {key!r}")
    regret = np.asarray(quantities["regret"], dtype=np.float64).ravel()
    groups = int(regret.size)
    if groups < 2:
        raise ValueError("confirm quantities need at least 2 groups")
    if int(quantities.get("n_groups", groups)) != groups:
        raise ValueError("n_groups does not match the regret axis")
    for key in ("group_seed", "y_high", "random_regret", "idle_regret", "ensemble_risk",
                "mu_only_ensemble", "class_only_ensemble", "tstar_layer",
                "interaction_layer", "a_hat", "a_star", "t_star",
                "interaction_min_distance"):
        arr = np.asarray(quantities[key]).ravel()
        if arr.shape != (groups,):
            raise ValueError(f"quantity {key!r} must align as ({groups},)")
    for key in ("trim1_keep", "trim5_keep"):
        arr = np.asarray(quantities[key])
        if arr.shape != (groups,) or arr.dtype != bool:
            raise ValueError(f"quantity {key!r} must be a bool ({groups},) keep mask")
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
    if j_hat.shape != (groups, prot.N_ACTIONS) or j_true.shape != j_hat.shape:
        raise ValueError("J_hat/J_true must align as (G, 10)")
    baselines = quantities.get("baseline_risks")
    if not isinstance(baselines, Mapping):
        raise ValueError("confirm quantities missing baseline_risks mapping")
    for name in ALL_BASELINE_NAMES:
        if name not in baselines:
            raise ValueError(f"baseline_risks missing {name!r}")
        risk = np.asarray(baselines[name], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"baseline risk {name!r} must be finite and group-aligned")
    strongest = str(quantities.get("strongest_baseline", ""))
    if strongest != PRIMARY_BASELINE:
        raise ValueError(
            f"the confirm never re-selects its baseline: strongest_baseline must be "
            f"{PRIMARY_BASELINE!r}, got {strongest!r}")
    qa = quantities.get("qa")
    if not isinstance(qa, Mapping) or any(key not in qa for key in QA_KEYS):
        raise ValueError(f"confirm quantities qa must carry {list(QA_KEYS)}")
    return groups


# --- decision quantities (section 3.2, verbatim) -------------------------------------------------


def _decision_quantities(j_hat: np.ndarray, j_true: np.ndarray) -> dict[str, np.ndarray]:
    """a_hat/a_star/regret/random/idle/y_high from the two consequence matrices.

    Tie-breaks are the fixed ACTION_PLANS order (np.argmin first occurrence; never
    regret-assisted). The ALWAYS_IDLE reference is ACTION_PLANS index 3 == (IDLE,3) —
    asserted, never (IDLE,1). RANDOM_ACTION is the mean over ALL TEN actions of the
    group. Regret uses the frozen clamp: [-1e-10, 0) -> 0, below -1e-10 raises,
    positive values are kept as-is.
    """
    if s4co.ACTION_PLANS[s4co.IDLE_PLAN_INDEX] != (1, 3):
        raise RuntimeError("ALWAYS_IDLE reference must stay (IDLE,3), never (IDLE,1)")
    j_hat = np.asarray(j_hat, dtype=np.float64)
    j_true = np.asarray(j_true, dtype=np.float64)
    if j_hat.shape != j_true.shape or j_hat.ndim != 2 \
            or j_hat.shape[1] != prot.N_ACTIONS:
        raise ValueError("J_hat/J_true must align as (G, 10)")
    rows = np.arange(j_true.shape[0])
    a_hat = np.argmin(j_hat, axis=-1)
    a_star = np.argmin(j_true, axis=-1)
    j_true_star = j_true[rows, a_star]
    regret = s4co.clamp_regret(j_true[rows, a_hat] - j_true_star)
    return {
        "a_hat": a_hat.astype(np.int64),
        "a_star": a_star.astype(np.int64),
        "regret": regret,
        "random_regret": s4co.clamp_regret(j_true.mean(axis=-1) - j_true_star),
        "idle_regret": s4co.clamp_regret(
            j_true[:, s4co.IDLE_PLAN_INDEX] - j_true_star),
        "y_high": (regret > prot.HIGH_REGRET_THRESHOLD).astype(np.int64),
    }


def _average_precision(risk: np.ndarray, label: np.ndarray) -> float:
    """Absolute average precision; a degenerate label yields the prevalence (PR == 0).

    The degenerate branch mirrors ``stage4cp_statistics.high_regret_pr`` and avoids the
    sklearn undefined-metrics warning (the suite runs with ``-W error``).
    """
    label = np.asarray(label, dtype=np.int64)
    prevalence = float(label.mean())
    if label.min() == label.max():
        return prevalence
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(label, np.asarray(risk, dtype=np.float64)))


# --- the joint quantity vector ------------------------------------------------------------------


def compute_confirm_quantity_vector(
    index: np.ndarray, quantities: Mapping[str, Any]
) -> np.ndarray:
    """Evaluate all 52 quantities on ONE whole-group resample ``index``.

    ``index`` is a with-replacement integer resample into the confirm groups, so every
    metric of the draw — primary endpoints, marginal strata, trims, five seeds,
    ablations AND the Table-C2-only MARGIN_ONLY / SEED_MARGIN_STD comparisons — stays
    coupled through the same group multiplicities. Improvement quantities are NaN when
    the frozen primary-baseline denominator is non-positive (the gate turns that into
    ``STAGE4CR2_CONFIRM_INVALID_ANALYSIS``); a stratum/trim subset with fewer than two
    resampled members yields NaN, which fails its gate condition. The efficiency
    normalization and action set are NEVER re-derived inside a draw: all ten actions
    per group enter through the pre-computed per-group quantities.
    """
    index = np.asarray(index, dtype=np.int64)
    reg = np.asarray(quantities["regret"], dtype=np.float64)[index]
    y_high = np.asarray(quantities["y_high"], dtype=np.int64)[index]
    ens = np.asarray(quantities["ensemble_risk"], dtype=np.float64)[index]
    baselines = quantities["baseline_risks"]
    base = np.asarray(baselines[PRIMARY_BASELINE], dtype=np.float64)[index]
    margin = np.asarray(baselines["MARGIN_ONLY"], dtype=np.float64)[index]
    std = np.asarray(baselines["SEED_MARGIN_STD"], dtype=np.float64)[index]
    combined = np.asarray(baselines["COMBINED_TOP2_MAX"], dtype=np.float64)[index]
    random_risk = np.asarray(baselines["CONSTANT_RANDOM"], dtype=np.float64)[index]
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)[:, index]
    mu_only = np.asarray(quantities["mu_only_ensemble"], dtype=np.float64)[index]
    class_only = np.asarray(quantities["class_only_ensemble"], dtype=np.float64)[index]

    order_ens = np.argsort(ens, kind="stable")
    order_base = np.argsort(base, kind="stable")
    order_margin = np.argsort(margin, kind="stable")
    order_std = np.argsort(std, kind="stable")
    a_base = s4cs.aurc_regret(order_base, reg)
    a_ens = s4cs.aurc_regret(order_ens, reg)
    r80_base = s4cs.regret_at_coverage(order_base, reg, COVERAGE)
    r80_ens = s4cs.regret_at_coverage(order_ens, reg, COVERAGE)
    a_margin = s4cs.aurc_regret(order_margin, reg)
    r80_margin = s4cs.regret_at_coverage(order_margin, reg, COVERAGE)
    a_std = s4cs.aurc_regret(order_std, reg)
    r80_std = s4cs.regret_at_coverage(order_std, reg, COVERAGE)
    order_combined = np.argsort(combined, kind="stable")
    order_random = np.argsort(random_risk, kind="stable")

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

    for slot, frac in ((Q_TRIM1, TRIM_FRACTIONS[0]), (Q_TRIM5, TRIM_FRACTIONS[1])):
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
    q[Q_HIGH_REGRET_FRAC] = float((reg > prot.HIGH_REGRET_THRESHOLD).mean())

    for i in range(N_SEEDS):
        order_seed = np.argsort(by_seed[i], kind="stable")
        q[Q_SEED_I_AURC + i] = s4cs.improvement(
            a_base, s4cs.aurc_regret(order_seed, reg))
        q[Q_SEED_I_R80 + i] = s4cs.improvement(
            r80_base, s4cs.regret_at_coverage(order_seed, reg, COVERAGE))
    q[Q_MU_I_AURC] = s4cs.improvement(
        a_base, s4cs.aurc_regret(np.argsort(mu_only, kind="stable"), reg))
    q[Q_MU_I_R80] = s4cs.improvement(
        r80_base, s4cs.regret_at_coverage(np.argsort(mu_only, kind="stable"), reg,
                                          COVERAGE))
    q[Q_CLS_I_AURC] = s4cs.improvement(
        a_base, s4cs.aurc_regret(np.argsort(class_only, kind="stable"), reg))
    q[Q_CLS_I_R80] = s4cs.improvement(
        r80_base, s4cs.regret_at_coverage(np.argsort(class_only, kind="stable"), reg,
                                          COVERAGE))

    # Table-C2-only comparisons on the SAME draws; never gate-relevant, never a swap.
    q[Q_MARGIN_I_AURC] = s4cs.improvement(a_margin, a_ens)
    q[Q_MARGIN_I_R80] = s4cs.improvement(r80_margin, r80_ens)
    q[Q_STD_I_AURC] = s4cs.improvement(a_std, a_ens)
    q[Q_STD_I_R80] = s4cs.improvement(r80_std, r80_ens)

    q[Q_AURC_MODEL] = a_ens
    q[Q_R80_MODEL] = r80_ens
    q[Q_AURC_MARGIN] = a_margin
    q[Q_R80_MARGIN] = r80_margin
    q[Q_AURC_STD] = a_std
    q[Q_R80_STD] = r80_std
    q[Q_AURC_COMBINED] = s4cs.aurc_regret(order_combined, reg)
    q[Q_R80_COMBINED] = s4cs.regret_at_coverage(order_combined, reg, COVERAGE)
    q[Q_AURC_RANDOM] = s4cs.aurc_regret(order_random, reg)
    q[Q_R80_RANDOM] = s4cs.regret_at_coverage(order_random, reg, COVERAGE)

    q[Q_AP_MODEL] = _average_precision(ens, y_high)
    q[Q_AP_MARGIN] = _average_precision(margin, y_high)
    q[Q_AP_SVE] = _average_precision(base, y_high)
    q[Q_AP_STD] = _average_precision(std, y_high)
    q[Q_AP_COMBINED] = _average_precision(combined, y_high)
    q[Q_AP_RANDOM] = _average_precision(random_risk, y_high)
    return q


# --- the one paired bootstrap ----------------------------------------------------------------------


def paired_group_bootstrap(
    quantities: Mapping[str, np.ndarray],
    *,
    seed: int = prot.BOOTSTRAP_ENTROPY,
    n_resamples: int = prot.BOOTSTRAP_RESAMPLES,
) -> BootstrapResult:
    """10,000-draw whole-group paired bootstrap with shared indices and 95% CIs.

    EVERY quantity of a draw is recomputed from the SAME whole-group resample index
    vector; the identity evaluation is the point estimate; the two-sided 95% CI is the
    (0.025, 0.975) resample quantile band. The confidence level is frozen at
    ``prot.CONFIDENCE`` (0.95) — never re-parameterised per call site and never the
    development 80% CI.
    """
    groups = _validate_quantities(quantities)
    generator = np.random.default_rng(int(seed))
    identity = np.arange(groups, dtype=np.int64)
    point = compute_confirm_quantity_vector(identity, quantities)
    draws = np.empty((int(n_resamples), N_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = compute_confirm_quantity_vector(index, quantities)
    alpha = (1.0 - prot.CONFIDENCE) / 2.0
    return BootstrapResult(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws,
        n_resamples=int(n_resamples),
        seed=int(seed),
        confidence=float(prot.CONFIDENCE),
    )


# --- the gates -------------------------------------------------------------------------------------


def _named(vector: np.ndarray) -> dict[str, float]:
    return {name: float(vector[i]) for i, name in enumerate(_QUANTITY_NAMES)}


def evaluate_confirmation_gates(
    quantities: Mapping[str, np.ndarray], bootstrap: BootstrapResult
) -> ConfirmDecision:
    """Adjudicate Gate TASK-CONFIRM and Gate RISK-DECISION-CONFIRM exactly once.

    Terminal-state priority (plan section 8, literal order): source/code/seed/shape/
    mask/identity/hash (QA) failure -> ``STAGE4CR2_CONFIRM_INVALID_ANALYSIS``; TASK
    condition 2/3/7 failure -> ``STAGE4CR2_CONFIRM_TASK_DEGENERATE``; condition 4/5/6
    failure -> ``INCONCLUSIVE_STAGE4CR2_CONFIRM_DECISION_MODEL_INSUFFICIENT``; a
    non-positive or non-finite primary-baseline denominator -> INVALID; any unblocked
    risk condition failure -> ``STAGE4CR2_DECISION_RISK_NOT_CONFIRMED``; all pass ->
    ``STAGE4CR2_DECISION_RISK_CONFIRMED``. Non-positive denominators are never repaired
    (no epsilon, no absolute difference, no baseline swap); NaN quantities fail their
    conditions by construction.
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
    if float(bootstrap.confidence) != float(prot.CONFIDENCE):
        raise ValueError("the confirm CI level is frozen at 0.95 (never the dev 0.80)")

    qa = {key: bool(quantities["qa"][key]) for key in QA_KEYS}
    qa_passed = all(qa.values())

    # --- Gate TASK-CONFIRM (plan section 6.2) --------------------------------------------
    oracle = np.asarray(quantities["a_star"], dtype=np.int64)
    oracle_shares = [float((oracle == a).mean()) for a in range(prot.N_ACTIONS)]
    n_oracle = int(sum(s >= prot.GATE_TASK_MIN_ORACLE_FRAC for s in oracle_shares))
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)
    stratum_counts = {
        "tstar_low": int((tstar == 0).sum()), "tstar_high": int((tstar == 1).sum()),
        "interaction_low": int((inter == 0).sum()),
        "interaction_high": int((inter == 1).sum()),
    }
    task_conditions = {
        "c1_all_complete_gate_q": bool(
            qa_passed and groups == prot.CONFIRM_ACCEPTED_GROUPS),
        "c2_high_regret_frac_ge_15pct": bool(
            point[Q_HIGH_REGRET_FRAC] >= prot.GATE_TASK_MIN_HIGH_REGRET_FRAC),
        "c3_at_least_3_oracle_actions_ge_5pct": bool(
            n_oracle >= prot.GATE_TASK_MIN_DISTINCT_ORACLE),
        "c4_branch_spearman_ci_lower_positive": bool(
            ci_lower[Q_SPEARMAN_BRANCH] > 0.0),
        "c5_base_vs_random_point_and_ci_positive": bool(
            point[Q_BASE_VS_RANDOM] > 0.0 and ci_lower[Q_BASE_VS_RANDOM] > 0.0),
        "c6_base_not_worse_than_idle": bool(
            np.isfinite([point[Q_BASE_MEAN_REGRET], point[Q_IDLE_MEAN_REGRET]]).all()
            and point[Q_BASE_MEAN_REGRET] <= point[Q_IDLE_MEAN_REGRET] + 1e-12),
        "c7_strata_ge_200_groups": bool(
            min(stratum_counts.values()) >= prot.GATE_TASK_MIN_STRATUM_GROUPS),
    }
    degenerate = not (task_conditions["c2_high_regret_frac_ge_15pct"]
                      and task_conditions["c3_at_least_3_oracle_actions_ge_5pct"]
                      and task_conditions["c7_strata_ge_200_groups"])
    insufficient = not (task_conditions["c4_branch_spearman_ci_lower_positive"]
                        and task_conditions["c5_base_vs_random_point_and_ci_positive"]
                        and task_conditions["c6_base_not_worse_than_idle"])
    if degenerate:
        task_status = prot.STATUS_TASK_DEGENERATE
    elif insufficient:
        task_status = prot.STATUS_MODEL_INSUFFICIENT
    else:
        task_status = STATUS_TASK_QUALIFIED

    # --- Gate RISK-DECISION-CONFIRM (plan section 6.3) ------------------------------------
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
        "r1_i_aurc_regret": bool(point[Q_I_AURC] >= prot.GATE_RISK_I_AURC_MIN
                                 and ci_lower[Q_I_AURC] > 0.0),
        "r2_i_regret80": bool(point[Q_I_REGRET80] >= prot.GATE_RISK_I_REGRET80_MIN
                              and ci_lower[Q_I_REGRET80] > 0.0),
        "r3_pr_auc_minus_prevalence": bool(point[Q_PR] > 0.0 and ci_lower[Q_PR] > 0.0),
        "r4_seeds_i_aurc": bool(seeds_i_aurc_positive
                                >= prot.GATE_RISK_MIN_POSITIVE_SEED_METRICS),
        "r5_seeds_i_regret80": bool(seeds_i_r80_positive
                                    >= prot.GATE_RISK_MIN_POSITIVE_SEED_METRICS),
        "r6_strata_floor": bool(
            all(np.isfinite(list(strata.values())))
            and min(strata.values()) >= prot.GATE_RISK_STRATA_I_AURC_FLOOR),
        "r7_trims_positive": bool(all(np.isfinite(list(trims.values())))
                                  and min(trims.values()) > 0.0),
        "r8_margin_spearman": bool(
            np.isfinite(point[Q_SPEARMAN_MARGIN])
            and abs(point[Q_SPEARMAN_MARGIN]) < prot.GATE_RISK_SPEARMAN_MARGIN_CEILING),
        "r9_qa_and_one_shot_locks": bool(qa_passed),
    }

    # --- terminal-state priority (plan section 8, literal order) ---------------------------
    if not qa_passed or not task_conditions["c1_all_complete_gate_q"]:
        # source/code/seed/shape/mask/identity/hash (incl. the frozen G=2,000 group-count
        # identity of condition 1) failures are INVALID before any task/risk verdict.
        status = prot.STATUS_INVALID
    elif degenerate:
        status = prot.STATUS_TASK_DEGENERATE
    elif insufficient:
        status = prot.STATUS_MODEL_INSUFFICIENT
    elif not baseline_valid:
        status = prot.STATUS_INVALID
    elif all(risk_conditions.values()):
        status = prot.STATUS_CONFIRMED
    else:
        status = prot.STATUS_NOT_CONFIRMED

    # --- non-blocking ablations + the three-baseline Table C2 block (plan 6.3 note) --------
    prevalence = float(np.asarray(quantities["y_high"], dtype=np.int64).mean())
    baseline_rows: dict[str, Any] = {}
    abs_slots = {
        "MARGIN_ONLY": (Q_AURC_MARGIN, Q_R80_MARGIN, Q_AP_MARGIN),
        "SEED_VOTE_ENTROPY": (Q_AURC_BASELINE, Q_R80_BASELINE, Q_AP_SVE),
        "SEED_MARGIN_STD": (Q_AURC_STD, Q_R80_STD, Q_AP_STD),
        "COMBINED_TOP2_MAX": (Q_AURC_COMBINED, Q_R80_COMBINED, Q_AP_COMBINED),
        "CONSTANT_RANDOM": (Q_AURC_RANDOM, Q_R80_RANDOM, Q_AP_RANDOM),
    }
    improvement_slots = {
        "MARGIN_ONLY": (Q_MARGIN_I_AURC, Q_MARGIN_I_R80),
        "SEED_VOTE_ENTROPY": (Q_I_AURC, Q_I_REGRET80),
        "SEED_MARGIN_STD": (Q_STD_I_AURC, Q_STD_I_R80),
    }
    for name in ALL_BASELINE_NAMES:
        a_slot, r_slot, ap_slot = abs_slots[name]
        entry: dict[str, Any] = {
            "aurc_regret": float(point[a_slot]),
            "regret_at_80": float(point[r_slot]),
            "average_precision": float(point[ap_slot]),
            "pr_auc_minus_prevalence": float(point[ap_slot]) - prevalence,
        }
        if name in improvement_slots:
            ia, ir = improvement_slots[name]
            entry["i_aurc_regret"] = float(point[ia])
            entry["i_aurc_regret_ci95"] = [float(ci_lower[ia]), float(ci_upper[ia])]
            entry["i_regret80"] = float(point[ir])
            entry["i_regret80_ci95"] = [float(ci_lower[ir]), float(ci_upper[ir])]
        if name == PRIMARY_BASELINE:
            entry["role"] = "primary_standard_baseline"
        elif name in prot.STANDARD_BASELINES:
            entry["role"] = "standard_baseline"
        elif name == "COMBINED_TOP2_MAX":
            entry["role"] = "historical_reference"
        else:
            entry["role"] = "curve_reference"
        baseline_rows[name] = entry
    ablations = {
        "blocking": False,
        "note": ("non-blocking ablations reported alongside the primary score; they "
                 "never enter the gates and never substitute for the primary score"),
        "MU_ONLY": {
            "i_aurc_regret": float(point[Q_MU_I_AURC]),
            "i_aurc_regret_ci95": [float(ci_lower[Q_MU_I_AURC]),
                                   float(ci_upper[Q_MU_I_AURC])],
            "i_regret80": float(point[Q_MU_I_R80]),
            "i_regret80_ci95": [float(ci_lower[Q_MU_I_R80]),
                                float(ci_upper[Q_MU_I_R80])],
        },
        "CLASS_ONLY": {
            "i_aurc_regret": float(point[Q_CLS_I_AURC]),
            "i_aurc_regret_ci95": [float(ci_lower[Q_CLS_I_AURC]),
                                   float(ci_upper[Q_CLS_I_AURC])],
            "i_regret80": float(point[Q_CLS_I_R80]),
            "i_regret80_ci95": [float(ci_lower[Q_CLS_I_R80]),
                                float(ci_upper[Q_CLS_I_R80])],
        },
        "baseline_curves": baseline_rows,
        "primary_model": {
            "name": "DIRECT_ENSEMBLE",
            "aurc_regret": float(point[Q_AURC_MODEL]),
            "regret_at_80": float(point[Q_R80_MODEL]),
            "average_precision": float(point[Q_AP_MODEL]),
            "pr_auc_minus_prevalence": float(point[Q_PR]),
        },
    }

    return ConfirmDecision(
        plan_id=prot.PLAN_ID,
        n_groups=int(groups),
        status=status,
        passed=bool(status == prot.STATUS_CONFIRMED),
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
                       "from the SAME resample index vector; two-sided 95% CI from the "
                       "resample quantiles (2.5%/97.5%)"),
        },
    )


# --- hash-chain verification (score side) ---------------------------------------------------------


def _assert_protocol_lock_binding(checked: Mapping[str, Any], paths: Mapping[str, Path]
                                  ) -> dict[str, Any]:
    """Cheap local guard: the formal lock must bind THIS config, schedule and code set.

    Mirrors the collector's ordering guard; the full source/seed re-verification stays
    with ``prot.verify_protocol`` / the runner's ``verify``. The code check is
    recovery-chain aware (plan section 8): the effective code set is the original
    lock's ``code_sha256`` extended by any immutable recovery locks chained from it
    (``prot.resolve_locked_code_hashes``); a broken chain or any uncovered drift fails
    closed.
    """
    lock_path = paths["protocol_lock"]
    if not lock_path.is_file():
        raise ValueError(
            f"the formal protocol lock is missing; freeze/burn require it first: "
            f"{lock_path}")
    payload = _read_json(lock_path)
    if payload.get("plan_id") != prot.PLAN_ID:
        raise ValueError(f"protocol lock plan_id must be {prot.PLAN_ID!r}")
    if payload.get("config_signature") != _config_signature(checked):
        raise ValueError("running config does not match the locked configuration")
    schedule = prot.build_confirm_seed_schedule(checked)
    digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    if payload.get("confirm_schedule_raw_sha256") != digest:
        raise ValueError("the locked confirm schedule raw hash drifted")
    locked_code = prot.resolve_locked_code_hashes(lock_path)
    for rel in prot.PROTOCOL_CODE_FILES:
        file_path = ROOT / "code" / rel
        if not file_path.is_file() or locked_code.get(rel) != sha256_file(file_path):
            raise ValueError(
                f"the protocol lock chain does not bind the current code hash of {rel} "
                "(code drift or a stale lock)")
    return payload


def _verify_inference_chain(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Re-hash every score-side artifact against its create-once receipt (fail-closed).

    Binds the collection receipt: it must exist, be the completed COLLECTED receipt,
    its ``inference_inputs_sha256`` must match the published legal inputs, and the
    inference receipt's ``data_sha256`` must match the same file. The sealed truth is
    NEVER opened here — only its receipt-recorded hash is carried through.
    """
    if not paths["inference_receipt"].is_file():
        raise ValueError(
            f"the confirm inference receipt is missing: {paths['inference_receipt']}")
    inference = _read_json(paths["inference_receipt"])
    for key, path in (("predictions_sha256", paths["predictions"]),
                      ("risk_scores_sha256", paths["risk_scores"])):
        if not path.is_file() or sha256_file(path) != inference.get(key):
            raise ValueError(f"confirm inference artifact drifted from its receipt: {path}")
    if not paths["inputs"].is_file() \
            or sha256_file(paths["inputs"]) != inference.get("data_sha256"):
        raise ValueError("confirm legal inputs drifted from the inference receipt")
    if not paths["collection_receipt"].is_file():
        raise ValueError(
            f"the confirm collection receipt is missing: {paths['collection_receipt']}")
    collection = _read_json(paths["collection_receipt"])
    if collection.get("status") != STATUS_COLLECTED:
        raise ValueError(
            f"the collection receipt status must be {STATUS_COLLECTED!r}; got "
            f"{collection.get('status')!r}")
    if collection.get("inference_inputs_sha256") != inference.get("data_sha256"):
        raise ValueError("collection receipt does not bind the inferred legal inputs")
    expected_truth = str(collection.get("confirmation_truth_sha256", ""))
    if not expected_truth:
        raise ValueError("collection receipt lost the sealed-truth hash binding")
    inference["collection_receipt_sha256"] = sha256_file(paths["collection_receipt"])
    inference["inference_receipt_sha256"] = sha256_file(paths["inference_receipt"])
    inference["collection_accepted_groups"] = int(collection.get("accepted_groups", -1))
    inference["expected_confirmation_truth_sha256"] = expected_truth
    inference["collection_truth_fields"] = list(collection.get("truth_fields", []))
    binding = collection.get("group_order_binding", {})
    inference["collection_same_order"] = bool(binding.get("same_order"))
    inference["collection_inputs_seed_sha256"] = str(
        binding.get("inference_inputs_group_seed_sha256", ""))
    return inference


def _verify_determinism(paths: Mapping[str, Path], inference: Mapping[str, Any]
                        ) -> dict[str, Any]:
    """The 32-group repeated-inference/scoring receipt must exist, pass and bind."""
    if not paths["determinism_receipt"].is_file():
        raise ValueError(
            "the determinism receipt is missing; run the fixed leading-groups repeated "
            "inference/scoring recheck before the score freeze")
    determinism = _read_json(paths["determinism_receipt"])
    if not determinism.get("bitwise_identical"):
        raise ValueError("the repeated inference/scoring was not bitwise identical")
    bindings = determinism.get("bindings", {})
    expected = {
        "predictions_sha256": inference["predictions_sha256"],
        "risk_scores_sha256": inference["risk_scores_sha256"],
        "inference_receipt_sha256": inference["inference_receipt_sha256"],
        "inference_inputs_sha256": inference["data_sha256"],
        "collection_receipt_sha256": inference["collection_receipt_sha256"],
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise ValueError(
                f"determinism receipt is not bound to the formal artifacts ({key})")
    return determinism


# --- plan 6.4: the fixed leading-groups repeated inference/scoring recheck -------------------------


def _resolve_hooks(run_inference_fn, build_features_fn, load_models_fn,
                   score_actions_fn, build_tokens_fn, score_fn) -> dict[str, Any]:
    """Default to the same frozen chain ``infer_and_score`` uses (read-only reuse)."""
    from ac_twin import stage4c_inference as s4ci
    from ac_twin import stage4cr2_training as s4cr2t
    return {
        "run_inference_fn": run_inference_fn or s4ci.run_grouped_frozen_inference,
        "build_features_fn": build_features_fn or s4ci.build_action_risk_features,
        "load_models_fn": load_models_fn or s4ci.load_frozen_models,
        "score_actions_fn": score_actions_fn or s4ci.score_actions,
        "build_tokens_fn": build_tokens_fn or s4cpf.build_relative_tokens,
        "score_fn": score_fn or s4cr2t.score_models,
        "real_chain": run_inference_fn is None,
    }


def _context_aligned_direct_scores(
    score_fn: Callable,
    inventory: Any,
    tokens: np.ndarray,
    global_features: np.ndarray,
    group_seed: np.ndarray,
    *,
    context_rows: int,
    device: str,
) -> Any:
    """Direct five-seed scoring reproduced in the role's ORIGINAL forward context.

    The frozen Direct scoring path (``stage4cr2_training.score_models`` ->
    ``stage4cp_models.predict_decision_risk`` -> ``DirectRiskModel.forward``) runs the
    WHOLE design matrix in ONE unbatched forward, so the GEMM kernel selection (hence
    the float32 accumulation order) depends on the total row count M — the same
    row-count-context dependence Task 1 documented for the e2e router and the
    COMBINED MLP. The formal inference scored all G groups in one call; a recheck
    scores only a leading subset, and M_formal != M_recheck diverges at the float32
    ULP level (probe: 146/160 risk_by_seed elements, max 5.992e-08). The Direct model
    is a strictly per-row MLP (Linear -> per-row LayerNorm -> ReLU; eval-mode dropout
    is the identity), so a row's output depends only on its own input row given a
    fixed kernel: zero-padding the inputs to the formal row count and slicing the
    leading rows back reproduces the formal scores bit-for-bit (probe-verified on all
    five seeds, the ensemble and both ablations). No tolerance is ever substituted
    for bitwise identity; the model math, the frozen scaler and the inputs are
    untouched — only the forward's row context is reproduced.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    global_features = np.asarray(global_features, dtype=np.float64)
    group_seed = np.asarray(group_seed, dtype=np.int64).ravel()
    rows = int(tokens.shape[0])
    if int(context_rows) < rows:
        raise RuntimeError(
            "direct-score context rows cannot be smaller than the group count")
    if int(context_rows) == rows:
        return score_fn(
            inventory,
            {"tokens": tokens, "global_features": global_features,
             "group_seed": group_seed},
            device=device)
    padded_tokens = np.zeros((int(context_rows),) + tokens.shape[1:], dtype=np.float64)
    padded_tokens[:rows] = tokens
    padded_global = np.zeros((int(context_rows), global_features.shape[1]),
                             dtype=np.float64)
    padded_global[:rows] = global_features
    padded_seed = np.zeros(int(context_rows), dtype=np.int64)
    padded_seed[:rows] = group_seed
    full = score_fn(
        inventory,
        {"tokens": padded_tokens, "global_features": padded_global,
         "group_seed": padded_seed},
        device=device)
    from ac_twin import stage4cr2_training as s4cr2t
    return s4cr2t.RiskScores(
        group_seed=group_seed,
        risk_ensemble=np.asarray(full.risk_ensemble, dtype=np.float64)[:rows],
        risk_by_seed=np.asarray(full.risk_by_seed, dtype=np.float64)[:, :rows],
        mu_only_ensemble=np.asarray(full.mu_only_ensemble, dtype=np.float64)[:rows],
        class_only_ensemble=np.asarray(full.class_only_ensemble, dtype=np.float64)[:rows],
        seeds=tuple(int(s) for s in full.seeds),
    )


def _recheck_chain_once(
    checked: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    context_rows: int,
    direct_context_rows: int,
    hooks: Mapping[str, Any],
    device: str,
    batch_size: int,
    binding: Mapping[str, Any],
    stage3r_dir: Path,
    inventory: Any,
    seed_role: str = "confirm",
) -> dict[str, Any]:
    """Run the frozen predictor/risk/Direct chain once over a leading-group sub-bundle.

    With the real chain, the THREE row-count-context-dependent forwards are recomputed
    in the role's ORIGINAL context, exactly as the Task-1 adapter check prescribes: the
    unbatched e2e router mixture via ``cinf._shape_aligned_router_e2e`` and the
    single-CPU-forward COMBINED MLP via ``cinf._context_aligned_combined_score``
    (``context_rows`` = the formal branch count), and the single-forward Direct head
    via ``_context_aligned_direct_scores`` (``direct_context_rows`` = the formal group
    count) — GEMM/MLP row independence; no tolerance is ever substituted for bitwise
    identity. ``seed_role`` selects the role's frozen action-level CONSTANT_RANDOM
    stream ("confirm" formal, "smoke" engineering).
    """
    groups = int(bundle["groups"])
    predictions = hooks["run_inference_fn"](
        bundle, binding=binding, stage3r_dir=stage3r_dir,
        device=device, batch_size=batch_size)
    cinf._validate_predictions(predictions, groups=groups, n_actions=prot.N_ACTIONS)
    predictions = dict(predictions)
    if hooks["real_chain"]:
        branches = int(bundle["branches"])
        e2e_aligned = np.empty_like(np.asarray(predictions["e2e"], dtype=np.float32))
        for pair_index in range(len(prot.EXPERT_ROUTER_SEEDS)):
            full_pair = np.asarray(
                predictions["full"][pair_index], dtype=np.float32).reshape(
                branches, prot.HORIZON, prot.OBJECT_CAPACITY, 2)
            history_pair = np.asarray(
                predictions["history"][pair_index], dtype=np.float32).reshape(
                branches, prot.HORIZON, prot.OBJECT_CAPACITY, 2)
            e2e_pair = cinf._shape_aligned_router_e2e(
                bundle, full_pair, history_pair, pair_index=pair_index,
                binding=binding, stage3r_dir=stage3r_dir, device=device,
                batch_size=batch_size, context_rows=context_rows)
            e2e_aligned[pair_index] = e2e_pair.reshape(
                groups, prot.N_ACTIONS, prot.HORIZON, prot.OBJECT_CAPACITY, 2)
        predictions["e2e"] = e2e_aligned
    features = hooks["build_features_fn"](bundle, predictions)
    frozen_models = hooks["load_models_fn"]()
    scored = dict(hooks["score_actions_fn"](
        features, frozen_models, groups=groups, n_actions=prot.N_ACTIONS,
        rng_seed=cinf._constant_random_seed(checked, seed_role)))
    if hooks["real_chain"]:
        combined_model = frozen_models["COMBINED_RISK_HEAD"]
        if str(combined_model.implementation) == "mlp":
            scored["COMBINED_RISK_HEAD"] = cinf._context_aligned_combined_score(
                features, combined_model, context_rows=context_rows,
                groups=groups, n_actions=prot.N_ACTIONS)
    action_risks = {name: np.asarray(scored[name], dtype=np.float64)
                    for name in s4cpp.RISK_CLASSES}
    tokens = hooks["build_tokens_fn"](
        np.asarray(predictions["e2e"], dtype=np.float64), bundle["factual_xy"],
        bundle["history_last_xy"], bundle["inference_mask"], action_risks,
        bundle["geometry_group"], bundle["t_star"], bundle["group_seed"])
    scores = _context_aligned_direct_scores(
        hooks["score_fn"], inventory,
        np.asarray(tokens["tokens"], dtype=np.float64),
        np.asarray(tokens["global_features"], dtype=np.float64),
        np.asarray(bundle["group_seed"], dtype=np.int64),
        context_rows=direct_context_rows, device=device)
    return {
        "predictions": predictions,
        "action_risks": action_risks,
        "tokens": tokens,
        "scores": scores,
    }


def run_determinism_recheck(
    config: Mapping[str, Any],
    *,
    run_inference_fn: Callable | None = None,
    build_features_fn: Callable | None = None,
    load_models_fn: Callable | None = None,
    score_actions_fn: Callable | None = None,
    build_tokens_fn: Callable | None = None,
    score_fn: Callable | None = None,
    inventory: Any = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Repeat inference+scoring on the fixed leading groups; require bitwise identity.

    Plan section 6.4 step 2. Two independent runs over the first
    ``min(determinism_checks, G)`` accepted groups must be elementwise identical, AND
    the first run must equal the corresponding LEADING SLICES of the formal
    ``predictions.npz`` / ``risk_scores.npz`` (all five pair predictor outputs, all five
    risk classes, all five Direct seed scores, all baselines, strata inputs) — never a
    two-small-batches-only comparison. Writes the create-once determinism receipt; a
    rerun re-verifies the bindings and never overwrites.
    """
    checked = prot.validate_config(config)
    paths = confirm_artifact_paths(checked)
    inference = _verify_inference_chain(paths)
    receipt_path = paths["determinism_receipt"]
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        _verify_determinism(paths, inference)
        return {**receipt, "verification": "existing_determinism_verified_no_overwrite"}

    with np.load(paths["inputs"], allow_pickle=False) as archive:
        legal = {key: np.asarray(archive[key]) for key in cinf.LEGAL_INPUT_KEYS}
    groups_total = int(legal["group_seed"].shape[0])
    n = min(int(checked["determinism_checks"]), groups_total)
    sliced = {key: value[:n] for key, value in legal.items()}
    bundle = cinf._bundle_from_arrays(
        sliced, config=checked, data_signature=sha256_file(paths["inputs"]),
        role="determinism_recheck")
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        formal_predictions = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        formal_scores = {key: np.asarray(archive[key]) for key in archive.files}
    context_rows = int(formal_predictions["e2e"].shape[1]) * prot.N_ACTIONS
    direct_context_rows = int(np.asarray(formal_scores["risk_by_seed"]).shape[1])
    if direct_context_rows != int(formal_predictions["e2e"].shape[1]):
        raise ValueError(
            "the formal predictions/risk_scores group counts diverge; the formal "
            "artifacts are inconsistent")

    import torch
    device = str(device or checked.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or checked.get("batch_size", 32))
    hooks = _resolve_hooks(run_inference_fn, build_features_fn, load_models_fn,
                           score_actions_fn, build_tokens_fn, score_fn)
    binding = _read_json(_resolve(checked["stage3b_binding"]))["receipt_payload"]
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    if inventory is None and hooks["real_chain"]:
        inventory = cinf.load_frozen_model_inventory(checked)

    first = _recheck_chain_once(
        checked, bundle, context_rows=context_rows,
        direct_context_rows=direct_context_rows, hooks=hooks, device=device,
        batch_size=batch_size, binding=binding, stage3r_dir=stage3r_dir,
        inventory=inventory)
    second = _recheck_chain_once(
        checked, bundle, context_rows=context_rows,
        direct_context_rows=direct_context_rows, hooks=hooks, device=device,
        batch_size=batch_size, binding=binding, stage3r_dir=stage3r_dir,
        inventory=inventory)

    # --- two independent runs: elementwise identical -------------------------------------
    for key in cinf.PREDICTION_KEYS:
        if not np.array_equal(np.asarray(first["predictions"][key]),
                              np.asarray(second["predictions"][key])):
            raise RuntimeError(
                f"repeated confirm inference is not elementwise reproducible ({key}); "
                "the one-shot confirmation halts before the lock is burned")
    for name in s4cpp.RISK_CLASSES:
        if not np.array_equal(first["action_risks"][name], second["action_risks"][name]):
            raise RuntimeError(
                f"repeated confirm action scoring is not elementwise reproducible "
                f"({name})")
    for key in ("risk_ensemble", "risk_by_seed", "mu_only_ensemble",
                "class_only_ensemble"):
        if not np.array_equal(np.asarray(getattr(first["scores"], key)),
                              np.asarray(getattr(second["scores"], key))):
            raise RuntimeError(
                f"repeated confirm Direct scoring is not elementwise reproducible "
                f"({key})")

    # --- run one vs the formal artifacts' leading slices ----------------------------------
    for key in cinf.PREDICTION_KEYS:
        formal_slice = np.asarray(formal_predictions[key])[:, :n]
        if not np.array_equal(np.asarray(first["predictions"][key]), formal_slice):
            raise RuntimeError(
                f"the recheck prediction {key!r} does not equal the formal "
                "predictions' leading slice (formal-output cross-check failed)")
    score_slices = {
        "risk_by_seed": np.asarray(first["scores"].risk_by_seed, dtype=np.float64),
        "risk_ensemble": np.asarray(first["scores"].risk_ensemble, dtype=np.float64),
        "mu_only_ensemble": np.asarray(first["scores"].mu_only_ensemble,
                                       dtype=np.float64),
        "class_only_ensemble": np.asarray(first["scores"].class_only_ensemble,
                                          dtype=np.float64),
    }
    for key, mine in score_slices.items():
        formal_slice = np.asarray(formal_scores[key])
        formal_slice = formal_slice[:, :n] if formal_slice.ndim == 2 else formal_slice[:n]
        if not np.array_equal(mine, formal_slice):
            raise RuntimeError(
                f"the recheck Direct score {key!r} does not equal the formal "
                "risk scores' leading slice")
    tokens = first["tokens"]
    seeds_slice = np.asarray(bundle["group_seed"], dtype=np.int64)
    formal_seed = np.asarray(formal_scores["group_seed"], dtype=np.int64)
    if not np.array_equal(seeds_slice, formal_seed[:n]):
        raise RuntimeError("the recheck group order diverges from the formal scores")
    for name, risk in tokens["baselines"].items():
        if not np.array_equal(np.asarray(risk, dtype=np.float64),
                              np.asarray(formal_scores[f"baseline_{name}"])[:n]):
            raise RuntimeError(
                f"the recheck baseline {name!r} does not equal the formal slice")
    if not np.array_equal(cinf._constant_random_group_scores(seeds_slice),
                          np.asarray(formal_scores["constant_random"])[:n]):
        raise RuntimeError("the recheck CONSTANT_RANDOM slice drifted from the formal")
    for key, mine in (("t_star", np.asarray(bundle["t_star"], dtype=np.int64)),
                      ("interaction_min_distance",
                       np.asarray(tokens["interaction_min_distance"], dtype=np.float64)),
                      ("reference_action",
                       np.asarray(tokens["reference_action"], dtype=np.int64)),
                      ("valid_mask", np.asarray(tokens["valid_mask"], dtype=bool))):
        if not np.array_equal(mine, np.asarray(formal_scores[key])[:n]):
            raise RuntimeError(
                f"the recheck {key!r} slice drifted from the formal risk scores")

    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "confirm",
        "bitwise_identical": True,
        "n_groups_rechecked": int(n),
        "group_indices": list(range(n)),
        "context_rows": int(context_rows),
        "direct_context_rows": int(direct_context_rows),
        "context_note": (
            "the frozen e2e router forward, the COMBINED MLP forward and the Direct "
            "five-seed scoring forward are row-count-context dependent; the recheck "
            "reproduces the role's original forward contexts (context_rows = formal "
            "branch count; direct_context_rows = formal group count) via zero-padding "
            "(GEMM/MLP row independence), so the comparison is a true bitwise one"),
        "blocks_compared": [
            "predictions(6 keys, 5 pairs)", "action_risks(5 classes)",
            "direct_scores(ensemble, 5 seeds, MU_ONLY, CLASS_ONLY)",
            "baselines(4)", "constant_random", "t_star", "interaction_min_distance",
            "reference_action", "valid_mask", "group_seed",
        ],
        "bindings": {
            "predictions_sha256": inference["predictions_sha256"],
            "risk_scores_sha256": inference["risk_scores_sha256"],
            "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
            "inference_inputs_sha256": inference["data_sha256"],
            "collection_receipt_sha256": inference["collection_receipt_sha256"],
        },
        "device": device,
        "dtype": "float32",
        "batch_size": batch_size,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return receipt


# --- Task 2 supplement (pre-lock): smoke engineering recheck + engineering-truth statistics ------
#
# ENGINEERING_SMOKE_ONLY end to end. These two functions close the smoke gap BEFORE the
# formal protocol lock exists: they repeat the 12-group smoke inference/scoring bitwise and
# exercise the REAL quantity-assembly / gate code path on the smoke truth as an engineering
# diagnostic. They NEVER touch a formal artifact path, NEVER burn or create any lock, NEVER
# call the judge-only ``load_confirmation_truth``, and NEVER produce a scientific verdict.


def smoke_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every smoke-scoped path (the ``smoke/`` engineering namespace ONLY)."""
    checked = prot.validate_config(config)
    predictions, risk_scores, inference_receipt = cinf.role_inference_paths(checked, "smoke")
    smoke_run = Path(inference_receipt).parent          # <run_dir>/smoke
    return {
        "inputs": cinf.role_input_path(checked, "smoke"),
        "truth": cinf.role_truth_path(checked, "smoke"),
        "predictions": predictions,
        "risk_scores": risk_scores,
        "inference_receipt": inference_receipt,
        "collection_receipt": smoke_run / "smoke_collection_receipt.json",
        "recheck_receipt": smoke_run / "smoke_recheck_receipt.json",
        "engineering_report": smoke_run / "smoke_engineering_report.json",
    }


def _assert_smoke_namespace(paths: Mapping[str, Path], checked: Mapping[str, Any]) -> None:
    """Fail closed if any smoke path collides with a formal artifact path or leaves the
    ``smoke/`` namespace (the smoke path must never touch formal artifact paths)."""
    formal_resolved = {Path(p).resolve() for p in confirm_artifact_paths(checked).values()}
    for key, path in paths.items():
        resolved = Path(path).resolve()
        if resolved in formal_resolved:
            raise ValueError(
                f"smoke path {key!r} collides with a formal artifact path: {path}")
        if Path(path).parent.name != "smoke":
            raise ValueError(
                f"smoke artifact {key!r} must live under a smoke/ namespace: {path}")


def _verify_smoke_inference_chain(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Smoke-scoped mirror of ``_verify_inference_chain`` (hash bindings, fail-closed).

    Binds the SMOKE collection receipt: it must exist, carry the ENGINEERING_SMOKE_ONLY
    status with the frozen 12 accepted groups, and hash-bind the published smoke legal
    inputs. The smoke truth is never opened here.
    """
    if not paths["inference_receipt"].is_file():
        raise ValueError(
            f"the smoke inference receipt is missing: {paths['inference_receipt']}")
    inference = _read_json(paths["inference_receipt"])
    if inference.get("status") != cinf.STATUS_SMOKE or inference.get("role") != "smoke":
        raise ValueError(
            "the smoke inference receipt must carry role 'smoke' and status "
            f"{cinf.STATUS_SMOKE!r}; got role={inference.get('role')!r} "
            f"status={inference.get('status')!r}")
    for key, path in (("predictions_sha256", paths["predictions"]),
                      ("risk_scores_sha256", paths["risk_scores"])):
        if not path.is_file() or sha256_file(path) != inference.get(key):
            raise ValueError(f"smoke inference artifact drifted from its receipt: {path}")
    if not paths["inputs"].is_file() \
            or sha256_file(paths["inputs"]) != inference.get("data_sha256"):
        raise ValueError("smoke legal inputs drifted from the smoke inference receipt")
    if not paths["collection_receipt"].is_file():
        raise ValueError(
            f"the smoke collection receipt is missing: {paths['collection_receipt']}")
    collection = _read_json(paths["collection_receipt"])
    if collection.get("status") != cinf.STATUS_SMOKE or collection.get("role") != "smoke":
        raise ValueError(
            f"the smoke collection receipt status must be {cinf.STATUS_SMOKE!r}; got "
            f"{collection.get('status')!r}")
    if collection.get("partial") or int(collection.get("accepted_groups", -1)) \
            != prot.SMOKE_ACCEPTED_GROUPS:
        raise ValueError(
            f"the smoke collection must be complete at {prot.SMOKE_ACCEPTED_GROUPS} "
            f"accepted groups; got {collection.get('accepted_groups')}")
    if collection.get("inference_inputs_sha256") != inference.get("data_sha256"):
        raise ValueError("smoke collection receipt does not bind the smoke legal inputs")
    inference["collection_receipt_sha256"] = sha256_file(paths["collection_receipt"])
    inference["inference_receipt_sha256"] = sha256_file(paths["inference_receipt"])
    return inference


def _smoke_recheck_bindings(inference: Mapping[str, Any]) -> dict[str, str]:
    return {
        "predictions_sha256": inference["predictions_sha256"],
        "risk_scores_sha256": inference["risk_scores_sha256"],
        "inference_receipt_sha256": inference["inference_receipt_sha256"],
        "inference_inputs_sha256": inference["data_sha256"],
        "collection_receipt_sha256": inference["collection_receipt_sha256"],
    }


def _verify_smoke_recheck_receipt(paths: Mapping[str, Path],
                                  inference: Mapping[str, Any]) -> dict[str, Any]:
    """The smoke recheck receipt must exist, record bitwise identity and bind the
    current smoke artifacts (never overwritten, never re-run silently)."""
    receipt_path = paths["recheck_receipt"]
    if not receipt_path.is_file():
        raise ValueError(
            "the smoke recheck receipt is missing; run the smoke repeated "
            "inference/scoring recheck first (run_smoke_recheck)")
    receipt = _read_json(receipt_path)
    if not receipt.get("bitwise_identical"):
        raise ValueError("the smoke repeated inference/scoring was not bitwise identical")
    if receipt.get("bindings") != _smoke_recheck_bindings(inference):
        raise ValueError(
            "the smoke recheck receipt is not bound to the current smoke artifacts "
            "(tampering or drift)")
    return receipt


def run_smoke_recheck(
    config: Mapping[str, Any],
    *,
    run_inference_fn: Callable | None = None,
    build_features_fn: Callable | None = None,
    load_models_fn: Callable | None = None,
    score_actions_fn: Callable | None = None,
    build_tokens_fn: Callable | None = None,
    score_fn: Callable | None = None,
    inventory: Any = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """ENGINEERING_SMOKE_ONLY: repeat smoke inference+scoring twice; bitwise identity.

    Mirrors ``run_determinism_recheck``'s mechanics but smoke-scoped: two independent
    runs over ALL 12 smoke groups under the same device/dtype/batch settings must be
    elementwise identical, AND the first run must equal the FULL persisted
    ``smoke_predictions.npz`` / ``smoke_risk_scores.npz`` arrays (all five pair outputs,
    all five risk classes, all five Direct seeds, all baselines — never a slice
    comparison, never a tolerance). The two row-count-context-dependent forwards are
    recomputed in the smoke role's ORIGINAL forward context (context_rows = the smoke
    branch count, 120) via the context-aligned helpers. Writes the create-once smoke
    recheck receipt; a rerun re-verifies the bindings and never overwrites. Never
    touches a formal artifact path, never burns any lock.
    """
    checked = prot.validate_config(config)
    paths = smoke_artifact_paths(checked)
    _assert_smoke_namespace(paths, checked)
    inference = _verify_smoke_inference_chain(paths)
    receipt_path = paths["recheck_receipt"]
    bindings = _smoke_recheck_bindings(inference)
    if receipt_path.exists():
        receipt = _verify_smoke_recheck_receipt(paths, inference)
        return {**receipt, "verification": "existing_smoke_recheck_verified_no_overwrite"}

    with np.load(paths["inputs"], allow_pickle=False) as archive:
        legal = {key: np.asarray(archive[key]) for key in cinf.LEGAL_INPUT_KEYS}
    groups = int(legal["group_seed"].shape[0])
    if groups != prot.SMOKE_ACCEPTED_GROUPS:
        raise ValueError(
            f"the smoke recheck covers ALL {prot.SMOKE_ACCEPTED_GROUPS} smoke groups; "
            f"the inputs carry {groups}")
    bundle = cinf._bundle_from_arrays(
        legal, config=checked, data_signature=sha256_file(paths["inputs"]),
        role="smoke_recheck")
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        persisted_predictions = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        persisted_scores = {key: np.asarray(archive[key]) for key in archive.files}
    context_rows = int(persisted_predictions["e2e"].shape[1]) * prot.N_ACTIONS
    if context_rows != int(bundle["branches"]):
        raise ValueError(
            "the persisted smoke predictions' branch context diverges from the smoke "
            "inputs; the smoke artifacts are inconsistent")
    direct_context_rows = int(np.asarray(persisted_scores["risk_by_seed"]).shape[1])
    if direct_context_rows != groups:
        raise ValueError(
            "the persisted smoke risk scores' group count diverges from the smoke "
            "inputs; the smoke artifacts are inconsistent")
    if not np.array_equal(np.asarray(persisted_predictions["group_seed"], dtype=np.int64),
                          np.asarray(bundle["group_seed"], dtype=np.int64)):
        raise ValueError("smoke predictions group order diverges from the smoke inputs")

    import torch
    device = str(device or checked.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    batch_size = int(batch_size or checked.get("batch_size", 32))
    hooks = _resolve_hooks(run_inference_fn, build_features_fn, load_models_fn,
                           score_actions_fn, build_tokens_fn, score_fn)
    binding = _read_json(_resolve(checked["stage3b_binding"]))["receipt_payload"]
    stage3r_dir = _resolve(checked["stage3r_run_dir"])
    if inventory is None and hooks["real_chain"]:
        inventory = cinf.load_frozen_model_inventory(checked)

    first = _recheck_chain_once(
        checked, bundle, context_rows=context_rows,
        direct_context_rows=direct_context_rows, hooks=hooks, device=device,
        batch_size=batch_size, binding=binding, stage3r_dir=stage3r_dir,
        inventory=inventory, seed_role="smoke")
    second = _recheck_chain_once(
        checked, bundle, context_rows=context_rows,
        direct_context_rows=direct_context_rows, hooks=hooks, device=device,
        batch_size=batch_size, binding=binding, stage3r_dir=stage3r_dir,
        inventory=inventory, seed_role="smoke")

    # --- two independent runs: elementwise identical -------------------------------------
    for key in cinf.PREDICTION_KEYS:
        if not np.array_equal(np.asarray(first["predictions"][key]),
                              np.asarray(second["predictions"][key])):
            raise RuntimeError(
                f"repeated smoke inference is not elementwise reproducible ({key})")
    for name in s4cpp.RISK_CLASSES:
        if not np.array_equal(first["action_risks"][name], second["action_risks"][name]):
            raise RuntimeError(
                f"repeated smoke action scoring is not elementwise reproducible ({name})")
    for key in ("risk_ensemble", "risk_by_seed", "mu_only_ensemble",
                "class_only_ensemble"):
        if not np.array_equal(np.asarray(getattr(first["scores"], key)),
                              np.asarray(getattr(second["scores"], key))):
            raise RuntimeError(
                f"repeated smoke Direct scoring is not elementwise reproducible ({key})")

    # --- run one vs the FULL persisted smoke artifacts (never a slice) --------------------
    for key in cinf.PREDICTION_KEYS:
        if not np.array_equal(np.asarray(first["predictions"][key]),
                              np.asarray(persisted_predictions[key])):
            raise RuntimeError(
                f"the smoke recheck prediction {key!r} does not equal the persisted "
                "smoke_predictions.npz (full-array cross-check failed)")
    score_blocks = {
        "risk_by_seed": np.asarray(first["scores"].risk_by_seed, dtype=np.float64),
        "risk_ensemble": np.asarray(first["scores"].risk_ensemble, dtype=np.float64),
        "mu_only_ensemble": np.asarray(first["scores"].mu_only_ensemble,
                                       dtype=np.float64),
        "class_only_ensemble": np.asarray(first["scores"].class_only_ensemble,
                                          dtype=np.float64),
    }
    for key, mine in score_blocks.items():
        if not np.array_equal(mine, np.asarray(persisted_scores[key])):
            raise RuntimeError(
                f"the smoke recheck Direct score {key!r} does not equal the persisted "
                "smoke_risk_scores.npz")
    tokens = first["tokens"]
    seeds = np.asarray(bundle["group_seed"], dtype=np.int64)
    if not np.array_equal(seeds, np.asarray(persisted_scores["group_seed"],
                                            dtype=np.int64)):
        raise RuntimeError("the smoke recheck group order diverges from the smoke scores")
    for name, risk in tokens["baselines"].items():
        if not np.array_equal(np.asarray(risk, dtype=np.float64),
                              np.asarray(persisted_scores[f"baseline_{name}"])):
            raise RuntimeError(
                f"the smoke recheck baseline {name!r} does not equal the persisted "
                "smoke_risk_scores.npz")
    if not np.array_equal(cinf._constant_random_group_scores(seeds),
                          np.asarray(persisted_scores["constant_random"])):
        raise RuntimeError("the smoke recheck CONSTANT_RANDOM drifted from the persisted")
    for key, mine in (("t_star", np.asarray(bundle["t_star"], dtype=np.int64)),
                      ("interaction_min_distance",
                       np.asarray(tokens["interaction_min_distance"], dtype=np.float64)),
                      ("reference_action",
                       np.asarray(tokens["reference_action"], dtype=np.int64)),
                      ("valid_mask", np.asarray(tokens["valid_mask"], dtype=bool))):
        if not np.array_equal(mine, np.asarray(persisted_scores[key])):
            raise RuntimeError(
                f"the smoke recheck {key!r} drifted from the persisted smoke risk scores")

    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "smoke",
        "status": cinf.STATUS_SMOKE,
        "bitwise_identical": True,
        "n_groups_rechecked": int(groups),
        "group_indices": list(range(groups)),
        "context_rows": int(context_rows),
        "direct_context_rows": int(direct_context_rows),
        "context_note": (
            "the frozen e2e router forward, the COMBINED MLP forward and the Direct "
            "five-seed scoring forward are row-count-context dependent; the smoke "
            "recheck reproduces the smoke role's original forward contexts "
            "(context_rows = smoke branch count, 120; direct_context_rows = smoke "
            "group count, 12) via zero-padding (GEMM/MLP row independence), so the "
            "comparison is a true bitwise one over ALL 12 smoke groups and the FULL "
            "persisted arrays"),
        "blocks_compared": [
            "predictions(6 keys, 5 pairs)", "action_risks(5 classes)",
            "direct_scores(ensemble, 5 seeds, MU_ONLY, CLASS_ONLY)",
            "baselines(4)", "constant_random", "t_star", "interaction_min_distance",
            "reference_action", "valid_mask", "group_seed",
        ],
        "bindings": bindings,
        "receipt_path": str(receipt_path),
        "device": device,
        "dtype": "float32",
        "batch_size": batch_size,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return receipt


def _load_smoke_truth_arrays(checked: Mapping[str, Any], paths: Mapping[str, Path]
                             ) -> dict[str, Any]:
    """Smoke-scoped truth load — ENGINEERING ONLY, never ``load_confirmation_truth``.

    The smoke truth binds ONLY to the ENGINEERING_SMOKE_ONLY smoke collection receipt
    (role "smoke"); the judge-only formal loader (which requires the burned one-shot
    lock) is never invoked on the smoke path. Hash/schema/integrity validation is
    REUSED from ``_load_truth_arrays``.
    """
    collection = _read_json(paths["collection_receipt"])
    if collection.get("role") != "smoke" or collection.get("status") != cinf.STATUS_SMOKE:
        raise ValueError(
            "the smoke truth binds only to the ENGINEERING_SMOKE_ONLY smoke collection "
            "receipt; refusing any formal-role binding")
    return _load_truth_arrays(checked, paths)


def run_smoke_engineering_statistics(
    config: Mapping[str, Any],
    *,
    n_resamples: int | None = None,
) -> dict[str, Any]:
    """ENGINEERING_SMOKE_ONLY: exercise the REAL quantity/gate path on the smoke truth.

    Loads the smoke truth (smoke role ONLY, via the smoke-scoped loader — never the
    judge-only ``load_confirmation_truth``), the smoke legal inputs and the smoke
    scores; assembles the per-group quantities with the SAME computation as
    ``assemble_confirm_quantities`` (in memory — the formal create-once quantities pair
    is never written); runs the ONE paired bootstrap (frozen entropy 2026093202, full
    10,000 resamples by default — a reduced count is permitted for speed ONLY and is
    recorded prominently); and evaluates the REAL gates. With 12 groups the formal gate
    thresholds cannot pass (strata need >=200 each, c1 needs the frozen 2,000), so the
    gate output is recorded as an engineering diagnostic and never raises on failing
    thresholds; non-finite/invalid computation still raises (that is the engineering
    bug class smoke exists to catch). Requires the smoke recheck receipt (ordering:
    smoke recheck BEFORE the engineering statistics). Writes the create-once
    ``smoke/smoke_engineering_report.json``; a rerun re-verifies and never overwrites.
    Never burns any lock, never writes a formal artifact, never produces a scientific
    verdict and never claims pass/fail of the method.
    """
    checked = prot.validate_config(config)
    paths = smoke_artifact_paths(checked)
    _assert_smoke_namespace(paths, checked)
    inference = _verify_smoke_inference_chain(paths)
    recheck = _verify_smoke_recheck_receipt(paths, inference)
    report_path = paths["engineering_report"]
    truth = _load_smoke_truth_arrays(checked, paths)
    bindings = {
        "inference_inputs_sha256": inference["data_sha256"],
        "predictions_sha256": inference["predictions_sha256"],
        "risk_scores_sha256": inference["risk_scores_sha256"],
        "smoke_truth_sha256": truth["sha256"],
        "collection_receipt_sha256": inference["collection_receipt_sha256"],
        "inference_receipt_sha256": inference["inference_receipt_sha256"],
        "recheck_receipt_sha256": sha256_file(paths["recheck_receipt"]),
    }
    if report_path.exists():
        report = _read_json(report_path)
        if report.get("status") != cinf.STATUS_SMOKE \
                or report.get("engineering_only") is not True:
            raise ValueError(
                "the smoke engineering report lost its ENGINEERING_SMOKE_ONLY label")
        if report.get("bindings") != bindings:
            raise ValueError(
                "the smoke engineering report is not bound to the current smoke "
                "artifacts (tampering or drift)")
        return {**report,
                "verification": "existing_smoke_engineering_report_verified_no_overwrite"}

    truth_arrays = truth["arrays"]
    groups = int(truth["n_groups"])
    if groups != prot.SMOKE_ACCEPTED_GROUPS:
        raise ValueError(
            f"the smoke truth must carry the {prot.SMOKE_ACCEPTED_GROUPS} smoke groups; "
            f"got {groups}")

    legal = cinf._read_legal_npz(paths["inputs"],
                                 {key: key for key in cinf.LEGAL_INPUT_KEYS})
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        predictions = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        scores = {key: np.asarray(archive[key]) for key in archive.files}

    # --- same-order group binding across all four smoke files -----------------------------
    inputs_seed = np.asarray(legal["group_seed"], dtype=np.int64).ravel()
    truth_seed = np.asarray(truth_arrays["group_seed"], dtype=np.int64).ravel()
    predictions_seed = np.asarray(predictions["group_seed"], dtype=np.int64).ravel()
    scores_seed = np.asarray(scores["group_seed"], dtype=np.int64).ravel()
    if not (np.array_equal(inputs_seed, truth_seed)
            and np.array_equal(inputs_seed, predictions_seed)
            and np.array_equal(inputs_seed, scores_seed)):
        raise ValueError(
            "smoke group_seed order diverges across inputs/predictions/risk_scores/"
            "truth; the engineering exercise fails closed, never realigns")
    # --- integrity copies inside the smoke truth must equal the smoke legal inputs --------
    for key in ("cf_actions", "cf_durations", "factual_actions", "t_star"):
        if not np.array_equal(np.asarray(truth_arrays[key]), np.asarray(legal[key])):
            raise ValueError(f"smoke truth integrity copy {key!r} diverges")
    if not np.array_equal(np.asarray(truth_arrays["inference_mask"], dtype=bool),
                          np.asarray(legal["inference_mask"], dtype=bool)):
        raise ValueError("smoke truth integrity copy 'inference_mask' diverges")

    # --- J_hat from the smoke predictions; cross-check the persisted reference action -----
    history = np.asarray(legal["object_history"], dtype=np.float64)
    factual_xy = np.asarray(legal["object_factual_future"], dtype=np.float64)[
        ..., POSITION_SLICE]
    history_last_xy = history[:, -1, s4co.EGO_SLOT, POSITION_SLICE]
    mask = np.asarray(legal["inference_mask"], dtype=bool)
    e2e = np.asarray(predictions["e2e"], dtype=np.float64)
    if e2e.shape != (N_SEEDS, groups, prot.N_ACTIONS, prot.HORIZON,
                     prot.OBJECT_CAPACITY, 2):
        raise ValueError(f"smoke predictions e2e must be (5,G,10,20,5,2); got {e2e.shape}")
    _, j_hat = s4cpf.compute_predicted_costs(e2e, factual_xy, history_last_xy, mask)
    a_hat = np.argmin(j_hat, axis=-1).astype(np.int64)
    reference_action = np.asarray(scores["reference_action"], dtype=np.int64).ravel()
    if not np.array_equal(a_hat, reference_action):
        raise ValueError(
            "recomputed J_hat argmin diverges from the persisted smoke reference_action")

    # --- J_true from the smoke truth with the frozen consequence function -----------------
    true_xy = np.asarray(truth_arrays["object_counterfactual_futures"],
                         dtype=np.float64)[..., POSITION_SLICE]
    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, prot.N_ACTIONS, 2))
    mask_b = np.broadcast_to(mask[:, None, :, :],
                             (groups, prot.N_ACTIONS) + mask.shape[1:])
    comp_true = s4co.trajectory_components(true_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))
    decision_arrays = _decision_quantities(j_hat, j_true)

    t_star = np.asarray(legal["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(scores["interaction_min_distance"], dtype=np.float64).ravel()
    tstar_layer = (t_star >= TSTAR_SPLIT).astype(np.int64)
    interaction_layer = (distance > float(prot.INTERACTION_THRESHOLD)).astype(np.int64)
    trim1_keep = s4cs.trim_high_regret(decision_arrays["regret"], TRIM_FRACTIONS[0])
    trim5_keep = s4cs.trim_high_regret(decision_arrays["regret"], TRIM_FRACTIONS[1])

    # --- honest QA flags for the smoke role (no lock exists by design) --------------------
    schedule = prot.build_smoke_seed_schedule(checked)
    domain = set(int(v) for v in schedule)
    role_seeds_valid = bool(
        np.unique(inputs_seed).size == groups
        and all(int(s) in domain for s in inputs_seed))
    finite_truth = bool(np.isfinite(j_true).all()
                        and np.isfinite(decision_arrays["regret"]).all())
    by_seed = np.asarray(scores["risk_by_seed"], dtype=np.float64)
    ensemble = np.asarray(scores["risk_ensemble"], dtype=np.float64)
    qa = {
        # 12 smoke groups are NEVER the frozen 2,000-group confirm task.
        "all_complete": bool(
            groups == prot.CONFIRM_ACCEPTED_GROUPS
            and j_true.shape == (groups, prot.N_ACTIONS)
            and j_hat.shape == (groups, prot.N_ACTIONS)
            and finite_truth),
        "hashes_match": True,   # the full smoke chain re-verified above
        "role_seeds_valid": role_seeds_valid,
        "determinism_passed": bool(recheck.get("bitwise_identical")),
        # smoke NEVER burns the one-shot lock — recorded honestly as False.
        "one_shot_locks": False,
        "ensemble_is_arithmetic_mean": bool(
            by_seed.shape == (N_SEEDS, groups)
            and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)),
    }

    baseline_risks: dict[str, np.ndarray] = {}
    for name in ALL_BASELINE_NAMES:
        key = "constant_random" if name == "CONSTANT_RANDOM" else f"baseline_{name}"
        risk = np.asarray(scores[key], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"smoke baseline risk {name!r} must be finite/aligned")
        baseline_risks[name] = risk

    quantities: dict[str, Any] = {
        "group_seed": inputs_seed,
        "regret": decision_arrays["regret"],
        "y_high": decision_arrays["y_high"],
        "random_regret": decision_arrays["random_regret"],
        "idle_regret": decision_arrays["idle_regret"],
        "a_hat": decision_arrays["a_hat"],
        "a_star": decision_arrays["a_star"],
        "J_hat": j_hat,
        "J_true": j_true,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": np.asarray(scores["mu_only_ensemble"], dtype=np.float64),
        "class_only_ensemble": np.asarray(scores["class_only_ensemble"],
                                          dtype=np.float64),
        "tstar_layer": tstar_layer,
        "interaction_layer": interaction_layer,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "trim1_keep": trim1_keep,
        "trim5_keep": trim5_keep,
        "baseline_risks": baseline_risks,
        "strongest_baseline": PRIMARY_BASELINE,
        "n_groups": int(groups),
        "interaction_threshold": float(prot.INTERACTION_THRESHOLD),
        "qa": qa,
    }
    _validate_quantities(quantities)  # fail closed: non-finite/misaligned raises

    resamples = prot.BOOTSTRAP_RESAMPLES if n_resamples is None else int(n_resamples)
    bootstrap = paired_group_bootstrap(
        quantities, seed=prot.BOOTSTRAP_ENTROPY, n_resamples=resamples)
    gates = evaluate_confirmation_gates(quantities, bootstrap)

    reduced = bool(resamples != prot.BOOTSTRAP_RESAMPLES)
    report = {
        "plan_id": prot.PLAN_ID,
        "role": "smoke",
        "status": cinf.STATUS_SMOKE,
        "engineering_only": True,
        "no_scientific_content": (
            "engineering exercise only, no scientific content: the gate outputs below "
            "are engineering diagnostics from the REAL quantity-assembly/gate code path "
            f"on {groups} smoke groups; the formal gate thresholds cannot pass at "
            f"{groups} groups (strata need >=200 each, condition c1 needs the frozen "
            "2,000); nothing here is a pass/fail verdict on the method and nothing "
            "here enters the formal confirmation"),
        "n_groups": int(groups),
        "bootstrap": {
            "seed": prot.BOOTSTRAP_ENTROPY,
            "n_resamples": int(resamples),
            "frozen_full_resamples": prot.BOOTSTRAP_RESAMPLES,
            "reduced_resamples": reduced,
            "note": ("REDUCED RESAMPLE COUNT — engineering speed only, never the formal "
                     "statistic" if reduced else
                     "full frozen 10,000 whole-group paired resamples"),
            "confidence": prot.CONFIDENCE,
        },
        "qa": qa,
        "gate_diagnostics": gates.to_dict(),
        "formal_side_effects": {
            "protocol_lock_created": False,
            "confirmation_used_lock_burned": False,
            "formal_artifacts_written": False,
            "formal_truth_loader_used": False,
        },
        "bindings": bindings,
        "report_path": str(report_path),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(report_path, report)
    return report


# --- plan 6.4 step 3: the score freeze ---------------------------------------------------------------


def freeze_scores(
    config: Mapping[str, Any],
    *,
    source_verify_fn: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Freeze every score-side artifact; write the create-once score freeze receipt.

    Refuses unless: the formal protocol lock binds this config/schedule/code set; the
    collection/inference receipts exist and hash-bind the published inputs, predictions
    and risk scores; the determinism recheck passed and is bound to the same artifacts;
    and ``source_verify_fn`` (default ``prot.verify_r2_sources`` over the configured R2
    development dir — the section-2.1 ten files plus the five relocated, hash-verified
    Direct checkpoints) passes. The sealed truth is NEVER opened here; only its
    receipt-recorded hash is carried into the receipt. A rerun re-verifies everything
    and never overwrites.
    """
    checked = prot.validate_config(config)
    paths = confirm_artifact_paths(checked)
    _assert_protocol_lock_binding(checked, paths)
    inference = _verify_inference_chain(paths)
    determinism = _verify_determinism(paths, inference)
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        groups = int(np.asarray(archive["group_seed"]).ravel().size)
    if inference["collection_accepted_groups"] != groups:
        raise ValueError("collection accepted group count does not match the scores")
    if not inference["collection_same_order"]:
        raise ValueError("the collection receipt does not attest same-order two-file "
                         "group binding")
    if source_verify_fn is None:
        source_verify_fn = lambda: prot.verify_r2_sources(  # noqa: E731
            _resolve(checked["r2_development_dir"]))
    source_verify_fn()

    receipt_path = paths["score_freeze_receipt"]
    bindings = {
        "inference_inputs_sha256": inference["data_sha256"],
        "predictions_sha256": inference["predictions_sha256"],
        "risk_scores_sha256": inference["risk_scores_sha256"],
        "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
        "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
        "collection_receipt_sha256": inference["collection_receipt_sha256"],
        "expected_confirmation_truth_sha256":
            inference["expected_confirmation_truth_sha256"],
        "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
    }
    if receipt_path.exists():
        receipt = _read_json(receipt_path)
        if receipt.get("plan_id") != prot.PLAN_ID \
                or receipt.get("bindings") != bindings:
            raise ValueError(
                "the score freeze receipt no longer matches the on-disk frozen "
                "artifacts (tampering or drift)")
        return {**receipt, "verification": "existing_score_freeze_verified_no_overwrite"}
    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "confirm",
        "n_groups": int(groups),
        "bindings": bindings,
        "model_seeds": [int(s) for s in prot.MODEL_SEEDS],
        "ensemble_definition": prot.ENSEMBLE_DEFINITION,
        "primary_baseline": PRIMARY_BASELINE,
        "standard_baselines": list(prot.STANDARD_BASELINES),
        "reference_baselines": list(prot.REFERENCE_BASELINES),
        "determinism_rechecked_groups": int(determinism["n_groups_rechecked"]),
        "bootstrap": {
            "entropy": prot.BOOTSTRAP_ENTROPY,
            "resamples": prot.BOOTSTRAP_RESAMPLES,
            "confidence": prot.CONFIDENCE,
        },
        "constant_random_salt": prot.CONSTANT_RANDOM_SALT,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(receipt_path, receipt)
    return receipt


# --- plan 6.4 step 5: burn the one-shot confirmation lock ----------------------------------------------


def burn_confirmation_lock(config: Mapping[str, Any]) -> str:
    """Create ``confirmation_used.lock`` exactly once, after the full score freeze.

    Refuses unless the score freeze receipt exists and every score-side binding
    re-verifies against the current files. The lock binds the sealed truth ONLY through
    the collection receipt's recorded hash — the burn NEVER reads the truth bytes. A
    second call re-verifies the bindings against the current files and never
    overwrites.
    """
    checked = prot.validate_config(config)
    paths = confirm_artifact_paths(checked)
    if not paths["score_freeze_receipt"].is_file():
        raise ValueError(
            "the score freeze receipt is missing; freeze_scores must run before the "
            "confirmation lock burns")
    freeze_receipt = freeze_scores(checked)  # re-verifies the full score-side chain
    bindings = dict(freeze_receipt["bindings"])
    bindings["score_freeze_receipt_sha256"] = sha256_file(paths["score_freeze_receipt"])
    bindings["statistics_code_sha256"] = sha256_file(
        ROOT / "code/ac_twin/stage4cr2_confirm_statistics.py")

    lock_path = paths["lock"]
    if lock_path.exists():
        payload = _read_json(lock_path)
        if payload.get("plan_id") != prot.PLAN_ID \
                or payload.get("bindings") != bindings:
            raise ValueError(
                "the burned confirmation lock no longer matches the on-disk frozen "
                "artifacts (tampering or drift); the one-shot confirmation is "
                "compromised")
        return str(lock_path)
    payload = {
        "plan_id": prot.PLAN_ID,
        "role": "confirm",
        "semantics": ("one-shot: the sealed confirmation truth may be read only after "
                      "this lock exists; all confirm scores are frozen and hash-bound "
                      "above"),
        "primary_baseline": PRIMARY_BASELINE,
        "interaction_threshold": float(prot.INTERACTION_THRESHOLD),
        "t_star_stratum_split": TSTAR_SPLIT,
        "coverage": COVERAGE,
        "bindings": bindings,
        "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path, payload)
    return str(lock_path)


def _verify_burned_lock(checked: Mapping[str, Any], paths: Mapping[str, Path]
                        ) -> dict[str, Any]:
    """The burned lock must exist and re-verify against the current frozen files."""
    if not paths["lock"].is_file():
        raise ValueError(
            "the sealed confirmation truth may not be read: confirmation_used.lock is "
            "not burned; freeze every score artifact and burn the lock first "
            "(plan section 6.4)")
    burn_confirmation_lock(checked)  # re-verifies the bindings, never overwrites
    return _read_json(paths["lock"])


# --- plan 6.4 step 6: the judge reads the sealed truth (only after the burn) ----------------------------


def _load_truth_arrays(checked: Mapping[str, Any], paths: Mapping[str, Path]
                       ) -> dict[str, Any]:
    """Open the sealed truth ONCE per transaction; verify hash, schema and integrity."""
    truth_path = paths["truth"]
    if not truth_path.is_file():
        raise FileNotFoundError(f"sealed confirmation truth missing: {truth_path}")
    collection = _read_json(paths["collection_receipt"])
    expected = str(collection.get("confirmation_truth_sha256", ""))
    digest = sha256_file(truth_path)
    if not expected or digest != expected:
        raise ValueError(
            "the sealed confirmation truth drifted from its collection-receipt hash")
    if list(collection.get("truth_fields", [])) != list(TRUTH_FIELDS):
        raise ValueError(
            "the collection receipt's truth_fields drifted from the frozen schema "
            "(collector/statistics contract drift)")
    with np.load(truth_path, allow_pickle=False) as archive:
        if set(archive.files) != set(TRUTH_FIELDS):
            raise ValueError(
                f"sealed truth archive key set drifted: {sorted(archive.files)}")
        arrays = {key: np.asarray(archive[key]) for key in TRUTH_FIELDS}
    futures = np.asarray(arrays["object_counterfactual_futures"])
    groups = futures.shape[0]
    if futures.shape != (groups, prot.N_ACTIONS, prot.HORIZON, prot.OBJECT_CAPACITY,
                         prot.N_OBJECT_FEATURES) or futures.dtype != np.float32:
        raise ValueError("object_counterfactual_futures must be (G,10,20,5,8) float32")
    if not np.isfinite(futures.astype(np.float64)).all():
        raise ValueError("non-finite sealed counterfactual truth")
    for key in ("cf_actions", "cf_durations"):
        value = np.asarray(arrays[key])
        if value.shape != (groups, prot.N_ACTIONS) or value.dtype != np.int64:
            raise ValueError(f"truth {key} must be int64 (G,10)")
    for g in range(groups):
        if [tuple(int(v) for v in pair) for pair in zip(arrays["cf_actions"][g],
                                                        arrays["cf_durations"][g])] \
                != [tuple(p) for p in prot.ACTION_PLANS]:
            raise ValueError("truth CF mechanism table drifted from ACTION_PLANS")
    factual_actions = np.asarray(arrays["factual_actions"])
    if factual_actions.shape != (groups, prot.HORIZON) \
            or factual_actions.dtype != np.int64:
        raise ValueError("truth factual_actions must be int64 (G,20)")
    t_star = np.asarray(arrays["t_star"])
    if t_star.shape != (groups,) or t_star.dtype != np.int64 \
            or int(t_star.min()) < prot.T_STAR_LOW \
            or int(t_star.max()) > prot.T_STAR_HIGH:
        raise ValueError("truth t_star must be int64 (G,) inside the frozen interval")
    group_seed = np.asarray(arrays["group_seed"])
    if group_seed.shape != (groups,) or group_seed.dtype != np.int64 \
            or np.unique(group_seed).size != groups:
        raise ValueError("truth group_seed must be unique int64 (G,)")
    mask = np.asarray(arrays["inference_mask"])
    if mask.shape != (groups, prot.HORIZON, prot.OBJECT_CAPACITY) \
            or mask.dtype != bool:
        raise ValueError("truth inference_mask must be bool (G,20,5)")
    return {"arrays": arrays, "path": str(truth_path), "sha256": digest,
            "n_groups": int(groups)}


def load_confirmation_truth(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed truth — legal ONLY after the burned confirmation lock.

    Verifies the lock bindings against the current frozen files first, then the truth
    file's hash against the collection receipt, the exact frozen field set, and every
    shape/dtype/finiteness/uniqueness contract. Read-only; never writes.
    """
    checked = prot.validate_config(config)
    paths = confirm_artifact_paths(checked)
    _verify_burned_lock(checked, paths)
    return _load_truth_arrays(checked, paths)


# --- plan 6.4 step 7: assemble the create-once per-group quantities --------------------------------------


def _quantities_receipt_bindings(checked: Mapping[str, Any], paths: Mapping[str, Path],
                                 truth_sha256: str) -> dict[str, str]:
    return {
        "inference_inputs_sha256": sha256_file(paths["inputs"]),
        "predictions_sha256": sha256_file(paths["predictions"]),
        "risk_scores_sha256": sha256_file(paths["risk_scores"]),
        "confirmation_truth_sha256": truth_sha256,
        "inference_receipt_sha256": sha256_file(paths["inference_receipt"]),
        "determinism_receipt_sha256": sha256_file(paths["determinism_receipt"]),
        "score_freeze_receipt_sha256": sha256_file(paths["score_freeze_receipt"]),
        "collection_receipt_sha256": sha256_file(paths["collection_receipt"]),
        "protocol_lock_sha256": sha256_file(paths["protocol_lock"]),
        "lock_sha256": sha256_file(paths["lock"]),
        "statistics_code_sha256": sha256_file(
            ROOT / "code/ac_twin/stage4cr2_confirm_statistics.py"),
    }


def _verify_quantities_receipt(checked: Mapping[str, Any], paths: Mapping[str, Path]
                               ) -> dict[str, Any]:
    """Verify an existing quantities NPZ + receipt (same-transaction resume proof).

    Any drift after the burn is UNPROVABLE -> ResourceAtRiskError, never a silent
    recompute.
    """
    receipt_path = paths["quantities_receipt"]
    npz_path = paths["quantities_npz"]
    if not receipt_path.is_file():
        raise ResourceAtRiskError(
            f"confirm quantities exist without their create-once receipt: {npz_path}")
    receipt = _read_json(receipt_path)
    if receipt.get("plan_id") != prot.PLAN_ID:
        raise ResourceAtRiskError("quantities receipt plan_id drifted")
    for key, expected in (("bootstrap_entropy", prot.BOOTSTRAP_ENTROPY),
                          ("bootstrap_resamples", prot.BOOTSTRAP_RESAMPLES)):
        if int(receipt.get(key, -1)) != expected:
            raise ResourceAtRiskError(
                f"quantities receipt {key} drifted from the frozen {expected}")
    if float(receipt.get("confidence", -1.0)) != prot.CONFIDENCE:
        raise ResourceAtRiskError("quantities receipt confidence drifted from 0.95")
    if receipt.get("primary_baseline") != PRIMARY_BASELINE:
        raise ResourceAtRiskError("quantities receipt primary baseline drifted")
    if not npz_path.is_file() or sha256_file(npz_path) \
            != receipt.get("confirm_group_quantities_sha256"):
        raise ResourceAtRiskError(
            "the confirm quantities NPZ drifted from its create-once receipt")
    if receipt.get("lock_sha256") != sha256_file(paths["lock"]):
        raise ResourceAtRiskError(
            "the quantities receipt is not bound to the burned confirmation lock")
    return receipt


def assemble_confirm_quantities(config: Mapping[str, Any]) -> str:
    """Read the sealed truth ONCE, build regret/strata, write create-once quantities.

    Plan section 6.4 steps 6-7. Refuses unless the one-shot lock is burned; refuses
    any second adjudication (the formal report already exists); re-verifies the full
    hash chain; cross-binds the same-order group_seed and the integrity copies across
    legal inputs / predictions / risk scores / truth (misorder or mismatch is INVALID,
    never repaired); recomputes J_hat from the frozen predictions and cross-checks the
    persisted reference action; computes J_true with the frozen Stage 4C consequence
    function over ALL TEN actions per group; writes ``confirm_group_quantities.npz``
    and its receipt exactly once. An existing quantities pair is verified and returned
    unchanged (same-transaction resume); any verification failure after the burn is
    ``STAGE4CR2_CONFIRM_RESOURCE_AT_RISK``.
    """
    checked = prot.validate_config(config)
    paths = confirm_artifact_paths(checked)
    if paths["report"].is_file():
        raise FileExistsError(
            f"a second confirmation adjudication is refused; the formal report "
            f"already exists: {paths['report']}")
    if paths["quantities_npz"].exists() or paths["quantities_receipt"].exists():
        _verify_quantities_receipt(checked, paths)
        return str(paths["quantities_npz"])

    lock_payload = _verify_burned_lock(checked, paths)
    inference = _verify_inference_chain(paths)
    truth = _load_truth_arrays(checked, paths)
    truth_arrays = truth["arrays"]
    groups = int(truth["n_groups"])

    legal = cinf._read_legal_npz(paths["inputs"],
                                 {key: key for key in cinf.LEGAL_INPUT_KEYS})
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        predictions = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        scores = {key: np.asarray(archive[key]) for key in archive.files}

    # --- same-order group binding across all four files (section 6.1) ---------------------
    inputs_seed = np.asarray(legal["group_seed"], dtype=np.int64).ravel()
    truth_seed = np.asarray(truth_arrays["group_seed"], dtype=np.int64).ravel()
    predictions_seed = np.asarray(predictions["group_seed"], dtype=np.int64).ravel()
    scores_seed = np.asarray(scores["group_seed"], dtype=np.int64).ravel()
    if not (np.array_equal(inputs_seed, truth_seed)
            and np.array_equal(inputs_seed, predictions_seed)
            and np.array_equal(inputs_seed, scores_seed)):
        raise ValueError(
            "group_seed order diverges across inputs/predictions/risk_scores/truth; "
            "missing/reordered groups are INVALID, never realigned")
    if inputs_seed.size != groups:
        raise ValueError("legal inputs and sealed truth group counts diverge")
    # --- integrity copies inside the sealed truth must equal the legal inputs -------------
    for key in ("cf_actions", "cf_durations", "factual_actions", "t_star"):
        if not np.array_equal(np.asarray(truth_arrays[key]), np.asarray(legal[key])):
            raise ValueError(
                f"truth integrity copy {key!r} diverges from the legal inputs; the "
                "pairing is unprovable -> INVALID")
    if not np.array_equal(np.asarray(truth_arrays["inference_mask"], dtype=bool),
                          np.asarray(legal["inference_mask"], dtype=bool)):
        raise ValueError(
            "truth integrity copy 'inference_mask' diverges from the legal inputs")

    # --- J_hat from the frozen predictions; cross-check the persisted reference action ----
    history = np.asarray(legal["object_history"], dtype=np.float64)
    factual_xy = np.asarray(legal["object_factual_future"], dtype=np.float64)[
        ..., POSITION_SLICE]
    history_last_xy = history[:, -1, s4co.EGO_SLOT, POSITION_SLICE]
    mask = np.asarray(legal["inference_mask"], dtype=bool)
    e2e = np.asarray(predictions["e2e"], dtype=np.float64)
    if e2e.shape != (N_SEEDS, groups, prot.N_ACTIONS, prot.HORIZON,
                     prot.OBJECT_CAPACITY, 2):
        raise ValueError(f"predictions e2e must be (5,G,10,20,5,2); got {e2e.shape}")
    _, j_hat = s4cpf.compute_predicted_costs(e2e, factual_xy, history_last_xy, mask)
    a_hat = np.argmin(j_hat, axis=-1).astype(np.int64)
    reference_action = np.asarray(scores["reference_action"], dtype=np.int64).ravel()
    if not np.array_equal(a_hat, reference_action):
        raise ValueError(
            "recomputed J_hat argmin diverges from the persisted reference_action; "
            "the prediction/score binding is broken")

    # --- J_true from the sealed truth with the frozen consequence function ----------------
    true_xy = np.asarray(truth_arrays["object_counterfactual_futures"],
                         dtype=np.float64)[..., POSITION_SLICE]
    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, prot.N_ACTIONS, 2))
    mask_b = np.broadcast_to(mask[:, None, :, :],
                             (groups, prot.N_ACTIONS) + mask.shape[1:])
    comp_true = s4co.trajectory_components(true_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))
    decision = _decision_quantities(j_hat, j_true)

    t_star = np.asarray(legal["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(scores["interaction_min_distance"], dtype=np.float64).ravel()
    tstar_layer = (t_star >= TSTAR_SPLIT).astype(np.int64)
    interaction_layer = (distance > float(prot.INTERACTION_THRESHOLD)).astype(np.int64)
    trim1_keep = s4cs.trim_high_regret(decision["regret"], TRIM_FRACTIONS[0])
    trim5_keep = s4cs.trim_high_regret(decision["regret"], TRIM_FRACTIONS[1])

    # --- Gate-Q assembly checks (recorded; evaluated inside the gates) --------------------
    schedule = prot.build_confirm_seed_schedule(checked)
    domain = set(int(v) for v in schedule)
    role_seeds_valid = bool(
        np.unique(inputs_seed).size == groups
        and all(int(s) in domain for s in inputs_seed))
    finite_truth = bool(np.isfinite(j_true).all()
                        and np.isfinite(decision["regret"]).all())
    all_complete = bool(
        groups == prot.CONFIRM_ACCEPTED_GROUPS
        and inference["collection_accepted_groups"] == prot.CONFIRM_ACCEPTED_GROUPS
        and j_true.shape == (groups, prot.N_ACTIONS)
        and j_hat.shape == (groups, prot.N_ACTIONS)
        and finite_truth)
    by_seed = np.asarray(scores["risk_by_seed"], dtype=np.float64)
    ensemble = np.asarray(scores["risk_ensemble"], dtype=np.float64)
    qa = {
        "all_complete": all_complete,
        "hashes_match": True,   # the full chain re-verified above
        "role_seeds_valid": role_seeds_valid,
        "determinism_passed": True,  # the lock re-verified the determinism receipt
        "one_shot_locks": True,
        "ensemble_is_arithmetic_mean": bool(
            by_seed.shape == (N_SEEDS, groups)
            and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)),
    }

    npz_path = paths["quantities_npz"]
    payload: dict[str, np.ndarray] = {
        "group_seed": inputs_seed,
        "regret": decision["regret"],
        "y_high": decision["y_high"],
        "random_regret": decision["random_regret"],
        "idle_regret": decision["idle_regret"],
        "a_hat": decision["a_hat"],
        "a_star": decision["a_star"],
        "J_hat": j_hat,
        "J_true": j_true,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": np.asarray(scores["mu_only_ensemble"], dtype=np.float64),
        "class_only_ensemble": np.asarray(scores["class_only_ensemble"],
                                          dtype=np.float64),
        "tstar_layer": tstar_layer,
        "interaction_layer": interaction_layer,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "trim1_keep": trim1_keep,
        "trim5_keep": trim5_keep,
        "n_groups": np.int64(groups),
        "interaction_threshold": np.float64(prot.INTERACTION_THRESHOLD),
    }
    for name in ALL_BASELINE_NAMES:
        if name == "CONSTANT_RANDOM":
            continue  # stored separately as ``constant_random`` by the inference stage
        risk = np.asarray(scores[f"baseline_{name}"], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"confirm baseline risk {name!r} must be finite/aligned")
        payload[_BASELINE_NPZ_PREFIX + name] = risk
    payload[_BASELINE_NPZ_PREFIX + "CONSTANT_RANDOM"] = np.asarray(
        scores["constant_random"], dtype=np.float64)
    atomic_save_npz(npz_path, **payload)

    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "confirm",
        "n_groups": int(groups),
        "confirm_group_quantities_path": str(npz_path),
        "confirm_group_quantities_sha256": sha256_file(npz_path),
        "primary_baseline": PRIMARY_BASELINE,
        "interaction_threshold": float(prot.INTERACTION_THRESHOLD),
        "interaction_threshold_source": "R2 train-only median, frozen by selection",
        "tstar_stratum_split": TSTAR_SPLIT,
        "strata": {
            "tstar_low": f"t_star < {TSTAR_SPLIT}",
            "tstar_high": f"t_star >= {TSTAR_SPLIT}",
            "interaction_low": (
                f"interaction_min_distance <= {prot.INTERACTION_THRESHOLD}"),
            "interaction_high": (
                f"interaction_min_distance > {prot.INTERACTION_THRESHOLD}"),
            "note": ("two marginal splits, NOT a 4-cell cross; R2 stratum "
                     "coding/naming kept; quantiles never refit on confirm data"),
        },
        "coverage": COVERAGE,
        "high_regret_threshold": prot.HIGH_REGRET_THRESHOLD,
        "trim_fractions": list(TRIM_FRACTIONS),
        "bootstrap_entropy": prot.BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": prot.BOOTSTRAP_RESAMPLES,
        "confidence": prot.CONFIDENCE,
        "qa": qa,
        "bindings": _quantities_receipt_bindings(checked, paths, truth["sha256"]),
        "lock_sha256": sha256_file(paths["lock"]),
        "lock_burned_at": lock_payload.get("burned_at", ""),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["quantities_receipt"], receipt)
    return str(npz_path)


def load_confirm_group_quantities(
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
    quantities["strongest_baseline"] = str(receipt["primary_baseline"])
    quantities["n_groups"] = int(np.asarray(data["n_groups"]).item())
    quantities["interaction_threshold"] = float(
        np.asarray(data["interaction_threshold"]).item())
    quantities["qa"] = {key: bool(value) for key, value in receipt["qa"].items()}
    _validate_quantities(quantities)
    return quantities


# --- section-7 output rendering (byte-deterministic) ----------------------------------------------------


def _fmt(value: Any) -> str:
    """Fixed CSV formatting (byte-deterministic tables)."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "%.17g" % float(value)
    return str(value)


def _csv_bytes(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)  # default lineterminator "\r\n": deterministic
    writer.writerow(header)
    for row in rows:
        writer.writerow([_fmt(value) for value in row])
    return buffer.getvalue().encode("utf-8")


def _write_bytes_once_or_verify(path: Path, blob: bytes) -> None:
    """Create-once bytes: a pre-existing file must be byte-identical, else the
    post-burn transaction is unprovable (``STAGE4CR2_CONFIRM_RESOURCE_AT_RISK``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as handle:
            handle.write(blob)
    except FileExistsError:
        if path.read_bytes() != blob:
            raise ResourceAtRiskError(
                f"a pre-existing output differs from the recomputed bytes and the "
                f"transaction cannot be proven identical: {path}")


def _render_table_c1(decision: ConfirmDecision) -> bytes:
    point, ci_lower = decision.metrics["point"], decision.metrics["ci_lower"]
    upper = decision.metrics["ci_upper"]
    task, risk = decision.task, decision.risk

    def row(gate, condition, metric=None, passed=None, note=""):
        if metric is None:
            return (gate, condition, None, None, None, passed, note)
        return (gate, condition, point[metric], ci_lower[metric], upper[metric],
                passed, note)

    rows: list[tuple[Any, ...]] = [
        ("TASK", "c1_all_complete_gate_q", None,
         task["conditions"]["c1_all_complete_gate_q"],
         f"n_groups={decision.n_groups}"),
        ("TASK", "c2_high_regret_frac_ge_15pct", "high_regret_frac",
         task["conditions"]["c2_high_regret_frac_ge_15pct"], ""),
        ("TASK", "c3_at_least_3_oracle_actions_ge_5pct", None,
         task["conditions"]["c3_at_least_3_oracle_actions_ge_5pct"],
         f"covered={task['oracle_actions_covered_ge_5pct']}"),
        ("TASK", "c4_branch_spearman_ci_lower_positive", "spearman_branch",
         task["conditions"]["c4_branch_spearman_ci_lower_positive"], ""),
        ("TASK", "c5_base_vs_random_point_and_ci_positive",
         "base_vs_random_improvement",
         task["conditions"]["c5_base_vs_random_point_and_ci_positive"], ""),
        ("TASK", "c6_base_not_worse_than_idle", "base_mean_regret",
         task["conditions"]["c6_base_not_worse_than_idle"],
         f"idle_mean_regret={_fmt(point['idle_mean_regret'])}"),
        ("TASK", "c7_strata_ge_200_groups", None,
         task["conditions"]["c7_strata_ge_200_groups"],
         json.dumps(task["stratum_counts"], sort_keys=True)),
        ("RISK", "r1_i_aurc_regret", "i_aurc_regret",
         risk["conditions"]["r1_i_aurc_regret"], "primary baseline SEED_VOTE_ENTROPY"),
        ("RISK", "r2_i_regret80", "i_regret80",
         risk["conditions"]["r2_i_regret80"], "primary baseline SEED_VOTE_ENTROPY"),
        ("RISK", "r3_pr_auc_minus_prevalence", "pr_auc_minus_prevalence",
         risk["conditions"]["r3_pr_auc_minus_prevalence"], ""),
        ("RISK", "r4_seeds_i_aurc", None, risk["conditions"]["r4_seeds_i_aurc"],
         f"positive={risk['seeds_i_aurc_positive']}/5"),
        ("RISK", "r5_seeds_i_regret80", None, risk["conditions"]["r5_seeds_i_regret80"],
         f"positive={risk['seeds_i_regret80_positive']}/5"),
        ("RISK", "r6_strata_floor", None, risk["conditions"]["r6_strata_floor"],
         json.dumps(risk["strata_i_aurc"], sort_keys=True)),
        ("RISK", "r7_trims_positive", None, risk["conditions"]["r7_trims_positive"],
         json.dumps(risk["trims_i_aurc"], sort_keys=True)),
        ("RISK", "r8_margin_spearman", "spearman_ensemble_margin",
         risk["conditions"]["r8_margin_spearman"], ""),
        ("RISK", "r9_qa_and_one_shot_locks", None,
         risk["conditions"]["r9_qa_and_one_shot_locks"], ""),
        ("TERMINAL", "status", None, decision.passed, decision.status),
    ]
    return _csv_bytes(
        ("gate", "condition", "point", "ci95_lower", "ci95_upper", "passed", "notes"),
        rows)


def _render_table_c2(decision: ConfirmDecision) -> bytes:
    """Primary model + three standard baselines + two references; the non-primary
    comparisons ride the SAME paired draws and never enter the Gate."""
    model = decision.ablations["primary_model"]
    rows: list[tuple[Any, ...]] = [(
        "DIRECT_ENSEMBLE", "primary_model", model["aurc_regret"], model["regret_at_80"],
        model["average_precision"], model["pr_auc_minus_prevalence"],
        None, None, None, None, None, None)]
    for name in ALL_BASELINE_NAMES:
        entry = decision.ablations["baseline_curves"][name]
        improvements = ("i_aurc_regret" in entry)
        rows.append((
            name, entry["role"], entry["aurc_regret"], entry["regret_at_80"],
            entry["average_precision"], entry["pr_auc_minus_prevalence"],
            entry.get("i_aurc_regret") if improvements else None,
            entry["i_aurc_regret_ci95"][0] if improvements else None,
            entry["i_aurc_regret_ci95"][1] if improvements else None,
            entry.get("i_regret80") if improvements else None,
            entry["i_regret80_ci95"][0] if improvements else None,
            entry["i_regret80_ci95"][1] if improvements else None,
        ))
    return _csv_bytes(
        ("name", "role", "aurc_regret", "regret_at_80", "average_precision",
         "pr_auc_minus_prevalence", "i_aurc_point", "i_aurc_ci95_lower",
         "i_aurc_ci95_upper", "i_regret80_point", "i_regret80_ci95_lower",
         "i_regret80_ci_upper"),
        rows)


def _render_table_c3(decision: ConfirmDecision) -> bytes:
    point, lower, upper = (decision.metrics["point"], decision.metrics["ci_lower"],
                           decision.metrics["ci_upper"])
    rows: list[tuple[Any, ...]] = []
    for seed in prot.MODEL_SEEDS:
        for metric in (f"seed{seed}_i_aurc", f"seed{seed}_i_regret80"):
            rows.append((f"seed_{seed}", metric.rsplit("_i_", 1)[-1],
                         point[metric], lower[metric], upper[metric]))
    for name in ("tstar_low_i_aurc", "tstar_high_i_aurc", "interaction_low_i_aurc",
                 "interaction_high_i_aurc", "trim1_i_aurc", "trim5_i_aurc"):
        kind = "stratum" if "i_aurc" in name and not name.startswith("trim") else "trim"
        rows.append((name, kind, point[name], lower[name], upper[name]))
    return _csv_bytes(("name", "kind", "point", "ci95_lower", "ci95_upper"), rows)


def _render_table_c4(decision: ConfirmDecision) -> bytes:
    rows: list[tuple[Any, ...]] = []
    for name in ("MU_ONLY", "CLASS_ONLY"):
        entry = decision.ablations[name]
        for metric in ("i_aurc_regret", "i_regret80"):
            ci = entry[f"{metric}_ci95"]
            rows.append((name, metric, entry[metric], ci[0], ci[1]))
    return _csv_bytes(("ablation", "metric", "point", "ci95_lower", "ci95_upper"),
                      rows)


def _within_group_rank_correlations(j_hat: np.ndarray, j_true: np.ndarray
                                    ) -> np.ndarray:
    """Per-group Spearman(J_hat, J_true) over the ten actions (a within-group
    diagnostic; the overall branch correlation never substitutes for it)."""
    groups = j_hat.shape[0]
    out = np.empty(groups, dtype=np.float64)
    for g in range(groups):
        out[g] = s4cs.spearman(j_hat[g], j_true[g])
    return out


def _render_table_c5(quantities: Mapping[str, Any], decision: ConfirmDecision,
                     within_corr: np.ndarray) -> bytes:
    regret = np.asarray(quantities["regret"], dtype=np.float64)
    point = decision.metrics["point"]
    rows: list[tuple[Any, ...]] = [
        ("regret_distribution", "mean", float(regret.mean())),
        ("regret_distribution", "median", float(np.median(regret))),
        ("regret_distribution", "p90", float(np.quantile(regret, 0.9))),
        ("regret_distribution", "p99", float(np.quantile(regret, 0.99))),
        ("regret_distribution", "max", float(regret.max())),
        ("regret_distribution", "high_regret_frac", point["high_regret_frac"]),
        ("reference_mean_regret", "base_action", point["base_mean_regret"]),
        ("reference_mean_regret", "random_action_ten_action_mean",
         point["random_mean_regret"]),
        ("reference_mean_regret", "always_idle_plan_(1,3)", point["idle_mean_regret"]),
        ("within_group_rank_corr", "mean", float(within_corr.mean())),
        ("within_group_rank_corr", "median", float(np.median(within_corr))),
        ("within_group_rank_corr", "p10", float(np.quantile(within_corr, 0.1))),
        ("within_group_rank_corr", "p90", float(np.quantile(within_corr, 0.9))),
        ("within_group_rank_corr", "frac_negative",
         float((within_corr < 0.0).mean())),
    ]
    oracle = np.asarray(quantities["a_star"], dtype=np.int64)
    selected = np.asarray(quantities["a_hat"], dtype=np.int64)
    for a, plan in enumerate(prot.ACTION_PLANS):
        rows.append(("oracle_action_frequency", f"a_star={tuple(plan)}",
                     float((oracle == a).mean())))
    for a, plan in enumerate(prot.ACTION_PLANS):
        rows.append(("selected_action_frequency", f"a_hat={tuple(plan)}",
                     float((selected == a).mean())))
    return _csv_bytes(("section", "key", "value"), rows)


def _render_secondary_analyses(quantities: Mapping[str, Any],
                               within_corr: np.ndarray) -> bytes:
    """Preregistered secondary analyses (plan section 7): layers by the base decision
    a_hat's action semantics and duration, plus the within-group J_hat/J_true rank
    correlation distribution. Layers with <100 groups are low-support exploratory and
    never endorse the main method."""
    regret = np.asarray(quantities["regret"], dtype=np.float64)
    y_high = np.asarray(quantities["y_high"], dtype=np.int64)
    a_hat = np.asarray(quantities["a_hat"], dtype=np.int64)
    groups = regret.size
    rows: list[tuple[Any, ...]] = []

    def layer_row(section, layer, mask):
        n = int(mask.sum())
        support = "preregistered_secondary" if n >= 100 else "low_support_exploratory"
        rows.append((
            section, layer, n,
            float(regret[mask].mean()) if n else None,
            float(y_high[mask].mean()) if n else None,
            float(within_corr[mask].mean()) if n else None,
            None, support))

    for action_id, name in ACTION_NAMES.items():
        mask = np.array([prot.ACTION_PLANS[int(a)][0] == action_id for a in a_hat])
        layer_row("a_hat_action_semantics", name, mask)
    for duration in (1, 3):
        mask = np.array([prot.ACTION_PLANS[int(a)][1] == duration for a in a_hat])
        layer_row("a_hat_duration", f"duration_{duration}", mask)
    for label, value in (
            ("mean", float(within_corr.mean())),
            ("median", float(np.median(within_corr))),
            ("p10", float(np.quantile(within_corr, 0.1))),
            ("p25", float(np.quantile(within_corr, 0.25))),
            ("p75", float(np.quantile(within_corr, 0.75))),
            ("p90", float(np.quantile(within_corr, 0.9))),
            ("frac_negative", float((within_corr < 0.0).mean()))):
        rows.append(("within_group_rank_corr_distribution", label, groups,
                     None, None, None, value, "preregistered_secondary"))
    return _csv_bytes(
        ("section", "layer", "n_groups", "mean_regret", "high_regret_frac",
         "mean_within_group_rank_corr", "value", "support"),
        rows)


def _render_fig_c1(quantities: Mapping[str, Any]) -> bytes:
    """Full-coverage regret curves for the ensemble and every baseline; fixed 80% mark."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regret = np.asarray(quantities["regret"], dtype=np.float64)
    groups = regret.size
    coverage = np.arange(1, groups + 1, dtype=np.float64) / groups
    curves = [("DIRECT_ENSEMBLE", np.asarray(quantities["ensemble_risk"],
                                            dtype=np.float64))]
    for name in ALL_BASELINE_NAMES:
        curves.append((name, np.asarray(quantities["baseline_risks"][name],
                                        dtype=np.float64)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, risk in curves:
        order = s4cs.order_by_risk(risk)
        r_k = np.cumsum(regret[order]) / np.arange(1, groups + 1)
        ax.plot(coverage, r_k, label=name, linewidth=0.9)
    ax.axvline(COVERAGE, color="black", linestyle="--", linewidth=0.8,
               label="coverage 0.80")
    ax.set_xlabel("coverage")
    ax.set_ylabel("R(k): mean regret of the k lowest-risk groups")
    ax.legend(fontsize=6)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150,
                metadata={"Software": "stage4cr2_confirm_statistics"})
    plt.close(fig)
    return buffer.getvalue()


def _render_decision_md(decision: ConfirmDecision,
                        quantities: Mapping[str, Any]) -> bytes:
    """The formal decision document (plan section 7 paper boundary, verbatim rules)."""
    point = decision.metrics["point"]
    lower, upper = decision.metrics["ci_lower"], decision.metrics["ci_upper"]
    lines = [
        "# Stage 4C-R2 Independent Confirmation — Decision",
        "",
        f"- plan_id: {decision.plan_id}",
        f"- n_groups: {decision.n_groups}",
        f"- terminal status: **{decision.status}**",
        f"- gate TASK-CONFIRM status: {decision.task['status']}",
        f"- primary baseline (validation-frozen, Gate-only): {PRIMARY_BASELINE}",
        "",
        "## Primary endpoints (95% CI, two-sided percentile, 10,000 paired draws)",
        "",
        f"- I_AURC_REGRET = {_fmt(point['i_aurc_regret'])} "
        f"[{_fmt(lower['i_aurc_regret'])}, {_fmt(upper['i_aurc_regret'])}]",
        f"- I_REGRET@80 = {_fmt(point['i_regret80'])} "
        f"[{_fmt(lower['i_regret80'])}, {_fmt(upper['i_regret80'])}]",
        f"- PR-AUC - prevalence = {_fmt(point['pr_auc_minus_prevalence'])} "
        f"[{_fmt(lower['pr_auc_minus_prevalence'])}, "
        f"{_fmt(upper['pr_auc_minus_prevalence'])}]",
        f"- AURC_REGRET (ensemble) = {_fmt(point['aurc_model'])}; "
        f"REGRET@80 (ensemble) = {_fmt(point['regret80_model'])}; "
        f"coverage = {_fmt(COVERAGE)}; "
        f"rejected fraction = {_fmt(1.0 - COVERAGE)}",
        "",
        "## Paper boundary (plan section 7)",
        "",
    ]
    if decision.status == prot.STATUS_CONFIRMED:
        lines += [SUCCESS_SENTENCE, ""]
    else:
        lines += [
            f"终态为 {decision.status}，成功句式不适用；不得改写为任何形式的确认成功。",
            "",
        ]
    lines += [
        ACTION_FREQUENCY_CAVEAT,
        "",
        PROHIBITED_CLAIMS,
        "",
        "Trim 稳健性依赖真值，仅为预注册稳健性分析，不是可部署的过滤器。单项 CI 是边际"
        " CI，不宣称所有次分析联合 95% 覆盖。无论确认成败，不改写 Stage 4B 或 R2 已有终态。",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


# --- plan 6.4 steps 8-9: the single adjudication + create-once formal outputs ----------------------------


def adjudicate_confirmation(
    config: Mapping[str, Any],
    *,
    bootstrap_fn: Callable[[Mapping[str, Any]], BootstrapResult] | None = None,
) -> dict[str, Any]:
    """Run the ONE confirmation adjudication and write the create-once formal outputs.

    Refuses a second adjudication (the formal report already exists). Requires the
    create-once quantities pair; verifies the NPZ hash, the frozen bootstrap constants
    and the lock binding — any post-burn drift is unprovable and stops at
    ``STAGE4CR2_CONFIRM_RESOURCE_AT_RISK``. The statistics run exactly once at the
    frozen entropy/resamples/95% confidence; the six section-7 files and the secondary
    analyses are byte-deterministic (a same-transaction resume re-derives identical
    bytes and never overwrites); the report JSON binds the quantities hash and is
    written LAST (its presence marks the completed transaction).
    """
    checked = prot.validate_config(config)
    paths = confirm_artifact_paths(checked)
    if paths["report"].is_file():
        raise FileExistsError(
            f"a second confirmation adjudication is refused; the formal report "
            f"already exists: {paths['report']}")
    if not paths["lock"].is_file():
        raise ValueError(
            "the confirmation lock is not burned; adjudication requires the one-shot "
            "lock (plan section 6.4)")
    if not paths["quantities_npz"].is_file() \
            and not paths["quantities_receipt"].is_file():
        raise ValueError(
            "the confirm quantities are missing; assemble_confirm_quantities must run "
            "before adjudication")
    receipt = _verify_quantities_receipt(checked, paths)
    quantities = load_confirm_group_quantities(paths["quantities_npz"],
                                               paths["quantities_receipt"])
    if bootstrap_fn is None:
        bootstrap = paired_group_bootstrap(
            quantities, seed=int(checked["bootstrap_entropy"]),
            n_resamples=int(checked["bootstrap_resamples"]))
    else:
        bootstrap = bootstrap_fn(quantities)
    decision = evaluate_confirmation_gates(quantities, bootstrap)

    within_corr = _within_group_rank_correlations(
        np.asarray(quantities["J_hat"], dtype=np.float64),
        np.asarray(quantities["J_true"], dtype=np.float64))
    outputs = {
        "table_c1": _render_table_c1(decision),
        "table_c2": _render_table_c2(decision),
        "fig_c1": _render_fig_c1(quantities),
        "table_c3": _render_table_c3(decision),
        "table_c4": _render_table_c4(decision),
        "table_c5": _render_table_c5(quantities, decision, within_corr),
        "secondary_csv": _render_secondary_analyses(quantities, within_corr),
        "decision": _render_decision_md(decision, quantities),
    }
    for key, blob in outputs.items():
        _write_bytes_once_or_verify(paths[key], blob)

    report = {
        "plan_id": prot.PLAN_ID,
        "role": "confirm",
        "status": decision.status,
        "passed": decision.passed,
        "n_groups": decision.n_groups,
        "decision": decision.to_dict(),
        "bindings": {
            "confirm_group_quantities_sha256":
                receipt["confirm_group_quantities_sha256"],
            "confirm_group_quantities_receipt_sha256":
                sha256_file(paths["quantities_receipt"]),
            "lock_sha256": sha256_file(paths["lock"]),
            "statistics_code_sha256": sha256_file(
                ROOT / "code/ac_twin/stage4cr2_confirm_statistics.py"),
        },
        "outputs": {key: {"path": str(paths[key]),
                          "sha256": hashlib.sha256(blob).hexdigest()}
                    for key, blob in outputs.items()},
        "paper_boundary": {
            "success_sentence_written": bool(decision.status == prot.STATUS_CONFIRMED),
            "action_frequency_caveat": ACTION_FREQUENCY_CAVEAT,
            "prohibited_claims": PROHIBITED_CLAIMS,
            "secondary_analyses_note": (
                "layers with <100 groups are low-support exploratory; analyses outside "
                "the section-7 list would be marked exploratory_not_preregistered and "
                "never enter the Gate or the main conclusion"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["report"], report)
    result = decision.to_dict()
    result["report_path"] = str(paths["report"])
    result["decision_path"] = str(paths["decision"])
    return result


__all__ = [
    "ALL_BASELINE_NAMES",
    "BootstrapResult",
    "COVERAGE",
    "ConfirmDecision",
    "N_QUANTITIES",
    "N_SEEDS",
    "POSITION_SLICE",
    "PRIMARY_BASELINE",
    "QA_KEYS",
    "Q_AP_COMBINED",
    "Q_AP_MARGIN",
    "Q_AP_MODEL",
    "Q_AP_RANDOM",
    "Q_AP_STD",
    "Q_AP_SVE",
    "Q_AURC_BASELINE",
    "Q_AURC_COMBINED",
    "Q_AURC_MARGIN",
    "Q_AURC_MODEL",
    "Q_AURC_RANDOM",
    "Q_AURC_STD",
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
    "Q_MARGIN_I_AURC",
    "Q_MARGIN_I_R80",
    "Q_MU_I_AURC",
    "Q_MU_I_R80",
    "Q_PR",
    "Q_R80_BASELINE",
    "Q_R80_COMBINED",
    "Q_R80_MARGIN",
    "Q_R80_MODEL",
    "Q_R80_RANDOM",
    "Q_R80_STD",
    "Q_RANDOM_MEAN_REGRET",
    "Q_SEED_I_AURC",
    "Q_SEED_I_R80",
    "Q_SPEARMAN_BRANCH",
    "Q_SPEARMAN_MARGIN",
    "Q_STD_I_AURC",
    "Q_STD_I_R80",
    "Q_TRIM1",
    "Q_TRIM5",
    "Q_TSTAR_HIGH",
    "Q_TSTAR_LOW",
    "REQUIRED_ARRAY_KEYS",
    "ResourceAtRiskError",
    "SEED_I_AURC_INDEX",
    "SEED_I_R80_INDEX",
    "STATUS_COLLECTED",
    "STATUS_TASK_QUALIFIED",
    "STRATA_INDEX",
    "TRIM_FRACTIONS",
    "TRIM_INDEX",
    "TSTAR_SPLIT",
    "TRUTH_FIELDS",
    "adjudicate_confirmation",
    "assemble_confirm_quantities",
    "burn_confirmation_lock",
    "compute_confirm_quantity_vector",
    "confirm_artifact_paths",
    "coverage_index",
    "evaluate_confirmation_gates",
    "freeze_scores",
    "load_confirm_group_quantities",
    "load_confirmation_truth",
    "paired_group_bootstrap",
    "run_determinism_recheck",
    "run_smoke_engineering_statistics",
    "run_smoke_recheck",
    "smoke_artifact_paths",
]



