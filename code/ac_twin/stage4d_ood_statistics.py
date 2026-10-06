"""Stage 4D D0 statistics: domain shift, task support, D0 Gate and mechanical selection.

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §5 (verbatim conditions) and §9.2.
This module implements the D0 pilot analysis ONLY:

* the six pre-intervention quantities (plan §5.2) computed by ONE shared function from the
  pre-``t_star`` history on both the ID (R2 train) and the OOD (pilot) side;
* the standardized mean differences vs the R2 train reference and the fixed 5-fold
  logistic domain classifier with the group-bootstrap two-sided 80 % AUC CI
  (5,000 resamples; per-profile entropies 2026104101/2026104102/2026104103);
* the four task-support conditions from the pilot task truth (regret / oracle /
  ``J_hat`` / ``J_true`` — plan §5.1 explicitly allows reading pilot truth for these);
* the four integrity conditions;
* ``evaluate_d0_profile`` (ANY single failed condition makes the profile unqualified)
  and ``select_ood_profile`` (JOINT > BEHAVIOR > GEOMETRY; the create-once
  ``ood_profile_selection.json``; ``STAGE4D_OOD_TASK_INSUFFICIENT`` when none qualifies).

D0 forbidden zone (plan §5.1, Global Constraint 3): this module NEVER loads a Direct
checkpoint, NEVER produces a risk score / AURC / risk-coverage / improvement quantity and
never reads the sealed Stage 4C-R2 confirm truth or the R2 train NPZ's CF-future/regret
fields. The domain classifier ONLY proves the shift; it never enters method scoring.
``J_hat`` is recomputed from the frozen decision-predictor ensemble (the R2 frozen
primitives, read-only) — never from any Direct artifact.
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
from scipy import stats as scipy_stats

from ac_twin import stage4c_objective as s4co  # frozen consequence function, read-only
from ac_twin import stage4cr2_power as s4cr2pw  # frozen R2 power primitives, read-only
from ac_twin import stage4cr2_protocol as s4cr2  # frozen R2 contract (bootstrap entropy)
from ac_twin import stage4cr2_statistics as s4cr2st  # frozen R2 screen statistics
from ac_twin import stage4cp_statistics as s4cs  # frozen risk-curve primitives, read-only
from ac_twin import stage4d_ood_env as ood
from ac_twin import stage4d_protocol as prot
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

ROOT = prot.ROOT

# --- object-tensor layout (frozen Stage 4 contract) -------------------------------------------
PRESENCE_IDX = 0
EGO_SLOT = s4co.EGO_SLOT
POSITION_SLICE = slice(2, 4)
VELOCITY_SLICE = slice(4, 6)

# --- the six pre-intervention quantities (plan §5.2, frozen order) ------------------------------
PRE_INTERVENTION_QUANTITIES = (
    "min_ego_distance",          # ego-to-other minimum distance (m)
    "mean_abs_relative_speed",   # mean |v_i - v_ego| over present non-ego (m/s)
    "merging_longitudinal_gap",  # x_merging - x_ego (m)
    "merging_relative_speed",    # |v_merging - v_ego| (m/s)
    "density_30m",               # non-ego vehicles within 30 m of the ego (count)
    "clip_ttc",                  # clipped longitudinal time-to-collision (s)
)

#: frozen definition constants of the six quantities (documentation-grade, never tuned).
DENSITY_RADIUS_M = 30.0       # mirrors the frozen R2 30 m interaction density
TTC_CLIP_MAX = 10.0           # clip ceiling (s) of the clipped TTC
TTC_EPS = 1e-9
#: the merging vehicle is identified as the non-ego vehicle with the largest lateral
#: coordinate at the FIRST history frame (the ramp lane sits ~10.5 m above the highway
#: lanes); the runner-up must be at least this far below for an unambiguous call.
MERGING_ID_MIN_MARGIN_M = 5.0

# --- domain classifier contract (plan §5.2, verbatim) -------------------------------------------
DOMAIN_CLASSIFIER_FOLDS = 5
DOMAIN_CLASSIFIER_MAX_ITER = 1000

# --- pilot role/profile wiring ------------------------------------------------------------------
PILOT_ORDER = ("behavior", "geometry", "joint")  # collection order (plan §10 Task 3)
PROFILE_OF_PILOT = {
    "behavior": "BEHAVIOR_OOD", "geometry": "GEOMETRY_OOD", "joint": "JOINT_OOD"}
PILOT_OF_PROFILE = {v: k for k, v in PROFILE_OF_PILOT.items()}

#: the R2 train NPZ key this stage may read (plan §2.2: pre-t_star history ONLY).
R2_TRAIN_LEGAL_KEYS = ("object_history",)

# --- D0 forbidden zone (plan §5.1; tests pin these) ----------------------------------------------
#: path tokens a D0 read must NEVER touch: Direct checkpoints / risk scores / AURC /
#: risk-coverage / Direct-vs-baseline improvement artifacts.
D0_FORBIDDEN_PATH_TOKENS = (
    ".pt", ".pth", ".ckpt",           # torch checkpoints (Direct / any model weights)
    "checkpoint",
    "risk_score", "risk_train", "risk_scores",
    "aurc",
    "coverage",
    "improvement",
    "direct",
)


class D0ForbiddenReadError(RuntimeError):
    """A D0 code path attempted to open a Direct/risk/improvement artifact."""


def assert_d0_read_legal(path: str | Path) -> None:
    """Fail immediately if a D0 read targets a Direct/risk/AURC/coverage artifact.

    D0 may read only frozen configs/receipts, the pilot two-file products and the R2
    train NPZ's pre-intervention history. Any Direct checkpoint, risk-score, AURC,
    risk-coverage or improvement-quantity path is refused before it is opened.
    """
    normalized = str(path).replace("\\", "/").lower()
    name = Path(str(path)).name.lower()
    for token in D0_FORBIDDEN_PATH_TOKENS:
        if token in name or (token.startswith(".") and name.endswith(token)):
            raise D0ForbiddenReadError(
                f"D0 must never open Direct/risk/AURC/coverage/improvement artifacts: "
                f"{path}")
        if token in ("direct", "checkpoint") and token in normalized:
            raise D0ForbiddenReadError(
                f"D0 must never open Direct/risk artifacts: {path}")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (ROOT / value)


def _json_safe(value: Any) -> Any:
    """Map numpy/non-finite scalars to JSON-safe values (inf/nan -> strings)."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


# --- the six pre-intervention quantities (ONE shared function for ID and OOD) --------------------


def _identify_merging_slot(history: np.ndarray) -> np.ndarray:
    """Per-group merging-vehicle slot: the present non-ego vehicle with the largest
    lateral coordinate at the FIRST history frame (the ramp lane); fail-closed when the
    call is ambiguous (runner-up within ``MERGING_ID_MIN_MARGIN_M``) or absent.
    """
    first = history[:, 0]                       # (G, N, F)
    present = first[..., PRESENCE_IDX] > 0
    present[:, EGO_SLOT] = False
    y = np.where(present, first[..., 3], -np.inf)
    order = np.argsort(-y, axis=1)              # descending y
    merging = order[:, 0]
    top = np.take_along_axis(y, order[:, :1], axis=1)[:, 0]
    runner_up = np.take_along_axis(y, order[:, 1:2], axis=1)[:, 0]
    if not np.all(np.isfinite(top)):
        raise ValueError("a group has no present non-ego vehicle at the first history "
                         "frame; the merging vehicle cannot be identified")
    margin = top - np.where(np.isfinite(runner_up), runner_up, -np.inf)
    if not np.all(margin >= MERGING_ID_MIN_MARGIN_M):
        raise ValueError(
            "ambiguous merging-vehicle identification: the two largest first-frame "
            f"lateral coordinates differ by less than {MERGING_ID_MIN_MARGIN_M} m")
    last = history[:, -1]
    if not np.all(last[np.arange(history.shape[0]), merging, PRESENCE_IDX] > 0):
        raise ValueError("the identified merging vehicle is absent at the last "
                         "history frame")
    return merging.astype(np.int64)


def pre_intervention_features(object_history: np.ndarray) -> dict[str, np.ndarray]:
    """The six plan-§5.2 pre-intervention quantities from the pre-``t_star`` history.

    ONE shared computation used for BOTH the ID (R2 train) and the OOD (pilot) side.
    All quantities are knowable before the intervention; nothing reads any future branch.
    Geometry quantities are evaluated at the last history frame (the decision point);
    the merging vehicle is identified at the first history frame (fail-closed).

    Returns ``{quantity_name: (G,) float64}`` in the frozen
    :data:`PRE_INTERVENTION_QUANTITIES` order.
    """
    history = np.asarray(object_history, dtype=np.float64)
    if history.ndim != 4 or history.shape[1:] != (
            prot.HISTORY_LEN, prot.OBJECT_CAPACITY, prot.N_OBJECT_FEATURES):
        raise ValueError(
            f"object_history must be (G,{prot.HISTORY_LEN},{prot.OBJECT_CAPACITY},"
            f"{prot.N_OBJECT_FEATURES}); got {history.shape}")
    if not np.isfinite(history).all():
        raise ValueError("non-finite pre-intervention history")
    groups = history.shape[0]
    if groups == 0:
        raise ValueError("object_history must contain at least one group")

    merging = _identify_merging_slot(history)
    rows = np.arange(groups)

    last = history[:, -1]                                # (G, N, F)
    present = last[..., PRESENCE_IDX] > 0
    present[:, EGO_SLOT] = False
    if not np.all(present.any(axis=1)):
        raise ValueError("a group has no present non-ego vehicle at the decision frame")
    xy = last[..., POSITION_SLICE]                       # (G, N, 2)
    vel = last[..., VELOCITY_SLICE]
    ego_xy = xy[:, EGO_SLOT]                             # (G, 2)
    ego_v = vel[:, EGO_SLOT]

    distance = np.linalg.norm(xy - ego_xy[:, None, :], axis=-1)     # (G, N)
    relspeed = np.linalg.norm(vel - ego_v[:, None, :], axis=-1)
    distance = np.where(present, distance, np.inf)
    relspeed = np.where(present, relspeed, np.nan)

    min_ego_distance = distance.min(axis=1)
    mean_abs_relative_speed = np.nanmean(relspeed, axis=1)
    density_30m = ((distance <= DENSITY_RADIUS_M)).sum(axis=1).astype(np.float64)

    merging_xy = xy[rows, merging]                       # (G, 2)
    merging_v = vel[rows, merging]
    merging_longitudinal_gap = merging_xy[:, 0] - ego_xy[:, 0]
    merging_relative_speed = np.linalg.norm(merging_v - ego_v, axis=-1)

    # clipped longitudinal TTC: min over present non-ego vehicles AHEAD of the ego with
    # positive closing speed of gap/closing; clipped to [0, TTC_CLIP_MAX]; the ceiling
    # when no such vehicle exists.
    gap = xy[..., 0] - ego_xy[:, None, 0]
    closing = ego_v[:, None, 0] - vel[..., 0]
    valid_ttc = present & (gap > 0.0) & (closing > 0.0)
    ttc = np.where(valid_ttc, gap / np.maximum(closing, TTC_EPS), np.inf)
    clip_ttc = np.clip(ttc.min(axis=1), 0.0, TTC_CLIP_MAX)

    out = {
        "min_ego_distance": min_ego_distance.astype(np.float64),
        "mean_abs_relative_speed": mean_abs_relative_speed.astype(np.float64),
        "merging_longitudinal_gap": merging_longitudinal_gap.astype(np.float64),
        "merging_relative_speed": merging_relative_speed.astype(np.float64),
        "density_30m": density_30m,
        "clip_ttc": clip_ttc.astype(np.float64),
    }
    return {name: out[name] for name in PRE_INTERVENTION_QUANTITIES}


def load_r2_train_id_features(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The ID reference features from the R2 train NPZ's pre-``t_star`` history ONLY.

    Plan §2.2: this stage may read ONLY the ``object_history`` key of the R2 train NPZ
    (CF futures / regret / effect fields are never touched). The NPZ path is taken from
    the pinned train collection receipt and the file hash is re-verified against the
    plan-§2.1 binding before any read.
    """
    checked = prot.validate_config(config)
    receipt_path = _resolve(prot.R2_TRAIN_COLLECTION_RECEIPT)
    if sha256_file(receipt_path) != prot.STAGE4D_PINNED_SOURCES[prot.R2_TRAIN_COLLECTION_RECEIPT]:
        raise prot.PreflightError("R2 train collection receipt hash drifted")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    train_npz = Path(receipt["data_output"])
    if str(receipt.get("data_output_sha256")) != prot.R2_TRAIN_NPZ_SHA256:
        raise prot.PreflightError("R2 train NPZ binding drifted in the pinned receipt")
    if not train_npz.is_file():
        raise FileNotFoundError(f"R2 train NPZ missing: {train_npz}")
    if sha256_file(train_npz) != prot.R2_TRAIN_NPZ_SHA256:
        raise prot.PreflightError(f"R2 train NPZ hash drifted: {train_npz}")
    assert_d0_read_legal(train_npz)
    with np.load(train_npz, allow_pickle=False) as archive:
        history = np.asarray(archive["object_history"], dtype=np.float32)
    features = pre_intervention_features(history)
    if any(int(v.size) != int(history.shape[0]) for v in features.values()):
        raise RuntimeError("ID reference feature extraction misaligned")
    _ = checked
    return features


# --- SMD (plan §5.2: >= 2 continuous quantities with |SMD| >= 0.50) ------------------------------


def standardized_mean_difference(id_values: np.ndarray, ood_values: np.ndarray) -> float:
    """SMD = (mean_ood - mean_id) / sqrt((var_id + var_ood) / 2), ddof=1 variances.

    A zero pooled denominator is fail-closed: 0 when the means also agree, otherwise a
    signed infinity (an infinite standardized shift). NaN inputs are rejected.
    """
    id_values = np.asarray(id_values, dtype=np.float64).ravel()
    ood_values = np.asarray(ood_values, dtype=np.float64).ravel()
    if id_values.size < 2 or ood_values.size < 2:
        raise ValueError("SMD requires at least two observations per side")
    if not np.isfinite(id_values).all() or not np.isfinite(ood_values).all():
        raise ValueError("SMD inputs must be finite")
    diff = float(ood_values.mean() - id_values.mean())
    denom = float(np.sqrt((id_values.var(ddof=1) + ood_values.var(ddof=1)) / 2.0))
    if denom <= 0.0:
        return 0.0 if diff == 0.0 else float(np.copysign(np.inf, diff))
    return diff / denom


def count_shifted_quantities(smds: Mapping[str, float]) -> int:
    """Count quantities with |SMD| >= the frozen 0.50 (plan §5.2). NaN is fail-closed."""
    count = 0
    for name, value in smds.items():
        value = float(value)
        if np.isnan(value):
            raise ValueError(f"non-finite SMD for {name!r}")
        if abs(value) >= prot.DOMAIN_SHIFT_SMD_MIN:
            count += 1
    return count


# --- fixed 5-fold logistic domain classifier + group-bootstrap AUC CI ---------------------------


def _feature_matrix(features: Mapping[str, np.ndarray] | np.ndarray) -> np.ndarray:
    """Normalize features to an (n, 6) matrix in the frozen quantity order."""
    if isinstance(features, Mapping):
        matrix = np.column_stack([
            np.asarray(features[name], dtype=np.float64).ravel()
            for name in PRE_INTERVENTION_QUANTITIES])
    else:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[1] != len(PRE_INTERVENTION_QUANTITIES):
        raise ValueError(
            f"domain features must be (n, {len(PRE_INTERVENTION_QUANTITIES)}); "
            f"got {matrix.shape}")
    if matrix.shape[0] < DOMAIN_CLASSIFIER_FOLDS:
        raise ValueError("too few rows for the fixed 5-fold classifier")
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite domain features")
    return matrix


def _oof_auc(matrix: np.ndarray, labels: np.ndarray, fold_seed: int) -> float:
    """Out-of-fold AUC of the fixed stratified-5-fold logistic domain classifier.

    The spec is frozen: features standardized with the pooled mean/std, then
    ``LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)``; the fold split is a
    shuffled ``StratifiedKFold`` seeded by ``fold_seed``. Deterministic.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    scaled = (matrix - mean) / std
    oof = np.full(labels.shape, np.nan, dtype=np.float64)
    folds = StratifiedKFold(
        n_splits=DOMAIN_CLASSIFIER_FOLDS, shuffle=True, random_state=int(fold_seed))
    for train_idx, test_idx in folds.split(scaled, labels):
        clf = LogisticRegression(C=1.0, solver="lbfgs",
                                 max_iter=DOMAIN_CLASSIFIER_MAX_ITER)
        clf.fit(scaled[train_idx], labels[train_idx])
        oof[test_idx] = clf.predict_proba(scaled[test_idx])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("domain classifier out-of-fold predictions incomplete")
    return float(roc_auc_score(labels, oof))


def _bootstrap_auc_ci(
    id_matrix: np.ndarray,
    ood_matrix: np.ndarray,
    bootstrap_seed: int,
    *,
    resamples: int,
    confidence: float,
) -> dict[str, float]:
    """Group-bootstrap the full 5-fold pipeline; percentile two-sided CI of the AUC.

    Each resample redraws the ID and OOD GROUPS separately (stratified group bootstrap —
    one feature row per group), refits the entire fixed pipeline (standardization +
    stratified 5-fold logistic + out-of-fold AUC), and the percentile interval over the
    ``resamples`` AUCs is the two-sided CI. The stream derives deterministically from
    the per-profile entropy (plan §5.2: 2026104101/2026104102/2026104103).
    """
    sequence = np.random.SeedSequence([int(bootstrap_seed)])
    fold_seq, boot_seq = sequence.spawn(2)
    labels = np.concatenate([np.zeros(len(id_matrix), dtype=np.int64),
                             np.ones(len(ood_matrix), dtype=np.int64)])
    pooled = np.vstack([id_matrix, ood_matrix])
    point = _oof_auc(pooled, labels, int(fold_seq.generate_state(1)[0]))

    boot_rng = np.random.default_rng(boot_seq)
    n_id, n_ood = len(id_matrix), len(ood_matrix)
    aucs = np.empty(int(resamples), dtype=np.float64)
    for b in range(int(resamples)):
        id_idx = boot_rng.integers(0, n_id, size=n_id)
        ood_idx = boot_rng.integers(0, n_ood, size=n_ood)
        resample = np.vstack([id_matrix[id_idx], ood_matrix[ood_idx]])
        resample_labels = np.concatenate(
            [np.zeros(n_id, dtype=np.int64), np.ones(n_ood, dtype=np.int64)])
        fold_seed = int(boot_rng.integers(0, 2**31 - 1))
        aucs[b] = _oof_auc(resample, resample_labels, fold_seed)
    alpha = 1.0 - float(confidence)
    lower, upper = np.percentile(aucs, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "auc_point": float(point),
        "auc_ci_lower": float(lower),
        "auc_ci_upper": float(upper),
        "resamples": int(resamples),
        "confidence": float(confidence),
    }


def compute_domain_shift(
    id_features: Mapping[str, np.ndarray] | np.ndarray,
    ood_features: Mapping[str, np.ndarray] | np.ndarray,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Plan-§5.2 domain shift: SMDs vs R2 train + the fixed 5-fold logistic AUC CI.

    The classifier ONLY proves the shift; it never enters method scoring. Uses the
    frozen constants: |SMD| >= 0.50 on >= 2 of the six quantities; 5-fold logistic;
    group-bootstrap two-sided 80 % AUC CI lower bound >= 0.70; 5,000 resamples;
    ``bootstrap_seed`` is the per-profile entropy (behavior=2026104101,
    geometry=2026104102, joint=2026104103).
    """
    id_matrix = _feature_matrix(id_features)
    ood_matrix = _feature_matrix(ood_features)
    smds = {
        name: standardized_mean_difference(id_matrix[:, j], ood_matrix[:, j])
        for j, name in enumerate(PRE_INTERVENTION_QUANTITIES)
    }
    n_shifted = count_shifted_quantities(smds)
    smd_pass = n_shifted >= prot.DOMAIN_SHIFT_MIN_SHIFTED
    auc = _bootstrap_auc_ci(
        id_matrix, ood_matrix, int(bootstrap_seed),
        resamples=prot.DOMAIN_CLASSIFIER_BOOTSTRAP_RESAMPLES,
        confidence=prot.DOMAIN_CLASSIFIER_CONFIDENCE)
    auc_pass = auc["auc_ci_lower"] >= prot.DOMAIN_CLASSIFIER_AUC_CI_LB_MIN
    return {
        "quantities": list(PRE_INTERVENTION_QUANTITIES),
        "smd": smds,
        "abs_smd": {name: abs(value) for name, value in smds.items()},
        "smd_min": prot.DOMAIN_SHIFT_SMD_MIN,
        "min_shifted": prot.DOMAIN_SHIFT_MIN_SHIFTED,
        "n_shifted": int(n_shifted),
        "smd_pass": bool(smd_pass),
        "domain_classifier": {
            "type": "stratified_5fold_logistic_pooled_standardization",
            "n_folds": DOMAIN_CLASSIFIER_FOLDS,
            "resamples": int(auc["resamples"]),
            "confidence": float(auc["confidence"]),
            "bootstrap_seed": int(bootstrap_seed),
            "n_id": int(len(id_matrix)),
            "n_ood": int(len(ood_matrix)),
            "auc_point": float(auc["auc_point"]),
            "auc_ci_lower": float(auc["auc_ci_lower"]),
            "auc_ci_upper": float(auc["auc_ci_upper"]),
            "ci_side": "two-sided percentile",
            "auc_ci_lb_min": prot.DOMAIN_CLASSIFIER_AUC_CI_LB_MIN,
            "scope": "proof of shift only; never enters method scoring",
        },
        "domain_auc_pass": bool(auc_pass),
        "shift_significant": bool(smd_pass and auc_pass),
    }


# --- task support (plan §5.2, four conditions) ----------------------------------------------------


def evaluate_task_support(
    regret: np.ndarray,
    a_star: np.ndarray,
    J_hat: np.ndarray,
    J_true: np.ndarray,
    idle_regret: np.ndarray,
    random_regret: np.ndarray,
) -> dict[str, Any]:
    """The four plan-§5.2 task-support conditions from the pilot task truth.

    1. ``regret > 0.01`` fraction >= 15 %;
    2. at least 3 distinct oracle action plans each >= 5 %;
    3. branch-level ``Spearman(J_hat, J_true)`` point estimate > 0 (pooled G*10 branches);
    4. base-action mean regret strictly better than RANDOM_ACTION and
       not above ``ALWAYS_IDLE + 1e-12``.
    """
    j_hat = np.asarray(J_hat, dtype=np.float64)
    j_true = np.asarray(J_true, dtype=np.float64)
    if j_hat.shape != j_true.shape or j_hat.ndim != 2 \
            or j_hat.shape[1] != prot.N_ACTIONS:
        raise ValueError("J_hat/J_true must align as (G, 10)")
    groups = j_hat.shape[0]
    regret = np.asarray(regret, dtype=np.float64).ravel()
    a_star = np.asarray(a_star, dtype=np.int64).ravel()
    idle = np.asarray(idle_regret, dtype=np.float64).ravel()
    random = np.asarray(random_regret, dtype=np.float64).ravel()
    for name, arr in (("regret", regret), ("a_star", a_star),
                      ("idle_regret", idle), ("random_regret", random)):
        if arr.shape != (groups,):
            raise ValueError(f"{name} must be (G,); got {arr.shape}")
    for name, arr in (("regret", regret), ("idle_regret", idle),
                      ("random_regret", random), ("J_hat", j_hat), ("J_true", j_true)):
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite task-support array: {name}")

    high_regret_frac = float(np.mean(regret > prot.HIGH_REGRET_THRESHOLD))
    cond1 = high_regret_frac >= prot.GATE_TASK_OOD["min_high_regret_frac"]

    oracle_counts = np.bincount(a_star, minlength=prot.N_ACTIONS)
    oracle_fracs = oracle_counts / float(groups)
    n_oracle_ge = int(np.sum(oracle_fracs >= prot.GATE_TASK_OOD["min_oracle_frac"]))
    cond2 = n_oracle_ge >= prot.GATE_TASK_OOD["min_distinct_oracle"]

    spearman = float(scipy_stats.spearmanr(j_hat.ravel(), j_true.ravel()).statistic)
    if not np.isfinite(spearman):
        raise ValueError("non-finite branch-level Spearman(J_hat, J_true)")
    cond3 = spearman > 0.0

    base_mean = float(regret.mean())
    random_mean = float(random.mean())
    idle_mean = float(idle.mean())
    base_beats_random = base_mean < random_mean
    base_not_above_idle = base_mean <= idle_mean + prot.BASELINE_TIE_TOLERANCE
    cond4 = bool(base_beats_random and base_not_above_idle)

    return {
        "n_groups": int(groups),
        "high_regret_frac": high_regret_frac,
        "regret_eps": prot.GATE_TASK_OOD["regret_eps"],
        "min_high_regret_frac": prot.GATE_TASK_OOD["min_high_regret_frac"],
        "high_regret_support": bool(cond1),
        "oracle_plan_fractions": {str(i): float(oracle_fracs[i])
                                  for i in range(prot.N_ACTIONS)},
        "n_oracle_plans_ge_min_frac": n_oracle_ge,
        "min_distinct_oracle": prot.GATE_TASK_OOD["min_distinct_oracle"],
        "min_oracle_frac": prot.GATE_TASK_OOD["min_oracle_frac"],
        "oracle_diversity": bool(cond2),
        "spearman_jhat_jtrue": spearman,
        "spearman_positive": bool(cond3),
        "base_mean_regret": base_mean,
        "random_mean_regret": random_mean,
        "idle_mean_regret": idle_mean,
        "base_beats_random": bool(base_beats_random),
        "base_not_above_idle": bool(base_not_above_idle),
        "base_action_beats_references": cond4,
        "task_supported": bool(cond1 and cond2 and cond3 and cond4),
    }


# --- integrity (plan §5.2, four conditions) --------------------------------------------------------


def evaluate_pilot_integrity(
    receipt: Mapping[str, Any],
    legal_arrays: Mapping[str, np.ndarray],
    truth_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """The four plan-§5.2 integrity conditions for one collected pilot.

    1. 300/300 complete groups, eleven-branch consistency evidence
       (``prefix_max_abs_diff`` within the frozen 1e-9 tolerance) and the receipt's
       same-order binding;
    2. entity-id / mask / action-plan / assignment integrity: unique same-order
       ``group_seed`` across both files, the truth file's integrity copies equal the
       legal inputs, the CF action/duration rows equal the frozen ten-action table,
       ``t_star`` in [10, 15], and the inference mask re-proven from the legal
       history/factual contract ONLY;
    3. no capacity overflow, no NaN/Inf, no unexplained early termination (exact shapes,
       finite arrays, at least one legal evaluation cell per group);
    4. the candidate pool was not exhausted (attempts <= 6,000).
    """
    legal = {key: np.asarray(value) for key, value in legal_arrays.items()}
    truth = {key: np.asarray(value) for key, value in truth_arrays.items()}
    groups = int(np.asarray(legal["group_seed"]).ravel().size)

    cond1 = bool(
        int(receipt.get("accepted_groups", -1)) == prot.PILOT_ACCEPTED_GROUPS
        and groups == prot.PILOT_ACCEPTED_GROUPS
        and int(receipt.get("counterfactual_branches", -1))
            == prot.PILOT_ACCEPTED_GROUPS * prot.N_ACTIONS
        and str(receipt.get("status")) == "COLLECTED"
        and not bool(receipt.get("partial", True))
        and bool(receipt.get("group_order_binding", {}).get("same_order"))
        and float(receipt.get("prefix_max_abs_diff", np.inf)) <= 1e-9)

    expected_actions = np.asarray([a for a, _ in prot.ACTION_PLANS], dtype=np.int64)
    expected_durations = np.asarray([d for _, d in prot.ACTION_PLANS], dtype=np.int64)
    legal_seed = np.asarray(legal["group_seed"], dtype=np.int64).ravel()
    truth_seed = np.asarray(truth["group_seed"], dtype=np.int64).ravel()
    recomputed_mask = prot.build_inference_mask(
        np.asarray(legal["object_history"], dtype=np.float32),
        np.asarray(legal["object_factual_future"], dtype=np.float32))
    cond2 = bool(
        np.unique(legal_seed).size == groups
        and np.array_equal(legal_seed, truth_seed)
        and np.all(np.asarray(legal["cf_actions"]) == expected_actions[None])
        and np.all(np.asarray(legal["cf_durations"]) == expected_durations[None])
        and all(np.array_equal(np.asarray(truth[key]), np.asarray(legal[key]))
                for key in ("cf_actions", "cf_durations", "factual_actions", "t_star",
                            "inference_mask"))
        and int(np.asarray(legal["t_star"]).min()) >= prot.T_STAR_LOW
        and int(np.asarray(legal["t_star"]).max()) <= prot.T_STAR_HIGH
        and np.array_equal(recomputed_mask, np.asarray(legal["inference_mask"], dtype=bool)))

    shapes_ok = (
        legal["object_history"].shape
            == (groups, prot.HISTORY_LEN, prot.OBJECT_CAPACITY, prot.N_OBJECT_FEATURES)
        and legal["object_factual_future"].shape
            == (groups, prot.HORIZON, prot.OBJECT_CAPACITY, prot.N_OBJECT_FEATURES)
        and truth["object_counterfactual_futures"].shape
            == (groups, prot.N_ACTIONS, prot.HORIZON, prot.OBJECT_CAPACITY,
                prot.N_OBJECT_FEATURES)
        and legal["inference_mask"].shape == (groups, prot.HORIZON, prot.OBJECT_CAPACITY))
    finite = all(np.isfinite(np.asarray(legal[key], dtype=np.float64)).all()
                 for key in ("object_history", "object_factual_future")) and \
        bool(np.isfinite(np.asarray(truth["object_counterfactual_futures"],
                                    dtype=np.float64)).all())
    legal_cells = int(np.asarray(legal["inference_mask"], dtype=bool)
                      .reshape(groups, -1).any(axis=1).sum())
    cond3 = bool(shapes_ok and finite and legal_cells == groups)

    cond4 = bool(int(receipt.get("attempts", prot.PILOT_MAX_CANDIDATE_SEEDS + 1))
                 <= prot.PILOT_MAX_CANDIDATE_SEEDS)

    return {
        "groups_complete": cond1,
        "entity_mask_action_integrity": cond2,
        "no_capacity_nan_early_termination": cond3,
        "candidate_pool_not_exhausted": cond4,
        "integrity_pass": bool(cond1 and cond2 and cond3 and cond4),
        "n_groups": int(groups),
        "attempts": int(receipt.get("attempts", -1)),
        "max_candidate_seeds": prot.PILOT_MAX_CANDIDATE_SEEDS,
    }


# --- D0 Gate: qualification + mechanical selection (plan §5.2/§5.3) -------------------------------


def evaluate_d0_profile(
    integrity: Mapping[str, Any],
    shift: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine the three §5.2 panels; ANY single failed condition -> unqualified."""
    integrity_conditions = (
        "groups_complete", "entity_mask_action_integrity",
        "no_capacity_nan_early_termination", "candidate_pool_not_exhausted")
    shift_conditions = ("smd_pass", "domain_auc_pass")
    task_conditions = (
        "high_regret_support", "oracle_diversity", "spearman_positive",
        "base_action_beats_references")
    failed: list[str] = []
    for panel, keys in ((integrity, integrity_conditions), (shift, shift_conditions),
                        (task, task_conditions)):
        for key in keys:
            if key not in panel:
                raise ValueError(f"panel is missing the §5.2 condition {key!r}")
            if not bool(panel[key]):
                failed.append(key)
    return {
        "qualified": not failed,
        "failed_conditions": failed,
        "integrity_pass": all(bool(integrity[k]) for k in integrity_conditions),
        "shift_pass": all(bool(shift[k]) for k in shift_conditions),
        "task_pass": all(bool(task[k]) for k in task_conditions),
    }


def select_ood_profile(profile_reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Plan §5.3: freeze the first qualified profile along JOINT > BEHAVIOR > GEOMETRY.

    Selection reads ONLY integrity / pre-intervention domain shift / task support — never
    any risk-model effect. With no qualified profile the result carries the
    pre-registered terminal state ``STAGE4D_OOD_TASK_INSUFFICIENT``.
    """
    missing = [p for p in ood.PROFILES if p not in profile_reports]
    if missing:
        raise ValueError(f"profile reports missing: {missing}; all three §4 profiles "
                         "must be evaluated independently before selection")
    qualified = {}
    for profile in ood.PROFILES:
        report = profile_reports[profile]
        if "qualified" not in report:
            raise ValueError(f"profile report for {profile} lacks 'qualified'")
        qualified[profile] = bool(report["qualified"])
    selected = next((p for p in ood.PROFILE_PRIORITY if qualified[p]), None)
    return {
        "plan_id": prot.PLAN_ID,
        "priority": list(ood.PROFILE_PRIORITY),
        "qualified": qualified,
        "selected_profile": selected,
        "status": prot.STATUS_D0_PROFILE_SELECTED if selected is not None
        else prot.STATUS_TASK_INSUFFICIENT,
        "selection_basis": (
            "integrity + pre-intervention domain shift + task support ONLY; "
            "no Direct/risk effect read (plan §5.3, Global Constraint 3)"),
    }


def write_profile_selection(path: str | Path, selection: Mapping[str, Any]) -> None:
    """Write the create-once ``ood_profile_selection.json``; a second write is refused."""
    payload = _json_safe(dict(selection))
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_once_json(Path(path), payload)


def write_profile_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write one pilot's create-once D0 profile report."""
    payload = _json_safe(dict(report))
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_once_json(Path(path), payload)


def selection_file_fingerprint(path: str | Path) -> str:
    """SHA256 of a written selection file (for downstream binding)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- pilot task-truth assembly (plan §5.1: pilot truth readable for these quantities) -------------


def assemble_pilot_task_quantities(
    legal_arrays: Mapping[str, np.ndarray],
    truth_arrays: Mapping[str, np.ndarray],
    e2e: np.ndarray,
) -> dict[str, np.ndarray]:
    """J_hat/J_true/regret/oracle for one pilot from legal inputs + pilot truth + the
    frozen decision-predictor ensemble's e2e effects (passed in; never a Direct output).

    The computation REUSES the frozen R2 primitives verbatim:
    ``stage4cp_features.compute_predicted_costs`` for the ensemble ``J_hat`` and
    ``stage4c_objective.trajectory_components``/``total_cost``/``clamp_regret`` for
    ``J_true``/regret, with the frozen (IDLE,3) reference and ten-action tie-break.
    """
    from ac_twin import stage4cp_features as s4cpf  # frozen R2 features, read-only

    history = np.asarray(legal_arrays["object_history"], dtype=np.float64)
    factual_xy = np.asarray(legal_arrays["object_factual_future"], dtype=np.float64)[
        ..., POSITION_SLICE]
    history_last_xy = history[:, -1, EGO_SLOT, POSITION_SLICE]
    mask = np.asarray(legal_arrays["inference_mask"], dtype=bool)
    groups = int(history.shape[0])

    e2e = np.asarray(e2e, dtype=np.float64)
    if e2e.shape != (len(prot.MODEL_SEEDS), groups, prot.N_ACTIONS, prot.HORIZON,
                     prot.OBJECT_CAPACITY, 2):
        raise ValueError(f"e2e must be (5,G,10,20,5,2); got {e2e.shape}")
    _, j_hat = s4cpf.compute_predicted_costs(e2e, factual_xy, history_last_xy, mask)

    true_xy = np.asarray(truth_arrays["object_counterfactual_futures"],
                         dtype=np.float64)[..., POSITION_SLICE]
    if true_xy.shape != (groups, prot.N_ACTIONS, prot.HORIZON,
                         prot.OBJECT_CAPACITY, 2):
        raise ValueError(f"truth CF futures must be (G,10,20,5,2); got {true_xy.shape}")
    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, prot.N_ACTIONS, 2))
    mask_b = np.broadcast_to(mask[:, None, :, :],
                             (groups, prot.N_ACTIONS) + mask.shape[1:])
    comp_true = s4co.trajectory_components(true_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))

    if s4co.ACTION_PLANS[s4co.IDLE_PLAN_INDEX] != (1, 3):
        raise RuntimeError("ALWAYS_IDLE reference must stay (IDLE,3), never (IDLE,1)")
    rows = np.arange(groups)
    a_hat = np.argmin(j_hat, axis=-1).astype(np.int64)
    a_star = np.argmin(j_true, axis=-1).astype(np.int64)
    j_true_star = j_true[rows, a_star]
    regret = s4co.clamp_regret(j_true[rows, a_hat] - j_true_star)
    idle = s4co.clamp_regret(j_true[:, s4co.IDLE_PLAN_INDEX] - j_true_star)
    random = s4co.clamp_regret(j_true.mean(axis=-1) - j_true_star)
    return {
        "J_hat": j_hat,
        "J_true": j_true,
        "a_hat": a_hat,
        "a_star": a_star,
        "regret": regret,
        "idle_regret": idle,
        "random_regret": random,
    }


# --- D1 confirmation power simulation (plan §6.1, verbatim) ------------------------------------
#
# Legal ONLY after D0 froze a unique profile (``D0_PROFILE_SELECTED``). The ONLY input is the
# create-once R2 development ``screen_group_quantities.npz`` (+ its receipt) — the power
# analysis never re-selects a model with it and never reads any Direct checkpoint, the sealed
# confirm truth, or the R2 train CF-future/regret fields.
#
# Frozen statistics (plan §6.1, verbatim):
# - the covariance comes DIRECTLY from this run's whole-group paired bootstrap draws,
#   recomputed deterministically with the frozen R2 screen entropy ``2026092702`` via
#   :func:`stage4cr2_statistics.paired_group_bootstrap`; a 95% SE is NEVER back-computed from
#   any historical CI width;
# - candidate accepted n = {1000, 2000, 3000, 4000}; the covariance scales by n_reference / n;
# - effect retention = {100%, 75%, 50%, 25%}: positive AND negative effects scale by the same
#   factor and signs are never flipped;
# - the ONLY blocking selection judge is the 50%-retention scenario; the minimal n whose joint
#   AND-chain pass rate reaches 0.80 is selected; 100%/75%/25% are reported side by side;
# - the simulated confirm gate is the FULL §6.1 AND-chain on each joint Monte-Carlo draw
#   (10,000 draws, entropy 2026104201), never a product of marginal powers;
# - n=4000 at 50% retention still insufficient -> ``STAGE4D_OOD_CONFIRMATION_UNDERPOWERED`` and
#   NO confirm resource (no protocol lock, no collection path) may be created.

Z_95 = 1.96  # confirm-side CI level

#: the §6.1 AND-chain quantity subset of the frozen R2 32-vector (stage4cr2_statistics layout),
#: in order: I_AURC, I_REGRET@80, PR-AUC-prev, five per-seed I_AURC, five per-seed I_REGRET@80.
D1_POWER_QUANTITIES = (
    s4cr2st.Q_I_AURC, s4cr2st.Q_I_REGRET80, s4cr2st.Q_PR,
) + s4cr2st.SEED_I_AURC_INDEX + s4cr2st.SEED_I_R80_INDEX
N_D1_POWER_QUANTITIES = len(D1_POWER_QUANTITIES)  # 13

#: local indices inside the power subset (lists: numpy fancy-indexing requires them).
P_I_AURC = 0
P_I_REGRET80 = 1
P_PR = 2
P_SEED_I_AURC = [3, 4, 5, 6, 7]
P_SEED_I_R80 = [8, 9, 10, 11, 12]

D1_POWER_QUANTITY_NAMES = tuple(
    s4cr2st._QUANTITY_NAMES[i] for i in D1_POWER_QUANTITIES)


def evaluate_d1_power_draws(
    draws: np.ndarray, standard_errors: np.ndarray
) -> dict[str, Any]:
    """Vectorised §6.1 AND-chain over joint Monte-Carlo draws of the power subset.

    ``draws`` is ``(replicates, 13)``; ``standard_errors`` is ``sqrt(diag(cov_n))`` for the
    same subset, taken DIRECTLY from the scaled draw covariance. The CI conditions use the
    confirm-side 95% lower bound (``estimate - 1.96*SE > 0``); the seed condition counts, per
    draw, the model seeds positive on I_AURC AND I_REGRET@80 SIMULTANEOUSLY (>= 4 of 5).
    """
    draws = np.asarray(draws, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != N_D1_POWER_QUANTITIES:
        raise ValueError(f"draws must align as (replicates, {N_D1_POWER_QUANTITIES})")
    if se.shape != (N_D1_POWER_QUANTITIES,):
        raise ValueError("standard_errors must align with the D1 power quantity subset")
    chain = prot.POWER_AND_CHAIN

    def ci95_lower(slot: int) -> np.ndarray:
        return draws[:, slot] - Z_95 * se[slot]

    conditions = {
        "i_aurc_regret": (draws[:, P_I_AURC] >= chain["i_aurc_min"])
        & (ci95_lower(P_I_AURC) > 0.0),
        "i_regret80": (draws[:, P_I_REGRET80] >= chain["i_regret80_min"])
        & (ci95_lower(P_I_REGRET80) > 0.0),
        "pr_auc_minus_prevalence": (draws[:, P_PR] > 0.0) & (ci95_lower(P_PR) > 0.0),
        "seeds_both_directions_positive": (
            ((draws[:, P_SEED_I_AURC] > 0.0) & (draws[:, P_SEED_I_R80] > 0.0)).sum(axis=1)
            >= chain["min_positive_seeds"]),
    }
    passed = np.logical_and.reduce(list(conditions.values()))
    return {"passed": passed, "conditions": conditions}


def simulate_d1_joint_power(
    theta_hat: np.ndarray,
    covariance: np.ndarray,
    *,
    candidate_ns,
    n_reference: int,
    retention_factors,
    monte_carlo: int,
    seed: int,
    selection_retention: float,
    target: float,
) -> dict[str, Any]:
    """Smallest n whose 50%-retention joint AND-chain power reaches the target.

    Covariance scales by ``n_reference / n``; effects scale to ``retention * theta`` with
    signs preserved (never flipped). ONLY the 50%-retention scenario selects; 100%/75%/25%
    are reported side by side as sensitivity. The gate is the FULL AND-chain on each joint
    draw from ONE generator (seeded directly with the reserved power entropy), so all
    n x retention cells share one deterministic stream.
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if theta_hat.shape != (N_D1_POWER_QUANTITIES,) \
            or covariance.shape != (N_D1_POWER_QUANTITIES, N_D1_POWER_QUANTITIES):
        raise ValueError("theta/covariance must match the D1 power quantity subset")
    if not candidate_ns:
        raise ValueError("candidate_ns must be non-empty")
    generator = np.random.default_rng(int(seed))
    max_n = int(max(candidate_ns))

    per_n: dict[int, Any] = {}
    selected_n: int | None = None
    for n in (int(v) for v in candidate_ns):
        cov_n = covariance * (float(n_reference) / n)
        se_n = np.sqrt(np.diag(cov_n))
        scenarios: dict[str, Any] = {}
        for retention in (float(r) for r in retention_factors):
            mean = float(retention) * theta_hat  # equal scaling; signs never flip
            draws = generator.multivariate_normal(mean, cov_n, size=int(monte_carlo))
            evaluation = evaluate_d1_power_draws(draws, se_n)
            scenarios[f"{retention:.2f}"] = {
                "joint_power": float(evaluation["passed"].mean()),
                "marginal_power": {
                    k: float(v.mean()) for k, v in evaluation["conditions"].items()},
            }
        main = scenarios[f"{float(selection_retention):.2f}"]
        binding = min(main["marginal_power"], key=main["marginal_power"].get)
        per_n[n] = {
            "scenarios": scenarios,
            "joint_power_at_selection": main["joint_power"],
            "binding_condition": binding,
            "binding_power": main["marginal_power"][binding],
        }
        if selected_n is None and main["joint_power"] >= float(target):
            selected_n = n

    return {
        "selected_n": selected_n,
        "underpowered": selected_n is None,
        "per_n": per_n,
        "n_reference": int(n_reference),
        "selection_retention": float(selection_retention),
        "target": float(target),
        "max_candidate_n": max_n,
        "sensitivity_at_max_n": {
            scenario: per_n[max_n]["scenarios"][scenario]["joint_power"]
            for scenario in (f"{float(r):.2f}" for r in retention_factors)
        },
    }


def _load_d1_power_inputs(npz_path: str | Path) -> dict[str, Any]:
    """Load the create-once R2 screen quantities NPZ + receipt; power reads ONLY these."""
    npz_path = Path(npz_path)
    receipt_path = npz_path.with_name("screen_group_quantities_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"screen quantities receipt missing (create-once pair): {receipt_path}")
    quantities = s4cr2st.load_screen_group_quantities(npz_path, receipt_path)
    return {
        "quantities": quantities,
        "npz_sha256": sha256_file(npz_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def _d1_bootstrap_power_inputs(
    quantities: Mapping[str, Any], *, seed: int, n_resamples: int
) -> dict[str, Any]:
    """Reproduce THIS RUN's whole-group paired bootstrap; take theta + covariance from draws.

    The draw stream is deterministic in (quantities, seed, n_resamples), reproducing the
    R2 screen adjudication's own bootstrap (frozen entropy ``2026092702``). The covariance
    is taken directly from the draws — never from any CI width.
    """
    bootstrap = s4cr2st.paired_group_bootstrap(
        quantities, seed=int(seed), n_resamples=int(n_resamples))
    draws = np.asarray(bootstrap.draws, dtype=np.float64)[:, list(D1_POWER_QUANTITIES)]
    theta_hat = np.asarray(bootstrap.point, dtype=np.float64)[list(D1_POWER_QUANTITIES)]
    if not np.isfinite(theta_hat).all():
        raise ValueError("non-finite identity quantity in the D1 power subset")
    return {
        "theta_hat": theta_hat,
        "covariance": s4cr2pw.symmetrised_psd_covariance(draws),
        "n_reference": int(np.asarray(quantities["regret"]).size),
        "bootstrap": bootstrap,
    }


def simulate_d1_power(
    group_quantities_path: str | Path,
    candidate_ns,
    retention_factors,
    seed: int,
    *,
    n_resamples: int = s4cr2.BOOTSTRAP_RESAMPLES,
    monte_carlo: int = prot.POWER_MONTE_CARLO,
    target: float = prot.POWER_TARGET,
    selection_retention: float = prot.POWER_SELECTION_RETENTION,
) -> dict[str, Any]:
    """Plan §6.1: conditional joint power of the D1 confirm AND-chain at each candidate n.

    Reads ONLY the create-once R2 screen quantities NPZ + receipt; recomputes the paired
    bootstrap deterministically for the covariance; simulates the FULL §6.1 AND-chain over
    ``monte_carlo`` joint draws (entropy ``seed`` = 2026104201); selects the minimum n whose
    50%-retention joint pass rate reaches ``target``. NEVER creates or consumes any confirm
    resource and never re-selects a model with the screen quantities.
    """
    inputs = _load_d1_power_inputs(group_quantities_path)
    power_inputs = _d1_bootstrap_power_inputs(
        inputs["quantities"], seed=s4cr2.BOOTSTRAP_ENTROPY, n_resamples=n_resamples)
    result = simulate_d1_joint_power(
        power_inputs["theta_hat"], power_inputs["covariance"],
        candidate_ns=tuple(int(n) for n in candidate_ns),
        n_reference=power_inputs["n_reference"],
        retention_factors=tuple(float(r) for r in retention_factors),
        monte_carlo=int(monte_carlo), seed=int(seed),
        selection_retention=float(selection_retention), target=float(target))
    theta_named = {name: float(power_inputs["theta_hat"][i])
                   for i, name in enumerate(D1_POWER_QUANTITY_NAMES)}
    se_named = {name: float(np.sqrt(power_inputs["covariance"][i, i]))
                for i, name in enumerate(D1_POWER_QUANTITY_NAMES)}
    underpowered = bool(result["underpowered"])
    return {
        "plan_id": prot.PLAN_ID,
        "status": (prot.STATUS_UNDERPOWERED if underpowered
                   else prot.STATUS_D1_POWER_FROZEN),
        "selected_n": result["selected_n"],
        "underpowered": underpowered,
        "target": float(target),
        "selection_retention": float(selection_retention),
        "n_reference": int(power_inputs["n_reference"]),
        "candidate_ns": [int(n) for n in candidate_ns],
        "retention_factors": [float(r) for r in retention_factors],
        "per_n": result["per_n"],
        "theta_hat": theta_named,
        "se_reference": se_named,
        "screen_quantities_sha256": inputs["npz_sha256"],
        "screen_quantities_receipt_sha256": inputs["receipt_sha256"],
        "bootstrap": {
            "seed": int(s4cr2.BOOTSTRAP_ENTROPY),
            "n_resamples": int(n_resamples),
            "confidence": float(s4cr2.CONFIDENCE_SCREEN),
            "design": ("covariance taken DIRECTLY from this run's whole-group paired "
                       "bootstrap draws (deterministically recomputed with the frozen R2 "
                       "screen entropy); a 95% SE is never derived from any CI width"),
        },
        "monte_carlo": {"entropy": int(seed), "n_draws": int(monte_carlo)},
        "max_candidate_n": result["max_candidate_n"],
        "sensitivity_at_max_n": result["sensitivity_at_max_n"],
    }


# ============================================================================================
# Task 6 (plan §6.4, §7, §8, §13): the ONE-SHOT D1 unseal + adjudication.
#
# The statistical unit is the accepted complete GROUP (2000). ONE whole-group paired
# bootstrap runs 10,000 resamples with entropy 2026110995; EVERY metric of a draw is
# recomputed from the SAME resample index vector and the two-sided 95% CI is read off the
# (2.5%, 97.5%) resample quantiles. Metric definitions are REUSED bitwise from
# ``stage4cp_statistics`` (AURC_REGRET, REGRET@80 via the frozen coverage_index,
# PR-AUC-prevalence, fail-closed relative improvement, Spearman); nothing is redefined.
#
# One-shot order (plan §6.4, enforced by IO): ``burn_confirmation_lock_d1`` (create-once;
# binds the truth hash FROM the collection receipt and NEVER reads the truth bytes) ->
# ``load_d1_confirmation_truth`` (judge-only, after the burn) -> ``assemble_d1_quantities``
# (create-once §13 per-group quantities receipt) -> ``adjudicate_d1`` (exactly once). A
# second adjudication with the formal report present is refused. Baseline denominators are
# fail-closed: a non-positive or non-finite primary-baseline AURC_REGRET / REGRET@80 yields
# NaN improvements and the gate returns ``INCONCLUSIVE_STAGE4D_OOD_INVALID`` — never an
# epsilon, an absolute difference or a baseline swap. The main baseline is SEED_VOTE_ENTROPY
# and ONLY it enters the Gate; MARGIN_ONLY / SEED_MARGIN_STD comparisons ride the SAME
# paired draws to Table_OOD_2 only.
#
# Circular-import note: ``stage4d_confirm_protocol`` already imports this module, so the
# score-side modules (``stage4d_confirm_protocol`` / ``stage4d_ood_inference`` /
# ``collect_stage4d_merge_ood`` / ``stage4cr2_confirm_inference``) are imported LAZILY inside
# the functions that need them, never at module top.

#: frozen quantity-vector layout (N_D1_QUANTITIES = 49). Mirrors the Stage 4C-R2 confirm
#: vector minus the R2-only trim/spearman-margin slots (not §7 gate conditions here).
Q_I_AURC = 0
Q_I_REGRET80 = 1
Q_PR = 2
Q_SPEARMAN_BRANCH = 3
Q_BASE_VS_RANDOM = 4
Q_TSTAR_LOW = 5
Q_TSTAR_HIGH = 6
Q_INTERACT_LOW = 7
Q_INTERACT_HIGH = 8
Q_AURC_BASELINE = 9
Q_R80_BASELINE = 10
Q_BASE_MEAN_REGRET = 11
Q_IDLE_MEAN_REGRET = 12
Q_RANDOM_MEAN_REGRET = 13
Q_HIGH_REGRET_FRAC = 14
Q_SEED_I_AURC = 15   # 15..19, in the frozen MODEL_SEEDS order
Q_SEED_I_R80 = 20    # 20..24
Q_MU_I_AURC = 25
Q_MU_I_R80 = 26
Q_CLS_I_AURC = 27
Q_CLS_I_R80 = 28
Q_MARGIN_I_AURC = 29   # Table_OOD_2 only — NEVER gate-relevant
Q_MARGIN_I_R80 = 30
Q_STD_I_AURC = 31      # Table_OOD_2 only
Q_STD_I_R80 = 32
Q_AURC_MODEL = 33
Q_R80_MODEL = 34
Q_AURC_MARGIN = 35
Q_R80_MARGIN = 36
Q_AURC_STD = 37
Q_R80_STD = 38
Q_AURC_COMBINED = 39
Q_R80_COMBINED = 40
Q_AURC_RANDOM = 41
Q_R80_RANDOM = 42
Q_AP_MODEL = 43
Q_AP_MARGIN = 44
Q_AP_SVE = 45
Q_AP_STD = 46
Q_AP_COMBINED = 47
Q_AP_RANDOM = 48
N_D1_QUANTITIES = 49
N_D1_SEEDS = len(prot.MODEL_SEEDS)

D1_STRATA_INDEX = (Q_TSTAR_LOW, Q_TSTAR_HIGH, Q_INTERACT_LOW, Q_INTERACT_HIGH)
D1_SEED_I_AURC_INDEX = tuple(range(Q_SEED_I_AURC, Q_SEED_I_AURC + N_D1_SEEDS))
D1_SEED_I_R80_INDEX = tuple(range(Q_SEED_I_R80, Q_SEED_I_R80 + N_D1_SEEDS))

_D1_QUANTITY_NAMES = (
    "i_aurc_regret", "i_regret80", "pr_auc_minus_prevalence", "spearman_branch",
    "base_vs_random_improvement", "tstar_low_i_aurc", "tstar_high_i_aurc",
    "interaction_low_i_aurc", "interaction_high_i_aurc", "aurc_baseline",
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
coverage_index = s4cs.coverage_index

COVERAGE = prot.COVERAGE            # 0.80, frozen
TSTAR_SPLIT = s4cr2st.TSTAR_SPLIT   # 13, frozen fixed split (never refit)

STATUS_D1_TASK_QUALIFIED = "TASK_OOD_QUALIFIED"

#: the ONLY baseline that may enter the Gate (validation-frozen; never re-selected).
D1_PRIMARY_BASELINE = prot.PRIMARY_BASELINE
D1_ALL_BASELINE_NAMES = tuple(prot.STANDARD_BASELINES) + tuple(prot.REFERENCE_BASELINES)

#: sealed-truth schema — MIRRORS collect_stage4d_merge_ood.TRUTH_FIELDS (the collector
#: pulls highway_env, so it is never imported here); the collection receipt's
#: ``truth_fields`` is cross-checked against this mirror at read time (fail-loud drift).
D1_TRUTH_FIELDS = (
    "object_counterfactual_futures",  # (G,10,20,5,8) float32 — the sealed CF truth
    "cf_actions",                     # (G,10) int64 — CF mechanism
    "cf_durations",                   # (G,10) int64 — CF mechanism
    "factual_actions",                # (G,20) int64 — integrity copy
    "t_star",                         # (G,)   int64 — integrity copy
    "group_seed",                     # (G,)   int64 — same-order binding
    "inference_mask",                 # (G,20,5) bool — integrity copy
)

#: per-group arrays the quantities mapping must carry (all aligned on the group axis).
D1_REQUIRED_ARRAY_KEYS = (
    "group_seed", "regret", "y_high", "random_regret", "idle_regret",
    "ensemble_risk", "risk_by_seed", "mu_only_ensemble", "class_only_ensemble",
    "tstar_layer", "interaction_layer", "J_hat", "J_true", "a_hat", "a_star",
    "t_star", "interaction_min_distance",
)

#: baseline risk arrays persisted in the quantities NPZ (prefix stripped on load).
_D1_BASELINE_NPZ_PREFIX = "baseline_risk__"

#: QA flags evaluated inside the gates (plan §7.1 condition 1 / §7.2 condition 5).
D1_QA_KEYS = (
    "all_complete", "hashes_match", "role_seeds_valid", "determinism_passed",
    "one_shot_locks", "ensemble_is_arithmetic_mean",
)

#: exact §8.2 paper-boundary sentences (verbatim; the success sentence is legal ONLY when
#: the terminal status is STAGE4D_OOD_ROBUSTNESS_CONFIRMED).
D1_SUCCESS_SENTENCE = "冻结 decision-risk 方法在预注册 Merge 生成机失配下仍获支持。"
D1_PROHIBITED_CLAIMS = (
    "禁止由此声称：跨平台外部有效性、真实道路泛化、线上安全提升、闭环策略已改善或"
    "对任意 OOD 鲁棒。")


class D1ResourceAtRiskError(RuntimeError):
    """A post-burn transaction cannot be proven consistent (plan §13).

    Raised instead of overwriting or re-deriving anything once the confirmation lock is
    burned; the only legal post-burn recovery is continuing the report from the SAME
    frozen per-group quantities.
    """

    status = prot.STATUS_INVALID


# --- result containers ---------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResultD1:
    """One whole-group paired bootstrap: identity point + 95% CI + raw draws."""
    point: np.ndarray        # (N_D1_QUANTITIES,)
    ci_lower: np.ndarray     # (N_D1_QUANTITIES,) two-sided 95% (2.5% quantile)
    ci_upper: np.ndarray     # (N_D1_QUANTITIES,) two-sided 95% (97.5% quantile)
    draws: np.ndarray        # (n_resamples, N_D1_QUANTITIES), shared indices per row
    n_resamples: int
    seed: int
    confidence: float


@dataclass(frozen=True)
class OODDecision:
    """The single D1 adjudication (Gate TASK-OOD + Gate OOD-RISK)."""
    plan_id: str
    n_groups: int
    status: str
    passed: bool
    task: dict[str, Any]
    risk: dict[str, Any]
    metrics: dict[str, Any]
    ablations: dict[str, Any]
    bootstrap: dict[str, Any]
    domain_shift: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "n_groups": int(self.n_groups),
            "status": self.status,
            "passed": bool(self.passed),
            "gate_task_ood": self.task,
            "gate_ood_risk": self.risk,
            "metrics": self.metrics,
            "ablations": self.ablations,
            "bootstrap": self.bootstrap,
            "domain_shift": self.domain_shift,
        }


# --- path contract ---------------------------------------------------------------------------


def d1_confirm_artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve every one-shot D1 confirm path (plan §8.1)."""
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
        "inference_receipt": _resolve(paths["inference_predictions"]).parent
            / "inference_receipt.json",
        "determinism_receipt": _resolve(paths["determinism_receipt"]),
        "score_freeze_receipt": _resolve(paths["score_freeze_receipt"]),
        "lock": _resolve(paths["confirmation_used_lock"]),
        "quantities_npz": report_dir / "d1_group_quantities.npz",
        "quantities_receipt": report_dir / "d1_group_quantities_receipt.json",
        "report": _resolve(paths["confirmatory_report"]),
        "decision": _resolve(paths["confirmatory_decision"]),
        "table_ood_1": report_dir / "Table_OOD_1_gate.csv",
        "table_ood_2": report_dir / "Table_OOD_2_baselines.csv",
        "table_ood_3": report_dir / "Table_OOD_3_seed_strata.csv",
        "fig_ood_1": report_dir / "Fig_OOD_1_risk_coverage.png",
        "fig_ood_2": report_dir / "Fig_OOD_2_domain_shift.png",
    }


# --- quantities validation ---------------------------------------------------------------------


def _validate_d1_quantities(quantities: Mapping[str, Any]) -> int:
    """Fail closed on any missing/misaligned/non-finite per-group quantity."""
    for key in D1_REQUIRED_ARRAY_KEYS:
        if key not in quantities:
            raise ValueError(f"D1 quantities missing array {key!r}")
    regret = np.asarray(quantities["regret"], dtype=np.float64).ravel()
    groups = int(regret.size)
    if groups < 2:
        raise ValueError("D1 quantities need at least 2 groups")
    if int(quantities.get("n_groups", groups)) != groups:
        raise ValueError("n_groups does not match the regret axis")
    for key in ("group_seed", "y_high", "random_regret", "idle_regret", "ensemble_risk",
                "mu_only_ensemble", "class_only_ensemble", "tstar_layer",
                "interaction_layer", "a_hat", "a_star", "t_star",
                "interaction_min_distance"):
        arr = np.asarray(quantities[key]).ravel()
        if arr.shape != (groups,):
            raise ValueError(f"quantity {key!r} must align as ({groups},)")
    for key in ("regret", "random_regret", "idle_regret", "ensemble_risk",
                "mu_only_ensemble", "class_only_ensemble", "J_hat", "J_true",
                "interaction_min_distance"):
        if not np.isfinite(np.asarray(quantities[key], dtype=np.float64)).all():
            raise ValueError(f"quantity {key!r} must be finite")
    by_seed = np.asarray(quantities["risk_by_seed"], dtype=np.float64)
    if by_seed.shape != (N_D1_SEEDS, groups) or not np.isfinite(by_seed).all():
        raise ValueError(
            f"risk_by_seed must be finite and align as ({N_D1_SEEDS}, {groups})")
    j_hat = np.asarray(quantities["J_hat"], dtype=np.float64)
    j_true = np.asarray(quantities["J_true"], dtype=np.float64)
    if j_hat.shape != (groups, prot.N_ACTIONS) or j_true.shape != j_hat.shape:
        raise ValueError("J_hat/J_true must align as (G, 10)")
    baselines = quantities.get("baseline_risks")
    if not isinstance(baselines, Mapping):
        raise ValueError("D1 quantities missing baseline_risks mapping")
    for name in D1_ALL_BASELINE_NAMES:
        if name not in baselines:
            raise ValueError(f"baseline_risks missing {name!r}")
        risk = np.asarray(baselines[name], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"baseline risk {name!r} must be finite and group-aligned")
    strongest = str(quantities.get("strongest_baseline", ""))
    if strongest != D1_PRIMARY_BASELINE:
        raise ValueError(
            f"the D1 confirm never re-selects its baseline: strongest_baseline must be "
            f"{D1_PRIMARY_BASELINE!r}, got {strongest!r}")
    qa = quantities.get("qa")
    if not isinstance(qa, Mapping) or any(key not in qa for key in D1_QA_KEYS):
        raise ValueError(f"D1 quantities qa must carry {list(D1_QA_KEYS)}")
    return groups


# --- decision quantities (plan §3.1, verbatim) -------------------------------------------------


def _d1_decision_quantities(j_hat: np.ndarray, j_true: np.ndarray) -> dict[str, np.ndarray]:
    """a_hat/a_star/regret/random/idle/y_high from the two consequence matrices.

    Tie-breaks are the fixed ACTION_PLANS order (np.argmin first occurrence; never
    regret-assisted). The ALWAYS_IDLE reference is ACTION_PLANS index 3 == (IDLE,3) —
    asserted, never (IDLE,1). RANDOM_ACTION is the mean over ALL TEN actions of the
    group. Regret uses the frozen clamp.
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


def _d1_average_precision(risk: np.ndarray, label: np.ndarray) -> float:
    """Absolute average precision; a degenerate label yields the prevalence (PR == 0)."""
    label = np.asarray(label, dtype=np.int64)
    prevalence = float(label.mean())
    if label.min() == label.max():
        return prevalence
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(label, np.asarray(risk, dtype=np.float64)))


# --- the joint quantity vector ------------------------------------------------------------------


def compute_d1_quantity_vector(
    index: np.ndarray, quantities: Mapping[str, Any]
) -> np.ndarray:
    """Evaluate all 49 quantities on ONE whole-group resample ``index``.

    ``index`` is a with-replacement integer resample into the D1 groups, so every metric of
    the draw — primary endpoints, marginal strata, five seeds, ablations AND the
    Table_OOD_2-only MARGIN_ONLY / SEED_MARGIN_STD comparisons — stays coupled through the
    same group multiplicities. Improvement quantities are NaN when the frozen
    primary-baseline denominator is non-positive (the gate turns that into
    ``INCONCLUSIVE_STAGE4D_OOD_INVALID``); a stratum subset with fewer than two resampled
    members yields NaN (descriptive only here — never a gate condition in this stage). The
    efficiency normalization and action set are NEVER re-derived inside a draw: all ten
    actions per group enter through the pre-computed per-group quantities.
    """
    index = np.asarray(index, dtype=np.int64)
    reg = np.asarray(quantities["regret"], dtype=np.float64)[index]
    y_high = np.asarray(quantities["y_high"], dtype=np.int64)[index]
    ens = np.asarray(quantities["ensemble_risk"], dtype=np.float64)[index]
    baselines = quantities["baseline_risks"]
    base = np.asarray(baselines[D1_PRIMARY_BASELINE], dtype=np.float64)[index]
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

    q = np.full(N_D1_QUANTITIES, np.nan, dtype=np.float64)
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

    for slot, layer_key in ((Q_TSTAR_LOW, "tstar_layer"),
                            (Q_INTERACT_LOW, "interaction_layer")):
        layer = np.asarray(quantities[layer_key], dtype=np.int64)[index]
        for level in (0, 1):
            mask = layer == level
            if mask.sum() < 2:
                continue  # stays NaN — descriptive stratum only, never gate-relevant here
            q[slot + level] = s4cs.improvement(
                s4cs.aurc_regret(np.argsort(base[mask], kind="stable"), reg[mask]),
                s4cs.aurc_regret(np.argsort(ens[mask], kind="stable"), reg[mask]))

    q[Q_AURC_BASELINE] = a_base
    q[Q_R80_BASELINE] = r80_base
    q[Q_BASE_MEAN_REGRET] = base_mean
    q[Q_IDLE_MEAN_REGRET] = float(idle.mean())
    q[Q_RANDOM_MEAN_REGRET] = rand_mean
    q[Q_HIGH_REGRET_FRAC] = float((reg > prot.HIGH_REGRET_THRESHOLD).mean())

    for i in range(N_D1_SEEDS):
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

    # Table_OOD_2-only comparisons on the SAME draws; never gate-relevant, never a swap.
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

    q[Q_AP_MODEL] = _d1_average_precision(ens, y_high)
    q[Q_AP_MARGIN] = _d1_average_precision(margin, y_high)
    q[Q_AP_SVE] = _d1_average_precision(base, y_high)
    q[Q_AP_STD] = _d1_average_precision(std, y_high)
    q[Q_AP_COMBINED] = _d1_average_precision(combined, y_high)
    q[Q_AP_RANDOM] = _d1_average_precision(random_risk, y_high)
    return q


# --- the one paired bootstrap -------------------------------------------------------------------


def paired_group_bootstrap_d1(
    quantities: Mapping[str, np.ndarray],
    *,
    seed: int = prot.D1_BOOTSTRAP_ENTROPY,
    n_resamples: int = prot.BOOTSTRAP_RESAMPLES,
) -> BootstrapResultD1:
    """10,000-draw whole-group paired bootstrap with shared indices and 95% CIs.

    EVERY quantity of a draw is recomputed from the SAME whole-group resample index
    vector; the identity evaluation is the point estimate; the two-sided 95% CI is the
    (0.025, 0.975) resample quantile band. The confidence level is frozen at
    ``prot.CONFIDENCE`` (0.95) — never re-parameterised and never the pilot 80% CI.
    """
    groups = _validate_d1_quantities(quantities)
    generator = np.random.default_rng(int(seed))
    identity = np.arange(groups, dtype=np.int64)
    point = compute_d1_quantity_vector(identity, quantities)
    draws = np.empty((int(n_resamples), N_D1_QUANTITIES), dtype=np.float64)
    for draw in range(int(n_resamples)):
        index = generator.integers(0, groups, size=groups)
        draws[draw] = compute_d1_quantity_vector(index, quantities)
    alpha = (1.0 - prot.CONFIDENCE) / 2.0
    return BootstrapResultD1(
        point=point,
        ci_lower=np.quantile(draws, alpha, axis=0),
        ci_upper=np.quantile(draws, 1.0 - alpha, axis=0),
        draws=draws,
        n_resamples=int(n_resamples),
        seed=int(seed),
        confidence=float(prot.CONFIDENCE),
    )


# --- the gates (plan §7.1 Gate TASK-OOD, §7.2 Gate OOD-RISK, §7.3 terminal states) ---------------


def _d1_named(vector: np.ndarray) -> dict[str, float]:
    return {name: float(vector[i]) for i, name in enumerate(_D1_QUANTITY_NAMES)}


def evaluate_gate_task_ood(
    quantities: Mapping[str, Any],
    bootstrap: BootstrapResultD1,
    metadata: Mapping[str, Any],
    confidence: float = prot.CONFIDENCE,
) -> dict[str, Any]:
    """Gate TASK-OOD (plan §7.1, all 8 conditions; §9.2 interface).

    ``quantities`` is the distilled create-once per-group receipt (the truth/predictions
    data flow, assembled once per §6.4/§13); ``metadata`` carries ``selected_n``, the
    ``domain_shift`` panel (condition 2) and the locked ``strata``; ``confidence`` is the
    frozen 0.95 CI level. The gate task panel is computed for every non-INVALID outcome; a
    QA/completeness failure is surfaced as INVALID by the caller, not adjudicated here.
    """
    groups = _validate_d1_quantities(quantities)
    if float(confidence) != float(prot.CONFIDENCE):
        raise ValueError("the D1 confirm CI level is frozen at 0.95 (never the pilot 0.80)")
    point = np.asarray(bootstrap.point, dtype=np.float64)
    ci_lower = np.asarray(bootstrap.ci_lower, dtype=np.float64)
    selected_n = int(metadata["selected_n"])
    domain_shift = metadata["domain_shift"]
    gate = prot.GATE_TASK_OOD

    qa = {key: bool(quantities["qa"][key]) for key in D1_QA_KEYS}
    qa_passed = all(qa.values())

    oracle = np.asarray(quantities["a_star"], dtype=np.int64)
    oracle_shares = [float((oracle == a).mean()) for a in range(prot.N_ACTIONS)]
    n_oracle = int(sum(s >= gate["min_oracle_frac"] for s in oracle_shares))
    tstar = np.asarray(quantities["tstar_layer"], dtype=np.int64)
    inter = np.asarray(quantities["interaction_layer"], dtype=np.int64)
    stratum_fracs = {
        "tstar_low": float((tstar == 0).mean()), "tstar_high": float((tstar == 1).mean()),
        "interaction_low": float((inter == 0).mean()),
        "interaction_high": float((inter == 1).mean()),
    }
    stratum_counts = {key: int(round(value * groups))
                      for key, value in stratum_fracs.items()}

    conditions = {
        "c1_complete_gate_q": bool(qa_passed and groups == selected_n),
        "c2_domain_shift": bool(domain_shift.get("smd_pass")
                                and domain_shift.get("domain_auc_pass")),
        "c3_high_regret_frac": bool(
            point[Q_HIGH_REGRET_FRAC] >= gate["min_high_regret_frac"]),
        "c4_oracle_diversity": bool(n_oracle >= gate["min_distinct_oracle"]),
        "c5_branch_spearman": bool(ci_lower[Q_SPEARMAN_BRANCH] > 0.0),
        "c6_base_beats_random": bool(point[Q_BASE_VS_RANDOM] > 0.0
                                     and ci_lower[Q_BASE_VS_RANDOM] > 0.0),
        "c7_base_not_worse_than_idle": bool(
            np.isfinite([point[Q_BASE_MEAN_REGRET], point[Q_IDLE_MEAN_REGRET]]).all()
            and point[Q_BASE_MEAN_REGRET]
            <= point[Q_IDLE_MEAN_REGRET] + prot.BASELINE_TIE_TOLERANCE),
        "c8_strata_min_10pct": bool(
            min(stratum_fracs.values()) >= gate["min_stratum_frac"]),
    }
    # §7.1: conditions 3/4/8 -> TASK_DEGENERATE; conditions 2/5/6/7 -> MODEL_INSUFFICIENT.
    degenerate = not (conditions["c3_high_regret_frac"]
                      and conditions["c4_oracle_diversity"]
                      and conditions["c8_strata_min_10pct"])
    insufficient = not (conditions["c2_domain_shift"]
                        and conditions["c5_branch_spearman"]
                        and conditions["c6_base_beats_random"]
                        and conditions["c7_base_not_worse_than_idle"])
    if degenerate:
        task_status = prot.STATUS_TASK_DEGENERATE
    elif insufficient:
        task_status = prot.STATUS_MODEL_INSUFFICIENT
    else:
        task_status = STATUS_D1_TASK_QUALIFIED
    return {
        "conditions": conditions,
        "status": task_status,
        "degenerate": bool(degenerate),
        "insufficient": bool(insufficient),
        "qa": qa,
        "qa_passed": bool(qa_passed),
        "high_regret_frac": float(point[Q_HIGH_REGRET_FRAC]),
        "oracle_action_shares": oracle_shares,
        "oracle_actions_covered_ge_min_frac": n_oracle,
        "stratum_fracs": stratum_fracs,
        "stratum_counts": stratum_counts,
    }


def evaluate_gate_ood_risk(
    quantities: Mapping[str, Any],
    bootstrap: BootstrapResultD1,
    metadata: Mapping[str, Any],
    confidence: float = prot.CONFIDENCE,
) -> dict[str, Any]:
    """Gate OOD-RISK (plan §7.2, all 5 conditions; §9.2 interface).

    Judged ONLY after Gate TASK-OOD fully passes (the caller enforces the order). The
    primary baseline is SEED_VOTE_ENTROPY and ONLY it enters the Gate. Denominators are
    fail-closed (a non-positive/non-finite baseline AURC/REGRET@80 fails validity, never an
    epsilon/absolute-difference/baseline-swap).
    """
    _validate_d1_quantities(quantities)
    if float(confidence) != float(prot.CONFIDENCE):
        raise ValueError("the D1 confirm CI level is frozen at 0.95 (never the pilot 0.80)")
    point = np.asarray(bootstrap.point, dtype=np.float64)
    ci_lower = np.asarray(bootstrap.ci_lower, dtype=np.float64)
    gate = prot.GATE_OOD_RISK

    qa_passed = all(bool(quantities["qa"][key]) for key in D1_QA_KEYS)
    baseline_valid = bool(
        np.isfinite([point[Q_AURC_BASELINE], point[Q_R80_BASELINE]]).all()
        and point[Q_AURC_BASELINE] > 0.0 and point[Q_R80_BASELINE] > 0.0)
    seeds_both = int(sum(
        point[D1_SEED_I_AURC_INDEX[i]] > 0.0 and point[D1_SEED_I_R80_INDEX[i]] > 0.0
        for i in range(N_D1_SEEDS)))
    conditions = {
        "r1_i_aurc_regret": bool(point[Q_I_AURC] >= gate["i_aurc_min"]
                                 and ci_lower[Q_I_AURC] > 0.0),
        "r2_i_regret80": bool(point[Q_I_REGRET80] >= gate["i_regret80_min"]
                              and ci_lower[Q_I_REGRET80] > 0.0),
        "r3_pr_auc_minus_prevalence": bool(point[Q_PR] > 0.0 and ci_lower[Q_PR] > 0.0),
        "r4_seed_consistency": bool(seeds_both >= gate["min_positive_seed_metrics"]),
        "r5_validity": bool(baseline_valid and qa_passed),
    }
    return {
        "conditions": conditions,
        "baseline_valid": baseline_valid,
        "baseline_denominators": {
            "aurc_regret": float(point[Q_AURC_BASELINE]),
            "regret_at_80": float(point[Q_R80_BASELINE]),
        },
        "seeds_both_directions_positive": seeds_both,
        "min_positive_seed_metrics": int(gate["min_positive_seed_metrics"]),
    }


def evaluate_d1_gates(
    quantities: Mapping[str, Any],
    bootstrap: BootstrapResultD1,
    *,
    selected_n: int,
    domain_shift: Mapping[str, Any],
) -> OODDecision:
    """Adjudicate Gate TASK-OOD then Gate OOD-RISK exactly once (plan §7, §7.3).

    Terminal-state priority (§7.3, literal order): QA/completeness (condition 1) failure ->
    ``INCONCLUSIVE_STAGE4D_OOD_INVALID``; TASK conditions 3/4/8 failure ->
    ``STAGE4D_OOD_TASK_DEGENERATE``; TASK conditions 2/5/6/7 failure ->
    ``INCONCLUSIVE_STAGE4D_OOD_DECISION_MODEL_INSUFFICIENT``; a non-positive/non-finite
    primary-baseline denominator -> ``INCONCLUSIVE_STAGE4D_OOD_INVALID``; any unblocked risk
    condition failure -> ``STAGE4D_OOD_ROBUSTNESS_NOT_CONFIRMED``; all pass ->
    ``STAGE4D_OOD_ROBUSTNESS_CONFIRMED``. A qualification failure NEVER claims the risk
    method is falsified (§7.1 note).
    """
    groups = _validate_d1_quantities(quantities)
    point = np.asarray(bootstrap.point, dtype=np.float64)
    ci_lower = np.asarray(bootstrap.ci_lower, dtype=np.float64)
    ci_upper = np.asarray(bootstrap.ci_upper, dtype=np.float64)
    draws = np.asarray(bootstrap.draws, dtype=np.float64)
    if point.shape != (N_D1_QUANTITIES,) or ci_lower.shape != (N_D1_QUANTITIES,) \
            or ci_upper.shape != (N_D1_QUANTITIES,) \
            or draws.shape != (int(bootstrap.n_resamples), N_D1_QUANTITIES):
        raise ValueError("bootstrap result does not match the frozen D1 quantity vector")
    if float(bootstrap.confidence) != float(prot.CONFIDENCE):
        raise ValueError("the D1 confirm CI level is frozen at 0.95 (never the pilot 0.80)")

    metadata = {"selected_n": int(selected_n), "domain_shift": domain_shift}
    task = evaluate_gate_task_ood(quantities, bootstrap, metadata,
                                  confidence=bootstrap.confidence)

    # --- terminal-state priority (§7.3, literal order) -------------------------------------
    qa_ok = task["qa_passed"]
    if not qa_ok or not task["conditions"]["c1_complete_gate_q"]:
        status = prot.STATUS_INVALID
    elif task["degenerate"]:
        status = prot.STATUS_TASK_DEGENERATE
    elif task["insufficient"]:
        status = prot.STATUS_MODEL_INSUFFICIENT
    else:
        # Gate TASK-OOD fully passed -> judge Gate OOD-RISK (§7.2) on the same draws.
        risk = evaluate_gate_ood_risk(quantities, bootstrap, metadata,
                                      confidence=bootstrap.confidence)
        if not risk["baseline_valid"]:
            status = prot.STATUS_INVALID
        elif all(risk["conditions"].values()):
            status = prot.STATUS_CONFIRMED
        else:
            status = prot.STATUS_NOT_CONFIRMED
    if status in (prot.STATUS_INVALID, prot.STATUS_TASK_DEGENERATE,
                  prot.STATUS_MODEL_INSUFFICIENT):
        # Gate OOD-RISK is not reached; surface the validity panel without adjudicating.
        risk = evaluate_gate_ood_risk(quantities, bootstrap, metadata,
                                      confidence=bootstrap.confidence)

    # --- descriptive baselines/ablations (Table_OOD_2/3; never the Gate) --------------------
    prevalence = float(np.asarray(quantities["y_high"], dtype=np.int64).mean())
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
    baseline_rows: dict[str, Any] = {}
    for name in D1_ALL_BASELINE_NAMES:
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
        if name == D1_PRIMARY_BASELINE:
            entry["role"] = "primary_standard_baseline"
        elif name in prot.STANDARD_BASELINES:
            entry["role"] = "standard_baseline"
        elif name == "COMBINED_TOP2_MAX":
            entry["role"] = "historical_reference"
        else:
            entry["role"] = "curve_reference"
        baseline_rows[name] = entry
    strata = {
        "tstar_low": float(point[Q_TSTAR_LOW]),
        "tstar_high": float(point[Q_TSTAR_HIGH]),
        "interaction_low": float(point[Q_INTERACT_LOW]),
        "interaction_high": float(point[Q_INTERACT_HIGH]),
    }
    ablations = {
        "blocking": False,
        "note": ("descriptive secondary analyses reported alongside the primary score; "
                 "they never enter the gates and never substitute for the primary score"),
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
        "strata_i_aurc": strata,
        "primary_model": {
            "name": "DIRECT_ENSEMBLE",
            "aurc_regret": float(point[Q_AURC_MODEL]),
            "regret_at_80": float(point[Q_R80_MODEL]),
            "average_precision": float(point[Q_AP_MODEL]),
            "pr_auc_minus_prevalence": float(point[Q_PR]),
        },
    }
    # carry the strata CI for Table_OOD_3.
    strata_ci = {
        "tstar_low": [float(ci_lower[Q_TSTAR_LOW]), float(ci_upper[Q_TSTAR_LOW])],
        "tstar_high": [float(ci_lower[Q_TSTAR_HIGH]), float(ci_upper[Q_TSTAR_HIGH])],
        "interaction_low": [float(ci_lower[Q_INTERACT_LOW]),
                            float(ci_upper[Q_INTERACT_LOW])],
        "interaction_high": [float(ci_lower[Q_INTERACT_HIGH]),
                             float(ci_upper[Q_INTERACT_HIGH])],
    }
    seed_rows = []
    for i, seed in enumerate(prot.MODEL_SEEDS):
        ia = D1_SEED_I_AURC_INDEX[i]
        ir = D1_SEED_I_R80_INDEX[i]
        seed_rows.append({
            "seed": int(seed),
            "i_aurc_regret": float(point[ia]),
            "i_aurc_regret_ci95": [float(ci_lower[ia]), float(ci_upper[ia])],
            "i_regret80": float(point[ir]),
            "i_regret80_ci95": [float(ci_lower[ir]), float(ci_upper[ir])],
        })
    ablations["seed_metrics"] = seed_rows
    ablations["strata_i_aurc_ci95"] = strata_ci

    return OODDecision(
        plan_id=prot.PLAN_ID,
        n_groups=int(groups),
        status=status,
        passed=bool(status == prot.STATUS_CONFIRMED),
        task=task,
        risk=risk,
        metrics={
            "point": _d1_named(point),
            "ci_lower": _d1_named(ci_lower),
            "ci_upper": _d1_named(ci_upper),
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
        domain_shift=dict(domain_shift),
    )


# --- plan §6.4: the one-shot burn / truth read / quantities assembly / adjudication --------------
#
# The score-side modules (``stage4d_ood_inference`` / ``collect_stage4d_merge_ood`` /
# ``stage4cr2_confirm_inference`` / ``stage4cp_features`` / ``stage4d_confirm_protocol``) are
# imported LAZILY inside the functions below: ``stage4d_confirm_protocol`` already imports this
# module at module top, so a top-level import here would be circular.


def _d1_read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_d1_domain_shift(
    id_features: Mapping[str, np.ndarray] | np.ndarray,
    ood_features: Mapping[str, np.ndarray] | np.ndarray,
) -> dict[str, Any]:
    """§7.1-condition-2 domain shift on the D1 confirm set, at the confirm 95% CI level.

    Reuses the frozen SMD / fixed 5-fold logistic pipeline bit-for-bit; ONLY the confidence
    level is raised from the D0 pilot 0.80 to the confirm 0.95 (plan §7.1). The bootstrap
    stream derives from the locked JOINT_OOD profile entropy (2026104103) — never the D1
    main paired bootstrap (reserved for entropy 2026110995). Proof of shift ONLY; the
    classifier never enters method scoring.
    """
    id_matrix = _feature_matrix(id_features)
    ood_matrix = _feature_matrix(ood_features)
    smds = {name: standardized_mean_difference(id_matrix[:, j], ood_matrix[:, j])
            for j, name in enumerate(PRE_INTERVENTION_QUANTITIES)}
    n_shifted = count_shifted_quantities(smds)
    smd_pass = n_shifted >= prot.DOMAIN_SHIFT_MIN_SHIFTED
    entropy = int(prot.DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES["joint"])  # locked JOINT_OOD
    auc = _bootstrap_auc_ci(id_matrix, ood_matrix, entropy,
                            resamples=prot.DOMAIN_CLASSIFIER_BOOTSTRAP_RESAMPLES,
                            confidence=prot.CONFIDENCE)  # 0.95 (confirm level)
    auc_pass = auc["auc_ci_lower"] >= prot.DOMAIN_CLASSIFIER_AUC_CI_LB_MIN
    return {
        "quantities": list(PRE_INTERVENTION_QUANTITIES),
        "smd": smds,
        "abs_smd": {name: abs(value) for name, value in smds.items()},
        "smd_min": prot.DOMAIN_SHIFT_SMD_MIN,
        "min_shifted": prot.DOMAIN_SHIFT_MIN_SHIFTED,
        "n_shifted": int(n_shifted),
        "smd_pass": bool(smd_pass),
        "domain_classifier": {
            "type": "stratified_5fold_logistic_pooled_standardization",
            "n_folds": DOMAIN_CLASSIFIER_FOLDS,
            "resamples": int(auc["resamples"]),
            "confidence": float(auc["confidence"]),
            "bootstrap_seed": int(entropy),
            "n_id": int(len(id_matrix)),
            "n_ood": int(len(ood_matrix)),
            "auc_point": float(auc["auc_point"]),
            "auc_ci_lower": float(auc["auc_ci_lower"]),
            "auc_ci_upper": float(auc["auc_ci_upper"]),
            "ci_side": "two-sided percentile",
            "auc_ci_lb_min": prot.DOMAIN_CLASSIFIER_AUC_CI_LB_MIN,
            "scope": "proof of shift only; never enters method scoring",
        },
        "domain_auc_pass": bool(auc_pass),
        "shift_significant": bool(smd_pass and auc_pass),
    }


def _d1_locked_strata(checked: Mapping[str, Any]) -> tuple[int, float]:
    """Read the frozen §7.1-condition-8 strata thresholds from the protocol lock.

    The t_star split must equal the frozen ``TSTAR_SPLIT``; the initial-distance threshold
    is read from the lock (R2-train-only median, frozen by selection; never refit here).
    """
    lock_path = d1_confirm_artifact_paths(checked)["protocol_lock"]
    payload = _d1_read_json(lock_path)
    strata = payload.get("strata", {})
    split = int(strata.get("t_star", {}).get("split", -1))
    threshold = float(strata.get("initial_distance", {}).get("threshold", float("nan")))
    if split != TSTAR_SPLIT:
        raise ValueError("the locked t_star split drifted from the frozen TSTAR_SPLIT")
    if not np.isfinite(threshold):
        raise ValueError("the locked initial-distance threshold is missing/non-finite")
    return split, threshold


def _d1_selected_n(checked: Mapping[str, Any]) -> int:
    payload = _d1_read_json(d1_confirm_artifact_paths(checked)["protocol_lock"])
    return int(payload.get("selected_n", -1))


def burn_confirmation_lock_d1(config: Mapping[str, Any]) -> str:
    """Create ``confirmation_used.lock`` exactly once, after the full score freeze.

    Refuses unless the score freeze receipt exists and every score-side binding re-verifies
    against the current files (``freeze_scores_d1`` is re-run as the verifier). The lock
    binds the sealed truth ONLY through the collection receipt's recorded hash — the burn
    NEVER reads the truth bytes. A second call re-verifies the bindings against the current
    files and never overwrites.
    """
    from ac_twin import stage4d_ood_inference as d1inf
    checked = prot.validate_config(config)
    paths = d1_confirm_artifact_paths(checked)
    if not paths["score_freeze_receipt"].is_file():
        raise ValueError(
            "the score freeze receipt is missing; freeze_scores_d1 must run before the "
            "confirmation lock burns")
    freeze_receipt = d1inf.freeze_scores_d1(checked)  # re-verifies the score-side chain
    bindings = dict(freeze_receipt["bindings"])
    bindings["score_freeze_receipt_sha256"] = sha256_file(paths["score_freeze_receipt"])
    bindings["statistics_code_sha256"] = sha256_file(
        ROOT / "code/ac_twin/stage4d_ood_statistics.py")

    lock_path = paths["lock"]
    if lock_path.exists():
        payload = _d1_read_json(lock_path)
        if payload.get("plan_id") != prot.PLAN_ID \
                or payload.get("bindings") != bindings:
            raise ValueError(
                "the burned confirmation lock no longer matches the on-disk frozen "
                "artifacts (tampering or drift); the one-shot confirmation is "
                "compromised")
        return str(lock_path)
    split, threshold = _d1_locked_strata(checked)
    payload = {
        "plan_id": prot.PLAN_ID,
        "role": "d1_confirm",
        "semantics": ("one-shot: the sealed confirmation truth may be read only after "
                      "this lock exists; all confirm scores are frozen and hash-bound "
                      "above"),
        "primary_baseline": D1_PRIMARY_BASELINE,
        "interaction_threshold": float(threshold),
        "t_star_stratum_split": int(split),
        "coverage": COVERAGE,
        "selected_n": _d1_selected_n(checked),
        "bindings": bindings,
        "burned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(lock_path, payload)
    return str(lock_path)


def _d1_verify_burned_lock(checked: Mapping[str, Any], paths: Mapping[str, Path]
                           ) -> dict[str, Any]:
    """The burned lock must exist and re-verify against the current frozen files."""
    if not paths["lock"].is_file():
        raise ValueError(
            "the sealed confirmation truth may not be read: confirmation_used.lock is "
            "not burned; freeze every score artifact and burn the lock first "
            "(plan section 6.4)")
    burn_confirmation_lock_d1(checked)  # re-verifies the bindings, never overwrites
    return _d1_read_json(paths["lock"])


def _d1_load_truth_arrays(checked: Mapping[str, Any], paths: Mapping[str, Path]
                          ) -> dict[str, Any]:
    """Open the sealed truth ONCE per transaction; verify hash, schema and integrity."""
    truth_path = paths["truth"]
    if not truth_path.is_file():
        raise FileNotFoundError(f"sealed confirmation truth missing: {truth_path}")
    collection = _d1_read_json(paths["collection_receipt"])
    expected = str(collection.get("confirmation_truth_sha256", ""))
    digest = sha256_file(truth_path)
    if not expected or digest != expected:
        raise ValueError(
            "the sealed confirmation truth drifted from its collection-receipt hash")
    if list(collection.get("truth_fields", [])) != list(D1_TRUTH_FIELDS):
        raise ValueError(
            "the collection receipt's truth_fields drifted from the frozen schema "
            "(collector/statistics contract drift)")
    with np.load(truth_path, allow_pickle=False) as archive:
        if set(archive.files) != set(D1_TRUTH_FIELDS):
            raise ValueError(
                f"sealed truth archive key set drifted: {sorted(archive.files)}")
        arrays = {key: np.asarray(archive[key]) for key in D1_TRUTH_FIELDS}
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


def load_d1_confirmation_truth(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read the sealed truth — legal ONLY after the burned confirmation lock.

    Verifies the lock bindings against the current frozen files first, then the truth
    file's hash against the collection receipt, the exact frozen field set, and every
    shape/dtype/finiteness/uniqueness contract. Read-only; never writes.
    """
    checked = prot.validate_config(config)
    paths = d1_confirm_artifact_paths(checked)
    _d1_verify_burned_lock(checked, paths)
    return _d1_load_truth_arrays(checked, paths)


def _d1_quantities_receipt_bindings(checked: Mapping[str, Any],
                                    paths: Mapping[str, Path],
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
            ROOT / "code/ac_twin/stage4d_ood_statistics.py"),
    }


def _verify_d1_quantities_receipt(checked: Mapping[str, Any], paths: Mapping[str, Path]
                                  ) -> dict[str, Any]:
    """Verify an existing quantities NPZ + receipt (same-transaction resume proof).

    Any drift after the burn is UNPROVABLE -> D1ResourceAtRiskError, never a silent
    recompute.
    """
    receipt_path = paths["quantities_receipt"]
    npz_path = paths["quantities_npz"]
    if not receipt_path.is_file():
        raise D1ResourceAtRiskError(
            f"D1 confirm quantities exist without their create-once receipt: {npz_path}")
    receipt = _d1_read_json(receipt_path)
    if receipt.get("plan_id") != prot.PLAN_ID:
        raise D1ResourceAtRiskError("quantities receipt plan_id drifted")
    for key, expected in (("bootstrap_entropy", prot.D1_BOOTSTRAP_ENTROPY),
                          ("bootstrap_resamples", prot.BOOTSTRAP_RESAMPLES)):
        if int(receipt.get(key, -1)) != expected:
            raise D1ResourceAtRiskError(
                f"quantities receipt {key} drifted from the frozen {expected}")
    if float(receipt.get("confidence", -1.0)) != prot.CONFIDENCE:
        raise D1ResourceAtRiskError("quantities receipt confidence drifted from 0.95")
    if receipt.get("primary_baseline") != D1_PRIMARY_BASELINE:
        raise D1ResourceAtRiskError("quantities receipt primary baseline drifted")
    if not npz_path.is_file() or sha256_file(npz_path) \
            != receipt.get("confirm_group_quantities_sha256"):
        raise D1ResourceAtRiskError(
            "the confirm quantities NPZ drifted from its create-once receipt")
    if receipt.get("lock_sha256") != sha256_file(paths["lock"]):
        raise D1ResourceAtRiskError(
            "the quantities receipt is not bound to the burned confirmation lock")
    return receipt


def assemble_d1_quantities(config: Mapping[str, Any]) -> str:
    """Read the sealed truth ONCE, build regret/strata, write create-once quantities.

    Plan §6.4 steps 6-7. Refuses unless the one-shot lock is burned; refuses any second
    adjudication (the formal report already exists); re-verifies the full hash chain;
    cross-binds the same-order group_seed and the integrity copies across legal inputs /
    predictions / risk scores / truth (misorder or mismatch is INVALID, never repaired);
    recomputes J_hat from the frozen predictions and cross-checks the persisted reference
    action; computes J_true with the frozen Stage 4C consequence function over ALL TEN
    actions per group; computes the §7.1-condition-2 domain shift on the D1 confirm set at
    the confirm 95% CI level; writes ``d1_group_quantities.npz`` and its receipt exactly
    once. An existing quantities pair is verified and returned unchanged (same-transaction
    resume); any verification failure after the burn is a resource-at-risk stop.
    """
    from ac_twin import stage4cr2_confirm_inference as cinf
    from ac_twin import stage4cp_features as s4cpf
    from ac_twin import stage4d_ood_inference as d1inf
    checked = prot.validate_config(config)
    paths = d1_confirm_artifact_paths(checked)
    if paths["report"].is_file():
        raise FileExistsError(
            f"a second confirmation adjudication is refused; the formal report "
            f"already exists: {paths['report']}")
    if paths["quantities_npz"].exists() or paths["quantities_receipt"].exists():
        _verify_d1_quantities_receipt(checked, paths)
        return str(paths["quantities_npz"])

    lock_payload = _d1_verify_burned_lock(checked, paths)
    inference = d1inf._verify_inference_chain(paths)
    d1inf._verify_determinism(paths, inference)
    truth = _d1_load_truth_arrays(checked, paths)
    truth_arrays = truth["arrays"]
    groups = int(truth["n_groups"])

    legal = cinf._read_legal_npz(paths["inputs"],
                                 {key: key for key in cinf.LEGAL_INPUT_KEYS})
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        predictions = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(paths["risk_scores"], allow_pickle=False) as archive:
        scores = {key: np.asarray(archive[key]) for key in archive.files}

    # --- same-order group binding across all four files (§6.1) ------------------------------
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
    # --- integrity copies inside the sealed truth must equal the legal inputs ---------------
    for key in ("cf_actions", "cf_durations", "factual_actions", "t_star"):
        if not np.array_equal(np.asarray(truth_arrays[key]), np.asarray(legal[key])):
            raise ValueError(
                f"truth integrity copy {key!r} diverges from the legal inputs; the "
                "pairing is unprovable -> INVALID")
    if not np.array_equal(np.asarray(truth_arrays["inference_mask"], dtype=bool),
                          np.asarray(legal["inference_mask"], dtype=bool)):
        raise ValueError(
            "truth integrity copy 'inference_mask' diverges from the legal inputs")

    # --- J_hat from the frozen predictions; cross-check the persisted reference action -------
    history = np.asarray(legal["object_history"], dtype=np.float64)
    factual_xy = np.asarray(legal["object_factual_future"], dtype=np.float64)[
        ..., POSITION_SLICE]
    history_last_xy = history[:, -1, EGO_SLOT, POSITION_SLICE]
    mask = np.asarray(legal["inference_mask"], dtype=bool)
    e2e = np.asarray(predictions["e2e"], dtype=np.float64)
    if e2e.shape != (N_D1_SEEDS, groups, prot.N_ACTIONS, prot.HORIZON,
                     prot.OBJECT_CAPACITY, 2):
        raise ValueError(f"predictions e2e must be (5,G,10,20,5,2); got {e2e.shape}")
    _, j_hat = s4cpf.compute_predicted_costs(e2e, factual_xy, history_last_xy, mask)
    a_hat = np.argmin(j_hat, axis=-1).astype(np.int64)
    reference_action = np.asarray(scores["reference_action"], dtype=np.int64).ravel()
    if not np.array_equal(a_hat, reference_action):
        raise ValueError(
            "recomputed J_hat argmin diverges from the persisted reference_action; "
            "the prediction/score binding is broken")

    # --- J_true from the sealed truth with the frozen consequence function -------------------
    true_xy = np.asarray(truth_arrays["object_counterfactual_futures"],
                         dtype=np.float64)[..., POSITION_SLICE]
    hist_b = np.broadcast_to(history_last_xy[:, None, :], (groups, prot.N_ACTIONS, 2))
    mask_b = np.broadcast_to(mask[:, None, :, :],
                             (groups, prot.N_ACTIONS) + mask.shape[1:])
    comp_true = s4co.trajectory_components(true_xy, hist_b, mask_b)
    j_true = s4co.total_cost(comp_true["interaction"],
                             s4co.efficiency_cost(comp_true["progress"]))
    decision = _d1_decision_quantities(j_hat, j_true)

    split, threshold = _d1_locked_strata(checked)
    t_star = np.asarray(legal["t_star"], dtype=np.int64).ravel()
    distance = np.asarray(scores["interaction_min_distance"], dtype=np.float64).ravel()
    tstar_layer = (t_star >= split).astype(np.int64)
    interaction_layer = (distance > float(threshold)).astype(np.int64)

    # --- Gate-Q assembly checks (recorded; evaluated inside the gates) -----------------------
    schedule = prot.make_candidate_seed_schedule(prot.D1_CANDIDATE_SEED_ENTROPY,
                                                 prot.D1_CANDIDATE_SCHEDULE_SIZE)
    domain = set(int(v) for v in schedule)
    role_seeds_valid = bool(
        np.unique(inputs_seed).size == groups
        and all(int(s) in domain for s in inputs_seed))
    finite_truth = bool(np.isfinite(j_true).all()
                        and np.isfinite(decision["regret"]).all())
    selected_n = _d1_selected_n(checked)
    all_complete = bool(
        groups == selected_n
        and inference["collection_accepted_groups"] == selected_n
        and j_true.shape == (groups, prot.N_ACTIONS)
        and j_hat.shape == (groups, prot.N_ACTIONS)
        and finite_truth)
    by_seed = np.asarray(scores["risk_by_seed"], dtype=np.float64)
    ensemble = np.asarray(scores["risk_ensemble"], dtype=np.float64)
    qa = {
        "all_complete": all_complete,
        "hashes_match": True,   # the full chain re-verified above
        "role_seeds_valid": role_seeds_valid,
        "determinism_passed": True,  # the determinism receipt re-verified above
        "one_shot_locks": True,
        "ensemble_is_arithmetic_mean": bool(
            by_seed.shape == (N_D1_SEEDS, groups)
            and np.allclose(ensemble, by_seed.mean(axis=0), rtol=0.0, atol=1e-15)),
    }

    # --- §7.1-condition-2 domain shift on the D1 confirm set (confirm 95% CI) ----------------
    ood_features = pre_intervention_features(history)
    id_features = load_r2_train_id_features(checked)
    domain_shift = compute_d1_domain_shift(id_features, ood_features)

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
        "n_groups": np.int64(groups),
        "interaction_threshold": np.float64(threshold),
    }
    for name in D1_ALL_BASELINE_NAMES:
        if name == "CONSTANT_RANDOM":
            risk = np.asarray(scores["constant_random"], dtype=np.float64).ravel()
        else:
            risk = np.asarray(scores[f"baseline_{name}"], dtype=np.float64).ravel()
        if risk.shape != (groups,) or not np.isfinite(risk).all():
            raise ValueError(f"confirm baseline risk {name!r} must be finite/aligned")
        payload[_D1_BASELINE_NPZ_PREFIX + name] = risk
    atomic_save_npz(npz_path, **payload)

    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "d1_confirm",
        "n_groups": int(groups),
        "confirm_group_quantities_path": str(npz_path),
        "confirm_group_quantities_sha256": sha256_file(npz_path),
        "primary_baseline": D1_PRIMARY_BASELINE,
        "selected_n": int(selected_n),
        "interaction_threshold": float(threshold),
        "interaction_threshold_source": "R2 train-only median, frozen by selection",
        "tstar_stratum_split": int(split),
        "strata": {
            "tstar_low": f"t_star < {split}",
            "tstar_high": f"t_star >= {split}",
            "interaction_low": f"interaction_min_distance <= {threshold}",
            "interaction_high": f"interaction_min_distance > {threshold}",
            "note": ("two marginal splits, NOT a 4-cell cross; quantiles never refit on "
                     "confirm data"),
        },
        "coverage": COVERAGE,
        "high_regret_threshold": prot.HIGH_REGRET_THRESHOLD,
        "bootstrap_entropy": prot.D1_BOOTSTRAP_ENTROPY,
        "bootstrap_resamples": prot.BOOTSTRAP_RESAMPLES,
        "confidence": prot.CONFIDENCE,
        "qa": qa,
        "domain_shift": _json_safe(domain_shift),
        "bindings": _d1_quantities_receipt_bindings(checked, paths, truth["sha256"]),
        "lock_sha256": sha256_file(paths["lock"]),
        "lock_burned_at": lock_payload.get("burned_at", ""),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["quantities_receipt"], receipt)
    return str(npz_path)


def load_d1_group_quantities(
    npz_path: str | Path, receipt_path: str | Path
) -> dict[str, Any]:
    """Load the create-once quantities NPZ + receipt into the gate-input mapping."""
    with np.load(Path(npz_path), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    receipt = _d1_read_json(Path(receipt_path))
    baselines = {}
    for key in list(data):
        if key.startswith(_D1_BASELINE_NPZ_PREFIX):
            baselines[key[len(_D1_BASELINE_NPZ_PREFIX):]] = data.pop(key)
    quantities: dict[str, Any] = dict(data)
    quantities["baseline_risks"] = baselines
    quantities["strongest_baseline"] = str(receipt["primary_baseline"])
    quantities["n_groups"] = int(np.asarray(data["n_groups"]).item())
    quantities["interaction_threshold"] = float(
        np.asarray(data["interaction_threshold"]).item())
    quantities["qa"] = {key: bool(value) for key, value in receipt["qa"].items()}
    quantities["domain_shift"] = receipt.get("domain_shift")
    _validate_d1_quantities(quantities)
    return quantities


# --- §8 output rendering (byte-deterministic) ----------------------------------------------------


def _d1_fmt(value: Any) -> str:
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


def _d1_csv_bytes(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)  # default lineterminator "\r\n": deterministic
    writer.writerow(header)
    for row in rows:
        writer.writerow([_d1_fmt(value) for value in row])
    return buffer.getvalue().encode("utf-8")


def _write_d1_bytes_once_or_verify(path: Path, blob: bytes) -> None:
    """Create-once bytes: a pre-existing file must be byte-identical, else the post-burn
    transaction is unprovable (resource-at-risk, never a silent overwrite)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as handle:
            handle.write(blob)
    except FileExistsError:
        if path.read_bytes() != blob:
            raise D1ResourceAtRiskError(
                f"a pre-existing output differs from the recomputed bytes and the "
                f"transaction cannot be proven identical: {path}")


def _render_table_ood_1(decision: OODDecision) -> bytes:
    """Table_OOD_1: Gate TASK-OOD (§7.1, 8 conditions) + Gate OOD-RISK (§7.2, 5) + terminal."""
    point = decision.metrics["point"]
    ci_lower = decision.metrics["ci_lower"]
    upper = decision.metrics["ci_upper"]
    task, risk = decision.task, decision.risk

    def row(gate, condition, metric=None, passed=None, note=""):
        if metric is None:
            return (gate, condition, None, None, None, passed, note)
        return (gate, condition, point[metric], ci_lower[metric], upper[metric],
                passed, note)

    ds = decision.domain_shift
    rows: list[tuple[Any, ...]] = [
        ("TASK", "c1_complete_gate_q", None,
         task["conditions"]["c1_complete_gate_q"], f"n_groups={decision.n_groups}"),
        ("TASK", "c2_domain_shift", None, task["conditions"]["c2_domain_shift"],
         f"n_shifted={ds.get('n_shifted')};auc_ci_lower="
         f"{_d1_fmt(ds.get('domain_classifier', {}).get('auc_ci_lower'))}"),
        ("TASK", "c3_high_regret_frac", "high_regret_frac",
         task["conditions"]["c3_high_regret_frac"], ""),
        ("TASK", "c4_oracle_diversity", None,
         task["conditions"]["c4_oracle_diversity"],
         f"covered={task['oracle_actions_covered_ge_min_frac']}"),
        ("TASK", "c5_branch_spearman", "spearman_branch",
         task["conditions"]["c5_branch_spearman"], ""),
        ("TASK", "c6_base_beats_random", "base_vs_random_improvement",
         task["conditions"]["c6_base_beats_random"], ""),
        ("TASK", "c7_base_not_worse_than_idle", "base_mean_regret",
         task["conditions"]["c7_base_not_worse_than_idle"],
         f"idle_mean_regret={_d1_fmt(point['idle_mean_regret'])}"),
        ("TASK", "c8_strata_min_10pct", None,
         task["conditions"]["c8_strata_min_10pct"],
         json.dumps(task["stratum_fracs"], sort_keys=True)),
        ("RISK", "r1_i_aurc_regret", "i_aurc_regret",
         risk["conditions"]["r1_i_aurc_regret"], "primary baseline SEED_VOTE_ENTROPY"),
        ("RISK", "r2_i_regret80", "i_regret80",
         risk["conditions"]["r2_i_regret80"], "primary baseline SEED_VOTE_ENTROPY"),
        ("RISK", "r3_pr_auc_minus_prevalence", "pr_auc_minus_prevalence",
         risk["conditions"]["r3_pr_auc_minus_prevalence"], ""),
        ("RISK", "r4_seed_consistency", None, risk["conditions"]["r4_seed_consistency"],
         f"both_directions_positive={risk['seeds_both_directions_positive']}/"
         f"{N_D1_SEEDS}"),
        ("RISK", "r5_validity", None, risk["conditions"]["r5_validity"],
         json.dumps(risk["baseline_denominators"], sort_keys=True)),
        ("TERMINAL", "status", None, decision.passed, decision.status),
    ]
    return _d1_csv_bytes(
        ("gate", "condition", "point", "ci95_lower", "ci95_upper", "passed", "notes"),
        rows)


def _render_table_ood_2(decision: OODDecision) -> bytes:
    """Table_OOD_2: primary model + 3 standard baselines + 2 references; the non-primary
    comparisons ride the SAME paired draws and never enter the Gate (no baseline swap)."""
    model = decision.ablations["primary_model"]
    rows: list[tuple[Any, ...]] = [(
        "DIRECT_ENSEMBLE", "primary_model", model["aurc_regret"], model["regret_at_80"],
        model["average_precision"], model["pr_auc_minus_prevalence"],
        None, None, None, None, None, None)]
    for name in D1_ALL_BASELINE_NAMES:
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
    return _d1_csv_bytes(
        ("name", "role", "aurc_regret", "regret_at_80", "average_precision",
         "pr_auc_minus_prevalence", "i_aurc_point", "i_aurc_ci95_lower",
         "i_aurc_ci95_upper", "i_regret80_point", "i_regret80_ci95_lower",
         "i_regret80_ci95_upper"),
        rows)


def _render_table_ood_3(decision: OODDecision) -> bytes:
    """Table_OOD_3: per-seed endpoint improvements + the two marginal strata (secondary)."""
    point = decision.metrics["point"]
    lower = decision.metrics["ci_lower"]
    upper = decision.metrics["ci_upper"]
    rows: list[tuple[Any, ...]] = []
    for seed in prot.MODEL_SEEDS:
        for metric in (f"seed{seed}_i_aurc", f"seed{seed}_i_regret80"):
            rows.append((f"seed_{seed}", metric.rsplit("_i_", 1)[-1],
                         point[metric], lower[metric], upper[metric]))
    for name in ("tstar_low_i_aurc", "tstar_high_i_aurc", "interaction_low_i_aurc",
                 "interaction_high_i_aurc"):
        rows.append((name, "stratum", point[name], lower[name], upper[name]))
    return _d1_csv_bytes(("name", "kind", "point", "ci95_lower", "ci95_upper"), rows)


def _render_fig_ood_1(quantities: Mapping[str, Any]) -> bytes:
    """Fig_OOD_1: full-coverage regret curves for the ensemble and every baseline; the
    fixed 80% coverage mark. Byte-deterministic (Agg + fixed metadata)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regret = np.asarray(quantities["regret"], dtype=np.float64)
    groups = regret.size
    coverage = np.arange(1, groups + 1, dtype=np.float64) / groups
    curves = [("DIRECT_ENSEMBLE", np.asarray(quantities["ensemble_risk"],
                                            dtype=np.float64))]
    for name in D1_ALL_BASELINE_NAMES:
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
                metadata={"Software": "stage4d_ood_statistics"})
    plt.close(fig)
    return buffer.getvalue()


def _render_fig_ood_2(domain_shift: Mapping[str, Any]) -> bytes:
    """Fig_OOD_2: |SMD| per pre-intervention quantity (0.50 mark) + the domain-classifier
    AUC with its 95% CI (0.70 mark). Proof of shift only. Byte-deterministic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(domain_shift.get("quantities", PRE_INTERVENTION_QUANTITIES))
    abs_smd = [float(domain_shift.get("abs_smd", {}).get(name, float("nan")))
               for name in names]
    classifier = domain_shift.get("domain_classifier", {})
    auc_point = float(classifier.get("auc_point", float("nan")))
    auc_lo = float(classifier.get("auc_ci_lower", float("nan")))
    auc_hi = float(classifier.get("auc_ci_upper", float("nan")))

    fig, (ax_smd, ax_auc) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    x = np.arange(len(names))
    ax_smd.bar(x, abs_smd, color="#4472a8")
    ax_smd.axhline(float(domain_shift.get("smd_min", 0.50)), color="black",
                   linestyle="--", linewidth=0.8, label="|SMD| 0.50")
    ax_smd.set_xticks(x)
    ax_smd.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax_smd.set_ylabel("|SMD| vs R2 train")
    ax_smd.legend(fontsize=6)
    ax_auc.errorbar([0], [auc_point],
                    yerr=[[auc_point - auc_lo], [auc_hi - auc_point]],
                    fmt="o", color="#4472a8", capsize=4)
    ax_auc.axhline(float(classifier.get("auc_ci_lb_min", 0.70)), color="black",
                   linestyle="--", linewidth=0.8, label="AUC CI lower 0.70")
    ax_auc.set_xlim(-0.5, 0.5)
    ax_auc.set_xticks([])
    ax_auc.set_ylabel("domain classifier AUC (95% CI)")
    ax_auc.legend(fontsize=6)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150,
                metadata={"Software": "stage4d_ood_statistics"})
    plt.close(fig)
    return buffer.getvalue()


def _render_decision_md_d1(decision: OODDecision) -> bytes:
    """The formal decision document (plan §8.2 paper boundary, verbatim rules)."""
    point = decision.metrics["point"]
    lower, upper = decision.metrics["ci_lower"], decision.metrics["ci_upper"]
    lines = [
        "# Stage 4D D1 Model-Mismatch OOD Confirmation — Decision",
        "",
        f"- plan_id: {decision.plan_id}",
        f"- n_groups: {decision.n_groups}",
        f"- terminal status: **{decision.status}**",
        f"- gate TASK-OOD status: {decision.task['status']}",
        f"- primary baseline (validation-frozen, Gate-only): {D1_PRIMARY_BASELINE}",
        "",
        "## Primary endpoints (95% CI, two-sided percentile, 10,000 paired draws)",
        "",
        f"- I_AURC_REGRET = {_d1_fmt(point['i_aurc_regret'])} "
        f"[{_d1_fmt(lower['i_aurc_regret'])}, {_d1_fmt(upper['i_aurc_regret'])}]",
        f"- I_REGRET@80 = {_d1_fmt(point['i_regret80'])} "
        f"[{_d1_fmt(lower['i_regret80'])}, {_d1_fmt(upper['i_regret80'])}]",
        f"- PR-AUC - prevalence = {_d1_fmt(point['pr_auc_minus_prevalence'])} "
        f"[{_d1_fmt(lower['pr_auc_minus_prevalence'])}, "
        f"{_d1_fmt(upper['pr_auc_minus_prevalence'])}]",
        f"- AURC_REGRET (ensemble) = {_d1_fmt(point['aurc_model'])}; "
        f"REGRET@80 (ensemble) = {_d1_fmt(point['regret80_model'])}; "
        f"coverage = {_d1_fmt(COVERAGE)}; "
        f"rejected fraction = {_d1_fmt(1.0 - COVERAGE)}",
        "",
        "## Paper boundary (plan §8.2)",
        "",
    ]
    if decision.status == prot.STATUS_CONFIRMED:
        lines += [D1_SUCCESS_SENTENCE, ""]
    else:
        lines += [
            f"终态为 {decision.status}，成功句式不适用；不得改写为任何形式的确认成功。",
            "",
        ]
    lines += [
        D1_PROHIBITED_CLAIMS,
        "",
        "单项 CI 是边际 CI，不宣称所有次分析联合 95% 覆盖。描述性次分析只入次表，不进主 "
        "AND Gate，不挽救主结果。无论裁判成败，不改写 Stage 4C-R2 或任何更早阶段已有终态。",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


# --- plan §6.4 steps 8-9: the single adjudication + create-once formal outputs ---------------------


def adjudicate_d1(
    config: Mapping[str, Any],
    *,
    bootstrap_fn: Callable[[Mapping[str, Any]], BootstrapResultD1] | None = None,
) -> dict[str, Any]:
    """Run the ONE confirmation adjudication and write the create-once formal outputs.

    Refuses a second adjudication (the formal report already exists). Requires the
    create-once quantities pair; verifies the NPZ hash, the frozen bootstrap constants and
    the lock binding — any post-burn drift is unprovable and stops at
    resource-at-risk. The statistics run exactly once at the frozen entropy/resamples/95%
    confidence; the §8.1 files are byte-deterministic (a same-transaction resume re-derives
    identical bytes and never overwrites); the report JSON binds the quantities hash and is
    written LAST (its presence marks the completed transaction).
    """
    checked = prot.validate_config(config)
    paths = d1_confirm_artifact_paths(checked)
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
            "the confirm quantities are missing; assemble_d1_quantities must run "
            "before adjudication")
    receipt = _verify_d1_quantities_receipt(checked, paths)
    quantities = load_d1_group_quantities(paths["quantities_npz"],
                                          paths["quantities_receipt"])
    if bootstrap_fn is None:
        bootstrap = paired_group_bootstrap_d1(
            quantities, seed=prot.D1_BOOTSTRAP_ENTROPY,
            n_resamples=prot.BOOTSTRAP_RESAMPLES)
    else:
        bootstrap = bootstrap_fn(quantities)
    selected_n = int(receipt.get("selected_n", quantities["n_groups"]))
    domain_shift = receipt.get("domain_shift")
    if not isinstance(domain_shift, Mapping):
        raise D1ResourceAtRiskError(
            "the quantities receipt lost its domain_shift panel")
    decision = evaluate_d1_gates(quantities, bootstrap,
                                 selected_n=selected_n, domain_shift=domain_shift)

    outputs = {
        "table_ood_1": _render_table_ood_1(decision),
        "table_ood_2": _render_table_ood_2(decision),
        "table_ood_3": _render_table_ood_3(decision),
        "fig_ood_1": _render_fig_ood_1(quantities),
        "fig_ood_2": _render_fig_ood_2(domain_shift),
        "decision": _render_decision_md_d1(decision),
    }
    for key, blob in outputs.items():
        _write_d1_bytes_once_or_verify(paths[key], blob)

    report = {
        "plan_id": prot.PLAN_ID,
        "role": "d1_confirm",
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
                ROOT / "code/ac_twin/stage4d_ood_statistics.py"),
        },
        "outputs": {key: {"path": str(paths[key]),
                          "sha256": hashlib.sha256(blob).hexdigest()}
                    for key, blob in outputs.items()},
        "paper_boundary": {
            "success_sentence_written": bool(decision.status == prot.STATUS_CONFIRMED),
            "success_sentence": D1_SUCCESS_SENTENCE,
            "prohibited_claims": D1_PROHIBITED_CLAIMS,
            "secondary_analyses_note": (
                "descriptive secondary analyses enter the secondary tables only; they "
                "never enter the main AND-gate and never rescue the primary result"),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(paths["report"], report)
    result = decision.to_dict()
    result["report_path"] = str(paths["report"])
    result["decision_path"] = str(paths["decision"])
    return result


__all__ = [
    "D0_FORBIDDEN_PATH_TOKENS",
    "D0ForbiddenReadError",
    "D1_POWER_QUANTITIES",
    "D1_POWER_QUANTITY_NAMES",
    "DENSITY_RADIUS_M",
    "DOMAIN_CLASSIFIER_FOLDS",
    "MERGING_ID_MIN_MARGIN_M",
    "N_D1_POWER_QUANTITIES",
    "PILOT_OF_PROFILE",
    "PILOT_ORDER",
    "POSITION_SLICE",
    "PRE_INTERVENTION_QUANTITIES",
    "PROFILE_OF_PILOT",
    "P_I_AURC",
    "P_I_REGRET80",
    "P_PR",
    "P_SEED_I_AURC",
    "P_SEED_I_R80",
    "R2_TRAIN_LEGAL_KEYS",
    "TTC_CLIP_MAX",
    "VELOCITY_SLICE",
    "Z_95",
    "assemble_pilot_task_quantities",
    "assert_d0_read_legal",
    "compute_domain_shift",
    "count_shifted_quantities",
    "evaluate_d0_profile",
    "evaluate_d1_power_draws",
    "evaluate_pilot_integrity",
    "evaluate_task_support",
    "load_r2_train_id_features",
    "pre_intervention_features",
    "select_ood_profile",
    "selection_file_fingerprint",
    "simulate_d1_joint_power",
    "simulate_d1_power",
    "standardized_mean_difference",
    "write_profile_report",
    "write_profile_selection",
    # --- Task 6 (§6.4/§7/§8/§13): the one-shot D1 unseal + adjudication ---------------------
    "BootstrapResultD1",
    "COVERAGE",
    "D1ResourceAtRiskError",
    "D1_ALL_BASELINE_NAMES",
    "D1_PRIMARY_BASELINE",
    "D1_QA_KEYS",
    "D1_REQUIRED_ARRAY_KEYS",
    "D1_SEED_I_AURC_INDEX",
    "D1_SEED_I_R80_INDEX",
    "D1_STRATA_INDEX",
    "D1_TRUTH_FIELDS",
    "N_D1_QUANTITIES",
    "N_D1_SEEDS",
    "OODDecision",
    "Q_I_AURC",
    "Q_I_REGRET80",
    "Q_PR",
    "Q_SPEARMAN_BRANCH",
    "Q_BASE_VS_RANDOM",
    "Q_HIGH_REGRET_FRAC",
    "Q_AURC_BASELINE",
    "Q_R80_BASELINE",
    "Q_BASE_MEAN_REGRET",
    "Q_IDLE_MEAN_REGRET",
    "Q_RANDOM_MEAN_REGRET",
    "Q_AURC_MODEL",
    "Q_R80_MODEL",
    "Q_SEED_I_AURC",
    "Q_SEED_I_R80",
    "STATUS_D1_TASK_QUALIFIED",
    "TSTAR_SPLIT",
    "adjudicate_d1",
    "assemble_d1_quantities",
    "burn_confirmation_lock_d1",
    "compute_d1_domain_shift",
    "compute_d1_quantity_vector",
    "coverage_index",
    "d1_confirm_artifact_paths",
    "evaluate_d1_gates",
    "evaluate_gate_ood_risk",
    "evaluate_gate_task_ood",
    "load_d1_confirmation_truth",
    "load_d1_group_quantities",
    "paired_group_bootstrap_d1",
]
