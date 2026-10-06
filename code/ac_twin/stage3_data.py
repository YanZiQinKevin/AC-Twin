"""Stage 3 data roles, response labels, masks and sealed-holdout rejection.

Stage 3 develops a response predictor (Stage 3A) and an evidence router
(Stage 3B) on the single development source
``data/object_centric_stage/merge_2000/object_twinworld_2000.npz``. The three
splits have fixed, non-interchangeable roles:

- split 0 (train): training, scaler fitting, class weights, cross-fit;
- split 1 (validation): early stopping, temperature calibration, bounded
  candidate selection;
- split 2 (development test): a single frozen-candidate development test,
  Go/No-Go and power estimation. It never reverse-selects expanded candidates.

The Stage 2 confirmatory holdout
(``data/object_centric_stage/stage2_merge_1200/...``) is permanently sealed. It
must never enter a Stage 3 loader, training, tuning, power, case selection or
threshold selection. :func:`assert_not_sealed_stage2` rejects it by normalized
absolute path, project-relative logical path and known SHA256, and the tests
cover case, separator and symlink/resolved-path bypasses.

This module never reads ``object_counterfactual_future`` into a model input;
the counterfactual branch is used exclusively to build the response truth, the
regime labels and the evaluation mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin.stage1_data import (
    FUTURE_STEPS,
    HISTORY_STEPS,
    N_FEATURES,
    POSITION_FEATURES,
    compute_history_agent_known,
    compute_valid_mask,
    sha256_file,
    validate_source_contract,
)
from ac_twin.stage1_statistics import assign_response_bin

# Frozen Stage 1 response-bin thresholds; must match response_bins.json exactly.
FROZEN_ZERO_TOLERANCE = 0.0001
FROZEN_T1 = 0.24843344965718578
FROZEN_T2 = 1.898992796585611
FROZEN_THRESHOLDS: dict[str, float] = {
    "zero_tolerance": FROZEN_ZERO_TOLERANCE,
    "t1": FROZEN_T1,
    "t2": FROZEN_T2,
}

# Split roles (integer labels stored in the npz ``split`` array).
SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_DEVELOPMENT_TEST = 2
SPLIT_ROLE_NAMES = {
    SPLIT_TRAIN: "train",
    SPLIT_VALIDATION: "validation",
    SPLIT_DEVELOPMENT_TEST: "development_test",
}

# Regime class ids (shared with Stage 1/2 ``assign_response_bin``).
REGIME_ZERO = 0
REGIME_LOW = 1
REGIME_MID = 2
REGIME_HIGH = 3
N_REGIMES = 4

# Development training must span exactly durations {1,3}; duration=2 is the
# training-unseen interpolation/extrapolation setting reserved for Gate C.
EXPECTED_TRAIN_DURATIONS = frozenset({1, 3})
UNSEEN_DURATION = 2

# Permanently sealed Stage 2 confirmatory holdout fingerprints.
STAGE2_HOLDOUT_LOGICAL_PATH = (
    "data/object_centric_stage/stage2_merge_1200/object_twinworld_stage2_merge_1200.npz"
)
STAGE2_HOLDOUT_FILE_SHA256 = (
    "0df40704efecf81d2752a8aea2b8c0558daea26c892f3aa310bc3e1c6473fd78"
)
STAGE2_HOLDOUT_CONTENT_SHA256 = (
    "3b5dce62b40ebef3b4d7750b00d7fb04fd4a2a86d2a6adc6d6f89c49399874be"
)
STAGE2_HOLDOUT_SHA256S = frozenset(
    {STAGE2_HOLDOUT_FILE_SHA256, STAGE2_HOLDOUT_CONTENT_SHA256}
)

# --- Stage 3A replacement development holdout (repair plan v0.1 §3) ---------------
# The old merge split 2 was adjudicated once and then inspected during
# debugging; it is permanently downgraded to HISTORICAL_INVALID_DIAGNOSTIC_ONLY.
# A fresh, independent, single-use replacement development holdout is collected
# with the SAME generation contract and a brand-new episode-seed universe.
REPLACEMENT_DEV_PURPOSE = "stage3a_replacement_development_test"
REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS = (300, 400, 500, 600, 700, 800)
REPLACEMENT_DEV_TARGET_POWER = 0.90
REPLACEMENT_DEV_POWER_SEED = 20260909
REPLACEMENT_DEV_COLLECTION_SEED = 20260919
REPLACEMENT_DEV_INFEASIBLE = "REPLACEMENT_DEV_POWER_INFEASIBLE"
REPLACEMENT_DEV_DATA_DIR = "data/object_centric_stage/stage3a_replacement_dev"
REPAIR_PLAN_ID = "AC-TWIN-RFER-STAGE3A-REPAIR-2026-09-09-V0.1"

# Frozen generation contract of the original merge development data
# (config/object_centric_merge_collect.json). The replacement holdout must
# replicate it EXACTLY: same scenario, action set, action intensities
# (probabilities), duration set {1,3}, onset window and acceptance rules; the
# only permitted change is a brand-new episode-seed universe (v0.1 §1.7).
MERGE_GENERATION_CONTRACT: dict[str, Any] = {
    "scene": "merge",
    "env_id": "merge-v0",
    "horizon": 20,
    "history_len": 10,
    "t_star_low": 10,
    "t_star_high": 15,
    "intervention_durations": [1, 3],
    "vehicles_count": 25,
    "observed_vehicles_count": 5,
    "duration": 100,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "determinism_checks": 20,
    "max_attempt_multiplier": 15,
    "action_weights": {
        "idle_maintain": 0.45,
        "faster": 0.18,
        "slower": 0.18,
        "other": 0.095,
    },
}

# The development source (old merge incl. the contaminated split 2) and the two
# future Gate C holdout families can never serve as the replacement gate input.
DEVELOPMENT_SOURCE_LOGICAL_PATH = (
    "data/object_centric_stage/merge_2000/object_twinworld_2000.npz"
)
_GATEC_PATH_MARKER = "gatec"


# --- Stage 3A Plus: calibration stability + new single-use qualification set ---
# (plan `AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1`). The old merge split 2 and
# the consumed replacement holdout are downgraded to HISTORICAL_DEVELOPMENT_
# DIAGNOSTIC roles — reusable for development/model selection but NEVER again a
# confirmatory adjudication. A brand-new, single-use qualification set is the
# only data allowed to carry Gate R+_qualification.
STAGE3A_PLUS_PLAN_ID = "AC-TWIN-RFER-STAGE3A-PLUS-2026-09-09-V0.1"
STAGE3A_PLUS_ROLE_QUALIFICATION = "qualification"

# Frozen seed sets (plan Global Constraint 6). The five production seeds are the
# Stage 3A retrain predictor seeds; the five stress seeds extend the stability
# observation. All ten enter the stability report and may never be dropped.
STAGE3A_PLUS_DEV_SEEDS = (101, 202, 303, 404, 505)
STAGE3A_PLUS_STRESS_SEEDS = (606, 707, 808, 909, 1010)
STAGE3A_PLUS_ALL_SEEDS = STAGE3A_PLUS_DEV_SEEDS + STAGE3A_PLUS_STRESS_SEEDS

# Frozen power / collection / gate constants (plan §6).
STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS = (400, 600, 800, 1000, 1200, 1600, 2000)
STAGE3A_PLUS_TARGET_POWER = 0.90
STAGE3A_PLUS_POWER_SEED = 20260930
STAGE3A_PLUS_COLLECTION_SEED = 20261009
STAGE3A_PLUS_GATE_BOOTSTRAP_SEED = 20260929
STAGE3A_PLUS_MONTE_CARLO_REPLICATES = 2000
STAGE3A_PLUS_INNER_BOOTSTRAP = 1000
STAGE3A_PLUS_FORMAL_BOOTSTRAP = 10000

STAGE3A_PLUS_QUALIFICATION_DATA_DIR = (
    "data/object_centric_stage/stage3a_plus_qualification"
)
STAGE3A_PLUS_QUALIFICATION_DIR_SEGMENT = "stage3a_plus_qualification"
STAGE3A_PLUS_QUALIFICATION_PURPOSE = "stage3a_plus_qualification"

# Known content fingerprints of viewed/contaminated Stage 3A sources, so a
# renamed or relocated copy is still refused as the qualification set. Stage 2
# fingerprints live in STAGE2_HOLDOUT_SHA256S.
OLD_DEVELOPMENT_SOURCE_FILE_SHA256 = (
    "eb6122ac62af20219fc0b27d346740a935587ff32d82e4000127a50a41466e2b"
)
REPLACEMENT_DEV_FILE_SHA256 = (
    "4031aaaeb98a51cb101c195adbd10b436a9e69f1addde1fb0f0900d6a442dd82"
)
STAGE3A_PLUS_VIEWED_SOURCE_SHA256S = frozenset(
    {OLD_DEVELOPMENT_SOURCE_FILE_SHA256, REPLACEMENT_DEV_FILE_SHA256}
)

# Old Stage 3A run/data locations are viewed development products. Exact path
# segments only — the new qualification directory (``stage3a_plus_qualification``)
# must NOT be caught by this set.
_STAGE3A_PLUS_FORBIDDEN_QUALIFICATION_SEGMENTS = frozenset(
    {"stage3a", "stage3a_retrain_v1", "stage3a_replacement_dev", "merge_2000"}
)


def _normalize_logical(path: str | Path) -> str:
    """Lowercase, forward-slashed, no-trailing-slash logical form of a path."""
    text = str(path).replace("\\", "/")
    # Collapse any accidental duplicate slashes and drop a trailing slash.
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/").lower()


def is_sealed_stage2_path(path: str | Path) -> bool:
    """True if ``path`` resolves to the sealed Stage 2 holdout npz.

    Compares both the project-relative logical form and the resolved absolute
    form (the latter defeats ``..``, case, separator and symlink/junction
    bypasses), case-insensitively.
    """
    logical = _normalize_logical(path)
    target = _normalize_logical(STAGE2_HOLDOUT_LOGICAL_PATH)
    if logical == target or logical.endswith("/" + target):
        return True
    try:
        resolved = _normalize_logical(Path(path).resolve())
    except (OSError, RuntimeError):
        resolved = logical
    if resolved == target or resolved.endswith("/" + target):
        return True
    return False


def assert_not_sealed_stage2(
    path: str | Path | None = None,
    *,
    file_sha256: str | None = None,
) -> None:
    """Raise ``ValueError`` if the sealed Stage 2 holdout is referenced.

    Rejects by normalized path (absolute or logical) and by known SHA256, so a
    renamed, relocated, case-flipped or symlinked copy of the sealed data is
    still refused as long as its content hash matches a registered fingerprint.
    """
    if path is not None and is_sealed_stage2_path(path):
        raise ValueError(
            "Stage 2 confirmatory holdout is permanently sealed and must not "
            f"enter Stage 3 (path: {path})"
        )
    if file_sha256 is not None and file_sha256.lower() in STAGE2_HOLDOUT_SHA256S:
        raise ValueError(
            "Stage 2 confirmatory holdout is permanently sealed and must not "
            f"enter Stage 3 (sha256: {file_sha256})"
        )


def scan_all_episode_seeds(
    data_root: str | Path,
    *,
    exclude_substring: str | None = None,
) -> np.ndarray:
    """Union of every ``episode_seed``/``episode_seeds`` array under ``data_root``.

    Unlike :func:`ac_twin.stage2_protocol.scan_historical_episode_seeds` this
    scan does NOT exclude ``stage2_`` products: the Stage 3A replacement
    development holdout must be disjoint from the sealed Stage 2 holdout too
    (repair plan v0.1 hard constraint 8). ``exclude_substring`` lets the
    replacement collector exempt its own (partial) products when re-scanning
    at collection/resume time; audits of foreign collections pass ``None``.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    collected: set[int] = set()
    for file_path in sorted(data_root.rglob("*.npz")):
        normalized = str(file_path).replace("\\", "/")
        if exclude_substring and exclude_substring in normalized:
            continue
        try:
            with np.load(file_path, allow_pickle=True) as archive:
                for key in ("episode_seed", "episode_seeds"):
                    if key in archive.files:
                        values = np.atleast_1d(np.asarray(archive[key])).ravel()
                        collected.update(int(v) for v in values)
        except Exception:
            continue
    if not collected:
        raise ValueError(f"no historical episode seeds found under {data_root}")
    return np.array(sorted(collected), dtype=np.int64)


def assert_replacement_source_allowed(
    path: str | Path | None,
    *,
    file_sha256: str | None = None,
) -> None:
    """Reject forbidden replacement-gate inputs (repair plan v0.1 §5 Task 6).

    The replacement development test may only read its own freshly collected
    holdout. Refused: the sealed Stage 2 holdout (path or known SHA256), the
    old development source (its split 2 is contaminated), and any Gate C
    holdout path (present or future).
    """
    assert_not_sealed_stage2(path, file_sha256=file_sha256)
    if path is None:
        return
    logical = _normalize_logical(path)
    target = _normalize_logical(DEVELOPMENT_SOURCE_LOGICAL_PATH)
    if logical == target or logical.endswith("/" + target):
        raise ValueError(
            "the old development source (merge_2000) contains the contaminated "
            "split 2 and can never be the replacement gate input"
        )
    for segment in logical.split("/"):
        if segment.startswith(_GATEC_PATH_MARKER):
            raise ValueError(
                f"Gate C holdout paths are forbidden as the replacement gate "
                f"input (path: {path})"
            )


def assert_stage3a_plus_source_role(
    path: str | Path | None,
    role: str,
    *,
    file_sha256: str | None = None,
) -> None:
    """Assert ``path`` may serve the Stage 3A Plus data ``role`` (else ValueError).

    The only role governed here is ``"qualification"``: the single-use,
    not-yet-collected Stage 3A Plus qualification holdout. It must live in the
    dedicated ``stage3a_plus_qualification`` directory and must never be the
    sealed Stage 2 holdout, the contaminated old development source
    (merge_2000), the consumed replacement development holdout, any old
    ``stage3a*`` run/data directory or any Gate C path — enforced by path AND by
    known content SHA256 so a renamed/relocated copy is still refused. The
    positive directory requirement is the final guard: only the new
    qualification location passes.
    """
    if role != STAGE3A_PLUS_ROLE_QUALIFICATION:
        raise ValueError(f"unknown Stage 3A Plus source role: {role!r}")
    assert_not_sealed_stage2(path, file_sha256=file_sha256)
    if file_sha256 is not None and file_sha256.lower() in STAGE3A_PLUS_VIEWED_SOURCE_SHA256S:
        raise ValueError(
            "a viewed/contaminated Stage 3A source (old merge split 2 or the "
            "consumed replacement development holdout) can never be the Stage 3A "
            f"Plus qualification set (sha256: {file_sha256})"
        )
    if path is None:
        raise ValueError("a Stage 3A Plus qualification source path is required")
    logical = _normalize_logical(path)
    segments = logical.split("/")
    for segment in segments:
        if segment in _STAGE3A_PLUS_FORBIDDEN_QUALIFICATION_SEGMENTS:
            raise ValueError(
                "old Stage 3A run/data locations and the contaminated old merge "
                "source are viewed development products and can never be the "
                f"Stage 3A Plus qualification set (path: {path})"
            )
        if segment.startswith(_GATEC_PATH_MARKER):
            raise ValueError(
                "Gate C holdout paths are forbidden as the Stage 3A Plus "
                f"qualification set (path: {path})"
            )
    if STAGE3A_PLUS_QUALIFICATION_DIR_SEGMENT not in segments:
        raise ValueError(
            "the Stage 3A Plus qualification set must live under "
            f"{STAGE3A_PLUS_QUALIFICATION_DATA_DIR} (path: {path})"
        )


def assert_train_duration_set(cf_duration_train: np.ndarray) -> None:
    """Assert development training durations are exactly ``{1, 3}``.

    Gate C treats duration=2 as training-unseen; that label is only meaningful
    if the development train split spans exactly ``{1, 3}``.
    """
    observed = frozenset(int(v) for v in np.unique(np.asarray(cf_duration_train)))
    if observed != EXPECTED_TRAIN_DURATIONS:
        raise ValueError(
            "Development train durations must be exactly "
            f"{sorted(EXPECTED_TRAIN_DURATIONS)}, got {sorted(observed)}; "
            "duration=2 cannot be labelled unseen otherwise"
        )


def validate_stage3_source(
    data: Mapping[str, np.ndarray],
    source_path: str | Path,
) -> dict[str, Any]:
    """Validate the Stage 3 development source; raise before any output.

    Extends the Stage 1 source contract with the sealed-holdout rejection and
    the train-duration assertion. Callers must run this first; no output
    directory is created here.
    """
    summary = validate_source_contract(data)
    assert_not_sealed_stage2(source_path)

    split = np.asarray(data["split"])
    cf_duration = np.asarray(data["cf_duration"])
    train_mask = split == SPLIT_TRAIN
    if not train_mask.any():
        raise ValueError("Stage 3 source has no split-0 (train) episodes")
    assert_train_duration_set(cf_duration[train_mask])
    summary["train_durations"] = sorted(EXPECTED_TRAIN_DURATIONS)
    summary["unseen_duration"] = UNSEEN_DURATION
    return summary


def split_role_indices(split: np.ndarray) -> dict[str, np.ndarray]:
    """Return the train/validation/development_test index arrays by role."""
    split = np.asarray(split)
    return {
        "train": np.flatnonzero(split == SPLIT_TRAIN),
        "validation": np.flatnonzero(split == SPLIT_VALIDATION),
        "development_test": np.flatnonzero(split == SPLIT_DEVELOPMENT_TEST),
    }


def assert_role_exclusive(*index_arrays: np.ndarray) -> None:
    """Raise if any episode index appears in more than one data role."""
    seen: set[int] = set()
    for array in index_arrays:
        for index in np.asarray(array).tolist():
            if index in seen:
                raise ValueError(f"Episode index {index} assigned to more than one role")
            seen.add(index)


def compute_response_pos(factual: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """Per-agent-time position response ``R_pos[e,i,t] = ||cf_xy - f_xy||_2``.

    Shape ``E×T×N`` float32. This is the only continuous response truth and is
    derived from the paired ground truth; it is never a model input.
    """
    factual_xy = np.asarray(factual, dtype=np.float32)[
        ..., POSITION_FEATURES[0] : POSITION_FEATURES[1] + 1
    ]
    counterfactual_xy = np.asarray(counterfactual, dtype=np.float32)[
        ..., POSITION_FEATURES[0] : POSITION_FEATURES[1] + 1
    ]
    return np.linalg.norm(counterfactual_xy - factual_xy, axis=-1).astype(np.float32)


def compute_regime_labels(
    r_pos: np.ndarray,
    thresholds: Mapping[str, Any] = FROZEN_THRESHOLDS,
) -> np.ndarray:
    """Per-agent-time regime labels 0=zero/1=low/2=mid/3=high (int64 E×T×N).

    Uses the frozen Stage 1 thresholds; they are never re-fit.
    """
    return assign_response_bin(r_pos, thresholds)


def compute_onset(
    r_pos: np.ndarray,
    valid_mask: np.ndarray,
    t1: float = FROZEN_T1,
) -> tuple[np.ndarray, np.ndarray]:
    """Material-response onset: first valid time with ``R_pos > t1``.

    Returns ``(onset, censored)`` with shape ``E×N`` each. ``onset`` is the
    integer time index of first exceedance, or ``-1`` when the threshold is
    never exceeded within the valid window; ``censored`` is True exactly there.
    Onset is a secondary metric and never carries H4 or Gate C.
    """
    r_pos = np.asarray(r_pos, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if r_pos.shape != valid.shape:
        raise ValueError("r_pos and valid_mask must share shape E×T×N")
    episodes, steps, agents = r_pos.shape
    exceeds = (r_pos > t1) & valid
    onset = np.full((episodes, agents), -1, dtype=np.int64)
    censored = np.ones((episodes, agents), dtype=bool)
    any_exceed = exceeds.any(axis=1)
    first_index = exceeds.argmax(axis=1)  # first True along time
    onset[any_exceed] = first_index[any_exceed]
    censored[any_exceed] = False
    return onset, censored


def compute_effect_eval_mask(
    history: np.ndarray,
    factual: np.ndarray,
    counterfactual: np.ndarray,
    *,
    exclude_ego: bool = True,
) -> np.ndarray:
    """Boolean E×T×N evaluation mask for the effect endpoint.

    The Stage 3 primary endpoint is computed over non-ego, factual/counterfactual
    jointly-visible agent-times: history-known AND factual-present AND
    counterfactual-present AND (optionally) non-ego.
    """
    valid = compute_valid_mask(history, factual, counterfactual)
    if exclude_ego:
        is_ego = np.asarray(factual)[..., 1] > 0
        valid = valid & ~is_ego
    return valid


def compute_non_ego_mask(factual: np.ndarray) -> np.ndarray:
    """Boolean E×T×N mask that is True for non-ego slots."""
    return ~(np.asarray(factual)[..., 1] > 0)


def history_agent_known(history: np.ndarray) -> np.ndarray:
    """Re-export of the Stage 1 history-known agent mask (E×N bool)."""
    return compute_history_agent_known(history)
