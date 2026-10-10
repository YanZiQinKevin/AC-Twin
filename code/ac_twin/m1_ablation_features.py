"""M1 supervised-baseline ablation: legal inputs, arm masks, scalers, identity-bound caches.

Plan ``AC-TWIN-RFER-M1-SUPERVISED-ABLATION-2026-10-07-V0.2`` sections 3/4/5/9.2.

The score side reuses the frozen Stage 5 build (``stage5_features.build_feature_context``),
which itself reuses the frozen R2 feature code path; it never opens the counterfactual
truth array. Labels/truth live in physically separate caches and are never consumed by a
score-side path.

Arm definitions (frozen; plan section 4):
  * FULL_MATCHED     — the original 315-dim input, main-fitted scaler.
  * BASELINE3_MATCHED — only the three standard baseline scalars
    ``[MARGIN_ONLY, SEED_VOTE_ENTROPY, SEED_MARGIN_STD]`` (taken verbatim from
    ``build_relative_tokens``'s ``baselines``), standardised with main-only mean/std
    (floor 1e-12), placed in the first three coordinates of a 315-dim vector; the
    remaining 312 coordinates are exact zeros.
  * NO_RISK_MATCHED  — the full input with the five explicit risk channels masked:
    per 34-dim token, columns 5..19 (the 15 ``*_risk_*`` continuous columns) are zeroed
    AFTER the main-fitted scaler transform; 135 of 315 coordinates masked, 180 kept.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from ac_twin import m1_ablation_protocol as m1p
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage5_features as s5feat
from ac_twin.checkpointing import atomic_save_npz
from ac_twin.stage1_data import sha256_file

FLAT_DIM = s4cp.N_COMPETITORS * s4cpf.TOKEN_DIM + s4cpf.GLOBAL_DIM  # 315
_TOKEN_FLAT_DIM = s4cp.N_COMPETITORS * s4cpf.TOKEN_DIM             # 306


def risk_token_columns() -> np.ndarray:
    """Token-local indices of the 15 explicit risk columns, asserted by schema NAME.

    The mask is located by the frozen schema names (``{kind}_risk_{class}``), never by
    trusting a bare column number (plan 4.2).
    """
    expected = [
        f"{kind}_risk_{k}" for k in s4cp.RISK_CLASSES
        for kind in ("selected", "competitor", "delta")]
    located = [s4cpf.TOKEN_CONT_NAMES.index(name) for name in expected]
    if located != list(range(5, 20)):
        raise RuntimeError(
            f"risk-channel schema drift: names resolve to {located}, expected 5..19")
    return np.asarray(located, dtype=np.int64)


RISK_TOKEN_COLS = risk_token_columns()
KEPT_TOKEN_COLS = np.asarray(
    [i for i in range(s4cpf.TOKEN_DIM) if i not in set(RISK_TOKEN_COLS.tolist())],
    dtype=np.int64)
assert KEPT_TOKEN_COLS.size == 19  # 5 cost/vote/rank + 14 one-hot


def risk_flat_columns() -> np.ndarray:
    """Flattened-input (315-dim) indices masked for NO_RISK: 9 tokens x 15 columns."""
    offsets = np.arange(s4cp.N_COMPETITORS, dtype=np.int64) * s4cpf.TOKEN_DIM
    return (offsets[:, None] + RISK_TOKEN_COLS[None, :]).ravel()


# --- legal role build ------------------------------------------------------------------------


def build_legal_role(sources: Mapping[str, Any], role: str,
                     *, expected_groups: int) -> dict[str, Any]:
    """Build the legal feature bundle for one role; fail closed on any invalid group.

    Returns raw (unscaled) tokens/global features, the three baseline scalars (verbatim
    from the frozen builder) and the in-memory group seeds (never persisted).
    """
    bundle, extras = s5feat.build_feature_context(sources, role)
    if not bool(bundle.valid_mask.all()):
        raise RuntimeError(
            f"{role}: non-valid groups present — the R2 roles are fully valid; "
            "any invalid group is a contract drift (fail closed)")
    tokens = np.asarray(bundle.tokens, dtype=np.float64)
    global_features = np.asarray(bundle.global_features, dtype=np.float64)
    if tokens.shape != (expected_groups, s4cp.N_COMPETITORS, s4cpf.TOKEN_DIM):
        raise RuntimeError(f"{role}: token shape drift {tokens.shape}")
    baselines = extras["baselines"]
    baseline3 = np.column_stack([
        np.asarray(baselines[name], dtype=np.float64) for name in m1p.BASELINE3_ORDER])
    for name, array in (("tokens", tokens), ("global_features", global_features),
                        ("baseline3", baseline3)):
        if not np.isfinite(array).all():
            raise RuntimeError(f"{role}: non-finite legal feature array {name}")
    return {
        "role": str(role),
        "tokens": tokens,
        "global_features": global_features,
        "baseline3": baseline3,
        "group_seed": np.asarray(bundle.group_seed, dtype=np.int64).ravel(),
    }


# --- scalers (main-only) ----------------------------------------------------------------------


def fit_baseline3_scaler(baseline3_main: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(baseline3_main, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(m1p.BASELINE3_ORDER) or values.shape[0] < 2:
        raise ValueError("baseline3 scaler needs (n>=2, 3) main rows")
    return {
        "mean": values.mean(axis=0),
        "std": np.maximum(values.std(axis=0), 1e-12),
    }


# --- arm inputs --------------------------------------------------------------------------------


def apply_no_risk_mask(tokens_scaled: np.ndarray) -> np.ndarray:
    """Zero the 15 explicit risk columns of every token AFTER standardisation."""
    masked = np.asarray(tokens_scaled, dtype=np.float64).copy()
    masked[:, :, RISK_TOKEN_COLS] = 0.0
    return masked


def arm_flat_inputs(arm: str,
                    tokens: np.ndarray,
                    global_features: np.ndarray,
                    baseline3: np.ndarray,
                    *,
                    full_scaler: s4cpf.FrozenFeatureScaler,
                    baseline3_scaler: Mapping[str, np.ndarray],
                    ) -> np.ndarray:
    """The 315-dim model input for one arm under the frozen definitions."""
    arm = str(arm)
    groups = np.asarray(tokens).shape[0]
    if arm == "FULL_MATCHED":
        x_tok = full_scaler.transform_tokens(tokens)
        x_glob = full_scaler.transform_global(global_features)
        flat = np.concatenate(
            [x_tok.reshape(groups, _TOKEN_FLAT_DIM), x_glob], axis=1)
    elif arm == "NO_RISK_MATCHED":
        x_tok = apply_no_risk_mask(full_scaler.transform_tokens(tokens))
        x_glob = full_scaler.transform_global(global_features)
        flat = np.concatenate(
            [x_tok.reshape(groups, _TOKEN_FLAT_DIM), x_glob], axis=1)
    elif arm == "BASELINE3_MATCHED":
        z = (np.asarray(baseline3, dtype=np.float64) - baseline3_scaler["mean"]) \
            / baseline3_scaler["std"]
        flat = np.zeros((groups, FLAT_DIM), dtype=np.float64)
        flat[:, : len(m1p.BASELINE3_ORDER)] = z
    else:
        raise ValueError(f"unknown arm: {arm!r}")
    if flat.shape != (groups, FLAT_DIM) or not np.isfinite(flat).all():
        raise RuntimeError(f"{arm}: illegal flat input shape/content {flat.shape}")
    return flat


def unflatten(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 315-dim flat input back into (G,9,34) tokens and (G,9) globals."""
    flat = np.asarray(flat, dtype=np.float64)
    groups = flat.shape[0]
    return (flat[:, :_TOKEN_FLAT_DIM].reshape(groups, s4cp.N_COMPETITORS, s4cpf.TOKEN_DIM),
            flat[:, _TOKEN_FLAT_DIM:])


def active_input_dims() -> dict[str, int]:
    """Active (non-zero-by-contract) input coordinates per arm (plan 4.3 reporting)."""
    return {
        "FULL_MATCHED": FLAT_DIM,
        "NO_RISK_MATCHED": FLAT_DIM - int(risk_flat_columns().size),
        "BASELINE3_MATCHED": len(m1p.BASELINE3_ORDER),
    }


# --- labels (truth side; physically separate from the score side) ----------------------------


def build_role_labels(sources: Mapping[str, Any], role: str) -> dict[str, np.ndarray]:
    """Training/evaluation labels via the frozen Stage 5 truth side (R2-checked)."""
    truth = s5feat.compute_group_truth(sources, role)
    regret = np.asarray(truth["regret"], dtype=np.float64).ravel()
    y_reg = np.log1p(regret / m1p.HIGH_REGRET_THRESHOLD)
    y_high = (regret > m1p.HIGH_REGRET_THRESHOLD).astype(np.float64)
    if not (np.isfinite(y_reg).all() and np.isfinite(y_high).all()):
        raise RuntimeError(f"{role}: non-finite labels")
    return {
        "y_reg": y_reg,
        "y_high": y_high,
        "regret": regret,
        "high_regret": y_high.astype(np.int64),
        "idle_regret": np.asarray(truth["idle_regret"], dtype=np.float64).ravel(),
        "random_regret": np.asarray(truth["random_regret"], dtype=np.float64).ravel(),
        "group_seed": np.asarray(truth["group_seed"], dtype=np.int64).ravel(),
    }


def _display_path(source_path: Path) -> str:
    try:
        return m1p._normalised(Path(source_path).relative_to(m1p.ROOT))
    except ValueError:
        return m1p._normalised(source_path)


# --- identity-bound caches (plan 9.2) ----------------------------------------------------------


def save_legal_cache(path: str | Path, role: str, source_path: Path,
                     built: Mapping[str, Any]) -> None:
    """Persist legal features WITHOUT raw group seeds, bound to the source identity."""
    group_seed = np.asarray(built["group_seed"], dtype=np.int64).ravel()
    row_index = np.arange(group_seed.size, dtype=np.int64)
    arrays = {
        "tokens": np.asarray(built["tokens"], dtype=np.float32),
        "global_features": np.asarray(built["global_features"], dtype=np.float32),
        "baseline3": np.asarray(built["baseline3"], dtype=np.float64),
        "source_row_index": row_index,
        "group_id": m1p.make_group_ids(role, row_index),
        "role": np.asarray(str(role)),
        "source_relpath": np.asarray(_display_path(source_path)),
        "source_sha256": np.asarray(sha256_file(source_path)),
        "source_identity_sha256": np.asarray(m1p.identity_digest(group_seed)),
    }
    m1p.assert_cache_keys_legal(list(arrays.keys()), context=f"legal cache {path}")
    atomic_save_npz(Path(path), **arrays)


def _load_bound_cache(path: str | Path, *, role: str, source_path: Path,
                      expected_rows: int, context: str) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        m1p.assert_cache_keys_legal(list(archive.files), context=context)
        record = {key: archive[key] for key in archive.files}
    binding = {
        "role": str(record["role"]),
        "source_sha256": str(record["source_sha256"]),
        "source_identity_sha256": str(record["source_identity_sha256"]),
        "source_row_index": record["source_row_index"],
        "group_id": record["group_id"],
    }
    group_seed = m1p.validate_binding(
        binding, role=role, source_path=source_path,
        expected_rows=expected_rows, context=context)
    return {"record": record, "group_seed": group_seed}


def load_legal_cache(path: str | Path, *, role: str, source_path: Path,
                     expected_rows: int) -> dict[str, Any]:
    loaded = _load_bound_cache(path, role=role, source_path=source_path,
                               expected_rows=expected_rows, context=f"legal cache {path}")
    record = loaded["record"]
    return {
        "role": role,
        "tokens": np.asarray(record["tokens"], dtype=np.float64),
        "global_features": np.asarray(record["global_features"], dtype=np.float64),
        "baseline3": np.asarray(record["baseline3"], dtype=np.float64),
        "group_seed": loaded["group_seed"],
    }


def save_label_cache(path: str | Path, role: str, source_path: Path,
                     labels: Mapping[str, np.ndarray]) -> None:
    """Persist truth-side labels, bound identically; never read by a score-side path."""
    group_seed = np.asarray(labels["group_seed"], dtype=np.int64).ravel()
    row_index = np.arange(group_seed.size, dtype=np.int64)
    arrays = {
        "y_reg": np.asarray(labels["y_reg"], dtype=np.float64),
        "y_high": np.asarray(labels["y_high"], dtype=np.float64),
        "regret": np.asarray(labels["regret"], dtype=np.float64),
        "idle_regret": np.asarray(labels["idle_regret"], dtype=np.float64),
        "random_regret": np.asarray(labels["random_regret"], dtype=np.float64),
        "source_row_index": row_index,
        "group_id": m1p.make_group_ids(role, row_index),
        "role": np.asarray(str(role)),
        "source_relpath": np.asarray(_display_path(source_path)),
        "source_sha256": np.asarray(sha256_file(source_path)),
        "source_identity_sha256": np.asarray(m1p.identity_digest(group_seed)),
        "truth_side": np.asarray("EVALUATION_ONLY"),
    }
    m1p.assert_cache_keys_legal(list(arrays.keys()), context=f"label cache {path}")
    atomic_save_npz(Path(path), **arrays)


def load_label_cache(path: str | Path, *, role: str, source_path: Path,
                     expected_rows: int) -> dict[str, Any]:
    loaded = _load_bound_cache(path, role=role, source_path=source_path,
                               expected_rows=expected_rows, context=f"label cache {path}")
    record = loaded["record"]
    if str(record.get("truth_side", "")) != "EVALUATION_ONLY":
        raise ValueError(f"label cache {path}: truth-side marker missing")
    return {
        "y_reg": np.asarray(record["y_reg"], dtype=np.float64),
        "y_high": np.asarray(record["y_high"], dtype=np.float64),
        "regret": np.asarray(record["regret"], dtype=np.float64),
        "idle_regret": np.asarray(record["idle_regret"], dtype=np.float64),
        "random_regret": np.asarray(record["random_regret"], dtype=np.float64),
        "group_seed": loaded["group_seed"],
    }


__all__ = [
    "FLAT_DIM", "KEPT_TOKEN_COLS", "RISK_TOKEN_COLS", "active_input_dims",
    "apply_no_risk_mask", "arm_flat_inputs", "build_legal_role", "build_role_labels",
    "fit_baseline3_scaler", "load_legal_cache", "load_label_cache", "risk_flat_columns",
    "risk_token_columns", "save_legal_cache", "save_label_cache", "unflatten",
]
