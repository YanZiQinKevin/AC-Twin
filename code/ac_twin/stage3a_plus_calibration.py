"""Stage 3A Plus calibrators (plan §4.1): four frozen calibration candidates.

Every calibrator fits ONLY on the validation split of the original
``merge_2000`` development source — dev/qualification labels never enter a fit
(plan Global Constraint 7). Each candidate is deterministic, persists its
parameters, fit-split, fit-sample count and input/output signatures, and a saved
calibration receipt can never be overwritten.

The four frozen candidates:

- ``CAL_SHARED_T``          : shared scalar temperature on the production-seed
                              mean-logit (``T in [0.25, 10]``);
- ``CAL_PER_SEED_T``        : one scalar temperature per seed, probs averaged;
- ``CAL_PER_SEED_VECTOR``   : per-seed positive-diagonal scale + bias on the
                              4-class logits (``z'_c = softplus(a_c) z_c + b_c``,
                              L2=1e-3, <=200 LBFGS iters, non-finite => fail);
- ``CAL_PER_SEED_T_SHRINK`` : per-seed temperature, then shrink toward the train
                              prevalence (``alpha in {0, .05, .10, .15, .20}``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.stage3_response import softmax

# Frozen candidate ids and hyper-grids (plan §4.1).
CAL_SHARED_T = "CAL_SHARED_T"
CAL_PER_SEED_T = "CAL_PER_SEED_T"
CAL_PER_SEED_VECTOR = "CAL_PER_SEED_VECTOR"
CAL_PER_SEED_T_SHRINK = "CAL_PER_SEED_T_SHRINK"
CALIBRATION_CANDIDATES = (
    CAL_SHARED_T, CAL_PER_SEED_T, CAL_PER_SEED_VECTOR, CAL_PER_SEED_T_SHRINK,
)
# Deterministic tie-break order when J_cal scores tie to 1e-8 (simpler first).
COMPLEXITY_ORDER = (CAL_SHARED_T, CAL_PER_SEED_T, CAL_PER_SEED_T_SHRINK, CAL_PER_SEED_VECTOR)

T_BOUNDS = (0.25, 10.0)
VECTOR_L2 = 1e-3
VECTOR_MAXITER = 200
SHRINKAGE_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)

_FIT_SPLIT = "validation"


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationArtifact:
    """A fitted calibrator with provenance (fit-split, sample count, signatures)."""

    method: str
    params: dict[str, Any]
    fit_split: str = _FIT_SPLIT
    n_fit: int = 0
    input_signature: str = ""
    output_signature: str = ""


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def calibration_signature(artifact: CalibrationArtifact) -> str:
    """Stable SHA-256 over method, params, fit-split/count and the input/output
    signatures — two fits that differ in ANY persisted field hash differently."""
    return canonical_fingerprint({
        "method": artifact.method,
        "params": _to_jsonable(artifact.params),
        "fit_split": artifact.fit_split,
        "n_fit": int(artifact.n_fit),
        "input_signature": artifact.input_signature,
        "output_signature": artifact.output_signature,
    })


def _array_signature(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        digest.update(a.tobytes())
        digest.update(str(a.shape).encode())
    return digest.hexdigest()


def _valid_flat(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    # A non-finite value ANYWHERE in the provided logits indicates corruption;
    # refuse even if it sits at a masked position.
    if not np.isfinite(logits).all():
        raise ValueError("calibration fit received non-finite logits")
    flat_logits = logits[mask]
    flat_labels = labels[mask]
    if flat_labels.size == 0:
        raise ValueError("calibration fit needs at least one valid sample")
    return flat_logits, flat_labels


def _nll_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    z = logits - logits.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(z).sum(axis=-1))
    p_true = np.exp(z[np.arange(labels.size), labels] - logsum)
    return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())


def _nll_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def _finish(method: str, params: dict[str, Any], n_fit: int,
            in_sig: str, out_sig: str) -> CalibrationArtifact:
    return CalibrationArtifact(
        method=method, params=params, fit_split=_FIT_SPLIT,
        n_fit=int(n_fit), input_signature=in_sig, output_signature=out_sig,
    )


# --------------------------------------------------------------------------- #
# Scalar temperature
# --------------------------------------------------------------------------- #
def fit_scalar_temperature(
    logits: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> CalibrationArtifact:
    """Fit one scalar temperature in ``[0.25, 10]`` minimising validation NLL."""
    flat_logits, flat_labels = _valid_flat(logits, labels, mask)
    in_sig = _array_signature(flat_logits, flat_labels)

    grid = np.concatenate([np.linspace(0.25, 4.0, 31), np.linspace(4.25, 10.0, 24)])
    best = min(grid, key=lambda t: _nll_from_logits(flat_logits / t, flat_labels))
    lo, hi = max(T_BOUNDS[0], best * 0.8), min(T_BOUNDS[1], best * 1.2)
    best = min(np.linspace(lo, hi, 21), key=lambda t: _nll_from_logits(flat_logits / t, flat_labels))
    best = float(np.clip(best, *T_BOUNDS))

    out_sig = _array_signature(softmax(flat_logits / best, axis=-1))
    return _finish(CAL_SHARED_T, {"T": best}, int(mask.sum()), in_sig, out_sig)


# --------------------------------------------------------------------------- #
# Vector scaling
# --------------------------------------------------------------------------- #
def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _softplus_inv(y: float) -> float:
    return float(np.log(np.expm1(y)))


def fit_vector_scaling(
    logits: np.ndarray, labels: np.ndarray, mask: np.ndarray, l2: float = VECTOR_L2
) -> CalibrationArtifact:
    """Fit per-class ``z'_c = softplus(a_c) z_c + b_c`` minimising validation NLL.

    ``a_c`` initialises so softplus(a)=1 (identity scale), ``b_c=0``; at most 200
    LBFGS iterations; any non-finite objective or result fails immediately.
    """
    flat_logits, flat_labels = _valid_flat(logits, labels, mask)
    in_sig = _array_signature(flat_logits, flat_labels)
    n_classes = flat_logits.shape[-1]

    def objective(theta: np.ndarray) -> float:
        a = theta[:n_classes]
        b = theta[n_classes:]
        scale = _softplus(a)
        z2 = flat_logits * scale + b
        val = _nll_from_logits(z2, flat_labels) + float(l2) * float(np.sum(a ** 2) + np.sum(b ** 2))
        return val if np.isfinite(val) else 1e12

    theta0 = np.concatenate([
        np.full(n_classes, _softplus_inv(1.0)), np.zeros(n_classes),
    ])
    result = minimize(
        objective, theta0, method="L-BFGS-B",
        options={"maxiter": VECTOR_MAXITER},
    )
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise FloatingPointError("vector-scaling fit produced a non-finite result")
    a = result.x[:n_classes]
    b = result.x[n_classes:]
    out_sig = _array_signature(softmax(flat_logits * _softplus(a) + b, axis=-1))
    return _finish(CAL_PER_SEED_VECTOR, {"a": a, "b": b}, int(mask.sum()), in_sig, out_sig)


# --------------------------------------------------------------------------- #
# Prevalence shrinkage (alpha-selection primitive)
# --------------------------------------------------------------------------- #
def fit_prevalence_shrinkage(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    grid: Sequence[float] = SHRINKAGE_GRID,
    *,
    train_prevalence: np.ndarray,
    base_temperature: float = 1.0,
) -> CalibrationArtifact:
    """Pick ``alpha`` from the frozen grid minimising validation NLL of the
    shrunk probabilities ``(1-alpha)*probs + alpha*train_prevalence``.

    ``probs`` are the already temperature-scaled probabilities; ``base_temperature``
    is recorded so :func:`apply_calibrator` can rebuild them from raw logits.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    prevalence = np.asarray(train_prevalence, dtype=np.float64)
    if prevalence.shape[-1] != probs.shape[-1]:
        raise ValueError("train_prevalence width must match the class count")
    flat_probs = probs[mask]
    flat_labels = labels[mask]
    if flat_labels.size == 0:
        raise ValueError("shrinkage fit needs valid samples")
    if not np.isfinite(flat_probs).all():
        raise ValueError("shrinkage fit received non-finite probs")
    in_sig = _array_signature(flat_probs, flat_labels, prevalence)

    def shrunk(alpha: float) -> np.ndarray:
        out = (1.0 - alpha) * flat_probs + alpha * prevalence
        return out / out.sum(axis=-1, keepdims=True)

    best_alpha = min((float(a) for a in grid), key=lambda a: _nll_from_probs(shrunk(a), flat_labels))
    out_sig = _array_signature(shrunk(best_alpha))
    return _finish(
        CAL_PER_SEED_T_SHRINK,
        {"T": float(base_temperature), "alpha": float(best_alpha),
         "train_prevalence": prevalence.tolist()},
        int(mask.sum()), in_sig, out_sig,
    )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def _apply_single(method: str, params: Mapping[str, Any], logits: np.ndarray,
                  prevalence: np.ndarray | None) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if method in (CAL_SHARED_T, CAL_PER_SEED_T):
        return softmax(logits / float(params["T"]), axis=-1)
    if method == CAL_PER_SEED_VECTOR:
        a = np.asarray(params["a"], dtype=np.float64)
        b = np.asarray(params["b"], dtype=np.float64)
        return softmax(logits * _softplus(a) + b, axis=-1)
    if method == CAL_PER_SEED_T_SHRINK:
        base = softmax(logits / float(params["T"]), axis=-1)
        prev = np.asarray(
            params.get("train_prevalence", prevalence), dtype=np.float64)
        if prev is None:
            raise ValueError("prevalence shrinkage needs train_prevalence")
        alpha = float(params["alpha"])
        out = (1.0 - alpha) * base + alpha * prev
        return out / out.sum(axis=-1, keepdims=True)
    raise ValueError(f"unknown calibration method: {method!r}")


def apply_calibrator(
    artifact: CalibrationArtifact,
    logits: np.ndarray,
    train_prevalence: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a fitted calibrator to raw logits -> probabilities summing to 1.

    When ``params`` holds a ``per_seed`` list, the leading axis of ``logits`` is
    the seed axis and each seed's own parameters are applied elementwise.
    """
    logits = np.asarray(logits, dtype=np.float64)
    params = artifact.params
    if "per_seed" in params:
        per_seed = params["per_seed"]
        if logits.shape[0] != len(per_seed):
            raise ValueError(
                f"per-seed calibrator has {len(per_seed)} seeds but logits have "
                f"leading dim {logits.shape[0]}"
            )
        # Shared keys (alpha, train_prevalence, ...) merge into each seed's own
        # parameters so per-seed temperature-shrinkage resolves both.
        shared = {k: v for k, v in params.items() if k != "per_seed"}
        return np.stack([
            _apply_single(artifact.method, {**shared, **per_seed[i]}, logits[i], train_prevalence)
            for i in range(len(per_seed))
        ], axis=0)
    return _apply_single(artifact.method, params, logits, train_prevalence)


# --------------------------------------------------------------------------- #
# Validation-only guard + persistence
# --------------------------------------------------------------------------- #
def fit_calibration_candidate(
    candidate_id: str,
    logits_seeds: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    train_prevalence: np.ndarray,
    *,
    n_production_seeds: int,
) -> CalibrationArtifact:
    """Fit one of the four frozen candidates on the validation split.

    ``logits_seeds`` is ``S×...×4`` over all ten fixed seeds; ``labels``/``mask``
    are the shared validation arrays. ``CAL_SHARED_T`` fits one temperature on
    the production-seed mean-logit; the per-seed candidates fit all ``S`` seeds.
    """
    from ac_twin.stage3_data import STAGE3A_PLUS_ALL_SEEDS

    logits_seeds = np.asarray(logits_seeds, dtype=np.float64)
    n_seeds = logits_seeds.shape[0]
    if n_seeds != len(STAGE3A_PLUS_ALL_SEEDS):
        raise ValueError(
            f"calibration candidates require all {len(STAGE3A_PLUS_ALL_SEEDS)} "
            f"fixed seeds, got {n_seeds}"
        )
    n_prod = int(n_production_seeds)
    prevalence = np.asarray(train_prevalence, dtype=np.float64)

    if candidate_id == CAL_SHARED_T:
        mean_logits = logits_seeds[:n_prod].mean(axis=0)
        return fit_scalar_temperature(mean_logits, labels, mask)

    if candidate_id == CAL_PER_SEED_T:
        per = [{"T": fit_scalar_temperature(logits_seeds[i], labels, mask).params["T"]}
               for i in range(n_seeds)]
        art = CalibrationArtifact(method=CAL_PER_SEED_T, params={"per_seed": per},
                                  n_fit=int(mask.sum()) * n_seeds)
        return art

    if candidate_id == CAL_PER_SEED_VECTOR:
        per = []
        for i in range(n_seeds):
            a = fit_vector_scaling(logits_seeds[i], labels, mask, l2=VECTOR_L2)
            per.append({"a": a.params["a"], "b": a.params["b"]})
        return CalibrationArtifact(method=CAL_PER_SEED_VECTOR, params={"per_seed": per},
                                   n_fit=int(mask.sum()) * n_seeds)

    if candidate_id == CAL_PER_SEED_T_SHRINK:
        per, scaled = [], []
        for i in range(n_seeds):
            t = fit_scalar_temperature(logits_seeds[i], labels, mask).params["T"]
            per.append({"T": t})
            scaled.append(softmax(logits_seeds[i] / t, axis=-1))
        pooled = np.stack(scaled, axis=0)
        s_labels = np.broadcast_to(np.asarray(labels), pooled.shape[:-1])
        s_mask = np.broadcast_to(np.asarray(mask, dtype=bool), pooled.shape[:-1])
        alpha_art = fit_prevalence_shrinkage(
            pooled, s_labels, s_mask, SHRINKAGE_GRID, train_prevalence=prevalence)
        return CalibrationArtifact(
            method=CAL_PER_SEED_T_SHRINK,
            params={"per_seed": per, "alpha": alpha_art.params["alpha"],
                    "train_prevalence": prevalence.tolist()},
            n_fit=int(mask.sum()) * n_seeds,
        )

    raise ValueError(f"unknown calibration candidate: {candidate_id!r}")


def fit_from_split(bundle: Mapping[str, Any], split: str = "validation") -> CalibrationArtifact:
    """Fit a shared temperature strictly from the validation split of ``bundle``.

    ``bundle`` carries ``logits``/``labels``/``mask`` (validation arrays) and
    ``validation_valid_count``. Any non-validation split is refused so dev or
    qualification labels can never leak into a fit.
    """
    if split != _FIT_SPLIT:
        raise ValueError(
            f"calibrators fit only on the validation split, got split={split!r}; "
            "dev/qualification labels never enter a fit"
        )
    artifact = fit_scalar_temperature(bundle["logits"], bundle["labels"], bundle["mask"])
    if int(artifact.n_fit) != int(bundle["validation_valid_count"]):
        raise ValueError("fit-sample count does not match the validation valid count")
    return artifact


def save_calibration_receipt(artifact: CalibrationArtifact, path: str | Path) -> Path:
    """Atomically write a calibration receipt; NEVER overwrite an existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"calibration receipt already exists: {path}")
    payload = {
        "method": artifact.method,
        "params": _to_jsonable(artifact.params),
        "fit_split": artifact.fit_split,
        "n_fit": int(artifact.n_fit),
        "input_signature": artifact.input_signature,
        "output_signature": artifact.output_signature,
        "calibration_signature": calibration_signature(artifact),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    import os
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
    tmp.unlink(missing_ok=True)
    return path


def load_calibration_receipt(path: str | Path) -> CalibrationArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibrationArtifact(
        method=payload["method"],
        params=payload["params"],
        fit_split=payload["fit_split"],
        n_fit=int(payload["n_fit"]),
        input_signature=payload.get("input_signature", ""),
        output_signature=payload.get("output_signature", ""),
    )
