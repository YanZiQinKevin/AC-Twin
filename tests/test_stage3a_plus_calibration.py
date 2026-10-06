"""Stage 3A Plus Task 3: calibrator fitting, application and signatures
(plan §4.1). All calibrators fit ONLY on the validation split; dev/qualification
labels never enter a fit. Every calibrator is deterministic and persists its
parameters, fit-split, fit-sample count and input/output signatures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3a_plus_calibration as cal  # noqa: E402
from ac_twin.stage3_response import softmax  # noqa: E402

rng = np.random.default_rng(7)
E, T, N, C = 6, 4, 3, 4
LOGITS = rng.normal(0, 2.0, size=(E, T, N, C))
LABELS = rng.integers(0, C, size=(E, T, N))
MASK = rng.random((E, T, N)) > 0.1
MASK[:, 0, 0] = False
TRAIN_PREVALENCE = np.array([0.55, 0.20, 0.15, 0.10])


def test_shared_temperature_bounded_deterministic_and_valid_probs():
    a = cal.fit_scalar_temperature(LOGITS, LABELS, MASK)
    b = cal.fit_scalar_temperature(LOGITS, LABELS, MASK)
    assert 0.25 <= a.params["T"] <= 10.0
    assert cal.calibration_signature(a) == cal.calibration_signature(b)
    probs = cal.apply_calibrator(a, LOGITS)
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-7)
    assert a.method == "CAL_SHARED_T"
    assert a.fit_split == "validation"
    assert a.n_fit == int(MASK.sum())


def test_vector_scaling_is_deterministic_and_probabilities_sum_to_one():
    a = cal.fit_vector_scaling(LOGITS, LABELS, MASK, l2=1e-3)
    b = cal.fit_vector_scaling(LOGITS, LABELS, MASK, l2=1e-3)
    assert cal.calibration_signature(a) == cal.calibration_signature(b)
    assert a.params["a"].shape == (C,) and a.params["b"].shape == (C,)
    np.testing.assert_allclose(cal.apply_calibrator(a, LOGITS).sum(-1), 1.0, atol=1e-7)
    # init scale == 1 => a such that softplus(a) == 1
    assert np.all(np.isfinite(a.params["a"])) and np.all(np.isfinite(a.params["b"]))


def test_vector_scaling_rejects_non_finite_objective():
    bad = LOGITS.copy()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises((ValueError, FloatingPointError)):
        cal.fit_vector_scaling(bad, LABELS, MASK, l2=1e-3)


def test_prevalence_shrinkage_picks_alpha_from_frozen_grid():
    base_probs = softmax(LOGITS / 1.5, axis=-1)
    art = cal.fit_prevalence_shrinkage(
        base_probs, LABELS, MASK, grid=(0.0, 0.05, 0.10, 0.15, 0.20),
        train_prevalence=TRAIN_PREVALENCE,
    )
    assert art.params["alpha"] in (0.0, 0.05, 0.10, 0.15, 0.20)
    # alpha = 0 reproduces the base probs; the base temperature is stored so
    # apply() can rebuild those base probs from raw logits.
    art0 = cal.fit_prevalence_shrinkage(
        base_probs, LABELS, MASK, grid=(0.0,), train_prevalence=TRAIN_PREVALENCE,
        base_temperature=1.5)
    np.testing.assert_allclose(
        cal.apply_calibrator(art0, LOGITS, TRAIN_PREVALENCE), base_probs, atol=1e-9)


def test_apply_shrinkage_shrinks_toward_prevalence():
    art = cal.CalibrationArtifact(
        method="CAL_PER_SEED_T_SHRINK",
        params={"T": 1.0, "alpha": 1.0, "train_prevalence": TRAIN_PREVALENCE.tolist()},
        fit_split="validation", n_fit=int(MASK.sum()),
    )
    probs = cal.apply_calibrator(art, LOGITS, TRAIN_PREVALENCE)
    np.testing.assert_allclose(
        probs, np.broadcast_to(TRAIN_PREVALENCE, probs.shape), atol=1e-9)


def test_apply_per_seed_collection():
    # A per-seed artifact applies each seed's own params along the leading axis.
    S = 3
    logits_seeds = rng.normal(0, 1.0, size=(S, E, T, N, C))
    art = cal.CalibrationArtifact(
        method="CAL_PER_SEED_T",
        params={"per_seed": [{"T": 1.0}, {"T": 2.0}, {"T": 0.5}]},
        fit_split="validation", n_fit=int(MASK.sum()) * S,
    )
    probs = cal.apply_calibrator(art, logits_seeds)
    assert probs.shape == logits_seeds.shape
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-7)
    np.testing.assert_allclose(probs[1], softmax(logits_seeds[1] / 2.0, axis=-1), atol=1e-9)


def test_calibrator_never_receives_dev_or_qualification_labels():
    bundle = {
        "logits": LOGITS, "labels": LABELS, "mask": MASK,
        "validation_valid_count": int(MASK.sum()),
    }
    artifact = cal.fit_from_split(bundle, split="validation")
    assert artifact.fit_split == "validation"
    assert artifact.n_fit == bundle["validation_valid_count"]
    for forbidden in ("train", "development_test", "qualification", "replacement"):
        with pytest.raises(ValueError):
            cal.fit_from_split(bundle, split=forbidden)


def test_signature_changes_with_params_and_covers_fit_count():
    a = cal.fit_scalar_temperature(LOGITS, LABELS, MASK)
    b = cal.fit_scalar_temperature(LOGITS * 2.0, LABELS, MASK)
    assert cal.calibration_signature(a) != cal.calibration_signature(b)
    # Same params but different n_fit -> different signature (tamper-evident).
    import dataclasses
    a2 = dataclasses.replace(a, n_fit=a.n_fit + 1)
    assert cal.calibration_signature(a) != cal.calibration_signature(a2)


def test_atomic_write_refuses_overwrite(tmp_path):
    a = cal.fit_scalar_temperature(LOGITS, LABELS, MASK)
    path = tmp_path / "cal.json"
    cal.save_calibration_receipt(a, path)
    assert path.is_file()
    with pytest.raises(FileExistsError):
        cal.save_calibration_receipt(a, path)  # never overwrite a calibration receipt
    loaded = cal.load_calibration_receipt(path)
    assert cal.calibration_signature(loaded) == cal.calibration_signature(a)
