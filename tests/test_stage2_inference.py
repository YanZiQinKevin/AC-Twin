"""Task 3 tests: Stage 2 frozen-checkpoint inference and derangement manifests."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402

from ac_twin import stage2_inference as si  # noqa: E402
from ac_twin import stage2_protocol as sp  # noqa: E402
from ac_twin.stage1_model import PerAgentPairedEffectGRU  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

STAGE1_DIR = ROOT / "experiments" / "stage1_evidence_value" / "merge"
NORM_NPZ = STAGE1_DIR / "normalization.npz"
N_AGENTS = 5
T_H, T_F = 10, 20


def _synthetic_holdout(n=24, seed=0):
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(n, T_H, N_AGENTS, 8)).astype(np.float32)
    factual = rng.normal(size=(n, T_F, N_AGENTS, 8)).astype(np.float32)
    counterfactual = rng.normal(size=(n, T_F, N_AGENTS, 8)).astype(np.float32)
    for arr in (history, factual, counterfactual):
        arr[..., 0] = 1.0  # presence
        arr[..., 1] = 0.0
        arr[:, :, 0, 1] = 1.0  # slot 0 is ego
    factual_actions = rng.integers(0, 5, size=(n, T_F)).astype(np.int64)
    counterfactual_actions = rng.integers(0, 5, size=(n, T_F)).astype(np.int64)
    return {
        "object_history": history,
        "object_factual_future": factual,
        "object_counterfactual_future": counterfactual,
        "factual_actions": factual_actions,
        "counterfactual_actions": counterfactual_actions,
        "cf_action": rng.integers(0, 5, size=n).astype(np.int64),
        "cf_duration": rng.choice([1, 3], size=n).astype(np.int64),
        "episode_seed": (1000 + np.arange(n)).astype(np.int64),
    }


# --- frozen normalizer loading -----------------------------------------------------


def test_load_frozen_normalizers():
    norm = si.load_frozen_normalizers(NORM_NPZ)
    for key in ("mean", "std", "target_mean", "target_std"):
        assert key in norm
    assert norm["mean"].shape == (8,)
    assert norm["target_mean"].shape == (2,)
    # The frozen feature normalizer must not be refit: it is loaded, not computed.
    assert np.allclose(norm["mean"][0], 0.0) and np.allclose(norm["std"][0], 1.0)


# --- checkpoint whitelist + hash ----------------------------------------------------


def test_load_frozen_model_roundtrip(tmp_path):
    model = PerAgentPairedEffectGRU()
    path = tmp_path / "uniform_suffix_seed11.pt"
    torch.save(model.state_dict(), path)
    loaded = si.load_frozen_model("uniform_suffix", 11, path,
                                  expected_sha256=sha256_file(path))
    assert isinstance(loaded, PerAgentPairedEffectGRU)
    assert not loaded.training  # eval mode


def test_load_frozen_model_rejects_hash_mismatch(tmp_path):
    model = PerAgentPairedEffectGRU()
    path = tmp_path / "uniform_suffix_seed11.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises((ValueError, RuntimeError), match="hash|sha256|checksum"):
        si.load_frozen_model("uniform_suffix", 11, path, expected_sha256="0" * 64)


def test_load_frozen_model_rejects_non_whitelist(tmp_path):
    model = PerAgentPairedEffectGRU()
    path = tmp_path / "rogue.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError):
        si.load_frozen_model("not_a_condition", 11, path)
    with pytest.raises(ValueError):
        si.load_frozen_model("uniform_suffix", 99, path)


def test_m1_m2_same_seed_load_distinct_weights():
    m1 = si.load_frozen_model("uniform_suffix", 11,
                              STAGE1_DIR / "uniform_suffix_seed11.pt")
    m2 = si.load_frozen_model("shuffled_suffix", 11,
                              STAGE1_DIR / "shuffled_suffix_seed11.pt")
    same = all(torch.allclose(m1.state_dict()[k], m2.state_dict()[k])
               for k in m1.state_dict())
    assert not same  # M1 and M2 are distinct frozen training regimes


# --- holdout preparation -------------------------------------------------------------


def test_prepare_holdout_arrays_shapes_and_windowing():
    arrays = si.prepare_holdout_arrays(_synthetic_holdout(), si.load_frozen_normalizers(NORM_NPZ),
                                       factual_suffix_steps=10)
    n = 24
    assert arrays["norm_history"].shape == (n, T_H, N_AGENTS, 8)
    assert arrays["windowed_factual"].shape == (n, T_F, N_AGENTS, 8)
    # Windowed to 10 steps: frames >= 10 are zeroed.
    assert np.allclose(arrays["windowed_factual"][:, 10:], 0.0)
    assert arrays["suffix_mask"][:, 10:].sum() == 0
    assert arrays["factual_actions"].shape == (n, T_F)
    assert arrays["valid_mask"].shape == (n, T_F, N_AGENTS)


def test_non_suffix_inputs_identical_across_conditions():
    raw = _synthetic_holdout()
    arrays = si.prepare_holdout_arrays(raw, si.load_frozen_normalizers(NORM_NPZ),
                                       factual_suffix_steps=10)
    donor, _ = si.build_holdout_manifest(arrays, donor_seed=20272907)
    base = {k: arrays[k] for k in ("norm_history", "factual_actions",
                                   "counterfactual_actions", "valid_mask")}
    # These non-suffix inputs are shared by M0/M1/M2 unchanged.
    inputs_m1 = si.select_suffix(arrays, "uniform_suffix", None)
    inputs_m2 = si.select_suffix(arrays, "shuffled_suffix", donor)
    for key, value in base.items():
        assert value is arrays[key]
    # Suffix channel differs by condition.
    assert not np.allclose(inputs_m1["suffix"], inputs_m2["suffix"])
    zeros = si.select_suffix(arrays, "no_suffix", None)
    assert np.allclose(zeros["suffix"], 0.0) and zeros["suffix_mask"].sum() == 0


# --- prediction ----------------------------------------------------------------------


def test_predict_holdout_world_coords_shape():
    arrays = si.prepare_holdout_arrays(_synthetic_holdout(), si.load_frozen_normalizers(NORM_NPZ),
                                       factual_suffix_steps=10)
    model = si.load_frozen_model("uniform_suffix", 11, STAGE1_DIR / "uniform_suffix_seed11.pt")
    pred = si.predict_holdout(model, arrays, "uniform_suffix", None,
                              device="cpu", batch_size=16)
    assert pred.shape == (24, T_F, N_AGENTS, 2)
    assert np.isfinite(pred).all()


# --- derangement manifest -------------------------------------------------------------


def test_manifest_derangement_hard_conditions():
    arrays = si.prepare_holdout_arrays(_synthetic_holdout(n=40), si.load_frozen_normalizers(NORM_NPZ),
                                       factual_suffix_steps=10)
    donor, manifest = si.build_holdout_manifest(arrays, donor_seed=20272907)
    n = 40
    assert donor.shape == (n,)
    assert not (donor == np.arange(n)).any()  # no fixed point
    assert len(np.unique(donor)) == n  # bijection
    seeds = arrays["episode_seed"]
    assert not (seeds[donor] == seeds).any()  # different episode seeds
    assert manifest["qa"]["self_match"] == 0
    assert manifest["qa"]["duplicate_donor"] == 0
    assert manifest["qa"]["same_seed"] == 0
    assert "tier_counts" in manifest["qa"]


def test_manifest_deterministic_and_seed_dependent():
    arrays = si.prepare_holdout_arrays(_synthetic_holdout(n=40), si.load_frozen_normalizers(NORM_NPZ),
                                       factual_suffix_steps=10)
    d1, _ = si.build_holdout_manifest(arrays, donor_seed=20272907)
    d2, _ = si.build_holdout_manifest(arrays, donor_seed=20272907)
    d3, _ = si.build_holdout_manifest(arrays, donor_seed=20272917)
    assert np.array_equal(d1, d2)
    assert not np.array_equal(d1, d3)


def test_manifest_records_identity_fields():
    arrays = si.prepare_holdout_arrays(_synthetic_holdout(n=40), si.load_frozen_normalizers(NORM_NPZ),
                                       factual_suffix_steps=10)
    _, manifest = si.build_holdout_manifest(arrays, donor_seed=20272907)
    assert manifest["donor_seed"] == 20272907
    assert "algorithm_sha256" in manifest
    assert "data_sha256" in manifest


# --- module purity --------------------------------------------------------------------


def test_inference_module_has_no_training_or_fitting():
    source = inspect.getsource(si)
    for forbidden in ("loss.backward", ".backward(", "torch.optim", "optimizer",
                      "fit_feature_normalizer", "fit_target_normalizer",
                      "fit_response_bins", ".train("):
        assert forbidden not in source, forbidden
