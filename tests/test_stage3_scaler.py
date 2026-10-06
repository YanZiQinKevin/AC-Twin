from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_data import MOTION_FEATURES, N_FEATURES  # noqa: E402
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402


def _block(episodes=4, steps=10, agents=5, seed=0):
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(episodes, steps, agents, N_FEATURES)).astype(np.float64)
    block[..., 0] = (block[..., 0] > 0).astype(np.float64)  # presence binary
    block[..., 1] = (block[..., 1] > 0).astype(np.float64)  # is_ego binary
    return block


def test_fit_standardizes_motion_features_only() -> None:
    block = _block()
    mask = np.ones(block.shape[:3], dtype=bool)
    scaler = FeatureScaler.fit({"history": block}, {"history": mask})
    out = scaler.transform(block)
    for feat in MOTION_FEATURES:
        vals = out[..., feat][mask]
        assert abs(vals.mean()) < 1e-6
        assert abs(vals.std() - 1.0) < 1e-6
    # non-motion features untouched
    for feat in (0, 1, 6, 7):
        np.testing.assert_allclose(out[..., feat], block[..., feat])


def test_fit_respects_valid_mask() -> None:
    block = _block()
    mask = np.zeros(block.shape[:3], dtype=bool)
    mask[:, :, 0] = True  # only agent 0 valid
    scaler = FeatureScaler.fit({"history": block}, {"history": mask})
    expected = block[:, :, 0, 2].mean()
    assert abs(scaler.mean[2] - expected) < 1e-9


def test_signature_stable_and_save_load_roundtrip(tmp_path) -> None:
    block = _block()
    mask = np.ones(block.shape[:3], dtype=bool)
    scaler = FeatureScaler.fit({"history": block}, {"history": mask}, fitted_on="train")
    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = FeatureScaler.load(path)
    assert loaded.signature() == scaler.signature()
    np.testing.assert_allclose(loaded.mean, scaler.mean)
    np.testing.assert_allclose(loaded.std, scaler.std)


def test_load_rejects_corrupted_scaler(tmp_path) -> None:
    block = _block()
    mask = np.ones(block.shape[:3], dtype=bool)
    scaler = FeatureScaler.fit({"history": block}, {"history": mask})
    path = tmp_path / "scaler.json"
    scaler.save(path)
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"][2] += 1.0  # tamper
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        FeatureScaler.load(path)


def test_refit_not_possible_on_loaded_scaler(tmp_path) -> None:
    block = _block()
    mask = np.ones(block.shape[:3], dtype=bool)
    scaler = FeatureScaler.fit({"history": block}, {"history": mask})
    assert scaler._frozen  # constructed scaler cannot be re-fit in place
