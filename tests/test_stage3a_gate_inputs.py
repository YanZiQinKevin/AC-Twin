from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage3_response import (  # noqa: E402
    MultinomialRegimeClassifier,
    TrainPrevalenceClassifier,
)
from ac_twin.stage3_scaler import FeatureScaler  # noqa: E402
from ac_twin.stage3_training import (  # noqa: E402
    STAGE3A_CONTRACT_ID,
    checkpoint_signature,
    compute_run_fingerprint,
    save_predictor_checkpoint,
    verify_checkpoint_integrity,
    load_predictor_state,
)
import torch  # noqa: E402


# --- classification baselines -------------------------------------------------- #
def _clf_features(episodes=6, horizon=10, agents=5, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "eval_shape": (episodes, horizon, agents),
        "n_eval": episodes,
        "horizon": horizon,
        "cf_action": rng.integers(0, 5, episodes),
        "cf_duration": rng.choice([1, 3], episodes).astype(float),
        "initial_distance": rng.uniform(0, 20, (episodes, agents)),  # per-vehicle (E,N)
    }


def test_prevalence_classifier_matches_train_distribution() -> None:
    feats = _clf_features()
    labels = np.zeros(feats["eval_shape"], dtype=np.int64)
    labels[..., 0] = 1  # some low
    valid = np.ones(feats["eval_shape"], dtype=bool)
    clf = TrainPrevalenceClassifier().fit(labels, feats, valid)
    probs = clf.predict_proba(feats)
    assert probs.shape == feats["eval_shape"] + (4,)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-9)
    # prevalence of class 0 dominates
    assert probs[..., 0].mean() > 0.79


def test_multinomial_classifier_cells_normalize() -> None:
    feats = _clf_features()
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, feats["eval_shape"])
    valid = np.ones(feats["eval_shape"], dtype=bool)
    clf = MultinomialRegimeClassifier().fit(labels, feats, valid)
    probs = clf.predict_proba(feats)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-9)


def test_multinomial_unseen_cell_falls_back_to_prevalence() -> None:
    feats = _clf_features(episodes=2)
    labels = np.zeros(feats["eval_shape"], dtype=np.int64)
    valid = np.ones(feats["eval_shape"], dtype=bool)
    clf = MultinomialRegimeClassifier().fit(labels, feats, valid)
    # query with an action never seen in train
    feats2 = dict(feats)
    feats2["cf_action"] = np.array([4, 4])
    feats2["cf_duration"] = np.array([3.0, 3.0])
    probs = clf.predict_proba(feats2)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-9)


# --- inference episode alignment (Task 2, repair plan v0.1) -------------------- #
HIST, FUT, N_AGENTS, N_FEAT = 6, 8, 4, 8


def _eval_tensors(episodes: int = 10) -> dict:
    """Synthetic scaled tensors with an episode marker in history[..., 0, 2]."""
    g = torch.Generator().manual_seed(7)
    tensors = {
        "history": torch.randn(episodes, HIST, N_AGENTS, N_FEAT, generator=g),
        "factual_suffix": torch.randn(episodes, FUT, N_AGENTS, N_FEAT, generator=g),
        "factual_actions": torch.randint(0, 5, (episodes, FUT), generator=g),
        "counterfactual_actions": torch.randint(0, 5, (episodes, FUT), generator=g),
        "cf_action": torch.randint(0, 5, (episodes,), generator=g),
        "cf_duration": torch.randint(0, 2, (episodes,), generator=g) * 2 + 1,  # {1,3}
        "t_star": torch.randint(2, 5, (episodes,), generator=g),
    }
    for e in range(episodes):
        tensors["history"][e, :, 0, 2] = float(e)  # unique episode marker
    return tensors


class _MarkerModel(torch.nn.Module):
    """Stub predictor echoing the episode marker into r_hat and logits."""

    def forward(self, history, factual_suffix, factual_actions,
                counterfactual_actions, cf_action, cf_duration, t_star):
        B, T, N, _ = factual_suffix.shape
        marker = history[:, -1, 0, 2].clamp_min(0.0)  # (B,)
        log1p_r = torch.log1p(marker).view(B, 1, 1).expand(B, T, N).contiguous()
        logits = torch.zeros(B, T, N, 4)
        logits[..., 1] = marker.view(B, 1, 1)  # raw marker in logits channel 1
        return {"log1p_r_hat": log1p_r, "regime_logits": logits}


def test_predict_split_preserves_requested_episode_order() -> None:
    import run_stage3a as drv

    tensors = _eval_tensors(10)
    requested = np.array([7, 2, 9, 1])  # shuffled, non-contiguous
    pred = drv._predict_split(_MarkerModel(), tensors, requested, "cpu")
    np.testing.assert_array_equal(pred["episode_indices"], requested)
    # r_hat carries the episode marker -> order must equal the request exactly
    np.testing.assert_allclose(pred["r_hat"][:, 0, 0], requested.astype(np.float64), rtol=1e-5)
    np.testing.assert_allclose(
        pred["regime_logits"][:, 0, 0, 1], requested.astype(np.float64), rtol=1e-5
    )


def test_predict_split_invariant_to_eval_batch_size() -> None:
    import run_stage3a as drv
    from ac_twin.stage3_response import Stage3ResponsePredictor

    torch.manual_seed(0)
    model = Stage3ResponsePredictor(hidden_dim=16, response_feature_dim=8)
    tensors = _eval_tensors(10)
    requested = np.array([7, 2, 9, 1, 0, 5])
    reference = drv._predict_split(model, tensors, requested, "cpu", batch_size=1)
    for bs in (2, 3, len(requested)):
        pred = drv._predict_split(model, tensors, requested, "cpu", batch_size=bs)
        np.testing.assert_array_equal(pred["episode_indices"], requested)
        # identical up to float32 batching non-associativity (~1e-8 level)
        np.testing.assert_allclose(pred["r_hat"], reference["r_hat"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(pred["regime_logits"], reference["regime_logits"], rtol=1e-6, atol=1e-6)


def test_assert_episode_order_rejects_misalignment() -> None:
    import run_stage3a as drv

    expected = np.array([0, 1, 2, 3])
    drv._assert_episode_order({"episode_indices": np.array([0, 1, 2, 3])}, expected)
    with pytest.raises(RuntimeError, match="episode order"):
        drv._assert_episode_order({"episode_indices": np.array([3, 2, 1, 0])}, expected)
    with pytest.raises(RuntimeError, match="episode order"):
        drv._assert_episode_order({"episode_indices": np.array([0, 1])}, expected)


# --- run fingerprint ------------------------------------------------------------- #
def test_fingerprint_covers_all_inputs() -> None:
    base = dict(
        data_signature="d", split_name="train", config={"a": 1},
        scaler_signature="s", class_weights=[1, 1, 1, 1],
    )
    fp = compute_run_fingerprint(**base)
    assert fp != compute_run_fingerprint(**{**base, "data_signature": "d2"})
    assert fp != compute_run_fingerprint(**{**base, "split_name": "validation"})
    assert fp != compute_run_fingerprint(**{**base, "config": {"a": 2}})
    assert fp != compute_run_fingerprint(**{**base, "scaler_signature": "s2"})
    assert fp != compute_run_fingerprint(**{**base, "class_weights": [2, 1, 1, 1]})
    assert fp != compute_run_fingerprint(**{**base, "contract_id": "other"})
    # deterministic
    assert fp == compute_run_fingerprint(**base)


# --- predictor checkpoint integrity --------------------------------------------- #
def _state(seed):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(3, 3, generator=g)}


def test_checkpoint_save_verify_roundtrip(tmp_path) -> None:
    sd = _state(0)
    rec = save_predictor_checkpoint(tmp_path / "m.pt", sd, meta={"seed": 0})
    out = verify_checkpoint_integrity(tmp_path / "m.pt", expected_signature=rec["signature"])
    assert out["signature"] == rec["signature"]
    loaded = load_predictor_state(tmp_path / "m.pt", expected_signature=rec["signature"])
    assert checkpoint_signature(loaded) == rec["signature"]


def test_verify_detects_wrong_expected_signature(tmp_path) -> None:
    save_predictor_checkpoint(tmp_path / "m.pt", _state(0), meta={})
    with pytest.raises(ValueError, match="does not match expected"):
        verify_checkpoint_integrity(tmp_path / "m.pt", expected_signature="0" * 64)


def test_verify_detects_tampered_state(tmp_path) -> None:
    path = tmp_path / "m.pt"
    rec = save_predictor_checkpoint(path, _state(0), meta={})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state_dict"]["w"] = torch.zeros(3, 3)  # tamper, keep stored sig
    torch.save(payload, path)
    with pytest.raises(ValueError, match="integrity failure"):
        verify_checkpoint_integrity(path)
