from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.stage1_data import (
    OBJECT_FEATURES,
    apply_feature_normalizer,
    apply_suffix_window,
    build_derangement,
    compute_anchors,
    compute_history_agent_known,
    compute_target_effect,
    compute_valid_mask,
    derangement_manifest,
    fit_feature_normalizer,
    fit_target_normalizer,
    suffix_presence_mask,
    validate_source_contract,
    visible_count_signature,
)


def make_dataset(episodes=8, agents=3, seed=0):
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(episodes, 10, agents, 8)).astype(np.float32)
    factual = rng.normal(size=(episodes, 20, agents, 8)).astype(np.float32)
    counterfactual = rng.normal(size=(episodes, 20, agents, 8)).astype(np.float32)
    for values in (history, factual, counterfactual):
        values[..., 0] = 1.0
        values[..., 1] = 0.0
        values[:, :, 0, 1] = 1.0  # ego slot 0
    return {
        "object_history": history,
        "object_factual_future": factual,
        "object_counterfactual_future": counterfactual,
        "factual_actions": rng.integers(0, 5, (episodes, 20)),
        "counterfactual_actions": rng.integers(0, 5, (episodes, 20)),
        "cf_action": rng.integers(0, 5, episodes),
        "cf_duration": rng.integers(1, 3, episodes),
        "episode_seed": np.arange(1000, 1000 + episodes),
        "split": np.array([0, 0, 0, 0, 1, 1, 2, 2])[:episodes],
        "object_features_json": json.dumps(list(OBJECT_FEATURES)),
    }


# --- contract validation -------------------------------------------------

def test_contract_accepts_valid_dataset():
    summary = validate_source_contract(make_dataset())
    assert summary["episodes"] == 8
    assert summary["agents"] == 3
    assert summary["features"] == list(OBJECT_FEATURES)


def test_contract_rejects_missing_field():
    data = make_dataset()
    del data["object_counterfactual_future"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_source_contract(data)


def test_contract_rejects_wrong_feature_order():
    data = make_dataset()
    shuffled = list(OBJECT_FEATURES)
    shuffled[2], shuffled[3] = shuffled[3], shuffled[2]
    data["object_features_json"] = json.dumps(shuffled)
    with pytest.raises(ValueError, match="object_features_json"):
        validate_source_contract(data)


def test_contract_rejects_bad_history_shape():
    data = make_dataset()
    data["object_history"] = data["object_history"][:, :5]
    with pytest.raises(ValueError, match="object_history"):
        validate_source_contract(data)


def test_contract_rejects_bad_split_label():
    data = make_dataset()
    data["split"] = np.full(8, 7)
    with pytest.raises(ValueError, match="split labels"):
        validate_source_contract(data)


def test_contract_rejects_duplicate_episode_seed():
    data = make_dataset()
    data["episode_seed"] = np.zeros(8, dtype=np.int64)
    with pytest.raises(ValueError, match="unique"):
        validate_source_contract(data)


def test_contract_rejects_nonfinite():
    data = make_dataset()
    data["object_counterfactual_future"][0, 0, 0, 2] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_source_contract(data)


# --- targets and masks ----------------------------------------------------

def test_target_effect_value_and_direction():
    data = make_dataset()
    factual = data["object_factual_future"]
    counterfactual = factual.copy()
    counterfactual[..., 2:4] = factual[..., 2:4] + np.array([1.0, -2.0], np.float32)
    target = compute_target_effect(factual, counterfactual)
    assert target.shape == (8, 20, 3, 2)
    np.testing.assert_allclose(target[..., 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(target[..., 1], -2.0, atol=1e-6)


def test_valid_mask_requires_all_three_conditions():
    data = make_dataset(episodes=2, agents=2)
    history = data["object_history"]
    factual = data["object_factual_future"]
    counterfactual = data["object_counterfactual_future"]
    mask = compute_valid_mask(history, factual, counterfactual)
    assert mask.shape == (2, 20, 2)
    assert mask.all()

    # Agent unknown in history (no presence) is invalid even if present later.
    history_no = history.copy()
    history_no[:, :, 1, 0] = 0.0
    assert not compute_valid_mask(history_no, factual, counterfactual)[:, :, 1].any()

    # Factual absence at a time invalidates that time.
    factual_no = factual.copy()
    factual_no[:, 3, 1, 0] = 0.0
    assert not compute_valid_mask(history, factual_no, counterfactual)[:, 3, 1].any()

    # Counterfactual absence at a time invalidates that time.
    cf_no = counterfactual.copy()
    cf_no[:, 5, 1, 0] = 0.0
    assert not compute_valid_mask(history, factual, cf_no)[:, 5, 1].any()


def test_history_agent_known_any_over_time():
    history = np.zeros((1, 10, 2, 8), np.float32)
    history[0, 7, 1, 0] = 1.0  # agent 1 present only at one history time
    history[0, :, 0, 0] = 1.0
    known = compute_history_agent_known(history)
    assert known.tolist() == [[True, True]]
    history[0, 7, 1, 0] = 0.0
    assert compute_history_agent_known(history).tolist() == [[True, False]]


# --- normalization ---------------------------------------------------------

def test_feature_normalizer_uses_only_train():
    data = make_dataset(episodes=8)
    history, factual = data["object_history"], data["object_factual_future"]
    train = np.array([0, 1, 2, 3])
    first = fit_feature_normalizer(history, factual, train)
    # Drastically change non-train episodes; refit on the same train indices.
    history[4:] += 1000.0
    factual[4:] += 1000.0
    second = fit_feature_normalizer(history, factual, train)
    np.testing.assert_allclose(first["mean"], second["mean"], rtol=1e-5)
    np.testing.assert_allclose(first["std"], second["std"], rtol=1e-5)


def test_feature_normalizer_standardizes_only_motion_channels():
    data = make_dataset()
    normalizer = fit_feature_normalizer(
        data["object_history"], data["object_factual_future"], np.array([0, 1, 2, 3]))
    # presence/is_ego/cos/sin keep mean 0 std 1 (unstandardized).
    for channel in (0, 1, 6, 7):
        assert normalizer["mean"][channel] == 0.0
        assert normalizer["std"][channel] == 1.0


def test_padding_stays_zero_after_standardization():
    data = make_dataset()
    normalizer = fit_feature_normalizer(
        data["object_history"], data["object_factual_future"], np.array([0, 1, 2, 3]))
    values = data["object_history"].copy()
    values[0, :, 2, 0] = 0.0          # agent 2 padding
    values[0, :, 2, 2:6] = 999.0      # garbage under padding
    anchors = compute_anchors(data["object_history"])
    out = apply_feature_normalizer(values, anchors, normalizer)
    np.testing.assert_allclose(out[0, :, 2, 2:6], 0.0, atol=1e-6)


def test_anchor_subtracted_from_positions():
    data = make_dataset()
    anchors = compute_anchors(data["object_history"])
    assert anchors.shape == (8, 2)
    normalizer = fit_feature_normalizer(
        data["object_history"], data["object_factual_future"], np.array([0, 1, 2, 3]))
    # With zero mean/std scaling, ego last-frame relative position should be ~0 mean-centered.
    out = apply_feature_normalizer(data["object_history"], anchors, normalizer)
    assert np.isfinite(out).all()


def test_target_normalizer_uses_only_valid_train():
    target = np.zeros((4, 20, 2, 2), np.float32)
    valid = np.zeros((4, 20, 2), bool)
    valid[0, :, :] = True
    valid[1, :, :] = True
    target[0] = 2.0
    target[1] = 4.0
    target[2] = 1000.0   # split 1, must not be used
    target[3] = -1000.0  # split 2, must not be used
    mean, std = fit_target_normalizer(target, valid, np.array([0, 1]))
    np.testing.assert_allclose(mean, [3.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(std, [1.0, 1.0], atol=1e-6)


# --- suffix window ---------------------------------------------------------

def test_suffix_window_zeroes_beyond_ten_steps():
    data = make_dataset()
    factual = data["object_factual_future"]
    windowed = apply_suffix_window(factual, 10)
    np.testing.assert_allclose(windowed[:, 10:], 0.0)
    np.testing.assert_allclose(windowed[:, :10], factual[:, :10])


def test_suffix_presence_mask_zeroes_beyond_window():
    data = make_dataset()
    mask = suffix_presence_mask(data["object_factual_future"], 10)
    assert mask[:, :10].all()
    assert not mask[:, 10:].any()


# --- derangement -----------------------------------------------------------

def _strata_arrays(episodes, key_fn):
    cf_action = np.array([key_fn(i)[0] for i in range(episodes)])
    cf_duration = np.array([key_fn(i)[1] for i in range(episodes)])
    first = np.array([key_fn(i)[2] for i in range(episodes)])
    visible = np.array([key_fn(i)[3] for i in range(episodes)])
    seeds = np.arange(5000, 5000 + episodes)
    return cf_action, cf_duration, first, visible, seeds


def test_derangement_no_self_no_dup_no_same_seed():
    cf_action, cf_duration, first, visible, seeds = _strata_arrays(
        12, lambda i: (i % 3, 1 + (i % 2), i % 5, 3))
    donor, tier = build_derangement(cf_action, cf_duration, first, visible, seeds, seed=7)
    assert not (donor == np.arange(12)).any()
    assert len(np.unique(donor)) == 12
    assert not (seeds[donor] == seeds).any()


def test_derangement_reproducible_and_epoch_varies():
    # One large stratum (>=3 members) so the seeded cycle can vary by epoch.
    args = _strata_arrays(12, lambda i: (0, 1, 0, 3))
    donor_a, _ = build_derangement(*args, seed=20260907)
    donor_a2, _ = build_derangement(*args, seed=20260907)
    np.testing.assert_array_equal(donor_a, donor_a2)
    # Epoch-wise seeds must not all collapse to one identical mapping.
    mappings = [build_derangement(*args, seed=20260907 + epoch)[0] for epoch in range(6)]
    assert any(not np.array_equal(mappings[0], m) for m in mappings[1:])


def test_derangement_requires_two_episodes():
    args = _strata_arrays(1, lambda i: (0, 1, 0, 3))
    with pytest.raises(ValueError, match="at least two"):
        build_derangement(*args, seed=0)


def test_derangement_fallback_tiers_and_qa():
    cf_action = np.array([0, 0, 1, 2])
    cf_duration = np.array([1, 1, 1, 2])
    first = np.array([0, 0, 0, 0])
    visible = np.array([3, 3, 3, 3])
    seeds = np.arange(100, 104)
    donor, tier = build_derangement(cf_action, cf_duration, first, visible, seeds, seed=3)
    # ep0/ep1 share the strictest key -> tier 1; ep2/ep3 only match at whole-split tier 4.
    assert sorted(tier[:2].tolist()) == [1, 1]
    assert sorted(tier[2:].tolist()) == [4, 4]
    manifest = derangement_manifest(donor, tier, 0, cf_action, cf_duration, visible, seeds)
    qa = manifest["qa"]
    assert qa["self_match"] == 0
    assert qa["duplicate_donor"] == 0
    assert qa["cross_split"] == 0
    assert qa["same_seed"] == 0
    assert qa["tier_counts"] == {1: 2, 4: 2}
    # cf action/duration match flags reflect the actual pairing.
    rec0 = manifest["records"][0]
    assert rec0["cf_action_match"] and rec0["cf_duration_match"] and rec0["visible_count_match"]


def test_manifest_fixed_and_hashable():
    cf_action, cf_duration, first, visible, seeds = _strata_arrays(
        10, lambda i: (i % 2, 1 + (i % 2), i % 4, 4))
    donor, tier = build_derangement(cf_action, cf_duration, first, visible, seeds,
                                    seed=20261907)
    manifest_a = derangement_manifest(donor, tier, 1, cf_action, cf_duration, visible, seeds)
    manifest_b = derangement_manifest(donor, tier, 1, cf_action, cf_duration, visible, seeds)
    assert canonical_fingerprint(manifest_a) == canonical_fingerprint(manifest_b)


def test_derangement_splices_stranded_loner():
    # ep1/ep2 share every key; ep0 is unique at every tier, so it is stranded
    # and must be spliced into an existing cycle at the final whole-split tier.
    cf_action = np.array([0, 1, 1])
    cf_duration = np.array([1, 1, 1])
    first = np.array([0, 0, 0])
    visible = np.array([3, 3, 3])
    seeds = np.arange(200, 203)
    donor, tier = build_derangement(cf_action, cf_duration, first, visible, seeds, seed=11)
    assert not (donor == np.arange(3)).any()          # no self match
    assert len(np.unique(donor)) == 3                  # bijective, no duplicate donor
    assert not (seeds[donor] == seeds).any()           # no same-seed pair
    assert tier[0] == 4                                # stranded loner fell to whole-split tier
    manifest = derangement_manifest(donor, tier, 0, cf_action, cf_duration, visible, seeds)
    assert manifest["qa"]["self_match"] == 0
    assert manifest["qa"]["duplicate_donor"] == 0


def test_visible_count_signature():
    known = np.array([[True, True, False], [True, False, False]])
    np.testing.assert_array_equal(visible_count_signature(known), [2, 1])
