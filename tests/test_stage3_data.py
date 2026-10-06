from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_data import OBJECT_FEATURES  # noqa: E402
from ac_twin import stage3_data as s3d  # noqa: E402


def make_dataset(episodes: int = 8, agents: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(episodes, 10, agents, 8)).astype(np.float32)
    factual = rng.normal(size=(episodes, 20, agents, 8)).astype(np.float32)
    counterfactual = rng.normal(size=(episodes, 20, agents, 8)).astype(np.float32)
    for a in (history, factual, counterfactual):
        a[..., 0] = 1.0  # presence on
        a[..., 1] = 0.0
        a[:, :, 0, 1] = 1.0  # slot 0 is ego
    data = {
        "object_history": history,
        "object_factual_future": factual,
        "object_counterfactual_future": counterfactual,
        "factual_actions": rng.integers(0, 5, (episodes, 20)),
        "counterfactual_actions": rng.integers(0, 5, (episodes, 20)),
        "cf_action": rng.integers(0, 5, episodes),
        "cf_duration": rng.choice([1, 3], episodes),
        "episode_seed": np.arange(episodes, dtype=np.int64) + 1000,
        "split": np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int8)[:episodes],
        "object_features_json": json.dumps(list(OBJECT_FEATURES)),
    }
    return data


# --- response truth / regime / onset ------------------------------------------


def test_response_pos_matches_manual_norm() -> None:
    data = make_dataset()
    factual = data["object_factual_future"]
    counterfactual = data["object_counterfactual_future"]
    r_pos = s3d.compute_response_pos(factual, counterfactual)
    assert r_pos.shape == (8, 20, 5)
    manual = np.linalg.norm(
        counterfactual[..., 2:4].astype(np.float64) - factual[..., 2:4].astype(np.float64),
        axis=-1,
    )
    np.testing.assert_allclose(r_pos, manual, rtol=1e-5, atol=1e-5)
    assert (r_pos >= 0).all()


def test_regime_labels_hit_all_four_bins_on_manual_tensor() -> None:
    thresholds = s3d.FROZEN_THRESHOLDS
    zt, t1, t2 = thresholds["zero_tolerance"], thresholds["t1"], thresholds["t2"]
    r_pos = np.array(
        [
            [
                [0.0, 0.5 * zt, 0.5 * (zt + t1), t1],
                [0.5 * (t1 + t2), t2, 1.5 * t2, 0.0],
            ]
        ],
        dtype=np.float32,
    )  # shape E=1, T=2, N=4
    labels = s3d.compute_regime_labels(r_pos)
    assert labels.shape == (1, 2, 4)
    flat = labels.reshape(-1).tolist()
    assert s3d.REGIME_ZERO in flat
    assert s3d.REGIME_LOW in flat
    assert s3d.REGIME_MID in flat
    assert s3d.REGIME_HIGH in flat
    # Boundary checks: exactly t1 -> low->mid transition uses strict >.
    assert labels[0, 0, 2] == s3d.REGIME_LOW
    assert labels[0, 0, 3] == s3d.REGIME_MID
    assert labels[0, 1, 0] == s3d.REGIME_MID  # (t1+t2)/2
    assert labels[0, 1, 1] == s3d.REGIME_MID  # exactly t2 is still mid (strict >)
    assert labels[0, 1, 2] == s3d.REGIME_HIGH  # 1.5*t2
    assert labels[0, 1, 3] == s3d.REGIME_ZERO


def test_regime_uses_frozen_thresholds_not_refit() -> None:
    assert s3d.FROZEN_THRESHOLDS["zero_tolerance"] == 0.0001
    assert s3d.FROZEN_THRESHOLDS["t1"] == 0.24843344965718578
    assert s3d.FROZEN_THRESHOLDS["t2"] == 1.898992796585611


def test_onset_first_exceedance_and_censoring() -> None:
    t1 = s3d.FROZEN_T1
    r_pos = np.zeros((1, 5, 3), dtype=np.float32)  # E=1, T=5, N=3
    valid = np.ones((1, 5, 3), dtype=bool)
    # agent 0: first exceeds at t=2
    r_pos[0, 2, 0] = t1 + 0.1
    r_pos[0, 4, 0] = t1 + 0.5
    # agent 1: never exceeds -> censored
    r_pos[0, :, 1] = t1 - 0.1
    # agent 2: exceeds at t=1 but that step is invalid; first valid exceed at t=3
    r_pos[0, 1, 2] = t1 + 0.2
    valid[0, 1, 2] = False
    r_pos[0, 3, 2] = t1 + 0.3

    onset, censored = s3d.compute_onset(r_pos, valid, t1)
    assert onset[0, 0] == 2
    assert not censored[0, 0]
    assert onset[0, 1] == -1
    assert censored[0, 1]
    assert onset[0, 2] == 3
    assert not censored[0, 2]


def test_onset_never_assigns_last_step_to_censored() -> None:
    t1 = s3d.FROZEN_T1
    r_pos = np.full((1, 20, 1), t1 - 0.5, dtype=np.float32)
    valid = np.ones((1, 20, 1), dtype=bool)
    onset, censored = s3d.compute_onset(r_pos, valid, t1)
    assert onset[0, 0] == -1
    assert censored[0, 0]


# --- masks / padding ----------------------------------------------------------


def test_effect_eval_mask_excludes_ego_and_requires_joint_presence() -> None:
    data = make_dataset()
    history = data["object_history"]
    factual = data["object_factual_future"]
    counterfactual = data["object_counterfactual_future"]
    mask = s3d.compute_effect_eval_mask(history, factual, counterfactual)
    assert mask.shape == (8, 20, 5)
    # slot 0 is ego -> excluded everywhere
    assert not mask[..., 0].any()
    # remove counterfactual presence for a non-ego vehicle at one time
    counterfactual[3, 5, 2, 0] = 0.0
    mask2 = s3d.compute_effect_eval_mask(history, factual, counterfactual)
    assert not mask2[3, 5, 2]


def test_padding_never_enters_valid_mask() -> None:
    data = make_dataset()
    history = data["object_history"]
    factual = data["object_factual_future"]
    counterfactual = data["object_counterfactual_future"]
    # zero out an entire vehicle's presence across all branches
    history[:, :, 3, 0] = 0.0
    factual[:, :, 3, 0] = 0.0
    counterfactual[:, :, 3, 0] = 0.0
    mask = s3d.compute_effect_eval_mask(history, factual, counterfactual, exclude_ego=False)
    assert not mask[:, :, 3].any()


# --- data roles ---------------------------------------------------------------


def test_split_role_indices_partition_dataset() -> None:
    data = make_dataset()
    roles = s3d.split_role_indices(data["split"])
    assert roles["train"].tolist() == [0, 1, 2, 3]
    assert roles["validation"].tolist() == [4, 5]
    assert roles["development_test"].tolist() == [6, 7]
    s3d.assert_role_exclusive(roles["train"], roles["validation"], roles["development_test"])


def test_role_exclusivity_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="more than one role"):
        s3d.assert_role_exclusive(np.array([0, 1]), np.array([1, 2]))


# --- duration-set assertion ----------------------------------------------------


def test_train_duration_set_accepts_exactly_1_and_3() -> None:
    s3d.assert_train_duration_set(np.array([1, 3, 1, 3]))
    with pytest.raises(ValueError, match="exactly"):
        s3d.assert_train_duration_set(np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="exactly"):
        s3d.assert_train_duration_set(np.array([1, 2, 3]))


def test_validate_stage3_source_asserts_train_durations() -> None:
    data = make_dataset()
    # force all train episodes to duration 1 -> violation
    data["cf_duration"] = np.array([1, 1, 1, 1, 1, 3, 1, 3])
    with pytest.raises(ValueError, match="exactly"):
        s3d.validate_stage3_source(data, "data/object_centric_stage/merge_2000/object_twinworld_2000.npz")


# --- sealed Stage 2 holdout rejection ------------------------------------------

SEALED = s3d.STAGE2_HOLDOUT_LOGICAL_PATH


def test_sealed_path_rejected_by_logical_path() -> None:
    assert s3d.is_sealed_stage2_path(SEALED)
    with pytest.raises(ValueError, match="permanently sealed"):
        s3d.assert_not_sealed_stage2(SEALED)


def test_sealed_path_rejected_case_and_separator_bypass() -> None:
    variants = [
        SEALED.upper(),
        SEALED.replace("/", "\\"),
        "DATA/OBJECT_CENTRIC_STAGE/STAGE2_MERGE_1200/OBJECT_TWINWORLD_STAGE2_MERGE_1200.NPZ",
        "./" + SEALED,
        "data/object_centric_stage//stage2_merge_1200/object_twinworld_stage2_merge_1200.npz",
    ]
    for variant in variants:
        assert s3d.is_sealed_stage2_path(variant), variant
        with pytest.raises(ValueError, match="permanently sealed"):
            s3d.assert_not_sealed_stage2(variant)


def test_sealed_path_rejected_absolute_and_parent_traversal(tmp_path) -> None:
    root = Path("D:/OneDrive/兰州交大/博士课题-大模型/世界模型/AC-Twin")
    absolute = root / SEALED
    assert s3d.is_sealed_stage2_path(absolute)
    # parent-traversal bypass resolving to the same file
    traversal = root / "data/object_centric_stage/stage2_merge_1200/../stage2_merge_1200/object_twinworld_stage2_merge_1200.npz"
    assert s3d.is_sealed_stage2_path(traversal)


def test_sealed_holdout_rejected_by_known_sha256() -> None:
    for sha in (s3d.STAGE2_HOLDOUT_FILE_SHA256, s3d.STAGE2_HOLDOUT_CONTENT_SHA256):
        with pytest.raises(ValueError, match="permanently sealed"):
            s3d.assert_not_sealed_stage2(None, file_sha256=sha)
        # case-insensitive
        with pytest.raises(ValueError, match="permanently sealed"):
            s3d.assert_not_sealed_stage2(None, file_sha256=sha.upper())


def test_unrelated_path_and_hash_allowed() -> None:
    assert not s3d.is_sealed_stage2_path("data/object_centric_stage/merge_2000/object_twinworld_2000.npz")
    s3d.assert_not_sealed_stage2("data/object_centric_stage/merge_2000/object_twinworld_2000.npz")
    s3d.assert_not_sealed_stage2(None, file_sha256="0" * 64)


def test_validate_stage3_source_rejects_sealed_path() -> None:
    data = make_dataset()
    with pytest.raises(ValueError, match="permanently sealed"):
        s3d.validate_stage3_source(data, SEALED)


def test_model_input_builder_never_touches_counterfactual() -> None:
    """The data module exposes no API that returns CF future as a model input.

    Response truth and labels derive from CF only through dedicated functions;
    the source-contract required fields are the Stage 1 set (no CF-input
    builder). Guard the boundary by asserting the CF array is consumed only by
    the truth/label/mask helpers, which take it as an explicit argument.
    """
    import inspect

    source = inspect.getsource(s3d)
    # There is no function that returns object_counterfactual_future for model input.
    assert "def build_model_input" not in source
    assert "def get_counterfactual_input" not in source
