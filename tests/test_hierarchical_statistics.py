from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ac_twin.statistics import hierarchical_paired_bootstrap_mean_difference


def test_hierarchical_bootstrap_reports_positive_reference_minus_candidate():
    reference = np.array([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    candidate = reference - 1.0

    result = hierarchical_paired_bootstrap_mean_difference(
        reference, candidate, n_resamples=2_000,
        confidence_level=0.95, seed=20260905,
    )

    assert result["independent_units"] == ["training_seed", "test_episode"]
    assert result["mean_paired_difference_reference_minus_candidate"] == pytest.approx(1.0)
    assert result["ci_lower"] > 0
    assert result["per_seed_mean_differences"] == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("varying_axis", ["seed_only", "episode_only"])
def test_hierarchical_interval_includes_each_independent_source_of_variation(varying_axis):
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    differences = np.tile(values[:, None] if varying_axis == "seed_only" else values[None, :],
                          (1, 8) if varying_axis == "seed_only" else (5, 1))
    result = hierarchical_paired_bootstrap_mean_difference(
        differences, np.zeros_like(differences), n_resamples=2000, confidence_level=0.95, seed=7)
    assert result["ci_lower"] < 2.0 < result["ci_upper"]
    assert result["ci_upper"] - result["ci_lower"] > 1.0


@pytest.mark.parametrize("candidate", [np.ones(3), np.ones((3, 4))])
def test_hierarchical_bootstrap_rejects_invalid_pairs(candidate):
    with pytest.raises(ValueError, match=r"identical \[S,N\] shape"):
        hierarchical_paired_bootstrap_mean_difference(
            np.ones((2, 3)), candidate,
            n_resamples=100, confidence_level=0.95, seed=1,
        )


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, True])
def test_hierarchical_bootstrap_rejects_invalid_resample_count(n_resamples):
    with pytest.raises(ValueError, match="n_resamples"):
        hierarchical_paired_bootstrap_mean_difference(
            np.ones((2, 2)), np.zeros((2, 2)),
            n_resamples=n_resamples, confidence_level=0.95, seed=1,
        )


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, float("nan"), True])
def test_hierarchical_bootstrap_rejects_invalid_confidence_level(confidence_level):
    with pytest.raises(ValueError, match="confidence_level"):
        hierarchical_paired_bootstrap_mean_difference(
            np.ones((2, 2)), np.zeros((2, 2)),
            n_resamples=100, confidence_level=confidence_level, seed=1,
        )


def test_hierarchical_bootstrap_is_reproducible_and_does_not_mutate_inputs():
    reference = np.array([[4.0, 7.0, 8.0], [5.0, 5.0, 9.0]])
    candidate = np.array([[2.0, 6.0, 5.0], [4.0, 2.0, 8.0]])
    reference_before, candidate_before = reference.copy(), candidate.copy()

    first = hierarchical_paired_bootstrap_mean_difference(
        reference, candidate, n_resamples=500, confidence_level=0.9, seed=7,
    )
    second = hierarchical_paired_bootstrap_mean_difference(
        reference, candidate, n_resamples=500, confidence_level=0.9, seed=7,
    )

    assert first == second
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(candidate, candidate_before)


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_hierarchical_bootstrap_requires_two_seeds_and_two_episodes(shape):
    with pytest.raises(ValueError, match="two seeds and two episodes"):
        hierarchical_paired_bootstrap_mean_difference(
            np.ones(shape), np.zeros(shape),
            n_resamples=100, confidence_level=0.95, seed=1,
        )


def _condition_result(training_seeds, ade, fde):
    return {
        "training_seeds": training_seeds,
        "ade_by_seed_episode": ade,
        "fde_by_seed_episode": fde,
    }


def test_hierarchical_report_uses_fixed_contrasts_and_preserves_seed_mapping():
    from run_identity_continuity_comparison import (
        PRIMARY_CONTRASTS,
        SECONDARY_CONTRASTS,
        build_hierarchical_contrast_report,
    )

    seeds = [101, 202]
    stable = _condition_result(seeds, [[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]])
    results = {
        "stable_temporal_65k": stable,
        "permuted_temporal_65k": _condition_result(seeds, [[3.0, 4.0], [6.0, 8.0]], [[4.0, 5.0], [7.0, 9.0]]),
        "distance_temporal_65k": _condition_result(seeds, [[2.0, 3.0], [5.0, 6.0]], [[3.0, 4.0], [6.0, 7.0]]),
        "set_pool_65k": _condition_result(seeds, [[1.5, 2.5], [3.5, 4.5]], [[2.5, 3.5], [4.5, 5.5]]),
    }

    report = build_hierarchical_contrast_report(
        results, {"bootstrap_resamples": 200, "confidence_level": 0.95},
    )

    assert PRIMARY_CONTRASTS == [
        ("permuted_temporal", "stable_temporal"),
        ("distance_temporal", "stable_temporal"),
    ]
    assert SECONDARY_CONTRASTS == [("set_pool", "stable_temporal")]
    tier = report["65k"]
    assert list(tier["primary"]) == [
        "permuted_temporal_minus_stable_temporal",
        "distance_temporal_minus_stable_temporal",
    ]
    comparison = tier["primary"]["permuted_temporal_minus_stable_temporal"]
    assert comparison["reference_condition"] == "permuted_temporal"
    assert comparison["candidate_condition"] == "stable_temporal"
    assert comparison["training_seeds"] == seeds
    assert comparison["ade"]["per_seed_mean_differences"] == pytest.approx([2.0, 3.5])
    assert comparison["ade"]["independent_units"] == ["training_seed", "test_episode"]


def test_hierarchical_report_rejects_misaligned_training_seed_order():
    from run_identity_continuity_comparison import build_hierarchical_contrast_report

    values = [[1.0, 2.0], [3.0, 4.0]]
    results = {
        f"{condition}_65k": _condition_result([11, 22], values, values)
        for condition in ("stable_temporal", "permuted_temporal", "distance_temporal", "set_pool")
    }
    results["permuted_temporal_65k"]["training_seeds"] = [22, 11]

    with pytest.raises(ValueError, match="training seed order"):
        build_hierarchical_contrast_report(
            results, {"bootstrap_resamples": 100, "confidence_level": 0.95},
        )


def test_hierarchical_report_marks_single_seed_as_unavailable_not_episode_only():
    from run_identity_continuity_comparison import build_hierarchical_contrast_report

    values = [[1.0, 2.0]]
    report = build_hierarchical_contrast_report({
        f"{condition}_65k": _condition_result([11], values, values)
        for condition in ("stable_temporal", "permuted_temporal", "distance_temporal", "set_pool")
    }, {"bootstrap_resamples": 100, "confidence_level": 0.95})

    comparison = report["65k"]["primary"]["permuted_temporal_minus_stable_temporal"]
    assert comparison["status"] == "unavailable"
    assert comparison["independent_units"] == ["training_seed", "test_episode"]
    assert "ci_lower" not in comparison
