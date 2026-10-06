"""Stage 3A Plus Task 2/6: unified ten-seed diagnostics, calibration-failure
classification, Gate R+ core/stability and hierarchical bootstrap.

Statistical contracts live in ``ac_twin.stage3a_plus_statistics`` and are pure
numpy so they are unit-testable without torch or training. The decomposition is
episode-aligned and calibrated-probability based; classification is mechanical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3a_plus_statistics as sps  # noqa: E402
from ac_twin.stage3_response import softmax  # noqa: E402

SEED_LABELS = {"101", "202", "303", "404", "505", "606", "707", "808", "909", "1010"}


def _tiny_arrays(n_episodes: int = 8, horizon: int = 4, agents: int = 3, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    E, T, N = n_episodes, horizon, agents
    r_true = rng.gamma(1.5, 0.4, size=(E, T, N)).astype(np.float64)
    regime = np.digitize(r_true, [0.25, 1.0, 1.9]).astype(np.int64)  # 0..3
    valid = rng.random((E, T, N)) > 0.1
    valid[:, 0, 0] = False  # ego slot excluded
    # Ten seeds of predictions: base + per-seed jitter.
    base_r = r_true + rng.normal(0, 0.05, size=(E, T, N))
    r_hat_seeds = np.stack([np.clip(base_r + rng.normal(0, 0.02, size=(E, T, N)), 0, None)
                            for _ in range(10)], axis=0)
    logits = rng.normal(0, 1.0, size=(E, T, N, 4))
    logits_seeds = np.stack([logits + rng.normal(0, 0.1, size=(E, T, N, 4))
                             for _ in range(10)], axis=0)
    calibrated = np.stack([softmax(logits_seeds[s], axis=-1) for s in range(10)], axis=0)
    episode_ids = np.arange(E)
    horizons = np.broadcast_to(np.arange(T)[None, :, None], (E, T, N)).copy()
    cf_actions = rng.integers(0, 5, size=E)
    cf_durations = rng.choice([1, 3], size=E)
    base_clf = np.broadcast_to(softmax(logits, axis=-1), (E, T, N, 4)).copy()
    baselines = {
        "continuous": {"strong": np.zeros((E, T, N))},
        "classification": {"strong": base_clf},
        "strongest_continuous": "strong",
        "strongest_brier": "strong",
    }
    return {
        "r_hat_seeds": r_hat_seeds,
        "logits_seeds": logits_seeds,
        "calibrated_probs_seeds": calibrated,
        "r_true": r_true,
        "regime_labels": regime,
        "valid_mask": valid,
        "episode_ids": episode_ids,
        "horizons": horizons,
        "cf_actions": cf_actions,
        "cf_durations": cf_durations,
        "baselines": baselines,
        "n_resamples": 200,
        "bootstrap_seed": 1234,
    }


# --------------------------------------------------------------------------- #
# decompose_seed_metrics
# --------------------------------------------------------------------------- #
def test_decompose_reports_all_ten_seeds_and_brier_cells():
    report = sps.decompose_seed_metrics(**_tiny_arrays())
    assert set(report["per_seed"]) == SEED_LABELS
    assert {"regime", "horizon", "action", "duration"} <= set(report["brier_decomposition"])
    for seed in SEED_LABELS:
        entry = report["per_seed"][seed]
        for key in ("mae", "spearman", "pr_auc_presence", "pr_auc_high_vs_lowmid",
                    "brier", "nll", "ece", "i_brier", "i_brier_ci_lower", "i_brier_ci_upper"):
            assert key in entry, f"per_seed[{seed}] missing {key}"


def test_decompose_per_seed_carries_gate_stability_fields():
    """Gate R+_stability consumes per-seed baseline-relative core metrics plus
    degeneracy/finiteness flags directly from the decomposition."""
    fx = _tiny_arrays()
    report = sps.decompose_seed_metrics(**fx)
    clf_base = fx["baselines"]["classification"]["strong"]
    labels, valid = fx["regime_labels"], fx["valid_mask"]
    presence_lab = sps.binary_labels(labels, (1, 2, 3))
    pos_mask = valid & (presence_lab == 1)
    high_lab = sps.binary_labels(labels, (3,))
    from ac_twin.stage3_response import pr_auc
    base_pr_presence = pr_auc(sps.presence_scores(clf_base), presence_lab, valid)
    base_pr_high = pr_auc(sps.high_vs_lowmid_scores(clf_base), high_lab, pos_mask)
    for i, seed in enumerate(sorted(SEED_LABELS, key=int)):
        entry = report["per_seed"][seed]
        for key in ("i_mae", "i_pr_presence", "i_pr_high_vs_lowmid",
                    "non_degenerate", "finite"):
            assert key in entry, f"per_seed[{seed}] missing {key}"
        s_idx = sps.STAGE3A_PLUS_ALL_SEEDS.index(int(seed))
        probs_s = fx["calibrated_probs_seeds"][s_idx]
        assert entry["i_pr_presence"] == pytest.approx(
            pr_auc(sps.presence_scores(probs_s), presence_lab, valid) - base_pr_presence,
            nan_ok=True)
        assert entry["i_pr_high_vs_lowmid"] == pytest.approx(
            pr_auc(sps.high_vs_lowmid_scores(probs_s), high_lab, pos_mask) - base_pr_high,
            nan_ok=True)
        assert entry["non_degenerate"] == bool(
            valid.any() and np.std(fx["r_hat_seeds"][s_idx][valid]) > sps.NONDEGENERATE_STD_FLOOR)
        assert entry["finite"] is True


def test_decompose_reliability_has_15_equal_mass_bins():
    report = sps.decompose_seed_metrics(**_tiny_arrays())
    rel = report["reliability"]
    assert rel["n_bins"] == 15
    assert len(rel["bins"]) == 15
    total = sum(b["count"] for b in rel["bins"])
    assert total > 0
    for b in rel["bins"]:
        assert {"count", "mean_confidence", "accuracy", "gap"} <= set(b)


def test_decompose_calibration_semantics_present():
    report = sps.decompose_seed_metrics(**_tiny_arrays())
    sem = report["calibration_semantics"]
    for key in ("total_variation", "argmax_flip_rate", "entropy_uncalibrated", "entropy_calibrated"):
        assert key in sem
    # Identity calibration (calibrated == softmax(logits)) => zero TV and flips.
    assert sem["total_variation"] == pytest.approx(0.0, abs=1e-9)
    assert sem["argmax_flip_rate"] == pytest.approx(0.0, abs=1e-9)


def test_decompose_requires_ten_seeds():
    fx = _tiny_arrays()
    fx["r_hat_seeds"] = fx["r_hat_seeds"][:5]
    fx["logits_seeds"] = fx["logits_seeds"][:5]
    fx["calibrated_probs_seeds"] = fx["calibrated_probs_seeds"][:5]
    with pytest.raises(ValueError):
        sps.decompose_seed_metrics(**fx)


def test_decompose_rejects_non_finite_predictions():
    fx = _tiny_arrays()
    fx["calibrated_probs_seeds"][0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        sps.decompose_seed_metrics(**fx)


def test_decompose_episode_alignment_hard_assert():
    fx = _tiny_arrays()
    fx["episode_ids"] = fx["episode_ids"][::-1].copy()  # order mismatch vs r_true
    fx["expected_episode_ids"] = np.arange(fx["r_true"].shape[0])
    with pytest.raises((ValueError, RuntimeError)):
        sps.decompose_seed_metrics(**fx)


# --------------------------------------------------------------------------- #
# classify_calibration_failure
# --------------------------------------------------------------------------- #
def _diag(*, eng=True, improve_a=False, improve_b=False, ensemble_ok=True,
          conc=0.3, ci_a=(-0.006, 0.001), ci_b=(-0.006, 0.001)):
    return {
        "engineering_ok": eng,
        "per_seed_cal_improves_303": {"A": improve_a, "B": improve_b},
        "ensemble_core_not_degraded_under_per_seed_cal": ensemble_ok,
        "max_subgroup_concentration": conc,
        "seed303": {
            "A": {"i_brier_ci_lower": ci_a[0], "i_brier_ci_upper": ci_a[1]},
            "B": {"i_brier_ci_lower": ci_b[0], "i_brier_ci_upper": ci_b[1]},
        },
    }


def test_classify_borderline_seed_variation():
    d = _diag(ci_a=(-0.006, 0.001), ci_b=(-0.006, 0.001))
    assert sps.classify_calibration_failure(d) == "BORDERLINE_SEED_VARIATION"


def test_classify_run_invalid_engineering():
    assert sps.classify_calibration_failure(_diag(eng=False)) == "RUN_INVALID_ENGINEERING"


def test_classify_shared_temperature_mismatch():
    d = _diag(improve_a=True, improve_b=True, ensemble_ok=True)
    assert sps.classify_calibration_failure(d) == "SHARED_TEMPERATURE_MISMATCH"
    # If the ensemble core IS degraded, it is not a clean shared-temperature fix.
    d2 = _diag(improve_a=True, improve_b=True, ensemble_ok=False, ci_a=(-0.006, 0.001))
    assert sps.classify_calibration_failure(d2) != "SHARED_TEMPERATURE_MISMATCH"


def test_classify_regime_localized():
    d = _diag(conc=0.85, ci_a=(-0.006, -0.001), ci_b=(-0.006, -0.001))
    assert sps.classify_calibration_failure(d) == "REGIME_LOCALIZED_CALIBRATION_FAILURE"


def test_classify_systematic():
    d = _diag(conc=0.3, ci_a=(-0.006, -0.001), ci_b=(-0.007, -0.002))
    assert sps.classify_calibration_failure(d) == "SYSTEMATIC_SEED_CALIBRATION_FAILURE"


def test_classify_mixed():
    # A is positive-and-excludes-0; B excludes 0 below but A does not -> not systematic,
    # no CI contains 0, no engineering/subgroup -> MIXED.
    d = _diag(conc=0.3, ci_a=(0.001, 0.005), ci_b=(-0.007, -0.002))
    assert sps.classify_calibration_failure(d) == "MIXED_CALIBRATION_FAILURE"


def test_direction_consistency_helper():
    a = {"per_seed": {s: {"i_brier": 0.01} for s in SEED_LABELS}}
    b = {"per_seed": {s: {"i_brier": 0.02} for s in SEED_LABELS}}
    assert sps.direction_consistency(a, b, metric="i_brier")["fraction_same_sign"] == 1.0
    b2 = {"per_seed": {s: {"i_brier": (-0.02 if s == "303" else 0.02)} for s in SEED_LABELS}}
    out = sps.direction_consistency(a, b2, metric="i_brier")
    assert out["fraction_same_sign"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Task 6: Gate R+ core/stability + hierarchical bootstrap
# --------------------------------------------------------------------------- #
def _stability_fixture(brier, baseline=0.06, hier_ci_lower=0.005, core_positive=True,
                       nondegenerate=True):
    seeds = [str(s) for s in sps.STAGE3A_PLUS_ALL_SEEDS]
    per_seed = {}
    for i, s in enumerate(seeds):
        per_seed[s] = {
            "i_mae": 0.05 if core_positive else -0.01,
            "spearman": 0.5 if core_positive else -0.1,
            "i_pr_presence": 0.4,
            "i_pr_high_vs_lowmid": 0.2,
            "i_brier": float(brier[i]),
            "non_degenerate": bool(nondegenerate),
            "finite": True,
        }
    return {
        "per_seed": per_seed,
        "baseline_brier": float(baseline),
        "hierarchical_i_brier": {
            "point": float(np.mean(brier)),
            "ci_lower": float(hier_ci_lower),
            "ci_upper": float(np.mean(brier) + 0.01),
        },
    }


def test_stability_allows_two_small_negative_brier_seeds_with_positive_population_ci():
    r = _stability_fixture(brier=[.01] * 8 + [-.001, -.001], baseline=.06)
    assert sps.evaluate_gate_r_plus_stability(r)["passed"]


def test_stability_rejects_materially_harmful_worst_seed():
    r = _stability_fixture(brier=[.01] * 9 + [-.004], baseline=.06)
    assert not sps.evaluate_gate_r_plus_stability(r)["checks"]["worst_seed_noninferior"]


def test_seven_positive_brier_seeds_never_passes():
    r = _stability_fixture(brier=[.01] * 7 + [-.001] * 3, baseline=.06)
    assert not sps.evaluate_gate_r_plus_stability(r)["checks"]["at_least_8_of_10_positive"]


def test_stability_requires_all_core_metrics_positive_per_seed():
    r = _stability_fixture(brier=[.01] * 10, core_positive=False)
    out = sps.evaluate_gate_r_plus_stability(r)
    assert not out["checks"]["per_seed_core_metrics_positive"]
    assert not out["passed"]


def test_stability_requires_positive_median_and_hierarchical_ci():
    r = _stability_fixture(brier=[.01] * 10, hier_ci_lower=-0.001)
    assert not sps.evaluate_gate_r_plus_stability(r)["checks"]["hierarchical_ci_lower_gt0"]
    # median exactly zero is not > 0
    r2 = _stability_fixture(brier=[.01] * 5 + [-.01] * 5, baseline=1.0)
    assert not sps.evaluate_gate_r_plus_stability(r2)["checks"]["median_i_brier_positive"]


def test_stability_requires_nondegenerate_seeds():
    r = _stability_fixture(brier=[.01] * 10, nondegenerate=False)
    assert not sps.evaluate_gate_r_plus_stability(r)["checks"]["all_seeds_nondegenerate"]


def test_hierarchical_bootstrap_resamples_seeds_and_episodes():
    rng = np.random.default_rng(0)
    values = rng.normal(0.01, 0.002, size=(10, 40))  # (S,E) per-seed per-episode I_Brier
    out = sps.hierarchical_seed_episode_bootstrap(values, n_resamples=500, seed=7)
    assert out["ci_lower"] < out["point"] < out["ci_upper"]
    assert out["n_resamples"] == 500
    # Deterministic given the seed.
    out2 = sps.hierarchical_seed_episode_bootstrap(values, n_resamples=500, seed=7)
    assert out["ci_lower"] == out2["ci_lower"]


def _core_fixture(ci_lower=0.01, ece=0.04, ece_baseline=0.05):
    boot = {"point": ci_lower + 0.01, "ci_lower": ci_lower, "ci_upper": ci_lower + 0.02}
    return {
        "i_mae": dict(boot), "spearman_boot": dict(boot), "i_pr_presence": dict(boot),
        "i_pr_high_vs_lowmid": dict(boot), "i_brier": dict(boot),
        "ece": ece, "ece_uncalibrated": ece_baseline,
    }


def test_gate_core_all_ci_lower_positive_and_ece():
    assert sps.evaluate_gate_r_plus_core(_core_fixture())["passed"]
    assert not sps.evaluate_gate_r_plus_core(_core_fixture(ci_lower=-0.001))["passed"]


def test_gate_core_ece_guard_or_relative_improvement():
    # ECE above 0.05 but >=20% improved over uncalibrated passes.
    ok = _core_fixture(ece=0.06, ece_baseline=0.10)  # 0.06 <= 0.8*0.10
    assert sps.evaluate_gate_r_plus_core(ok)["checks"]["ece_calibrated"]
    bad = _core_fixture(ece=0.09, ece_baseline=0.10)  # 0.09 > 0.08 and > 0.05
    assert not sps.evaluate_gate_r_plus_core(bad)["checks"]["ece_calibrated"]


def test_gate_r_plus_and_of_core_stability_q():
    good_core = _core_fixture()
    good_stab = _stability_fixture(brier=[.01] * 10)
    report = {"core": good_core, "stability": good_stab, "gate_q_passed": True}
    assert sps.evaluate_gate_r_plus(report)["passed"]
    report_q_fail = {"core": good_core, "stability": good_stab, "gate_q_passed": False}
    assert not sps.evaluate_gate_r_plus(report_q_fail)["passed"]
    report_core_fail = {"core": _core_fixture(ci_lower=-1), "stability": good_stab,
                        "gate_q_passed": True}
    assert not sps.evaluate_gate_r_plus(report_core_fail)["passed"]
