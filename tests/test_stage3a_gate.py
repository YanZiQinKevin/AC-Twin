from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage3a_gate import (  # noqa: E402
    check_five_seed_consistent,
    check_non_degenerate,
    evaluate_h4_gate,
    select_strongest_baselines,
)


def _make_truth(episodes=40, steps=10, agents=5, seed=0):
    rng = np.random.default_rng(seed)
    # Response truth with structure: some zero, some spread across regimes.
    r_true = rng.gamma(shape=1.5, scale=0.6, size=(episodes, steps, agents))
    zero_mask = rng.random((episodes, steps, agents)) < 0.4
    r_true[zero_mask] = 0.0
    regime = np.zeros((episodes, steps, agents), dtype=np.int64)
    regime[(r_true > 0.0001) & (r_true <= 0.25)] = 1
    regime[(r_true > 0.25) & (r_true <= 1.9)] = 2
    regime[r_true > 1.9] = 3
    valid = np.ones((episodes, steps, agents), dtype=bool)
    return r_true, regime, valid


def _logits_from_labels(regime, temperature_noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    logits = np.zeros(regime.shape + (4,))
    for c in range(4):
        logits[..., c] = (regime == c) * 6.0
    logits += rng.normal(0, temperature_noise, logits.shape)
    return logits


def _baselines(r_true, regime, valid, feats):
    from ac_twin.stage3_response import (
        ConstantZeroBaseline,
        TrainMeanByHorizonBaseline,
        TrainPrevalenceClassifier,
        MultinomialRegimeClassifier,
    )
    cont = {
        "constant_zero": ConstantZeroBaseline().fit(r_true, feats, valid).predict(feats),
        "train_mean": TrainMeanByHorizonBaseline().fit(r_true, feats, valid).predict(feats),
    }
    clf = {
        "prevalence": TrainPrevalenceClassifier().fit(regime, feats, valid).predict_proba(feats),
        "multinomial": MultinomialRegimeClassifier().fit(regime, feats, valid).predict_proba(feats),
    }
    return cont, clf


def _feats(shape):
    e, t, n = shape
    return {
        "eval_shape": shape, "n_eval": e, "horizon": t,
        "cf_action": np.zeros(e, dtype=int), "cf_duration": np.ones(e),
        "initial_distance": np.zeros((e, n)),  # per-vehicle (E,N)
    }


def test_strong_model_passes_gate() -> None:
    r_true, regime, valid = _make_truth()
    feats = _feats(r_true.shape)
    cont, clf = _baselines(r_true, regime, valid, feats)
    strongest = select_strongest_baselines(
        r_true=r_true, regime_labels=regime, valid_mask=valid,
        continuous_baselines=cont, classification_baselines=clf,
    )
    # five seeds: r_hat close to truth, logits aligned with labels
    seeds_r = np.stack([r_true + np.random.default_rng(s).normal(0, 0.05, r_true.shape) for s in range(5)]).clip(0)
    seeds_logits = np.stack([_logits_from_labels(regime, seed=s) for s in range(5)])
    report = evaluate_h4_gate(
        model_r_hat_seeds=seeds_r,
        model_regime_logits_seeds=seeds_logits,
        temperature=1.0,
        r_true=r_true, regime_labels=regime, valid_mask=valid,
        continuous_baselines=cont, classification_baselines=clf,
        strongest_baselines=strongest,
        n_resamples=200, confidence_level=0.95, bootstrap_seed=7,
    )
    assert report["verdict"]["passed"], report["verdict"]["checks"]
    assert report["metrics"]["i_mae"]["ci_lower"] > 0
    # per-seed consistency now spans all five metrics
    assert set(report["five_seed"]["per_seed"]) == {
        "i_mae", "spearman", "i_pr_presence", "i_pr_high_vs_lowmid", "i_brier"
    }


def test_degenerate_constant_model_fails_gate() -> None:
    r_true, regime, valid = _make_truth()
    feats = _feats(r_true.shape)
    cont, clf = _baselines(r_true, regime, valid, feats)
    strongest = select_strongest_baselines(
        r_true=r_true, regime_labels=regime, valid_mask=valid,
        continuous_baselines=cont, classification_baselines=clf,
    )
    # constant-mean model: no better than the train-mean baseline + degenerate
    const = np.full_like(r_true, r_true[valid].mean())
    seeds_r = np.stack([const.copy() for _ in range(5)])
    # uninformative logits -> prevalence-like
    seeds_logits = np.zeros((5,) + regime.shape + (4,))
    report = evaluate_h4_gate(
        model_r_hat_seeds=seeds_r,
        model_regime_logits_seeds=seeds_logits,
        temperature=1.0,
        r_true=r_true, regime_labels=regime, valid_mask=valid,
        continuous_baselines=cont, classification_baselines=clf,
        strongest_baselines=strongest,
        n_resamples=200, confidence_level=0.95, bootstrap_seed=7,
    )
    assert not report["verdict"]["passed"]


def test_five_seed_consistency_requires_all_metrics() -> None:
    good = {k: np.full(5, 0.1) for k in ("i_mae", "spearman", "i_pr_presence", "i_pr_high_vs_lowmid", "i_brier")}
    assert check_five_seed_consistent(good)
    # one metric flips sign on one seed -> not consistent
    bad = dict(good)
    bad["i_pr_presence"] = np.array([0.1, 0.1, -0.2, 0.1, 0.1])
    assert not check_five_seed_consistent(bad)
    # MAE-only is no longer sufficient (missing metrics)
    assert not check_five_seed_consistent({"i_mae": np.full(5, 0.1)})


def test_non_degenerate_checks() -> None:
    valid = np.ones((2, 10, 5), dtype=bool)
    const_zero = np.zeros((2, 10, 5))
    const_mean = np.full((2, 10, 5), 0.7)
    rng = np.random.default_rng(0)
    varying = rng.random((2, 10, 5))
    assert not check_non_degenerate(const_zero, valid)
    assert not check_non_degenerate(const_mean, valid)
    assert check_non_degenerate(varying, valid)
