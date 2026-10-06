"""Stage 4B power tests: joint AND-chain selection, roles==2 restriction, Gate-0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4_risk as s4r  # noqa: E402
from ac_twin import stage4b_power as s4bpw  # noqa: E402
from ac_twin import stage4b_protocol as s4bp  # noqa: E402


def _synthetic_inputs(episodes=200, seed=0):
    """Strong-signal synthetic screen: combined ranks error far better than baselines."""
    rng = np.random.default_rng(seed)
    risk_target = rng.gamma(2.0, 0.2, size=episodes)
    unc = risk_target + rng.normal(0, 0.6, size=episodes)          # weak
    geo = risk_target + rng.normal(0, 0.9, size=episodes)          # weaker
    comb = risk_target + rng.normal(0, 0.15, size=episodes)        # strong
    primary = np.stack([risk_target + rng.normal(0, 0.05, episodes) for _ in range(5)])
    cf_action = np.tile(np.arange(5), episodes // 5 + 1)[:episodes]
    cf_duration = np.tile(np.repeat([1, 3], 5), episodes // 10 + 1)[:episodes]
    return {
        "comb_score": comb, "geo_score": geo, "unc_score": unc,
        "risk_target": risk_target, "primary_error_by_pair": primary,
        "cf_action": cf_action, "cf_duration": cf_duration,
        "episode_seed": np.arange(episodes), "top_error_threshold": 0.4,
    }


# --- quantity vector layout + direction -------------------------------------------------


def test_quantity_vector_layout_and_direction():
    inputs = _synthetic_inputs()
    identity = np.arange(inputs["risk_target"].size)
    q = s4bpw.compute_quantity_vector(
        identity, inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
        inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"])
    assert q.shape == (s4bpw.N_QUANTITIES,)
    assert np.isfinite(q).all()
    # Strong-signal fixture: combined beats both baselines on every axis.
    assert q[s4bpw.Q_I_AURC_UNC] > 0 and q[s4bpw.Q_I_C80_UNC] > 0
    assert q[s4bpw.Q_I_AURC_GEO] > 0 and q[s4bpw.Q_I_C80_GEO] > 0
    assert q[s4bpw.Q_PR] > 0
    assert (q[s4bpw.PAIR_INDEX] > 0).all()


def test_bootstrap_preserves_episode_multiplicity():
    # A resample index with replacement repeats episodes; the statistic must reflect
    # the multiplicity (a duplicated heavy episode shifts the mean error).
    inputs = _synthetic_inputs(episodes=40)
    heavy = int(np.argmax(inputs["risk_target"]))
    index = np.concatenate([np.arange(40), np.full(40, heavy)])
    # Pad/trim to keep _validate_pair >=2; use the resampled set directly.
    q_dup = s4bpw.compute_quantity_vector(
        index, inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
        inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"])
    q_plain = s4bpw.compute_quantity_vector(
        np.arange(40), inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
        inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"])
    # Duplicating the single heaviest episode 40x changes the statistics.
    assert not np.allclose(q_dup, q_plain)


# --- covariance is empirical (NOT back-derived from an 80% CI) ---------------------------


def test_covariance_is_empirical_bootstrap():
    inputs = _synthetic_inputs()
    out = s4bpw.joint_bootstrap(inputs, n_resamples=300, seed=123)
    # Recompute the empirical covariance independently; must match symmetrised form.
    rng = np.random.default_rng(123)
    episodes = inputs["risk_target"].size
    draws = np.empty((300, s4bpw.N_QUANTITIES))
    for b in range(300):
        idx = rng.integers(0, episodes, size=episodes)
        draws[b] = s4bpw.compute_quantity_vector(
            idx, inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
            inputs["risk_target"], inputs["primary_error_by_pair"],
            inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"])
    emp = np.cov(draws, rowvar=False)
    assert np.allclose(out["covariance"], emp, atol=1e-6)
    # Positive semi-definite after clipping.
    assert np.linalg.eigvalsh(out["covariance"]).min() >= 0
    # theta_hat is the full-screen point estimate.
    assert np.allclose(out["theta_hat"], s4bpw.compute_quantity_vector(
        np.arange(episodes), inputs["comb_score"], inputs["geo_score"], inputs["unc_score"],
        inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"]))


def test_no_ci_width_backwards_derivation():
    # The module must not expose any helper that converts a CI width to an SE via a
    # z-multiplier (plan section 5.3: SE only from the new bootstrap distribution).
    assert not any("ci" in name.lower() and "se" in name.lower()
                   for name in dir(s4bpw))


# --- mechanical AND-chain selection -------------------------------------------------------


def test_selects_smallest_n_meeting_target():
    inputs = _synthetic_inputs()
    boot = s4bpw.joint_bootstrap(inputs, n_resamples=300, seed=7)
    sel = s4bpw.select_confirmation_size(
        boot["theta_hat"], boot["covariance"], candidate_ns=(2000, 3000, 4000, 5000),
        monte_carlo=2000, seed=99, n_reference=inputs["risk_target"].size,
        shrink_scenarios=(1.0, 0.75, 0.5), shrink_main=0.75, target=0.80)
    # Strong signal: smallest n should already pass; sensitivity at max n reported.
    assert sel["selected_n"] == 2000
    assert set(sel["sensitivity_at_max_n"]) == {"1.00", "0.75", "0.50"}
    assert sel["per_n"][2000]["binding_condition"] in sel["per_n"][2000]["scenarios"]["0.75"]["marginal_power"]


def test_underpowered_when_signal_zero():
    episodes = 200
    rng = np.random.default_rng(3)
    risk_target = rng.gamma(2.0, 0.2, size=episodes)
    noise = lambda s: rng.normal(0, s, episodes)  # noqa: E731
    inputs = {
        "comb_score": risk_target + noise(1.0),  # no better than baselines
        "geo_score": risk_target + noise(1.0),
        "unc_score": risk_target + noise(1.0),
        "risk_target": risk_target,
        "primary_error_by_pair": np.stack([risk_target + noise(0.05) for _ in range(5)]),
        "cf_action": np.tile(np.arange(5), episodes // 5 + 1)[:episodes],
        "cf_duration": np.tile(np.repeat([1, 3], 5), episodes // 10 + 1)[:episodes],
        "episode_seed": np.arange(episodes), "top_error_threshold": 0.4,
    }
    boot = s4bpw.joint_bootstrap(inputs, n_resamples=200, seed=11)
    sel = s4bpw.select_confirmation_size(
        boot["theta_hat"], boot["covariance"], candidate_ns=(2000, 3000, 4000, 5000),
        monte_carlo=2000, seed=5, n_reference=episodes,
        shrink_scenarios=(1.0, 0.75, 0.5), shrink_main=0.75, target=0.80)
    assert sel["selected_n"] is None
    assert sel["underpowered"] is True


# --- Gate-0 self-check ---------------------------------------------------------------------


def _fake_report(point):
    strata = {}
    for a, slot in s4bpw.ACTION_INDEX.items():
        strata[f"action_{a}"] = {"i_aurc": float(point[slot])}
    for d, slot in s4bpw.DURATION_INDEX.items():
        strata[f"duration_{d}"] = {"i_aurc": float(point[slot])}
    return {
        "improvement_bootstrap": {"i_aurc": float(point[s4bpw.Q_I_AURC_UNC]),
                                  "i_c80": float(point[s4bpw.Q_I_C80_UNC])},
        "finalist_report": {"pr_auc_minus_prevalence": float(point[s4bpw.Q_PR])},
        "per_pair_i_aurc": [float(point[s4bpw.Q_PAIR_START + p]) for p in range(5)],
        "stratified": {"strata": strata},
    }


def test_gate0_passes_on_exact_match():
    inputs = _synthetic_inputs()
    point = s4bpw.compute_quantity_vector(
        np.arange(inputs["risk_target"].size), inputs["comb_score"], inputs["geo_score"],
        inputs["unc_score"], inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"])
    out = s4bpw.self_check_vs_report(point, _fake_report(point))
    assert out["passed"] and out["max_abs_delta"] == 0.0


def test_gate0_rejects_drift_beyond_1e_8():
    inputs = _synthetic_inputs()
    point = s4bpw.compute_quantity_vector(
        np.arange(inputs["risk_target"].size), inputs["comb_score"], inputs["geo_score"],
        inputs["unc_score"], inputs["risk_target"], inputs["primary_error_by_pair"],
        inputs["cf_action"], inputs["cf_duration"], inputs["top_error_threshold"])
    report = _fake_report(point)
    report["improvement_bootstrap"]["i_aurc"] += 1e-6  # drift past 1e-8
    with pytest.raises(ValueError, match="Gate-0"):
        s4bpw.self_check_vs_report(point, report)
