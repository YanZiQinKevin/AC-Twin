"""Focused blocking-invariant tests for the D2E plan-4.2 screen power pre-check (torch-free math).

The distinct NEW invariants (the reused D2 screen quantity vector + gate thresholds are tested
upstream in ``stage4d_external_statistics``): (1) the extended power vector's first 32 slots are
the frozen D2 screen vector bitwise and the appended structural fractions are correct; (2) the
draw AND-chain mirrors the D2 screen gate's statistical conditions (point + 80% CI-lower + the
base<=idle line + 5-seed consistency + strata/oracle/reactive support); (3) ``simulate_power``
selects the smallest n whose 75%-shrink joint power reaches the target and scales covariance by
``n_reference / n``; (4) the covariance is symmetrised PSD; (5) ``estimate_screen_power`` wires
bootstrap -> subset -> covariance -> MVN end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2e_power as POW  # noqa: E402
from ac_twin import stage4d_external_statistics as D2STAT  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402

G = 240
BASELINES = ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD")


def _synth_quantities(seed: int = 0) -> dict:
    """A valid screen-quantities mapping where the repaired pipeline passes every condition."""
    rng = np.random.default_rng(seed)
    regret = rng.uniform(0.005, 0.06, size=G)                 # strictly positive
    noise = rng.normal(scale=0.01, size=G)
    ensemble = regret + noise                                  # strong risk -> regret correlation
    risk_by_seed = ensemble[None] + rng.normal(scale=0.01, size=(len(T.MODEL_SEEDS), G))
    baseline_risks = {
        name: rng.permutation(regret) + rng.normal(scale=0.02, size=G) for name in BASELINES}
    j_true = np.abs(rng.normal(size=(G, 10))) + 0.05
    best = rng.integers(0, 4, size=G)                          # oracle spread over 4 actions
    j_true[np.arange(G), best] -= 0.04                          # make ``best`` the argmin
    j_hat = j_true + rng.normal(scale=0.01, size=(G, 10))      # correlated -> spearman > 0
    return {
        "group_seed": np.arange(G, dtype=np.int64) + 1,
        "regret": regret,
        "y_high": (regret > 0.01).astype(np.int64),
        "random_regret": regret + rng.uniform(0.01, 0.03, size=G),
        "idle_regret": regret + rng.uniform(0.005, 0.02, size=G),
        "ensemble_risk": ensemble,
        "risk_by_seed": risk_by_seed,
        "J_hat": j_hat,
        "J_true": j_true,
        "a_star": np.argmin(j_true, axis=-1),
        "tstar_layer": (np.arange(G) % 2).astype(np.int64),
        "interaction_layer": ((np.arange(G) // 2) % 2).astype(np.int64),
        "t_star": np.where(np.arange(G) % 2 == 0, 10, 14).astype(np.int64),
        "interaction_min_distance": rng.uniform(1.0, 30.0, size=G),
        "reactive": (np.arange(G) % 10) < 3,                    # 30% reactive
        "baseline_risks": baseline_risks,
        "strongest_baseline": "SEED_MARGIN_STD",
        "n_groups": G,
    }


def _good_power_row() -> np.ndarray:
    """One power-subset row that passes every statistical condition with margin."""
    row = np.zeros(POW.N_POWER_QUANTITIES)
    row[POW.P_I_AURC] = 0.20
    row[POW.P_I_REGRET80] = 0.12
    row[POW.P_PR] = 0.15
    row[POW.P_SPEARMAN] = 0.50
    row[POW.P_BASE_VS_RANDOM] = 0.30
    row[POW.P_BASE_MEAN] = 0.05
    row[POW.P_IDLE_MEAN] = 0.10
    row[POW.P_HIGH_REGRET] = 0.30
    row[POW.P_SEED_I_AURC] = 0.10
    row[POW.P_SEED_I_R80] = 0.06
    row[POW.P_ORACLE] = 0.12                                   # all 10 >= 0.05 -> coverage 10
    row[POW.P_REACTIVE] = 0.40
    row[POW.P_STRATA] = 0.25
    return row


def test_extended_vector_prefix_matches_frozen_d2_vector() -> None:
    q = _synth_quantities()
    identity = np.arange(G, dtype=np.int64)
    vec = POW.compute_d2e_power_vector(identity, q)
    assert vec.shape == (POW.N_EXTENDED,)
    np.testing.assert_allclose(
        vec[:D2STAT.N_QUANTITIES], D2STAT.compute_screen_quantity_vector(identity, q),
        rtol=1e-12, atol=1e-12,
        err_msg="the first 32 slots must be the frozen D2 screen vector bitwise")
    assert np.isfinite(vec).all()


def test_extended_vector_structural_fractions() -> None:
    q = _synth_quantities()
    identity = np.arange(G, dtype=np.int64)
    vec = POW.compute_d2e_power_vector(identity, q)
    oracle = vec[POW.X_ORACLE:POW.X_ORACLE + T.N_ACTIONS]
    np.testing.assert_allclose(oracle.sum(), 1.0, rtol=1e-9,
                               err_msg="oracle action shares must sum to 1")
    a_star = q["a_star"]
    np.testing.assert_allclose(
        oracle, [float((a_star == a).mean()) for a in range(T.N_ACTIONS)], rtol=1e-12)
    assert vec[POW.X_REACTIVE] == float(np.asarray(q["reactive"], dtype=bool).mean())
    strata = vec[POW.X_STRATA:POW.X_STRATA + 4]
    np.testing.assert_allclose(strata[0] + strata[1], 1.0, rtol=1e-9)   # tstar low+high
    np.testing.assert_allclose(strata[2] + strata[3], 1.0, rtol=1e-9)   # interaction low+high
    assert len(POW.POWER_SLOTS) == POW.N_POWER_QUANTITIES == 33
    assert len(POW.POWER_QUANTITY_NAMES) == POW.N_POWER_QUANTITIES


def test_evaluate_screen_draws_and_chain() -> None:
    se = np.full(POW.N_POWER_QUANTITIES, 0.01)
    good = _good_power_row()[None].repeat(8, axis=0)
    out = POW.evaluate_screen_draws(good, se)
    assert out["passed"].all(), "a fully-passing draw must pass the AND-chain"
    assert all(bool(v.all()) for v in out["conditions"].values())

    # base worse than idle -> c5 fails (the base<=idle line is simulated, not rescued).
    bad_idle = good.copy()
    bad_idle[:, POW.P_BASE_MEAN] = 0.20
    bad_idle[:, POW.P_IDLE_MEAN] = 0.10
    out = POW.evaluate_screen_draws(bad_idle, se)
    assert not out["passed"].any()
    assert not out["conditions"]["c5_base_vs_random_and_not_worse_than_idle"].any()

    # point above the materiality line but a wide SE -> CI lower crosses 0 -> r1 fails.
    big_se = se.copy()
    big_se[POW.P_I_AURC] = 0.20                                  # ci = 0.20 - 1.2816*0.20 < 0
    out = POW.evaluate_screen_draws(good, big_se)
    assert not out["conditions"]["r1_i_aurc_regret"].any()
    assert not out["passed"].any()

    # fewer than 4 seeds positive on both improvements -> r4 fails.
    bad_seed = good.copy()
    bad_seed[:, POW.P_SEED_I_AURC[0]] = -0.01
    bad_seed[:, POW.P_SEED_I_AURC[1]] = -0.01
    out = POW.evaluate_screen_draws(bad_seed, se)
    assert not out["conditions"]["r4_seed_consistency_both_positive"].any()


def test_symmetrised_psd_covariance() -> None:
    rng = np.random.default_rng(3)
    draws = rng.normal(size=(500, POW.N_POWER_QUANTITIES))
    cov = POW.symmetrised_psd_covariance(draws)
    assert cov.shape == (POW.N_POWER_QUANTITIES, POW.N_POWER_QUANTITIES)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-10, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() >= 0.0


def test_simulate_power_selects_smallest_n_and_scales_covariance() -> None:
    theta = _good_power_row()
    cov = np.diag(np.full(POW.N_POWER_QUANTITIES, 0.02 ** 2))
    result = POW.simulate_power(
        theta, cov, candidate_ns=(1000, 1500, 2000, 2500), n_reference=1500,
        monte_carlo=4000, seed=11, shrink_scenarios=(1.0, 0.75, 0.5),
        selection_scenario=0.75, target=0.80)
    assert result["selected_n"] == 1000, "a strong effect must select the smallest candidate n"
    assert not result["underpowered"]
    for n in (1000, 1500, 2000, 2500):
        assert result["per_n"][n]["joint_power_at_selection"] >= 0.80

    # a materiality-line failure (i_aurc below 0.05) can never be rescued by a larger n.
    weak = theta.copy()
    weak[POW.P_I_AURC] = 0.04
    result = POW.simulate_power(
        weak, cov, candidate_ns=(1000, 1500, 2000, 2500), n_reference=1500,
        monte_carlo=4000, seed=11, shrink_scenarios=(1.0, 0.75, 0.5),
        selection_scenario=0.75, target=0.80)
    assert result["underpowered"] and result["selected_n"] is None
    assert result["per_n"][2500]["binding_condition"] == "r1_i_aurc_regret"


def test_estimate_screen_power_end_to_end() -> None:
    q = _synth_quantities()
    result = POW.estimate_screen_power(
        q, bootstrap_seed=123, bootstrap_resamples=200, monte_carlo=2000,
        candidate_ns=(1000, 1500), shrink_scenarios=(1.0, 0.75, 0.5),
        selection_scenario=0.75, target=0.80, data_sha256="ab" * 32)
    assert result["n_reference"] == G
    assert set(result["per_n"]) == {1000, 1500}
    assert set(result["theta_hat"]) == set(POW.POWER_QUANTITY_NAMES)
    assert set(result["se_reference"]) == set(POW.POWER_QUANTITY_NAMES)
    assert all(np.isfinite(list(result["theta_hat"].values())))
    for n in (1000, 1500):
        for scenario in ("1.00", "0.75", "0.50"):
            assert 0.0 <= result["per_n"][n]["scenarios"][scenario]["joint_power"] <= 1.0
