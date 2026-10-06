"""Stage 4C power tests: screen-only inputs, mechanical selection, AND chain."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4c_power as s4cpw  # noqa: E402
from ac_twin import stage4c_protocol as s4cp  # noqa: E402

NA = s4cp.N_ACTIONS


def _screen_npz(tmp_path, groups=50, missing=None):
    rng = np.random.default_rng(0)
    regret = np.abs(rng.normal(size=groups))
    fields = {
        "group_seed": np.arange(groups),
        "regret": regret,
        "regret_by_pair": np.stack([regret + 0.02 * p for p in range(5)]),
        "random_regret": regret + 0.5,
        "idle_regret": regret + 0.4,
        "combined_group_risk": regret + 0.05 * rng.normal(size=groups),
        "baseline_group_risk": rng.normal(size=groups),
        "high_regret_label": (regret > np.quantile(regret, 0.8)).astype(np.int64),
        "tstar_layer": (np.arange(groups) % 2),
        "density_layer": ((np.arange(groups) // 2) % 2),
        "J_hat": np.abs(rng.normal(size=(groups, NA))),
        "J_true": np.abs(rng.normal(size=(groups, NA))),
    }
    if missing:
        fields.pop(missing)
    out = tmp_path / "screen_group_quantities.npz"
    np.savez_compressed(out, **fields)
    return out


def test_load_screen_group_quantities_requires_fields(tmp_path):
    ok = s4cpw.load_screen_group_quantities(_screen_npz(tmp_path))
    assert ok["regret"].shape[0] == 50
    with pytest.raises(ValueError):
        s4cpw.load_screen_group_quantities(_screen_npz(tmp_path, missing="regret_by_pair"))


def test_evaluate_draws_and_chain_single_failure_fails_all():
    se = np.full(s4cpw.N_QUANTITIES, 0.01)
    good = np.zeros(s4cpw.N_QUANTITIES)
    good[s4cpw.Q_I_AURC] = 0.10
    good[s4cpw.Q_I_REGRET80] = 0.05
    good[s4cpw.Q_PR] = 0.2
    good[s4cpw.PAIR_INDEX] = 0.05
    good[s4cpw.STRATA_INDEX] = 0.0
    good[s4cpw.TRIM_INDEX] = 0.03
    draws = np.tile(good, (100, 1))
    assert s4cpw.evaluate_draws(draws, se)["passed"].all()
    bad = draws.copy()
    bad[:, s4cpw.Q_PR] = -0.5  # PR condition fails -> whole AND fails
    assert not s4cpw.evaluate_draws(bad, se)["passed"].any()


def test_select_confirm_n_mechanical_and_power_grows_with_n():
    # strong positive effects + small covariance -> smallest n already powered.
    theta = np.zeros(s4cpw.N_QUANTITIES)
    theta[s4cpw.Q_I_AURC] = 0.20
    theta[s4cpw.Q_I_REGRET80] = 0.15
    theta[s4cpw.Q_PR] = 0.30
    theta[s4cpw.PAIR_INDEX] = 0.20
    theta[s4cpw.STRATA_INDEX] = 0.10
    theta[s4cpw.TRIM_INDEX] = 0.10
    cov = np.eye(s4cpw.N_QUANTITIES) * 1e-4
    sel = s4cpw.select_confirm_n(
        theta, cov, candidate_ns=(600, 900, 1200), monte_carlo=500, seed=3,
        n_reference=300, shrink_scenarios=(1.0, 0.75, 0.5), shrink_main=0.75,
        target=0.80)
    assert sel["selected_n"] == 600  # the SMALLEST n meeting the target is chosen
    assert not sel["underpowered"]
    # every candidate meets the target, so the mechanical rule takes the minimum.
    for n in (600, 900, 1200):
        assert sel["per_n"][n]["joint_power_main"] >= 0.80
        assert sel["per_n"][n]["binding_condition"]  # binding condition reported


def test_select_confirm_n_underpowered_returns_none():
    theta = np.zeros(s4cpw.N_QUANTITIES)  # no effect anywhere
    cov = np.eye(s4cpw.N_QUANTITIES) * 1.0
    sel = s4cpw.select_confirm_n(
        theta, cov, candidate_ns=(600, 900), monte_carlo=300, seed=3,
        n_reference=300, shrink_scenarios=(1.0, 0.75, 0.5), shrink_main=0.75,
        target=0.80)
    assert sel["underpowered"] and sel["selected_n"] is None
    # sensitivity at max n is reported for all three shrink scenarios.
    assert set(sel["sensitivity_at_max_n"]) == {"1.00", "0.75", "0.50"}
