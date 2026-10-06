"""Blocking tests for the Stage 5 power layer (plan v0.3 section 6.2, Task 4).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Pins the corrected
power calibre (the D2F lesson): model/decision EFFECT centres shrink by rho while
scenario/task SUPPORT proportions keep their coupled resampled values; structural
identities hold on every draw; the full randomizable SOTA-M AND-chain is simulated
jointly (never marginal multiplication); CI lower bounds use the resampling SE at
the current n; the materiality early-stop fires before any n grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage5_power as s5p  # noqa: E402


def make_quantities(g: int = 400, seed: int = 0, effect: str = "large"):
    rng = np.random.RandomState(seed)
    regret = rng.rand(g) * 0.3 + 0.005
    sota = regret + rng.randn(g) * 0.10
    if effect == "large":
        actwin = regret + rng.randn(g) * 0.02      # clearly better ranking
    else:
        actwin = sota + rng.randn(g) * 0.001       # no material difference
    actwin_by_seed = actwin[None, :] + rng.randn(5, g) * 0.005
    sota_by_seed = sota[None, :] + rng.randn(5, g) * 0.005
    t_star = rng.randint(10, 16, g).astype(np.int64)
    interaction = rng.rand(g) * 60
    return {
        "regret": regret,
        "y_high": (regret > 0.01).astype(np.int64),
        "idle_regret": regret + 0.05,
        "random_regret": regret + 0.10,
        "a_star": rng.randint(0, 10, g).astype(np.int64),
        "t_star": t_star,
        "interaction_min_distance": interaction,
        "tstar_layer": (t_star >= 13).astype(np.int64),
        "interaction_layer": (interaction > 29.129).astype(np.int64),
        "actwin_ensemble": actwin,
        "actwin_by_seed": actwin_by_seed,
        "sota_ensemble": sota,
        "sota_by_seed": sota_by_seed,
    }


# --- slot typing --------------------------------------------------------------------------------


def test_slot_typing_is_exact_partition():
    slots = s5p.classify_power_slots()
    union = sorted(slots["effect"] + slots["anchor"] + slots["support"])
    assert union == list(range(s5p.N_QUANTITIES))


def test_support_slots_never_shrunk():
    q = make_quantities()
    ref = s5p.reference_vector(q)
    draws = s5p.resample_draws(q, n=200, n_resamples=20, seed=123)
    transformed, _ = s5p.transform_draws(draws, ref, rho=0.5)
    slots = s5p.classify_power_slots()
    for slot in slots["support"]:
        assert np.array_equal(transformed[:, slot], draws[:, slot])
    for slot in slots["anchor"]:
        assert np.array_equal(transformed[:, slot], draws[:, slot])
    # effect slots DID move (unless the reference effect is exactly zero)
    moved = any(not np.allclose(transformed[:, s], draws[:, s])
                for s in slots["effect"] if abs(ref[s]) > 1e-12)
    assert moved


def test_structural_identities_enforced():
    q = make_quantities()
    draws = s5p.resample_draws(q, n=200, n_resamples=20, seed=7)
    s5p.assert_structural_identities(draws)  # oracle/stratum shares sum to 1
    broken = draws.copy()
    broken[0, s5p.Q_ORACLE] += 0.5
    with pytest.raises(ValueError):
        s5p.assert_structural_identities(broken)


# --- materiality early stop ------------------------------------------------------------------------


def test_materiality_early_stop():
    assert s5p.check_materiality(0.07, 0.05) is True      # 0.75x stays above both lines
    assert s5p.check_materiality(0.06, 0.05) is False     # 0.75*0.06=0.045 < 0.05
    assert s5p.check_materiality(0.07, 0.039) is False    # 0.75*0.039=0.02925 < 0.03
    assert s5p.check_materiality(float("nan"), 0.05) is False  # fail closed


# --- gate evaluation ---------------------------------------------------------------------------------


def test_evaluate_draws_and_chain():
    q = make_quantities(effect="large")
    ref = s5p.reference_vector(q)
    draws = s5p.resample_draws(q, n=300, n_resamples=50, seed=11)
    se = draws.std(axis=0, ddof=1)
    transformed, d_idle = s5p.transform_draws(draws, ref, rho=1.0)
    evaluation = s5p.evaluate_gate_draws(transformed, d_idle, se)
    assert set(evaluation["conditions"]) == {
        "r1_i_aurc", "r2_i_regret80", "r3_seed_consistency",
        "r4_strata_protection", "r5_task_contract"}
    assert evaluation["passed"].mean() > 0.5  # large effect mostly passes
    # r1 fails when every draw's i_aurc sits below the 0.05 line
    low = transformed.copy()
    low[:, s5p.Q_I_AURC] = 0.01
    evaluation_low = s5p.evaluate_gate_draws(low, d_idle, se)
    assert not evaluation_low["conditions"]["r1_i_aurc"].any()
    assert not evaluation_low["passed"].any()


# --- joint simulation ----------------------------------------------------------------------------------


def test_simulate_power_selects_minimal_n_or_underpowered():
    q = make_quantities(effect="large")
    result = s5p.simulate_power(q, candidate_ns=(100, 200), n_resamples=200, seed=5)
    assert result["selected_n"] in (100, 200)
    # joint power grows with n; the larger n clears the 0.90 bar for a large effect
    assert result["per_n"][200]["scenarios"]["0.75"]["joint_power"] >= 0.90
    assert (result["per_n"][200]["scenarios"]["0.75"]["joint_power"]
            >= result["per_n"][100]["scenarios"]["0.75"]["joint_power"])
    # no material effect -> the AND chain cannot pass at any n
    q_flat = make_quantities(effect="none")
    result_flat = s5p.simulate_power(q_flat, candidate_ns=(100, 200),
                                     n_resamples=200, seed=5)
    assert result_flat["selected_n"] is None
    assert result_flat["underpowered"] is True


def test_reference_vector_fail_closed_on_nonfinite():
    q = make_quantities()
    q["regret"] = q["regret"].copy()
    q["regret"][3] = np.nan
    with pytest.raises(ValueError):
        s5p.reference_vector(q)
