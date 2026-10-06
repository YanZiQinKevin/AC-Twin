"""Focused blocking-invariant tests for the D2F corrected power calibre (plan section 2.1).

The distinct NEW invariants (the reused extended 47-slot vector + gate thresholds are tested
upstream in ``test_stage4d_d2e_power`` / ``test_stage4d_external_statistics``):

1. the 33 power quantities are typed EXACTLY (15 model/decision effects + 2 report anchors +
   16 scenario/task support + the derived D_idle difference);
2. the corrected transform shrinks ONLY effect centres (E_b + (rho-1)*E_ref) and the D_idle
   DIFFERENCE — support and anchor slots are bitwise untouched, so the support-condition
   marginal powers are IDENTICAL across rho scenarios (the old 0.75x-everything defect can
   never reappear: 0.75x oracle shares summing to 0.75 are rejected fail-closed);
3. c5 tests ONLY the transformed difference D_idle^(rho) >= -tie_tolerance (never the
   separately-shrunk absolute regrets);
4. the fresh entropy 2026122406 is registered and the per-n resample streams are
   deterministic; the n selection respects the pre-frozen capacity-admissible set;
5. the report is create-once, requires a passing capacity audit, quotes the old D2E terminal
   state, and writes ONLY inside the D2F namespace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4d_d2f_power as DPOW  # noqa: E402
from ac_twin import stage4d_d2f_protocol as PROT  # noqa: E402
from ac_twin import stage4d_d2e_power as POW  # noqa: E402
from ac_twin import stage4d_external_statistics as D2STAT  # noqa: E402
from ac_twin import stage4d_external_training as T  # noqa: E402

G = 240
BASELINES = ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD")


def _synth_quantities(seed: int = 0, correlated: bool = True) -> dict:
    """A valid screen-quantities mapping; with ``correlated`` the pipeline passes every gate."""
    rng = np.random.default_rng(seed)
    regret = rng.uniform(0.005, 0.06, size=G)
    noise = rng.normal(scale=0.01, size=G)
    ensemble = regret + noise if correlated else rng.uniform(0.005, 0.06, size=G)
    risk_by_seed = ensemble[None] + rng.normal(scale=0.01, size=(len(T.MODEL_SEEDS), G))
    baseline_risks = {
        name: rng.permutation(regret) + rng.normal(scale=0.02, size=G) for name in BASELINES}
    j_true = np.abs(rng.normal(size=(G, 10))) + 0.05
    best = rng.integers(0, 4, size=G)
    j_true[np.arange(G), best] -= 0.04
    j_hat = j_true + rng.normal(scale=0.01, size=(G, 10))
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
        "reactive": (np.arange(G) % 10) < 3,
        "baseline_risks": baseline_risks,
        "strongest_baseline": "SEED_MARGIN_STD",
        "n_groups": G,
    }


def _good_row() -> np.ndarray:
    """One power-subset row passing every statistical condition with margin."""
    row = np.zeros(DPOW.N_POWER_QUANTITIES)
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
    row[POW.P_ORACLE] = 0.10
    row[POW.P_REACTIVE] = 0.40
    row[POW.P_STRATA] = 0.25
    return row


# --- 1. exact typing of the 33 quantities -----------------------------------------------------------------


def test_slot_typing_is_exact_partition() -> None:
    slots = DPOW.classify_power_slots()
    assert len(slots["effect"]) == 15
    assert len(slots["anchor"]) == 2
    assert len(slots["support"]) == 16
    assert sorted(slots["effect"] + slots["anchor"] + slots["support"]) == list(range(33))
    # support is EXACTLY the scenario/task proportions (never shrunk with effect retention).
    assert set(slots["support"]) == {POW.P_HIGH_REGRET, POW.P_REACTIVE} \
        | set(POW.P_ORACLE) | set(POW.P_STRATA)
    # effects include the decision improvements and all ten per-seed slots.
    assert set(slots["effect"]) == {POW.P_I_AURC, POW.P_I_REGRET80, POW.P_PR, POW.P_SPEARMAN,
                                    POW.P_BASE_VS_RANDOM} \
        | set(POW.P_SEED_I_AURC) | set(POW.P_SEED_I_R80)
    # the absolute regrets are report anchors only (never gate inputs).
    assert set(slots["anchor"]) == {POW.P_BASE_MEAN, POW.P_IDLE_MEAN}


# --- 2. the corrected transform ----------------------------------------------------------------------------


def test_transform_shrinks_only_effect_centers_and_didle_difference() -> None:
    rng = np.random.default_rng(7)
    reference = _good_row()
    draws = reference[None] + rng.normal(scale=0.01, size=(64, DPOW.N_POWER_QUANTITIES))
    rho = 0.75
    transformed, d_idle_t = DPOW.transform_draws(draws, reference, rho)
    slots = DPOW.classify_power_slots()
    for slot in slots["effect"]:
        np.testing.assert_allclose(
            transformed[:, slot], draws[:, slot] + (rho - 1.0) * reference[slot],
            rtol=1e-12, atol=1e-15,
            err_msg=f"effect slot {slot} must shift by exactly (rho-1)*E_ref")
    for slot in slots["support"] + slots["anchor"]:
        np.testing.assert_array_equal(
            transformed[:, slot], draws[:, slot],
            err_msg=f"support/anchor slot {slot} must stay bitwise untouched")
    raw_diff = draws[:, POW.P_IDLE_MEAN] - draws[:, POW.P_BASE_MEAN]
    ref_diff = reference[POW.P_IDLE_MEAN] - reference[POW.P_BASE_MEAN]
    np.testing.assert_allclose(d_idle_t, raw_diff + (rho - 1.0) * ref_diff,
                               rtol=1e-12, atol=1e-15,
                               err_msg="D_idle must shrink as a DIFFERENCE")


def test_support_marginal_powers_are_identical_across_rho() -> None:
    """The crisp anti-defect invariant: support conditions never see the effect shrink."""
    q = _synth_quantities()
    result = DPOW.simulate_corrected_power(
        q, candidate_ns=(1000, 1500), n_resamples=200, seed=99,
        shrink_scenarios=(1.0, 0.75, 0.5), selection_scenario=0.75, target=0.80)
    for n in (1000, 1500):
        scenarios = result["per_n"][n]["scenarios"]
        for condition in ("c2_high_regret_frac", "c3_oracle_coverage", "c6_reactive_frac",
                          "c7_strata_floor"):
            values = {scenarios[s]["marginal_power"][condition] for s in scenarios}
            assert len(values) == 1, (
                f"{condition} marginal power must be IDENTICAL across rho scenarios at n={n} "
                f"(support proportions are never multiplied by rho): {values}")


def test_structural_identities_hold_and_reject_shrunk_shares() -> None:
    q = _synth_quantities()
    draws = DPOW.resample_draws(q, 120, n_resamples=50, seed=5)
    DPOW.assert_structural_identities(draws)
    corrupted = draws.copy()
    corrupted[:, POW.X_ORACLE:POW.X_ORACLE + 10] *= 0.75      # the old D2E defect shape
    with pytest.raises(ValueError, match="oracle action shares"):
        DPOW.assert_structural_identities(corrupted)


# --- 3. c5 tests ONLY the transformed D_idle difference -------------------------------------------------------


def test_c5_uses_difference_shrink_not_absolute_product_shrink() -> None:
    se = np.full(DPOW.N_POWER_QUANTITIES, 0.005)
    reference = _good_row()                                    # D_ref = 0.10 - 0.05 = +0.05
    row = _good_row()
    row[POW.P_BASE_MEAN] = 0.10
    row[POW.P_IDLE_MEAN] = 0.11                                # raw draw D_b = +0.01 (>0)
    draws = row[None].repeat(4, axis=0)

    # Under the OLD product semantics (0.75*base <= 0.75*idle) this draw passes; under the
    # corrected DIFFERENCE shrink D^(0.75) = 0.01 - 0.25*0.05 = -0.0025 < -tol -> c5 FAILS.
    old_out = POW.evaluate_screen_draws(draws, se)
    assert old_out["conditions"]["c5_base_vs_random_and_not_worse_than_idle"].all(), (
        "the frozen OLD gate passes c5 here (product semantics); the difference is the "
        "calibre correction")
    transformed, d_idle_t = DPOW.transform_draws(draws, reference, 0.75)
    assert np.all(d_idle_t < -D2STAT.BASELINE_TIE_TOLERANCE)
    out = DPOW.evaluate_corrected_draws(transformed, d_idle_t, se)
    assert not out["conditions"]["c5_base_vs_random_and_not_worse_than_idle"].any()
    assert not out["passed"].any()

    # control: at rho=1.0 the raw positive difference survives and every condition passes.
    transformed1, d_idle1 = DPOW.transform_draws(draws, reference, 1.0)
    out1 = DPOW.evaluate_corrected_draws(transformed1, d_idle1, se)
    assert out1["conditions"]["c5_base_vs_random_and_not_worse_than_idle"].all()
    assert out1["passed"].all()


# --- 4. seed registry + deterministic per-n streams + admissible-set selection ---------------------------------


def test_seed_registered_and_streams_deterministic() -> None:
    assert DPOW.POWER_SEED == 2026122406
    assert DPOW.POWER_SEED != POW.POWER_BOOTSTRAP_SEED        # 2026122405 (the D2E power seed)
    q = _synth_quantities()
    a = DPOW.resample_draws(q, 120, n_resamples=20, seed=DPOW.POWER_SEED)
    b = DPOW.resample_draws(q, 120, n_resamples=20, seed=DPOW.POWER_SEED)
    np.testing.assert_array_equal(a, b, err_msg="same n must reproduce the same stream")
    c = DPOW.resample_draws(q, 130, n_resamples=20, seed=DPOW.POWER_SEED)
    assert not np.array_equal(a, c[:, : a.shape[1]]) or c.shape[1] != a.shape[1]


def test_selection_respects_admissible_set_and_underpowered_stop() -> None:
    q = _synth_quantities()
    result = DPOW.simulate_corrected_power(
        q, candidate_ns=(1500, 2000), n_resamples=200, seed=13,
        shrink_scenarios=(1.0, 0.75, 0.5), selection_scenario=0.75, target=0.80)
    assert result["selected_n"] in (1500, 2000), (
        "the selection must never leave the pre-frozen capacity-admissible set")
    assert result["selected_n"] == 1500, "a strong synthetic effect selects the smallest n"
    with pytest.raises(ValueError, match="no admissible candidate n"):
        DPOW.simulate_corrected_power(q, candidate_ns=(), n_resamples=50, seed=13)

    weak = _synth_quantities(seed=1, correlated=False)         # risk ~ independent of regret
    result = DPOW.simulate_corrected_power(
        weak, candidate_ns=(1000, 1500), n_resamples=200, seed=13,
        shrink_scenarios=(1.0, 0.75, 0.5), selection_scenario=0.75, target=0.80)
    assert result["underpowered"] and result["selected_n"] is None


def test_stress_table_is_non_selective_and_monotone() -> None:
    q = _synth_quantities()
    result = DPOW.simulate_corrected_power(
        q, candidate_ns=(1000,), n_resamples=200, seed=17,
        shrink_scenarios=(1.0, 0.75, 0.5), selection_scenario=0.75, target=0.80)
    stress = result["per_n"][1000]["stress_table"]
    assert "NON-SELECTIVE" in stress["label"]
    assert set(stress["reactive_frac"]["scenarios"]) == {"0.24", "0.22", "0.20"}
    assert set(stress["high_regret_frac"]["scenarios"]) == {"0.30", "0.20", "0.15"}
    c6 = [stress["reactive_frac"]["scenarios"][key]["c6_marginal_power"]
          for key in ("0.24", "0.22", "0.20")]
    assert c6[0] >= c6[1] >= c6[2], "lowering the reactive centre cannot raise c6 power"
    c2 = [stress["high_regret_frac"]["scenarios"][key]["c2_marginal_power"]
          for key in ("0.30", "0.20", "0.15")]
    assert c2[0] >= c2[1] >= c2[2]


# --- 5. the create-once report, audit prerequisite and namespace isolation ---------------------------------------


def _fake_audit(tmp_path: Path, status: str, admissible: list[int]) -> None:
    binding = tmp_path / PROT.SOURCE_BINDING_NAME
    binding.write_text(json.dumps({"plan_id": PROT.PLAN_ID, "status": PROT.STATUS_AUDIT_PASS}),
                       encoding="utf-8")
    audit = {
        "plan_id": PROT.PLAN_ID,
        "status": status,
        "capacity": {"screen_admissible_ns": admissible,
                     "confirm_admissible_ns": admissible,
                     "confirm_max_admissible": max(admissible) if admissible else None},
    }
    (tmp_path / PROT.POOL_AUDIT_NAME).write_text(json.dumps(audit), encoding="utf-8")


def test_report_requires_audit_pass_and_is_create_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(PROT, "D2F_DIR", tmp_path)
    monkeypatch.setattr(DPOW, "POWER_DIR", tmp_path / "power")
    monkeypatch.setattr(DPOW, "load_d2e_validation_quantities",
                        lambda: _synth_quantities())
    monkeypatch.setattr(DPOW, "assert_old_report_parity",
                        lambda q: {"old_status": "D2E_SCREEN_UNDERPOWERED",
                                   "old_joint_power_075_at_n2500": 0.6759,
                                   "note": "quoted old terminal state"})
    written: list[str] = []
    real_write = DPOW.write_once_json
    monkeypatch.setattr(DPOW, "write_once_json",
                        lambda path, payload: (written.append(str(path)),
                                               real_write(path, payload))[1])
    monkeypatch.setattr(DPOW, "POWER_RESAMPLES", 150)

    _fake_audit(tmp_path, PROT.STATUS_CAPACITY_INSUFFICIENT, [])
    with pytest.raises(RuntimeError, match="must not run"):
        DPOW.run_power_correction()

    _fake_audit(tmp_path, PROT.STATUS_AUDIT_PASS, [1000, 1500])
    report = DPOW.run_power_correction()
    assert report["status"] == DPOW.STATUS_POWER_PASS
    assert report["selected_n"] == 1000
    assert report["power_seed"] == 2026122406
    assert report["old_algorithm"]["old_joint_power_075_at_n2500"] == 0.6759
    assert report["old_algorithm"]["old_status"] == "D2E_SCREEN_UNDERPOWERED"
    # every write lands inside the D2F namespace only.
    assert written and all(str(tmp_path) in path for path in written)

    again = DPOW.run_power_correction()
    assert again["written_at"] == report["written_at"], (
        "a rerun reloads the create-once report; it never rewrites it")
