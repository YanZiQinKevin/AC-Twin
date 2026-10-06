"""Stage 4C-R2 confirmation power diagnostic tests (plan sections 10, 11, 14).

Blocking invariants proved here (task-5 brief / plan section 14.13):
- the diagnostic reads ONLY the create-once per-group quantities NPZ (+ its receipt) —
  never raw predictions, never rebuilt truth;
- the covariance comes DIRECTLY from this run's whole-group paired bootstrap draws
  (entropy 2026092702); a 95% SE is NEVER derived from the 80% CI width;
- the confirm gate is simulated as the FULL AND-chain on joint Monte-Carlo draws, never
  as a product of marginal powers;
- the 75% shrink scenario is the ONLY blocking selection scenario (100%/50% are
  sensitivity); positive and negative effects shrink equally and signs are never flipped;
- when no candidate n is viable the diagnostic reports STAGE4CR2_CONFIRMATION_UNDERPOWERED
  and creates NO confirm resource (no lock, no schedule, no collection);
- the diagnostic JSON is create-once and refuses a second write;
- the diagnostic is refused unless the one-shot screen PASSED
  (STAGE4CR2_DEV_READY_FOR_CONFIRMATION).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_statistics as s4cr2st  # noqa: E402
from ac_twin import stage4cr2_power as s4cr2pw  # noqa: E402

G = 400  # small synthetic screen; the statistical unit is still the whole group


def _synthetic_quantities(*, groups=G, seed=7, strong=True):
    """Synthetic screen quantities. ``strong`` gives a near-perfect ensemble vs a
    pure-noise strongest baseline (large positive effects); ``strong=False`` makes the
    ensemble baseline-quality noise (no effect anywhere)."""
    rng = np.random.default_rng(seed)
    regret = rng.gamma(shape=1.0, scale=0.01, size=groups)
    if strong:
        ensemble = regret + rng.normal(0.0, 1e-3, size=groups)
    else:
        ensemble = rng.random(groups)
    by_seed = ensemble[None, :] + rng.normal(0.0, 2e-3, size=(5, groups))
    baselines = {
        "MARGIN_ONLY": regret + rng.normal(0.0, 0.05, size=groups),
        "SEED_VOTE_ENTROPY": rng.random(groups),
        "SEED_MARGIN_STD": rng.random(groups),
        "COMBINED_TOP2_MAX": regret.copy(),
        "CONSTANT_RANDOM": rng.random(groups),
    }
    j_true = rng.gamma(shape=2.0, scale=0.5, size=(groups, s4cr2.N_ACTIONS))
    j_hat = j_true + rng.normal(0.0, 0.05, size=(groups, s4cr2.N_ACTIONS))
    idx = np.arange(groups)
    return {
        "group_seed": idx.astype(np.int64),
        "regret": regret,
        "y_high": (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64),
        "random_regret": regret + 0.05,
        "idle_regret": regret + 0.08,
        "ensemble_risk": ensemble,
        "risk_by_seed": by_seed,
        "mu_only_ensemble": ensemble.copy(),
        "class_only_ensemble": ensemble.copy(),
        "baseline_risks": baselines,
        "strongest_baseline": "SEED_VOTE_ENTROPY",
        "tstar_layer": (idx % 2).astype(np.int64),
        "interaction_layer": ((idx // 2) % 2).astype(np.int64),
        "J_hat": j_hat,
        "J_true": j_true,
        "a_star": idx % s4cr2.N_ACTIONS,
        "t_star": np.where(idx % 2 == 0, 11, 14),
        "interaction_min_distance": np.where((idx // 2) % 2 == 0, 10.0, 40.0),
        "n_groups": int(groups),
        "interaction_threshold": 25.0,
        "qa": {
            "all_complete": True,
            "hashes_match": True,
            "role_seeds_valid": True,
            "determinism_passed": True,
            "one_shot_locks": True,
            "ensemble_is_arithmetic_mean": True,
        },
    }


def _write_screen_npz(tmp_path, *, strong=True, groups=G, seed=7):
    """Persist ONLY the create-once pair: quantities NPZ + receipt (nothing else)."""
    quantities = _synthetic_quantities(groups=groups, seed=seed, strong=strong)
    payload = {}
    for key, value in quantities.items():
        if key in ("baseline_risks", "qa", "strongest_baseline"):
            continue
        payload[key] = value
    for name, risk in quantities["baseline_risks"].items():
        payload[f"baseline_risk__{name}"] = risk
    npz_path = tmp_path / "screen_group_quantities.npz"
    np.savez_compressed(npz_path, **payload)
    receipt = {
        "plan_id": s4cr2.PLAN_ID,
        "strongest_standard_baseline": quantities["strongest_baseline"],
        "qa": quantities["qa"],
    }
    receipt_path = tmp_path / "screen_group_quantities_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return npz_path


def _estimate(npz_path, **overrides):
    kwargs = dict(
        candidate_ns=(2000, 3000),
        n_resamples=150,
        monte_carlo=400,
    )
    kwargs.update(overrides)
    return s4cr2pw.estimate_confirm_power(npz_path, **kwargs)


# --- input discipline ------------------------------------------------------------------------


def test_power_reads_only_per_group_quantities(tmp_path):
    """Only the NPZ + receipt exist; no predictions/raw data/report are present or read."""
    npz_path = _write_screen_npz(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "screen_group_quantities.npz", "screen_group_quantities_receipt.json"}
    report = _estimate(npz_path)
    assert report.n_reference == G
    assert report.plan_id == s4cr2.PLAN_ID
    assert report.screen_quantities_sha256
    assert set(report.per_n) == {2000, 3000}


def test_covariance_comes_from_bootstrap_draws_never_ci_width(tmp_path):
    """The SE scale must equal sqrt(diag(cov)) of an INDEPENDENTLY recomputed paired
    bootstrap of the same quantities; the 80% CI width is not an input anywhere."""
    npz_path = _write_screen_npz(tmp_path)
    n_resamples = 150
    report = _estimate(npz_path, n_resamples=n_resamples)
    quantities = s4cr2st.load_screen_group_quantities(
        npz_path, npz_path.with_name("screen_group_quantities_receipt.json"))
    independent = s4cr2st.paired_group_bootstrap(
        quantities, seed=s4cr2.BOOTSTRAP_ENTROPY, n_resamples=n_resamples)
    cov = s4cr2pw.symmetrised_psd_covariance(
        independent.draws[:, list(s4cr2pw.POWER_QUANTITIES)])
    expected_se = np.sqrt(np.diag(cov))
    got = np.array([report.se_reference[name] for name in s4cr2pw.POWER_QUANTITY_NAMES])
    assert np.allclose(got, expected_se, rtol=1e-12, atol=0.0)
    expected_theta = np.array([report.theta_hat[name]
                               for name in s4cr2pw.POWER_QUANTITY_NAMES])
    assert np.allclose(
        expected_theta, independent.point[list(s4cr2pw.POWER_QUANTITIES)],
        rtol=0.0, atol=0.0)
    # fully deterministic: identical inputs -> identical report (no CI-width channel).
    again = _estimate(npz_path, n_resamples=n_resamples)
    assert report.to_dict() == again.to_dict()


# --- the full AND-chain on joint draws ---------------------------------------------------------


def _good_draws(rows=200):
    draws = np.zeros((rows, s4cr2pw.N_POWER_QUANTITIES))
    draws[:, s4cr2pw.P_I_AURC] = 0.20
    draws[:, s4cr2pw.P_I_REGRET80] = 0.15
    draws[:, s4cr2pw.P_PR] = 0.30
    draws[:, s4cr2pw.P_TRIM] = 0.10
    draws[:, s4cr2pw.P_STRATA] = 0.05
    draws[:, s4cr2pw.P_SEED_I_AURC] = 0.10
    draws[:, s4cr2pw.P_SEED_I_R80] = 0.10
    return draws


def test_and_chain_single_failure_fails_whole_chain():
    se = np.full(s4cr2pw.N_POWER_QUANTITIES, 0.01)
    draws = _good_draws()
    assert s4cr2pw.evaluate_confirm_draws(draws, se)["passed"].all()
    bad = draws.copy()
    bad[:, s4cr2pw.P_PR] = -0.5  # one condition fails -> the whole AND-chain fails
    assert not s4cr2pw.evaluate_confirm_draws(bad, se)["passed"].any()


def test_joint_power_is_not_a_product_of_marginals():
    """Conditions fail on DISJOINT halves of the draws: each marginal power is 0.5 (product
    0.25) but the joint AND power is exactly 0."""
    se = np.full(s4cr2pw.N_POWER_QUANTITIES, 0.01)
    draws = _good_draws(rows=200)
    draws[:100, s4cr2pw.P_I_AURC] = -1.0          # condition A fails on first half
    draws[100:, s4cr2pw.P_TRIM[0]] = -1.0         # condition B fails on second half
    evaluation = s4cr2pw.evaluate_confirm_draws(draws, se)
    assert evaluation["conditions"]["i_aurc_regret"].mean() == pytest.approx(0.5)
    assert evaluation["conditions"]["trims_positive"].mean() == pytest.approx(0.5)
    assert float(evaluation["passed"].mean()) == 0.0  # AND-chain, not 0.5 * 0.5


# --- 75% selection scenario and shrink semantics -------------------------------------------------


def _theta_and_cov(i_aurc, se_aurc, *, seed_effect=0.10):
    theta = np.zeros(s4cr2pw.N_POWER_QUANTITIES)
    theta[s4cr2pw.P_I_AURC] = i_aurc
    theta[s4cr2pw.P_I_REGRET80] = 0.15
    theta[s4cr2pw.P_PR] = 0.30
    theta[s4cr2pw.P_TRIM] = 0.10
    theta[s4cr2pw.P_STRATA] = 0.05
    theta[s4cr2pw.P_SEED_I_AURC] = seed_effect
    theta[s4cr2pw.P_SEED_I_R80] = seed_effect
    cov = np.eye(s4cr2pw.N_POWER_QUANTITIES) * 1e-6
    cov[s4cr2pw.P_I_AURC, s4cr2pw.P_I_AURC] = se_aurc ** 2
    return theta, cov


def test_selection_scenario_75pct_is_the_only_binding_judge():
    """100% shrink passes at every n, but 75% fails -> NO viable n is reported."""
    # SE at n=2000 is se_ref*2 (covariance scales n_ref/n); at n=3000 it is se_ref*1.633.
    # 100% shrink: per-draw CI95 pass prob ~0.84 at n=2000 (passes the 0.80 target);
    # 75% shrink: mean 0.075 -> pass prob ~0.77 even at the largest candidate (fails).
    theta, cov = _theta_and_cov(0.10, 0.017)
    result = s4cr2pw.simulate_power(
        theta, cov, candidate_ns=(2000, 3000), n_reference=8000,
        monte_carlo=500, seed=3, shrink_scenarios=(1.0, 0.75, 0.5),
        selection_scenario=0.75, target=0.80)
    assert result["per_n"][2000]["scenarios"]["1.00"]["joint_power"] >= 0.80
    assert result["per_n"][2000]["scenarios"]["0.75"]["joint_power"] < 0.80
    assert result["selected_n"] is None
    assert result["underpowered"] is True


def test_shrink_scales_positive_and_negative_equally_never_flipping_signs():
    """Negative effects stay negative under every shrink; they can never pass a '>0'
    condition, so a screen with two negative seed effects is unpowered everywhere."""
    theta, cov = _theta_and_cov(0.20, 0.001, seed_effect=0.10)
    theta[s4cr2pw.P_SEED_I_AURC[3]] = -0.05
    theta[s4cr2pw.P_SEED_I_AURC[4]] = -0.05  # only 3/5 seeds positive at ANY shrink
    result = s4cr2pw.simulate_power(
        theta, cov, candidate_ns=(2000,), n_reference=8000,
        monte_carlo=300, seed=5, shrink_scenarios=(1.0, 0.75, 0.5),
        selection_scenario=0.75, target=0.80)
    for n_entry in result["per_n"].values():
        for scenario in n_entry["scenarios"].values():
            assert scenario["marginal_power"]["seeds_i_aurc_positive"] == 0.0
            assert scenario["joint_power"] == 0.0
    assert result["underpowered"] is True


def test_smallest_viable_n_is_selected():
    theta, cov = _theta_and_cov(0.20, 0.010)
    result = s4cr2pw.simulate_power(
        theta, cov, candidate_ns=(2000, 3000, 4000, 5000), n_reference=8000,
        monte_carlo=500, seed=11, shrink_scenarios=(1.0, 0.75, 0.5),
        selection_scenario=0.75, target=0.80)
    assert result["selected_n"] == 2000  # the MINIMUM n meeting the target
    assert result["underpowered"] is False
    assert set(result["sensitivity_at_max_n"]) == {"1.00", "0.75", "0.50"}


# --- driver-level: create-once, gating, no confirm resources ------------------------------------


def _tmp_config(tmp_path):
    """Real frozen config values with run_dir/data_root redirected to tmp space."""
    config = json.loads((ROOT / "experiments/stage4_reliability/stage4c_r2/"
                         "stage4cr2_config.json").read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "stage4c_r2")
    config["data_root"] = str(tmp_path / "stage4cr2_merge_multiaction_dev")
    return config


def _fabricate_passed_screen(config, *, strong=True):
    dev = Path(config["run_dir"]) / "development"
    dev.mkdir(parents=True, exist_ok=True)
    npz_path = _write_screen_npz(dev, strong=strong)
    report = {
        "plan_id": s4cr2.PLAN_ID,
        "terminal_state": s4cr2.STATUS_DEV_READY,
        "passed": True,
    }
    (dev / "development_screen_report.json").write_text(
        json.dumps(report), encoding="utf-8")
    return npz_path


def _files_under(root):
    return {str(p.relative_to(root)).replace("\\", "/")
            for p in Path(root).rglob("*") if p.is_file()}


def test_power_diagnostic_writes_create_once_and_no_confirm_resources(tmp_path):
    config = _tmp_config(tmp_path)
    _fabricate_passed_screen(config, strong=True)
    before = _files_under(tmp_path)
    outcome = s4cr2pw.run_confirmation_power_diagnostic(
        config, n_resamples=100, monte_carlo=200)
    after = _files_under(tmp_path)
    created = after - before
    assert created == {"stage4c_r2/development/confirmation_power_diagnostic.json"}
    # NO confirm lock / schedule / collection resource may appear (plan section 10).
    assert not any("confirm" in name and "confirmation_power_diagnostic" not in name
                   for name in created)
    diagnostic = json.loads(
        (Path(config["run_dir"]) / "development" /
         "confirmation_power_diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["plan_id"] == s4cr2.PLAN_ID
    assert diagnostic["status"] in (s4cr2.STATUS_DEV_READY, s4cr2.STATUS_UNDERPOWERED)
    assert diagnostic["confirm_resources_created"] is False
    assert outcome["diagnostic_path"].endswith("confirmation_power_diagnostic.json")


def test_power_diagnostic_underpowered_reports_status_and_no_confirm_resources(tmp_path):
    config = _tmp_config(tmp_path)
    _fabricate_passed_screen(config, strong=False)  # no effect anywhere
    before = _files_under(tmp_path)
    outcome = s4cr2pw.run_confirmation_power_diagnostic(
        config, n_resamples=100, monte_carlo=200)
    diagnostic = json.loads(Path(outcome["diagnostic_path"]).read_text(encoding="utf-8"))
    assert diagnostic["selected_n"] is None
    assert diagnostic["underpowered"] is True
    assert diagnostic["status"] == s4cr2.STATUS_UNDERPOWERED
    assert (_files_under(tmp_path) - before) == {
        "stage4c_r2/development/confirmation_power_diagnostic.json"}


def test_power_diagnostic_is_create_once(tmp_path):
    config = _tmp_config(tmp_path)
    _fabricate_passed_screen(config, strong=True)
    s4cr2pw.run_confirmation_power_diagnostic(config, n_resamples=50, monte_carlo=100)
    with pytest.raises(FileExistsError):
        s4cr2pw.run_confirmation_power_diagnostic(config, n_resamples=50, monte_carlo=100)


def test_power_diagnostic_refused_unless_screen_passed(tmp_path):
    config = _tmp_config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    dev.mkdir(parents=True, exist_ok=True)
    # no screen report at all -> refused
    with pytest.raises(ValueError):
        s4cr2pw.run_confirmation_power_diagnostic(config, n_resamples=50, monte_carlo=100)
    # a failed screen (line closed) -> refused; no diagnostic file may appear
    (dev / "development_screen_report.json").write_text(json.dumps({
        "plan_id": s4cr2.PLAN_ID,
        "terminal_state": s4cr2.STATUS_LINE_CLOSED,
        "passed": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        s4cr2pw.run_confirmation_power_diagnostic(config, n_resamples=50, monte_carlo=100)
    assert not (dev / "confirmation_power_diagnostic.json").exists()
