"""Blocking tests for the Stage 4D D1 joint power simulation (plan §6.1, §11 item 7).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 4. These tests pin:

* the ONLY power input is the R2 development ``screen_group_quantities.npz`` (+ its
  receipt); the power module never re-selects a model with it;
* SE and covariance come DIRECTLY from per-group whole-group paired bootstrap draws
  (recomputed with the frozen R2 screen entropy); an SE is NEVER back-computed from any
  historical CI width (plan §6.1, verbatim);
* the confirm gate is simulated as the FULL AND-chain on joint Monte-Carlo draws, never
  a product of marginal powers; the AND chain is §6.1 verbatim (``I_AURC>=0.05``,
  ``I_REGRET@80>=0.03``, both 95% CI lower bounds > 0, ``PR-AUC-prev`` 95% CI lower
  bound > 0, and at least 4/5 model seeds positive in BOTH directions on the SAME seed);
* the ONLY blocking selection judge is the 50%-retention scenario; the minimal n with
  joint pass rate >= 0.80 is selected; 100%/75%/25% are reported side by side;
* retention scales positive and negative effects equally and never flips a sign;
* n=4000 still insufficient -> ``STAGE4D_OOD_CONFIRMATION_UNDERPOWERED`` and NO confirm
  resource (no protocol lock, no collection path) is created;
* ``power_report.json`` is create-once.
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
from ac_twin import stage4d_ood_statistics as d0  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402

import run_stage4d_ood as runner  # noqa: E402

G = 400  # small synthetic screen; the statistical unit is still the whole group


def _synthetic_quantities(*, groups=G, seed=7, strong=True):
    """Synthetic screen quantities (same contract as the R2 screen NPZ).

    ``strong`` gives a near-perfect ensemble vs a pure-noise primary baseline (large
    positive effects); ``strong=False`` makes the ensemble baseline-quality noise (no
    effect anywhere)."""
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


def _simulate(npz_path, **overrides):
    kwargs = dict(
        candidate_ns=(1000, 2000),
        retention_factors=(1.0, 0.75, 0.5, 0.25),
        seed=prot.POWER_ENTROPY,
        n_resamples=150,
        monte_carlo=400,
    )
    kwargs.update(overrides)
    return d0.simulate_d1_power(npz_path, **kwargs)


# --- quantity subset and input discipline --------------------------------------------------


def test_power_quantity_subset_is_the_and_chain_vector():
    """The §6.1 AND-chain quantity subset: I_AURC, I_REGRET@80, PR-AUC-prev, five per-seed
    I_AURC and five per-seed I_REGRET@80 — 13 quantities in the frozen order."""
    assert tuple(d0.D1_POWER_QUANTITIES) == (
        s4cr2st.Q_I_AURC, s4cr2st.Q_I_REGRET80, s4cr2st.Q_PR,
        *s4cr2st.SEED_I_AURC_INDEX, *s4cr2st.SEED_I_R80_INDEX)
    assert d0.N_D1_POWER_QUANTITIES == 13
    assert tuple(d0.D1_POWER_QUANTITY_NAMES) == (
        "i_aurc_regret", "i_regret80", "pr_auc_minus_prevalence",
        "seed8101_i_aurc", "seed8202_i_aurc", "seed8303_i_aurc",
        "seed8404_i_aurc", "seed8505_i_aurc",
        "seed8101_i_regret80", "seed8202_i_regret80", "seed8303_i_regret80",
        "seed8404_i_regret80", "seed8505_i_regret80")


def test_power_reads_only_screen_quantities_npz_and_receipt(tmp_path):
    """Only the NPZ + receipt exist; the report binds both hashes; n_reference = groups."""
    npz_path = _write_screen_npz(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "screen_group_quantities.npz", "screen_group_quantities_receipt.json"}
    report = _simulate(npz_path)
    assert report["n_reference"] == G
    assert report["plan_id"] == prot.PLAN_ID
    assert report["screen_quantities_sha256"]
    assert report["screen_quantities_receipt_sha256"]
    assert set(report["per_n"]) == {1000, 2000}


def test_and_chain_thresholds_are_the_pinned_plan_values():
    """§6.1 verbatim: I_AURC>=0.05, I_REGRET@80>=0.03, >=4/5 seeds both-positive."""
    assert prot.POWER_AND_CHAIN == {
        "i_aurc_min": 0.05, "i_regret80_min": 0.03, "min_positive_seeds": 4}
    assert prot.POWER_SELECTION_RETENTION == 0.50
    assert prot.POWER_TARGET == 0.80
    assert prot.CANDIDATE_NS == (1000, 2000, 3000, 4000)
    assert prot.RETENTION_FACTORS == (1.0, 0.75, 0.5, 0.25)
    assert prot.POWER_MONTE_CARLO == 10000
    assert prot.POWER_ENTROPY == 2026104201


# --- SE/covariance from bootstrap draws ONLY (never CI-width back-calculation) --------------


def test_se_and_theta_come_from_bootstrap_draws_never_ci_width(tmp_path):
    """The SE scale must equal sqrt(diag(cov)) of an INDEPENDENTLY recomputed paired
    bootstrap of the same quantities (frozen R2 screen entropy); no CI-width channel
    exists anywhere in the computation."""
    npz_path = _write_screen_npz(tmp_path)
    n_resamples = 150
    report = _simulate(npz_path, n_resamples=n_resamples)
    quantities = s4cr2st.load_screen_group_quantities(
        npz_path, npz_path.with_name("screen_group_quantities_receipt.json"))
    independent = s4cr2st.paired_group_bootstrap(
        quantities, seed=s4cr2.BOOTSTRAP_ENTROPY, n_resamples=n_resamples)
    from ac_twin import stage4cr2_power as s4cr2pw
    cov = s4cr2pw.symmetrised_psd_covariance(
        independent.draws[:, list(d0.D1_POWER_QUANTITIES)])
    expected_se = np.sqrt(np.diag(cov))
    got = np.array([report["se_reference"][name]
                    for name in d0.D1_POWER_QUANTITY_NAMES])
    assert np.allclose(got, expected_se, rtol=1e-12, atol=0.0)
    expected_theta = independent.point[list(d0.D1_POWER_QUANTITIES)]
    got_theta = np.array([report["theta_hat"][name]
                          for name in d0.D1_POWER_QUANTITY_NAMES])
    assert np.allclose(got_theta, expected_theta, rtol=0.0, atol=0.0)
    # fully deterministic: identical inputs -> identical report (no CI-width channel).
    again = _simulate(npz_path, n_resamples=n_resamples)
    assert report == again


# --- the FULL AND-chain on joint draws ------------------------------------------------------


def _good_draws(rows=200):
    draws = np.zeros((rows, d0.N_D1_POWER_QUANTITIES))
    draws[:, d0.P_I_AURC] = 0.20
    draws[:, d0.P_I_REGRET80] = 0.15
    draws[:, d0.P_PR] = 0.30
    draws[:, d0.P_SEED_I_AURC] = 0.10
    draws[:, d0.P_SEED_I_R80] = 0.10
    return draws


def test_and_chain_single_failure_fails_whole_chain():
    se = np.full(d0.N_D1_POWER_QUANTITIES, 0.01)
    draws = _good_draws()
    assert d0.evaluate_d1_power_draws(draws, se)["passed"].all()
    bad = draws.copy()
    bad[:, d0.P_PR] = -0.5  # one condition fails -> the whole AND-chain fails
    assert not d0.evaluate_d1_power_draws(bad, se)["passed"].any()


def test_joint_power_is_not_a_product_of_marginals():
    """Conditions fail on DISJOINT halves of the draws: each marginal power is 0.5
    (product 0.25) but the joint AND power is exactly 0."""
    se = np.full(d0.N_D1_POWER_QUANTITIES, 0.01)
    draws = _good_draws(rows=200)
    draws[:100, d0.P_I_AURC] = -1.0     # condition A fails on the first half
    draws[100:, d0.P_I_REGRET80] = -1.0  # condition B fails on the second half
    evaluation = d0.evaluate_d1_power_draws(draws, se)
    assert evaluation["conditions"]["i_aurc_regret"].mean() == pytest.approx(0.5)
    assert evaluation["conditions"]["i_regret80"].mean() == pytest.approx(0.5)
    assert float(evaluation["passed"].mean()) == 0.0  # AND-chain, not 0.5 * 0.5


def test_seed_condition_requires_same_seed_positive_in_both_directions():
    """§6.1 'at least 4/5 model-seed both-direction consistent': the count is over seeds
    positive on I_AURC AND I_REGRET@80 SIMULTANEOUSLY — not two marginal counts."""
    se = np.full(d0.N_D1_POWER_QUANTITIES, 0.01)
    draws = _good_draws(rows=50)
    # seeds {0,1,2,3} positive on I_AURC; seeds {1,2,3,4} positive on I_REGRET@80:
    # marginal counts are 4 and 4, but only seeds {1,2,3} are positive in BOTH -> fail.
    draws[:, d0.P_SEED_I_AURC[4]] = -0.10
    draws[:, d0.P_SEED_I_R80[0]] = -0.10
    evaluation = d0.evaluate_d1_power_draws(draws, se)
    assert not evaluation["conditions"]["seeds_both_directions_positive"].any()
    assert not evaluation["passed"].any()
    # repairing seed 0's I_REGRET@80 makes seeds {0,1,2,3} both-positive -> pass.
    fixed = draws.copy()
    fixed[:, d0.P_SEED_I_R80[0]] = 0.10
    evaluation2 = d0.evaluate_d1_power_draws(fixed, se)
    assert evaluation2["conditions"]["seeds_both_directions_positive"].all()
    assert evaluation2["passed"].all()


def test_ci_lower_bound_is_estimate_minus_196_se_from_draw_covariance():
    """The 95% CI lower bound inside the chain is draw - 1.96*SE with SE from the
    scaled draw covariance; a draw exactly at the bound FAILS (strict > 0)."""
    se = np.full(d0.N_D1_POWER_QUANTITIES, 0.05)
    draws = _good_draws(rows=10)
    # I_AURC draw = 1.96*SE exactly -> CI lower == 0 -> fails (strict inequality).
    draws[:, d0.P_I_AURC] = 1.96 * 0.05
    evaluation = d0.evaluate_d1_power_draws(draws, se)
    assert not evaluation["conditions"]["i_aurc_regret"].any()
    # epsilon above -> passes (and the >= 0.05 point threshold holds too).
    draws[:, d0.P_I_AURC] = 1.96 * 0.05 + 1e-9
    evaluation2 = d0.evaluate_d1_power_draws(draws, se)
    assert evaluation2["conditions"]["i_aurc_regret"].all()


# --- retention semantics and the 50%-retention selection rule --------------------------------


def _theta_and_cov(i_aurc, se_aurc, *, seed_effect=0.10):
    theta = np.zeros(d0.N_D1_POWER_QUANTITIES)
    theta[d0.P_I_AURC] = i_aurc
    theta[d0.P_I_REGRET80] = 0.15
    theta[d0.P_PR] = 0.30
    theta[d0.P_SEED_I_AURC] = seed_effect
    theta[d0.P_SEED_I_R80] = seed_effect
    cov = np.eye(d0.N_D1_POWER_QUANTITIES) * 1e-6
    cov[d0.P_I_AURC, d0.P_I_AURC] = se_aurc ** 2
    return theta, cov


def _simulate_joint(theta, cov, **overrides):
    kwargs = dict(
        candidate_ns=(1000, 2000, 3000, 4000), n_reference=8000,
        retention_factors=(1.0, 0.75, 0.5, 0.25), monte_carlo=500, seed=3,
        selection_retention=0.50, target=0.80)
    kwargs.update(overrides)
    return d0.simulate_d1_joint_power(theta, cov, **kwargs)


def test_selection_scenario_50pct_is_the_only_blocking_judge():
    """100%/75% pass at every n, but 50% fails -> NO viable n (100%/75%/25% are
    sensitivity only; the 50% AND-chain is the ONLY blocking judge)."""
    # 100%: pass prob ~0.84 at n=1000 (passes 0.80); 50%: mean 0.05 -> ~0.55 (fails).
    theta, cov = _theta_and_cov(0.10, 0.012)
    result = _simulate_joint(theta, cov)
    assert result["per_n"][1000]["scenarios"]["1.00"]["joint_power"] >= 0.80
    assert result["per_n"][4000]["scenarios"]["0.50"]["joint_power"] < 0.80
    assert result["selected_n"] is None
    assert result["underpowered"] is True


def test_smallest_viable_n_is_selected_at_50pct_retention():
    theta, cov = _theta_and_cov(0.20, 0.010)
    result = _simulate_joint(theta, cov)
    assert result["selected_n"] == 1000  # the MINIMUM n meeting the 0.80 target
    assert result["underpowered"] is False
    for n in (1000, 2000, 3000, 4000):
        assert set(result["per_n"][n]["scenarios"]) == {"1.00", "0.75", "0.50", "0.25"}
        assert result["per_n"][n]["binding_condition"] in \
            result["per_n"][n]["scenarios"]["0.50"]["marginal_power"]


def test_retention_scales_positive_and_negative_equally_never_flipping_signs():
    """Negative effects stay negative under every retention; they can never pass a '>0'
    condition, so a screen with two negative seed effects is unpowered everywhere."""
    theta, cov = _theta_and_cov(0.20, 0.001, seed_effect=0.10)
    theta[d0.P_SEED_I_AURC[3]] = -0.05
    theta[d0.P_SEED_I_AURC[4]] = -0.05  # only 3/5 seeds both-positive at ANY retention
    result = _simulate_joint(theta, cov, candidate_ns=(1000,))
    for n_entry in result["per_n"].values():
        for scenario in n_entry["scenarios"].values():
            assert scenario["marginal_power"]["seeds_both_directions_positive"] == 0.0
            assert scenario["joint_power"] == 0.0
    assert result["underpowered"] is True


def test_underpowered_when_max_candidate_n_falls_short():
    theta, cov = _theta_and_cov(0.06, 0.030)  # weak effect, large SE
    result = _simulate_joint(theta, cov)
    assert result["selected_n"] is None
    assert result["underpowered"] is True
    assert result["sensitivity_at_max_n"]["0.50"] < 0.80


# --- driver-level: the runner power phase ---------------------------------------------------


def _tmp_config(tmp_path):
    """Real frozen config values with run_dir/data_root/r2 paths redirected to tmp."""
    config = json.loads(prot.CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "stage4d_ood"
    config["run_dir"] = str(run_dir)
    config["data_root"] = str(tmp_path / "data")
    config["r2_development_dir"] = str(tmp_path / "r2_dev")
    config["paths"] = dict(config["paths"])
    config["paths"]["run_dir"] = str(run_dir)
    for key in list(config["paths"]):
        config["paths"][key] = str(run_dir / Path(key).name)
    config["paths"]["power_report"] = str(run_dir / "power" / "power_report.json")
    config["paths"]["ood_profile_selection"] = str(
        run_dir / "pilot" / "ood_profile_selection.json")
    config["paths"]["protocol_lock"] = str(run_dir / "protocol" / "protocol_lock.json")
    return config


def _fabricate_power_inputs(config, *, strong=True):
    """Synthetic screen NPZ (hash-bound into the config) + a D0-selected profile."""
    r2_dev = Path(config["r2_development_dir"])
    r2_dev.mkdir(parents=True, exist_ok=True)
    npz_path = _write_screen_npz(r2_dev, strong=strong)
    from ac_twin.stage1_data import sha256_file
    config["r2_pinned_sources"] = dict(config["r2_pinned_sources"])
    config["r2_pinned_sources"][
        "experiments/stage4_reliability/stage4c_r2/development/"
        "screen_group_quantities.npz"] = sha256_file(npz_path)
    selection_path = Path(config["paths"]["ood_profile_selection"])
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({
        "plan_id": prot.PLAN_ID,
        "selected_profile": "JOINT_OOD",
        "status": prot.STATUS_D0_PROFILE_SELECTED,
    }), encoding="utf-8")
    return npz_path


def _files_under(root):
    return {str(p.relative_to(root)).replace("\\", "/")
            for p in Path(root).rglob("*") if p.is_file()}


def test_power_phase_writes_create_once_report_and_freezes_n(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    config = _tmp_config(tmp_path)
    _fabricate_power_inputs(config, strong=True)
    before = _files_under(tmp_path)
    outcome = runner.run_phase("power", config=config)
    after = _files_under(tmp_path)
    assert after - before == {"stage4d_ood/power/power_report.json"}
    report = json.loads(
        Path(config["paths"]["power_report"]).read_text(encoding="utf-8"))
    assert report["plan_id"] == prot.PLAN_ID
    assert report["status"] == prot.STATUS_D1_POWER_FROZEN
    assert report["selected_n"] in prot.CANDIDATE_NS
    assert report["underpowered"] is False
    assert outcome["selected_n"] == report["selected_n"]
    # the four retention scenarios are reported side by side for every candidate n.
    for n_entry in report["per_n"].values():
        assert set(n_entry["scenarios"]) == {"1.00", "0.75", "0.50", "0.25"}
    # create-once resume: verifies, never rewrites.
    again = runner.run_phase("power", config=config)
    assert again["verification"] == "existing_power_report_verified_no_overwrite"


def test_power_phase_underpowered_writes_terminal_state_and_no_confirm_resources(
        tmp_path, monkeypatch):
    """n=4000 still insufficient -> STAGE4D_OOD_CONFIRMATION_UNDERPOWERED hard stop:
    the report is written honestly and NO protocol lock / collection path is created."""
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    config = _tmp_config(tmp_path)
    _fabricate_power_inputs(config, strong=False)  # no effect anywhere
    before = _files_under(tmp_path)
    with pytest.raises(runner.TerminalState) as excinfo:
        runner.run_phase("power", config=config)
    assert excinfo.value.status == prot.STATUS_UNDERPOWERED
    created = _files_under(tmp_path) - before
    assert created == {"stage4d_ood/power/power_report.json"}
    report = json.loads(
        Path(config["paths"]["power_report"]).read_text(encoding="utf-8"))
    assert report["status"] == prot.STATUS_UNDERPOWERED
    assert report["selected_n"] is None
    assert report["underpowered"] is True
    # no confirm resource: no lock, no collection/inference directories.
    assert not Path(config["paths"]["protocol_lock"]).exists()
    assert not (Path(config["run_dir"]) / "collection").exists()
    # the hard stop is absorbing: resume re-raises, never proceeds to the lock.
    with pytest.raises(runner.TerminalState) as excinfo2:
        runner.run_phase("power", config=config)
    assert excinfo2.value.status == prot.STATUS_UNDERPOWERED


def test_power_phase_refused_without_d0_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    config = _tmp_config(tmp_path)
    r2_dev = Path(config["r2_development_dir"])
    r2_dev.mkdir(parents=True, exist_ok=True)
    _write_screen_npz(r2_dev, strong=True)
    # no ood_profile_selection.json at all -> refused before any simulation.
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run_phase("power", config=config)
    assert not Path(config["paths"]["power_report"]).exists()


def test_power_phase_rejects_screen_quantities_hash_drift(tmp_path, monkeypatch):
    """The power input must be the hash-pinned R2 screen quantities; drift fails closed."""
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    config = _tmp_config(tmp_path)
    _fabricate_power_inputs(config, strong=True)
    config["r2_pinned_sources"][
        "experiments/stage4_reliability/stage4c_r2/development/"
        "screen_group_quantities.npz"] = "0" * 64  # drifted pin
    with pytest.raises(Exception, match="hash|drift"):
        runner.run_phase("power", config=config)
    assert not Path(config["paths"]["power_report"]).exists()


def test_power_phase_entropy_is_the_reserved_power_entropy(tmp_path, monkeypatch):
    """The Monte-Carlo entropy is the reserved 2026104201, recorded in the report."""
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    config = _tmp_config(tmp_path)
    _fabricate_power_inputs(config, strong=True)
    runner.run_phase("power", config=config)
    report = json.loads(
        Path(config["paths"]["power_report"]).read_text(encoding="utf-8"))
    assert report["monte_carlo"]["entropy"] == prot.POWER_ENTROPY == 2026104201
    assert report["monte_carlo"]["n_draws"] == prot.POWER_MONTE_CARLO
    assert report["bootstrap"]["seed"] == s4cr2.BOOTSTRAP_ENTROPY
    assert "never" in report["bootstrap"]["design"]  # no CI-width SE back-calculation
