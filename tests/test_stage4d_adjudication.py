"""Stage 4D D1 one-shot adjudication Gate-math tests (plan §7, §11 items 8/9/10).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 6. These tests pin the
Gate TASK-OOD (§7.1, 8 conditions) and Gate OOD-RISK (§7.2, 5 conditions) math on
constructed known-answer per-group quantities, the frozen whole-group paired bootstrap
(10,000 resamples, entropy 2026110995, two-sided 95% CI), the fail-closed baseline
denominator rule (INVALID, never an epsilon/absolute difference), the §7.3 terminal-state
priority, and the one-shot lock rules (a second adjudication is refused; a resume never
re-runs a create-once phase and never starts D2).

The statistical unit is the accepted complete GROUP. All quantities are constructed
synthetically with known answers; no Direct checkpoint, no frozen score artifact and no
sealed truth is read here (the gates consume the distilled create-once quantities).
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_stage4d_ood as runner  # noqa: E402
from ac_twin import stage4d_ood_statistics as d0  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402
from ac_twin import stage4cp_statistics as s4cs  # noqa: E402  frozen primitives

NA = prot.N_ACTIONS
THRESHOLD = 29.128996473819946  # the locked R2-train-only initial-distance threshold


# --- synthetic per-group quantities (known-answer, passing by default) ---------------------


def _make_quantities(G=240, seed=0):
    """A valid, gate-PASSING quantities mapping with a clear positive risk signal.

    The ensemble/per-seed/ablation risks track the regret (a good ordering -> positive
    improvement); the primary baseline SEED_VOTE_ENTROPY and the other baselines are
    uncorrelated (a random ordering -> their AURC ~ mean regret), so the ensemble beats
    them. Everything the §7.1/§7.2 gates read is set to a passing value.
    """
    rng = np.random.default_rng(seed)
    n_high = int(round(0.30 * G))
    regret = np.zeros(G, dtype=np.float64)
    regret[:n_high] = rng.uniform(0.05, 0.40, size=n_high)     # clearly high regret
    regret[n_high:] = rng.uniform(0.0, 0.005, size=G - n_high)  # low regret
    rng.shuffle(regret)

    ensemble_risk = regret + rng.normal(0.0, 0.005, size=G)   # near-perfect ranking
    risk_by_seed = np.stack([regret + rng.normal(0.0, 0.02, size=G)
                             for _ in range(len(prot.MODEL_SEEDS))])
    mu_only = regret + rng.normal(0.0, 0.01, size=G)
    class_only = regret + rng.normal(0.0, 0.01, size=G)
    baselines = {
        "SEED_VOTE_ENTROPY": rng.uniform(0.0, 1.0, size=G),   # uncorrelated (primary)
        "MARGIN_ONLY": rng.uniform(0.0, 1.0, size=G),
        "SEED_MARGIN_STD": rng.uniform(0.0, 1.0, size=G),
        "COMBINED_TOP2_MAX": rng.uniform(0.0, 1.0, size=G),
        "CONSTANT_RANDOM": np.full(G, 0.5),                   # ties -> identity order
    }

    j_true = rng.uniform(0.4, 1.6, size=(G, NA))
    j_hat = j_true + rng.normal(0.0, 0.03, size=(G, NA))      # correlated -> Spearman>0
    a_star = np.tile(np.arange(NA, dtype=np.int64), G // NA + 1)[:G]  # 10 plans, 10% each
    a_hat = np.argmin(j_hat, axis=1).astype(np.int64)

    t_star = np.where(np.arange(G) % 2 == 0, 12, 14).astype(np.int64)  # 50/50 layers
    distance = np.where(np.arange(G) % 2 == 0, THRESHOLD - 5.0,
                        THRESHOLD + 5.0).astype(np.float64)
    return {
        "group_seed": np.arange(1000, 1000 + G, dtype=np.int64),
        "regret": regret,
        "y_high": (regret > prot.HIGH_REGRET_THRESHOLD).astype(np.int64),
        "random_regret": regret + rng.uniform(0.02, 0.08, size=G),  # base beats random
        "idle_regret": regret + 0.001,                            # base <= idle
        "ensemble_risk": ensemble_risk,
        "risk_by_seed": risk_by_seed,
        "mu_only_ensemble": mu_only,
        "class_only_ensemble": class_only,
        "tstar_layer": (t_star >= 13).astype(np.int64),
        "interaction_layer": (distance > THRESHOLD).astype(np.int64),
        "J_hat": j_hat,
        "J_true": j_true,
        "a_hat": a_hat,
        "a_star": a_star,
        "t_star": t_star,
        "interaction_min_distance": distance,
        "baseline_risks": baselines,
        "strongest_baseline": "SEED_VOTE_ENTROPY",
        "n_groups": int(G),
        "interaction_threshold": THRESHOLD,
        "qa": {key: True for key in d0.D1_QA_KEYS},
    }


def _domain_shift(smd_pass=True, auc_pass=True):
    return {
        "smd_pass": bool(smd_pass),
        "domain_auc_pass": bool(auc_pass),
        "shift_significant": bool(smd_pass and auc_pass),
        "n_shifted": 3 if smd_pass else 0,
        "domain_classifier": {"auc_ci_lower": 0.9 if auc_pass else 0.5,
                              "confidence": 0.95},
    }


def _bootstrap(quantities, n_resamples=400, seed=prot.D1_BOOTSTRAP_ENTROPY):
    return d0.paired_group_bootstrap_d1(
        quantities, seed=seed, n_resamples=n_resamples)


def _override(bootstrap, *, point=None, ci_lower=None, ci_upper=None):
    """A copy of a real bootstrap result with selected slots replaced (terminal tests)."""
    return dataclasses.replace(
        bootstrap,
        point=point if point is not None else bootstrap.point,
        ci_lower=ci_lower if ci_lower is not None else bootstrap.ci_lower,
        ci_upper=ci_upper if ci_upper is not None else bootstrap.ci_upper)


def _evaluate(quantities, bootstrap, *, selected_n=None, domain_shift=None):
    return d0.evaluate_d1_gates(
        quantities, bootstrap,
        selected_n=int(selected_n if selected_n is not None
                       else quantities["n_groups"]),
        domain_shift=domain_shift if domain_shift is not None else _domain_shift())


# --- §11 item 8: AURC direction + coverage index (known answers) ---------------------------


def _tiny_quantities():
    """Four groups with a hand-computable AURC / REGRET@80 (ensemble = ascending regret,
    primary baseline = descending regret)."""
    regret = np.array([0.0, 0.1, 0.2, 0.3])
    G = 4
    baselines = {
        "SEED_VOTE_ENTROPY": np.array([0.3, 0.2, 0.1, 0.0]),   # reversed ordering
        "MARGIN_ONLY": np.array([0.3, 0.2, 0.1, 0.0]),
        "SEED_MARGIN_STD": np.array([0.3, 0.2, 0.1, 0.0]),
        "COMBINED_TOP2_MAX": np.array([0.3, 0.2, 0.1, 0.0]),
        "CONSTANT_RANDOM": np.full(G, 0.5),
    }
    j_true = np.arange(G * NA, dtype=np.float64).reshape(G, NA) + 1.0
    return {
        "group_seed": np.arange(1000, 1000 + G, dtype=np.int64),
        "regret": regret,
        "y_high": (regret > prot.HIGH_REGRET_THRESHOLD).astype(np.int64),
        "random_regret": regret + 0.05,
        "idle_regret": regret + 0.01,
        "ensemble_risk": regret.copy(),                        # perfect ascending order
        "risk_by_seed": np.stack([regret.copy() for _ in range(len(prot.MODEL_SEEDS))]),
        "mu_only_ensemble": regret.copy(),
        "class_only_ensemble": regret.copy(),
        "tstar_layer": np.array([0, 0, 1, 1]),
        "interaction_layer": np.array([0, 0, 1, 1]),
        "J_hat": j_true.copy(),                                 # perfect Spearman
        "J_true": j_true,
        "a_hat": np.zeros(G, dtype=np.int64),
        "a_star": np.zeros(G, dtype=np.int64),
        "t_star": np.array([11, 12, 13, 14]),
        "interaction_min_distance": np.array([1.0, 2.0, 40.0, 50.0]),
        "baseline_risks": baselines,
        "strongest_baseline": "SEED_VOTE_ENTROPY",
        "n_groups": G,
        "interaction_threshold": THRESHOLD,
        "qa": {key: True for key in d0.D1_QA_KEYS},
    }


def test_aurc_direction_and_value_known_answer():
    """A better (ascending-regret) ordering lowers AURC -> positive relative improvement."""
    quantities = _tiny_quantities()
    vector = d0.compute_d1_quantity_vector(np.arange(4), quantities)
    # ensemble order [0,1,2,3]: R(k)=cumsum([0,.1,.2,.3])/[1..4]=[0,.05,.1,.15] -> 0.075
    assert vector[d0.Q_AURC_MODEL] == pytest.approx(0.075)
    # baseline order [3,2,1,0]: regret[[3,2,1,0]]=[.3,.2,.1,0]; R(k)=[.3,.25,.2,.15] -> 0.225
    assert vector[d0.Q_AURC_BASELINE] == pytest.approx(0.225)
    # improvement = (0.225 - 0.075) / 0.225 = 2/3; a better ordering gives I_AURC > 0.
    assert vector[d0.Q_I_AURC] == pytest.approx(2.0 / 3.0)
    assert vector[d0.Q_I_AURC] > 0.0


def test_coverage_index_regret_at_80_known_answer():
    """REGRET@80 uses the frozen round(coverage*G) index, never an own floor/ceil."""
    assert s4cs.coverage_index(0.80, 4) == 3  # round(3.2) == 3, clamped to [1, 4]
    quantities = _tiny_quantities()
    vector = d0.compute_d1_quantity_vector(np.arange(4), quantities)
    # ensemble: mean of the 3 lowest-risk regrets = mean([0,.1,.2]) = 0.1
    assert vector[d0.Q_R80_MODEL] == pytest.approx(0.1)
    # baseline: mean of regret[[3,2,1,0]][:3] = mean([.3,.2,.1]) = 0.2
    assert vector[d0.Q_R80_BASELINE] == pytest.approx(0.2)
    assert vector[d0.Q_I_REGRET80] == pytest.approx(0.5)


# --- §11 item 8: paired bootstrap determinism (fixed entropy recompute elementwise-equal) ---


def test_paired_bootstrap_deterministic_same_entropy():
    quantities = _make_quantities(G=120, seed=1)
    first = d0.paired_group_bootstrap_d1(quantities, seed=prot.D1_BOOTSTRAP_ENTROPY,
                                         n_resamples=60)
    second = d0.paired_group_bootstrap_d1(quantities, seed=prot.D1_BOOTSTRAP_ENTROPY,
                                          n_resamples=60)
    assert np.array_equal(first.draws, second.draws)   # elementwise identical
    assert np.array_equal(first.point, second.point)
    assert first.seed == prot.D1_BOOTSTRAP_ENTROPY == 2026110995
    assert first.confidence == prot.CONFIDENCE == 0.95
    # the identity point estimate equals a direct identity evaluation.
    identity = d0.compute_d1_quantity_vector(np.arange(120), quantities)
    assert np.array_equal(first.point, identity)


def test_paired_bootstrap_draws_share_index_vector():
    """Every quantity of a draw is recomputed from the SAME resample index vector."""
    quantities = _make_quantities(G=80, seed=2)
    bootstrap = d0.paired_group_bootstrap_d1(quantities, seed=prot.D1_BOOTSTRAP_ENTROPY,
                                             n_resamples=25)
    assert bootstrap.draws.shape == (25, d0.N_D1_QUANTITIES)
    # regenerate the draw index vectors with the same generator and re-check draw 0.
    generator = np.random.default_rng(prot.D1_BOOTSTRAP_ENTROPY)
    index0 = generator.integers(0, 80, size=80)
    expected0 = d0.compute_d1_quantity_vector(index0, quantities)
    assert np.array_equal(bootstrap.draws[0], expected0)


# --- §11 item 8: model-seed consistency count -------------------------------------------------


def test_seed_consistency_counts_both_directions():
    quantities = _make_quantities(G=200, seed=3)
    bootstrap = _bootstrap(quantities, n_resamples=300)
    decision = _evaluate(quantities, bootstrap)
    # all five seeds track the regret -> all five positive on both metrics.
    assert decision.risk["seeds_both_directions_positive"] == len(prot.MODEL_SEEDS)
    assert decision.risk["conditions"]["r4_seed_consistency"] is True

    # flip two seeds to a WORSE-than-baseline ordering (negative improvement both ways).
    broken = _make_quantities(G=200, seed=3)
    baseline = broken["baseline_risks"]["SEED_VOTE_ENTROPY"]
    for i in (0, 1):  # reverse the regret ranking -> improvement < 0 vs a random baseline
        broken["risk_by_seed"][i] = -broken["regret"] + baseline * 1e-6
    decision2 = _evaluate(broken, _bootstrap(broken, n_resamples=300))
    assert decision2.risk["seeds_both_directions_positive"] == 3
    assert decision2.risk["conditions"]["r4_seed_consistency"] is False


# --- §7.1 Gate TASK-OOD: the eight conditions -------------------------------------------------


def test_gate_task_ood_all_eight_conditions_pass():
    quantities = _make_quantities(G=240, seed=4)
    decision = _evaluate(quantities, _bootstrap(quantities))
    task = decision.task
    assert task["status"] == d0.STATUS_D1_TASK_QUALIFIED
    expected = (
        "c1_complete_gate_q", "c2_domain_shift", "c3_high_regret_frac",
        "c4_oracle_diversity", "c5_branch_spearman", "c6_base_beats_random",
        "c7_base_not_worse_than_idle", "c8_strata_min_10pct")
    assert tuple(task["conditions"]) == expected
    assert all(task["conditions"].values())


def test_gate_task_ood_degenerate_conditions_map_to_task_degenerate():
    """§7.1: conditions 3/4/8 failure -> STAGE4D_OOD_TASK_DEGENERATE."""
    # condition 3 (high-regret fraction) fails.
    quantities = _make_quantities(G=240, seed=5)
    bootstrap = _bootstrap(quantities)
    point = bootstrap.point.copy()
    point[d0.Q_HIGH_REGRET_FRAC] = 0.05  # < 0.15
    decision = _evaluate(quantities, _override(bootstrap, point=point))
    assert decision.status == prot.STATUS_TASK_DEGENERATE

    # condition 4 (oracle diversity) fails: collapse the oracle to a single action.
    degenerate = _make_quantities(G=240, seed=5)
    degenerate["a_star"] = np.zeros(240, dtype=np.int64)
    decision2 = _evaluate(degenerate, _bootstrap(degenerate))
    assert decision2.status == prot.STATUS_TASK_DEGENERATE

    # condition 8 (strata) fails: one stratum empties.
    strata = _make_quantities(G=240, seed=5)
    strata["tstar_layer"] = np.ones(240, dtype=np.int64)  # no t_star-low stratum
    decision3 = _evaluate(strata, _bootstrap(strata))
    assert decision3.status == prot.STATUS_TASK_DEGENERATE


def test_gate_task_ood_insufficient_conditions_map_to_model_insufficient():
    """§7.1: conditions 2/5/6/7 failure -> INCONCLUSIVE_..._DECISION_MODEL_INSUFFICIENT."""
    # condition 2 (domain shift) fails.
    quantities = _make_quantities(G=240, seed=6)
    decision = _evaluate(quantities, _bootstrap(quantities),
                         domain_shift=_domain_shift(smd_pass=False))
    assert decision.status == prot.STATUS_MODEL_INSUFFICIENT

    # condition 5 (branch Spearman CI lower > 0) fails.
    bootstrap = _bootstrap(quantities)
    ci_lower = bootstrap.ci_lower.copy()
    ci_lower[d0.Q_SPEARMAN_BRANCH] = -0.01
    decision2 = _evaluate(quantities, _override(bootstrap, ci_lower=ci_lower))
    assert decision2.status == prot.STATUS_MODEL_INSUFFICIENT

    # condition 7 (base not worse than idle) fails: idle strictly better than base.
    worse = _make_quantities(G=240, seed=6)
    worse["idle_regret"] = np.maximum(worse["regret"] - 0.05, 0.0)
    decision3 = _evaluate(worse, _bootstrap(worse))
    assert decision3.status == prot.STATUS_MODEL_INSUFFICIENT


# --- §7.2 Gate OOD-RISK + §7.3 terminal priority -----------------------------------------------


def test_gate_ood_risk_all_pass_confirmed():
    quantities = _make_quantities(G=240, seed=7)
    decision = _evaluate(quantities, _bootstrap(quantities))
    assert decision.task["status"] == d0.STATUS_D1_TASK_QUALIFIED
    assert all(decision.risk["conditions"].values())
    assert decision.status == prot.STATUS_CONFIRMED
    assert decision.passed is True
    # the primary endpoints meet the frozen thresholds with positive CI lower bounds.
    point, lower = decision.metrics["point"], decision.metrics["ci_lower"]
    assert point["i_aurc_regret"] >= prot.GATE_OOD_RISK["i_aurc_min"]
    assert lower["i_aurc_regret"] > 0.0
    assert point["i_regret80"] >= prot.GATE_OOD_RISK["i_regret80_min"]
    assert lower["i_regret80"] > 0.0
    assert point["pr_auc_minus_prevalence"] > 0.0
    assert lower["pr_auc_minus_prevalence"] > 0.0


def test_gate_ood_risk_failure_is_not_confirmed_never_falsified():
    """Valid TASK + valid denominators + an unmet risk condition -> NOT_CONFIRMED."""
    quantities = _make_quantities(G=240, seed=8)
    bootstrap = _bootstrap(quantities)
    point = bootstrap.point.copy()
    point[d0.Q_I_AURC] = 0.01  # below the 0.05 threshold
    decision = _evaluate(quantities, _override(bootstrap, point=point))
    assert decision.task["status"] == d0.STATUS_D1_TASK_QUALIFIED
    assert decision.risk["baseline_valid"] is True
    assert decision.status == prot.STATUS_NOT_CONFIRMED
    assert decision.passed is False


# --- §11 item 8: degenerate denominator -> INVALID (fail-closed, no epsilon) -------------------


def test_degenerate_baseline_denominator_is_invalid_analysis():
    """A non-positive/non-finite primary-baseline AURC/REGRET@80 -> INVALID_ANALYSIS.

    TASK otherwise passes; the denominator check sits after the task verdict in the §7.3
    priority, so a degenerate denominator is INVALID even though the task qualified.
    """
    quantities = _make_quantities(G=240, seed=9)
    bootstrap = _bootstrap(quantities)
    for slot in (d0.Q_AURC_BASELINE, d0.Q_R80_BASELINE):
        point = bootstrap.point.copy()
        point[slot] = 0.0  # degenerate denominator
        decision = _evaluate(quantities, _override(bootstrap, point=point))
        assert decision.status == prot.STATUS_INVALID
        assert decision.risk["baseline_valid"] is False
        assert decision.risk["conditions"]["r5_validity"] is False


def test_terminal_priority_qa_failure_is_invalid_before_any_verdict():
    """QA / condition-1 (completeness/hash/lock) failure -> INVALID, highest priority."""
    quantities = _make_quantities(G=240, seed=10)
    quantities["qa"]["hashes_match"] = False
    decision = _evaluate(quantities, _bootstrap(quantities))
    assert decision.status == prot.STATUS_INVALID

    # a group count below the frozen selected_n fails condition 1 -> INVALID.
    decision2 = _evaluate(_make_quantities(G=240, seed=10),
                          _bootstrap(_make_quantities(G=240, seed=10)),
                          selected_n=2400)
    assert decision2.status == prot.STATUS_INVALID


def test_terminal_priority_degenerate_beats_insufficient_and_invalid_denominator():
    """§7.3 literal order: degenerate (3/4/8) outranks insufficient and denominator."""
    quantities = _make_quantities(G=240, seed=11)
    bootstrap = _bootstrap(quantities)
    point = bootstrap.point.copy()
    point[d0.Q_HIGH_REGRET_FRAC] = 0.0     # degenerate (condition 3)
    point[d0.Q_AURC_BASELINE] = 0.0        # also a degenerate denominator
    ci_lower = bootstrap.ci_lower.copy()
    ci_lower[d0.Q_SPEARMAN_BRANCH] = -0.5  # also insufficient (condition 5)
    decision = _evaluate(quantities,
                         _override(bootstrap, point=point, ci_lower=ci_lower))
    assert decision.status == prot.STATUS_TASK_DEGENERATE


# --- quantities validation (fail-closed) -------------------------------------------------------


def test_validate_quantities_refuses_misaligned_or_nonfinite():
    good = _make_quantities(G=60, seed=12)
    assert d0._validate_d1_quantities(good) == 60
    bad = _make_quantities(G=60, seed=12)
    bad["regret"] = bad["regret"][:-1]  # misaligned
    with pytest.raises(ValueError):
        d0._validate_d1_quantities(bad)
    bad2 = _make_quantities(G=60, seed=12)
    bad2["ensemble_risk"][3] = np.nan
    with pytest.raises(ValueError):
        d0._validate_d1_quantities(bad2)
    bad3 = _make_quantities(G=60, seed=12)
    bad3["strongest_baseline"] = "MARGIN_ONLY"  # the baseline is never re-selected
    with pytest.raises(ValueError):
        d0._validate_d1_quantities(bad3)


# --- §11 items 9/10: one-shot lock refusal + run-all resume -----------------------------------


def _config(tmp_path) -> dict:
    config = json.loads(prot.CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run" / "stage4d_ood"
    config["run_dir"] = str(run_dir)
    config["data_root"] = str(tmp_path / "data" / "object_centric_stage"
                              / "stage4d_merge_ood")
    config["paths"] = dict(config["paths"])
    prefix = "experiments/stage4_reliability/stage4d_ood/"
    for key in list(config["paths"]):
        value = str(config["paths"][key])
        if value.startswith(prefix):
            config["paths"][key] = str(run_dir / value[len(prefix):])
    return config


def test_burn_lock_refuses_without_score_freeze(tmp_path):
    """The one-shot lock burns only after the full score freeze (fail-closed)."""
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="score freeze"):
        d0.burn_confirmation_lock_d1(config)


def test_adjudication_refuses_a_second_run(tmp_path):
    """§11 item 9: an existing formal report refuses any second adjudication."""
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    paths = d0.d1_confirm_artifact_paths(checked)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps({"status": prot.STATUS_CONFIRMED}),
                               encoding="utf-8")
    with pytest.raises(FileExistsError):
        d0.adjudicate_d1(config)


def test_run_all_has_no_d2_and_ends_at_reports_status_sync():
    """§11 item 10 + Global Constraint 9: the pipeline stops; D2/MetaDrive never start."""
    assert runner.PIPELINE[-1] == "reports-status-sync"
    for phase in runner.PIPELINE:
        low = phase.lower()
        assert "d2" not in low and "metadrive" not in low


def test_runner_adjudicate_resume_never_reburns(tmp_path, monkeypatch):
    """An existing report short-circuits: burn/assemble/adjudicate are never re-run."""
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    paths = d0.d1_confirm_artifact_paths(checked)
    # a consistent create-once fixture: quantities + receipt + burned lock + report.
    paths["quantities_npz"].parent.mkdir(parents=True, exist_ok=True)
    lock_payload = {"plan_id": prot.PLAN_ID}
    paths["lock"].write_text(json.dumps(lock_payload), encoding="utf-8")
    import numpy as _np
    _np.savez(paths["quantities_npz"], regret=_np.zeros(4))
    from ac_twin.stage1_data import sha256_file
    receipt = {"plan_id": prot.PLAN_ID,
               "confirm_group_quantities_sha256": sha256_file(paths["quantities_npz"]),
               "lock_sha256": sha256_file(paths["lock"])}
    paths["quantities_receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    report = {"plan_id": prot.PLAN_ID, "status": prot.STATUS_CONFIRMED,
              "bindings": {"confirm_group_quantities_sha256":
                           receipt["confirm_group_quantities_sha256"]}}
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(json.dumps(report), encoding="utf-8")

    calls = []
    monkeypatch.setattr(runner.d0, "burn_confirmation_lock_d1",
                        lambda cfg: calls.append("burn"))
    monkeypatch.setattr(runner.d0, "assemble_d1_quantities",
                        lambda cfg: calls.append("assemble"))
    monkeypatch.setattr(runner.d0, "adjudicate_d1",
                        lambda cfg: calls.append("adjudicate"))
    out = runner.run_phase("adjudicate", config)
    assert calls == []  # no create-once phase re-ran
    assert out["action"] == "verified_terminal"
    assert out["status"] == prot.STATUS_CONFIRMED
