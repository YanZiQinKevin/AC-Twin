"""Stage 4C-R2 validation-gate / baseline-selection / freeze tests (plan sections 4.4, 8).

Blocking invariants proved here (plan section 14 / task-3 brief):
- STRONGEST_STANDARD_BASELINE is the unique lowest validation AURC_REGRET among
  MARGIN_ONLY / SEED_VOTE_ENTROPY / SEED_MARGIN_STD; ties within 1e-12 break in that
  listed order; COMBINED_TOP2_MAX (historical) and CONSTANT_RANDOM (curve reference) are
  never eligible; the screen may never re-select;
- the validation gate's EIGHT conditions are each independently enforced (plan 8.1);
- interaction strata below 20% in validation terminate as STAGE4CR2_STRATUM_IMBALANCED;
- the freeze writes the five create-once artifacts and refuses a second write.
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

from ac_twin import stage4cp_models as s4cm  # noqa: E402
from ac_twin import stage4cp_statistics as s4cs  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_selection as s4cr2s  # noqa: E402
from ac_twin import stage4cr2_training as s4cr2t  # noqa: E402

CANDIDATES = s4cr2.STRONGEST_BASELINE_CANDIDATES


def _risk_scores(ensemble, by_seed, group_seed):
    return s4cr2t.RiskScores(
        group_seed=np.asarray(group_seed, dtype=np.int64),
        risk_ensemble=np.asarray(ensemble, dtype=np.float64),
        risk_by_seed=np.asarray(by_seed, dtype=np.float64),
        mu_only_ensemble=np.asarray(ensemble, dtype=np.float64),
        class_only_ensemble=np.asarray(ensemble, dtype=np.float64),
        seeds=tuple(int(s) for s in s4cr2.MODEL_SEEDS),
    )


def _passing_fixture(groups=400, seed=7):
    """scores + truth that pass all eight gate conditions with balanced strata.

    Ensemble is near-perfect; the three eligible baselines are pure noise (the strongest
    is whichever of them happens to rank best); COMBINED_TOP2_MAX is PERFECT and must
    never be selected; MARGIN_ONLY stays weakly correlated so the Spearman ceiling holds.
    """
    rng = np.random.default_rng(seed)
    regret = rng.gamma(shape=1.5, scale=0.01, size=groups)
    y_high = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)
    ens = regret + rng.normal(0.0, 1e-3, size=groups)
    by_seed = ens[None, :] + rng.normal(0.0, 2e-3, size=(5, groups))
    ens = by_seed.mean(axis=0)
    baseline_risks = {
        "MARGIN_ONLY": regret + rng.normal(0.0, 0.05, size=groups),
        "SEED_VOTE_ENTROPY": rng.random(groups),
        "SEED_MARGIN_STD": rng.random(groups),
        "COMBINED_TOP2_MAX": regret.copy(),  # perfect: historical reference ONLY
    }
    validation = {
        "regret": regret,
        "y_high": y_high,
        "baseline_risks": baseline_risks,
        "group_seed": np.arange(groups, dtype=np.int64),
    }
    receipt = s4cr2s.select_strongest_standard_baseline(validation)
    assert receipt.strongest != "COMBINED_TOP2_MAX"
    distance = np.where(np.arange(groups) % 2 == 0, 10.0, 40.0)  # 50/50 strata
    truth = {
        **validation,
        "baseline_receipt": receipt,
        "interaction_min_distance": distance,
        "interaction_threshold": 25.0,
        "bindings": {"train_data_sha256": "aa", "validation_data_sha256": "bb"},
    }
    scores = _risk_scores(ens, by_seed, np.arange(groups, dtype=np.int64))
    return scores, truth


# --- strongest-baseline selection --------------------------------------------------------------


def test_strongest_baseline_is_unique_lowest_validation_aurc():
    rng = np.random.default_rng(11)
    groups = 500
    regret = rng.gamma(shape=1.5, scale=0.01, size=groups)
    risks = {
        "MARGIN_ONLY": rng.random(groups),              # worst
        "SEED_VOTE_ENTROPY": regret + rng.normal(0, 0.02, size=groups),  # best
        "SEED_MARGIN_STD": regret + rng.normal(0, 0.05, size=groups),   # middle
        "COMBINED_TOP2_MAX": regret.copy(),             # perfect but INELIGIBLE
    }
    receipt = s4cr2s.select_strongest_standard_baseline({
        "regret": regret, "y_high": (regret > 0.01).astype(np.int64),
        "baseline_risks": risks, "group_seed": np.arange(groups)})
    assert receipt.strongest == "SEED_VOTE_ENTROPY"
    aurcs = {name: s4cs.aurc_regret(s4cs.order_by_risk(risks[name]), regret)
             for name in CANDIDATES}
    assert receipt.aurc_regret[receipt.strongest] == pytest.approx(
        min(aurcs.values()))
    # the receipt records every candidate's three metrics for offline audit.
    for name in CANDIDATES:
        assert name in receipt.aurc_regret
        assert name in receipt.regret_at_80
        assert name in receipt.pr_auc_minus_prevalence
    assert receipt.tie_break_order == tuple(CANDIDATES)


def test_baseline_tie_break_within_1e_12_in_listed_order():
    rng = np.random.default_rng(13)
    groups = 300
    base = rng.gamma(shape=1.5, scale=0.01, size=groups) * 1e-12  # tiny regrets
    order = np.argsort(base, kind="stable")
    perfect = base.copy()
    swapped = base.copy()
    i, j = order[5], order[6]  # adjacent-in-rank swap: AURC worsens by ~1e-15
    swapped[i], swapped[j] = swapped[j], swapped[i]
    risks = {"MARGIN_ONLY": swapped, "SEED_VOTE_ENTROPY": perfect,
             "SEED_MARGIN_STD": rng.random(groups)}
    aurc_margin = s4cs.aurc_regret(s4cs.order_by_risk(swapped), base)
    aurc_vote = s4cs.aurc_regret(s4cs.order_by_risk(perfect), base)
    assert 0.0 < aurc_margin - aurc_vote < s4cr2.BASELINE_TIE_TOLERANCE
    receipt = s4cr2s.select_strongest_standard_baseline({
        "regret": base, "y_high": (base > 0.01).astype(np.int64),
        "baseline_risks": risks, "group_seed": np.arange(groups)})
    assert receipt.strongest == "MARGIN_ONLY"  # tie -> earlier in the frozen order

    big = base * 1e8  # same rankings, difference now >> 1e-12: the better one wins
    receipt_big = s4cr2s.select_strongest_standard_baseline({
        "regret": big, "y_high": (big > 0.01).astype(np.int64),
        "baseline_risks": risks, "group_seed": np.arange(groups)})
    assert receipt_big.strongest == "SEED_VOTE_ENTROPY"


def test_identical_baselines_break_to_first_listed():
    rng = np.random.default_rng(17)
    groups = 200
    regret = rng.gamma(shape=1.5, scale=0.01, size=groups)
    same = regret + rng.normal(0, 0.01, size=groups)
    risks = {"MARGIN_ONLY": same, "SEED_VOTE_ENTROPY": same.copy(),
             "SEED_MARGIN_STD": same.copy()}
    receipt = s4cr2s.select_strongest_standard_baseline({
        "regret": regret, "y_high": (regret > 0.01).astype(np.int64),
        "baseline_risks": risks, "group_seed": np.arange(groups)})
    assert receipt.strongest == "MARGIN_ONLY"


# --- validation gate: pass fixture ----------------------------------------------------------------


def test_validation_gate_passes_on_qualified_fixture():
    scores, truth = _passing_fixture()
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert len(report.conditions) == 8
    assert all(report.conditions.values()), report.conditions
    assert report.passed is True
    assert report.status == "VALIDATION_QUALIFIED"
    assert report.strata_balanced is True


# --- validation gate: each of the eight conditions independently enforced -------------------------


def test_gate_c1_i_aurc_regret_enforced():
    scores, truth = _passing_fixture()
    base = truth["baseline_risks"][truth["baseline_receipt"].strongest]
    by_seed = np.tile(base, (5, 1))  # ensemble == baseline -> I_AURC = 0 < 0.05
    scores = _risk_scores(by_seed.mean(axis=0), by_seed, scores.group_seed)
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.conditions["c1_i_aurc_regret"] is False
    assert report.passed is False
    assert report.status == s4cr2.STATUS_MODEL_DEV_INSUFFICIENT


def test_gate_c2_i_regret80_enforced():
    # Ensemble fixes ONLY the top-20% tail: I_AURC > 0 but I_REGRET@80 = 0 < 0.03.
    groups = 400
    low = np.random.default_rng(19).uniform(0.0, 0.005, size=320)
    high = np.random.default_rng(23).uniform(0.05, 0.30, size=80)
    regret = np.concatenate([np.sort(low), np.sort(high)])  # ascending
    y_high = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)
    ens = regret + np.linspace(0, 1e-6, groups)  # near-perfect ascending risk
    base_risk = regret.copy()
    base_risk[320:] = regret[320:][::-1]  # reversed tail, same lowest-80% set
    by_seed = np.tile(ens, (5, 1))
    risks = {"MARGIN_ONLY": np.random.default_rng(29).random(groups),
             "SEED_VOTE_ENTROPY": np.random.default_rng(31).random(groups),
             "SEED_MARGIN_STD": base_risk}
    validation = {"regret": regret, "y_high": y_high, "baseline_risks": risks,
                  "group_seed": np.arange(groups, dtype=np.int64)}
    receipt = s4cr2s.select_strongest_standard_baseline(validation)
    assert receipt.strongest == "SEED_MARGIN_STD"  # the crafted imperfect baseline
    truth = {**validation, "baseline_receipt": receipt,
             "interaction_min_distance": np.where(np.arange(groups) % 2, 10.0, 40.0),
             "interaction_threshold": 25.0}
    scores = _risk_scores(by_seed.mean(axis=0), by_seed, np.arange(groups, dtype=np.int64))
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.metrics["i_regret80"] == pytest.approx(0.0, abs=1e-9)
    assert report.conditions["c2_i_regret80"] is False
    assert report.conditions["c1_i_aurc_regret"] is True  # isolation: c1 still passes
    assert report.passed is False


def test_gate_c3_pr_auc_enforced():
    scores, truth = _passing_fixture()
    perfect = truth["regret"].copy()
    truth["baseline_risks"] = dict(truth["baseline_risks"])
    truth["baseline_risks"]["SEED_MARGIN_STD"] = perfect  # perfect strongest baseline
    for name in ("MARGIN_ONLY", "SEED_VOTE_ENTROPY"):
        truth["baseline_risks"][name] = np.random.default_rng(37).random(perfect.size)
    receipt = s4cr2s.select_strongest_standard_baseline(
        {k: truth[k] for k in ("regret", "y_high", "baseline_risks", "group_seed")})
    truth["baseline_receipt"] = receipt
    assert receipt.strongest == "SEED_MARGIN_STD"
    # ensemble PR-AUC-prevalence > 0 but strictly below the perfect baseline.
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.metrics["pr_auc_minus_prevalence_ensemble"] > 0.0
    assert report.conditions["c3_pr_auc"] is False
    assert report.passed is False


def test_gate_c4_seed_i_aurc_enforced():
    scores, truth = _passing_fixture()
    rng = np.random.default_rng(41)
    good = truth["regret"] + rng.normal(0, 1e-3, size=truth["regret"].size)
    bad = -truth["regret"] + rng.normal(0, 1e-3, size=truth["regret"].size)
    by_seed = np.stack([good, good, good, bad, bad])  # only 3/5 positive
    scores = _risk_scores(by_seed.mean(axis=0), by_seed, scores.group_seed)
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.metrics["seeds_i_aurc_positive"] == 3
    assert report.conditions["c4_seeds_i_aurc"] is False
    assert report.passed is False


def test_gate_c5_seed_i_regret80_enforced():
    scores, truth = _passing_fixture()
    rng = np.random.default_rng(43)
    good = truth["regret"] + rng.normal(0, 1e-3, size=truth["regret"].size)
    bad = -truth["regret"] + rng.normal(0, 1e-3, size=truth["regret"].size)
    by_seed = np.stack([good, good, bad, bad, good])  # only 3/5 positive
    scores = _risk_scores(by_seed.mean(axis=0), by_seed, scores.group_seed)
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.metrics["seeds_i_regret80_positive"] == 3
    assert report.conditions["c5_seeds_i_regret80"] is False
    assert report.passed is False


def test_gate_c6_trims_enforced():
    scores, truth = _passing_fixture()
    regret = truth["regret"]
    top = np.argsort(regret, kind="stable")[-4:]  # the top-1% highest-regret groups
    regret2 = regret.copy()
    regret2[top] = 5.0  # make the trimmed tail dominate the untrimmed improvement
    truth = dict(truth)
    truth["regret"] = regret2
    truth["y_high"] = (regret2 > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.int64)
    ens = regret2 + np.random.default_rng(47).normal(0, 1e-3, size=regret2.size)
    base_risk = ens.copy()
    base_risk[top] = ens.min() - 1.0  # baseline hides the top-1% at the safest ranks
    # the other two baselines are anti-perfect (strictly worse AURC than hiding four
    # groups), so the crafted SEED_MARGIN_STD is mechanically the strongest.
    truth["baseline_risks"] = {"MARGIN_ONLY": -regret2,
                               "SEED_VOTE_ENTROPY": -regret2 + 1e-3,
                               "SEED_MARGIN_STD": base_risk}
    receipt = s4cr2s.select_strongest_standard_baseline(
        {k: truth[k] for k in ("regret", "y_high", "baseline_risks", "group_seed")})
    truth["baseline_receipt"] = receipt
    by_seed = np.tile(ens, (5, 1))
    scores = _risk_scores(by_seed.mean(axis=0), by_seed, scores.group_seed)
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.conditions["c1_i_aurc_regret"] is True  # untrimmed improvement is large
    assert report.conditions["c6_trims_positive"] is False  # gone after both trims
    assert report.passed is False


def test_gate_c7_margin_spearman_enforced():
    scores, truth = _passing_fixture()
    truth = dict(truth)
    truth["baseline_risks"] = dict(truth["baseline_risks"])
    truth["baseline_risks"]["MARGIN_ONLY"] = scores.risk_ensemble.copy()  # identical
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert abs(report.metrics["spearman_ensemble_margin"]) == pytest.approx(1.0)
    assert report.conditions["c7_margin_spearman"] is False
    assert report.passed is False


def test_gate_c8_finite_and_matching_enforced():
    scores, truth = _passing_fixture()
    bad = scores.risk_ensemble.copy()
    bad[0] = np.nan
    report = s4cr2s.evaluate_validation_gate(
        _risk_scores(bad, scores.risk_by_seed, scores.group_seed), truth)
    assert report.conditions["c8_finite_matching"] is False
    assert report.passed is False

    shifted = scores.risk_by_seed.mean(axis=0) + 1e-9  # ensemble != arithmetic mean
    report = s4cr2s.evaluate_validation_gate(
        _risk_scores(shifted, scores.risk_by_seed, scores.group_seed), truth)
    assert report.conditions["c8_finite_matching"] is False
    assert report.passed is False

    wrong_seeds = scores.group_seed + 1  # role/group mismatch
    report = s4cr2s.evaluate_validation_gate(
        _risk_scores(scores.risk_ensemble, scores.risk_by_seed, wrong_seeds), truth)
    assert report.conditions["c8_finite_matching"] is False
    assert report.passed is False


def test_gate_stratum_imbalance_is_its_own_terminal_state():
    scores, truth = _passing_fixture()
    truth = dict(truth)
    truth["interaction_min_distance"] = np.full(scores.group_seed.size, 5.0)  # one stratum
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.strata_balanced is False
    assert report.passed is False
    assert report.status == s4cr2.STATUS_STRATUM_IMBALANCED


# --- create-once freeze ---------------------------------------------------------------------------


def _inventory(tmp_path):
    config = json.loads(
        (ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json")
        .read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "run")
    config["data_root"] = str(
        tmp_path / "data" / "object_centric_stage" / s4cr2.PURPOSE_DEV)
    config["device"] = "cpu"
    groups = 64
    rng = np.random.default_rng(5)
    bundle = {
        "tokens": rng.normal(size=(groups, s4cr2.N_COMPETITORS, s4cr2.TOKEN_DIM)),
        "global_features": rng.normal(size=(groups, s4cr2.GLOBAL_DIM)),
        "y_reg": rng.random(groups),
        "y_high": (rng.random(groups) > 0.5).astype(np.float64),
        "group_seed": np.arange(groups, dtype=np.int64),
        "interaction_min_distance": rng.uniform(5.0, 60.0, size=groups),
    }

    def fake_fit(seed, main, monitor, *, pos_weight, config, device):
        model = s4cm.build_model(
            s4cr2.ARCH_DIRECT, hidden_dim=s4cr2.DIRECT_HIDDEN_WIDTH,
            dropout=s4cr2.DROPOUT, direct_width=s4cr2.DIRECT_HIDDEN_WIDTH)
        return {
            "arch": s4cr2.ARCH_DIRECT, "hidden_dim": s4cr2.DIRECT_HIDDEN_WIDTH,
            "direct_width": s4cr2.DIRECT_HIDDEN_WIDTH, "lambda_cls": s4cr2.LAMBDA_CLS,
            "seed": int(seed), "pos_weight": float(pos_weight),
            "dropout": s4cr2.DROPOUT, "token_dim": s4cr2.TOKEN_DIM,
            "global_dim": s4cr2.GLOBAL_DIM, "n_competitors": s4cr2.N_COMPETITORS,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "train_loss": 0.5, "best_internal_monitor_loss": 0.6,
            "n_train_groups": int(main["x_tok"].shape[0]),
        }

    return s4cr2t.fit_five_direct_models(bundle, config, fit_one_fn=fake_fit,
                                         device="cpu")


def test_freeze_final_selection_writes_five_artifacts_create_once(tmp_path):
    inventory = _inventory(tmp_path)
    scores, truth = _passing_fixture()
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert report.passed
    path = s4cr2s.freeze_final_selection(report, inventory)
    dev = Path(inventory.run_dir) / "development"
    for name in ("feature_schema.json", "feature_scaler.npz", "model_inventory.json",
                 "validation_gate_report.json", "final_selection.json"):
        assert (dev / name).is_file(), name
    assert path == dev / "final_selection.json"

    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["param_count"] == 15740
    assert final["architecture"] == s4cr2.ARCH_DIRECT
    assert final["strongest_standard_baseline"] == report.baseline.strongest
    assert final["interaction_threshold"] == pytest.approx(inventory.interaction_threshold)
    assert [int(s) for s in final["model_seeds"]] == list(s4cr2.MODEL_SEEDS)
    assert len(final["checkpoints"]) == 5
    for seed, entry in final["checkpoints"].items():
        assert entry["sha256"] == inventory.checkpoint_sha256[int(seed)]
    assert final["train_data_sha256"] == "aa"
    assert final["validation_data_sha256"] == "bb"
    assert "risk_score" in final["primary_score_definition"]
    # the choice is offline-auditable: all three candidates' metrics + tie-break order.
    for name in CANDIDATES:
        assert name in final["baseline_validation_metrics"]
        metrics = final["baseline_validation_metrics"][name]
        assert {"aurc_regret", "regret_at_80", "pr_auc_minus_prevalence"} <= set(metrics)
    assert final["baseline_tie_break_order"] == list(CANDIDATES)
    assert final["baseline_tie_tolerance"] == s4cr2.BASELINE_TIE_TOLERANCE

    with pytest.raises(FileExistsError):
        s4cr2s.freeze_final_selection(report, inventory)


def test_freeze_refuses_failed_gate(tmp_path):
    inventory = _inventory(tmp_path)
    scores, truth = _passing_fixture()
    truth = dict(truth)
    truth["baseline_risks"] = dict(truth["baseline_risks"])
    truth["baseline_risks"]["MARGIN_ONLY"] = scores.risk_ensemble.copy()  # breaks c7
    report = s4cr2s.evaluate_validation_gate(scores, truth)
    assert not report.passed
    with pytest.raises(ValueError):
        s4cr2s.freeze_final_selection(report, inventory)
