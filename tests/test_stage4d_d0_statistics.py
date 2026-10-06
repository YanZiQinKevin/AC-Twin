"""Blocking tests for the Stage 4D D0 statistics module (plan §5.2/§5.3, §11 items 3/4).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 3. These tests pin:

* the six pre-intervention quantities (one shared computation function for the ID and
  OOD sides), including the ramp-y merging-vehicle identification with a fail-closed
  separation check;
* SMD math on constructed data with exact known values, and the >=2-quantity
  |SMD|>=0.50 rule;
* the fixed 5-fold logistic domain classifier with the group-bootstrap two-sided 80%
  AUC CI (5,000 resamples, per-profile entropies) — separable domains pass,
  identical domains fail;
* the four task-support conditions, each failing independently;
* ``evaluate_d0_profile``: ANY single failed condition makes the profile unqualified;
* ``select_ood_profile``: JOINT > BEHAVIOR > GEOMETRY priority and the
  ``STAGE4D_OOD_TASK_INSUFFICIENT`` hard stop;
* the create-once guard on ``ood_profile_selection.json`` (a second write is refused);
* the D0 forbidden zone: opening a Direct checkpoint / risk-score / AURC / coverage /
  improvement artifact path fails immediately, and the D0 statistics module never
  imports the Direct/risk machinery.
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

from ac_twin import stage4d_ood_statistics as d0  # noqa: E402
from ac_twin import stage4d_ood_env as ood  # noqa: E402
from ac_twin import stage4d_protocol as prot  # noqa: E402


# --- constructed history fixture ---------------------------------------------------------

T = prot.HISTORY_LEN
N = prot.OBJECT_CAPACITY
F = prot.N_OBJECT_FEATURES


def _frame(presence, is_ego, x, y, vx, vy):
    return np.array([presence, is_ego, x, y, vx, vy, 1.0, 0.0], dtype=np.float64)


def _constructed_history():
    """One group with known pre-intervention geometry (see test expectations below).

    ego (slot 0): (0, 0), v=(10, 0).
    slot 1: (12, 0), v=(7, 0)  — ahead, closing 3 m/s -> TTC 4.
    slot 2: (0, 4),  v=(10, 0) — lateral, distance 4, relspeed 0.
    slot 3: (-25, 0), v=(10, 0) — behind, distance 25 (< 30 -> density).
    slot 4 (merging): first frame y=14.5 (ramp), last frame (55, 10), v=(12, 0).
    """
    history = np.zeros((1, T, N, F), dtype=np.float64)
    for t in range(T):
        history[0, t, 0] = _frame(1, 1, 0.0, 0.0, 10.0, 0.0)
        history[0, t, 1] = _frame(1, 0, 12.0, 0.0, 7.0, 0.0)
        history[0, t, 2] = _frame(1, 0, 0.0, 4.0, 10.0, 0.0)
        history[0, t, 3] = _frame(1, 0, -25.0, 0.0, 10.0, 0.0)
        if t == 0:
            history[0, t, 4] = _frame(1, 0, 40.0, 14.5, 12.0, 0.0)
        else:
            history[0, t, 4] = _frame(1, 0, 55.0, 10.0, 12.0, 0.0)
    return history


# --- pre-intervention feature extraction ---------------------------------------------------


def test_pre_intervention_quantities_exist_and_order_frozen():
    assert tuple(d0.PRE_INTERVENTION_QUANTITIES) == (
        "min_ego_distance",
        "mean_abs_relative_speed",
        "merging_longitudinal_gap",
        "merging_relative_speed",
        "density_30m",
        "clip_ttc",
    )


def test_pre_intervention_features_exact_constructed():
    feats = d0.pre_intervention_features(_constructed_history())
    assert set(feats) == set(d0.PRE_INTERVENTION_QUANTITIES)
    assert np.isclose(feats["min_ego_distance"][0], 4.0)
    assert np.isclose(feats["mean_abs_relative_speed"][0], (3.0 + 0.0 + 0.0 + 2.0) / 4.0)
    assert np.isclose(feats["merging_longitudinal_gap"][0], 55.0)
    assert np.isclose(feats["merging_relative_speed"][0], 2.0)
    assert np.isclose(feats["density_30m"][0], 3.0)
    assert np.isclose(feats["clip_ttc"][0], 4.0)


def test_pre_intervention_features_clip_ttc_ceiling_when_no_closing():
    history = _constructed_history()
    history[0, :, 1] = _frame(1, 0, 12.0, 0.0, 30.0, 0.0)  # ahead but faster than ego
    feats = d0.pre_intervention_features(history)
    assert np.isclose(feats["clip_ttc"][0], d0.TTC_CLIP_MAX)


def test_merging_identification_requires_clear_ramp_separation():
    history = _constructed_history()
    # a second non-ego vehicle near the ramp y at the first frame -> ambiguous.
    history[0, 0, 3] = _frame(1, 0, -25.0, 13.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="merging"):
        d0.pre_intervention_features(history)


def test_pre_intervention_features_reject_bad_shape():
    with pytest.raises(ValueError):
        d0.pre_intervention_features(np.zeros((2, T, N, F + 1)))


# --- SMD math ------------------------------------------------------------------------------


def test_smd_exact_value_constructed():
    id_values = np.array([0.0, 2.0])
    ood_values = np.array([2.0, 4.0])
    # means 1 vs 3, ddof=1 variances 2 vs 2 -> smd = 2 / sqrt(2).
    smd = d0.standardized_mean_difference(id_values, ood_values)
    assert np.isclose(smd, np.sqrt(2.0))


def test_smd_zero_for_identical_distributions():
    values = np.linspace(-1.0, 1.0, 11)
    assert np.isclose(d0.standardized_mean_difference(values, values.copy()), 0.0)


def test_smd_sign_and_absolute_rule():
    id_values = np.array([0.0, 2.0])
    ood_values = np.array([2.0, 4.0])
    assert d0.standardized_mean_difference(id_values, ood_values) > 0
    assert d0.standardized_mean_difference(ood_values, id_values) < 0


def test_smd_shift_count_threshold():
    smds = {"a": 0.6, "b": -0.5, "c": 0.49, "d": 0.0, "e": 0.1, "f": 0.2}
    assert d0.count_shifted_quantities(smds) == 2
    assert d0.count_shifted_quantities({"a": 0.6, "b": 0.1}) == 1
    with pytest.raises(ValueError):
        d0.count_shifted_quantities({"a": float("nan")})


# --- domain classifier + bootstrap CI --------------------------------------------------------


def _separable_features(n_per_side=60, seed=7):
    rng = np.random.default_rng(seed)
    id_features = rng.normal(0.0, 1.0, size=(n_per_side, 6))
    ood_features = id_features.copy()
    ood_features[:, 0] += 10.0
    ood_features[:, 1] += 10.0
    return id_features, ood_features


def test_domain_classifier_full_spec_on_separable_data():
    id_features, ood_features = _separable_features()
    out = d0.compute_domain_shift(
        id_features, ood_features, bootstrap_seed=prot.DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES["behavior"])
    clf = out["domain_classifier"]
    assert clf["n_folds"] == 5
    assert clf["resamples"] == prot.DOMAIN_CLASSIFIER_BOOTSTRAP_RESAMPLES == 5000
    assert clf["confidence"] == prot.DOMAIN_CLASSIFIER_CONFIDENCE == 0.80
    assert clf["bootstrap_seed"] == 2026104101
    assert np.isclose(clf["auc_point"], 1.0)
    assert clf["auc_ci_lower"] >= prot.DOMAIN_CLASSIFIER_AUC_CI_LB_MIN
    assert clf["auc_ci_lower"] <= clf["auc_point"] <= clf["auc_ci_upper"]
    assert out["domain_auc_pass"] is True
    assert out["smd_pass"] is True
    assert out["shift_significant"] is True


def test_domain_classifier_identical_domains_fail():
    rng = np.random.default_rng(11)
    id_features = rng.normal(0.0, 1.0, size=(60, 6))
    ood_features = rng.normal(0.0, 1.0, size=(60, 6))
    out = d0.compute_domain_shift(
        id_features, ood_features, bootstrap_seed=prot.DOMAIN_CLASSIFIER_BOOTSTRAP_ENTROPIES["geometry"])
    assert out["domain_classifier"]["auc_ci_lower"] < prot.DOMAIN_CLASSIFIER_AUC_CI_LB_MIN
    assert out["domain_auc_pass"] is False
    assert out["smd_pass"] is False
    assert out["shift_significant"] is False


def test_domain_classifier_streams_deterministic_and_seed_bound():
    # overlapping (non-separable) domains so the bootstrap distribution is non-degenerate
    rng = np.random.default_rng(31)
    id_m = rng.normal(0.0, 1.0, size=(40, 6))
    ood_m = rng.normal(0.8, 1.0, size=(40, 6))
    out_b = d0._bootstrap_auc_ci(id_m, ood_m, 2026104101, resamples=50, confidence=0.80)
    out_b2 = d0._bootstrap_auc_ci(id_m, ood_m, 2026104101, resamples=50, confidence=0.80)
    out_g = d0._bootstrap_auc_ci(id_m, ood_m, 2026104102, resamples=50, confidence=0.80)
    assert out_b == out_b2  # deterministic per entropy
    assert out_b != out_g   # distinct per-profile entropies -> distinct streams


def test_compute_domain_shift_rejects_bad_inputs():
    with pytest.raises(ValueError):
        d0.compute_domain_shift(np.zeros((10, 5)), np.zeros((10, 6)), bootstrap_seed=1)
    with pytest.raises(ValueError):
        d0.compute_domain_shift(np.zeros((0, 6)), np.zeros((10, 6)), bootstrap_seed=1)


# --- task support (plan §5.2, four conditions) ----------------------------------------------


def _passing_task_arrays(n_groups=100, seed=3):
    rng = np.random.default_rng(seed)
    j_true = rng.uniform(0.1, 1.0, size=(n_groups, prot.N_ACTIONS))
    j_hat = j_true + 0.001 * rng.uniform(size=j_true.shape)
    flip = np.arange(n_groups) < max(1, int(round(0.20 * n_groups)))
    j_hat[flip] = -j_true[flip]  # those groups pick the worst action -> regret > 0.01
    return j_hat, j_true


def _task_inputs(j_hat, j_true):
    a_hat = np.argmin(j_hat, axis=-1)
    a_star = np.argmin(j_true, axis=-1)
    rows = np.arange(j_true.shape[0])
    regret = np.clip(j_true[rows, a_hat] - j_true[rows, a_star], 0.0, None)
    idle = np.clip(j_true[:, prot.IDLE_PLAN_INDEX] - j_true[rows, a_star], 0.0, None)
    random = np.clip(j_true.mean(axis=-1) - j_true[rows, a_star], 0.0, None)
    return {
        "regret": regret, "a_star": a_star, "J_hat": j_hat, "J_true": j_true,
        "idle_regret": idle, "random_regret": random,
    }


def test_task_support_all_conditions_pass():
    j_hat, j_true = _passing_task_arrays()
    out = d0.evaluate_task_support(**_task_inputs(j_hat, j_true))
    assert out["high_regret_support"] is True
    assert out["oracle_diversity"] is True
    assert out["spearman_positive"] is True
    assert out["base_action_beats_references"] is True
    assert out["task_supported"] is True
    assert out["min_high_regret_frac"] == 0.15
    assert out["regret_eps"] == 0.01
    assert out["min_distinct_oracle"] == 3
    assert out["min_oracle_frac"] == 0.05


def test_task_support_high_regret_fraction_fails():
    rng = np.random.default_rng(5)
    j_true = rng.uniform(0.1, 1.0, size=(100, prot.N_ACTIONS))
    j_hat = j_true + 0.001 * rng.uniform(size=j_true.shape)  # no flips -> regret ~ 0
    out = d0.evaluate_task_support(**_task_inputs(j_hat, j_true))
    assert out["high_regret_frac"] < 0.15
    assert out["high_regret_support"] is False
    assert out["task_supported"] is False


def test_task_support_oracle_diversity_fails():
    rng = np.random.default_rng(9)
    j_true = rng.uniform(0.1, 1.0, size=(100, prot.N_ACTIONS))
    j_true[:, 0] = 0.05  # plan 0 is always the oracle
    j_hat, _ = _passing_task_arrays()
    j_hat = j_true + 0.001 * rng.uniform(size=j_true.shape)
    j_hat[:20] = -j_true[:20]
    out = d0.evaluate_task_support(**_task_inputs(j_hat, j_true))
    assert out["n_oracle_plans_ge_min_frac"] < 3
    assert out["oracle_diversity"] is False
    assert out["task_supported"] is False


def test_task_support_spearman_fails_when_anti_correlated():
    rng = np.random.default_rng(13)
    j_true = rng.uniform(0.1, 1.0, size=(100, prot.N_ACTIONS))
    j_hat = -j_true  # perfectly anti-correlated at the branch level
    out = d0.evaluate_task_support(**_task_inputs(j_hat, j_true))
    assert out["spearman_jhat_jtrue"] < 0
    assert out["spearman_positive"] is False
    assert out["task_supported"] is False


def test_task_support_base_not_better_than_random_fails():
    rng = np.random.default_rng(17)
    j_true = rng.uniform(0.1, 1.0, size=(100, prot.N_ACTIONS))
    j_hat = -j_true  # base action = argmax(J_true): mean regret >= random's
    out = d0.evaluate_task_support(**_task_inputs(j_hat, j_true))
    assert out["base_beats_random"] is False
    assert out["base_action_beats_references"] is False
    assert out["task_supported"] is False


def test_task_support_base_above_idle_fails():
    # base regret equal to random (uniform actions) but idle is the oracle (idle regret 0).
    rng = np.random.default_rng(21)
    j_true = rng.uniform(0.1, 1.0, size=(100, prot.N_ACTIONS))
    j_true[:, prot.IDLE_PLAN_INDEX] = 0.0  # ALWAYS_IDLE is always optimal
    j_hat = rng.uniform(0.1, 1.0, size=(100, prot.N_ACTIONS))  # uncorrelated base choice
    out = d0.evaluate_task_support(**_task_inputs(j_hat, j_true))
    assert out["base_not_above_idle"] is False
    assert out["base_action_beats_references"] is False
    assert out["task_supported"] is False


def test_task_support_rejects_misaligned_shapes():
    j_hat, j_true = _passing_task_arrays()
    inputs = _task_inputs(j_hat, j_true)
    inputs["J_true"] = j_true[:, :5]
    with pytest.raises(ValueError):
        d0.evaluate_task_support(**inputs)


# --- evaluate_d0_profile: any single failure -> unqualified ---------------------------------


def _all_pass_panels():
    integrity = {
        "groups_complete": True,
        "entity_mask_action_integrity": True,
        "no_capacity_nan_early_termination": True,
        "candidate_pool_not_exhausted": True,
    }
    shift = {"smd_pass": True, "domain_auc_pass": True, "shift_significant": True}
    task = {
        "high_regret_support": True,
        "oracle_diversity": True,
        "spearman_positive": True,
        "base_action_beats_references": True,
        "task_supported": True,
    }
    return integrity, shift, task


def test_evaluate_d0_profile_all_pass_qualified():
    integrity, shift, task = _all_pass_panels()
    out = d0.evaluate_d0_profile(integrity, shift, task)
    assert out["qualified"] is True
    assert out["failed_conditions"] == []


@pytest.mark.parametrize("panel,key", [
    ("integrity", "groups_complete"),
    ("integrity", "entity_mask_action_integrity"),
    ("integrity", "no_capacity_nan_early_termination"),
    ("integrity", "candidate_pool_not_exhausted"),
    ("shift", "smd_pass"),
    ("shift", "domain_auc_pass"),
    ("task", "high_regret_support"),
    ("task", "oracle_diversity"),
    ("task", "spearman_positive"),
    ("task", "base_action_beats_references"),
])
def test_evaluate_d0_profile_any_single_failure_unqualified(panel, key):
    integrity, shift, task = _all_pass_panels()
    panels = {"integrity": integrity, "shift": shift, "task": task}
    panels[panel][key] = False
    if panel == "shift":
        panels["shift"]["shift_significant"] = False
    if panel == "task":
        panels["task"]["task_supported"] = False
    out = d0.evaluate_d0_profile(panels["integrity"], panels["shift"], panels["task"])
    assert out["qualified"] is False
    assert key in out["failed_conditions"]


def test_evaluate_d0_profile_rejects_missing_condition():
    integrity, shift, task = _all_pass_panels()
    del integrity["groups_complete"]
    with pytest.raises(ValueError):
        d0.evaluate_d0_profile(integrity, shift, task)


# --- select_ood_profile: priority + hard stop + create-once ---------------------------------


def _report(qualified):
    return {"qualified": bool(qualified)}


def test_select_ood_profile_priority_joint_first():
    reports = {"BEHAVIOR_OOD": _report(True), "GEOMETRY_OOD": _report(True),
               "JOINT_OOD": _report(True)}
    out = d0.select_ood_profile(reports)
    assert out["selected_profile"] == "JOINT_OOD"
    assert out["status"] == prot.STATUS_D0_PROFILE_SELECTED
    assert list(out["priority"]) == ["JOINT_OOD", "BEHAVIOR_OOD", "GEOMETRY_OOD"]


def test_select_ood_profile_falls_to_behavior_then_geometry():
    reports = {"BEHAVIOR_OOD": _report(True), "GEOMETRY_OOD": _report(True),
               "JOINT_OOD": _report(False)}
    assert d0.select_ood_profile(reports)["selected_profile"] == "BEHAVIOR_OOD"
    reports = {"BEHAVIOR_OOD": _report(False), "GEOMETRY_OOD": _report(True),
               "JOINT_OOD": _report(False)}
    assert d0.select_ood_profile(reports)["selected_profile"] == "GEOMETRY_OOD"


def test_select_ood_profile_none_qualified_hard_stop():
    reports = {"BEHAVIOR_OOD": _report(False), "GEOMETRY_OOD": _report(False),
               "JOINT_OOD": _report(False)}
    out = d0.select_ood_profile(reports)
    assert out["selected_profile"] is None
    assert out["status"] == prot.STATUS_TASK_INSUFFICIENT


def test_select_ood_profile_requires_all_three_reports():
    with pytest.raises(ValueError):
        d0.select_ood_profile({"JOINT_OOD": _report(True)})


def test_write_profile_selection_create_once(tmp_path):
    reports = {"BEHAVIOR_OOD": _report(True), "GEOMETRY_OOD": _report(True),
               "JOINT_OOD": _report(True)}
    selection = d0.select_ood_profile(reports)
    path = tmp_path / "ood_profile_selection.json"
    d0.write_profile_selection(path, selection)
    assert json.loads(path.read_text(encoding="utf-8"))["selected_profile"] == "JOINT_OOD"
    with pytest.raises(FileExistsError):
        d0.write_profile_selection(path, selection)


# --- D0 forbidden zone (plan §5.1, Global Constraint 3) --------------------------------------


def test_d0_forbidden_read_direct_checkpoint_fails(tmp_path):
    ckpt = tmp_path / "models" / "direct_seed8101.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake checkpoint bytes")
    with pytest.raises(d0.D0ForbiddenReadError):
        d0.assert_d0_read_legal(ckpt)


def test_d0_forbidden_read_risk_score_files_fail(tmp_path):
    for name in ("risk_scores.npz", "stage4c_risk_train.npz", "aurc_table.csv",
                 "risk_coverage.png", "direct_vs_baseline_improvement.json"):
        path = tmp_path / name
        path.write_bytes(b"x")
        with pytest.raises(d0.D0ForbiddenReadError):
            d0.assert_d0_read_legal(path)


def test_d0_forbidden_read_allows_legal_pilot_files(tmp_path):
    for name in ("inference_inputs.npz", "confirmation_truth.npz",
                 "collection_receipt.json"):
        path = tmp_path / "pilot" / "behavior" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        d0.assert_d0_read_legal(path)  # must not raise


def test_d0_statistics_module_never_imports_direct_or_risk_machinery():
    source = (CODE / "ac_twin" / "stage4d_ood_statistics.py").read_text(encoding="utf-8")
    for forbidden in (
        "torch.load",                  # any checkpoint load
        "load_frozen_model_inventory", # Direct checkpoint inventory loader
        "load_frozen_models",          # risk-model loader
        "stage4cr2_training",          # Direct scoring module
        "score_models",                # Direct scoring entry
    ):
        assert forbidden not in source, forbidden


# --- pilot integrity evaluation ---------------------------------------------------------------


def _integrity_arrays(n_groups=300, seed=23):
    rng = np.random.default_rng(seed)
    history = np.zeros((n_groups, T, N, F), dtype=np.float32)
    history[..., 0] = 1.0  # presence
    history[..., 2:8] = rng.normal(size=(n_groups, T, N, 6)).astype(np.float32)
    factual = np.zeros((n_groups, prot.HORIZON, N, F), dtype=np.float32)
    factual[..., 0] = 1.0
    factual[..., 2:8] = rng.normal(size=(n_groups, prot.HORIZON, N, 6)).astype(np.float32)
    truth_futures = np.zeros(
        (n_groups, prot.N_ACTIONS, prot.HORIZON, N, F), dtype=np.float32)
    truth_futures[..., 0] = 1.0
    truth_futures[..., 2:8] = rng.normal(
        size=(n_groups, prot.N_ACTIONS, prot.HORIZON, N, 6)).astype(np.float32)
    factual_actions = rng.integers(0, 5, size=(n_groups, prot.HORIZON))
    mask = prot.build_inference_mask(history, factual)
    legal = {
        "object_history": history,
        "object_factual_future": factual,
        "factual_actions": factual_actions,
        "cf_actions": np.tile([a for a, _ in prot.ACTION_PLANS], (n_groups, 1)),
        "cf_durations": np.tile([d for _, d in prot.ACTION_PLANS], (n_groups, 1)),
        "t_star": rng.integers(prot.T_STAR_LOW, prot.T_STAR_HIGH + 1, size=n_groups),
        "group_seed": np.arange(n_groups, dtype=np.int64) + 1000,
        "inference_mask": mask,
    }
    truth = {
        "object_counterfactual_futures": truth_futures,
        "cf_actions": legal["cf_actions"].copy(),
        "cf_durations": legal["cf_durations"].copy(),
        "factual_actions": factual_actions.copy(),
        "t_star": legal["t_star"].copy(),
        "group_seed": legal["group_seed"].copy(),
        "inference_mask": mask.copy(),
    }
    receipt = {
        "accepted_groups": n_groups,
        "counterfactual_branches": n_groups * prot.N_ACTIONS,
        "attempts": 400,
        "status": "COLLECTED",
        "partial": False,
        "prefix_max_abs_diff": 0.0,
        "group_order_binding": {"same_order": True, "n_groups": n_groups},
    }
    return receipt, legal, truth


def test_pilot_integrity_all_pass():
    receipt, legal, truth = _integrity_arrays()
    out = d0.evaluate_pilot_integrity(receipt, legal, truth)
    assert out["groups_complete"] is True
    assert out["entity_mask_action_integrity"] is True
    assert out["no_capacity_nan_early_termination"] is True
    assert out["candidate_pool_not_exhausted"] is True
    assert out["integrity_pass"] is True


def test_pilot_integrity_incomplete_groups_fail():
    receipt, legal, truth = _integrity_arrays()
    receipt["accepted_groups"] = 299
    out = d0.evaluate_pilot_integrity(receipt, legal, truth)
    assert out["groups_complete"] is False
    assert out["integrity_pass"] is False


def test_pilot_integrity_mask_mismatch_fails():
    receipt, legal, truth = _integrity_arrays()
    truth["inference_mask"] = ~truth["inference_mask"]
    out = d0.evaluate_pilot_integrity(receipt, legal, truth)
    assert out["entity_mask_action_integrity"] is False
    assert out["integrity_pass"] is False


def test_pilot_integrity_non_finite_fails():
    receipt, legal, truth = _integrity_arrays()
    legal["object_factual_future"][0, 0, 0, 2] = np.nan
    out = d0.evaluate_pilot_integrity(receipt, legal, truth)
    assert out["no_capacity_nan_early_termination"] is False
    assert out["integrity_pass"] is False


def test_pilot_integrity_exhausted_pool_fails():
    receipt, legal, truth = _integrity_arrays()
    receipt["attempts"] = prot.PILOT_MAX_CANDIDATE_SEEDS + 1
    out = d0.evaluate_pilot_integrity(receipt, legal, truth)
    assert out["candidate_pool_not_exhausted"] is False
    assert out["integrity_pass"] is False
