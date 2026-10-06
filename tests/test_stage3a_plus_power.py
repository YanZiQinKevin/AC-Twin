"""Stage 3A Plus Task 8: full-Gate joint power estimation + qualification
artifact freeze (three-way hash binding).

The estimator is pure numpy over a precomputed per-episode statistic bundle;
test fixtures supply cheap synthetic bundles. Frozen values (candidates, MC
replicates, inner bootstrap, target power, power seed) are validated exactly —
any deviation is rejected (plan Task 8 step 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import prepare_stage3a_plus_power as pp  # noqa: E402
from ac_twin import stage3_data as s3d  # noqa: E402

FROZEN = dict(
    candidates=list(s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS),
    monte_carlo_replicates=s3d.STAGE3A_PLUS_MONTE_CARLO_REPLICATES,
    inner_bootstrap_resamples=s3d.STAGE3A_PLUS_INNER_BOOTSTRAP,
    target_power=s3d.STAGE3A_PLUS_TARGET_POWER,
    seed=s3d.STAGE3A_PLUS_POWER_SEED,
)
S = 10  # ten fixed seeds


def _synthetic_set_bundle(E=60, *, mean_i_brier=None, strong=True, seed=0,
                          ece=(0.04, 0.06), weak_seed_rank=None):
    """A cheap but structurally complete per-set statistic bundle.

    ``mean_i_brier`` gives the (S,) per-seed mean per-episode I_Brier; other
    effects are comfortably positive when ``strong``. ``weak_seed_rank=(m, sd)``
    replaces seed S-1's per-seed Spearman pseudo-values with N(m, sd) so the
    per-seed core-point condition binds at small n but concentrates at large n.
    """
    rng = np.random.default_rng(seed)
    if mean_i_brier is None:
        mean_i_brier = np.full(S, 0.02 if strong else -0.005)
    scale = 0.01 if strong else 0.02
    i_brier_ep_seeds = np.stack([
        rng.normal(m, scale, size=E) for m in mean_i_brier], axis=0)
    # Per-episode sufficient sums consistent with the per-episode values:
    # single-episode I_Brier = (base_ss - model_ss) / (4 * n_ep).
    n_ep = np.full(E, 100.0)
    base_ss_ep = np.full(E, 100.0)
    model_ss_ep = base_ss_ep - 4 * n_ep * i_brier_ep_seeds.mean(axis=0)
    brier_seeds_model_ss = np.stack(
        [base_ss_ep - 4 * n_ep * i_brier_ep_seeds[s] for s in range(S)], axis=0)
    i_mae_ep = rng.normal(0.05 if strong else -0.01, 0.02, size=E)
    i_mae_ep_seeds = np.stack(
        [rng.normal(0.05 if strong else -0.01, 0.02, size=E) for _ in range(S)], axis=0)

    rank_pseudo = rng.normal(0.10 if strong else -0.02, 0.05, size=E)

    def _rank_fn(pseudo):
        return lambda idx: float(np.mean(pseudo[np.asarray(idx)]))

    rank_fns = {name: _rank_fn(rank_pseudo + off)
                for name, off in (("spearman", 0.0), ("i_pr_presence", 0.01),
                                  ("i_pr_high_vs_lowmid", 0.005))}
    rank_fns_seeds = {
        name: [_rank_fn(rank_pseudo + off + 0.002 * s) for s in range(S)]
        for name, off in (("spearman", 0.0), ("i_pr_presence", 0.01),
                          ("i_pr_high_vs_lowmid", 0.005))}
    if weak_seed_rank is not None:
        m, sd = weak_seed_rank
        rank_fns_seeds["spearman"][S - 1] = _rank_fn(rng.normal(m, sd, size=E))
    return {
        "n_episodes": E,
        "i_mae_ep": i_mae_ep,
        "i_mae_ep_seeds": i_mae_ep_seeds,
        "brier": {"model_ss_ep": model_ss_ep, "base_ss_ep": base_ss_ep, "n_ep": n_ep},
        "brier_seeds": {"model_ss_ep": brier_seeds_model_ss, "n_ep": np.broadcast_to(
            n_ep, (S, E)).copy()},
        "i_brier_ep_seeds": i_brier_ep_seeds,
        "rank_fns": rank_fns,
        "rank_fns_seeds": rank_fns_seeds,
        "ece_fn": lambda idx: ece,
        "r_ensemble_std_fn": lambda idx: 0.5,
    }


def _bundle(**kw_a):
    return {"A": _synthetic_set_bundle(seed=0, **kw_a),
            "B": _synthetic_set_bundle(seed=1, **kw_a)}


# --------------------------------------------------------------------------- #
# Step 1: frozen values enforced
# --------------------------------------------------------------------------- #
def test_estimate_rejects_any_deviation_from_frozen_values():
    bundle = _bundle()
    for key, bad in (("candidates", [400, 600]),
                     ("monte_carlo_replicates", 100),
                     ("inner_bootstrap_resamples", 100),
                     ("target_power", 0.8),
                     ("seed", 123)):
        kwargs = dict(FROZEN)
        kwargs[key] = bad
        with pytest.raises(ValueError, match="frozen"):
            pp.estimate_gate_r_plus_joint_power(bundle, **kwargs)


def test_estimate_requires_both_dev_sets():
    kwargs = dict(FROZEN)
    with pytest.raises(ValueError):
        pp.estimate_gate_r_plus_joint_power({"A": _synthetic_set_bundle()}, **kwargs)


# --------------------------------------------------------------------------- #
# Step 2: joint (not marginal) selection
# --------------------------------------------------------------------------- #
def test_joint_power_selects_by_weakest_condition_not_marginals():
    # All five core marginals are strongly positive, but seed 9's per-seed
    # Spearman point is a hair above zero with large spread: at small n the
    # per-seed core-point condition fails often and only concentration at
    # larger n rescues it. The JOINT power — never any marginal — is what
    # pushes the selection upward.
    bundle = {
        "A": _synthetic_set_bundle(E=400, seed=2, weak_seed_rank=(0.004, 0.08)),
        "B": _synthetic_set_bundle(E=400, seed=3, weak_seed_rank=(0.004, 0.08)),
    }
    out = pp.estimate_gate_r_plus_joint_power(bundle, **FROZEN)
    selected = out["selected_episode_count"]
    assert selected is None or selected > FROZEN["candidates"][0]
    per_candidate = out["per_candidate"]
    n0 = FROZEN["candidates"][0]
    entry = per_candidate[str(n0)] if str(n0) in per_candidate else per_candidate[n0]
    rates = entry["condition_pass_rates"]
    core_rates = [rates[c] for c in
                  ("i_mae_ci", "spearman_ci", "i_pr_presence_ci",
                   "i_pr_high_vs_lowmid_ci", "i_brier_ci")]
    # core marginals all ~1 at n0, but joint is dragged below by stability
    assert min(core_rates) > entry["joint_power"]
    assert rates["per_seed_core_positive"] < 0.90  # the binding condition
    assert entry["weakest_condition"]["condition"] == "per_seed_core_positive"
    assert entry["joint_power"] == pytest.approx(
        min(entry["set_joint_power"]["A"], entry["set_joint_power"]["B"]))


def test_strong_bundle_selects_smallest_candidate():
    out = pp.estimate_gate_r_plus_joint_power(_bundle(), **FROZEN)
    assert out["selected_episode_count"] == FROZEN["candidates"][0]
    assert out["status"] == "POWER_FEASIBLE"


# --------------------------------------------------------------------------- #
# Step 3: infeasible at n=2000 -> no artifacts
# --------------------------------------------------------------------------- #
def test_infeasible_power_blocks_freeze(tmp_path):
    out = pp.estimate_gate_r_plus_joint_power(_bundle(strong=False), **FROZEN)
    assert out["selected_episode_count"] is None
    assert out["status"] == "STAGE3A_PLUS_QUALIFICATION_POWER_INFEASIBLE"
    with pytest.raises(RuntimeError, match="INFEASIBLE"):
        pp.freeze_qualification_artifacts(
            tmp_path, power_report=out, historical_seeds=np.arange(10),
            collection_seed=s3d.STAGE3A_PLUS_COLLECTION_SEED,
            power_report_sha256="sha")
    assert not (tmp_path / "qualification").exists()


# --------------------------------------------------------------------------- #
# Step 4: NaN draws fail; worse-of-A/B drives the joint power
# --------------------------------------------------------------------------- #
def test_nan_draws_count_as_failures():
    bundle = _bundle()
    calls = {"n": 0}

    def flaky_ece(idx):
        calls["n"] += 1
        return (float("nan"), 0.06) if calls["n"] % 2 == 0 else (0.04, 0.06)

    bundle["A"]["ece_fn"] = flaky_ece
    bundle["B"]["ece_fn"] = flaky_ece
    out = pp.estimate_gate_r_plus_joint_power(bundle, **FROZEN)
    n0 = FROZEN["candidates"][0]
    entry = out["per_candidate"].get(str(n0), out["per_candidate"][n0])
    assert entry["condition_pass_rates"]["ece_guard"] < 1.0
    assert entry["joint_power"] < 1.0


def test_worse_of_ab_drives_joint_power():
    strong = _synthetic_set_bundle(seed=4)
    weak = _synthetic_set_bundle(E=400, seed=5, weak_seed_rank=(0.004, 0.08))
    out = pp.estimate_gate_r_plus_joint_power({"A": strong, "B": weak}, **FROZEN)
    n0 = FROZEN["candidates"][0]
    entry = out["per_candidate"].get(str(n0), out["per_candidate"][n0])
    assert entry["set_joint_power"]["A"] >= entry["set_joint_power"]["B"]
    assert entry["joint_power"] == pytest.approx(entry["set_joint_power"]["B"])


# --------------------------------------------------------------------------- #
# Step 5: freeze + three-way hash binding
# --------------------------------------------------------------------------- #
def _feasible_report():
    return {
        "plan_id": s3d.STAGE3A_PLUS_PLAN_ID,
        "artifact_type": "power_report",
        "status": "POWER_FEASIBLE",
        "selected_episode_count": 400,
        "candidates": list(s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS),
        "target_power": s3d.STAGE3A_PLUS_TARGET_POWER,
        "monte_carlo_replicates": s3d.STAGE3A_PLUS_MONTE_CARLO_REPLICATES,
        "inner_bootstrap_resamples": s3d.STAGE3A_PLUS_INNER_BOOTSTRAP,
        "seed": s3d.STAGE3A_PLUS_POWER_SEED,
    }


def test_freeze_writes_three_bound_artifacts(tmp_path):
    import hashlib
    import json
    from ac_twin.stage1_data import sha256_file
    out = pp.freeze_qualification_artifacts(
        tmp_path, power_report=_feasible_report(),
        historical_seeds=np.arange(1000),
        collection_seed=s3d.STAGE3A_PLUS_COLLECTION_SEED,
        power_report_sha256="powersha")
    config_path = out["config_path"]
    schedule_path = out["seed_schedule_path"]
    assert config_path.is_file() and schedule_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    # config binds to the power report and the schedule
    assert config["power_report_sha256"] == "powersha"
    assert config["seed_schedule_sha256"] == hashlib.sha256(
        np.array(schedule["schedule"], dtype=np.int64).tobytes()).hexdigest()
    # schedule binds back to the power report AND the config file
    assert schedule["power_report_sha256"] == "powersha"
    assert schedule["config_file_sha256"] == sha256_file(config_path)
    # generation contract: ten equal strata, schedule = 15x, merge-v0 scene
    assert config["n_pairs"] == 400
    assert config["episodes_per_stratum"] == 40
    assert len(schedule["schedule"]) == 400 * 15
    assert config["scene"] == "merge-v0"
    assert config["actions"] == [0, 1, 2, 3, 4]
    assert config["intervention_durations"] == [1, 3]
    assert config["collection_seed"] == s3d.STAGE3A_PLUS_COLLECTION_SEED
    # single-shot: second freeze refused
    with pytest.raises(FileExistsError):
        pp.freeze_qualification_artifacts(
            tmp_path, power_report=_feasible_report(),
            historical_seeds=np.arange(1000),
            collection_seed=s3d.STAGE3A_PLUS_COLLECTION_SEED,
            power_report_sha256="powersha")


def test_freeze_rejects_bad_seed_and_untied_report(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        pp.freeze_qualification_artifacts(
            tmp_path, power_report=_feasible_report(),
            historical_seeds=np.arange(10), collection_seed=999,
            power_report_sha256="powersha")
    report = dict(_feasible_report(), selected_episode_count=455)  # not divisible by 10
    with pytest.raises(ValueError):
        pp.freeze_qualification_artifacts(
            tmp_path / "x", power_report=report,
            historical_seeds=np.arange(10),
            collection_seed=s3d.STAGE3A_PLUS_COLLECTION_SEED,
            power_report_sha256="powersha")
    # schedule overlapping historical seeds is refused
    from ac_twin.stage2_protocol import make_seed_schedule
    overlapping = make_seed_schedule(s3d.STAGE3A_PLUS_COLLECTION_SEED, size=400 * 15)
    with pytest.raises((ValueError, RuntimeError)):
        pp.freeze_qualification_artifacts(
            tmp_path / "y", power_report=_feasible_report(),
            historical_seeds=np.asarray(overlapping),
            collection_seed=s3d.STAGE3A_PLUS_COLLECTION_SEED,
            power_report_sha256="powersha")
