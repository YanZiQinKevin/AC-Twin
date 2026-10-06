"""Stage 3A Plus qualification power sizing — full-Gate JOINT power (plan §6.1).

Unlike the repair-phase power script (five marginals, weakest sizes n), this
estimator simulates the COMPLETE Gate R+ AND-chain on every Monte Carlo draw:

- the five core CI conditions (``estimate - Z_95 * SE > 0``);
- the ECE guard (<= 0.05 or >= 20% relative improvement over uncalibrated);
- the stability block: per-seed core point positivity (10 seeds x 4 metrics),
  >= 8/10 positive I_Brier, median I_Brier > 0, worst-seed non-inferiority
  (delta_B = 5% of the draw's baseline Brier), hierarchical seed x episode CI
  lower > 0, and the non-degeneracy floor.

Gate Q+ is an engineering/protocol check and is assumed to pass in the power
simulation (it carries no sampling variance).

The effect distributions come from BOTH historical dev sets (A = old merge
split 2, B = replacement holdout); the joint power at each candidate n is the
MINIMUM over the two sets (the worse effect distribution sizes the study,
plan §6.1 "基于开发集 A/B 中较差的效应分布").

Frozen constants (any deviation is rejected): candidates
[400,600,800,1000,1200,1600,2000], 2,000 Monte Carlo replicates, 1,000 inner
bootstrap resamples, target power 0.90, power seed 20260930. The formal
adjudication still uses 10,000 bootstraps — the power stage never feeds back
into the gate.

CI mechanics (documented approximation): mean-decomposable statistics
(I_MAE, I_Brier, per-seed I_Brier, hierarchical I_Brier) use the closed-form
bootstrap SE of the episode mean (equivalent to a 1,000-resample inner
bootstrap); the three rank statistics (pooled Spearman, two PR-AUC contrasts)
are evaluated EXACTLY on each draw, with SE from a 1,000-resample episode
bootstrap on the full pool, sqrt(E/n)-scaled to the draw size. NaN draws count
as failures. Selection is by JOINT power only — never by marginals.

Outputs (single-shot, hash-bound, under the stage3a_plus run dir):
``power/power_report.json`` first; only when feasible,
``qualification/config.json`` then ``qualification/seed_schedule.json``.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage1_plus_statistics import Z_95  # noqa: E402
from ac_twin.stage2_protocol import make_seed_schedule, validate_seed_schedule  # noqa: E402
from ac_twin.stage3a_plus_statistics import (  # noqa: E402
    ECE_GUARD,
    ECE_RELATIVE_IMPROVEMENT,
    MIN_POSITIVE_BRIER_SEEDS,
    NONDEGENERATE_STD_FLOOR,
    WORST_SEED_NONINFERIORITY_FRAC,
)

PLAN_ID = s3d.STAGE3A_PLUS_PLAN_ID
POWER_INFEASIBLE = "STAGE3A_PLUS_QUALIFICATION_POWER_INFEASIBLE"
RANK_STATS = ("spearman", "i_pr_presence", "i_pr_high_vs_lowmid")
CONDITIONS = (
    "i_mae_ci", "spearman_ci", "i_pr_presence_ci", "i_pr_high_vs_lowmid_ci",
    "i_brier_ci", "ece_guard", "per_seed_core_positive", "at_least_8_of_10_positive",
    "median_i_brier_positive", "worst_seed_noninferior", "hierarchical_ci",
    "non_degenerate",
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Joint power estimation
# --------------------------------------------------------------------------- #
def _validate_frozen(candidates, monte_carlo_replicates, inner_bootstrap_resamples,
                     target_power, seed) -> None:
    if tuple(int(n) for n in candidates) != tuple(s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS):
        raise ValueError(
            f"candidate episode counts are frozen to "
            f"{tuple(s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS)}")
    if int(monte_carlo_replicates) != s3d.STAGE3A_PLUS_MONTE_CARLO_REPLICATES:
        raise ValueError(
            f"monte_carlo_replicates is frozen to {s3d.STAGE3A_PLUS_MONTE_CARLO_REPLICATES}")
    if int(inner_bootstrap_resamples) != s3d.STAGE3A_PLUS_INNER_BOOTSTRAP:
        raise ValueError(
            f"inner_bootstrap_resamples is frozen to {s3d.STAGE3A_PLUS_INNER_BOOTSTRAP}")
    if float(target_power) != s3d.STAGE3A_PLUS_TARGET_POWER:
        raise ValueError(f"target_power is frozen to {s3d.STAGE3A_PLUS_TARGET_POWER}")
    if int(seed) != s3d.STAGE3A_PLUS_POWER_SEED:
        raise ValueError(f"power seed is frozen to {s3d.STAGE3A_PLUS_POWER_SEED}")


def _rank_pool_ses(bundle_set: Mapping[str, Any], inner: int, seed: int) -> dict[str, float]:
    """Full-pool episode-bootstrap SE for each ensemble rank statistic, using
    exactly ``inner`` resamples (the frozen inner bootstrap count)."""
    rng = np.random.default_rng(seed)
    E = int(bundle_set["n_episodes"])
    out = {}
    for name in RANK_STATS:
        fn = bundle_set["rank_fns"][name]
        draws = np.array([fn(rng.integers(0, E, size=E)) for _ in range(inner)])
        finite = draws[np.isfinite(draws)]
        out[name] = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
    return out


def _draw_conditions(b: Mapping[str, Any], idx: np.ndarray, *,
                     rank_se_pool: Mapping[str, float], n: int) -> dict[str, bool]:
    """Evaluate all twelve Gate R+ sampling conditions on one episode draw.

    Every non-finite component fails its condition (NaN draws count as
    failures)."""
    idx = np.asarray(idx)
    E = int(b["n_episodes"])
    conds: dict[str, bool] = {}

    # --- core: mean-decomposable statistics (closed-form bootstrap SE) ------
    i_mae_vals = b["i_mae_ep"][idx]
    i_mae_pt = float(np.mean(i_mae_vals))
    i_mae_se = float(np.std(i_mae_vals, ddof=1) / np.sqrt(idx.size))
    conds["i_mae_ci"] = bool(i_mae_pt - Z_95 * i_mae_se > 0.0)

    ss_m = float(b["brier"]["model_ss_ep"][idx].sum())
    ss_b = float(b["brier"]["base_ss_ep"][idx].sum())
    n_pts = float(b["brier"]["n_ep"][idx].sum())
    model_brier = ss_m / (4.0 * n_pts)
    base_brier = ss_b / (4.0 * n_pts)
    i_brier_pt = base_brier - model_brier
    i_brier_ep = (b["brier"]["base_ss_ep"] - b["brier"]["model_ss_ep"]) / (
        4.0 * b["brier"]["n_ep"])
    i_brier_se = float(np.std(i_brier_ep[idx], ddof=1) / np.sqrt(idx.size))
    conds["i_brier_ci"] = bool(i_brier_pt - Z_95 * i_brier_se > 0.0)

    # --- core: rank statistics (exact point on the draw, pool-SE scaled) ----
    scale = float(np.sqrt(E / idx.size))
    for name in RANK_STATS:
        pt = float(b["rank_fns"][name](idx))
        conds[f"{name}_ci"] = bool(pt - Z_95 * float(rank_se_pool[name]) * scale > 0.0)

    # --- ECE guard -----------------------------------------------------------
    ece_cal, ece_uncal = b["ece_fn"](idx)
    conds["ece_guard"] = bool(
        np.isfinite(ece_cal) and np.isfinite(ece_uncal)
        and (ece_cal <= ECE_GUARD
             or ece_cal <= (1.0 - ECE_RELATIVE_IMPROVEMENT) * ece_uncal))

    # --- stability: per-seed core point positivity ---------------------------
    per_seed_ok = True
    n_seeds = len(b["i_mae_ep_seeds"])
    i_brier_seeds = np.empty(n_seeds, dtype=np.float64)
    for s in range(n_seeds):
        if float(np.mean(b["i_mae_ep_seeds"][s][idx])) <= 0.0:
            per_seed_ok = False
        for name in RANK_STATS:
            if float(b["rank_fns_seeds"][name][s](idx)) <= 0.0:
                per_seed_ok = False
        ss_ms = float(b["brier_seeds"]["model_ss_ep"][s][idx].sum())
        n_s = float(b["brier_seeds"]["n_ep"][s][idx].sum())
        i_brier_seeds[s] = (ss_b / (4.0 * n_s)) - (ss_ms / (4.0 * n_s))
    conds["per_seed_core_positive"] = bool(per_seed_ok)

    # --- stability: I_Brier direction conditions ------------------------------
    conds["at_least_8_of_10_positive"] = bool(
        (i_brier_seeds > 0.0).sum() >= MIN_POSITIVE_BRIER_SEEDS)
    conds["median_i_brier_positive"] = bool(np.median(i_brier_seeds) > 0.0)
    conds["worst_seed_noninferior"] = bool(
        i_brier_seeds.min() >= -WORST_SEED_NONINFERIORITY_FRAC * base_brier)

    # --- stability: hierarchical seed x episode CI (seed-cluster formula) -----
    vals = b["i_brier_ep_seeds"][:, idx]  # (S, n)
    hier_pt = float(vals.mean())
    per_seed_means = vals.mean(axis=1)
    hier_se = float(np.std(per_seed_means, ddof=1) / np.sqrt(per_seed_means.size))
    conds["hierarchical_ci"] = bool(hier_pt - Z_95 * hier_se > 0.0)

    # --- stability: non-degeneracy --------------------------------------------
    conds["non_degenerate"] = bool(
        float(b["r_ensemble_std_fn"](idx)) > NONDEGENERATE_STD_FLOOR)
    return conds


def estimate_gate_r_plus_joint_power(
    statistic_bundle: Mapping[str, Any],
    *,
    candidates,
    monte_carlo_replicates: int,
    inner_bootstrap_resamples: int,
    target_power: float,
    seed: int,
    progress_prefix: str = "3A+-power",
) -> dict[str, Any]:
    """Joint (full AND-gate) Monte Carlo power per frozen candidate size.

    ``statistic_bundle`` carries BOTH dev sets (``"A"``/``"B"``); the reported
    joint power at each n is the minimum over the two sets.
    """
    _validate_frozen(candidates, monte_carlo_replicates, inner_bootstrap_resamples,
                     target_power, seed)
    candidates = [int(n) for n in candidates]
    replicates = int(monte_carlo_replicates)
    inner = int(inner_bootstrap_resamples)
    if set(statistic_bundle) != {"A", "B"}:
        raise ValueError("statistic_bundle must carry both dev sets 'A' and 'B'")

    rank_se_pool = {
        name: _rank_pool_ses(statistic_bundle[name], inner, seed + 1000 + i)
        for i, name in enumerate(("A", "B"))
    }

    rng = np.random.default_rng(seed)
    reporter = ProgressReporter(total=len(candidates) * 2, every=1,
                                prefix=progress_prefix, heartbeat_seconds=30.0)
    per_candidate: dict[int, dict[str, Any]] = {}
    done = 0
    for n in candidates:
        started = time.perf_counter()
        set_results: dict[str, Any] = {}
        for set_name in ("A", "B"):
            b = statistic_bundle[set_name]
            E = int(b["n_episodes"])
            pass_matrix = np.zeros((len(CONDITIONS), replicates), dtype=bool)
            for r in range(replicates):
                idx = rng.integers(0, E, size=n)
                conds = _draw_conditions(b, idx, rank_se_pool=rank_se_pool[set_name], n=n)
                pass_matrix[:, r] = [conds[c] for c in CONDITIONS]
            rates = {c: float(pass_matrix[i].mean()) for i, c in enumerate(CONDITIONS)}
            set_results[set_name] = {
                "joint_power": float(pass_matrix.all(axis=0).mean()),
                "condition_pass_rates": rates,
            }
            done += 1
            reporter.update(done, details=f"n={n} set={set_name} "
                            f"joint={set_results[set_name]['joint_power']:.3f} "
                            f"candidate_seconds={time.perf_counter() - started:.1f}")
        joint_min = min(set_results["A"]["joint_power"], set_results["B"]["joint_power"])
        cond_rates = {c: min(set_results["A"]["condition_pass_rates"][c],
                             set_results["B"]["condition_pass_rates"][c])
                      for c in CONDITIONS}
        weakest = min(CONDITIONS, key=lambda c: cond_rates[c])
        per_candidate[n] = {
            "joint_power": joint_min,
            "set_joint_power": {s: set_results[s]["joint_power"] for s in ("A", "B")},
            "condition_pass_rates": cond_rates,
            "weakest_condition": {"condition": weakest, "pass_rate": cond_rates[weakest]},
        }

    selected = next((n for n in candidates
                     if per_candidate[n]["joint_power"] >= float(target_power)), None)
    return {
        "per_candidate": per_candidate,
        "selected_episode_count": selected,
        "status": "POWER_FEASIBLE" if selected is not None else POWER_INFEASIBLE,
        "candidates": candidates,
        "target_power": float(target_power),
        "monte_carlo_replicates": replicates,
        "inner_bootstrap_resamples": inner,
        "seed": int(seed),
        "z": float(Z_95),
        "rank_se_pool": rank_se_pool,
        "rule": (
            "joint power = fraction of draws where ALL twelve Gate R+ sampling "
            "conditions hold, minimised over dev sets A/B (worse effect "
            "distribution sizes the study); CI conditions use estimate - Z_95*SE "
            "> 0 with closed-form bootstrap SE for episode-mean statistics and "
            "pool-bootstrap SE * sqrt(E/n) for rank statistics; NaN draws fail; "
            "selection is by joint power, never marginals"),
    }


# --------------------------------------------------------------------------- #
# Qualification artifact freeze (single-shot, three-way hash binding)
# --------------------------------------------------------------------------- #
def qualification_paths(run_dir: str | Path, *, data_root: str | Path | None = None
                        ) -> dict[str, Path]:
    run_dir = Path(run_dir)
    data_root = Path("data") if data_root is None else Path(data_root)
    qual_dir = run_dir / "qualification"
    data_dir = data_root / "object_centric_stage" / s3d.STAGE3A_PLUS_QUALIFICATION_DIR_SEGMENT
    return {
        "power_report_path": run_dir / "power" / "power_report.json",
        "config_path": qual_dir / "config.json",
        "seed_schedule_path": qual_dir / "seed_schedule.json",
        "output_dir": qual_dir / "collection",
        "data_output": data_dir / "object_twinworld_stage3a_plus_qualification.npz",
        "rejection_ledger": data_dir / "rejection_ledger.jsonl",
        "collection_receipt": qual_dir / "collection_receipt.json",
        "historical_data_root": data_root,
    }


def freeze_qualification_artifacts(
    run_dir: str | Path,
    power_report: Mapping[str, Any],
    historical_seeds: np.ndarray,
    collection_seed: int,
    *,
    power_report_sha256: str,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Freeze the qualification config + seed schedule after a feasible joint
    power run. Hard-stops on infeasibility; refuses non-frozen seeds, counts
    not divisible by ten strata, seed-schedule overlap with the historical
    universe, and any pre-existing artifact (single-shot)."""
    if int(collection_seed) != int(s3d.STAGE3A_PLUS_COLLECTION_SEED):
        raise ValueError(
            f"collection seed is frozen to {s3d.STAGE3A_PLUS_COLLECTION_SEED}")
    if tuple(int(n) for n in power_report.get("candidates", ())) != tuple(
            s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS):
        raise ValueError("power report candidates deviate from the frozen list")
    selected = power_report.get("selected_episode_count")
    if selected is None or power_report.get("status") == POWER_INFEASIBLE:
        raise RuntimeError(
            f"{POWER_INFEASIBLE}: joint power never reaches "
            f"{s3d.STAGE3A_PLUS_TARGET_POWER} within "
            f"{list(s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS)}; stop and return to "
            "user review — do not relax the gate")
    n_pairs = int(selected)
    if n_pairs % 10 != 0:
        raise ValueError(
            f"qualification n={n_pairs} must be divisible by the ten "
            "action x duration strata")

    paths = dict(paths) if paths is not None else qualification_paths(run_dir)
    for key in ("config_path", "seed_schedule_path"):
        if paths[key].exists():
            raise FileExistsError(
                f"frozen qualification artifact already exists: {paths[key]}; "
                "the freeze is single-shot and never overwritten")

    schedule = make_seed_schedule(int(collection_seed), size=n_pairs * 15)
    validate_seed_schedule(schedule, np.asarray(historical_seeds, dtype=np.int64))
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()

    contract = s3d.MERGE_GENERATION_CONTRACT
    config = {
        "experiment_name": "stage3a_plus_qualification",
        "analysis_status": s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE,
        "purpose": s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE,
        "plan_id": PLAN_ID,
        "scene": "merge-v0",
        "env_id": contract["env_id"],
        "n_pairs": n_pairs,
        "actions": [0, 1, 2, 3, 4],
        "intervention_durations": [1, 3],
        "episodes_per_stratum": n_pairs // 10,
        "stratified_equal_counts": True,
        "no_posthoc_selection": True,
        "collection_seed": int(collection_seed),
        "seed_schedule_size": int(schedule.size),
        "max_attempts": int(schedule.size),
        "horizon": contract["horizon"],
        "history_len": contract["history_len"],
        "t_star_low": contract["t_star_low"],
        "t_star_high": contract["t_star_high"],
        "vehicles_count": contract["vehicles_count"],
        "observed_vehicles_count": contract["observed_vehicles_count"],
        "duration": contract["duration"],
        "simulation_frequency": contract["simulation_frequency"],
        "policy_frequency": contract["policy_frequency"],
        "determinism_checks": contract["determinism_checks"],
        "action_weights": dict(contract["action_weights"]),
        "factual_suffix_steps": 10,
        "checkpoint_every_accepted": 25,
        "output_dir": str(paths["output_dir"]),
        "data_output": str(paths["data_output"]),
        "rejection_ledger": str(paths["rejection_ledger"]),
        "collection_receipt": str(paths["collection_receipt"]),
        "historical_data_root": str(paths["historical_data_root"]),
        "power_report_sha256": str(power_report_sha256),
        "seed_schedule_sha256": schedule_sha,
    }
    _atomic_write_json(paths["config_path"], config)
    config_file_sha = sha256_file(paths["config_path"])

    historical = np.asarray(historical_seeds, dtype=np.int64)
    schedule_doc = {
        "purpose": s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE,
        "plan_id": PLAN_ID,
        "collection_seed": int(collection_seed),
        "size": int(schedule.size),
        "schedule": [int(v) for v in schedule],
        "schedule_sha256": schedule_sha,
        "historical_seed_count": int(historical.size),
        "historical_seed_universe_sha256": hashlib.sha256(historical.tobytes()).hexdigest(),
        "power_report_sha256": str(power_report_sha256),
        "config_fingerprint": canonical_fingerprint(json.loads(json.dumps(config))),
        "config_file_sha256": config_file_sha,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_write_json(paths["seed_schedule_path"], schedule_doc)
    return {
        "config_path": paths["config_path"],
        "seed_schedule_path": paths["seed_schedule_path"],
        "schedule": schedule,
    }


# --------------------------------------------------------------------------- #
# Production bundle builder (torch; isolated from the pure estimator)
# --------------------------------------------------------------------------- #
def build_set_bundle(
    eval_set: Mapping[str, Any],
    cal_probs_seeds: np.ndarray,
    uncal_probs_seeds: np.ndarray,
    *,
    n_production_seeds: int,
) -> dict[str, Any]:
    """Per-episode statistic bundle for one dev set, built from the frozen
    ten-seed predictions with the frozen calibrator already applied.

    All statistic factories are the SAME ones the gate uses; the closures take
    episode-index vectors exactly as the episode bootstrap resamples them.
    """
    from ac_twin.stage3_response import (  # local: keeps the estimator torch-free
        expected_calibration_error,
    )
    from ac_twin.stage3_statistics import (
        mae_per_episode,
        make_brier_difference_statistic,
        make_pooled_spearman_statistic,
        make_pr_auc_difference_statistic,
    )
    from ac_twin.stage3a_plus_statistics import (
        binary_labels,
        high_vs_lowmid_scores,
        presence_scores,
        _HIGH,
        _PRESENCE_POSITIVE,
    )

    r_true = np.asarray(eval_set["r_true"], dtype=np.float64)
    labels = np.asarray(eval_set["regime_labels"], dtype=np.int64)
    valid = np.asarray(eval_set["valid_mask"], dtype=bool)
    r_seeds = np.asarray(eval_set["r_hat_seeds"], dtype=np.float64)
    cal_seeds = np.asarray(cal_probs_seeds, dtype=np.float64)
    uncal_seeds = np.asarray(uncal_probs_seeds, dtype=np.float64)
    n_prod = int(n_production_seeds)
    S = r_seeds.shape[0]
    E = r_true.shape[0]

    cont_base = np.asarray(
        eval_set["baselines"]["continuous"][eval_set["baselines"]["strongest_continuous"]],
        dtype=np.float64)
    clf_base = np.asarray(
        eval_set["baselines"]["classification"][eval_set["baselines"]["strongest_brier"]],
        dtype=np.float64)

    r_ens = r_seeds[:n_prod].mean(axis=0)
    cal_ens = cal_seeds[:n_prod].mean(axis=0)
    uncal_ens = uncal_seeds[:n_prod].mean(axis=0)

    base_mae_ep = mae_per_episode(cont_base, r_true, valid)
    model_mae_ep = mae_per_episode(r_ens, r_true, valid)
    i_mae_ep = base_mae_ep - model_mae_ep
    i_mae_ep_seeds = np.stack(
        [base_mae_ep - mae_per_episode(r_seeds[s], r_true, valid) for s in range(S)],
        axis=0)

    onehot = np.eye(cal_ens.shape[-1], dtype=np.float64)[labels]  # E,T,N,4

    def _ss_per_episode(probs):
        sq = ((probs - onehot) ** 2).sum(axis=-1)  # E,T,N (sum over classes)
        sq = np.where(valid, sq, 0.0)
        return sq.sum(axis=(1, 2))  # (E,)

    n_ep = valid.sum(axis=(1, 2)).astype(np.float64)  # (E,)
    model_ss_ep = _ss_per_episode(cal_ens)
    base_ss_ep = _ss_per_episode(clf_base)
    seeds_model_ss = np.stack([_ss_per_episode(cal_seeds[s]) for s in range(S)], axis=0)
    seeds_n = np.broadcast_to(n_ep, (S, E)).copy()

    i_brier_ep_seeds = np.empty((S, E), dtype=np.float64)
    for s in range(S):
        stat = make_brier_difference_statistic(cal_seeds[s], clf_base, labels, valid)
        for e in range(E):
            i_brier_ep_seeds[s, e] = stat(np.array([e]))

    presence_lab = binary_labels(labels, _PRESENCE_POSITIVE)
    pos_mask = valid & (presence_lab == 1)
    high_lab = binary_labels(labels, (_HIGH,))

    rank_fns = {
        "spearman": make_pooled_spearman_statistic(r_ens, r_true, valid),
        "i_pr_presence": make_pr_auc_difference_statistic(
            presence_scores(cal_ens), presence_lab,
            presence_scores(clf_base), presence_lab, valid),
        "i_pr_high_vs_lowmid": make_pr_auc_difference_statistic(
            high_vs_lowmid_scores(cal_ens), high_lab,
            high_vs_lowmid_scores(clf_base), high_lab, pos_mask),
    }
    rank_fns_seeds = {
        "spearman": [make_pooled_spearman_statistic(r_seeds[s], r_true, valid)
                     for s in range(S)],
        "i_pr_presence": [make_pr_auc_difference_statistic(
            presence_scores(cal_seeds[s]), presence_lab,
            presence_scores(clf_base), presence_lab, valid) for s in range(S)],
        "i_pr_high_vs_lowmid": [make_pr_auc_difference_statistic(
            high_vs_lowmid_scores(cal_seeds[s]), high_lab,
            high_vs_lowmid_scores(clf_base), high_lab, pos_mask) for s in range(S)],
    }

    def ece_fn(idx):
        idx = np.asarray(idx)
        probs_c = np.concatenate([cal_ens[e][valid[e]] for e in idx], axis=0)
        probs_u = np.concatenate([uncal_ens[e][valid[e]] for e in idx], axis=0)
        lab = np.concatenate([labels[e][valid[e]] for e in idx], axis=0)
        v = np.ones(lab.shape, dtype=bool)
        return (expected_calibration_error(probs_c, lab, v, n_bins=15, equal_mass=True),
                expected_calibration_error(probs_u, lab, v, n_bins=15, equal_mass=True))

    def r_ensemble_std_fn(idx):
        idx = np.asarray(idx)
        pooled = np.concatenate([r_ens[e][valid[e]] for e in idx], axis=0)
        return float(np.std(pooled)) if pooled.size else 0.0

    return {
        "n_episodes": E,
        "i_mae_ep": i_mae_ep,
        "i_mae_ep_seeds": i_mae_ep_seeds,
        "brier": {"model_ss_ep": model_ss_ep, "base_ss_ep": base_ss_ep, "n_ep": n_ep},
        "brier_seeds": {"model_ss_ep": seeds_model_ss, "n_ep": seeds_n},
        "i_brier_ep_seeds": i_brier_ep_seeds,
        "rank_fns": rank_fns,
        "rank_fns_seeds": rank_fns_seeds,
        "ece_fn": ece_fn,
        "r_ensemble_std_fn": r_ensemble_std_fn,
    }


# --------------------------------------------------------------------------- #
# main (autonomous under run-all; state-gated)
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3A Plus qualification power sizing")
    parser.add_argument("--config", default="experiments/stage3_routing/stage3a_plus/dev_config.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import run_stage3a_plus as drv  # local: torch driver

    args = parse_args(argv)
    config = drv.load_dev_config(args.config)
    run_dir = _resolve(config["run_dir"])
    state = drv.load_state(run_dir)["state"]
    if state != "COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS":
        print(f"[3A+-power] REFUSED: state {state} != COMPUTE_BUDGET_RECORDED_STAGE3A_PLUS",
              flush=True)
        return 2
    paths = qualification_paths(run_dir)
    if paths["power_report_path"].exists():
        print(f"[3A+-power] REFUSED: {paths['power_report_path']} exists (single-shot)",
              flush=True)
        return 2

    freeze_receipt = json.loads(
        (run_dir / "freeze" / "final_freeze_receipt.json").read_text(encoding="utf-8"))
    payload = freeze_receipt["receipt_payload"]
    if freeze_receipt["receipt_sha256"] != canonical_fingerprint(payload):
        print("[3A+-power] freeze receipt integrity failure", flush=True)
        return 2
    artifact = drv.calibrator_from_selection_receipt(json.loads(
        drv._calibration_selection_path(run_dir, int(payload["round"]))
        .read_text(encoding="utf-8")))

    # Ten-seed inference on dev A and B with the frozen checkpoints.
    prior = _resolve(config["prior_run_dir"])
    bundle = drv.s3a.load_development_tensors(_resolve(config["development_source"]))
    tensors, roles = bundle["tensors"], bundle["roles"]
    scaler = drv.s3a.prepare_scaler(tensors, roles, prior)
    scaled = drv.s3a.scale_tensors(tensors, scaler)
    repl = drv.s3a.load_replacement_tensors(_resolve(config["replacement_source"]))
    repl_scaled = drv.s3a.scale_tensors(repl["tensors"], scaler)
    checkpoints = {int(s): rec["path"] for s, rec in payload["checkpoints"].items()}
    for seed, sig in payload["checkpoints"].items():
        drv.s3t.load_predictor_state(checkpoints[int(seed)], expected_signature=sig["signature"])

    cfg = config["final_config"]
    strongest = drv._strongest_basenames(prior)
    train_prev = drv._train_prevalence(tensors, roles)
    from ac_twin.stage3_response import softmax  # local torch-adjacent import

    eval_sets = {}
    for name, tns, raw, idx in (
            ("A", tensors, bundle["raw_factual_suffix"], roles["development_test"]),
            ("B", repl["tensors"], repl["raw_factual_suffix"], repl["roles"]["eval"])):
        sc = scaled if name == "A" else repl_scaled
        r_seeds, logit_seeds = drv._infer_all_seeds(checkpoints, cfg, sc, idx, config["device"])
        baselines = drv._build_eval_baselines(tensors, bundle["raw_factual_suffix"], roles,
                                              tns, raw, idx, strongest)
        es = drv._make_eval_set(name, r_seeds, logit_seeds, tns, idx, baselines)
        cal_probs = drv.cal.apply_calibrator(artifact, logit_seeds, train_prev)
        uncal_probs = np.stack([softmax(logit_seeds[s], axis=-1)
                                for s in range(logit_seeds.shape[0])], axis=0)
        eval_sets[name] = build_set_bundle(
            es, cal_probs, uncal_probs,
            n_production_seeds=int(config.get("n_production_seeds", 5)))
        print(f"[3A+-power] bundle built for dev set {name}", flush=True)

    result = estimate_gate_r_plus_joint_power(
        eval_sets,
        candidates=s3d.STAGE3A_PLUS_CANDIDATE_EPISODE_COUNTS,
        monte_carlo_replicates=s3d.STAGE3A_PLUS_MONTE_CARLO_REPLICATES,
        inner_bootstrap_resamples=s3d.STAGE3A_PLUS_INNER_BOOTSTRAP,
        target_power=s3d.STAGE3A_PLUS_TARGET_POWER,
        seed=s3d.STAGE3A_PLUS_POWER_SEED)

    report = {
        "plan_id": PLAN_ID,
        "artifact_type": "power_report",
        "purpose": s3d.STAGE3A_PLUS_QUALIFICATION_PURPOSE,
        **result,
        "freeze_receipt_sha256": freeze_receipt["receipt_sha256"],
        "data_signatures": {"development": bundle["data_signature"],
                            "replacement": repl["data_signature"]},
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_write_json(paths["power_report_path"], report)
    report_sha = sha256_file(paths["power_report_path"])
    print(f"[3A+-power] status={result['status']} "
          f"selected={result['selected_episode_count']} -> {paths['power_report_path']}",
          flush=True)

    if result["selected_episode_count"] is None:
        # Hard stop (plan §9.3 condition 4): no state advance, no collection
        # artifacts — report and wait for user review.
        print(f"[3A+-power] {POWER_INFEASIBLE}: joint power below "
              f"{s3d.STAGE3A_PLUS_TARGET_POWER} at every candidate; collection is "
              "forbidden — stop and report", flush=True)
        return 3

    historical = s3d.scan_all_episode_seeds(
        ROOT / "data", exclude_substring=s3d.STAGE3A_PLUS_QUALIFICATION_DIR_SEGMENT)
    artifacts = freeze_qualification_artifacts(
        run_dir, report, historical, s3d.STAGE3A_PLUS_COLLECTION_SEED,
        power_report_sha256=report_sha)
    drv.advance_state(run_dir, "QUALIFICATION_POWER_FROZEN",
                      note=f"n={result['selected_episode_count']}")
    print(f"[3A+-power] froze n={result['selected_episode_count']} -> "
          f"{artifacts['config_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
