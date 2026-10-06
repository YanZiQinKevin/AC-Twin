"""Stage 3A replacement development holdout — power sizing (repair plan v0.1 §3.2).

Uses the OLD, invalidated run's final config, five checkpoints, frozen baselines
and the old merge split 2 — with the repaired episode alignment — purely to plan
the replacement sample size. This forensic use is explicitly sanctioned for power
planning ONLY: it must not change the model, thresholds, search space or gate.

Mechanical rule (frozen):
- episode-resampling Monte Carlo power for each of ``I_MAE``, ``Spearman``,
  ``I_PR_presence``, ``I_PR_high_vs_lowmid``, ``I_Brier``;
- candidate episode counts frozen to [300, 400, 500, 600, 700, 800];
- a draw "passes" when ``estimate - Z_95 * SE_n > 0`` (same rule as the Stage 1
  Plus power curve); NaN draws count as failures;
- each marginal targets power 0.90; the AND chain is sized by its weakest metric;
- the FIRST candidate reaching 0.90 on all five is frozen; if none does, the
  status is ``REPLACEMENT_DEV_POWER_INFEASIBLE`` and no collection artifacts are
  written (hard stop, back to user review);
- the Monte Carlo seed is frozen to 20260909.

Outputs (all under ``<run-dir>/``): ``power/power_report.json`` first, then — only
when feasible — ``replacement_dev/replacement_dev_config.json`` and
``replacement_dev/replacement_dev_seed_schedule.json``, hash-chained to the power
report. Existing artifacts are never overwritten.

This is a manually-started task with a 30-second progress heartbeat.
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
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_data as s3d  # noqa: E402
from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402
from ac_twin.progress import ProgressReporter  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402
from ac_twin.stage1_plus_statistics import Z_95  # noqa: E402
from ac_twin.stage2_protocol import make_seed_schedule, validate_seed_schedule  # noqa: E402
from ac_twin.stage3_routing import PREDICTOR_SEEDS  # noqa: E402

REPAIR_PLAN_ID = s3d.REPAIR_PLAN_ID
POWER_STATISTIC_NAMES = (
    "i_mae", "spearman", "i_pr_presence", "i_pr_high_vs_lowmid", "i_brier",
)
DEFAULT_MONTE_CARLO_REPLICATES = 2000
INVALIDATION_RECORD_NAME = "INVALIDATION_RECORD_20260909.json"

DEFAULT_OLD_RUN_DIR = "experiments/stage3_routing/stage3a"
DEFAULT_RUN_DIR = "experiments/stage3_routing/stage3a_retrain_v1"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


# --- power statistics over the old split-2 pool -------------------------------------


def build_power_statistics(
    *,
    model_r_hat: np.ndarray,
    model_probs: np.ndarray,
    r_true: np.ndarray,
    regime_labels: np.ndarray,
    valid_mask: np.ndarray,
    baseline_r_hat: np.ndarray,
    baseline_probs_presence: np.ndarray,
    baseline_probs_high: np.ndarray,
    baseline_probs_brier: np.ndarray,
) -> dict[str, Callable[[np.ndarray], float]]:
    """The five episode-clustered improvement statistics, built EXACTLY as the
    gate builds them (same makers, same masks, same sign conventions), but as
    index-resampleable functions for the power Monte Carlo.

    The model arrays are the five-seed ensemble; the baseline arrays are the
    per-metric strongest baselines frozen on validation (never re-picked here).
    """
    from ac_twin.stage3a_gate import (  # local import: keeps module torch-free
        _HIGH,
        _PRESENCE_POSITIVE,
        binary_labels,
        high_vs_lowmid_scores,
        presence_scores,
    )
    from ac_twin.stage3_statistics import (
        mae_per_episode,
        make_brier_difference_statistic,
        make_paired_contrast_statistic,
        make_pooled_spearman_statistic,
        make_pr_auc_difference_statistic,
    )

    model_r = np.asarray(model_r_hat, dtype=np.float64)
    base_r = np.asarray(baseline_r_hat, dtype=np.float64)
    r_true = np.asarray(r_true, dtype=np.float64)
    labels = np.asarray(regime_labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)

    model_mae_ep = mae_per_episode(model_r, r_true, valid)
    base_mae_ep = mae_per_episode(base_r, r_true, valid)

    presence_lab = binary_labels(labels, _PRESENCE_POSITIVE)
    pos_mask = valid & (presence_lab == 1)
    high_lab = binary_labels(labels, (_HIGH,))

    return {
        "i_mae": make_paired_contrast_statistic(base_mae_ep, model_mae_ep),
        "spearman": make_pooled_spearman_statistic(model_r, r_true, valid),
        "i_pr_presence": make_pr_auc_difference_statistic(
            presence_scores(model_probs), presence_lab,
            presence_scores(baseline_probs_presence), presence_lab, valid,
        ),
        "i_pr_high_vs_lowmid": make_pr_auc_difference_statistic(
            high_vs_lowmid_scores(model_probs), high_lab,
            high_vs_lowmid_scores(baseline_probs_high), high_lab, pos_mask,
        ),
        "i_brier": make_brier_difference_statistic(
            model_probs, baseline_probs_brier, labels, valid,
        ),
    }


def select_episode_count(
    per_candidate: Mapping[int, Mapping[str, Any]],
    *,
    candidates: tuple[int, ...] = s3d.REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS,
    target_power: float = s3d.REPLACEMENT_DEV_TARGET_POWER,
) -> int | None:
    """First candidate whose FIVE marginal powers all reach the target; else None."""
    for n in candidates:
        entry = per_candidate.get(n, per_candidate.get(str(n)))
        if entry is None:
            raise ValueError(f"per_candidate is missing candidate n={n}")
        marginal = entry["marginal_power"]
        if all(float(marginal[name]) >= target_power for name in POWER_STATISTIC_NAMES):
            return int(n)
    return None


def estimate_replacement_dev_power(
    statistics: Mapping[str, Callable[[np.ndarray], float]],
    *,
    pool_size: int,
    candidates: tuple[int, ...] | list[int] = s3d.REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS,
    monte_carlo_replicates: int = DEFAULT_MONTE_CARLO_REPLICATES,
    seed: int = s3d.REPLACEMENT_DEV_POWER_SEED,
    target_power: float = s3d.REPLACEMENT_DEV_TARGET_POWER,
    progress_prefix: str = "3A-power",
) -> dict[str, Any]:
    """Monte-Carlo episode-resampling marginal power for the five gate metrics.

    For each candidate size n, ``monte_carlo_replicates`` draws of n episodes are
    sampled with replacement from the pool and every statistic is evaluated on
    the draw. A draw passes when ``statistic - Z_95 * SE_n > 0`` where ``SE_n``
    is the cross-draw standard deviation at size n; non-finite draws fail. The
    marginal power is the passing fraction; the weakest metric sizes the chain.
    """
    candidates = tuple(int(n) for n in candidates)
    if candidates != tuple(s3d.REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS):
        raise ValueError(
            f"candidate episode counts are frozen to "
            f"{tuple(s3d.REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS)}, got {candidates}"
        )
    if int(seed) != int(s3d.REPLACEMENT_DEV_POWER_SEED):
        raise ValueError(f"power seed is frozen to {s3d.REPLACEMENT_DEV_POWER_SEED}")
    names = set(statistics)
    missing = [k for k in POWER_STATISTIC_NAMES if k not in names]
    extra = sorted(names - set(POWER_STATISTIC_NAMES))
    if missing or extra:
        raise ValueError(
            f"power statistics must be exactly {list(POWER_STATISTIC_NAMES)}; "
            f"missing={missing} extra={extra}"
        )
    pool = int(pool_size)
    if pool < 2:
        raise ValueError("pool_size must be >= 2")
    replicates = int(monte_carlo_replicates)
    if replicates < 50:
        raise ValueError("monte_carlo_replicates must be >= 50")

    rng = np.random.default_rng(int(seed))
    reporter = ProgressReporter(
        total=len(candidates), every=1, prefix=progress_prefix, heartbeat_seconds=30.0)
    per_candidate: dict[int, dict[str, Any]] = {}
    for done, n in enumerate(candidates, start=1):
        started = time.perf_counter()
        draws = {name: np.empty(replicates, dtype=np.float64) for name in POWER_STATISTIC_NAMES}
        for r in range(replicates):
            indices = rng.integers(0, pool, size=n)
            for name in POWER_STATISTIC_NAMES:
                draws[name][r] = statistics[name](indices)
        entry: dict[str, Any] = {"marginal_power": {}, "se": {}}
        for name in POWER_STATISTIC_NAMES:
            values = draws[name]
            finite = np.isfinite(values)
            se = float(np.std(values[finite], ddof=1)) if finite.sum() >= 2 else float("nan")
            passed = finite & ((values - Z_95 * se) > 0.0)
            entry["se"][name] = se
            entry["marginal_power"][name] = float(np.mean(passed))
        weakest = min(POWER_STATISTIC_NAMES, key=lambda k: entry["marginal_power"][k])
        entry["weakest"] = {
            "condition": weakest,
            "power": entry["marginal_power"][weakest],
        }
        per_candidate[n] = entry
        reporter.update(
            done,
            details=(f"n={n} weakest={weakest} "
                     f"power={entry['marginal_power'][weakest]:.3f} "
                     f"candidate_seconds={time.perf_counter() - started:.1f}"),
        )

    selected = select_episode_count(
        per_candidate, candidates=candidates, target_power=target_power)
    return {
        "per_candidate": per_candidate,
        "selected_episode_count": selected,
        "status": "POWER_FEASIBLE" if selected is not None else s3d.REPLACEMENT_DEV_INFEASIBLE,
        "candidates": list(candidates),
        "target_power": float(target_power),
        "monte_carlo_replicates": replicates,
        "seed": int(seed),
        "z": float(Z_95),
        "pool_size": pool,
        "rule": "pass = statistic - Z_95 * SE_n > 0; marginal power = passing fraction; "
                "weakest of the five metrics sizes the AND chain; NaN draws fail",
    }


# --- frozen collection artifacts -----------------------------------------------------


def replacement_dev_paths(run_dir: str | Path, *, data_root: str | Path | None = None) -> dict[str, Path]:
    """Canonical locations of every replacement-holdout artifact.

    Paths are ROOT-relative by default (matching the repo's config convention);
    tests pass an absolute ``data_root`` and ``run_dir`` to redirect into tmp.
    """
    run_dir = Path(run_dir)
    data_root = Path("data") if data_root is None else Path(data_root)
    repl_dir = run_dir / "replacement_dev"
    data_dir = data_root / "object_centric_stage" / "stage3a_replacement_dev"
    return {
        "power_report_path": run_dir / "power" / "power_report.json",
        "config_path": repl_dir / "replacement_dev_config.json",
        "seed_schedule_path": repl_dir / "replacement_dev_seed_schedule.json",
        "output_dir": repl_dir / "collection",
        "data_output": data_dir / "object_twinworld_stage3a_replacement_dev.npz",
        "rejection_ledger": data_dir / "rejection_ledger.jsonl",
        "collection_receipt": repl_dir / "replacement_dev_collection_receipt.json",
        "historical_data_root": data_root,
    }


def freeze_replacement_artifacts(
    run_dir: str | Path,
    *,
    selected_episode_count: int | None,
    power_report_sha256: str,
    historical_seeds: np.ndarray,
    collection_seed: int = s3d.REPLACEMENT_DEV_COLLECTION_SEED,
    paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Freeze the collection config + seed schedule after a feasible power run.

    Refuses (hard stop) when the power run was infeasible, when any artifact
    already exists, or when the fresh seed schedule overlaps the historical
    episode-seed universe (Stage 1, Stage 1 Plus, Stage 2, old Stage 3 dev,
    pilot, replication — every manifest under ``data/``).
    """
    if selected_episode_count is None:
        raise RuntimeError(
            f"{s3d.REPLACEMENT_DEV_INFEASIBLE}: the weakest metric never reaches "
            f"power {s3d.REPLACEMENT_DEV_TARGET_POWER} within "
            f"{list(s3d.REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS)}; "
            "stop and return to user review — do not relax the gate"
        )
    n_pairs = int(selected_episode_count)
    run_dir = Path(run_dir)
    paths = dict(paths) if paths is not None else replacement_dev_paths(run_dir)
    # The frozen count must be the one the power report carries; the estimator is
    # the only place the frozen candidate list is enforced, and the report is its
    # tamper-evident carrier.
    report_path = _resolve(paths["power_report_path"])
    if not report_path.is_file():
        raise ValueError(f"power report missing: {report_path}")
    if sha256_file(report_path) != str(power_report_sha256):
        raise ValueError("power report sha256 mismatch — refusing to freeze against "
                         "an edited power report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") == s3d.REPLACEMENT_DEV_INFEASIBLE:
        raise RuntimeError(
            f"{s3d.REPLACEMENT_DEV_INFEASIBLE}: the power report is infeasible; "
            "stop and return to user review")
    if int(report.get("selected_episode_count", -1)) != n_pairs:
        raise ValueError(
            f"selected episode count {n_pairs} does not match the power report's "
            f"{report.get('selected_episode_count')}")
    for key in ("config_path", "seed_schedule_path"):
        if paths[key].exists():
            raise FileExistsError(
                f"frozen replacement artifact already exists: {paths[key]}; "
                "the freeze is single-shot and never overwritten"
            )

    contract = s3d.MERGE_GENERATION_CONTRACT
    max_attempts = n_pairs * int(contract["max_attempt_multiplier"])
    schedule = make_seed_schedule(collection_seed, size=max_attempts)
    validate_seed_schedule(schedule, np.asarray(historical_seeds, dtype=np.int64))
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()

    config = {
        "experiment_name": "stage3a_replacement_dev",
        "analysis_status": s3d.REPLACEMENT_DEV_PURPOSE,
        "purpose": s3d.REPLACEMENT_DEV_PURPOSE,
        "plan_id": s3d.REPAIR_PLAN_ID,
        "scene": contract["scene"],
        "env_id": contract["env_id"],
        "n_pairs": n_pairs,
        "collection_seed": int(collection_seed),
        "seed_schedule_size": max_attempts,
        "max_attempts": max_attempts,
        "horizon": contract["horizon"],
        "history_len": contract["history_len"],
        "t_star_low": contract["t_star_low"],
        "t_star_high": contract["t_star_high"],
        "intervention_durations": list(contract["intervention_durations"]),
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
        "purpose": s3d.REPLACEMENT_DEV_PURPOSE,
        "plan_id": s3d.REPAIR_PLAN_ID,
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
        "config": config,
        "schedule": schedule,
    }


# --- forensic pool from the OLD (invalidated) run, with repaired alignment ---------


def build_forensic_power_pool(
    old_run_dir: str | Path,
    source: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Five-seed ensemble predictions on old split 2 with the REPAIRED alignment.

    Sanctioned forensic use (v0.1 §0.3/§3.2): the invalidated run's config,
    checkpoints, scaler and frozen baselines are loaded READ-ONLY; inference goes
    through the repaired unshuffled ``_predict_split`` with the hard episode-order
    assertion. Nothing here may feed back into model, threshold or gate choices.
    """
    import torch  # local: the pure-numpy power path stays torch-free for tests

    import run_stage3a as drv
    from ac_twin.stage3_response import Stage3ResponsePredictor, fit_temperature, softmax

    old_run_dir = _resolve(old_run_dir)
    record = old_run_dir / INVALIDATION_RECORD_NAME
    if not record.is_file():
        raise RuntimeError(
            f"old run dir must carry {INVALIDATION_RECORD_NAME} "
            "(the forensic pool may only be built from the invalidated run)"
        )
    selection = json.loads((old_run_dir / "final_selection.json").read_text(encoding="utf-8"))
    cfg = selection["final_config"]
    strongest = selection["strongest_baselines"]
    signatures = list(selection["checkpoint_signatures"])
    finalist = selection["finalist_name"]

    bundle = drv.load_development_tensors(_resolve(source))
    scaler = drv.FeatureScaler.load(old_run_dir / "scaler.json")
    scaled = drv.scale_tensors(bundle["tensors"], scaler)
    tensors, roles = bundle["tensors"], bundle["roles"]
    test_idx = roles["development_test"]
    val_idx = roles["validation"]

    r_hat_seeds, logit_seeds, val_logit_seeds = [], [], []
    for position, seed in enumerate(PREDICTOR_SEEDS):
        state = s3t.load_predictor_state(
            old_run_dir / f"{finalist}_seed{seed}.pt",
            expected_signature=signatures[position],
        )
        model = Stage3ResponsePredictor(
            hidden_dim=int(cfg["hidden_dim"]),
            response_feature_dim=int(cfg["response_feature_dim"]),
        )
        model.load_state_dict(state)
        ev = drv._predict_split(model, scaled, test_idx, device)
        drv._assert_episode_order(ev, test_idx)
        vp = drv._predict_split(model, scaled, val_idx, device)
        drv._assert_episode_order(vp, val_idx)
        r_hat_seeds.append(ev["r_hat"])
        logit_seeds.append(ev["regime_logits"])
        val_logit_seeds.append(vp["regime_logits"])
        print(f"[3A-power] forensic inference seed {seed} done", flush=True)

    val_logits = np.stack(val_logit_seeds, axis=0).mean(axis=0)
    temperature = fit_temperature(
        val_logits,
        tensors["regime_labels"][val_idx].numpy(),
        tensors["valid"][val_idx].numpy(),
    )
    model_r = np.stack(r_hat_seeds, axis=0).mean(axis=0)
    model_probs = softmax(np.stack(logit_seeds, axis=0).mean(axis=0) / float(temperature), axis=-1)

    cont, clf = drv._build_baselines(
        tensors, bundle["raw_factual_suffix"], roles["train"],
        tensors, bundle["raw_factual_suffix"], test_idx,
    )

    return {
        "model_r_hat": model_r,
        "model_probs": model_probs,
        "r_true": tensors["r_true"][test_idx].numpy(),
        "regime_labels": tensors["regime_labels"][test_idx].numpy(),
        "valid_mask": tensors["valid"][test_idx].numpy(),
        "baseline_r_hat": np.asarray(cont[strongest["strongest_continuous"]], dtype=np.float64),
        "baseline_probs_presence": clf[strongest["strongest_presence"]],
        "baseline_probs_high": clf[strongest["strongest_high_vs_lowmid"]],
        "baseline_probs_brier": clf[strongest["strongest_brier"]],
        "n_episodes": int(len(test_idx)),
        "temperature": float(temperature),
        "input_summary": {
            "source": str(source),
            "source_sha256": sha256_file(_resolve(source)),
            "old_run_dir": str(old_run_dir),
            "old_final_selection_sha256": sha256_file(old_run_dir / "final_selection.json"),
            "old_scaler_sha256": sha256_file(old_run_dir / "scaler.json"),
            "old_checkpoint_sha256": {
                f"{finalist}_seed{seed}.pt": sha256_file(old_run_dir / f"{finalist}_seed{seed}.pt")
                for seed in PREDICTOR_SEEDS
            },
            "old_checkpoint_signatures": signatures,
            "invalidation_record_sha256": sha256_file(record),
            "final_config": dict(cfg),
            "strongest_baselines": dict(strongest),
        },
    }


# --- main (manually started) ---------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3A replacement dev power sizing")
    parser.add_argument("--config", default="experiments/stage3_routing/stage3a_config.json")
    parser.add_argument("--old-run-dir", default=DEFAULT_OLD_RUN_DIR)
    parser.add_argument("--source", default="data/object_centric_stage/merge_2000/object_twinworld_2000.npz")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--replicates", type=int, default=DEFAULT_MONTE_CARLO_REPLICATES)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = _resolve(args.run_dir)
    paths = replacement_dev_paths(args.run_dir)
    report_path = _resolve(paths["power_report_path"])
    if report_path.exists():
        print(f"[3A-power] REFUSED: {report_path} already exists; the power freeze "
              "is single-shot", flush=True)
        return 3

    pool = build_forensic_power_pool(args.old_run_dir, args.source, device=args.device)
    statistics = build_power_statistics(
        model_r_hat=pool["model_r_hat"], model_probs=pool["model_probs"],
        r_true=pool["r_true"], regime_labels=pool["regime_labels"],
        valid_mask=pool["valid_mask"], baseline_r_hat=pool["baseline_r_hat"],
        baseline_probs_presence=pool["baseline_probs_presence"],
        baseline_probs_high=pool["baseline_probs_high"],
        baseline_probs_brier=pool["baseline_probs_brier"],
    )
    result = estimate_replacement_dev_power(
        statistics, pool_size=pool["n_episodes"], monte_carlo_replicates=args.replicates)

    report = {
        "plan_id": s3d.REPAIR_PLAN_ID,
        "purpose": s3d.REPLACEMENT_DEV_PURPOSE,
        "stage": "3A-replacement-power",
        "forensic_use_only": True,
        **result,
        "temperature": pool["temperature"],
        "input_summary": pool["input_summary"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_write_json(report_path, report)
    report_sha = sha256_file(report_path)
    print(f"[3A-power] status={result['status']} selected={result['selected_episode_count']} "
          f"-> {report_path}", flush=True)

    if result["selected_episode_count"] is None:
        print(f"[3A-power] {s3d.REPLACEMENT_DEV_INFEASIBLE}: no candidate in "
              f"{list(s3d.REPLACEMENT_DEV_CANDIDATE_EPISODE_COUNTS)} reaches power "
              f"{s3d.REPLACEMENT_DEV_TARGET_POWER} on all five metrics; stop and "
              "return to user review", flush=True)
        return 2

    historical = s3d.scan_all_episode_seeds(ROOT / "data")
    artifacts = freeze_replacement_artifacts(
        args.run_dir,
        selected_episode_count=result["selected_episode_count"],
        power_report_sha256=report_sha,
        historical_seeds=historical,
    )
    print(f"[3A-power] froze n={result['selected_episode_count']} -> "
          f"{artifacts['config_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
