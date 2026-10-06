"""Stage 3A Plus compute-budget benchmark (plan Task 11).

Measures the four cost drivers on THIS machine and prints a JSON block used to
author ``experiments/stage3_routing/stage3a_plus/COMPUTE_BUDGET.md``:

1. training: one epoch on 12 development-train episodes (real pipeline code,
   final architecture h96/r32, plan device);
2. inference: one seed over a 50-episode eval slice;
3. gate bootstrap: ``decompose_seed_metrics`` at replacement-holdout scale with
   50 resamples (linear in resamples; the formal gate uses 10,000);
4. collection: real ``merge-v0`` attempts through the Stage 2 engine rollout.

Run: ``python code/benchmark_stage3a_plus_budget.py`` (CUDA per dev_config).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import run_stage3a as s3a  # noqa: E402
import run_stage3a_plus as drv  # noqa: E402
from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin import stage3a_plus_statistics as sps  # noqa: E402
from ac_twin.stage3_response import (  # noqa: E402
    Stage3ResponsePredictor,
    compute_class_weights,
    response_masked_loss,
    softmax,
)


def _bench_training(config, scaled, roles, device) -> dict:
    train_idx = np.asarray(roles["train"])[:12]
    val_idx = np.asarray(roles["validation"])[:12]
    cfg, tr = config["final_config"], config["training"]
    s3t.seed_all(42)
    class_weights = torch_weights = compute_class_weights(
        scaled["regime_labels"][roles["train"]].numpy(),
        scaled["valid"][roles["train"]].numpy())
    import torch
    weights = torch.as_tensor(class_weights)

    def loss_fn(model, batch):
        out = model(
            batch["history"], batch["factual_suffix"], batch["factual_actions"],
            batch["counterfactual_actions"], batch["cf_action"], batch["cf_duration"],
            batch["t_star"])
        return response_masked_loss(
            out, batch["r_true"], batch["regime_labels"], batch["valid"],
            weights.to(out["regime_logits"].device), lambda_cls=float(cfg["lambda_cls"]),
            lambda_brier=0.10, label_smoothing=0.0)

    sums_fn = s3a._response_sums_fn(weights)
    batches = s3a.make_batches(scaled, train_idx, int(tr["batch_size"]), seed=42)
    val_batches = [b for _, b in s3a.make_eval_batches(scaled, val_idx, int(tr["batch_size"]))]
    model = Stage3ResponsePredictor(hidden_dim=int(cfg["hidden_dim"]),
                                    response_feature_dim=int(cfg["response_feature_dim"]))
    start = time.perf_counter()
    result = s3t.train_model(
        model, loss_fn, batches, val_batches, seed=42, max_epochs=1,
        early_stopping_patience=1, learning_rate=float(tr["learning_rate"]),
        device=device, weight_decay=0.0, progress_prefix="budget-bench",
        selection_fn=s3a.selection_score, val_sums_fn=sums_fn)
    elapsed = time.perf_counter() - start
    return {"episodes": 12, "epochs": 1, "seconds": elapsed,
            "seconds_per_epoch_per_episode": elapsed / 12.0,
            "train_split_size": int(len(roles["train"])),
            "validation_split_size": int(len(roles["validation"]))}


def _bench_inference(config, scaled, roles, device) -> dict:
    import torch
    cfg = config["final_config"]
    eval_idx = np.asarray(roles["development_test"])[:50]
    torch.manual_seed(0)
    model = Stage3ResponsePredictor(hidden_dim=int(cfg["hidden_dim"]),
                                    response_feature_dim=int(cfg["response_feature_dim"]))
    start = time.perf_counter()
    s3a._predict_split(model, scaled, eval_idx, device)
    elapsed = time.perf_counter() - start
    return {"episodes": 50, "seconds": elapsed,
            "seconds_per_seed_per_episode": elapsed / 50.0,
            "dev_a_episodes": int(len(roles["development_test"]))}


def _bench_decompose(n_episodes: int, n_resamples: int = 50) -> dict:
    rng = np.random.default_rng(0)
    S, T, N, C = 10, 20, 5, 4
    E = n_episodes
    r_true = np.abs(rng.normal(0, 1, (E, T, N)))
    valid = rng.random((E, T, N)) > 0.2
    labels = rng.integers(0, C, (E, T, N))
    r_seeds = r_true[None] + rng.normal(0, 0.3, (S, E, T, N))
    logits = rng.normal(0, 1, (S, E, T, N, C))
    cal = np.stack([softmax(logits[s], axis=-1) for s in range(S)], axis=0)
    cont = np.repeat(r_true[valid].mean(), E * T * N).reshape(E, T, N)
    clf = np.zeros((E, T, N, C))
    for c in range(C):
        clf[..., c] = float((labels[valid] == c).mean())
    baselines = {"continuous": {"b": cont}, "classification": {"c": clf},
                 "strongest_continuous": "b", "strongest_brier": "c"}
    start = time.perf_counter()
    sps.decompose_seed_metrics(
        r_hat_seeds=r_seeds, logits_seeds=logits, calibrated_probs_seeds=cal,
        r_true=r_true, regime_labels=labels, valid_mask=valid,
        episode_ids=np.arange(E), horizons=drv._horizon_grid(valid),
        cf_actions=np.zeros(E, dtype=np.int64),
        cf_durations=np.ones(E, dtype=np.int64), baselines=baselines,
        n_resamples=n_resamples, bootstrap_seed=20260929, n_production_seeds=5)
    elapsed = time.perf_counter() - start
    return {"episodes": E, "resamples": n_resamples, "seconds": elapsed,
            "seconds_per_resample_at_E": elapsed / n_resamples}


def _bench_collection(n_attempts: int = 3) -> dict:
    import collect_stage2_holdout as engine
    config = {
        "env_id": "merge-v0", "horizon": 20, "history_len": 10,
        "t_star_low": 10, "t_star_high": 15, "vehicles_count": 25,
        "observed_vehicles_count": 5, "duration": 100,
        "simulation_frequency": 15, "policy_frequency": 5,
        "collection_seed": 20261009,
    }
    action_ids, action_probs, capacity = engine._real_probe(config)
    rng = np.random.default_rng(20261009)
    seconds = []
    for _ in range(n_attempts):
        seed = int(rng.integers(1, 2**31 - 1))
        t_star = int(rng.integers(config["t_star_low"], config["t_star_high"] + 1))
        prefix = rng.choice(action_ids, size=t_star, p=action_probs).astype(np.int64)
        factual = rng.choice(action_ids, size=config["horizon"], p=action_probs).astype(np.int64)
        cf = factual.copy()
        cf[:1] = (int(factual[0]) + 1) % len(action_ids)
        env_cfg = engine._env_config(config)
        start = time.perf_counter()
        engine.rollout_branch_real(config["env_id"], env_cfg, seed, prefix, factual, capacity)
        engine.rollout_branch_real(config["env_id"], env_cfg, seed, prefix, cf, capacity)
        seconds.append(time.perf_counter() - start)
    return {"attempts": n_attempts, "seconds_per_attempt": float(np.mean(seconds))}


def main() -> int:
    import torch  # noqa: F401
    config = drv.load_dev_config(ROOT / "experiments/stage3_routing/stage3a_plus/dev_config.json")
    device = config["device"] if torch.cuda.is_available() else "cpu"
    print(f"[budget-bench] device={device}", flush=True)

    bundle = s3a.load_development_tensors(ROOT / config["development_source"])
    scaler = s3a.prepare_scaler(bundle["tensors"], bundle["roles"],
                                ROOT / config["prior_run_dir"])
    scaled = s3a.scale_tensors(bundle["tensors"], scaler)
    with np.load(ROOT / config["replacement_source"], allow_pickle=True) as z:
        replacement_episodes = int(z["episode_seed"].shape[0])

    out = {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "replacement_episodes": replacement_episodes,
        "training": _bench_training(config, scaled, bundle["roles"], device),
        "inference": _bench_inference(config, scaled, bundle["roles"], device),
        "decompose": _bench_decompose(replacement_episodes),
        "collection": _bench_collection(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
