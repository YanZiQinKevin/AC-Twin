"""Train Stage 2B intersection models and adjudicate Gate S on development.

Only the fresh v0.4 train/validation dataset is accepted.  This entry point
never opens a confirm dataset and never computes Gate A/B.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.checkpointing import canonical_fingerprint
from ac_twin.progress import format_duration
from ac_twin.stage1_data import OBJECT_FEATURES, sha256_file
from ac_twin.stage1_model import PerAgentPairedEffectGRU
from ac_twin.stage2b_data import validate_stage2b_source
from ac_twin import stage2b_training as st
from run_stage1_evidence_value import (
    _denormalize_effect,
    _model_inputs,
    _predict,
    build_split_derangement,
    train_stage1_run,
)

CONDITIONS = ("no_suffix", "uniform_suffix", "shuffled_suffix")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _signature(config: dict, condition: str, seed: int, source_sha: str,
               donor_sha: str) -> str:
    return canonical_fingerprint({
        "config": config, "condition": condition, "seed": seed,
        "source_sha256": source_sha, "validation_donor_sha256": donor_sha,
        "implementation": {
            "runner": sha256_file(Path(__file__)),
            "training": sha256_file(ROOT / "code/ac_twin/stage2b_training.py"),
            "data": sha256_file(ROOT / "code/ac_twin/stage1_data.py"),
            "statistics": sha256_file(ROOT / "code/ac_twin/stage1_statistics.py"),
            "model": sha256_file(ROOT / "code/ac_twin/stage1_model.py"),
            "stage1_training": sha256_file(ROOT / "code/run_stage1_evidence_value.py"),
        },
    })


def _suffix_swap(model, arrays: dict, validation_donor: np.ndarray,
                 batch_size: int, device: torch.device, sealed_norm: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    val_idx = arrays["indices"]["validation"]
    model.eval()
    with torch.no_grad():
        correct_inputs = _model_inputs(arrays, val_idx, "uniform_suffix", None)
        correct_norm = _predict(model, correct_inputs, batch_size, device)
        max_diff = float(np.max(np.abs(correct_norm - sealed_norm)))
        if not np.allclose(correct_norm, sealed_norm, atol=1e-4, rtol=1e-3):
            raise RuntimeError(f"M1 correct-suffix replay mismatch: {max_diff:.3e}")
        wrong_inputs = _model_inputs(
            arrays, val_idx, "shuffled_suffix", validation_donor)
        wrong_norm = _predict(model, wrong_inputs, batch_size, device)
    if not np.isfinite(wrong_norm).all():
        raise RuntimeError("M1 swapped-suffix prediction is non-finite")
    return correct_norm, wrong_norm, max_diff


def run(config_path: Path, *, device_name: str | None, resume: bool) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    st.validate_training_config(config)
    source = _resolve(config["source_data"])
    receipt_path = _resolve(config["source_receipt"])
    if not source.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("v0.4 development dataset or receipt is missing")
    source_sha = sha256_file(source)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("data_output_sha256") != source_sha or receipt.get("partial", True):
        raise ValueError("development source does not match its completed receipt")
    with np.load(source, allow_pickle=True) as archive:
        data = {name: archive[name] for name in archive.files}
    validate_stage2b_source(data, expected_split_labels={0, 1},
                            expected_episodes=6000)
    arrays, thresholds = st.prepare_development_arrays(
        data, suffix_steps=int(config["factual_suffix_steps"]))
    indices = arrays["indices"]
    train_indices = indices["train"]
    val_indices = indices["validation"]
    # train_stage1_run asks for a test view; alias validation only for its
    # second read-only prediction return and discard it. Gate S reads val only.
    train_indices_contract = {"train": train_indices,
                              "validation": val_indices, "test": val_indices}
    validation_donor, validation_manifest = build_split_derangement(
        arrays, val_indices, 1, int(config["derangement"]["validation_seed"]))
    donor_sha = canonical_fingerprint(validation_manifest)
    train_base_seed = int(config["derangement"]["train_base_seed"])

    output = _resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    model_config = dict(config["training"])
    model_config.update(config["architecture"])

    np.savez_compressed(
        output / "normalization.npz", **arrays["normalizer"],
        target_mean=arrays["target_mean"], target_std=arrays["target_std"],
        train_indices=train_indices, validation_indices=val_indices,
        feature_names=np.asarray(list(OBJECT_FEATURES)),
        source_data_sha256=np.asarray(source_sha))
    _write_json(output / "response_bins.json", dict(thresholds))
    _write_json(output / "validation_manifest.json", validation_manifest)

    def train_donor(epoch: int) -> np.ndarray:
        donor, _ = build_split_derangement(
            arrays, train_indices, 0, train_base_seed + int(epoch))
        return donor

    predictions = {name: [] for name in CONDITIONS}
    total_jobs = len(CONDITIONS) * len(config["training_seeds"])
    overall_start = time.perf_counter()
    completed_jobs = 0
    for condition in CONDITIONS:
        for seed in config["training_seeds"]:
            stem = f"{condition}_seed{seed}"
            files = {
                "model": output / f"{stem}.pt",
                "log": output / f"{stem}_training.json",
                "val": output / f"{stem}_val_prediction.npy",
                "meta": output / f"{stem}_run.json",
                "swap": output / f"{stem}_val_swapped_suffix.npy",
            }
            signature = _signature(config, condition, int(seed), source_sha, donor_sha)
            complete = False
            if files["meta"].exists():
                metadata = json.loads(files["meta"].read_text(encoding="utf-8"))
                if metadata.get("signature") != signature:
                    raise ValueError(f"run signature drift for {stem}")
                complete = metadata.get("status") == "complete"
                if complete and not all(files[k].is_file() for k in ("model", "log", "val")):
                    raise FileNotFoundError(f"completed run missing artifacts: {stem}")
            elif any(files[k].exists() for k in ("model", "log", "val")):
                raise ValueError(f"unsigned partial artifacts for {stem}")
            if complete and not resume:
                raise FileExistsError(f"run already complete: {stem}; use --resume")

            model = PerAgentPairedEffectGRU(
                feature_dim=len(OBJECT_FEATURES), action_dim=5,
                object_embedding_dim=int(config["architecture"]["object_embedding_dim"]),
                hidden_dim=int(config["architecture"]["hidden_dim"]),
                action_embedding_dim=int(config["architecture"]["action_embedding_dim"]))
            parameters = sum(p.numel() for p in model.parameters())
            prefix = (f"[Stage2B {config['variant']}] job={completed_jobs + 1}/{total_jobs} "
                      f"condition={condition} seed={seed}")
            if complete:
                model.load_state_dict(torch.load(files["model"], map_location="cpu",
                                                 weights_only=True))
                model = model.to(device)
                val_norm = np.load(files["val"], allow_pickle=False)
                log = json.loads(files["log"].read_text(encoding="utf-8"))
                print(f"{prefix} resumed=complete", flush=True)
            else:
                _write_json(files["meta"], {"signature": signature, "status": "running"})
                job_start = time.perf_counter()

                def report_epoch(record: dict) -> None:
                    epoch = int(record["epoch"])
                    elapsed = time.perf_counter() - job_start
                    eta = elapsed * max(int(record["max_epochs"]) - epoch, 0) / max(epoch, 1)
                    print(f"{prefix} epoch={epoch}/{record['max_epochs']} "
                          f"train={record['train_mse']:.6f} val={record['val_mse']:.6f} "
                          f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                          flush=True)

                _seed_all(int(seed))
                model, val_norm, _discard, log = train_stage1_run(
                    model, arrays, train_indices_contract, condition, train_donor,
                    validation_donor, validation_donor, model_config, int(seed),
                    device, progress_callback=report_epoch)
                best = min(log, key=lambda row: row["val_mse"])
                torch.save(model.state_dict(), files["model"])
                _write_json(files["log"], log)
                np.save(files["val"], val_norm, allow_pickle=False)
                _write_json(files["meta"], {
                    "signature": signature, "status": "complete", "seed": int(seed),
                    "parameters": int(parameters), "epochs_run": len(log),
                    "best_epoch": int(best["epoch"]),
                    "best_val_mse": float(best["val_mse"]),
                    "runtime_seconds": time.perf_counter() - job_start,
                })

            entry = {"val": _denormalize_effect(val_norm, arrays),
                     "log": log, "parameters": parameters}
            if condition == "uniform_suffix":
                if files["swap"].is_file():
                    wrong_norm = np.load(files["swap"], allow_pickle=False)
                    correct_norm = val_norm
                else:
                    correct_norm, wrong_norm, replay_diff = _suffix_swap(
                        model, arrays, validation_donor,
                        int(model_config["batch_size"]), device, val_norm)
                    np.save(files["swap"], wrong_norm, allow_pickle=False)
                    print(f"{prefix} suffix_replay_max_abs_diff={replay_diff:.3e}", flush=True)
                entry["val_suffix_swap_correct"] = _denormalize_effect(correct_norm, arrays)
                entry["val_suffix_swap_shuffled"] = _denormalize_effect(wrong_norm, arrays)
            predictions[condition].append(entry)
            completed_jobs += 1
            elapsed_total = time.perf_counter() - overall_start
            eta_total = elapsed_total * (total_jobs - completed_jobs) / completed_jobs
            print(f"{prefix} complete overall={completed_jobs}/{total_jobs} "
                  f"percent={100*completed_jobs/total_jobs:.1f}% "
                  f"elapsed={format_duration(elapsed_total)} eta={format_duration(eta_total)}",
                  flush=True)

    report = st.evaluate_gate_s(
        arrays, predictions, thresholds,
        n_resamples=int(config["statistics"]["bootstrap_resamples"]),
        confidence=float(config["statistics"]["confidence_level"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]))
    report.update({
        "plan_id": config["plan_id"], "variant": config["variant"],
        "source_data": str(source), "source_data_sha256": source_sha,
        "validation_manifest_sha256": donor_sha,
        "normalization_sha256": sha256_file(output / "normalization.npz"),
        "response_bins_sha256": sha256_file(output / "response_bins.json"),
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__)),
            "training": sha256_file(ROOT / "code/ac_twin/stage2b_training.py"),
            "data": sha256_file(ROOT / "code/ac_twin/stage1_data.py"),
            "statistics": sha256_file(ROOT / "code/ac_twin/stage1_statistics.py"),
            "model": sha256_file(ROOT / "code/ac_twin/stage1_model.py"),
            "stage1_training": sha256_file(ROOT / "code/run_stage1_evidence_value.py"),
        },
        "training_seeds": config["training_seeds"], "device": str(device),
        "n_train": len(train_indices), "n_validation": len(val_indices),
        "response_bins": thresholds,
    })
    gate_path = output / "gate_s_report.json"
    if gate_path.exists():
        raise FileExistsError(f"Gate S report is create-once: {gate_path}")
    _write_json(gate_path, report)
    print(f"[Stage2B {config['variant']}] GateS={report['status']} "
          f"elapsed={format_duration(time.perf_counter()-overall_start)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(_resolve(args.config), device_name=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
