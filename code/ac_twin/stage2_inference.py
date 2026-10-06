"""Stage 2 frozen-checkpoint inference on the confirmatory holdout.

Loads the frozen Stage 1 feature/target normalizers and the 15 frozen
checkpoints, then produces, for the new holdout:

- 15 main predictions (M0/M1/M2 x 5 seeds; M2 uses the primary manifest donor);
- 5 fixed-M1 suffix-swap diagnostic predictions (correct vs primary-shuffled);
- 20 alternate-M2 donor sensitivity predictions (4 alternate manifests x 5 seeds).

This module performs no training of any kind and never fits normalization or
response bins. Checkpoint SHA256 are verified against the protocol-lock whitelist
before loading. All predictions are returned in world coordinates (de-normalized
with the frozen target statistics).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402

from ac_twin import stage2_protocol as sp  # noqa: E402
from ac_twin.stage1_data import (  # noqa: E402
    apply_feature_normalizer,
    apply_suffix_window,
    build_derangement,
    compute_anchors,
    compute_history_agent_known,
    compute_target_effect,
    compute_valid_mask,
    derangement_manifest,
    sha256_file,
    suffix_presence_mask,
    visible_count_signature,
)
from ac_twin.stage1_model import PerAgentPairedEffectGRU  # noqa: E402
from ac_twin import stage1_statistics as _s1  # noqa: E402
from ac_twin.checkpointing import canonical_fingerprint  # noqa: E402

CONDITIONS = sp.CONDITIONS
TRAINING_SEEDS = sp.TRAINING_SEEDS
DONOR_SEEDS = (sp.PRIMARY_DERANGEMENT_SEED, *sp.ALTERNATE_DONOR_SEEDS)
_CONFIRMATORY_SPLIT_LABEL = 2


# --- frozen artifact loading ---------------------------------------------------------


def load_frozen_normalizers(path: str | Path) -> dict[str, np.ndarray]:
    """Load (never refit) the frozen Stage 1 feature + target normalizers."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {
            "mean": np.asarray(archive["mean"], dtype=np.float32),
            "std": np.asarray(archive["std"], dtype=np.float32),
            "target_mean": np.asarray(archive["target_mean"], dtype=np.float32),
            "target_std": np.asarray(archive["target_std"], dtype=np.float32),
        }


def load_frozen_model(condition: str, seed: int, checkpoint_path: str | Path,
                      *, expected_sha256: str | None = None,
                      model_config: Mapping[str, Any] | None = None) -> PerAgentPairedEffectGRU:
    """Load one whitelisted frozen checkpoint in eval mode.

    ``condition`` must be one of the three frozen conditions and ``seed`` one of
    the five frozen training seeds. When ``expected_sha256`` is given, the
    checkpoint file hash must match exactly or loading fails.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} is not a frozen Stage 2 condition")
    if int(seed) not in TRAINING_SEEDS:
        raise ValueError(f"seed {seed} is not a frozen Stage 2 training seed")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint_path}")
    if expected_sha256 is not None:
        actual = sha256_file(checkpoint_path)
        if actual != expected_sha256:
            raise ValueError(
                f"checkpoint hash mismatch for {checkpoint_path.name}:\n"
                f"  expected {expected_sha256}\n  actual   {actual}")
    model_config = model_config or {}
    model = PerAgentPairedEffectGRU(
        object_embedding_dim=int(model_config.get("object_embedding_dim", 64)),
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        action_embedding_dim=int(model_config.get("action_embedding_dim", 8)))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# --- holdout preparation --------------------------------------------------------------


def prepare_holdout_arrays(data: Mapping[str, np.ndarray],
                           normalizer: Mapping[str, np.ndarray],
                           *, factual_suffix_steps: int = 10) -> dict[str, Any]:
    """Normalize the holdout with the frozen feature normalizer; build targets/masks.

    The counterfactual future is used only for the target/response/mask, never as
    a model input. Returns world-coordinate targets and frozen-normalized inputs.
    """
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    counterfactual = np.asarray(data["object_counterfactual_future"], dtype=np.float32)

    anchors = compute_anchors(history)
    feature_normalizer = {"mean": normalizer["mean"], "std": normalizer["std"]}
    norm_history = apply_feature_normalizer(history, anchors, feature_normalizer)
    norm_factual = apply_feature_normalizer(factual, anchors, feature_normalizer)
    windowed_factual = apply_suffix_window(norm_factual, factual_suffix_steps)
    suffix_mask = suffix_presence_mask(factual, factual_suffix_steps)

    target = compute_target_effect(factual, counterfactual)
    valid_mask = compute_valid_mask(history, factual, counterfactual)
    history_known = compute_history_agent_known(history)
    visible_count = visible_count_signature(history_known)
    r_pos, _ = _s1.response_magnitudes(counterfactual, factual)

    return {
        "norm_history": norm_history,
        "windowed_factual": windowed_factual,
        "suffix_mask": suffix_mask,
        "target": target,
        "valid_mask": valid_mask,
        "history_known": history_known,
        "visible_count": visible_count,
        "r_pos": r_pos,
        "factual_actions": np.asarray(data["factual_actions"], dtype=np.int64),
        "counterfactual_actions": np.asarray(data["counterfactual_actions"], dtype=np.int64),
        "cf_action": np.asarray(data["cf_action"]),
        "cf_duration": np.asarray(data["cf_duration"]),
        "episode_seed": np.asarray(data["episode_seed"]),
        "target_mean": np.asarray(normalizer["target_mean"], dtype=np.float64),
        "target_std": np.asarray(normalizer["target_std"], dtype=np.float64),
    }


def select_suffix(arrays: Mapping[str, Any], condition: str,
                  donor_local: np.ndarray | None) -> dict[str, np.ndarray]:
    """Select (suffix_features, suffix_mask) for a condition; shares all other inputs."""
    if condition == "no_suffix":
        return {"suffix": np.zeros_like(arrays["windowed_factual"]),
                "suffix_mask": np.zeros_like(arrays["suffix_mask"])}
    if condition == "uniform_suffix":
        return {"suffix": arrays["windowed_factual"], "suffix_mask": arrays["suffix_mask"]}
    if condition == "shuffled_suffix":
        if donor_local is None:
            raise ValueError("shuffled_suffix requires a derangement donor mapping")
        donor = np.asarray(donor_local, dtype=np.int64)
        return {"suffix": arrays["windowed_factual"][donor],
                "suffix_mask": arrays["suffix_mask"][donor]}
    raise ValueError(f"unknown condition {condition!r}")


def predict_holdout(model: PerAgentPairedEffectGRU, arrays: Mapping[str, Any],
                    condition: str, donor_index: np.ndarray | None, *,
                    device: str = "cpu", batch_size: int = 128) -> np.ndarray:
    """World-coordinate per-agent effect prediction (E,T,N,2) for one condition."""
    selected = select_suffix(arrays, condition, donor_index)
    model = model.to(device)
    model.eval()
    history = arrays["norm_history"]
    n = len(history)
    outputs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sel = slice(start, min(start + batch_size, n))
            pred = model(
                torch.from_numpy(history[sel]).float().to(device),
                torch.from_numpy(selected["suffix"][sel]).float().to(device),
                torch.from_numpy(arrays["factual_actions"][sel]).long().to(device),
                torch.from_numpy(arrays["counterfactual_actions"][sel]).long().to(device),
                suffix_mask=torch.from_numpy(selected["suffix_mask"][sel]).to(device),
            )
            outputs.append(pred.cpu().numpy())
    normalized = np.concatenate(outputs, axis=0)
    return normalized.astype(np.float64) * arrays["target_std"] + arrays["target_mean"]


# --- derangement manifest ----------------------------------------------------------------


def build_holdout_manifest(arrays: Mapping[str, Any], donor_seed: int) -> tuple[np.ndarray, dict]:
    """Stratified derangement over the holdout reusing Stage 1 ``build_derangement``.

    Hard conditions (no fixed point, donor bijection, distinct episode seeds) are
    enforced by ``build_derangement`` and re-checked here; the manifest records the
    donor seed, the derangement algorithm hash and the data hash that together fix
    the manifest identity.
    """
    cf_action = np.asarray(arrays["cf_action"])
    cf_duration = np.asarray(arrays["cf_duration"])
    factual_first_action = np.asarray(arrays["factual_actions"])[:, 0]
    visible_count = np.asarray(arrays["visible_count"])
    episode_seed = np.asarray(arrays["episode_seed"])
    donor, tier = build_derangement(cf_action, cf_duration, factual_first_action,
                                    visible_count, episode_seed, seed=int(donor_seed))
    n = len(donor)
    if (donor == np.arange(n)).any():
        raise RuntimeError("manifest produced a self match")
    if len(np.unique(donor)) != n:
        raise RuntimeError("manifest reused a donor")
    if (episode_seed[donor] == episode_seed).any():
        raise RuntimeError("manifest matched identical episode seeds")

    manifest = derangement_manifest(donor, tier, _CONFIRMATORY_SPLIT_LABEL,
                                    cf_action, cf_duration, visible_count, episode_seed)
    manifest["donor_seed"] = int(donor_seed)
    manifest["algorithm_sha256"] = sha256_file(ROOT / "code" / "ac_twin" / "stage1_data.py")
    manifest["data_sha256"] = canonical_fingerprint({
        "cf_action": cf_action.tolist(),
        "cf_duration": cf_duration.tolist(),
        "factual_first_action": factual_first_action.tolist(),
        "visible_count": visible_count.tolist(),
        "episode_seed": episode_seed.tolist(),
    })
    return donor, manifest


# --- frozen probe summaries (used by the protocol lock's params provider) ----------------


def compute_train_factual_summaries(config: Mapping[str, Any],
                                    *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Seed-mean factual summaries + active mask on the Stage 1 train split.

    Runs the five frozen M1 checkpoints' factual encoder over the Stage 1 train
    inputs (frozen feature normalizer + 10-step window). Read-only; no training.
    """
    data = np.load(sp._resolve(config["stage1_source_data"]), allow_pickle=True)
    normalization = np.load(sp._resolve(config["normalization_npz"]))
    train_idx = np.asarray(normalization["train_indices"], dtype=np.int64)
    normalizer = {"mean": normalization["mean"].astype(np.float32),
                  "std": normalization["std"].astype(np.float32)}
    history = np.asarray(data["object_history"], dtype=np.float32)
    factual = np.asarray(data["object_factual_future"], dtype=np.float32)
    anchors = compute_anchors(history)
    norm_factual = apply_feature_normalizer(factual, anchors, normalizer)
    windowed = apply_suffix_window(norm_factual, int(config["factual_suffix_steps"]))
    suffix_mask = suffix_presence_mask(factual, int(config["factual_suffix_steps"]))

    stage1_dir = sp._resolve(config["stage1_output_dir"])
    summaries, actives = [], []
    for seed in TRAINING_SEEDS:
        model = load_frozen_model("uniform_suffix", seed,
                                  stage1_dir / f"uniform_suffix_seed{seed}.pt")
        model = model.to(device)
        with torch.no_grad():
            summary, active = model.encode_factual_agents(
                torch.from_numpy(windowed[train_idx]).float().to(device),
                torch.from_numpy(suffix_mask[train_idx]).to(device))
        summaries.append(summary.cpu().numpy())
        actives.append(active.cpu().numpy())
    return np.stack(summaries).mean(axis=0), np.stack(actives).any(axis=0)


# --- orchestration (the runner's `infer` phase) ------------------------------------------


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array)
    Path(tmp).replace(path)


def expected_prediction_files(config: Mapping[str, Any]) -> dict[str, str]:
    """Map every prediction filename to its kind, for receipt verification."""
    files: dict[str, str] = {}
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            files[f"{condition}_seed{seed}_prediction.npy"] = "main"
    for seed in TRAINING_SEEDS:
        files[f"uniform_suffix_seed{seed}_primary_swap_prediction.npy"] = "m1_swap"
    for donor_seed in sp.ALTERNATE_DONOR_SEEDS:
        for seed in TRAINING_SEEDS:
            files[f"shuffled_suffix_seed{seed}_donor{donor_seed}_prediction.npy"] = "alt_donor"
    files["uniform_suffix_factual_summaries_seed_mean.npy"] = "probe_summary"
    files["uniform_suffix_factual_active.npy"] = "probe_summary"
    return files


def run_inference(config: Mapping[str, Any], *, device: str = "cpu",
                  output_dir: str | Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Produce all Stage 2 predictions + manifests + a prediction receipt.

    If a valid prediction receipt already exists and every artifact hash verifies,
    this verifies and returns without overwriting (no re-inference, no donor swap).
    """
    output_dir = Path(output_dir)
    receipt_path = output_dir / "prediction_receipt.json"
    expected_files = expected_prediction_files(config)

    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        ok = all((output_dir / name).is_file()
                 and sha256_file(output_dir / name) == receipt["prediction_sha256"].get(name)
                 for name in expected_files)
        ok = ok and all((output_dir / m).is_file()
                        and sha256_file(output_dir / m) == receipt["manifest_sha256"].get(m)
                        for m in receipt.get("manifest_sha256", {}))
        if ok:
            receipt["verification"] = "existing_predictions_verified_no_overwrite"
            return receipt
        raise RuntimeError(
            "prediction receipt exists but artifacts fail hash verification; "
            "refusing to overwrite or re-infer")

    normalizer = load_frozen_normalizers(sp._resolve(config["normalization_npz"]))
    with np.load(sp._resolve(config["data_output"]), allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    arrays = prepare_holdout_arrays(data, normalizer,
                                    factual_suffix_steps=int(config["factual_suffix_steps"]))
    batch_size = int(config["model"]["batch_size"])
    stage1_dir = sp._resolve(config["stage1_output_dir"])
    checkpoint_hashes = lock["checkpoint_sha256"]

    # Build and write the five manifests (primary + four alternates).
    donors: dict[int, np.ndarray] = {}
    manifest_sha: dict[str, str] = {}
    for donor_seed in DONOR_SEEDS:
        donor, manifest = build_holdout_manifest(arrays, donor_seed)
        donors[int(donor_seed)] = donor
        name = f"confirmatory_manifest_donor{donor_seed}.json"
        path = output_dir / name
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(path)
        manifest_sha[name] = sha256_file(path)
    primary_donor = donors[sp.PRIMARY_DERANGEMENT_SEED]

    prediction_sha: dict[str, str] = {}

    def save(name: str, array: np.ndarray) -> None:
        _atomic_save_npy(output_dir / name, array)
        prediction_sha[name] = sha256_file(output_dir / name)

    # 15 main predictions.
    for condition in CONDITIONS:
        donor = primary_donor if condition == "shuffled_suffix" else None
        for seed in TRAINING_SEEDS:
            model = load_frozen_model(
                condition, seed, stage1_dir / f"{condition}_seed{seed}.pt",
                expected_sha256=checkpoint_hashes[f"{condition}_seed{seed}.pt"],
                model_config=config["model"])
            pred = predict_holdout(model, arrays, condition, donor,
                                   device=device, batch_size=batch_size)
            save(f"{condition}_seed{seed}_prediction.npy", pred)

    # Holdout factual summaries (seed-mean over the five frozen M1 checkpoints) for
    # the frozen S3* representation probe applied at analysis time.
    summaries, actives = [], []
    for seed in TRAINING_SEEDS:
        model = load_frozen_model(
            "uniform_suffix", seed, stage1_dir / f"uniform_suffix_seed{seed}.pt",
            expected_sha256=checkpoint_hashes[f"uniform_suffix_seed{seed}.pt"],
            model_config=config["model"])
        model = model.to(device)
        with torch.no_grad():
            summary, active = model.encode_factual_agents(
                torch.from_numpy(arrays["windowed_factual"]).float().to(device),
                torch.from_numpy(arrays["suffix_mask"]).to(device))
        summaries.append(summary.cpu().numpy())
        actives.append(active.cpu().numpy())
    holdout_summaries = np.stack(summaries).mean(axis=0)
    holdout_active = np.stack(actives).any(axis=0)
    _atomic_save_npy(output_dir / "uniform_suffix_factual_summaries_seed_mean.npy",
                     holdout_summaries)
    _atomic_save_npy(output_dir / "uniform_suffix_factual_active.npy",
                     holdout_active.astype(bool))
    prediction_sha["uniform_suffix_factual_summaries_seed_mean.npy"] = sha256_file(
        output_dir / "uniform_suffix_factual_summaries_seed_mean.npy")
    prediction_sha["uniform_suffix_factual_active.npy"] = sha256_file(
        output_dir / "uniform_suffix_factual_active.npy")

    # 5 fixed-M1 suffix-swap diagnostics (correct pass must match the main M1).
    for seed in TRAINING_SEEDS:
        model = load_frozen_model(
            "uniform_suffix", seed, stage1_dir / f"uniform_suffix_seed{seed}.pt",
            expected_sha256=checkpoint_hashes[f"uniform_suffix_seed{seed}.pt"],
            model_config=config["model"])
        correct = predict_holdout(model, arrays, "uniform_suffix", None,
                                  device=device, batch_size=batch_size)
        main_path = output_dir / f"uniform_suffix_seed{seed}_prediction.npy"
        main = np.load(main_path, allow_pickle=False)
        if not np.allclose(correct, main, atol=1e-4, rtol=1e-3):
            raise RuntimeError(
                f"M1 suffix-swap correct pass disagrees with the main M1 prediction (seed {seed})")
        swapped = predict_holdout(model, arrays, "shuffled_suffix", primary_donor,
                                  device=device, batch_size=batch_size)
        save(f"uniform_suffix_seed{seed}_primary_swap_prediction.npy", swapped)

    # 20 alternate-M2 donor sensitivity predictions.
    for donor_seed in sp.ALTERNATE_DONOR_SEEDS:
        donor = donors[int(donor_seed)]
        for seed in TRAINING_SEEDS:
            model = load_frozen_model(
                "shuffled_suffix", seed, stage1_dir / f"shuffled_suffix_seed{seed}.pt",
                expected_sha256=checkpoint_hashes[f"shuffled_suffix_seed{seed}.pt"],
                model_config=config["model"])
            pred = predict_holdout(model, arrays, "shuffled_suffix", donor,
                                   device=device, batch_size=batch_size)
            save(f"shuffled_suffix_seed{seed}_donor{donor_seed}_prediction.npy", pred)

    receipt = {
        "experiment_name": config["experiment_name"],
        "analysis_status": config["analysis_status"],
        "training_performed": False,
        "checkpoint_modified": False,
        "device": device,
        "n_episodes": int(len(arrays["episode_seed"])),
        "prediction_sha256": prediction_sha,
        "manifest_sha256": manifest_sha,
        "primary_derangement_seed": sp.PRIMARY_DERANGEMENT_SEED,
        "alternate_donor_seeds": list(sp.ALTERNATE_DONOR_SEEDS),
        "lock_sha256": lock["lock_sha256"],
        "config_signature": canonical_fingerprint(json.loads(json.dumps(config))),
        "data_output_sha256": sha256_file(sp._resolve(config["data_output"])),
    }
    tmp = receipt_path.with_name(receipt_path.name + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(receipt_path)
    return receipt
