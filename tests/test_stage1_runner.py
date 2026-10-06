from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin.stage1_data import OBJECT_FEATURES
from run_stage1_evidence_value import (
    SMOKE_DECISION,
    _denormalize_effect,
    _evaluate_gate_s,
    _gate_a_robustness,
    _gate_s2,
    _gate_s3,
    _low_high_diff_scalar,
    _manifest_qa_pass,
    _model_inputs,
    _nonego_valid_mask,
    analyze_stage1,
    prepare_stage1_arrays,
    run_signature,
    run_stage1,
    validate_stage1_config,
)


def _synthetic_dataset(episodes=24, agents=3, seed=0):
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(episodes, 10, agents, 8)).astype(np.float32)
    factual = rng.normal(size=(episodes, 20, agents, 8)).astype(np.float32)
    counterfactual = rng.normal(size=(episodes, 20, agents, 8)).astype(np.float32)
    for values in (history, factual, counterfactual):
        values[..., 0] = 1.0
        values[..., 1] = 0.0
        values[:, :, 0, 1] = 1.0
        values[..., 2:4] *= 20.0
    # Give the counterfactual a real positional divergence for response signal.
    counterfactual[..., 2:4] += rng.normal(scale=2.0, size=(episodes, 20, agents, 2)).astype(np.float32)
    return {
        "object_history": history,
        "object_factual_future": factual,
        "object_counterfactual_future": counterfactual,
        "factual_actions": rng.integers(0, 5, (episodes, 20)),
        "counterfactual_actions": rng.integers(0, 5, (episodes, 20)),
        "cf_action": rng.integers(0, 5, episodes),
        "cf_duration": rng.integers(1, 3, episodes),
        "episode_seed": np.arange(10_000, 10_000 + episodes),
        "split": np.array([0] * 12 + [1] * 6 + [2] * 6),
        "object_features_json": np.asarray(json.dumps(list(OBJECT_FEATURES))),
    }


def _write_npz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimal_config(tmp_path: Path, smoke=True):
    source = tmp_path / "source.npz"
    _write_npz(source, _synthetic_dataset())
    config = {
        "experiment_name": "stage1_unit",
        "scene": "merge",
        "source_data": str(source),
        "output_dir": str(tmp_path / "out"),
        "evaluation_mode": "development_test",
        "factual_suffix_steps": 10,
        "progress_every_epochs": 5,
        "conditions": [
            {"name": "no_suffix", "suffix_mode": "none"},
            {"name": "uniform_suffix", "suffix_mode": "correct"},
            {"name": "shuffled_suffix", "suffix_mode": "shuffled"},
        ],
        "derangement": {"base_seed": 20260907, "validation_seed": 20261907, "test_seed": 20262907},
        "model": {
            "object_embedding_dim": 8, "hidden_dim": 8, "action_embedding_dim": 4,
            "batch_size": 8, "max_epochs": 2, "early_stopping_patience": 2,
            "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip_norm": 5.0,
            "training_seeds": [11],
        },
        "statistics": {
            "bootstrap_resamples": 50, "confidence_level": 0.95,
            "zero_response_tolerance_m": 0.0001,
            "positive_response_quantiles": [1 / 3, 2 / 3],
            "eta_epsilon_m": 0.0001, "eta_sensitivity_epsilons_m": [1e-6, 1e-3],
            "bootstrap_seed": 20260907,
        },
    }
    if smoke:
        config["smoke_test"] = True
        config["sample_limits"] = {"train": 12, "validation": 6, "test": 6}
    return config


# --- config validation -------------------------------------------------------

def test_validate_accepts_minimal_smoke(tmp_path):
    validate_stage1_config(minimal_config(tmp_path))


@pytest.mark.parametrize("bad", ["rename", "missing", "duplicate"])
def test_validate_rejects_bad_conditions(tmp_path, bad):
    config = minimal_config(tmp_path)
    if bad == "rename":
        config["conditions"][0]["name"] = "wrong"
    elif bad == "missing":
        config["conditions"].pop()
    else:
        config["conditions"].append(config["conditions"][0].copy())
    with pytest.raises(ValueError, match="condition"):
        validate_stage1_config(config)


def test_validate_rejects_wrong_suffix_mode(tmp_path):
    config = minimal_config(tmp_path)
    config["conditions"][0]["suffix_mode"] = "correct"
    with pytest.raises(ValueError, match="suffix_mode"):
        validate_stage1_config(config)


def test_formal_requires_five_seeds(tmp_path):
    config = minimal_config(tmp_path, smoke=False)
    with pytest.raises(ValueError, match="five seeds"):
        validate_stage1_config(config)


def test_formal_forbids_frozen_test_data(tmp_path):
    config = minimal_config(tmp_path, smoke=False)
    config["model"]["training_seeds"] = [11, 22, 33, 44, 55]
    config["statistics"]["bootstrap_resamples"] = 10000
    config["frozen_test_data"] = "somewhere.npz"
    with pytest.raises(ValueError, match="frozen_test_data"):
        validate_stage1_config(config)


def test_evaluation_mode_must_be_development_test(tmp_path):
    config = minimal_config(tmp_path)
    config["evaluation_mode"] = "frozen_test"
    with pytest.raises(ValueError, match="development_test"):
        validate_stage1_config(config)


def test_signature_changes_with_source_hash(tmp_path):
    config = minimal_config(tmp_path)
    sig_a = run_signature(config, "no_suffix", 11, "aaa", "vvv", "ttt")
    sig_b = run_signature(config, "no_suffix", 11, "bbb", "vvv", "ttt")
    assert sig_a != sig_b


# --- full smoke run + resume --------------------------------------------------

def test_smoke_run_produces_complete_resumable_artifacts(tmp_path, capsys):
    config = minimal_config(tmp_path)
    report = run_stage1(config, resume=False, device="cpu")
    assert report["decision"] == SMOKE_DECISION
    assert report["analysis"]["decision"] == SMOKE_DECISION
    out = Path(config["output_dir"])
    for condition in ("no_suffix", "uniform_suffix", "shuffled_suffix"):
        stem = f"{condition}_seed11"
        for suffix in (".pt", "_training.json", "_val_prediction.npy",
                       "_test_prediction.npy", "_run.json"):
            assert (out / f"{stem}{suffix}").is_file(), suffix
    for shared in ("normalization.npz", "response_bins.json", "validation_manifest.json",
                   "test_manifest.json", "report.json"):
        assert (out / shared).is_file(), shared
    # Smoke must not print development-test metrics or a scientific conclusion.
    captured = capsys.readouterr().out
    assert "ADE" not in captured and "FDE" not in captured

    # Resume: identical command must not retrain; weights + predictions unchanged.
    before = {p.name: _sha256(p) for p in out.iterdir() if p.suffix in {".pt", ".npy"}}
    report2 = run_stage1(config, resume=True, device="cpu")
    after = {p.name: _sha256(p) for p in out.iterdir() if p.suffix in {".pt", ".npy"}}
    assert before == after
    assert report2["decision"] == SMOKE_DECISION


def test_resume_rejected_when_source_changes(tmp_path):
    config = minimal_config(tmp_path)
    run_stage1(config, resume=False, device="cpu")
    # Rewrite the source with a perturbed value -> different sha256.
    data = _synthetic_dataset()
    data["object_history"][0, 0, 0, 2] += 1.0
    _write_npz(Path(config["source_data"]), data)
    with pytest.raises(ValueError, match="does not match"):
        run_stage1(config, resume=True, device="cpu")


def test_orphan_artifact_is_not_treated_as_complete(tmp_path):
    config = minimal_config(tmp_path)
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "no_suffix_seed11.pt").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="no signature"):
        run_stage1(config, resume=True, device="cpu")


def test_existing_artifacts_without_resume_fail(tmp_path):
    config = minimal_config(tmp_path)
    run_stage1(config, resume=False, device="cpu")
    with pytest.raises(FileExistsError, match="--resume"):
        run_stage1(config, resume=False, device="cpu")


# --- analyze_stage1 executes on synthetic predictions -------------------------

def test_model_inputs_exclude_counterfactual_future(tmp_path):
    config = minimal_config(tmp_path)
    with np.load(config["source_data"], allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    arrays = prepare_stage1_arrays(config, data)
    idx = arrays["indices"]["train"]
    for condition in ("no_suffix", "uniform_suffix", "shuffled_suffix"):
        donor = None
        if condition == "shuffled_suffix":
            from run_stage1_evidence_value import build_split_derangement
            donor, _ = build_split_derangement(arrays, idx, 0, 20260907)
        inputs = _model_inputs(arrays, idx, condition, donor)
        # counterfactual_actions (the queried intervention ACF) is a legal input;
        # the counterfactual future trajectory must never be an input.
        assert "object_counterfactual_future" not in inputs
        assert not any(key.endswith("_future") or "cf_future" in key for key in inputs)

    # M0 must use an all-zero suffix from the training input onward.
    m0 = _model_inputs(arrays, idx, "no_suffix", None)
    assert np.count_nonzero(m0["suffix"]) == 0
    assert not m0["suffix_mask"].any()


def _prepared_arrays(config):
    with np.load(config["source_data"], allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    arrays = prepare_stage1_arrays(config, data)
    arrays["_history_full"] = np.asarray(data["object_history"], dtype=np.float32)
    arrays["_indices_train"] = arrays["indices"]["train"]
    from ac_twin.stage1_statistics import fit_response_bins
    train_nonego = _nonego_valid_mask(arrays, arrays["indices"]["train"])
    arrays["_response_thresholds"] = fit_response_bins(
        arrays["r_pos"][arrays["indices"]["train"]][train_nonego], 0.0001, (1 / 3, 2 / 3))
    return arrays


def test_denormalize_effect_returns_world_coordinates():
    arrays = {"target_mean": np.array([1.0, 2.0]), "target_std": np.array([2.0, 0.5])}
    pred_norm = np.array([[[[1.0, 2.0]]]])  # normalized units
    world = _denormalize_effect(pred_norm, arrays)
    np.testing.assert_allclose(world[0, 0, 0], [1.0 * 2.0 + 1.0, 2.0 * 0.5 + 2.0])


def test_gate_s2_passes_when_model_beats_baselines(tmp_path):
    """Regression: S2 must pass when the model's R_hat matches the true response.

    With the reversed sign this returned False even for a near-perfect model.
    """
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    indices = arrays["indices"]
    val_idx = indices["validation"]
    val_nonego = _nonego_valid_mask(arrays, val_idx)
    r_pos_val = arrays["r_pos"][val_idx]
    n_val, T, N = r_pos_val.shape
    # Model prediction whose norm equals the true response -> MAE ~ 0, Spearman 1.
    perfect = np.zeros((n_val, T, N, 2))
    perfect[..., 0] = r_pos_val
    predictions = {"uniform_suffix": [{"val": perfect}]}
    passed, detail = _gate_s2(arrays, val_idx, val_nonego, predictions, config, r_pos_val)
    assert detail["spearman"]["point"] > 0.9
    assert detail["model_mae"] < detail["baseline_action_horizon_mae"]
    assert detail["model_mae"] < detail["baseline_action_horizon_distance_mae"]
    assert passed, f"S2 should pass for a near-perfect model, got detail={detail}"


def test_analyze_stage1_executes(tmp_path):
    config = minimal_config(tmp_path)
    source = Path(config["source_data"])
    with np.load(source, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}
    arrays = prepare_stage1_arrays(config, data)
    arrays["_history_full"] = np.asarray(data["object_history"], dtype=np.float32)
    arrays["_indices_train"] = arrays["indices"]["train"]
    from ac_twin.stage1_statistics import fit_response_bins
    from run_stage1_evidence_value import _nonego_valid_mask
    train_nonego = _nonego_valid_mask(arrays, arrays["indices"]["train"])
    arrays["_response_thresholds"] = fit_response_bins(
        arrays["r_pos"][arrays["indices"]["train"]][train_nonego], 0.0001, (1 / 3, 2 / 3))

    rng = np.random.default_rng(0)
    indices = arrays["indices"]
    n_val, n_test = len(indices["validation"]), len(indices["test"])
    T, N = arrays["target"].shape[1], arrays["target"].shape[2]

    def fake_split(n):
        return rng.normal(scale=0.5, size=(n, T, N, 2)).astype(np.float32)

    predictions = {}
    for ci, condition in enumerate(("no_suffix", "uniform_suffix", "shuffled_suffix")):
        entries = []
        for _ in range(2):  # two seeds
            summaries = {
                "train": rng.normal(size=(len(indices["train"]), N, 8)),
                "val": rng.normal(size=(n_val, N, 8)),
            }
            actives = {
                "train": np.ones((len(indices["train"]), N), bool),
                "val": np.ones((n_val, N), bool),
            }
            entries.append({
                "val": fake_split(n_val) + ci * 0.3,
                "test": fake_split(n_test) + ci * 0.3,
                "log": [{"epoch": 1, "train_mse": 1.0, "val_mse": 1.0}],
                "parameters": 123,
                "factual_summaries": summaries,
                "factual_active": actives,
            })
        predictions[condition] = entries

    result = analyze_stage1(arrays, indices, predictions, config)
    assert result["gate_s"] in {"PASS", "FAIL"}
    assert result["gate_a"] in {"DEVELOPMENT_GO_TO_NEW_HOLDOUT", "DEVELOPMENT_NO_GO",
                                "BLOCKED_AWAITING_GATE_S"}
    assert result["gate_b"] in {"SUPPORTED", "NOT_SUPPORTED", "UNDERPOWERED",
                                "BLOCKED_AWAITING_GATE_S"}
    assert "bins" in result and "gate_s_detail" in result


# --- Protocol-fix tests (Stage 1 closure plan) --------------------------------

def _full_predictions(arrays, seed_count=2, hidden=8):
    """A complete predictions dict with every key the Gate S/A/B analysis reads."""
    rng = np.random.default_rng(5)
    indices = arrays["indices"]
    n_tr, n_val, n_test = (len(indices[k]) for k in ("train", "validation", "test"))
    T, N = arrays["target"].shape[1], arrays["target"].shape[2]

    def eff(n):
        return rng.normal(size=(n, T, N, 2))

    predictions = {}
    for condition in ("no_suffix", "uniform_suffix", "shuffled_suffix"):
        entries = []
        for _ in range(seed_count):
            entry = {"val": eff(n_val), "test": eff(n_test),
                     "log": [{"epoch": 1, "train_mse": 1.0, "val_mse": 1.0}],
                     "parameters": 1000}
            if condition == "uniform_suffix":
                entry["factual_summaries"] = {"train": rng.normal(size=(n_tr, N, hidden)),
                                              "val": rng.normal(size=(n_val, N, hidden))}
                entry["factual_active"] = {"train": np.ones((n_tr, N), bool),
                                           "val": np.ones((n_val, N), bool)}
                entry["val_suffix_swap_correct"] = eff(n_val)
                entry["val_suffix_swap_shuffled"] = eff(n_val)
            entries.append(entry)
        predictions[condition] = entries
    return predictions


def test_gate_s_does_not_read_development_test(tmp_path):
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    predictions = _full_predictions(arrays)
    status_a, detail_a = _evaluate_gate_s(arrays, arrays["indices"], predictions, config,
                                          arrays["_response_thresholds"])
    # Replace every development-test prediction with extreme values; S must not change.
    import copy
    corrupted = copy.deepcopy(predictions)
    for condition in corrupted:
        for entry in corrupted[condition]:
            entry["test"] = np.full_like(entry["test"], 1e9)
    status_b, detail_b = _evaluate_gate_s(arrays, arrays["indices"], corrupted, config,
                                          arrays["_response_thresholds"])
    assert status_a == status_b
    assert detail_a["S1"] == detail_b["S1"]
    assert detail_a["S1_detail"]["ego_m1_minus_m0_point"] == detail_b["S1_detail"]["ego_m1_minus_m0_point"]
    assert detail_a["data_scope"] == "train_validation_only"


def _gate_s1_ego_control(arrays, config, uniform_val, no_suffix_val):
    predictions = _full_predictions(arrays)
    for entry in predictions["uniform_suffix"]:
        entry["val"] = uniform_val.copy()
    for entry in predictions["no_suffix"]:
        entry["val"] = no_suffix_val.copy()
    _, detail = _evaluate_gate_s(arrays, arrays["indices"], predictions, config,
                                 arrays["_response_thresholds"])
    return detail["S1_detail"]


def test_ego_positive_control_strict_direction(tmp_path):
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    val_idx = arrays["indices"]["validation"]
    target_val = arrays["target"][val_idx]
    # Case 1: M1 perfect, M0 off on ego -> M1 ego error < M0 -> pass.
    uniform_val = target_val.copy()
    no_suffix_val = target_val.copy()
    no_suffix_val[:, :, 0, :] += 5.0  # ego worse for M0
    detail = _gate_s1_ego_control(arrays, config, uniform_val, no_suffix_val)
    assert detail["ego_control_direction"] is True
    assert detail["ego_m1_minus_m0_point"] < 0
    # Case 2: M1 worse than M0 by any amount (e.g. 1.49x) -> must NOT pass (no 1.5x tol).
    uniform_val2 = target_val.copy()
    no_suffix_val2 = target_val.copy()
    no_suffix_val2[:, :, 0, :] += 2.0
    uniform_val2[:, :, 0, :] += 2.0 * 1.49  # M1 1.49x worse
    detail2 = _gate_s1_ego_control(arrays, config, uniform_val2, no_suffix_val2)
    assert detail2["ego_control_direction"] is False


def test_low_high_diff_scalar_direction_and_zero_denominator():
    # correct suffix much better in low than high -> positive diff
    assert _low_high_diff_scalar(1.0, 4.0, 3.0, 4.0) > 0
    # zero shuffled denominator -> unavailable (nan), not a silent epsilon
    assert np.isnan(_low_high_diff_scalar(1.0, 0.0, 1.0, 4.0))
    assert np.isnan(_low_high_diff_scalar(1.0, 4.0, 1.0, 0.0))


def test_gate_a_robustness_leave_one_out(tmp_path):
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    indices = arrays["indices"]
    test_idx = indices["test"]
    test_nonego = _nonego_valid_mask(arrays, test_idx)
    n_test, T, N = arrays["target"][test_idx].shape[0], arrays["target"].shape[1], arrays["target"].shape[2]
    bin_id = np.zeros((n_test, T, N), dtype=np.int64)
    bin_id[:, : T // 2] = 0   # low bins (0)
    bin_id[:, T // 2:] = 3    # high bins (3)
    # Correct suffix clearly better in low than in high, for every seed.
    errors_by_seed = {"uniform_suffix": [], "shuffled_suffix": []}
    rng = np.random.default_rng(0)
    for _ in range(3):
        errors_by_seed["uniform_suffix"].append(np.where(bin_id == 0, 1.0, 3.0) + rng.normal(0, 1e-4, bin_id.shape))
        errors_by_seed["shuffled_suffix"].append(np.where(bin_id == 0, 4.0, 4.0) + rng.normal(0, 1e-4, bin_id.shape))
    errors_by_seed = {k: np.stack(v) for k, v in errors_by_seed.items()}
    l_suffix = errors_by_seed["uniform_suffix"].mean(0)
    l_shuffled = errors_by_seed["shuffled_suffix"].mean(0)
    cf_action = arrays["cf_action"][test_idx]
    result = _gate_a_robustness(errors_by_seed, l_suffix, l_shuffled, bin_id, test_nonego,
                                cf_action, (0, 1), (3,), 50, 0.95, 20260907)
    assert result["not_single_seed"] is True
    assert result["not_single_action_type"] is True
    assert result["not_high_influence_episodes"] is True

    # Now make one seed reverse the direction -> not_single_seed must fail.
    errors_by_seed["uniform_suffix"][0] = np.where(bin_id == 0, 4.0, 1.0)
    result2 = _gate_a_robustness(errors_by_seed, l_suffix, l_shuffled, bin_id, test_nonego,
                                 cf_action, (0, 1), (3,), 50, 0.95, 20260907)
    assert result2["not_single_seed"] is False


def test_manifest_qa_pass():
    clean = {"validation": {"qa": {"self_match": 0, "same_seed": 0, "cross_split": 0, "duplicate_donor": 0}},
             "test": {"qa": {"self_match": 0, "same_seed": 0, "cross_split": 0, "duplicate_donor": 0}}}
    ok, _ = _manifest_qa_pass(clean)
    assert ok is True
    bad = {"validation": {"qa": {"self_match": 1, "same_seed": 0, "cross_split": 0, "duplicate_donor": 0}},
           "test": {"qa": {"self_match": 0, "same_seed": 0, "cross_split": 0, "duplicate_donor": 0}}}
    ok2, _ = _manifest_qa_pass(bad)
    assert ok2 is False
    ok3, _ = _manifest_qa_pass(None)
    assert ok3 is False


def test_gate_s3_does_not_use_m2(tmp_path):
    """Sentinel M2 predictions must not change S3 (it uses fixed-M1 swap only)."""
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    predictions = _full_predictions(arrays)
    _, detail_a = _gate_s3(arrays, arrays["indices"], predictions, config)
    import copy
    corrupted = copy.deepcopy(predictions)
    for entry in corrupted["shuffled_suffix"]:
        entry["val"] = np.full_like(entry["val"], 1e9)
        entry["test"] = np.full_like(entry["test"], 1e9)
    _, detail_b = _gate_s3(arrays, arrays["indices"], corrupted, config)
    assert detail_a["prediction_evidence"] == detail_b["prediction_evidence"]
    assert detail_a["method"] == "fixed_m1_checkpoint_suffix_swap"


def _motion_summary(factual):
    presence = factual[..., 0] > 0
    count = presence.sum(1)
    disp = factual[:, -1, :, 2:4] - factual[:, 0, :, 2:4]
    vel_sum = (factual[..., 4:6] * presence[..., None]).sum(1)
    mean_vel = np.where(count[..., None] > 0, vel_sum / np.maximum(count, 1)[..., None], 0.0)
    final_vel = factual[:, -1, :, 4:6]
    return np.concatenate([disp, mean_vel, final_vel], axis=-1)


def test_s3_probe_episode_bootstrap_ci(tmp_path):
    from run_stage1_evidence_value import _gate_s3_probe
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    indices = arrays["indices"]
    n_tr, n_val = len(indices["train"]), len(indices["validation"])
    hidden = 8
    y_tr = _motion_summary(arrays["object_factual_future"][indices["train"]])
    y_val = _motion_summary(arrays["object_factual_future"][indices["validation"]])
    N = y_tr.shape[1]
    rng = np.random.default_rng(0)

    def strong(y):
        s = np.zeros((y.shape[0], N, hidden))
        s[..., :6] = y + rng.normal(0, 0.01, y.shape)
        return s

    entry = {"factual_summaries": {"train": strong(y_tr), "val": strong(y_val)},
             "factual_active": {"train": np.ones((n_tr, N), bool), "val": np.ones((n_val, N), bool)}}
    passed, detail = _gate_s3_probe(arrays, indices, {"uniform_suffix": [entry]}, config)
    assert passed, detail
    assert detail["probe_minus_shuffled_label"]["ci_upper"] < 0
    assert detail["probe_minus_scene_mean"]["ci_upper"] < 0
    assert detail["n_episodes"] >= 2

    # A representation that does not beat the scene-mean baseline must fail.
    rng2 = np.random.default_rng(1)
    weak = {"factual_summaries": {"train": rng2.normal(size=(n_tr, N, hidden)),
                                  "val": rng2.normal(size=(n_val, N, hidden))},
            "factual_active": {"train": np.ones((n_tr, N), bool), "val": np.ones((n_val, N), bool)}}
    passed2, detail2 = _gate_s3_probe(arrays, indices, {"uniform_suffix": [weak]}, config)
    assert passed2 is False


def test_suffix_swap_single_checkpoint_and_determinism(tmp_path):
    import torch
    from ac_twin.stage1_model import PerAgentPairedEffectGRU
    from run_stage1_evidence_value import (
        _predict, _suffix_swap_inference, build_split_derangement)
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    indices = arrays["indices"]
    torch.manual_seed(0)
    model = PerAgentPairedEffectGRU(object_embedding_dim=8, hidden_dim=8, action_embedding_dim=4)
    model.eval()
    device = torch.device("cpu")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    val_idx = indices["validation"]
    # Seal a "correct-suffix" validation prediction from this checkpoint.
    correct_inputs = _model_inputs(arrays, val_idx, "uniform_suffix", None)
    with torch.no_grad():
        sealed = _predict(model, correct_inputs, int(config["model"]["batch_size"]), device)
    np.save(out / "uniform_suffix_seed11_val_prediction.npy", sealed, allow_pickle=False)
    torch.save(model.state_dict(), out / "uniform_suffix_seed11.pt")
    validation_donor, _ = build_split_derangement(arrays, val_idx, 1, 20261907)
    correct_world, shuffled_world = _suffix_swap_inference(
        model, arrays, indices, validation_donor, config, 11, device, out, "merge",
        "sourcehash", "valmanifesthash")
    # Same checkpoint + identical H/AF/ACF: correct pass reproduces the sealed prediction.
    np.testing.assert_allclose(correct_world, _denormalize_effect(sealed, arrays),
                               rtol=1e-4, atol=1e-4)
    # Only the suffix was swapped: shuffled output differs from correct.
    assert not np.allclose(shuffled_world, correct_world)
    # New inference artifact + metadata written; no training performed.
    assert (out / "uniform_suffix_seed11_val_prediction_with_frozen_shuffled_suffix.npy").is_file()
    meta = json.loads((out / "uniform_suffix_seed11_val_suffix_swap_metadata.json").read_text())
    assert meta["training_performed"] is False
    assert meta["checkpoint_modified"] is False
    assert meta["correct_suffix_matches_sealed_val"] is True
    assert meta["mask_qa"]["self_match"] == 0
    # Fixed seed / same checkpoint rerun is bit-identical (read-only forward, CPU).
    _, shuffled_world2 = _suffix_swap_inference(
        model, arrays, indices, validation_donor, config, 11, device, out, "merge",
        "sourcehash", "valmanifesthash")
    np.testing.assert_allclose(shuffled_world, shuffled_world2)


def test_shuffled_suffix_swaps_donor_features_and_mask_together(tmp_path):
    config = minimal_config(tmp_path)
    arrays = _prepared_arrays(config)
    from run_stage1_evidence_value import build_split_derangement
    val_idx = arrays["indices"]["validation"]
    donor, _ = build_split_derangement(arrays, val_idx, 1, 20261907)
    shuffled = _model_inputs(arrays, val_idx, "shuffled_suffix", donor)
    donor_global = val_idx[donor]
    # suffix features and suffix mask both come from the donor episode, aligned.
    np.testing.assert_array_equal(shuffled["suffix"],
                                  arrays["windowed_factual"][donor_global])
    np.testing.assert_array_equal(shuffled["suffix_mask"],
                                  arrays["suffix_mask"][donor_global])
    # history and actions stay with the recipient.
    np.testing.assert_array_equal(shuffled["history"], arrays["norm_history"][val_idx])
