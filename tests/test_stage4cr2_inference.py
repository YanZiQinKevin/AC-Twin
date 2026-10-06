"""Stage 4C-R2 frozen-inference tests: expansion, outputs contract, determinism."""
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

import collect_stage4cr2_merge_multiaction as coll  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402
from ac_twin import stage4cr2_inference as s4cr2i  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402

NA = s4cr2.N_ACTIONS
CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"


def _config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "run")
    config["data_root"] = str(
        tmp_path / "data" / "object_centric_stage" / s4cr2.PURPOSE_DEV)
    return config


def _write_role_npz(config, role, groups=4):
    """Write a minimal synthetic grouped npz exactly where the collector would put it."""
    rng = np.random.default_rng(0)
    capacity, hl, hz, nf = 5, 10, 20, 8
    history = rng.normal(size=(groups, hl, capacity, nf)).astype(np.float32)
    factual = rng.normal(size=(groups, hz, capacity, nf)).astype(np.float32)
    cf = rng.normal(size=(groups, NA, hz, capacity, nf)).astype(np.float32)
    for arr in (history, factual, cf):
        arr[..., 0] = 0.0
        arr[..., :4, 0] = 1.0
    cf_actions = np.tile([a for a, _ in s4cr2.ACTION_PLANS], (groups, 1)).astype(np.int64)
    cf_durations = np.tile([d for _, d in s4cr2.ACTION_PLANS], (groups, 1)).astype(np.int64)
    factual_actions = rng.integers(0, 5, size=(groups, hz)).astype(np.int64)
    inference_mask = np.zeros((groups, hz, capacity), dtype=bool)
    inference_mask[..., 1:4] = True
    data_output, _, _ = coll.role_paths(config, role)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_output, object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf, cf_actions=cf_actions,
        cf_durations=cf_durations, factual_actions=factual_actions,
        t_star=np.full(groups, 12), episode_seed=np.arange(100, 100 + groups),
        n_mask_cells=np.full(groups, 60), inference_mask=inference_mask)
    return data_output


# --- fake frozen-chain callables (the real ones need the GPU checkpoints) -----------------


def _fake_predictions(bundle, *, binding, stage3r_dir, device, batch_size):
    groups, hz, cap = int(bundle["groups"]), 20, 5
    shape6 = (5, groups, NA, hz, cap, 2)
    shape5 = (5, groups, NA, hz, cap)
    return {
        "e2e": np.zeros(shape6, dtype=np.float32),
        "full": np.zeros(shape6, dtype=np.float32),
        "history": np.zeros(shape6, dtype=np.float32),
        "r_hat": np.zeros(shape5, dtype=np.float32),
        "presence_prob": np.zeros(shape5, dtype=np.float32),
        "high_prob": np.zeros(shape5, dtype=np.float32),
    }


def _fake_features(bundle, predictions):
    return {"U_seed_mean": np.zeros(int(bundle["branches"]), dtype=np.float64)}


class _FixedModel:
    def predict(self, matrix):
        return np.zeros(matrix.shape[0])


def _fake_frozen_models():
    return {"COMBINED_RISK_HEAD": _FixedModel(), "GEOMETRY_ONLY": _FixedModel(),
            "RESPONSE_ONLY": _FixedModel()}


def _fake_score_actions(features, frozen, *, groups, n_actions, rng_seed):
    scores = {k: np.zeros((groups, n_actions), dtype=np.float64) for k in s4cpp.RISK_CLASSES}
    scores["CONSTANT_RANDOM"] = np.full((groups, n_actions), 0.5)
    return scores


def _infer_kwargs():
    return {
        "run_inference_fn": _fake_predictions,
        "build_features_fn": _fake_features,
        "load_models_fn": _fake_frozen_models,
        "score_actions_fn": _fake_score_actions,
    }


# --- bundle loading + guards ---------------------------------------------------------------


def test_load_role_bundle_expansion_order(tmp_path):
    config = _config(tmp_path)
    groups = 3
    _write_role_npz(config, "development_train", groups=groups)
    bundle = s4cr2i.load_role_bundle(config, "development_train")
    assert bundle["groups"] == groups and bundle["branches"] == groups * NA
    cf_action = bundle["tensors"]["cf_action"].numpy().reshape(groups, NA)
    cf_duration = bundle["tensors"]["cf_duration"].numpy().reshape(groups, NA)
    for g in range(groups):
        assert [int(a) for a in cf_action[g]] == [a for a, _ in s4cr2.ACTION_PLANS]
        assert [int(d) for d in cf_duration[g]] == [d for _, d in s4cr2.ACTION_PLANS]
    cf_plans = bundle["tensors"]["counterfactual_actions"].numpy().reshape(groups, NA, 20)
    data_output, _, _ = coll.role_paths(config, "development_train")
    factual_actions = np.load(data_output, allow_pickle=True)["factual_actions"]
    for g in range(groups):
        for a in range(NA):
            act, dur = s4cr2.ACTION_PLANS[a]
            assert np.array_equal(
                cf_plans[g, a], s4cr2.materialize_action_plan(factual_actions[g], act, dur))


def test_load_role_bundle_refuses_forbidden_sources(tmp_path):
    config = _config(tmp_path)
    config["data_root"] = str(
        tmp_path / "data" / "object_centric_stage" / "stage4c_merge_multiaction_dev")
    with pytest.raises(ValueError):
        s4cr2i.load_role_bundle(config, "development_train")
    # the Stage 4C one-shot screen products stay permanently unreadable.
    with pytest.raises(ValueError):
        s4cr2.assert_allowed_stage4cr2_data(
            ROOT / "experiments/stage4_reliability/stage4c/development"
            / "predictions_development_screen.npz")


# --- infer_role output contract -------------------------------------------------------------


def test_infer_role_outputs_feed_feature_contract(tmp_path):
    config = _config(tmp_path)
    groups = 4
    _write_role_npz(config, "development_train", groups=groups)
    receipt = s4cr2i.infer_role(config, "development_train", **_infer_kwargs())
    assert receipt.groups == groups
    assert receipt.status == "INFERENCE_COMPLETE"

    pred_path, score_path, _ = s4cr2i.role_inference_paths(config, "development_train")
    with np.load(pred_path) as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(score_path) as z:
        scores = {k: z[k] for k in z.files}

    # predictions carry the per-seed predictor outputs; scores carry exactly the five
    # action-level risk classes the 34-dim token contract consumes — nothing else.
    assert predictions["e2e"].shape == (5, groups, NA, 20, 5, 2)
    assert set(scores) == set(s4cpp.RISK_CLASSES) | {"group_seed"}
    for key in s4cpp.RISK_CLASSES:
        assert scores[key].shape == (groups, NA)
    # no truth-side array, label or adaptive statistic may leave the inference layer.
    for archive in (predictions, scores):
        for key in archive:
            low = key.lower()
            for banned in ("regret", "j_true", "label", "y_reg", "y_high",
                           "counterfactual_xy", "oracle", "scaler", "class_weight"):
                assert banned not in low, key

    # the saved arrays plug directly into the frozen 9x34-token + 9-global builder.
    bundle = s4cr2i.load_role_bundle(config, "development_train")
    action_risks = {k: scores[k].astype(np.float64) for k in s4cpp.RISK_CLASSES}
    out = s4cpf.build_relative_tokens(
        predictions["e2e"], bundle["factual_xy"], bundle["history_last_xy"],
        bundle["inference_mask"], action_risks, bundle["geometry_group"],
        bundle["t_star"], bundle["episode_seed"])
    assert out["tokens"].shape == (groups, s4cr2.N_COMPETITORS, s4cr2.TOKEN_DIM)
    assert out["global_features"].shape == (groups, s4cr2.GLOBAL_DIM)
    assert bool(out["valid_mask"].all())


def test_infer_role_create_once_never_overwrites(tmp_path):
    config = _config(tmp_path)
    _write_role_npz(config, "development_train", groups=2)
    first = s4cr2i.infer_role(config, "development_train", **_infer_kwargs())
    pred_path, _, receipt_path = s4cr2i.role_inference_paths(config, "development_train")
    receipt_text = receipt_path.read_text(encoding="utf-8")

    def louder_predictions(bundle, *, binding, stage3r_dir, device, batch_size):
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        out["e2e"] = out["e2e"] + 1.0  # a rerun must NOT land on disk
        return out

    second = s4cr2i.infer_role(config, "development_train",
                               **{**_infer_kwargs(), "run_inference_fn": louder_predictions})
    assert second.verification == "existing_inference_verified_no_overwrite"
    assert second.predictions_sha256 == first.predictions_sha256
    assert receipt_path.read_text(encoding="utf-8") == receipt_text
    with np.load(pred_path) as z:
        assert float(np.abs(z["e2e"]).max()) == 0.0


def test_infer_role_rejects_non_finite_predictions(tmp_path):
    config = _config(tmp_path)
    _write_role_npz(config, "development_train", groups=2)

    def bad_predictions(bundle, *, binding, stage3r_dir, device, batch_size):
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        out["e2e"][0, 0, 0, 0, 0, 0] = np.nan
        return out

    with pytest.raises(RuntimeError, match="non-finite"):
        s4cr2i.infer_role(config, "development_train",
                          **{**_infer_kwargs(), "run_inference_fn": bad_predictions})


# --- repeated-inference determinism ---------------------------------------------------------


def _bundle_for_determinism(config, groups=4):
    _write_role_npz(config, "development_screen", groups=groups)
    return s4cr2i.load_role_bundle(config, "development_screen")


def _fake_pair_inference(config, scaled, indices, *, pair_index, binding, stage3r_dir):
    idx = np.asarray(indices, dtype=np.int64)
    base = idx.astype(np.float64)
    n = idx.size
    return {
        "e2e": np.repeat(base[:, None], 2, axis=1).reshape(n, 1, 1, 2).astype(np.float32),
        "full": np.zeros((n, 1, 1, 2), dtype=np.float32),
        "history": np.zeros((n, 1, 1, 2), dtype=np.float32),
        "r_hat": base.astype(np.float32),
        "presence_prob": np.zeros(n, dtype=np.float32),
        "high_prob": np.zeros(n, dtype=np.float32),
    }


def test_verify_repeated_inference_elementwise_identical(tmp_path):
    config = _config(tmp_path)
    bundle = _bundle_for_determinism(config, groups=4)
    receipt = s4cr2i.verify_repeated_inference(
        "development_screen", np.array([0, 2, 3]),
        config=config, bundle=bundle, pair_inference_fn=_fake_pair_inference)
    assert receipt.status == "DETERMINISM_VERIFIED"
    assert receipt.bitwise_identical and receipt.max_abs_diff == 0.0
    assert receipt.n_groups == 3
    assert list(receipt.group_seeds) == [100, 102, 103]


def test_verify_repeated_inference_fails_closed_on_drift(tmp_path):
    config = _config(tmp_path)
    bundle = _bundle_for_determinism(config, groups=4)
    calls = {"n": 0}

    def flaky(config_, scaled, indices, *, pair_index, binding, stage3r_dir):
        calls["n"] += 1
        out = _fake_pair_inference(config_, scaled, indices, pair_index=pair_index,
                                   binding=binding, stage3r_dir=stage3r_dir)
        if calls["n"] == 2:
            out["e2e"] = out["e2e"] + 1e-3
        return out

    with pytest.raises(RuntimeError, match="elementwise"):
        s4cr2i.verify_repeated_inference(
            "development_screen", np.array([0, 1]),
            config=config, bundle=bundle, pair_inference_fn=flaky)


def test_verify_repeated_inference_validates_indices(tmp_path):
    config = _config(tmp_path)
    bundle = _bundle_for_determinism(config, groups=4)
    with pytest.raises(ValueError):
        s4cr2i.verify_repeated_inference(
            "development_screen", np.array([0, 4]),  # 4 is out of range
            config=config, bundle=bundle, pair_inference_fn=_fake_pair_inference)
    with pytest.raises(ValueError):
        s4cr2i.verify_repeated_inference(
            "development_screen", np.array([], dtype=np.int64),
            config=config, bundle=bundle, pair_inference_fn=_fake_pair_inference)
    with pytest.raises(ValueError):
        s4cr2i.verify_repeated_inference(
            "development_screen", np.array([1, 1]),  # duplicated group
            config=config, bundle=bundle, pair_inference_fn=_fake_pair_inference)
