"""Stage 4C-R2 independent-confirmation inference tests: truth isolation + frozen scoring.

Plan ``AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`` sections 2.2, 3.3, 5.1, 9 Task 1.
The score side must never open ``confirmation_truth.npz`` (or any counterfactual-truth
key), never refit the scaler, and reproduce R2's frozen implementation bit-for-bit on
the fixed first-32 R2 train groups (legal inputs + on-disk predictions only).
"""
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_confirm_protocol as prot  # noqa: E402
from ac_twin import stage4cr2_confirm_inference as cinf  # noqa: E402
from ac_twin import stage4cr2_training as s4cr2t  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402

NA = prot.N_ACTIONS
CONFIG_PATH = (
    ROOT / "experiments/stage4_reliability/stage4cr2_confirm/stage4cr2_confirm_config.json")
R2_DEV = ROOT / "experiments/stage4_reliability/stage4c_r2/development"
R2_TRAIN_DATA = (
    ROOT / "data/object_centric_stage/stage4cr2_merge_multiaction_dev"
    / "object_twinworld_stage4cr2_train.npz")

TRUTH_KEY_SUBSTRINGS = cinf.FORBIDDEN_INPUT_KEY_TOKENS


def _config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    data_root = tmp_path / "data" / "object_centric_stage" / prot.PURPOSE_CONFIRM
    config["run_dir"] = str(run_dir)
    config["data_root"] = str(data_root)
    for key, value in config["paths"].items():
        if value.startswith("experiments/stage4_reliability/stage4cr2_confirm/"):
            rel = value.removeprefix("experiments/stage4_reliability/stage4cr2_confirm/")
            config["paths"][key] = str(run_dir / rel)
        elif value.startswith("data/object_centric_stage/stage4cr2_merge_multiaction_confirm/"):
            rel = value.removeprefix(
                "data/object_centric_stage/stage4cr2_merge_multiaction_confirm/")
            config["paths"][key] = str(data_root / rel)
    return config


def _write_legal_inputs(config, role="confirm", groups=4):
    """Write a minimal synthetic §5.1 legal-inputs npz exactly where expected."""
    rng = np.random.default_rng(0)
    capacity, hl, hz, nf = 5, 10, 20, 8
    history = rng.normal(size=(groups, hl, capacity, nf)).astype(np.float32)
    factual = rng.normal(size=(groups, hz, capacity, nf)).astype(np.float32)
    for arr in (history, factual):
        arr[..., 0] = 0.0
        arr[..., :4, 0] = 1.0
    cf_actions = np.tile([a for a, _ in prot.ACTION_PLANS], (groups, 1)).astype(np.int64)
    cf_durations = np.tile([d for _, d in prot.ACTION_PLANS], (groups, 1)).astype(np.int64)
    factual_actions = rng.integers(0, 5, size=(groups, hz)).astype(np.int64)
    inference_mask = np.zeros((groups, hz, capacity), dtype=bool)
    inference_mask[..., 1:4] = True
    input_path = cinf.role_input_path(config, role)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        input_path, object_history=history, object_factual_future=factual,
        factual_actions=factual_actions, cf_actions=cf_actions,
        cf_durations=cf_durations, t_star=np.full(groups, 12),
        group_seed=np.arange(100, 100 + groups, dtype=np.int64),
        inference_mask=inference_mask)
    return input_path


# --- fake frozen-chain callables (the real ones need the full checkpoint chain) ----------


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


def _fake_score_fn(inventory, bundle, *, device):
    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64).ravel()
    n = group_seed.size
    base = (group_seed % 97).astype(np.float64) / 97.0
    by_seed = np.stack([base + 0.001 * i for i in range(5)])
    return s4cr2t.RiskScores(
        group_seed=group_seed,
        risk_ensemble=by_seed.mean(axis=0),
        risk_by_seed=by_seed,
        mu_only_ensemble=base * 0.5,
        class_only_ensemble=base * 0.25,
        seeds=tuple(int(s) for s in prot.MODEL_SEEDS),
    )


def _infer_kwargs():
    return {
        "run_inference_fn": _fake_predictions,
        "build_features_fn": _fake_features,
        "load_models_fn": _fake_frozen_models,
        "score_actions_fn": _fake_score_actions,
        "score_fn": _fake_score_fn,
    }


# --- legal-input bundle loading ---------------------------------------------------------


def test_load_inference_bundle_fixed_contract(tmp_path):
    config = _config(tmp_path)
    groups = 3
    input_path = _write_legal_inputs(config, groups=groups)
    bundle = cinf.load_inference_bundle(str(input_path), config)
    assert bundle["groups"] == groups
    assert bundle["n_actions"] == NA == 10  # fixed contract, never read from CF truth
    assert bundle["branches"] == groups * NA
    # the score-side bundle must NOT carry any counterfactual-truth array.
    forbidden = {"counterfactual_xy", "J_true", "regret", "a_star", "true_response",
                 "object_counterfactual_futures"}
    assert forbidden.isdisjoint(bundle)
    tensors = bundle["tensors"]
    cf_action = tensors["cf_action"].numpy().reshape(groups, NA)
    cf_duration = tensors["cf_duration"].numpy().reshape(groups, NA)
    for g in range(groups):
        assert [int(a) for a in cf_action[g]] == [a for a, _ in prot.ACTION_PLANS]
        assert [int(d) for d in cf_duration[g]] == [d for _, d in prot.ACTION_PLANS]
    cf_plans = tensors["counterfactual_actions"].numpy().reshape(groups, NA, 20)
    with np.load(input_path, allow_pickle=True) as z:
        factual_actions = z["factual_actions"]
    for g in range(groups):
        for a in range(NA):
            act, dur = prot.ACTION_PLANS[a]
            assert np.array_equal(
                cf_plans[g, a], prot.materialize_action_plan(factual_actions[g], act, dur))


def test_load_inference_bundle_rejects_truth_keys(tmp_path):
    config = _config(tmp_path)
    input_path = _write_legal_inputs(config, groups=2)
    with np.load(input_path) as z:
        legal = {k: z[k] for k in z.files}
    for bad_key in ("object_counterfactual_futures", "counterfactual_xy", "J_true",
                    "regret", "a_star", "true_response"):
        payload = dict(legal)
        payload[bad_key] = np.zeros((2, NA, 20, 5, 8), dtype=np.float32)
        np.savez(input_path, **payload)
        with pytest.raises(ValueError):
            cinf.load_inference_bundle(str(input_path), config)
    np.savez(input_path, **legal)  # restore


def test_load_inference_bundle_rejects_missing_legal_keys(tmp_path):
    config = _config(tmp_path)
    input_path = _write_legal_inputs(config, groups=2)
    with np.load(input_path) as z:
        legal = {k: z[k] for k in z.files}
    for dropped in ("object_history", "object_factual_future", "factual_actions",
                    "cf_actions", "cf_durations", "t_star", "group_seed", "inference_mask"):
        payload = {k: v for k, v in legal.items() if k != dropped}
        np.savez(input_path, **payload)
        with pytest.raises(ValueError):
            cinf.load_inference_bundle(str(input_path), config)
    np.savez(input_path, **legal)


def test_load_inference_bundle_refuses_truth_path(tmp_path):
    """Pointing the score side at the sealed truth file is refused by path alone."""
    config = _config(tmp_path)
    truth_path = cinf.role_truth_path(config, "confirm")
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(truth_path, object_counterfactual_futures=np.zeros((2, NA, 20, 5, 8)))
    with pytest.raises(ValueError):
        cinf.load_inference_bundle(str(truth_path), config)


def test_load_inference_bundle_rejects_wrong_action_axis(tmp_path):
    config = _config(tmp_path)
    input_path = _write_legal_inputs(config, groups=2)
    with np.load(input_path) as z:
        legal = {k: z[k] for k in z.files}
    legal["cf_actions"] = legal["cf_actions"][:, :9]  # not the fixed ten-action contract
    np.savez(input_path, **legal)
    with pytest.raises(ValueError):
        cinf.load_inference_bundle(str(input_path), config)


def test_score_bundle_excludes_truth(tmp_path):
    """The brief's canonical isolation shape: legal bundle disjoint from truth keys."""
    config = _config(tmp_path)
    input_path = _write_legal_inputs(config, groups=4)
    legal_bundle = cinf.load_inference_bundle(str(input_path), config)
    forbidden = {"counterfactual_xy", "J_true", "regret", "a_star", "true_response"}
    assert forbidden.isdisjoint(legal_bundle)
    receipt = cinf.infer_and_score(config, "confirm", **_infer_kwargs())
    assert receipt["groups"] == 4
    with np.load(Path(receipt["risk_scores_path"])) as z:
        assert np.isfinite(z["risk_ensemble"]).all()


# --- truth-file access isolation (spy + physical absence) --------------------------------


class _TruthAccessGuard:
    """Wrap a numpy NpzFile so any access to a truth-side key fails the test."""

    def __init__(self, archive):
        self._archive = archive

    def __getitem__(self, key):
        low = str(key).lower()
        if any(tok in low for tok in TRUTH_KEY_SUBSTRINGS):
            raise AssertionError(f"score-side code read a truth key: {key!r}")
        return self._archive[key]

    def __getattr__(self, name):
        return getattr(self._archive, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._archive.__exit__(*exc)


def _install_truth_spy(monkeypatch):
    real_open = builtins.open
    real_np_load = np.load

    def guarded_open(file, *args, **kwargs):
        if "confirmation_truth" in str(file).replace("\\", "/").lower():
            raise AssertionError(f"score-side code opened the sealed truth file: {file}")
        return real_open(file, *args, **kwargs)

    def guarded_np_load(file, *args, **kwargs):
        if "confirmation_truth" in str(file).replace("\\", "/").lower():
            raise AssertionError(f"score-side code np.load'ed the sealed truth file: {file}")
        return _TruthAccessGuard(real_np_load(file, *args, **kwargs))

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(np, "load", guarded_np_load)


def test_infer_and_score_never_reads_truth_and_truth_absent(tmp_path, monkeypatch):
    """Spy: opening confirmation_truth fails. Fixture: the truth file never exists —
    inference must complete with the sealed truth physically removed/absent."""
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=3)
    truth_path = cinf.role_truth_path(config, "confirm")
    assert not truth_path.exists()
    _install_truth_spy(monkeypatch)
    receipt = cinf.infer_and_score(config, "confirm", **_infer_kwargs())
    assert receipt["status"] == cinf.STATUS_INFERENCE
    assert not truth_path.exists()  # still absent after the full score path


def test_scaler_is_never_refit(tmp_path, monkeypatch):
    """Any fit of the feature scaler in the confirm path is a defect."""
    def _no_fit(*args, **kwargs):
        raise AssertionError("FrozenFeatureScaler.fit called in the confirm path")

    monkeypatch.setattr(s4cpf.FrozenFeatureScaler, "fit", staticmethod(_no_fit))
    monkeypatch.setattr(s4cpf, "fit_feature_scaler", _no_fit)
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=2)
    cinf.infer_and_score(config, "confirm", **_infer_kwargs())


def test_no_training_entrypoint_used(tmp_path, monkeypatch):
    """fit/optimizer entrypoints of the R2 training module are never invoked."""
    def _no_train(*args, **kwargs):
        raise AssertionError("training entrypoint called in the confirm path")

    monkeypatch.setattr(s4cr2t, "fit_five_direct_models", _no_train)
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=2)
    cinf.infer_and_score(config, "confirm", **_infer_kwargs())


# --- infer_and_score output contract ------------------------------------------------------


def test_infer_and_score_outputs_contract(tmp_path):
    config = _config(tmp_path)
    groups = 4
    _write_legal_inputs(config, groups=groups)
    receipt = cinf.infer_and_score(config, "confirm", **_infer_kwargs())
    assert receipt["groups"] == groups and receipt["n_actions"] == NA
    assert receipt["branches"] == groups * NA
    assert receipt["model_seeds"] == [8101, 8202, 8303, 8404, 8505]
    assert receipt["pair_indices"] == [0, 1, 2, 3, 4]

    pred_path, score_path, _ = cinf.role_inference_paths(config, "confirm")
    with np.load(pred_path) as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(score_path) as z:
        scores = {k: z[k] for k in z.files}

    assert predictions["e2e"].shape == (5, groups, NA, 20, 5, 2)
    # five Direct seeds + ensemble + ablations + all baselines (§10 risk_scores.npz).
    assert scores["risk_by_seed"].shape == (5, groups)
    assert scores["risk_ensemble"].shape == (groups,)
    assert np.allclose(scores["risk_ensemble"], scores["risk_by_seed"].mean(axis=0),
                       rtol=0.0, atol=1e-15)
    for key in ("mu_only_ensemble", "class_only_ensemble", "constant_random",
                        "baseline_MARGIN_ONLY", "baseline_SEED_VOTE_ENTROPY",
                        "baseline_SEED_MARGIN_STD", "baseline_COMBINED_TOP2_MAX",
                        "t_star", "interaction_min_distance", "reference_action",
                        "valid_mask", "group_seed"):
        assert key in scores, key
    # no truth-side array, label or adaptive statistic may leave the score layer.
    for archive in (predictions, scores):
        for key in archive:
            low = key.lower()
            for banned in ("regret", "j_true", "label", "y_reg", "y_high",
                           "counterfactual", "oracle", "a_star", "true_"):
                assert banned not in low, key


def test_constant_random_streams_mirror_r2_frozen_constructions(tmp_path):
    """Controller ruling: mirror R2's frozen CONSTANT_RANDOM constructions exactly.

    Group level: SeedSequence([int(group_seed)]).generate_state(1)[0] / 2**32, NO salt
    (exactly stage4cr2_statistics._constant_random_scores). Action level:
    SeedSequence([*action_stream_root, salt]).generate_state(1)[0] (exactly
    stage4cr2_inference._curve_reference_seed with the confirm salt).
    """
    from ac_twin import stage4cr2_statistics as s4cr2s

    config = _config(tmp_path)
    groups = 3
    _write_legal_inputs(config, groups=groups)
    receipt = cinf.infer_and_score(config, "confirm", **_infer_kwargs())
    _, score_path, _ = cinf.role_inference_paths(config, "confirm")
    with np.load(score_path) as z:
        group_seed = np.asarray(z["group_seed"], dtype=np.int64)
        constant_random = np.asarray(z["constant_random"], dtype=np.float64)

    # group level: bitwise equal to R2's frozen helper on the same seeds.
    expected = np.asarray(s4cr2s._constant_random_scores(group_seed), dtype=np.float64)
    assert np.array_equal(constant_random, expected)
    for i, seed in enumerate(group_seed):
        manual = np.random.SeedSequence([int(seed)]).generate_state(1)[0] / float(2**32)
        assert constant_random[i] == manual  # NO salt anywhere in the derivation

    # action level: the role's full action stream root + the frozen salt.
    expected_confirm = int(np.random.SeedSequence(
        [2026093012, 0, 2026093203]).generate_state(1)[0])
    expected_smoke = int(np.random.SeedSequence(
        [2026093100, 12, 2026093203]).generate_state(1)[0])
    assert cinf._constant_random_seed(config, "confirm") == expected_confirm
    assert cinf._constant_random_seed(config, "smoke") == expected_smoke

    # receipt fidelity: both derivations and their seeds are recorded, so every
    # persisted constant-random array is regenerable from the receipt alone.
    streams = receipt["constant_random_streams"]
    assert streams["action_level"]["rng_seed"] == expected_confirm
    assert streams["action_level"]["action_stream_root"] == [2026093012, 0]
    assert streams["action_level"]["constant_random_salt"] == 2026093203
    assert "NO salt" in streams["group_level"]["derivation"]
    assert streams["action_level"]["mirrors"].startswith("stage4cr2_inference")
    assert streams["group_level"]["mirrors"].startswith("stage4cr2_statistics")


def test_infer_and_score_create_once_never_overwrites(tmp_path):
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=2)
    first = cinf.infer_and_score(config, "confirm", **_infer_kwargs())
    _, _, receipt_path = cinf.role_inference_paths(config, "confirm")
    receipt_text = receipt_path.read_text(encoding="utf-8")

    def louder(bundle, *, binding, stage3r_dir, device, batch_size):
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        out["e2e"] = out["e2e"] + 1.0  # a rerun must NOT land on disk
        return out

    second = cinf.infer_and_score(config, "confirm",
                                  **{**_infer_kwargs(), "run_inference_fn": louder})
    assert second["verification"] == "existing_inference_verified_no_overwrite"
    assert second["predictions_sha256"] == first["predictions_sha256"]
    assert receipt_path.read_text(encoding="utf-8") == receipt_text
    pred_path, _, _ = cinf.role_inference_paths(config, "confirm")
    with np.load(pred_path) as z:
        assert float(np.abs(z["e2e"]).max()) == 0.0


def test_infer_and_score_rejects_non_finite_predictions(tmp_path):
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=2)

    def bad(bundle, *, binding, stage3r_dir, device, batch_size):
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        out["e2e"][0, 0, 0, 0, 0, 0] = np.nan
        return out

    with pytest.raises(RuntimeError, match="non-finite"):
        cinf.infer_and_score(config, "confirm",
                             **{**_infer_kwargs(), "run_inference_fn": bad})


def test_smoke_role_isolated_under_smoke_dir(tmp_path):
    config = _config(tmp_path)
    _write_legal_inputs(config, role="smoke", groups=2)
    receipt = cinf.infer_and_score(config, "smoke", **_infer_kwargs())
    assert receipt["status"] == cinf.STATUS_SMOKE
    pred_path, score_path, receipt_path = cinf.role_inference_paths(config, "smoke")
    for path in (pred_path, score_path, receipt_path):
        assert path.is_file()
        assert "smoke" in str(path).replace("\\", "/").split("/")
    # smoke never touches the formal inference artifact paths.
    formal = cinf.role_inference_paths(config, "confirm")
    assert not any(p.exists() for p in formal)


# --- frozen model inventory (read-only construction) ---------------------------------------


def test_load_frozen_model_inventory_real_r2():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    inventory = cinf.load_frozen_model_inventory(config)
    assert isinstance(inventory, s4cr2t.ModelInventory)
    assert tuple(inventory.seeds) == prot.MODEL_SEEDS
    assert inventory.param_count == prot.EXPECTED_TRAINABLE_PARAMETERS
    assert inventory.arch == prot.ARCH_DIRECT
    # paths were relocated then hash-verified; every checkpoint exists on disk.
    for seed in inventory.seeds:
        path = Path(inventory.checkpoint_paths[int(seed)])
        assert path.is_file()
        assert path.parent.name == "models"
    # the scaler comes from the pinned frozen feature_scaler.npz, never refit.
    frozen = s4cpf.FrozenFeatureScaler.load(R2_DEV / "feature_scaler.npz")
    assert np.array_equal(inventory.scaler.token_cont_mean, frozen.token_cont_mean)
    assert np.array_equal(inventory.scaler.token_cont_std, frozen.token_cont_std)
    assert np.array_equal(inventory.scaler.global_mean, frozen.global_mean)
    assert np.array_equal(inventory.scaler.global_std, frozen.global_std)


def test_load_frozen_model_inventory_detects_checkpoint_drift(tmp_path):
    import shutil
    fake = tmp_path / "r2dev"
    (fake / "models").mkdir(parents=True)
    for name in ("final_selection.json", "model_inventory.json", "feature_scaler.npz"):
        shutil.copy2(R2_DEV / name, fake / name)
    for seed in prot.MODEL_SEEDS:
        shutil.copy2(R2_DEV / "models" / f"direct_seed{seed}.pt",
                     fake / "models" / f"direct_seed{seed}.pt")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["r2_development_dir"] = str(fake)
    inventory = cinf.load_frozen_model_inventory(config)  # clean copy loads
    assert inventory.param_count == 15740
    target = fake / "models" / "direct_seed8101.pt"
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        cinf.load_frozen_model_inventory(config)


# --- adapter consistency: R2 train first-32 groups, legal inputs only --------------------


def test_adapter_curve_seed_matches_r2_train_role():
    """The adapter check reuses R2's frozen train-role CONSTANT_RANDOM stream root."""
    import collect_stage4cr2_merge_multiaction as r2_collector
    from ac_twin import stage4cr2_inference as s4cr2i
    r2_config = s4cr2.validate_config(json.loads(
        (ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json")
        .read_text(encoding="utf-8")))
    expected = s4cr2i._curve_reference_seed(r2_config, "development_train")
    assert cinf._r2_train_curve_reference_seed() == expected
    assert cinf._R2_CURVE_REFERENCE_SALT == s4cr2i._CURVE_REFERENCE_SALT
    assert r2_collector.action_stream_root(r2_config, "development_train") == (
        s4cr2.DEV_SEED_ENTROPY, s4cr2.TRAIN_SEGMENT[0])


def test_compare_adapter_outputs_detects_drift():
    """The comparator itself fails closed on any prediction/feature drift."""
    rng = np.random.default_rng(0)
    reference = {k: rng.normal(size=(5, 2, NA, 20, 5, 2)).astype(np.float32)
                 for k in ("e2e", "full", "history")}
    reference.update({k: rng.normal(size=(5, 2, NA, 20, 5)).astype(np.float32)
                      for k in ("r_hat", "presence_prob", "high_prob")})
    ours = {k: v.copy() for k, v in reference.items()}
    cinf._compare_prediction_arrays(ours, reference)  # identical passes
    ours["e2e"][0, 0, 0, 0, 0, 0] += 1e-6
    with pytest.raises(RuntimeError):
        cinf._compare_prediction_arrays(ours, reference)


def test_adapter_consistency_r2_train_first32(monkeypatch):
    """Real frozen chain: 32 R2 train groups, legal inputs only, bitwise vs on-disk.

    Reads ONLY the legal input keys of the R2 train archive (the counterfactual-truth
    key is never accessed — a guarded np.load proves it) and R2's on-disk predictions/
    scores. No CF truth, no label, no new scientific metric.
    """
    if not R2_TRAIN_DATA.is_file():
        pytest.skip("R2 train data not present")
    real_np_load = np.load

    def guarded_np_load(file, *args, **kwargs):
        return _TruthAccessGuard(real_np_load(file, *args, **kwargs))

    monkeypatch.setattr(np, "load", guarded_np_load)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    receipt = cinf.verify_adapter_consistency(config, n_groups=32)
    assert receipt["n_groups"] == 32
    assert receipt["predictions_bitwise_identical"] is True
    assert receipt["action_scores_bitwise_identical"] is True
    assert receipt["features_bitwise_identical"] is True
    assert receipt["direct_scores_bitwise_identical"] is True
    assert receipt["max_abs_diff"] == 0.0
