"""Stage 4D D1 frozen inference/scoring + recovery-chain tests (plan §6.4, §11 6/8, §13).

Plan ``AC-TWIN-RFER-STAGE4D-D0D1-OOD-2026-09-20-V1.0`` §10 Task 5. The score side reads
ONLY the published legal-input file (``inference_inputs.npz``); the sealed
``confirmation_truth.npz`` is never opened (spy-pinned). The risk-score math is the frozen
Stage 4C-R2 implementation reused read-only (risk formula pinned verbatim on constructed
tensors); the D1 action-level CONSTANT_RANDOM stream derives from the D1 action-stream
root [2026110994, 4] + the frozen salt. The §13 recovery chain extends the locked code
registry (cinf + the new adapter + the runner phases) without overwriting the write-once
lock. ``confirmation_used.lock`` must NOT exist at any score-side step (Task 6 only).
"""
from __future__ import annotations

import builtins
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_stage4d_ood as runner  # noqa: E402
from ac_twin import stage4cr2_confirm_inference as cinf  # noqa: E402  frozen, read-only
from ac_twin import stage4cr2_confirm_protocol as r2confirm  # noqa: E402  frozen
from ac_twin import stage4cr2_training as s4cr2t  # noqa: E402  frozen scoring entry
from ac_twin import stage4d_confirm_protocol as confirm_prot  # noqa: E402
from ac_twin import stage4d_ood_inference as d1inf  # noqa: E402  Task 5 adapter
from ac_twin import stage4d_protocol as prot  # noqa: E402
from ac_twin import stage4cp_models as s4cm  # noqa: E402  frozen Direct model
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402  RISK_CLASSES
from ac_twin.stage1_data import sha256_file  # noqa: E402

NA = prot.N_ACTIONS
TRUTH_KEY_SUBSTRINGS = cinf.FORBIDDEN_INPUT_KEY_TOKENS


# --- fixtures -----------------------------------------------------------------------------


def _config(tmp_path) -> dict:
    """The frozen Stage 4D config with run/data roots and output paths under tmp."""
    config = json.loads(prot.CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run" / "stage4d_ood"
    config["run_dir"] = str(run_dir)
    config["data_root"] = str(tmp_path / "data" / "object_centric_stage"
                              / "stage4d_merge_ood")
    config["paths"] = dict(config["paths"])
    prefix = "experiments/stage4_reliability/stage4d_ood/"
    for key in list(config["paths"]):
        value = str(config["paths"][key])
        if value.startswith(prefix):
            config["paths"][key] = str(run_dir / value[len(prefix):])
    return config


def _write_legal_inputs(config, groups=4):
    """A minimal synthetic legal-inputs npz exactly where the D1 score side expects it."""
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
    input_path = Path(prot.validate_config(config)["paths"]["inference_inputs"])
    input_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        input_path, object_history=history, object_factual_future=factual,
        factual_actions=factual_actions, cf_actions=cf_actions,
        cf_durations=cf_durations, t_star=np.full(groups, 12, dtype=np.int64),
        group_seed=np.arange(100, 100 + groups, dtype=np.int64),
        inference_mask=inference_mask)
    return input_path


def _write_d1_lock(config, *, selected_n=2000):
    """Stand-in D1 protocol lock binding the running config + both code registries."""
    checked = prot.validate_config(config)
    payload = {
        "plan_id": prot.PLAN_ID,
        "config_signature": prot._config_signature(checked),
        "selected_n": int(selected_n),
        "ood_profile": "JOINT_OOD",
        "code_sha256": prot.protocol_code_hashes(),
        "extended_code_sha256": confirm_prot.confirm_code_hashes(),
    }
    lock_path = Path(checked["run_dir"]) / "protocol" / "protocol_lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    return lock_path


def _write_collection_receipt(config, groups, inputs_path):
    """A completed COLLECTED D1 collection receipt (carries the sealed-truth hash only)."""
    checked = prot.validate_config(config)
    truth_path = Path(checked["paths"]["confirmation_truth"])
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(inputs_path) as z:
        seeds = np.asarray(z["group_seed"], dtype=np.int64)
        cf_actions = np.asarray(z["cf_actions"])
        cf_durations = np.asarray(z["cf_durations"])
        factual_actions = np.asarray(z["factual_actions"])
        t_star = np.asarray(z["t_star"])
        mask = np.asarray(z["inference_mask"])
    np.savez_compressed(
        truth_path,
        object_counterfactual_futures=np.zeros((groups, NA, 20, 5, 8), np.float32),
        cf_actions=cf_actions, cf_durations=cf_durations,
        factual_actions=factual_actions, t_star=t_star, group_seed=seeds,
        inference_mask=mask)
    receipt = {
        "plan_id": prot.PLAN_ID,
        "role": "d1_confirm",
        "status": "COLLECTED",
        "accepted_groups": int(groups),
        "inference_inputs_sha256": sha256_file(inputs_path),
        "confirmation_truth_sha256": sha256_file(truth_path),
        "truth_fields": ["object_counterfactual_futures", "cf_actions", "cf_durations",
                         "factual_actions", "t_star", "group_seed", "inference_mask"],
        "group_order_binding": {
            "order": "acceptance_order",
            "n_groups": int(groups),
            "inference_inputs_group_seed_sha256": hashlib.sha256(
                seeds.tobytes()).hexdigest(),
            "confirmation_truth_group_seed_sha256": hashlib.sha256(
                seeds.tobytes()).hexdigest(),
            "same_order": True,
        },
    }
    receipt_path = Path(checked["paths"]["collection_receipt"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return receipt_path


# --- fake frozen-chain callables (the real ones need the full checkpoint chain) -----------


def _fake_predictions(bundle, *, binding, stage3r_dir, device, batch_size):
    groups = int(bundle["groups"])
    shape6 = (5, groups, NA, 20, 5, 2)
    shape5 = (5, groups, NA, 20, 5)
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
    scores = {k: np.zeros((groups, n_actions), dtype=np.float64)
              for k in s4cpp.RISK_CLASSES}
    scores["CONSTANT_RANDOM"] = np.full((groups, n_actions), 0.5)
    return scores


def _fake_score_fn(inventory, bundle, *, device):
    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64).ravel()
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


def _hooks():
    return {
        "run_inference_fn": _fake_predictions,
        "build_features_fn": _fake_features,
        "load_models_fn": _fake_frozen_models,
        "score_actions_fn": _fake_score_actions,
        "score_fn": _fake_score_fn,
    }


def _build_score_side(tmp_path, groups=4):
    """Full synthetic score side: datasets + lock + collection receipt + inference."""
    config = _config(tmp_path)
    inputs_path = _write_legal_inputs(config, groups=groups)
    _write_d1_lock(config, selected_n=groups)
    _write_collection_receipt(config, groups, inputs_path)
    receipt = d1inf.infer_and_score_d1(config, **_hooks())
    return {"config": config, "inputs": inputs_path, "inference_receipt": receipt}


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


# --- engine binding (the frozen stage4cr2 chain the adapter reuses read-only) --------------


def test_engine_config_bound_to_stage4cr2_lock_chain():
    """The engine config is the frozen stage4cr2 confirm config, validated and chain-bound."""
    engine = d1inf.load_engine_config()
    assert engine["plan_id"] == "AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1"
    assert engine["architecture"] == "SAME_INFORMATION_DIRECT"
    # cinf (the reused frozen scoring module) is hash-bound by the stage4cr2 lock chain.
    chain = r2confirm.resolve_locked_code_hashes(
        r2confirm.ROOT / "experiments/stage4_reliability/stage4cr2_confirm/protocol"
        / "protocol_lock.json")
    assert chain["ac_twin/stage4cr2_confirm_inference.py"] == sha256_file(
        CODE / "ac_twin/stage4cr2_confirm_inference.py")


def test_registry_extension_covers_cinf_adapter_runner():
    """Controller ruling: the recovery-chain registry extension binds cinf + the new code."""
    registry = set(confirm_prot.CONFIRM_REGISTRY)
    assert "ac_twin/stage4cr2_confirm_inference.py" in registry  # cinf
    assert "ac_twin/stage4d_ood_inference.py" in registry  # this stage's thin adapter
    assert "run_stage4d_ood.py" in registry
    assert "ac_twin/stage4d_confirm_protocol.py" in registry


# --- truth isolation (plan §11 item 6) -----------------------------------------------------


def test_infer_d1_refuses_truth_path(tmp_path):
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    truth_path = Path(checked["paths"]["confirmation_truth"])
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(truth_path, object_counterfactual_futures=np.zeros((2, NA, 20, 5, 8)))
    config["paths"]["inference_inputs"] = str(truth_path)  # pointed at the sealed truth
    with pytest.raises(ValueError):
        d1inf.infer_and_score_d1(config, **_hooks())


def test_infer_d1_rejects_truth_keys_in_input_archive(tmp_path):
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
            d1inf.infer_and_score_d1(config, **_hooks())
    np.savez(input_path, **legal)  # restore


def test_infer_d1_never_reads_truth_and_truth_absent(tmp_path, monkeypatch):
    """Spy: opening confirmation_truth fails. Fixture: the truth file never exists."""
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=3)
    truth_path = Path(prot.validate_config(config)["paths"]["confirmation_truth"])
    assert not truth_path.exists()
    _install_truth_spy(monkeypatch)
    receipt = d1inf.infer_and_score_d1(config, **_hooks())
    assert receipt["status"] == d1inf.STATUS_INFERENCE
    assert not truth_path.exists()  # still absent after the full score path


def test_infer_d1_reads_only_the_eight_legal_keys(tmp_path, monkeypatch):
    config = _config(tmp_path)
    input_path = _write_legal_inputs(config, groups=2)
    accessed = []
    real_np_load = np.load

    class Recorder(_TruthAccessGuard):
        def __getitem__(self, key):
            accessed.append(str(key))
            return super().__getitem__(key)

    def recording_np_load(file, *args, **kwargs):
        archive = real_np_load(file, *args, **kwargs)
        if Path(str(file)) == input_path:
            return Recorder(archive)
        return archive

    monkeypatch.setattr(np, "load", recording_np_load)
    d1inf.infer_and_score_d1(config, **_hooks())
    assert set(accessed) == set(d1inf.LEGAL_INPUT_KEYS)
    assert len(set(accessed)) == 8


# --- output contract + frozen score math ----------------------------------------------------


def test_infer_d1_outputs_contract(tmp_path):
    config = _config(tmp_path)
    groups = 4
    _write_legal_inputs(config, groups=groups)
    receipt = d1inf.infer_and_score_d1(config, **_hooks())
    assert receipt["plan_id"] == prot.PLAN_ID
    assert receipt["role"] == "d1_confirm"
    assert receipt["groups"] == groups and receipt["n_actions"] == NA
    assert receipt["model_seeds"] == [8101, 8202, 8303, 8404, 8505]
    paths = d1inf.inference_artifact_paths(prot.validate_config(config))
    with np.load(paths["predictions"]) as z:
        predictions = {k: z[k] for k in z.files}
    with np.load(paths["risk_scores"]) as z:
        scores = {k: z[k] for k in z.files}
    assert predictions["e2e"].shape == (5, groups, NA, 20, 5, 2)
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


def test_risk_score_formula_verbatim_on_constructed_tensors(tmp_path):
    """risk_s = sigmoid(logit_high)*0.01*expm1(softplus(raw_mu)); ensemble = arithmetic mean.

    Hand-computed from each checkpoint's raw heads on constructed tensors, compared
    bitwise-anchored (allclose at float64) against the adapter's frozen scoring entry
    (``stage4cr2_training.score_models`` — reused read-only, never re-implemented).
    """
    torch = pytest.importorskip("torch")
    from ac_twin import stage4cp_features as s4cpf

    rng = np.random.default_rng(7)
    n = 6
    tokens = rng.normal(size=(n, s4cpp.N_COMPETITORS, s4cpf.TOKEN_DIM))
    glob = rng.normal(size=(n, s4cpf.GLOBAL_DIM))
    mask = np.ones((n, s4cpp.N_COMPETITORS), dtype=bool)
    y_reg = rng.gamma(1.5, 0.02, size=n)
    y_high = (y_reg > 0.01).astype(np.int64)
    cfg = json.loads((ROOT / "experiments/stage4_reliability/stage4c_plus"
                      "/stage4cp_config.json").read_text(encoding="utf-8"))
    cfg["device"] = "cpu"
    checkpoint_paths, checkpoints = {}, {}
    for seed in prot.MODEL_SEEDS:
        ckpt = s4cm.fit_decision_risk_model(
            s4cpp.ARCH_DIRECT, tokens, glob, mask, y_reg, y_high, hidden_dim=64,
            lambda_cls=0.25, seed=int(seed) % 6151, config=cfg, direct_width=43)
        path = tmp_path / f"direct_seed{seed}.pt"
        s4cm.save_checkpoint(ckpt, path)
        checkpoint_paths[int(seed)] = str(path)
        checkpoints[int(seed)] = ckpt
    scaler = s4cpf.FrozenFeatureScaler.fit(tokens, glob)
    inventory = s4cr2t.ModelInventory(
        plan_id="t", arch=s4cpp.ARCH_DIRECT, seeds=tuple(int(s) for s in prot.MODEL_SEEDS),
        param_count=15740, run_dir=str(tmp_path), checkpoint_paths=checkpoint_paths,
        checkpoint_sha256={s: sha256_file(p) for s, p in checkpoint_paths.items()},
        scaler=scaler, pos_weight=1.0, interaction_threshold=29.128996473819946,
        main_indices=(0, 1), monitor_indices=(2,), train_losses={}, monitor_losses={},
        n_train_groups=n, written_at="t")
    scores = s4cr2t.score_models(
        inventory, {"tokens": tokens, "global_features": glob,
                    "group_seed": np.arange(n, dtype=np.int64)}, device="cpu")
    # per-seed: hand-computed from each checkpoint's OWN raw heads (sigmoid/expm1/softplus).
    x_tok = scaler.transform_tokens(tokens)
    x_glob = scaler.transform_global(glob)
    for i, seed in enumerate(prot.MODEL_SEEDS):
        model = s4cm.build_model(s4cpp.ARCH_DIRECT, hidden_dim=64, dropout=0.0,
                                 direct_width=43).eval()
        model.load_state_dict(checkpoints[int(seed)]["state_dict"])
        with torch.no_grad():
            flat = torch.as_tensor(
                scaler.transform_tokens(tokens), dtype=torch.float32).flatten(start_dim=1)
            raw_mu, logit_high = model(
                flat, torch.as_tensor(x_glob, dtype=torch.float32))
        raw_mu = raw_mu.numpy().astype(np.float64)
        logit_high = logit_high.numpy().astype(np.float64)
        mu_pos = np.log1p(np.exp(raw_mu))  # softplus, verbatim
        sig = 1.0 / (1.0 + np.exp(-logit_high))  # sigmoid, verbatim
        expected = sig * 0.01 * np.expm1(mu_pos)  # the frozen primary score, verbatim
        np.testing.assert_allclose(scores.risk_by_seed[i], expected, rtol=1e-10,
                                   atol=1e-12)
        assert scores.seeds[i] == int(seed)
        _ = x_tok  # transform path exercised via predict_decision_risk inside score_models
    # ensemble = the exact arithmetic mean of the five seeds, never anything else.
    np.testing.assert_allclose(scores.risk_ensemble, scores.risk_by_seed.mean(axis=0),
                               rtol=0.0, atol=1e-15)


def test_constant_random_streams_use_d1_root_and_frozen_salt(tmp_path):
    """Action level: SeedSequence([2026110994, 4, salt]); group level: pure group seed."""
    config = _config(tmp_path)
    groups = 3
    _write_legal_inputs(config, groups=groups)
    receipt = d1inf.infer_and_score_d1(config, **_hooks())
    paths = d1inf.inference_artifact_paths(prot.validate_config(config))
    with np.load(paths["risk_scores"]) as z:
        group_seed = np.asarray(z["group_seed"], dtype=np.int64)
        constant_random = np.asarray(z["constant_random"], dtype=np.float64)
    # group level: bitwise equal to the frozen helper on the same seeds (NO salt).
    expected = np.asarray(cinf._constant_random_group_scores(group_seed), dtype=np.float64)
    assert np.array_equal(constant_random, expected)
    # action level: THIS stage's D1 action-stream root + the frozen salt.
    expected_seed = int(np.random.SeedSequence(
        [2026110994, 4, r2confirm.CONSTANT_RANDOM_SALT]).generate_state(1)[0])
    assert d1inf.constant_random_seed_d1(prot.validate_config(config)) == expected_seed
    streams = receipt["constant_random_streams"]
    assert streams["action_level"]["rng_seed"] == expected_seed
    assert streams["action_level"]["action_stream_root"] == [2026110994, 4]
    assert streams["action_level"]["constant_random_salt"] == r2confirm.CONSTANT_RANDOM_SALT


def test_infer_d1_create_once_never_overwrites(tmp_path):
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=2)
    first = d1inf.infer_and_score_d1(config, **_hooks())
    paths = d1inf.inference_artifact_paths(prot.validate_config(config))
    receipt_text = paths["inference_receipt"].read_text(encoding="utf-8")

    def louder(bundle, *, binding, stage3r_dir, device, batch_size):
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        out["e2e"] = out["e2e"] + 1.0  # a rerun must NOT land on disk
        return out

    second = d1inf.infer_and_score_d1(config, **{**_hooks(), "run_inference_fn": louder})
    assert second["verification"] == "existing_inference_verified_no_overwrite"
    assert second["predictions_sha256"] == first["predictions_sha256"]
    assert paths["inference_receipt"].read_text(encoding="utf-8") == receipt_text


def test_infer_d1_refuses_when_confirmation_used_lock_exists(tmp_path):
    config = _config(tmp_path)
    _write_legal_inputs(config, groups=2)
    used_lock = Path(prot.validate_config(config)["paths"]["confirmation_used_lock"])
    used_lock.parent.mkdir(parents=True, exist_ok=True)
    used_lock.write_text("burned", encoding="utf-8")  # Task-6-only artifact
    with pytest.raises(ValueError, match="confirmation_used"):
        d1inf.infer_and_score_d1(config, **_hooks())


# --- determinism recheck (plan §6.4 step 2) -------------------------------------------------


def test_determinism_recheck_create_once_and_formal_slice_binding(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    paths = d1inf.inference_artifact_paths(prot.validate_config(config))
    receipt = d1inf.run_determinism_recheck_d1(config, **_hooks())
    assert receipt["bitwise_identical"] is True
    assert receipt["n_groups_rechecked"] == 4  # min(DETERMINISM_GROUPS, G)
    assert receipt["context_rows"] == 4 * NA  # the role's full branch count as context
    assert receipt["direct_context_rows"] == 4  # the formal group count
    assert paths["determinism_receipt"].is_file()
    text = paths["determinism_receipt"].read_text(encoding="utf-8")
    again = d1inf.run_determinism_recheck_d1(config, **_hooks())
    assert again["verification"] == "existing_determinism_verified_no_overwrite"
    assert paths["determinism_receipt"].read_text(encoding="utf-8") == text

    # a non-deterministic chain is a hard stop, never a tolerance.
    fixture = _build_score_side(tmp_path / "b")
    calls = {"n": 0}

    def noisy(bundle, *, binding, stage3r_dir, device, batch_size):
        calls["n"] += 1
        out = _fake_predictions(bundle, binding=binding, stage3r_dir=stage3r_dir,
                                device=device, batch_size=batch_size)
        if calls["n"] % 2 == 0:
            out["e2e"] = out["e2e"] + 1e-6
        return out

    with pytest.raises(RuntimeError):
        d1inf.run_determinism_recheck_d1(fixture["config"],
                                         **{**_hooks(), "run_inference_fn": noisy})

    # recheck results must match the FORMAL artifacts' leading slices: tamper -> refuse.
    fixture = _build_score_side(tmp_path / "c")
    config = fixture["config"]
    paths = d1inf.inference_artifact_paths(prot.validate_config(config))
    with np.load(paths["predictions"]) as z:
        payload = {k: z[k] for k in z.files}
    payload["e2e"] = payload["e2e"] + 1e-6
    np.savez_compressed(paths["predictions"], **payload)
    with pytest.raises((ValueError, RuntimeError)):
        d1inf.run_determinism_recheck_d1(config, **_hooks())


def _context_sensitive_score_fn(inventory, bundle, *, device):
    """Emulates the row-count-context-dependent Direct forward (every row shifts with M)."""
    tokens = np.asarray(bundle["tokens"], dtype=np.float64)
    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64).ravel()
    m = tokens.shape[0]
    base = (group_seed % 97).astype(np.float64) / 97.0 + 1e-9 * m
    by_seed = np.stack([base + 0.001 * i for i in range(5)])
    return s4cr2t.RiskScores(
        group_seed=group_seed,
        risk_ensemble=by_seed.mean(axis=0),
        risk_by_seed=by_seed,
        mu_only_ensemble=base * 0.5,
        class_only_ensemble=base * 0.25,
        seeds=tuple(int(s) for s in prot.MODEL_SEEDS),
    )


def test_determinism_recheck_pads_direct_scoring_to_formal_context(tmp_path):
    """With a row-count-context-sensitive Direct forward, only the padded recheck matches."""
    fixture = _build_score_side(tmp_path, groups=5)
    config = fixture["config"]
    hooks = {**_hooks(), "score_fn": _context_sensitive_score_fn}
    # rebuild the formal side WITH the context-sensitive scoring (M=5 context).
    fixture = _build_score_side(tmp_path / "d", groups=5)
    config = fixture["config"]
    # the formal artifacts above were written by _fake_score_fn; rewrite them with the
    # context-sensitive fn so the formal M=5 context differs from a standalone M=4.
    config2 = fixture["config"]
    paths = d1inf.inference_artifact_paths(prot.validate_config(config2))
    for key in ("predictions", "risk_scores", "inference_receipt"):
        paths[key].unlink()
    d1inf.infer_and_score_d1(config2, **hooks)
    receipt = d1inf.run_determinism_recheck_d1(config2, n_groups=4, **hooks)
    assert receipt["bitwise_identical"] is True
    assert receipt["direct_context_rows"] == 5  # padded to the formal group context


# --- score freeze (plan §6.4 step 3) ---------------------------------------------------------


def test_freeze_create_once_and_full_binding(tmp_path):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    checked = prot.validate_config(config)
    paths = d1inf.inference_artifact_paths(checked)
    d1inf.run_determinism_recheck_d1(config, **_hooks())
    receipt = d1inf.freeze_scores_d1(config)
    assert receipt["plan_id"] == prot.PLAN_ID
    assert receipt["n_groups"] == 4
    assert receipt["primary_baseline"] == "SEED_VOTE_ENTROPY"
    assert receipt["standard_baselines"] == ["MARGIN_ONLY", "SEED_VOTE_ENTROPY",
                                             "SEED_MARGIN_STD"]
    assert receipt["bootstrap"]["entropy"] == prot.D1_BOOTSTRAP_ENTROPY == 2026110995
    bindings = receipt["bindings"]
    assert bindings["predictions_sha256"] == sha256_file(paths["predictions"])
    assert bindings["risk_scores_sha256"] == sha256_file(paths["risk_scores"])
    assert bindings["determinism_receipt_sha256"] == sha256_file(
        paths["determinism_receipt"])
    assert bindings["protocol_lock_sha256"] == sha256_file(paths["protocol_lock"])
    assert bindings["expected_confirmation_truth_sha256"]  # receipt-carried, never read
    assert paths["score_freeze_receipt"].is_file()
    text = paths["score_freeze_receipt"].read_text(encoding="utf-8")
    again = d1inf.freeze_scores_d1(config)
    assert again["verification"] == "existing_score_freeze_verified_no_overwrite"
    assert paths["score_freeze_receipt"].read_text(encoding="utf-8") == text


def test_freeze_refuses_without_determinism_or_with_used_lock(tmp_path):
    fixture = _build_score_side(tmp_path / "a")
    with pytest.raises(ValueError, match="determinism"):
        d1inf.freeze_scores_d1(fixture["config"])  # determinism receipt missing

    fixture = _build_score_side(tmp_path / "b")
    config = fixture["config"]
    d1inf.run_determinism_recheck_d1(config, **_hooks())
    used_lock = Path(prot.validate_config(config)["paths"]["confirmation_used_lock"])
    used_lock.parent.mkdir(parents=True, exist_ok=True)
    used_lock.write_text("burned", encoding="utf-8")
    with pytest.raises(ValueError, match="confirmation_used"):
        d1inf.freeze_scores_d1(config)


def test_freeze_reverification_allowed_after_lock_burned(tmp_path):
    """Re-verifying an existing freeze receipt AFTER the one-shot lock burned is legal.

    Regression for a latent defect found at the Task 6 burn: the lock-unused guard must
    guard only a FIRST freeze (the create-once receipt already prevents any re-freeze). The
    Task 6 judge re-verifies the frozen score side around/after the burn, so
    ``freeze_scores_d1`` must not be blocked by the burned lock once the receipt exists.
    """
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    checked = prot.validate_config(config)
    paths = d1inf.inference_artifact_paths(checked)
    d1inf.run_determinism_recheck_d1(config, **_hooks())
    d1inf.freeze_scores_d1(config)   # the create-once freeze (receipt now exists)
    # the one-shot unseal lock burns (score side already frozen); the judge re-verifies.
    paths["used_lock"].parent.mkdir(parents=True, exist_ok=True)
    paths["used_lock"].write_text(json.dumps({"plan_id": prot.PLAN_ID}), encoding="utf-8")
    again = d1inf.freeze_scores_d1(config)   # must NOT be blocked by the burned lock
    assert again["verification"] == "existing_score_freeze_verified_no_overwrite"


def test_freeze_refuses_lock_code_drift(tmp_path, monkeypatch):
    fixture = _build_score_side(tmp_path)
    config = fixture["config"]
    d1inf.run_determinism_recheck_d1(config, **_hooks())
    # drift a registered code file -> the freeze's lock binding fails closed.
    target = CODE / "collect_stage4d_merge_ood.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# drift\n")
        with pytest.raises(ValueError, match="code|drift|hash|mismatch"):
            d1inf.freeze_scores_d1(config)
    finally:
        target.write_bytes(original)


# --- §13 recovery chain (registry extension; the write-once lock is never overwritten) ------


def _tmp_lock_env(tmp_path, monkeypatch):
    """A tmp D1 lock binding the CURRENT on-disk registries (recovery-chain testbed)."""
    config = _config(tmp_path)
    checked = prot.validate_config(config)
    real_lock = json.loads(
        (prot.ROOT / "experiments/stage4_reliability/stage4d_ood/protocol"
         / "protocol_lock.json").read_text(encoding="utf-8"))
    power_path = Path(checked["paths"]["power_report"])
    power_path.parent.mkdir(parents=True, exist_ok=True)
    power_path.write_text(json.dumps({
        "plan_id": prot.PLAN_ID, "status": prot.STATUS_D1_POWER_FROZEN,
        "selected_n": 2000, "underpowered": False}), encoding="utf-8")
    selection_path = Path(checked["paths"]["ood_profile_selection"])
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps({
        "plan_id": prot.PLAN_ID, "selected_profile": "JOINT_OOD",
        "status": prot.STATUS_D0_PROFILE_SELECTED}), encoding="utf-8")
    code_inventory_path = Path(checked["paths"]["code_inventory"])
    code_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    code_inventory_path.write_text(json.dumps({"marker": "tmp"}), encoding="utf-8")
    for key in ("source_inventory", "seed_inventory"):
        sidecar = Path(checked["paths"][key])
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({"marker": key}), encoding="utf-8")
    payload = {
        "plan_id": prot.PLAN_ID,
        "config_signature": prot._config_signature(checked),
        "selected_n": 2000,
        "ood_profile": "JOINT_OOD",
        "ood_profile_selection_sha256": sha256_file(selection_path),
        "power_report_sha256": sha256_file(power_path),
        "code_inventory_sha256": sha256_file(code_inventory_path),
        "strata": real_lock["strata"],
        "frozen_source_inventory": real_lock["frozen_source_inventory"],
        "seed_audit": {"zero_overlap": True, "overlap": []},
        "code_sha256": prot.protocol_code_hashes(),
        "extended_code_sha256": confirm_prot.confirm_code_hashes(),
    }
    lock_path = Path(checked["run_dir"]) / "protocol" / "protocol_lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(prot, "CONFIG_PATH", config_path)
    audit = {"zero_overlap": True, "overlap": [], "mutual_disjoint": True,
             "per_domain": {}, "inventory_sha256": "b" * 64, "n_inventory": 1}
    monkeypatch.setattr(confirm_prot, "run_real_seed_audit",
                        lambda cfg, roots=None: (audit, {}))
    return config, lock_path


def _write_recovery(lock_path, *, changed=None, added=None, binds=None,
                    name="recovery_lock_r1.json", kind="pure_engineering_recovery",
                    declared=None):
    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    code = dict(payload["extended_code_sha256"])
    for rel, record in (changed or {}).items():
        code[rel] = record["after_sha256"]
    for rel, record in (added or {}).items():
        code[rel] = record["sha256"]
    recovery = {
        "plan_id": prot.PLAN_ID,
        "kind": kind,
        "reason": "test recovery",
        "binds_previous_lock_sha256": binds or sha256_file(lock_path),
        "changed_files": changed or {},
        "added_files": added or {},
        "extended_code_sha256": declared if declared is not None else code,
    }
    path = Path(lock_path).parent / name
    path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def test_resolve_without_recovery_returns_locked_extended_set(tmp_path, monkeypatch):
    _config_env, lock_path = _tmp_lock_env(tmp_path, monkeypatch)
    assert confirm_prot.recovery_lock_paths(lock_path) == []
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert confirm_prot.resolve_locked_extended_code_hashes(lock_path) == \
        dict(payload["extended_code_sha256"])


def test_recovery_chain_roundtrip_binds_runner_repair(tmp_path, monkeypatch):
    """A drifted stage-own file fails closed; the recovery lock re-binds it; lock untouched."""
    config, lock_path = _tmp_lock_env(tmp_path, monkeypatch)
    probe = CODE / "_stage4d_recovery_probe.py"
    probe.write_text("# stage4d recovery probe v1\n", encoding="utf-8")
    monkeypatch.setattr(confirm_prot, "CONFIRM_CODE_FILES",
                        tuple(confirm_prot.CONFIRM_CODE_FILES) + (probe.name,))
    monkeypatch.setattr(confirm_prot, "CONFIRM_REGISTRY",
                        tuple(confirm_prot.CONFIRM_REGISTRY) + (probe.name,))
    try:
        # re-bind the lock to a registry that includes the probe (v1).
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        payload["extended_code_sha256"] = confirm_prot.confirm_code_hashes()
        lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        v1 = sha256_file(probe)
        probe.write_text("# stage4d recovery probe v2 - repaired after lock\n",
                         encoding="utf-8")
        v2 = sha256_file(probe)
        with pytest.raises(ValueError):  # drift without a recovery lock: fail closed
            confirm_prot.verify_confirm_protocol_lock(Path(config["run_dir"]))
        created = confirm_prot.create_recovery_lock(
            config, changed_files={probe.name: "probe repair"},
            added_files={}, reason="unit-test recovery",
            base_verify_fn=lambda run_dir: {})
        assert Path(created).name == "recovery_lock_r1.json"
        resolved = confirm_prot.resolve_locked_extended_code_hashes(lock_path)
        assert resolved[probe.name] == v2
        # the original write-once lock file still binds v1 (never overwritten).
        assert json.loads(lock_path.read_text(
            encoding="utf-8"))["extended_code_sha256"][probe.name] == v1
        verified = confirm_prot.verify_confirm_protocol_lock(Path(config["run_dir"]))
        assert verified["plan_id"] == prot.PLAN_ID
    finally:
        probe.unlink(missing_ok=True)
        created_path = lock_path.parent / "recovery_lock_r1.json"
        created_path.unlink(missing_ok=True)


def test_recovery_chain_refuses_tampered_binding(tmp_path, monkeypatch):
    _config_env, lock_path = _tmp_lock_env(tmp_path, monkeypatch)
    rel = "run_stage4d_ood.py"
    current = confirm_prot.confirm_code_hashes()[rel]
    _write_recovery(lock_path, changed={rel: {"before_sha256": current,
                                              "after_sha256": "1" * 64}},
                    binds="f" * 64)
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)


def test_recovery_chain_refuses_before_after_and_declared_mismatch(tmp_path, monkeypatch):
    _config_env, lock_path = _tmp_lock_env(tmp_path, monkeypatch)
    rel = "run_stage4d_ood.py"
    current = confirm_prot.confirm_code_hashes()[rel]
    # before mismatch
    _write_recovery(lock_path, changed={rel: {"before_sha256": "0" * 64,
                                              "after_sha256": "1" * 64}})
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    # no-op change
    _write_recovery(lock_path, changed={rel: {"before_sha256": current,
                                              "after_sha256": current}})
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    # declared set disagrees with the applied changes
    _write_recovery(lock_path, changed={rel: {"before_sha256": current,
                                              "after_sha256": "1" * 64}},
                    declared={})
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)


def test_recovery_chain_refuses_non_own_or_base_or_duplicate_changes(tmp_path, monkeypatch):
    _config_env, lock_path = _tmp_lock_env(tmp_path, monkeypatch)
    hashes = confirm_prot.confirm_code_hashes()
    # a frozen reused dependency can never be re-bound (changed).
    frozen_rel = "ac_twin/stage4c_inference.py"  # in CONFIRM_EXTRA_REUSED_FROZEN_CODE_FILES
    assert frozen_rel in hashes
    _write_recovery(lock_path, changed={frozen_rel: {
        "before_sha256": hashes[frozen_rel], "after_sha256": "2" * 64}})
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    # an unregistered file can never be re-bound.
    _write_recovery(lock_path, changed={"ac_twin/not_in_lock.py": {
        "before_sha256": "0" * 64, "after_sha256": "1" * 64}})
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    # an added file already in the locked set is refused.
    _write_recovery(lock_path, added={"run_stage4d_ood.py": {
        "sha256": "3" * 64, "role": "stage4d_own", "diff_summary": "dup"}})
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    # an added file in the BASE registry is refused (the base set is immutable).
    base_rel = next(iter(prot.PROTOCOL_CODE_FILES))
    _write_recovery(lock_path, added={base_rel: {
        "sha256": "4" * 64, "role": "stage4d_own", "diff_summary": "base"}})
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    # wrong kind / non-contiguous chain.
    rel = "run_stage4d_ood.py"
    _write_recovery(lock_path, changed={rel: {"before_sha256": hashes[rel],
                                              "after_sha256": "1" * 64}},
                    kind="silent_rewrite")
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    (Path(lock_path).parent / "recovery_lock_r1.json").unlink()
    _write_recovery(lock_path, changed={rel: {"before_sha256": hashes[rel],
                                              "after_sha256": "1" * 64}},
                    name="recovery_lock_r2.json")  # gap: no r1
    with pytest.raises(ValueError):
        confirm_prot.recovery_lock_paths(lock_path)


def test_recovery_chain_reused_frozen_addition_cross_bound(tmp_path, monkeypatch):
    """A reused-frozen addition (cinf) must equal the stage4cr2 chain-resolved hash."""
    _config_env, lock_path = _tmp_lock_env(tmp_path, monkeypatch)
    cinf_rel = "ac_twin/stage4cr2_confirm_inference.py"
    # pretend the locked set predates the cinf registration.
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    locked = dict(payload["extended_code_sha256"])
    locked.pop(cinf_rel)
    payload["extended_code_sha256"] = locked
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    real = sha256_file(CODE / cinf_rel)
    declared = dict(locked)
    declared[cinf_rel] = real
    _write_recovery(lock_path, added={cinf_rel: {
        "sha256": real, "role": "reused_frozen",
        "bound_by": "stage4cr2_confirm_lock_chain", "diff_summary": "register cinf"}},
        declared=declared)
    resolved = confirm_prot.resolve_locked_extended_code_hashes(lock_path)
    assert resolved[cinf_rel] == real
    # a wrong hash for the reused-frozen addition fails closed.
    (lock_path.parent / "recovery_lock_r1.json").unlink()
    declared_bad = dict(locked)
    declared_bad[cinf_rel] = "9" * 64
    _write_recovery(lock_path, added={cinf_rel: {
        "sha256": "9" * 64, "role": "reused_frozen",
        "bound_by": "stage4cr2_confirm_lock_chain", "diff_summary": "bad"}},
        declared=declared_bad)
    with pytest.raises(ValueError):
        confirm_prot.resolve_locked_extended_code_hashes(lock_path)


# --- runner phase wiring ---------------------------------------------------------------------


def test_runner_implements_task5_phases():
    for phase in ("collect-d1", "inference-scoring", "score-freeze"):
        assert phase in runner.PIPELINE
        assert phase in runner._PHASE_FUNCTIONS
    # Task 6 implemented burn-confirmation-lock / adjudicate / reports-status-sync: every
    # §12 phase is now registered (none left unimplemented, none silently skipped).
    for phase in ("burn-confirmation-lock", "adjudicate", "reports-status-sync"):
        assert phase in runner.PIPELINE
        assert phase in runner._PHASE_FUNCTIONS
        assert callable(runner._PHASE_FUNCTIONS[phase])


def test_runner_collect_d1_calls_collector_with_d1_role(tmp_path, monkeypatch):
    calls = []

    def fake_collect_role(config, role, **kwargs):
        calls.append(role)
        return {"role": role, "profile": "JOINT_OOD", "status": "COLLECTED",
                "accepted_groups": 2000, "attempts": 6800, "partial": False}

    monkeypatch.setattr(runner.collector, "collect_role", fake_collect_role)
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    monkeypatch.setattr(runner.prot, "verify_protocol_lock", lambda run_dir: {
        "selected_n": 2000, "ood_profile": "JOINT_OOD"})
    monkeypatch.setattr(runner.confirm_prot, "verify_confirm_protocol_lock",
                        lambda run_dir: {"selected_n": 2000})
    out = runner.run_phase("collect-d1", config={
        "run_dir": str(tmp_path), "data_root": "y",
        "paths": {"confirmation_used_lock": str(tmp_path / "confirmation_used.lock")}})
    assert calls == ["d1_confirm"]
    assert out["accepted_groups"] == 2000


def test_runner_collect_d1_refuses_existing_confirmation_used_lock(tmp_path, monkeypatch):
    used_lock = tmp_path / "confirmation_used.lock"
    used_lock.write_text("burned", encoding="utf-8")
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    with pytest.raises(ValueError, match="confirmation_used"):
        runner.run_phase("collect-d1", config={
            "run_dir": str(tmp_path), "data_root": "y",
            "paths": {"confirmation_used_lock": str(used_lock)}})


def test_runner_inference_and_freeze_phases_call_adapter(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner.prot, "validate_config", lambda cfg: dict(cfg))
    monkeypatch.setattr(runner.confirm_prot, "verify_confirm_code_chain",
                        lambda run_dir: {"ok": True})
    monkeypatch.setattr(runner.d1_inference, "infer_and_score_d1",
                        lambda config: calls.append("infer") or {"groups": 2000})
    monkeypatch.setattr(runner.d1_inference, "run_determinism_recheck_d1",
                        lambda config: calls.append("recheck") or {"bitwise_identical": True})
    monkeypatch.setattr(runner.d1_inference, "freeze_scores_d1",
                        lambda config: calls.append("freeze") or {"n_groups": 2000})
    config = {"run_dir": str(tmp_path), "data_root": "y",
              "paths": {"confirmation_used_lock": str(tmp_path / "no.lock")}}
    runner.run_phase("inference-scoring", config)
    runner.run_phase("score-freeze", config)
    assert calls == ["infer", "recheck", "freeze"]
