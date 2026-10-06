"""Stage 4C-R2 one-shot screen order tests (plan sections 9.3, 11, 14; task-4 brief).

Blocking invariants proved here:
- screen collection/adjudication is refused unless the validation gate PASSED and
  ``final_selection.json`` is frozen (VALIDATION_QUALIFIED);
- the one-shot order is enforced: frozen predictions/scores + hashed Direct score freeze
  + determinism receipts must exist BEFORE ``development_screen_used.lock`` can burn,
  and reading screen truth (``assemble_screen_group_quantities``) is REFUSED until the
  lock is burned;
- every adjudication-side artifact is create-once: a second adjudication is refused;
- after a terminal screen report exists, any rerun/continue request is refused
  (line closed; no edits, no re-adjudication);
- the screen receipt binds the USER-CONFIRMED train-support recovery report
  (``train_support_report_r1.json``, path + SHA256) and names the superseded base
  report, so the recovery is offline-auditable.
"""
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
import drive_stage4cr2_task4 as drive  # noqa: E402
from ac_twin import stage4cp_protocol as s4cpp  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_statistics as s4cr2st  # noqa: E402
from ac_twin import stage4cr2_training as s4cr2t  # noqa: E402
from ac_twin.checkpointing import atomic_save_npz  # noqa: E402
from ac_twin.stage1_data import sha256_file  # noqa: E402

NA = s4cr2.N_ACTIONS
CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"


def _config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "run")
    config["data_root"] = str(
        tmp_path / "data" / "object_centric_stage" / s4cr2.PURPOSE_DEV)
    config["device"] = "cpu"
    return config


def _screen_segment_seeds(config, groups):
    schedule = s4cr2.build_seed_schedule(config)
    lo, hi = s4cr2.SCREEN_SEGMENT
    return schedule[lo:lo + groups]


def _write_final_selection(dev, *, status="VALIDATION_QUALIFIED"):
    payload = {
        "plan_id": s4cr2.PLAN_ID,
        "status": status,
        "strongest_standard_baseline": "SEED_VOTE_ENTROPY",
        "interaction_threshold": 25.0,
        "checkpoints": {str(s): {"path": f"models/direct_seed{s}.pt", "sha256": "ab"}
                        for s in s4cr2.MODEL_SEEDS},
        "model_seeds": [int(s) for s in s4cr2.MODEL_SEEDS],
    }
    (dev / "final_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    inventory = {"plan_id": s4cr2.PLAN_ID, "seeds": [int(s) for s in s4cr2.MODEL_SEEDS]}
    (dev / "model_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8")
    return dev / "final_selection.json"


def _write_train_support_reports(dev):
    base = {"plan_id": s4cr2.PLAN_ID, "status": "STAGE4CR2_TRAIN_SUPPORT_INSUFFICIENT"}
    (dev / "train_support_report.json").write_text(
        json.dumps(base, indent=2), encoding="utf-8")
    r1 = {"plan_id": s4cr2.PLAN_ID, "status": "TRAIN_SUPPORT_OK",
          "recovery": {"recovered_from": "train_support_report.json"}}
    (dev / "train_support_report_r1.json").write_text(
        json.dumps(r1, indent=2), encoding="utf-8")
    return dev / "train_support_report_r1.json"


def _write_data_npz(config, groups, seeds):
    rng = np.random.default_rng(0)
    capacity, hl, hz, nf = 5, 10, 20, 8
    history = rng.normal(size=(groups, hl, capacity, nf)).astype(np.float32)
    factual = rng.normal(size=(groups, hz, capacity, nf)).astype(np.float32)
    cf = rng.normal(size=(groups, NA, hz, capacity, nf)).astype(np.float32)
    for arr in (history, factual, cf):
        arr[..., 0] = 0.0
        arr[..., :4, 0] = 1.0
    mask = np.zeros((groups, hz, capacity), dtype=bool)
    mask[..., 1:4] = True
    data_output, _, _ = coll.role_paths(config, "development_screen")
    data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_output, object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf,
        cf_actions=np.tile([a for a, _ in s4cr2.ACTION_PLANS], (groups, 1)).astype(np.int64),
        cf_durations=np.tile([d for _, d in s4cr2.ACTION_PLANS],
                             (groups, 1)).astype(np.int64),
        factual_actions=rng.integers(0, 5, size=(groups, hz)).astype(np.int64),
        t_star=np.where(np.arange(groups) % 2 == 0, 11, 14),
        episode_seed=np.asarray(seeds, dtype=np.int64),
        n_mask_cells=np.full(groups, 60),
        inference_mask=mask)
    return data_output


def _fake_features(groups, seeds):
    """Score-side features exactly as ``build_relative_tokens`` would return them."""
    rng = np.random.default_rng(3)
    baselines = {
        "MARGIN_ONLY": rng.random(groups),
        "SEED_VOTE_ENTROPY": rng.random(groups),
        "SEED_MARGIN_STD": rng.random(groups),
        "COMBINED_TOP2_MAX": rng.random(groups),
    }
    return {
        "tokens": rng.normal(size=(groups, s4cr2.N_COMPETITORS, s4cr2.TOKEN_DIM)),
        "global_features": rng.normal(size=(groups, s4cr2.GLOBAL_DIM)),
        "valid_mask": np.ones(groups, dtype=bool),
        "group_seed": np.asarray(seeds, dtype=np.int64),
        "t_star": np.where(np.arange(groups) % 2 == 0, 11, 14),
        "j_hat": rng.random((groups, NA)),
        "baselines": baselines,
        "interaction_min_distance": np.where(
            (np.arange(groups) // 2) % 2 == 0, 10.0, 40.0),
    }


def _fake_score_fn(bundle):
    """Deterministic stand-in for the five-checkpoint Direct scoring (test hook)."""
    group_seed = np.asarray(bundle["group_seed"], dtype=np.int64)
    groups = group_seed.size
    base = np.sin(group_seed.astype(np.float64)) * 0.5 + 0.5
    by_seed = np.stack([base + 0.01 * i for i in range(5)])
    return s4cr2t.RiskScores(
        group_seed=group_seed,
        risk_ensemble=by_seed.mean(axis=0),
        risk_by_seed=by_seed,
        mu_only_ensemble=base.copy(),
        class_only_ensemble=base.copy(),
        seeds=tuple(int(s) for s in s4cr2.MODEL_SEEDS),
    )


def _write_inference_side(config, groups, seeds):
    """The Task-2 predictor-side artifacts + receipts with real hashes."""
    paths = s4cr2st.screen_artifact_paths(config)
    dev = paths["dev_dir"]
    dev.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    atomic_save_npz(paths["predictions"], **{
        k: np.zeros((5, groups, NA, 20, 5, 2), dtype=np.float32)
        for k in ("e2e", "full", "history")},
        **{k: np.zeros((5, groups, NA, 20, 5), dtype=np.float32)
           for k in ("r_hat", "presence_prob", "high_prob")},
        group_seed=np.asarray(seeds, dtype=np.int64))
    atomic_save_npz(paths["scores"],
                    group_seed=np.asarray(seeds, dtype=np.int64),
                    **{k: rng.random((groups, NA)) for k in s4cpp.RISK_CLASSES})
    data_sha = sha256_file(paths["data_output"])
    paths["collection_receipt"].write_text(json.dumps({
        "plan_id": s4cr2.PLAN_ID,
        "role": "development_screen",
        "status": "COLLECTION_COMPLETE",
        "accepted_groups": int(groups),
        "data_output_sha256": data_sha,
    }, indent=2), encoding="utf-8")
    receipt = {
        "plan_id": s4cr2.PLAN_ID,
        "role": "development_screen",
        "status": "INFERENCE_COMPLETE",
        "groups": int(groups),
        "data_sha256": data_sha,
        "predictions_sha256": sha256_file(paths["predictions"]),
        "scores_sha256": sha256_file(paths["scores"]),
    }
    paths["inference_receipt"].write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    determinism = {
        "plan_id": s4cr2.PLAN_ID,
        "role": "development_screen",
        "status": "DETERMINISM_VERIFIED",
        "n_groups": 32,
        "bitwise_identical": True,
        "max_abs_diff": 0.0,
        "data_sha256": data_sha,
    }
    paths["determinism_receipt"].write_text(
        json.dumps(determinism, indent=2), encoding="utf-8")


def _setup(tmp_path, groups=48):
    """Full score-side state: frozen selection + reports + data + inference artifacts."""
    config = _config(tmp_path)
    paths = s4cr2st.screen_artifact_paths(config)
    paths["dev_dir"].mkdir(parents=True, exist_ok=True)
    seeds = _screen_segment_seeds(config, groups)
    _write_final_selection(paths["dev_dir"])
    _write_train_support_reports(paths["dev_dir"])
    _write_data_npz(config, groups, seeds)
    _write_inference_side(config, groups, seeds)
    return config, paths, seeds


def _truth_bundle(groups, seeds):
    """Truth-side bundle as ``load_role_bundle`` would return (injected in tests)."""
    rng = np.random.default_rng(11)
    cf = rng.gamma(2.0, 4.0, size=(groups, NA, 20, 5, 2))
    cf[..., 0, :] *= 0.0  # keep the ego trajectory modest
    mask = np.zeros((groups, 20, 5), dtype=bool)
    mask[..., 1:4] = True
    return {
        "groups": int(groups),
        "n_actions": NA,
        "counterfactual_xy": cf,
        "history_last_xy": rng.normal(size=(groups, 2)),
        "inference_mask": mask,
        "episode_seed": np.asarray(seeds, dtype=np.int64),
    }


# --- precondition: validation must have passed and be frozen --------------------------------


def test_collect_screen_refused_without_frozen_selection(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError):
        drive.require_frozen_selection(config)


def test_collect_screen_refused_when_validation_failed(tmp_path):
    config = _config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    dev.mkdir(parents=True)
    _write_final_selection(dev, status=s4cr2.STATUS_MODEL_DEV_INSUFFICIENT)
    with pytest.raises(ValueError):
        drive.require_frozen_selection(config)


# --- score freeze + lock burn order ---------------------------------------------------------


def test_burn_lock_refused_before_direct_scores_frozen(tmp_path):
    config, paths, _ = _setup(tmp_path)
    with pytest.raises(ValueError):
        s4cr2st.burn_screen_lock(config)
    assert not paths["lock"].exists()


def test_freeze_direct_scores_create_once_and_bitwise_recheck(tmp_path):
    config, paths, seeds = _setup(tmp_path)
    calls = []

    def counting_score_fn(bundle):
        calls.append(int(np.asarray(bundle["group_seed"]).size))
        return _fake_score_fn(bundle)

    path = s4cr2st.freeze_screen_direct_scores(config, features=_fake_features(48, seeds),
                                               score_fn=counting_score_fn, device="cpu")
    assert path == paths["direct_scores_npz"] and path.is_file()
    # one full pass + two 32-group bitwise recheck passes.
    assert calls == [48, 32, 32]
    receipt = json.loads(paths["direct_scores_receipt"].read_text(encoding="utf-8"))
    assert receipt["direct_recheck"]["bitwise_identical"] is True
    assert receipt["direct_recheck"]["n_groups"] == 32
    assert receipt["predictions_sha256"] == sha256_file(paths["predictions"])
    assert receipt["scores_sha256"] == sha256_file(paths["scores"])
    assert receipt["final_selection_sha256"] == sha256_file(paths["final_selection"])
    digest = sha256_file(path)
    # create-once: a second call verifies and never overwrites.
    again = s4cr2st.freeze_screen_direct_scores(
        config, features=_fake_features(48, seeds), score_fn=counting_score_fn,
        device="cpu")
    assert again == path and sha256_file(path) == digest
    assert calls == [48, 32, 32]  # not rescored


def test_freeze_refuses_hash_drift(tmp_path):
    config, paths, seeds = _setup(tmp_path)
    atomic_save_npz(paths["scores"], group_seed=np.asarray(seeds, dtype=np.int64),
                    **{k: np.zeros((48, NA)) for k in s4cpp.RISK_CLASSES})
    with pytest.raises(ValueError):
        s4cr2st.freeze_screen_direct_scores(
            config, features=_fake_features(48, seeds), score_fn=_fake_score_fn,
            device="cpu")


def test_burn_lock_create_once_and_binding_verified(tmp_path):
    config, paths, seeds = _setup(tmp_path)
    s4cr2st.freeze_screen_direct_scores(config, features=_fake_features(48, seeds),
                                        score_fn=_fake_score_fn, device="cpu")
    lock = s4cr2st.burn_screen_lock(config)
    assert lock == paths["lock"] and lock.is_file()
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["bindings"]["direct_scores_sha256"] == sha256_file(
        paths["direct_scores_npz"])
    first_read = lock.read_bytes()
    again = s4cr2st.burn_screen_lock(config)  # idempotent verify, never overwrite
    assert again == lock and lock.read_bytes() == first_read
    # tampering with a frozen score artifact after the burn is detected.
    atomic_save_npz(paths["direct_scores_npz"], group_seed=np.zeros(48, dtype=np.int64))
    with pytest.raises(ValueError):
        s4cr2st.burn_screen_lock(config)


# --- truth read strictly after the burned lock; single adjudication -------------------------


def test_assemble_refused_until_lock_is_burned(tmp_path):
    config, paths, seeds = _setup(tmp_path)
    s4cr2st.freeze_screen_direct_scores(config, features=_fake_features(48, seeds),
                                        score_fn=_fake_score_fn, device="cpu")
    # scores frozen but the lock is NOT burned: reading truth must be refused.
    with pytest.raises(ValueError, match="lock"):
        s4cr2st.assemble_screen_group_quantities(config, bundle=_truth_bundle(48, seeds))
    assert not paths["quantities_npz"].exists()


def test_assemble_writes_create_once_and_binds_recovery_report(tmp_path):
    config, paths, seeds = _setup(tmp_path)
    s4cr2st.freeze_screen_direct_scores(config, features=_fake_features(48, seeds),
                                        score_fn=_fake_score_fn, device="cpu")
    s4cr2st.burn_screen_lock(config)
    npz_path = s4cr2st.assemble_screen_group_quantities(
        config, bundle=_truth_bundle(48, seeds))
    assert npz_path == paths["quantities_npz"] and npz_path.is_file()
    receipt = json.loads(paths["quantities_receipt"].read_text(encoding="utf-8"))
    assert receipt["n_groups"] == 48
    assert receipt["strongest_standard_baseline"] == "SEED_VOTE_ENTROPY"
    assert receipt["interaction_threshold"] == pytest.approx(25.0)
    assert receipt["bootstrap_entropy"] == s4cr2.BOOTSTRAP_ENTROPY == 2026092702
    assert receipt["bootstrap_resamples"] == s4cr2.BOOTSTRAP_RESAMPLES == 10000
    assert receipt["confidence_screen"] == s4cr2.CONFIDENCE_SCREEN == 0.80
    assert receipt["screen_group_quantities_sha256"] == sha256_file(npz_path)
    assert receipt["lock_sha256"] == sha256_file(paths["lock"])
    # the user-confirmed train-support recovery is offline-auditable.
    recovery = receipt["train_support_binding"]
    assert recovery["report_path"].endswith("train_support_report_r1.json")
    assert recovery["report_sha256"] == sha256_file(
        paths["dev_dir"] / "train_support_report_r1.json")
    assert recovery["report_status"] == "TRAIN_SUPPORT_OK"
    assert recovery["superseded_base_report_path"].endswith("train_support_report.json")
    assert recovery["superseded_base_report_sha256"] == sha256_file(
        paths["dev_dir"] / "train_support_report.json")
    assert recovery["superseded_base_report_status"] == "STAGE4CR2_TRAIN_SUPPORT_INSUFFICIENT"

    quantities = s4cr2st.load_screen_group_quantities(npz_path, paths["quantities_receipt"])
    assert quantities["strongest_baseline"] == "SEED_VOTE_ENTROPY"
    assert quantities["regret"].shape == (48,)
    assert quantities["risk_by_seed"].shape == (5, 48)
    assert set(quantities["baseline_risks"]) == {
        "MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD",
        "COMBINED_TOP2_MAX", "CONSTANT_RANDOM"}
    assert quantities["qa"]["role_seeds_valid"] is True

    # a second adjudication is refused.
    with pytest.raises(FileExistsError):
        s4cr2st.assemble_screen_group_quantities(config, bundle=_truth_bundle(48, seeds))


def test_assemble_refused_when_report_already_exists(tmp_path):
    config, paths, seeds = _setup(tmp_path)
    s4cr2st.freeze_screen_direct_scores(config, features=_fake_features(48, seeds),
                                        score_fn=_fake_score_fn, device="cpu")
    s4cr2st.burn_screen_lock(config)
    paths["report"].write_text(json.dumps(
        {"plan_id": s4cr2.PLAN_ID,
         "terminal_state": s4cr2.STATUS_LINE_CLOSED}), encoding="utf-8")
    with pytest.raises(FileExistsError):
        s4cr2st.assemble_screen_group_quantities(config, bundle=_truth_bundle(48, seeds))


def test_screen_rerun_refused_after_terminal_report(tmp_path):
    config = _config(tmp_path)
    dev = Path(config["run_dir"]) / "development"
    dev.mkdir(parents=True)
    _write_final_selection(dev)
    report = dev / "development_screen_report.json"
    report.write_text(json.dumps(
        {"plan_id": s4cr2.PLAN_ID, "terminal_state": s4cr2.STATUS_LINE_CLOSED}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="closed|refused|already"):
        drive.guard_screen_not_adjudicated(config)
    # the same refusal holds after a PASS: a screen is judged exactly once.
    report.write_text(json.dumps(
        {"plan_id": s4cr2.PLAN_ID, "terminal_state": s4cr2.STATUS_DEV_READY}),
        encoding="utf-8")
    with pytest.raises(ValueError):
        drive.guard_screen_not_adjudicated(config)
