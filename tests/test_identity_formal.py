from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_identity_experiment_config import dataset, minimal_config, _trainer_without_optimization
import run_identity_continuity_comparison as runner


def formal_fixture(tmp_path, dataset):
    config = minimal_config()
    config.update(smoke_test=False, capacity_tiers=["65k", "83k"], scene="merge",
                  source_data=str(tmp_path / "source.npz"),
                  frozen_test_data=str(tmp_path / "frozen.npz"),
                  output_dir=str(tmp_path / "output"),
                  statistics={"bootstrap_resamples": 10000, "confidence_level": 0.95})
    config["model"]["training_seeds"] = [11, 22, 33, 44, 55]
    source = copy.deepcopy(dataset)
    source.update(config_json=np.asarray(json.dumps({"collection_seed": 10, "env_id": "merge-v0",
                          "history_len": 3, "horizon": 4})),
                  object_capacity=np.asarray(4),
                  object_features_json=np.asarray(json.dumps(runner.FEATURE_NAMES)))
    frozen = copy.deepcopy(source)
    frozen["episode_seed"] += 10000
    frozen["config_json"] = np.asarray(json.dumps({"collection_seed": 20, "env_id": "merge-v0",
                                                  "history_len": 3, "horizon": 4}))
    np.savez(config["source_data"], **source)
    np.savez(config["frozen_test_data"], **frozen)
    return config, source, frozen


@pytest.mark.parametrize("scene", ["merge", "roundabout"])
def test_formal_configs_bind_frozen_paths_without_requiring_files(scene):
    config = json.loads((runner.ROOT / "config" / f"identity_continuity_{scene}.json").read_text())
    runner.validate_identity_config(config)
    assert config.get("frozen_test_data") == f"data/identity_continuity_stage/{scene}_holdout_1000/object_twinworld_holdout_1000.npz"
    assert config["output_dir"] == f"experiments/identity_continuity_stage/{scene}"


def test_missing_frozen_fails_before_model_or_output_creation(tmp_path, dataset, monkeypatch):
    config, _, _ = formal_fixture(tmp_path, dataset)
    Path(config["frozen_test_data"]).unlink()
    monkeypatch.setattr(runner, "build_capacity_matched_model", lambda **kwargs: pytest.fail("model created"))
    with pytest.raises(FileNotFoundError, match="冻结数据缺失"):
        runner.run_experiment(config, resume=True, device="cpu")
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("mutation, message", [
    ("collection_seed", "collection seed"), ("episode_seed", "episode seed"),
    ("features", "feature"), ("capacity", "capacity"),
    ("history", "history"), ("horizon", "horizon"), ("scene", "scene"),
])
def test_preflight_rejects_incompatible_frozen_archive(tmp_path, dataset, mutation, message):
    config, source, frozen = formal_fixture(tmp_path, dataset)
    metadata = json.loads(str(frozen["config_json"]))
    if mutation == "collection_seed":
        metadata["collection_seed"] = 10
    elif mutation == "episode_seed":
        frozen["episode_seed"][0] = source["episode_seed"][-1]
    elif mutation == "features":
        frozen["object_features_json"] = np.asarray(json.dumps(runner.FEATURE_NAMES[::-1]))
    elif mutation == "capacity":
        frozen["object_capacity"] = np.asarray(5)
    elif mutation == "history":
        frozen["object_history"] = frozen["object_history"][:, :-1]
        metadata["history_len"] = 2
    elif mutation == "horizon":
        frozen["object_factual_future"] = frozen["object_factual_future"][:, :-1]
        metadata["horizon"] = 3
    else:
        metadata["env_id"] = "roundabout-v0"
    frozen["config_json"] = np.asarray(json.dumps(metadata))
    np.savez(config["frozen_test_data"], **frozen)
    with pytest.raises(ValueError, match=message):
        runner.preflight_identity_run(config)
    assert not Path(config["output_dir"]).exists()


def test_preflight_returns_bound_hashes_and_accepts_legacy_object_json(tmp_path, dataset):
    config, source, frozen = formal_fixture(tmp_path, dataset)
    for archive, filename in ((source, "source_data"), (frozen, "frozen_test_data")):
        for key in ("config_json", "object_features_json"):
            archive[key] = archive[key].astype(object)
        np.savez(config[filename], **archive)
    result = runner.preflight_identity_run(config)
    assert result["source_data_sha256"] == runner.sha256_file(Path(config["source_data"]))
    assert result["frozen_test_data_sha256"] == runner.sha256_file(Path(config["frozen_test_data"]))
    assert result["data_consistency"] is True
    assert not Path(config["output_dir"]).exists()


def test_formal_cannot_fall_back_to_development_test(tmp_path, dataset):
    config, _, _ = formal_fixture(tmp_path, dataset)
    del config["frozen_test_data"]
    with pytest.raises(ValueError, match="frozen_test_data"):
        runner.validate_identity_config(config)


@pytest.mark.parametrize("filename", ["report.json", "orphan.pt", "experiment_manifest.json"])
def test_preflight_rejects_unsigned_or_mismatching_formal_output(tmp_path, dataset, filename):
    config, _, _ = formal_fixture(tmp_path, dataset)
    output = Path(config["output_dir"])
    output.mkdir()
    (output / filename).write_text(json.dumps({"signature": "wrong"}))
    with pytest.raises(ValueError, match="signature"):
        runner.preflight_identity_run(config)
    assert len(list(output.iterdir())) == 1


@pytest.mark.parametrize("boundary", ["manifest", "seed_metadata"])
def test_statistics_implementation_change_rejects_resume_before_training_or_writes(
        tmp_path, dataset, monkeypatch, boundary):
    config, _, _ = formal_fixture(tmp_path, dataset)
    manifest = runner.preflight_identity_run(config)
    old_seed_signature = runner.run_signature(
        config["conditions"][0], config, manifest["source_data_sha256"],
        manifest["frozen_test_data_sha256"], capacity_tier="65k", seed=11)
    output = Path(config["output_dir"])
    output.mkdir()
    manifest_path = output / "experiment_manifest.json"
    metadata_path = output / "stable_temporal_65k_seed11_run.json"
    metadata_path.write_text(json.dumps({"signature": old_seed_signature,
                                          "summary": {"status": "running"}}))
    statistics_path = runner.ROOT / "code" / "ac_twin" / "statistics.py"
    changed_statistics = tmp_path / "changed_statistics.py"
    changed_statistics.write_text(statistics_path.read_text(encoding="utf-8") + "\n# regression change\n")
    actual_sha256 = runner.sha256_file
    monkeypatch.setattr(runner, "sha256_file", lambda filename: actual_sha256(
        changed_statistics if Path(filename).resolve() == statistics_path.resolve() else filename))
    if boundary == "seed_metadata":
        # A current manifest does not make stale per-seed metadata reusable.
        manifest["signature"] = runner.run_signature(
            {}, config, manifest["source_data_sha256"], manifest["frozen_test_data_sha256"],
            capacity_tier="all", seed=0)
    manifest_path.write_text(json.dumps(manifest))
    before = {file.name: file.read_bytes() for file in output.iterdir()}
    monkeypatch.setattr(runner, "build_capacity_matched_model", lambda **kwargs: pytest.fail("model created"))
    monkeypatch.setattr(runner, "train_one", lambda *args, **kwargs: pytest.fail("training started"))
    with pytest.raises(ValueError, match="signature|match"):
        runner.run_experiment(config, resume=True, device="cpu")
    assert {file.name: file.read_bytes() for file in output.iterdir()} == before


@pytest.mark.parametrize("mutation, message", [
    ("other_seeds", "training_seeds"), ("reordered_seeds", "training_seeds"),
    ("resamples", "bootstrap_resamples"), ("float_resamples", "bootstrap_resamples"),
    ("confidence", "confidence_level"),
    ("missing_statistics", "statistics"),
])
def test_formal_protocol_settings_rejected_before_training(tmp_path, dataset, monkeypatch, mutation, message):
    config, _, _ = formal_fixture(tmp_path, dataset)
    if mutation == "other_seeds":
        config["model"]["training_seeds"] = [1, 2, 3, 4, 5]
    elif mutation == "reordered_seeds":
        config["model"]["training_seeds"] = [55, 44, 33, 22, 11]
    elif mutation == "resamples":
        config["statistics"]["bootstrap_resamples"] = 20
    elif mutation == "float_resamples":
        config["statistics"]["bootstrap_resamples"] = 10000.0
    elif mutation == "confidence":
        config["statistics"]["confidence_level"] = 0.90
    else:
        del config["statistics"]
    monkeypatch.setattr(runner, "build_capacity_matched_model", lambda **kwargs: pytest.fail("model created"))
    monkeypatch.setattr(runner, "train_one", lambda *args, **kwargs: pytest.fail("training started"))
    for check in (runner.validate_identity_config, runner.preflight_identity_run):
        with pytest.raises(ValueError, match=message):
            check(config)
    with pytest.raises(ValueError, match=message):
        runner.run_experiment(config, resume=True, device="cpu")
    assert not Path(config["output_dir"]).exists()


def test_smoke_schema_still_allows_short_statistics():
    config = minimal_config()
    config["statistics"] = {"bootstrap_resamples": 20, "confidence_level": 0.90}
    runner.validate_identity_config(config)


def report_fixture(scene):
    metric = {"independent_units": ["training_seed", "test_episode"], "seed_count": 5,
              "episode_count": 1000, "bootstrap_resamples": 10000, "confidence_level": 0.95,
              "ci_lower": 0.5, "ci_upper": 1.5,
              "mean_paired_difference_reference_minus_candidate": 1.0,
              "per_seed_mean_differences": [1.0] * 5}
    contrast = {"status": "available", "training_seeds": [11, 22, 33, 44, 55],
                "fde": metric}
    tier = {"primary": {f"{name}_minus_stable_temporal": copy.deepcopy(contrast)
                        for name in ("permuted_temporal", "distance_temporal")},
            "secondary": {"set_pool_minus_stable_temporal": copy.deepcopy(contrast)}}
    return {"scene": scene, "evaluation_mode": "frozen_test", "training_seeds": [11, 22, 33, 44, 55],
            "frozen_test_data_sha256": scene + "hash", "n_frozen_test": 1000,
            "qa": {key: True for key in ("fairness", "object_multiset", "ego_slot", "permutation",
                   "data_consistency", "capacity", "finite_values", "resume", "holdout_collection")},
            "hierarchical_paired_contrasts": {key: copy.deepcopy(tier) for key in ("65k", "83k")}}


def test_decision_never_passes_single_scene():
    assert runner.decide_identity_continuity(report_fixture("merge"), None) == "PENDING_SECOND_SCENE"


def test_decision_requires_both_scenes_both_tiers_and_all_primary_contrasts():
    merge, roundabout = report_fixture("merge"), report_fixture("roundabout")
    assert runner.decide_identity_continuity(merge, roundabout) == "PASS_IDENTITY_CONTINUITY"
    roundabout["hierarchical_paired_contrasts"]["83k"]["primary"]["permuted_temporal_minus_stable_temporal"]["fde"]["ci_lower"] = -0.1
    assert runner.decide_identity_continuity(merge, roundabout) == "SCENE_LIMITED_IDENTITY_EVIDENCE"


@pytest.mark.parametrize("problem", ["missing_qa", "null_qa", "failed_qa", "development", "same_scene", "same_hash"])
def test_decision_rejects_missing_or_failed_qa_and_nonformal_data(problem):
    merge, other = report_fixture("merge"), report_fixture("roundabout")
    if problem == "missing_qa":
        del other["qa"]["resume"]
    elif problem == "null_qa":
        other["qa"] = None
    elif problem == "failed_qa":
        other["qa"]["finite_values"] = False
    elif problem == "development":
        other["evaluation_mode"] = "development_test"
    elif problem == "same_scene":
        other["scene"] = "merge"
    else:
        other["frozen_test_data_sha256"] = merge["frozen_test_data_sha256"]
    assert runner.decide_identity_continuity(merge, other) == "QA_FAIL"


@pytest.mark.parametrize("problem", ["single_seed", "missing_tier", "unavailable", "nan", "episode_only", "wrong_ci", "wrong_seed_order"])
def test_incomplete_or_seed_driven_statistics_cannot_pass(problem):
    reports = [report_fixture("merge"), report_fixture("roundabout")]
    for report in reports:
        contrast = report["hierarchical_paired_contrasts"]["65k"]["primary"]["permuted_temporal_minus_stable_temporal"]
        if problem == "single_seed":
            contrast["fde"]["per_seed_mean_differences"] = [5, 0, 0, 0, 0]
        elif problem == "missing_tier":
            del report["hierarchical_paired_contrasts"]["83k"]
        elif problem == "unavailable":
            contrast["status"] = "unavailable"
        elif problem == "nan":
            contrast["fde"]["ci_lower"] = float("nan")
        elif problem == "episode_only":
            contrast["fde"]["independent_units"] = ["test_episode"]
        elif problem == "wrong_ci":
            contrast["fde"]["confidence_level"] = 0.5
        else:
            contrast["training_seeds"] = [55, 44, 33, 22, 11]
    assert runner.decide_identity_continuity(*reports) == "INCONCLUSIVE"


def test_distance_only_does_not_support_identity():
    reports = [report_fixture("merge"), report_fixture("roundabout")]
    for report in reports:
        for tier in report["hierarchical_paired_contrasts"].values():
            metric = tier["primary"]["permuted_temporal_minus_stable_temporal"]["fde"]
            metric.update(ci_lower=-0.5, ci_upper=0.5,
                          mean_paired_difference_reference_minus_candidate=0,
                          per_seed_mean_differences=[0] * 5)
    assert runner.decide_identity_continuity(*reports) == "DISTANCE_ONLY_DIFFERENCE_NOT_IDENTITY"


def test_set_pool_without_reliable_difference_takes_priority_over_identity_pass():
    reports = [report_fixture("merge"), report_fixture("roundabout")]
    for report in reports:
        for tier in report["hierarchical_paired_contrasts"].values():
            tier["secondary"]["set_pool_minus_stable_temporal"]["fde"].update(ci_lower=-0.5, ci_upper=0.5)
    assert runner.decide_identity_continuity(*reports) == "PREFER_SET_POOL"


def test_formal_run_reports_pending_and_does_not_print_test_metrics_during_training(tmp_path, dataset, monkeypatch, capsys):
    config, _, _ = formal_fixture(tmp_path, dataset)
    monkeypatch.setattr(runner, "train_one", _trainer_without_optimization)
    # This routing/IO test keeps the real formal schema; statistics have their
    # own numerical tests, so avoid bootstrap work without changing the config.
    monkeypatch.setattr(runner, "build_hierarchical_contrast_report", lambda results, statistics: {})
    report = runner.run_experiment(config, resume=True, device="cpu")
    assert report["decision"] == "PENDING_SECOND_SCENE"
    assert report["qa"]["data_consistency"] is True
    assert report["qa"]["resume"] == "PENDING"
    assert "ADE=" not in capsys.readouterr().out
    manifest = json.loads((Path(config["output_dir"]) / "experiment_manifest.json").read_text())
    assert manifest["source_data_sha256"] == report["source_data_sha256"]
    assert manifest["frozen_test_data_sha256"] == report["frozen_test_data_sha256"]
    runner.preflight_identity_run(config)


def test_best_epoch_matches_trainer_improvement_threshold_and_partial_restart_is_visible(tmp_path, dataset, monkeypatch, capsys):
    from test_identity_experiment_config import _save_experiment_fixture
    config = _save_experiment_fixture(tmp_path, dataset)

    def trainer(*args, **kwargs):
        model, prediction, log = _trainer_without_optimization(*args, **kwargs)
        log[0]["val_mse"], log[1]["val_mse"] = 1.0, 1.0 - 5e-8
        return model, prediction, log

    monkeypatch.setattr(runner, "train_one", trainer)
    result = runner.run_experiment(config, resume=True, device="cpu")
    assert all(value["runs"][0]["best_epoch"] == 1 for value in result["results"].values())
    metadata_path = Path(config["output_dir"]) / "stable_temporal_65k_seed11_run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["summary"]["status"] = "running"
    metadata_path.write_text(json.dumps(metadata))
    runner.run_experiment(config, resume=True, device="cpu")
    assert "resumed=incomplete restart_seed=true" in capsys.readouterr().out
