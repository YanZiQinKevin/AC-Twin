from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_stage3a as drv  # noqa: E402
from ac_twin import stage3_training as s3t  # noqa: E402


def _valid_config() -> dict:
    return {
        "search_space": {
            "hidden_dim": [64, 96],
            "response_feature_dim": [32, 64],
            "lambda_cls": [0.10, 0.25],
        },
        "max_finalist_configs": 2,
        "max_epochs": 5,
        "early_stopping_patience": 2,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "device": "cuda",
    }


def _benchmarks(device="cuda") -> dict:
    return {
        "device": device,
        "date": "2026-09-09",
        "response": {"epoch_seconds": 2.0, "peak_memory_gb": 3.0, "device": device, "episodes": 1400},
        "router": {"epoch_seconds": 3.0, "peak_memory_gb": 4.0, "device": device, "episodes": 1400},
    }


# --- config validation ------------------------------------------------------- #
def test_validate_config_accepts_registered_space() -> None:
    drv.validate_stage3a_config(_valid_config())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["search_space"].update(hidden_dim=[64, 96, 128]),
        lambda c: c["search_space"].update(lambda_cls=[0.10, 0.25, 0.5]),
        lambda c: c.update(max_finalist_configs=3),
        lambda c: c.pop("max_epochs"),
    ],
)
def test_validate_config_rejects_expansion_or_missing(mutate) -> None:
    config = _valid_config()
    mutate(config)
    with pytest.raises(ValueError):
        drv.validate_stage3a_config(config)


# --- batching (episode is the sampling unit) ---------------------------------- #
def test_make_batches_partitions_episodes_without_reuse() -> None:
    tensors = {"x": torch.arange(20).view(20, 1)}
    indices = np.arange(20)
    batches = drv.make_batches(tensors, indices, batch_size=6, seed=0)
    seen = torch.cat([b["x"].view(-1) for b in batches]).tolist()
    assert sorted(seen) == list(range(20))  # disjoint, complete
    assert all(len(b["x"]) <= 6 for b in batches)


def test_make_batches_deterministic_per_seed() -> None:
    tensors = {"x": torch.arange(20).view(20, 1)}
    a = drv.make_batches(tensors, np.arange(20), 6, seed=7)
    b = drv.make_batches(tensors, np.arange(20), 6, seed=7)
    assert [bb["x"].tolist() for bb in a] == [bb["x"].tolist() for bb in b]


# --- budget subcommand --------------------------------------------------------- #
def test_budget_command_writes_all_sections(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_benchmarks("cuda")), encoding="utf-8")
    out = tmp_path / "COMPUTE_BUDGET.md"
    args = drv.build_parser().parse_args([
        "budget", "--config", str(config_path), "--benchmark", str(benchmark_path),
        "--output", str(out),
    ])
    assert args.func(args) == 0
    text = out.read_text(encoding="utf-8")
    for section in ("估算来源与外推", "单 seed 有界筛选", "cross-fit 额外成本",
                    "Stage 3B 分项预算", "失败、重试与断点续跑策略", "总训练次数",
                    "正式集合", "典型总时长", "最坏总时长"):
        assert section in text


def test_budget_command_fails_on_device_mismatch(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    cfg = _valid_config()
    cfg["device"] = "cpu"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(_benchmarks("cuda")), encoding="utf-8")
    args = drv.build_parser().parse_args([
        "budget", "--config", str(config_path), "--benchmark", str(benchmark_path),
        "--output", str(tmp_path / "B.md"),
    ])
    with pytest.raises(ValueError, match="device mismatch"):
        args.func(args)


# --- readiness ------------------------------------------------------------------ #
def test_readiness_prints_checklist(capsys) -> None:
    args = drv.build_parser().parse_args(["readiness"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "checklist" in out


# --- search finalist selection (stubbed trainer) -------------------------------- #
def _fake_summary(cfg, seed, sel_score, best_val=None):
    return {
        "run": "x", "state": "COMPLETE", "seed": seed, "config": dict(cfg),
        "best_val_loss": best_val if best_val is not None else sel_score,
        "train_lambda_cls": cfg["lambda_cls"], "selection_lambda_cls": 0.25,
        "validation_reg_loss": sel_score, "validation_ce_loss": 0.0,
        "validation_selection_score": sel_score,
        "epochs_run": 1, "signature": "s", "checkpoint": "x.pt",
    }


def test_search_selects_two_finalists_by_validation_loss(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_valid_config()), encoding="utf-8")

    fake_bundle = {
        "scaled": {}, "roles": {"train": np.arange(4), "validation": np.arange(4)},
        "scaler": None, "data_signature": "d",
    }
    monkeypatch.setattr(drv, "_load_bundle_and_scaler", lambda a: fake_bundle)
    monkeypatch.setattr(drv.ProgressReporter, "update", lambda *a, **k: None)
    scores = {tuple(sorted(c.items())): 1.0 + i for i, c in enumerate(s3t.search_space_configs())}

    def fake_train(config, scaled, roles, cfg, seed, run_dir, resume, *, scaler, data_signature):
        return _fake_summary(cfg, seed, scores[tuple(sorted(cfg.items()))])

    monkeypatch.setattr(drv, "_train_one_config", fake_train)
    args = drv.build_parser().parse_args([
        "search", "--config", str(config_path), "--run-dir", str(tmp_path / "runs"),
    ])
    assert args.func(args) == 0
    results = json.loads((tmp_path / "runs" / "search" / "search_results.json").read_text(encoding="utf-8"))
    assert len(results["finalists"]) == s3t.MAX_FINALIST_CONFIGS
    assert "selection score" in results["selection_basis"]
    vals = sorted(r["validation_selection_score"] for r in results["results"])
    assert vals[0] < vals[-1]


def test_search_ranks_by_selection_score_not_combined_train_lambda_loss(tmp_path, monkeypatch) -> None:
    """v0.1 §2.3/§2.5: finalists follow validation_selection_score even when the
    candidate's own combined-loss value (best_val_loss) ranks in the OPPOSITE order."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    fake_bundle = {
        "scaled": {}, "roles": {"train": np.arange(4), "validation": np.arange(4)},
        "scaler": None, "data_signature": "d",
    }
    monkeypatch.setattr(drv, "_load_bundle_and_scaler", lambda a: fake_bundle)
    monkeypatch.setattr(drv.ProgressReporter, "update", lambda *a, **k: None)
    configs = s3t.search_space_configs()

    def fake_train(config, scaled, roles, cfg, seed, run_dir, resume, *, scaler, data_signature):
        i = [tuple(sorted(c.items())) for c in configs].index(tuple(sorted(cfg.items())))
        # selection score ascending with i; combined loss DESCENDING (opposite)
        return _fake_summary(cfg, seed, sel_score=1.0 + i, best_val=100.0 - i)

    monkeypatch.setattr(drv, "_train_one_config", fake_train)
    args = drv.build_parser().parse_args([
        "search", "--config", str(config_path), "--run-dir", str(tmp_path / "runs"),
    ])
    assert args.func(args) == 0
    results = json.loads((tmp_path / "runs" / "search" / "search_results.json").read_text(encoding="utf-8"))
    finalist_scores = sorted(
        r["validation_selection_score"] for r in results["results"]
        if r["config"] in [dict(f) for f in results["finalists"]]
    )
    # the two finalists are the two LOWEST selection scores (1.0, 2.0) even though
    # they carry the two HIGHEST combined-loss values (100.0, 99.0)
    assert finalist_scores == [1.0, 2.0]


def test_select_final_ranks_by_five_seed_selection_score(tmp_path, monkeypatch) -> None:
    """v0.1 §2.5.6: the final config follows the mean five-seed selection score,
    not the candidates' own combined training losses."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_valid_config()), encoding="utf-8")
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    seeds = [101, 202, 303, 404, 505]
    cfg_x = {"hidden_dim": 64, "response_feature_dim": 32, "lambda_cls": 0.1}
    cfg_y = {"hidden_dim": 96, "response_feature_dim": 32, "lambda_cls": 0.1}
    fit5 = {"finalists": [
        {"config": cfg_x, "per_seed": [
            {"seed": s, "signature": f"x{s}", "path": "x.pt",
             "best_val_loss": 0.05, "validation_selection_score": 0.50} for s in seeds]},
        {"config": cfg_y, "per_seed": [
            {"seed": s, "signature": f"y{s}", "path": "y.pt",
             "best_val_loss": 0.50, "validation_selection_score": 0.40} for s in seeds]},
    ]}
    (run_dir / "fit_five").mkdir()
    (run_dir / "fit_five" / "fit_five_results.json").write_text(json.dumps(fit5), encoding="utf-8")
    import torch

    fake_bundle = {
        "tensors": {
            "r_true": torch.zeros(4, 2, 2), "regime_labels": torch.zeros(4, 2, 2, dtype=torch.long),
            "valid": torch.ones(4, 2, 2, dtype=torch.bool),
        },
        "roles": {"train": np.arange(2), "validation": np.arange(2, 4)},
        "raw_factual_suffix": np.zeros((4, 2, 2, 8)),
        "data_signature": "d",
    }
    monkeypatch.setattr(drv, "_load_bundle_and_scaler", lambda a: fake_bundle)
    monkeypatch.setattr(drv, "_build_baselines", lambda *a: ({}, {}))
    monkeypatch.setattr(drv, "select_strongest_baselines", lambda **k: {
        "strongest_continuous": "constant_zero", "strongest_presence": "prevalence",
        "strongest_high_vs_lowmid": "prevalence", "strongest_brier": "prevalence",
    })
    args = drv.build_parser().parse_args([
        "select-final", "--config", str(config_path), "--run-dir", str(run_dir),
    ])
    assert args.func(args) == 0
    receipt = json.loads((run_dir / "selection" / "final_selection.json").read_text(encoding="utf-8"))
    # X wins on combined loss (0.05 < 0.50) but Y wins on selection score (0.40 < 0.50)
    assert receipt["finalist_name"] == "h96_r32_l0.1"
    assert receipt["per_finalist_mean_validation_selection_score"] == {
        "h64_r32_l0.1": 0.50, "h96_r32_l0.1": 0.40,
    }
