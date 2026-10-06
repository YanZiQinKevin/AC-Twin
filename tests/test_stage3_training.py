from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage3_training as s3t  # noqa: E402
from ac_twin.stage3_routing import SEED_PAIRS  # noqa: E402


# --- deterministic seeding (Task 3, repair plan v0.1) --------------------------- #


def test_seed_all_deterministic_model_init_and_cuda_flags() -> None:
    from ac_twin.stage3_response import Stage3ResponsePredictor

    s3t.seed_all(0)
    m1 = Stage3ResponsePredictor(hidden_dim=16, response_feature_dim=8)
    s3t.seed_all(0)
    m2 = Stage3ResponsePredictor(hidden_dim=16, response_feature_dim=8)
    assert s3t.checkpoint_signature(m1.state_dict()) == s3t.checkpoint_signature(m2.state_dict())
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # strict mode: a non-deterministic CUDA op must raise, not warn
    assert torch.are_deterministic_algorithms_enabled()


def test_seed_all_reseeds_all_three_rngs() -> None:
    import random

    s3t.seed_all(42)
    snapshot = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    s3t.seed_all(42)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == snapshot


# --- checkpoint signatures -------------------------------------------------------


def _tiny_state(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "layer.weight": torch.randn(4, 4, generator=g),
        "layer.bias": torch.randn(4, generator=g),
    }


def test_checkpoint_signature_deterministic() -> None:
    sd = _tiny_state(0)
    assert s3t.checkpoint_signature(sd) == s3t.checkpoint_signature(sd)


def test_five_checkpoint_signatures_distinct() -> None:
    states = [_tiny_state(seed) for seed in (101, 202, 303, 404, 505)]
    out = s3t.five_checkpoint_signatures(states)
    assert out["all_distinct"]
    assert len(set(out["signatures"])) == 5


def test_five_checkpoint_signatures_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="five"):
        s3t.five_checkpoint_signatures([_tiny_state(1)] * 2)


def test_five_checkpoint_signatures_rejects_duplicates() -> None:
    states = [_tiny_state(1)] * 5
    with pytest.raises(ValueError, match="distinct"):
        s3t.five_checkpoint_signatures(states)


# --- seed-pair enforcement ---------------------------------------------------------


def test_seed_pair_for_index_frozen() -> None:
    assert s3t.seed_pair_for_index(0) == (101, 1101)
    assert s3t.seed_pair_for_index(4) == (505, 1505)
    assert [s3t.seed_pair_for_index(i) for i in range(5)] == list(SEED_PAIRS)


def test_seed_pair_for_index_out_of_range() -> None:
    with pytest.raises(ValueError):
        s3t.seed_pair_for_index(5)
    with pytest.raises(ValueError):
        s3t.seed_pair_for_index(-1)


# --- cross-fit splits -----------------------------------------------------------------


def test_crossfit_splits_no_episode_leakage() -> None:
    train_indices = np.arange(30)
    rng = np.random.default_rng(0)
    cf_action = rng.integers(0, 5, 30)
    cf_duration = rng.choice([1, 3], 30)
    episode_seed = np.arange(30) + 900
    splits = s3t.build_crossfit_splits(train_indices, cf_action, cf_duration, episode_seed)
    fold0 = set(splits["fold0"].tolist())
    fold1 = set(splits["fold1"].tolist())
    assert fold0.isdisjoint(fold1)
    assert fold0 | fold1 == set(train_indices.tolist())


def test_crossfit_fold_masks_cover_and_disjoint() -> None:
    cf_action = np.zeros(20, dtype=int)
    cf_duration = np.ones(20, dtype=int)
    episode_seed = np.arange(20)
    fold0, fold1 = s3t.crossfit_fold_masks(cf_action, cf_duration, episode_seed)
    assert not (fold0 & fold1).any()
    assert (fold0 | fold1).all()


# --- compute budget ----------------------------------------------------------------------


def _benchmarks(device="cuda"):
    return {
        "response": {
            "epoch_seconds": 2.0, "peak_memory_gb": 3.5, "device": device,
            "date": "2026-09-09", "source": "full-epoch response benchmark",
        },
        "router": {
            "epoch_seconds": 3.0, "peak_memory_gb": 4.5, "device": device,
            "date": "2026-09-09", "source": "full-epoch router benchmark",
        },
        "collection_episodes_per_second": 20.0,
        "inference_episodes_per_second": 50.0,
        "bootstrap_resamples_per_second": 2000.0,
    }


def test_search_space_is_bounded_and_registered() -> None:
    configs = s3t.search_space_configs()
    assert len(configs) == 8  # 2 x 2 x 2
    for cfg in configs:
        assert cfg["hidden_dim"] in (64, 96)
        assert cfg["response_feature_dim"] in (32, 64)
        assert cfg["lambda_cls"] in (0.10, 0.25)


def test_budget_estimate_structure_and_math() -> None:
    est = s3t.build_budget_estimate(
        _benchmarks(), train_episodes=1400, benchmark_episodes=1400,
        assumed_epochs=60, checkpoint_gb_per_run=0.5,
    )
    # 8 single-seed screen + 2 finalists x 5 = 18 Stage 3A runs
    assert est["stage3a"]["single_seed_screen"]["runs"] == 8
    assert est["stage3a"]["five_seed_finalists"]["runs"] == 10
    assert est["stage3a"]["total_runs"] == 18
    # cross-fit (router): Direct+Dual x 5 = 10; Stage 3B two-tier: 4+6=10 conds x 5 = 50
    assert est["crossfit"]["runs"] == 10
    assert est["stage3b"]["tier1"]["runs"] == 20
    assert est["stage3b"]["tier2"]["runs"] == 30
    assert est["stage3b"]["runs"] == 50
    assert est["totals"]["total_runs"] == 78
    # uncertainty band wraps the point estimate
    band = est["stage3a"]["total_device_hours"]
    assert band["low"] == band["point"] * 0.5
    assert band["high"] == band["point"] * 2.0
    # per-config entries carry per-seed duration and memory
    assert len(est["stage3a"]["per_config"]) == 8
    assert est["stage3a"]["per_config"][0]["peak_memory_gb"] == 3.5
    # typical/worst totals
    assert est["totals"]["worst_hours"] == est["totals"]["typical_hours"] * 2.0
    # formal section present with three cost blocks
    for block in ("collection", "inference", "bootstrap"):
        assert block in est["formal"]
    # Stage 3B uses router timing, not response timing
    assert est["stage3b"]["device_hours"]["point"] != est["stage3a"]["total_device_hours"]["point"]


def test_budget_device_labels_cpu_vs_cuda() -> None:
    cuda = s3t.build_budget_estimate(
        _benchmarks("cuda"), train_episodes=100, benchmark_episodes=100,
        assumed_epochs=10, checkpoint_gb_per_run=0.5,
    )
    cpu = s3t.build_budget_estimate(
        _benchmarks("cpu"), train_episodes=100, benchmark_episodes=100,
        assumed_epochs=10, checkpoint_gb_per_run=0.5,
    )
    assert cuda["unit_labels"]["hours"] == "GPU 小时"
    assert cpu["unit_labels"]["hours"] == "CPU 小时"
    assert cpu["unit_labels"]["memory"] == "峰值内存(CPU RSS)"
    assert cuda["unit_labels"]["memory"] == "峰值显存(VRAM)"


def test_compute_budget_contains_all_required_sections(tmp_path) -> None:
    est = s3t.build_budget_estimate(
        _benchmarks(), train_episodes=1400, benchmark_episodes=1400,
        assumed_epochs=60, checkpoint_gb_per_run=0.5,
    )
    text = s3t.generate_compute_budget(est, tmp_path / "COMPUTE_BUDGET.md")
    assert (tmp_path / "COMPUTE_BUDGET.md").exists()
    for section in (
        "估算来源与外推", "每个配置、每个 seed", "单 seed 有界筛选",
        "入围配置补五 seed", "cross-fit 额外成本", "Stage 3B 分项预算",
        "失败、重试与断点续跑策略", "总训练次数", "存储需求",
        "tier-1", "tier-2", "正式集合", "典型总时长", "最坏总时长",
        "10,000× bootstrap",
    ):
        assert section in text, section


def test_compute_budget_cpu_never_claims_gpu(tmp_path) -> None:
    est = s3t.build_budget_estimate(
        _benchmarks("cpu"), train_episodes=100, benchmark_episodes=100,
        assumed_epochs=10, checkpoint_gb_per_run=0.5,
    )
    text = s3t.generate_compute_budget(est, tmp_path / "B.md")
    assert "GPU 小时" not in text
    assert "峰值显存" not in text
    assert "CPU 小时" in text
    assert "峰值内存(CPU RSS)" in text


def test_compute_budget_rejects_missing_fields(tmp_path) -> None:
    est = s3t.build_budget_estimate(
        _benchmarks(), train_episodes=1400, benchmark_episodes=1400,
        assumed_epochs=60, checkpoint_gb_per_run=0.5,
    )
    del est["crossfit"]
    with pytest.raises(ValueError, match="missing required fields"):
        s3t.generate_compute_budget(est, tmp_path / "COMPUTE_BUDGET.md")


# --- training checkpoint / resume -------------------------------------------------


def test_train_model_checkpoint_and_resume(tmp_path) -> None:
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 32).view(-1, 1)
    y = 2 * x
    batches = [{"x": x, "y": y}]
    val_batches = [{"x": x, "y": y}]

    def loss_fn(m, batch):
        return torch.nn.functional.mse_loss(m(batch["x"]), batch["y"])

    ckpt = tmp_path / "run.ckpt"
    model = torch.nn.Linear(1, 1)
    out1 = s3t.train_model(
        model, loss_fn, batches, val_batches, seed=0, max_epochs=3,
        early_stopping_patience=10, learning_rate=0.1, checkpoint_path=ckpt,
        run_fingerprint="test-run",
    )
    assert ckpt.exists()
    assert out1["epochs_run"] == 3

    # resume and run more epochs; must continue from epoch 3
    model2 = torch.nn.Linear(1, 1)
    out2 = s3t.train_model(
        model2, loss_fn, batches, val_batches, seed=0, max_epochs=6,
        early_stopping_patience=10, learning_rate=0.1, checkpoint_path=ckpt,
        resume=True, run_fingerprint="test-run",
    )
    assert out2["epochs_run"] == 6  # 3 + 3 more
    assert out2["best_val_loss"] <= out1["best_val_loss"] + 1e-6


def test_resume_refuses_wrong_fingerprint(tmp_path) -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters())
    ckpt = tmp_path / "run.ckpt"
    s3t.save_train_checkpoint(
        ckpt, model=model, optimizer=opt, epoch=0, best_val=1.0,
        best_state=None, history=[], run_fingerprint="run-A",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        s3t.load_train_checkpoint(
            ckpt, model=model, optimizer=opt, run_fingerprint="run-B"
        )


# --- training loop smoke -------------------------------------------------------------------


def test_train_model_reduces_loss_and_selects_best() -> None:
    torch.manual_seed(0)
    # tiny linear regression: model learns y = 2x
    model = torch.nn.Linear(1, 1)
    x = torch.linspace(-1, 1, 32).view(-1, 1)
    y = 2 * x
    batches = [{"x": x, "y": y}]
    val_batches = [{"x": x, "y": y}]

    def loss_fn(m, batch):
        return torch.nn.functional.mse_loss(m(batch["x"]), batch["y"])

    out = s3t.train_model(
        model, loss_fn, batches, val_batches,
        seed=0, max_epochs=200, early_stopping_patience=20,
        learning_rate=0.1, device="cpu",
    )
    assert out["best_val_loss"] < 0.05
    assert out["signature"]
    assert out["epochs_run"] <= 200
    # best val is the minimum over history
    vals = [h["val_loss"] for h in out["history"]]
    assert out["best_val_loss"] <= min(vals) + 1e-6


# --- global validation aggregation (Task 4, repair plan v0.1) ------------------- #


def _sum_squares_fn(model, batch):
    x = batch["x"].double()
    return {
        "reg_sum": x.pow(2).sum(),
        "ce_sum": x.abs().sum(),
        "valid_count": torch.tensor(float(x.numel()), dtype=torch.float64),
    }


def test_global_validation_metrics_invariant_to_batching_and_order() -> None:
    from ac_twin import stage3_response as s3r

    torch.manual_seed(0)
    values = torch.randn(37, 5)
    model = torch.nn.Identity()

    def batches(order, batch_size):
        return [{"x": values[order[i:i + batch_size]]} for i in range(0, len(order), batch_size)]

    base = s3t.global_validation_metrics(
        model, batches(np.arange(37), 32), _sum_squares_fn, s3r.selection_score, "cpu"
    )
    assert set(base) == {
        "validation_reg_loss", "validation_ce_loss", "validation_selection_score",
    }
    rng = np.random.default_rng(0)
    for batch_size in (1, 17, 32, 37):
        for _ in range(3):
            order = rng.permutation(37)
            out = s3t.global_validation_metrics(
                model, batches(order, batch_size), _sum_squares_fn, s3r.selection_score, "cpu"
            )
            for key, ref in base.items():
                assert abs(out[key] - ref) < 1e-7, f"{key} drifted at batch_size={batch_size}"


def test_train_model_early_stops_on_selection_score_with_sums_fn(tmp_path) -> None:
    """With val_sums_fn, best_val is the globally aggregated selection score."""
    from ac_twin import stage3_response as s3r

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 3)
    batches = [{"x": torch.randn(4, 3), "y": torch.randn(4, 3)} for _ in range(3)]
    val_batches = [{"x": torch.randn(9, 3), "y": torch.randn(9, 3)} for _ in range(2)]

    def loss_fn(m, b):
        return ((m(b["x"]) - b["y"]) ** 2).mean()

    def sums_fn(m, b):
        err = (m(b["x"]) - b["y"]).double()
        return {
            "reg_sum": err.pow(2).sum(),
            "ce_sum": err.abs().sum(),
            "valid_count": torch.tensor(float(err.numel()), dtype=torch.float64),
        }

    out = s3t.train_model(
        model, loss_fn, batches, val_batches, seed=0, max_epochs=2,
        early_stopping_patience=2, learning_rate=1e-3,
        val_sums_fn=sums_fn, selection_fn=s3r.selection_score,
    )
    assert "validation_reg_loss" in out
    assert "validation_ce_loss" in out
    assert out["best_val_loss"] == pytest.approx(out["validation_selection_score"])
    # selection score equals the fixed-coefficient combination of the breakdown
    assert out["validation_selection_score"] == pytest.approx(
        out["validation_reg_loss"] + 0.25 * out["validation_ce_loss"]
    )


# --- Stage 3A Plus Task 5: optimizer weight_decay (§4.3) --------------------------- #


def _linear_fixture():
    x = torch.linspace(-1, 1, 32).view(-1, 1)
    y = 2 * x
    batches = [{"x": x, "y": y}]
    val_batches = [{"x": x, "y": y}]

    def loss_fn(m, batch):
        return torch.nn.functional.mse_loss(m(batch["x"]), batch["y"])

    return batches, val_batches, loss_fn


def test_train_model_default_weight_decay_matches_explicit_zero() -> None:
    """Omitting weight_decay (default 0.0) must equal passing 0.0 exactly."""
    batches, val_batches, loss_fn = _linear_fixture()
    torch.manual_seed(7)
    m_default = torch.nn.Linear(1, 1)
    torch.manual_seed(7)
    m_zero = torch.nn.Linear(1, 1)
    out_default = s3t.train_model(
        m_default, loss_fn, batches, val_batches, seed=0, max_epochs=4,
        early_stopping_patience=10, learning_rate=0.05,
    )
    out_zero = s3t.train_model(
        m_zero, loss_fn, batches, val_batches, seed=0, max_epochs=4,
        early_stopping_patience=10, learning_rate=0.05, weight_decay=0.0,
    )
    assert out_default["signature"] == out_zero["signature"]
    assert out_default["best_val_loss"] == pytest.approx(out_zero["best_val_loss"], abs=1e-12)


def test_train_model_positive_weight_decay_changes_result() -> None:
    """weight_decay > 0 must reach the optimizer and change the trained weights."""
    batches, val_batches, loss_fn = _linear_fixture()
    torch.manual_seed(7)
    m_zero = torch.nn.Linear(1, 1)
    torch.manual_seed(7)
    m_decay = torch.nn.Linear(1, 1)
    out_zero = s3t.train_model(
        m_zero, loss_fn, batches, val_batches, seed=0, max_epochs=8,
        early_stopping_patience=10, learning_rate=0.05, weight_decay=0.0,
    )
    out_decay = s3t.train_model(
        m_decay, loss_fn, batches, val_batches, seed=0, max_epochs=8,
        early_stopping_patience=10, learning_rate=0.05, weight_decay=0.1,
    )
    assert out_zero["signature"] != out_decay["signature"]
    # Decoupled from the loss: L2 shrinkage must not increase the data-fit loss
    # here, but it must reduce the parameter L2 norm of the selected best state.
    norm_zero = sum(float(v.pow(2).sum()) for v in out_zero["state_dict"].values())
    norm_decay = sum(float(v.pow(2).sum()) for v in out_decay["state_dict"].values())
    assert norm_decay < norm_zero
