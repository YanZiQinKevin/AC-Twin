from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

SPEC = importlib.util.spec_from_file_location("run_stage3r", CODE / "run_stage3r.py")
assert SPEC is not None and SPEC.loader is not None
run_stage3r = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_stage3r
SPEC.loader.exec_module(run_stage3r)


CONFIG = ROOT / "experiments" / "stage3_routing" / "stage3r" / "stage3r_config.json"


def test_validate_config_accepts_frozen_contract():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = run_stage3r.validate_config(config, project_root=ROOT)
    assert validated["k_folds"] == 5
    assert validated["lambda_cew"] == [0.25, 1.0]
    assert validated["lambda_rank"] == [0.0, 0.1]


def test_validate_config_rejects_historical_split_as_train():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["train_split"] = 2
    with pytest.raises(ValueError, match="frozen split roles"):
        run_stage3r.validate_config(config, project_root=ROOT)


def test_preflight_verifies_real_source_and_binding():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = run_stage3r.preflight(config, project_root=ROOT)
    assert report["status"] == "STAGE3R_PREFLIGHT_PASS"
    assert report["development_source_exists"] is True
    assert report["binding_exists"] is True
    assert report["historical_split_not_selectable"] is True


def test_engineering_smoke_runs_full_history_router_and_direct_on_cpu():
    report = run_stage3r.engineering_smoke(device="cpu", episodes=2, horizon=4)
    assert report["status"] == "ENGINEERING_SMOKE_ONLY"
    assert report["finite"] is True
    assert report["full_shape"] == [2, 4, 3, 2]
    assert report["history_suffix_invariant"] is True
    assert report["router_convex"] is True


def test_masked_expert_mse_ignores_invalid_cells():
    prediction = torch.tensor([[[[1.0, 0.0], [100.0, 100.0]]]])
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[[True, False]]])
    loss = run_stage3r.masked_expert_mse(prediction, target, valid)
    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_validate_oof_coverage_requires_each_train_episode_once():
    train_indices = np.array([0, 1, 2, 3, 4])
    run_stage3r.validate_oof_coverage(
        train_indices,
        [np.array([3, 0]), np.array([4]), np.array([1, 2])],
    )
    with pytest.raises(ValueError, match="exactly once"):
        run_stage3r.validate_oof_coverage(
            train_indices,
            [np.array([0, 1]), np.array([1, 2]), np.array([3, 4])],
        )


def test_expert_loss_and_ordered_prediction_use_registered_mask():
    generator = torch.Generator().manual_seed(8)
    tensors = {
        "history": torch.randn(4, 5, 3, 8, generator=generator),
        "factual_suffix": torch.randn(4, 5, 3, 8, generator=generator),
        "factual_actions": torch.randint(0, 5, (4, 4), generator=generator),
        "counterfactual_actions": torch.randint(0, 5, (4, 4), generator=generator),
        "effect_target": torch.randn(4, 4, 3, 2, generator=generator),
        "valid": torch.ones(4, 4, 3, dtype=torch.bool),
    }
    tensors["history"][..., 0] = 1.0
    tensors["factual_suffix"][..., 0] = 1.0
    model = run_stage3r.Stage3RExpert(
        "full", feature_dim=8, object_embedding_dim=12, hidden_dim=12
    )
    batch = {key: value[:2] for key, value in tensors.items()}
    loss = run_stage3r.expert_loss(model, batch)
    assert torch.isfinite(loss)
    prediction = run_stage3r.predict_expert_ordered(
        model, tensors, np.array([3, 1]), device="cpu", batch_size=1
    )
    assert prediction.shape == (2, 4, 3, 2)


def test_parser_accepts_resumable_oof_command():
    args = run_stage3r.build_parser().parse_args(
        ["oof-experts", "--config", str(CONFIG), "--resume"]
    )
    assert args.command == "oof-experts"
    assert args.resume is True


def test_parser_accepts_phase0_command():
    args = run_stage3r.build_parser().parse_args(
        ["phase0-headroom", "--config", str(CONFIG)]
    )
    assert args.command == "phase0-headroom"


def test_parser_accepts_oof_smoke_command():
    args = run_stage3r.build_parser().parse_args(
        ["oof-smoke", "--config", str(CONFIG)]
    )
    assert args.command == "oof-smoke"


def test_phase0_from_validation_arrays_selects_constant_and_soft_headroom():
    history = np.zeros((5, 40, 2, 1, 2), dtype=np.float32)
    full = np.zeros_like(history)
    full[..., 0] = 2.0
    target = np.zeros((40, 2, 1, 2), dtype=np.float32)
    target[:20, ..., 0] = 0.25
    target[20:, ..., 0] = 1.75
    valid = np.ones((40, 2, 1), dtype=bool)
    report = run_stage3r.phase0_from_validation_arrays(
        full, history, target, valid, n_resamples=500, seed=13
    )
    assert np.isclose(
        np.arange(0.0, 1.0001, 0.05), report["best_constant_w"]
    ).any()
    assert report["status"] == "STAGE3R_SOFT_HEADROOM_CONFIRMED"


def test_parser_accepts_router_development_commands():
    for command in (
        "train-router-candidates",
        "phase1-predictability",
        "select-finalist",
    ):
        arguments = [command, "--config", str(CONFIG)]
        if command == "train-router-candidates":
            arguments.append("--resume")
        args = run_stage3r.build_parser().parse_args(arguments)
        assert args.command == command


def test_router_objective_is_finite_and_backpropagates():
    generator = torch.Generator().manual_seed(29)
    shape = (3, 4, 2)
    model = run_stage3r.CEWRouter(7, 8)
    features = torch.randn(*shape, 7, generator=generator)
    full = torch.randn(*shape, 2, generator=generator)
    history = torch.randn(*shape, 2, generator=generator)
    target = torch.randn(*shape, 2, generator=generator)
    valid = torch.ones(shape, dtype=torch.bool)
    teacher_cew = torch.rand(shape, generator=generator)
    teacher_cea = torch.randn(shape, generator=generator)
    loss, weight = run_stage3r.router_objective(
        model,
        features,
        full,
        history,
        target,
        valid,
        teacher_cew,
        teacher_cea,
        valid,
        valid.float(),
        lambda_cew=0.25,
        lambda_rank=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert weight.shape == shape
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_parser_accepts_direct_and_power_commands():
    direct = run_stage3r.build_parser().parse_args(
        ["train-direct", "--config", str(CONFIG), "--resume"]
    )
    power = run_stage3r.build_parser().parse_args(
        ["power-rscreen", "--config", str(CONFIG)]
    )
    assert direct.resume is True
    assert power.command == "power-rscreen"


def test_parser_accepts_rscreen_freeze_inference_and_adjudication():
    for command in ("freeze-rscreen", "adjudicate-rscreen"):
        args = run_stage3r.build_parser().parse_args(
            [command, "--config", str(CONFIG)]
        )
        assert args.command == command
    inference = run_stage3r.build_parser().parse_args(
        ["infer-rscreen", "--config", str(CONFIG), "--resume"]
    )
    assert inference.resume is True
