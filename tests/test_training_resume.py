from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_architecture_comparison import (  # noqa: E402
    build_model,
    load_run_metadata,
    parse_args,
    run_signature,
    sha256_file,
    train_one,
    write_run_metadata,
)


CONDITION = {
    "name": "row_65k",
    "encoder_type": "row_sorted_gru",
    "hidden_dim": 54,
    "capacity_tier": "65k",
    "suffix_source": "same_episode",
}
CONFIG = {
    "factual_suffix_steps": 10,
    "shuffle_seed": 7,
    "model": {
        "action_embedding_dim": 8,
        "batch_size": 128,
        "max_epochs": 200,
        "early_stopping_patience": 25,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "training_seeds": [11, 22, 33, 44, 55],
    },
    "statistics": {"bootstrap_resamples": 10000, "confidence_level": 0.95},
}


def test_training_parser_accepts_resume() -> None:
    args = parse_args(["--config", "x.json", "--resume"])

    assert isinstance(args, argparse.Namespace)
    assert args.config == "x.json"
    assert args.resume is True


def test_run_signature_changes_with_dataset_or_condition() -> None:
    first = run_signature(CONDITION, CONFIG, "aaa")

    assert first != run_signature(CONDITION, CONFIG, "bbb")
    assert first != run_signature({**CONDITION, "hidden_dim": 55}, CONFIG, "aaa")


def test_sha256_file_hashes_content(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    target.write_bytes(b"abc")

    assert sha256_file(target) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_run_metadata_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "row_65k_seed11_run.json"
    summary = {"status": "complete", "seed": 11, "FDE": 1.25}

    write_run_metadata(target, signature="correct", summary=summary)
    loaded = load_run_metadata(target, expected_signature="correct")

    assert loaded["signature"] == "correct"
    assert loaded["summary"] == summary


def test_resume_rejects_incompatible_run_metadata(tmp_path: Path) -> None:
    target = tmp_path / "row_65k_seed11_run.json"
    write_run_metadata(target, signature="old", summary={"status": "complete"})

    with pytest.raises(ValueError, match="does not match"):
        load_run_metadata(target, expected_signature="new")


def test_resume_rejects_incomplete_run_metadata(tmp_path: Path) -> None:
    target = tmp_path / "row_65k_seed11_run.json"
    target.write_text(
        json.dumps({"signature": "correct", "summary": {"status": "running"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not complete"):
        load_run_metadata(target, expected_signature="correct")


def test_train_one_reports_epoch_metrics() -> None:
    generator = np.random.default_rng(7)
    history = generator.normal(size=(6, 10, 26)).astype(np.float32)
    suffix = generator.normal(size=(6, 20, 26)).astype(np.float32)
    factual_actions = np.zeros((6, 20), np.int64)
    counterfactual_actions = np.ones((6, 20), np.int64)
    targets = generator.normal(size=(6, 20, 2)).astype(np.float32)
    model_config = {
        "action_embedding_dim": 8,
        "batch_size": 2,
        "max_epochs": 1,
        "early_stopping_patience": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    model = build_model({**CONDITION, "hidden_dim": 8}, CONFIG)
    records: list[dict] = []

    _, prediction, log = train_one(
        model,
        history,
        suffix,
        factual_actions,
        counterfactual_actions,
        targets,
        np.asarray([0, 1, 2, 3]),
        np.asarray([4]),
        np.asarray([5]),
        model_config,
        seed=11,
        device=torch.device("cpu"),
        progress_callback=records.append,
    )

    assert prediction.shape == (1, 20, 2)
    assert len(log) == 1
    assert len(records) == 1
    assert set(records[0]) >= {
        "epoch",
        "max_epochs",
        "train_mse",
        "val_mse",
        "best_val_mse",
    }
