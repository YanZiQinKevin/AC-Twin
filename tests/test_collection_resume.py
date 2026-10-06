from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import collect_pilot as collector  # noqa: E402

from collect_pilot import (  # noqa: E402
    empty_collection_state,
    pack_collection_state,
    parse_args,
    record_future_short_counts,
    restore_collection_state,
    stratified_split,
)


def _synthetic_state() -> dict:
    return {
        "histories": [np.zeros((10, 25), np.float32), np.ones((10, 25), np.float32)],
        "factual_futures": [np.zeros((20, 25), np.float32), np.ones((20, 25), np.float32)],
        "counterfactual_futures": [
            np.full((20, 25), 2, np.float32),
            np.full((20, 25), 3, np.float32),
        ],
        "factual_action_plans": [np.zeros(20, np.int64), np.ones(20, np.int64)],
        "counterfactual_action_plans": [np.ones(20, np.int64), np.full(20, 2, np.int64)],
        "cf_action_values": [1, 2],
        "cf_duration_values": [1, 3],
        "episode_seeds": [101, 202],
        "t_stars": [10, 12],
        "prefix_action_values": [np.arange(10, dtype=np.int64), np.arange(12, dtype=np.int64)],
        "object_histories": [
            np.zeros((10, 5, 8), np.float32),
            np.ones((10, 5, 8), np.float32),
        ],
        "object_factual_futures": [
            np.zeros((20, 5, 8), np.float32),
            np.ones((20, 5, 8), np.float32),
        ],
        "object_counterfactual_futures": [
            np.full((20, 5, 8), 2, np.float32),
            np.full((20, 5, 8), 3, np.float32),
        ],
        "accepted_strata": Counter({(1, 1): 1, (2, 3): 1}),
        "skip_counts": Counter({"short_future": 4}),
        "prefix_max_abs_diff": 0.0,
        "identity_max_abs_diff": 0.0,
        "attempts": 6,
    }


def test_collection_parser_accepts_resume() -> None:
    args = parse_args(["--config", "x.json", "--resume"])

    assert args.config == "x.json"
    assert args.resume is True


def test_roundabout_collection_config_is_object_centric() -> None:
    config = json.loads(
        (ROOT / "config" / "object_centric_roundabout_collect.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["env_id"] == "roundabout-v0"
    assert config["save_object_centric"] is True
    assert config["n_pairs"] == 2000
    assert config["paths"]["data"].endswith(
        "roundabout_2000/object_twinworld_2000.npz"
    )


@pytest.mark.parametrize(
    ("filename", "env_id", "seed", "data_path"),
    [
        (
            "identity_holdout_merge_collect.json",
            "merge-v0",
            20261011,
            "data/identity_continuity_stage/merge_holdout_1000/"
            "object_twinworld_holdout_1000.npz",
        ),
        (
            "identity_holdout_roundabout_collect.json",
            "roundabout-v0",
            20261013,
            "data/identity_continuity_stage/roundabout_holdout_1000/"
            "object_twinworld_holdout_1000.npz",
        ),
    ],
)
def test_holdout_collection_configs_are_complete_and_frozen(
    filename: str, env_id: str, seed: int, data_path: str
) -> None:
    """Validate collection settings without constructing a highway environment."""
    config = json.loads((ROOT / "config" / filename).read_text(encoding="utf-8"))

    assert config["env_id"] == env_id
    assert config["collection_seed"] == seed
    assert config["n_pairs"] == 1000
    assert config["horizon"] == 20
    assert config["history_len"] == 10
    assert config["t_star_low"] == 10
    assert config["t_star_high"] == 15
    assert config["intervention_durations"] == [1, 3]
    assert config["vehicles_count"] == 25
    assert config["observed_vehicles_count"] == 5
    assert config["save_object_centric"] is True
    assert config["checkpoint_every_accepted"] == 100
    assert config["progress_every_accepted"] == 20
    assert config["paths"]["data"] == data_path
    for required in (
        "experiment_name",
        "duration",
        "simulation_frequency",
        "policy_frequency",
        "max_attempt_multiplier",
        "determinism_checks",
        "split",
        "paths",
    ):
        assert required in config


def test_collection_state_starts_with_explicit_termination_reason_counts() -> None:
    state = empty_collection_state()

    assert state["skip_counts"] == {
        "prefix_terminated": 0,
        "factual_future_short": 0,
        "counterfactual_future_short": 0,
    }


def test_future_short_counts_keep_factual_and_counterfactual_distinct() -> None:
    skip_counts = empty_collection_state()["skip_counts"]

    assert record_future_short_counts(skip_counts, factual_length=19,
                                      counterfactual_length=18, horizon=20)
    assert skip_counts["factual_future_short"] == 1
    assert skip_counts["counterfactual_future_short"] == 1
    assert not record_future_short_counts(skip_counts, factual_length=20,
                                          counterfactual_length=20, horizon=20)


def _minimal_final_payload(n_pairs: int = 2) -> dict[str, np.ndarray]:
    sample = np.zeros((n_pairs, 1), dtype=np.float32)
    actions = np.zeros((n_pairs, 1), dtype=np.int64)
    return {
        "history": sample,
        "factual_future": sample,
        "counterfactual_future": sample,
        "factual_actions": actions,
        "counterfactual_actions": actions,
        "cf_action": np.zeros(n_pairs, dtype=np.int64),
        "cf_duration": np.ones(n_pairs, dtype=np.int64),
        "episode_seed": np.arange(n_pairs, dtype=np.int64),
        "t_star": np.full(n_pairs, 10, dtype=np.int64),
        "prefix_actions": actions,
        "prefix_action_lengths": np.ones(n_pairs, dtype=np.int64),
        "split": np.zeros(n_pairs, dtype=np.int8),
        "factual_ego_xy": np.zeros((n_pairs, 1, 2), dtype=np.float32),
        "counterfactual_ego_xy": np.zeros((n_pairs, 1, 2), dtype=np.float32),
        "effect_xy": np.zeros((n_pairs, 1, 2), dtype=np.float32),
        "obs_shape": np.asarray([1, 1], dtype=np.int64),
        "action_mapping_json": np.asarray("{}", dtype=object),
        "env_config_json": np.asarray("{}", dtype=object),
        "collection_stats_json": np.asarray("{}", dtype=object),
        "config_json": np.asarray("{}", dtype=object),
    }


def _final_stats(n_pairs: int = 2) -> dict[str, int]:
    return {"requested_pairs": n_pairs, "accepted_pairs": n_pairs}


def _complete_collection_config(data_path: str = "data/holdout.npz") -> dict:
    return {
        "experiment_name": "holdout_unit_test",
        "env_id": "merge-v0",
        "collection_seed": 20261011,
        "save_object_centric": False,
        "n_pairs": 2,
        "horizon": 20,
        "history_len": 10,
        "t_star_low": 10,
        "t_star_high": 15,
        "intervention_durations": [1, 3],
        "vehicles_count": 25,
        "observed_vehicles_count": 5,
        "duration": 100,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "max_attempt_multiplier": 15,
        "determinism_checks": 20,
        "checkpoint_every_accepted": 100,
        "progress_every_accepted": 20,
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "paths": {"data": data_path, "output_dir": "experiments/holdout"},
    }


def test_completed_output_requires_the_current_complete_config(tmp_path: Path) -> None:
    assert hasattr(collector, "canonical_config_json")
    out_path = tmp_path / "holdout.npz"
    stats_path = out_path.with_suffix(".stats.json")
    expected = _complete_collection_config()
    payload = _minimal_final_payload()
    payload["config_json"] = np.asarray(json.dumps(expected), dtype=object)
    payload["collection_stats_json"] = np.asarray(
        json.dumps(_final_stats()), dtype=object
    )
    np.savez_compressed(out_path, **payload)
    stats_path.write_text(json.dumps(_final_stats()), encoding="utf-8")

    assert collector.resolve_collection_start(
        out_path, stats_path, tmp_path / "holdout.partial.npz", False,
        current_config=expected,
    ) == "complete"
    changed_seed = {**expected, "collection_seed": 20261012}
    with pytest.raises(RuntimeError, match="configuration"):
        collector.resolve_collection_start(
            out_path, stats_path, tmp_path / "holdout.partial.npz", False,
            current_config=changed_seed,
        )
    changed_paths = {
        **expected,
        "paths": {**expected["paths"], "data": "data/other_holdout.npz"},
    }
    with pytest.raises(RuntimeError, match="configuration"):
        collector.resolve_collection_start(
            out_path, stats_path, tmp_path / "holdout.partial.npz", False,
            current_config=changed_paths,
        )


def test_completed_output_requires_sidecar_stats_to_match_embedded_stats(
    tmp_path: Path,
) -> None:
    assert hasattr(collector, "canonical_config_json")
    out_path = tmp_path / "holdout.npz"
    stats_path = out_path.with_suffix(".stats.json")
    config = _complete_collection_config()
    payload = _minimal_final_payload()
    payload["config_json"] = np.asarray(json.dumps(config), dtype=object)
    payload["collection_stats_json"] = np.asarray(
        json.dumps({**_final_stats(), "attempts": 3}), dtype=object
    )
    np.savez_compressed(out_path, **payload)
    stats_path.write_text(
        json.dumps({**_final_stats(), "attempts": 4}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="embedded"):
        collector.resolve_collection_start(
            out_path, stats_path, tmp_path / "holdout.partial.npz", False,
            current_config=config,
        )


def test_existing_output_is_complete_only_when_npz_and_stats_validate(tmp_path: Path) -> None:
    assert hasattr(collector, "resolve_collection_start")
    out_path = tmp_path / "holdout.npz"
    stats_path = out_path.with_suffix(".stats.json")
    config = _complete_collection_config()
    payload = _minimal_final_payload()
    payload["config_json"] = np.asarray(json.dumps(config), dtype=object)
    payload["collection_stats_json"] = np.asarray(json.dumps(_final_stats()), dtype=object)
    np.savez_compressed(out_path, **payload)
    stats_path.write_text(json.dumps(_final_stats()), encoding="utf-8")

    assert collector.resolve_collection_start(
        out_path, stats_path, tmp_path / "holdout.partial.npz", False,
        current_config=config,
    ) == "complete"
    assert collector.resolve_collection_start(
        out_path, stats_path, tmp_path / "holdout.partial.npz", True,
        current_config=config,
    ) == "complete"
    stats_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        collector.resolve_collection_start(
            out_path, stats_path, tmp_path / "holdout.partial.npz", False,
            current_config=config,
        )
    stats_path.write_text(json.dumps(_final_stats()), encoding="utf-8")
    np.savez_compressed(out_path, **{"history": np.zeros((2, 1), dtype=np.float32)})
    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        collector.resolve_collection_start(
            out_path, stats_path, tmp_path / "holdout.partial.npz", False,
            current_config=config,
        )


def test_resume_mode_requires_a_partial_checkpoint_when_final_is_absent(tmp_path: Path) -> None:
    assert hasattr(collector, "resolve_collection_start")
    out_path = tmp_path / "holdout.npz"
    partial_path = out_path.with_suffix(".partial.npz")
    config = _complete_collection_config()

    assert collector.resolve_collection_start(
        out_path, out_path.with_suffix(".stats.json"), partial_path, False,
        current_config=config,
    ) == "new"
    partial_path.write_bytes(b"checkpoint")
    assert collector.resolve_collection_start(
        out_path, out_path.with_suffix(".stats.json"), partial_path, False,
        current_config=config,
    ) == "resume"
    partial_path.unlink()
    with pytest.raises(FileNotFoundError, match="--resume"):
        collector.resolve_collection_start(
            out_path, out_path.with_suffix(".stats.json"), partial_path, True,
            current_config=config,
        )


@pytest.mark.parametrize("failed_writer", ["npz", "stats"])
def test_final_commit_failure_keeps_partial_and_hides_final_outputs(
    tmp_path: Path, failed_writer: str
) -> None:
    assert hasattr(collector, "commit_collection_outputs")
    out_path = tmp_path / "holdout.npz"
    partial_path = out_path.with_suffix(".partial.npz")
    partial_path.write_bytes(b"usable checkpoint remains managed by the collector")
    partial_before = partial_path.read_bytes()

    def fail_npz(*args, **kwargs):
        raise OSError("npz write interrupted")

    def fail_stats(*args, **kwargs):
        raise OSError("stats write interrupted")

    with pytest.raises(OSError, match="interrupted"):
        collector.commit_collection_outputs(
            out_path,
            _minimal_final_payload(),
            _final_stats(),
            current_config=_complete_collection_config(),
            npz_writer=fail_npz if failed_writer == "npz" else np.savez_compressed,
            stats_writer=fail_stats if failed_writer == "stats" else None,
        )
    assert partial_path.exists()
    assert partial_path.read_bytes() == partial_before
    assert not out_path.exists()
    assert not out_path.with_suffix(".stats.json").exists()


def test_rollout_marks_only_prefix_termination_as_prefix_terminated(monkeypatch) -> None:
    class FakeEnv:
        def __init__(self, terminal_step: int):
            self.terminal_step = terminal_step
            self.steps = 0

        def reset(self, seed: int):
            return np.zeros((1, 5), dtype=np.float32), {}

        def step(self, action: int):
            self.steps += 1
            return (
                np.full((1, 5), self.steps, dtype=np.float32),
                0.0,
                self.steps == self.terminal_step,
                False,
                {},
            )

        def close(self):
            pass

    monkeypatch.setattr(collector, "make_env", lambda *args: FakeEnv(terminal_step=2))
    _, _, prefix_terminated = collector.rollout(
        "fake-v0", {}, 1, np.array([0]), np.array([0])
    )
    assert not prefix_terminated

    monkeypatch.setattr(collector, "make_env", lambda *args: FakeEnv(terminal_step=2))
    _, _, prefix_terminated = collector.rollout(
        "fake-v0", {}, 1, np.array([0, 0]), np.array([0])
    )
    assert prefix_terminated


def test_full_length_terminal_prefix_is_counted_before_future_short() -> None:
    assert hasattr(collector, "record_prefix_termination")
    skip_counts = empty_collection_state()["skip_counts"]

    assert collector.record_prefix_termination(
        skip_counts,
        factual_prefix_terminated=True,
        counterfactual_prefix_terminated=False,
        factual_prefix_length=11,
        counterfactual_prefix_length=11,
        t_star=10,
    )
    assert skip_counts["prefix_terminated"] == 1
    assert skip_counts["factual_future_short"] == 0
    assert skip_counts["counterfactual_future_short"] == 0


def test_collection_state_round_trip_preserves_variable_prefixes() -> None:
    state = _synthetic_state()

    metadata, arrays = pack_collection_state(state)
    restored = restore_collection_state(arrays, metadata)

    assert restored["attempts"] == state["attempts"]
    assert restored["accepted_strata"] == state["accepted_strata"]
    assert restored["skip_counts"] == state["skip_counts"]
    assert restored["episode_seeds"] == state["episode_seeds"]
    assert [len(x) for x in restored["prefix_action_values"]] == [10, 12]
    for expected, actual in zip(
        state["prefix_action_values"], restored["prefix_action_values"]
    ):
        np.testing.assert_array_equal(actual, expected)
    for key in (
        "histories",
        "factual_futures",
        "counterfactual_futures",
        "object_histories",
        "object_factual_futures",
        "object_counterfactual_futures",
    ):
        np.testing.assert_array_equal(np.stack(restored[key]), np.stack(state[key]))


def test_restore_rejects_inconsistent_accepted_count() -> None:
    metadata, arrays = pack_collection_state(_synthetic_state())
    metadata["accepted"] = 3

    with pytest.raises(ValueError, match="accepted count"):
        restore_collection_state(arrays, metadata)


def test_smoke_split_has_train_validation_and_test_samples() -> None:
    actions = np.repeat(np.arange(5), 8)
    durations = np.tile(np.repeat([1, 3], 4), 5)

    split = stratified_split(actions, durations, 0.50, 0.25, seed=7)

    assert np.bincount(split, minlength=3).tolist() == [20, 10, 10]
