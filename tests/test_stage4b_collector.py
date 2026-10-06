"""Stage 4B collector tests: no-outcome rejection, quota, single-role payload, resume."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import collect_stage4b_merge_confirm as collector  # noqa: E402
from ac_twin import stage4b_protocol as s4bp  # noqa: E402


# --- rejection reasons carry no outcome-based filter -------------------------------------


def test_rejection_reasons_have_no_outcome_based_filter():
    forbidden_fragments = (
        "response", "effect", "error", "risk", "collision", "near",
        "crash", "expert", "better", "oracle", "label", "outcome",
    )
    for reason in collector._REASONS:
        for fragment in forbidden_fragments:
            assert fragment not in reason.lower(), (
                f"rejection reason {reason!r} looks outcome-based ({fragment})"
            )
    assert "stratum_quota_full" in collector._REASONS


# --- stratum quota --------------------------------------------------------------------------


def test_select_stratum_picks_least_filled_under_quota():
    rng = np.random.default_rng(0)
    state = {"accepted_strata": Counter({(0, 1): 2, (1, 1): 0, (2, 1): 1})}
    for _ in range(20):
        assert collector._select_stratum(state, [(0, 1), (1, 1), (2, 1)], rng, 2) == (1, 1)


def test_select_stratum_none_when_all_full():
    rng = np.random.default_rng(0)
    state = {"accepted_strata": Counter({(0, 1): 1, (1, 1): 1})}
    assert collector._select_stratum(state, [(0, 1), (1, 1)], rng, 1) is None


def test_all_strata_is_ten_cells():
    config = {"cf_actions": [0, 1, 2, 3, 4], "intervention_durations": [1, 3]}
    assert len(collector._all_strata(config)) == 10


# --- single-role payload: NO split column ----------------------------------------------------


def _accepted_state(n=2):
    state = collector._empty_state()
    capacity, hist, fut = 5, 10, 20
    for i in range(n):
        seed = 1000 + i
        state["object_history"].append(np.full((hist, capacity, 8), seed, dtype=np.float32))
        state["object_factual_future"].append(np.zeros((fut, capacity, 8), dtype=np.float32))
        state["object_counterfactual_future"].append(np.ones((fut, capacity, 8), dtype=np.float32))
        state["factual_actions"].append(np.zeros(fut, dtype=np.int64))
        state["counterfactual_actions"].append(np.ones(fut, dtype=np.int64))
        for prefix in ("factual", "counterfactual"):
            state[f"{prefix}_lane_index"].append(np.zeros(hist + fut, dtype=np.int32))
            state[f"{prefix}_target_lane_index"].append(np.zeros(hist + fut, dtype=np.int32))
            state[f"{prefix}_cmd_accel"].append(np.zeros(hist + fut, dtype=np.float32))
            state[f"{prefix}_cmd_steer"].append(np.zeros(hist + fut, dtype=np.float32))
            state[f"{prefix}_crashed"].append(np.zeros(hist + fut, dtype=bool))
            state[f"{prefix}_terminated"].append(False)
            state[f"{prefix}_truncated"].append(False)
            state[f"{prefix}_first_contact_step"].append(-1)
            state[f"{prefix}_termination_reason"].append("")
            state[f"{prefix}_fingerprints_json"].append("[]")
        state["cf_action"].append(1)
        state["cf_duration"].append(3)
        state["episode_seed"].append(seed)
        state["t_star"].append(12)
        state["initial_fingerprint_consistent"].append(True)
        state["vehicle_metadata_json"].append("[]")
        state["accepted_strata"][(1, 3)] += 1
        state["attempts"] += 1
    return state


def test_payload_has_no_role_split_column():
    import collect_stage2_holdout as engine

    state = _accepted_state(2)
    config = {"cf_actions": [0, 1, 2, 3, 4], "intervention_durations": [1, 3]}
    payload = collector._build_payload(config, state, capacity=5, dt=0.2)
    # Stage 4B has a single confirm role -> no per-episode split is written.
    assert "split" not in payload
    assert payload["object_history"].shape[0] == 2


def test_pack_restore_round_trip_preserves_counts():
    state = _accepted_state(2)
    metadata, arrays = collector._pack_state(state)
    restored = collector._restore_state(arrays, metadata)
    assert len(restored["object_history"]) == 2
    assert restored["accepted_strata"][(1, 3)] == 2
    assert restored["episode_seed"] == [1000, 1001]


def test_pack_state_rejects_inconsistent_accepted_count():
    state = collector._empty_state()
    state["object_history"].append(np.zeros((10, 5, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="inconsistent accepted count"):
        collector._pack_state(state)
