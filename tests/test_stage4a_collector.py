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

import collect_stage4a_merge_dev as collector  # noqa: E402
from ac_twin import stage4_protocol as s4p  # noqa: E402


# --- rejection reasons carry no outcome-based filter ---------------------------------


def test_rejection_reasons_have_no_outcome_based_filter():
    # Accept/reject may only use pre-intervention quota + simulation validity;
    # there must be NO reason that reads the response, the effect error, which
    # expert is better, a collision/near-collision label or any risk score.
    forbidden_fragments = (
        "response", "effect", "error", "risk", "collision", "near",
        "crash", "expert", "better", "oracle", "label", "outcome",
    )
    for reason in collector._REASONS:
        for fragment in forbidden_fragments:
            assert fragment not in reason.lower(), (
                f"rejection reason {reason!r} looks outcome-based ({fragment})"
            )
    # The Stage 4A-specific reason is the pure quota rule.
    assert "stratum_quota_full" in collector._REASONS


# --- stratum quota selection -----------------------------------------------------------


def _state_with_counts(counts: dict[tuple[int, int], int]) -> dict:
    return {"accepted_strata": Counter(counts)}


def test_select_stratum_picks_least_filled_under_quota():
    rng = np.random.default_rng(0)
    quota = 2
    eligible = [(0, 1), (1, 1), (2, 1)]
    state = _state_with_counts({(0, 1): 2, (1, 1): 0, (2, 1): 1})
    for _ in range(20):
        chosen = collector._select_stratum(state, eligible, rng, quota)
        # (0,1) is full -> never returned; (1,1) is least-filled -> always chosen.
        assert chosen == (1, 1)


def test_select_stratum_returns_none_when_all_full():
    rng = np.random.default_rng(0)
    quota = 1
    eligible = [(0, 1), (1, 1)]
    state = _state_with_counts({(0, 1): 1, (1, 1): 1})
    assert collector._select_stratum(state, eligible, rng, quota) is None


def test_select_stratum_never_overfills_a_stratum():
    rng = np.random.default_rng(1)
    quota = 3
    strata = [(a, d) for a in s4p.CF_ACTIONS for d in s4p.CF_DURATIONS]
    state = _state_with_counts({key: 0 for key in strata})
    # Fill every stratum to quota via repeated selection; counts must never exceed.
    picks = 0
    while True:
        chosen = collector._select_stratum(state, strata, rng, quota)
        if chosen is None:
            break
        state["accepted_strata"][chosen] += 1
        picks += 1
        assert state["accepted_strata"][chosen] <= quota
    assert picks == quota * len(strata)  # exactly 3*10 selections then None
    assert collector._all_full(state, strata, quota)


def test_all_full_requires_every_stratum_at_quota():
    strata = [(0, 1), (1, 1)]
    quota = 2
    state = _state_with_counts({(0, 1): 2, (1, 1): 1})
    assert not collector._all_full(state, strata, quota)
    state["accepted_strata"][(1, 1)] = 2
    assert collector._all_full(state, strata, quota)


# --- ten-stratum definition -------------------------------------------------------------


def test_all_strata_is_ten_cells():
    config = {"cf_actions": [0, 1, 2, 3, 4], "intervention_durations": [1, 3]}
    strata = collector._all_strata(config)
    assert len(strata) == 10
    assert set(strata) == {(a, d) for a in (0, 1, 2, 3, 4) for d in (1, 3)}


# --- double-accept guard on the pack/restore round-trip ----------------------------------


def test_pack_state_rejects_inconsistent_accepted_count():
    state = collector._empty_state()
    # Simulate one accepted episode in the stacked fields but not the scalars.
    state["object_history"].append(np.zeros((10, 5, 8), dtype=np.float32))
    state["object_factual_future"].append(np.zeros((20, 5, 8), dtype=np.float32))
    state["object_counterfactual_future"].append(np.zeros((20, 5, 8), dtype=np.float32))
    # factual_actions etc. left empty -> inconsistent -> must raise (double-accept guard).
    with pytest.raises(ValueError, match="inconsistent accepted count"):
        collector._pack_state(state)


def test_pack_restore_round_trip_preserves_counts():
    state = collector._empty_state()
    capacity, hist, fut = 5, 10, 20
    # Two fully-specified accepted episodes.
    for seed in (101, 202):
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

    metadata, arrays = collector._pack_state(state)
    restored = collector._restore_state(arrays, metadata)
    assert len(restored["object_history"]) == 2
    assert restored["accepted_strata"][(1, 3)] == 2
    assert int(restored["attempts"]) == 2
    assert restored["episode_seed"] == [101, 202]
