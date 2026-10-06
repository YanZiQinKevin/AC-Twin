"""Blocking tests for the Stage 4D-D2 scenario split protocol.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` sections 4.2,
4.3 and 12.7: role zero-overlap, one-scenario-one-context, frozen rank rule,
frozen t_star candidate-order rule, primary/reserve disjointness, create-once.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ac_twin import stage4d_external_split as split
from ac_twin.stage4d_external_protocol import role_rank_key

ROLE_SALT = "AC-TWIN-D2-V0.1|"
ROLE_TARGETS = {
    "smoke": 12,
    "pilot": 100,
    "train": 5000,
    "validation": 1500,
    "development_screen": 4000,
}


def _record(scenario_id: str, shard: str = "training_20s.tfrecord-00000-of-01000",
            timesteps: int = 199, tracks: int = 20, maps: int = 100,
            sdc: int | None = 7) -> dict:
    return {
        "scenario_id": scenario_id,
        "source_shard": shard,
        "num_timesteps": timesteps,
        "num_tracks": tracks,
        "num_map_features": maps,
        "sdc_track_id": sdc,
        "sdc_track_index": 0,
    }


# --- t_star candidate-order rule -------------------------------------------------


def test_t_star_order_is_deterministic_permutation():
    candidates = [10, 11, 12, 13, 14, 15]
    order1 = split.t_star_candidate_order("abcdef0123456789", ROLE_SALT, candidates)
    order2 = split.t_star_candidate_order("abcdef0123456789", ROLE_SALT, candidates)
    assert order1 == order2
    assert sorted(order1) == candidates


def test_t_star_order_varies_across_scenarios():
    candidates = [10, 11, 12, 13, 14, 15]
    orders = {
        tuple(split.t_star_candidate_order(f"scen{i:016x}", ROLE_SALT, candidates))
        for i in range(50)
    }
    # 720 permutations exist; 50 random scenarios must not collapse to one order.
    assert len(orders) > 1


def test_t_star_order_rejects_empty_candidates():
    with pytest.raises(ValueError):
        split.t_star_candidate_order("abcdef0123456789", ROLE_SALT, [])


# --- index pre-filter -------------------------------------------------------------


def test_prefilter_accepts_nominal_record():
    assert split.passes_index_prefilter(_record("a" * 16))


@pytest.mark.parametrize(
    "field, value",
    [
        ("timesteps", split.PREFILTER_MIN_TIMESTEPS - 1),
        ("tracks", split.PREFILTER_MIN_TRACKS - 1),
        ("maps", split.PREFILTER_MIN_MAP_FEATURES - 1),
    ],
)
def test_prefilter_rejects_degenerate_counts(field, value):
    kwargs = {field: value}
    assert not split.passes_index_prefilter(_record("b" * 16, **kwargs))


def test_prefilter_rejects_missing_sdc():
    assert not split.passes_index_prefilter(_record("c" * 16, sdc=None))


def test_prefilter_boundary_values_accepted():
    assert split.passes_index_prefilter(
        _record(
            "d" * 16,
            timesteps=split.PREFILTER_MIN_TIMESTEPS,
            tracks=split.PREFILTER_MIN_TRACKS,
            maps=split.PREFILTER_MIN_MAP_FEATURES,
        )
    )


# --- role pool assignment ----------------------------------------------------------

# Small synthetic quotas (the frozen CONFIRM_POOL_QUOTA default is exercised only
# against the real 70k-scenario index; synthetic pools use a scaled-down contract).
SMALL_TARGETS = {
    "smoke": 2,
    "pilot": 4,
    "train": 20,
    "validation": 6,
    "development_screen": 16,
}
SMALL_CONFIRM_QUOTA = 12
SMALL_TOTAL = sum(SMALL_TARGETS.values()) + SMALL_CONFIRM_QUOTA  # 60


def _small_pools(records):
    return split.assign_role_pools(
        records, ROLE_SALT, SMALL_TARGETS, confirm_quota=SMALL_CONFIRM_QUOTA
    )


def _synthetic_index(n: int) -> list[dict]:
    return [_record(f"{i:064x}", shard=f"training_20s.tfrecord-{i % 1000:05d}-of-01000")
            for i in range(n)]


def _small_quota(role: str) -> int:
    return SMALL_CONFIRM_QUOTA if role == "confirm" else SMALL_TARGETS[role]


def test_role_pools_disjoint_and_cover_all_survivors():
    records = _synthetic_index(300)
    rows = _small_pools(records)
    assert len(rows) == 300  # all synthetic records pass the pre-filter
    ids = [r["scenario_id"] for r in rows]
    assert len(set(ids)) == len(ids)  # one-scenario-one-pool (zero overlap)
    assert [r["global_rank"] for r in rows] == list(range(len(rows)))


def test_role_pools_follow_frozen_hash_rank_order():
    records = _synthetic_index(300)
    rows = _small_pools(records)
    keys = [role_rank_key(r["scenario_id"], ROLE_SALT) for r in rows]
    assert keys == sorted(keys)


def test_role_primary_counts_match_quotas_and_reserve_is_tail():
    records = _synthetic_index(600)
    rows = _small_pools(records)
    stats = split._split_stats(rows)
    for role in split.ROLE_ORDER:
        quota = _small_quota(role)
        assert stats[role]["primary"] == quota
        assert stats[role]["pool_size"] >= quota
    # primary rows are exactly the first `quota` rows of each role segment
    for role in split.ROLE_ORDER:
        role_rows = [r for r in rows if r["role"] == role]
        quota = _small_quota(role)
        assert all(r["pool"] == "primary" for r in role_rows[:quota])
        assert all(r["pool"] == "reserve" for r in role_rows[quota:])
    # role segments are contiguous and follow the frozen role order
    bounds = [(stats[r]["rank_lo"], stats[r]["rank_hi"]) for r in split.ROLE_ORDER]
    for (lo, hi), (next_lo, _next_hi) in zip(bounds, bounds[1:]):
        assert hi + 1 == next_lo
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(rows) - 1


def test_prefilter_survivors_only_enter_pools():
    records = _synthetic_index(300)
    records.append(_record("f" * 64, tracks=2))  # ineligible at index level
    rows = _small_pools(records)
    assert len(rows) == 300
    assert all(r["scenario_id"] != "f" * 64 for r in rows)


def test_assignment_rejects_too_few_survivors():
    records = _synthetic_index(SMALL_TOTAL - 1)
    with pytest.raises(ValueError):
        _small_pools(records)


def test_assignment_deterministic_across_calls():
    records = _synthetic_index(300)
    rows1 = _small_pools(records)
    rows2 = _small_pools(list(reversed(records)))
    assert rows1 == rows2


# --- manifest build / verify (real workspace, create-once) -------------------------

PLATFORM_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments/stage4_reliability/stage4d_external/platform"
)


def test_frozen_split_manifest_verifies_against_real_index():
    """If the split has been frozen in this workspace, it must re-verify exactly."""
    manifest_path = PLATFORM_DIR / split.MANIFEST_FILE
    if not manifest_path.is_file():
        pytest.skip("scenario split not yet frozen in this workspace")
    from ac_twin.stage4d_external_protocol import load_config

    config = load_config()
    result = split.verify_scenario_split(config)
    assert result["status"] == "SPLIT_VERIFIED"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "SCENARIO_SPLIT_FROZEN"
    assert manifest["plan_id"] == "AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2"


def test_split_manifest_is_create_once():
    manifest_path = PLATFORM_DIR / split.MANIFEST_FILE
    if not manifest_path.is_file():
        pytest.skip("scenario split not yet frozen in this workspace")
    from ac_twin.stage4d_external_protocol import load_config

    config = load_config()
    with pytest.raises(FileExistsError):
        split.build_scenario_split_manifest(
            config, code_self_path=Path(__file__).resolve().parents[1]
            / "code/ac_twin/stage4d_external_split.py"
        )
