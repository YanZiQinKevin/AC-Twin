"""Integration test for the Stage 4D-D2 MetaDrive adapter on REAL converted scenarios.

Plan ``AC-TWIN-RFER-STAGE4D-D2-METADRIVE-EXTERNAL-2026-09-21-V0.2`` section 12.7.
This exercises the full 11-branch group runner against a throwaway mini-DB built
from already-converted pkls. It is an ENGINEERING test (tracking/contract/determinism
correctness); it consumes no pilot/screen/confirm resource and never rejects on
outcome. Skips cleanly if the DB has no eligible scenario yet.
"""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from ac_twin import stage4d_external_features as F
import stage4d_metadrive_adapter as A

ROLE_SALT = "AC-TWIN-D2-V0.1|"
DB_DIR = Path(r"D:\MetaDrive\scenarionet_waymo_training_20s")

pytestmark = pytest.mark.skipif(not DB_DIR.is_dir(), reason="conversion DB not present")


def _iter_pkls(limit: int):
    count = 0
    for p in sorted(DB_DIR.glob("sd_waymo_v1.2_*.pkl")):
        yield p
        count += 1
        if count >= limit:
            return


def _load(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def _find_eligible(max_scan: int = 250):
    """Scan converted pkls for the first eligible scenario (plan 4.2)."""
    for p in _iter_pkls(max_scan):
        try:
            sd = _load(p)
        except Exception:  # noqa: BLE001 - skip a concurrently-written file
            continue
        sid = str(sd.get("metadata", {}).get("scenario_id", sd.get("id", p.stem)))
        t_star, reason, info, _ = F.select_t_star(sd, sid, ROLE_SALT)
        if t_star is not None:
            return p, sd, sid, t_star, info
    return None


def _build_mini_db(tmp_path: Path, pkls: list[Path]) -> Path:
    db = tmp_path / "mini_db"
    db.mkdir(parents=True)
    summary, mapping = {}, {}
    for src in pkls:
        dst = db / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        sd = _load(dst)
        meta = sd.get("metadata", {})
        summary[src.name] = {
            "id": str(meta.get("scenario_id", sd.get("id"))),
            "scenario_id": str(meta.get("scenario_id", sd.get("id"))),
            "coordinate": meta.get("coordinate", "waymo"),
            "source_file": str(meta.get("source_file", "")),
        }
        mapping[src.name] = ""
    with open(db / "dataset_summary.pkl", "wb") as fh:
        pickle.dump(summary, fh)
    with open(db / "dataset_mapping.pkl", "wb") as fh:
        pickle.dump(mapping, fh)
    return db


@pytest.fixture(scope="module")
def eligible_group(tmp_path_factory):
    found = _find_eligible()
    if found is None:
        pytest.skip("no eligible scenario among converted pkls yet")
    p, sd, sid, t_star, info = found
    tmp = tmp_path_factory.mktemp("stage4d_adapter")
    db = _build_mini_db(tmp, [p])
    return db, p, sd, sid, t_star, info


def test_group_context_builds(eligible_group):
    _db, _p, sd, sid, t_star, info = eligible_group
    ctx = A.prepare_group_context(sd, sid, ROLE_SALT)
    assert ctx.t_star == t_star
    assert ctx.ego_lane_id == info["ego_lane_id"]
    assert len(ctx.eval_track_ids) == F.OBJECT_CAPACITY
    assert ctx.eval_track_ids[0] == str(sd["metadata"]["sdc_id"])
    assert ctx.factual_plan.shape == (F.FUTURE_LEN,)
    assert ctx.ego_lane_id in ctx.target_lanes
    # ego lane adjacency pinned to eligibility neighbours
    assert ctx.adjacency[ctx.ego_lane_id] == (ctx.left_lane_id, ctx.right_lane_id)


def test_history_matches_recorded_track(eligible_group):
    _db, _p, sd, sid, t_star, _info = eligible_group
    ctx = A.prepare_group_context(sd, sid, ROLE_SALT)
    hist = A.history_features_from_track(sd, ctx)
    assert hist.shape == (F.HISTORY_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
    # last history frame is t_star; the ego (slot 0) projects to the frame origin in s,
    # and its d equals its true (small) lateral offset from the reference centreline.
    assert hist[-1, 0, 0] == 1.0  # presence
    assert hist[-1, 0, 1] == 1.0  # is_ego
    assert abs(hist[-1, 0, 2]) < 1e-4  # s == 0 at the ego-projection origin
    assert abs(hist[-1, 0, 3]) < F.EGO_LANE_MAX_LATERAL_M  # on the ego lane
    # ego moves forward along +s over the history (monotone-ish)
    ego_s = hist[:, 0, 2]
    assert ego_s[-1] > ego_s[0]


def test_full_group_run_shapes_and_determinism(eligible_group):
    db, _p, sd, sid, t_star, _info = eligible_group
    ctx = A.prepare_group_context(sd, sid, ROLE_SALT)
    index = A.scenario_id_to_index(str(db))[sid]
    env = A.build_env(str(db), num_scenarios=1)
    try:
        out1 = A.run_group(env, sd, index, ctx)
    finally:
        env.close()
    # Double-replay determinism gate semantics: a fresh env (clean start, as a full
    # collection re-run produces) must reproduce the tensors exactly. (Same-env
    # back-to-back repetition accumulates engine state and is NOT the gate.)
    env2 = A.build_env(str(db), num_scenarios=1)
    try:
        out2 = A.run_group(env2, sd, index, ctx)
    finally:
        env2.close()

    G_future = out1["object_factual_future"]
    assert out1["object_history"].shape == (F.HISTORY_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
    assert G_future.shape == (F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
    assert out1["object_counterfactual_futures"].shape == (
        F.N_ACTIONS, F.FUTURE_LEN, F.OBJECT_CAPACITY, F.N_OBJECT_FEATURES)
    assert out1["factual_actions"].shape == (F.FUTURE_LEN,)
    assert out1["cf_actions"].shape == (F.N_ACTIONS,)
    assert out1["cf_durations"].shape == (F.N_ACTIONS,)
    assert out1["inference_mask"].shape == (F.FUTURE_LEN, F.OBJECT_CAPACITY)
    assert out1["inference_mask"].dtype == bool
    assert out1["branches_completed"]

    # determinism: identical rerun -> identical tensors (double-replay gate)
    for key in ("object_history", "object_factual_future", "object_counterfactual_futures"):
        assert np.allclose(out1[key], out2[key], atol=1e-5), key

    # the ego is present and moving forward in the factual future
    ego_s = G_future[:, 0, 2]
    assert (G_future[:, 0, 0] > 0).all()  # ego present every future frame
    assert ego_s[-1] > ego_s[0]


def test_counterfactual_branches_diverge_from_factual(eligible_group):
    db, _p, sd, sid, _t, _i = eligible_group
    ctx = A.prepare_group_context(sd, sid, ROLE_SALT)
    index = A.scenario_id_to_index(str(db))[sid]
    env = A.build_env(str(db), num_scenarios=1)
    try:
        out = A.run_group(env, sd, index, ctx)
    finally:
        env.close()
    factual = out["object_factual_future"][:, 0, :]  # ego features over future
    cf = out["object_counterfactual_futures"][:, :, 0, :]  # ego per CF branch
    # at least one counterfactual action must move the ego off the factual path
    diverged = any(
        not np.allclose(factual[:, 2:4], cf[a, :, 2:4], atol=1e-3)
        for a in range(F.N_ACTIONS)
    )
    assert diverged, "no counterfactual branch diverged from the factual ego path"
