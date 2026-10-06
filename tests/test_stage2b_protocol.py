"""Task 1 blocking tests: Stage 2B protocol — config validation, seed schedules,
historical/cross-role disjointness, collection manifests, the mechanical pilot
profile-selection rule and the P0 movement/conflict lookup.

No highway_env import happens here (pytest runs under ``-W error``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage2b_protocol as sp  # noqa: E402
from ac_twin import stage2b_data as sd  # noqa: E402

SMOKE_CONFIG = ROOT / "config" / "stage2b_intersection_smoke.json"
PILOT_CONFIG = ROOT / "config" / "stage2b_intersection_pilot.json"
DEV_CONFIG = ROOT / "config" / "stage2b_intersection_development.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _smoke() -> dict:
    return _load(SMOKE_CONFIG)


# --- config validation ---------------------------------------------------------


def test_smoke_config_validates():
    sp.validate_stage2b_config(_smoke())


def test_pilot_container_validates_and_resolves():
    container = _load(PILOT_CONFIG)
    sp.validate_pilot_container_config(container)
    for letter, count in (("A", 10), ("B", 15)):
        effective = sp.resolve_pilot_profile_config(container, letter)
        assert effective["role"] == f"pilot_{letter.lower()}"
        assert effective["initial_vehicle_count"] == count
        assert effective["n_pairs"] == 600 and effective["per_stratum"] == 100
        assert effective["max_attempts"] == 12000
        sp.validate_stage2b_config(effective)


def test_development_config_validates_after_profile_injection():
    config = _load(DEV_CONFIG)
    assert config["initial_vehicle_count"] is None
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(config)  # unresolved profile must not validate
    selection = {"selected_profile": "B", "initial_vehicle_count": 15}
    resolved = sp.resolve_development_config(config, selection)
    assert resolved["initial_vehicle_count"] == 15
    assert resolved["selected_profile"] == "B"
    sp.validate_stage2b_config(resolved)


def test_development_rejects_unselected_profile():
    config = _load(DEV_CONFIG)
    with pytest.raises(ValueError):
        sp.resolve_development_config(
            config, {"selected_profile": None, "initial_vehicle_count": None})


@pytest.mark.parametrize("key,value", [
    ("spawn_probability", 0.5),
    ("policy_frequency", 10),
    ("simulation_frequency", 10),
    ("duration", 13),
    ("object_capacity", 21),
    ("observed_vehicles_count", 15),
    ("controlled_vehicles", 2),
    ("horizon", 10),
    ("history_len", 5),
])
def test_frozen_keys_reject_deviation(key, value):
    cfg = _smoke()
    cfg[key] = value
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)


def test_rejects_wrong_env_scene_plan():
    cfg = _smoke()
    cfg["env_id"] = "merge-v0"
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)
    cfg = _smoke()
    cfg["scene"] = "merge"
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)
    cfg = _smoke()
    cfg["plan_id"] = "AC-TWIN-RFER-STAGE2-HOLDOUT-2026-09-07-V1"
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)


def test_rejects_wrong_role_and_handmade_confirm():
    cfg = _smoke()
    cfg["role"] = "not_a_role"
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)
    cfg = _smoke()
    cfg["role"] = "confirm"
    cfg["collection_seed"] = sp.COLLECTION_SEEDS["confirm"]
    with pytest.raises(ValueError, match="P3|generated"):
        sp.validate_stage2b_config(cfg)


def test_rejects_collection_seed_of_another_role():
    cfg = _smoke()
    cfg["collection_seed"] = sp.COLLECTION_SEEDS["pilot_a"]
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)


def test_rejects_top_level_retrain_keys():
    cfg = _smoke()
    cfg["learning_rate"] = 1e-3
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)


def test_development_rejects_modified_training_block():
    config = _load(DEV_CONFIG)
    selection = {"selected_profile": "A", "initial_vehicle_count": 10}
    resolved = sp.resolve_development_config(config, selection)
    resolved["training"]["hidden_dim"] = 96
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(resolved)


def test_development_rejects_modified_c1_block():
    config = _load(DEV_CONFIG)
    selection = {"selected_profile": "A", "initial_vehicle_count": 10}
    resolved = sp.resolve_development_config(config, selection)
    resolved["augmentation_c1"]["object_embedding_dim"] = 128
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(resolved)


def test_n_pairs_must_match_six_strata():
    cfg = _smoke()
    cfg["n_pairs"] = 10
    with pytest.raises(ValueError):
        sp.validate_stage2b_config(cfg)


def test_confirm_candidate_sizes_frozen():
    assert sp.CONFIRM_CANDIDATE_N == (1200, 1800, 2400, 3600)
    assert all(n % 6 == 0 for n in sp.CONFIRM_CANDIDATE_N)


# --- seed schedule -------------------------------------------------------------


def test_seed_schedule_range_unique():
    schedule = sp.make_seed_schedule(sp.COLLECTION_SEEDS["smoke"], 512)
    assert schedule.shape == (512,)
    assert len(np.unique(schedule)) == 512
    assert schedule.min() >= sp.SEED_LOW
    assert schedule.max() <= sp.SEED_HIGH_EXCLUSIVE - 1


def test_seed_schedule_rejects_historical_overlap():
    schedule = sp.make_seed_schedule(sp.COLLECTION_SEEDS["smoke"], 512)
    historical = np.array([schedule[0], 999999], dtype=np.int64)
    with pytest.raises(ValueError, match="overlap"):
        sp.validate_seed_schedule(schedule, historical)


def test_seed_schedule_rejects_cross_role_overlap():
    schedule = sp.make_seed_schedule(sp.COLLECTION_SEEDS["smoke"], 512)
    other = np.array([schedule[5]], dtype=np.int64)
    with pytest.raises(ValueError, match="role"):
        sp.validate_seed_schedule(schedule, np.array([7], dtype=np.int64),
                                  {"pilot_a": other})
    # Disjoint other-role schedule passes.
    sp.validate_seed_schedule(schedule, np.array([7], dtype=np.int64),
                              {"pilot_a": np.array([8, 9], dtype=np.int64)})


def test_frozen_collection_seeds_disjoint_from_live_history_and_each_other():
    """Every role's frozen collection seed must yield a schedule disjoint from
    the live historical seed pool and from every other role (plan 3.2)."""
    historical = sp.scan_historical_episode_seeds(ROOT / "data", ROOT / "experiments")
    sizes = {"smoke": sp.SMOKE_SEED_SCHEDULE_SIZE,
             "pilot_a": sp.PILOT_MAX_ATTEMPTS, "pilot_b": sp.PILOT_MAX_ATTEMPTS,
             "development": sp.DEV_SEED_SCHEDULE_SIZE,
             "confirm": sp.DEV_SEED_SCHEDULE_SIZE}  # conservative cap for confirm
    schedules = {role: sp.make_seed_schedule(sp.COLLECTION_SEEDS[role], size)
                 for role, size in sizes.items()}
    assert len(set(sp.COLLECTION_SEEDS.values())) == len(sp.COLLECTION_SEEDS)
    for role, schedule in schedules.items():
        sp.validate_seed_schedule(schedule, historical)
    roles = sorted(schedules)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            overlap = np.intersect1d(schedules[a], schedules[b])
            assert not overlap.size, f"roles {a}/{b} overlap: {overlap[:5]}"


# --- historical scan ------------------------------------------------------------


def test_historical_scan_excludes_stage2b_products(tmp_path):
    data = tmp_path / "data"
    (data / "object_centric_stage" / "merge_2000").mkdir(parents=True)
    (data / "object_centric_stage" / "stage2b_intersection_pilot_a").mkdir(parents=True)
    np.savez(data / "object_centric_stage" / "merge_2000" / "old.npz",
             episode_seed=np.array([11, 22, 33]))
    np.savez(data / "object_centric_stage" / "stage2b_intersection_pilot_a" / "own.npz",
             episode_seed=np.array([44, 55]))
    experiments = tmp_path / "experiments"
    (experiments / "stage2_confirmatory" / "merge").mkdir(parents=True)
    (experiments / "stage2b_intersection" / "smoke").mkdir(parents=True)
    np.save(experiments / "stage2_confirmatory" / "merge" / "seed_schedule.npy",
            np.array([66, 77]))
    np.save(experiments / "stage2b_intersection" / "smoke" / "seed_schedule.npy",
            np.array([88]))
    found = sp.scan_historical_episode_seeds(data, experiments)
    assert set(found.tolist()) == {11, 22, 33, 66, 77}


# --- collection manifest --------------------------------------------------------


@pytest.fixture()
def manifest_env(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "scan_historical_episode_seeds",
                        lambda *a, **k: np.array([1, 2, 3], dtype=np.int64))
    config = _smoke()
    config["output_dir"] = str(tmp_path / "smoke")
    config["data_output"] = str(tmp_path / "data" / "s.npz")
    config["rejection_ledger"] = str(tmp_path / "data" / "l.jsonl")
    config["collection_receipt"] = str(tmp_path / "data" / "r.json")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    return config, cfg_path, tmp_path / "smoke"


def test_prepare_collection_create_once(manifest_env, tmp_path):
    config, cfg_path, out_dir = manifest_env
    manifest = sp.prepare_collection(config, out_dir, config_path=cfg_path,
                                     stage2b_root=tmp_path / "no_roles")
    assert manifest["zero_overlap"]["disjoint"] is True
    assert (out_dir / "seed_schedule.npy").is_file()
    with pytest.raises(FileExistsError):
        sp.prepare_collection(config, out_dir, config_path=cfg_path,
                              stage2b_root=tmp_path / "no_roles")


def test_verify_manifest_detects_schedule_drift(manifest_env, tmp_path):
    config, cfg_path, out_dir = manifest_env
    sp.prepare_collection(config, out_dir, config_path=cfg_path,
                          stage2b_root=tmp_path / "no_roles")
    sp.verify_collection_manifest(out_dir, config)
    schedule = np.load(out_dir / "seed_schedule.npy")
    schedule = schedule.copy()
    schedule[0] = schedule[-1]
    np.save(out_dir / "seed_schedule.npy", schedule)
    with pytest.raises(ValueError, match="drift"):
        sp.verify_collection_manifest(out_dir, config)


def test_verify_manifest_detects_config_change(manifest_env, tmp_path):
    config, cfg_path, out_dir = manifest_env
    sp.prepare_collection(config, out_dir, config_path=cfg_path,
                          stage2b_root=tmp_path / "no_roles")
    changed = dict(config)
    changed["determinism_checks"] = 3
    with pytest.raises(ValueError, match="config"):
        sp.verify_collection_manifest(out_dir, changed)


def test_prepare_collection_detects_other_role_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "scan_historical_episode_seeds",
                        lambda *a, **k: np.array([1, 2, 3], dtype=np.int64))
    root = tmp_path / "stage2b_intersection"
    smoke_cfg = _smoke()
    smoke_cfg["output_dir"] = str(root / "smoke")
    smoke_cfg["data_output"] = str(tmp_path / "d1.npz")
    smoke_cfg["rejection_ledger"] = str(tmp_path / "l1.jsonl")
    smoke_cfg["collection_receipt"] = str(tmp_path / "r1.json")
    sp.prepare_collection(smoke_cfg, root / "smoke", stage2b_root=root)

    # A second role whose schedule collides with smoke's must fail wholesale.
    pilot_cfg = sp.resolve_pilot_profile_config(_load(PILOT_CONFIG), "A")
    pilot_cfg["output_dir"] = str(root / "pilot_a")
    colliding = np.load(root / "smoke" / "seed_schedule.npy")
    monkeypatch.setattr(sp, "make_seed_schedule", lambda seed, size: colliding)
    with pytest.raises(ValueError, match="pilot_a|role|overlap"):
        sp.prepare_collection(pilot_cfg, root / "pilot_a", stage2b_root=root)


def test_manifest_binding_seals_and_verifies(manifest_env, tmp_path):
    config, cfg_path, out_dir = manifest_env
    binding = tmp_path / "lookup.json"
    binding.write_text(json.dumps({"a": 1}), encoding="utf-8")
    config["movement_lookup_json"] = str(binding)
    sp.prepare_collection(config, out_dir, config_path=cfg_path,
                          extra_bindings={"movement_lookup": str(binding)},
                          stage2b_root=tmp_path / "no_roles")
    sp.verify_collection_manifest(out_dir, config)
    binding.write_text(json.dumps({"a": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="binding"):
        sp.verify_collection_manifest(out_dir, config)


# --- mechanical profile selection (never reads V_evt) ---------------------------


def _qualifying_metrics(weakest=120):
    return {
        "accepted_pairs": 600,
        "per_stratum_min": 100,
        "prefix_identity_qa_pass": True,
        "support_rate": 0.55,
        "positive_episodes": 330,
        "median_paired_valid_non_ego": 4.0,
        "bin_episode_counts": {"positive_low": weakest + 10,
                               "positive_mid": weakest + 5,
                               "positive_high": weakest},
    }


def test_select_profile_picks_max_weakest_bin():
    metrics_a = _qualifying_metrics(weakest=110)
    metrics_b = _qualifying_metrics(weakest=150)
    decision = sp.select_profile({"A": metrics_a, "B": metrics_b})
    assert decision["status"] == "PROFILE_SELECTED"
    assert decision["selected_profile"] == "B"
    assert decision["initial_vehicle_count"] == 15


def test_select_profile_tie_breaks_to_a():
    decision = sp.select_profile({"A": _qualifying_metrics(), "B": _qualifying_metrics()})
    assert decision["selected_profile"] == "A"


def test_select_profile_rejects_unqualified():
    metrics_a = _qualifying_metrics()
    metrics_a["support_rate"] = 0.10  # below the 0.40 floor
    decision = sp.select_profile({"A": metrics_a, "B": _qualifying_metrics()})
    assert decision["selected_profile"] == "B"


def test_select_profile_none_qualified_is_inconclusive():
    metrics_a = _qualifying_metrics()
    metrics_a["positive_episodes"] = 100
    decision = sp.select_profile({"A": metrics_a, "B": None})
    assert decision["status"] == "INCONCLUSIVE_INTERSECTION_RESPONSE_SUPPORT"
    assert decision["selected_profile"] is None


def test_profile_selection_rule_needs_no_value_quantities():
    """The mechanical rule consumes only support metrics; a metrics dict that
    provably contains no V_evt / model-error / Gate fields is sufficient."""
    metrics = _qualifying_metrics()
    forbidden = {"v_evt", "c_presence", "c_high_low", "c_high_mid", "l_suffix",
                 "l_shuffled", "l_no", "model_error", "gate_a"}
    assert not (forbidden & set(metrics))
    decision = sp.select_profile({"A": metrics, "B": None})
    assert decision["selected_profile"] == "A"


def test_v04_profile_b_qualifies_without_redundant_300_episode_floor():
    """v0.4 fixes B from the completed v0.3 pilot.  With 600 pilot
    episodes, the old absolute floor of 300 silently replaced the registered
    40% support floor by 50%; it must not participate in the recovery rule."""
    metrics = _qualifying_metrics(weakest=173)
    metrics["support_rate"] = 0.43
    metrics["positive_episodes"] = 258
    metrics["bin_episode_counts"] = {
        "positive_low": 250, "positive_mid": 240, "positive_high": 173,
    }
    decision = sp.build_v04_profile_b_recovery(metrics)
    assert decision["status"] == "V04_PROFILE_B_FROZEN"
    assert decision["selected_profile"] == "B"
    assert decision["initial_vehicle_count"] == 15
    assert decision["metrics"]["positive_episodes"] == 258
    assert "minimum_positive_episodes" not in decision["qualification_rule"]


def test_v04_profile_b_still_requires_registered_support_and_bin_floors():
    metrics = _qualifying_metrics(weakest=173)
    metrics["support_rate"] = 0.39
    metrics["positive_episodes"] = 234
    with pytest.raises(ValueError, match="support requirements"):
        sp.build_v04_profile_b_recovery(metrics)

    metrics = _qualifying_metrics(weakest=99)
    metrics["support_rate"] = 0.43
    metrics["positive_episodes"] = 258
    with pytest.raises(ValueError, match="support requirements"):
        sp.build_v04_profile_b_recovery(metrics)


def test_v04_development_config_uses_separate_plan_and_collection_seed():
    config = json.loads(DEV_CONFIG.read_text(encoding="utf-8"))
    config.update({
        "plan_id": sp.V04_PLAN_ID,
        "collection_seed": 2026091609,
        "initial_vehicle_count": 15,
        "experiment_name": "stage2b_intersection_development_v04",
        "output_dir": "experiments/stage2b_intersection/development_v04",
        "data_output": ("data/object_centric_stage/"
                        "stage2b_intersection_development_v04/"
                        "object_twinworld_stage2b_intersection_development_v04.npz"),
        "rejection_ledger": ("data/object_centric_stage/"
                             "stage2b_intersection_development_v04/"
                             "rejection_ledger.jsonl"),
        "collection_receipt": ("data/object_centric_stage/"
                               "stage2b_intersection_development_v04/"
                               "collection_receipt.json"),
        "profile_selection_json": (
            "experiments/stage2b_intersection/profile_selection_v04.json"),
    })
    sp.validate_stage2b_config(config)


def test_write_profile_selection_create_once(tmp_path):
    path = tmp_path / "profile_selection.json"
    record = sp.write_profile_selection(
        path, sp.select_profile({"A": _qualifying_metrics(), "B": None}))
    assert record["selection_sha256"]
    with pytest.raises(FileExistsError):
        sp.write_profile_selection(
            path, sp.select_profile({"A": _qualifying_metrics(), "B": None}))


# --- development split assignment -------------------------------------------------


def test_development_split_label_mechanical():
    assert sd.development_split_label(0) == 0
    assert sd.development_split_label(799) == 0
    assert sd.development_split_label(800) == 1
    assert sd.development_split_label(999) == 1
    with pytest.raises(ValueError):
        sd.development_split_label(1000)


# --- movement signature + conflict lookup ------------------------------------------


def test_movement_signature_from_route():
    route = [["o0", "ir0", "0"], ["ir0", "il2", "0"], ["il2", "o2", "0"]]
    assert sd.movement_signature_from_route(route) == (0, 2)
    # Routes recorded after reset's simulation steps may have dropped the
    # approach segment; the incoming arm is then read from the ir node.
    inner_start = [["ir3", "il2", "None"], ["il2", "o2", "None"]]
    assert sd.movement_signature_from_route(inner_start) == (3, 2)
    assert sd.movement_signature_from_route([]) is None
    assert sd.movement_signature_from_route(None) is None
    assert sd.movement_signature_from_route([["x0", "y1", "0"]]) is None
    same_arm = [["o1", "ir1", "0"], ["ir1", "il1", "0"]]
    assert sd.movement_signature_from_route(same_arm) is None


def _rot(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def _synthetic_polylines() -> dict[tuple[str, str, int], np.ndarray]:
    """Faithful-enough copy of the intersection-v0 lane topology/geometry."""
    lanes: dict[tuple[str, str, int], np.ndarray] = {}
    for i in range(4):
        R = _rot(90 * i)
        lanes[("o" + str(i), "ir" + str(i), 0)] = np.array(
            [R @ [2.0, 111.0], R @ [2.0, 11.0]])
        lanes[("il" + str(i), "o" + str(i), 0)] = np.array(
            [R @ [11.0, 2.0], R @ [111.0, 2.0]])
        for j in range(4):
            if i == j:
                continue
            turn = (j - i) % 4
            if turn == 2:  # straight
                inner = np.array([R @ [2.0, 11.0], R @ [2.0, -11.0]])
            elif turn == 3:  # right turn: center R@[11,11], r=9, 180->270 deg
                center = R @ np.array([11.0, 11.0])
                theta = np.radians(np.linspace(90 * i + 180, 90 * i + 270, 8))
                inner = center + 9.0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
            else:  # left turn: center R@[-11,11], r=13, 0->-90 deg
                center = R @ np.array([-11.0, 11.0])
                theta = np.radians(np.linspace(90 * i + 0, 90 * i - 90, 8))
                inner = center + 13.0 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
            lanes[("ir" + str(i), "il" + str(j), 0)] = inner
    return lanes


def test_movement_lookup_crossing_and_shared_edge():
    lookup = sd.build_movement_lookup(_synthetic_polylines())
    assert len(lookup["movements"]) == 12
    # Straight-from-north vs straight-from-west cross inside the intersection.
    assert lookup["crossing"]["0->2"]["1->3"] is True
    # Movements from the same arm share the incoming directed edge: no crossing.
    assert lookup["shared_edge"]["0->2"]["0->1"] is True
    assert lookup["crossing"]["0->2"]["0->1"] is False
    assert lookup["crossing"]["0->2"]["0->2"] is False


def test_movement_lookup_hash_stable_and_sensitive():
    first = sd.build_movement_lookup(_synthetic_polylines())
    second = sd.build_movement_lookup(_synthetic_polylines())
    assert first["lookup_sha256"] == second["lookup_sha256"]
    shifted = _synthetic_polylines()
    shifted[("ir0", "il2", 0)] = shifted[("ir0", "il2", 0)] + 5.0
    third = sd.build_movement_lookup(shifted)
    assert third["lookup_sha256"] != first["lookup_sha256"]


# --- pilot support metrics (closed list) -------------------------------------------


def _synthetic_npz_fields(positive_episodes: int, episodes: int = 6):
    """Small synthetic paired dataset: N=3 (ego + 2), T=20, all present."""
    rng = np.random.default_rng(0)
    n, t = 3, 20
    history = rng.normal(size=(episodes, 10, n, 8)).astype(np.float32)
    factual = rng.normal(size=(episodes, t, n, 8)).astype(np.float32)
    counterfactual = factual.copy()
    history[..., 0] = 1.0
    factual[..., 0] = 1.0
    counterfactual[..., 0] = 1.0
    # Episodes below `positive_episodes` get a strong position response on a
    # non-ego agent; the rest have zero response.
    for e in range(positive_episodes):
        counterfactual[e, :, 1, 2:4] += (1.0 + 0.5 * e)
    return {"object_history": history, "object_factual_future": factual,
            "object_counterfactual_future": counterfactual}


def test_pilot_support_metrics_counts():
    data = _synthetic_npz_fields(positive_episodes=4, episodes=6)
    metrics = sd.pilot_support_metrics(data)
    assert metrics["accepted_pairs"] == 6
    assert metrics["positive_episodes"] == 4
    assert abs(metrics["support_rate"] - 4.0 / 6.0) < 1e-9
    assert metrics["median_paired_valid_non_ego"] == 2.0
    bins = metrics["bin_episode_counts"]
    assert bins["positive_low"] + 0 >= 1  # terciles over 4 positive episodes
    assert bins["positive_high"] >= 1


def test_pilot_support_metrics_zero_response():
    data = _synthetic_npz_fields(positive_episodes=0, episodes=3)
    metrics = sd.pilot_support_metrics(data)
    assert metrics["support_rate"] == 0.0
    assert metrics["bin_episode_counts"] == {"positive_low": 0, "positive_mid": 0,
                                             "positive_high": 0}


# --- NPZ contract -------------------------------------------------------------------


def test_validate_stage2b_source_rejects_contract_violations():
    data = _synthetic_npz_fields(positive_episodes=2, episodes=3)
    episodes = 3
    data.update({
        "factual_actions": np.zeros((episodes, 20), dtype=np.int64),
        "counterfactual_actions": np.zeros((episodes, 20), dtype=np.int64),
        "cf_action": np.ones(episodes, dtype=np.int64),
        "cf_duration": np.ones(episodes, dtype=np.int64),
        "episode_seed": np.arange(episodes, dtype=np.int64) + 1,
        "t_star": np.full(episodes, 12, dtype=np.int64),
        "split": np.full(episodes, 4, dtype=np.int8),
        "initial_identity_count": np.full(episodes, 3, dtype=np.int64),
        "max_road_vehicles": np.full(episodes, 3, dtype=np.int64),
        "object_capacity": np.asarray(20, dtype=np.int64),
        "object_features_json": np.asarray(json.dumps(
            ["presence", "is_ego", "x", "y", "vx", "vy",
             "cos_heading", "sin_heading"]), dtype=object),
        "vehicle_metadata_json": np.asarray(["[]"] * episodes, dtype=object),
    })
    # Pad the object tensors to the capacity-20 width.
    for key in ("object_history", "object_factual_future", "object_counterfactual_future"):
        value = data[key]
        padded = np.zeros((episodes, value.shape[1], 20, 8), dtype=np.float32)
        padded[:, :, :3, :] = value
        data[key] = padded
    summary = sd.validate_stage2b_source(
        data, expected_split_labels={4}, expected_episodes=3)
    assert summary["episodes"] == 3

    bad = dict(data)
    bad["split"] = np.full(episodes, 2, dtype=np.int8)
    with pytest.raises(ValueError, match="split"):
        sd.validate_stage2b_source(bad, expected_split_labels={4}, expected_episodes=3)

    bad = dict(data)
    bad["cf_action"] = data["factual_actions"][:, 0].copy()
    with pytest.raises(ValueError, match="differ"):
        sd.validate_stage2b_source(bad, expected_split_labels={4}, expected_episodes=3)

    bad = dict(data)
    bad["max_road_vehicles"] = np.full(episodes, 21, dtype=np.int64)
    with pytest.raises(ValueError, match="capacity"):
        sd.validate_stage2b_source(bad, expected_split_labels={4}, expected_episodes=3)
