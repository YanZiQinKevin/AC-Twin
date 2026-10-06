from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import run_stage4a as driver  # noqa: E402
from ac_twin import stage4_protocol as s4p  # noqa: E402
from ac_twin import stage4_risk as s4r  # noqa: E402
from ac_twin.stage3r_models import CEWRouter  # noqa: E402


# --- e2e router device-placement regression -------------------------------------
#
# infer-dev originally crashed on pair0: `_load_e2e_router` returns the model on
# CPU (map_location="cpu") and the router was never moved to the configured device
# while its input `features` was moved to CUDA -> RuntimeError (mat1 cuda vs cpu).
# The fix is the single `.to(torch.device(device))` on the router. These tests lock
# both the device-match invariant and the presence of that fix in cmd_infer_dev.


def _router_device() -> "object":
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_e2e_router_params_and_inputs_share_device():
    import torch

    device = _router_device()
    router = CEWRouter(70, 64)  # input_dim=70 (frozen router contract), hidden=64
    features = torch.zeros(4, 70)
    # The original bug: router on CPU, features moved to a (CUDA) device -> mismatch.
    if device.type == "cuda":
        with pytest.raises(RuntimeError):
            router(features.to(device))
    # The fixed pattern: move the router to the SAME device as its input.
    router = router.to(device)
    moved_features = features.to(device)
    out = router(moved_features)
    assert next(router.parameters()).device == moved_features.device
    assert out.device == moved_features.device
    assert out.device.type == device.type


def test_e2e_router_real_checkpoint_device_match():
    """Small-scale infer smoke on the configured device with the real frozen router."""
    import torch

    device = _router_device()
    stage3r_dir = ROOT / "experiments" / "stage3_routing" / "stage3r"
    router = driver._load_e2e_router(stage3r_dir, 0, s4p.EXPERT_ROUTER_SEEDS[0]).to(device)
    features = torch.zeros(2, 70).to(device)  # one small batch on the device
    weight = router(features)
    assert next(router.parameters()).device == features.device
    assert weight.shape == (2, 1)
    assert torch.isfinite(weight).all()


def test_infer_dev_moves_router_to_device():
    """Regression guard: cmd_infer_dev must move the e2e router onto the device."""
    import inspect

    source = inspect.getsource(driver.cmd_infer_dev)
    assert "_load_e2e_router(stage3r_dir, pair_index, seed).to(torch.device(device))" in source


# --- w=0.60 constant mixture -------------------------------------------------------


def test_constant_mixture_is_exactly_060_full_040_history():
    full = np.random.default_rng(1).normal(size=(5, 3, 4, 2, 2))
    history = np.random.default_rng(2).normal(size=(5, 3, 4, 2, 2))
    constant = s4p.PRIMARY_CONSTANT_W * full + (1.0 - s4p.PRIMARY_CONSTANT_W) * history
    assert s4p.PRIMARY_CONSTANT_W == 0.60
    assert np.allclose(constant, 0.60 * full + 0.40 * history)


# --- PRIMARY_PREDICTOR selection (risk-train only) ---------------------------------


def test_primary_predictor_prefers_lower_risk_train_ade():
    constant = np.full((5, 10), 1.0)   # mean 1.0
    e2e = np.full((5, 10), 2.0)        # mean 2.0
    out = driver._select_primary_predictor(constant, e2e)
    assert out["primary_predictor"] == "BEST_CONSTANT_W"
    assert out["constant_w"] == 0.60


def test_primary_predictor_switches_when_e2e_better():
    constant = np.full((5, 10), 3.0)
    e2e = np.full((5, 10), 1.0)
    out = driver._select_primary_predictor(constant, e2e)
    assert out["primary_predictor"] == "END_TO_END_MIXTURE"


def test_primary_predictor_tie_goes_to_best_constant_w():
    constant = np.full((5, 10), 1.5)
    e2e = np.full((5, 10), 1.5)
    out = driver._select_primary_predictor(constant, e2e)
    assert out["primary_predictor"] == "BEST_CONSTANT_W"


def test_primary_predictor_uses_only_passed_train_slice():
    # Whatever the caller passes IS the risk-train slice; the function must not
    # reach outside it. A train slice favouring constant must select constant even
    # if a hypothetical held-out slice would favour e2e.
    constant_train = np.full((5, 300), 1.0)
    e2e_train = np.full((5, 300), 2.0)
    assert driver._select_primary_predictor(
        constant_train, e2e_train)["primary_predictor"] == "BEST_CONSTANT_W"


# --- nested learning-curve subsets ----------------------------------------------------


def _ten_stratum_train(per_stratum: int = 6):
    cf_action, cf_duration, episode_seed, train_idx = [], [], [], []
    seed = 1000
    index = 0
    for action in s4p.CF_ACTIONS:
        for duration in s4p.CF_DURATIONS:
            for _ in range(per_stratum):
                cf_action.append(action)
                cf_duration.append(duration)
                episode_seed.append(seed)
                train_idx.append(index)
                seed += 1
                index += 1
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(cf_action, dtype=np.int64),
        np.asarray(cf_duration, dtype=np.int64),
        np.asarray(episode_seed, dtype=np.int64),
    )


def test_nested_train_subsets_are_strictly_nested():
    train_idx, cf_action, cf_duration, episode_seed = _ten_stratum_train(per_stratum=6)
    subsets = driver._nested_train_subsets(
        train_idx, cf_action, cf_duration, episode_seed, (2, 4, 6)
    )
    assert [s.size for s in subsets] == [2 * 10, 4 * 10, 6 * 10]
    assert set(subsets[0].tolist()) <= set(subsets[1].tolist())
    assert set(subsets[1].tolist()) <= set(subsets[2].tolist())


def test_nested_train_subsets_take_seed_hash_order_per_stratum():
    train_idx, cf_action, cf_duration, episode_seed = _ten_stratum_train(per_stratum=6)
    subsets = driver._nested_train_subsets(
        train_idx, cf_action, cf_duration, episode_seed, (2, 4)
    )
    members = driver._stratum_members_seed_order(train_idx, cf_action, cf_duration, episode_seed)
    # Each 2-per-stratum subset is exactly the first-2 (seed-hash) of the 4-subset,
    # per stratum.
    small = subsets[0]
    for key, cell in members.items():
        first_two = set(cell[:2])
        in_small = first_two & set(small.tolist())
        assert in_small == first_two


def test_nested_train_subsets_reject_oversized_request():
    train_idx, cf_action, cf_duration, episode_seed = _ten_stratum_train(per_stratum=6)
    with pytest.raises(ValueError, match="cannot take"):
        driver._nested_train_subsets(train_idx, cf_action, cf_duration, episode_seed, (7,))


# --- single-use development-screen lock ----------------------------------------------


def test_acquire_screen_lock_is_single_use(tmp_path: Path):
    lock = tmp_path / "development_screen_used.lock"
    driver._acquire_screen_lock(lock)
    assert lock.is_file()
    with pytest.raises(RuntimeError, match="already consumed"):
        driver._acquire_screen_lock(lock)


# --- engineering smoke -----------------------------------------------------------------


def _smoke_config(tmp_path: Path) -> dict:
    return {"run_dir": str(tmp_path / "run")}


def test_cmd_smoke_writes_only_to_smoke_dir(tmp_path: Path):
    report = driver.cmd_smoke(_smoke_config(tmp_path))
    assert report["status"] == driver.SMOKE_STATUS == "ENGINEERING_SMOKE_ONLY"
    report_path = tmp_path / "run" / "smoke" / "smoke_report.json"
    assert report_path.is_file()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk["status"] == "ENGINEERING_SMOKE_ONLY"
    # Nothing outside smoke/ was created in the run dir.
    run_dir = tmp_path / "run"
    assert {p.name for p in run_dir.iterdir()} == {"smoke"}


def test_cmd_smoke_whitelist_rejects_truth_input(tmp_path: Path):
    report = driver.cmd_smoke(_smoke_config(tmp_path))
    assert report["whitelist_rejects_truth_input"] is True
    assert report["risk_target_finite"] is True


def test_cmd_smoke_is_deterministic(tmp_path: Path):
    first = driver.cmd_smoke(_smoke_config(tmp_path / "a"))
    second = driver.cmd_smoke(_smoke_config(tmp_path / "b"))
    assert first["feature_names"] == second["feature_names"]
    assert first["feature_count"] == second["feature_count"]
    assert first["feature_count"] == len(s4r.ALL_FEATURES)
