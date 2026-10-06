"""Stage 4C-R2 fixed Direct training + train-support-gate tests (plan sections 4.2, 7).

Blocking invariants proved here (plan section 14 / task-3 brief):
- the one and only model is SAME_INFORMATION_DIRECT, width 43, dropout 0.10,
  lambda_cls 0.25, with EXACTLY 15,740 trainable parameters (recounted per layer);
- the internal-monitor split is a fixed 10% of train, a pure function of the group seeds
  (never of a model seed), and stays disjoint from the main body;
- the feature scaler and the class weight are fit on the train MAIN BODY ONLY — mutating
  monitor rows (or any non-train rows) cannot change them;
- the ensemble is a plain arithmetic mean over EXACTLY the five frozen seeds; all five
  checkpoints are kept; no median, no seed-picking, no validation weighting;
- the train support hard gate (plan 7.1) enforces all seven conditions independently.
"""
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

from ac_twin import stage4cp_features as s4cpf  # noqa: E402
from ac_twin import stage4cp_models as s4cm  # noqa: E402
from ac_twin import stage4cr2_protocol as s4cr2  # noqa: E402
from ac_twin import stage4cr2_training as s4cr2t  # noqa: E402

CONFIG_PATH = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"


def _config(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["run_dir"] = str(tmp_path / "run")
    config["data_root"] = str(
        tmp_path / "data" / "object_centric_stage" / s4cr2.PURPOSE_DEV)
    config["device"] = "cpu"
    return config


def _synthetic_train_bundle(groups=160, seed=0):
    """Random but well-posed train bundle: both label classes present, all continuous
    columns varying. Row identity is encoded in the last (categorical) token column,
    which the scaler must never standardise."""
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(groups, s4cr2.N_COMPETITORS, s4cr2.TOKEN_DIM))
    tokens[:, :, 20:] = np.round(rng.normal(size=(groups, s4cr2.N_COMPETITORS, 14)))
    tokens[:, 0, -1] = np.arange(groups, dtype=np.float64)  # row-id probe (unscaled)
    global_features = rng.normal(size=(groups, s4cr2.GLOBAL_DIM))
    regret = np.concatenate([
        np.zeros(groups // 2),
        rng.gamma(shape=2.0, scale=0.02, size=groups - groups // 2),
    ])
    y_reg = np.log1p(regret / s4cr2.HIGH_REGRET_THRESHOLD)
    y_high = (regret > s4cr2.HIGH_REGRET_THRESHOLD).astype(np.float64)
    return {
        "tokens": tokens,
        "global_features": global_features,
        "y_reg": y_reg,
        "y_high": y_high,
        "group_seed": np.arange(10_000, 10_000 + groups, dtype=np.int64),
        "interaction_min_distance": rng.uniform(5.0, 60.0, size=groups),
    }


def _fake_checkpoint(seed, n_main, scores=None):
    """Minimal Direct checkpoint payload (real architecture state, fixed test scores)."""
    model = s4cm.build_model(
        s4cr2.ARCH_DIRECT, hidden_dim=s4cr2.DIRECT_HIDDEN_WIDTH,
        dropout=s4cr2.DROPOUT, direct_width=s4cr2.DIRECT_HIDDEN_WIDTH)
    return {
        "arch": s4cr2.ARCH_DIRECT,
        "hidden_dim": s4cr2.DIRECT_HIDDEN_WIDTH,
        "direct_width": s4cr2.DIRECT_HIDDEN_WIDTH,
        "lambda_cls": s4cr2.LAMBDA_CLS,
        "seed": int(seed),
        "pos_weight": 1.0,
        "dropout": s4cr2.DROPOUT,
        "token_dim": s4cr2.TOKEN_DIM,
        "global_dim": s4cr2.GLOBAL_DIM,
        "n_competitors": s4cr2.N_COMPETITORS,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "train_loss": 0.0,
        "best_internal_monitor_loss": 0.0,
        "n_train_groups": int(n_main),
        "test_risk": scores,  # test hook consumed by the injected predict_fn
    }


def _capturing_fit(captured, score_table):
    """fit_one_fn test double: records exactly what each seed is trained on."""
    def fake_fit(seed, main, monitor, *, pos_weight, config, device):
        captured.setdefault("pos_weight", []).append(float(pos_weight))
        captured.setdefault("main_rows", []).append(main["x_tok"][:, 0, -1].copy())
        captured.setdefault("monitor_rows", []).append(monitor["x_tok"][:, 0, -1].copy())
        captured.setdefault("seeds", []).append(int(seed))
        return _fake_checkpoint(seed, main["x_tok"].shape[0],
                                scores=score_table[int(seed)])
    return fake_fit


def _fixed_predict(checkpoint, tokens, global_features, token_mask, *, device):
    n = tokens.shape[0]
    risk = np.asarray(checkpoint["test_risk"], dtype=np.float64)
    assert risk.shape == (n,)
    return {"risk_score": risk, "mu_only": risk, "class_only": risk,
            "raw_mu": risk, "logit_high": risk}


# --- fixed architecture / parameter count ---------------------------------------------------


def test_direct_param_count_exactly_15740_recounted_per_layer():
    in_dim = s4cr2.N_COMPETITORS * s4cr2.TOKEN_DIM + s4cr2.GLOBAL_DIM  # 315
    assert in_dim == 315
    w = s4cr2.DIRECT_HIDDEN_WIDTH  # 43
    per_layer = {
        "linear1": in_dim * w + w,      # 13,588
        "layernorm1": 2 * w,            # 86
        "linear2": w * w + w,           # 1,892
        "layernorm2": 2 * w,            # 86
        "linear3": w * 2 + 2,           # 88
    }
    assert per_layer == {"linear1": 13588, "layernorm1": 86, "linear2": 1892,
                         "layernorm2": 86, "linear3": 88}
    assert sum(per_layer.values()) == 15740
    model = s4cm.build_model(
        s4cr2.ARCH_DIRECT, hidden_dim=w, dropout=s4cr2.DROPOUT, direct_width=w)
    assert s4cm.count_parameters(model) == s4cr2.EXPECTED_TRAINABLE_PARAMETERS == 15740
    # dropout carries no parameters; the frozen contract is exact, not +/-5%.
    assert s4cr2t.count_direct_parameters() == 15740


def test_fit_rejects_any_frozen_hyperparameter_drift(tmp_path):
    bundle = _synthetic_train_bundle()
    for key, bad in (("direct_hidden_width", 44), ("lambda_cls", 0.3),
                     ("dropout", 0.2), ("learning_rate", 0.01),
                     ("train_batch_size", 64), ("max_epochs", 50),
                     ("early_stop_patience", 6), ("internal_monitor_frac", 0.2)):
        config = _config(tmp_path / f"drift_{key}")
        config[key] = bad
        with pytest.raises(ValueError):
            s4cr2t.fit_five_direct_models(bundle, config, device="cpu")
    config = _config(tmp_path / "drift_seeds")
    config["model_seeds"] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        s4cr2t.fit_five_direct_models(bundle, config, device="cpu")


# --- internal-monitor split ------------------------------------------------------------------


def test_internal_monitor_split_fixed_by_group_seed_only():
    bundle = _synthetic_train_bundle(groups=200)
    seeds = bundle["group_seed"]
    main1, mon1 = s4cr2t.internal_monitor_split(seeds)
    main2, mon2 = s4cr2t.internal_monitor_split(seeds)
    np.testing.assert_array_equal(main1, main2)
    np.testing.assert_array_equal(mon1, mon2)
    assert mon1.size == int(round(s4cr2.INTERNAL_MONITOR_FRAC * seeds.size)) == 20
    assert np.intersect1d(main1, mon1).size == 0
    assert np.sort(np.concatenate([main1, mon1])).tolist() == list(range(seeds.size))
    # membership is a per-group-seed property: permuting the rows must not change WHICH
    # group seeds land in the monitor set (fixed across the five model seeds, since the
    # split takes no model seed).
    order = np.random.default_rng(1).permutation(seeds.size)
    main_p, mon_p = s4cr2t.internal_monitor_split(seeds[order])
    assert set(int(s) for s in seeds[mon1]) == set(int(s) for s in seeds[order][mon_p])


# --- scaler / class weight fit on the train main body ONLY ------------------------------------


def test_scaler_and_class_weight_come_from_main_body_only(tmp_path):
    bundle = _synthetic_train_bundle(groups=160)
    score_table = {int(s): np.zeros(4) for s in s4cr2.MODEL_SEEDS}
    captured = {}
    config = _config(tmp_path)
    inventory = s4cr2t.fit_five_direct_models(
        bundle, config, fit_one_fn=_capturing_fit(captured, score_table), device="cpu")

    main_idx, mon_idx = s4cr2t.internal_monitor_split(bundle["group_seed"])
    expected = s4cpf.FrozenFeatureScaler.fit(
        bundle["tokens"][main_idx], bundle["global_features"][main_idx])
    np.testing.assert_array_equal(
        inventory.scaler.token_cont_mean, expected.token_cont_mean)
    np.testing.assert_array_equal(
        inventory.scaler.token_cont_std, expected.token_cont_std)
    np.testing.assert_array_equal(inventory.scaler.global_mean, expected.global_mean)
    np.testing.assert_array_equal(inventory.scaler.global_std, expected.global_std)

    n_main = main_idx.size
    n_pos = float(bundle["y_high"][main_idx].sum())
    expected_pw = (n_main - n_pos) / max(n_pos, 1.0)
    assert captured["pos_weight"] == [pytest.approx(expected_pw)] * 5
    assert inventory.pos_weight == pytest.approx(expected_pw)

    # the monitor rows reached the trainer untouched by training and disjoint from main.
    for rows_main, rows_mon in zip(captured["main_rows"], captured["monitor_rows"]):
        assert set(int(v) for v in rows_main) == set(int(v) for v in main_idx)
        assert set(int(v) for v in rows_mon) == set(int(v) for v in mon_idx)

    # mutating the monitor rows (or any non-main rows) cannot move the scaler/weight.
    poisoned = {k: np.array(v, copy=True) for k, v in bundle.items()}
    poisoned["tokens"][mon_idx] = 1e6
    poisoned["global_features"][mon_idx] = -1e6
    poisoned["y_high"][mon_idx] = 1.0 - poisoned["y_high"][mon_idx]
    captured2 = {}
    config2 = _config(tmp_path / "second")
    inventory2 = s4cr2t.fit_five_direct_models(
        poisoned, config2, fit_one_fn=_capturing_fit(captured2, score_table),
        device="cpu")
    np.testing.assert_array_equal(
        inventory2.scaler.token_cont_mean, expected.token_cont_mean)
    np.testing.assert_array_equal(inventory2.scaler.global_mean, expected.global_mean)
    assert captured2["pos_weight"] == captured["pos_weight"]


# --- five seeds kept; ensemble is the plain arithmetic mean ------------------------------------


def test_ensemble_is_arithmetic_mean_of_exactly_five_seeds(tmp_path):
    groups_eval = 7
    rng = np.random.default_rng(3)
    # per-seed score table crafted so that mean != median.
    score_table = {}
    for i, seed in enumerate(s4cr2.MODEL_SEEDS):
        score_table[int(seed)] = rng.normal(loc=i * 2.0, scale=0.1, size=groups_eval)
    stacked = np.stack([score_table[int(s)] for s in s4cr2.MODEL_SEEDS])
    assert not np.allclose(stacked.mean(axis=0), np.median(stacked, axis=0))

    bundle = _synthetic_train_bundle(groups=64, seed=5)
    config = _config(tmp_path)
    captured = {}
    inventory = s4cr2t.fit_five_direct_models(
        bundle, config, fit_one_fn=_capturing_fit(captured, score_table), device="cpu")

    assert inventory.seeds == tuple(int(s) for s in s4cr2.MODEL_SEEDS)
    assert len(inventory.checkpoint_paths) == 5  # ALL five checkpoints kept
    for seed, path in inventory.checkpoint_paths.items():
        assert Path(path).is_file(), seed

    eval_bundle = {
        "tokens": rng.normal(size=(groups_eval, s4cr2.N_COMPETITORS, s4cr2.TOKEN_DIM)),
        "global_features": rng.normal(size=(groups_eval, s4cr2.GLOBAL_DIM)),
        "group_seed": np.arange(groups_eval, dtype=np.int64),
    }
    scores = s4cr2t.score_models(inventory, eval_bundle,
                                 predict_fn=_fixed_predict, device="cpu")
    assert scores.risk_by_seed.shape == (5, groups_eval)
    for i, seed in enumerate(s4cr2.MODEL_SEEDS):
        np.testing.assert_array_equal(scores.risk_by_seed[i], score_table[int(seed)])
    np.testing.assert_array_equal(  # exact arithmetic mean, not median/weighted/picked
        scores.risk_ensemble, stacked.mean(axis=0))
    assert scores.seeds == tuple(int(s) for s in s4cr2.MODEL_SEEDS)


def test_fit_is_create_once_and_never_retrains(tmp_path):
    bundle = _synthetic_train_bundle(groups=64, seed=7)
    score_table = {int(s): np.zeros(2) for s in s4cr2.MODEL_SEEDS}
    captured = {}
    config = _config(tmp_path)
    first = s4cr2t.fit_five_direct_models(
        bundle, config, fit_one_fn=_capturing_fit(captured, score_table), device="cpu")
    assert len(captured["seeds"]) == 5
    second = s4cr2t.fit_five_direct_models(
        bundle, config, fit_one_fn=_capturing_fit(captured, score_table), device="cpu")
    assert len(captured["seeds"]) == 5  # no retraining on resume
    assert second.verification == "existing_inventory_verified_no_overwrite"
    assert second.checkpoint_sha256 == first.checkpoint_sha256


# --- real fixed training loop (small bundle, CPU) ---------------------------------------------


def test_real_fit_trains_all_five_and_scores_finitely(tmp_path):
    bundle = _synthetic_train_bundle(groups=160, seed=11)
    config = _config(tmp_path)
    inventory = s4cr2t.fit_five_direct_models(bundle, config, device="cpu")
    assert inventory.param_count == 15740
    assert inventory.seeds == tuple(int(s) for s in s4cr2.MODEL_SEEDS)
    for seed in s4cr2.MODEL_SEEDS:
        assert np.isfinite(inventory.train_losses[int(seed)])
        assert np.isfinite(inventory.monitor_losses[int(seed)])
    scores = s4cr2t.score_models(inventory, bundle, device="cpu")
    assert np.isfinite(scores.risk_ensemble).all()
    assert scores.risk_by_seed.shape == (5, 160)
    np.testing.assert_array_equal(
        scores.risk_ensemble, scores.risk_by_seed.mean(axis=0))
    # frozen interaction threshold = train median of preintervention_min_distance.
    assert inventory.interaction_threshold == pytest.approx(
        float(np.median(bundle["interaction_min_distance"])))


# --- train support hard gate (plan section 7.1) ------------------------------------------------


def _support_fixture(groups=5000, seed=21):
    """A PASSING train-support fixture built on the real frozen schedule segment."""
    rng = np.random.default_rng(seed)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    schedule = s4cr2.build_seed_schedule(config)
    group_seed = schedule[:groups]
    regret = np.zeros(groups)
    n_high = 800
    n_nonzero = 1600
    regret[:n_nonzero] = rng.uniform(1e-4, 5e-3, size=n_nonzero)
    regret[:n_high] = rng.uniform(0.02, 0.2, size=n_high)
    rng.shuffle(regret)
    oracle = rng.choice(10, size=groups, p=[0.12, 0.12, 0.12, 0.12, 0.12,
                                          0.08, 0.08, 0.08, 0.08, 0.08])
    tokens = rng.normal(size=(groups, s4cr2.N_COMPETITORS, s4cr2.TOKEN_DIM))
    tokens[:, :, 20:] = np.round(rng.normal(size=(groups, s4cr2.N_COMPETITORS, 14)))
    global_features = rng.normal(size=(groups, s4cr2.GLOBAL_DIM))
    return {
        "groups": groups,
        "n_actions": 10,
        "complete_groups": groups,
        "regret": regret,
        "oracle_action": oracle,
        "base_mean_regret": 0.004,
        "random_mean_regret": 0.010,
        "idle_mean_regret": 0.006,
        "tokens": tokens,
        "global_features": global_features,
        "group_seed": group_seed,
    }


def test_train_support_gate_passes_on_qualified_fixture():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    report = s4cr2t.evaluate_train_support_gate(_support_fixture(), config)
    assert report["passed"] is True
    assert all(report["conditions"].values())
    assert len(report["conditions"]) == 7
    assert report["status"] == "TRAIN_SUPPORT_OK"


@pytest.mark.parametrize("condition", [
    "all_complete_10_action",
    "high_regret_groups_ge_750",
    "nonzero_regret_groups_ge_1250",
    "at_least_3_oracle_actions_ge_5pct",
    "continuous_features_finite_with_variance",
    "base_beats_random_not_worse_than_idle",
    "role_seed_disjointness",
])
def test_train_support_gate_each_condition_independently_enforced(condition):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fixture = _support_fixture()
    groups = fixture["groups"]
    if condition == "all_complete_10_action":
        fixture["complete_groups"] = groups - 1
    elif condition == "high_regret_groups_ge_750":
        fixture["regret"] = np.where(fixture["regret"] > 0.01, 0.005, fixture["regret"])
        fixture["regret"][:749] = 0.02  # only 749 high-regret groups
    elif condition == "nonzero_regret_groups_ge_1250":
        fixture["regret"] = 0.0 * fixture["regret"]
        fixture["regret"][:1249] = 0.002
        fixture["regret"][:800] = 0.02  # high-regret count still satisfied
    elif condition == "at_least_3_oracle_actions_ge_5pct":
        fixture["oracle_action"] = np.zeros(groups, dtype=np.int64)  # one oracle only
    elif condition == "continuous_features_finite_with_variance":
        fixture["global_features"][:, 0] = 3.14  # zero-variance continuous column
    elif condition == "base_beats_random_not_worse_than_idle":
        fixture["base_mean_regret"] = fixture["random_mean_regret"]  # no improvement
    elif condition == "role_seed_disjointness":
        schedule = s4cr2.build_seed_schedule(config)
        bad = np.array(fixture["group_seed"], copy=True)
        bad[0] = schedule[s4cr2.VALIDATION_SEGMENT[0]]  # a validation-segment seed
        fixture["group_seed"] = bad
    report = s4cr2t.evaluate_train_support_gate(fixture, config)
    assert report["conditions"][condition] is False, report["conditions"]
    assert report["passed"] is False
    assert report["status"] == s4cr2.STATUS_TRAIN_SUPPORT_INSUFFICIENT


def test_train_support_gate_rejects_non_finite_and_nan():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fixture = _support_fixture()
    fixture["tokens"][0, 0, 0] = np.nan
    report = s4cr2t.evaluate_train_support_gate(fixture, config)
    assert report["conditions"]["continuous_features_finite_with_variance"] is False
    assert report["status"] == s4cr2.STATUS_TRAIN_SUPPORT_INSUFFICIENT


def test_train_support_gate_constant_contract_exemption_is_narrow():
    """visible_vehicle_count is constant BY CONTRACT (capacity-5 observation, ego+4 slots,
    all known in every accepted group — verified constant in the R2 train 5,000 and the
    historical Stage 4C selection 500); the plan-7.1 exemption covers ONLY it."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    idx = list(s4cpf.GLOBAL_NAMES).index("visible_vehicle_count")
    fixture = _support_fixture()
    fixture["global_features"][:, idx] = 4.0  # contractual constant: exempt
    report = s4cr2t.evaluate_train_support_gate(fixture, config)
    assert report["conditions"]["continuous_features_finite_with_variance"] is True
    assert report["passed"] is True

    fixture2 = _support_fixture(seed=22)
    fixture2["global_features"][:, idx] = 4.0
    fixture2["tokens"][:, :, 0] = 1.0  # delta_j_mean constant: NOT a constant contract
    report2 = s4cr2t.evaluate_train_support_gate(fixture2, config)
    assert report2["conditions"]["continuous_features_finite_with_variance"] is False
    assert report2["values"]["zero_variance_continuous_columns"] == ["delta_j_mean"]
    assert report2["status"] == s4cr2.STATUS_TRAIN_SUPPORT_INSUFFICIENT
