"""M1 supervised-baseline ablation: key-invariant tests (plan section 10, P1).

Synthetic-data tests for the frozen contract; no real R2 per-group product is opened
except the two read-only config/report JSONs needed for interface parity. Each
distinct critical invariant is proven once (risk-tiered principle).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from ac_twin import m1_ablation_features as m1f
from ac_twin import m1_ablation_protocol as m1p
from ac_twin import m1_ablation_statistics as m1st
from ac_twin import m1_ablation_training as m1t
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_protocol as s4cp
from ac_twin import stage4cr2_training as s4cr2t
from ac_twin import stage5_features as s5feat

ROOT = m1p.ROOT


def _config() -> dict:
    return m1p.load_config(ROOT / m1p.OUTPUT_DIR / "config.json")


# --- synthetic R2-shaped sources ----------------------------------------------------------


def _synthetic_sources(tmp_path: Path, groups: int = 8, seeds: int = 5):
    """Minimal train-role shaped data/predictions/scores triple (legal side only)."""
    rng = np.random.default_rng(7)
    capacity, hist, horizon, nfeat, actions = 5, 10, 20, 8, 10
    group_seed = np.arange(1001, 1001 + groups, dtype=np.int64)
    history = rng.normal(size=(groups, hist, capacity, nfeat)).astype(np.float32)
    history[..., 0] = np.abs(history[..., 0]) + 1.0  # presence channel positive
    factual = rng.normal(size=(groups, horizon, capacity, nfeat)).astype(np.float32)
    factual[..., 0] = np.abs(factual[..., 0]) + 1.0
    cf = rng.normal(size=(groups, actions, horizon, capacity, nfeat)).astype(np.float32)
    cf[..., 0] = np.abs(cf[..., 0]) + 1.0
    data = tmp_path / "data.npz"
    np.savez_compressed(
        data, object_history=history, object_factual_future=factual,
        object_counterfactual_futures=cf,
        inference_mask=np.ones((groups, horizon, capacity), dtype=bool),
        cf_actions=np.tile(np.arange(actions), (groups, 1)).astype(np.int64),
        cf_durations=np.tile(np.repeat([1, 3], 5), (groups, 1)).astype(np.int64),
        t_star=np.full(groups, 10, dtype=np.int64),
        episode_seed=group_seed,
        factual_actions=np.zeros((groups, horizon), dtype=np.int64),
        n_actions=np.int64(actions), accepted_groups=np.int64(groups),
        n_mask_cells=np.zeros(groups, dtype=np.int64),
        object_capacity=np.int64(capacity),
    )
    e2e = rng.normal(scale=0.1, size=(seeds, groups, actions, horizon, capacity, 2)) \
        .astype(np.float32)
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(predictions, e2e=e2e, group_seed=group_seed)
    scores = tmp_path / "scores.npz"
    score_arrays = {name: rng.random((groups, actions)) for name in s4cp.RISK_CLASSES}
    np.savez_compressed(scores, group_seed=group_seed, **score_arrays)
    return {"train_data": data, "train_predictions": predictions,
            "train_scores": scores}, group_seed


# --- 1. whitelist + output isolation --------------------------------------------------------


def test_whitelist_rejects_screen_confirm_paths():
    cfg = _config()
    from ac_twin import stage5_protocol as s5prot
    for bad in ("experiments/stage4_reliability/stage4c_r2/development/screen_scores.npz",
                "experiments/stage4_reliability/stage4cr2_confirm/confirm_scores.npz",
                "experiments/stage4_reliability/stage4b/reports/x.json"):
        with pytest.raises(ValueError):
            s5prot.assert_allowed_stage5_data(ROOT / bad)
    out = m1p.assert_output_isolation(cfg)
    assert m1p._normalised(out).endswith("m1_supervised_ablation_v02")
    evil = dict(cfg)
    evil["output_dir"] = "experiments/stage4_reliability/m1_evil"
    with pytest.raises(ValueError):
        m1p.validate_config(evil)
    with pytest.raises(ValueError):
        m1p.assert_output_isolation({**cfg, "output_dir": "paper/m1_evil"})


# --- 2. score side is truth-free --------------------------------------------------------------


def test_legal_features_ignore_truth_array(tmp_path):
    sources, _ = _synthetic_sources(tmp_path)
    bundle_a, _ = s5feat.build_feature_context(sources, "train")
    # corrupt the truth array in a copy: the legal build must be bit-identical
    corrupted = tmp_path / "data_corrupted.npz"
    with np.load(sources["train_data"], allow_pickle=True) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["object_counterfactual_futures"] = \
        arrays["object_counterfactual_futures"] * 1000.0 + 7.0
    np.savez_compressed(corrupted, **arrays)
    bundle_b, _ = s5feat.build_feature_context(sources, "train", data_path=corrupted)
    assert np.array_equal(np.asarray(bundle_a.tokens), np.asarray(bundle_b.tokens))
    assert np.array_equal(np.asarray(bundle_a.global_features),
                          np.asarray(bundle_b.global_features))


# --- 3. arm masks --------------------------------------------------------------------------------


def test_arm_masks_exact():
    rng = np.random.default_rng(3)
    groups = 16
    tokens = rng.normal(size=(groups, 9, 34))
    glob = rng.normal(size=(groups, 9))
    b3 = rng.normal(size=(groups, 3))
    scaler = s4cpf.FrozenFeatureScaler.fit(tokens, glob)
    b3_scaler = m1f.fit_baseline3_scaler(b3)

    assert m1f.RISK_TOKEN_COLS.tolist() == list(range(5, 20))
    assert m1f.risk_flat_columns().size == 135

    no_risk = m1f.arm_flat_inputs("NO_RISK_MATCHED", tokens, glob, b3,
                                  full_scaler=scaler, baseline3_scaler=b3_scaler)
    assert no_risk.shape == (groups, 315)
    assert np.all(no_risk[:, m1f.risk_flat_columns()] == 0.0)
    kept = np.setdiff1d(np.arange(315), m1f.risk_flat_columns())
    full = m1f.arm_flat_inputs("FULL_MATCHED", tokens, glob, b3,
                               full_scaler=scaler, baseline3_scaler=b3_scaler)
    assert np.array_equal(no_risk[:, kept], full[:, kept])

    # masking AFTER standardisation: mutating the pre-mask risk values cannot change
    # the NO_RISK model input at all
    tokens_mut = tokens.copy()
    tokens_mut[:, :, m1f.RISK_TOKEN_COLS] = 1e6
    no_risk_mut = m1f.arm_flat_inputs("NO_RISK_MATCHED", tokens_mut, glob, b3,
                                      full_scaler=scaler, baseline3_scaler=b3_scaler)
    assert np.array_equal(no_risk, no_risk_mut)

    base = m1f.arm_flat_inputs("BASELINE3_MATCHED", tokens, glob, b3,
                               full_scaler=scaler, baseline3_scaler=b3_scaler)
    assert np.all(base[:, 3:] == 0.0)
    expected = (b3 - b3_scaler["mean"]) / b3_scaler["std"]
    assert np.allclose(base[:, :3], expected)
    assert m1f.active_input_dims() == {
        "FULL_MATCHED": 315, "NO_RISK_MATCHED": 180, "BASELINE3_MATCHED": 3}


# --- 4. main-only statistics are role-local ------------------------------------------------------


def test_main_only_scaler_split_posweight(tmp_path):
    sources, group_seed = _synthetic_sources(tmp_path, groups=40)
    built = m1f.build_legal_role(sources, "train", expected_groups=40)
    main_idx, monitor_idx = s4cr2t.internal_monitor_split(built["group_seed"])
    assert main_idx.size + monitor_idx.size == 40
    assert set(main_idx.tolist()).isdisjoint(set(monitor_idx.tolist()))
    scaler = s4cpf.FrozenFeatureScaler.fit(
        built["tokens"][main_idx], built["global_features"][main_idx])
    # mutating monitor/validation values must not move the main-fitted scaler
    tokens2 = built["tokens"].copy()
    tokens2[monitor_idx] = 1e6
    scaler2 = s4cpf.FrozenFeatureScaler.fit(
        tokens2[main_idx], built["global_features"][main_idx])
    assert np.array_equal(scaler.token_cont_mean, scaler2.token_cont_mean)
    y_high = np.zeros(40)
    y_high[main_idx[:4]] = 1.0
    n_main = main_idx.size
    n_pos = y_high[main_idx].sum()
    assert (n_main - n_pos) / max(n_pos, 1.0) == pytest.approx(8.0)
    # split is a pure function of the group seeds, identical for every model seed
    m2, mon2 = s4cr2t.internal_monitor_split(built["group_seed"])
    assert np.array_equal(main_idx, m2) and np.array_equal(monitor_idx, mon2)


# --- 5. trainer mirrors the frozen R2 loop ----------------------------------------------------------


def _r2_config() -> dict:
    path = ROOT / "experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_trainer_matches_frozen_r2_loop():
    rng = np.random.default_rng(11)
    n_main, n_mon = 96, 16
    tokens = rng.normal(size=(n_main + n_mon, 9, 34))
    glob = rng.normal(size=(n_main + n_mon, 9))
    y_reg = np.abs(rng.normal(size=n_main + n_mon))
    y_high = (y_reg > 0.5).astype(np.float64)
    config = _r2_config()
    main = {"x_tok": tokens[:n_main], "x_glob": glob[:n_main],
            "y_reg": y_reg[:n_main], "y_high": y_high[:n_main]}
    monitor = {"x_tok": tokens[n_main:], "x_glob": glob[n_main:],
               "y_reg": y_reg[n_main:], "y_high": y_high[n_main:]}
    original = s4cr2t._fit_direct_seed(8101, main, monitor, pos_weight=2.0,
                                       config=config, device="cpu")
    flat = np.concatenate([tokens.reshape(n_main + n_mon, 306), glob], axis=1)
    matched = m1t.fit_arm_seed("FULL_MATCHED", 8101, flat[:n_main], y_reg[:n_main],
                               y_high[:n_main], flat[n_main:], y_reg[n_main:],
                               y_high[n_main:], pos_weight=2.0, device="cpu")
    assert original["train_loss"] == pytest.approx(matched["train_loss"], abs=0.0)
    assert original["best_internal_monitor_loss"] == pytest.approx(
        matched["best_internal_monitor_loss"], abs=0.0)
    assert original["epochs_run"] == matched["epochs_run"]
    for key, value in original["state_dict"].items():
        assert torch.equal(value, matched["state_dict"][key])
    assert m1t.count_parameters() == 15740


def test_same_seed_identical_initialisation_across_arms():
    torch.manual_seed(8101)
    from ac_twin import stage4cp_models as s4cm
    m1 = s4cm.build_model(s4cp.ARCH_DIRECT, hidden_dim=43, dropout=0.1, direct_width=43)
    torch.manual_seed(8101)
    m2 = s4cm.build_model(s4cp.ARCH_DIRECT, hidden_dim=43, dropout=0.1, direct_width=43)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(v1, v2)


def test_score_formula_and_ensemble():
    from ac_twin import stage4cp_models as s4cm
    raw_mu = np.array([-2.0, 0.0, 3.0])
    logit = np.array([0.0, 1.0, -1.0])
    scores = s4cm.heads_to_scores(raw_mu, logit)
    mu_pos = np.log1p(np.exp(raw_mu))
    expected = (1.0 / (1.0 + np.exp(-logit))) * 0.01 * np.expm1(mu_pos)
    assert np.allclose(scores["risk_score"], expected)


# --- 6. statistics contract ---------------------------------------------------------------------------


def test_statistics_pairing_intervals_and_degenerates():
    rng = np.random.default_rng(5)
    groups = 200
    regret = np.abs(rng.normal(size=groups))
    scores = {m: rng.normal(size=groups) for m in m1p.ALL_METHODS}
    # identical scores -> every contrast is exactly zero, intervals degenerate to zero
    same = {m: scores["FULL_MATCHED"].copy() for m in m1p.ALL_METHODS}
    boot = m1st.paired_bootstrap(same, regret, n_draws=50, entropy=1)
    names = boot["names"]
    for cid, _, _ in m1p.CONTRASTS:
        for metric in (m1st.METRIC_AURC, m1st.METRIC_R80):
            column = names.index(f"{cid}_{metric}_improvement")
            assert np.allclose(boot["draws"][:, column], 0.0)
            column_d = names.index(f"{cid}_{metric}_absdiff")
            assert np.allclose(boot["draws"][:, column_d], 0.0)
    table = m1st.contrast_table(boot)
    assert len(table) == 20  # 5 contrasts x 2 metrics x 2 forms
    main_rows = [r for r in table
                 if r["metric"] == m1st.METRIC_AURC and r["form"] == "improvement"]
    assert len(main_rows) == m1p.MAIN_FAMILY_SIZE == 5
    for row in table:
        assert row["status"] == "OK"
        assert row["ci99_lower"] <= row["ci99_upper"]
    # quantity names: 7 methods x 2 + 20 contrasts
    assert len(names) == 14 + 20
    # degenerate denominator -> NOT_IDENTIFIABLE relative, absolute still reported
    improvement, diff = m1st.improvement_and_diff(0.0, 1.0)
    assert np.isnan(improvement) and diff == -1.0
    improvement, diff = m1st.improvement_and_diff(1e-13, 0.0)
    assert np.isnan(improvement)
    improvement, diff = m1st.improvement_and_diff(2.0, 1.0)
    assert improvement == pytest.approx(0.5) and diff == pytest.approx(1.0)


def test_stable_tie_order_and_coverage_rounding():
    regret = np.arange(10, dtype=np.float64)
    tied = np.zeros(10)
    order = m1st.s4cs.order_by_risk(tied)
    assert order.tolist() == list(range(10))  # source row order kept among ties
    from ac_twin import stage4_statistics as s4s
    assert s4s.coverage_index(0.8, 1500) == 1200
    assert s4s.coverage_index(0.8, 3) == 2
    assert s4s.coverage_index(0.01, 1500) == 15


# --- 7. resume / drift refusal -----------------------------------------------------------------------------


def test_fit_resume_and_drift_refusal(tmp_path, monkeypatch):
    rng = np.random.default_rng(13)
    groups = 64
    x = rng.normal(size=(groups, 315))
    y_reg = np.abs(rng.normal(size=groups))
    y_high = (y_reg > 0.5).astype(np.float64)
    main_idx = np.arange(48)
    monitor_idx = np.arange(48, 64)
    model_root = tmp_path / "models"
    calls = {"n": 0}
    real_fit = m1t.fit_arm_seed

    def counting_fit(*args, **kwargs):
        calls["n"] += 1
        kwargs.pop("progress", None)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(m1p, "MAX_EPOCHS", 2)
    monkeypatch.setattr(m1t, "fit_arm_seed", counting_fit)
    arm_inputs = {arm: x.copy() for arm in m1p.ARMS}
    inventory = m1t.fit_all_arms(
        arm_inputs, main_idx=main_idx, monitor_idx=monitor_idx,
        y_reg=y_reg, y_high=y_high, pos_weight=1.0, model_root=model_root,
        device="cpu")
    assert calls["n"] == 15
    assert len(inventory["checkpoints"]) == 15
    # rerun: the create-once inventory verifies hashes and never retrains
    inventory2 = m1t.fit_all_arms(
        arm_inputs, main_idx=main_idx, monitor_idx=monitor_idx,
        y_reg=y_reg, y_high=y_high, pos_weight=1.0, model_root=model_root,
        device="cpu")
    assert calls["n"] == 15
    assert inventory2["written_at"] == inventory["written_at"]
    # checkpoint drift -> refusal
    entry = next(iter(inventory["checkpoints"].values()))
    path = Path(entry["path"])
    blob = path.read_bytes()
    path.write_bytes(blob + b"corruption")
    try:
        with pytest.raises(ValueError):
            m1t.load_arm_checkpoints(inventory, "FULL_MATCHED")
    finally:
        path.write_bytes(blob)


# --- 8. identity contract -----------------------------------------------------------------------------------


def test_identity_contract_roundtrip_and_tamper(tmp_path):
    sources, group_seed = _synthetic_sources(tmp_path, groups=12)
    built = m1f.build_legal_role(sources, "train", expected_groups=12)
    cache = tmp_path / "train_legal.npz"
    m1f.save_legal_cache(cache, "train", Path(sources["train_data"]), built)
    loaded = m1f.load_legal_cache(cache, role="train",
                                  source_path=Path(sources["train_data"]),
                                  expected_rows=12)
    assert np.array_equal(loaded["group_seed"], group_seed)
    assert np.array_equal(loaded["tokens"], built["tokens"])
    # scanner-visible keys absent; the mirrored scanner extracts nothing
    assert m1p.scanner_npz_seeds(cache) == set()
    with np.load(cache, allow_pickle=False) as archive:
        for key in archive.files:
            assert key not in m1p.FORBIDDEN_NPZ_KEYS
    # tampering with the identity digest must fail
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["source_identity_sha256"] = np.asarray("0" * 64)
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError):
        m1f.load_legal_cache(bad, role="train",
                             source_path=Path(sources["train_data"]), expected_rows=12)
    # duplicated / shifted row indices must fail
    arrays["source_identity_sha256"] = np.asarray(m1p.identity_digest(group_seed))
    arrays["source_row_index"] = np.arange(1, 13, dtype=np.int64)
    bad2 = tmp_path / "bad2.npz"
    np.savez_compressed(bad2, **arrays)
    with pytest.raises(ValueError):
        m1f.load_legal_cache(bad2, role="train",
                             source_path=Path(sources["train_data"]), expected_rows=12)
    # source file hash tamper must fail
    other = tmp_path / "other_data.npz"
    with np.load(sources["train_data"], allow_pickle=True) as archive:
        mutated = {k: archive[k] for k in archive.files}
    mutated["object_history"] = mutated["object_history"] + 1.0
    np.savez_compressed(other, **mutated)
    with pytest.raises(ValueError):
        m1f.load_legal_cache(cache, role="train", source_path=other, expected_rows=12)


def test_json_seed_free_guard():
    m1p.assert_json_seed_free({"main_row_index": [1, 2, 3],
                               "model_seeds": [8101, 8202]}, context="ok")
    with pytest.raises(ValueError):
        m1p.assert_json_seed_free({"group_seed": [123]}, context="bad")
    with pytest.raises(ValueError):
        m1p.assert_json_seed_free({"episode_seeds": [1, 2]}, context="bad")
    with pytest.raises(ValueError):
        m1p.assert_json_seed_free({"group_seed": 123}, context="bad")
    # metadata ints under non-hint keys are fine (mirrors the frozen scanner rules)
    m1p.assert_json_seed_free({"bootstrap_entropy": 2026100701,
                               "dev_seed_entropy": 2026092880}, context="ok")


def test_protected_manuscript_hashes_stable_and_detect_drift():
    protected = m1p.manuscript_protected_files()
    assert any("ac-twin_论文初稿_中文.md" in key for key in protected)
    assert not any("~$" in key for key in protected)
    hashes = m1p.hash_protected_files()
    assert m1p.verify_protected_hashes(hashes) == []
    drift = dict(hashes)
    first = next(iter(drift))
    drift[first] = "0" * 64
    assert m1p.verify_protected_hashes(drift) == [first]
