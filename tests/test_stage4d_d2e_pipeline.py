"""Focused blocking-invariant tests for the D2E repaired-prediction pipeline (plan 3.2/5).

The distinct NEW invariants here (the reused D2/D2R functions are tested upstream): the ego-slot
repair flows into the efficiency/progress term (the documented failure mechanism), the Direct head
and the three standard baselines share the identical repaired ``J_hat``, and the variant dispatch
behaves (kinematic floor action-agnostic, oracle = truth, slots 1-4 always preserved).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from ac_twin import stage4c_objective as s4co  # noqa: E402
from ac_twin import stage4d_d2e_pipeline as PIPE  # noqa: E402
from ac_twin import stage4cp_features as s4cpf  # noqa: E402

G, A, T, N = 4, 10, 20, 5
E = G * A


def _bundle() -> SimpleNamespace:
    """Minimal branch-bundle namespace; history is PER-GROUP (shared across the 10 actions)."""
    rng = np.random.default_rng(7)
    hist_xy = rng.normal(size=(G, 2)).astype(np.float64)
    factual_xy = (hist_xy[:, None, None, :] + rng.normal(size=(G, T, N, 2)) * 0.1
                  + np.arange(T)[None, :, None, None] * 0.05)
    cf_xy = factual_xy[:, None].repeat(A, axis=1) + rng.normal(size=(G, A, T, N, 2)) * 0.02
    inference_mask = np.ones((G, T, N), dtype=bool)
    inference_mask[:, :, 0] = False                       # ego slot excluded (collector contract)
    group_history = rng.normal(size=(G, 10, N, 8)).astype(np.float32)
    return SimpleNamespace(
        groups=G, n_actions=A, branches=E,
        history_last_xy=hist_xy,
        factual_xy=factual_xy.astype(np.float64),
        counterfactual_xy=cf_xy.astype(np.float64),
        inference_mask=inference_mask,
        valid_eval=np.repeat(inference_mask, A, axis=0),
        effect_target=(cf_xy.reshape(E, T, N, 2)
                       - np.repeat(factual_xy, A, axis=0)).astype(np.float32),
        history=np.repeat(group_history, A, axis=0),      # (E,10,N,8), per-group
    )


def test_repair_flows_into_progress_term() -> None:
    """Regression guard: trajectory_components reads ego progress from slot 0 UNCONDITIONALLY.

    The whole D2E repair theory rests on this frozen behaviour — if slot-0 progress were ever
    masked out, replacing slot 0 would stop fixing the efficiency term. (The interaction term
    legitimately also uses the slot-0 position as the proximity reference, so it changes too.)
    """
    rng = np.random.default_rng(1)
    base = rng.normal(size=(G, A, T, N, 2))
    moved = base.copy()
    moved[..., 0, 0] += 5.0                               # shift ego x by 5 m in every frame
    hist = np.zeros((G, A, 2))
    mask = np.ones((G, A, T, N), dtype=bool)
    mask[..., 0] = False                                  # ego excluded as a proximity TARGET
    c0 = s4co.trajectory_components(base, hist, mask)
    c1 = s4co.trajectory_components(moved, hist, mask)
    assert not np.allclose(c0["progress"], c1["progress"]), \
        "progress must respond to slot 0 (the repair mechanism)"
    np.testing.assert_allclose(c1["progress"], c0["progress"] + 5.0, rtol=1e-6,
                               err_msg="progress must shift by exactly the ego x displacement")


def test_direct_and_baselines_share_repaired_jhat() -> None:
    """The baselines + reference action are derived from the SAME repaired j_hat (fair share)."""
    rng = np.random.default_rng(2)
    e2e = rng.normal(size=(5, G, A, T, N, 2)).astype(np.float32) * 0.1   # NOTE: (S,G,A,T,N,2)
    b = _bundle()
    action_risks = {k: rng.normal(size=(G, A)) for k in
                    ("COMBINED_RISK_HEAD", "UNCERTAINTY_ONLY", "GEOMETRY_ONLY",
                     "RESPONSE_ONLY", "DISAGREEMENT_ONLY")}
    geometry = {"distance_to_ego": rng.uniform(1, 30, size=(G, N)),
                "relative_speed": rng.normal(size=(G, N)),
                "visible_count": np.full(G, float(N))}
    direct = s4cpf.build_relative_tokens(
        e2e=e2e, factual_xy=b.factual_xy, history_last_xy=b.history_last_xy,
        inference_mask=b.inference_mask, action_risks=action_risks, geometry=geometry,
        t_star=np.full(G, 12), group_seed=np.arange(G) + 1)
    j_hat = direct["j_hat"]                                # (G,A)
    rows = np.arange(G)
    order = np.argsort(j_hat, axis=-1, kind="stable")
    top1_top2 = j_hat[rows, order[:, 1]] - j_hat[rows, order[:, 0]]
    np.testing.assert_allclose(direct["baselines"]["MARGIN_ONLY"], -top1_top2, rtol=1e-9,
                               err_msg="MARGIN_ONLY must derive from the repaired J_hat")
    assert np.array_equal(direct["reference_action"], np.argmin(j_hat, axis=-1))
    for name in ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD"):
        assert name in direct["baselines"]


def test_variant_dispatch() -> None:
    b = _bundle()
    rng = np.random.default_rng(3)
    e2e_old = rng.normal(size=(5, E, T, N, 2)).astype(np.float32)
    # kinematic floor: identical delta for every action in a group (action-agnostic)
    kin = PIPE.kinematic_e2e(b, e2e_old)
    kin_slot0 = kin[:, :, :, 0, :].reshape(5, G, A, T, 2)
    for a in range(1, A):
        np.testing.assert_allclose(kin_slot0[:, :, 0], kin_slot0[:, :, a], rtol=1e-5,
                                   err_msg="kinematic floor must be action-agnostic")
    # oracle: slot 0 == the true ego displacement
    orc = PIPE.oracle_e2e(b, e2e_old)
    true_ego = np.asarray(b.effect_target[:, :, 0, :], dtype=np.float32)
    np.testing.assert_allclose(orc[:, :, :, 0, :],
                               np.broadcast_to(true_ego[None], (5,) + true_ego.shape), rtol=1e-5)
    # slots 1-4 preserved in both variants
    np.testing.assert_array_equal(kin[:, :, :, 1:, :], e2e_old[:, :, :, 1:, :])
    np.testing.assert_array_equal(orc[:, :, :, 1:, :], e2e_old[:, :, :, 1:, :])


def test_ego_quality_structure() -> None:
    b = _bundle()
    rng = np.random.default_rng(4)
    e2e = rng.normal(size=(5, E, T, N, 2)).astype(np.float32) * 0.1
    ego_mask = np.ones((E, T), dtype=bool)
    q = PIPE.ego_quality(b, e2e, ego_mask)
    for key in ("ego_final_progress_mae", "ego_progress_rank_spearman",
                "progress_constant_group_frac", "ego_effect_ade", "non_ego_effect_ade"):
        assert key in q, key
    assert np.isfinite(q["ego_final_progress_mae"]) and q["ego_final_progress_mae"] >= 0.0
    assert np.isfinite(q["ego_effect_ade"]) and np.isfinite(q["non_ego_effect_ade"])
    # oracle (perfect slot 0) drives the ego ADE + final MAE to ~0
    q_oracle = PIPE.ego_quality(b, PIPE.oracle_e2e(b, e2e), ego_mask)
    assert q_oracle["ego_effect_ade"] < 1e-4
    assert q_oracle["ego_final_progress_mae"] < 1e-4
