"""Stage 2 non-monotonic statistics and mechanical Gates (S*/A/B).

All statistics operate on world-coordinate, de-normalized arrays with an
``episode × time × agent`` alignment. The independent unit is the episode: the
five model seeds are averaged per agent-time first, and every confirmatory CI is
an episode-cluster bootstrap over the frozen bootstrap seed. Training-seed
randomness is described only by leave-one-model-seed-out and is never folded into
the CI. ``C_rebound`` is a readable summary point estimate only; the裁判 CIs are
carried by the smooth contrasts ``C_high_low`` and ``C_high_mid``.

This module is pure NumPy: no torch, no training, no fitting on the holdout. The
Gate S* baselines and representation probe use only coefficients frozen on the
Stage 1 train split and sealed in the protocol lock.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ac_twin import stage1_statistics as _s1
from ac_twin import stage1_plus_statistics as _plus
from ac_twin.stage2_protocol import motion_summary

# Final decision vocabulary (frozen; exactly one is emitted).
INVALID_TECHNICAL = "INVALID_TECHNICAL"
INCONCLUSIVE_MODEL = "INCONCLUSIVE_FROZEN_MODEL_OUT_OF_SAMPLE_INSUFFICIENT"
GATE_A_CONFIRMED = "GATE_A_CONFIRMED_NONMONOTONIC"
GATE_A_RESPONSE_SUPPORT = "GATE_A_NOT_CONFIRMED_RESPONSE_SUPPORT"
GATE_A_NOT_CONFIRMED = "GATE_A_NOT_CONFIRMED"

GATE_B_SUPPORTED = "GATE_B_SUPPORTED"
GATE_B_NOT_SUPPORTED = "GATE_B_NOT_SUPPORTED"
GATE_B_UNDERPOWERED = "GATE_B_UNDERPOWERED"

BIN_NAMES = ("zero", "positive_low", "positive_mid", "positive_high")
# The three smooth contrasts that carry the practical point screen.
CONTRASTS = ("c_presence", "c_high_low", "c_high_mid")

# Bootstrap-seed offsets (base = config statistics.bootstrap_seed = 20270908).
_OFF = {
    "v_zero": 1, "v_positive_all": 2, "v_low": 3, "v_mid": 4, "v_high": 5,
    "c_presence": 20, "c_high_low": 21, "c_high_mid": 22,
    "gate_b": 30, "s2_spearman": 40, "s2_mae2": 41, "s2_mae3": 42,
    "s3_gap": 50, "s3_probe_shuffled": 60, "s3_probe_scene": 61,
    "sec_spearman": 70, "sec_trend": 71, "sec_low_high": 72,
}


# --- per-episode helpers ------------------------------------------------------------


def _episode_group_sums(errors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    sums = np.zeros(errors.shape[0], dtype=np.float64)
    np.add.at(sums, np.nonzero(mask)[0], errors[mask])
    return sums


def _episode_group_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    sums = np.zeros(values.shape[0])
    counts = np.zeros(values.shape[0])
    np.add.at(sums, np.nonzero(mask)[0], values[mask])
    np.add.at(counts, np.nonzero(mask)[0], 1.0)
    out = np.full(values.shape[0], np.nan)
    nonzero = counts > 0
    out[nonzero] = sums[nonzero] / counts[nonzero]
    return out


def _per_episode_bin_means(values, bin_id, valid_mask, bins):
    values = np.asarray(values, dtype=np.float64)
    sel = np.asarray(valid_mask, dtype=bool) & np.isin(bin_id, list(bins))
    sums = np.zeros(values.shape[0])
    counts = np.zeros(values.shape[0])
    np.add.at(sums, np.nonzero(sel)[0], values[sel])
    np.add.at(counts, np.nonzero(sel)[0], 1.0)
    eligible = np.nonzero(counts > 0)[0]
    per_episode = np.full(values.shape[0], np.nan)
    per_episode[eligible] = sums[eligible] / counts[eligible]
    return per_episode, eligible, int(sel.sum())


# --- bin episode sums + three contrasts ---------------------------------------------


def _bin_episode_sums(l_suffix, l_shuffled, bin_id, valid_mask):
    """Per-episode suffix/shuffled sums for bins 0,1,2,3 and the positive union."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    groups = {"zero": 0, "positive_low": 1, "positive_mid": 2, "positive_high": 3}
    suf, shf = {}, {}
    for name, b in groups.items():
        mask = valid_mask & (bin_id == b)
        suf[name] = _episode_group_sums(l_suffix, mask)
        shf[name] = _episode_group_sums(l_shuffled, mask)
    positive_mask = valid_mask & (bin_id >= 1)
    suf["positive_all"] = _episode_group_sums(l_suffix, positive_mask)
    shf["positive_all"] = _episode_group_sums(l_shuffled, positive_mask)
    return suf, shf


def _v_evt(suf, shf, indices):
    denom = shf[indices].sum()
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - suf[indices].sum() / denom)


def three_contrasts(suf: Mapping[str, np.ndarray], shf: Mapping[str, np.ndarray],
                    indices: np.ndarray) -> dict[str, float]:
    """The three smooth contrasts as point estimates over a subset of episodes.

    C_presence = V(zero) - V(positive_all); C_high_low = V(high) - V(low);
    C_high_mid = V(high) - V(mid).
    """
    indices = np.asarray(indices)
    v_zero = _v_evt(suf["zero"], shf["zero"], indices)
    v_pos = _v_evt(suf["positive_all"], shf["positive_all"], indices)
    v_low = _v_evt(suf["positive_low"], shf["positive_low"], indices)
    v_mid = _v_evt(suf["positive_mid"], shf["positive_mid"], indices)
    v_high = _v_evt(suf["positive_high"], shf["positive_high"], indices)
    return {
        "c_presence": v_zero - v_pos,
        "c_high_low": v_high - v_low,
        "c_high_mid": v_high - v_mid,
        "v_zero": v_zero, "v_positive_all": v_pos,
        "v_low": v_low, "v_mid": v_mid, "v_high": v_high,
    }


# --- main computation ------------------------------------------------------------------


def _seedmean_errors(predictions_main: Mapping[str, np.ndarray], target: np.ndarray):
    """Per-seed and seed-mean position errors per condition (seeds not samples)."""
    errors_by_seed, errors = {}, {}
    for condition, preds in predictions_main.items():
        preds = np.asarray(preds, dtype=np.float64)
        if preds.ndim != 5:
            raise ValueError(f"{condition} predictions must be S×E×T×N×2")
        per_seed = np.stack([_s1.position_errors(preds[s], target)
                             for s in range(preds.shape[0])])
        errors_by_seed[condition] = per_seed
        errors[condition] = _s1.seed_mean(per_seed)
    return errors_by_seed, errors


def _initial_distance_to_ego(history: np.ndarray) -> np.ndarray:
    presence = history[..., 0] > 0
    ego_last = history[:, -1, 0, 2:4]
    episodes, _, agents, _ = history.shape
    distance = np.full((episodes, agents), np.nan)
    for e in range(episodes):
        for i in range(agents):
            present = np.nonzero(presence[e, :, i])[0]
            if len(present):
                distance[e, i] = np.linalg.norm(history[e, present[-1], i, 2:4] - ego_last[e])
    return distance


def compute_stage2_statistics(truth: Mapping[str, Any],
                              predictions: Mapping[str, Any],
                              frozen_params: Mapping[str, np.ndarray],
                              manifest: Mapping[str, Any],
                              config: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the full Stage 2 statistical report (gates evaluated separately)."""
    stats = config["statistics"]
    n_resamples = int(stats["bootstrap_resamples"])
    confidence = float(stats["confidence_level"])
    base = int(stats["bootstrap_seed"])
    point_screen = float(stats["practical_point_screen"])
    min_bin_episodes = int(stats["min_bin_episodes"])
    eta_eps = float(stats["eta_epsilon_m"])

    r_pos = np.asarray(truth["r_pos"], dtype=np.float64)
    target = np.asarray(truth["target"], dtype=np.float64)
    valid_mask = np.asarray(truth["valid_mask"], dtype=bool).copy()
    valid_mask[:, :, 0] = False  # exclude ego from the main analysis
    cf_action = np.asarray(truth["cf_action"])
    episode_seed = np.asarray(truth["episode_seed"])
    n_episodes = valid_mask.shape[0]

    thresholds = {"zero_tolerance": float(config["response_bins"]["zero_tolerance"]),
                  "t1": float(config["response_bins"]["t1"]),
                  "t2": float(config["response_bins"]["t2"])}
    bin_id = _s1.assign_response_bin(r_pos, thresholds)

    errors_by_seed, errors = _seedmean_errors(predictions["main"], target)
    l_no = errors["no_suffix"]
    l_suffix = errors["uniform_suffix"]
    l_shuffled = errors["shuffled_suffix"]

    bins = _s1.aggregate_bins(l_no, l_suffix, l_shuffled, bin_id, valid_mask)
    metrics = _s1.value_metrics(l_suffix, l_shuffled, l_no, eta_eps)

    bin_masks = {b: valid_mask & (bin_id == b) for b in (0, 1, 2, 3)}
    positive_mask = valid_mask & (bin_id >= 1)
    suf, shf = _bin_episode_sums(l_suffix, l_shuffled, bin_id, valid_mask)
    arange = np.arange(n_episodes)

    def boot(stat_fn, offset):
        return _s1.episode_bootstrap(n_episodes, stat_fn, n_resamples, confidence, base + offset)

    def unavailable(offset):
        return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_resamples": n_resamples, "n_finite_draws": 0,
                "confidence_level": confidence, "bootstrap_seed": base + offset,
                "n_episodes": int(n_episodes)}

    def v_boot(selection, offset):
        if not np.asarray(selection).any():
            return unavailable(offset)
        return boot(_plus.make_group_v_evt_statistic(l_suffix, l_shuffled, selection), offset)

    def contrast_boot(left, right, offset):
        if not (np.asarray(left).any() and np.asarray(right).any()):
            return unavailable(offset)
        return boot(_plus.make_group_contrast_statistic(l_suffix, l_shuffled, left, right), offset)

    v_zero = v_boot(bin_masks[0], _OFF["v_zero"])
    v_positive_all = v_boot(positive_mask, _OFF["v_positive_all"])
    v_low = v_boot(bin_masks[1], _OFF["v_low"])
    v_mid = v_boot(bin_masks[2], _OFF["v_mid"])
    v_high = v_boot(bin_masks[3], _OFF["v_high"])
    c_presence = contrast_boot(bin_masks[0], positive_mask, _OFF["c_presence"])
    c_high_low = contrast_boot(bin_masks[3], bin_masks[1], _OFF["c_high_low"])
    c_high_mid = contrast_boot(bin_masks[3], bin_masks[2], _OFF["c_high_mid"])
    # C_rebound: readable summary point estimate only (no judgment CI).
    full_contrasts = three_contrasts(suf, shf, arange)
    c_rebound_point = float(min(full_contrasts["c_high_low"], full_contrasts["c_high_mid"]))

    bin_episode_counts = {name: int(bins[name]["n_episodes"])
                          for name in ("positive_low", "positive_mid", "positive_high")}

    robustness = _robustness(errors_by_seed, l_suffix, l_shuffled, bin_id, valid_mask,
                             cf_action, n_episodes)
    alternate = _alternate_donor(predictions.get("alt_donor", {}), l_suffix, target,
                                 bin_id, valid_mask)
    gate_b, gate_b_detail = _gate_b(l_no, l_suffix, bin_id, valid_mask, n_episodes,
                                    int(stats["gate_b_min_episodes"]), n_resamples,
                                    confidence, base + _OFF["gate_b"])
    gate_s_star = _gate_s_star(truth, predictions, errors, frozen_params, bin_id,
                               valid_mask, thresholds, n_resamples, confidence, base)
    secondary = _secondary(truth, errors, metrics, bin_id, valid_mask, positive_mask,
                           n_episodes, n_resamples, confidence, base)
    manifest_qa_pass, manifest_qa_detail = _manifest_qa(manifest)

    point_screen_checks = {name: bool(full_contrasts[name] >= point_screen)
                           for name in CONTRASTS}

    report = {
        "n_episodes": int(n_episodes),
        "n_agent_time_valid": int(valid_mask.sum()),
        "response_bins": thresholds,
        "bins": bins,
        "bin_episode_counts": bin_episode_counts,
        "v_zero": v_zero, "v_positive_all": v_positive_all,
        "v_low": v_low, "v_mid": v_mid, "v_high": v_high,
        "c_presence": c_presence, "c_high_low": c_high_low, "c_high_mid": c_high_mid,
        "c_rebound_point": c_rebound_point,
        "point_screen": point_screen,
        "point_screen_checks": point_screen_checks,
        "robustness": robustness,
        "alternate_donor": alternate,
        "gate_b": gate_b,
        "gate_b_detail": gate_b_detail,
        "gate_s_star": gate_s_star,
        "secondary": secondary,
        "manifest_qa_pass": manifest_qa_pass,
        "manifest_qa_detail": manifest_qa_detail,
        "episode_seed_unique": bool(len(np.unique(episode_seed)) == n_episodes),
        "_min_bin_episodes": min_bin_episodes,
    }
    return report


# --- robustness ---------------------------------------------------------------------


def _robustness(errors_by_seed, l_suffix, l_shuffled, bin_id, valid_mask, cf_action, n_episodes):
    """Leave-one-seed/action/episode-out and 1%/5% high-error deletion on the three
    contrasts; every variant must stay strictly positive (no reversal to negative)."""
    arange = np.arange(n_episodes)
    n_seeds = errors_by_seed["uniform_suffix"].shape[0]

    # Leave-one-model-seed-out.
    loso_seed = {name: [] for name in CONTRASTS}
    for s in range(n_seeds):
        keep = [i for i in range(n_seeds) if i != s]
        ls = errors_by_seed["uniform_suffix"][keep].mean(axis=0)
        lh = errors_by_seed["shuffled_suffix"][keep].mean(axis=0)
        suf_s, shf_s = _bin_episode_sums(ls, lh, bin_id, valid_mask)
        c = three_contrasts(suf_s, shf_s, arange)
        for name in CONTRASTS:
            loso_seed[name].append(float(c[name]))

    # Seed-mean per-bin episode sums for action/episode/deletion robustness.
    suf, shf = _bin_episode_sums(l_suffix, l_shuffled, bin_id, valid_mask)

    # Leave-one-action-type-out.
    loso_action = {}
    for action in sorted(np.unique(cf_action).tolist()):
        keep = np.flatnonzero(cf_action != action)
        c = three_contrasts(suf, shf, keep)
        loso_action[int(action)] = {name: float(c[name]) for name in CONTRASTS}
        loso_action[int(action)]["n_episodes"] = int(len(keep))

    # Leave-one-episode-out.
    eligible = np.nonzero(valid_mask.any(axis=(1, 2)))[0]
    loeo = {name: [] for name in CONTRASTS}
    for e in eligible:
        keep = eligible[eligible != e]
        c = three_contrasts(suf, shf, keep)
        for name in CONTRASTS:
            loeo[name].append(float(c[name]))

    # High-error deletion by per-episode mean(L_suffix + L_shuffled).
    score = _episode_group_means(l_suffix + l_shuffled, valid_mask)
    deletion = {}
    for frac in (0.01, 0.05):
        k = max(1, int(np.floor(frac * len(eligible))))
        order = eligible[np.argsort(-score[eligible])]
        kept = np.setdiff1d(eligible, order[:k])
        c = three_contrasts(suf, shf, kept)
        deletion[f"drop_top_{int(round(frac * 100))}pct"] = {
            **{name: float(c[name]) for name in CONTRASTS},
            "n_episodes": int(len(kept)), "n_episodes_removed": int(k),
        }

    def all_positive(seq):
        return bool(len(seq) and np.isfinite(seq).all() and (np.asarray(seq) > 0).all())

    checks = {
        "leave_one_model_seed_out": {name: all_positive(loso_seed[name]) for name in CONTRASTS},
        "leave_one_action_type_out": {
            name: all(v[name] > 0 and np.isfinite(v[name]) for v in loso_action.values())
            for name in CONTRASTS},
        "leave_one_episode_out": {name: all_positive(loeo[name]) for name in CONTRASTS},
        "high_error_deletion": {
            name: all(v[name] > 0 and np.isfinite(v[name]) for v in deletion.values())
            for name in CONTRASTS},
    }
    return {
        "leave_one_model_seed_out": loso_seed,
        "leave_one_action_type_out": loso_action,
        "leave_one_episode_out_min": {name: float(np.min(loeo[name])) if loeo[name] else float("nan")
                                      for name in CONTRASTS},
        "high_error_deletion": deletion,
        "checks": checks,
        "all_positive": bool(all(all(v.values()) for v in checks.values())),
    }


def _alternate_donor(alt_donor, l_suffix, target, bin_id, valid_mask):
    """Point-estimate sensitivity of the three contrasts to each alternate donor.

    ``l_suffix`` is donor-independent (the correct-suffix M1 channel); only the
    shuffled channel changes with the donor manifest. Reports point estimates only.
    """
    out = {}
    arange = np.arange(valid_mask.shape[0])
    for donor_seed, preds in alt_donor.items():
        preds = np.asarray(preds, dtype=np.float64)
        l_shuffled_alt = _s1.seed_mean(np.stack([
            _s1.position_errors(preds[s], target) for s in range(preds.shape[0])]))
        suf, shf = _bin_episode_sums(l_suffix, l_shuffled_alt, bin_id, valid_mask)
        contrasts = three_contrasts(suf, shf, arange)
        out[str(donor_seed)] = {name: float(contrasts[name]) for name in CONTRASTS}
    return out


def _gate_b(l_no, l_suffix, bin_id, valid_mask, n_episodes, min_episodes,
            n_resamples, confidence, seed):
    """Gate B: positive_high Delta = L_no - L_suffix with cluster CI fully below 0."""
    delta = l_no - l_suffix
    high_mask = valid_mask & (bin_id == 3)
    high_episodes = np.unique(np.nonzero(high_mask)[0])
    n_high = int(len(high_episodes))
    episode_axis = np.arange(n_episodes)[:, None, None]
    if n_high >= 2:
        per_ep = np.array([np.nanmean(delta[high_mask & (episode_axis == e)])
                           for e in high_episodes])

        def stat(idx):
            return float(np.mean(per_ep[idx]))
        boot = _s1.episode_bootstrap(n_high, stat, n_resamples, confidence, seed)
        ci_upper, point = boot["ci_upper"], boot["point"]
    else:
        boot = {"n_episodes": n_high}
        ci_upper, point = None, None
    if n_high < min_episodes or ci_upper is None:
        status = GATE_B_UNDERPOWERED
    elif point < 0 and ci_upper < 0:
        status = GATE_B_SUPPORTED
    else:
        status = GATE_B_NOT_SUPPORTED
    return status, boot


# --- Gate S* ------------------------------------------------------------------------


def _gate_s_star(truth, predictions, errors, frozen_params, bin_id, valid_mask,
                 thresholds, n_resamples, confidence, base):
    s1, s1_detail = _s1_star(predictions["main"], valid_mask)
    s2, s2_detail = _s2_star(truth, predictions, frozen_params, valid_mask,
                             n_resamples, confidence, base)
    s3, s3_detail = _s3_star(truth, predictions, frozen_params, bin_id, valid_mask,
                             thresholds, n_resamples, confidence, base)
    return {"S1": bool(s1), "S2": bool(s2), "S3": bool(s3),
            "pass": bool(s1 and s2 and s3),
            "S1_detail": s1_detail, "S2_detail": s2_detail, "S3_detail": s3_detail}


def _s1_star(main, valid_mask):
    """S1*: 15 main predictions finite, non-constant, non-single-vehicle template."""
    finite = True
    non_constant = True
    for condition, preds in main.items():
        preds = np.asarray(preds, dtype=np.float64)
        finite = finite and bool(np.isfinite(preds).all())
    m1 = _s1.seed_mean(np.asarray(main["uniform_suffix"], dtype=np.float64))
    if np.allclose(m1, m1.flat[0]):
        non_constant = False
    # Non-single-vehicle-template: not all vehicles produce identical outputs.
    n_agents = m1.shape[2]
    template = all(np.allclose(m1[:, :, i], m1[:, :, 1]) for i in range(2, n_agents))
    passed = bool(finite and non_constant and not template)
    return passed, {"finite": bool(finite), "non_constant": bool(non_constant),
                    "single_vehicle_template": bool(template)}


def _baseline_design_holdout(truth, valid_mask, use_distance):
    """Design for the frozen S2* baselines on the holdout (valid non-ego rows)."""
    n, T, N = valid_mask.shape
    fa = np.asarray(truth["factual_actions"])
    ca = np.asarray(truth["counterfactual_actions"])
    action_diff = np.broadcast_to(np.abs(fa - ca)[:, :, None].astype(np.float64), (n, T, N))
    horizon = np.broadcast_to(np.arange(T)[None, :, None], (n, T, N)).astype(np.float64)
    cols = [np.ones((n, T, N)), action_diff, horizon]
    if use_distance:
        distance = _initial_distance_to_ego(np.asarray(truth["history"], dtype=np.float64))
        cols.append(np.broadcast_to(distance[:, None, :], (n, T, N)).astype(np.float64))
    X = np.column_stack([c.reshape(-1) for c in cols])
    return X


def _s2_star(truth, predictions, frozen_params, valid_mask, n_resamples, confidence, base):
    """S2*: M1 R_hat tracks true R (Spearman CI>0) and beats frozen MAE baselines."""
    r_pos = np.asarray(truth["r_pos"], dtype=np.float64)
    m1 = _s1.seed_mean(np.asarray(predictions["main"]["uniform_suffix"], dtype=np.float64))
    r_hat = np.linalg.norm(m1, axis=-1)  # E,T,N
    n_episodes = valid_mask.shape[0]

    spear = _s1.episode_bootstrap(
        n_episodes, _s1.make_spearman_statistic(r_hat, r_pos, valid_mask),
        n_resamples, confidence, base + _OFF["s2_spearman"])
    spearman_ok = bool(np.isfinite(spear["ci_lower"]) and spear["ci_lower"] > 0)

    X2 = _baseline_design_holdout(truth, valid_mask, False)
    X3 = _baseline_design_holdout(truth, valid_mask, True)
    pred2 = (X2 @ np.asarray(frozen_params["beta_action_horizon"])).reshape(r_pos.shape)
    pred3 = (X3 @ np.asarray(frozen_params["beta_action_horizon_distance"])).reshape(r_pos.shape)

    def mae(arr):
        return float(np.abs(arr - r_pos)[valid_mask].mean())

    def mae_diff_boot(base_pred, offset):
        base_abs = np.abs(base_pred - r_pos)
        mod_abs = np.abs(r_hat - r_pos)
        base_rows = _episode_group_means(base_abs, valid_mask)
        mod_rows = _episode_group_means(mod_abs, valid_mask)

        def stat(indices):
            return float(np.nanmean(mod_rows[indices] - base_rows[indices]))
        return _s1.episode_bootstrap(n_episodes, stat, n_resamples, confidence, base + offset)

    diff2 = mae_diff_boot(pred2, _OFF["s2_mae2"])
    diff3 = mae_diff_boot(pred3, _OFF["s2_mae3"])
    mae_ok = bool(np.isfinite(diff2["ci_upper"]) and diff2["ci_upper"] < 0
                  and np.isfinite(diff3["ci_upper"]) and diff3["ci_upper"] < 0)
    passed = bool(spearman_ok and mae_ok)
    return passed, {"spearman": spear, "model_mae": mae(r_hat),
                    "baseline_action_horizon_mae": mae(pred2),
                    "baseline_action_horizon_distance_mae": mae(pred3),
                    "mae_diff_vs_baseline2": diff2, "mae_diff_vs_baseline3": diff3}


def _s3_star(truth, predictions, frozen_params, bin_id, valid_mask, thresholds,
             n_resamples, confidence, base):
    """S3*: fixed-M1 suffix-swap prediction evidence + frozen representation probe."""
    target = np.asarray(truth["target"], dtype=np.float64)
    n_episodes = valid_mask.shape[0]
    correct_world = _s1.seed_mean(np.asarray(predictions["main"]["uniform_suffix"], dtype=np.float64))
    swapped_world = _s1.seed_mean(np.asarray(predictions["m1_swap"], dtype=np.float64))
    l_correct = _s1.position_errors(correct_world, target)
    l_swapped = _s1.position_errors(swapped_world, target)
    diff = l_swapped - l_correct  # positive => correct suffix beats swapped suffix

    change = np.abs(swapped_world - correct_world)
    nonzero_change = bool(np.count_nonzero(change[valid_mask]) > 0)
    change_mag = np.linalg.norm(change, axis=-1)
    agent_change = (np.where(valid_mask, change_mag, 0.0).sum(axis=(0, 1))
                    / np.maximum(valid_mask.sum(axis=(0, 1)), 1))
    vehicle_specific = bool(agent_change.size > 1
                            and not np.allclose(agent_change, agent_change[0]))

    gap_bins = {"zero": (0,), "positive_low": (1,), "positive_mid": (2,)}
    gap_stats, gap_ok = {}, False
    for name, bins in gap_bins.items():
        per_episode, eligible, n_agent_time = _per_episode_bin_means(diff, bin_id, valid_mask, bins)
        if len(eligible) < 2:
            gap_stats[name] = None
            continue

        def stat(idx, per_episode=per_episode, eligible=eligible):
            return float(np.nanmean(per_episode[eligible[idx]]))
        boot = _s1.episode_bootstrap(len(eligible), stat, n_resamples, confidence,
                                     base + _OFF["s3_gap"])
        boot["n_episodes"] = int(len(eligible))
        boot["n_agent_time"] = n_agent_time
        gap_stats[name] = boot
        if np.isfinite(boot["ci_lower"]) and boot["ci_lower"] > 0:
            gap_ok = True
    prediction_evidence = bool(nonzero_change and vehicle_specific and gap_ok)

    representation_evidence, rep_detail = _s3_probe_holdout(
        truth, predictions, frozen_params, n_episodes, n_resamples, confidence, base)
    passed = bool(prediction_evidence and representation_evidence)
    return passed, {"prediction_evidence": bool(prediction_evidence),
                    "nonzero_prediction_change": bool(nonzero_change),
                    "vehicle_specific_change": bool(vehicle_specific),
                    "bin_gaps": gap_stats,
                    "representation_evidence": bool(representation_evidence),
                    "representation_detail": rep_detail}


def _s3_probe_holdout(truth, predictions, frozen_params, n_episodes,
                      n_resamples, confidence, base):
    """Apply the frozen Stage 1 probe to holdout factual summaries; beat baselines."""
    if "factual_summaries" not in predictions or "factual_active" not in predictions:
        return False, {"reason": "holdout factual summaries not provided"}
    summaries = np.asarray(predictions["factual_summaries"], dtype=np.float64)  # E,N,h
    active = np.asarray(predictions["factual_active"], dtype=bool)               # E,N
    y = motion_summary(np.asarray(truth["factual"], dtype=np.float64))           # E,N,6
    n_val, N, hidden = summaries.shape
    Xv = np.column_stack([np.ones(n_val * N), summaries.reshape(-1, hidden)])

    pred_probe = (Xv @ np.asarray(frozen_params["probe_coefficients"])).reshape(n_val, N, 6)
    pred_shuffled = (Xv @ np.asarray(frozen_params["probe_coefficients_shuffled"])).reshape(n_val, N, 6)
    pred_scene = np.broadcast_to(np.asarray(frozen_params["probe_scene_mean"]), (n_val, N, 6))

    se_probe = ((pred_probe - y) ** 2).mean(-1)
    se_shuffled = ((pred_shuffled - y) ** 2).mean(-1)
    se_scene = ((pred_scene - y) ** 2).mean(-1)

    def per_episode(se):
        count = active.sum(axis=1)
        out = np.full(n_val, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[count > 0] = (se * active).sum(axis=1)[count > 0] / count[count > 0]
        return out

    pe_probe, pe_shuffled, pe_scene = per_episode(se_probe), per_episode(se_shuffled), per_episode(se_scene)
    diff_shuffled = pe_probe - pe_shuffled
    diff_scene = pe_probe - pe_scene
    eligible = np.nonzero(np.isfinite(diff_shuffled) & np.isfinite(diff_scene))[0]
    n_eligible = len(eligible)

    def boot(diff, offset):
        if n_eligible < 2:
            return {"point": float("nan"), "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "n_episodes": int(n_eligible)}
        return _s1.episode_bootstrap(
            n_eligible, lambda idx: float(np.nanmean(diff[eligible[idx]])),
            n_resamples, confidence, base + offset)

    boot_shuffled = boot(diff_shuffled, _OFF["s3_probe_shuffled"])
    boot_scene = boot(diff_scene, _OFF["s3_probe_scene"])
    passed = bool(n_eligible >= 2
                  and np.isfinite(boot_shuffled["ci_upper"]) and boot_shuffled["ci_upper"] < 0
                  and np.isfinite(boot_scene["ci_upper"]) and boot_scene["ci_upper"] < 0)
    return passed, {"n_episodes": int(n_eligible),
                    "probe_minus_shuffled_label": boot_shuffled,
                    "probe_minus_scene_mean": boot_scene}


# --- secondary (descriptive only) ----------------------------------------------------


def _secondary(truth, errors, metrics, bin_id, valid_mask, positive_mask,
               n_episodes, n_resamples, confidence, base):
    r_pos = np.asarray(truth["r_pos"], dtype=np.float64)
    l_no = errors["no_suffix"]
    l_suffix = errors["uniform_suffix"]
    l_shuffled = errors["shuffled_suffix"]
    eta = metrics["eta_evt"]
    arange = np.arange(n_episodes)

    pos_spearman = _s1.episode_bootstrap(
        n_episodes, _s1.make_spearman_statistic(r_pos, eta, positive_mask),
        n_resamples, confidence, base + _OFF["sec_spearman"])

    cov = _broadcast_covariates(truth, valid_mask)
    standardizer = {}
    for name in ("horizon_index", "cf_duration", "initial_distance",
                 "history_visible_ratio", "visible_count"):
        values = cov[name][positive_mask]
        standardizer[name] = {"mean": float(values.mean()),
                              "std": float(max(values.std(), 1e-6))}
    X_pos, y_pos, feature_names, log_col = _s1.build_trend_design(
        eta, r_pos, cov["horizon_index"], cov["cf_action"], cov["cf_duration"],
        cov["initial_distance"], cov["history_visible_ratio"], cov["visible_count"],
        positive_mask, standardizer)
    row_episode_pos = np.nonzero(positive_mask)[0]
    pos_trend = _s1.episode_bootstrap(
        n_episodes, _s1.make_trend_coefficient_statistic(X_pos, y_pos, log_col, row_episode_pos, n_episodes),
        n_resamples, confidence, base + _OFF["sec_trend"])

    original_low = valid_mask & np.isin(bin_id, [0, 1])
    high_mask = valid_mask & (bin_id == 3)
    if original_low.any() and high_mask.any():
        low_high = _s1.episode_bootstrap(
            n_episodes, _plus.make_group_contrast_statistic(l_suffix, l_shuffled, original_low,
                                                            high_mask),
            n_resamples, confidence, base + _OFF["sec_low_high"])
    else:
        low_high = {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                    "n_episodes": int(n_episodes)}

    return {
        "label": "secondary_only",
        "positive_only_spearman": pos_spearman,
        "positive_only_trend_log_r": {**pos_trend, "feature_names": feature_names},
        "original_v_evt_low_minus_high": low_high,
        "d_evt_mean": float(np.nanmean(metrics["D_evt"][valid_mask])),
        "rho_no_mean": float(np.nanmean(1.0 - l_suffix[valid_mask] / np.maximum(l_no[valid_mask], 1e-12))),
        "eta_evt_mean": float(np.nanmean(eta[valid_mask])),
    }


def _broadcast_covariates(truth, valid_mask):
    n, T, N = valid_mask.shape
    history = np.asarray(truth["history"], dtype=np.float64)
    distance = _initial_distance_to_ego(history)
    presence = (history[..., 0] > 0)
    visible_ratio = presence.mean(axis=1)
    visible_count = np.asarray(truth["visible_count"], dtype=np.float64)
    horizon = np.broadcast_to(np.arange(T)[None, :, None], (n, T, N)).astype(np.float64)
    return {
        "horizon_index": horizon,
        "cf_action": np.broadcast_to(np.asarray(truth["cf_action"])[:, None, None], (n, T, N)),
        "cf_duration": np.broadcast_to(np.asarray(truth["cf_duration"])[:, None, None], (n, T, N)).astype(np.float64),
        "initial_distance": np.broadcast_to(distance[:, None, :], (n, T, N)),
        "history_visible_ratio": np.broadcast_to(visible_ratio[:, None, :], (n, T, N)),
        "visible_count": np.broadcast_to(visible_count[:, None, None], (n, T, N)).astype(np.float64),
    }


def _manifest_qa(manifest):
    if not manifest:
        return False, {}
    detail, ok = {}, True
    primary = manifest.get("primary", {})
    qa = primary.get("qa", {})
    detail["primary"] = qa
    ok = all(int(qa.get(key, -1)) == 0
             for key in ("self_match", "same_seed", "cross_split", "duplicate_donor"))
    return bool(ok), detail


# --- mechanical gate judges ------------------------------------------------------------


def judge_gate_a(report: Mapping[str, Any], *, gate_q_pass: bool) -> str:
    """Mechanical Gate A / final-status judge with fixed priority.

    Priority: Gate Q fail -> INVALID_TECHNICAL; Gate S* fail -> INCONCLUSIVE; any
    positive bin < min episodes -> RESPONSE_SUPPORT; any Gate A condition fails ->
    NOT_CONFIRMED; all pass -> CONFIRMED_NONMONOTONIC.
    """
    if not gate_q_pass:
        return INVALID_TECHNICAL
    if not report["gate_s_star"]["pass"]:
        return INCONCLUSIVE_MODEL

    stats_needed = report
    min_bin = report.get("_min_bin_episodes", 100)
    counts = report["bin_episode_counts"]
    if any(counts[name] < min_bin for name in
           ("positive_low", "positive_mid", "positive_high")):
        return GATE_A_RESPONSE_SUPPORT

    def ci_lower_pos(entry):
        return bool(np.isfinite(entry["ci_lower"]) and entry["ci_lower"] > 0)

    screen = report["point_screen"]
    conditions = {
        "c_presence_ci": ci_lower_pos(report["c_presence"]),
        "c_presence_screen": bool(report["c_presence"]["point"] >= screen),
        "v_zero_ci": ci_lower_pos(report["v_zero"]),
        "c_high_low_ci": ci_lower_pos(report["c_high_low"]),
        "c_high_low_screen": bool(report["c_high_low"]["point"] >= screen),
        "c_high_mid_ci": ci_lower_pos(report["c_high_mid"]),
        "c_high_mid_screen": bool(report["c_high_mid"]["point"] >= screen),
        "v_high_ci": ci_lower_pos(report["v_high"]),
        "robustness": bool(report["robustness"]["all_positive"]),
        "manifest_qa": bool(report["manifest_qa_pass"]),
    }
    return GATE_A_CONFIRMED if all(conditions.values()) else GATE_A_NOT_CONFIRMED


def judge_gate_b(report: Mapping[str, Any]) -> str:
    """Gate B is independent of Gate A; report its frozen status verbatim."""
    return str(report["gate_b"])
