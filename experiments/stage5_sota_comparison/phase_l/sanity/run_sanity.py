"""Stage 5 Phase L: toy/synthetic formula sanity checks for P1-P5 candidate methods.

AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3, section 3.4/5.1.

Each check builds a small synthetic problem, implements the PUBLISHED core
formula/loss in its minimal form (no AC-Twin data, no truth access at score
time beyond what the paper allows), and asserts pre-registered direction /
numeric properties. Results are written to sanity_report.json and folded
into literature/compatibility_decision.json by Task 1.

These checks are engineering verification only (ENGINEERING_SANITY_ONLY):
they prove the formula as implemented behaves as the paper claims on toy
data. They produce NO scientific result about AC-Twin.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT_DIR = Path(__file__).resolve().parent
DEVICE = "cpu"
TORCH_SEED = 2026092711


def _set_seed(seed: int = TORCH_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _train(net: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn, epochs: int,
           lr: float = 1e-3, batch: int = 256, tag: str = "") -> None:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = x.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(net, x[idx], y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    [{tag}] epoch {ep + 1}/{epochs} loss={tot / n:.4f}", flush=True)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r)


def _auc_rank(risk: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC of `risk` (higher=riskier) for binary `label` (1=positive/failure)."""
    order = np.argsort(risk, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(risk) + 1)
    pos = label == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------- P1 ConfidNet
def sanity_confidnet_tcp() -> dict:
    """Corbiere et al. 2019: confidence model predicts TCP = p_model(true class);
    risk = 1 - predicted_TCP must rank failures above successes."""
    print("[P1] ConfidNet TCP toy ...", flush=True)
    _set_seed()
    from sklearn.datasets import make_moons
    n_tr, n_te = 5000, 1500
    x_tr, y_tr = make_moons(n_samples=n_tr, noise=0.28, random_state=7)
    x_te, y_te = make_moons(n_samples=n_te, noise=0.28, random_state=8)
    # label noise -> genuine failures the base model cannot fix
    rng = np.random.RandomState(9)
    flip_tr = rng.rand(n_tr) < 0.08
    y_tr = y_tr.copy(); y_tr[flip_tr] ^= 1
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    x_te_t = torch.tensor(x_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    clf = _mlp(2, 64, 2)
    ce = nn.CrossEntropyLoss()
    _train(clf, x_tr_t, y_tr_t, lambda m, xb, yb: ce(m(xb), yb), epochs=60, tag="P1-base")
    for p in clf.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        probs_tr = torch.softmax(clf(x_tr_t), dim=1)
        probs_te = torch.softmax(clf(x_te_t), dim=1)
        pred_te = probs_te.argmax(1)
    tcp_tr = probs_tr.gather(1, y_tr_t.unsqueeze(1)).squeeze(1)  # train-time oracle class prob

    conf = _mlp(4, 64, 1)  # input: x + logits (legal, pre-truth at inference)
    feats_tr = torch.cat([x_tr_t, torch.log(probs_tr + 1e-9)], dim=1)
    feats_te = torch.cat([x_te_t, torch.log(probs_te + 1e-9)], dim=1)
    _train(conf, feats_tr, tcp_tr, lambda m, xb, yb: ((m(xb).squeeze(1) - yb) ** 2).mean(),
           epochs=80, tag="P1-conf")
    with torch.no_grad():
        tcp_hat_te = conf(feats_te).squeeze(1).numpy()
    risk = 1.0 - tcp_hat_te
    err = (pred_te != y_te_t).numpy().astype(int)
    auc = _auc_rank(risk, err)
    sp = _spearman(risk, err)
    ok = auc >= 0.70 and sp > 0.20
    print(f"    [P1] error-detection AUC={auc:.3f} (>=0.70), Spearman(risk,error)={sp:.3f} (>0.20) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"method_id": "CONFIDNET_TCP_ADAPTED", "family": "learned/driving calibrated confidence",
            "auc_error_detection": auc, "spearman_risk_error": sp,
            "criteria": {"auc_error_detection_gte": 0.70, "spearman_gt": 0.20}, "pass": bool(ok)}


# -------------------------------------------------------------- P2 SelectiveNet
def sanity_selectivenet_regret() -> dict:
    """Geifman & El-Yaniv 2019 (regression adaptation): three heads (prediction f,
    selection g, auxiliary h); loss = selective_risk/coverage + lambda*(coverage-mean g)^2
    + aux; empirical coverage near target, selective risk < full risk."""
    print("[P2] SelectiveNet regret toy ...", flush=True)
    _set_seed()
    rng = np.random.RandomState(11)
    n_tr, n_te = 6000, 2000
    x_tr = rng.uniform(-1, 1, size=(n_tr, 2)).astype(np.float32)
    x_te = rng.uniform(-1, 1, size=(n_te, 2)).astype(np.float32)

    def gen(x):
        base = np.sin(3 * x[:, 0]) + x[:, 1]
        sigma = 0.05 + 0.45 * (x[:, 0] > 0)  # heteroscedastic difficulty
        return base, sigma
    b_tr, s_tr = gen(x_tr); b_te, s_te = gen(x_te)
    y_tr = (b_tr + s_tr * rng.randn(n_tr)).astype(np.float32)
    y_te = (b_te + s_te * rng.randn(n_te)).astype(np.float32)

    x_tr_t = torch.tensor(x_tr); y_tr_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te); y_te_t = torch.tensor(y_te)

    body = _mlp(2, 64, 64, depth=2)
    head_f = nn.Linear(64, 1)
    head_g = nn.Linear(64, 1)
    head_h = nn.Linear(64, 1)
    params = list(body.parameters()) + list(head_f.parameters()) + list(head_g.parameters()) + list(head_h.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    target_c, lam = 0.80, 32.0
    epochs = 120
    for ep in range(epochs):
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, 256):
            idx = perm[i:i + 256]
            z = body(x_tr_t[idx])
            f, g, h = head_f(z).squeeze(1), torch.sigmoid(head_g(z)).squeeze(1), head_h(z).squeeze(1)
            yb = y_tr_t[idx]
            sel_risk = (g * (f - yb) ** 2).mean() / (g.mean() + 1e-8)
            cov_pen = lam * (target_c - g.mean()) ** 2
            aux = ((h - yb) ** 2).mean()
            loss = sel_risk + cov_pen + 0.5 * aux
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 24 == 0:
            print(f"    [P2] epoch {ep + 1}/{epochs} loss={float(loss):.4f}", flush=True)

    with torch.no_grad():
        z = body(x_te_t)
        f_te = head_f(z).squeeze(1).numpy()
        g_te = torch.sigmoid(head_g(z)).squeeze(1).numpy()
    thr = np.quantile(g_te, 1 - target_c)
    covered = g_te >= thr
    coverage = float(covered.mean())
    se = (f_te - y_te) ** 2
    full_mse = float(se.mean())
    sel_mse = float(se[covered].mean())
    hi_noise = x_te[:, 0] > 0
    g_hi, g_lo = float(g_te[hi_noise].mean()), float(g_te[~hi_noise].mean())
    risk = 1.0 - g_te
    sp = _spearman(risk, np.abs(f_te - y_te))
    ok = (0.70 <= coverage <= 0.90) and (sel_mse < full_mse) and (g_hi < g_lo) and (sp > 0.20)
    print(f"    [P2] coverage={coverage:.3f} in [0.70,0.90], sel_mse={sel_mse:.4f}<full={full_mse:.4f}, "
          f"g_hi={g_hi:.3f}<g_lo={g_lo:.3f}, Spearman={sp:.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"method_id": "SELECTIVENET_REGRET_ADAPTED", "family": "joint selective/rejection",
            "coverage": coverage, "selective_mse": sel_mse, "full_mse": full_mse,
            "g_high_noise": g_hi, "g_low_noise": g_lo, "spearman_risk_abserr": sp,
            "criteria": {"coverage_in": [0.70, 0.90], "selective_lt_full": True,
                         "g_high_lt_g_low": True, "spearman_gt": 0.20}, "pass": bool(ok)}


# ------------------------------------------------------------------- P3 RWR kNN
def sanity_rwr_nn_knn() -> dict:
    """Li et al. 2024: regressor trained on ALL data (no-rejection learning);
    rejection score = local residual level of k nearest neighbours in the frozen
    learned representation; score must rank true errors."""
    print("[P3] RWR NN+kNN rejection toy ...", flush=True)
    _set_seed()
    rng = np.random.RandomState(13)
    n_tr, n_te = 6000, 2000
    x_tr = rng.uniform(-1, 1, size=(n_tr, 2)).astype(np.float32)
    x_te = rng.uniform(-1, 1, size=(n_te, 2)).astype(np.float32)

    def gen(x):
        base = np.sin(3 * x[:, 0]) + x[:, 1]
        sigma = 0.05 + 0.45 * (x[:, 0] > 0)
        return base, sigma
    b_tr, s_tr = gen(x_tr); b_te, s_te = gen(x_te)
    y_tr = (b_tr + s_tr * rng.randn(n_tr)).astype(np.float32)
    y_te = (b_te + s_te * rng.randn(n_te)).astype(np.float32)
    x_tr_t = torch.tensor(x_tr); y_tr_t = torch.tensor(y_tr)
    x_te_t = torch.tensor(x_te); y_te_t = torch.tensor(y_te)

    reg = _mlp(2, 64, 1, depth=2)
    _train(reg, x_tr_t, y_tr_t, lambda m, xb, yb: ((m(xb).squeeze(1) - yb) ** 2).mean(),
           epochs=80, tag="P3-reg")
    for p in reg.parameters():
        p.requires_grad_(False)
    feat = nn.Sequential(*list(reg.children())[:-1])  # penultimate representation
    with torch.no_grad():
        z_tr = feat(x_tr_t)
        z_te = feat(x_te_t)
        res_tr = (reg(x_tr_t).squeeze(1) - y_tr_t).abs()
        pred_te = reg(x_te_t).squeeze(1).numpy()
    k = 32
    # kNN residual score in representation space
    d = torch.cdist(z_te, z_tr)
    nn_idx = d.topk(k, largest=False).indices
    score = res_tr[nn_idx].mean(dim=1).numpy()
    abs_err = np.abs(pred_te - y_te)
    sp = _spearman(score, abs_err)
    thr = np.quantile(score, 1 - 0.80)
    kept = score <= thr
    full_mse = float(abs_err.mean() ** 2) if False else float((abs_err ** 2).mean())
    kept_mse = float((abs_err[kept] ** 2).mean())
    ok = (sp > 0.30) and (kept_mse <= 0.90 * full_mse)
    print(f"    [P3] Spearman(kNN score,|err|)={sp:.3f} (>0.30), kept80 MSE={kept_mse:.4f} vs full={full_mse:.4f} "
          f"(ratio {kept_mse / full_mse:.3f} <= 0.90) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"method_id": "RWR_NN_KNN_REJECT_ADAPTED", "family": "joint selective/rejection (technical alternate)",
            "k": k, "spearman_score_abserr": sp, "kept80_mse": kept_mse, "full_mse": full_mse,
            "criteria": {"spearman_gt": 0.30, "kept80_mse_ratio_lte": 0.90}, "pass": bool(ok)}


# ------------------------------------------------------- P4 conformal + reject
def sanity_conformal_reject() -> dict:
    """Johansson et al. 2024 style: regressor on fit split; Mondrian split-conformal
    calibration on a DISJOINT calibration split; risk = per-bin interval width
    (frozen difficulty ordering); marginal coverage ~1-alpha; width ranks error."""
    print("[P4] Conformal regression with reject option toy ...", flush=True)
    _set_seed()
    rng = np.random.RandomState(17)
    n_all, n_te = 8000, 2000
    x_all = rng.uniform(-1, 1, size=(n_all, 2)).astype(np.float32)
    x_te = rng.uniform(-1, 1, size=(n_te, 2)).astype(np.float32)

    def gen(x):
        base = np.sin(3 * x[:, 0]) + x[:, 1]
        sigma = 0.05 + 0.45 * (x[:, 0] > 0)
        return base, sigma
    b_all, s_all = gen(x_all); b_te, s_te = gen(x_te)
    y_all = (b_all + s_all * rng.randn(n_all)).astype(np.float32)
    y_te = (b_te + s_te * rng.randn(n_te)).astype(np.float32)

    # deterministic 80/20 fit/calibration split by index hash
    h = (np.arange(n_all) * 2654435761) % (2 ** 31)
    is_fit = (h % 10) < 8
    fit_idx, cal_idx = np.where(is_fit)[0], np.where(~is_fit)[0]
    assert len(set(fit_idx.tolist()) & set(cal_idx.tolist())) == 0

    x_fit_t = torch.tensor(x_all[fit_idx]); y_fit_t = torch.tensor(y_all[fit_idx])
    x_cal = torch.tensor(x_all[cal_idx])
    x_te_t = torch.tensor(x_te)
    reg = _mlp(2, 64, 1, depth=2)
    _train(reg, x_fit_t, y_fit_t, lambda m, xb, yb: ((m(xb).squeeze(1) - yb) ** 2).mean(),
           epochs=80, tag="P4-reg")
    with torch.no_grad():
        cal_pred = reg(x_cal).squeeze(1).numpy()
        te_pred = reg(x_te_t).squeeze(1).numpy()
    cal_nc = np.abs(cal_pred - y_all[cal_idx])
    alpha = 0.10
    # Mondrian bins by difficulty region (sigma structure), frozen rule: x0>0
    cal_bin = (x_all[cal_idx, 0] > 0).astype(int)
    te_bin = (x_te[:, 0] > 0).astype(int)
    qhat = {}
    for b in (0, 1):
        nc_b = np.sort(cal_nc[cal_bin == b])
        q_level = min(1.0, math.ceil((len(nc_b) + 1) * (1 - alpha)) / len(nc_b))
        qhat[b] = float(np.quantile(nc_b, q_level, method="higher"))
    width = np.array([qhat[b] for b in te_bin]) * 2.0
    covered = np.abs(te_pred - y_te) <= width / 2.0
    coverage = float(covered.mean())
    abs_err = np.abs(te_pred - y_te)
    bin_err = [float(abs_err[te_bin == b].mean()) for b in (0, 1)]
    bin_width = [float(width[te_bin == b].mean()) for b in (0, 1)]
    risk = width  # frozen interval-width ordering
    thr = np.quantile(risk, 1 - 0.80)
    kept = risk <= thr
    kept_mse = float((abs_err[kept] ** 2).mean()); full_mse = float((abs_err ** 2).mean())
    sp = _spearman(risk, abs_err)
    ok = (0.85 <= coverage <= 0.95) and (bin_width[1] > bin_width[0]) and (bin_err[1] > bin_err[0]) \
        and (kept_mse < full_mse) and (sp > 0.20)
    print(f"    [P4] coverage={coverage:.3f} in [0.85,0.95], widths {bin_width} match error order {bin_err}, "
          f"kept80 MSE={kept_mse:.4f}<full={full_mse:.4f}, Spearman={sp:.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"method_id": "CONFORMAL_REGRESSION_REJECT_ADAPTED", "family": "conformal/risk-control rejection",
            "alpha": alpha, "marginal_coverage": coverage, "bin_widths": bin_width, "bin_mean_abs_err": bin_err,
            "kept80_mse": kept_mse, "full_mse": full_mse, "spearman_width_abserr": sp,
            "fit_cal_disjoint": True,
            "criteria": {"coverage_in": [0.85, 0.95], "width_order_matches_error": True,
                         "kept80_lt_full": True, "spearman_gt": 0.20}, "pass": bool(ok)}


# -------------------------------------------------------------------- P5 CCTR
def sanity_cctr_temperature() -> dict:
    """Cao et al. 2024 (CCTR) style: ensemble disagreement + context -> sample-specific
    temperature for calibrated uncertainty; calibrated predictive NLL improves and
    the uncertainty ranks errors (usable as group risk)."""
    print("[P5] CCTR sample-specific temperature toy ...", flush=True)
    _set_seed()
    rng = np.random.RandomState(19)
    n_tr, n_cal, n_te = 5000, 1500, 2000
    x_tr = rng.uniform(-1, 1, size=(n_tr, 2)).astype(np.float32)
    x_cal = rng.uniform(-1, 1, size=(n_cal, 2)).astype(np.float32)
    x_te = rng.uniform(-1, 1, size=(n_te, 2)).astype(np.float32)

    def gen(x):
        base = np.sin(3 * x[:, 0]) + x[:, 1]
        sigma = 0.05 + 0.45 * (x[:, 0] > 0)
        return base, sigma
    b_tr, s_tr = gen(x_tr); b_cal, s_cal = gen(x_cal); b_te, s_te = gen(x_te)
    y_tr = (b_tr + s_tr * rng.randn(n_tr)).astype(np.float32)
    y_cal = (b_cal + s_cal * rng.randn(n_cal)).astype(np.float32)
    y_te = (b_te + s_te * rng.randn(n_te)).astype(np.float32)

    preds_tr, preds_cal, preds_te = [], [], []
    for sd in (8101, 8202, 8303, 8404, 8505):
        torch.manual_seed(sd)
        # bootstrap resample per member: members fit different noise realisations,
        # so ensemble disagreement tracks the heteroscedastic difficulty sigma(x)
        boot = np.random.RandomState(sd).randint(0, n_tr, size=n_tr)
        xt = torch.tensor(x_tr[boot]); yt = torch.tensor(y_tr[boot])
        net = _mlp(2, 64, 1, depth=2)
        _train(net, xt, yt, lambda m, xb, yb: ((m(xb).squeeze(1) - yb) ** 2).mean(),
               epochs=60, tag=f"P5-ens{sd}")
        with torch.no_grad():
            preds_tr.append(net(torch.tensor(x_tr)).squeeze(1).numpy())
            preds_cal.append(net(torch.tensor(x_cal)).squeeze(1).numpy())
            preds_te.append(net(torch.tensor(x_te)).squeeze(1).numpy())
    p_cal = np.stack(preds_cal); p_te = np.stack(preds_te)
    mean_cal, mean_te = p_cal.mean(0), p_te.mean(0)
    dis_cal, dis_te = p_cal.std(0), p_te.std(0)
    # sample-specific temperature: sigma(x) = exp(a * disagreement + b), with
    # (a, b) fitted by minimizing the actual Gaussian NLL on the calibration
    # split; temperature clipped to [T_min, T_max] as in the CCTR recipe
    from scipy.optimize import minimize
    t_min, t_max = math.log(1e-2), math.log(2.0)
    err_cal = np.abs(y_cal - mean_cal)

    def nll_obj(ab):
        ls = np.clip(ab[0] * dis_cal + ab[1], t_min, t_max)
        sig = np.exp(ls)
        return float((ls + math.log(math.sqrt(2 * math.pi)) + err_cal ** 2 / (2 * sig ** 2)).mean())

    b0 = math.log(float(np.std(y_cal - mean_cal)) + 1e-9)
    opt_res = minimize(nll_obj, x0=np.array([0.0, b0]), method="Nelder-Mead",
                       options={"maxiter": 500})
    coef = opt_res.x
    sig_te = np.exp(np.clip(coef[0] * dis_te + coef[1], t_min, t_max))
    const_sig = np.full_like(sig_te, np.std(y_cal - mean_cal))

    def nll(y, mu, sig):
        return float((0.5 * np.log(2 * np.pi * sig ** 2) + (y - mu) ** 2 / (2 * sig ** 2)).mean())
    nll_cal_model, nll_const = nll(y_te, mean_te, sig_te), nll(y_te, mean_te, const_sig)
    abs_err = np.abs(mean_te - y_te)
    sp = _spearman(sig_te, abs_err)
    risk = sig_te
    thr = np.quantile(risk, 1 - 0.80)
    kept = risk <= thr
    kept_mse, full_mse = float((abs_err[kept] ** 2).mean()), float((abs_err ** 2).mean())
    ok = (nll_cal_model < nll_const) and (sp > 0.30) and (kept_mse < full_mse) and bool(np.all(np.isfinite(sig_te))) and bool(np.all(sig_te > 0))
    print(f"    [P5] calibrated NLL={nll_cal_model:.4f} < const={nll_const:.4f}, Spearman={sp:.3f} (>0.30), "
          f"kept80 MSE={kept_mse:.4f}<full={full_mse:.4f}, sigma finite&positive -> {'PASS' if ok else 'FAIL'}", flush=True)
    return {"method_id": "CCTR_DRIVING_CALIBRATION_ADAPTED", "family": "learned/driving calibrated confidence (technical alternate)",
            "nll_sample_temperature": nll_cal_model, "nll_constant": nll_const,
            "spearman_sigma_abserr": sp, "kept80_mse": kept_mse, "full_mse": full_mse,
            "sigma_finite_positive": bool(np.all(np.isfinite(sig_te)) and np.all(sig_te > 0)),
            "criteria": {"nll_lt_constant": True, "spearman_gt": 0.30, "kept80_lt_full": True,
                         "sigma_finite_positive": True}, "pass": bool(ok)}


def main() -> int:
    t0 = time.time()
    print("Stage 5 Phase L sanity runner (ENGINEERING_SANITY_ONLY)", flush=True)
    checks = [sanity_confidnet_tcp, sanity_selectivenet_regret, sanity_rwr_nn_knn,
              sanity_conformal_reject, sanity_cctr_temperature]
    results = []
    for i, fn in enumerate(checks, 1):
        print(f"[{i}/{len(checks)}] running {fn.__name__} ...", flush=True)
        results.append(fn())
        print(f"    done ({time.time() - t0:.0f}s elapsed)", flush=True)
    report = {
        "report_type": "STAGE5_PHASE_L_SANITY",
        "status": "ENGINEERING_SANITY_ONLY",
        "torch_seed": TORCH_SEED,
        "n_checks": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "all_pass": all(r["pass"] for r in results),
        "families_verified": sorted({r["family"] for r in results if r["pass"]}),
        "results": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out = OUT_DIR / "sanity_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SUMMARY: {report['n_pass']}/{report['n_checks']} checks passed; families verified: "
          f"{report['families_verified']}; wrote {out}", flush=True)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
