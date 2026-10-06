"""Stage 5 fit/score contract for the published-method adapters (plan v0.3, Task 3).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``.

``fit_method`` trains one (method, grid-config, seed) on the TRAIN role only and
persists one state file; ``score_method`` reconstructs a pure scoring function
from that receipt and a legal bundle — the truth dict is never an input at
score time. Truth fields used at fit time: ``oracle_action`` (P1 TCP target
only) and ``regret`` (P2/P3/P4/P5 regression target). Scalers, reference sets
and conformal states are all fit on the train role and frozen inside the state.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ac_twin import stage5_methods as s5m
from ac_twin.stage1_data import sha256_file

DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 20
COVERAGE_TARGET = 0.80  # frozen main coverage (plan 4.2)


@dataclass(frozen=True)
class MethodReceipt:
    """Frozen handle to one fitted (method, config, seed) state."""
    method_id: str
    seed: int
    state_path: str
    state_sha256: str
    param_count: int
    notes: dict[str, Any] = field(default_factory=dict)


# --- feature tensors -----------------------------------------------------------------


def _token_scaler(tokens: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    sel = tokens[mask]  # (N, token_dim)
    mean = sel.mean(axis=0)
    std = sel.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _column_scaler(x: np.ndarray) -> dict[str, np.ndarray]:
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": x.mean(axis=0).astype(np.float32), "std": std.astype(np.float32)}


def _apply_scaler(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _encoder_tensors(bundle: Any, scaler: dict[str, dict[str, np.ndarray]],
                     device: str):
    tokens = np.asarray(bundle.tokens, dtype=np.float32)
    mask = np.asarray(bundle.token_mask, dtype=bool)
    tokens = _apply_scaler(tokens, scaler["tokens"])
    tokens = tokens * mask[:, :, None]
    globals_ = _apply_scaler(np.asarray(bundle.global_features, dtype=np.float32),
                             scaler["globals"])
    return (torch.from_numpy(tokens).to(device),
            torch.from_numpy(mask.astype(np.float32)).to(device),
            torch.from_numpy(globals_).to(device))


# --- generic deterministic training loop ----------------------------------------------


def _train(model: torch.nn.Module, loss_fn, n: int, device: str,
           max_epochs: int, patience: int, seed: int,
           batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
    gen = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = None
    stale = 0
    for _ in range(int(max_epochs)):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        model.train()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size].to(device)
            loss = loss_fn(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.numel()
        avg = total / max(n, 1)
        if avg < best_loss - 1e-9:
            best_loss = avg
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    model.load_state_dict(best_state)
    model.eval()


def _save_state(output_dir: Path, method_id: str, seed: int, payload: dict) -> tuple[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{method_id}_seed{seed}.pt"
    torch.save(payload, path)
    return str(path), sha256_file(path)


def _load_state(receipt: MethodReceipt) -> dict:
    path = Path(receipt.state_path)
    if sha256_file(path) != receipt.state_sha256:
        raise RuntimeError(f"state file hash mismatch for {receipt.method_id}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


# --- fit ---------------------------------------------------------------------------------


def fit_method(method_id: str, bundle: Any, truth: dict, config: dict, seed: int,
               output_dir, device: str = "cpu", max_epochs: int = DEFAULT_MAX_EPOCHS,
               patience: int = DEFAULT_PATIENCE) -> MethodReceipt:
    """Fit one (method, config, seed) on the train-role bundle + truth; persist state."""
    param_count = s5m.check_capacity(method_id, config)
    n = int(bundle.group_seed.size)
    regret = np.asarray(truth["regret"], dtype=np.float32)
    notes: dict[str, Any] = {}

    if method_id == "CONFIDNET_TCP_ADAPTED":
        x = s5m.p1_inputs(bundle, config)
        scaler = {"x": _column_scaler(x)}
        xs = torch.from_numpy(_apply_scaler(x, scaler["x"])).to(device)
        y = torch.from_numpy(
            s5m.tcp_target(bundle.j_hat, truth["oracle_action"],
                           float(config["temperature"])).astype(np.float32)).to(device)
        model = s5m.build_model(method_id, config, seed)
        mse = torch.nn.MSELoss()
        _train(model, lambda idx: mse(model(xs[idx]), y[idx]), n, device,
               max_epochs, patience, seed)
        payload = {"method_id": method_id, "config": dict(config), "seed": int(seed),
                   "model_state": model.state_dict(), "scaler": scaler}
        state_path, sha = _save_state(Path(output_dir), method_id, seed, payload)

    elif method_id == "SELECTIVENET_REGRET_ADAPTED":
        tokens = np.asarray(bundle.tokens, dtype=np.float32)
        mask = np.asarray(bundle.token_mask, dtype=bool)
        scaler = {"tokens": _token_scaler(tokens, mask),
                  "globals": _column_scaler(np.asarray(bundle.global_features, dtype=np.float32))}
        tt, tm, tg = _encoder_tensors(bundle, scaler, device)
        y = torch.from_numpy(regret).to(device)
        model = s5m.build_model(method_id, config, seed)

        def sn_loss(idx):
            f, g, h = model(tt[idx], tm[idx], tg[idx])
            total, _ = s5m.selectivenet_loss(f, g, h, y[idx],
                                             target_coverage=COVERAGE_TARGET)
            return total
        _train(model, sn_loss, n, device, max_epochs, patience, seed)
        notes["target_coverage"] = COVERAGE_TARGET
        payload = {"method_id": method_id, "config": dict(config), "seed": int(seed),
                   "model_state": model.state_dict(), "scaler": scaler}
        state_path, sha = _save_state(Path(output_dir), method_id, seed, payload)

    elif method_id == "RWR_NN_KNN_REJECT_ADAPTED":
        tokens = np.asarray(bundle.tokens, dtype=np.float32)
        mask = np.asarray(bundle.token_mask, dtype=bool)
        scaler = {"tokens": _token_scaler(tokens, mask),
                  "globals": _column_scaler(np.asarray(bundle.global_features, dtype=np.float32))}
        tt, tm, tg = _encoder_tensors(bundle, scaler, device)
        y = torch.from_numpy(regret).to(device)
        model = s5m.build_model(method_id, config, seed)
        mse = torch.nn.MSELoss()
        _train(model, lambda idx: mse(model(tt[idx], tm[idx], tg[idx]), y[idx]),
               n, device, max_epochs, patience, seed)
        # frozen in-sample train-residual reference (validation is reserved — registered deviation)
        with torch.no_grad():
            rep = model.representation(tt, tm, tg).cpu().numpy().astype(np.float64)
            pred = model(tt, tm, tg).cpu().numpy().astype(np.float64)
        residuals = (pred - regret.astype(np.float64)) ** 2
        rejector = s5m.KNNRejector(k=int(config["k"])).fit(rep, residuals)
        notes["reference_groups"] = n
        payload = {"method_id": method_id, "config": dict(config), "seed": int(seed),
                   "model_state": model.state_dict(), "scaler": scaler,
                   "rejector": rejector.state_dict()}
        state_path, sha = _save_state(Path(output_dir), method_id, seed, payload)

    elif method_id == "CONFORMAL_REGRESSION_REJECT_ADAPTED":
        fit_mask = s5m.conformal_fit_mask(bundle.group_seed, fit_fraction=0.8)
        n_fit = int(fit_mask.sum())
        n_cal = n - n_fit
        if n_fit == 0 or n_cal == 0:
            raise RuntimeError("degenerate conformal split")
        tokens = np.asarray(bundle.tokens, dtype=np.float32)
        mask = np.asarray(bundle.token_mask, dtype=bool)
        scaler = {"tokens": _token_scaler(tokens[fit_mask], mask[fit_mask]),
                  "globals": _column_scaler(
                      np.asarray(bundle.global_features, dtype=np.float32)[fit_mask])}
        tt, tm, tg = _encoder_tensors(bundle, scaler, device)
        y = torch.from_numpy(regret).to(device)
        fit_idx = torch.from_numpy(np.where(fit_mask)[0]).to(device)
        model = s5m.build_model(method_id, config, seed)
        mse = torch.nn.MSELoss()

        def p4_loss(batch_pos):
            idx = fit_idx[batch_pos]
            return mse(model(tt[idx], tm[idx], tg[idx]), y[idx])
        _train(model, p4_loss, n_fit, device, max_epochs, patience, seed)
        with torch.no_grad():
            cal_idx = torch.from_numpy(np.where(~fit_mask)[0]).to(device)
            pred_cal = model(tt[cal_idx], tm[cal_idx], tg[cal_idx]).cpu().numpy().astype(np.float64)
        res_cal = np.abs(pred_cal - regret[~fit_mask].astype(np.float64))
        difficulty = s5m.difficulty_from_bundle(bundle)
        mondrian = s5m.MondrianConformalState.fit(
            res_cal, difficulty[~fit_mask], n_bins=int(config["difficulty_bins"]), alpha=0.10)
        notes["fit_groups"] = n_fit
        notes["calibration_groups"] = n_cal
        notes["fit_calibration_disjoint"] = True
        notes["qhat_monotone"] = bool(np.all(np.diff(mondrian.qhat) >= 0.0))
        payload = {"method_id": method_id, "config": dict(config), "seed": int(seed),
                   "model_state": model.state_dict(), "scaler": scaler,
                   "mondrian": mondrian.state_dict()}
        state_path, sha = _save_state(Path(output_dir), method_id, seed, payload)

    elif method_id == "CCTR_DRIVING_CALIBRATION_ADAPTED":
        x = s5m.p5_inputs(bundle)
        scaler = {"x": _column_scaler(x)}
        xs = torch.from_numpy(_apply_scaler(x, scaler["x"])).to(device)
        y = torch.from_numpy(regret).to(device)
        model = s5m.build_model(method_id, config, seed)

        def nll_loss(idx):
            mu, sigma = model(xs[idx])
            return s5m.gaussian_nll(mu, sigma, y[idx])
        _train(model, nll_loss, n, device, max_epochs, patience, seed)
        payload = {"method_id": method_id, "config": dict(config), "seed": int(seed),
                   "model_state": model.state_dict(), "scaler": scaler}
        state_path, sha = _save_state(Path(output_dir), method_id, seed, payload)

    else:
        raise ValueError(f"unknown Stage 5 method: {method_id!r}")

    return MethodReceipt(method_id=method_id, seed=int(seed), state_path=state_path,
                         state_sha256=sha, param_count=param_count, notes=notes)


# --- score (truth-free) --------------------------------------------------------------------


def score_method(receipt: MethodReceipt, bundle: Any, device: str = "cpu") -> np.ndarray:
    """Pure scoring from a receipt + legal bundle; higher = riskier. No truth inputs."""
    state = _load_state(receipt)
    method_id = receipt.method_id
    config = state["config"]

    if method_id == "CONFIDNET_TCP_ADAPTED":
        x = _apply_scaler(s5m.p1_inputs(bundle, config), state["scaler"]["x"])
        model = s5m.build_model(method_id, config, receipt.seed)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        with torch.no_grad():
            conf = model(torch.from_numpy(x).to(device)).cpu().numpy()
        return (1.0 - conf).astype(np.float64)

    if method_id == "SELECTIVENET_REGRET_ADAPTED":
        tt, tm, tg = _encoder_tensors(bundle, state["scaler"], device)
        model = s5m.build_model(method_id, config, receipt.seed)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        with torch.no_grad():
            _, g, _ = model(tt, tm, tg)
        return (1.0 - g.cpu().numpy()).astype(np.float64)

    if method_id == "RWR_NN_KNN_REJECT_ADAPTED":
        tt, tm, tg = _encoder_tensors(bundle, state["scaler"], device)
        model = s5m.build_model(method_id, config, receipt.seed)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        with torch.no_grad():
            rep = model.representation(tt, tm, tg).cpu().numpy().astype(np.float64)
        rejector = s5m.KNNRejector.from_state_dict(state["rejector"])
        return rejector.score(rep).astype(np.float64)

    if method_id == "CONFORMAL_REGRESSION_REJECT_ADAPTED":
        mondrian = s5m.MondrianConformalState.from_state_dict(state["mondrian"])
        return mondrian.score(s5m.difficulty_from_bundle(bundle)).astype(np.float64)

    if method_id == "CCTR_DRIVING_CALIBRATION_ADAPTED":
        x = _apply_scaler(s5m.p5_inputs(bundle), state["scaler"]["x"])
        model = s5m.build_model(method_id, config, receipt.seed)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        with torch.no_grad():
            _, sigma = model(torch.from_numpy(x).to(device))
        return sigma.cpu().numpy().astype(np.float64)

    raise ValueError(f"unknown Stage 5 method: {method_id!r}")


def raw_outputs_method(receipt: MethodReceipt, bundle: Any,
                       device: str = "cpu") -> dict | None:
    """Method-native probability/interval outputs for the calibration report.

    P1: ``tcp`` (predicted true-class probability). P4: ``mu`` + ``half_width``.
    P5: ``mu`` + ``sigma``. P2/P3 have no native probability/interval -> None
    (the integrity report marks them ``NOT_APPLICABLE``).
    """
    state = _load_state(receipt)
    method_id = receipt.method_id
    config = state["config"]
    if method_id == "CONFIDNET_TCP_ADAPTED":
        x = _apply_scaler(s5m.p1_inputs(bundle, config), state["scaler"]["x"])
        model = s5m.build_model(method_id, config, receipt.seed)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        with torch.no_grad():
            tcp = model(torch.from_numpy(x).to(device)).cpu().numpy()
        return {"tcp": tcp.astype(np.float64)}
    if method_id == "CONFORMAL_REGRESSION_REJECT_ADAPTED":
        tt, tm, tg = _encoder_tensors(bundle, state["scaler"], device)
        model = s5m.build_model(method_id, config, receipt.seed)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        with torch.no_grad():
            mu = model(tt, tm, tg).cpu().numpy().astype(np.float64)
        mondrian = s5m.MondrianConformalState.from_state_dict(state["mondrian"])
        half = mondrian.score(s5m.difficulty_from_bundle(bundle)) / 2.0
        return {"mu": mu, "half_width": half.astype(np.float64)}
    if method_id == "CCTR_DRIVING_CALIBRATION_ADAPTED":
        x = _apply_scaler(s5m.p5_inputs(bundle), state["scaler"]["x"])
        model = s5m.build_model(method_id, config, receipt.seed)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        with torch.no_grad():
            mu, sigma = model(torch.from_numpy(x).to(device))
        return {"mu": mu.cpu().numpy().astype(np.float64),
                "sigma": sigma.cpu().numpy().astype(np.float64)}
    return None


__all__ = ["MethodReceipt", "fit_method", "score_method", "raw_outputs_method",
           "COVERAGE_TARGET"]
