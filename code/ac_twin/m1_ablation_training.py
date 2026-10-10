"""M1 supervised-baseline ablation: three-arm matched training (plan sections 4/5/12).

Every arm trains the SAME frozen Direct architecture (315 -> 43 -> 43 -> 2, exactly
15,740 parameters) with the SAME mechanics as the frozen Stage 4C-R2 loop
(``stage4cr2_training._fit_direct_seed``): identical seeding, batch generator, loss,
early-stop tolerance and best-state restore. The loop is re-implemented here (never by
calling the R2 entry point, which is bound to the old plan and write paths) and is
verified against the original on synthetic data in ``tests/test_m1_ablation.py``.

Resume contract (plan 12): a completed fit is verified read-only (checkpoint hash
against the fit ledger) and skipped; an interrupted fit leaves no usable training
state, so that (arm, seed) is explicitly RESTARTED and the restart is logged — a
completed seed is never replaced and results are never picked.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from ac_twin import m1_ablation_protocol as m1p
from ac_twin import stage4cp_features as s4cpf
from ac_twin import stage4cp_models as s4cm
from ac_twin import stage4cp_protocol as s4cp
from ac_twin.stage1_data import sha256_file
from ac_twin.stage3r_protocol import write_once_json

FIT_LEDGER_NAME = "fit_ledger.json"
MODEL_INVENTORY_NAME = "model_inventory.json"


def count_parameters() -> int:
    model = s4cm.build_model(
        s4cp.ARCH_DIRECT, hidden_dim=m1p.DIRECT_HIDDEN_WIDTH,
        dropout=m1p.DROPOUT, direct_width=m1p.DIRECT_HIDDEN_WIDTH)
    count = s4cm.count_parameters(model)
    if count != m1p.EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"Direct trainable parameters {count} != "
                           f"{m1p.EXPECTED_TRAINABLE_PARAMETERS}")
    return count


def fit_arm_seed(arm: str,
                 seed: int,
                 x_main: np.ndarray,
                 y_reg_main: np.ndarray,
                 y_high_main: np.ndarray,
                 x_monitor: np.ndarray,
                 y_reg_monitor: np.ndarray,
                 y_high_monitor: np.ndarray,
                 *,
                 pos_weight: float,
                 device: str,
                 progress: Callable[[str], None] | None = None,
                 ) -> dict[str, Any]:
    """Train ONE (arm, seed) on the main rows; early-stop on the fixed monitor rows.

    Mirrors the frozen R2 loop exactly: ``torch.manual_seed(seed)``,
    ``np.random.seed(seed % (2**31 - 1))``, a ``torch.Generator`` batch order seeded by
    the model seed, AdamW(lr=1e-3, wd=1e-4), SmoothL1(mu_pos, y_reg) + lambda_cls *
    weighted BCE(logit_high, y_high), monitor total-loss early stopping with tolerance
    1e-10 and best-state restore. ``x_*`` are the arm's 315-dim flat inputs (already
    standardised/masked per the arm definition).
    """
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))

    model = s4cm.build_model(
        s4cp.ARCH_DIRECT, hidden_dim=m1p.DIRECT_HIDDEN_WIDTH,
        dropout=m1p.DROPOUT, direct_width=m1p.DIRECT_HIDDEN_WIDTH).to(dev)

    x_m = torch.as_tensor(np.asarray(x_main, dtype=np.float64), dtype=torch.float32)
    y_mr = torch.as_tensor(np.asarray(y_reg_main, dtype=np.float64), dtype=torch.float32)
    y_mh = torch.as_tensor(np.asarray(y_high_main, dtype=np.float64), dtype=torch.float32)
    x_v = torch.as_tensor(np.asarray(x_monitor, dtype=np.float64), dtype=torch.float32).to(dev)
    y_vr = torch.as_tensor(np.asarray(y_reg_monitor, dtype=np.float64), dtype=torch.float32).to(dev)
    y_vh = torch.as_tensor(np.asarray(y_high_monitor, dtype=np.float64), dtype=torch.float32).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=m1p.LEARNING_RATE,
                                  weight_decay=m1p.WEIGHT_DECAY)
    smooth_l1 = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(
        float(pos_weight), dtype=torch.float32, device=dev))

    def forward(flat_tokens):
        split = flat_tokens.shape[1] - s4cpf.GLOBAL_DIM
        return model(flat_tokens[:, :split], flat_tokens[:, split:])

    def total_loss(x, y_r, y_h) -> float:
        model.eval()
        with torch.no_grad():
            raw_mu, logit = forward(x)
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_r) + m1p.LAMBDA_CLS * bce(logit, y_h)
        return float(loss.item())

    n_main = x_m.shape[0]
    best_state = None
    best_val = float("inf")
    wait = 0
    started = time.perf_counter()
    batch_generator = torch.Generator().manual_seed(int(seed))
    epochs_run = 0
    for epoch in range(m1p.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_main, generator=batch_generator).numpy()
        for start in range(0, n_main, m1p.TRAIN_BATCH_SIZE):
            sel = order[start:start + m1p.TRAIN_BATCH_SIZE]
            optimizer.zero_grad()
            raw_mu, logit = forward(x_m[sel].to(dev))
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_mr[sel].to(dev)) \
                + m1p.LAMBDA_CLS * bce(logit, y_mh[sel].to(dev))
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1
        val_loss = total_loss(x_v, y_vr, y_vh)
        if val_loss < best_val - m1p.EARLY_STOP_TOLERANCE:
            best_val = val_loss
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if progress is not None:
            progress(f"arm={arm} seed={seed} epoch={epochs_run}/{m1p.MAX_EPOCHS} "
                     f"monitor_loss={val_loss:.6f} best={best_val:.6f} "
                     f"elapsed={time.perf_counter() - started:.1f}s")
        if wait >= m1p.EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    train_loss = total_loss(x_m.to(dev), y_mr.to(dev), y_mh.to(dev))
    return {
        "arch": s4cp.ARCH_DIRECT,
        "arm": str(arm),
        "hidden_dim": m1p.DIRECT_HIDDEN_WIDTH,
        "direct_width": m1p.DIRECT_HIDDEN_WIDTH,
        "lambda_cls": m1p.LAMBDA_CLS,
        "seed": int(seed),
        "pos_weight": float(pos_weight),
        "dropout": m1p.DROPOUT,
        "token_dim": s4cpf.TOKEN_DIM,
        "global_dim": s4cpf.GLOBAL_DIM,
        "n_competitors": s4cp.N_COMPETITORS,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "train_loss": float(train_loss),
        "best_internal_monitor_loss": float(best_val),
        "epochs_run": int(epochs_run),
        "n_train_groups": int(n_main),
    }


def score_arm_checkpoints(checkpoints: Mapping[int, Mapping[str, Any]],
                          flat: np.ndarray,
                          *,
                          device: str) -> dict[str, np.ndarray]:
    """Score groups with every seed checkpoint -> per-seed scores + arithmetic mean.

    The ensemble is exactly ``risk_by_seed.mean(axis=0)``; never median, never
    seed-picked. Scoring reuses the frozen ``stage4cp_models.predict_decision_risk``
    (the flat input is split back into tokens + globals first).
    """
    from ac_twin import m1_ablation_features as m1f
    flat = np.asarray(flat, dtype=np.float64)
    tokens, global_features = m1f.unflatten(flat)
    token_mask = np.ones(tokens.shape[:2], dtype=bool)
    seeds = sorted(int(s) for s in checkpoints.keys())
    per_seed = np.empty((len(seeds), flat.shape[0]), dtype=np.float64)
    for i, seed in enumerate(seeds):
        scores = s4cm.predict_decision_risk(
            checkpoints[seed], tokens, global_features, token_mask, device=device)
        risk = np.asarray(scores["risk_score"], dtype=np.float64)
        if risk.shape != (flat.shape[0],) or not np.isfinite(risk).all():
            raise RuntimeError(f"non-finite risk_score from seed {seed}")
        per_seed[i] = risk
    return {
        "risk_by_seed": per_seed,
        "risk_ensemble": per_seed.mean(axis=0),
        "seeds": np.asarray(seeds, dtype=np.int64),
    }


# --- orchestration with resume ----------------------------------------------------------------


def _checkpoint_path(model_root: Path, arm: str, seed: int) -> Path:
    return model_root / arm / f"seed{int(seed)}.pt"


def _load_ledger(model_root: Path) -> dict[str, Any]:
    path = model_root / FIT_LEDGER_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": {}, "restarts": []}


def _write_ledger(model_root: Path, ledger: Mapping[str, Any]) -> None:
    path = model_root / FIT_LEDGER_NAME
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(ledger), indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def fit_all_arms(arm_inputs: Mapping[str, np.ndarray],
                 *,
                 main_idx: np.ndarray,
                 monitor_idx: np.ndarray,
                 y_reg: np.ndarray,
                 y_high: np.ndarray,
                 pos_weight: float,
                 model_root: Path,
                 device: str,
                 progress: Callable[[str], None] | None = None,
                 ) -> dict[str, Any]:
    """Run the frozen 15-fit matrix (seed outer loop, arm order FULL->BASELINE3->NO_RISK).

    Completed fits are verified read-only and skipped; a missing/partial checkpoint is
    retrained as an explicit restart of that (arm, seed) and logged. After all fifteen
    checkpoints exist, the create-once ``model_inventory.json`` is written.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    count_parameters()
    model_root = Path(model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(model_root)
    y_reg = np.asarray(y_reg, dtype=np.float64).ravel()
    y_high = np.asarray(y_high, dtype=np.float64).ravel()
    main_idx = np.asarray(main_idx, dtype=np.int64)
    monitor_idx = np.asarray(monitor_idx, dtype=np.int64)

    inventory_path = model_root / MODEL_INVENTORY_NAME
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for entry in inventory["checkpoints"].values():
            path = Path(entry["path"])
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"M1 checkpoint drifted from its inventory: {path}")
        return inventory

    fits_done = 0
    total_fits = len(m1p.MODEL_SEEDS) * len(m1p.ARMS)
    for seed in m1p.MODEL_SEEDS:
        for arm in m1p.ARMS:
            key = f"{arm}:{int(seed)}"
            target = _checkpoint_path(model_root, arm, seed)
            recorded = ledger["completed"].get(key)
            if recorded is not None and target.is_file() \
                    and sha256_file(target) == recorded["sha256"]:
                fits_done += 1
                if progress is not None:
                    progress(f"[fit {fits_done}/{total_fits}] {key} already complete "
                             "(verified read-only); skipped")
                continue
            if recorded is None and target.exists():
                raise FileExistsError(
                    f"checkpoint without ledger entry: {target}; remove the incomplete "
                    "arm directory to re-run")
            restart = recorded is not None
            if restart:
                ledger["restarts"].append(
                    {"fit": key, "reason": "checkpoint missing or hash drift; explicit "
                     "restart of the incomplete seed", "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            x = np.asarray(arm_inputs[arm], dtype=np.float64)
            checkpoint = fit_arm_seed(
                arm, int(seed), x[main_idx], y_reg[main_idx], y_high[main_idx],
                x[monitor_idx], y_reg[monitor_idx], y_high[monitor_idx],
                pos_weight=pos_weight, device=device, progress=progress)
            if int(checkpoint["seed"]) != int(seed) or str(checkpoint["arm"]) != arm:
                raise RuntimeError(f"trainer returned the wrong fit: {key}")
            for name in ("train_loss", "best_internal_monitor_loss"):
                if not np.isfinite(float(checkpoint[name])):
                    raise RuntimeError(f"non-finite {name} for {key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            s4cm.save_checkpoint(checkpoint, target)
            ledger["completed"][key] = {
                "path": str(target), "sha256": sha256_file(target),
                "train_loss": float(checkpoint["train_loss"]),
                "best_internal_monitor_loss": float(checkpoint["best_internal_monitor_loss"]),
                "epochs_run": int(checkpoint["epochs_run"]),
            }
            _write_ledger(model_root, ledger)
            fits_done += 1
            if progress is not None:
                progress(f"[fit {fits_done}/{total_fits}] {key} done "
                         f"(epochs={checkpoint['epochs_run']}, "
                         f"monitor={checkpoint['best_internal_monitor_loss']:.6f})"
                         + (" [restart]" if restart else ""))

    inventory = {
        "plan_id": m1p.PLAN_ID,
        "arch": s4cp.ARCH_DIRECT,
        "arms": list(m1p.ARMS),
        "seeds": [int(s) for s in m1p.MODEL_SEEDS],
        "param_count": m1p.EXPECTED_TRAINABLE_PARAMETERS,
        "pos_weight": float(pos_weight),
        "device": device,
        "checkpoints": ledger["completed"],
        "restarts": ledger["restarts"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_once_json(inventory_path, inventory)
    return inventory


def load_arm_checkpoints(inventory: Mapping[str, Any], arm: str) -> dict[int, dict[str, Any]]:
    """Load one arm's five seed checkpoints (verified against the frozen inventory)."""
    out: dict[int, dict[str, Any]] = {}
    for seed in m1p.MODEL_SEEDS:
        key = f"{arm}:{int(seed)}"
        entry = inventory["checkpoints"].get(key)
        if entry is None:
            raise ValueError(f"inventory missing fit {key}")
        path = Path(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"M1 checkpoint drifted from its inventory: {path}")
        out[int(seed)] = s4cm.load_checkpoint(path)
    return out


__all__ = [
    "FIT_LEDGER_NAME", "MODEL_INVENTORY_NAME", "count_parameters", "fit_all_arms",
    "fit_arm_seed", "load_arm_checkpoints", "score_arm_checkpoints",
]
