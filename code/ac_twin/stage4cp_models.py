"""Stage 4C Plus decision-risk model architectures, training and scoring (plan section 4).

Two candidate heads share the SAME supervision and the SAME per-group information:

* ``DECISION_COMPARATIVE_RISK`` (DCR): a shared two-layer MLP ``phi`` encodes each of the
  nine competitor tokens; a permutation-invariant mean/max pool aggregates them; a
  two-layer ``head`` reads ``concat(mean, max, global)`` -> ``(raw_mu, logit_high)``.
* ``SAME_INFORMATION_DIRECT`` (Direct): the nine tokens (frozen ACTION_PLANS order) are
  flattened with the globals into a plain MLP; its hidden width is chosen mechanically so
  its trainable-parameter count is within ``+/-5%`` of the matching DCR (audit recorded).

Both emit ``(raw_mu, logit_high)``; the primary risk score and the two pre-registered
non-blocking ablations derive from one forward pass (plan section 3.3):
    mu_pos     = softplus(raw_mu)
    risk_score = sigmoid(logit_high) * 0.01 * expm1(mu_pos)
    MU_ONLY    = mu_pos
    CLASS_ONLY = sigmoid(logit_high)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ac_twin import stage4cp_features as s4cf
from ac_twin import stage4cp_protocol as s4cp

SOFTPLUS_BETA = 1.0


# --- architectures ---------------------------------------------------------------------


class DCRiskModel(nn.Module):
    """Permutation-invariant decision-comparative risk head."""

    def __init__(self, token_dim: int, global_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.phi = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, tokens, token_mask, global_features):
        """tokens (B,K,TD), token_mask (B,K), global (B,GD) -> (raw_mu, logit_high)."""
        h = self.phi(tokens)                                   # (B,K,H)
        mask = token_mask.to(h.dtype).unsqueeze(-1)            # (B,K,1)
        h_masked = h * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = h_masked.sum(dim=1) / denom                     # (B,H)
        neg = torch.finfo(h.dtype).min
        h_for_max = h_masked + (1.0 - mask) * neg
        mx = h_for_max.max(dim=1).values                       # (B,H)
        hg = torch.cat([mean, mx, global_features], dim=-1)    # (B,2H+GD)
        out = self.head(hg)                                    # (B,2)
        return out[:, 0], out[:, 1]


class DirectRiskModel(nn.Module):
    """Same-information direct (flattened, order-sensitive) risk head."""

    def __init__(self, flat_dim: int, global_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(flat_dim + global_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, flat_tokens, global_features):
        """flat_tokens (B,K*TD), global (B,GD) -> (raw_mu, logit_high)."""
        x = torch.cat([flat_tokens, global_features], dim=-1)
        out = self.net(x)
        return out[:, 0], out[:, 1]


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_model(
    arch: str,
    *,
    hidden_dim: int,
    dropout: float,
    direct_width: int | None = None,
    n_competitors: int = s4cp.N_COMPETITORS,
    token_dim: int = s4cf.TOKEN_DIM,
    global_dim: int = s4cf.GLOBAL_DIM,
) -> nn.Module:
    """Build one candidate model. Direct requires its pre-computed ``direct_width``."""
    if arch == s4cp.ARCH_DCR:
        return DCRiskModel(token_dim, global_dim, hidden_dim, dropout)
    if arch == s4cp.ARCH_DIRECT:
        if direct_width is None:
            raise ValueError("Direct model requires a pre-computed direct_width")
        return DirectRiskModel(n_competitors * token_dim, global_dim, direct_width, dropout)
    raise ValueError(f"unknown architecture: {arch}")


def select_direct_width(
    dcr_param_count: int,
    *,
    n_competitors: int = s4cp.N_COMPETITORS,
    token_dim: int = s4cf.TOKEN_DIM,
    global_dim: int = s4cf.GLOBAL_DIM,
    tolerance: float = s4cp.PARAM_COUNT_TOLERANCE,
) -> tuple[int, int]:
    """Mechanically pick the Direct hidden width within ``tolerance`` of the DCR params.

    Builds candidate Direct MLPs of increasing width and returns the width whose
    trainable-parameter count is closest to ``dcr_param_count``; raises if none is within
    tolerance. Returns ``(width, direct_param_count)``.
    """
    flat_dim = n_competitors * token_dim
    best: tuple[int, int] | None = None
    for width in range(4, 4096):
        model = DirectRiskModel(flat_dim, global_dim, width, dropout=0.0)
        params = count_parameters(model)
        if abs(params - dcr_param_count) / dcr_param_count <= tolerance:
            if best is None or abs(params - dcr_param_count) < abs(best[1] - dcr_param_count):
                best = (width, params)
    if best is None:
        raise ValueError(
            f"no Direct width within {tolerance:.0%} of DCR param count {dcr_param_count}")
    return best


# --- score derivation --------------------------------------------------------------------


def heads_to_scores(raw_mu: np.ndarray, logit_high: np.ndarray) -> dict[str, np.ndarray]:
    """Derive the primary risk score and the two non-blocking ablations."""
    raw_mu = np.asarray(raw_mu, dtype=np.float64)
    logit_high = np.asarray(logit_high, dtype=np.float64)
    mu_pos = np.log1p(np.exp(-np.abs(raw_mu))) + np.maximum(raw_mu, 0.0)  # softplus, stable
    class_only = 1.0 / (1.0 + np.exp(-logit_high))                        # sigmoid
    risk_score = class_only * 0.01 * np.expm1(mu_pos)
    for name, value in (("mu_pos", mu_pos), ("class_only", class_only), ("risk_score", risk_score)):
        if not np.isfinite(value).all():
            raise RuntimeError(f"non-finite Stage 4C Plus score: {name}")
    return {"risk_score": risk_score, "mu_only": mu_pos, "class_only": class_only}


# --- training -------------------------------------------------------------------------------


def _internal_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 90/10 train/internal-val split of the training rows (train-only)."""
    generator = np.random.default_rng(int(seed))
    permutation = generator.permutation(n)
    n_val = max(1, int(round(0.1 * n)))
    return permutation[n_val:], permutation[:n_val]


def fit_decision_risk_model(
    arch: str,
    tokens: np.ndarray,
    global_features: np.ndarray,
    token_mask: np.ndarray,
    y_reg: np.ndarray,
    y_high: np.ndarray,
    *,
    hidden_dim: int,
    lambda_cls: float,
    seed: int,
    config: Mapping[str, Any],
    direct_width: int | None = None,
) -> dict[str, Any]:
    """Train one decision-risk head; return a serialisable checkpoint payload.

    Class weight (``pos_weight``) is computed from the TRAIN rows only. Early stopping
    monitors total loss on a deterministic internal 90/10 split of the training rows
    (train-only; never the validation/screen/confirm roles).
    """
    checked = s4cp.validate_config(config)
    tokens = np.asarray(tokens, dtype=np.float64)
    global_features = np.asarray(global_features, dtype=np.float64)
    token_mask = np.asarray(token_mask, dtype=bool)
    y_reg = np.asarray(y_reg, dtype=np.float64).ravel()
    y_high = np.asarray(y_high, dtype=np.float64).ravel()
    n = tokens.shape[0]
    if not (global_features.shape[0] == token_mask.shape[0] == y_reg.shape[0] == y_high.shape[0] == n):
        raise ValueError("tokens/global/mask/labels must align on the group axis")
    if n < 2:
        raise ValueError("need >=2 training groups")
    n_pos = float(y_high.sum())
    pos_weight = (n - n_pos) / max(n_pos, 1.0)  # train-prevalence class weight

    device = torch.device(str(checked["device"]) if torch.cuda.is_available()
                          or str(checked["device"]) == "cpu" else "cpu")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**31 - 1))

    model = build_model(arch, hidden_dim=int(hidden_dim),
                        dropout=float(checked["dropout"]), direct_width=direct_width).to(device)

    tr_idx, val_idx = _internal_split(n, int(seed))
    x_tok = torch.as_tensor(tokens, dtype=torch.float32)
    x_glob = torch.as_tensor(global_features, dtype=torch.float32)
    x_mask = torch.as_tensor(token_mask, dtype=torch.float32)
    y_r = torch.as_tensor(y_reg, dtype=torch.float32)
    y_h = torch.as_tensor(y_high, dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(checked["learning_rate"]),
                                  weight_decay=float(checked["weight_decay"]))
    smooth_l1 = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32,
                                                          device=device))
    batch_size = int(checked["train_batch_size"])
    max_epochs = int(checked["max_epochs"])
    patience = int(checked["early_stop_patience"])

    def total_loss(idx: np.ndarray) -> float:
        model.eval()
        with torch.no_grad():
            raw_mu, logit = _forward(model, arch, x_tok[idx].to(device),
                                     x_mask[idx].to(device), x_glob[idx].to(device))
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_r[idx].to(device)) \
                + float(lambda_cls) * bce(logit, y_h[idx].to(device))
        return float(loss.item())

    best_state = None
    best_val = float("inf")
    wait = 0
    batch_generator = torch.Generator().manual_seed(int(seed))
    for _ in range(max_epochs):
        model.train()
        order = torch.randperm(tr_idx.shape[0], generator=batch_generator).numpy()
        for start in range(0, tr_idx.shape[0], batch_size):
            sel = tr_idx[order[start:start + batch_size]]
            optimizer.zero_grad()
            raw_mu, logit = _forward(model, arch, x_tok[sel].to(device),
                                     x_mask[sel].to(device), x_glob[sel].to(device))
            mu_pos = torch.nn.functional.softplus(raw_mu)
            loss = smooth_l1(mu_pos, y_r[sel].to(device)) \
                + float(lambda_cls) * bce(logit, y_h[sel].to(device))
            loss.backward()
            optimizer.step()
        val_loss = total_loss(val_idx)
        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    train_loss = total_loss(tr_idx)
    return {
        "arch": arch,
        "hidden_dim": int(hidden_dim),
        "direct_width": (int(direct_width) if arch == s4cp.ARCH_DIRECT else None),
        "lambda_cls": float(lambda_cls),
        "seed": int(seed),
        "pos_weight": float(pos_weight),
        "dropout": float(checked["dropout"]),
        "token_dim": s4cf.TOKEN_DIM,
        "global_dim": s4cf.GLOBAL_DIM,
        "n_competitors": s4cp.N_COMPETITORS,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "train_loss": float(train_loss),
        "best_internal_val_loss": float(best_val),
        "n_train_groups": int(n),
    }


def _forward(model: nn.Module, arch: str, tokens, token_mask, global_features):
    if arch == s4cp.ARCH_DCR:
        return model(tokens, token_mask, global_features)
    flat = tokens.flatten(start_dim=1)
    return model(flat, global_features)


# --- prediction ---------------------------------------------------------------------------


def _load_model_from_checkpoint(checkpoint: Mapping[str, Any]) -> nn.Module:
    model = build_model(
        str(checkpoint["arch"]), hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        direct_width=(None if checkpoint["arch"] == s4cp.ARCH_DCR
                      else int(checkpoint["direct_width"])),
        n_competitors=int(checkpoint["n_competitors"]),
        token_dim=int(checkpoint["token_dim"]), global_dim=int(checkpoint["global_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_decision_risk(
    checkpoint: Mapping[str, Any],
    tokens: np.ndarray,
    global_features: np.ndarray,
    token_mask: np.ndarray,
    *,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Score groups with a trained checkpoint -> risk_score / mu_only / class_only (G,)."""
    tokens = np.asarray(tokens, dtype=np.float64)
    global_features = np.asarray(global_features, dtype=np.float64)
    token_mask = np.asarray(token_mask, dtype=bool)
    model = _load_model_from_checkpoint(checkpoint)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)
    with torch.no_grad():
        raw_mu, logit = _forward(
            model, str(checkpoint["arch"]),
            torch.as_tensor(tokens, dtype=torch.float32, device=dev),
            torch.as_tensor(token_mask, dtype=torch.float32, device=dev),
            torch.as_tensor(global_features, dtype=torch.float32, device=dev))
    scores = heads_to_scores(raw_mu.cpu().numpy(), logit.cpu().numpy())
    scores["raw_mu"] = raw_mu.cpu().numpy()
    scores["logit_high"] = logit.cpu().numpy()
    return scores


# --- persistence -------------------------------------------------------------------------


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary)
    temporary.replace(target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


__all__ = [
    "DCRiskModel",
    "DirectRiskModel",
    "build_model",
    "count_parameters",
    "fit_decision_risk_model",
    "heads_to_scores",
    "load_checkpoint",
    "predict_decision_risk",
    "save_checkpoint",
    "select_direct_width",
]
