"""Stage 5 published-method adapters (plan v0.3 section 2.2, Task 3).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3``. Minimal faithful
implementations of the published formulas on the AC-Twin legal input contract:

- P1 ``CONFIDNET_TCP_ADAPTED`` (Corbiere et al. 2019): TCP target from tempered
  J_hat action probabilities (oracle action only at TRAIN time); confidence MLP;
  risk = 1 - predicted_TCP.
- P2 ``SELECTIVENET_REGRET_ADAPTED`` (Geifman & El-Yaniv 2019): shared legal-feature
  encoder with regret-regression / selection / auxiliary heads; selective-risk +
  quadratic coverage penalty + auxiliary loss at target coverage 0.80;
  risk = 1 - selection probability.
- P3 ``RWR_NN_KNN_REJECT_ADAPTED`` (Li et al. 2024): regressor on ALL train groups
  (no-rejection learning); kNN conditional-loss rejector in the frozen learned
  representation; risk = local residual level.
- P4 ``CONFORMAL_REGRESSION_REJECT_ADAPTED`` (Johansson et al. 2024): regressor on a
  deterministic 80% fit split; Mondrian split-conformal on the disjoint 20%
  calibration split with difficulty-based bins; risk = interval width.
- P5 ``CCTR_DRIVING_CALIBRATION_ADAPTED`` (Cao et al. 2024): heteroscedastic
  Gaussian NLL with a sample-specific Softplus temperature over seed-disagreement
  and legal context features; risk = calibrated sigma.

All score outputs are float64 (G,), higher = riskier. Nothing here reads J_true,
the oracle action, true regret or any truth-derived quantity at score time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

DIRECT_PARAM_COUNT = 15740  # frozen SAME_INFORMATION_DIRECT(width=43)
CAPACITY_LO = 0.5 * DIRECT_PARAM_COUNT
CAPACITY_HI = 2.0 * DIRECT_PARAM_COUNT

TOKEN_DIM = 34
GLOBAL_DIM = 9
N_ACTIONS = 10


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    family: str
    legal_fields: tuple[str, ...]
    grid: tuple[dict[str, Any], ...]


METHOD_SPECS: dict[str, MethodSpec] = {
    "CONFIDNET_TCP_ADAPTED": MethodSpec(
        "CONFIDNET_TCP_ADAPTED", "learned/driving calibrated confidence",
        ("j_hat", "global_features", "selected_action"),
        tuple({"temperature": t} for t in (0.5, 1.0, 2.0))),
    "SELECTIVENET_REGRET_ADAPTED": MethodSpec(
        "SELECTIVENET_REGRET_ADAPTED", "joint selective/rejection",
        ("tokens", "token_mask", "global_features", "j_hat", "selected_action"),
        tuple({"hidden_width": w} for w in (64, 96))),
    "RWR_NN_KNN_REJECT_ADAPTED": MethodSpec(
        "RWR_NN_KNN_REJECT_ADAPTED", "joint selective/rejection",
        ("tokens", "token_mask", "global_features", "j_hat", "selected_action"),
        tuple({"k": k} for k in (16, 32, 64))),
    "CONFORMAL_REGRESSION_REJECT_ADAPTED": MethodSpec(
        "CONFORMAL_REGRESSION_REJECT_ADAPTED", "conformal/risk-control rejection",
        ("tokens", "token_mask", "global_features", "j_hat", "selected_action", "strata"),
        tuple({"difficulty_bins": b} for b in (4, 8))),
    "CCTR_DRIVING_CALIBRATION_ADAPTED": MethodSpec(
        "CCTR_DRIVING_CALIBRATION_ADAPTED", "learned/driving calibrated confidence",
        ("j_hat_by_seed", "global_features", "selected_action", "strata"),
        ({"hidden_width": 64, "depth": 3}, {"hidden_width": 96, "depth": 2})),
}


# --- P1: tempered action probabilities + TCP target --------------------------------


def tempered_action_probs(j_hat: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(-J_hat / temperature) row-wise; legal (prediction side only)."""
    z = -np.asarray(j_hat, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def tcp_target(j_hat: np.ndarray, oracle_action: np.ndarray, temperature: float) -> np.ndarray:
    """ConfidNet TCP target: tempered probability of the (TRAIN-TIME) oracle action."""
    probs = tempered_action_probs(j_hat, temperature)
    rows = np.arange(probs.shape[0])
    return probs[rows, np.asarray(oracle_action, dtype=np.int64)]


def p1_inputs(bundle: Any, config: dict) -> np.ndarray:
    """(G, 19): tempered action probabilities + global features (all legal)."""
    probs = tempered_action_probs(bundle.j_hat, float(config["temperature"]))
    return np.concatenate([probs, np.asarray(bundle.global_features, dtype=np.float64)],
                          axis=1).astype(np.float32)


class ConfidNet(nn.Module):
    """Confidence MLP -> sigmoid scalar (predicted TCP)."""

    def __init__(self, input_dim: int = 19, hidden_width: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_width), nn.ReLU()]
            d = hidden_width
        layers += [nn.Linear(d, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# --- shared token encoder (P2 / P3 / P4) --------------------------------------------


class TokenEncoder(nn.Module):
    """Per-token MLP + masked mean pooling + global-feature fusion -> embedding."""

    def __init__(self, hidden_width: int, token_dim: int = TOKEN_DIM, global_dim: int = GLOBAL_DIM):
        super().__init__()
        self.token_layer = nn.Linear(token_dim, hidden_width)
        self.joint1 = nn.Linear(hidden_width + global_dim, hidden_width)
        self.joint2 = nn.Linear(hidden_width, hidden_width)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor,
                global_features: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.token_layer(tokens))                    # (G,9,h)
        mask = token_mask.unsqueeze(-1).to(h.dtype)                 # (G,9,1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        z = torch.cat([pooled, global_features], dim=1)
        return torch.relu(self.joint2(torch.relu(self.joint1(z))))  # (G,h)


class SelectiveNetModel(nn.Module):
    """SelectiveNet: shared encoder + prediction f, selection g (sigmoid), auxiliary h."""

    def __init__(self, hidden_width: int):
        super().__init__()
        self.encoder = TokenEncoder(hidden_width)
        self.head_f = nn.Linear(hidden_width, 1)
        self.head_g = nn.Linear(hidden_width, 1)
        self.head_h = nn.Linear(hidden_width, 1)

    def forward(self, tokens, token_mask, global_features):
        z = self.encoder(tokens, token_mask, global_features)
        f = self.head_f(z).squeeze(1)
        g = torch.sigmoid(self.head_g(z)).squeeze(1)
        h = self.head_h(z).squeeze(1)
        return f, g, h


def selectivenet_loss(f: torch.Tensor, g: torch.Tensor, h: torch.Tensor,
                      y: torch.Tensor, target_coverage: float = 0.80,
                      lam: float = 32.0, alpha: float = 0.5):
    """Paper Sec. 4: L = alpha*(selective_risk + lambda*Psi(c-phi)) + (1-alpha)*L_aux,
    Psi(a) = max(0, a)^2; selective_risk = E[g*SE] / E[g]."""
    se = (f - y) ** 2
    selective_risk = (g * se).mean() / g.mean().clamp_min(1e-8)
    coverage_penalty = lam * torch.clamp(target_coverage - g.mean(), min=0.0) ** 2
    auxiliary = ((h - y) ** 2).mean()
    total = alpha * (selective_risk + coverage_penalty) + (1 - alpha) * auxiliary
    parts = {"selective_risk": selective_risk.detach(),
             "coverage_penalty": coverage_penalty.detach(),
             "auxiliary": auxiliary.detach()}
    return total, parts


class RegretRegressor(nn.Module):
    """Shared encoder + scalar regret head (P3 main regressor, P4 fit regressor)."""

    def __init__(self, hidden_width: int = 64):
        super().__init__()
        self.encoder = TokenEncoder(hidden_width)
        self.head = nn.Linear(hidden_width, 1)

    def forward(self, tokens, token_mask, global_features):
        return self.head(self.encoder(tokens, token_mask, global_features)).squeeze(1)

    def representation(self, tokens, token_mask, global_features):
        return self.encoder(tokens, token_mask, global_features)


# --- P3: kNN rejector ------------------------------------------------------------------


class KNNRejector:
    """kNN conditional-loss rejection score in a frozen representation (kNNRej idea).

    The reference set (representations + residuals) is fixed at fit time from the
    TRAIN role only; scoring is a pure function of that state and the input
    representation.
    """

    def __init__(self, k: int):
        self.k = int(k)
        self._reference: np.ndarray | None = None
        self._residuals: np.ndarray | None = None

    def fit(self, reference: np.ndarray, residuals: np.ndarray) -> "KNNRejector":
        reference = np.asarray(reference, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        if reference.shape[0] != residuals.shape[0]:
            raise ValueError("reference/residual size mismatch")
        self._reference = reference
        self._residuals = residuals
        return self

    def score(self, representation: np.ndarray) -> np.ndarray:
        if self._reference is None or self._residuals is None:
            raise RuntimeError("KNNRejector is not fitted")
        rep = np.asarray(representation, dtype=np.float64)
        ref_t = torch.from_numpy(self._reference)
        rep_t = torch.from_numpy(rep)
        d = torch.cdist(rep_t, ref_t)
        k = min(self.k, ref_t.shape[0])
        nn_idx = d.topk(k, largest=False).indices
        return self._residuals[nn_idx.numpy()].mean(axis=1)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"k": np.int64(self.k), "reference": self._reference,
                "residuals": self._residuals}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "KNNRejector":
        rej = cls(k=int(state["k"]))
        rej.fit(state["reference"], state["residuals"])
        return rej


# --- P4: deterministic fit/calibration split + Mondrian conformal ------------------------


def conformal_fit_mask(group_seed: np.ndarray, fit_fraction: float = 0.8) -> np.ndarray:
    """Deterministic group-seed hash split (plan 6.1): ~fit_fraction True (fit).

    splitmix64 finalizer for decorrelation: a bare multiplicative hash degenerates
    when the seeds themselves carry multiplicative structure.
    """
    x = np.asarray(group_seed, dtype=np.uint64).copy()
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    threshold = int(round(fit_fraction * 10))
    return (x % np.uint64(10)) < threshold


@dataclass(frozen=True)
class MondrianConformalState:
    """Frozen Mondrian split-conformal rejection state (interval-width risk)."""
    edges: np.ndarray       # (n_bins + 1,) difficulty bin edges
    qhat: np.ndarray        # (n_bins,) cumulative-max half-widths (monotone)
    alpha: float

    @classmethod
    def fit(cls, residuals: np.ndarray, difficulty: np.ndarray,
            n_bins: int, alpha: float = 0.10) -> "MondrianConformalState":
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        difficulty = np.asarray(difficulty, dtype=np.float64).ravel()
        if residuals.shape[0] != difficulty.shape[0]:
            raise ValueError("residual/difficulty size mismatch")
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(difficulty, qs)
        edges[0], edges[-1] = -np.inf, np.inf
        bin_idx = np.clip(np.digitize(difficulty, edges[1:-1]), 0, n_bins - 1)
        global_level = min(1.0, np.ceil((residuals.size + 1) * (1 - alpha)) / residuals.size)
        global_q = float(np.quantile(residuals, global_level, method="higher"))
        qhat = np.empty(n_bins, dtype=np.float64)
        for b in range(n_bins):
            res_b = residuals[bin_idx == b]
            if res_b.size == 0:
                qhat[b] = global_q
                continue
            level = min(1.0, np.ceil((res_b.size + 1) * (1 - alpha)) / res_b.size)
            qhat[b] = float(np.quantile(res_b, level, method="higher"))
        qhat = np.maximum.accumulate(qhat)  # frozen monotone post-rule
        return cls(edges=edges, qhat=qhat, alpha=float(alpha))

    def score(self, difficulty: np.ndarray) -> np.ndarray:
        """Interval WIDTH risk (higher = riskier), a frozen function of difficulty."""
        difficulty = np.asarray(difficulty, dtype=np.float64).ravel()
        bin_idx = np.clip(np.digitize(difficulty, self.edges[1:-1]), 0, self.qhat.size - 1)
        return 2.0 * self.qhat[bin_idx]

    def state_dict(self) -> dict[str, Any]:
        return {"edges": self.edges, "qhat": self.qhat, "alpha": self.alpha}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MondrianConformalState":
        return cls(edges=np.asarray(state["edges"]), qhat=np.asarray(state["qhat"]),
                   alpha=float(state["alpha"]))


def difficulty_from_bundle(bundle: Any) -> np.ndarray:
    """Frozen legal difficulty proxy: seed std of J_hat for the SELECTED action."""
    jbs = np.asarray(bundle.j_hat_by_seed, dtype=np.float64)      # (S,G,A)
    rows = np.arange(jbs.shape[1])
    selected = np.asarray(bundle.selected_action, dtype=np.int64)
    return jbs[:, rows, selected].std(axis=0)


# --- P5: CCTR-style heteroscedastic temperature net ---------------------------------------


class CCTRUncertaintyNet(nn.Module):
    """MLP -> (mu, sigma); sigma = Softplus + eps (sample-specific temperature)."""

    def __init__(self, input_dim: int = 20, hidden_width: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_width), nn.ReLU()]
            d = hidden_width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        out = self.head(self.body(x))
        mu = out[:, 0]
        sigma = self.softplus(out[:, 1]) + 1e-4
        return mu, sigma


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    import math
    return (torch.log(sigma) + 0.5 * math.log(2 * math.pi)
            + (y - mu) ** 2 / (2 * sigma ** 2)).mean()


def p5_inputs(bundle: Any) -> np.ndarray:
    """(G, 20): global features + J_hat row + selected-action seed disagreement."""
    globals_ = np.asarray(bundle.global_features, dtype=np.float64)
    j_hat = np.asarray(bundle.j_hat, dtype=np.float64)
    difficulty = difficulty_from_bundle(bundle)[:, None]
    return np.concatenate([globals_, j_hat, difficulty], axis=1).astype(np.float32)


# --- model construction / capacity ---------------------------------------------------------


def build_model(method_id: str, config: dict, seed: int) -> nn.Module:
    """Deterministically-seeded model for one grid config of one method."""
    torch.manual_seed(int(seed))
    if method_id == "CONFIDNET_TCP_ADAPTED":
        return ConfidNet(input_dim=19, hidden_width=64, depth=3)
    if method_id == "SELECTIVENET_REGRET_ADAPTED":
        return SelectiveNetModel(hidden_width=int(config["hidden_width"]))
    if method_id == "RWR_NN_KNN_REJECT_ADAPTED":
        return RegretRegressor(hidden_width=64)
    if method_id == "CONFORMAL_REGRESSION_REJECT_ADAPTED":
        return RegretRegressor(hidden_width=64)
    if method_id == "CCTR_DRIVING_CALIBRATION_ADAPTED":
        return CCTRUncertaintyNet(input_dim=20, hidden_width=int(config["hidden_width"]),
                                  depth=int(config["depth"]))
    raise ValueError(f"unknown Stage 5 method: {method_id!r}")


def method_param_count(method_id: str, config: dict) -> int:
    model = build_model(method_id, config, seed=0)
    return int(sum(p.numel() for p in model.parameters()))


def check_capacity(method_id: str, config: dict) -> int:
    """Frozen 0.5x-2.0x Direct capacity band (plan 3.2); raises outside the band."""
    count = method_param_count(method_id, config)
    if not (CAPACITY_LO <= count <= CAPACITY_HI):
        raise ValueError(
            f"{method_id} config {config} has {count} params, outside the frozen "
            f"capacity band [{int(CAPACITY_LO)}, {int(CAPACITY_HI)}]")
    return count


__all__ = [
    "DIRECT_PARAM_COUNT", "CAPACITY_LO", "CAPACITY_HI", "MethodSpec", "METHOD_SPECS",
    "tempered_action_probs", "tcp_target", "p1_inputs", "ConfidNet",
    "TokenEncoder", "SelectiveNetModel", "selectivenet_loss", "RegretRegressor",
    "KNNRejector", "conformal_fit_mask", "MondrianConformalState",
    "difficulty_from_bundle", "CCTRUncertaintyNet", "gaussian_nll", "p5_inputs",
    "build_model", "method_param_count", "check_capacity",
]
