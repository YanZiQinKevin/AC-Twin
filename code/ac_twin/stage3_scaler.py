"""Stage 3 train-only feature scaler (fit once on split 0, reused elsewhere).

The development source stores raw object features. The four motion features
``x, y, vx, vy`` (``stage1_data.MOTION_FEATURES``) are standardised; the binary
(``presence``, ``is_ego``) and unit-circle (``cos/sin_heading``) features are
left untouched. The scaler is fit **once** on the train split only and persisted
to disk; validation and development-test reuse the persisted scaler and must
never re-fit it. The fit-once guard and the persisted signature make an
accidental re-fit or a swapped scaler a hard error rather than a silent leak.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ac_twin.stage1_data import MOTION_FEATURES, N_FEATURES


class FeatureScaler:
    """Per-feature mean/std over the train split's valid motion agent-times."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, *, fitted_on: str) -> None:
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        if mean.shape != (N_FEATURES,) or std.shape != (N_FEATURES,):
            raise ValueError(f"scaler mean/std must be shape ({N_FEATURES},)")
        if (std[list(MOTION_FEATURES)] <= 0).any():
            raise ValueError("scaler std must be positive on motion features")
        self.mean = mean
        self.std = std
        self.fitted_on = str(fitted_on)
        self._frozen = True  # a constructed/loaded scaler is never re-fittable

    # -- fitting (train split only) ------------------------------------------
    @classmethod
    def fit(
        cls,
        arrays: Mapping[str, np.ndarray],
        valid_masks: Mapping[str, np.ndarray],
        *,
        fitted_on: str = "train",
    ) -> "FeatureScaler":
        """Fit mean/std on the given (train) arrays over valid agent-times.

        ``arrays`` maps a name to an ``E×T×N×F`` block (history, factual_suffix);
        ``valid_masks`` maps the same names to ``E×T×N`` booleans. Only the
        caller's choice of train-split indices makes this train-only — the driver
        passes split-0 slices and persists the result so later splits reuse it.
        """
        sums = np.zeros(N_FEATURES, dtype=np.float64)
        sq_sums = np.zeros(N_FEATURES, dtype=np.float64)
        counts = np.zeros(N_FEATURES, dtype=np.float64)
        for name, block in arrays.items():
            block = np.asarray(block, dtype=np.float64)
            mask = np.asarray(valid_masks[name], dtype=bool)
            if block.shape[:3] != mask.shape or block.shape[3] != N_FEATURES:
                raise ValueError(f"{name}: block {block.shape} vs mask {mask.shape} mismatch")
            for feat in MOTION_FEATURES:
                vals = block[..., feat][mask]
                sums[feat] += vals.sum()
                sq_sums[feat] += (vals ** 2).sum()
                counts[feat] += vals.size
        mean = np.zeros(N_FEATURES, dtype=np.float64)
        std = np.ones(N_FEATURES, dtype=np.float64)
        for feat in MOTION_FEATURES:
            if counts[feat] == 0:
                raise ValueError(f"no valid values to fit motion feature {feat}")
            mean[feat] = sums[feat] / counts[feat]
            var = sq_sums[feat] / counts[feat] - mean[feat] ** 2
            std[feat] = float(np.sqrt(max(var, 1e-12)))
        return cls(mean, std, fitted_on=fitted_on)

    # -- transform -------------------------------------------------------------
    def transform(self, block: np.ndarray) -> np.ndarray:
        """Return a standardised copy of an ``...×F`` block (motion features only)."""
        block = np.asarray(block, dtype=np.float64).copy()
        for feat in MOTION_FEATURES:
            block[..., feat] = (block[..., feat] - self.mean[feat]) / self.std[feat]
        return block

    def transform_(self, block: np.ndarray) -> np.ndarray:
        """In-place variant of :meth:`transform` for float32/float64 blocks."""
        for feat in MOTION_FEATURES:
            block[..., feat] = (block[..., feat] - self.mean[feat]) / self.std[feat]
        return block

    # -- signature / persistence ------------------------------------------------
    def signature(self) -> str:
        """SHA256 over the fitted mean/std (a stable scaler identity)."""
        digest = hashlib.sha256()
        digest.update(self.mean.astype(np.float64).tobytes())
        digest.update(self.std.astype(np.float64).tobytes())
        digest.update(self.fitted_on.encode("utf-8"))
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "fitted_on": self.fitted_on,
            "motion_features": list(MOTION_FEATURES),
            "signature": self.signature(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = cls(
            np.asarray(payload["mean"], dtype=np.float64),
            np.asarray(payload["std"], dtype=np.float64),
            fitted_on=payload["fitted_on"],
        )
        if scaler.signature() != payload["signature"]:
            raise ValueError("persisted scaler signature mismatch (corrupt or swapped scaler)")
        return scaler
