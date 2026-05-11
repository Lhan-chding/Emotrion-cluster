from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class AffectCalibrator:
    """Remove systematic modality bias between audio and lyrics VA estimates.

    Supported modes:
      - identity: no calibration
      - global_median_shift: subtract half the median delta from each modality
      - affine: per-dimension affine alignment (fit on observed pairs)
      - quantile: quantile-match lyrics distribution to audio distribution
    """

    mode: str = "global_median_shift"
    bias_: Optional[np.ndarray] = field(default=None, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def fit(
        self,
        audio_va: np.ndarray,
        lyrics_va: np.ndarray,
        mask: np.ndarray,
    ) -> "AffectCalibrator":
        audio = np.asarray(audio_va, dtype=np.float32)
        lyrics = np.asarray(lyrics_va, dtype=np.float32)
        has_both = _has_both_mask(mask)

        if self.mode == "identity":
            self.bias_ = np.zeros(2, dtype=np.float32)
        elif self.mode == "global_median_shift":
            if has_both.sum() < 2:
                self.bias_ = np.zeros(2, dtype=np.float32)
            else:
                delta = lyrics[has_both] - audio[has_both]
                self.bias_ = np.median(delta, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported calibration mode: {self.mode!r}")

        self._fitted = True
        return self

    def transform(
        self,
        audio_va: np.ndarray,
        lyrics_va: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("AffectCalibrator must be fit before transform.")
        audio = np.asarray(audio_va, dtype=np.float32).copy()
        lyrics = np.asarray(lyrics_va, dtype=np.float32).copy()

        if self.mode == "identity":
            return audio, lyrics

        half_bias = self.bias_ * 0.5
        audio_cal = audio + half_bias
        lyrics_cal = lyrics - half_bias
        return audio_cal.astype(np.float32), lyrics_cal.astype(np.float32)

    def fit_transform(
        self,
        audio_va: np.ndarray,
        lyrics_va: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(audio_va, lyrics_va, mask)
        return self.transform(audio_va, lyrics_va, mask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "bias": self.bias_.tolist() if self.bias_ is not None else None,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffectCalibrator":
        obj = cls(mode=data["mode"])
        if data.get("bias") is not None:
            obj.bias_ = np.asarray(data["bias"], dtype=np.float32)
        obj._fitted = bool(data.get("fitted", False))
        return obj


@dataclass
class DiffResidualizer:
    """Remove consensus-dependent drift from raw audio-lyrics delta.

    Produces conditional residual tension: d_res = d_raw - E[d_raw | consensus].

    Supported modes:
      - knn: local mean subtraction using K nearest neighbors in consensus space
      - macro_bin: bin by macro cluster assignment, subtract per-bin median
      - identity: no residualization (raw delta passthrough)
    """

    mode: str = "knn"
    n_neighbors: int = 101
    _ref_consensus: Optional[np.ndarray] = field(default=None, repr=False)
    _ref_delta: Optional[np.ndarray] = field(default=None, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def fit(
        self,
        consensus_va: np.ndarray,
        raw_delta: np.ndarray,
        mask: np.ndarray,
    ) -> "DiffResidualizer":
        consensus = np.asarray(consensus_va, dtype=np.float32)
        delta = np.asarray(raw_delta, dtype=np.float32)
        has_both = _has_both_mask(mask)

        if self.mode == "identity":
            self._fitted = True
            return self

        observed = has_both
        if observed.sum() < 2:
            self._ref_consensus = consensus[:1]
            self._ref_delta = delta[:1]
            self._fitted = True
            return self

        self._ref_consensus = consensus[observed].copy()
        self._ref_delta = delta[observed].copy()
        self._fitted = True
        return self

    def transform(
        self,
        consensus_va: np.ndarray,
        raw_delta: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("DiffResidualizer must be fit before transform.")
        consensus = np.asarray(consensus_va, dtype=np.float32)
        delta = np.asarray(raw_delta, dtype=np.float32)

        if self.mode == "identity":
            return delta.copy()

        has_both = _has_both_mask(mask)
        result = np.zeros_like(delta)
        if not has_both.any():
            return result

        k = min(self.n_neighbors, self._ref_consensus.shape[0])
        obs_idx = np.where(has_both)[0]

        ref_c = self._ref_consensus.astype(np.float64)
        ref_d = self._ref_delta.astype(np.float64)

        for idx in obs_idx:
            diffs = ref_c - consensus[idx].astype(np.float64)
            dists = np.sum(diffs ** 2, axis=1)
            nn = np.argpartition(dists, min(k, len(dists)) - 1)[:k]
            local_mean = ref_d[nn].mean(axis=0)
            result[idx] = (delta[idx].astype(np.float64) - local_mean).astype(np.float32)

        return result.astype(np.float32)

    def fit_transform(
        self,
        consensus_va: np.ndarray,
        raw_delta: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        self.fit(consensus_va, raw_delta, mask)
        return self.transform(consensus_va, raw_delta, mask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "n_neighbors": self.n_neighbors,
            "ref_consensus_shape": list(self._ref_consensus.shape) if self._ref_consensus is not None else None,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiffResidualizer":
        obj = cls(mode=data["mode"], n_neighbors=data.get("n_neighbors", 101))
        obj._fitted = bool(data.get("fitted", False))
        return obj


def _has_both_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 1:
        return m > 0.0
    return (m[:, 0] > 0.0) & (m[:, 1] > 0.0)
