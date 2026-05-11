from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


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
class BalanceAlphaLearner:
    """Learn a global audio/lyrics balance weight from VA clusterability."""

    mode: str = "clusterability_alpha"
    alpha_min: float = 0.20
    alpha_max: float = 0.90
    alpha_step: float = 0.05
    search_k_min: int = 4
    search_k_max: int = 8
    regularization: float = 0.05
    random_state: int = 42
    alpha_: Optional[float] = None
    scores_: List[Dict[str, float]] = field(default_factory=list, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def fit(
        self,
        audio_cal: np.ndarray,
        lyrics_cal: np.ndarray,
        mask: np.ndarray,
    ) -> "BalanceAlphaLearner":
        audio = np.asarray(audio_cal, dtype=np.float32)
        lyrics = np.asarray(lyrics_cal, dtype=np.float32)
        view_mask = np.asarray(mask, dtype=np.float32)
        has_any = _has_any_mask(view_mask)

        mode = str(self.mode or "clusterability_alpha").strip().lower()
        if mode == "global_alpha":
            self.alpha_ = float(np.clip(0.5 if self.alpha_ is None else self.alpha_, 0.0, 1.0))
            self._fitted = True
            return self
        if mode != "clusterability_alpha":
            raise ValueError(f"Unsupported balance alpha mode: {self.mode!r}")

        if int(has_any.sum()) < max(4, int(self.search_k_min)):
            self.alpha_ = 0.5
            self.scores_ = []
            self._fitted = True
            return self

        best_alpha = 0.5
        best_score = -float("inf")
        self.scores_ = []
        for alpha in _alpha_grid(float(self.alpha_min), float(self.alpha_max), float(self.alpha_step)):
            consensus = _balanced_consensus(audio, lyrics, view_mask, alpha)
            matrix = consensus[has_any].astype(np.float64)
            alpha_best_score = -float("inf")
            alpha_best_k = -1
            alpha_best_silhouette = float("nan")
            alpha_best_stability = 0.0
            alpha_best_size_balance = 0.0
            k_upper = min(int(self.search_k_max), int(matrix.shape[0]) - 1)
            for k in range(max(2, int(self.search_k_min)), k_upper + 1):
                metrics = self._score_candidate(matrix, int(k), float(alpha))
                if metrics is None:
                    continue
                if metrics["score"] > alpha_best_score:
                    alpha_best_score = float(metrics["score"])
                    alpha_best_k = int(k)
                    alpha_best_silhouette = float(metrics["silhouette"])
                    alpha_best_stability = float(metrics["stability"])
                    alpha_best_size_balance = float(metrics["size_balance"])
            if np.isfinite(alpha_best_score):
                self.scores_.append(
                    {
                        "alpha": float(alpha),
                        "best_k": float(alpha_best_k),
                        "score": float(alpha_best_score),
                        "silhouette": float(alpha_best_silhouette),
                        "stability": float(alpha_best_stability),
                        "size_balance": float(alpha_best_size_balance),
                    }
                )
                if alpha_best_score > best_score:
                    best_score = float(alpha_best_score)
                    best_alpha = float(alpha)

        self.alpha_ = float(best_alpha)
        self._fitted = True
        return self

    def transform(
        self,
        audio_cal: np.ndarray,
        lyrics_cal: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("BalanceAlphaLearner must be fit or initialized with alpha_ before transform.")
        audio = np.asarray(audio_cal, dtype=np.float32)
        lyrics = np.asarray(lyrics_cal, dtype=np.float32)
        return _balanced_consensus(audio, lyrics, np.asarray(mask, dtype=np.float32), float(self.alpha_))

    def fit_transform(
        self,
        audio_cal: np.ndarray,
        lyrics_cal: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        self.fit(audio_cal, lyrics_cal, mask)
        return self.transform(audio_cal, lyrics_cal, mask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "alpha_step": float(self.alpha_step),
            "search_k_min": int(self.search_k_min),
            "search_k_max": int(self.search_k_max),
            "regularization": float(self.regularization),
            "random_state": int(self.random_state),
            "alpha": float(self.alpha_) if self.alpha_ is not None else None,
            "fitted": bool(self._fitted),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BalanceAlphaLearner":
        obj = cls(
            mode=str(data.get("mode", "clusterability_alpha")),
            alpha_min=float(data.get("alpha_min", 0.20)),
            alpha_max=float(data.get("alpha_max", 0.90)),
            alpha_step=float(data.get("alpha_step", 0.05)),
            search_k_min=int(data.get("search_k_min", 4)),
            search_k_max=int(data.get("search_k_max", 8)),
            regularization=float(data.get("regularization", 0.05)),
            random_state=int(data.get("random_state", 42)),
            alpha_=data.get("alpha"),
        )
        obj._fitted = bool(data.get("fitted", obj.alpha_ is not None))
        return obj

    def _score_candidate(self, matrix: np.ndarray, k: int, alpha: float) -> Optional[Dict[str, float]]:
        if matrix.shape[0] <= k:
            return None
        try:
            model = GaussianMixture(
                n_components=int(k),
                covariance_type="diag",
                reg_covar=1e-5,
                n_init=3,
                random_state=int(self.random_state) + int(k),
            ).fit(matrix)
            labels = model.predict(matrix)
            if np.unique(labels).size < 2:
                return None
            silhouette = float(silhouette_score(matrix, labels))
        except Exception:
            return None

        counts = np.bincount(labels, minlength=int(k)).astype(np.float64)
        if counts.min() <= 0:
            return None
        size_balance = float(counts.min() / max(counts.max(), 1.0))
        stability_scores: List[float] = []
        for run_idx in range(1, 4):
            try:
                alt = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    reg_covar=1e-5,
                    n_init=1,
                    random_state=int(self.random_state) + int(k) * 100 + run_idx,
                ).fit(matrix)
                alt_labels = alt.predict(matrix)
                if np.unique(alt_labels).size >= 2:
                    stability_scores.append(float(adjusted_rand_score(labels, alt_labels)))
            except Exception:
                continue
        stability = float(np.mean(stability_scores)) if stability_scores else 0.0
        score = (
            silhouette
            + 0.20 * stability
            + 0.10 * size_balance
            - float(self.regularization) * float((alpha - 0.5) ** 2)
        )
        return {
            "score": float(score),
            "silhouette": float(silhouette),
            "stability": float(stability),
            "size_balance": float(size_balance),
        }


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


def _has_any_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 1:
        return m > 0.0
    return (m[:, 0] > 0.0) | (m[:, 1] > 0.0)


def _alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> np.ndarray:
    if alpha_step <= 0:
        raise ValueError("alpha_step must be positive.")
    lo = min(float(alpha_min), float(alpha_max))
    hi = max(float(alpha_min), float(alpha_max))
    count = int(np.floor((hi - lo) / float(alpha_step))) + 1
    values = lo + np.arange(count + 1, dtype=np.float64) * float(alpha_step)
    values = values[values <= hi + 1e-9]
    if values.size == 0:
        values = np.asarray([lo], dtype=np.float64)
    return np.round(values, 10)


def _balanced_consensus(audio: np.ndarray, lyrics: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    audio_matrix = np.asarray(audio, dtype=np.float32)
    lyrics_matrix = np.asarray(lyrics, dtype=np.float32)
    view_mask = np.asarray(mask, dtype=np.float32)
    if audio_matrix.shape != lyrics_matrix.shape:
        raise ValueError(f"audio and lyrics must have matching shapes, got {audio_matrix.shape} and {lyrics_matrix.shape}.")
    if audio_matrix.ndim != 2 or audio_matrix.shape[1] != 2:
        raise ValueError(f"audio and lyrics must have shape [N, 2], got {audio_matrix.shape}.")
    if view_mask.ndim != 2 or view_mask.shape[0] != audio_matrix.shape[0] or view_mask.shape[1] < 2:
        raise ValueError(f"mask must have shape [N, >=2], got {view_mask.shape}.")

    has_audio = view_mask[:, 0:1] > 0.0
    has_lyrics = view_mask[:, 1:2] > 0.0
    has_both = has_audio & has_lyrics
    clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
    consensus = np.full(audio_matrix.shape, 0.5, dtype=np.float32)
    consensus[has_both[:, 0]] = (
        clipped_alpha * audio_matrix[has_both[:, 0]]
        + (1.0 - clipped_alpha) * lyrics_matrix[has_both[:, 0]]
    ).astype(np.float32)
    audio_only = has_audio[:, 0] & ~has_lyrics[:, 0]
    lyrics_only = has_lyrics[:, 0] & ~has_audio[:, 0]
    consensus[audio_only] = audio_matrix[audio_only]
    consensus[lyrics_only] = lyrics_matrix[lyrics_only]
    return consensus.astype(np.float32)
