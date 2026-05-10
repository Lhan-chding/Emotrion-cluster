from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cluster.backends.gmm_convergence import fit_gaussian_mixture_robust


BlockSlice = Tuple[int, int]


def _as_block_slices(block_slices: Sequence[Sequence[int]], n_features: int) -> List[BlockSlice]:
    slices: List[BlockSlice] = []
    for raw in block_slices:
        start, stop = int(raw[0]), int(raw[1])
        if start < 0 or stop <= start or stop > int(n_features):
            raise ValueError(f"Invalid block slice ({start}, {stop}) for feature_dim={n_features}.")
        slices.append((start, stop))
    if not slices:
        raise ValueError("At least one block slice is required.")
    return slices


def _normalize_block_mask(block_mask: Optional[np.ndarray], n_samples: int, n_blocks: int) -> np.ndarray:
    if block_mask is None:
        return np.ones((n_samples, n_blocks), dtype=bool)
    mask = np.asarray(block_mask, dtype=bool)
    if mask.shape != (n_samples, n_blocks):
        raise ValueError(f"block_mask must have shape [{n_samples}, {n_blocks}], got {mask.shape}.")
    return mask


@dataclass
class PartialGaussianMixture:
    n_components: int
    block_slices: Sequence[Sequence[int]]
    covariance_type: str = "diag"
    random_state: int = 42
    n_init: int = 10
    reg_covar: float = 1e-5

    def fit(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> "PartialGaussianMixture":
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"features must be 2D, got {matrix.shape}.")
        if str(self.covariance_type).lower() != "diag":
            raise ValueError("PartialGaussianMixture currently supports covariance_type='diag' only.")
        self.block_slices_ = _as_block_slices(self.block_slices, matrix.shape[1])
        mask = _normalize_block_mask(block_mask, matrix.shape[0], len(self.block_slices_))
        fit_matrix = np.array(matrix, copy=True)
        self.block_fill_values_ = []
        for block_id, (start, stop) in enumerate(self.block_slices_):
            observed = mask[:, block_id]
            if observed.any():
                fill = fit_matrix[observed, start:stop].mean(axis=0).astype(np.float32)
            else:
                fill = np.zeros(stop - start, dtype=np.float32)
            fit_matrix[~observed, start:stop] = fill
            self.block_fill_values_.append(fill)

        self.model_ = fit_gaussian_mixture_robust(
            fit_matrix,
            n_components=int(self.n_components),
            covariance_type="diag",
            reg_covar=float(self.reg_covar),
            n_init=int(self.n_init),
            max_iter=300,
            random_state=int(self.random_state),
            require_converged=True,
            context="partial Gaussian mixture",
        )
        self.means_ = self.model_.means_.astype(np.float64)
        self.covariances_ = np.maximum(self.model_.covariances_.astype(np.float64), float(self.reg_covar))
        self.weights_ = np.maximum(self.model_.weights_.astype(np.float64), 1e-12)
        self.weights_ = self.weights_ / self.weights_.sum()
        return self

    def _check_is_fit(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("PartialGaussianMixture must be fit before predict.")

    def predict_proba(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_is_fit()
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.means_.shape[1]:
            raise ValueError(f"features must have shape [N, {self.means_.shape[1]}], got {matrix.shape}.")
        mask = _normalize_block_mask(block_mask, matrix.shape[0], len(self.block_slices_))
        logp = np.tile(np.log(self.weights_).reshape(1, -1), (matrix.shape[0], 1))

        for row_idx in range(matrix.shape[0]):
            for block_id, (start, stop) in enumerate(self.block_slices_):
                if not mask[row_idx, block_id]:
                    continue
                x = matrix[row_idx : row_idx + 1, start:stop]
                mean = self.means_[:, start:stop]
                var = self.covariances_[:, start:stop]
                diff = x - mean
                block_dim = stop - start
                block_logp = -0.5 * (
                    block_dim * np.log(2.0 * np.pi)
                    + np.log(var).sum(axis=1)
                    + ((diff * diff) / var).sum(axis=1)
                )
                logp[row_idx] += block_logp

        max_log = np.max(logp, axis=1, keepdims=True)
        probs = np.exp(logp - max_log)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return probs.astype(np.float32)

    def predict(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.argmax(self.predict_proba(features, block_mask=block_mask), axis=1).astype(np.int64)
