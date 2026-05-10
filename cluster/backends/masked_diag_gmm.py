from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from cluster.backends.gmm_convergence import fit_gaussian_mixture_robust


BlockSlice = Tuple[int, int]


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    return (max_values + np.log(np.maximum(stable.sum(axis=axis, keepdims=True), 1e-300))).squeeze(axis)


def _as_block_slices(block_slices: Sequence[Sequence[int]], n_features: int) -> list[BlockSlice]:
    slices: list[BlockSlice] = []
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


def _feature_mask_from_blocks(block_mask: np.ndarray, block_slices: Sequence[BlockSlice], n_features: int) -> np.ndarray:
    feature_mask = np.zeros((block_mask.shape[0], int(n_features)), dtype=bool)
    for block_id, (start, stop) in enumerate(block_slices):
        feature_mask[:, start:stop] = block_mask[:, block_id : block_id + 1]
    return feature_mask


@dataclass
class MaskedDiagonalGMM:
    """Diagonal GMM trained and scored with block-level missingness.

    Missing blocks do not contribute to the likelihood, E-step, or M-step
    sufficient statistics. This keeps K search and final assignment aligned
    when the clustering feature space has consensus/tension/metadata blocks.
    """

    n_components: int
    block_slices: Sequence[Sequence[int]]
    covariance_type: str = "diag"
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 100
    tol: float = 1e-3
    reg_covar: float = 1e-5

    def fit(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> "MaskedDiagonalGMM":
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"features must be 2D, got {matrix.shape}.")
        if str(self.covariance_type).lower() != "diag":
            raise ValueError("MaskedDiagonalGMM supports covariance_type='diag' only.")
        self.block_slices_ = _as_block_slices(self.block_slices, matrix.shape[1])
        mask = _normalize_block_mask(block_mask, matrix.shape[0], len(self.block_slices_))
        feature_mask = _feature_mask_from_blocks(mask, self.block_slices_, matrix.shape[1])
        if not feature_mask.any(axis=1).all():
            raise ValueError("Every row must have at least one observed feature block.")

        init_matrix = np.array(matrix, copy=True)
        for block_id, (start, stop) in enumerate(self.block_slices_):
            observed = mask[:, block_id]
            fill = init_matrix[observed, start:stop].mean(axis=0) if observed.any() else np.zeros(stop - start)
            init_matrix[~observed, start:stop] = fill

        init = fit_gaussian_mixture_robust(
            init_matrix,
            n_components=int(self.n_components),
            covariance_type="diag",
            reg_covar=float(self.reg_covar),
            n_init=int(self.n_init),
            max_iter=max(20, int(self.max_iter)),
            tol=float(self.tol),
            random_state=int(self.random_state),
            context="masked diagonal GMM initializer",
        )
        self.means_ = init.means_.astype(np.float64)
        self.covariances_ = np.maximum(init.covariances_.astype(np.float64), float(self.reg_covar))
        self.weights_ = np.maximum(init.weights_.astype(np.float64), 1e-12)
        self.weights_ = self.weights_ / self.weights_.sum()

        previous_ll = -np.inf
        for iteration in range(int(self.max_iter)):
            log_prob = self._component_log_prob(matrix, feature_mask)
            log_norm = _logsumexp(log_prob, axis=1)
            resp = np.exp(log_prob - log_norm.reshape(-1, 1))

            nk = np.maximum(resp.sum(axis=0), 1e-12)
            self.weights_ = nk / matrix.shape[0]
            for component in range(int(self.n_components)):
                for feature in range(matrix.shape[1]):
                    observed = feature_mask[:, feature]
                    denom = float((resp[observed, component]).sum())
                    if denom <= 1e-12:
                        continue
                    values = matrix[observed, feature]
                    weights = resp[observed, component]
                    mean = float((weights * values).sum() / denom)
                    var = float((weights * (values - mean) ** 2).sum() / denom)
                    self.means_[component, feature] = mean
                    self.covariances_[component, feature] = max(var, float(self.reg_covar))

            current_ll = float(log_norm.sum())
            if iteration > 0 and abs(current_ll - previous_ll) <= float(self.tol) * max(abs(previous_ll), 1.0):
                break
            previous_ll = current_ll

        self.n_iter_ = iteration + 1
        self.lower_bound_ = previous_ll / max(matrix.shape[0], 1)
        self.labels_ = self.predict(matrix, block_mask=mask)
        return self

    def _check_is_fit(self) -> None:
        if not hasattr(self, "means_"):
            raise RuntimeError("MaskedDiagonalGMM must be fit before scoring or prediction.")

    def _component_log_prob(self, matrix: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
        n_samples, n_features = matrix.shape
        log_prob = np.tile(np.log(np.maximum(self.weights_, 1e-12)).reshape(1, -1), (n_samples, 1))
        for component in range(int(self.n_components)):
            diff = matrix - self.means_[component].reshape(1, n_features)
            var = np.maximum(self.covariances_[component].reshape(1, n_features), float(self.reg_covar))
            term = np.where(
                feature_mask,
                np.log(2.0 * np.pi * var) + (diff * diff) / var,
                0.0,
            )
            log_prob[:, component] += -0.5 * term.sum(axis=1)
        return log_prob

    def score_samples(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_is_fit()
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.means_.shape[1]:
            raise ValueError(f"features must have shape [N, {self.means_.shape[1]}], got {matrix.shape}.")
        mask = _normalize_block_mask(block_mask, matrix.shape[0], len(self.block_slices_))
        feature_mask = _feature_mask_from_blocks(mask, self.block_slices_, matrix.shape[1])
        return _logsumexp(self._component_log_prob(matrix, feature_mask), axis=1).astype(np.float64)

    def predict_proba(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_is_fit()
        matrix = np.asarray(features, dtype=np.float64)
        mask = _normalize_block_mask(block_mask, matrix.shape[0], len(self.block_slices_))
        feature_mask = _feature_mask_from_blocks(mask, self.block_slices_, matrix.shape[1])
        log_prob = self._component_log_prob(matrix, feature_mask)
        log_norm = _logsumexp(log_prob, axis=1).reshape(-1, 1)
        return np.exp(log_prob - log_norm).astype(np.float32)

    def predict(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.argmax(self.predict_proba(features, block_mask=block_mask), axis=1).astype(np.int64)

    def score(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> float:
        return float(np.mean(self.score_samples(features, block_mask=block_mask)))

    def _n_parameters(self, n_features: int) -> int:
        return (int(self.n_components) - 1) + 2 * int(self.n_components) * int(n_features)

    def bic(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> float:
        matrix = np.asarray(features, dtype=np.float64)
        ll = float(self.score_samples(matrix, block_mask=block_mask).sum())
        return -2.0 * ll + self._n_parameters(matrix.shape[1]) * np.log(max(matrix.shape[0], 1))

    def aic(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> float:
        matrix = np.asarray(features, dtype=np.float64)
        ll = float(self.score_samples(matrix, block_mask=block_mask).sum())
        return -2.0 * ll + 2.0 * self._n_parameters(matrix.shape[1])
