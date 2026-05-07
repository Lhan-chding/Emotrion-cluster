from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


class TorchGaussianMixture:
    """Diagonal-covariance Gaussian mixture with a sklearn-like API."""

    def __init__(
        self,
        n_components: int,
        *,
        covariance_type: str = "diag",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if str(covariance_type).lower() != "diag":
            raise ValueError("TorchGaussianMixture currently supports covariance_type='diag' only.")
        self.n_components = int(n_components)
        self.covariance_type = "diag"
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.random_state = int(random_state)
        self.device = str(device)
        self.dtype = dtype
        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.lower_bound_: float = float("-inf")
        self.converged_: bool = False
        self.n_iter_: int = 0

    def _tensor(self, X: np.ndarray | torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.dtype
        if torch.is_tensor(X):
            return X.to(device=self.device, dtype=dtype)
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        return torch.as_tensor(np.asarray(X, dtype=np_dtype), device=self.device, dtype=dtype)

    def _initial_means(self, X: torch.Tensor, run: int) -> torch.Tensor:
        generator = torch.Generator(device=X.device)
        generator.manual_seed(self.random_state + run * 1009)
        indices = torch.randperm(X.shape[0], generator=generator, device=X.device)[: self.n_components]
        if indices.numel() < self.n_components:
            repeats = self.n_components - int(indices.numel())
            indices = torch.cat([indices, indices[:1].repeat(repeats)])
        return X[indices].clone()

    def _estimate_log_gaussian_prob(self, X: torch.Tensor, means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
        n_samples = int(X.shape[0])
        n_components = int(means.shape[0])
        log_prob = torch.empty((n_samples, n_components), device=X.device, dtype=X.dtype)
        log_det = torch.log(variances).sum(dim=1)
        constant = X.shape[1] * math.log(2.0 * math.pi)
        for cluster_id in range(n_components):
            diff = X - means[cluster_id]
            mahal = (diff * diff / variances[cluster_id]).sum(dim=1)
            log_prob[:, cluster_id] = -0.5 * (constant + log_det[cluster_id] + mahal)
        return log_prob

    def _fit_once(self, X: torch.Tensor, run: int):
        n_samples, n_features = X.shape
        means = self._initial_means(X, run)
        global_var = torch.var(X, dim=0, unbiased=False).clamp_min(self.reg_covar)
        variances = global_var.repeat(self.n_components, 1)
        weights = torch.full((self.n_components,), 1.0 / self.n_components, device=X.device, dtype=X.dtype)
        previous_lower = torch.tensor(float("-inf"), device=X.device, dtype=X.dtype)
        current_lower = previous_lower
        converged = False
        n_iter = 0

        for iteration in range(1, self.max_iter + 1):
            log_prob = self._estimate_log_gaussian_prob(X, means, variances) + torch.log(weights.clamp_min(1e-12))[None, :]
            log_norm = torch.logsumexp(log_prob, dim=1)
            responsibilities = torch.softmax(log_prob, dim=1)
            nk = responsibilities.sum(dim=0).clamp_min(1e-8)
            weights = nk / float(n_samples)
            means = responsibilities.T @ X / nk[:, None]
            variance_rows = []
            for cluster_id in range(self.n_components):
                centered = X - means[cluster_id]
                weighted_square = responsibilities[:, cluster_id : cluster_id + 1] * centered * centered
                variance_rows.append(weighted_square.sum(dim=0) / nk[cluster_id])
            variances = torch.stack(variance_rows, dim=0)
            variances = variances.clamp_min(self.reg_covar)

            lower = log_norm.mean()
            current_lower = lower
            change = torch.abs(lower - previous_lower)
            n_iter = iteration
            if torch.isfinite(previous_lower) and float(change.detach().cpu()) < self.tol:
                converged = True
                break
            previous_lower = lower

        return means, variances, weights, float(current_lower.detach().cpu()), converged, n_iter

    def fit(self, X: np.ndarray | torch.Tensor):
        features = self._tensor(X)
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {tuple(features.shape)}.")
        if features.shape[0] < self.n_components:
            raise ValueError("n_samples must be >= n_components.")

        best = None
        for run in range(max(self.n_init, 1)):
            current = self._fit_once(features, run)
            if best is None or current[3] > best[3]:
                best = current
        if best is None:
            raise RuntimeError("Torch GMM fit did not produce a model.")

        means, variances, weights, lower, converged, n_iter = best
        self.means_ = means.detach().cpu().numpy().astype(np.float32)
        self.covariances_ = variances.detach().cpu().numpy().astype(np.float32)
        self.weights_ = weights.detach().cpu().numpy().astype(np.float32)
        self.lower_bound_ = float(lower)
        self.converged_ = bool(converged)
        self.n_iter_ = int(n_iter)
        return self

    def _require_fit(self):
        if self.means_ is None or self.covariances_ is None or self.weights_ is None:
            raise RuntimeError("TorchGaussianMixture must be fit before prediction.")

    def _estimate_weighted_log_prob(
        self,
        X: np.ndarray | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self._require_fit()
        dtype = dtype or self.dtype
        features = self._tensor(X, dtype=dtype)
        means = self._tensor(self.means_, dtype=dtype)
        variances = self._tensor(self.covariances_, dtype=dtype)
        weights = self._tensor(self.weights_, dtype=dtype)
        min_weight = 1e-300 if dtype == torch.float64 else 1e-12
        return self._estimate_log_gaussian_prob(features, means, variances) + torch.log(weights.clamp_min(min_weight))[None, :]

    def score_samples(self, X: np.ndarray | torch.Tensor, *, dtype: torch.dtype | None = None) -> np.ndarray:
        log_prob = self._estimate_weighted_log_prob(X, dtype=dtype)
        return torch.logsumexp(log_prob, dim=1).detach().cpu().numpy().astype(np.float64)

    def score(self, X: np.ndarray | torch.Tensor) -> float:
        return float(np.mean(self.score_samples(X)))

    def predict_proba(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        log_prob = self._estimate_weighted_log_prob(X)
        return torch.softmax(log_prob, dim=1).detach().cpu().numpy().astype(np.float32)

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def fit_predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.fit(X).predict(X)

    def _n_parameters(self, n_features: int) -> int:
        return (self.n_components - 1) + self.n_components * n_features + self.n_components * n_features

    def bic(self, X: np.ndarray | torch.Tensor) -> float:
        features = self._tensor(X, dtype=torch.float64)
        log_prob = self._estimate_weighted_log_prob(features, dtype=torch.float64)
        log_likelihood = float(torch.logsumexp(log_prob, dim=1).sum().detach().cpu())
        return -2.0 * log_likelihood + self._n_parameters(int(features.shape[1])) * math.log(int(features.shape[0]))

    def aic(self, X: np.ndarray | torch.Tensor) -> float:
        features = self._tensor(X, dtype=torch.float64)
        log_prob = self._estimate_weighted_log_prob(features, dtype=torch.float64)
        log_likelihood = float(torch.logsumexp(log_prob, dim=1).sum().detach().cpu())
        return -2.0 * log_likelihood + 2.0 * self._n_parameters(int(features.shape[1]))
