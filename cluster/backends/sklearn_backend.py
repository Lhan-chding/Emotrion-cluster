from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score

from cluster.backends.gmm_convergence import fit_gaussian_mixture_robust


class SklearnBackend:
    name = "sklearn"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def fit_predict(
        self,
        X: np.ndarray,
        algorithm: str,
        n_clusters: int | None = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Any]:
        algo = str(algorithm).strip().lower()
        features = np.asarray(X, dtype=np.float32)
        if algo == "kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters is required for kmeans.")
            model = KMeans(
                n_clusters=int(n_clusters),
                n_init=int(kwargs.get("n_init", 10)),
                max_iter=int(kwargs.get("max_iter", 300)),
                random_state=int(kwargs.get("random_state", 42)),
            )
            labels = model.fit_predict(features)
            return labels.astype(np.int64), model
        if algo == "gmm":
            if n_clusters is None:
                raise ValueError("n_clusters is required for gmm.")
            model = fit_gaussian_mixture_robust(
                features,
                n_components=int(n_clusters),
                covariance_type=str(kwargs.get("covariance_type", "full")),
                reg_covar=float(kwargs.get("reg_covar", 1e-5)),
                n_init=int(kwargs.get("n_init", 10)),
                max_iter=int(kwargs.get("max_iter", 300)),
                tol=float(kwargs.get("tol", 1e-3)),
                random_state=int(kwargs.get("random_state", 42)),
                require_converged=True,
                context="sklearn backend GMM",
            )
            labels = model.predict(features)
            return labels.astype(np.int64), model
        if algo == "hdbscan":
            try:
                from sklearn.cluster import HDBSCAN
            except ImportError as exc:
                raise RuntimeError("sklearn HDBSCAN is not available in this scikit-learn build.") from exc
            model = HDBSCAN(
                min_cluster_size=int(kwargs.get("min_cluster_size", 20)),
                min_samples=kwargs.get("min_samples"),
            )
            labels = model.fit_predict(features)
            return labels.astype(np.int64), model
        raise ValueError(f"Unsupported sklearn clustering algorithm: {algorithm}")

    def score_silhouette(self, X: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float:
        features = np.asarray(X, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if len(np.unique(y)) < 2:
            return float("nan")
        sample_size = kwargs.get("sample_size")
        if sample_size is not None and int(sample_size) > 0 and int(sample_size) < features.shape[0]:
            return float(
                silhouette_score(
                    features,
                    y,
                    sample_size=int(sample_size),
                    random_state=int(kwargs.get("random_state", 42)),
                )
            )
        return float(silhouette_score(features, y))

    def pairwise_distances(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return pairwise_distances(np.asarray(X, dtype=np.float32), metric=kwargs.get("metric", "euclidean"))
