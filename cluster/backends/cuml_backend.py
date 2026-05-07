from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class CuMLBackend:
    name = "cuml"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cupy  # noqa: F401
            import cuml  # noqa: F401
        except Exception:
            return False
        return True

    def fit_predict(
        self,
        X: np.ndarray,
        algorithm: str,
        n_clusters: int | None = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Any]:
        import cupy as cp

        algo = str(algorithm).strip().lower()
        X_gpu = cp.asarray(X, dtype=cp.float32)
        if algo == "kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters is required for kmeans.")
            from cuml.cluster import KMeans

            model = KMeans(
                n_clusters=int(n_clusters),
                n_init=int(kwargs.get("n_init", 64)),
                max_iter=int(kwargs.get("max_iter", 300)),
                random_state=int(kwargs.get("random_state", 42)),
            )
            labels = model.fit_predict(X_gpu)
            return cp.asnumpy(labels).astype(np.int64), model
        if algo == "hdbscan":
            from cuml.cluster import HDBSCAN

            model = HDBSCAN(
                min_cluster_size=int(kwargs.get("min_cluster_size", 20)),
                min_samples=kwargs.get("min_samples"),
            )
            labels = model.fit_predict(X_gpu)
            return cp.asnumpy(labels).astype(np.int64), model
        if algo == "spectral":
            if n_clusters is None:
                raise ValueError("n_clusters is required for spectral clustering.")
            from cuml.cluster import SpectralClustering

            model = SpectralClustering(
                n_clusters=int(n_clusters),
                random_state=int(kwargs.get("random_state", 42)),
            )
            labels = model.fit_predict(X_gpu)
            return cp.asnumpy(labels).astype(np.int64), model
        if algo == "gmm":
            raise ValueError("cuML backend does not provide GMM here; use cluster_backend=torch or sklearn.")
        raise ValueError(f"Unsupported cuML clustering algorithm: {algorithm}")

    def score_silhouette(self, X: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float:
        import cupy as cp
        from cuml.metrics.cluster import silhouette_score

        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(labels)
        return float(silhouette_score(X_gpu, y_gpu))

    def pairwise_distances(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        import cupy as cp
        from cuml.metrics import pairwise_distances

        X_gpu = cp.asarray(X, dtype=cp.float32)
        return cp.asnumpy(pairwise_distances(X_gpu, metric=kwargs.get("metric", "euclidean")))

