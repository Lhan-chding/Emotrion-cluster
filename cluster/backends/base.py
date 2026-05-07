from __future__ import annotations

from typing import Any, Protocol, Tuple

import numpy as np


class ClusterBackend(Protocol):
    name: str

    @classmethod
    def is_available(cls) -> bool:
        ...

    def fit_predict(
        self,
        X: np.ndarray,
        algorithm: str,
        n_clusters: int | None = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Any]:
        ...

    def score_silhouette(self, X: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float:
        ...

    def pairwise_distances(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        ...


def resolve_cluster_backend(name: str = "auto", *, device: str = "cpu") -> ClusterBackend:
    requested = str(name or "auto").strip().lower()
    device_name = str(device or "cpu").strip().lower()

    from cluster.backends.cuml_backend import CuMLBackend
    from cluster.backends.sklearn_backend import SklearnBackend
    from cluster.backends.torch_backend import TorchBackend

    if requested == "auto":
        if device_name.startswith("cuda") and CuMLBackend.is_available():
            return CuMLBackend()
        if device_name.startswith("cuda") and TorchBackend.is_available():
            return TorchBackend(device=device_name)
        return SklearnBackend()
    if requested == "sklearn":
        return SklearnBackend()
    if requested == "torch":
        if not TorchBackend.is_available():
            raise RuntimeError("PyTorch backend requested but torch is not available.")
        return TorchBackend(device=device_name)
    if requested == "cuml":
        if not CuMLBackend.is_available():
            raise RuntimeError("cuML backend requested but RAPIDS cuML/cupy are not available.")
        return CuMLBackend()
    raise ValueError("Unsupported cluster backend '{}'. Expected auto, sklearn, torch, or cuml.".format(name))

