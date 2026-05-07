from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch

from cluster.backends.torch_gmm_backend import TorchGaussianMixture
from cluster.backends.torch_metrics import torch_silhouette_score_chunked


@dataclass
class TorchKMeansModel:
    centers_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = torch.as_tensor(np.asarray(X, dtype=np.float32))
        centers = torch.as_tensor(self.centers_.astype(np.float32))
        distances = torch.cdist(features, centers)
        return torch.argmin(distances, dim=1).cpu().numpy().astype(np.int64)


class TorchBackend:
    name = "torch"

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def _kmeans(
        self,
        X: np.ndarray,
        *,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 42,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, TorchKMeansModel]:
        features = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        best_labels = None
        best_centers = None
        best_inertia = float("inf")
        for run in range(max(int(n_init), 1)):
            generator = torch.Generator(device=features.device)
            generator.manual_seed(int(random_state) + run * 997)
            indices = torch.randperm(features.shape[0], generator=generator, device=features.device)[: int(n_clusters)]
            centers = features[indices].clone()
            previous_inertia = None
            labels = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
            for _ in range(int(max_iter)):
                distances = torch.cdist(features, centers)
                labels = torch.argmin(distances, dim=1)
                new_centers = centers.clone()
                for cluster_id in range(int(n_clusters)):
                    mask = labels == cluster_id
                    if bool(mask.any()):
                        new_centers[cluster_id] = features[mask].mean(dim=0)
                inertia = float(torch.sum((features - new_centers[labels]) ** 2).detach().cpu())
                if previous_inertia is not None and abs(previous_inertia - inertia) < float(tol):
                    centers = new_centers
                    break
                previous_inertia = inertia
                centers = new_centers
            inertia = float(torch.sum((features - centers[labels]) ** 2).detach().cpu())
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.detach().cpu().numpy().astype(np.int64)
                best_centers = centers.detach().cpu().numpy().astype(np.float32)
        if best_labels is None or best_centers is None:
            raise RuntimeError("Torch KMeans failed to produce labels.")
        return best_labels, TorchKMeansModel(centers_=best_centers)

    def fit_predict(
        self,
        X: np.ndarray,
        algorithm: str,
        n_clusters: int | None = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Any]:
        algo = str(algorithm).strip().lower()
        if n_clusters is None:
            raise ValueError(f"n_clusters is required for torch {algo}.")
        if algo == "gmm":
            model = TorchGaussianMixture(
                n_components=int(n_clusters),
                covariance_type=str(kwargs.get("covariance_type", "diag")),
                n_init=int(kwargs.get("n_init", 10)),
                max_iter=int(kwargs.get("max_iter", 300)),
                tol=float(kwargs.get("tol", 1e-4)),
                reg_covar=float(kwargs.get("reg_covar", 1e-6)),
                random_state=int(kwargs.get("random_state", 42)),
                device=self.device,
            )
            labels = model.fit_predict(X)
            return labels.astype(np.int64), model
        if algo == "kmeans":
            return self._kmeans(
                X,
                n_clusters=int(n_clusters),
                n_init=int(kwargs.get("n_init", 10)),
                max_iter=int(kwargs.get("max_iter", 300)),
                random_state=int(kwargs.get("random_state", 42)),
                tol=float(kwargs.get("tol", 1e-4)),
            )
        raise ValueError(f"Unsupported torch clustering algorithm: {algorithm}")

    def score_silhouette(self, X: np.ndarray, labels: np.ndarray, **kwargs: Any) -> float:
        return torch_silhouette_score_chunked(
            X,
            labels,
            chunk_size=int(kwargs.get("chunk_size", 4096)),
            device=self.device,
        )

    def pairwise_distances(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        features = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        return torch.cdist(features, features).detach().cpu().numpy().astype(np.float32)

