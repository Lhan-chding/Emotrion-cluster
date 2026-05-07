from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch


def _as_tensor(X: np.ndarray | torch.Tensor, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(X):
        return X.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(X), device=device, dtype=dtype)


def torch_pairwise_distances_chunked(
    X: np.ndarray | torch.Tensor,
    *,
    chunk_size: int = 4096,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Iterator[Tuple[int, torch.Tensor]]:
    features = _as_tensor(X, device=device, dtype=dtype)
    n_rows = int(features.shape[0])
    step = max(int(chunk_size), 1)
    for start in range(0, n_rows, step):
        yield start, torch.cdist(features[start : start + step], features)


def torch_silhouette_score_chunked(
    X: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    chunk_size: int = 4096,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> float:
    features = _as_tensor(X, device=device, dtype=dtype)
    y = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    unique = torch.unique(y)
    if int(unique.numel()) < 2:
        return float("nan")

    n_rows = int(features.shape[0])
    if n_rows <= 1:
        return float("nan")

    cluster_masks = [(y == cluster_id) for cluster_id in unique]
    silhouettes = torch.zeros(n_rows, device=features.device, dtype=dtype)

    for start, distances in torch_pairwise_distances_chunked(
        features,
        chunk_size=chunk_size,
        device=str(features.device),
        dtype=dtype,
    ):
        end = start + distances.shape[0]
        chunk_labels = y[start:end]
        a_values = torch.zeros(distances.shape[0], device=features.device, dtype=dtype)
        b_values = torch.full((distances.shape[0],), float("inf"), device=features.device, dtype=dtype)
        singleton = torch.zeros(distances.shape[0], device=features.device, dtype=torch.bool)

        for cluster_id, mask in zip(unique, cluster_masks):
            in_cluster = chunk_labels == cluster_id
            count = int(mask.sum().item())
            if count <= 0:
                continue
            mean_dist = distances[:, mask].mean(dim=1)
            b_values = torch.where(~in_cluster, torch.minimum(b_values, mean_dist), b_values)
            if bool(in_cluster.any()):
                if count <= 1:
                    singleton = torch.where(in_cluster, torch.ones_like(singleton), singleton)
                    a_values = torch.where(in_cluster, torch.zeros_like(a_values), a_values)
                else:
                    same_sum = distances[:, mask].sum(dim=1)
                    a_cluster = same_sum / float(count - 1)
                    a_values = torch.where(in_cluster, a_cluster, a_values)

        denom = torch.maximum(a_values, b_values)
        scores = torch.where(denom > 0, (b_values - a_values) / denom, torch.zeros_like(denom))
        scores = torch.where(singleton, torch.zeros_like(scores), scores)
        silhouettes[start:end] = scores

    return float(silhouettes.mean().detach().cpu().item())

