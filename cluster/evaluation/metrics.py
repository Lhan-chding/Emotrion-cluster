from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score


BlockSlice = Tuple[int, int]


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


def masked_pairwise_distances(
    features: np.ndarray,
    *,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
) -> np.ndarray:
    """Pairwise Euclidean distances using only commonly observed feature blocks."""

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"features must be 2D, got {matrix.shape}.")
    slices = _as_block_slices(block_slices, matrix.shape[1])
    mask = np.asarray(block_mask, dtype=bool)
    if mask.shape != (matrix.shape[0], len(slices)):
        raise ValueError(f"block_mask must have shape [{matrix.shape[0]}, {len(slices)}], got {mask.shape}.")

    n_samples = matrix.shape[0]
    squared = np.zeros((n_samples, n_samples), dtype=np.float64)
    observed_dims = np.zeros((n_samples, n_samples), dtype=np.float64)
    for block_id, (start, stop) in enumerate(slices):
        observed_pair = np.logical_and.outer(mask[:, block_id], mask[:, block_id])
        if not observed_pair.any():
            continue
        block_sq = pairwise_distances(matrix[:, start:stop], metric="sqeuclidean")
        squared += np.where(observed_pair, block_sq, 0.0)
        observed_dims += np.where(observed_pair, float(stop - start), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = squared / observed_dims * float(matrix.shape[1])
    no_overlap = observed_dims <= 0.0
    finite = normalized[np.isfinite(normalized) & ~no_overlap]
    fallback = float(finite.max()) if finite.size else 1.0
    normalized = np.where(no_overlap, fallback, normalized)
    normalized = np.maximum(normalized, 0.0)
    distances = np.sqrt(normalized, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def masked_silhouette_score(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
) -> float:
    """Silhouette score on a precomputed masked distance matrix."""

    y = np.asarray(labels, dtype=np.int64)
    if np.unique(y).size < 2 or y.size < 3:
        return 0.0
    distances = masked_pairwise_distances(features, block_mask=block_mask, block_slices=block_slices)
    try:
        return float(silhouette_score(distances, y, metric="precomputed"))
    except Exception:
        return 0.0
