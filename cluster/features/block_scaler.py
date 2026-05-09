from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


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


@dataclass
class BlockwiseObservedScaler:
    """Standardize each feature block using only rows where that block is observed."""

    block_slices: Sequence[Sequence[int]]
    eps: float = 1e-6

    def fit(self, features: np.ndarray, block_mask: np.ndarray) -> "BlockwiseObservedScaler":
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"features must be 2D, got {matrix.shape}.")
        self.block_slices_ = _as_block_slices(self.block_slices, matrix.shape[1])
        mask = np.asarray(block_mask, dtype=bool)
        if mask.shape != (matrix.shape[0], len(self.block_slices_)):
            raise ValueError(f"block_mask must have shape [{matrix.shape[0]}, {len(self.block_slices_)}], got {mask.shape}.")

        self.mean_ = np.zeros(matrix.shape[1], dtype=np.float32)
        self.scale_ = np.ones(matrix.shape[1], dtype=np.float32)
        counts: list[int] = []
        for block_id, (start, stop) in enumerate(self.block_slices_):
            observed = mask[:, block_id]
            counts.append(int(observed.sum()))
            if not observed.any():
                continue
            block = matrix[observed, start:stop]
            mean = block.mean(axis=0)
            scale = block.std(axis=0)
            scale = np.where(scale <= float(self.eps), 1.0, scale)
            self.mean_[start:stop] = mean.astype(np.float32)
            self.scale_[start:stop] = scale.astype(np.float32)
        self.observed_counts_ = np.asarray(counts, dtype=np.int64)
        return self

    def transform(self, features: np.ndarray, block_mask: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("BlockwiseObservedScaler must be fit before transform.")
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"features must have shape [N, {self.mean_.shape[0]}], got {matrix.shape}.")
        mask = np.asarray(block_mask, dtype=bool)
        if mask.shape != (matrix.shape[0], len(self.block_slices_)):
            raise ValueError(f"block_mask must have shape [{matrix.shape[0]}, {len(self.block_slices_)}], got {mask.shape}.")
        transformed = ((matrix - self.mean_.reshape(1, -1)) / self.scale_.reshape(1, -1)).astype(np.float32)
        for block_id, (start, stop) in enumerate(self.block_slices_):
            transformed[~mask[:, block_id], start:stop] = 0.0
        return transformed

    def fit_transform(self, features: np.ndarray, block_mask: np.ndarray) -> np.ndarray:
        return self.fit(features, block_mask=block_mask).transform(features, block_mask=block_mask)
