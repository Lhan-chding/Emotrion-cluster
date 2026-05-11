from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cluster.backends.gmm_convergence import fit_gaussian_mixture_robust
from cluster.backends.masked_diag_gmm import MaskedDiagonalGMM
from cluster.evaluation.metrics import masked_silhouette_score


BlockSlice = Tuple[int, int]


def _as_block_slices(block_slices: Sequence[Sequence[int]], n_features: int) -> List[BlockSlice]:
    slices: List[BlockSlice] = []
    for raw in block_slices:
        start, stop = int(raw[0]), int(raw[1])
        if start < 0 or stop <= start or stop > int(n_features):
            raise ValueError(f"Invalid block slice ({start}, {stop}) for feature_dim={n_features}.")
        slices.append((start, stop))
    if len(slices) not in {2, 3}:
        raise ValueError("MacroMicroClusterer requires two or three blocks: consensus, tension[, metadata].")
    return slices


def _normalize_block_mask(block_mask: Optional[np.ndarray], n_samples: int, n_blocks: int) -> np.ndarray:
    if block_mask is None:
        return np.ones((n_samples, n_blocks), dtype=bool)
    mask = np.asarray(block_mask, dtype=bool)
    if mask.shape != (n_samples, n_blocks):
        raise ValueError(f"block_mask must have shape [{n_samples}, {n_blocks}], got {mask.shape}.")
    fixed = np.array(mask, copy=True)
    empty_rows = ~fixed.any(axis=1)
    if empty_rows.any():
        fixed[empty_rows, 0] = True
    return fixed


@dataclass
class MacroMicroClusterer:
    """Two-stage macro/micro clusterer for diff-aware three-block features.

    The first feature block defines macro affect regions. Within each macro,
    micro clusters are fit on tension, metadata residual, and local affect
    residual blocks using masked diagonal GMM likelihood.
    """

    macro_k: int
    block_slices: Sequence[Sequence[int]]
    covariance_type: str = "diag"
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 100
    reg_covar: float = 1e-5
    micro_k_min: int = 1
    micro_k_max: int = 5
    min_cluster_size: int = 20

    def fit(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> "MacroMicroClusterer":
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"features must be 2D, got {matrix.shape}.")
        self.block_slices_ = _as_block_slices(self.block_slices, matrix.shape[1])
        mask = _normalize_block_mask(block_mask, matrix.shape[0], len(self.block_slices_))
        consensus, _tension, metadata = self._split_blocks(matrix)

        self.macro_model_ = fit_gaussian_mixture_robust(
            consensus,
            n_components=int(self.macro_k),
            covariance_type=str(self.covariance_type),
            reg_covar=float(self.reg_covar),
            n_init=int(self.n_init),
            max_iter=max(20, int(self.max_iter)),
            random_state=int(self.random_state),
            require_converged=True,
            context=f"macro GMM macro_k={int(self.macro_k)}",
        )
        macro_labels = self.macro_model_.predict(consensus).astype(np.int64)
        self.macro_labels_ = macro_labels

        self.consensus_centroids_ = np.zeros((int(self.macro_k), consensus.shape[1]), dtype=np.float32)
        self.metadata_centroids_ = np.zeros((int(self.macro_k), metadata.shape[1]), dtype=np.float32)
        self.micro_models_: Dict[int, MaskedDiagonalGMM] = {}
        self.micro_k_by_macro_: Dict[int, int] = {}
        self.micro_label_offsets_: Dict[int, Dict[int, int]] = {}
        self.label_names: Dict[int, str] = {}
        self.info: Dict[str, Any] = {"macro_details": {}, "micro_details": {}}

        labels = np.full(matrix.shape[0], -1, dtype=np.int64)
        global_label = 0
        for macro_id in range(int(self.macro_k)):
            macro_mask = macro_labels == macro_id
            macro_indices = np.where(macro_mask)[0]
            if macro_indices.size == 0:
                self.micro_k_by_macro_[macro_id] = 0
                continue

            self.consensus_centroids_[macro_id] = consensus[macro_indices].mean(axis=0)
            metadata_observed = mask[macro_indices, 2] if mask.shape[1] > 2 else np.zeros(macro_indices.size, dtype=bool)
            if metadata.shape[1] > 0 and metadata_observed.any():
                self.metadata_centroids_[macro_id] = metadata[macro_indices][metadata_observed].mean(axis=0)

            micro_features = self._micro_features_for_macro(matrix[macro_indices], macro_id)
            micro_mask = self._micro_block_mask(mask[macro_indices])
            micro_k, micro_model, micro_labels, micro_sil = self._fit_micro(micro_features, micro_mask, macro_id)
            self.micro_k_by_macro_[macro_id] = int(micro_k)
            offsets: Dict[int, int] = {}
            for micro_id in range(int(micro_k)):
                offsets[micro_id] = global_label
                local_indices = macro_indices[micro_labels == micro_id]
                labels[local_indices] = global_label
                suffix = chr(ord("a") + micro_id) if micro_id < 26 else str(micro_id)
                self.label_names[global_label] = f"M{macro_id + 1}-{suffix}" if micro_k > 1 else f"M{macro_id + 1}"
                global_label += 1
            self.micro_label_offsets_[macro_id] = offsets
            if micro_model is not None:
                self.micro_models_[macro_id] = micro_model
            self.info["macro_details"][f"macro_{macro_id}"] = {
                "size": int(macro_indices.size),
                "micro_k": int(micro_k),
            }
            self.info["micro_details"][f"macro_{macro_id}"] = {
                "micro_k": int(micro_k),
                "micro_silhouette": float(micro_sil),
                "sizes": [int((micro_labels == item).sum()) for item in range(int(micro_k))],
            }

        if np.any(labels < 0):
            raise RuntimeError("MacroMicroClusterer failed to assign every sample.")
        self.labels_ = labels
        self.n_components = int(global_label)
        self.info.update(
            {
                "macro_k": int(self.macro_k),
                "total_clusters": int(self.n_components),
                "min_cluster_size": int(np.bincount(labels, minlength=int(self.n_components)).min()),
                "label_names": {str(key): value for key, value in self.label_names.items()},
            }
        )
        return self

    def predict(self, features: np.ndarray, block_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if not hasattr(self, "macro_model_"):
            raise RuntimeError("MacroMicroClusterer must be fit before prediction.")
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"features must be 2D, got {matrix.shape}.")
        mask = _normalize_block_mask(block_mask, matrix.shape[0], len(self.block_slices_))
        consensus, _tension, _metadata = self._split_blocks(matrix)
        macro_labels = self.macro_model_.predict(consensus).astype(np.int64)
        labels = np.full(matrix.shape[0], -1, dtype=np.int64)

        for macro_id in range(int(self.macro_k)):
            macro_mask = macro_labels == macro_id
            if not macro_mask.any():
                continue
            offsets = self.micro_label_offsets_.get(macro_id, {0: 0})
            micro_k = int(self.micro_k_by_macro_.get(macro_id, 1))
            if micro_k <= 1 or macro_id not in self.micro_models_:
                labels[macro_mask] = offsets.get(0, 0)
                continue
            micro_features = self._micro_features_for_macro(matrix[macro_mask], macro_id)
            micro_mask = self._micro_block_mask(mask[macro_mask])
            micro_labels = self.micro_models_[macro_id].predict(micro_features, block_mask=micro_mask)
            for micro_id, global_label in offsets.items():
                labels[np.where(macro_mask)[0][micro_labels == micro_id]] = int(global_label)

        if np.any(labels < 0):
            raise RuntimeError("MacroMicroClusterer failed to predict every sample.")
        return labels.astype(np.int64)

    def _split_blocks(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        blocks = [matrix[:, start:stop] for start, stop in self.block_slices_]
        if len(blocks) == 2:
            metadata = np.zeros((matrix.shape[0], 0), dtype=np.float32)
            return blocks[0], blocks[1], metadata
        return blocks[0], blocks[1], blocks[2]

    def _micro_features_for_macro(self, matrix: np.ndarray, macro_id: int) -> np.ndarray:
        consensus, tension, metadata = self._split_blocks(matrix)
        local_consensus = consensus - self.consensus_centroids_[int(macro_id)].reshape(1, -1)
        metadata_residual = metadata - self.metadata_centroids_[int(macro_id)].reshape(1, -1)
        return np.concatenate([tension, metadata_residual, local_consensus], axis=1).astype(np.float32)

    def _micro_block_mask(self, block_mask: np.ndarray) -> np.ndarray:
        if block_mask.shape[1] == 2:
            micro_mask = np.stack([block_mask[:, 1], block_mask[:, 0]], axis=1).astype(bool)
            empty_rows = ~micro_mask.any(axis=1)
            if empty_rows.any():
                micro_mask[empty_rows, 1] = True
            return micro_mask
        micro_mask = np.stack([block_mask[:, 1], block_mask[:, 2], block_mask[:, 0]], axis=1).astype(bool)
        empty_rows = ~micro_mask.any(axis=1)
        if empty_rows.any():
            micro_mask[empty_rows, 2] = True
        return micro_mask

    def _micro_block_slices(self) -> List[BlockSlice]:
        consensus_start, consensus_stop = self.block_slices_[0]
        tension_start, tension_stop = self.block_slices_[1]
        tension_dim = int(tension_stop - tension_start)
        consensus_dim = int(consensus_stop - consensus_start)
        if len(self.block_slices_) == 2:
            return [
                (0, tension_dim),
                (tension_dim, tension_dim + consensus_dim),
            ]
        metadata_start, metadata_stop = self.block_slices_[2]
        metadata_dim = int(metadata_stop - metadata_start)
        return [
            (0, tension_dim),
            (tension_dim, tension_dim + metadata_dim),
            (tension_dim + metadata_dim, tension_dim + metadata_dim + consensus_dim),
        ]

    def _fit_micro(
        self,
        micro_features: np.ndarray,
        micro_mask: np.ndarray,
        macro_id: int,
    ) -> Tuple[int, Optional[MaskedDiagonalGMM], np.ndarray, float]:
        n_samples = int(micro_features.shape[0])
        min_size = max(int(self.min_cluster_size), 1)
        max_k_by_size = max(1, n_samples // min_size)
        upper = min(int(self.micro_k_max), max_k_by_size, n_samples)
        candidate_ks = [1]
        lower = max(2, int(self.micro_k_min))
        if upper >= lower:
            candidate_ks.extend(range(lower, upper + 1))

        best_k = 1
        best_model: Optional[MaskedDiagonalGMM] = None
        best_labels = np.zeros(n_samples, dtype=np.int64)
        best_score = 0.0
        for k in candidate_ks:
            if k == 1:
                continue
            model = MaskedDiagonalGMM(
                n_components=int(k),
                block_slices=self._micro_block_slices(),
                covariance_type="diag",
                random_state=int(self.random_state) + int(macro_id) * 1000 + int(k),
                n_init=max(1, int(self.n_init)),
                max_iter=int(self.max_iter),
                reg_covar=float(self.reg_covar),
            ).fit(micro_features, block_mask=micro_mask)
            labels = model.predict(micro_features, block_mask=micro_mask).astype(np.int64)
            sizes = np.bincount(labels, minlength=int(k))
            if sizes.min() < min_size:
                continue
            score = masked_silhouette_score(
                micro_features,
                labels,
                block_mask=micro_mask,
                block_slices=self._micro_block_slices(),
            )
            if score > best_score:
                best_k = int(k)
                best_model = model
                best_labels = labels
                best_score = float(score)
        return best_k, best_model, best_labels, best_score
