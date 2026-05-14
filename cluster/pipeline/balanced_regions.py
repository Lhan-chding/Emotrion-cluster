from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from cluster.backends.torch_metrics import torch_silhouette_score_chunked

from cluster.pipeline.k_selection import (
    KSearchResult,
    KSelectionConfig,
    compute_affect_purity_metrics,
    compute_overlap_gate_metrics,
)


class BalancedVARegionClusterer:
    """Size-constrained region clustering on a 2-D VA plane."""

    def __init__(
        self,
        n_components: int,
        *,
        min_cluster_size: int,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 100,
        backend: str = "sklearn",
        device: str = "cpu",
    ) -> None:
        self.n_components = int(n_components)
        self.min_cluster_size = int(min_cluster_size)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.backend = str(backend or "sklearn").strip().lower()
        self.device = str(device or "cpu")
        self.actual_backend_ = "balanced_va_region_kmeans"

    def fit(self, values: np.ndarray) -> "BalancedVARegionClusterer":
        matrix = self._primary_va(values)
        if matrix.shape[0] < self.n_components:
            raise ValueError("n_components exceeds sample count.")
        if matrix.shape[0] < self.n_components * self.min_cluster_size:
            raise ValueError(
                f"Cannot fit {self.n_components} clusters with min_cluster_size={self.min_cluster_size} "
                f"from {matrix.shape[0]} samples."
            )
        if self.backend == "torch":
            return self._fit_torch(matrix)
        return self._fit_sklearn(matrix)

    def _fit_sklearn(self, matrix: np.ndarray) -> "BalancedVARegionClusterer":
        best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        rng = np.random.default_rng(self.random_state)
        for init_idx in range(max(1, self.n_init)):
            seed = int(rng.integers(0, 2**31 - 1))
            centers = KMeans(
                n_clusters=self.n_components,
                n_init=1,
                random_state=seed,
            ).fit(matrix).cluster_centers_.astype(np.float64)
            labels = np.full(matrix.shape[0], -1, dtype=np.int64)
            for _ in range(max(1, self.max_iter)):
                distances = self._distances(matrix, centers)
                new_labels = np.argmin(distances, axis=1).astype(np.int64)
                new_labels = self._repair_min_size(new_labels, distances)
                new_centers = np.vstack(
                    [
                        matrix[new_labels == cluster_id].mean(axis=0)
                        for cluster_id in range(self.n_components)
                    ]
                )
                if np.array_equal(new_labels, labels):
                    centers = new_centers
                    labels = new_labels
                    break
                centers = new_centers
                labels = new_labels
            distances = self._distances(matrix, centers)
            inertia = float(np.sum(distances[np.arange(matrix.shape[0]), labels]))
            if best is None or inertia < best[0]:
                best = (inertia, centers.copy(), labels.copy())

        if best is None:
            raise RuntimeError("BalancedVARegionClusterer failed to fit.")
        self.inertia_ = float(best[0])
        self.cluster_centers_ = best[1].astype(np.float64)
        self.labels_ = best[2].astype(np.int64)
        self.actual_backend_ = "balanced_va_region_kmeans"
        return self

    def _fit_torch(self, matrix: np.ndarray) -> "BalancedVARegionClusterer":
        device = torch.device(self.device if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        features = torch.as_tensor(matrix.astype(np.float32), device=device)
        best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        for init_idx in range(max(1, self.n_init)):
            generator = torch.Generator(device=features.device)
            generator.manual_seed(int(self.random_state) + 9973 * init_idx)
            indices = torch.randperm(features.shape[0], generator=generator, device=features.device)[: self.n_components]
            centers = features[indices].clone()
            labels = np.full(matrix.shape[0], -1, dtype=np.int64)
            for _ in range(max(1, self.max_iter)):
                distances_t = self._torch_distances(features, centers)
                new_labels = torch.argmin(distances_t, dim=1).detach().cpu().numpy().astype(np.int64)
                if np.bincount(new_labels, minlength=self.n_components).min() < self.min_cluster_size:
                    new_labels = self._repair_min_size(
                        new_labels,
                        distances_t.detach().cpu().numpy().astype(np.float64),
                    )
                labels_t = torch.as_tensor(new_labels, device=features.device, dtype=torch.long)
                new_centers = centers.clone()
                for cluster_id in range(self.n_components):
                    mask = labels_t == int(cluster_id)
                    if bool(mask.any()):
                        new_centers[cluster_id] = features[mask].mean(dim=0)
                if np.array_equal(new_labels, labels):
                    centers = new_centers
                    labels = new_labels
                    break
                centers = new_centers
                labels = new_labels
            labels_t = torch.as_tensor(labels, device=features.device, dtype=torch.long)
            final_distances = self._torch_distances(features, centers)
            inertia = float(final_distances[torch.arange(features.shape[0], device=features.device), labels_t].sum().detach().cpu().item())
            centers_np = centers.detach().cpu().numpy().astype(np.float64)
            if best is None or inertia < best[0]:
                best = (inertia, centers_np.copy(), labels.copy())
        if best is None:
            raise RuntimeError("BalancedVARegionClusterer torch backend failed to fit.")
        self.inertia_ = float(best[0])
        self.cluster_centers_ = best[1].astype(np.float64)
        self.labels_ = best[2].astype(np.int64)
        self.actual_backend_ = "balanced_va_region_torch_kmeans"
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        matrix = self._primary_va(values)
        distances = self._distances(matrix, self.cluster_centers_)
        return np.argmin(distances, axis=1).astype(np.int64)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = self._primary_va(values)
        distances = self._distances(matrix, self.cluster_centers_)
        logits = -distances
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    def _repair_min_size(self, labels: np.ndarray, distances: np.ndarray) -> np.ndarray:
        repaired = np.asarray(labels, dtype=np.int64).copy()
        sizes = np.bincount(repaired, minlength=self.n_components)
        while sizes.min() < self.min_cluster_size:
            deficit_clusters = np.where(sizes < self.min_cluster_size)[0]
            moved = False
            for target in deficit_clusters.tolist():
                need = int(self.min_cluster_size - sizes[target])
                for _ in range(need):
                    donors = np.where(sizes > self.min_cluster_size)[0]
                    if donors.size == 0:
                        break
                    donor_mask = np.isin(repaired, donors)
                    donor_indices = np.where(donor_mask)[0]
                    current = repaired[donor_indices]
                    penalty = distances[donor_indices, target] - distances[donor_indices, current]
                    order = np.argsort(penalty, kind="stable")
                    chosen = -1
                    for rel_idx in order.tolist():
                        idx = int(donor_indices[rel_idx])
                        source = int(repaired[idx])
                        if sizes[source] > self.min_cluster_size:
                            chosen = idx
                            break
                    if chosen < 0:
                        break
                    source = int(repaired[chosen])
                    repaired[chosen] = int(target)
                    sizes[source] -= 1
                    sizes[target] += 1
                    moved = True
            if not moved:
                break
        return repaired

    @staticmethod
    def _primary_va(values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError(f"VA region clustering requires [N, >=2] features, got {matrix.shape}.")
        return matrix[:, :2]

    @staticmethod
    def _distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
        diff = values[:, None, :] - centers[None, :, :]
        return np.sum(diff**2, axis=2)

    @staticmethod
    def _torch_distances(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        diff = values[:, None, :] - centers[None, :, :]
        return torch.sum(diff * diff, dim=2)


def _min_size_threshold(config: KSelectionConfig, n_samples: int) -> int:
    return max(
        int(config.min_cluster_size),
        int(np.ceil(float(config.min_cluster_size_ratio) * float(n_samples))),
    )


def _tension_diagnostics(tension: Optional[np.ndarray], labels: np.ndarray) -> Dict[str, float]:
    if tension is None:
        return {
            "tension_norm_mean": float("nan"),
            "tension_effect_ratio": float("nan"),
        }
    matrix = np.asarray(tension, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != labels.shape[0] or matrix.shape[1] == 0:
        return {
            "tension_norm_mean": float("nan"),
            "tension_effect_ratio": float("nan"),
        }
    norms = np.linalg.norm(matrix, axis=1)
    global_center = matrix.mean(axis=0)
    within = 0.0
    between = 0.0
    total = 0
    for cluster_id in np.unique(labels):
        group = matrix[labels == int(cluster_id)]
        if group.size == 0:
            continue
        center = group.mean(axis=0)
        within += float(np.sum((group - center.reshape(1, -1)) ** 2))
        between += float(group.shape[0] * np.sum((center - global_center) ** 2))
        total += int(group.shape[0])
    return {
        "tension_norm_mean": float(np.mean(norms)),
        "tension_effect_ratio": float(between / max(within, 1e-12)) if total else float("nan"),
    }


def _stability(
    primary_va: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    min_cluster_size: int,
    config: KSelectionConfig,
) -> Tuple[float, float]:
    runs = max(0, int(config.stability_runs))
    if runs <= 1:
        return 1.0, 0.0
    fit_va = primary_va
    fit_labels = labels
    fit_min_cluster_size = int(min_cluster_size)
    sample_size = int(getattr(config, "stability_sample_size", 0))
    if sample_size > 0 and primary_va.shape[0] > sample_size:
        rng = np.random.default_rng(int(config.random_state) + 7919)
        indices = np.sort(rng.choice(primary_va.shape[0], size=int(sample_size), replace=False))
        fit_va = primary_va[indices]
        fit_labels = labels[indices]
        scale = float(fit_va.shape[0]) / float(primary_va.shape[0])
        fit_min_cluster_size = max(1, int(np.floor(float(min_cluster_size) * scale)))
    scores: List[float] = []
    for run_idx in range(1, runs):
        try:
            model = BalancedVARegionClusterer(
                n_components=int(k),
                min_cluster_size=int(fit_min_cluster_size),
                random_state=int(config.random_state) + 1543 * run_idx,
                n_init=max(1, min(5, int(config.n_init))),
                max_iter=max(10, int(config.region_max_iter)),
                backend=_balanced_backend_name(config),
                device=str(config.device),
            ).fit(fit_va)
            scores.append(float(adjusted_rand_score(fit_labels, model.labels_)))
        except Exception:
            continue
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def _balanced_backend_name(config: KSelectionConfig) -> str:
    backend = str(getattr(config, "cluster_backend", "sklearn") or "sklearn").strip().lower()
    return "torch" if backend == "torch" else "sklearn"


def _score_balanced_silhouette(va: np.ndarray, labels: np.ndarray, config: KSelectionConfig) -> float:
    if np.unique(labels).size <= 1:
        return 0.0
    sample_size = int(getattr(config, "silhouette_sample_size", 0))
    eval_va = va
    eval_labels = labels
    if sample_size > 0 and va.shape[0] > sample_size:
        rng = np.random.default_rng(int(config.random_state))
        sample_idx = np.sort(rng.choice(va.shape[0], size=int(sample_size), replace=False))
        eval_va = va[sample_idx]
        eval_labels = labels[sample_idx]
    elif sample_size <= 0 and va.shape[0] > 20000:
        rng = np.random.default_rng(int(config.random_state))
        sample_idx = np.sort(rng.choice(va.shape[0], size=10000, replace=False))
        eval_va = va[sample_idx]
        eval_labels = labels[sample_idx]
    if np.unique(eval_labels).size < 2:
        return float("nan")
    use_torch = str(config.eval_backend).strip().lower() == "torch" or str(config.silhouette_mode).strip().lower() == "torch_chunked"
    if use_torch:
        return torch_silhouette_score_chunked(
            eval_va.astype(np.float32),
            eval_labels,
            chunk_size=int(config.silhouette_chunk_size),
            device=str(config.device),
        )
    return float(silhouette_score(eval_va.astype(np.float64), eval_labels))


def search_balanced_va_regions(
    features: np.ndarray,
    config: KSelectionConfig,
    *,
    primary_va: Optional[np.ndarray] = None,
    affect_labels: Optional[np.ndarray] = None,
) -> KSearchResult:
    matrix = np.asarray(features, dtype=np.float32)
    va = np.asarray(primary_va if primary_va is not None else matrix[:, :2], dtype=np.float32)
    if va.ndim != 2 or va.shape[0] != matrix.shape[0] or va.shape[1] < 2:
        raise ValueError(f"primary_va must have shape [N, >=2], got {va.shape}.")
    va = va[:, :2]
    tension = matrix[:, 2:] if matrix.ndim == 2 and matrix.shape[1] > 2 else None
    min_size_threshold = _min_size_threshold(config, int(matrix.shape[0]))
    rows: List[Dict[str, Any]] = []
    candidates: Dict[int, BalancedVARegionClusterer] = {}

    for k in range(int(config.k_min), int(config.k_max) + 1):
        if int(k) <= 1 or int(k) * min_size_threshold > int(matrix.shape[0]):
            rows.append(
                {
                    "k": int(k),
                    "balanced_region_score": -float("inf"),
                    "min_cluster_size": 0,
                    "min_size_ok": False,
                    "size_balance": 0.0,
                    "skipped": True,
                }
            )
            continue
        model = BalancedVARegionClusterer(
            n_components=int(k),
            min_cluster_size=int(min_size_threshold),
            random_state=int(config.random_state),
            n_init=max(1, int(config.n_init)),
            max_iter=max(10, int(config.region_max_iter)),
            backend=_balanced_backend_name(config),
            device=str(config.device),
        ).fit(va)
        labels = model.labels_.astype(np.int64)
        sizes = np.bincount(labels, minlength=int(k))
        try:
            silhouette = _score_balanced_silhouette(va, labels, config)
        except Exception:
            silhouette = float("nan")
        stability_mean, stability_std = _stability(
            va,
            labels,
            k=int(k),
            min_cluster_size=int(min_size_threshold),
            config=config,
        )
        overlap_metrics = compute_overlap_gate_metrics(
            va,
            labels,
            min_va_knn_purity=float(config.min_va_knn_purity),
            min_va_center_sep=float(config.min_va_center_sep),
            max_negative_silhouette_fraction=float(config.max_va_negative_silhouette_fraction),
            silhouette_sample_size=int(getattr(config, "silhouette_sample_size", 0)),
            eval_backend=str(config.eval_backend),
            device=str(config.device),
            chunk_size=int(config.silhouette_chunk_size),
            random_state=int(config.random_state),
        )
        tension_metrics = _tension_diagnostics(tension, labels)
        affect_metrics: Dict[str, Any] = {}
        if config.affect_gate_enabled:
            affect_metrics = compute_affect_purity_metrics(
                labels,
                affect_labels,
                min_dominant_ratio=float(config.min_affect_dominant_ratio),
                max_mixed_cluster_fraction=float(config.max_affect_mixed_cluster_fraction),
                min_weighted_purity=float(config.min_affect_weighted_purity),
                min_valid_fraction=float(config.min_affect_valid_fraction),
            )
        size_balance = float(sizes.min() / max(sizes.max(), 1))
        score = (
            0.32 * float(np.nan_to_num(silhouette, nan=-1.0))
            + 0.23 * float(stability_mean)
            + 0.15 * float(size_balance)
            + 0.12 * float(np.nan_to_num(overlap_metrics["va_knn_purity_20"], nan=0.0))
            + 0.13 * min(1.0, float(np.nan_to_num(overlap_metrics["va_center_radius_sep"], nan=0.0)))
            - 0.05 * float(np.nan_to_num(overlap_metrics["va_negative_silhouette_fraction"], nan=1.0))
        )
        candidates[int(k)] = model
        rows.append(
            {
                "k": int(k),
                "balanced_region_score": float(score),
                "va_silhouette": float(silhouette),
                "seed_ari_mean": float(stability_mean),
                "seed_ari_std": float(stability_std),
                "min_cluster_size": int(sizes.min()),
                "min_size_ok": bool(sizes.min() >= min_size_threshold),
                "size_balance": float(size_balance),
                "inertia": float(model.inertia_),
                "skipped": False,
                **overlap_metrics,
                **tension_metrics,
                **affect_metrics,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("k", kind="stable").reset_index(drop=True)
    eligible = (~metrics.get("skipped", False).to_numpy(dtype=bool)) & metrics["min_size_ok"].to_numpy(dtype=bool)
    if bool(config.overlap_gate_enabled) and "overlap_gate_ok" in metrics.columns:
        eligible &= metrics["overlap_gate_ok"].fillna(False).to_numpy(dtype=bool)
    if not eligible.any():
        candidates_text = ", ".join(
            (
                f"k={int(row.k)}:min_size={int(getattr(row, 'min_cluster_size', 0))},"
                f"min_ok={bool(getattr(row, 'min_size_ok', False))},"
                f"overlap_ok={bool(getattr(row, 'overlap_gate_ok', True))},"
                f"sil={float(getattr(row, 'va_silhouette', float('nan'))):.3f},"
                f"score={float(getattr(row, 'balanced_region_score', float('nan'))):.3f}"
            )
            for row in metrics.itertuples(index=False)
        )
        raise ValueError(
            "No balanced_va_regions candidate satisfied min-size and overlap hard gates "
            f"(k_min={int(config.k_min)}, k_max={int(config.k_max)}, "
            f"min_cluster_size_threshold={min_size_threshold}). Candidates: {candidates_text}."
        )
    selected_idx = int(metrics[eligible]["balanced_region_score"].idxmax())
    selected_row = metrics.loc[selected_idx]
    selected_k = int(selected_row["k"])
    best_model = candidates[selected_k]
    selection_info = {
        "selected_k": selected_k,
        "selection_mode": "balanced_va_regions",
        "balanced_region_score": float(selected_row["balanced_region_score"]),
        "va_silhouette": float(selected_row["va_silhouette"]),
        "seed_ari_mean": float(selected_row["seed_ari_mean"]),
        "seed_ari_std": float(selected_row["seed_ari_std"]),
        "min_cluster_size_threshold": int(min_size_threshold),
        "min_cluster_size": int(selected_row["min_cluster_size"]),
        "min_size_ok": bool(selected_row["min_size_ok"]),
        "size_balance": float(selected_row["size_balance"]),
        "anti_collapse_min_size_hard_gate": True,
        "tension_diagnostic_only": True,
        "tension_norm_mean": float(selected_row.get("tension_norm_mean", float("nan"))),
        "tension_effect_ratio": float(selected_row.get("tension_effect_ratio", float("nan"))),
        "stability_sample_size": int(config.stability_sample_size),
        "silhouette_sample_size": int(config.silhouette_sample_size),
        "va_silhouette_sample_size": int(selected_row.get("va_silhouette_sample_size", 0)),
        "overlap_gate_enabled": bool(config.overlap_gate_enabled),
        "overlap_gate_ok": bool(selected_row.get("overlap_gate_ok", True)),
        "va_knn_purity_10": float(selected_row.get("va_knn_purity_10", float("nan"))),
        "va_knn_purity_20": float(selected_row.get("va_knn_purity_20", float("nan"))),
        "va_center_radius_sep": float(selected_row.get("va_center_radius_sep", float("nan"))),
        "va_negative_silhouette_fraction": float(selected_row.get("va_negative_silhouette_fraction", float("nan"))),
        "va_mean_silhouette": float(selected_row.get("va_mean_silhouette", float("nan"))),
        "cluster_backend": str(config.cluster_backend),
        "eval_backend": config.eval_backend,
        "actual_cluster_backend": str(getattr(best_model, "actual_backend_", "balanced_va_region_kmeans")),
        "actual_eval_backend": "torch" if str(config.eval_backend).strip().lower() == "torch" else config.eval_backend,
        "device": str(config.device),
        "region_max_iter": int(config.region_max_iter),
        "affect_gate_diagnostic_only": bool(config.affect_gate_enabled),
    }
    if config.affect_gate_enabled:
        for field in (
            "affect_valid_fraction",
            "affect_weighted_dominant_ratio",
            "affect_min_dominant_ratio",
            "affect_mixed_cluster_fraction",
            "affect_nmi",
            "affect_min_dominant_gate_ok",
            "affect_worst_cluster_id",
            "affect_worst_cluster_size",
            "affect_gate_ok",
        ):
            value = selected_row.get(field, float("nan"))
            if field.endswith("_ok"):
                selection_info[field] = bool(value)
            elif field.endswith("_id") or field.endswith("_size"):
                selection_info[field] = int(value) if pd.notna(value) else -1
            else:
                selection_info[field] = float(value) if pd.notna(value) else float("nan")
    return KSearchResult(
        best_k=selected_k,
        best_model=best_model,
        metrics=metrics,
        selection_info=selection_info,
    )
