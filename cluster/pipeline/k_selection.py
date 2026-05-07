"""K-selection strategies for GMM clustering.

Provides composite multi-metric scoring, BIC elbow detection,
hierarchical two-level clustering, and stability analysis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture

from cluster.backends import resolve_cluster_backend


@dataclass(frozen=True)
class KSelectionConfig:
    k_min: int = 4
    k_max: int = 24
    covariance_type: str = "full"
    n_init: int = 10
    min_cluster_size: int = 20
    min_cluster_size_ratio: float = 0.01
    random_state: int = 42
    # Composite weights
    w_bic: float = 0.30
    w_silhouette: float = 0.30
    w_min_size: float = 0.20
    w_stability: float = 0.20
    stability_runs: int = 5
    cluster_backend: str = "sklearn"
    eval_backend: str = "sklearn"
    device: str = "cpu"
    silhouette_mode: str = "full"
    silhouette_sample_size: int = 0
    silhouette_chunk_size: int = 4096
    # Hierarchical
    macro_k_min: int = 4
    macro_k_max: int = 8
    micro_k_min: int = 2
    micro_k_max: int = 5


@dataclass
class KSearchResult:
    best_k: int
    best_model: GaussianMixture
    metrics: pd.DataFrame
    selection_info: Dict[str, Any]


@dataclass
class HierarchicalClusterResult:
    macro_k: int
    macro_labels: np.ndarray
    micro_labels: np.ndarray  # combined label per sample, e.g. 0, 1, 2, ...
    label_names: Dict[int, str]  # e.g. {0: "M1-a", 1: "M1-b", 2: "M2-a", ...}
    total_clusters: int
    macro_model: GaussianMixture
    micro_models: Dict[int, GaussianMixture]
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _normalize_series(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if constant."""
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Stability scoring
# ---------------------------------------------------------------------------

def _n_parallel_jobs() -> int:
    """Return number of parallel jobs, respecting env override."""
    env_val = os.environ.get("CLUSTER_N_JOBS", "")
    if env_val:
        return int(env_val)
    return -1  # use all CPUs


def _fit_stability_run(
    features: np.ndarray,
    k: int,
    covariance_type: str,
    random_state: int,
    cluster_backend: str = "sklearn",
    device: str = "cpu",
) -> np.ndarray:
    """Fit a single GMM run for stability scoring."""
    backend = resolve_cluster_backend(cluster_backend, device=device)
    labels, _model = backend.fit_predict(
        features,
        algorithm="gmm",
        n_clusters=k,
        covariance_type=covariance_type,
        reg_covar=1e-5,
        n_init=1,
        random_state=random_state,
    )
    return labels.astype(np.int64)


def compute_stability_score(
    features: np.ndarray,
    k: int,
    n_runs: int = 5,
    covariance_type: str = "full",
    random_state: int = 42,
    cluster_backend: str = "sklearn",
    device: str = "cpu",
) -> float:
    """Run GMM *n_runs* times in parallel and return mean pairwise ARI."""
    if n_runs < 2:
        return 1.0
    all_labels: List[np.ndarray] = Parallel(n_jobs=min(n_runs, _n_parallel_jobs()))(
        delayed(_fit_stability_run)(
            features,
            k,
            covariance_type,
            random_state + run * 1000,
            cluster_backend,
            device,
        )
        for run in range(n_runs)
    )
    ari_scores: List[float] = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))
    return float(np.mean(ari_scores)) if ari_scores else 1.0


# ---------------------------------------------------------------------------
# BIC elbow detection
# ---------------------------------------------------------------------------

def detect_bic_elbow(bic_values: Dict[int, float]) -> int:
    """Second-derivative method to find BIC diminishing-returns point."""
    if len(bic_values) < 3:
        return min(bic_values, key=bic_values.get)
    ks = sorted(bic_values.keys())
    bics = np.array([bic_values[k] for k in ks], dtype=np.float64)
    # First derivative (lower BIC is better, so negative slope = improving)
    d1 = np.diff(bics)
    # Second derivative (acceleration)
    d2 = np.diff(d1)
    # Elbow: where acceleration is most positive (improvement slowing fastest)
    if len(d2) == 0:
        return ks[int(np.argmin(bics))]
    elbow_idx = int(np.argmax(d2)) + 1  # +1 because d2 is offset by 1 from d1
    return ks[elbow_idx]


# ---------------------------------------------------------------------------
# Composite K search
# ---------------------------------------------------------------------------

def _fit_single_k(
    features: np.ndarray,
    k: int,
    config: KSelectionConfig,
    dist_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit GMM for a single K value.

    Parameters
    ----------
    dist_matrix : optional precomputed pairwise distance matrix.
        When provided, silhouette_score uses metric='precomputed' to
        avoid recomputing distances for every K.
    """
    cluster_backend = resolve_cluster_backend(config.cluster_backend, device=config.device)
    labels, gmm = cluster_backend.fit_predict(
        features,
        algorithm="gmm",
        n_clusters=k,
        covariance_type=config.covariance_type,
        reg_covar=1e-5,
        n_init=config.n_init,
        random_state=config.random_state,
    )
    n_unique = len(np.unique(labels))
    sizes = np.bincount(labels, minlength=k)
    min_size_threshold = max(
        config.min_cluster_size,
        int(np.ceil(config.min_cluster_size_ratio * features.shape[0])),
    )
    if n_unique > 1:
        if dist_matrix is not None and str(config.eval_backend).lower() == "sklearn":
            sil = _safe_metric(silhouette_score, dist_matrix, labels, metric="precomputed")
        else:
            sil = _score_silhouette(features, labels, config)
        db = _safe_metric(davies_bouldin_score, features, labels)
    else:
        sil = float("nan")
        db = float("nan")
    return {
        "k": k,
        "gmm": gmm,
        "labels": labels,
        "bic": float(gmm.bic(features)),
        "aic": float(gmm.aic(features)),
        "silhouette": sil,
        "davies_bouldin": db,
        "min_cluster_size": int(sizes.min()),
        "min_size_ok": bool(sizes.min() >= min_size_threshold),
        "min_size_threshold": min_size_threshold,
    }


def _score_silhouette(features: np.ndarray, labels: np.ndarray, config: KSelectionConfig) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    mode = str(config.silhouette_mode).strip().lower()
    eval_backend_name = "torch" if mode == "torch_chunked" else config.eval_backend
    backend = resolve_cluster_backend(eval_backend_name, device=config.device)
    if mode == "sampled":
        return backend.score_silhouette(
            features,
            labels,
            sample_size=int(config.silhouette_sample_size),
            random_state=int(config.random_state),
        )
    return backend.score_silhouette(
        features,
        labels,
        chunk_size=int(config.silhouette_chunk_size),
    )


def search_gmm_composite(
    features: np.ndarray,
    config: KSelectionConfig,
) -> KSearchResult:
    """Multi-metric K selection with composite scoring.

    Optimisations:
    - Precomputes pairwise distance matrix once for silhouette scoring.
    - Parallelises GMM fitting across K values using joblib.
    - Parallelises stability scoring within each K.
    """
    # Precompute distance matrix once only for the sklearn full silhouette path.
    use_precomputed = (
        str(config.eval_backend).lower() == "sklearn"
        and str(config.silhouette_mode).lower() == "full"
    )
    dist_matrix = pairwise_distances(features, metric="euclidean") if use_precomputed else None

    k_range = list(range(config.k_min, config.k_max + 1))

    # Parallel GMM fitting across K values
    results: List[Dict[str, Any]] = Parallel(n_jobs=_n_parallel_jobs())(
        delayed(_fit_single_k)(features, k, config, dist_matrix)
        for k in k_range
    )

    # Stability scoring (each call is already internally parallelised)
    for result in results:
        result["stability"] = compute_stability_score(
            features, result["k"],
            n_runs=config.stability_runs,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            cluster_backend=config.cluster_backend,
            device=config.device,
        )

    if not results:
        raise RuntimeError("GMM composite search produced no results.")

    # Build metrics DataFrame
    rows = []
    for r in results:
        rows.append({
            "k": r["k"],
            "bic": r["bic"],
            "aic": r["aic"],
            "silhouette": r["silhouette"],
            "davies_bouldin": r["davies_bouldin"],
            "min_cluster_size": r["min_cluster_size"],
            "min_size_ok": r["min_size_ok"],
            "stability": r["stability"],
        })
    metrics = pd.DataFrame(rows).sort_values("k", kind="stable").reset_index(drop=True)

    # Composite scoring
    bic_arr = np.array([r["bic"] for r in results], dtype=np.float64)
    sil_arr = np.array([r["silhouette"] for r in results], dtype=np.float64)
    stab_arr = np.array([r["stability"] for r in results], dtype=np.float64)

    # Normalize BIC (lower is better → invert)
    bic_norm = 1.0 - _normalize_series(bic_arr)
    sil_norm = _normalize_series(np.nan_to_num(sil_arr, nan=0.0))
    stab_norm = _normalize_series(stab_arr)

    # Size penalty: 1.0 if min_size >= threshold, else ratio
    size_scores = np.array([
        min(1.0, r["min_cluster_size"] / max(r["min_size_threshold"], 1))
        for r in results
    ], dtype=np.float64)

    composite = (
        config.w_bic * bic_norm
        + config.w_silhouette * sil_norm
        + config.w_min_size * size_scores
        + config.w_stability * stab_norm
    )
    metrics["composite_score"] = composite

    best_idx = int(np.argmax(composite))
    best_result = results[best_idx]

    # Also compute BIC elbow for reference
    bic_elbow_k = detect_bic_elbow({r["k"]: r["bic"] for r in results})

    return KSearchResult(
        best_k=best_result["k"],
        best_model=best_result["gmm"],
        metrics=metrics,
        selection_info={
            "selected_k": best_result["k"],
            "selection_mode": "composite",
            "composite_score": float(composite[best_idx]),
            "bic_elbow_k": bic_elbow_k,
            "min_cluster_size_threshold": best_result["min_size_threshold"],
            "stability_runs": config.stability_runs,
            "cluster_backend": config.cluster_backend,
            "eval_backend": config.eval_backend,
            "silhouette_mode": config.silhouette_mode,
        },
    )


# ---------------------------------------------------------------------------
# Legacy BIC-only search (backward compatible)
# ---------------------------------------------------------------------------

def search_gmm_bic_only(
    features: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    min_cluster_size_abs: int,
    min_cluster_size_ratio: float,
    covariance_type: str = "diag",
    n_init: int = 10,
    config: Optional[KSelectionConfig] = None,
) -> KSearchResult:
    """Original BIC + size-constraint selection (backward compatible)."""
    effective_config = config or KSelectionConfig(
        k_min=k_min,
        k_max=k_max,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=random_state,
        min_cluster_size=min_cluster_size_abs,
        min_cluster_size_ratio=min_cluster_size_ratio,
    )
    rows: List[Dict[str, float]] = []
    fitted_models: Dict[int, Any] = {}
    cluster_backend = resolve_cluster_backend(effective_config.cluster_backend, device=effective_config.device)
    for k in range(int(k_min), int(k_max) + 1):
        labels, model = cluster_backend.fit_predict(
            features,
            algorithm="gmm",
            n_clusters=int(k),
            covariance_type=covariance_type,
            reg_covar=1e-5,
            n_init=int(n_init),
            random_state=int(random_state),
        )
        fitted_models[int(k)] = model
        n_unique = len(np.unique(labels))
        rows.append({
            "k": float(k),
            "bic": float(model.bic(features)),
            "aic": float(model.aic(features)),
            "silhouette": _safe_metric(_score_silhouette, features, labels, effective_config) if n_unique > 1 else float("nan"),
            "davies_bouldin": _safe_metric(davies_bouldin_score, features, labels) if n_unique > 1 else float("nan"),
            "min_cluster_size": float(np.bincount(labels, minlength=k).min()),
        })
    if not fitted_models:
        raise RuntimeError("GMM search did not produce a model.")
    metrics = pd.DataFrame(rows).sort_values("k", kind="stable").reset_index(drop=True)
    min_size_threshold = max(int(min_cluster_size_abs), int(np.ceil(float(min_cluster_size_ratio) * float(features.shape[0]))))
    metrics["eligible_under_size_constraint"] = metrics["min_cluster_size"] >= float(min_size_threshold)
    eligible = metrics[metrics["eligible_under_size_constraint"]].copy()
    selection_mode = "bic_with_size_constraint"
    if eligible.empty:
        eligible = metrics.copy()
        selection_mode = "bic_fallback_no_size_feasible"
    selected_idx = eligible["bic"].idxmin()
    selected_row = metrics.loc[selected_idx]
    selected_k = int(selected_row["k"])
    return KSearchResult(
        best_k=selected_k,
        best_model=fitted_models[selected_k],
        metrics=metrics,
        selection_info={
            "selected_k": float(selected_k),
            "selection_mode": selection_mode,
            "min_cluster_size_threshold": float(min_size_threshold),
            "cluster_backend": effective_config.cluster_backend,
            "eval_backend": effective_config.eval_backend,
            "silhouette_mode": effective_config.silhouette_mode,
        },
    )


# ---------------------------------------------------------------------------
# Hierarchical (two-level) clustering
# ---------------------------------------------------------------------------

def hierarchical_cluster(
    features: np.ndarray,
    config: KSelectionConfig,
) -> HierarchicalClusterResult:
    """Two-level clustering: macro clusters then micro sub-clusters.

    Level 1: GMM with macro_k on full features (emotion regions).
    Level 2: For each macro cluster, GMM with micro_k selected by silhouette.
    Labels: "M1-a", "M1-b", "M2-a", etc.

    Macro K range is derived from config.k_min / config.k_max when they
    differ from the legacy defaults, falling back to macro_k_min / macro_k_max.
    Micro K range always uses micro_k_min / micro_k_max.
    """
    # Derive effective macro range: honour CLI k_min/k_max when explicitly set
    macro_lo = config.macro_k_min
    macro_hi = config.macro_k_max
    if config.k_min != 4 or config.k_max != 24:  # non-default → user overrode
        macro_lo = config.k_min
        macro_hi = config.k_max

    # Precompute distance matrix for silhouette scoring
    dist_matrix = pairwise_distances(features, metric="euclidean")

    # Step 1: Find best macro K
    best_macro_k = macro_lo
    best_macro_sil = -1.0
    best_macro_gmm: Optional[GaussianMixture] = None
    best_macro_labels: Optional[np.ndarray] = None

    for mk in range(macro_lo, macro_hi + 1):
        gmm = GaussianMixture(
            n_components=mk,
            covariance_type=config.covariance_type,
            reg_covar=1e-5,
            n_init=config.n_init,
            random_state=config.random_state,
        )
        labels = gmm.fit_predict(features)
        if len(np.unique(labels)) > 1:
            sil = _safe_metric(silhouette_score, dist_matrix, labels, metric="precomputed")
            if sil > best_macro_sil:
                best_macro_sil = sil
                best_macro_k = mk
                best_macro_gmm = gmm
                best_macro_labels = labels

    if best_macro_gmm is None or best_macro_labels is None:
        raise RuntimeError("Hierarchical macro clustering failed.")

    # Step 2: Micro-cluster within each macro cluster
    micro_models: Dict[int, GaussianMixture] = {}
    combined_labels = np.full(len(features), -1, dtype=np.int64)
    label_names: Dict[int, str] = {}
    global_label = 0
    info_micro: Dict[str, Any] = {}

    for macro_id in range(best_macro_k):
        mask = best_macro_labels == macro_id
        cluster_features = features[mask]
        cluster_size = int(mask.sum())

        if cluster_size < config.micro_k_min * 2:
            # Too small to sub-cluster
            combined_labels[mask] = global_label
            label_names[global_label] = f"M{macro_id + 1}"
            info_micro[f"macro_{macro_id}"] = {"micro_k": 1, "size": cluster_size}
            global_label += 1
            continue

        best_micro_k = 1
        best_micro_sil = -1.0
        best_micro_gmm: Optional[GaussianMixture] = None
        best_micro_labels: Optional[np.ndarray] = None

        # Precompute micro distance matrix for this macro cluster
        micro_dist = pairwise_distances(cluster_features, metric="euclidean")

        for micro_k in range(config.micro_k_min, min(config.micro_k_max + 1, cluster_size // 2 + 1)):
            gmm = GaussianMixture(
                n_components=micro_k,
                covariance_type=config.covariance_type,
                reg_covar=1e-5,
                n_init=config.n_init,
                random_state=config.random_state,
            )
            micro_labels = gmm.fit_predict(cluster_features)
            if len(np.unique(micro_labels)) > 1:
                sil = _safe_metric(silhouette_score, micro_dist, micro_labels, metric="precomputed")
                if sil > best_micro_sil:
                    best_micro_sil = sil
                    best_micro_k = micro_k
                    best_micro_gmm = gmm
                    best_micro_labels = micro_labels

        if best_micro_gmm is not None and best_micro_labels is not None:
            micro_models[macro_id] = best_micro_gmm
            sub_letters = "abcdefghijklmnopqrstuvwxyz"
            for sub_id in range(best_micro_k):
                sub_mask = best_micro_labels == sub_id
                indices = np.where(mask)[0][sub_mask]
                combined_labels[indices] = global_label
                suffix = sub_letters[sub_id] if sub_id < len(sub_letters) else str(sub_id)
                label_names[global_label] = f"M{macro_id + 1}-{suffix}"
                global_label += 1
        else:
            combined_labels[mask] = global_label
            label_names[global_label] = f"M{macro_id + 1}"
            global_label += 1

        info_micro[f"macro_{macro_id}"] = {
            "micro_k": best_micro_k,
            "size": cluster_size,
            "micro_silhouette": best_micro_sil,
        }

    return HierarchicalClusterResult(
        macro_k=best_macro_k,
        macro_labels=best_macro_labels,
        micro_labels=combined_labels,
        label_names=label_names,
        total_clusters=global_label,
        macro_model=best_macro_gmm,
        micro_models=micro_models,
        info={
            "macro_k": best_macro_k,
            "macro_silhouette": best_macro_sil,
            "micro_details": info_micro,
            "total_clusters": global_label,
        },
    )
