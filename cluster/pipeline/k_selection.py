"""K-selection strategies for GMM clustering.

Provides composite multi-metric scoring, BIC elbow detection,
hierarchical two-level clustering, and stability analysis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture

from cluster.backends import resolve_cluster_backend
from cluster.backends.gmm_convergence import fit_gaussian_mixture_robust
from cluster.backends.masked_diag_gmm import MaskedDiagonalGMM
from cluster.evaluation.metrics import masked_silhouette_score
from cluster.pipeline.macro_micro import MacroMicroClusterer


@dataclass(frozen=True)
class KSelectionConfig:
    k_min: int = 4
    k_max: int = 24
    covariance_type: str = "diag"
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
    cluster_backend: str = "auto"
    eval_backend: str = "auto"
    device: str = "cpu"
    silhouette_mode: str = "full"
    silhouette_sample_size: int = 0
    silhouette_chunk_size: int = 4096
    # Mask purity
    w_mask_purity: float = 0.15
    strict_min_size: bool = True
    # Hierarchical
    macro_k_min: int = 4
    macro_k_max: int = 8
    micro_k_min: int = 2
    micro_k_max: int = 5
    # Affect-first hard gates
    affect_gate_enabled: bool = False
    min_affect_dominant_ratio: float = 0.70
    max_affect_mixed_cluster_fraction: float = 0.15
    min_affect_weighted_purity: float = 0.80
    min_affect_valid_fraction: float = 0.95


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


def _min_size_threshold(config: KSelectionConfig, n_samples: int) -> int:
    return max(
        int(config.min_cluster_size),
        int(np.ceil(float(config.min_cluster_size_ratio) * float(n_samples))),
    )


def _mask_patterns(view_mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if view_mask is None:
        return None
    mask = np.asarray(view_mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[1] < 2:
        return None
    return np.array([
        "".join(["1" if value > 0 else "0" for value in row[:2]])
        for row in mask
    ])


def _mask_nmi(labels: np.ndarray, view_mask: Optional[np.ndarray]) -> float:
    patterns = _mask_patterns(view_mask)
    if patterns is None:
        return float("nan")
    return float(normalized_mutual_info_score(np.asarray(labels, dtype=np.int64), patterns))


def _max_cluster_mask_purity(labels: np.ndarray, view_mask: Optional[np.ndarray]) -> float:
    patterns = _mask_patterns(view_mask)
    if patterns is None:
        return float("nan")
    y = np.asarray(labels, dtype=np.int64)
    purities: List[float] = []
    for cluster_id in np.unique(y):
        cluster_patterns = patterns[y == cluster_id]
        if cluster_patterns.size == 0:
            continue
        _, counts = np.unique(cluster_patterns, return_counts=True)
        purities.append(float(counts.max() / counts.sum()))
    return float(max(purities)) if purities else float("nan")


def compute_affect_purity_metrics(
    cluster_labels: np.ndarray,
    affect_labels: Optional[np.ndarray],
    *,
    min_dominant_ratio: float,
    max_mixed_cluster_fraction: float,
    min_weighted_purity: float,
    min_valid_fraction: float,
) -> Dict[str, Any]:
    """Measure whether discovered clusters remain coherent on the affect quadrant plane."""
    if affect_labels is None:
        return {
            "affect_valid_fraction": float("nan"),
            "affect_weighted_dominant_ratio": float("nan"),
            "affect_min_dominant_ratio": float("nan"),
            "affect_mixed_cluster_fraction": float("nan"),
            "affect_nmi": float("nan"),
            "affect_gate_ok": False,
        }

    clusters = np.asarray(cluster_labels, dtype=np.int64)
    labels = np.asarray(affect_labels, dtype=np.int64)
    if clusters.shape[0] != labels.shape[0]:
        raise ValueError(
            f"affect_labels length {labels.shape[0]} does not match cluster labels length {clusters.shape[0]}."
        )
    valid = (labels >= 0) & (labels < 4)
    valid_count = int(valid.sum())
    valid_fraction = float(valid_count / max(labels.shape[0], 1))
    if valid_count == 0:
        return {
            "affect_valid_fraction": valid_fraction,
            "affect_weighted_dominant_ratio": float("nan"),
            "affect_min_dominant_ratio": float("nan"),
            "affect_mixed_cluster_fraction": float("nan"),
            "affect_nmi": float("nan"),
            "affect_gate_ok": False,
        }

    weighted_dominant = 0.0
    mixed_count = 0
    dominant_ratios: List[float] = []
    for cluster_id in np.unique(clusters[valid]):
        cluster_mask = valid & (clusters == int(cluster_id))
        cluster_size = int(cluster_mask.sum())
        if cluster_size == 0:
            continue
        counts = np.bincount(labels[cluster_mask], minlength=4).astype(np.float64)
        dominant = float(counts.max() / max(cluster_size, 1))
        dominant_ratios.append(dominant)
        weighted_dominant += float(counts.max())
        if dominant < float(min_dominant_ratio):
            mixed_count += cluster_size

    weighted_ratio = float(weighted_dominant / max(valid_count, 1))
    min_ratio = float(min(dominant_ratios)) if dominant_ratios else float("nan")
    mixed_fraction = float(mixed_count / max(valid_count, 1))
    affect_nmi = float(normalized_mutual_info_score(clusters[valid], labels[valid]))
    gate_ok = (
        valid_fraction >= float(min_valid_fraction)
        and weighted_ratio >= float(min_weighted_purity)
        and min_ratio >= float(min_dominant_ratio)
        and mixed_fraction <= float(max_mixed_cluster_fraction)
    )
    return {
        "affect_valid_fraction": valid_fraction,
        "affect_weighted_dominant_ratio": weighted_ratio,
        "affect_min_dominant_ratio": min_ratio,
        "affect_mixed_cluster_fraction": mixed_fraction,
        "affect_nmi": affect_nmi,
        "affect_gate_ok": bool(gate_ok),
    }


def _add_affect_selection_info(
    info: Dict[str, Any],
    row: pd.Series,
    config: KSelectionConfig,
) -> None:
    if not bool(config.affect_gate_enabled):
        return
    info["affect_gate_enabled"] = True
    info["min_affect_dominant_ratio"] = float(config.min_affect_dominant_ratio)
    info["max_affect_mixed_cluster_fraction"] = float(config.max_affect_mixed_cluster_fraction)
    info["min_affect_weighted_purity"] = float(config.min_affect_weighted_purity)
    info["min_affect_valid_fraction"] = float(config.min_affect_valid_fraction)
    info["affect_gate_ok"] = bool(row.get("affect_gate_ok", False))
    for field in (
        "affect_valid_fraction",
        "affect_weighted_dominant_ratio",
        "affect_min_dominant_ratio",
        "affect_mixed_cluster_fraction",
        "affect_nmi",
    ):
        value = row.get(field, float("nan"))
        info[field] = float(value) if pd.notna(value) else float("nan")


def _prefix_affect_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _select_best_index(
    metrics: pd.DataFrame,
    scores: np.ndarray,
    config: KSelectionConfig,
    *,
    selection_mode: str,
) -> int:
    eligible = metrics["min_size_ok"].to_numpy(dtype=bool)
    if "affect_gate_ok" in metrics.columns:
        eligible = eligible & metrics["affect_gate_ok"].to_numpy(dtype=bool)
    if not bool(eligible.any()):
        threshold = _min_size_threshold(config, int(metrics.attrs.get("n_samples", 0)))
        min_sizes = ", ".join(
            f"K={int(row.k)}:{int(row.min_cluster_size)}"
            for row in metrics[["k", "min_cluster_size"]].itertuples(index=False)
        )
        affect_detail = ""
        if "affect_gate_ok" in metrics.columns:
            affect_detail = (
                " Affect gate candidates: "
                + ", ".join(
                    f"K={int(row.k)}:ok={bool(row.affect_gate_ok)},"
                    f"weighted={float(row.affect_weighted_dominant_ratio):.3f},"
                    f"min={float(row.affect_min_dominant_ratio):.3f},"
                    f"mixed={float(row.affect_mixed_cluster_fraction):.3f}"
                    for row in metrics[
                        [
                            "k",
                            "affect_gate_ok",
                            "affect_weighted_dominant_ratio",
                            "affect_min_dominant_ratio",
                            "affect_mixed_cluster_fraction",
                        ]
                    ].itertuples(index=False)
                )
                + "."
            )
        raise ValueError(
            f"No K candidate satisfied min_cluster_size_threshold={threshold} "
            f"for {selection_mode}. Candidate minimum sizes: {min_sizes}.{affect_detail}"
        )
    masked_scores = np.where(eligible, scores, -np.inf)
    return int(np.argmax(masked_scores))


# ---------------------------------------------------------------------------
# Stability scoring
# ---------------------------------------------------------------------------

def _n_parallel_jobs() -> int:
    """Return number of parallel jobs, respecting env override."""
    env_val = os.environ.get("CLUSTER_N_JOBS", "")
    if env_val:
        return int(env_val)
    return -1  # use all CPUs


def _is_accelerated_backend_requested(cluster_backend: str, device: str) -> bool:
    backend_name = str(cluster_backend or "").strip().lower()
    device_name = str(device or "").strip().lower()
    return backend_name in {"torch", "cuml"} or device_name.startswith("cuda")


def _parallel_kwargs(cluster_backend: str, device: str, max_jobs: Optional[int] = None) -> Dict[str, Any]:
    if _is_accelerated_backend_requested(cluster_backend, device):
        return {"n_jobs": 1}
    n_jobs = _n_parallel_jobs()
    if max_jobs is not None:
        n_jobs = min(int(max_jobs), int(n_jobs)) if int(n_jobs) > 0 else int(max_jobs)
    return {"n_jobs": n_jobs}


def _backend_runtime_info(config: KSelectionConfig, *, actual_cluster_backend: str, actual_eval_backend: str) -> Dict[str, Any]:
    return {
        "cluster_backend": config.cluster_backend,
        "eval_backend": config.eval_backend,
        "actual_cluster_backend": actual_cluster_backend,
        "actual_eval_backend": actual_eval_backend,
        "device": str(config.device),
        "silhouette_mode": config.silhouette_mode,
        "silhouette_sample_size": int(config.silhouette_sample_size),
        "silhouette_chunk_size": int(config.silhouette_chunk_size),
    }


def _resolve_eval_backend_name(config: KSelectionConfig) -> str:
    mode = str(config.silhouette_mode).strip().lower()
    if mode == "sampled":
        return "sklearn"
    backend_name = "torch" if mode in {"torch_chunked", "masked_torch_chunked"} else config.eval_backend
    return resolve_cluster_backend(backend_name, device=config.device).name


def _fit_stability_run(
    features: np.ndarray,
    k: int,
    covariance_type: str,
    random_state: int,
    cluster_backend: str = "sklearn",
    device: str = "cpu",
) -> np.ndarray:
    """Fit a single GMM run for stability scoring."""
    backend = resolve_cluster_backend(
        cluster_backend,
        device=device,
        algorithm="gmm",
        covariance_type=covariance_type,
    )
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
    covariance_type: str = "diag",
    random_state: int = 42,
    cluster_backend: str = "sklearn",
    device: str = "cpu",
) -> float:
    """Run GMM *n_runs* times in parallel and return mean pairwise ARI."""
    if n_runs < 2:
        return 1.0
    all_labels: List[np.ndarray] = Parallel(**_parallel_kwargs(cluster_backend, device, max_jobs=n_runs))(
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
    cluster_backend = resolve_cluster_backend(
        config.cluster_backend,
        device=config.device,
        algorithm="gmm",
        covariance_type=config.covariance_type,
    )
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
    min_size_threshold = _min_size_threshold(config, int(features.shape[0]))
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
        "actual_cluster_backend": cluster_backend.name,
    }


def _score_silhouette(features: np.ndarray, labels: np.ndarray, config: KSelectionConfig) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    mode = str(config.silhouette_mode).strip().lower()
    if mode == "sampled":
        backend = resolve_cluster_backend("sklearn", device=config.device)
        return backend.score_silhouette(
            features,
            labels,
            sample_size=int(config.silhouette_sample_size),
            random_state=int(config.random_state),
        )
    eval_backend_name = "torch" if mode in {"torch_chunked", "masked_torch_chunked"} else config.eval_backend
    backend = resolve_cluster_backend(eval_backend_name, device=config.device)
    return backend.score_silhouette(
        features,
        labels,
        chunk_size=int(config.silhouette_chunk_size),
    )


def _score_masked_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return masked_silhouette_score(features, labels, block_mask=block_mask, block_slices=block_slices)


def _cluster_jaccard_summary(reference_labels: np.ndarray, candidate_labels: np.ndarray) -> Tuple[float, float]:
    ref = np.asarray(reference_labels, dtype=np.int64)
    cand = np.asarray(candidate_labels, dtype=np.int64)
    scores: List[float] = []
    for ref_label in np.unique(ref):
        ref_mask = ref == int(ref_label)
        best = 0.0
        for cand_label in np.unique(cand):
            cand_mask = cand == int(cand_label)
            union = np.logical_or(ref_mask, cand_mask).sum()
            if union == 0:
                continue
            best = max(best, float(np.logical_and(ref_mask, cand_mask).sum() / union))
        scores.append(best)
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.min(scores))


def compute_macro_micro_bootstrap_stability(
    features: np.ndarray,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
    base_labels: np.ndarray,
    macro_k: int,
    config: KSelectionConfig,
    n_runs: int,
) -> Dict[str, float]:
    if int(n_runs) < 2:
        return {
            "seed_ari_mean": 1.0,
            "seed_ari_std": 0.0,
            "cluster_jaccard_mean": 1.0,
            "cluster_jaccard_min": 1.0,
            "bootstrap_valid_rate": 1.0,
        }

    matrix = np.asarray(features, dtype=np.float32)
    mask = np.asarray(block_mask, dtype=bool)
    base = np.asarray(base_labels, dtype=np.int64)
    rng = np.random.default_rng(int(config.random_state) + int(macro_k) * 10007)
    sample_size = max(
        int(min(matrix.shape[0], max(int(config.min_cluster_size) * int(macro_k), round(matrix.shape[0] * 0.8)))),
        min(matrix.shape[0], int(macro_k)),
    )
    ari_scores: List[float] = []
    jaccard_means: List[float] = []
    jaccard_mins: List[float] = []
    attempts = int(n_runs)
    for run in range(attempts):
        sample_idx = rng.choice(matrix.shape[0], size=sample_size, replace=True)
        try:
            model = MacroMicroClusterer(
                macro_k=int(macro_k),
                block_slices=block_slices,
                covariance_type=str(config.covariance_type),
                random_state=int(config.random_state) + 7919 * (run + 1),
                n_init=1,
                max_iter=80,
                micro_k_min=int(config.micro_k_min),
                micro_k_max=int(config.micro_k_max),
                min_cluster_size=max(1, int(config.min_cluster_size)),
            ).fit(matrix[sample_idx], block_mask=mask[sample_idx])
            predicted = model.predict(matrix, block_mask=mask)
        except Exception:
            continue
        ari_scores.append(float(adjusted_rand_score(base, predicted)))
        j_mean, j_min = _cluster_jaccard_summary(base, predicted)
        jaccard_means.append(j_mean)
        jaccard_mins.append(j_min)

    if not ari_scores:
        return {
            "seed_ari_mean": 0.0,
            "seed_ari_std": 0.0,
            "cluster_jaccard_mean": 0.0,
            "cluster_jaccard_min": 0.0,
            "bootstrap_valid_rate": 0.0,
        }
    return {
        "seed_ari_mean": float(np.mean(ari_scores)),
        "seed_ari_std": float(np.std(ari_scores)),
        "cluster_jaccard_mean": float(np.mean(jaccard_means)),
        "cluster_jaccard_min": float(np.min(jaccard_mins)),
        "bootstrap_valid_rate": float(len(ari_scores) / max(attempts, 1)),
    }


def search_gmm_composite(
    features: np.ndarray,
    config: KSelectionConfig,
    view_mask: Optional[np.ndarray] = None,
    affect_labels: Optional[np.ndarray] = None,
) -> KSearchResult:
    """Multi-metric K selection with composite scoring.

    Optimisations:
    - Precomputes pairwise distance matrix once for silhouette scoring.
    - Parallelises GMM fitting across K values using joblib.
    - Parallelises stability scoring within each K.

    Parameters
    ----------
    view_mask : ndarray [N, >=2] or None
        If provided, NMI between cluster labels and mask patterns is
        computed for each K and subtracted as a penalty in composite score.
    """
    # Precompute distance matrix once only for the sklearn full silhouette path.
    use_precomputed = (
        str(config.eval_backend).lower() == "sklearn"
        and str(config.silhouette_mode).lower() == "full"
    )
    dist_matrix = pairwise_distances(features, metric="euclidean") if use_precomputed else None

    k_range = list(range(config.k_min, config.k_max + 1))

    # Parallel GMM fitting across K values
    results: List[Dict[str, Any]] = Parallel(**_parallel_kwargs(config.cluster_backend, config.device))(
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

    mask_nmi_arr: Dict[int, float] = {}
    max_mask_purity_arr: Dict[int, float] = {}
    if view_mask is not None and config.w_mask_purity > 0:
        for r in results:
            lbls = r["labels"]
            mask_nmi_arr[r["k"]] = _mask_nmi(lbls, view_mask)
            max_mask_purity_arr[r["k"]] = _max_cluster_mask_purity(lbls, view_mask)

    # Build metrics DataFrame
    rows = []
    for r in results:
        affect_metrics: Dict[str, Any] = {}
        if config.affect_gate_enabled:
            affect_metrics = compute_affect_purity_metrics(
                r["labels"],
                affect_labels,
                min_dominant_ratio=float(config.min_affect_dominant_ratio),
                max_mixed_cluster_fraction=float(config.max_affect_mixed_cluster_fraction),
                min_weighted_purity=float(config.min_affect_weighted_purity),
                min_valid_fraction=float(config.min_affect_valid_fraction),
            )
        rows.append({
            "k": r["k"],
            "bic": r["bic"],
            "aic": r["aic"],
            "silhouette": r["silhouette"],
            "davies_bouldin": r["davies_bouldin"],
            "min_cluster_size": r["min_cluster_size"],
            "min_size_ok": r["min_size_ok"],
            "stability": r["stability"],
            "mask_nmi": mask_nmi_arr.get(r["k"], float("nan")),
            "max_mask_purity": max_mask_purity_arr.get(r["k"], float("nan")),
            **affect_metrics,
        })
    metrics = pd.DataFrame(rows).sort_values("k", kind="stable").reset_index(drop=True)
    metrics.attrs["n_samples"] = int(features.shape[0])

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

    raw_composite = (
        config.w_bic * bic_norm
        + config.w_silhouette * sil_norm
        + config.w_min_size * size_scores
        + config.w_stability * stab_norm
    )
    # Subtract mask-purity penalty if view_mask was provided
    mask_penalty_arr: Optional[np.ndarray] = None
    if mask_nmi_arr and config.w_mask_purity > 0:
        mask_penalty_arr = np.array([
            0.5 * mask_nmi_arr.get(r["k"], 0.0)
            + 0.5 * max_mask_purity_arr.get(r["k"], 0.0)
            for r in results
        ], dtype=np.float64)
        composite = raw_composite - config.w_mask_purity * mask_penalty_arr
    else:
        composite = raw_composite
    metrics["composite_score"] = composite

    best_idx = _select_best_index(metrics, composite, config, selection_mode="composite")
    best_result = results[best_idx]
    actual_cluster_backend = str(best_result.get("actual_cluster_backend", config.cluster_backend))
    actual_eval_backend = _resolve_eval_backend_name(config)

    # Also compute BIC elbow for reference
    bic_elbow_k = detect_bic_elbow({r["k"]: r["bic"] for r in results})

    selection_info: Dict[str, Any] = {
        "selected_k": best_result["k"],
        "selection_mode": "composite",
        "composite_score": float(composite[best_idx]),
        "bic_elbow_k": bic_elbow_k,
        "min_cluster_size_threshold": best_result["min_size_threshold"],
        "stability_runs": config.stability_runs,
        "w_mask_purity": config.w_mask_purity,
        "mask_nmi": mask_nmi_arr.get(best_result["k"], float("nan")),
        "max_mask_purity": max_mask_purity_arr.get(best_result["k"], float("nan")),
        "raw_composite_score": float(raw_composite[best_idx]),
        "eligible_candidate_count": int(metrics["min_size_ok"].sum()),
        "candidate_count": int(len(metrics)),
        **_backend_runtime_info(
            config,
            actual_cluster_backend=actual_cluster_backend,
            actual_eval_backend=actual_eval_backend,
        ),
    }
    if mask_penalty_arr is not None:
        selection_info["mask_penalty_applied"] = float(config.w_mask_purity * mask_penalty_arr[best_idx])
    if config.affect_gate_enabled:
        _add_affect_selection_info(selection_info, metrics.loc[best_idx], config)

    return KSearchResult(
        best_k=best_result["k"],
        best_model=best_result["gmm"],
        metrics=metrics,
        selection_info=selection_info,
    )


def search_gmm_semantic_composite(
    features: np.ndarray,
    config: KSelectionConfig,
    view_mask: Optional[np.ndarray] = None,
    affect_labels: Optional[np.ndarray] = None,
) -> KSearchResult:
    """Semantic composite K search with stricter leakage and support terms."""
    semantic_config = replace(
        config,
        w_bic=0.25,
        w_silhouette=0.25,
        w_min_size=0.25,
        w_stability=0.25,
        w_mask_purity=max(float(config.w_mask_purity), 0.25),
    )
    result = search_gmm_composite(features, semantic_config, view_mask=view_mask, affect_labels=affect_labels)
    metrics = result.metrics.copy()
    metrics["semantic_composite_score"] = metrics["composite_score"]
    info = dict(result.selection_info)
    info["selection_mode"] = "semantic_composite"
    info["semantic_terms"] = {
        "bic": semantic_config.w_bic,
        "silhouette": semantic_config.w_silhouette,
        "min_size": semantic_config.w_min_size,
        "stability": semantic_config.w_stability,
        "mask_purity_penalty": semantic_config.w_mask_purity,
    }
    return KSearchResult(
        best_k=result.best_k,
        best_model=result.best_model,
        metrics=metrics,
        selection_info=info,
    )


def _fit_single_masked_k(
    features: np.ndarray,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
    k: int,
    config: KSelectionConfig,
) -> Dict[str, Any]:
    model = MaskedDiagonalGMM(
        n_components=int(k),
        block_slices=block_slices,
        covariance_type="diag",
        reg_covar=1e-5,
        n_init=int(config.n_init),
        max_iter=100,
        random_state=int(config.random_state),
    ).fit(features, block_mask=block_mask)
    labels = model.predict(features, block_mask=block_mask).astype(np.int64)
    sizes = np.bincount(labels, minlength=int(k))
    n_unique = len(np.unique(labels))
    if n_unique > 1:
        sil = _score_masked_silhouette(features, labels, block_mask, block_slices)
        db = _safe_metric(davies_bouldin_score, features, labels)
    else:
        sil = float("nan")
        db = float("nan")
    min_size_threshold = _min_size_threshold(config, int(features.shape[0]))
    bic = float(model.bic(features, block_mask=block_mask))
    aic = float(model.aic(features, block_mask=block_mask))
    return {
        "k": int(k),
        "gmm": model,
        "labels": labels,
        "bic": bic,
        "aic": aic,
        "masked_bic": bic,
        "masked_aic": aic,
        "silhouette": sil,
        "davies_bouldin": db,
        "min_cluster_size": int(sizes.min()),
        "min_size_ok": bool(sizes.min() >= min_size_threshold),
        "min_size_threshold": int(min_size_threshold),
        "actual_cluster_backend": "masked_diag_gmm",
    }


def compute_masked_stability_score(
    features: np.ndarray,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
    k: int,
    n_runs: int = 5,
    random_state: int = 42,
) -> float:
    if n_runs < 2:
        return 1.0
    labels_by_run: List[np.ndarray] = []
    for run in range(int(n_runs)):
        model = MaskedDiagonalGMM(
            n_components=int(k),
            block_slices=block_slices,
            random_state=int(random_state) + run * 1000,
            n_init=1,
            max_iter=80,
        ).fit(features, block_mask=block_mask)
        labels_by_run.append(model.predict(features, block_mask=block_mask))
    scores: List[float] = []
    for i in range(len(labels_by_run)):
        for j in range(i + 1, len(labels_by_run)):
            scores.append(adjusted_rand_score(labels_by_run[i], labels_by_run[j]))
    return float(np.mean(scores)) if scores else 1.0


def search_masked_diag_gmm_composite(
    features: np.ndarray,
    config: KSelectionConfig,
    *,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
    view_mask: Optional[np.ndarray] = None,
    semantic: bool = False,
    affect_labels: Optional[np.ndarray] = None,
) -> KSearchResult:
    """K search using the same masked likelihood model used for final assignment."""
    if str(config.covariance_type).lower() != "diag":
        raise ValueError("MaskedDiagonalGMM K search requires covariance_type='diag'.")
    matrix = np.asarray(features, dtype=np.float32)
    mask = np.asarray(block_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != matrix.shape[0]:
        raise ValueError(f"block_mask must have shape [N, B], got {mask.shape} for features {matrix.shape}.")
    effective_config = (
        replace(
            config,
            w_bic=0.25,
            w_silhouette=0.25,
            w_min_size=0.25,
            w_stability=0.25,
            w_mask_purity=max(float(config.w_mask_purity), 0.25),
        )
        if semantic
        else config
    )
    results = [
        _fit_single_masked_k(matrix, mask, block_slices, k, effective_config)
        for k in range(int(effective_config.k_min), int(effective_config.k_max) + 1)
    ]
    if not results:
        raise RuntimeError("MaskedDiagonalGMM search produced no results.")
    for result in results:
        result["stability"] = compute_masked_stability_score(
            matrix,
            mask,
            block_slices,
            int(result["k"]),
            n_runs=int(effective_config.stability_runs),
            random_state=int(effective_config.random_state),
        )

    mask_nmi_arr: Dict[int, float] = {}
    max_mask_purity_arr: Dict[int, float] = {}
    if view_mask is not None and effective_config.w_mask_purity > 0:
        for result in results:
            mask_nmi_arr[int(result["k"])] = _mask_nmi(result["labels"], view_mask)
            max_mask_purity_arr[int(result["k"])] = _max_cluster_mask_purity(result["labels"], view_mask)

    rows = []
    for result in results:
        affect_metrics: Dict[str, Any] = {}
        if effective_config.affect_gate_enabled:
            affect_metrics = compute_affect_purity_metrics(
                result["labels"],
                affect_labels,
                min_dominant_ratio=float(effective_config.min_affect_dominant_ratio),
                max_mixed_cluster_fraction=float(effective_config.max_affect_mixed_cluster_fraction),
                min_weighted_purity=float(effective_config.min_affect_weighted_purity),
                min_valid_fraction=float(effective_config.min_affect_valid_fraction),
            )
        rows.append({
            "k": int(result["k"]),
            "bic": float(result["bic"]),
            "aic": float(result["aic"]),
            "masked_bic": float(result["masked_bic"]),
            "masked_aic": float(result["masked_aic"]),
            "silhouette": float(result["silhouette"]),
            "davies_bouldin": float(result["davies_bouldin"]),
            "min_cluster_size": int(result["min_cluster_size"]),
            "min_size_ok": bool(result["min_size_ok"]),
            "stability": float(result["stability"]),
            "mask_nmi": mask_nmi_arr.get(int(result["k"]), float("nan")),
            "max_mask_purity": max_mask_purity_arr.get(int(result["k"]), float("nan")),
            **affect_metrics,
        })
    metrics = pd.DataFrame(rows).sort_values("k", kind="stable").reset_index(drop=True)
    metrics.attrs["n_samples"] = int(matrix.shape[0])

    bic_norm = 1.0 - _normalize_series(metrics["masked_bic"].to_numpy(dtype=np.float64))
    sil_norm = _normalize_series(np.nan_to_num(metrics["silhouette"].to_numpy(dtype=np.float64), nan=0.0))
    stab_norm = _normalize_series(metrics["stability"].to_numpy(dtype=np.float64))
    threshold = _min_size_threshold(effective_config, int(matrix.shape[0]))
    size_scores = np.minimum(1.0, metrics["min_cluster_size"].to_numpy(dtype=np.float64) / max(threshold, 1))
    raw_composite = (
        effective_config.w_bic * bic_norm
        + effective_config.w_silhouette * sil_norm
        + effective_config.w_min_size * size_scores
        + effective_config.w_stability * stab_norm
    )
    if mask_nmi_arr and effective_config.w_mask_purity > 0:
        mask_penalty = np.asarray(
            [
                0.5 * mask_nmi_arr.get(int(k), 0.0)
                + 0.5 * max_mask_purity_arr.get(int(k), 0.0)
                for k in metrics["k"].tolist()
            ],
            dtype=np.float64,
        )
    else:
        mask_penalty = np.zeros(len(metrics), dtype=np.float64)
    composite = raw_composite - effective_config.w_mask_purity * mask_penalty
    score_column = "semantic_composite_score" if semantic else "composite_score"
    metrics["composite_score"] = composite
    metrics[score_column] = composite

    mode = "masked_semantic_composite" if semantic else "masked_composite"
    best_idx = _select_best_index(metrics, composite, effective_config, selection_mode=mode)
    best_result = results[int(best_idx)]
    selected_k = int(best_result["k"])
    info: Dict[str, Any] = {
        "selected_k": selected_k,
        "selection_mode": mode,
        "composite_score": float(composite[best_idx]),
        "raw_composite_score": float(raw_composite[best_idx]),
        "bic_elbow_k": detect_bic_elbow({int(r["k"]): float(r["masked_bic"]) for r in results}),
        "min_cluster_size_threshold": int(threshold),
        "stability_runs": int(effective_config.stability_runs),
        "w_mask_purity": float(effective_config.w_mask_purity),
        "mask_nmi": mask_nmi_arr.get(selected_k, float("nan")),
        "max_mask_purity": max_mask_purity_arr.get(selected_k, float("nan")),
        "partial_likelihood": True,
        "feature_block_slices": [(int(start), int(stop)) for start, stop in block_slices],
        "eligible_candidate_count": int(metrics["min_size_ok"].sum()),
        "candidate_count": int(len(metrics)),
        "cluster_backend": "masked_diag_gmm",
        "eval_backend": effective_config.eval_backend,
        "actual_cluster_backend": "masked_diag_gmm",
        "actual_eval_backend": _resolve_eval_backend_name(effective_config),
        "device": str(effective_config.device),
        "silhouette_mode": effective_config.silhouette_mode,
        "silhouette_sample_size": int(effective_config.silhouette_sample_size),
        "silhouette_chunk_size": int(effective_config.silhouette_chunk_size),
        "mask_penalty_applied": float(effective_config.w_mask_purity * mask_penalty[best_idx]),
    }
    if semantic:
        info["semantic_terms"] = {
            "masked_bic": effective_config.w_bic,
            "silhouette": effective_config.w_silhouette,
            "min_size": effective_config.w_min_size,
            "stability": effective_config.w_stability,
            "mask_purity_penalty": effective_config.w_mask_purity,
        }
    if effective_config.affect_gate_enabled:
        _add_affect_selection_info(info, metrics.loc[best_idx], effective_config)
    return KSearchResult(
        best_k=selected_k,
        best_model=best_result["gmm"],
        metrics=metrics,
        selection_info=info,
    )


def search_macro_micro_diffaware(
    features: np.ndarray,
    config: KSelectionConfig,
    *,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
    view_mask: Optional[np.ndarray] = None,
    affect_labels: Optional[np.ndarray] = None,
) -> KSearchResult:
    """True macro/micro search for two/three-block diff-aware feature matrices."""
    if str(config.covariance_type).lower() != "diag":
        raise ValueError("macro_micro K search requires covariance_type='diag'.")
    matrix = np.asarray(features, dtype=np.float32)
    mask = np.asarray(block_mask, dtype=bool)
    if len(block_slices) not in {2, 3}:
        raise ValueError("macro_micro K search requires block_slices: consensus, tension[, metadata].")
    if mask.ndim != 2 or mask.shape != (matrix.shape[0], len(block_slices)):
        raise ValueError(
            f"block_mask must have shape [N, {len(block_slices)}], got {mask.shape} for features {matrix.shape}."
        )

    min_size_threshold = _min_size_threshold(config, int(matrix.shape[0]))
    rows: List[Dict[str, Any]] = []
    candidates: Dict[int, MacroMicroClusterer] = {}
    consensus_start, consensus_stop = int(block_slices[0][0]), int(block_slices[0][1])
    consensus = matrix[:, consensus_start:consensus_stop]

    for macro_k in range(int(config.macro_k_min), int(config.macro_k_max) + 1):
        model = MacroMicroClusterer(
            macro_k=int(macro_k),
            block_slices=block_slices,
            covariance_type=str(config.covariance_type),
            random_state=int(config.random_state),
            n_init=int(config.n_init),
            max_iter=100,
            micro_k_min=int(config.micro_k_min),
            micro_k_max=int(config.micro_k_max),
            min_cluster_size=int(min_size_threshold),
        ).fit(matrix, block_mask=mask)
        labels = model.labels_.astype(np.int64)
        macro_labels = model.macro_labels_.astype(np.int64)
        sizes = np.bincount(labels, minlength=int(model.n_components))
        total_k_ok = int(config.k_min) <= int(model.n_components) <= int(config.k_max)
        macro_sil = _score_silhouette(consensus, macro_labels, config) if len(np.unique(macro_labels)) > 1 else 0.0
        final_sil = _score_masked_silhouette(matrix, labels, mask, block_slices) if len(np.unique(labels)) > 1 else 0.0
        stability = compute_macro_micro_bootstrap_stability(
            matrix,
            mask,
            block_slices,
            labels,
            int(macro_k),
            config,
            int(config.stability_runs),
        )
        mask_nmi = _mask_nmi(labels, view_mask) if view_mask is not None else float("nan")
        max_mask_purity = _max_cluster_mask_purity(labels, view_mask) if view_mask is not None else float("nan")
        affect_metrics: Dict[str, Any] = {}
        if config.affect_gate_enabled:
            macro_affect_metrics = compute_affect_purity_metrics(
                macro_labels,
                affect_labels,
                min_dominant_ratio=float(config.min_affect_dominant_ratio),
                max_mixed_cluster_fraction=float(config.max_affect_mixed_cluster_fraction),
                min_weighted_purity=float(config.min_affect_weighted_purity),
                min_valid_fraction=float(config.min_affect_valid_fraction),
            )
            final_affect_metrics = compute_affect_purity_metrics(
                labels,
                affect_labels,
                min_dominant_ratio=float(config.min_affect_dominant_ratio),
                max_mixed_cluster_fraction=float(config.max_affect_mixed_cluster_fraction),
                min_weighted_purity=float(config.min_affect_weighted_purity),
                min_valid_fraction=float(config.min_affect_valid_fraction),
            )
            affect_metrics = {
                **macro_affect_metrics,
                **_prefix_affect_metrics(final_affect_metrics, "final"),
                "affect_gate_level": "macro",
            }
        score = 0.35 * float(macro_sil) + 0.25 * float(final_sil) + 0.25 * float(stability["seed_ari_mean"])
        if np.isfinite(mask_nmi):
            score -= 0.25 * float(mask_nmi)
        candidates[int(macro_k)] = model
        rows.append(
            {
                "macro_k": int(macro_k),
                "total_clusters": int(model.n_components),
                "total_k_ok": bool(total_k_ok),
                "macro_silhouette": float(macro_sil),
                "final_silhouette": float(final_sil),
                **stability,
                "macro_micro_score": float(score),
                "min_cluster_size": int(sizes.min()),
                "min_size_ok": bool(sizes.min() >= min_size_threshold),
                "min_size_threshold": int(min_size_threshold),
                "mask_nmi": float(mask_nmi),
                "max_mask_purity": float(max_mask_purity),
                **affect_metrics,
            }
        )

    metrics = pd.DataFrame(rows).sort_values("macro_k", kind="stable").reset_index(drop=True)
    eligible = metrics["min_size_ok"].to_numpy(dtype=bool) & metrics["total_k_ok"].to_numpy(dtype=bool)
    if "affect_gate_ok" in metrics.columns:
        eligible = eligible & metrics["affect_gate_ok"].to_numpy(dtype=bool)
    if not eligible.any():
        candidates_text = ", ".join(
            (
                f"macro_k={int(row.macro_k)}:total_k={int(row.total_clusters)},"
                f"min_size={int(row.min_cluster_size)},"
                f"affect_ok={bool(getattr(row, 'affect_gate_ok', True))},"
                f"weighted={float(getattr(row, 'affect_weighted_dominant_ratio', float('nan'))):.3f},"
                f"min={float(getattr(row, 'affect_min_dominant_ratio', float('nan'))):.3f},"
                f"mixed={float(getattr(row, 'affect_mixed_cluster_fraction', float('nan'))):.3f}"
            )
            for row in metrics.itertuples(index=False)
        )
        raise ValueError(
            "No macro_micro candidate satisfied total K and min-size hard gates "
            f"(k_min={int(config.k_min)}, k_max={int(config.k_max)}, "
            f"min_cluster_size_threshold={min_size_threshold}). Candidates: {candidates_text}."
        )
    eligible_metrics = metrics[eligible]
    selected_idx = int(eligible_metrics["macro_micro_score"].idxmax())
    selected_row = metrics.loc[selected_idx]
    selected_macro_k = int(selected_row["macro_k"])
    best_model = candidates[selected_macro_k]
    selection_info = {
        "selected_k": int(best_model.n_components),
        "selection_mode": "macro_micro_diffaware",
        "macro_k": int(best_model.macro_k),
        "micro_details": best_model.info.get("micro_details", {}),
        "label_names": best_model.label_names,
        "macro_micro_score": float(selected_row["macro_micro_score"]),
        "total_k_ok": bool(selected_row["total_k_ok"]),
        "min_cluster_size_threshold": int(min_size_threshold),
        "min_cluster_size": int(selected_row["min_cluster_size"]),
        "min_size_ok": bool(selected_row["min_size_ok"]),
        "seed_ari_mean": float(selected_row["seed_ari_mean"]),
        "seed_ari_std": float(selected_row["seed_ari_std"]),
        "cluster_jaccard_mean": float(selected_row["cluster_jaccard_mean"]),
        "cluster_jaccard_min": float(selected_row["cluster_jaccard_min"]),
        "bootstrap_valid_rate": float(selected_row["bootstrap_valid_rate"]),
        "stability_runs": int(config.stability_runs),
        "partial_likelihood": True,
        "feature_block_slices": [(int(start), int(stop)) for start, stop in block_slices],
        "cluster_backend": "macro_micro_masked_diag_gmm",
        "eval_backend": config.eval_backend,
        "actual_cluster_backend": "macro_micro_masked_diag_gmm",
        "actual_eval_backend": _resolve_eval_backend_name(config),
        "device": str(config.device),
        "silhouette_mode": str(config.silhouette_mode),
        "silhouette_sample_size": int(config.silhouette_sample_size),
        "silhouette_chunk_size": int(config.silhouette_chunk_size),
    }
    if config.affect_gate_enabled:
        _add_affect_selection_info(selection_info, selected_row, config)
        selection_info["affect_gate_level"] = str(selected_row.get("affect_gate_level", "macro"))
        for field in (
            "final_affect_valid_fraction",
            "final_affect_weighted_dominant_ratio",
            "final_affect_min_dominant_ratio",
            "final_affect_mixed_cluster_fraction",
            "final_affect_nmi",
        ):
            value = selected_row.get(field, float("nan"))
            selection_info[field] = float(value) if pd.notna(value) else float("nan")
        selection_info["final_affect_gate_ok"] = bool(selected_row.get("final_affect_gate_ok", False))
    return KSearchResult(
        best_k=int(best_model.n_components),
        best_model=best_model,
        metrics=metrics,
        selection_info=selection_info,
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
    cluster_backend = resolve_cluster_backend(
        effective_config.cluster_backend,
        device=effective_config.device,
        algorithm="gmm",
        covariance_type=covariance_type,
    )
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
        min_sizes = ", ".join(
            f"K={int(row.k)}:{int(row.min_cluster_size)}"
            for row in metrics[["k", "min_cluster_size"]].itertuples(index=False)
        )
        raise ValueError(
            f"No K candidate satisfied min_cluster_size_threshold={min_size_threshold} "
            f"for bic_only. Candidate minimum sizes: {min_sizes}."
        )
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
            **_backend_runtime_info(
                effective_config,
                actual_cluster_backend=cluster_backend.name,
                actual_eval_backend=_resolve_eval_backend_name(effective_config),
            ),
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
        gmm = fit_gaussian_mixture_robust(
            features,
            n_components=mk,
            covariance_type=config.covariance_type,
            reg_covar=1e-5,
            n_init=config.n_init,
            max_iter=300,
            random_state=config.random_state,
            require_converged=True,
            context=f"hierarchical macro GMM k={mk}",
        )
        labels = gmm.predict(features)
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
            gmm = fit_gaussian_mixture_robust(
                cluster_features,
                n_components=micro_k,
                covariance_type=config.covariance_type,
                reg_covar=1e-5,
                n_init=config.n_init,
                max_iter=300,
                random_state=config.random_state,
                require_converged=True,
                context=f"hierarchical micro GMM macro={macro_id} k={micro_k}",
            )
            micro_labels = gmm.predict(cluster_features)
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
            **_backend_runtime_info(
                config,
                actual_cluster_backend="sklearn",
                actual_eval_backend="sklearn",
            ),
        },
    )
