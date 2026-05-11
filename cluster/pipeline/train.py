from __future__ import annotations

import argparse
import colorsys
import json
import os
import pickle
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cluster.config import MUSIC_LABEL_NAMES, parse_split_protocol
from cluster.backends.gmm_convergence import fit_gaussian_mixture_robust
from cluster.features.block_scaler import BlockwiseObservedScaler
from cluster.features.va_geometry import (
    BALANCED_VA_DIFF_DIM,
    VA_GEOMETRY_FEATURE_NAMES,
    VA_GEOMETRY_OBSERVED_NAMES,
    VA_GEOMETRY_OBSERVED_DIM,
    build_balanced_va_diff_features,
    impute_unobserved_pairwise,
)
from cluster.pipeline.k_selection import (
    KSelectionConfig,
    KSearchResult,
    HierarchicalClusterResult,
    search_gmm_composite,
    search_gmm_semantic_composite,
    search_macro_micro_diffaware,
    search_masked_diag_gmm_composite,
    search_gmm_bic_only,
    hierarchical_cluster,
)
from cluster.models.discovery_net import (
    MusicMetadataDiscoveryNet,
    create_music_discovery_datasets,
    create_music_discovery_loader,
    extract_split_embeddings,
    initialize_discovery_runtime,
    music_discovery_dataset_filter_summary,
    save_discovery_checkpoint,
    train_music_discovery_model,
)
from cluster.data.metadata import (
    build_canonical_metadata,
    build_metadata_features,
    save_metadata_feature_bundle,
)

METADATA_POLICY_CHOICES: Tuple[str, ...] = (
    "all_metadata_upper_bound",
    "non_affective_metadata",
    "affective_va_only",
    "report_only",
)
_AFFECTIVE_METADATA_PREFIXES: Tuple[str, ...] = (
    "Moods::",
    "MoodsAll::",
    "Themes::",
    "tempo_bin::",
)
_AFFECTIVE_METADATA_EXACT: Tuple[str, ...] = (
    "numeric::num_MoodsAll",
    "cross::energy_valence",
)
_AFFECTIVE_METADATA_TERMS: Tuple[str, ...] = (
    "mood",
    "theme",
    "valence",
    "arousal",
    "energy",
    "tempo",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end music discovery pipeline: metadata build, multiview training, variable-K clustering, and reports."
    )
    parser.add_argument("--aligned_root", type=str, default=None,
                        help="Required only when --metadata_mode=rebuild_from_aligned")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split_protocol", type=str, default="70_15_15")
    parser.add_argument("--search_split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--eval_splits", type=str, default="train,val,test,all")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--run_stage",
        "--stage",
        dest="run_stage",
        type=str,
        default="train",
        choices=["train", "pretrain", "discover", "search", "finetune", "cluster_finetune", "evaluate", "full"],
    )
    parser.add_argument("--pretrain_epochs", type=int, default=None)
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--metadata_hidden_dim", type=int, default=128)
    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--metadata_aux_scale", type=float, default=0.60)
    parser.add_argument("--metadata_recon_weight", type=float, default=0.35)
    parser.add_argument("--metadata_recon_loss", type=str, default="mse", choices=["mse", "bce"])
    parser.add_argument("--fused_recon_weight", type=float, default=0.50)
    parser.add_argument("--align_weight", type=float, default=0.20)
    parser.add_argument("--metadata_align_weight", type=float, default=0.08)
    parser.add_argument("--metadata_mode", type=str, default="processed",
                        choices=["processed", "rebuild_from_aligned", "none"],
                        help="processed: use existing metadata.npy; rebuild_from_aligned: rebuild from aligned_root; none: zero metadata (ablation)")
    parser.add_argument("--min_token_freq", type=int, default=3)
    parser.add_argument("--max_tokens_per_field", type=int, default=128)
    parser.add_argument("--k_min", "--total_k_min", dest="k_min", type=int, default=4)
    parser.add_argument("--k_max", "--total_k_max", dest="k_max", type=int, default=12)
    parser.add_argument("--min_cluster_size_abs", type=int, default=20)
    parser.add_argument("--min_cluster_size_ratio", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--metadata_logit_offset", type=float, default=-0.5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", type=str, default="true", help="Enable mixed precision training (true/false)")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="0 to disable")
    parser.add_argument("--scheduler_T0", type=int, default=20, help="CosineAnnealingWarmRestarts T_0; 0 to disable")
    parser.add_argument("--scheduler_Tmult", type=int, default=2)
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-6)
    parser.add_argument("--gate_entropy_weight", type=float, default=0.01)
    parser.add_argument("--cluster_head_k", type=int, default=0,
                        help="Enable DEC/CVCL cluster head with this K; 0 disables it.")
    parser.add_argument("--auto_cluster_head", type=str, default="false")
    parser.add_argument("--cluster_head_init_from", type=str, default="random")
    parser.add_argument("--cluster_loss_warmup_epochs", type=int, default=0)
    parser.add_argument("--cluster_temperature", type=float, default=1.0,
                        help="Soft assignment temperature for the optional DEC/CVCL cluster head.")
    parser.add_argument("--cluster_loss_weight", type=float, default=0.0,
                        help="Weight for fused DEC target-distribution KL loss.")
    parser.add_argument("--cvcl_loss_weight", type=float, default=0.0,
                        help="Weight for per-view assignment alignment to fused assignments.")
    parser.add_argument("--assignment_balance_weight", type=float, default=0.0,
                        help="Weight for balanced cluster assignment regularization.")
    parser.add_argument("--consensus_va_weight", type=float, default=0.1,
                        help="Weight for consensus VA preservation loss (MSE between va_pred and mean_va)")
    parser.add_argument("--diff_preserve_weight", type=float, default=0.05,
                        help="Weight for disagreement preservation loss (SmoothL1 between latent and VA distances)")
    parser.add_argument("--diff_input_dim", type=int, default=26,
                        help="Input dimension for DiffEncoder (diff geometry features)")
    parser.add_argument("--k_strategy", type=str, default="composite",
                        choices=["composite", "semantic_composite", "macro_micro", "bic_only", "hierarchical"],
                        help="K-selection strategy: composite, semantic_composite, macro_micro, bic_only, or hierarchical")
    parser.add_argument("--covariance_type", type=str, default="diag",
                        choices=["full", "diag", "tied", "spherical"],
                        help="GMM covariance type (diag recommended; full prone to overfitting with missingness artifacts)")
    parser.add_argument("--stability_runs", type=int, default=5,
                        help="Number of GMM runs for stability scoring (composite strategy)")
    parser.add_argument("--cluster_backend", type=str, default="auto",
                        choices=["auto", "sklearn", "torch", "cuml"],
                        help="Clustering backend. auto uses GPU-capable backends when compatible with the algorithm.")
    parser.add_argument("--eval_backend", type=str, default="auto",
                        choices=["auto", "sklearn", "torch", "cuml"],
                        help="Backend used for clustering metrics such as silhouette.")
    parser.add_argument("--silhouette_mode", type=str, default="full",
                        choices=["full", "sampled", "torch_chunked", "masked_torch_chunked"],
                        help="Silhouette evaluation mode. torch_chunked avoids a full pairwise matrix.")
    parser.add_argument("--silhouette_sample_size", type=int, default=0)
    parser.add_argument("--silhouette_chunk_size", type=int, default=4096)
    parser.add_argument("--cluster_feature_strategy", type=str, default="full",
                        choices=[
                            "full",
                            "fused_residual",
                            "fused_only",
                            "fused_va_geometry",
                            "masked_diffaware",
                            "macro_micro_diffaware",
                            "partial_gmm_diffaware",
                            "mean_va",
                            "audio_va",
                            "lyrics_va",
                            "balanced_va_diff",
                            "va_geometry",
                            "mean_va_diff",
                            "original_va",
                            "metadata_only",
                            "pca_reduced",
                        ],
                        help="Clustering feature strategy")
    parser.add_argument("--pca_target_dim", type=int, default=32,
                        help="Target dimensionality for PCA reduction (pca_reduced strategy)")
    parser.add_argument("--plot_va_source", type=str, default="mean",
                        choices=["mean", "original"],
                        help="VA coordinates used in cluster scatter and summaries.")
    parser.add_argument(
        "--metadata_policy",
        type=str,
        default="all_metadata_upper_bound",
        choices=list(METADATA_POLICY_CHOICES),
        help=(
            "Controls whether metadata embeddings may enter clustering. "
            "all_metadata_upper_bound preserves legacy behavior; "
            "affective_va_only removes the metadata block from clustering; "
            "non_affective_metadata requires processed metadata without affective fields."
        ),
    )
    parser.add_argument("--metadata_cluster_weight", type=float, default=0.75)
    parser.add_argument("--conflict_cluster_weight", type=float, default=0.40)
    parser.add_argument("--gate_cluster_weight", type=float, default=0.20)
    parser.add_argument("--diff_cluster_weight", type=float, default=0.35)
    parser.add_argument("--block_scaler", type=str, default="auto", choices=["auto", "standard", "observed"])
    parser.add_argument("--run_topconf_audit", type=str, default="false")
    parser.add_argument(
        "--require_both_va",
        type=str,
        default="false",
        help="When true, keep only samples with both audio and lyrics VA in train/search/eval splits.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cluster_assignment_mode", type=str, default="joint",
                        choices=["joint", "complete_first", "partial_likelihood"],
                        help="joint: GMM fit on all samples; complete_first: fit only on both-pair samples; partial_likelihood: ignore unobserved feature blocks during prediction")
    parser.add_argument("--macro_k_min", type=int, default=4)
    parser.add_argument("--macro_k_max", type=int, default=6)
    parser.add_argument("--micro_k_min", type=int, default=1)
    parser.add_argument("--micro_k_max", type=int, default=5)
    parser.add_argument(
        "--affect_gate",
        type=str,
        default="true",
        help="When true, K selection rejects candidates with mixed VA/quadrant clusters.",
    )
    parser.add_argument("--min_affect_dominant_ratio", type=float, default=0.70)
    parser.add_argument("--max_affect_mixed_cluster_fraction", type=float, default=0.15)
    parser.add_argument("--min_affect_weighted_purity", type=float, default=0.80)
    parser.add_argument("--min_affect_valid_fraction", type=float, default=0.95)
    parser.add_argument(
        "--affect_boundary_margin",
        type=float,
        default=0.0,
        help="Exclude VA points within this distance of the 0.5 valence/arousal axes from affect hard-gate purity.",
    )
    parser.add_argument("--min_final_clusters", type=int, default=0)
    parser.add_argument("--max_final_clusters", type=int, default=0)
    parser.add_argument("--micro_min_cluster_size_abs", type=int, default=0)
    parser.add_argument("--micro_min_cluster_size_ratio", type=float, default=0.0)
    return parser


# ---------------------------------------------------------------------------
# Mask-purity diagnostics
# ---------------------------------------------------------------------------

def compute_mask_purity(
    assignments: np.ndarray,
    view_mask: np.ndarray,
) -> Dict[str, Any]:
    """Compute mask-pattern diagnostics for cluster assignments.

    Returns a dict with:
    - nmi: normalized mutual information between assignments and mask patterns
    - global_mask_distribution: baseline pattern frequencies
    - clusters: per-cluster enrichment data
    """
    from sklearn.metrics import normalized_mutual_info_score

    mask_patterns = np.array([
        "".join(["1" if v > 0 else "0" for v in row[:2]])
        for row in view_mask
    ])

    nmi = float(normalized_mutual_info_score(assignments, mask_patterns))

    global_unique, global_counts = np.unique(mask_patterns, return_counts=True)
    global_dist = {str(p): int(c) for p, c in zip(global_unique.tolist(), global_counts.tolist())}
    global_fracs = {str(p): float(c) / len(mask_patterns) for p, c in zip(global_unique.tolist(), global_counts.tolist())}

    clusters = []
    for cluster_id in sorted(np.unique(assignments).tolist()):
        cluster_mask = assignments == cluster_id
        cluster_size = int(cluster_mask.sum())
        if cluster_size == 0:
            continue
        cluster_patterns = mask_patterns[cluster_mask]
        unique_patterns, counts = np.unique(cluster_patterns, return_counts=True)
        distribution = {str(p): int(c) for p, c in zip(unique_patterns.tolist(), counts.tolist())}
        dominant_idx = int(counts.argmax())
        dominant_combo = str(unique_patterns[dominant_idx])
        purity = float(counts[dominant_idx]) / cluster_size
        baseline_frac = global_fracs.get(dominant_combo, 0.0)
        enrichment = purity / baseline_frac if baseline_frac > 0 else 0.0
        clusters.append({
            "cluster_id": int(cluster_id),
            "size": cluster_size,
            "mask_purity": round(purity, 4),
            "dominant_mask_combo": dominant_combo,
            "enrichment_vs_baseline": round(enrichment, 4),
            "mask_combo_distribution": distribution,
            "has_audio_ratio": round(float((view_mask[cluster_mask, 0] > 0).mean()), 4),
            "has_lyrics_ratio": round(float((view_mask[cluster_mask, 1] > 0).mean()), 4),
        })
    return {
        "nmi": round(nmi, 4),
        "global_mask_distribution": global_dist,
        "clusters": clusters,
    }


# ---------------------------------------------------------------------------
# Clustering feature strategies
# ---------------------------------------------------------------------------

class ClusterFeatureStrategy(Enum):
    FULL = "full"                     # z_fused + z_audio + z_lyrics + z_metadata + gate + conflict
    FUSED_RESIDUAL = "fused_residual" # z_fused + residuals + gate + conflict
    FUSED_ONLY = "fused_only"         # z_fused + gate + conflict
    FUSED_VA_GEOMETRY = "fused_va_geometry" # z_fused + VA geometry, with VA as primary axes
    MEAN_VA = "mean_va"               # raw audio/lyrics mean VA; unsupervised legacy baseline
    AUDIO_VA = "audio_va"             # raw audio VA baseline
    LYRICS_VA = "lyrics_va"           # raw lyrics VA baseline
    VA_GEOMETRY = "va_geometry"       # mean VA + circumplex audio/lyrics disagreement geometry
    MEAN_VA_DIFF = "mean_va_diff"     # legacy alias for va_geometry
    BALANCED_VA_DIFF = "balanced_va_diff"  # consensus VA + compact disagreement encoding
    ORIGINAL_VA = "original_va"       # original VA only; sanity baseline for VA-derived labels
    METADATA_ONLY = "metadata_only"   # metadata-only leakage/upper-bound baseline
    PCA_REDUCED = "pca_reduced"       # any strategy above -> PCA to target_dim
    MACRO_MICRO_DIFFAWARE = "macro_micro_diffaware"
    PARTIAL_GMM_DIFFAWARE = "partial_gmm_diffaware"


def _mean_va_features(embeddings: Dict[str, Any]) -> np.ndarray:
    mean_va = embeddings.get("mean_va")
    if mean_va is None:
        raise ValueError("cluster_feature_strategy='mean_va' requires mean_va embeddings.")
    features = np.asarray(mean_va, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(f"mean_va must have shape [N, 2], got {features.shape}.")
    return features


def _view_va_features(embeddings: Dict[str, Any], key: str, view_index: int) -> np.ndarray:
    values = embeddings.get(key)
    if values is None:
        raise ValueError(f"cluster_feature_strategy='{key}' requires {key} embeddings.")
    features = np.asarray(values, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(f"{key} must have shape [N, 2], got {features.shape}.")
    view_mask = embeddings.get("view_mask")
    if view_mask is not None and "mean_va" in embeddings:
        mask = np.asarray(view_mask, dtype=np.float32)
        missing = mask[:, int(view_index)] <= 0.0
        if missing.any():
            features = np.array(features, copy=True)
            features[missing] = _mean_va_features(embeddings)[missing]
    return features.astype(np.float32)


def _va_geometry_features(embeddings: Dict[str, Any]) -> np.ndarray:
    """Return only the 14 observed geometry features for clustering (no mask dims)."""
    geometry = embeddings.get("va_geometry")
    if geometry is None:
        raise ValueError("cluster_feature_strategy='va_geometry' requires va_geometry embeddings.")
    features = np.asarray(geometry, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"va_geometry must be 2D, got shape {features.shape}.")
    if features.shape[1] == len(VA_GEOMETRY_FEATURE_NAMES):
        # Full 17-dim: strip the last 3 mask columns for clustering
        return features[:, :VA_GEOMETRY_OBSERVED_DIM]
    elif features.shape[1] == VA_GEOMETRY_OBSERVED_DIM:
        return features
    else:
        raise ValueError(
            f"va_geometry must have shape [N, {len(VA_GEOMETRY_FEATURE_NAMES)}] or [N, {VA_GEOMETRY_OBSERVED_DIM}], got {features.shape}."
        )


def _original_va_features(embeddings: Dict[str, Any]) -> np.ndarray:
    original_va = embeddings.get("original_va")
    if original_va is None:
        raise ValueError("cluster_feature_strategy='original_va' requires original_va embeddings.")
    features = np.asarray(original_va, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(f"original_va must have shape [N, 2], got {features.shape}.")
    return features


def _balanced_va_diff_features(embeddings: Dict[str, Any], view_mask: np.ndarray) -> np.ndarray:
    audio = embeddings.get("audio_va")
    lyrics = embeddings.get("lyrics_va")
    if audio is None or lyrics is None:
        raise ValueError("cluster_feature_strategy='balanced_va_diff' requires audio_va and lyrics_va embeddings.")
    return build_balanced_va_diff_features(
        np.asarray(audio, dtype=np.float32),
        np.asarray(lyrics, dtype=np.float32),
        np.asarray(view_mask, dtype=np.float32),
    )


def _metadata_only_features(embeddings: Dict[str, Any], view_mask: np.ndarray) -> np.ndarray:
    z_metadata = embeddings.get("z_metadata")
    if z_metadata is None:
        raise ValueError("cluster_feature_strategy='metadata_only' requires z_metadata embeddings.")
    features = np.asarray(z_metadata, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"z_metadata must be 2D, got shape {features.shape}.")
    metadata_missing = np.asarray(view_mask, dtype=np.float32)[:, 2:3] <= 0.0
    return np.where(metadata_missing, 0.0, features).astype(np.float32)


def _conflict_features(embeddings: Dict[str, Any], view_mask: np.ndarray) -> np.ndarray:
    geometry = embeddings.get("va_geometry")
    if geometry is not None:
        features = np.asarray(geometry, dtype=np.float32)
        if features.ndim == 2 and features.shape[1] == len(VA_GEOMETRY_FEATURE_NAMES):
            return features[:, :VA_GEOMETRY_OBSERVED_DIM]
        elif features.ndim == 2 and features.shape[1] == VA_GEOMETRY_OBSERVED_DIM:
            return features
    # Legacy fallback: consistency + |va_diff|
    raw = np.concatenate(
        [embeddings["consistency"], np.abs(embeddings["va_diff"])],
        axis=1,
    ).astype(np.float32)
    # Per-row conditional imputation via z_fused (or mean_va) to avoid zero-variance columns
    has_both = (view_mask[:, 0] > 0) & (view_mask[:, 1] > 0)
    if has_both.any() and not has_both.all():
        consensus = embeddings.get("z_fused", embeddings.get("mean_va"))
        if consensus is None or consensus.shape[1] < 1:
            # No consensus anchor — add tiny per-row noise to break zero-variance
            rng = np.random.default_rng(42)
            for i in np.where(~has_both)[0]:
                fill = raw[has_both].mean(axis=0) + rng.normal(0, 1e-6, raw.shape[1])
                raw[i] = fill
        else:
            obs_idx = np.where(has_both)[0]
            unobs_idx = np.where(~has_both)[0]
            obs_consensus = np.asarray(consensus[obs_idx], dtype=np.float64)
            obs_values = raw[obs_idx].astype(np.float64)
            k = min(5, len(obs_idx))
            for i in unobs_idx:
                diff = obs_consensus - np.asarray(consensus[i], dtype=np.float64)
                nn = np.argpartition(np.sum(diff ** 2, axis=1), k - 1)[:k]
                raw[i] = obs_values[nn].mean(axis=0).astype(np.float32)
    return raw


def _impute_unobserved_diff_latents(
    z_diff: np.ndarray,
    view_mask: np.ndarray,
    anchors: np.ndarray,
    fitted_state: Optional[Any] = None,
    fit_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    values = np.array(z_diff, dtype=np.float32, copy=True)
    mask = np.asarray(view_mask, dtype=np.float32)
    has_both = (mask[:, 0] > 0.0) & (mask[:, 1] > 0.0)
    anchor_matrix = np.asarray(anchors, dtype=np.float32)
    if anchor_matrix.ndim != 2 or anchor_matrix.shape[0] != values.shape[0]:
        anchor_matrix = np.arange(values.shape[0], dtype=np.float32).reshape(-1, 1)

    if isinstance(fitted_state, dict) and fitted_state.get("kind") == "diff_latent_knn":
        state = fitted_state
    else:
        reference_mask = has_both.copy()
        if fit_mask is not None:
            reference_mask &= np.asarray(fit_mask, dtype=bool)
        reference_idx = np.where(reference_mask)[0]
        if reference_idx.size == 0:
            reference_idx = np.where(has_both)[0]
        if reference_idx.size == 0:
            state = {
                "kind": "diff_latent_knn",
                "anchors": np.zeros((1, anchor_matrix.shape[1]), dtype=np.float32),
                "values": np.zeros((1, values.shape[1]), dtype=np.float32),
                "k": 1,
            }
        else:
            state = {
                "kind": "diff_latent_knn",
                "anchors": anchor_matrix[reference_idx].astype(np.float32),
                "values": values[reference_idx].astype(np.float32),
                "k": int(min(5, reference_idx.size)),
            }

    unobserved_idx = np.where(~has_both)[0]
    if unobserved_idx.size:
        ref_anchors = np.asarray(state["anchors"], dtype=np.float32)
        ref_values = np.asarray(state["values"], dtype=np.float32)
        k = max(1, min(int(state.get("k", 1)), ref_values.shape[0]))
        for idx in unobserved_idx.tolist():
            diff = ref_anchors.astype(np.float64) - anchor_matrix[idx].astype(np.float64)
            nearest = np.argpartition(np.sum(diff ** 2, axis=1), k - 1)[:k]
            values[idx] = ref_values[nearest].mean(axis=0).astype(np.float32)
    return values.astype(np.float32), state


def build_cluster_features(
    embeddings: Dict[str, Any],
    metadata_cluster_weight: float,
    conflict_cluster_weight: float,
    gate_cluster_weight: float,
    strategy: str = "full",
    pca_target_dim: int = 32,
    fitted_pca: Optional[PCA] = None,
    fitted_imputation: Optional[Any] = None,
    fit_mask: Optional[np.ndarray] = None,
    diff_cluster_weight: float = 0.35,
) -> Tuple[np.ndarray, Optional[PCA], Optional[Any]]:
    """Build clustering features from embeddings using the specified strategy.

    When *strategy* is ``pca_reduced``:
    - If *fitted_pca* is ``None`` a new PCA is fit on the input (search split).
    - If *fitted_pca* is provided it is reused via ``transform`` so that all
      splits share the same feature space.

    Pairwise geometry imputation:
    - If *fitted_imputation* is ``None``, fill values are computed from observed
      rows (fit mode, use on search split).
    - If provided, those values are reused (transform mode, use on eval splits).

    Returns ``(features, pca_model, imputation_state)`` where *pca_model* is the
    fitted PCA (or ``None``). Geometry strategies return a fill vector; the
    masked_diffaware strategy returns a KNN imputation state for z_diff.
    """
    base_strategy = strategy.lower().replace("pca_reduced_", "")
    use_pca = strategy.lower().startswith("pca_reduced") or strategy.lower() == "pca_reduced"

    view_mask = embeddings.get("view_mask")
    if view_mask is None:
        n = next(iter(embeddings.values())).shape[0]
        view_mask = np.ones((n, 3), dtype=np.float32)
    view_mask = np.asarray(view_mask, dtype=np.float32)

    imputation_fill: Optional[np.ndarray] = None

    if base_strategy in {"masked_diffaware", "macro_micro_diffaware", "partial_gmm_diffaware"}:
        if base_strategy == "masked_diffaware":
            z_fused = embeddings.get("z_fused")
        else:
            z_fused = embeddings.get("z_affect")
            if z_fused is None:
                z_fused = embeddings.get("z_fused")
        if z_fused is None:
            raise ValueError(f"cluster_feature_strategy='{base_strategy}' requires z_fused or z_affect embeddings.")
        z_fused = z_fused.astype(np.float32)
        z_tension = embeddings.get("z_tension", embeddings.get("z_diff"))
        if z_tension is None:
            raise ValueError(f"cluster_feature_strategy='{base_strategy}' requires z_tension or z_diff embeddings.")
        z_tension = z_tension.astype(np.float32)
        z_tension, imputation_fill = _impute_unobserved_diff_latents(
            z_diff=z_tension,
            view_mask=view_mask,
            anchors=z_fused,
            fitted_state=fitted_imputation,
            fit_mask=fit_mask,
        )
        z_meta = embeddings["z_metadata"].astype(np.float32)
        metadata_missing = view_mask[:, 2:3] <= 0.0
        z_meta = np.where(metadata_missing, 0.0, z_meta)
        features = np.concatenate(
            [
                z_fused,
                z_tension,
                z_meta,
            ],
            axis=1,
        )
    elif base_strategy == "mean_va":
        features = _mean_va_features(embeddings)
    elif base_strategy == "audio_va":
        features = _view_va_features(embeddings, "audio_va", 0)
    elif base_strategy == "lyrics_va":
        features = _view_va_features(embeddings, "lyrics_va", 1)
    elif base_strategy == "balanced_va_diff":
        features = _balanced_va_diff_features(embeddings, view_mask)
    elif base_strategy in {"va_geometry", "mean_va_diff"}:
        raw_geom = _va_geometry_features(embeddings)
        features, imputation_fill = impute_unobserved_pairwise(raw_geom, view_mask, fitted_imputation)
    elif base_strategy == "original_va":
        features = _original_va_features(embeddings)
    elif base_strategy == "metadata_only":
        features = _metadata_only_features(embeddings, view_mask)
    else:
        features = None

    if features is None:
        z_fused = embeddings["z_fused"].astype(np.float32)
        z_audio = embeddings["z_audio"].astype(np.float32)
        z_lyrics = embeddings["z_lyrics"].astype(np.float32)
        z_metadata = float(metadata_cluster_weight) * embeddings["z_metadata"].astype(np.float32)
        # Impute missing-view embeddings with z_fused to avoid zero-vector leakage
        audio_missing = view_mask[:, 0:1] <= 0.0
        lyrics_missing = view_mask[:, 1:2] <= 0.0
        metadata_missing = view_mask[:, 2:3] <= 0.0
        z_audio = np.where(audio_missing, z_fused, z_audio)
        z_lyrics = np.where(lyrics_missing, z_fused, z_lyrics)
        z_metadata = np.where(metadata_missing, 0.0, z_metadata)
        gate = float(gate_cluster_weight) * embeddings["gate_weights"].astype(np.float32)
        raw_conflict = _conflict_features(embeddings, view_mask)
        if raw_conflict.shape[1] == VA_GEOMETRY_OBSERVED_DIM:
            raw_conflict, imputation_fill = impute_unobserved_pairwise(raw_conflict, view_mask, fitted_imputation)
        conflict = float(conflict_cluster_weight) * raw_conflict

        if base_strategy == "fused_residual":
            residual_audio = z_audio - z_fused
            residual_lyrics = z_lyrics - z_fused
            features = np.concatenate(
                [z_fused, residual_audio, residual_lyrics, z_metadata, gate, conflict],
                axis=1,
            )
        elif base_strategy == "fused_only":
            features = np.concatenate([z_fused, gate, conflict], axis=1)
        elif base_strategy == "fused_va_geometry":
            raw_geom = _va_geometry_features(embeddings)
            imputed_geom, imputation_fill = impute_unobserved_pairwise(raw_geom, view_mask, fitted_imputation)
            features = np.concatenate([z_fused, imputed_geom], axis=1)
        else:  # "full", "pca_reduced", or default
            features = np.concatenate(
                [z_fused, z_audio, z_lyrics, z_metadata, gate, conflict],
                axis=1,
            )

    pca_model: Optional[PCA] = fitted_pca
    if use_pca:
        target = min(pca_target_dim, features.shape[1], features.shape[0])
        if target < features.shape[1]:
            if pca_model is None:
                pca_model = PCA(n_components=target, random_state=42)
                if fit_mask is not None:
                    pca_model.fit(features[fit_mask])
                    features = pca_model.transform(features).astype(np.float32)
                else:
                    features = pca_model.fit_transform(features).astype(np.float32)
            else:
                features = pca_model.transform(features).astype(np.float32)

    return features.astype(np.float32), pca_model, imputation_fill


def cluster_feature_weights(
    strategy: str,
    feature_dim: int,
    *,
    conflict_cluster_weight: float,
    gate_cluster_weight: float,
    metadata_cluster_weight: float = 1.0,
    diff_cluster_weight: float = 1.0,
) -> np.ndarray:
    base_strategy = str(strategy or "full").strip().lower().replace("pca_reduced_", "")
    weights = np.ones(int(feature_dim), dtype=np.float32)
    if base_strategy in {"va_geometry", "mean_va_diff"} and int(feature_dim) == VA_GEOMETRY_OBSERVED_DIM:
        weights[0:2] = 2.0
        weights[2:VA_GEOMETRY_OBSERVED_DIM] = float(conflict_cluster_weight)
    elif base_strategy == "balanced_va_diff" and int(feature_dim) == BALANCED_VA_DIFF_DIM:
        weights[0:2] = 2.5
        weights[2:BALANCED_VA_DIFF_DIM] = float(diff_cluster_weight)
    elif base_strategy == "fused_va_geometry" and int(feature_dim) > VA_GEOMETRY_OBSERVED_DIM:
        latent_dim = int(feature_dim) - VA_GEOMETRY_OBSERVED_DIM
        weights[:latent_dim] = 0.5
        weights[latent_dim : latent_dim + 2] = 2.0
        weights[latent_dim + 2 : latent_dim + VA_GEOMETRY_OBSERVED_DIM] = float(conflict_cluster_weight)
    elif base_strategy in {"masked_diffaware", "macro_micro_diffaware", "partial_gmm_diffaware"} and int(feature_dim) % 3 == 0:
        latent_dim = int(feature_dim) // 3
        weights[latent_dim : 2 * latent_dim] = float(diff_cluster_weight)
        weights[2 * latent_dim : 3 * latent_dim] = float(metadata_cluster_weight)
    return weights.astype(np.float32)


def cluster_feature_block_slices(strategy: str, feature_dim: int) -> List[Tuple[int, int]]:
    base_strategy = str(strategy or "full").strip().lower().replace("pca_reduced_", "")
    if base_strategy == "balanced_va_diff" and int(feature_dim) == BALANCED_VA_DIFF_DIM:
        return [(0, 2), (2, int(feature_dim))]
    if base_strategy in {"masked_diffaware", "macro_micro_diffaware", "partial_gmm_diffaware"} and int(feature_dim) % 3 == 0:
        latent_dim = int(feature_dim) // 3
        return [(0, latent_dim), (latent_dim, 2 * latent_dim), (2 * latent_dim, 3 * latent_dim)]
    return [(0, int(feature_dim))]


def cluster_feature_block_mask(strategy: str, view_mask: Optional[np.ndarray], n_samples: int) -> np.ndarray:
    base_strategy = str(strategy or "full").strip().lower().replace("pca_reduced_", "")
    if view_mask is None:
        return np.ones((int(n_samples), len(cluster_feature_block_slices(base_strategy, 1))), dtype=bool)
    mask = np.asarray(view_mask, dtype=np.float32)
    if base_strategy == "balanced_va_diff":
        has_audio = mask[:, 0] > 0.0
        has_lyrics = mask[:, 1] > 0.0
        has_consensus = has_audio | has_lyrics
        block_mask = np.stack([has_consensus, has_audio & has_lyrics], axis=1).astype(bool)
        empty_rows = ~block_mask.any(axis=1)
        if empty_rows.any():
            block_mask = np.array(block_mask, copy=True)
            block_mask[empty_rows, 0] = True
        return block_mask
    if base_strategy in {"masked_diffaware", "macro_micro_diffaware", "partial_gmm_diffaware"}:
        has_audio = mask[:, 0] > 0.0
        has_lyrics = mask[:, 1] > 0.0
        has_metadata = mask[:, 2] > 0.0 if mask.shape[1] > 2 else np.ones(mask.shape[0], dtype=bool)
        has_consensus = has_audio | has_lyrics | has_metadata
        block_mask = np.stack(
            [
                has_consensus,
                has_audio & has_lyrics,
                has_metadata,
            ],
            axis=1,
        ).astype(bool)
        empty_rows = ~block_mask.any(axis=1)
        if empty_rows.any():
            block_mask = np.array(block_mask, copy=True)
            block_mask[empty_rows, 0] = True
        return block_mask
    return np.ones((mask.shape[0], 1), dtype=bool)


def _affective_metadata_features(metadata_feature_names: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for name in metadata_feature_names:
        text = str(name)
        lowered = text.lower()
        if text in _AFFECTIVE_METADATA_EXACT:
            matches.append(text)
            continue
        if any(text.startswith(prefix) for prefix in _AFFECTIVE_METADATA_PREFIXES):
            matches.append(text)
            continue
        if any(term in lowered for term in _AFFECTIVE_METADATA_TERMS):
            matches.append(text)
    return matches


def resolve_metadata_policy(
    metadata_policy: str,
    *,
    metadata_feature_names: Sequence[str],
    requested_metadata_cluster_weight: float,
) -> Dict[str, Any]:
    requested_policy = str(metadata_policy or "all_metadata_upper_bound").strip().lower()
    if requested_policy not in METADATA_POLICY_CHOICES:
        raise ValueError(f"Unsupported metadata_policy={metadata_policy!r}; expected one of {METADATA_POLICY_CHOICES}.")
    policy = "affective_va_only" if requested_policy == "report_only" else requested_policy
    requested_weight = float(requested_metadata_cluster_weight)
    affective_features = _affective_metadata_features(metadata_feature_names)
    if policy == "non_affective_metadata" and affective_features:
        examples = ", ".join(affective_features[:8])
        raise ValueError(
            "metadata_policy='non_affective_metadata' requires a processed metadata schema with affective fields removed. "
            f"Found {len(affective_features)} affective/leaky metadata features, examples: {examples}"
        )
    effective_weight = 0.0 if policy == "affective_va_only" else requested_weight
    return {
        "metadata_policy": requested_policy,
        "effective_metadata_policy": policy,
        "requested_metadata_cluster_weight": requested_weight,
        "effective_metadata_cluster_weight": float(effective_weight),
        "metadata_block_used_for_clustering": bool(effective_weight > 0.0),
        "affective_metadata_feature_count": int(len(affective_features)),
        "affective_metadata_examples": affective_features[:12],
    }


def apply_metadata_policy_to_block_mask(
    block_mask: np.ndarray,
    *,
    metadata_cluster_weight: float,
    diff_cluster_weight: float = 1.0,
) -> np.ndarray:
    mask = np.asarray(block_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"block_mask must be 2-D, got shape {mask.shape}.")
    if (
        (mask.shape[1] < 2 or float(diff_cluster_weight) > 0.0)
        and (mask.shape[1] < 3 or float(metadata_cluster_weight) > 0.0)
    ):
        return np.array(mask, copy=True)
    updated = np.array(mask, copy=True)
    if mask.shape[1] >= 2 and float(diff_cluster_weight) <= 0.0:
        updated[:, 1] = False
    if mask.shape[1] >= 3 and float(metadata_cluster_weight) <= 0.0:
        updated[:, 2] = False
    empty_rows = ~updated.any(axis=1)
    if empty_rows.any():
        updated[empty_rows, 0] = True
    return updated


def apply_cluster_feature_weights(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    vector = np.asarray(weights, dtype=np.float32).reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != vector.shape[1]:
        raise ValueError(f"Feature weights shape {vector.shape} does not match features shape {matrix.shape}.")
    return (matrix * vector).astype(np.float32)


def _uses_blockwise_observed_scaler(strategy: str, block_slices: Sequence[Sequence[int]]) -> bool:
    base_strategy = str(strategy or "full").strip().lower().replace("pca_reduced_", "")
    return base_strategy in {"masked_diffaware", "macro_micro_diffaware", "partial_gmm_diffaware"} and len(block_slices) > 1


def fit_cluster_scaler(
    strategy: str,
    features_raw: np.ndarray,
    *,
    block_mask: np.ndarray,
    block_slices: Sequence[Sequence[int]],
    fit_mask: Optional[np.ndarray] = None,
    block_scaler: str = "auto",
) -> Any:
    scaler_mode = str(block_scaler or "auto").strip().lower()
    if scaler_mode not in {"auto", "standard", "observed"}:
        raise ValueError("block_scaler must be one of: auto, standard, observed.")
    use_observed = (
        len(block_slices) > 1
        and scaler_mode == "observed"
        or (scaler_mode == "auto" and _uses_blockwise_observed_scaler(strategy, block_slices))
    )
    if use_observed:
        if fit_mask is not None:
            return BlockwiseObservedScaler(block_slices=block_slices).fit(
                np.asarray(features_raw)[fit_mask],
                block_mask=np.asarray(block_mask, dtype=bool)[fit_mask],
            )
        return BlockwiseObservedScaler(block_slices=block_slices).fit(features_raw, block_mask=np.asarray(block_mask, dtype=bool))
    if fit_mask is not None:
        return StandardScaler().fit(np.asarray(features_raw)[fit_mask])
    return StandardScaler().fit(features_raw)


def transform_cluster_features(scaler: Any, features_raw: np.ndarray, *, block_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if isinstance(scaler, BlockwiseObservedScaler):
        if block_mask is None:
            raise ValueError("BlockwiseObservedScaler transform requires block_mask.")
        return scaler.transform(features_raw, block_mask=np.asarray(block_mask, dtype=bool)).astype(np.float32)
    return scaler.transform(features_raw).astype(np.float32)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_eval_splits(text: str, search_split: str) -> List[str]:
    raw = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    valid = {"all", "train", "val", "test"}
    out: List[str] = []
    for item in raw:
        if item not in valid:
            raise ValueError(f"Unsupported split '{item}'. Expected one of {sorted(valid)}.")
        if item not in out:
            out.append(item)
    if search_split not in out:
        out.insert(0, search_split)
    return out


def parse_bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# K-selection dispatcher
# ---------------------------------------------------------------------------

def run_k_selection(
    features: np.ndarray,
    k_strategy: str,
    k_min: int,
    k_max: int,
    random_state: int,
    min_cluster_size_abs: int,
    min_cluster_size_ratio: float,
    covariance_type: str = "diag",
    stability_runs: int = 5,
    cluster_backend: str = "auto",
    eval_backend: str = "auto",
    device: str = "cpu",
    silhouette_mode: str = "full",
    silhouette_sample_size: int = 0,
    silhouette_chunk_size: int = 4096,
    view_mask: Optional[np.ndarray] = None,
    assignment_mode: str = "joint",
    block_mask: Optional[np.ndarray] = None,
    block_slices: Optional[Sequence[Sequence[int]]] = None,
    macro_k_min: int = 4,
    macro_k_max: int = 8,
    micro_k_min: int = 1,
    micro_k_max: int = 5,
    affect_labels: Optional[np.ndarray] = None,
    affect_gate_enabled: bool = False,
    min_affect_dominant_ratio: float = 0.70,
    max_affect_mixed_cluster_fraction: float = 0.15,
    min_affect_weighted_purity: float = 0.80,
    min_affect_valid_fraction: float = 0.95,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """Dispatch to the appropriate K-selection strategy.

    Returns (gmm_model_or_result, metrics_df, selection_info).
    For 'hierarchical', the first element is a HierarchicalClusterResult.
    """
    k_strategy = str(k_strategy).strip().lower()
    config = KSelectionConfig(
        k_min=k_min,
        k_max=k_max,
        covariance_type=covariance_type,
        random_state=random_state,
        min_cluster_size=min_cluster_size_abs,
        min_cluster_size_ratio=min_cluster_size_ratio,
        stability_runs=stability_runs,
        cluster_backend=cluster_backend,
        eval_backend=eval_backend,
        device=device,
        silhouette_mode=silhouette_mode,
        silhouette_sample_size=silhouette_sample_size,
        silhouette_chunk_size=silhouette_chunk_size,
        macro_k_min=int(macro_k_min),
        macro_k_max=int(macro_k_max),
        micro_k_min=int(micro_k_min),
        micro_k_max=int(micro_k_max),
        affect_gate_enabled=bool(affect_gate_enabled),
        min_affect_dominant_ratio=float(min_affect_dominant_ratio),
        max_affect_mixed_cluster_fraction=float(max_affect_mixed_cluster_fraction),
        min_affect_weighted_purity=float(min_affect_weighted_purity),
        min_affect_valid_fraction=float(min_affect_valid_fraction),
    )
    assignment_mode = str(assignment_mode or "joint").strip().lower()
    if k_strategy == "macro_micro":
        if assignment_mode == "complete_first":
            raise ValueError("macro_micro K search is incompatible with complete_first; use partial_likelihood or joint.")
        if block_mask is None or block_slices is None:
            raise ValueError("macro_micro K search requires block_mask and block_slices.")
        result = search_macro_micro_diffaware(
            features,
            config,
            block_mask=np.asarray(block_mask, dtype=bool),
            block_slices=block_slices,
            view_mask=view_mask,
            affect_labels=affect_labels,
        )
        return result.best_model, result.metrics, result.selection_info

    if assignment_mode == "partial_likelihood":
        if block_mask is None or block_slices is None:
            raise ValueError("partial_likelihood K search requires block_mask and block_slices.")
        result = search_masked_diag_gmm_composite(
            features,
            config,
            block_mask=np.asarray(block_mask, dtype=bool),
            block_slices=block_slices,
            view_mask=view_mask,
            semantic=k_strategy == "semantic_composite",
            affect_labels=affect_labels,
        )
        return result.best_model, result.metrics, result.selection_info

    if k_strategy == "composite":
        result = search_gmm_composite(features, config, view_mask=view_mask, affect_labels=affect_labels)
        return result.best_model, result.metrics, result.selection_info
    elif k_strategy == "semantic_composite":
        result = search_gmm_semantic_composite(features, config, view_mask=view_mask, affect_labels=affect_labels)
        return result.best_model, result.metrics, result.selection_info

    elif k_strategy == "hierarchical":
        hier_result = hierarchical_cluster(features, config)
        # Build a summary metrics DataFrame for reporting
        info = hier_result.info
        metrics = pd.DataFrame([{
            "macro_k": info["macro_k"],
            "macro_silhouette": info["macro_silhouette"],
            "total_clusters": info["total_clusters"],
        }])
        selection_info = {
            "selected_k": hier_result.total_clusters,
            "selection_mode": "hierarchical",
            "macro_k": hier_result.macro_k,
            "label_names": hier_result.label_names,
            "cluster_backend": info.get("cluster_backend", cluster_backend),
            "eval_backend": info.get("eval_backend", eval_backend),
            "actual_cluster_backend": info.get("actual_cluster_backend", "sklearn"),
            "actual_eval_backend": info.get("actual_eval_backend", "sklearn"),
            "device": info.get("device", device),
            "silhouette_mode": info.get("silhouette_mode", silhouette_mode),
            "silhouette_sample_size": info.get("silhouette_sample_size", silhouette_sample_size),
            "silhouette_chunk_size": info.get("silhouette_chunk_size", silhouette_chunk_size),
        }
        return hier_result, metrics, selection_info

    else:  # bic_only (legacy)
        result = search_gmm_bic_only(
            features=features,
            k_min=k_min,
            k_max=k_max,
            random_state=random_state,
            min_cluster_size_abs=min_cluster_size_abs,
            min_cluster_size_ratio=min_cluster_size_ratio,
            covariance_type=covariance_type,
            n_init=10,
            config=config,
        )
        return result.best_model, result.metrics, result.selection_info


def _safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _cluster_palette(num_clusters: int) -> Dict[int, str]:
    palette: Dict[int, str] = {}
    if num_clusters <= 0:
        return palette
    golden_ratio = 0.618033988749895
    for idx in range(int(num_clusters)):
        hue = (idx * golden_ratio) % 1.0
        sat = 0.72 if idx % 2 == 0 else 0.88
        val = 0.92 if idx % 3 != 0 else 0.78
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette[idx] = "#{:02x}{:02x}{:02x}".format(
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
        )
    return palette


def _legend_handles(palette: Dict[int, str], cluster_ids: Sequence[int]) -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[int(cluster_id)],
            markeredgecolor="none",
            markersize=7,
            label=f"Cluster {int(cluster_id)}",
        )
        for cluster_id in cluster_ids
    ]


def _build_cluster_features(
    embeddings: Dict[str, Any],
    metadata_cluster_weight: float,
    conflict_cluster_weight: float,
    gate_cluster_weight: float,
) -> np.ndarray:
    view_mask = embeddings.get("view_mask")
    if view_mask is None:
        view_mask = np.ones((embeddings["z_fused"].shape[0], 3), dtype=np.float32)
    view_mask = view_mask.astype(np.float32)
    z_fused = embeddings["z_fused"].astype(np.float32)
    z_audio = embeddings["z_audio"].astype(np.float32)
    z_lyrics = embeddings["z_lyrics"].astype(np.float32)
    z_metadata = float(metadata_cluster_weight) * embeddings["z_metadata"].astype(np.float32)
    z_audio = np.where(view_mask[:, 0:1] <= 0.0, z_fused, z_audio)
    z_lyrics = np.where(view_mask[:, 1:2] <= 0.0, z_fused, z_lyrics)
    z_metadata = np.where(view_mask[:, 2:3] <= 0.0, 0.0, z_metadata)
    conflict = _conflict_features(embeddings, view_mask)
    return np.concatenate(
        [
            z_fused,
            z_audio,
            z_lyrics,
            z_metadata,
            float(gate_cluster_weight) * embeddings["gate_weights"].astype(np.float32),
            float(conflict_cluster_weight) * conflict,
        ],
        axis=1,
    ).astype(np.float32)


# Keep legacy alias for backward compatibility
def _search_gmm(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    min_cluster_size_abs: int,
    min_cluster_size_ratio: float,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """Legacy wrapper — delegates to run_k_selection with bic_only strategy."""
    return run_k_selection(
        features=X,
        k_strategy="bic_only",
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
        min_cluster_size_abs=min_cluster_size_abs,
        min_cluster_size_ratio=min_cluster_size_ratio,
    )


def _plot_bic_curve(metrics: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics["k"], metrics["bic"], marker="o", label="BIC")
    ax.plot(metrics["k"], metrics["aic"], marker="s", label="AIC")
    if "eligible_under_size_constraint" in metrics.columns:
        eligible = metrics[metrics["eligible_under_size_constraint"]]
        if not eligible.empty:
            ax.scatter(
                eligible["k"],
                eligible["bic"],
                color="#2ca02c",
                s=45,
                label="Eligible under size constraint",
                zorder=3,
            )
    ax.set_xlabel("Number of clusters (K)")
    ax.set_ylabel("Criterion")
    ax.set_title("Variable-K GMM Search")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_training_curves(history: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss")
    ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Discovery Training Curves")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cluster_scatter(
    mean_va: np.ndarray,
    assignments: np.ndarray,
    out_path: str,
    palette: Dict[int, str],
) -> None:
    cluster_ids = sorted(np.unique(assignments).tolist())
    colors = [palette[int(item)] for item in assignments.tolist()]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        mean_va[:, 0],
        mean_va[:, 1],
        c=colors,
        s=36,
        alpha=0.85,
        edgecolors="none",
    )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Valence")
    ax.set_ylabel("Mean Arousal")
    ax.set_title("Discovered Clusters on Mean VA Plane")
    ax.legend(
        handles=_legend_handles(palette, cluster_ids),
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cluster_latent_pca(
    z_fused: np.ndarray,
    assignments: np.ndarray,
    out_path: str,
    palette: Dict[int, str],
) -> None:
    n_components = min(2, z_fused.shape[1], z_fused.shape[0])
    if n_components < 2:
        return
    coords = PCA(n_components=2, random_state=42).fit_transform(z_fused)
    cluster_ids = sorted(np.unique(assignments).tolist())
    colors = [palette[int(item)] for item in assignments.tolist()]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=34,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title("Clusters on Fused Latent PCA")
    ax.legend(
        handles=_legend_handles(palette, cluster_ids),
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cluster_feature_pca(
    features: np.ndarray,
    assignments: np.ndarray,
    out_path: str,
    palette: Dict[int, str],
) -> bool:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"cluster features must be 2D, got shape {matrix.shape}.")
    n_components = min(2, matrix.shape[1], matrix.shape[0])
    if n_components < 2:
        return False
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    cluster_ids = sorted(np.unique(assignments).tolist())
    colors = [palette[int(item)] for item in assignments.tolist()]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=34,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlabel("Feature PCA-1")
    ax.set_ylabel("Feature PCA-2")
    ax.set_title("Clusters on Actual GMM Feature PCA")
    ax.legend(
        handles=_legend_handles(palette, cluster_ids),
        title="Cluster",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def _plot_cluster_size_bar(assignments: np.ndarray, out_path: str, palette: Dict[int, str]) -> None:
    unique, counts = np.unique(assignments, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [str(int(item)) for item in unique.tolist()],
        counts.tolist(),
        color=[palette[int(item)] for item in unique.tolist()],
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Samples")
    ax.set_title("Cluster Size Distribution")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _quadrant_heatmap_matrix(assignments: np.ndarray, labels: np.ndarray) -> Tuple[List[int], np.ndarray, int]:
    cluster_ids = sorted(np.unique(assignments).tolist())
    heatmap = np.zeros((len(cluster_ids), 4), dtype=np.float32)
    valid_total = 0
    for row_idx, cluster_id in enumerate(cluster_ids):
        mask = assignments == cluster_id
        valid_labels = labels[mask].astype(np.int64)
        valid_labels = valid_labels[(valid_labels >= 0) & (valid_labels < 4)]
        valid_total += int(valid_labels.shape[0])
        counts = np.bincount(valid_labels, minlength=4).astype(np.float32)
        total = max(float(valid_labels.shape[0]), 1.0)
        heatmap[row_idx] = counts / total
    return [int(item) for item in cluster_ids], heatmap, valid_total


def _plot_quadrant_heatmap(assignments: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    cluster_ids, heatmap, valid_total = _quadrant_heatmap_matrix(assignments, labels)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.45 * len(cluster_ids))))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", norm=Normalize(vmin=0.0, vmax=1.0))
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([MUSIC_LABEL_NAMES[idx] for idx in range(4)])
    ax.set_yticks(np.arange(len(cluster_ids)))
    ax.set_yticklabels([str(int(item)) for item in cluster_ids])
    ax.set_xlabel("Quadrant")
    ax.set_ylabel("Cluster")
    ax.set_title("Quadrant Composition per Cluster")
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    if valid_total == 0:
        ax.set_title("Quadrant Composition per Cluster (no valid labels)")
        ax.text(
            1.5,
            max((len(cluster_ids) - 1) / 2.0, 0.0),
            "No valid quadrant labels",
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_diff_scatter(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    assignments: np.ndarray,
    out_path: str,
    palette: Dict[int, str],
    max_arrows: int = 300,
) -> None:
    """Scatter plot with arrows from audio VA → lyrics VA per track.

    Only plots tracks where both audio and lyrics are available.
    Arrow direction and length encode cross-modal affective tension.
    """
    both_mask = (view_mask[:, 0] > 0) & (view_mask[:, 1] > 0)
    indices = np.where(both_mask)[0]
    n_plot = min(len(indices), max_arrows)
    if n_plot == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No tracks with both audio+lyrics", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Audio -> Lyrics VA Delta")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return
    plot_idx = np.random.RandomState(42).choice(indices, size=n_plot, replace=False) if n_plot < len(indices) else indices

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx in plot_idx:
        cid = int(assignments[idx])
        color = palette.get(cid, "#888888")
        ax.annotate(
            "",
            xy=(float(lyrics_va[idx, 0]), float(lyrics_va[idx, 1])),
            xytext=(float(audio_va[idx, 0]), float(audio_va[idx, 1])),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.6),
        )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Audio → Lyrics VA Delta per Track")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cluster_ids = sorted(np.unique(assignments).tolist())
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=palette[cid], markersize=6, label=f"Cluster {cid}")
        for cid in cluster_ids if cid in palette
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_mask_distribution(
    assignments: np.ndarray,
    view_mask: np.ndarray,
    out_path: str,
    palette: Dict[int, str],
) -> None:
    """Stacked bar chart: mutually-exclusive mask-pattern composition per cluster."""
    cluster_ids = sorted(np.unique(assignments).tolist())
    n_bars = len(cluster_ids)
    has_audio = view_mask[:, 0] > 0
    has_lyrics = view_mask[:, 1] > 0
    pattern_groups = [
        ("both", has_audio & has_lyrics),
        ("audio_only", has_audio & ~has_lyrics),
        ("lyrics_only", ~has_audio & has_lyrics),
        ("neither", ~has_audio & ~has_lyrics),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(n_bars)
    bar_width = 0.6
    bottoms = np.zeros(n_bars, dtype=np.float32)

    for label, pat_mask in pattern_groups:
        values = np.zeros(n_bars)
        for ci, cid in enumerate(cluster_ids):
            cluster_mask = assignments == cid
            if cluster_mask.any():
                values[ci] = float(pat_mask[cluster_mask].mean())
        ax.bar(x, values, bar_width, bottom=bottoms, label=label)
        bottoms += values

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.set_title("Mask Pattern per Cluster")
    ax.set_xticks(x)
    ax.set_xticklabels([str(cid) for cid in cluster_ids])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_gate_profiles(assignments: np.ndarray, gate_weights: np.ndarray, out_path: str) -> None:
    cluster_ids = sorted(np.unique(assignments).tolist())
    means = []
    for cluster_id in cluster_ids:
        mask = assignments == cluster_id
        means.append(gate_weights[mask].mean(axis=0))
    mean_array = np.asarray(means, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(cluster_ids))))
    y = np.arange(len(cluster_ids))
    ax.barh(y, mean_array[:, 0], color="#4C78A8", label="audio")
    ax.barh(y, mean_array[:, 1], left=mean_array[:, 0], color="#F58518", label="lyrics")
    ax.barh(y, mean_array[:, 2], left=mean_array[:, 0] + mean_array[:, 1], color="#54A24B", label="metadata")
    ax.set_yticks(y)
    ax.set_yticklabels([str(int(item)) for item in cluster_ids])
    ax.set_xlabel("Mean gate weight")
    ax.set_ylabel("Cluster")
    ax.set_title("Average View Weight per Cluster")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _dataset_mean_va(dataset) -> np.ndarray:
    audio = dataset.raw_audio.astype(np.float32)
    lyrics = dataset.raw_lyrics.astype(np.float32)
    view_mask = getattr(dataset, "view_mask", np.ones((audio.shape[0], 3), dtype=np.float32)).astype(np.float32)
    weights = view_mask[:, 0:1] + view_mask[:, 1:2]
    summed = audio * view_mask[:, 0:1] + lyrics * view_mask[:, 1:2]
    mean_va = np.divide(
        summed,
        np.maximum(weights, 1.0),
        out=np.full_like(summed, 0.5, dtype=np.float32),
        where=weights > 0,
    )
    return mean_va.astype(np.float32)


def _dataset_plot_va(dataset, source: str = "mean") -> np.ndarray:
    source_name = str(source or "mean").strip().lower()
    if source_name == "original":
        original_va = getattr(dataset, "original_va", None)
        if original_va is None:
            raise ValueError("plot_va_source='original' requires original_va.npy in the processed dataset.")
        original_va = np.asarray(original_va, dtype=np.float32)
        if original_va.ndim != 2 or original_va.shape[1] != 2:
            raise ValueError(f"original_va must have shape [N, 2], got {original_va.shape}.")
        return original_va
    if source_name != "mean":
        raise ValueError("plot_va_source must be 'mean' or 'original'.")
    return _dataset_mean_va(dataset)


def _va_quadrant_labels(va: np.ndarray, boundary_margin: float = 0.0) -> np.ndarray:
    """Derive Q1/Q2/Q3/Q4 labels from VA-plane coordinates using 0.5 as the axes.

    Points inside ``boundary_margin`` of either axis are marked -1 because their
    quadrant assignment is inherently ambiguous on a continuous VA plane.
    """
    coords = np.asarray(va, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"VA coordinates must have shape [N, 2], got {coords.shape}.")
    margin = max(float(boundary_margin), 0.0)
    valence_high = coords[:, 0] >= 0.5
    arousal_high = coords[:, 1] >= 0.5
    labels = np.full(coords.shape[0], -1, dtype=np.int64)
    away_from_boundary = (np.abs(coords[:, 0] - 0.5) >= margin) & (np.abs(coords[:, 1] - 0.5) >= margin)
    labels[away_from_boundary & valence_high & arousal_high] = 0
    labels[away_from_boundary & ~valence_high & arousal_high] = 1
    labels[away_from_boundary & ~valence_high & ~arousal_high] = 2
    labels[away_from_boundary & valence_high & ~arousal_high] = 3
    return labels


METADATA_MIN_GLOBAL_SUPPORT = 10
METADATA_MIN_CLUSTER_SUPPORT = 5


def _split_metadata_feature_name(name: str) -> Tuple[str, str]:
    text = str(name)
    if "::" not in text:
        return "metadata", text
    field, token = text.split("::", 1)
    return field, token


def _benjamini_hochberg_qvalues(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    n = float(values.size)
    adjusted = ranked * n / np.arange(1, values.size + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _cluster_summary(
    assignments: np.ndarray,
    dataset,
    metadata_feature_names: Sequence[str],
    plot_va_source: str = "mean",
) -> List[Dict[str, Any]]:
    raw_audio = dataset.raw_audio
    raw_lyrics = dataset.raw_lyrics
    raw_metadata = getattr(dataset, "raw_metadata_report", dataset.raw_metadata)
    if raw_metadata.shape[1] != len(metadata_feature_names):
        raw_metadata = dataset.raw_metadata
    view_mask = getattr(dataset, "view_mask", np.ones((raw_audio.shape[0], 3), dtype=np.float32)).astype(np.float32)
    mean_va = _dataset_plot_va(dataset, plot_va_source)
    va_quadrants = _va_quadrant_labels(mean_va)
    numeric_indices = [idx for idx, name in enumerate(metadata_feature_names) if str(name).startswith("numeric::")]
    token_indices = [idx for idx, name in enumerate(metadata_feature_names) if not str(name).startswith("numeric::")]
    token_global_rates = {}
    token_global_supports = {}
    if token_indices:
        token_values = raw_metadata[:, token_indices] > 0.0
        global_rates = token_values.mean(axis=0)
        global_supports = token_values.sum(axis=0)
        token_global_rates = {
            feature_idx: float(global_rates[rel_idx])
            for rel_idx, feature_idx in enumerate(token_indices)
        }
        token_global_supports = {
            feature_idx: int(global_supports[rel_idx])
            for rel_idx, feature_idx in enumerate(token_indices)
        }

    summaries: List[Dict[str, Any]] = []
    for cluster_id in sorted(np.unique(assignments).tolist()):
        mask = assignments == cluster_id
        cluster_size = int(np.sum(mask))
        valid_labels = va_quadrants[mask].astype(np.int64)
        valid_labels = valid_labels[(valid_labels >= 0) & (valid_labels < 4)]
        label_counts = np.bincount(valid_labels, minlength=4)
        total = max(cluster_size, 1)
        dominant_idx = int(label_counts.argmax()) if valid_labels.size else -1

        top_tokens: List[Dict[str, Any]] = []
        if token_indices:
            cluster_token_values = raw_metadata[mask][:, token_indices] > 0.0
            cluster_rates = cluster_token_values.mean(axis=0)
            supports = cluster_token_values.sum(axis=0)
            eligible_entries: List[Dict[str, Any]] = []
            for rel_idx in range(len(token_indices)):
                cluster_rate = float(cluster_rates[rel_idx])
                if cluster_rate <= 0:
                    continue
                feature_idx = token_indices[rel_idx]
                cluster_support = int(supports[rel_idx])
                global_support = int(token_global_supports.get(feature_idx, 0))
                if (
                    cluster_support < METADATA_MIN_CLUSTER_SUPPORT
                    or global_support < METADATA_MIN_GLOBAL_SUPPORT
                ):
                    continue
                population_rate = float(token_global_rates[feature_idx])
                p_value = float(
                    hypergeom.sf(
                        cluster_support - 1,
                        len(assignments),
                        global_support,
                        cluster_size,
                    )
                )
                field, token = _split_metadata_feature_name(str(metadata_feature_names[feature_idx]))
                eligible_entries.append(
                    {
                        "feature": str(metadata_feature_names[feature_idx]),
                        "field": field,
                        "token": token,
                        "cluster_rate": cluster_rate,
                        "population_rate": population_rate,
                        "enrichment": float(cluster_rate / max(population_rate, 1e-8)),
                        "support": cluster_support,
                        "global_support": global_support,
                        "p_value": p_value,
                        "mean_weight": float(raw_metadata[mask][:, feature_idx].mean()),
                    }
                )
            q_values = _benjamini_hochberg_qvalues([entry["p_value"] for entry in eligible_entries])
            for entry, q_value in zip(eligible_entries, q_values.tolist()):
                entry["q_value"] = float(q_value)
            top_tokens = sorted(
                eligible_entries,
                key=lambda item: (
                    item["q_value"],
                    item["p_value"],
                    -item["enrichment"],
                    -item["support"],
                    item["feature"],
                ),
            )[:10]

        numeric_means: Dict[str, float] = {}
        for feature_idx in numeric_indices:
            numeric_means[str(metadata_feature_names[feature_idx])] = float(raw_metadata[mask][:, feature_idx].mean())

        example_tracks: List[Dict[str, Any]] = []
        example_indices = np.where(mask)[0][:8].tolist()
        for idx in example_indices:
            item = {
                "identifier": str(dataset.identifiers[idx]),
                "lyric_identifier": str(dataset.lyric_identifiers[idx]),
            }
            if not dataset.canonical_metadata.empty:
                for field in ("Artist", "Title", "Quadrant"):
                    if field in dataset.canonical_metadata.columns:
                        item[field.lower()] = str(dataset.canonical_metadata.iloc[idx][field])
            example_tracks.append(item)

        both_mask = (view_mask[:, 0] > 0) & (view_mask[:, 1] > 0)
        pair_mask = mask & both_mask
        if pair_mask.any():
            va_delta = raw_audio[pair_mask] - raw_lyrics[pair_mask]
            diff_magnitude = float(np.mean(np.linalg.norm(va_delta, axis=1)))
            delta_direction = float(np.mean(np.abs(np.arctan2(va_delta[:, 1], va_delta[:, 0]))))
        else:
            diff_magnitude = float("nan")
            delta_direction = float("nan")

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "num_samples": cluster_size,
                "sample_fraction": float(cluster_size / len(assignments)),
                "dominant_quadrant": MUSIC_LABEL_NAMES.get(dominant_idx, str(dominant_idx)),
                "dominant_quadrant_ratio": float(label_counts[dominant_idx] / total) if dominant_idx >= 0 else 0.0,
                "mean_audio_valence": float(raw_audio[mask & (view_mask[:, 0] > 0), 0].mean()) if np.any(mask & (view_mask[:, 0] > 0)) else float("nan"),
                "mean_audio_arousal": float(raw_audio[mask & (view_mask[:, 0] > 0), 1].mean()) if np.any(mask & (view_mask[:, 0] > 0)) else float("nan"),
                "mean_lyrics_valence": float(raw_lyrics[mask & (view_mask[:, 1] > 0), 0].mean()) if np.any(mask & (view_mask[:, 1] > 0)) else float("nan"),
                "mean_lyrics_arousal": float(raw_lyrics[mask & (view_mask[:, 1] > 0), 1].mean()) if np.any(mask & (view_mask[:, 1] > 0)) else float("nan"),
                "mean_valence": float(mean_va[mask, 0].mean()),
                "mean_arousal": float(mean_va[mask, 1].mean()),
                "mean_diff_magnitude": diff_magnitude,
                "delta_direction_angle": delta_direction,
                "has_audio_ratio": round(float((view_mask[mask, 0] > 0).mean()), 4),
                "has_lyrics_ratio": round(float((view_mask[mask, 1] > 0).mean()), 4),
                "diff_observed_ratio": round(float((both_mask[mask]).mean()), 4),
                "quadrant_distribution": {
                    MUSIC_LABEL_NAMES[idx]: {
                        "count": int(label_counts[idx]),
                        "ratio": float(label_counts[idx] / total),
                    }
                    for idx in range(4)
                },
                "top_metadata_tokens": top_tokens,
                "numeric_metadata_means": numeric_means,
                "example_tracks": example_tracks,
            }
        )
    return summaries


def _normalize_cluster_label_names(cluster_label_names: Optional[Dict[Any, Any]]) -> Dict[int, str]:
    if not cluster_label_names:
        return {}
    normalized: Dict[int, str] = {}
    for key, value in dict(cluster_label_names).items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _cluster_label_names_for_outputs(k_result: Any, selection_info: Dict[str, Any]) -> Dict[int, str]:
    if isinstance(k_result, HierarchicalClusterResult):
        return _normalize_cluster_label_names(k_result.label_names)
    return _normalize_cluster_label_names(selection_info.get("label_names"))


def _parse_macro_micro_label(label: str) -> Optional[Tuple[int, int, str]]:
    text = str(label or "").strip()
    if not text.startswith("M"):
        return None
    body = text[1:]
    if "-" in body:
        macro_text, micro_label = body.split("-", 1)
    else:
        macro_text, micro_label = body, ""
    try:
        macro_id = int(macro_text)
    except ValueError:
        return None
    if micro_label == "":
        micro_id = 0
    elif len(micro_label) == 1 and micro_label.isalpha():
        micro_id = ord(micro_label.lower()) - ord("a") + 1
    else:
        try:
            micro_id = int(micro_label)
        except ValueError:
            micro_id = 0
    return macro_id, micro_id, micro_label


def _label_metadata(cluster_id: int, cluster_label_names: Dict[int, str]) -> Dict[str, Any]:
    label_name = cluster_label_names.get(int(cluster_id), str(int(cluster_id)))
    parsed = _parse_macro_micro_label(label_name)
    if parsed is None:
        return {
            "label_name": label_name,
            "macro_id": -1,
            "micro_id": -1,
            "micro_label": "",
        }
    macro_id, micro_id, micro_label = parsed
    return {
        "label_name": label_name,
        "macro_id": int(macro_id),
        "micro_id": int(micro_id),
        "micro_label": micro_label,
    }


def _build_assignment_frame(
    assignments: np.ndarray,
    dataset,
    embeddings: Dict[str, Any],
    plot_va_source: str = "mean",
    cluster_label_names: Optional[Dict[Any, Any]] = None,
) -> pd.DataFrame:
    mean_va = _dataset_plot_va(dataset, plot_va_source)
    label_map = _normalize_cluster_label_names(cluster_label_names)
    frame = pd.DataFrame(
        {
            "identifier": dataset.identifiers,
            "lyric_identifier": dataset.lyric_identifiers,
            "cluster_id": assignments.astype(int),
            "raw_label_id": dataset.labels.astype(int),
            "raw_label_name": [MUSIC_LABEL_NAMES.get(int(item), str(item)) for item in dataset.labels.tolist()],
            "audio_valence": dataset.raw_audio[:, 0],
            "audio_arousal": dataset.raw_audio[:, 1],
            "lyrics_valence": dataset.raw_lyrics[:, 0],
            "lyrics_arousal": dataset.raw_lyrics[:, 1],
            "mean_valence": mean_va[:, 0],
            "mean_arousal": mean_va[:, 1],
            "has_audio": getattr(dataset, "view_mask", np.ones((len(assignments), 3)))[:, 0].astype(bool),
            "has_lyrics": getattr(dataset, "view_mask", np.ones((len(assignments), 3)))[:, 1].astype(bool),
            "has_metadata": getattr(dataset, "view_mask", np.ones((len(assignments), 3)))[:, 2].astype(bool),
            "consistency": dataset.consistency,
            "gate_audio": embeddings["gate_weights"][:, 0],
            "gate_lyrics": embeddings["gate_weights"][:, 1],
            "gate_metadata": embeddings["gate_weights"][:, 2],
        }
    )
    if label_map:
        label_rows = [_label_metadata(int(cluster_id), label_map) for cluster_id in assignments.tolist()]
        frame["label_name"] = [row["label_name"] for row in label_rows]
        frame["macro_id"] = [row["macro_id"] for row in label_rows]
        frame["micro_id"] = [row["micro_id"] for row in label_rows]
        frame["micro_label"] = [row["micro_label"] for row in label_rows]
    if not dataset.canonical_metadata.empty:
        for field in ("Artist", "Title", "Quadrant"):
            if field in dataset.canonical_metadata.columns:
                frame[field.lower()] = dataset.canonical_metadata[field].astype(str).tolist()
    return frame


def _metadata_token_stats_text(tokens: Sequence[Dict[str, Any]], limit: int = 5) -> str:
    return "; ".join(
        (
            f"{entry.get('field', 'metadata')}::{entry.get('token', str(entry['feature']).split('::', 1)[-1])}"
            f"(n={int(entry.get('support', 0))},"
            f"N={int(entry.get('global_support', 0))},"
            f"q={float(entry.get('q_value', 1.0)):.3g})"
        )
        for entry in list(tokens)[:limit]
    )


MACRO_MICRO_ENRICHMENT_COLUMNS = [
    "cluster_id",
    "label_name",
    "macro_id",
    "micro_id",
    "micro_label",
    "feature",
    "field",
    "token",
    "support",
    "global_support",
    "cluster_rate",
    "population_rate",
    "enrichment",
    "p_value",
    "q_value",
]


def _write_macro_micro_artifacts(
    out_dir: str,
    dataset,
    assignments: np.ndarray,
    summary: Sequence[Dict[str, Any]],
    cluster_label_names: Dict[int, str],
    palette: Dict[int, str],
) -> Dict[str, Any]:
    label_rows = {
        int(item["cluster_id"]): _label_metadata(int(item["cluster_id"]), cluster_label_names)
        for item in summary
    }
    macro_ids = sorted(
        {
            int(row["macro_id"])
            for row in label_rows.values()
            if int(row.get("macro_id", -1)) > 0
        }
    )
    if not macro_ids:
        return {}

    macro_dir = os.path.join(out_dir, "macro_micro")
    _ensure_dir(macro_dir)
    summary_rows: List[Dict[str, Any]] = []
    enrichment_rows: List[Dict[str, Any]] = []
    for item in summary:
        cluster_id = int(item["cluster_id"])
        label_data = label_rows[cluster_id]
        if int(label_data["macro_id"]) <= 0:
            continue
        token_stats = _metadata_token_stats_text(item.get("top_metadata_tokens", []))
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                **label_data,
                "num_samples": int(item["num_samples"]),
                "sample_fraction": float(item["sample_fraction"]),
                "dominant_quadrant": str(item["dominant_quadrant"]),
                "dominant_quadrant_ratio": float(item["dominant_quadrant_ratio"]),
                "mean_valence": float(item["mean_valence"]),
                "mean_arousal": float(item["mean_arousal"]),
                "mean_diff_magnitude": float(item.get("mean_diff_magnitude", float("nan"))),
                "delta_direction_angle": float(item.get("delta_direction_angle", float("nan"))),
                "diff_observed_ratio": float(item.get("diff_observed_ratio", 0.0)),
                "has_audio_ratio": float(item.get("has_audio_ratio", 0.0)),
                "has_lyrics_ratio": float(item.get("has_lyrics_ratio", 0.0)),
                "top_metadata_token_stats": token_stats,
            }
        )
        for token in item.get("top_metadata_tokens", []):
            enrichment_rows.append(
                {
                    "cluster_id": cluster_id,
                    **label_data,
                    "feature": str(token["feature"]),
                    "field": str(token.get("field", "")),
                    "token": str(token.get("token", "")),
                    "support": int(token.get("support", 0)),
                    "global_support": int(token.get("global_support", 0)),
                    "cluster_rate": float(token.get("cluster_rate", 0.0)),
                    "population_rate": float(token.get("population_rate", 0.0)),
                    "enrichment": float(token.get("enrichment", 0.0)),
                    "p_value": float(token.get("p_value", 1.0)),
                    "q_value": float(token.get("q_value", 1.0)),
                }
            )

    macro_summary_path = os.path.join(out_dir, "macro_micro_summary.csv")
    macro_enrichment_path = os.path.join(out_dir, "macro_micro_metadata_enrichment.csv")
    pd.DataFrame(summary_rows).to_csv(macro_summary_path, index=False, encoding="utf-8")
    enrichment_frame = pd.DataFrame(enrichment_rows, columns=MACRO_MICRO_ENRICHMENT_COLUMNS)
    enrichment_frame.to_csv(macro_enrichment_path, index=False, encoding="utf-8")

    macro_diff_paths: Dict[str, str] = {}
    macro_metadata_paths: Dict[str, str] = {}
    view_mask = dataset.view_mask if hasattr(dataset, "view_mask") else np.ones((len(assignments), 3), dtype=np.float32)
    for macro_id in macro_ids:
        cluster_ids = [
            cluster_id
            for cluster_id, label_data in label_rows.items()
            if int(label_data["macro_id"]) == int(macro_id)
        ]
        macro_mask = np.isin(assignments, np.asarray(cluster_ids, dtype=np.int64))
        if macro_mask.any():
            diff_path = os.path.join(macro_dir, f"macro_{macro_id}_diff_arrow.png")
            _plot_diff_scatter(
                dataset.raw_audio[macro_mask],
                dataset.raw_lyrics[macro_mask],
                view_mask[macro_mask],
                assignments[macro_mask],
                diff_path,
                {cluster_id: palette.get(cluster_id, "#888888") for cluster_id in cluster_ids},
            )
            macro_diff_paths[str(macro_id)] = diff_path
        macro_metadata_path = os.path.join(macro_dir, f"macro_{macro_id}_metadata_enrichment.csv")
        macro_enrichment = enrichment_frame[enrichment_frame["macro_id"] == int(macro_id)] if not enrichment_frame.empty else enrichment_frame
        macro_enrichment.to_csv(macro_metadata_path, index=False, encoding="utf-8")
        macro_metadata_paths[str(macro_id)] = macro_metadata_path

    return {
        "macro_micro_dir": macro_dir,
        "macro_micro_summary": macro_summary_path,
        "macro_micro_metadata_enrichment": macro_enrichment_path,
        "macro_diff_arrows": macro_diff_paths,
        "macro_metadata_tables": macro_metadata_paths,
    }


def _write_split_outputs(
    out_dir: str,
    split: str,
    dataset,
    embeddings: Dict[str, Any],
    assignments: np.ndarray,
    metadata_feature_names: Sequence[str],
    selected_k: int,
    feature_dim: int,
    cluster_features: Optional[np.ndarray] = None,
    search_metrics: Optional[pd.DataFrame] = None,
    plot_va_source: str = "mean",
    cluster_label_names: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Any]:
    _ensure_dir(out_dir)
    mean_va = _dataset_plot_va(dataset, plot_va_source)
    palette = _cluster_palette(int(selected_k))
    label_map = _normalize_cluster_label_names(cluster_label_names)
    summary = _cluster_summary(
        assignments=assignments,
        dataset=dataset,
        metadata_feature_names=metadata_feature_names,
        plot_va_source=plot_va_source,
    )
    assignment_frame = _build_assignment_frame(
        assignments=assignments,
        dataset=dataset,
        embeddings=embeddings,
        plot_va_source=plot_va_source,
        cluster_label_names=label_map,
    )
    catalog_frame = pd.DataFrame(
        [
            {
                "cluster_id": int(item["cluster_id"]),
                "num_samples": int(item["num_samples"]),
                "sample_fraction": float(item["sample_fraction"]),
                "dominant_quadrant": str(item["dominant_quadrant"]),
                "dominant_quadrant_ratio": float(item["dominant_quadrant_ratio"]),
                "mean_valence": float(item["mean_valence"]),
                "mean_arousal": float(item["mean_arousal"]),
                "top_metadata_tokens": ", ".join(
                    str(entry["feature"]).split("::", 1)[-1]
                    for entry in item.get("top_metadata_tokens", [])[:5]
                ),
                "top_metadata_token_stats": _metadata_token_stats_text(item.get("top_metadata_tokens", [])),
            }
            for item in summary
        ]
    )
    if label_map and not catalog_frame.empty:
        catalog_label_frame = pd.DataFrame(
            [_label_metadata(int(cluster_id), label_map) for cluster_id in catalog_frame["cluster_id"].tolist()]
        )
        insert_at = 1
        for column in ("label_name", "macro_id", "micro_id", "micro_label"):
            catalog_frame.insert(insert_at, column, catalog_label_frame[column].tolist())
            insert_at += 1

    assignment_path = os.path.join(out_dir, "cluster_assignments.csv")
    catalog_path = os.path.join(out_dir, "cluster_catalog.csv")
    scatter_path = os.path.join(out_dir, "cluster_scatter_mean_va.png")
    latent_pca_path = os.path.join(out_dir, "cluster_pca_fused.png")
    feature_pca_path = os.path.join(out_dir, "cluster_pca_features.png")
    size_bar_path = os.path.join(out_dir, "cluster_size_bar.png")
    quadrant_heatmap_path = os.path.join(out_dir, "cluster_quadrant_heatmap.png")
    gate_profile_path = os.path.join(out_dir, "cluster_gate_profile.png")
    diff_scatter_path = os.path.join(out_dir, "cluster_diff_arrow.png")
    mask_dist_path = os.path.join(out_dir, "cluster_mask_distribution.png")
    palette_path = os.path.join(out_dir, "cluster_palette.json")
    summary_path = os.path.join(out_dir, "cluster_summary.json")

    assignment_frame.to_csv(assignment_path, index=False, encoding="utf-8")
    catalog_frame.to_csv(catalog_path, index=False, encoding="utf-8")
    _plot_cluster_scatter(mean_va, assignments, scatter_path, palette)
    _plot_cluster_latent_pca(embeddings["z_fused"], assignments, latent_pca_path, palette)
    feature_pca_written = False
    if cluster_features is not None:
        feature_pca_written = _plot_cluster_feature_pca(cluster_features, assignments, feature_pca_path, palette)
    _plot_cluster_size_bar(assignments, size_bar_path, palette)
    _plot_quadrant_heatmap(assignments, _va_quadrant_labels(mean_va), quadrant_heatmap_path)
    _plot_gate_profiles(assignments, embeddings["gate_weights"], gate_profile_path)
    _dataset_view_mask = dataset.view_mask if hasattr(dataset, "view_mask") else np.ones((len(assignments), 3), dtype=np.float32)
    _plot_diff_scatter(dataset.raw_audio if hasattr(dataset, "raw_audio") else np.zeros((len(assignments), 2)),
                       dataset.raw_lyrics if hasattr(dataset, "raw_lyrics") else np.zeros((len(assignments), 2)),
                       _dataset_view_mask, assignments, diff_scatter_path, palette)
    _plot_mask_distribution(assignments, _dataset_view_mask, mask_dist_path, palette)
    macro_micro_outputs = (
        _write_macro_micro_artifacts(
            out_dir=out_dir,
            dataset=dataset,
            assignments=assignments,
            summary=summary,
            cluster_label_names=label_map,
            palette=palette,
        )
        if label_map
        else {}
    )
    with open(palette_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in palette.items()}, f, ensure_ascii=False, indent=2)

    bic_curve_path = None
    search_metrics_path = None
    if search_metrics is not None:
        search_metrics_path = os.path.join(out_dir, "cluster_search_metrics.csv")
        search_metrics.to_csv(search_metrics_path, index=False, encoding="utf-8")
        # Only plot BIC curve when metrics contain the required k/bic/aic columns
        if {"k", "bic", "aic"}.issubset(search_metrics.columns):
            bic_curve_path = os.path.join(out_dir, "bic_curve.png")
            _plot_bic_curve(search_metrics, bic_curve_path)

    payload = {
        "split": split,
        "selected_k": int(selected_k),
        "feature_dim": int(feature_dim),
        "num_samples": int(len(assignments)),
        "plot_va_source": str(plot_va_source),
        "metadata_token_thresholds": {
            "min_global_support": int(METADATA_MIN_GLOBAL_SUPPORT),
            "min_cluster_support": int(METADATA_MIN_CLUSTER_SUPPORT),
            "fdr_method": "benjamini_hochberg",
            "p_value_test": "hypergeometric_right_tail",
        },
        "cluster_summary": summary,
        "output_files": {
            "cluster_assignments": assignment_path,
            "cluster_catalog": catalog_path,
            "cluster_summary": summary_path,
            "cluster_scatter": scatter_path,
            "cluster_pca_fused": latent_pca_path,
            "cluster_pca_features": feature_pca_path if feature_pca_written else None,
            "cluster_size_bar": size_bar_path,
            "cluster_quadrant_heatmap": quadrant_heatmap_path,
            "cluster_gate_profile": gate_profile_path,
            "cluster_diff_arrow": diff_scatter_path,
            "cluster_mask_distribution": mask_dist_path,
            "cluster_palette": palette_path,
            "search_metrics": search_metrics_path,
            "bic_curve": bic_curve_path,
            **macro_micro_outputs,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _write_pipeline_report(out_path: str, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Music Discovery Training Pipeline")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Processed dataset: `{summary['processed_dir']}`")
    lines.append(f"- Search split: `{summary['search_split']}`")
    lines.append(f"- Evaluated splits: `{', '.join(summary['eval_splits'])}`")
    lines.append(f"- Selected K: `{summary['selected_k']}`")
    if "selection_mode" in summary:
        lines.append(f"- K selection mode: `{summary['selection_mode']}`")
    if "actual_cluster_backend" in summary:
        lines.append(f"- Cluster backend: `{summary.get('cluster_backend')}` -> `{summary['actual_cluster_backend']}`")
    if "actual_eval_backend" in summary:
        lines.append(f"- Eval backend: `{summary.get('eval_backend')}` -> `{summary['actual_eval_backend']}`")
    if "min_cluster_size_threshold" in summary:
        lines.append(f"- Min cluster size threshold: `{summary['min_cluster_size_threshold']}`")
    lines.append(f"- Training epochs: `{summary['epochs']}`")
    lines.append(f"- Latent dim: `{summary['latent_dim']}`")
    lines.append(f"- DEC/CVCL head K: `{summary.get('cluster_head_k', 0)}`")
    if "cluster_feature_strategy" in summary:
        lines.append(f"- Cluster feature strategy: `{summary['cluster_feature_strategy']}`")
    if "metadata_policy" in summary:
        policy = summary["metadata_policy"]
        if isinstance(policy, dict):
            lines.append(
                f"- Metadata policy: `{policy.get('metadata_policy')}` "
                f"(cluster weight={float(policy.get('effective_metadata_cluster_weight', 0.0)):.3g})"
            )
        else:
            lines.append(f"- Metadata policy: `{policy}`")
    if "require_both_va" in summary:
        lines.append(f"- Complete audio+lyrics VA subset: `{bool(summary['require_both_va'])}`")
        filter_summary = summary.get("dataset_filter_summary", {}) or {}
        split_counts = filter_summary.get("splits", {}) if isinstance(filter_summary, dict) else {}
        if split_counts:
            count_text = ", ".join(
                f"{split}: {int(values.get('kept_samples', 0))}/{int(values.get('original_samples', 0))}"
                for split, values in split_counts.items()
            )
            lines.append(f"- Filtered sample counts: `{count_text}`")
    if "plot_va_source" in summary:
        lines.append(f"- Plot VA source: `{summary['plot_va_source']}`")
    lines.append(f"- Metadata feature dim: `{summary['metadata_feature_dim']}`")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Model checkpoint: `{summary['checkpoint_path']}`")
    lines.append(f"- GMM bundle: `{summary['gmm_bundle_path']}`")
    lines.append(f"- Training history: `{summary['history_path']}`")
    lines.append("")
    lines.append("## Per-Split Cluster Snapshot")
    lines.append("")
    for split, payload in summary["split_outputs"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- Samples: `{payload['num_samples']}`")
        feature_pca = payload.get("output_files", {}).get("cluster_pca_features")
        if feature_pca:
            lines.append(f"- Actual GMM feature PCA: `{feature_pca}`")
        macro_summary = payload.get("output_files", {}).get("macro_micro_summary")
        if macro_summary:
            lines.append(f"- Macro/micro summary: `{macro_summary}`")
        cluster_preview = payload["cluster_summary"][:5]
        for cluster in cluster_preview:
            tokens = ", ".join(
                (
                    f"{str(item['feature']).split('::', 1)[-1]}"
                    f" (n={int(item.get('support', 0))}, q={float(item.get('q_value', 1.0)):.3g})"
                )
                for item in cluster.get("top_metadata_tokens", [])[:3]
            )
            diff_ratio = cluster.get("diff_observed_ratio", -1)
            mask_warning = ""
            if diff_ratio >= 0 and diff_ratio < 0.15:
                mask_warning = " **[WARNING: possible missingness cluster]**"
            elif cluster.get("has_lyrics_ratio", 1.0) < 0.15:
                mask_warning = " **[WARNING: possible missingness cluster]**"
            lines.append(
                f"- Cluster {cluster['cluster_id']}: size={cluster['num_samples']}, "
                f"dominant={cluster['dominant_quadrant']} ({cluster['dominant_quadrant_ratio']:.2%}), "
                f"mean_va=({cluster['mean_valence']:.3f}, {cluster['mean_arousal']:.3f}), "
                f"diff_obs={diff_ratio:.2%}, "
                f"tokens={tokens or 'n/a'}{mask_warning}"
            )
        lines.append("")
    mask_purity_data = summary.get("mask_purity_diagnostics")
    if mask_purity_data:
        nmi = mask_purity_data.get("nmi", 0.0)
        lines.append("## Mask-Purity Diagnostics")
        lines.append("")
        lines.append(f"- NMI(assignments, mask_pattern): **{nmi:.4f}**")
        global_dist = mask_purity_data.get("global_mask_distribution", {})
        if global_dist:
            dist_str = ", ".join(f"{k}={v}" for k, v in sorted(global_dist.items()))
            lines.append(f"- Global mask distribution: {dist_str}")
        lines.append("")
        for entry in mask_purity_data.get("clusters", []):
            enrichment = entry.get("enrichment_vs_baseline", 0.0)
            warning = ""
            if enrichment > 2.0 and entry["mask_purity"] > 0.7:
                warning = " **[enriched]**"
            lines.append(
                f"- Cluster {entry['cluster_id']}: purity={entry['mask_purity']:.4f}, "
                f"dominant={entry['dominant_mask_combo']}, enrichment={enrichment:.2f}x, "
                f"size={entry['size']}{warning}"
            )
        if nmi > 0.15:
            lines.append("")
            lines.append(
                f"> **WARNING**: NMI={nmi:.4f} > 0.15 indicates moderate correlation "
                f"between cluster assignments and mask patterns. Clustering may be "
                f"partially driven by data availability rather than emotion."
            )
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)
    models_dir = os.path.join(out_dir, "models")
    _ensure_dir(models_dir)

    metadata_mode = str(getattr(args, "metadata_mode", "processed")).strip().lower()
    if metadata_mode == "rebuild_from_aligned":
        if not args.aligned_root:
            parser.error("--aligned_root is required when --metadata_mode=rebuild_from_aligned")
        canonical = build_canonical_metadata(
            aligned_root=str(args.aligned_root),
            processed_dir=str(args.processed_dir),
        )
        metadata_bundle = build_metadata_features(
            canonical_metadata=canonical,
            min_token_freq=int(args.min_token_freq),
            max_tokens_per_field=int(args.max_tokens_per_field),
        )
        written_files = save_metadata_feature_bundle(metadata_bundle, out_dir=str(args.processed_dir))
        metadata_summary = {
            "num_samples": int(metadata_bundle.features.shape[0]),
            "feature_dim": int(metadata_bundle.features.shape[1]),
            "min_token_freq": int(args.min_token_freq),
            "max_tokens_per_field": int(args.max_tokens_per_field),
            "written_files": written_files,
            "metadata_mode": "rebuild_from_aligned",
        }
    elif metadata_mode == "none":
        metadata_path = os.path.join(str(args.processed_dir), "metadata.npy")
        if os.path.exists(metadata_path):
            existing = np.load(metadata_path)
            n_samples = existing.shape[0]
        else:
            n_samples = 0
        zero_metadata = np.zeros((n_samples, 1), dtype=np.float32)
        np.save(metadata_path, zero_metadata)
        names_path = os.path.join(str(args.processed_dir), "metadata_feature_names.json")
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(["numeric::zero"], f)
        metadata_summary = {
            "num_samples": n_samples,
            "feature_dim": 1,
            "metadata_mode": "none",
        }
    else:
        metadata_path = os.path.join(str(args.processed_dir), "metadata.npy")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"metadata_mode='processed' but {metadata_path} does not exist. "
                f"Run prepare_unimodal_dataset.py first or use --metadata_mode rebuild_from_aligned."
            )
        existing = np.load(metadata_path)
        metadata_summary = {
            "num_samples": int(existing.shape[0]),
            "feature_dim": int(existing.shape[1]),
            "metadata_mode": "processed",
        }
    metadata_summary_path = os.path.join(out_dir, "metadata_build_summary.json")
    with open(metadata_summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, ensure_ascii=False, indent=2)

    split_protocol = parse_split_protocol(str(args.split_protocol))
    require_both_va = parse_bool_text(args.require_both_va)
    datasets = create_music_discovery_datasets(
        data_dir=str(args.processed_dir),
        split_protocol=split_protocol,
        require_both_va=require_both_va,
    )
    dataset_filter_summary = music_discovery_dataset_filter_summary(datasets)
    device = initialize_discovery_runtime(seed=int(args.seed), gpu=str(args.gpu))

    train_loader = create_music_discovery_loader(datasets.train_dataset, batch_size=int(args.batch_size), shuffle=True)
    val_loader = create_music_discovery_loader(datasets.val_dataset, batch_size=int(args.batch_size), shuffle=False)
    eval_loaders = {
        "train": create_music_discovery_loader(datasets.train_dataset, batch_size=int(args.batch_size), shuffle=False),
        "val": create_music_discovery_loader(datasets.val_dataset, batch_size=int(args.batch_size), shuffle=False),
        "test": create_music_discovery_loader(datasets.test_dataset, batch_size=int(args.batch_size), shuffle=False),
        "all": create_music_discovery_loader(datasets.all_dataset, batch_size=int(args.batch_size), shuffle=False),
    }
    eval_datasets = {
        "train": datasets.train_dataset,
        "val": datasets.val_dataset,
        "test": datasets.test_dataset,
        "all": datasets.all_dataset,
    }
    metadata_policy_info = resolve_metadata_policy(
        str(args.metadata_policy),
        metadata_feature_names=datasets.metadata_feature_names,
        requested_metadata_cluster_weight=float(args.metadata_cluster_weight),
    )
    effective_metadata_cluster_weight = float(metadata_policy_info["effective_metadata_cluster_weight"])

    _use_amp = parse_bool_text(args.use_amp)
    train_epochs = int(args.pretrain_epochs) if args.pretrain_epochs is not None else int(args.epochs)

    model = MusicMetadataDiscoveryNet(
        audio_dim=int(datasets.train_dataset.audio.shape[1]),
        lyrics_dim=int(datasets.train_dataset.lyrics.shape[1]),
        metadata_dim=int(datasets.train_dataset.metadata.shape[1]),
        metadata_recon_dim=int(datasets.train_dataset.raw_metadata_recon_target.shape[1]),
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        metadata_hidden_dim=int(args.metadata_hidden_dim),
        gate_hidden_dim=int(args.gate_hidden_dim),
        metadata_aux_scale=float(args.metadata_aux_scale),
        dropout=float(args.dropout),
        metadata_logit_offset=float(args.metadata_logit_offset),
        cluster_head_k=int(args.cluster_head_k),
        cluster_temperature=float(args.cluster_temperature),
        diff_input_dim=int(args.diff_input_dim),
    ).to(device)

    best_state, history, best_metrics = train_music_discovery_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=train_epochs,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        metadata_recon_weight=float(args.metadata_recon_weight),
        fused_recon_weight=float(args.fused_recon_weight),
        align_weight=float(args.align_weight),
        metadata_align_weight=float(args.metadata_align_weight),
        gate_entropy_weight=float(args.gate_entropy_weight),
        cluster_loss_weight=float(args.cluster_loss_weight),
        cvcl_loss_weight=float(args.cvcl_loss_weight),
        assignment_balance_weight=float(args.assignment_balance_weight),
        consensus_va_weight=float(args.consensus_va_weight),
        diff_preserve_weight=float(args.diff_preserve_weight),
        metadata_recon_loss=str(args.metadata_recon_loss),
        grad_clip_norm=float(args.grad_clip_norm),
        use_amp=_use_amp,
        early_stopping_patience=int(args.early_stopping_patience),
        scheduler_T0=int(args.scheduler_T0),
        scheduler_Tmult=int(args.scheduler_Tmult),
        scheduler_eta_min=float(args.scheduler_eta_min),
    )
    model.load_state_dict(best_state, strict=True)

    checkpoint_path = os.path.join(models_dir, "music_discovery_model_best.pth")
    processed_meta_path = os.path.join(str(args.processed_dir), "meta.json")
    processed_schema_path = os.path.join(str(args.processed_dir), "schema.json")
    metadata_schema_path = os.path.join(str(args.processed_dir), "metadata_schema.json")
    processed_meta = {}
    processed_schema = {}
    metadata_schema = {}
    if os.path.exists(processed_meta_path):
        with open(processed_meta_path, "r", encoding="utf-8") as f:
            processed_meta = json.load(f)
    if os.path.exists(processed_schema_path):
        with open(processed_schema_path, "r", encoding="utf-8") as f:
            processed_schema = json.load(f)
    if os.path.exists(metadata_schema_path):
        with open(metadata_schema_path, "r", encoding="utf-8") as f:
            metadata_schema = json.load(f)
    save_discovery_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        scaler_state=datasets.scaler_state,
        config={
            "split_protocol": split_protocol,
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
            "metadata_hidden_dim": int(args.metadata_hidden_dim),
            "gate_hidden_dim": int(args.gate_hidden_dim),
            "gate_context_dim": int(model.gate_context_dim),
            "metadata_aux_scale": float(args.metadata_aux_scale),
            "metadata_recon_weight": float(args.metadata_recon_weight),
            "metadata_recon_loss": str(args.metadata_recon_loss),
            "metadata_recon_dim": int(model.metadata_recon_dim),
            "fused_recon_weight": float(args.fused_recon_weight),
            "align_weight": float(args.align_weight),
            "metadata_align_weight": float(args.metadata_align_weight),
            "batch_size": int(args.batch_size),
            "epochs": int(train_epochs),
            "run_stage": str(args.run_stage),
            "pretrain_epochs": int(args.pretrain_epochs) if args.pretrain_epochs is not None else None,
            "finetune_epochs": int(args.finetune_epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "metadata_logit_offset": float(args.metadata_logit_offset),
            "grad_clip_norm": float(args.grad_clip_norm),
            "use_amp": _use_amp,
            "early_stopping_patience": int(args.early_stopping_patience),
            "scheduler_T0": int(args.scheduler_T0),
            "scheduler_Tmult": int(args.scheduler_Tmult),
            "scheduler_eta_min": float(args.scheduler_eta_min),
            "gate_entropy_weight": float(args.gate_entropy_weight),
            "cluster_head_k": int(args.cluster_head_k),
            "cluster_temperature": float(args.cluster_temperature),
            "cluster_loss_weight": float(args.cluster_loss_weight),
            "cvcl_loss_weight": float(args.cvcl_loss_weight),
            "assignment_balance_weight": float(args.assignment_balance_weight),
            "consensus_va_weight": float(args.consensus_va_weight),
            "diff_preserve_weight": float(args.diff_preserve_weight),
            "diff_input_dim": int(args.diff_input_dim),
            "diff_encoder_trained": float(args.diff_preserve_weight) > 0.0,
            "min_token_freq": int(args.min_token_freq),
            "max_tokens_per_field": int(args.max_tokens_per_field),
            "mask_aware_gate": True,
            "dec_cvcl_head": int(args.cluster_head_k) > 0,
            "require_both_va": bool(require_both_va),
            "dataset_filter_summary": dataset_filter_summary,
        },
        best_metrics=best_metrics,
        dataset_version=processed_meta.get("dataset_version"),
        dataset_hash=processed_meta.get("dataset_hash"),
        schema_hash=processed_schema.get("schema_hash", processed_meta.get("schema_hash")),
        metadata_schema=metadata_schema,
    )

    history_frame = pd.DataFrame(history)
    history_path = os.path.join(out_dir, "training_history.csv")
    history_frame.to_csv(history_path, index=False, encoding="utf-8")
    _plot_training_curves(history_frame, os.path.join(out_dir, "training_curves.png"))

    eval_splits = _parse_eval_splits(str(args.eval_splits), search_split=str(args.search_split).strip().lower())
    embeddings_by_split = {
        split: extract_split_embeddings(model=model, loader=eval_loaders[split], device=device)
        for split in eval_splits
    }

    search_split = str(args.search_split).strip().lower()
    feature_strategy = str(args.cluster_feature_strategy).strip().lower()
    if feature_strategy == "masked_diffaware" and float(args.diff_preserve_weight) <= 0.0:
        raise ValueError(
            "cluster_feature_strategy='masked_diffaware' requires --diff_preserve_weight > 0 "
            "so DiffEncoder is trained before clustering."
        )

    # Determine complete_first fit mask early so PCA/scaler fit on complete-pair rows only
    assignment_mode = str(getattr(args, "cluster_assignment_mode", "joint")).strip().lower()
    search_view_mask = embeddings_by_split[search_split].get("view_mask")
    both_mask: Optional[np.ndarray] = None
    if assignment_mode == "complete_first":
        if search_view_mask is None:
            raise ValueError(
                "complete_first mode requires view_mask in embeddings for search split "
                f"'{search_split}' — got None."
            )
        both_mask = (search_view_mask[:, 0] > 0) & (search_view_mask[:, 1] > 0)
        if not both_mask.any():
            raise ValueError(
                "complete_first mode requires at least one track with both audio+lyrics "
                f"in search split '{search_split}', but none found."
            )
    search_block_mask = cluster_feature_block_mask(
        feature_strategy,
        search_view_mask,
        int(embeddings_by_split[search_split]["z_fused"].shape[0]) if "z_fused" in embeddings_by_split[search_split] else 0,
    )
    search_block_mask = apply_metadata_policy_to_block_mask(
        search_block_mask,
        metadata_cluster_weight=effective_metadata_cluster_weight,
        diff_cluster_weight=float(args.diff_cluster_weight),
    )

    search_features_raw, search_pca, search_imputation = build_cluster_features(
        embeddings=embeddings_by_split[search_split],
        metadata_cluster_weight=effective_metadata_cluster_weight,
        conflict_cluster_weight=float(args.conflict_cluster_weight),
        gate_cluster_weight=float(args.gate_cluster_weight),
        strategy=feature_strategy,
        pca_target_dim=int(args.pca_target_dim),
        fit_mask=both_mask,
        diff_cluster_weight=float(args.diff_cluster_weight),
    )
    raw_block_slices = cluster_feature_block_slices(feature_strategy, int(search_features_raw.shape[1]))
    cluster_scaler = fit_cluster_scaler(
        feature_strategy,
        search_features_raw,
        block_mask=search_block_mask,
        block_slices=raw_block_slices,
        fit_mask=both_mask,
        block_scaler=str(args.block_scaler),
    )
    feature_weights = cluster_feature_weights(
        feature_strategy,
        int(search_features_raw.shape[1]),
        conflict_cluster_weight=float(args.conflict_cluster_weight),
        gate_cluster_weight=float(args.gate_cluster_weight),
        metadata_cluster_weight=effective_metadata_cluster_weight,
        diff_cluster_weight=float(args.diff_cluster_weight),
    )
    search_features = apply_cluster_feature_weights(
        transform_cluster_features(cluster_scaler, search_features_raw, block_mask=search_block_mask),
        feature_weights,
    )
    k_strategy = str(args.k_strategy).strip().lower()
    block_slices = cluster_feature_block_slices(feature_strategy, int(search_features.shape[1]))
    selection_info_metadata_policy = dict(metadata_policy_info)
    k_result, search_metrics, selection_info = run_k_selection(
        features=search_features,
        k_strategy=k_strategy,
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        random_state=int(args.random_state),
        min_cluster_size_abs=int(args.min_cluster_size_abs),
        min_cluster_size_ratio=float(args.min_cluster_size_ratio),
        covariance_type=str(args.covariance_type),
        stability_runs=int(args.stability_runs),
        cluster_backend=str(args.cluster_backend),
        eval_backend=str(args.eval_backend),
        device=str(device),
        silhouette_mode=str(args.silhouette_mode),
        silhouette_sample_size=int(args.silhouette_sample_size),
        silhouette_chunk_size=int(args.silhouette_chunk_size),
        view_mask=search_view_mask,
        assignment_mode=assignment_mode,
        block_mask=search_block_mask,
        block_slices=block_slices,
        macro_k_min=int(args.macro_k_min),
        macro_k_max=int(args.macro_k_max),
        micro_k_min=int(args.micro_k_min),
        micro_k_max=int(args.micro_k_max),
        affect_labels=_va_quadrant_labels(
            _dataset_plot_va(eval_datasets[search_split], str(args.plot_va_source)),
            boundary_margin=float(args.affect_boundary_margin),
        ),
        affect_gate_enabled=parse_bool_text(args.affect_gate),
        min_affect_dominant_ratio=float(args.min_affect_dominant_ratio),
        max_affect_mixed_cluster_fraction=float(args.max_affect_mixed_cluster_fraction),
        min_affect_weighted_purity=float(args.min_affect_weighted_purity),
        min_affect_valid_fraction=float(args.min_affect_valid_fraction),
    )
    selection_info["metadata_policy"] = selection_info_metadata_policy

    # Resolve GMM model and selected_k depending on strategy
    is_hierarchical = isinstance(k_result, HierarchicalClusterResult)
    cluster_output_label_names = _cluster_label_names_for_outputs(k_result, selection_info)

    # Block hierarchical + complete_first combination
    if assignment_mode in {"complete_first", "partial_likelihood"} and is_hierarchical:
        raise ValueError(
            f"cluster_assignment_mode='{assignment_mode}' is incompatible with "
            "k_strategy='hierarchical'. Hierarchical clustering uses two-level "
            "macro/micro GMMs that cannot be refitted on complete-pair subset. "
            "Use k_strategy='composite' or 'bic_only' instead."
        )

    if is_hierarchical:
        gmm_model = k_result.macro_model
        selected_k = k_result.total_clusters
    else:
        gmm_model = k_result
        selected_k = int(gmm_model.n_components)

    # Complete-pair-first: re-fit GMM on both-pair samples only
    if assignment_mode == "complete_first":
        assert both_mask is not None
        if both_mask.sum() < selected_k:
            raise ValueError(
                f"complete_first mode: selected_k={selected_k} exceeds "
                f"complete-pair sample count ({both_mask.sum()}) in search split "
                f"'{search_split}'. Reduce k_max (currently {args.k_max}) or use "
                f"cluster_assignment_mode='joint'."
            )
        if both_mask.any() and not both_mask.all():
            refitted = fit_gaussian_mixture_robust(
                search_features[both_mask],
                n_components=selected_k,
                covariance_type=str(args.covariance_type),
                reg_covar=1e-5,
                n_init=10,
                max_iter=300,
                random_state=int(args.random_state),
                require_converged=True,
                context="complete_first refit GMM",
            )
            gmm_model = refitted
            selection_info["complete_first_refit"] = True
            selection_info["complete_pair_samples"] = int(both_mask.sum())
            selection_info["total_samples"] = int(search_features.shape[0])
    elif assignment_mode == "partial_likelihood":
        if str(args.covariance_type) != "diag":
            raise ValueError("cluster_assignment_mode='partial_likelihood' requires --covariance_type diag.")
        selection_info["partial_likelihood"] = True
        selection_info["feature_block_slices"] = block_slices

    gmm_bundle_path = os.path.join(out_dir, "discovery_gmm_bundle.pkl")
    with open(gmm_bundle_path, "wb") as f:
        pickle.dump(
            {
                "cluster_scaler": cluster_scaler,
                "gmm_model": gmm_model,
                "k_strategy": k_strategy,
                "hierarchical_result": k_result if is_hierarchical else None,
                "search_pca": search_pca,
                "search_imputation": search_imputation,
                "feature_weights": feature_weights,
                "config": {
                    "search_split": search_split,
                    "k_strategy": k_strategy,
                    "k_min": int(args.k_min),
                    "k_max": int(args.k_max),
                    "macro_k_min": int(args.macro_k_min),
                    "macro_k_max": int(args.macro_k_max),
                    "micro_k_min": int(args.micro_k_min),
                    "micro_k_max": int(args.micro_k_max),
                    "min_cluster_size_abs": int(args.min_cluster_size_abs),
                    "min_cluster_size_ratio": float(args.min_cluster_size_ratio),
                    "covariance_type": str(args.covariance_type),
                    "stability_runs": int(args.stability_runs),
                    "cluster_backend": str(args.cluster_backend),
                    "eval_backend": str(args.eval_backend),
                    "silhouette_mode": str(args.silhouette_mode),
                    "silhouette_sample_size": int(args.silhouette_sample_size),
                    "silhouette_chunk_size": int(args.silhouette_chunk_size),
                    "affect_gate": parse_bool_text(args.affect_gate),
                    "min_affect_dominant_ratio": float(args.min_affect_dominant_ratio),
                    "max_affect_mixed_cluster_fraction": float(args.max_affect_mixed_cluster_fraction),
                    "min_affect_weighted_purity": float(args.min_affect_weighted_purity),
                    "min_affect_valid_fraction": float(args.min_affect_valid_fraction),
                    "affect_boundary_margin": float(args.affect_boundary_margin),
                    "metadata_policy": metadata_policy_info,
                    "metadata_cluster_weight": effective_metadata_cluster_weight,
                    "requested_metadata_cluster_weight": float(args.metadata_cluster_weight),
                    "conflict_cluster_weight": float(args.conflict_cluster_weight),
                    "gate_cluster_weight": float(args.gate_cluster_weight),
                    "cluster_feature_strategy": feature_strategy,
                    "cluster_feature_weights": feature_weights.tolist(),
                    "block_scaler": str(args.block_scaler),
                    "cluster_assignment_mode": assignment_mode,
                    "feature_block_slices": cluster_feature_block_slices(feature_strategy, int(search_features.shape[1])),
                    "plot_va_source": str(args.plot_va_source),
                    "pca_target_dim": int(args.pca_target_dim),
                    "selection_info": selection_info,
                },
            },
            f,
        )

    split_outputs: Dict[str, Dict[str, Any]] = {}
    split_assignments: Dict[str, np.ndarray] = {}
    for split in eval_splits:
        features_raw, _, _ = build_cluster_features(
            embeddings=embeddings_by_split[split],
            metadata_cluster_weight=effective_metadata_cluster_weight,
            conflict_cluster_weight=float(args.conflict_cluster_weight),
            gate_cluster_weight=float(args.gate_cluster_weight),
            strategy=feature_strategy,
            pca_target_dim=int(args.pca_target_dim),
            fitted_pca=search_pca,
            fitted_imputation=search_imputation,
            diff_cluster_weight=float(args.diff_cluster_weight),
        )
        split_block_mask = cluster_feature_block_mask(
            feature_strategy,
            embeddings_by_split[split].get("view_mask"),
            int(features_raw.shape[0]),
        )
        split_block_mask = apply_metadata_policy_to_block_mask(
            split_block_mask,
            metadata_cluster_weight=effective_metadata_cluster_weight,
            diff_cluster_weight=float(args.diff_cluster_weight),
        )
        features = apply_cluster_feature_weights(
            transform_cluster_features(cluster_scaler, features_raw, block_mask=split_block_mask),
            feature_weights,
        )
        if is_hierarchical:
            # For hierarchical: predict macro, then apply micro sub-labels
            macro_labels = k_result.macro_model.predict(features).astype(np.int64)
            # Re-run micro models per macro cluster to get combined labels
            assignments = np.full(len(features), -1, dtype=np.int64)
            global_label = 0
            for macro_id in range(k_result.macro_k):
                mask = macro_labels == macro_id
                if not mask.any():
                    # This macro cluster has no samples in this split — still
                    # advance global_label to keep label numbering consistent.
                    if macro_id in k_result.micro_models:
                        global_label += k_result.micro_models[macro_id].n_components
                    else:
                        global_label += 1
                    continue
                if macro_id in k_result.micro_models:
                    micro_gmm = k_result.micro_models[macro_id]
                    micro_labels = micro_gmm.predict(features[mask])
                    for sub_id in range(micro_gmm.n_components):
                        sub_mask = micro_labels == sub_id
                        indices = np.where(mask)[0][sub_mask]
                        assignments[indices] = global_label
                        global_label += 1
                else:
                    assignments[mask] = global_label
                    global_label += 1
        else:
            if assignment_mode == "partial_likelihood":
                assignments = gmm_model.predict(features, block_mask=split_block_mask).astype(np.int64)
            else:
                assignments = gmm_model.predict(features).astype(np.int64)

        split_assignments[split] = assignments
        split_dir = os.path.join(out_dir, split)
        payload = _write_split_outputs(
            out_dir=split_dir,
            split=split,
            dataset=eval_datasets[split],
            embeddings=embeddings_by_split[split],
            assignments=assignments,
            metadata_feature_names=datasets.metadata_feature_names,
            selected_k=selected_k,
            feature_dim=int(features.shape[1]),
            cluster_features=features,
            search_metrics=search_metrics if split == search_split else None,
            plot_va_source=str(args.plot_va_source),
            cluster_label_names=cluster_output_label_names,
        )
        split_outputs[split] = payload

    if cluster_output_label_names:
        label_names_path = os.path.join(out_dir, "cluster_label_names.json")
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in cluster_output_label_names.items()}, f, ensure_ascii=False, indent=2)

    pipeline_summary = {
        "processed_dir": str(args.processed_dir),
        "aligned_root": str(args.aligned_root),
        "search_split": search_split,
        "eval_splits": eval_splits,
        "selected_k": selected_k,
        "k_strategy": k_strategy,
        "selection_mode": str(selection_info.get("selection_mode", k_strategy)),
        "epochs": int(train_epochs),
        "latent_dim": int(args.latent_dim),
        "cluster_head_k": int(args.cluster_head_k),
        "cluster_temperature": float(args.cluster_temperature),
        "cluster_loss_weight": float(args.cluster_loss_weight),
        "cvcl_loss_weight": float(args.cvcl_loss_weight),
        "assignment_balance_weight": float(args.assignment_balance_weight),
        "metadata_feature_dim": int(metadata_summary["feature_dim"]),
        "metadata_policy": metadata_policy_info,
        "metadata_cluster_weight": effective_metadata_cluster_weight,
        "requested_metadata_cluster_weight": float(args.metadata_cluster_weight),
        "checkpoint_path": checkpoint_path,
        "gmm_bundle_path": gmm_bundle_path,
        "history_path": history_path,
        "metadata_summary_path": metadata_summary_path,
        "require_both_va": bool(require_both_va),
        "dataset_filter_summary": dataset_filter_summary,
        "cluster_feature_strategy": feature_strategy,
        "cluster_feature_weights": feature_weights.tolist(),
        "block_scaler": str(args.block_scaler),
        "cluster_assignment_mode": assignment_mode,
        "plot_va_source": str(args.plot_va_source),
        "pca_target_dim": int(args.pca_target_dim),
        "cluster_backend": str(args.cluster_backend),
        "eval_backend": str(args.eval_backend),
        "actual_cluster_backend": str(selection_info.get("actual_cluster_backend", args.cluster_backend)),
        "actual_eval_backend": str(selection_info.get("actual_eval_backend", args.eval_backend)),
        "device": str(selection_info.get("device", device)),
        "silhouette_mode": str(args.silhouette_mode),
        "silhouette_sample_size": int(args.silhouette_sample_size),
        "silhouette_chunk_size": int(args.silhouette_chunk_size),
        "cluster_aware_finetune": bool(
            int(args.cluster_head_k) > 0
            and (
                float(args.cluster_loss_weight) > 0.0
                or float(args.cvcl_loss_weight) > 0.0
                or float(args.assignment_balance_weight) > 0.0
            )
        ),
        "selection_info": selection_info,
        "split_outputs": split_outputs,
    }
    if is_hierarchical:
        pipeline_summary["macro_k"] = k_result.macro_k
    elif "macro_k" in selection_info:
        pipeline_summary["macro_k"] = int(selection_info["macro_k"])
    if cluster_output_label_names:
        pipeline_summary["label_names"] = {str(k): v for k, v in cluster_output_label_names.items()}
    if "min_cluster_size_threshold" in selection_info:
        pipeline_summary["min_cluster_size_threshold"] = int(selection_info["min_cluster_size_threshold"])

    if search_split in split_assignments:
        search_dataset = eval_datasets[search_split]
        search_view_mask = getattr(search_dataset, "view_mask", np.ones((len(search_dataset), 3), dtype=np.float32))
        mask_purity_results = compute_mask_purity(split_assignments[search_split], search_view_mask)
        pipeline_summary["mask_purity_diagnostics"] = mask_purity_results

    pipeline_summary_path = os.path.join(out_dir, "pipeline_summary.json")
    with open(pipeline_summary_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, ensure_ascii=False, indent=2)
    _write_pipeline_report(os.path.join(out_dir, "pipeline_report.md"), pipeline_summary)
    audit_result = None
    if parse_bool_text(args.run_topconf_audit):
        from cluster.evaluation.topconf_audit import audit_run, write_audit_outputs

        audit_result = audit_run(out_dir)
        write_audit_outputs(audit_result, out_dir)

    print(f"[Pipeline] Wrote full discovery-training outputs to {out_dir}")
    print(f"  - checkpoint: {checkpoint_path}")
    print(f"  - k_strategy: {k_strategy}")
    print(f"  - selected_k: {selected_k}")
    if is_hierarchical:
        print(f"  - macro_k: {k_result.macro_k}")
    elif "macro_k" in selection_info:
        print(f"  - macro_k: {selection_info['macro_k']}")
    if cluster_output_label_names:
        print(f"  - label_names: {cluster_output_label_names}")
    print("  - training_history.csv")
    print("  - training_curves.png")
    print("  - pipeline_summary.json")
    print("  - pipeline_report.md")
    if audit_result is not None:
        print(f"  - topconf_audit_report.md: ready={bool(audit_result.get('overall_ready'))}")
        if not bool(audit_result.get("overall_ready")):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
