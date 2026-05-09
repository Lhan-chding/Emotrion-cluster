from cluster.pipeline.k_selection import (
    HierarchicalClusterResult,
    KSearchResult,
    KSelectionConfig,
    compute_stability_score,
    detect_bic_elbow,
    hierarchical_cluster,
    search_gmm_bic_only,
    search_gmm_composite,
    search_gmm_semantic_composite,
    search_masked_diag_gmm_composite,
)
from cluster.pipeline.train import (
    ClusterFeatureStrategy,
    _build_cluster_features,
    _ensure_dir,
    _parse_eval_splits,
    _search_gmm,
    _write_pipeline_report,
    _write_split_outputs,
    build_cluster_features,
    run_k_selection,
)

__all__ = [
    # K-selection
    "KSelectionConfig",
    "KSearchResult",
    "HierarchicalClusterResult",
    "search_gmm_composite",
    "search_gmm_semantic_composite",
    "search_masked_diag_gmm_composite",
    "search_gmm_bic_only",
    "hierarchical_cluster",
    "compute_stability_score",
    "detect_bic_elbow",
    "run_k_selection",
    # Feature strategies
    "ClusterFeatureStrategy",
    "build_cluster_features",
    # Train utilities (legacy)
    "_build_cluster_features",
    "_ensure_dir",
    "_parse_eval_splits",
    "_search_gmm",
    "_write_pipeline_report",
    "_write_split_outputs",
]
