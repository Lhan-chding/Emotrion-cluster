from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np

from cluster.config import parse_split_protocol
from cluster.features.va_geometry import build_va_geometry_features
from cluster.models.discovery_net import (
    create_music_discovery_datasets,
    create_music_discovery_loader,
    extract_split_embeddings,
    initialize_discovery_runtime,
    load_music_discovery_checkpoint,
)
from cluster.pipeline.k_selection import HierarchicalClusterResult
from cluster.pipeline.train import (
    _build_cluster_features,
    _cluster_label_names_for_outputs,
    _dataset_mean_va,
    _ensure_dir,
    _parse_eval_splits,
    _write_pipeline_report,
    _write_split_outputs,
    apply_cluster_feature_weights,
    apply_metadata_policy_to_block_mask,
    build_cluster_features,
    cluster_feature_block_mask,
    cluster_feature_block_slices,
    cluster_feature_weights,
    compute_mask_purity,
    fit_cluster_scaler,
    parse_bool_text,
    resolve_metadata_policy,
    run_k_selection,
    transform_cluster_features,
)

RAW_ONLY_CLUSTER_FEATURE_STRATEGIES = frozenset(
    {"mean_va", "audio_va", "lyrics_va", "va_geometry", "mean_va_diff", "original_va", "metadata_only"}
)


def _base_feature_strategy(strategy: str) -> str:
    return str(strategy or "full").strip().lower().replace("pca_reduced_", "")


def _strategy_requires_checkpoint(strategy: str) -> bool:
    return _base_feature_strategy(strategy) not in RAW_ONLY_CLUSTER_FEATURE_STRATEGIES


def _availability_gate(view_mask: np.ndarray) -> np.ndarray:
    """Uniform gate weights for raw-feature mode (no learned model).

    Returns uniform 1/3 weights regardless of view availability to avoid
    leaking missingness patterns into cluster features.
    """
    n = view_mask.shape[0]
    n_views = view_mask.shape[1] if view_mask.ndim == 2 else 3
    return np.full((n, n_views), 1.0 / n_views, dtype=np.float32)


def _raw_feature_embeddings(dataset) -> Dict[str, Any]:
    view_mask = getattr(dataset, "view_mask", np.ones((len(dataset), 3), dtype=np.float32)).astype(np.float32)
    both_audio_lyrics = ((view_mask[:, 0] > 0.0) & (view_mask[:, 1] > 0.0)).reshape(-1, 1)
    signed_va_diff = (dataset.raw_audio.astype(np.float32) - dataset.raw_lyrics.astype(np.float32))
    signed_va_diff = np.where(both_audio_lyrics, signed_va_diff, 0.0).astype(np.float32)
    mean_va = _dataset_mean_va(dataset)
    return {
        "audio_va": dataset.raw_audio.astype(np.float32),
        "lyrics_va": dataset.raw_lyrics.astype(np.float32),
        "mean_va": mean_va,
        "va_geometry": build_va_geometry_features(dataset.raw_audio, dataset.raw_lyrics, view_mask),
        "original_va": dataset.original_va.astype(np.float32),
        "view_mask": view_mask,
        "consistency": dataset.consistency.astype(np.float32).reshape(-1, 1),
        "va_diff": dataset.va_diff.astype(np.float32),
        "signed_va_diff": signed_va_diff,
        "z_fused": mean_va.astype(np.float32),
        "z_metadata": dataset.metadata.astype(np.float32),
        "gate_weights": _availability_gate(view_mask),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse an existing music discovery checkpoint and rerun larger variable-K cluster search with richer visualizations."
    )
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Existing discovery run directory containing models/music_discovery_model_best.pth. Required for learned feature strategies, not for raw VA strategies.",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split_protocol", type=str, default="70_15_15")
    parser.add_argument("--search_split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--eval_splits", type=str, default="train,val,test,all")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k_min", "--total_k_min", dest="k_min", type=int, default=4)
    parser.add_argument("--k_max", "--total_k_max", dest="k_max", type=int, default=20)
    parser.add_argument("--min_cluster_size_abs", type=int, default=20)
    parser.add_argument("--min_cluster_size_ratio", type=float, default=0.01)
    parser.add_argument("--metadata_cluster_weight", type=float, default=0.75)
    parser.add_argument("--conflict_cluster_weight", type=float, default=0.40)
    parser.add_argument("--gate_cluster_weight", type=float, default=0.20)
    parser.add_argument("--diff_cluster_weight", type=float, default=0.35)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--k_strategy", type=str, default="composite",
                        choices=["composite", "semantic_composite", "macro_micro", "bic_only", "hierarchical"],
                        help="K-selection strategy")
    parser.add_argument("--covariance_type", type=str, default="diag",
                        choices=["full", "diag", "tied", "spherical"])
    parser.add_argument("--stability_runs", type=int, default=5)
    parser.add_argument("--cluster_backend", type=str, default="auto",
                        choices=["auto", "sklearn", "torch", "cuml"])
    parser.add_argument("--eval_backend", type=str, default="auto",
                        choices=["auto", "sklearn", "torch", "cuml"])
    parser.add_argument("--silhouette_mode", type=str, default="full",
                        choices=["full", "sampled", "torch_chunked", "masked_torch_chunked"])
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
                            "va_geometry",
                            "mean_va_diff",
                            "original_va",
                            "metadata_only",
                            "pca_reduced",
                        ],
                        help="Clustering feature strategy")
    parser.add_argument("--pca_target_dim", type=int, default=32,
                        help="Target dimensionality for PCA reduction")
    parser.add_argument("--plot_va_source", type=str, default="mean",
                        choices=["mean", "original"],
                        help="VA coordinates used in cluster scatter and summaries.")
    parser.add_argument(
        "--metadata_policy",
        type=str,
        default="all_metadata_upper_bound",
        choices=["all_metadata_upper_bound", "non_affective_metadata", "affective_va_only", "report_only"],
        help="Controls whether metadata embeddings may enter clustering.",
    )
    parser.add_argument("--block_scaler", type=str, default="auto", choices=["auto", "standard", "observed"])
    parser.add_argument("--run_topconf_audit", type=str, default="false")
    parser.add_argument("--allow_incompatible_checkpoint", type=str, default="false",
                        help="Allow checkpoint with mismatched metadata_dim (true/false)")
    parser.add_argument("--cluster_assignment_mode", type=str, default="joint",
                        choices=["joint", "complete_first", "partial_likelihood"],
                        help="joint: GMM fit on all samples; complete_first: fit only on both-pair samples; partial_likelihood: ignore unobserved feature blocks during prediction")
    parser.add_argument("--macro_k_min", type=int, default=4)
    parser.add_argument("--macro_k_max", type=int, default=8)
    parser.add_argument("--micro_k_min", type=int, default=1)
    parser.add_argument("--micro_k_max", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)

    feature_strategy = str(args.cluster_feature_strategy).strip().lower()
    requires_checkpoint = _strategy_requires_checkpoint(feature_strategy)
    device = initialize_discovery_runtime(seed=int(args.seed), gpu=str(args.gpu))
    checkpoint_path = None
    sidecar: Dict[str, Any] = {"config": {}}
    model = None
    if requires_checkpoint:
        if not args.run_dir:
            parser.error(
                f"--run_dir is required when --cluster_feature_strategy={feature_strategy!r} uses learned embeddings."
            )
        checkpoint_path = os.path.join(str(args.run_dir), "models", "music_discovery_model_best.pth")
        model, sidecar = load_music_discovery_checkpoint(checkpoint_path=checkpoint_path, device=device)
        if feature_strategy == "masked_diffaware":
            compatibility = sidecar.get("checkpoint_compatibility", {})
            initialized = compatibility.get("initialized_missing_modules", [])
            config = sidecar.get("config", {})
            if initialized or not bool(config.get("diff_encoder_trained", False)):
                raise ValueError(
                    "cluster_feature_strategy='masked_diffaware' requires a checkpoint "
                    "trained with DiffEncoder preservation loss. Retrain with "
                    "--diff_preserve_weight > 0 before using this strategy."
                )
        _allow_incompat = str(getattr(args, "allow_incompatible_checkpoint", "false")).strip().lower() in {"1", "true", "yes"}

        # Schema-level validation: compare feature names, not just dimension
        checkpoint_schema = sidecar.get("metadata_schema") or {}
        checkpoint_feature_names = checkpoint_schema.get("feature_names", [])
        current_names_path = os.path.join(str(args.processed_dir), "metadata_feature_names.json")
        if os.path.exists(current_names_path) and checkpoint_feature_names:
            with open(current_names_path, "r", encoding="utf-8") as f:
                current_feature_names = json.load(f)
            if current_feature_names != checkpoint_feature_names and not _allow_incompat:
                n_show = min(5, max(len(current_feature_names), len(checkpoint_feature_names)))
                raise ValueError(
                    f"Checkpoint metadata feature names differ from current processed dataset.\n"
                    f"  Checkpoint (first {n_show}): {checkpoint_feature_names[:n_show]}\n"
                    f"  Current   (first {n_show}): {current_feature_names[:n_show]}\n"
                    f"Same dimension but different semantics will produce wrong cluster assignments. "
                    f"Retrain or pass --allow_incompatible_checkpoint true."
                )
        else:
            # Fallback: dimension-only check when schema is unavailable
            checkpoint_metadata_dim = len(sidecar.get("scaler_state", {}).get("metadata", {}).get("mean", []))
            current_metadata_path = os.path.join(str(args.processed_dir), "metadata.npy")
            if os.path.exists(current_metadata_path) and checkpoint_metadata_dim > 0:
                current_metadata_dim = int(np.load(current_metadata_path).shape[1])
                if current_metadata_dim != checkpoint_metadata_dim and not _allow_incompat:
                    raise ValueError(
                        f"Checkpoint metadata_dim={checkpoint_metadata_dim} != processed metadata_dim={current_metadata_dim}. "
                        f"The checkpoint was trained with different metadata. Either retrain or pass "
                        f"--allow_incompatible_checkpoint true to proceed anyway."
                    )

    split_protocol = parse_split_protocol(
        str(sidecar.get("config", {}).get("split_protocol", str(args.split_protocol)))
    )
    datasets = create_music_discovery_datasets(
        data_dir=str(args.processed_dir),
        split_protocol=split_protocol,
        scaler_state=sidecar.get("scaler_state"),
    )
    eval_splits = _parse_eval_splits(str(args.eval_splits), search_split=str(args.search_split).strip().lower())

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

    if requires_checkpoint:
        embeddings_by_split = {
            split: extract_split_embeddings(model=model, loader=eval_loaders[split], device=device)
            for split in eval_splits
        }
    else:
        embeddings_by_split = {
            split: _raw_feature_embeddings(eval_datasets[split])
            for split in eval_splits
        }

    search_split = str(args.search_split).strip().lower()
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
    search_block_mask = cluster_feature_block_mask(feature_strategy, search_view_mask, int(search_features_raw.shape[0]))
    search_block_mask = apply_metadata_policy_to_block_mask(
        search_block_mask,
        metadata_cluster_weight=effective_metadata_cluster_weight,
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
    )
    selection_info["metadata_policy"] = dict(metadata_policy_info)

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
            from sklearn.mixture import GaussianMixture
            refitted = GaussianMixture(
                n_components=selected_k,
                covariance_type=str(args.covariance_type),
                reg_covar=1e-5,
                n_init=10,
                random_state=int(args.random_state),
            )
            refitted.fit(search_features[both_mask])
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
                    "checkpoint_path": checkpoint_path,
                    "selection_info": selection_info,
                },
            },
            f,
        )

    split_outputs: Dict[str, Dict[str, Any]] = {}
    search_assignments: Optional[np.ndarray] = None
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
            macro_labels = k_result.macro_model.predict(features).astype(np.int64)
            assignments = np.full(len(features), -1, dtype=np.int64)
            global_label = 0
            for macro_id in range(k_result.macro_k):
                mask = macro_labels == macro_id
                if not mask.any():
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

        payload = _write_split_outputs(
            out_dir=os.path.join(out_dir, split),
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
        if split == search_split:
            search_assignments = assignments

    if cluster_output_label_names:
        label_names_path = os.path.join(out_dir, "cluster_label_names.json")
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in cluster_output_label_names.items()}, f, ensure_ascii=False, indent=2)

    summary = {
        "processed_dir": str(args.processed_dir),
        "source_run_dir": str(args.run_dir) if args.run_dir else None,
        "search_split": search_split,
        "eval_splits": eval_splits,
        "selected_k": selected_k,
        "k_strategy": k_strategy,
        "selection_mode": str(selection_info.get("selection_mode", k_strategy)),
        "cluster_backend": str(args.cluster_backend),
        "eval_backend": str(args.eval_backend),
        "actual_cluster_backend": str(selection_info.get("actual_cluster_backend", args.cluster_backend)),
        "actual_eval_backend": str(selection_info.get("actual_eval_backend", args.eval_backend)),
        "device": str(selection_info.get("device", device)),
        "silhouette_mode": str(args.silhouette_mode),
        "silhouette_sample_size": int(args.silhouette_sample_size),
        "silhouette_chunk_size": int(args.silhouette_chunk_size),
        "selection_info": selection_info,
        "cluster_feature_strategy": feature_strategy,
        "cluster_feature_weights": feature_weights.tolist(),
        "block_scaler": str(args.block_scaler),
        "metadata_policy": metadata_policy_info,
        "metadata_cluster_weight": effective_metadata_cluster_weight,
        "requested_metadata_cluster_weight": float(args.metadata_cluster_weight),
        "cluster_assignment_mode": assignment_mode,
        "plot_va_source": str(args.plot_va_source),
        "checkpoint_path": checkpoint_path,
        "gmm_bundle_path": gmm_bundle_path,
        "split_outputs": split_outputs,
    }
    if is_hierarchical:
        summary["macro_k"] = k_result.macro_k
    elif "macro_k" in selection_info:
        summary["macro_k"] = int(selection_info["macro_k"])
    if cluster_output_label_names:
        summary["label_names"] = {str(k): v for k, v in cluster_output_label_names.items()}
    if "min_cluster_size_threshold" in selection_info:
        summary["min_cluster_size_threshold"] = int(selection_info["min_cluster_size_threshold"])

    mask_purity_diagnostics = None
    if search_assignments is not None:
        search_dataset = eval_datasets[search_split]
        search_view_mask = getattr(search_dataset, "view_mask", np.ones((len(search_dataset), 3), dtype=np.float32))
        mask_purity_diagnostics = compute_mask_purity(search_assignments, search_view_mask)
        summary["mask_purity_diagnostics"] = mask_purity_diagnostics

    summary_path = os.path.join(out_dir, "rerun_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    report_data = {
        "processed_dir": str(args.processed_dir),
        "search_split": search_split,
        "eval_splits": eval_splits,
        "selected_k": selected_k,
        "k_strategy": k_strategy,
        "selection_mode": str(selection_info.get("selection_mode", k_strategy)),
        "cluster_backend": str(args.cluster_backend),
        "eval_backend": str(args.eval_backend),
        "actual_cluster_backend": str(selection_info.get("actual_cluster_backend", args.cluster_backend)),
        "actual_eval_backend": str(selection_info.get("actual_eval_backend", args.eval_backend)),
        "epochs": sidecar.get("config", {}).get("epochs", "raw_feature_only" if not requires_checkpoint else "reused"),
        "latent_dim": sidecar.get("config", {}).get("latent_dim", "raw_feature_only" if not requires_checkpoint else "reused"),
        "metadata_feature_dim": int(datasets.train_dataset.metadata.shape[1]),
        "metadata_policy": metadata_policy_info,
        "checkpoint_path": checkpoint_path,
        "gmm_bundle_path": gmm_bundle_path,
        "history_path": os.path.join(str(args.run_dir), "training_history.csv") if args.run_dir else None,
        "split_outputs": split_outputs,
    }
    if mask_purity_diagnostics is not None:
        report_data["mask_purity_diagnostics"] = mask_purity_diagnostics
    if "min_cluster_size_threshold" in selection_info:
        report_data["min_cluster_size_threshold"] = int(selection_info["min_cluster_size_threshold"])
    _write_pipeline_report(os.path.join(out_dir, "rerun_report.md"), report_data)
    audit_result = None
    if parse_bool_text(args.run_topconf_audit):
        from cluster.evaluation.topconf_audit import audit_run, write_audit_outputs

        audit_result = audit_run(out_dir)
        write_audit_outputs(audit_result, out_dir)

    print(f"[Rerun] Wrote expanded K-search outputs to {out_dir}")
    print(f"  - k_strategy: {k_strategy}")
    print(f"  - selected_k: {selected_k}")
    if checkpoint_path is None:
        print("  - checkpoint reused: none (raw feature strategy)")
    else:
        print(f"  - checkpoint reused: {checkpoint_path}")
    print("  - rerun_summary.json")
    print("  - rerun_report.md")
    if audit_result is not None:
        print(f"  - topconf_audit_report.md: ready={bool(audit_result.get('overall_ready'))}")
        if not bool(audit_result.get("overall_ready")):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
