from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from cluster.config import parse_split_protocol
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
    _ensure_dir,
    _parse_eval_splits,
    _write_pipeline_report,
    _write_split_outputs,
    build_cluster_features,
    run_k_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse an existing music discovery checkpoint and rerun larger variable-K cluster search with richer visualizations."
    )
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True, help="Existing discovery run directory containing models/music_discovery_model_best.pth")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split_protocol", type=str, default="70_15_15")
    parser.add_argument("--search_split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--eval_splits", type=str, default="train,val,test,all")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k_min", type=int, default=4)
    parser.add_argument("--k_max", type=int, default=20)
    parser.add_argument("--min_cluster_size_abs", type=int, default=20)
    parser.add_argument("--min_cluster_size_ratio", type=float, default=0.01)
    parser.add_argument("--metadata_cluster_weight", type=float, default=0.75)
    parser.add_argument("--conflict_cluster_weight", type=float, default=0.40)
    parser.add_argument("--gate_cluster_weight", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--k_strategy", type=str, default="composite",
                        choices=["composite", "bic_only", "hierarchical"],
                        help="K-selection strategy")
    parser.add_argument("--covariance_type", type=str, default="full",
                        choices=["full", "diag", "tied", "spherical"])
    parser.add_argument("--stability_runs", type=int, default=5)
    parser.add_argument("--cluster_backend", type=str, default="auto",
                        choices=["auto", "sklearn", "torch", "cuml"])
    parser.add_argument("--eval_backend", type=str, default="auto",
                        choices=["auto", "sklearn", "torch", "cuml"])
    parser.add_argument("--silhouette_mode", type=str, default="full",
                        choices=["full", "sampled", "torch_chunked"])
    parser.add_argument("--silhouette_sample_size", type=int, default=0)
    parser.add_argument("--silhouette_chunk_size", type=int, default=4096)
    parser.add_argument("--cluster_feature_strategy", type=str, default="full",
                        choices=["full", "fused_residual", "fused_only", "pca_reduced"],
                        help="Clustering feature strategy")
    parser.add_argument("--pca_target_dim", type=int, default=32,
                        help="Target dimensionality for PCA reduction")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)

    checkpoint_path = os.path.join(str(args.run_dir), "models", "music_discovery_model_best.pth")
    device = initialize_discovery_runtime(seed=int(args.seed), gpu=str(args.gpu))
    model, sidecar = load_music_discovery_checkpoint(checkpoint_path=checkpoint_path, device=device)

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

    embeddings_by_split = {
        split: extract_split_embeddings(model=model, loader=eval_loaders[split], device=device)
        for split in eval_splits
    }

    search_split = str(args.search_split).strip().lower()
    feature_strategy = str(args.cluster_feature_strategy).strip().lower()
    search_features_raw, search_pca = build_cluster_features(
        embeddings=embeddings_by_split[search_split],
        metadata_cluster_weight=float(args.metadata_cluster_weight),
        conflict_cluster_weight=float(args.conflict_cluster_weight),
        gate_cluster_weight=float(args.gate_cluster_weight),
        strategy=feature_strategy,
        pca_target_dim=int(args.pca_target_dim),
    )
    cluster_scaler = StandardScaler().fit(search_features_raw)
    search_features = cluster_scaler.transform(search_features_raw).astype(np.float32)
    k_strategy = str(args.k_strategy).strip().lower()
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
    )

    is_hierarchical = isinstance(k_result, HierarchicalClusterResult)
    if is_hierarchical:
        gmm_model = k_result.macro_model
        selected_k = k_result.total_clusters
    else:
        gmm_model = k_result
        selected_k = int(gmm_model.n_components)

    gmm_bundle_path = os.path.join(out_dir, "discovery_gmm_bundle.pkl")
    with open(gmm_bundle_path, "wb") as f:
        pickle.dump(
            {
                "cluster_scaler": cluster_scaler,
                "gmm_model": gmm_model,
                "k_strategy": k_strategy,
                "hierarchical_result": k_result if is_hierarchical else None,
                "search_pca": search_pca,
                "config": {
                    "search_split": search_split,
                    "k_strategy": k_strategy,
                    "k_min": int(args.k_min),
                    "k_max": int(args.k_max),
                    "min_cluster_size_abs": int(args.min_cluster_size_abs),
                    "min_cluster_size_ratio": float(args.min_cluster_size_ratio),
                    "covariance_type": str(args.covariance_type),
                    "stability_runs": int(args.stability_runs),
                    "cluster_backend": str(args.cluster_backend),
                    "eval_backend": str(args.eval_backend),
                    "silhouette_mode": str(args.silhouette_mode),
                    "silhouette_sample_size": int(args.silhouette_sample_size),
                    "silhouette_chunk_size": int(args.silhouette_chunk_size),
                    "metadata_cluster_weight": float(args.metadata_cluster_weight),
                    "conflict_cluster_weight": float(args.conflict_cluster_weight),
                    "gate_cluster_weight": float(args.gate_cluster_weight),
                    "cluster_feature_strategy": feature_strategy,
                    "pca_target_dim": int(args.pca_target_dim),
                    "checkpoint_path": checkpoint_path,
                    "selection_info": selection_info,
                },
            },
            f,
        )

    split_outputs: Dict[str, Dict[str, Any]] = {}
    for split in eval_splits:
        features_raw, _ = build_cluster_features(
            embeddings=embeddings_by_split[split],
            metadata_cluster_weight=float(args.metadata_cluster_weight),
            conflict_cluster_weight=float(args.conflict_cluster_weight),
            gate_cluster_weight=float(args.gate_cluster_weight),
            strategy=feature_strategy,
            pca_target_dim=int(args.pca_target_dim),
            fitted_pca=search_pca,
        )
        features = cluster_scaler.transform(features_raw).astype(np.float32)

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
            search_metrics=search_metrics if split == search_split else None,
        )
        split_outputs[split] = payload

    if is_hierarchical:
        label_names_path = os.path.join(out_dir, "hierarchical_label_names.json")
        with open(label_names_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in k_result.label_names.items()}, f, ensure_ascii=False, indent=2)

    summary = {
        "processed_dir": str(args.processed_dir),
        "source_run_dir": str(args.run_dir),
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
        "checkpoint_path": checkpoint_path,
        "gmm_bundle_path": gmm_bundle_path,
        "split_outputs": split_outputs,
    }
    if is_hierarchical:
        summary["macro_k"] = k_result.macro_k
        summary["label_names"] = {str(k): v for k, v in k_result.label_names.items()}
    if "min_cluster_size_threshold" in selection_info:
        summary["min_cluster_size_threshold"] = int(selection_info["min_cluster_size_threshold"])
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
        "epochs": sidecar.get("config", {}).get("epochs", "reused"),
        "latent_dim": sidecar.get("config", {}).get("latent_dim", "reused"),
        "metadata_feature_dim": len(datasets.metadata_feature_names),
        "checkpoint_path": checkpoint_path,
        "gmm_bundle_path": gmm_bundle_path,
        "history_path": os.path.join(str(args.run_dir), "training_history.csv"),
        "split_outputs": split_outputs,
    }
    if "min_cluster_size_threshold" in selection_info:
        report_data["min_cluster_size_threshold"] = int(selection_info["min_cluster_size_threshold"])
    _write_pipeline_report(os.path.join(out_dir, "rerun_report.md"), report_data)

    print(f"[Rerun] Wrote expanded K-search outputs to {out_dir}")
    print(f"  - k_strategy: {k_strategy}")
    print(f"  - selected_k: {selected_k}")
    print(f"  - checkpoint reused: {checkpoint_path}")
    print("  - rerun_summary.json")
    print("  - rerun_report.md")


if __name__ == "__main__":
    main()
