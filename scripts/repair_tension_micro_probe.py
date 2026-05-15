"""Regenerate report-only residual tension artifacts for an existing run."""

from pathlib import Path
import argparse
import json
import pickle
import sys
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.models.discovery_net import create_music_discovery_datasets, initialize_discovery_runtime
from cluster.pipeline.rerun import _raw_feature_embeddings
from cluster.pipeline.train import (
    _write_tension_micro_probe_artifacts,
    _write_tension_substructure_artifacts,
    build_cluster_features,
    parse_bool_text,
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bundle_config(run_dir: Path) -> Dict[str, Any]:
    bundle_path = run_dir / "discovery_gmm_bundle.pkl"
    if not bundle_path.exists():
        return {}
    with bundle_path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        return dict(payload["config"])
    return {}


def _config_value(*configs: Mapping[str, Any], key: str, default: Any) -> Any:
    for config in configs:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _tension_config(summary: Mapping[str, Any], bundle_config: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    raw = _config_value(summary, bundle_config, key="tension_micro_probe", default={})
    config = dict(raw) if isinstance(raw, dict) else {}
    config.update(
        {
            "enabled": parse_bool_text(str(config.get("enabled", "true"))),
            "source": str(args.tension_micro_source or config.get("source", "residualized")),
            "k_max": int(args.tension_micro_k_max if args.tension_micro_k_max is not None else config.get("k_max", 3)),
            "min_cluster_size": int(
                args.tension_micro_min_cluster_size_abs
                if args.tension_micro_min_cluster_size_abs is not None
                else config.get("min_cluster_size", 30)
            ),
            "min_silhouette": float(
                args.tension_micro_min_silhouette
                if args.tension_micro_min_silhouette is not None
                else config.get("min_silhouette", 0.10)
            ),
            "min_effect": float(
                args.tension_micro_min_effect
                if args.tension_micro_min_effect is not None
                else config.get("min_effect", 0.25)
            ),
            "stability_runs": int(
                args.tension_micro_stability_runs
                if args.tension_micro_stability_runs is not None
                else config.get("stability_runs", 5)
            ),
            "random_state": int(_config_value(summary, bundle_config, key="random_state", default=42)),
        }
    )
    return config


def _assignments_for_split(run_dir: Path, split: str, expected_n: int) -> np.ndarray:
    path = run_dir / split / "cluster_assignments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster assignments: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "cluster_id" not in frame.columns:
        raise ValueError(f"{path} must contain a cluster_id column.")
    labels = frame["cluster_id"].to_numpy(dtype=np.int64)
    if labels.shape[0] != int(expected_n):
        raise ValueError(f"{path} has {labels.shape[0]} rows but split dataset has {expected_n}.")
    return labels


def _merge_output_files(
    payload: Mapping[str, Any],
    tension_micro_outputs: Mapping[str, Any],
    tension_substructure_outputs: Mapping[str, Any],
) -> Dict[str, Any]:
    output_files = dict(payload.get("output_files", {})) if isinstance(payload.get("output_files"), dict) else {}
    micro_files = tension_micro_outputs.get("output_files", {}) if isinstance(tension_micro_outputs, Mapping) else {}
    if isinstance(micro_files, Mapping):
        output_files.update(dict(micro_files))
    if isinstance(tension_substructure_outputs, Mapping):
        output_files.update(dict(tension_substructure_outputs))
    return output_files


def _update_repaired_summary_jsons(
    run_dir: Path,
    split: str,
    *,
    tension_micro_outputs: Mapping[str, Any],
    tension_substructure_outputs: Mapping[str, Any],
) -> Dict[str, bool]:
    split_dir = Path(run_dir) / str(split)
    fresh_probe = tension_micro_outputs.get("tension_micro_probe") if isinstance(tension_micro_outputs, Mapping) else None
    updated = {
        "split_cluster_summary": False,
        "rerun_summary": False,
        "pipeline_summary": False,
    }

    split_summary_path = split_dir / "cluster_summary.json"
    split_payload: Dict[str, Any] = {}
    if split_summary_path.exists():
        split_payload = _load_json(split_summary_path)
        split_payload["tension_micro_probe"] = fresh_probe
        split_payload["output_files"] = _merge_output_files(
            split_payload,
            tension_micro_outputs,
            tension_substructure_outputs,
        )
        split_summary_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        updated["split_cluster_summary"] = True

    for summary_name in ("rerun_summary.json", "pipeline_summary.json"):
        summary_path = Path(run_dir) / summary_name
        if not summary_path.exists():
            continue
        root_payload = _load_json(summary_path)
        split_outputs = root_payload.get("split_outputs")
        if not isinstance(split_outputs, dict):
            continue
        current_split_payload = split_outputs.get(str(split), {})
        if not isinstance(current_split_payload, dict):
            current_split_payload = {}
        replacement = dict(current_split_payload)
        replacement["tension_micro_probe"] = fresh_probe
        replacement["output_files"] = _merge_output_files(
            replacement,
            tension_micro_outputs,
            tension_substructure_outputs,
        )
        if split_payload:
            for key in ("split", "selected_k", "feature_dim", "num_samples", "plot_va_source", "balance_alpha"):
                if key in split_payload:
                    replacement[key] = split_payload[key]
        split_outputs[str(split)] = replacement
        root_payload["split_outputs"] = split_outputs
        summary_path.write_text(json.dumps(root_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        updated["rerun_summary" if summary_name == "rerun_summary.json" else "pipeline_summary"] = True
    return updated


def _build_features_for_split(
    embeddings: Dict[str, Any],
    *,
    fitted_state: Any,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray, Any]:
    features, _pca, state = build_cluster_features(
        embeddings=embeddings,
        metadata_cluster_weight=0.0,
        conflict_cluster_weight=float(_config_value(config, key="conflict_cluster_weight", default=0.40)),
        gate_cluster_weight=float(_config_value(config, key="gate_cluster_weight", default=0.20)),
        strategy=str(_config_value(config, key="cluster_feature_strategy", default="calibrated_va_tension")),
        pca_target_dim=int(_config_value(config, key="pca_target_dim", default=32)),
        fitted_imputation=fitted_state,
        diff_cluster_weight=float(_config_value(config, key="diff_cluster_weight", default=0.0)),
        consensus_mode=str(_config_value(config, key="consensus_mode", default="clusterability_alpha")),
        consensus_alpha=float(_config_value(config, key="consensus_alpha", default=0.5)),
        alpha_search_min=float(_config_value(config, key="alpha_search_min", default=0.20)),
        alpha_search_max=float(_config_value(config, key="alpha_search_max", default=0.90)),
        alpha_search_step=float(_config_value(config, key="alpha_search_step", default=0.05)),
        alpha_search_k_min=int(_config_value(config, key="alpha_search_k_min", default=4)),
        alpha_search_k_max=int(_config_value(config, key="alpha_search_k_max", default=8)),
        calibration_mode=str(_config_value(config, key="calibration_mode", default="global_median_shift")),
        diff_residual_mode=str(_config_value(config, key="diff_residual_mode", default="knn")),
        diff_residual_neighbors=int(_config_value(config, key="diff_residual_neighbors", default=101)),
        compute_device=str(device),
        compute_chunk_size=int(args.silhouette_chunk_size),
        compute_sample_size=int(args.silhouette_sample_size),
        tension_encoding=str(_config_value(config, key="tension_encoding", default="residual_3d")),
    )
    return features, state


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate residual tension micro-probe artifacts in-place.")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--split_protocol", default=None)
    parser.add_argument("--search_split", default=None, choices=["train", "val", "test", "all"])
    parser.add_argument("--eval_splits", default=None)
    parser.add_argument("--require_both_va", default=None)
    parser.add_argument("--silhouette_sample_size", type=int, default=50000)
    parser.add_argument("--silhouette_chunk_size", type=int, default=16384)
    parser.add_argument("--eval_backend", default=None)
    parser.add_argument("--tension_micro_source", default="residualized", choices=["residualized", "raw_delta"])
    parser.add_argument("--tension_micro_k_max", type=int, default=None)
    parser.add_argument("--tension_micro_min_cluster_size_abs", type=int, default=None)
    parser.add_argument("--tension_micro_min_silhouette", type=float, default=None)
    parser.add_argument("--tension_micro_min_effect", type=float, default=None)
    parser.add_argument("--tension_micro_stability_runs", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    summary = _load_json(run_dir / "rerun_summary.json")
    bundle_config = _load_bundle_config(run_dir)
    merged_config = {**bundle_config, **summary}
    split_protocol = str(args.split_protocol or _config_value(summary, bundle_config, key="split_protocol", default="70_15_15"))
    search_split = str(args.search_split or _config_value(summary, bundle_config, key="search_split", default="train"))
    raw_eval_splits = args.eval_splits or _config_value(summary, bundle_config, key="eval_splits", default="train,val,test,all")
    if isinstance(raw_eval_splits, (list, tuple)):
        eval_splits = [str(item).strip() for item in raw_eval_splits if str(item).strip()]
    else:
        eval_splits = [item.strip() for item in str(raw_eval_splits).split(",") if item.strip()]
    require_both = (
        parse_bool_text(args.require_both_va)
        if args.require_both_va is not None
        else bool(_config_value(summary, bundle_config, key="require_both_va", default=True))
    )
    eval_backend = str(args.eval_backend or _config_value(summary, bundle_config, key="eval_backend", default="torch"))

    device = initialize_discovery_runtime(seed=int(_config_value(summary, bundle_config, key="seed", default=42)), gpu=str(args.gpu))
    datasets = create_music_discovery_datasets(
        str(Path(args.processed_dir).expanduser()),
        split_protocol,
        require_both_va=bool(require_both),
    )
    dataset_by_split = {
        "train": datasets.train_dataset,
        "val": datasets.val_dataset,
        "test": datasets.test_dataset,
        "all": datasets.all_dataset,
    }
    embeddings_by_split = {split: _raw_feature_embeddings(dataset_by_split[split]) for split in set(eval_splits + [search_split])}
    _search_features, fitted_state = _build_features_for_split(
        embeddings_by_split[search_split],
        fitted_state=None,
        config=merged_config,
        args=args,
        device=str(device),
    )
    tension_config = _tension_config(summary, bundle_config, args)
    search_feature_state = dict(fitted_state) if isinstance(fitted_state, dict) else {}
    search_feature_state["tension_micro_probe_config"] = tension_config

    for split in eval_splits:
        dataset = dataset_by_split[split]
        features_raw, split_state = _build_features_for_split(
            embeddings_by_split[split],
            fitted_state=fitted_state,
            config=merged_config,
            args=args,
            device=str(device),
        )
        feature_state = dict(split_state) if isinstance(split_state, dict) else dict(search_feature_state)
        feature_state["tension_micro_probe_config"] = tension_config
        assignments = _assignments_for_split(run_dir, split, len(dataset))
        split_dir = run_dir / split
        outputs = _write_tension_micro_probe_artifacts(
            out_dir=str(split_dir),
            dataset=dataset,
            assignments=assignments,
            feature_state=feature_state,
            cluster_features=None,
            tension_features=features_raw,
            eval_backend=eval_backend,
            device=str(device),
            silhouette_sample_size=int(args.silhouette_sample_size),
            silhouette_chunk_size=int(args.silhouette_chunk_size),
        )
        subtype_outputs = _write_tension_substructure_artifacts(
            out_dir=str(split_dir),
            dataset=dataset,
            metadata_feature_names=datasets.metadata_feature_names,
            tension_micro_outputs=outputs,
        )
        updated_jsons = _update_repaired_summary_jsons(
            run_dir,
            split,
            tension_micro_outputs=outputs,
            tension_substructure_outputs=subtype_outputs,
        )
        assignment_path = split_dir / "tension_micro_probe" / "tension_micro_assignments.csv"
        frame = pd.read_csv(assignment_path, low_memory=False)
        probe_frame = pd.read_csv(split_dir / "tension_micro_probe" / "tension_micro_probe.csv", low_memory=False)
        print(
            f"[repair] {split}: rows={len(frame)} source={frame['tension_micro_source'].iloc[0]} "
            f"norm_mean={frame['tension_norm'].mean():.4f} norm_std={frame['tension_norm'].std():.4f} "
            f"micro_splits={int((probe_frame['selected_micro_k'] > 1).sum())} "
            f"subtype_report={bool(subtype_outputs)} "
            f"summary_jsons={','.join(name for name, ok in updated_jsons.items() if ok) or 'none'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
