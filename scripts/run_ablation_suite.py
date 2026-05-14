from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import pandas as pd
import numpy as np


DEFAULT_CONFIGS: Tuple[str, ...] = (
    "audio_va",
    "lyrics_va",
    "raw_mean_va",
    "calibrated_mean_alpha_0_5",
    "clusterability_alpha",
    "raw_mean_plus_signed_diff",
    "calibrated_va_tension_final_report_only",
    "latent_two_view_va_gmm",
    "metadata_only_report_diagnostic",
)

CONFIG_REGISTRY: Dict[str, Dict[str, Any]] = {
    "audio_va": {
        "strategy": "audio_va",
        "k_strategy": "balanced_va_regions",
        "requires_run_dir": False,
        "plot_va_source": "cluster_consensus",
    },
    "lyrics_va": {
        "strategy": "lyrics_va",
        "k_strategy": "balanced_va_regions",
        "requires_run_dir": False,
        "plot_va_source": "cluster_consensus",
    },
    "raw_mean_va": {
        "strategy": "mean_va",
        "k_strategy": "balanced_va_regions",
        "requires_run_dir": False,
        "plot_va_source": "cluster_consensus",
    },
    "calibrated_mean_alpha_0_5": {
        "strategy": "calibrated_va_tension",
        "k_strategy": "balanced_va_regions",
        "requires_run_dir": False,
        "plot_va_source": "cluster_consensus",
        "extra_args": ["--consensus_mode", "global_alpha", "--consensus_alpha", "0.5"],
    },
    "clusterability_alpha": {
        "strategy": "calibrated_va_tension",
        "k_strategy": "balanced_va_regions",
        "requires_run_dir": False,
        "plot_va_source": "cluster_consensus",
        "extra_args": ["--consensus_mode", "clusterability_alpha"],
    },
    "raw_mean_plus_signed_diff": {
        "strategy": "mean_va_diff",
        "k_strategy": "composite",
        "requires_run_dir": False,
        "plot_va_source": "mean",
        "diagnostic_allow_failed_gates": True,
    },
    "calibrated_va_tension_final_report_only": {
        "strategy": "calibrated_va_tension",
        "k_strategy": "balanced_va_regions",
        "requires_run_dir": False,
        "metadata_policy": "report_only",
        "plot_va_source": "cluster_consensus",
        "extra_args": [
            "--consensus_mode",
            "clusterability_alpha",
            "--calibration_mode",
            "global_median_shift",
            "--diff_residual_mode",
            "knn",
            "--diff_residual_neighbors",
            "101",
            "--tension_encoding",
            "residual_3d",
            "--run_tension_micro_probe",
            "true",
        ],
    },
    "latent_two_view_va_gmm": {
        "strategy": "latent_two_view_va",
        "k_strategy": "latent_va_gmm",
        "requires_run_dir": False,
        "metadata_policy": "report_only",
        "plot_va_source": "latent_consensus",
        "k_min": 4,
        "k_max": 8,
        "extra_args": [
            "--latent_learn_view_bias",
            "true",
            "--latent_share_view_noise",
            "false",
            "--latent_alpha_prior_strength",
            "0.2",
            "--latent_max_iter",
            "200",
        ],
        "diagnostic_allow_failed_gates": True,
    },
    "metadata_only_report_diagnostic": {
        "strategy": "metadata_only",
        "k_strategy": "composite",
        "requires_run_dir": False,
        "metadata_policy": "all_metadata_upper_bound",
        "require_both_va": False,
        "plot_va_source": "mean",
        "diagnostic_allow_failed_gates": True,
    },
}

MAIN_CONFIG = "calibrated_va_tension_final_report_only"


class SuiteArgs(NamedTuple):
    processed_dir: str
    base_run_dir: Optional[str]
    out_dir: str
    gpu: str
    batch_size: int
    stability_runs: int
    stability_sample_size: int
    k_min: int
    k_max: int
    macro_k_min: int
    macro_k_max: int
    micro_k_min: int
    micro_k_max: int
    min_cluster_size_abs: int
    metadata_policy: str
    require_both_va: bool


def _repo_script_path(script_name: str) -> str:
    return str(Path(__file__).resolve().parent / script_name)


def parse_configs(text: str) -> List[str]:
    configs = [item.strip() for item in str(text).split(",") if item.strip()]
    if not configs:
        raise ValueError("At least one ablation config is required.")
    unknown = [item for item in configs if item not in CONFIG_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown ablation config(s): {unknown}. Available: {sorted(CONFIG_REGISTRY)}")
    return configs


def build_rerun_command(args: SuiteArgs, config_name: str, run_out_dir: Path) -> List[str]:
    config = CONFIG_REGISTRY[config_name]
    if bool(config.get("requires_run_dir")) and not args.base_run_dir:
        raise ValueError(f"Config '{config_name}' requires --base_run_dir.")
    metadata_policy = str(config.get("metadata_policy", args.metadata_policy))
    diff_weight = float(config.get("diff_cluster_weight", 0.35))
    k_min = int(config.get("k_min", args.k_min))
    k_max = int(config.get("k_max", args.k_max))
    require_both_va = bool(config.get("require_both_va", args.require_both_va))
    assignment_mode = "partial_likelihood" if str(config["k_strategy"]) in {"macro_micro", "constrained_macro_micro"} else "joint"
    command = [
        sys.executable,
        _repo_script_path("rerun_search.py"),
        "--processed_dir",
        str(args.processed_dir),
        "--out_dir",
        str(run_out_dir),
        "--gpu",
        str(args.gpu),
        "--batch_size",
        str(int(args.batch_size)),
        "--cluster_feature_strategy",
        str(config["strategy"]),
        "--cluster_assignment_mode",
        assignment_mode,
        "--k_strategy",
        str(config["k_strategy"]),
        "--total_k_min",
        str(k_min),
        "--total_k_max",
        str(k_max),
        "--macro_k_min",
        str(int(args.macro_k_min)),
        "--macro_k_max",
        str(int(args.macro_k_max)),
        "--micro_k_min",
        str(int(args.micro_k_min)),
        "--micro_k_max",
        str(int(args.micro_k_max)),
        "--min_cluster_size_abs",
        str(int(args.min_cluster_size_abs)),
        "--stability_runs",
        str(int(args.stability_runs)),
        "--stability_sample_size",
        str(int(args.stability_sample_size)),
        "--metadata_policy",
        metadata_policy,
        "--require_both_va",
        "true" if require_both_va else "false",
        "--diff_cluster_weight",
        f"{diff_weight:g}",
        "--covariance_type",
        "diag",
        "--affect_gate",
        "true",
        "--plot_va_source",
        str(config.get("plot_va_source", "mean")),
    ]
    if bool(config.get("diagnostic_allow_failed_gates", False)):
        command.extend(["--diagnostic_allow_failed_gates", "true"])
    command.extend([str(item) for item in config.get("extra_args", [])])
    if bool(config.get("requires_run_dir")):
        command.extend(["--run_dir", str(args.base_run_dir)])
    return command


def _first_existing_metric_path(run_dir: Path) -> Optional[Path]:
    for split in ("all", "train", "val", "test"):
        path = run_dir / split / "cluster_search_metrics.csv"
        if path.exists():
            return path
    return None


def _selected_metric_row(metrics: pd.DataFrame, selected_k: int) -> pd.Series:
    for column in ("total_clusters", "k"):
        if column in metrics.columns:
            matches = metrics[metrics[column].astype(int) == int(selected_k)]
            if not matches.empty:
                return matches.iloc[0]
    score_cols = [
        col
        for col in ("balanced_region_score", "latent_va_score", "macro_micro_score", "composite_score", "semantic_composite_score", "silhouette")
        if col in metrics.columns
    ]
    if score_cols:
        return metrics.sort_values(score_cols[0], ascending=False).iloc[0]
    return metrics.iloc[0]


def _score_from_row(row: pd.Series) -> float:
    for column in ("balanced_region_score", "latent_va_score", "macro_micro_score", "composite_score", "semantic_composite_score", "silhouette"):
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return float("nan")


def _affect_penalty_from_row(row: pd.Series) -> float:
    if row.empty:
        return 0.0
    penalty = 0.0
    weighted = row.get("affect_weighted_dominant_ratio", float("nan"))
    mixed = row.get("affect_mixed_cluster_fraction", float("nan"))
    min_ratio = row.get("affect_min_dominant_ratio", float("nan"))
    if pd.notna(weighted):
        penalty += max(0.0, 0.80 - float(weighted))
    if pd.notna(mixed):
        penalty += max(0.0, float(mixed) - 0.15)
    if pd.notna(min_ratio):
        penalty += max(0.0, 0.70 - float(min_ratio))
    return float(penalty)


def _claim_score_from_row(
    row: pd.Series,
    mask_nmi: float,
    max_mask_enrichment: float,
    *,
    include_affect_penalty: bool = True,
) -> float:
    if row.empty:
        return float("nan")
    if "va_mean_silhouette" in row and pd.notna(row["va_mean_silhouette"]):
        separation = float(row["va_mean_silhouette"])
    elif "va_silhouette" in row and pd.notna(row["va_silhouette"]):
        separation = float(row["va_silhouette"])
    elif "latent_consensus_silhouette" in row and pd.notna(row["latent_consensus_silhouette"]):
        separation = float(row["latent_consensus_silhouette"])
    elif "final_silhouette" in row and pd.notna(row["final_silhouette"]):
        separation = float(row["final_silhouette"])
    elif "silhouette" in row and pd.notna(row["silhouette"]):
        separation = float(row["silhouette"])
    else:
        return float("nan")
    leakage_penalty = 0.0 if not np.isfinite(mask_nmi) else max(0.0, float(mask_nmi) - 0.05)
    enrichment_penalty = 0.0 if not np.isfinite(max_mask_enrichment) else max(0.0, float(max_mask_enrichment) - 1.30) * 0.05
    affect_penalty = _affect_penalty_from_row(row) if bool(include_affect_penalty) else 0.0
    return float(separation - leakage_penalty - enrichment_penalty - affect_penalty)


def _run_error_path(run_dir: Path) -> Path:
    return run_dir / "run_error.json"


def _write_run_error(run_dir: Path, config_name: str, command: Sequence[str], error: BaseException) -> Path:
    payload: Dict[str, Any] = {
        "config": config_name,
        "status": "failed",
        "command": list(command),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    returncode = getattr(error, "returncode", None)
    if returncode is not None:
        payload["returncode"] = int(returncode)
    path = _run_error_path(run_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _summarize_failed_run(run_dir: Path, config_name: str) -> Dict[str, Any]:
    error_path = _run_error_path(run_dir)
    error_payload: Dict[str, Any] = {}
    if error_path.exists():
        error_payload = json.loads(error_path.read_text(encoding="utf-8"))
    return {
        "config": config_name,
        "status": "failed",
        "run_dir": str(run_dir),
        "selected_k": float("nan"),
        "k_strategy": CONFIG_REGISTRY.get(config_name, {}).get("k_strategy"),
        "cluster_feature_strategy": CONFIG_REGISTRY.get(config_name, {}).get("strategy"),
        "score": float("nan"),
        "claim_score": float("nan"),
        "silhouette": float("nan"),
        "va_silhouette": float("nan"),
        "knn_purity_10": float("nan"),
        "knn_purity_20": float("nan"),
        "center_radius_sep": float("nan"),
        "negative_silhouette_fraction": float("nan"),
        "macro_silhouette": float("nan"),
        "final_silhouette": float("nan"),
        "affect_weighted_dominant_ratio": float("nan"),
        "affect_min_dominant_ratio": float("nan"),
        "affect_mixed_cluster_fraction": float("nan"),
        "affect_gate_ok": False,
        "seed_ari_mean": float("nan"),
        "seed_ari_std": float("nan"),
        "size_balance": float("nan"),
        "min_cluster_size": float("nan"),
        "cluster_jaccard_min": float("nan"),
        "mask_nmi": float("nan"),
        "metadata_policy": CONFIG_REGISTRY.get(config_name, {}).get("metadata_policy", "report_only"),
        "max_mask_enrichment": float("nan"),
        "diagnostic_failed_gate_override": False,
        "error_type": error_payload.get("error_type"),
        "error_message": error_payload.get("error_message", f"Missing rerun summary for config '{config_name}'."),
        "returncode": error_payload.get("returncode"),
    }


def _float_from_row(row: pd.Series, *columns: str) -> float:
    if row.empty:
        return float("nan")
    for column in columns:
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return float("nan")


def _metadata_policy_text(summary: Mapping[str, Any], config_name: str) -> str:
    policy = summary.get("metadata_policy")
    if isinstance(policy, Mapping):
        policy_name = str(policy.get("metadata_policy", ""))
        effective_weight = policy.get("effective_metadata_cluster_weight")
        if effective_weight is not None:
            return f"{policy_name}:weight={float(effective_weight):.4g}"
        return policy_name
    if policy:
        return str(policy)
    return str(CONFIG_REGISTRY.get(config_name, {}).get("metadata_policy", "report_only"))


def summarize_run(run_dir: Path, config_name: str) -> Dict[str, Any]:
    summary_path = run_dir / "rerun_summary.json"
    if not summary_path.exists():
        return _summarize_failed_run(run_dir, config_name)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_k = int(summary.get("selected_k", -1))
    selection = summary.get("selection_info", {}) or {}
    metric_path = _first_existing_metric_path(run_dir)
    metric_row = pd.Series(dtype=float)
    if metric_path is not None:
        metrics = pd.read_csv(metric_path)
        if not metrics.empty:
            metric_row = _selected_metric_row(metrics, selected_k)
    mask_diag = summary.get("mask_purity_diagnostics", {}) or {}
    clusters = mask_diag.get("clusters", []) or []
    mask_nmi = float(mask_diag.get("nmi", float("nan")))
    max_mask_enrichment = max([float(item.get("enrichment_vs_baseline", 0.0)) for item in clusters], default=float("nan"))
    selection_mode = str(summary.get("selection_mode", selection.get("selection_mode", summary.get("k_strategy", "")))).strip().lower()
    affect_is_diagnostic = selection_mode in {"balanced_va_regions", "latent_va_gmm"}
    return {
        "config": config_name,
        "status": "ok",
        "run_dir": str(run_dir),
        "selected_k": selected_k,
        "k_strategy": summary.get("k_strategy"),
        "cluster_feature_strategy": summary.get("cluster_feature_strategy"),
        "score": _score_from_row(metric_row) if not metric_row.empty else float("nan"),
        "claim_score": _claim_score_from_row(
            metric_row,
            mask_nmi,
            max_mask_enrichment,
            include_affect_penalty=not affect_is_diagnostic,
        ),
        "silhouette": float(metric_row.get("silhouette", float("nan"))) if not metric_row.empty else float("nan"),
        "va_silhouette": _float_from_row(metric_row, "va_mean_silhouette", "va_silhouette", "latent_consensus_silhouette"),
        "knn_purity_10": _float_from_row(metric_row, "va_knn_purity_10"),
        "knn_purity_20": _float_from_row(metric_row, "va_knn_purity_20"),
        "center_radius_sep": _float_from_row(metric_row, "va_center_radius_sep"),
        "negative_silhouette_fraction": _float_from_row(metric_row, "va_negative_silhouette_fraction"),
        "macro_silhouette": float(metric_row.get("macro_silhouette", float("nan"))) if not metric_row.empty else float("nan"),
        "final_silhouette": float(metric_row.get("final_silhouette", float("nan"))) if not metric_row.empty else float("nan"),
        "affect_weighted_dominant_ratio": float(metric_row.get("affect_weighted_dominant_ratio", float("nan"))) if not metric_row.empty else float("nan"),
        "affect_min_dominant_ratio": float(metric_row.get("affect_min_dominant_ratio", float("nan"))) if not metric_row.empty else float("nan"),
        "affect_mixed_cluster_fraction": float(metric_row.get("affect_mixed_cluster_fraction", float("nan"))) if not metric_row.empty else float("nan"),
        "affect_gate_ok": bool(metric_row.get("affect_gate_ok", False)) if not metric_row.empty and pd.notna(metric_row.get("affect_gate_ok", float("nan"))) else False,
        "seed_ari_mean": float(selection.get("seed_ari_mean", metric_row.get("seed_ari_mean", float("nan")))),
        "seed_ari_std": float(selection.get("seed_ari_std", metric_row.get("seed_ari_std", float("nan")))),
        "size_balance": _float_from_row(metric_row, "size_balance"),
        "min_cluster_size": _float_from_row(metric_row, "min_cluster_size"),
        "cluster_jaccard_min": float(selection.get("cluster_jaccard_min", metric_row.get("cluster_jaccard_min", float("nan")))),
        "mask_nmi": mask_nmi,
        "metadata_policy": _metadata_policy_text(summary, config_name),
        "max_mask_enrichment": max_mask_enrichment,
        "diagnostic_failed_gate_override": bool(selection.get("diagnostic_failed_gate_override", False)),
        "error_type": None,
        "error_message": None,
        "returncode": None,
    }


def write_suite_reports(out_dir: Path | str, configs: Sequence[str]) -> Tuple[Path, Path]:
    root = Path(out_dir)
    rows = [summarize_run(root / config, config) for config in configs]
    baseline = pd.DataFrame(rows)
    baseline_path = root / "baseline_comparison.csv"
    baseline.to_csv(baseline_path, index=False)
    score_column = "claim_score" if "claim_score" in baseline.columns else "score"
    proposed = baseline.loc[baseline["config"] == MAIN_CONFIG, score_column]
    proposed_score = float(proposed.iloc[0]) if not proposed.empty else float("nan")
    ablation = baseline.copy()
    ablation[f"delta_claim_score_vs_{MAIN_CONFIG}"] = pd.to_numeric(ablation[score_column], errors="coerce") - proposed_score
    if "score" in ablation.columns:
        proposed_internal = baseline.loc[baseline["config"] == MAIN_CONFIG, "score"]
        proposed_internal_score = float(proposed_internal.iloc[0]) if not proposed_internal.empty else float("nan")
        ablation[f"delta_score_vs_{MAIN_CONFIG}"] = pd.to_numeric(ablation["score"], errors="coerce") - proposed_internal_score
    ablation_path = root / "ablation_report.csv"
    ablation.to_csv(ablation_path, index=False)
    return baseline_path, ablation_path


def copy_reports_to_base_run(suite_dir: Path | str, base_run_dir: Path | str) -> List[str]:
    suite = Path(suite_dir)
    base = Path(base_run_dir)
    base.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in ("baseline_comparison.csv", "ablation_report.csv"):
        source = suite / name
        if not source.exists():
            raise FileNotFoundError(f"Cannot copy missing ablation report: {source}")
        shutil.copyfile(source, base / name)
        copied.append(name)
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run top-conference baseline and ablation reruns.")
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--base_run_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--stability_runs", type=int, default=80)
    parser.add_argument("--stability_sample_size", type=int, default=0)
    parser.add_argument("--total_k_min", "--k_min", dest="k_min", type=int, default=8)
    parser.add_argument("--total_k_max", "--k_max", dest="k_max", type=int, default=16)
    parser.add_argument("--macro_k_min", type=int, default=3)
    parser.add_argument("--macro_k_max", type=int, default=6)
    parser.add_argument("--micro_k_min", type=int, default=1)
    parser.add_argument("--micro_k_max", type=int, default=5)
    parser.add_argument("--min_cluster_size_abs", type=int, default=40)
    parser.add_argument("--metadata_policy", default="report_only")
    parser.add_argument(
        "--require_both_va",
        choices=("true", "false"),
        default="true",
        help="Run every ablation on the same complete audio+lyrics VA subset.",
    )
    parser.add_argument(
        "--skip_existing",
        choices=("true", "false"),
        default="true",
        help="Skip configs whose rerun_summary.json already exists. Failed or incomplete configs are retried.",
    )
    return parser


def main() -> None:
    ns = build_parser().parse_args()
    configs = parse_configs(ns.configs)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = SuiteArgs(
        processed_dir=str(ns.processed_dir),
        base_run_dir=str(ns.base_run_dir) if ns.base_run_dir else None,
        out_dir=str(ns.out_dir),
        gpu=str(ns.gpu),
        batch_size=int(ns.batch_size),
        stability_runs=int(ns.stability_runs),
        stability_sample_size=int(ns.stability_sample_size),
        k_min=int(ns.k_min),
        k_max=int(ns.k_max),
        macro_k_min=int(ns.macro_k_min),
        macro_k_max=int(ns.macro_k_max),
        micro_k_min=int(ns.micro_k_min),
        micro_k_max=int(ns.micro_k_max),
        min_cluster_size_abs=int(ns.min_cluster_size_abs),
        metadata_policy=str(ns.metadata_policy),
        require_both_va=str(ns.require_both_va).lower() == "true",
    )
    for config in configs:
        run_out_dir = out_dir / config
        run_out_dir.mkdir(parents=True, exist_ok=True)
        if str(ns.skip_existing).lower() == "true" and (run_out_dir / "rerun_summary.json").exists():
            print(f"[Ablation] Skip existing successful config: {config}", flush=True)
            continue
        command = build_rerun_command(args, config, run_out_dir)
        print("[Ablation]", " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            error_path = _write_run_error(run_out_dir, config, command, exc)
            print(f"[Ablation][FAILED] {config}: wrote {error_path}; continuing.", flush=True)
    baseline_path, ablation_path = write_suite_reports(out_dir, configs)
    print(f"[Ablation] Wrote {baseline_path}")
    print(f"[Ablation] Wrote {ablation_path}")
    if args.base_run_dir:
        copied = copy_reports_to_base_run(out_dir, args.base_run_dir)
        print(f"[Ablation] Copied reports to base run: {', '.join(copied)}")


if __name__ == "__main__":
    main()
