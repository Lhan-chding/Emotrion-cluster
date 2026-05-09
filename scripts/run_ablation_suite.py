from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_CONFIGS: Tuple[str, ...] = (
    "mean_va",
    "audio_va",
    "lyrics_va",
    "mean_va_diff",
    "va_geometry",
    "metadata_only",
    "proposed_no_diff",
    "proposed_no_metadata",
    "proposed_full",
)

CONFIG_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mean_va": {"strategy": "mean_va", "k_strategy": "composite", "requires_run_dir": False},
    "audio_va": {"strategy": "audio_va", "k_strategy": "composite", "requires_run_dir": False},
    "lyrics_va": {"strategy": "lyrics_va", "k_strategy": "composite", "requires_run_dir": False},
    "mean_va_diff": {"strategy": "mean_va_diff", "k_strategy": "composite", "requires_run_dir": False},
    "va_geometry": {"strategy": "va_geometry", "k_strategy": "composite", "requires_run_dir": False},
    "metadata_only": {"strategy": "metadata_only", "k_strategy": "composite", "requires_run_dir": False},
    "proposed_no_diff": {
        "strategy": "macro_micro_diffaware",
        "k_strategy": "macro_micro",
        "requires_run_dir": True,
        "diff_cluster_weight": 0.0,
    },
    "proposed_no_metadata": {
        "strategy": "macro_micro_diffaware",
        "k_strategy": "macro_micro",
        "requires_run_dir": True,
        "metadata_policy": "report_only",
    },
    "proposed_full": {
        "strategy": "macro_micro_diffaware",
        "k_strategy": "macro_micro",
        "requires_run_dir": True,
        "metadata_policy": "report_only",
    },
}


class SuiteArgs(NamedTuple):
    processed_dir: str
    base_run_dir: Optional[str]
    out_dir: str
    gpu: str
    batch_size: int
    stability_runs: int
    k_min: int
    k_max: int
    macro_k_min: int
    macro_k_max: int
    micro_k_min: int
    micro_k_max: int
    min_cluster_size_abs: int
    metadata_policy: str


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
        "partial_likelihood" if str(config["k_strategy"]) == "macro_micro" else "joint",
        "--k_strategy",
        str(config["k_strategy"]),
        "--total_k_min",
        str(int(args.k_min)),
        "--total_k_max",
        str(int(args.k_max)),
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
        "--metadata_policy",
        metadata_policy,
        "--diff_cluster_weight",
        f"{diff_weight:g}",
        "--covariance_type",
        "diag",
    ]
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
    score_cols = [col for col in ("macro_micro_score", "composite_score", "semantic_composite_score", "silhouette") if col in metrics.columns]
    if score_cols:
        return metrics.sort_values(score_cols[0], ascending=False).iloc[0]
    return metrics.iloc[0]


def _score_from_row(row: pd.Series) -> float:
    for column in ("macro_micro_score", "composite_score", "semantic_composite_score", "silhouette"):
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return float("nan")


def summarize_run(run_dir: Path, config_name: str) -> Dict[str, Any]:
    summary_path = run_dir / "rerun_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing rerun summary for config '{config_name}': {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_k = int(summary.get("selected_k", -1))
    metric_path = _first_existing_metric_path(run_dir)
    metric_row = pd.Series(dtype=float)
    if metric_path is not None:
        metrics = pd.read_csv(metric_path)
        if not metrics.empty:
            metric_row = _selected_metric_row(metrics, selected_k)
    mask_diag = summary.get("mask_purity_diagnostics", {}) or {}
    clusters = mask_diag.get("clusters", []) or []
    return {
        "config": config_name,
        "run_dir": str(run_dir),
        "selected_k": selected_k,
        "k_strategy": summary.get("k_strategy"),
        "cluster_feature_strategy": summary.get("cluster_feature_strategy"),
        "score": _score_from_row(metric_row) if not metric_row.empty else float("nan"),
        "silhouette": float(metric_row.get("silhouette", float("nan"))) if not metric_row.empty else float("nan"),
        "seed_ari_mean": float((summary.get("selection_info", {}) or {}).get("seed_ari_mean", metric_row.get("seed_ari_mean", float("nan")))),
        "cluster_jaccard_min": float((summary.get("selection_info", {}) or {}).get("cluster_jaccard_min", metric_row.get("cluster_jaccard_min", float("nan")))),
        "mask_nmi": float(mask_diag.get("nmi", float("nan"))),
        "max_mask_enrichment": max([float(item.get("enrichment_vs_baseline", 0.0)) for item in clusters], default=float("nan")),
    }


def write_suite_reports(out_dir: Path | str, configs: Sequence[str]) -> Tuple[Path, Path]:
    root = Path(out_dir)
    rows = [summarize_run(root / config, config) for config in configs]
    baseline = pd.DataFrame(rows)
    baseline_path = root / "baseline_comparison.csv"
    baseline.to_csv(baseline_path, index=False)
    proposed = baseline.loc[baseline["config"] == "proposed_full", "score"]
    proposed_score = float(proposed.iloc[0]) if not proposed.empty else float("nan")
    ablation = baseline.copy()
    ablation["delta_score_vs_proposed_full"] = ablation["score"] - proposed_score
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
    parser.add_argument("--total_k_min", "--k_min", dest="k_min", type=int, default=8)
    parser.add_argument("--total_k_max", "--k_max", dest="k_max", type=int, default=16)
    parser.add_argument("--macro_k_min", type=int, default=3)
    parser.add_argument("--macro_k_max", type=int, default=6)
    parser.add_argument("--micro_k_min", type=int, default=1)
    parser.add_argument("--micro_k_max", type=int, default=5)
    parser.add_argument("--min_cluster_size_abs", type=int, default=40)
    parser.add_argument("--metadata_policy", default="report_only")
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
        k_min=int(ns.k_min),
        k_max=int(ns.k_max),
        macro_k_min=int(ns.macro_k_min),
        macro_k_max=int(ns.macro_k_max),
        micro_k_min=int(ns.micro_k_min),
        micro_k_max=int(ns.micro_k_max),
        min_cluster_size_abs=int(ns.min_cluster_size_abs),
        metadata_policy=str(ns.metadata_policy),
    )
    for config in configs:
        run_out_dir = out_dir / config
        run_out_dir.mkdir(parents=True, exist_ok=True)
        command = build_rerun_command(args, config, run_out_dir)
        print("[Ablation]", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    baseline_path, ablation_path = write_suite_reports(out_dir, configs)
    print(f"[Ablation] Wrote {baseline_path}")
    print(f"[Ablation] Wrote {ablation_path}")
    if args.base_run_dir:
        copied = copy_reports_to_base_run(out_dir, args.base_run_dir)
        print(f"[Ablation] Copied reports to base run: {', '.join(copied)}")


if __name__ == "__main__":
    main()
