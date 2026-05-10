from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd


PASS = "pass"
FAIL = "fail"
WARN = "warn"


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    for name in ("rerun_summary.json", "pipeline_summary.json"):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing rerun_summary.json or pipeline_summary.json under {run_dir}.")


def _load_search_metrics(run_dir: Path) -> pd.DataFrame:
    for path in (
        run_dir / "train" / "cluster_search_metrics.csv",
        run_dir / "all" / "cluster_search_metrics.csv",
    ):
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _gate(status: str, detail: str, value: Any = None) -> Dict[str, Any]:
    return {"status": status, "detail": detail, "value": value}


def _max_mask_enrichment(summary: Mapping[str, Any]) -> float:
    clusters = summary.get("mask_purity_diagnostics", {}).get("clusters", [])
    values = [float(item.get("enrichment_vs_baseline", 0.0)) for item in clusters]
    return max(values) if values else float("nan")


def _macro_micro_artifacts_present(run_dir: Path) -> bool:
    required = [
        run_dir / "all" / "macro_micro_summary.csv",
        run_dir / "all" / "macro_micro_metadata_enrichment.csv",
    ]
    macro_dir = run_dir / "all" / "macro_micro"
    return all(path.exists() for path in required) and macro_dir.exists() and any(macro_dir.glob("macro_*_diff_arrow.png"))


def _split_semantic_consistency(run_dir: Path) -> Dict[str, Any]:
    split_tables: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test", "all"):
        path = run_dir / split / "macro_micro_summary.csv"
        if path.exists():
            split_tables[split] = pd.read_csv(path)
    if len(split_tables) < 4:
        return {"status": FAIL, "detail": "Missing macro_micro_summary.csv for at least one split.", "value": 0.0}
    all_labels = set(split_tables["all"]["label_name"].astype(str).tolist())
    if not all_labels:
        return {"status": FAIL, "detail": "No labels in all/macro_micro_summary.csv.", "value": 0.0}
    consistent = 0
    total = 0
    all_quadrants = {
        str(row.label_name): str(row.dominant_quadrant)
        for row in split_tables["all"][["label_name", "dominant_quadrant"]].itertuples(index=False)
    }
    for split in ("train", "val", "test"):
        split_frame = split_tables[split]
        split_labels = set(split_frame["label_name"].astype(str).tolist())
        if split_labels != all_labels:
            return {
                "status": FAIL,
                "detail": f"{split} labels differ from all split.",
                "value": sorted(all_labels.symmetric_difference(split_labels)),
            }
        for row in split_frame[["label_name", "dominant_quadrant"]].itertuples(index=False):
            total += 1
            consistent += int(str(row.dominant_quadrant) == all_quadrants[str(row.label_name)])
    rate = consistent / max(total, 1)
    status = PASS if rate >= 0.75 else WARN
    return {"status": status, "detail": "Dominant quadrant consistency across train/val/test.", "value": round(rate, 4)}


def _required_ablations_gate(run_dir: Path) -> Dict[str, Any]:
    missing_files = [name for name in ("ablation_report.csv", "baseline_comparison.csv") if not (run_dir / name).exists()]
    if missing_files:
        return _gate(
            FAIL,
            "baseline_comparison.csv and ablation_report.csv are required for top-conference main-result claims.",
            {"missing_files": missing_files},
        )

    required_configs = {
        "mean_va",
        "audio_va",
        "lyrics_va",
        "mean_va_diff",
        "va_geometry",
        "metadata_only",
        "proposed_no_diff",
        "proposed_no_metadata",
        "proposed_full",
    }
    baseline = pd.read_csv(run_dir / "baseline_comparison.csv")
    ablation = pd.read_csv(run_dir / "ablation_report.csv")
    observed = set(baseline.get("config", pd.Series(dtype=str)).astype(str).tolist())
    observed |= set(ablation.get("config", pd.Series(dtype=str)).astype(str).tolist())
    missing_configs = sorted(required_configs - observed)
    if missing_configs:
        return _gate(
            FAIL,
            "Ablation reports exist but do not cover the required baseline matrix.",
            {"missing_configs": missing_configs},
        )
    if "status" in baseline.columns:
        status = baseline["status"].astype(str).str.lower()
        required_rows = baseline[baseline["config"].astype(str).isin(required_configs)]
        failed_required = required_rows[~status.loc[required_rows.index].isin({"ok", "pass", "success", "completed"})]
        if not failed_required.empty:
            value: Dict[str, Any] = {"failed_configs": failed_required["config"].astype(str).tolist()}
            if "error_message" in failed_required.columns:
                value["errors"] = {
                    str(row.config): str(row.error_message)
                    for row in failed_required[["config", "error_message"]].itertuples(index=False)
                }
            return _gate(
                FAIL,
                "Required ablation configs must complete successfully before main-result claims.",
                value,
            )
    score_column = "claim_score" if "claim_score" in baseline.columns else "score"
    if score_column in baseline.columns:
        proposed = baseline.loc[baseline["config"].astype(str) == "proposed_full", score_column]
        if proposed.empty or pd.isna(proposed.iloc[0]):
            return _gate(
                FAIL,
                f"proposed_full must have a valid {score_column} before it can be claimed as the main result.",
            )
        proposed_score = float(proposed.iloc[0])
        competitors = baseline[baseline["config"].astype(str) != "proposed_full"].copy()
        competitors = competitors[pd.to_numeric(competitors[score_column], errors="coerce").notna()]
        non_improved = competitors[pd.to_numeric(competitors[score_column], errors="coerce") >= proposed_score]
        if not non_improved.empty:
            return _gate(
                FAIL,
                "proposed_full must outperform every required baseline before it can be claimed as the main result.",
                {
                    "proposed_full_score": proposed_score,
                    "score_column": score_column,
                    "non_improved_configs": non_improved["config"].astype(str).tolist(),
                },
            )
    return _gate(PASS, "Required ablation/baseline matrix is present and proposed_full is above listed baselines.")


def audit_run(run_dir: str | Path) -> Dict[str, Any]:
    root = Path(run_dir)
    summary = _load_summary(root)
    selection = summary.get("selection_info", {})
    metrics = _load_search_metrics(root)
    gates: Dict[str, Dict[str, Any]] = {}

    if not metrics.empty and "total_k_ok" in metrics.columns:
        selected_k = int(summary.get("selected_k", selection.get("selected_k", -1)))
        selected_rows = metrics[metrics.get("total_clusters", pd.Series(dtype=int)) == selected_k]
        if selected_rows.empty:
            selected_rows = metrics
        ok = bool(selected_rows["total_k_ok"].astype(bool).any())
        gates["total_k_constraint_honored"] = _gate(PASS if ok else FAIL, "Selected macro/micro candidate must satisfy total K bounds.", selected_k)
    else:
        gates["total_k_constraint_honored"] = _gate(FAIL, "Missing total_k_ok column in cluster_search_metrics.csv.")

    stability_fields = ("seed_ari_mean", "cluster_jaccard_min", "bootstrap_valid_rate")
    stability_present = all(field in selection for field in stability_fields)
    if not stability_present and not metrics.empty:
        stability_present = all(field in metrics.columns for field in stability_fields)
    gates["bootstrap_stability_present"] = _gate(
        PASS if stability_present else FAIL,
        "macro_micro search must report bootstrap/seed stability metrics.",
    )

    mask_nmi = float(summary.get("mask_purity_diagnostics", {}).get("nmi", float("nan")))
    gates["mask_nmi_below_0_05"] = _gate(PASS if mask_nmi < 0.05 else FAIL, "Mask NMI must remain below 0.05.", mask_nmi)
    max_enrichment = _max_mask_enrichment(summary)
    gates["max_mask_enrichment_below_1_30"] = _gate(
        PASS if max_enrichment <= 1.30 else WARN,
        "Max mask-pattern enrichment should remain below 1.30.",
        None if pd.isna(max_enrichment) else round(max_enrichment, 4),
    )
    gates["macro_micro_artifacts_present"] = _gate(
        PASS if _macro_micro_artifacts_present(root) else FAIL,
        "Macro/micro summary, enrichment tables, and per-macro diff plots must exist.",
    )
    gates["split_semantic_consistency_quantified"] = _split_semantic_consistency(root)
    gates["required_ablations_present"] = _required_ablations_gate(root)
    gates["cluster_aware_finetune_enabled"] = _gate(
        PASS if bool(summary.get("cluster_aware_finetune", False) or selection.get("cluster_aware_finetune", False)) else FAIL,
        "Main result must include cluster-aware fine-tuning evidence.",
    )
    gates["metadata_policy_declared"] = _gate(
        PASS if bool(summary.get("metadata_policy") or selection.get("metadata_policy")) else FAIL,
        "Run must declare metadata policy: affective_va_only, non_affective_metadata, or all_metadata_upper_bound.",
    )
    filter_summary = summary.get("dataset_filter_summary", {}) or {}
    require_both_va = bool(summary.get("require_both_va", filter_summary.get("require_both_va", False)))
    gates["complete_audio_lyrics_va_subset"] = _gate(
        PASS if require_both_va else FAIL,
        "Main result must use the complete audio+lyrics VA subset unless explicitly running an incomplete-view ablation.",
        filter_summary.get("splits"),
    )

    hard_failures = {
        name: gate
        for name, gate in gates.items()
        if gate["status"] == FAIL
    }
    return {
        "run_dir": str(root),
        "overall_ready": not hard_failures,
        "num_failures": len(hard_failures),
        "gates": gates,
    }


def write_audit_outputs(result: Mapping[str, Any], out_dir: str | Path) -> Dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "topconf_audit_summary.json"
    report_path = root / "topconf_audit_report.md"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Top-Conference Readiness Audit", ""]
    lines.append(f"- Run dir: `{result['run_dir']}`")
    lines.append(f"- Overall ready: `{bool(result['overall_ready'])}`")
    lines.append(f"- Failure count: `{int(result['num_failures'])}`")
    lines.append("")
    lines.append("| Gate | Status | Value | Detail |")
    lines.append("|---|---|---:|---|")
    for name, gate in result["gates"].items():
        lines.append(f"| `{name}` | `{gate['status']}` | `{gate.get('value', '')}` | {gate['detail']} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": str(summary_path), "report": str(report_path)}
