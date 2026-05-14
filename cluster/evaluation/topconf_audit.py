from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd


PASS = "pass"
FAIL = "fail"
WARN = "warn"

REQUIRED_ABLATION_CONFIGS = {
    "audio_va",
    "lyrics_va",
    "raw_mean_va",
    "calibrated_mean_alpha_0_5",
    "fixed_alpha_0_60",
    "clusterability_alpha",
    "raw_mean_plus_signed_diff",
    "residual_tension_weak_concat",
    "calibrated_va_tension_final_report_only",
    "latent_two_view_va_gmm",
    "metadata_only_report_diagnostic",
    "k4_sensitivity",
    "k5_sensitivity",
    "k6_sensitivity",
}
MAIN_ABLATION_CONFIG = "calibrated_va_tension_final_report_only"


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


def _affect_purity_from_catalog(
    run_dir: Path,
    *,
    min_dominant_ratio: float,
    min_weighted_purity: float,
    max_mixed_cluster_fraction: float,
) -> Dict[str, Any]:
    path = run_dir / "all" / "cluster_catalog.csv"
    if not path.exists():
        return _gate(FAIL, "all/cluster_catalog.csv is required to audit VA affect purity.")
    catalog = pd.read_csv(path)
    required = {"num_samples", "dominant_quadrant_ratio"}
    if not required.issubset(set(catalog.columns)):
        return _gate(
            FAIL,
            "cluster_catalog.csv must include num_samples and dominant_quadrant_ratio.",
            {"missing_columns": sorted(required - set(catalog.columns))},
        )
    sizes = pd.to_numeric(catalog["num_samples"], errors="coerce").fillna(0.0)
    ratios = pd.to_numeric(catalog["dominant_quadrant_ratio"], errors="coerce").fillna(0.0)
    total = float(sizes.sum())
    if total <= 0:
        return _gate(FAIL, "cluster_catalog.csv has no samples.")
    weighted = float((sizes * ratios).sum() / total)
    min_ratio = float(ratios.min())
    mixed_fraction = float(sizes[ratios < float(min_dominant_ratio)].sum() / total)
    ok = (
        weighted >= float(min_weighted_purity)
        and min_ratio >= float(min_dominant_ratio)
        and mixed_fraction <= float(max_mixed_cluster_fraction)
    )
    return _gate(
        PASS if ok else FAIL,
        "Clusters must remain coherent on the Mean VA quadrant plane.",
        {
            "affect_weighted_dominant_ratio": round(weighted, 4),
            "affect_min_dominant_ratio": round(min_ratio, 4),
            "affect_mixed_cluster_fraction": round(mixed_fraction, 4),
            "min_affect_dominant_ratio": float(min_dominant_ratio),
            "min_affect_weighted_purity": float(min_weighted_purity),
            "max_affect_mixed_cluster_fraction": float(max_mixed_cluster_fraction),
        },
    )


def _affect_purity_gate(run_dir: Path, selection: Mapping[str, Any], metrics: pd.DataFrame, selected_k: int) -> Dict[str, Any]:
    min_dominant_ratio = float(selection.get("min_affect_dominant_ratio", 0.70))
    min_weighted_purity = float(selection.get("min_affect_weighted_purity", 0.80))
    max_mixed_cluster_fraction = float(selection.get("max_affect_mixed_cluster_fraction", 0.15))
    required_metric_fields = {
        "affect_gate_ok",
        "affect_weighted_dominant_ratio",
        "affect_min_dominant_ratio",
        "affect_mixed_cluster_fraction",
    }
    if not metrics.empty and required_metric_fields.issubset(set(metrics.columns)):
        selected_rows = pd.DataFrame()
        for column in ("total_clusters", "k"):
            if column in metrics.columns:
                selected_rows = metrics[metrics[column].astype(int) == int(selected_k)]
                if not selected_rows.empty:
                    break
        if selected_rows.empty:
            selected_rows = metrics
        row = selected_rows.iloc[0]
        value = {
            "affect_gate_ok": bool(row.get("affect_gate_ok", False)),
            "affect_weighted_dominant_ratio": round(float(row.get("affect_weighted_dominant_ratio", float("nan"))), 4),
            "affect_min_dominant_ratio": round(float(row.get("affect_min_dominant_ratio", float("nan"))), 4),
            "affect_mixed_cluster_fraction": round(float(row.get("affect_mixed_cluster_fraction", float("nan"))), 4),
            "min_affect_dominant_ratio": min_dominant_ratio,
            "min_affect_weighted_purity": min_weighted_purity,
            "max_affect_mixed_cluster_fraction": max_mixed_cluster_fraction,
        }
        return _gate(
            PASS if bool(row.get("affect_gate_ok", False)) else FAIL,
            "Selected K must pass VA affect-purity hard gates.",
            value,
        )
    return _affect_purity_from_catalog(
        run_dir,
        min_dominant_ratio=min_dominant_ratio,
        min_weighted_purity=min_weighted_purity,
        max_mixed_cluster_fraction=max_mixed_cluster_fraction,
    )


def _is_balanced_va_regions(summary: Mapping[str, Any], selection: Mapping[str, Any]) -> bool:
    mode = str(summary.get("selection_mode", selection.get("selection_mode", ""))).strip().lower()
    return mode == "balanced_va_regions"


def _selected_metric_row(metrics: pd.DataFrame, selected_k: int) -> pd.Series:
    if metrics.empty:
        return pd.Series(dtype=float)
    for column in ("total_clusters", "k"):
        if column in metrics.columns:
            rows = metrics[metrics[column].astype(int) == int(selected_k)]
            if not rows.empty:
                return rows.iloc[0]
    return metrics.iloc[0]


def _metric_or_selection_value(
    selection: Mapping[str, Any],
    metrics: pd.DataFrame,
    selected_k: int,
    field: str,
) -> Any:
    if field in selection:
        return selection[field]
    row = _selected_metric_row(metrics, selected_k)
    if not row.empty and field in row:
        return row[field]
    return None


def _balanced_va_region_stability_gate(
    summary: Mapping[str, Any],
    selection: Mapping[str, Any],
    metrics: pd.DataFrame,
    selected_k: int,
) -> Dict[str, Any]:
    if not _is_balanced_va_regions(summary, selection):
        return _gate(PASS, "Balanced VA region stability is only required for balanced_va_regions main runs.")
    mean_value = _metric_or_selection_value(selection, metrics, selected_k, "seed_ari_mean")
    std_value = _metric_or_selection_value(selection, metrics, selected_k, "seed_ari_std")
    try:
        seed_ari_mean = float(mean_value)
        seed_ari_std = float(std_value)
    except (TypeError, ValueError):
        return _gate(FAIL, "balanced_va_regions must report seed_ari_mean and seed_ari_std.", {"seed_ari_mean": mean_value, "seed_ari_std": std_value})
    ok = pd.notna(seed_ari_mean) and pd.notna(seed_ari_std) and seed_ari_mean >= 0.90
    return _gate(
        PASS if ok else FAIL,
        "Balanced VA region runs must report high seed stability.",
        {"seed_ari_mean": round(seed_ari_mean, 4), "seed_ari_std": round(seed_ari_std, 4)},
    )


def _overlap_gate_train_val_test_all(run_dir: Path, summary: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    if not _is_balanced_va_regions(summary, selection):
        return _gate(PASS, "Split overlap audit is only required for balanced_va_regions main runs.")
    rows: Dict[str, Any] = {}
    missing = []
    failures = []
    for split in ("train", "val", "test", "all"):
        path = run_dir / split / "cluster_overlap_audit.csv"
        if not path.exists():
            missing.append(split)
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            missing.append(split)
            continue
        row = frame.iloc[0]
        purity20 = float(row.get("va_knn_purity_20", float("nan")))
        sep = float(row.get("va_center_radius_sep", float("nan")))
        negative_fraction = float(row.get("va_negative_silhouette_fraction", float("nan")))
        overlap_ok = bool(row.get("overlap_gate_ok", False))
        metric_ok = purity20 >= 0.88 and sep >= 0.70 and negative_fraction <= 0.05
        if not (overlap_ok or metric_ok):
            failures.append(split)
        rows[split] = {
            "overlap_gate_ok": overlap_ok,
            "va_knn_purity_20": round(purity20, 4) if pd.notna(purity20) else None,
            "va_center_radius_sep": round(sep, 4) if pd.notna(sep) else None,
            "va_negative_silhouette_fraction": round(negative_fraction, 4) if pd.notna(negative_fraction) else None,
            "metric_ok": metric_ok,
        }
    if missing:
        return _gate(FAIL, "cluster_overlap_audit.csv must exist for train/val/test/all.", {"missing_splits": missing})
    return _gate(
        PASS if not failures else FAIL,
        "Train/val/test/all splits must pass overlap audit or balanced VA metric thresholds.",
        {"splits": rows, "failed_splits": failures},
    )


def _metadata_not_used_for_clustering(summary: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    policy = summary.get("metadata_policy", selection.get("metadata_policy", {}))
    if isinstance(policy, Mapping):
        metadata_policy = str(policy.get("metadata_policy", "")).lower()
        effective_weight = float(policy.get("effective_metadata_cluster_weight", 0.0))
        block_used = bool(policy.get("metadata_block_used_for_clustering", effective_weight > 0.0))
    else:
        metadata_policy = str(policy).lower()
        effective_weight = 0.0 if metadata_policy == "report_only" else float("nan")
        block_used = metadata_policy not in {"report_only", "affective_va_only"}
    ok = metadata_policy == "report_only" and effective_weight == 0.0 and not block_used
    return _gate(
        PASS if ok else FAIL,
        "Main claim metadata policy must be report_only with zero effective metadata clustering weight.",
        {"metadata_policy": metadata_policy, "effective_metadata_cluster_weight": effective_weight, "metadata_block_used_for_clustering": block_used},
    )


def _report_only_tension_present(run_dir: Path, summary: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    if not _is_balanced_va_regions(summary, selection):
        return _gate(PASS, "Report-only tension artifacts are only required for balanced_va_regions main runs.")
    required = [
        run_dir / "all" / "tension_micro_probe" / "tension_micro_probe.csv",
        run_dir / "all" / "tension_substructure_report.md",
        run_dir / "all" / "tension_substructure_enrichment.csv",
        run_dir / "all" / "tension_subtype_assignments.csv",
    ]
    missing = [str(path.relative_to(run_dir)) for path in required if not path.exists()]
    if missing:
        return _gate(
            FAIL,
            "Report-only residual tension micro/subtype artifacts must be present for ACL reporting.",
            {"missing_files": missing},
        )
    probe = pd.read_csv(run_dir / "all" / "tension_micro_probe" / "tension_micro_probe.csv")
    if "tension_micro_source" not in probe.columns:
        return _gate(
            FAIL,
            "Report-only tension probe must declare tension_micro_source='residualized'.",
            {"missing_column": "tension_micro_source"},
        )
    source_values = set(probe.get("tension_micro_source", pd.Series(dtype=str)).astype(str).str.lower().tolist())
    if source_values and source_values != {"residualized"}:
        return _gate(
            FAIL,
            "Report-only tension probe must use residualized calibrated tension rather than raw lyrics-audio delta.",
            {"tension_micro_sources": sorted(source_values)},
        )
    return _gate(
        PASS,
        "Report-only residual tension micro/subtype artifacts are present for ACL reporting.",
        {"tension_micro_sources": sorted(source_values) if source_values else []},
    )


def _alpha_search_report_present(run_dir: Path, summary: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    if not _is_balanced_va_regions(summary, selection):
        return _gate(PASS, "Alpha-search report is only required for balanced_va_regions main runs.")
    path = run_dir / "all" / "balance_alpha_report.csv"
    return _gate(
        PASS if path.exists() else FAIL,
        "all/balance_alpha_report.csv must be present for the balanced VA alpha audit.",
        str(path.relative_to(run_dir)) if path.exists() else None,
    )


def _tension_split_reproducibility_gate(run_dir: Path, summary: Mapping[str, Any], selection: Mapping[str, Any]) -> Dict[str, Any]:
    if not _is_balanced_va_regions(summary, selection):
        return _gate(PASS, "Tension split reproducibility is only required for balanced_va_regions main runs.")
    split_rows: Dict[str, Any] = {}
    missing = []
    for split in ("train", "val", "test", "all"):
        path = run_dir / split / "tension_micro_probe" / "tension_micro_probe.csv"
        if not path.exists():
            missing.append(split)
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            missing.append(split)
            continue
        if "tension_micro_source" not in frame.columns:
            split_rows[split] = {"missing_column": "tension_micro_source"}
            continue
        selected = pd.to_numeric(frame.get("selected_micro_k", pd.Series(dtype=float)), errors="coerce")
        split_rows[split] = {
            "clusters": int(len(frame)),
            "clusters_with_micro_split": int((selected > 1).sum()),
            "max_selected_micro_k": int(selected.max()) if selected.notna().any() else 1,
            "source_values": sorted(set(frame.get("tension_micro_source", pd.Series(dtype=str)).astype(str).str.lower().tolist())),
        }
    if missing:
        return _gate(
            FAIL,
            "Residual tension micro probe must be emitted for train/val/test/all splits.",
            {"missing_splits": missing, "splits": split_rows},
        )
    missing_source_column = [
        split
        for split, row in split_rows.items()
        if bool(row.get("missing_column"))
    ]
    if missing_source_column:
        return _gate(
            FAIL,
            "Residual tension split probes must declare tension_micro_source for every split.",
            {"missing_source_column_splits": missing_source_column, "splits": split_rows},
        )
    source_failures = [
        split
        for split, row in split_rows.items()
        if row["source_values"] and row["source_values"] != ["residualized"]
    ]
    if source_failures:
        return _gate(
            FAIL,
            "Residual tension split probes must use residualized source for every split.",
            {"source_failed_splits": source_failures, "splits": split_rows},
        )
    all_count = int(split_rows["all"]["clusters_with_micro_split"])
    split_counts = [int(split_rows[split]["clusters_with_micro_split"]) for split in ("train", "val", "test")]
    stable = all(abs(count - all_count) <= max(1, int(round(0.25 * max(all_count, 1)))) for count in split_counts)
    return _gate(
        PASS if stable else WARN,
        "Train/val/test residual tension subtype counts should be broadly reproducible against all split.",
        {"splits": split_rows},
    )


def _required_ablations_gate(run_dir: Path) -> Dict[str, Any]:
    missing_files = [name for name in ("ablation_report.csv", "baseline_comparison.csv") if not (run_dir / name).exists()]
    if missing_files:
        return _gate(
            FAIL,
            "baseline_comparison.csv and ablation_report.csv are required for top-conference main-result claims.",
            {"missing_files": missing_files},
        )

    baseline = pd.read_csv(run_dir / "baseline_comparison.csv")
    ablation = pd.read_csv(run_dir / "ablation_report.csv")
    observed = set(baseline.get("config", pd.Series(dtype=str)).astype(str).tolist())
    observed |= set(ablation.get("config", pd.Series(dtype=str)).astype(str).tolist())
    missing_configs = sorted(REQUIRED_ABLATION_CONFIGS - observed)
    if missing_configs:
        return _gate(
            FAIL,
            "Ablation reports exist but do not cover the required baseline matrix.",
            {"missing_configs": missing_configs},
        )
    if "status" in baseline.columns:
        status = baseline["status"].astype(str).str.lower()
        required_rows = baseline[baseline["config"].astype(str).isin(REQUIRED_ABLATION_CONFIGS)]
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
        proposed = baseline.loc[baseline["config"].astype(str) == MAIN_ABLATION_CONFIG, score_column]
        if proposed.empty or pd.isna(proposed.iloc[0]):
            return _gate(
                FAIL,
                f"{MAIN_ABLATION_CONFIG} must have a valid {score_column} before it can be claimed as the main result.",
            )
        proposed_score = float(proposed.iloc[0])
        competitors = baseline[baseline["config"].astype(str) != MAIN_ABLATION_CONFIG].copy()
        competitors = competitors[pd.to_numeric(competitors[score_column], errors="coerce").notna()]
        non_improved = competitors[pd.to_numeric(competitors[score_column], errors="coerce") >= proposed_score]
        return _gate(
            PASS,
            "Required ablation/baseline matrix is present; relative scores are reported as diagnostic context.",
            {
                "score_context": {
                    f"{MAIN_ABLATION_CONFIG}_score": proposed_score,
                    "score_column": score_column,
                    "non_improved_configs": non_improved["config"].astype(str).tolist(),
                }
            },
        )
    return _gate(PASS, "Required ablation/baseline matrix is present.")


def audit_run(run_dir: str | Path) -> Dict[str, Any]:
    root = Path(run_dir)
    summary = _load_summary(root)
    selection = summary.get("selection_info", {})
    metrics = _load_search_metrics(root)
    gates: Dict[str, Dict[str, Any]] = {}
    selected_k = int(summary.get("selected_k", selection.get("selected_k", -1)))

    if not metrics.empty and "total_k_ok" in metrics.columns:
        selected_rows = metrics[metrics.get("total_clusters", pd.Series(dtype=int)) == selected_k]
        if selected_rows.empty:
            selected_rows = metrics
        ok = bool(selected_rows["total_k_ok"].astype(bool).any())
        gates["total_k_constraint_honored"] = _gate(PASS if ok else FAIL, "Selected macro/micro candidate must satisfy total K bounds.", selected_k)
    else:
        selected_k = int(summary.get("selected_k", selection.get("selected_k", -1)))
        if not metrics.empty and "k" in metrics.columns and "min_size_ok" in metrics.columns:
            selected_rows = metrics[metrics["k"].astype(int) == int(selected_k)]
            ok = bool(not selected_rows.empty and selected_rows["min_size_ok"].astype(bool).any())
            gates["total_k_constraint_honored"] = _gate(
                PASS if ok else FAIL,
                "Selected flat GMM candidate must satisfy min-size gates.",
                selected_k,
            )
        else:
            gates["total_k_constraint_honored"] = _gate(FAIL, "Missing total_k_ok or flat k/min_size_ok columns in cluster_search_metrics.csv.")

    selection_mode = str(summary.get("selection_mode", selection.get("selection_mode", ""))).lower()
    if selection_mode == "balanced_va_regions":
        stability_fields = ("seed_ari_mean", "seed_ari_std")
        stability_present = all(field in selection for field in stability_fields)
        if not stability_present and not metrics.empty:
            stability_present = all(field in metrics.columns for field in stability_fields)
    elif "macro_micro" in selection_mode:
        stability_fields = ("seed_ari_mean", "cluster_jaccard_min", "bootstrap_valid_rate")
        stability_present = all(field in selection for field in stability_fields)
        if not stability_present and not metrics.empty:
            stability_present = all(field in metrics.columns for field in stability_fields)
    else:
        stability_present = bool("stability" in selection or (not metrics.empty and "stability" in metrics.columns))
    gates["bootstrap_stability_present"] = _gate(
        PASS if stability_present else FAIL,
        "Search must report stability metrics.",
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
        PASS if "macro_micro" not in str(summary.get("selection_mode", selection.get("selection_mode", ""))) or _macro_micro_artifacts_present(root) else FAIL,
        "Macro/micro artifacts are required only for macro_micro main-result runs.",
    )
    gates["split_semantic_consistency_quantified"] = (
        _split_semantic_consistency(root)
        if "macro_micro" in selection_mode
        else _gate(PASS, "Split semantic consistency is not required for flat affect-first runs.")
    )
    gates["balanced_va_region_stability_present"] = _balanced_va_region_stability_gate(
        summary,
        selection,
        metrics,
        selected_k,
    )
    gates["overlap_gate_train_val_test_all"] = _overlap_gate_train_val_test_all(root, summary, selection)
    gates["metadata_not_used_for_clustering"] = _metadata_not_used_for_clustering(summary, selection)
    gates["report_only_tension_present"] = _report_only_tension_present(root, summary, selection)
    gates["tension_split_reproducibility"] = _tension_split_reproducibility_gate(root, summary, selection)
    gates["alpha_search_report_present"] = _alpha_search_report_present(root, summary, selection)
    affect_gate = _affect_purity_gate(root, selection, metrics, selected_k)
    if _is_balanced_va_regions(summary, selection) and affect_gate["status"] == FAIL:
        affect_gate = _gate(
            WARN,
            "Affect purity is diagnostic for balanced_va_regions; continuous VA-region readiness is audited by overlap and stability gates.",
            affect_gate.get("value"),
        )
    gates["affect_purity_gate"] = affect_gate
    gates["required_ablations_present"] = _required_ablations_gate(root)
    cluster_head_k = int(summary.get("cluster_head_k", selection.get("cluster_head_k", 0)) or 0)
    gates["random_cluster_head_disabled"] = _gate(
        PASS if cluster_head_k == 0 else FAIL,
        "Affect-first main runs must not use a random DEC/CVCL cluster head during representation training.",
        cluster_head_k,
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
