import json
from pathlib import Path

import pandas as pd

from cluster.evaluation.topconf_audit import audit_run


def test_topconf_audit_fails_when_required_ablations_are_missing(tmp_path):
    run_dir = tmp_path
    (run_dir / "all" / "macro_micro").mkdir(parents=True)
    (run_dir / "train").mkdir()
    summary = {
        "selected_k": 9,
        "selection_mode": "macro_micro_diffaware",
        "selection_info": {
            "selected_k": 9,
            "stability_runs": 5,
            "seed_ari_mean": 0.8,
            "cluster_jaccard_min": 0.5,
            "bootstrap_valid_rate": 1.0,
        },
        "mask_purity_diagnostics": {
            "nmi": 0.02,
            "clusters": [{"enrichment_vs_baseline": 1.1}],
        },
        "label_names": {"0": "M1-a"},
    }
    (run_dir / "rerun_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "macro_k": 4,
                "total_clusters": 9,
                "total_k_ok": True,
                "min_size_ok": True,
                "seed_ari_mean": 0.8,
            }
        ]
    ).to_csv(run_dir / "train" / "cluster_search_metrics.csv", index=False)
    pd.DataFrame([{"label_name": "M1-a", "dominant_quadrant": "Q1"}]).to_csv(
        run_dir / "all" / "macro_micro_summary.csv",
        index=False,
    )
    (run_dir / "all" / "macro_micro_metadata_enrichment.csv").write_text("macro_id,token\n1,happy\n", encoding="utf-8")
    (run_dir / "all" / "macro_micro" / "macro_1_diff_arrow.png").write_bytes(b"png")

    result = audit_run(run_dir)

    assert result["overall_ready"] is False
    assert result["gates"]["required_ablations_present"]["status"] == "fail"
    assert result["gates"]["total_k_constraint_honored"]["status"] == "pass"
    assert result["gates"]["bootstrap_stability_present"]["status"] == "pass"


def test_topconf_audit_detects_total_k_constraint_failure(tmp_path):
    run_dir = tmp_path
    (run_dir / "train").mkdir()
    summary = {"selected_k": 9, "selection_info": {}}
    (run_dir / "rerun_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame([{"macro_k": 4, "total_clusters": 9, "total_k_ok": False, "min_size_ok": True}]).to_csv(
        run_dir / "train" / "cluster_search_metrics.csv",
        index=False,
    )

    result = audit_run(run_dir)

    assert result["gates"]["total_k_constraint_honored"]["status"] == "fail"


def test_topconf_audit_requires_named_ablation_configs_and_proposed_gain(tmp_path):
    run_dir = tmp_path
    (run_dir / "train").mkdir()
    summary = {"selected_k": 9, "selection_info": {}}
    (run_dir / "rerun_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame([{"macro_k": 4, "total_clusters": 9, "total_k_ok": True, "min_size_ok": True}]).to_csv(
        run_dir / "train" / "cluster_search_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"config": "raw_mean_va", "score": 0.40},
            {"config": "calibrated_va_tension_final_report_only", "score": 0.30},
        ]
    ).to_csv(run_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "config": "raw_mean_va",
                "score": 0.40,
                "delta_score_vs_calibrated_va_tension_final_report_only": 0.10,
            },
            {
                "config": "calibrated_va_tension_final_report_only",
                "score": 0.30,
                "delta_score_vs_calibrated_va_tension_final_report_only": 0.0,
            },
        ]
    ).to_csv(run_dir / "ablation_report.csv", index=False)

    result = audit_run(run_dir)

    gate = result["gates"]["required_ablations_present"]
    assert gate["status"] == "fail"
    assert "required baseline matrix" in gate["detail"].lower() or "outperform" in gate["detail"].lower()


def test_topconf_audit_fails_when_required_ablation_config_failed(tmp_path):
    run_dir = tmp_path
    (run_dir / "train").mkdir()
    summary = {"selected_k": 9, "selection_info": {}}
    (run_dir / "rerun_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame([{"macro_k": 4, "total_clusters": 9, "total_k_ok": True, "min_size_ok": True}]).to_csv(
        run_dir / "train" / "cluster_search_metrics.csv",
        index=False,
    )
    rows = [
        {"config": "audio_va", "status": "ok", "score": 0.10},
        {"config": "lyrics_va", "status": "ok", "score": 0.10},
        {"config": "raw_mean_va", "status": "ok", "score": 0.10},
        {"config": "calibrated_mean_alpha_0_5", "status": "ok", "score": 0.11},
        {"config": "clusterability_alpha", "status": "ok", "score": 0.12},
        {"config": "raw_mean_plus_signed_diff", "status": "ok", "score": 0.13},
        {"config": "calibrated_va_tension_final_report_only", "status": "ok", "score": 0.30},
        {"config": "latent_two_view_va_gmm", "status": "ok", "score": 0.14},
        {
            "config": "metadata_only_report_diagnostic",
            "status": "failed",
            "score": float("nan"),
            "error_message": "No K candidate satisfied min_cluster_size_threshold=40",
        },
    ]
    pd.DataFrame(rows).to_csv(run_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "ablation_report.csv", index=False)

    result = audit_run(run_dir)

    gate = result["gates"]["required_ablations_present"]
    assert gate["status"] == "fail"
    assert gate["value"]["failed_configs"] == ["metadata_only_report_diagnostic"]


def test_topconf_audit_uses_common_claim_score_to_reject_weaker_proposed_model(tmp_path):
    run_dir = tmp_path
    (run_dir / "train").mkdir()
    summary = {"selected_k": 9, "selection_info": {}, "require_both_va": True}
    (run_dir / "rerun_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame([{"macro_k": 4, "total_clusters": 9, "total_k_ok": True, "min_size_ok": True}]).to_csv(
        run_dir / "train" / "cluster_search_metrics.csv",
        index=False,
    )
    rows = [
        {"config": "audio_va", "status": "ok", "score": 0.80, "claim_score": 0.30},
        {"config": "lyrics_va", "status": "ok", "score": 0.10, "claim_score": 0.06},
        {"config": "raw_mean_va", "status": "ok", "score": 0.10, "claim_score": 0.08},
        {"config": "calibrated_mean_alpha_0_5", "status": "ok", "score": 0.11, "claim_score": 0.07},
        {"config": "clusterability_alpha", "status": "ok", "score": 0.12, "claim_score": 0.05},
        {"config": "raw_mean_plus_signed_diff", "status": "ok", "score": 0.13, "claim_score": 0.09},
        {
            "config": "calibrated_va_tension_final_report_only",
            "status": "ok",
            "score": 0.90,
            "claim_score": 0.20,
        },
        {"config": "latent_two_view_va_gmm", "status": "ok", "score": 0.14, "claim_score": 0.10},
        {"config": "metadata_only_report_diagnostic", "status": "ok", "score": 0.12, "claim_score": 0.04},
    ]
    pd.DataFrame(rows).to_csv(run_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "ablation_report.csv", index=False)

    result = audit_run(run_dir)

    gate = result["gates"]["required_ablations_present"]
    assert gate["status"] == "fail"
    assert gate["value"]["score_column"] == "claim_score"
    assert gate["value"]["non_improved_configs"] == ["audio_va"]


def test_topconf_audit_fails_large_mixed_affect_cluster(tmp_path):
    run_dir = tmp_path
    (run_dir / "train").mkdir()
    (run_dir / "all").mkdir()
    summary = {
        "selected_k": 12,
        "require_both_va": True,
        "cluster_head_k": 0,
        "selection_info": {
            "affect_gate_enabled": True,
            "min_affect_dominant_ratio": 0.70,
            "max_affect_mixed_cluster_fraction": 0.15,
            "min_affect_weighted_purity": 0.80,
            "seed_ari_mean": 0.8,
            "cluster_jaccard_min": 0.5,
            "bootstrap_valid_rate": 1.0,
        },
        "mask_purity_diagnostics": {"nmi": 0.0, "clusters": [{"enrichment_vs_baseline": 1.0}]},
        "metadata_policy": {"metadata_policy": "report_only"},
    }
    (run_dir / "rerun_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "macro_k": 5,
                "total_clusters": 12,
                "total_k_ok": True,
                "min_size_ok": True,
                "affect_gate_ok": False,
                "seed_ari_mean": 0.8,
                "cluster_jaccard_min": 0.5,
                "bootstrap_valid_rate": 1.0,
            }
        ]
    ).to_csv(run_dir / "train" / "cluster_search_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"cluster_id": 0, "num_samples": 100, "dominant_quadrant_ratio": 0.95},
            {"cluster_id": 1, "num_samples": 200, "dominant_quadrant_ratio": 0.42},
        ]
    ).to_csv(run_dir / "all" / "cluster_catalog.csv", index=False)

    result = audit_run(run_dir)

    gate = result["gates"]["affect_purity_gate"]
    assert gate["status"] == "fail"
    assert gate["value"]["affect_mixed_cluster_fraction"] > 0.15


def test_topconf_audit_balanced_va_region_gates_and_affect_diagnostic(tmp_path):
    run_dir = tmp_path
    for split in ("train", "val", "test", "all"):
        (run_dir / split).mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "va_knn_purity_10": 0.93,
                    "va_knn_purity_20": 0.89,
                    "va_center_radius_sep": 1.02,
                    "va_negative_silhouette_fraction": 0.02,
                    "va_mean_silhouette": 0.37,
                    "overlap_gate_ok": False if split == "val" else True,
                }
            ]
        ).to_csv(run_dir / split / "cluster_overlap_audit.csv", index=False)
    (run_dir / "all" / "tension_micro_probe").mkdir()
    (run_dir / "all" / "tension_micro_probe" / "tension_micro_probe.csv").write_text(
        "cluster_id,selected_micro_k\n0,2\n",
        encoding="utf-8",
    )
    (run_dir / "all" / "tension_substructure_report.md").write_text("# Tension\n", encoding="utf-8")
    (run_dir / "all" / "balance_alpha_report.csv").write_text("alpha,score\n0.6,0.9\n", encoding="utf-8")
    summary = {
        "selected_k": 4,
        "selection_mode": "balanced_va_regions",
        "cluster_feature_strategy": "calibrated_va_tension",
        "require_both_va": True,
        "cluster_head_k": 0,
        "metadata_policy": {
            "metadata_policy": "report_only",
            "effective_metadata_policy": "affective_va_only",
            "effective_metadata_cluster_weight": 0.0,
            "metadata_block_used_for_clustering": False,
        },
        "selection_info": {
            "selected_k": 4,
            "selection_mode": "balanced_va_regions",
            "seed_ari_mean": 0.98,
            "seed_ari_std": 0.01,
            "affect_gate_ok": False,
        },
        "mask_purity_diagnostics": {"nmi": 0.0, "clusters": [{"enrichment_vs_baseline": 1.0}]},
    }
    (run_dir / "rerun_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "k": 4,
                "min_size_ok": True,
                "seed_ari_mean": 0.98,
                "seed_ari_std": 0.01,
                "affect_gate_ok": False,
                "affect_weighted_dominant_ratio": 0.73,
                "affect_min_dominant_ratio": 0.49,
                "affect_mixed_cluster_fraction": 0.47,
            }
        ]
    ).to_csv(run_dir / "train" / "cluster_search_metrics.csv", index=False)
    required = [
        "audio_va",
        "lyrics_va",
        "raw_mean_va",
        "calibrated_mean_alpha_0_5",
        "clusterability_alpha",
        "raw_mean_plus_signed_diff",
        "calibrated_va_tension_final_report_only",
        "latent_two_view_va_gmm",
        "metadata_only_report_diagnostic",
    ]
    rows = [{"config": name, "status": "ok", "claim_score": 0.1 + idx * 0.01} for idx, name in enumerate(required)]
    pd.DataFrame(rows).to_csv(run_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "ablation_report.csv", index=False)

    result = audit_run(run_dir)

    assert result["gates"]["balanced_va_region_stability_present"]["status"] == "pass"
    assert result["gates"]["overlap_gate_train_val_test_all"]["status"] == "pass"
    assert result["gates"]["metadata_not_used_for_clustering"]["status"] == "pass"
    assert result["gates"]["report_only_tension_present"]["status"] == "pass"
    assert result["gates"]["alpha_search_report_present"]["status"] == "pass"
    assert result["gates"]["affect_purity_gate"]["status"] == "warn"
    assert "affect purity is diagnostic" in result["gates"]["affect_purity_gate"]["detail"].lower()
