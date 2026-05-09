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
            {"config": "mean_va", "score": 0.40},
            {"config": "proposed_full", "score": 0.30},
        ]
    ).to_csv(run_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"config": "mean_va", "score": 0.40, "delta_score_vs_proposed_full": 0.10},
            {"config": "proposed_full", "score": 0.30, "delta_score_vs_proposed_full": 0.0},
        ]
    ).to_csv(run_dir / "ablation_report.csv", index=False)

    result = audit_run(run_dir)

    gate = result["gates"]["required_ablations_present"]
    assert gate["status"] == "fail"
    assert "required baseline matrix" in gate["detail"].lower() or "outperform" in gate["detail"].lower()
