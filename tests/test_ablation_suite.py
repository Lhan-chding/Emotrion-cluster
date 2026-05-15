import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_ablation_suite_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation_suite.py"
    spec = importlib.util.spec_from_file_location("run_ablation_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ablation_suite_builds_report_only_final_command(tmp_path):
    module = _load_ablation_suite_module()
    args = module.SuiteArgs(
        processed_dir="processed",
        base_run_dir="base",
        out_dir=str(tmp_path),
        gpu="0",
        batch_size=512,
        stability_runs=80,
        stability_sample_size=0,
        k_min=8,
        k_max=16,
        macro_k_min=3,
        macro_k_max=6,
        micro_k_min=1,
        micro_k_max=5,
        min_cluster_size_abs=40,
        metadata_policy="report_only",
        require_both_va=True,
    )

    command = module.build_rerun_command(
        args,
        "calibrated_va_tension_final_report_only",
        tmp_path / "calibrated_va_tension_final_report_only",
    )

    assert "--run_dir" not in command
    assert command[command.index("--cluster_feature_strategy") + 1] == "calibrated_va_tension"
    assert command[command.index("--metadata_policy") + 1] == "report_only"
    assert command[command.index("--require_both_va") + 1] == "true"
    assert command[command.index("--k_strategy") + 1] == "balanced_va_regions"
    assert command[command.index("--cluster_assignment_mode") + 1] == "joint"
    assert command[command.index("--plot_va_source") + 1] == "cluster_consensus"
    assert command[command.index("--total_k_min") + 1] == "8"
    assert command[command.index("--total_k_max") + 1] == "16"
    assert command[command.index("--cluster_backend") + 1] == "torch"
    assert command[command.index("--eval_backend") + 1] == "torch"
    assert command[command.index("--silhouette_mode") + 1] == "torch_chunked"
    assert command[command.index("--silhouette_sample_size") + 1] == "50000"
    assert command[command.index("--silhouette_chunk_size") + 1] == "16384"
    assert command[command.index("--affect_gate") + 1] == "true"


def test_ablation_suite_uses_diff_contamination_as_composite_baseline(tmp_path):
    module = _load_ablation_suite_module()
    args = module.SuiteArgs(
        processed_dir="processed",
        base_run_dir="base",
        out_dir=str(tmp_path),
        gpu="0",
        batch_size=512,
        stability_runs=80,
        stability_sample_size=0,
        k_min=4,
        k_max=12,
        macro_k_min=3,
        macro_k_max=6,
        micro_k_min=1,
        micro_k_max=5,
        min_cluster_size_abs=40,
        metadata_policy="report_only",
        require_both_va=True,
    )

    command = module.build_rerun_command(args, "raw_mean_plus_signed_diff", tmp_path / "raw_mean_plus_signed_diff")

    assert "--run_dir" not in command
    assert command[command.index("--cluster_feature_strategy") + 1] == "mean_va_diff"
    assert command[command.index("--k_strategy") + 1] == "composite"
    assert command[command.index("--cluster_assignment_mode") + 1] == "joint"
    assert command[command.index("--diagnostic_allow_failed_gates") + 1] == "true"
    assert command[command.index("--cluster_backend") + 1] == "torch"


def test_ablation_suite_builds_latent_two_view_gmm_command(tmp_path):
    module = _load_ablation_suite_module()
    args = module.SuiteArgs(
        processed_dir="processed",
        base_run_dir=None,
        out_dir=str(tmp_path),
        gpu="0",
        batch_size=512,
        stability_runs=80,
        stability_sample_size=0,
        k_min=4,
        k_max=12,
        macro_k_min=3,
        macro_k_max=6,
        micro_k_min=1,
        micro_k_max=5,
        min_cluster_size_abs=40,
        metadata_policy="report_only",
        require_both_va=True,
    )

    command = module.build_rerun_command(args, "latent_two_view_va_gmm", tmp_path / "latent_two_view_va_gmm")

    assert command[command.index("--cluster_feature_strategy") + 1] == "latent_two_view_va"
    assert command[command.index("--k_strategy") + 1] == "latent_va_gmm"
    assert command[command.index("--cluster_assignment_mode") + 1] == "joint"
    assert command[command.index("--total_k_min") + 1] == "4"
    assert command[command.index("--total_k_max") + 1] == "8"
    assert command[command.index("--latent_learn_view_bias") + 1] == "true"
    assert command[command.index("--latent_share_view_noise") + 1] == "false"
    assert command[command.index("--latent_alpha_prior_strength") + 1] == "0.2"
    assert command[command.index("--latent_max_iter") + 1] == "200"
    assert command[command.index("--plot_va_source") + 1] == "latent_consensus"
    assert command[command.index("--diagnostic_allow_failed_gates") + 1] == "true"
    assert command[command.index("--eval_backend") + 1] == "torch"


def test_ablation_suite_accepts_dataset_l_required_configs_and_builds_sensitivity_commands(tmp_path):
    module = _load_ablation_suite_module()
    expected = {
        "audio_va",
        "lyrics_va",
        "raw_mean_va",
        "calibrated_mean_alpha_0_5",
        "fixed_alpha_0_60",
        "clusterability_alpha",
        "raw_mean_plus_signed_diff",
        "residual_tension_weak_concat",
        "calibrated_va_tension_final_report_only",
        "metadata_only_report_diagnostic",
        "k4_sensitivity",
        "k5_sensitivity",
        "k6_sensitivity",
    }
    assert expected.issubset(set(module.DEFAULT_CONFIGS))

    args = module.SuiteArgs(
        processed_dir="processed",
        base_run_dir=None,
        out_dir=str(tmp_path),
        gpu="3",
        batch_size=4096,
        stability_runs=40,
        stability_sample_size=50000,
        k_min=4,
        k_max=16,
        macro_k_min=3,
        macro_k_max=6,
        micro_k_min=1,
        micro_k_max=5,
        min_cluster_size_abs=800,
        metadata_policy="report_only",
        require_both_va=True,
    )

    fixed_alpha = module.build_rerun_command(args, "fixed_alpha_0_60", tmp_path / "fixed_alpha_0_60")
    residual = module.build_rerun_command(args, "residual_tension_weak_concat", tmp_path / "residual_tension_weak_concat")
    k6 = module.build_rerun_command(args, "k6_sensitivity", tmp_path / "k6_sensitivity")

    assert fixed_alpha[fixed_alpha.index("--consensus_mode") + 1] == "global_alpha"
    assert fixed_alpha[fixed_alpha.index("--consensus_alpha") + 1] == "0.6"
    assert residual[residual.index("--k_strategy") + 1] == "composite"
    assert residual[residual.index("--cluster_feature_strategy") + 1] == "calibrated_va_tension"
    assert residual[residual.index("--diff_cluster_weight") + 1] == "0.1"
    assert k6[k6.index("--total_k_min") + 1] == "6"
    assert k6[k6.index("--total_k_max") + 1] == "6"
    assert fixed_alpha[fixed_alpha.index("--silhouette_sample_size") + 1] == "50000"


def test_ablation_suite_builds_metadata_diagnostic_with_failed_gate_override(tmp_path):
    module = _load_ablation_suite_module()
    args = module.SuiteArgs(
        processed_dir="processed",
        base_run_dir=None,
        out_dir=str(tmp_path),
        gpu="0",
        batch_size=512,
        stability_runs=80,
        stability_sample_size=0,
        k_min=4,
        k_max=12,
        macro_k_min=3,
        macro_k_max=6,
        micro_k_min=1,
        micro_k_max=5,
        min_cluster_size_abs=40,
        metadata_policy="report_only",
        require_both_va=True,
    )

    command = module.build_rerun_command(
        args,
        "metadata_only_report_diagnostic",
        tmp_path / "metadata_only_report_diagnostic",
    )

    assert command[command.index("--cluster_feature_strategy") + 1] == "metadata_only"
    assert command[command.index("--metadata_policy") + 1] == "all_metadata_upper_bound"
    assert command[command.index("--require_both_va") + 1] == "false"
    assert command[command.index("--diagnostic_allow_failed_gates") + 1] == "true"


def test_ablation_suite_writes_required_comparison_reports(tmp_path):
    module = _load_ablation_suite_module()
    for name, score in [("raw_mean_va", 0.1), ("calibrated_va_tension_final_report_only", 0.3)]:
        run_dir = tmp_path / name
        all_dir = run_dir / "all"
        all_dir.mkdir(parents=True)
        (run_dir / "rerun_summary.json").write_text(
            json.dumps(
                {
                    "selected_k": 9,
                    "k_strategy": "balanced_va_regions",
                    "cluster_feature_strategy": "calibrated_va_tension" if name == "calibrated_va_tension_final_report_only" else "mean_va",
                    "metadata_policy": {"metadata_policy": "report_only", "effective_metadata_cluster_weight": 0.0},
                    "selection_info": {"seed_ari_mean": 0.8, "seed_ari_std": 0.05, "cluster_jaccard_min": 0.5},
                    "mask_purity_diagnostics": {"nmi": 0.02, "clusters": [{"enrichment_vs_baseline": 1.1}]},
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "k": 9,
                    "balanced_region_score": score,
                    "va_mean_silhouette": score / 10.0,
                    "va_knn_purity_10": 0.90,
                    "va_knn_purity_20": 0.88,
                    "va_center_radius_sep": 0.95,
                    "va_negative_silhouette_fraction": 0.02,
                    "size_balance": 0.75,
                    "min_cluster_size": 60,
                }
            ]
        ).to_csv(
            all_dir / "cluster_search_metrics.csv",
            index=False,
        )

    baseline_path, ablation_path = module.write_suite_reports(
        tmp_path,
        ["raw_mean_va", "calibrated_va_tension_final_report_only"],
    )

    baseline = pd.read_csv(baseline_path)
    ablation = pd.read_csv(ablation_path)
    assert set(baseline["config"]) == {"raw_mean_va", "calibrated_va_tension_final_report_only"}
    assert "score" in baseline.columns
    assert "claim_score" in baseline.columns
    assert "va_silhouette" in baseline.columns
    assert "knn_purity_20" in baseline.columns
    assert "center_radius_sep" in baseline.columns
    assert "negative_silhouette_fraction" in baseline.columns
    assert "seed_ari_std" in baseline.columns
    assert "size_balance" in baseline.columns
    assert "metadata_policy" in baseline.columns
    assert "delta_score_vs_calibrated_va_tension_final_report_only" in ablation.columns
    assert "delta_claim_score_vs_calibrated_va_tension_final_report_only" in ablation.columns
    assert (
        float(
            ablation.loc[
                ablation["config"] == "raw_mean_va",
                "delta_score_vs_calibrated_va_tension_final_report_only",
            ].iloc[0]
        )
        < 0.0
    )


def test_ablation_suite_balanced_va_claim_score_ignores_diagnostic_affect_penalty(tmp_path):
    module = _load_ablation_suite_module()
    run_dir = tmp_path / "clusterability_alpha"
    all_dir = run_dir / "all"
    all_dir.mkdir(parents=True)
    (run_dir / "rerun_summary.json").write_text(
        json.dumps(
            {
                "selected_k": 4,
                "k_strategy": "balanced_va_regions",
                "selection_mode": "balanced_va_regions",
                "cluster_feature_strategy": "calibrated_va_tension",
                "metadata_policy": {"metadata_policy": "report_only", "effective_metadata_cluster_weight": 0.0},
                "selection_info": {"seed_ari_mean": 0.9, "seed_ari_std": 0.01},
                "mask_purity_diagnostics": {"nmi": 0.0, "clusters": []},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "k": 4,
                "balanced_region_score": 0.7,
                "va_mean_silhouette": 0.38,
                "affect_weighted_dominant_ratio": 0.50,
                "affect_min_dominant_ratio": 0.40,
                "affect_mixed_cluster_fraction": 0.60,
            }
        ]
    ).to_csv(all_dir / "cluster_search_metrics.csv", index=False)

    row = module.summarize_run(run_dir, "clusterability_alpha")

    assert row["claim_score"] == 0.38


def test_ablation_suite_records_failed_configs_without_stopping_report(tmp_path):
    module = _load_ablation_suite_module()
    ok_dir = tmp_path / "raw_mean_va"
    ok_all = ok_dir / "all"
    ok_all.mkdir(parents=True)
    (ok_dir / "rerun_summary.json").write_text(
        json.dumps(
            {
                "selected_k": 9,
                "k_strategy": "balanced_va_regions",
                "cluster_feature_strategy": "mean_va",
                "selection_info": {},
                "mask_purity_diagnostics": {"nmi": 0.01, "clusters": []},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"k": 9, "composite_score": 0.2, "silhouette": 0.05}]).to_csv(
        ok_all / "cluster_search_metrics.csv",
        index=False,
    )
    failed_dir = tmp_path / "metadata_only_report_diagnostic"
    failed_dir.mkdir()
    (failed_dir / "run_error.json").write_text(
        json.dumps(
            {
                "config": "metadata_only_report_diagnostic",
                "status": "failed",
                "error_type": "CalledProcessError",
                "error_message": "No K candidate satisfied min_cluster_size_threshold=40",
                "returncode": 1,
            }
        ),
        encoding="utf-8",
    )

    baseline_path, ablation_path = module.write_suite_reports(tmp_path, ["raw_mean_va", "metadata_only_report_diagnostic"])

    baseline = pd.read_csv(baseline_path)
    ablation = pd.read_csv(ablation_path)
    failed = baseline.loc[baseline["config"] == "metadata_only_report_diagnostic"].iloc[0]
    assert failed["status"] == "failed"
    assert pd.isna(failed["score"])
    assert "min_cluster_size_threshold=40" in failed["error_message"]
    assert set(ablation["config"]) == {"raw_mean_va", "metadata_only_report_diagnostic"}


def test_ablation_suite_can_copy_reports_back_to_base_run(tmp_path):
    module = _load_ablation_suite_module()
    suite_dir = tmp_path / "suite"
    base_dir = tmp_path / "base"
    suite_dir.mkdir()
    base_dir.mkdir()
    (suite_dir / "baseline_comparison.csv").write_text(
        "config,score\ncalibrated_va_tension_final_report_only,1.0\n",
        encoding="utf-8",
    )
    (suite_dir / "ablation_report.csv").write_text(
        "config,score\ncalibrated_va_tension_final_report_only,1.0\n",
        encoding="utf-8",
    )

    copied = module.copy_reports_to_base_run(suite_dir, base_dir)

    assert (base_dir / "baseline_comparison.csv").read_text(encoding="utf-8").startswith("config")
    assert (base_dir / "ablation_report.csv").exists()
    assert set(copied) == {"baseline_comparison.csv", "ablation_report.csv"}
