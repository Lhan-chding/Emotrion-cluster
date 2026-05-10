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


def test_ablation_suite_builds_report_only_proposed_command(tmp_path):
    module = _load_ablation_suite_module()
    args = module.SuiteArgs(
        processed_dir="processed",
        base_run_dir="base",
        out_dir=str(tmp_path),
        gpu="0",
        batch_size=512,
        stability_runs=80,
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

    command = module.build_rerun_command(args, "proposed_no_metadata", tmp_path / "proposed_no_metadata")

    assert "--run_dir" in command
    assert "base" in command
    assert command[command.index("--cluster_feature_strategy") + 1] == "macro_micro_diffaware"
    assert command[command.index("--metadata_policy") + 1] == "report_only"
    assert command[command.index("--require_both_va") + 1] == "true"
    assert command[command.index("--k_strategy") + 1] == "macro_micro"
    assert command[command.index("--total_k_min") + 1] == "8"
    assert command[command.index("--total_k_max") + 1] == "16"


def test_ablation_suite_writes_required_comparison_reports(tmp_path):
    module = _load_ablation_suite_module()
    for name, score in [("mean_va", 0.1), ("proposed_full", 0.3)]:
        run_dir = tmp_path / name
        all_dir = run_dir / "all"
        all_dir.mkdir(parents=True)
        (run_dir / "rerun_summary.json").write_text(
            json.dumps(
                {
                    "selected_k": 9,
                    "k_strategy": "macro_micro" if name == "proposed_full" else "composite",
                    "cluster_feature_strategy": "macro_micro_diffaware" if name == "proposed_full" else "mean_va",
                    "selection_info": {"seed_ari_mean": 0.8, "cluster_jaccard_min": 0.5},
                    "mask_purity_diagnostics": {"nmi": 0.02, "clusters": [{"enrichment_vs_baseline": 1.1}]},
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"k": 9, "composite_score": score, "silhouette": score / 10.0}]).to_csv(
            all_dir / "cluster_search_metrics.csv",
            index=False,
        )

    baseline_path, ablation_path = module.write_suite_reports(tmp_path, ["mean_va", "proposed_full"])

    baseline = pd.read_csv(baseline_path)
    ablation = pd.read_csv(ablation_path)
    assert set(baseline["config"]) == {"mean_va", "proposed_full"}
    assert "score" in baseline.columns
    assert "claim_score" in baseline.columns
    assert "delta_score_vs_proposed_full" in ablation.columns
    assert "delta_claim_score_vs_proposed_full" in ablation.columns
    assert float(ablation.loc[ablation["config"] == "mean_va", "delta_score_vs_proposed_full"].iloc[0]) < 0.0


def test_ablation_suite_records_failed_configs_without_stopping_report(tmp_path):
    module = _load_ablation_suite_module()
    ok_dir = tmp_path / "mean_va"
    ok_all = ok_dir / "all"
    ok_all.mkdir(parents=True)
    (ok_dir / "rerun_summary.json").write_text(
        json.dumps(
            {
                "selected_k": 9,
                "k_strategy": "composite",
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
    failed_dir = tmp_path / "metadata_only"
    failed_dir.mkdir()
    (failed_dir / "run_error.json").write_text(
        json.dumps(
            {
                "config": "metadata_only",
                "status": "failed",
                "error_type": "CalledProcessError",
                "error_message": "No K candidate satisfied min_cluster_size_threshold=40",
                "returncode": 1,
            }
        ),
        encoding="utf-8",
    )

    baseline_path, ablation_path = module.write_suite_reports(tmp_path, ["mean_va", "metadata_only"])

    baseline = pd.read_csv(baseline_path)
    ablation = pd.read_csv(ablation_path)
    failed = baseline.loc[baseline["config"] == "metadata_only"].iloc[0]
    assert failed["status"] == "failed"
    assert pd.isna(failed["score"])
    assert "min_cluster_size_threshold=40" in failed["error_message"]
    assert set(ablation["config"]) == {"mean_va", "metadata_only"}


def test_ablation_suite_can_copy_reports_back_to_base_run(tmp_path):
    module = _load_ablation_suite_module()
    suite_dir = tmp_path / "suite"
    base_dir = tmp_path / "base"
    suite_dir.mkdir()
    base_dir.mkdir()
    (suite_dir / "baseline_comparison.csv").write_text("config,score\nproposed_full,1.0\n", encoding="utf-8")
    (suite_dir / "ablation_report.csv").write_text("config,score\nproposed_full,1.0\n", encoding="utf-8")

    copied = module.copy_reports_to_base_run(suite_dir, base_dir)

    assert (base_dir / "baseline_comparison.csv").read_text(encoding="utf-8").startswith("config")
    assert (base_dir / "ablation_report.csv").exists()
    assert set(copied) == {"baseline_comparison.csv", "ablation_report.csv"}
