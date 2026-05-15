import json

from scripts.repair_tension_micro_probe import _update_repaired_summary_jsons


def test_update_repaired_summary_jsons_rewrites_split_and_root_payloads(tmp_path):
    run_dir = tmp_path / "run"
    split_dir = run_dir / "all"
    split_dir.mkdir(parents=True)
    stale_probe = {"summary": [{"cluster_id": 0, "mean_tension_norm": 9.9}]}
    fresh_probe = {"summary": [{"cluster_id": 0, "mean_tension_norm": 0.11}]}
    (split_dir / "cluster_summary.json").write_text(
        json.dumps(
            {
                "split": "all",
                "tension_micro_probe": stale_probe,
                "output_files": {"tension_micro_probe": "old.csv", "cluster_summary": "keep.json"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "rerun_summary.json").write_text(
        json.dumps(
            {
                "selected_k": 5,
                "split_outputs": {
                    "all": {
                        "tension_micro_probe": stale_probe,
                        "output_files": {"tension_micro_probe": "old.csv", "cluster_summary": "keep.json"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    _update_repaired_summary_jsons(
        run_dir,
        "all",
        tension_micro_outputs={
            "tension_micro_probe": fresh_probe,
            "output_files": {"tension_micro_probe": "new.csv", "tension_micro_assignments": "assign.csv"},
        },
        tension_substructure_outputs={"tension_substructure_report": "report.md"},
    )

    split_payload = json.loads((split_dir / "cluster_summary.json").read_text(encoding="utf-8"))
    root_payload = json.loads((run_dir / "rerun_summary.json").read_text(encoding="utf-8"))
    root_split = root_payload["split_outputs"]["all"]

    assert split_payload["tension_micro_probe"] == fresh_probe
    assert split_payload["output_files"]["cluster_summary"] == "keep.json"
    assert split_payload["output_files"]["tension_micro_probe"] == "new.csv"
    assert split_payload["output_files"]["tension_substructure_report"] == "report.md"
    assert root_split["tension_micro_probe"] == fresh_probe
    assert root_split["output_files"]["tension_micro_assignments"] == "assign.csv"
    assert root_split["output_files"]["tension_substructure_report"] == "report.md"
