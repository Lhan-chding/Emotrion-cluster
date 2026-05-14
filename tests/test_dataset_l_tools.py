import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_raw_csv_audit_writes_expected_outputs_for_dataset_l_scale_fixture(tmp_path):
    module = _load_script("audit_raw_unimodal_csv.py")
    rows = 10020
    frame = pd.DataFrame(
        {
            "Song": [f"song-{idx:05d}" for idx in range(rows)],
            "Artist": [f"artist-{idx % 23}" for idx in range(rows)],
            "Title": [f"title-{idx:05d}" for idx in range(rows)],
            "Quadrant": [f"Q{(idx % 4) + 1}" for idx in range(rows)],
            "Original_Arousal": np.linspace(0.1, 0.9, rows),
            "Original_Valence": np.linspace(0.9, 0.1, rows),
            "Audio_Arousal": (np.arange(rows) % 997) / 996.0,
            "Audio_Valence": (np.arange(rows) % 1009) / 1008.0,
            "Lyrics_Arousal": (np.arange(rows) % 991) / 990.0,
            "Lyrics_Valence": (np.arange(rows) % 983) / 982.0,
            "Genres": ["rock, pop"] * rows,
            "Moods": [f"mood-{idx % 37}" for idx in range(rows)],
            "MoodsAll": [f"mood-{idx % 37}, calm" for idx in range(rows)],
            "Themes": ["memory"] * rows,
            "Styles": ["alt"] * rows,
        }
    )
    source = tmp_path / "dataset_l.csv"
    frame.to_csv(source, index=False)

    summary = module.audit_raw_csv(source, tmp_path / "audit")

    assert summary["passed"] is True
    assert summary["rows"] == rows
    assert summary["both_audio_lyrics_complete_fraction"] == 1.0
    assert summary["unique_audio_va_pairs"] >= 1000
    assert (tmp_path / "audit" / "raw_csv_audit_report.md").exists()
    assert (tmp_path / "audit" / "raw_csv_column_summary.csv").exists()
    assert (tmp_path / "audit" / "metadata_coverage.csv").exists()


def test_raw_csv_audit_flags_duplicate_and_tiny_inputs(tmp_path):
    module = _load_script("audit_raw_unimodal_csv.py")
    source = tmp_path / "bad.csv"
    pd.DataFrame(
        [
            {"Song": "dup", "Audio_Arousal": 0.1, "Audio_Valence": 0.2, "Lyrics_Arousal": 0.3, "Lyrics_Valence": 0.4},
            {"Song": "dup", "Audio_Arousal": 0.1, "Audio_Valence": 0.2, "Lyrics_Arousal": 0.3, "Lyrics_Valence": 0.4},
        ]
    ).to_csv(source, index=False)

    summary = module.audit_raw_csv(source, tmp_path / "audit")
    saved = json.loads((tmp_path / "audit" / "raw_csv_audit_summary.json").read_text(encoding="utf-8"))

    assert summary["passed"] is False
    assert summary["duplicate_song_count"] == 2
    assert saved["failures"]


def test_compare_dataset_s_l_writes_alignment_outputs(tmp_path):
    module = _load_script("compare_dataset_s_l_results.py")
    s_dir = tmp_path / "dataset_s"
    l_dir = tmp_path / "dataset_l"
    for root, centers in (
        (s_dir, [(0, "Warm", 0.7, 0.3, 100), (1, "Dark", 0.2, 0.8, 120)]),
        (l_dir, [(0, "Warm-L", 0.72, 0.31, 1000), (1, "Dark-L", 0.21, 0.79, 1100)]),
    ):
        (root / "all").mkdir(parents=True)
        (root / "rerun_summary.json").write_text(
            json.dumps({"selected_k": 2, "selection_info": {"balance_alpha": 0.5, "balanced_region_score": 0.7}}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "cluster_id": cid,
                    "canonical_name": name,
                    "balanced_valence": valence,
                    "balanced_arousal": arousal,
                    "size": size,
                }
                for cid, name, valence, arousal, size in centers
            ]
        ).to_csv(root / "all" / "canonical_affect_regions.csv", index=False)

    summary = module.compare_runs(s_dir, l_dir, tmp_path / "comparison")
    alignment = pd.read_csv(tmp_path / "comparison" / "dataset_s_l_region_alignment.csv")

    assert summary["dataset_l_selected_k"] == 2
    assert len(alignment) == 2
    assert (tmp_path / "comparison" / "dataset_s_l_comparison_report.md").exists()


def test_compile_dataset_l_report_writes_representatives_and_missing_outputs(tmp_path):
    module = _load_script("compile_dataset_l_paper_report.py")
    run_dir = tmp_path / "run"
    all_dir = run_dir / "all"
    all_dir.mkdir(parents=True)
    (run_dir / "rerun_summary.json").write_text(
        json.dumps(
            {
                "selected_k": 1,
                "selection_info": {
                    "balanced_region_score": 0.6,
                    "balance_alpha": 0.5,
                    "va_mean_silhouette": 0.4,
                    "va_knn_purity_20": 0.9,
                    "va_center_radius_sep": 0.8,
                    "va_negative_silhouette_fraction": 0.05,
                    "seed_ari_mean": 0.95,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "canonical_name": "Warm Calm-Positive",
                "size": 2,
                "balanced_valence": 0.7,
                "balanced_arousal": 0.3,
                "top_tokens": "warm",
            }
        ]
    ).to_csv(all_dir / "canonical_affect_regions.csv", index=False)
    pd.DataFrame(
        [
            {"identifier": "song-a", "cluster_id": 0, "artist": "A", "title": "T", "balanced_valence": 0.71, "balanced_arousal": 0.29},
            {"identifier": "song-b", "cluster_id": 0, "artist": "B", "title": "U", "balanced_valence": 0.75, "balanced_arousal": 0.28},
        ]
    ).to_csv(all_dir / "cluster_assignments.csv", index=False)
    pd.DataFrame(
        [{"identifier": "song-a", "cluster_id": 0, "tension_micro_id": 0, "tension_subtype_label": "modality-consistent", "tension_norm": 0.1}]
    ).to_csv(all_dir / "tension_subtype_assignments.csv", index=False)
    (all_dir / "tension_substructure_report.md").write_text("# Tension\n", encoding="utf-8")
    pd.DataFrame([{"cluster_id": 0, "mean_tension_norm": 0.1}]).to_csv(all_dir / "tension_substructure_enrichment.csv", index=False)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "raw_csv_audit_summary.json").write_text(json.dumps({"rows": 2, "unique_song_count": 2}), encoding="utf-8")

    summary = module.compile_report(run_dir, raw_audit_dir=raw_dir, comparison_dir=run_dir)

    assert summary["representative_rows"] == 2
    assert (run_dir / "dataset_L_representative_tracks.csv").exists()
    assert (run_dir / "dataset_L_tension_substructure_report.md").exists()
    assert (run_dir / "final_paper_ready_report.md").exists()
    assert (run_dir / "missing_outputs.md").exists()
