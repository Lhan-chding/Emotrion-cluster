import json

import pandas as pd

from scripts.posthoc_song_affective_profile import run_posthoc_profile


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_posthoc_profile_writes_all_selected_report_and_sanity_check(tmp_path):
    run_dir = tmp_path / "run"
    all_dir = run_dir / "all"
    all_dir.mkdir(parents=True)

    assignments = [
        ("S001", 0, 0.20, 0.20),
        ("S002", 0, 0.24, 0.22),
        ("S003", 1, 0.70, 0.25),
        ("S004", 1, 0.68, 0.28),
        ("S005", 2, 0.25, 0.82),
        ("S006", 2, 0.28, 0.78),
        ("S007", 3, 0.82, 0.76),
        ("S008", 3, 0.78, 0.80),
    ]
    _write_csv(
        all_dir / "cluster_assignments.csv",
        [
            {
                "identifier": song_id,
                "cluster_id": cluster_id,
                "balanced_valence": valence,
                "balanced_arousal": arousal,
            }
            for song_id, cluster_id, valence, arousal in assignments
        ],
    )
    tension_rows = [
        ("S001", 0, "C0-T0", "modality-consistent", 0.01, 0.01, 0.014),
        ("S002", 0, "C0-T1", "lyric-brightened", 0.25, 0.05, 0.255),
        ("S003", 1, "C1-T0", "modality-consistent", -0.01, 0.01, 0.014),
        ("S004", 1, "C1-T1", "lyric-softened", -0.05, -0.22, 0.226),
        ("S005", 2, "C2-T0", "modality-consistent", 0.02, -0.01, 0.022),
        ("S006", 2, "C2-T1", "lyric-darkened", -0.24, 0.02, 0.241),
        ("S007", 3, "C3-T0", "modality-consistent", 0.01, 0.02, 0.022),
        ("S008", 3, "C3-T1", "lyric-intensified", 0.04, 0.28, 0.283),
    ]
    _write_csv(
        all_dir / "tension_subtype_assignments.csv",
        [
            {
                "identifier": song_id,
                "cluster_id": cluster_id,
                "tension_label": label,
                "tension_subtype_label": subtype,
                "tension_dv": dv,
                "tension_da": da,
                "tension_norm": norm,
            }
            for song_id, cluster_id, label, subtype, dv, da, norm in tension_rows
        ],
    )
    metadata_csv = tmp_path / "metadata.csv"
    _write_csv(
        metadata_csv,
        [
            {"Song": song_id, "Title": f"Title {song_id}", "Artist": f"Artist {song_id}"}
            for song_id, *_rest in assignments
        ],
    )
    selected_csv = tmp_path / "selected.csv"
    _write_csv(
        selected_csv,
        [
            {"identifier": "S002", "song": "Artist S002 - Title S002"},
            {"identifier": "S999", "song": "Missing Song"},
        ],
    )

    summary = run_posthoc_profile(
        run_dir=run_dir,
        song_metadata_csv=metadata_csv,
        selected_songs_csv=selected_csv,
        out_dir=tmp_path / "out",
        make_figures=False,
    )

    out_dir = tmp_path / "out"
    all_profile = pd.read_csv(out_dir / "song_affective_profile_all.csv")
    selected_profile = pd.read_csv(out_dir / "song_affective_profile_selected.csv")
    missing_selected = pd.read_csv(out_dir / "missing_selected_songs.csv")
    sanity = json.loads((out_dir / "sanity_check.json").read_text(encoding="utf-8"))

    assert summary["total_songs"] == 8
    assert len(all_profile) == 8
    assert selected_profile["song_id"].tolist() == ["S002"]
    assert missing_selected["identifier"].tolist() == ["S999"]
    assert sanity["selected_found_count"] == 1
    assert sanity["selected_missing_count"] == 1
    assert all_profile.set_index("song_id").loc["S002", "cluster_id"] == 0
    assert all_profile.set_index("song_id").loc["S002", "tension_label"] == "C0-T1"

    weight_cols = ["w_region_C0", "w_region_C1", "w_region_C2", "w_region_C3"]
    assert all_profile[weight_cols].sum(axis=1).round(6).eq(1.0).all()
    assert all_profile["region_typicality"].between(0.0, 1.0).all()
    assert all_profile["tension_strength_percentile"].between(0.0, 1.0).all()

    descriptors = json.loads(selected_profile.iloc[0]["top_descriptors_json"])
    assert 1 <= len(descriptors) <= 8
    assert max(item["weight"] for item in descriptors) <= 1.0
    assert any("lyric valence uplift" == item["descriptor"] for item in descriptors)
    assert "Subdued Melancholy" in selected_profile.iloc[0]["english_interpretation"]
    assert "calibrated cross-modal tension" in selected_profile.iloc[0]["english_interpretation"]
    assert "Subdued Melancholy" in selected_profile.iloc[0]["chinese_interpretation"]

    assert (out_dir / "song_affective_profile_report.md").exists()
    assert (out_dir / "descriptor_weights_selected.json").exists()
