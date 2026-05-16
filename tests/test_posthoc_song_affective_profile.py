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
        output_suffix="_v2",
    )

    out_dir = tmp_path / "out"
    all_profile = pd.read_csv(out_dir / "song_affective_profile_all_v2.csv")
    selected_profile = pd.read_csv(out_dir / "song_affective_profile_selected_v2.csv")
    missing_selected = pd.read_csv(out_dir / "missing_selected_songs_v2.csv")
    sanity = json.loads((out_dir / "sanity_check_v2.json").read_text(encoding="utf-8"))

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
    assert max(item["display_descriptor_weight"] for item in descriptors) <= 1.0
    assert all("raw_descriptor_score" in item for item in descriptors)
    assert any("lyric valence uplift" == item["descriptor"] for item in descriptors)
    assert "Subdued Melancholy" in selected_profile.iloc[0]["english_interpretation"]
    assert "calibrated cross-modal tension" in selected_profile.iloc[0]["english_interpretation"]
    assert "Subdued Melancholy" in selected_profile.iloc[0]["chinese_interpretation"]

    assert {"region_role", "boundary_flag", "descriptor_conflict_flag", "main_text_eligible"}.issubset(
        all_profile.columns
    )
    assert (out_dir / "song_affective_profile_report_v2.md").exists()
    assert (out_dir / "descriptor_weights_selected_v2.json").exists()


def test_posthoc_profile_applies_review_gates_and_report_sections(tmp_path):
    run_dir = tmp_path / "run"
    all_dir = run_dir / "all"
    all_dir.mkdir(parents=True)

    assignments = [
        ("P0A", 0, 0.10, 0.10),
        ("P0B", 0, 0.12, 0.12),
        ("P0C", 0, 0.14, 0.14),
        ("B0", 0, 0.89, 0.89),
        ("P1A", 1, 0.90, 0.90),
        ("P1B", 1, 0.88, 0.88),
        ("P1C", 1, 0.86, 0.86),
        ("C1", 1, 0.91, 0.91),
        ("X1", 1, 0.89, 0.89),
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
    _write_csv(
        all_dir / "tension_subtype_assignments.csv",
        [
            {
                "identifier": "P0A",
                "cluster_id": 0,
                "tension_label": "C0-T0",
                "tension_subtype_label": "modality-consistent",
                "tension_dv": 0.00,
                "tension_da": 0.00,
                "tension_norm": 0.01,
            },
            {
                "identifier": "P0B",
                "cluster_id": 0,
                "tension_label": "C0-T0",
                "tension_subtype_label": "modality-consistent",
                "tension_dv": 0.01,
                "tension_da": 0.01,
                "tension_norm": 0.02,
            },
            {
                "identifier": "P0C",
                "cluster_id": 0,
                "tension_label": "C0-T0",
                "tension_subtype_label": "modality-consistent",
                "tension_dv": -0.01,
                "tension_da": -0.01,
                "tension_norm": 0.03,
            },
            {
                "identifier": "B0",
                "cluster_id": 0,
                "tension_label": "C0-T1",
                "tension_subtype_label": "lyric-brightened",
                "tension_dv": 0.40,
                "tension_da": 0.10,
                "tension_norm": 0.41,
            },
            {
                "identifier": "P1A",
                "cluster_id": 1,
                "tension_label": "C1-T0",
                "tension_subtype_label": "modality-consistent",
                "tension_dv": 0.00,
                "tension_da": 0.00,
                "tension_norm": 0.01,
            },
            {
                "identifier": "P1B",
                "cluster_id": 1,
                "tension_label": "C1-T0",
                "tension_subtype_label": "modality-consistent",
                "tension_dv": 0.01,
                "tension_da": 0.01,
                "tension_norm": 0.02,
            },
            {
                "identifier": "P1C",
                "cluster_id": 1,
                "tension_label": "C1-T0",
                "tension_subtype_label": "modality-consistent",
                "tension_dv": -0.01,
                "tension_da": -0.01,
                "tension_norm": 0.03,
            },
            {
                "identifier": "C1",
                "cluster_id": 1,
                "tension_label": "C1-T0",
                "tension_subtype_label": "modality-consistent",
                "tension_dv": 0.02,
                "tension_da": 0.02,
                "tension_norm": 0.04,
            },
            {
                "identifier": "X1",
                "cluster_id": 1,
                "tension_label": "C1-T1",
                "tension_subtype_label": "lyric-intensified",
                "tension_dv": 0.60,
                "tension_da": 0.80,
                "tension_norm": 1.00,
            },
        ],
    )
    metadata_csv = tmp_path / "metadata.csv"
    _write_csv(
        metadata_csv,
        [
            {
                "Song": song_id,
                "Title": "I Fucking Hate You" if song_id == "X1" else f"Title {song_id}",
                "Artist": "Artist",
            }
            for song_id, *_rest in assignments
        ],
    )
    selected_csv = tmp_path / "selected.csv"
    _write_csv(selected_csv, [{"identifier": song_id} for song_id in ["P0A", "B0", "X1"]])

    summary = run_posthoc_profile(
        run_dir=run_dir,
        song_metadata_csv=metadata_csv,
        selected_songs_csv=selected_csv,
        out_dir=tmp_path / "out",
        make_figures=False,
        output_suffix="_v2",
    )

    out_dir = tmp_path / "out"
    all_profile = pd.read_csv(out_dir / "song_affective_profile_all_v2.csv").set_index("song_id")
    selected_profile = pd.read_csv(out_dir / "song_affective_profile_selected_v2.csv").set_index("song_id")
    report = (out_dir / "song_affective_profile_report_v2.md").read_text(encoding="utf-8-sig")
    sanity = json.loads((out_dir / "sanity_check_v2.json").read_text(encoding="utf-8"))

    assert all_profile.loc["B0", "region_role"] == "boundary"
    assert bool(all_profile.loc["B0", "boundary_flag"])
    assert not bool(all_profile.loc["B0", "main_text_eligible"])
    boundary_descriptors = json.loads(all_profile.loc["B0", "top_descriptors_json"])
    assert "boundary between assigned region and nearest alternative" == boundary_descriptors[0]["descriptor"]
    assert not any(item["descriptor"] == "playful vitality" for item in boundary_descriptors[:3])

    consistent_descriptors = json.loads(all_profile.loc["P1A", "top_descriptors_json"])
    forbidden = {"lyric arousal intensification", "lyric valence tempering", "high cross-modal tension"}
    assert forbidden.isdisjoint({item["descriptor"] for item in consistent_descriptors[:5]})
    assert "audio-lyric agreement" in {item["descriptor"] for item in consistent_descriptors}

    assert not bool(selected_profile.loc["X1", "main_text_eligible"])
    assert bool(selected_profile.loc["X1", "explicit_title_flag"])
    assert summary["negative_margin_count"] >= 1
    assert sanity["explicit_or_encoding_issue_count"] >= 1
    assert "Region prototype songs" in report
    assert "Tension case-study songs" in report
    assert "Boundary / ambiguity cases" in report
    assert "Appendix-only candidates" in report
    assert "For each candidate region k" in report
