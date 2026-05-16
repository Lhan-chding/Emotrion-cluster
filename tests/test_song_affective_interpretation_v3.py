import json

import pandas as pd

from scripts.build_song_affective_interpretation_v3 import run_interpretation_v3


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_row(song_id, cluster_id, cluster_name, **overrides):
    weights = {
        "w_region_C0": 0.02,
        "w_region_C1": 0.02,
        "w_region_C2": 0.02,
        "w_region_C3": 0.02,
    }
    weights[f"w_region_C{cluster_id}"] = overrides.pop("assigned_weight", 0.88)
    nearest_alt_cluster = overrides.pop("nearest_alt_cluster", "C1 Gentle Warmth")
    nearest_alt_id = int(nearest_alt_cluster.split()[0].replace("C", ""))
    weights[f"w_region_C{nearest_alt_id}"] = overrides.pop("nearest_alt_weight", 0.10)
    row = {
        "song_id": song_id,
        "title": f"Title {song_id}",
        "artist": "Artist",
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "region_typicality": 0.86,
        "region_confidence": 0.91,
        "region_margin": 0.75,
        "nearest_alt_cluster": nearest_alt_cluster,
        "nearest_alt_cluster_weight": weights[f"w_region_C{nearest_alt_id}"],
        "tension_label": f"C{cluster_id}-T0",
        "tension_name": "modality-consistent",
        "tension_dv": 0.0,
        "tension_da": 0.0,
        "tension_strength_percentile": 0.12,
        "selected_role": "region_prototype",
        "descriptor_conflict_flag": False,
        "encoding_issue_flag": False,
        "explicit_title_flag": False,
        "top_descriptors_json": json.dumps([{"descriptor": "audio-lyric agreement", "weight": 1.0}]),
        **weights,
    }
    row.update(overrides)
    return row


def test_interpretation_v3_writes_rule_outputs_and_quality_gates(tmp_path):
    all_rows = [
        _base_row(
            "PROTO",
            0,
            "Subdued Melancholy",
            region_typicality=0.90,
            tension_strength_percentile=0.10,
            nearest_alt_cluster="C1 Gentle Warmth",
        ),
        _base_row(
            "BOUNDARY",
            3,
            "Playful Vitality",
            assigned_weight=0.52,
            nearest_alt_weight=0.45,
            region_typicality=0.48,
            region_confidence=0.52,
            region_margin=-0.02,
            nearest_alt_cluster="C1 Gentle Warmth",
            tension_name="lyric-darkened",
            tension_dv=-0.30,
            tension_da=-0.12,
            tension_strength_percentile=0.83,
            selected_role="boundary_case",
        ),
        _base_row(
            "MILD",
            2,
            "Volatile Intensity",
            assigned_weight=0.70,
            nearest_alt_weight=0.24,
            region_typicality=0.66,
            nearest_alt_cluster="C0 Subdued Melancholy",
            tension_name="lyric-brightened",
            tension_dv=0.32,
            tension_da=-0.06,
            tension_strength_percentile=0.72,
            selected_role="tension_case",
        ),
        _base_row(
            "LOW_DIR",
            1,
            "Gentle Warmth",
            region_typicality=0.88,
            nearest_alt_cluster="C0 Subdued Melancholy",
            tension_name="lyric-brightened",
            tension_dv=0.55,
            tension_da=0.33,
            tension_strength_percentile=0.20,
            selected_role="region_prototype",
        ),
        _base_row(
            "FLAGGED",
            0,
            "Subdued Melancholy",
            nearest_alt_cluster="C1 Gentle Warmth",
            explicit_title_flag=True,
            selected_role="region_prototype",
        ),
    ]
    profile_all = tmp_path / "song_affective_profile_all_v2.csv"
    profile_selected = tmp_path / "song_affective_profile_selected_v2.csv"
    descriptor_json = tmp_path / "descriptor_weights_selected_v2.json"
    _write_csv(profile_all, all_rows)
    _write_csv(profile_selected, [all_rows[0], all_rows[1], all_rows[2]])
    descriptor_json.write_text(
        json.dumps(
            {
                "PROTO": {"descriptors": [{"descriptor": "subdued melancholy", "weight": 1.0}]},
                "BOUNDARY": {"descriptors": [{"descriptor": "bright vitality", "weight": 0.9}]},
            }
        ),
        encoding="utf-8",
    )

    sanity = run_interpretation_v3(
        profile_all_csv=profile_all,
        profile_selected_csv=profile_selected,
        descriptor_json=descriptor_json,
        out_dir=tmp_path / "out",
    )

    out_dir = tmp_path / "out"
    all_v3 = pd.read_csv(out_dir / "song_affective_interpretation_all_v3.csv").set_index("song_id")
    selected_v3 = pd.read_csv(out_dir / "song_affective_interpretation_selected_v3.csv").set_index("song_id")
    rules = json.loads((out_dir / "interpretation_rules_v3.json").read_text(encoding="utf-8"))
    report = (out_dir / "song_affective_interpretation_report_v3.md").read_text(encoding="utf-8-sig")
    sanity_file = json.loads((out_dir / "sanity_check_interpretation_v3.json").read_text(encoding="utf-8"))

    assert sanity == sanity_file
    assert rules["post_hoc_only"] is True
    assert all_v3.loc["PROTO", "final_interpretation_label"] == "Concordant Subdued Melancholy"
    assert all_v3.loc["PROTO", "geometry_role"] == "prototype"
    assert all_v3.loc["BOUNDARY", "region_mixture_type"] == "boundary_blend"
    assert all_v3.loc["BOUNDARY", "geometry_role"] == "boundary"
    assert all_v3.loc["BOUNDARY", "final_interpretation_label"] == "Playful Vitality / Gentle Warmth Boundary Blend"
    assert bool(selected_v3.loc["BOUNDARY", "main_text_interpretation_eligible"])
    assert all_v3.loc["MILD", "secondary_affect"] == "with a mild Subdued Melancholy undertone"
    assert all_v3.loc["MILD", "tension_overlay"] == "Valence-Reframed Intensity"
    assert all_v3.loc["MILD", "cross_modal_relation"] == "lyric_valence_uplift; lyric_arousal_softening"
    assert all_v3.loc["LOW_DIR", "cross_modal_relation"] == "affective_concordance"
    assert "Lyrical" not in all_v3.loc["LOW_DIR", "final_interpretation_label"]
    assert not bool(all_v3.loc["FLAGGED", "main_text_interpretation_eligible"])
    assert sanity["total_songs"] == 5
    assert sanity["boundary_blend_count"] == 1
    assert sanity["low_tension_directional_violation_count"] == 0
    assert sanity["main_text_interpretation_eligible_count"] == 4
    assert "## Cluster Affect Lexicon" in report
    assert "## Selected Songs Interpretation Table" in report
    assert "BOUNDARY" in report


def test_interpretation_v3_detects_missing_required_fields(tmp_path):
    profile_all = tmp_path / "all.csv"
    profile_selected = tmp_path / "selected.csv"
    descriptor_json = tmp_path / "descriptor.json"
    _write_csv(profile_all, [{"song_id": "BROKEN", "cluster_id": 0}])
    _write_csv(profile_selected, [{"song_id": "BROKEN", "cluster_id": 0}])
    descriptor_json.write_text("{}", encoding="utf-8")

    try:
        run_interpretation_v3(
            profile_all_csv=profile_all,
            profile_selected_csv=profile_selected,
            descriptor_json=descriptor_json,
            out_dir=tmp_path / "out",
        )
    except ValueError as exc:
        assert "missing required v2 fields" in str(exc)
        assert "w_region_C0" in str(exc)
    else:
        raise AssertionError("Expected missing-field validation to fail.")


def test_selected_boundary_role_can_come_from_selected_profile(tmp_path):
    boundary = _base_row(
        "BOUNDARY_ONLY_SELECTED",
        3,
        "Playful Vitality",
        assigned_weight=0.50,
        nearest_alt_weight=0.46,
        region_margin=-0.01,
        nearest_alt_cluster="C1 Gentle Warmth",
    )
    boundary.pop("selected_role")
    selected_boundary = dict(boundary)
    selected_boundary["selected_role"] = "boundary_case"

    profile_all = tmp_path / "all.csv"
    profile_selected = tmp_path / "selected.csv"
    descriptor_json = tmp_path / "descriptor.json"
    _write_csv(profile_all, [boundary])
    _write_csv(profile_selected, [selected_boundary])
    descriptor_json.write_text("{}", encoding="utf-8")

    run_interpretation_v3(
        profile_all_csv=profile_all,
        profile_selected_csv=profile_selected,
        descriptor_json=descriptor_json,
        out_dir=tmp_path / "out",
    )

    all_v3 = pd.read_csv(tmp_path / "out" / "song_affective_interpretation_all_v3.csv").set_index("song_id")
    selected_v3 = pd.read_csv(tmp_path / "out" / "song_affective_interpretation_selected_v3.csv").set_index("song_id")

    assert not bool(all_v3.loc["BOUNDARY_ONLY_SELECTED", "main_text_interpretation_eligible"])
    assert bool(selected_v3.loc["BOUNDARY_ONLY_SELECTED", "main_text_interpretation_eligible"])
    assert selected_v3.loc["BOUNDARY_ONLY_SELECTED", "selected_role"] == "boundary_case"
