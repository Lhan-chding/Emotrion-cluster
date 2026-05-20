import json

import pandas as pd

from scripts.build_external_review_labels import build_external_review_labels
from scripts.evaluate_affect_retrieval import evaluate_retrieval_results
from scripts.run_affect_retrieval_eval import run_affect_retrieval_eval


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _interpretation_row(song_id, title, artist, cluster_id, valence, arousal, **overrides):
    cluster_names = {
        0: "Subdued Melancholy",
        1: "Gentle Warmth",
        2: "Volatile Intensity",
        3: "Playful Vitality",
    }
    weights = {
        "w_region_C0": 0.02,
        "w_region_C1": 0.02,
        "w_region_C2": 0.02,
        "w_region_C3": 0.02,
    }
    weights[f"w_region_C{cluster_id}"] = overrides.pop("assigned_weight", 0.90)
    row = {
        "song_id": song_id,
        "title": title,
        "artist": artist,
        "cluster_id": cluster_id,
        "cluster_name": cluster_names[cluster_id],
        "balanced_valence": valence,
        "balanced_arousal": arousal,
        "region_typicality": overrides.pop("region_typicality", 0.85),
        "region_confidence": overrides.pop("region_confidence", 0.90),
        "region_margin": overrides.pop("region_margin", 0.70),
        "nearest_alt_cluster": overrides.pop("nearest_alt_cluster", "C1 Gentle Warmth"),
        "tension_label": overrides.pop("tension_label", f"C{cluster_id}-T0"),
        "tension_name": overrides.pop("tension_name", "modality-consistent"),
        "tension_dv": overrides.pop("tension_dv", 0.0),
        "tension_da": overrides.pop("tension_da", 0.0),
        "tension_norm": overrides.pop("tension_norm", 0.01),
        "affective_complexity_score": overrides.pop("affective_complexity_score", 0.20),
        "complexity_level": overrides.pop("complexity_level", "simple / concordant"),
        "final_interpretation_label": overrides.pop("final_interpretation_label", "Concordant region"),
        **weights,
    }
    row.update(overrides)
    return row


def test_run_affect_retrieval_eval_builds_outputs_without_external_labels(tmp_path):
    cluster_rows = [
        {"identifier": "VOL_HIGH", "cluster_id": 2, "balanced_v": 0.20, "balanced_a": 0.86},
        {"identifier": "VOL_LOW", "cluster_id": 2, "balanced_v": 0.21, "balanced_a": 0.84},
        {"identifier": "MEL_LIFT", "cluster_id": 0, "balanced_v": 0.18, "balanced_a": 0.22},
        {"identifier": "WARM", "cluster_id": 1, "balanced_v": 0.62, "balanced_a": 0.30},
        {"identifier": "BRIGHT_DARK", "cluster_id": 3, "balanced_v": 0.86, "balanced_a": 0.82},
        {"identifier": "PROTO", "cluster_id": 1, "balanced_v": 0.65, "balanced_a": 0.28},
    ]
    tension_rows = [
        {
            "identifier": "VOL_HIGH",
            "tension_label": "C2-T1",
            "tension_name": "lyric-intensified",
            "delta_v": 0.05,
            "delta_a": 0.90,
            "tension_norm": 0.90,
        },
        {
            "identifier": "VOL_LOW",
            "tension_label": "C2-T0",
            "tension_name": "modality-consistent",
            "delta_v": 0.01,
            "delta_a": 0.02,
            "tension_norm": 0.02,
        },
        {
            "identifier": "MEL_LIFT",
            "tension_label": "C0-T1",
            "tension_name": "lyric-brightened",
            "delta_v": 0.40,
            "delta_a": 0.30,
            "tension_norm": 0.50,
        },
        {
            "identifier": "WARM",
            "tension_label": "C1-T1",
            "tension_name": "lyric-brightened",
            "delta_v": 0.22,
            "delta_a": 0.20,
            "tension_norm": 0.30,
        },
        {
            "identifier": "BRIGHT_DARK",
            "tension_label": "C3-T1",
            "tension_name": "lyric-darkened",
            "delta_v": -0.35,
            "delta_a": -0.20,
            "tension_norm": 0.40,
        },
        {
            "identifier": "PROTO",
            "tension_label": "C1-T0",
            "tension_name": "modality-consistent",
            "delta_v": 0.00,
            "delta_a": 0.00,
            "tension_norm": 0.01,
        },
    ]
    interpretation_rows = [
        _interpretation_row(
            "VOL_HIGH",
            "High Volatile",
            "Artist A",
            2,
            0.20,
            0.86,
            tension_dv=0.05,
            tension_da=0.90,
            tension_norm=0.90,
            affective_complexity_score=0.91,
            complexity_level="highly complex / ambivalent",
            final_interpretation_label="Volatile Intensity with High Cross-modal Tension",
        ),
        _interpretation_row("VOL_LOW", "Low Volatile", "Artist B", 2, 0.21, 0.84),
        _interpretation_row(
            "MEL_LIFT",
            "Lifted Melancholy",
            "Artist C",
            0,
            0.18,
            0.22,
            tension_dv=0.40,
            tension_da=0.30,
            tension_norm=0.50,
            affective_complexity_score=0.65,
            complexity_level="complex",
        ),
        _interpretation_row("WARM", "Warm Song", "Artist D", 1, 0.62, 0.30),
        _interpretation_row(
            "BRIGHT_DARK",
            "Bright Dark",
            "Artist E",
            3,
            0.86,
            0.82,
            tension_dv=-0.35,
            tension_da=-0.20,
            tension_norm=0.40,
            affective_complexity_score=0.82,
            complexity_level="highly complex / ambivalent",
        ),
        _interpretation_row("PROTO", "Prototype", "Artist F", 1, 0.65, 0.28),
    ]

    cluster_csv = tmp_path / "dataset_s_cluster_assignments.csv"
    tension_csv = tmp_path / "dataset_s_tension_assignments.csv"
    interpretation_csv = tmp_path / "song_affective_interpretation_all_v3.csv"
    _write_csv(cluster_csv, cluster_rows)
    _write_csv(tension_csv, tension_rows)
    _write_csv(interpretation_csv, interpretation_rows)

    sanity = run_affect_retrieval_eval(
        cluster_csv=cluster_csv,
        tension_csv=tension_csv,
        interpretation_csv=interpretation_csv,
        out_dir=tmp_path / "out",
        top_k=(2, 3),
        retrieval_depth=3,
        make_figures=False,
        mirror_report_path=None,
    )

    out_dir = tmp_path / "out"
    features = pd.read_csv(out_dir / "song_retrieval_features.csv")
    rankings = pd.read_csv(out_dir / "retrieval_results_all.csv")
    annotation_pool = pd.read_csv(out_dir / "external_annotation_pool.csv")
    search_queries = pd.read_csv(out_dir / "source_search_queries.csv")
    sanity_file = json.loads((out_dir / "sanity_check_retrieval.json").read_text(encoding="utf-8"))

    assert sanity == sanity_file
    assert len(features) == 6
    assert features.set_index("song_id").loc["VOL_HIGH", "tension_strength_percentile"] == 1.0
    full_top = rankings[
        (rankings["query_id"] == "volatile_high_tension")
        & (rankings["system"] == "va_tension_complexity")
        & (rankings["rank"] == 1)
    ].iloc[0]
    assert full_top["song_id"] == "VOL_HIGH"
    assert "high cross-modal tension" in full_top["model_retrieval_reason"]
    assert annotation_pool["annotation_status"].eq("needs_external_review").all()
    assert search_queries["search_query"].str.contains("review", case=False).all()
    assert sanity["whether_any_external_label_used_in_scoring"] is False
    assert sanity["metrics_available"] is False


def test_external_review_labels_and_metrics_use_posthoc_evidence_only(tmp_path):
    pool = pd.DataFrame(
        [
            {
                "query_id": "volatile_high_tension",
                "song_id": "S_STRONG",
                "title": "Strong",
                "artist": "Artist",
                "cluster_name": "Volatile Intensity",
            },
            {
                "query_id": "volatile_high_tension",
                "song_id": "S_WEAK",
                "title": "Weak",
                "artist": "Artist",
                "cluster_name": "Volatile Intensity",
            },
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "song_id": "S_STRONG",
                "source_name": "Pitchfork",
                "source_url": "https://example.com/review",
                "external_summary": "Professional review describes aggression, political anger, tension, and harsh confrontational energy.",
            }
        ]
    )
    labels = build_external_review_labels(pool, evidence)

    strong = labels.set_index("song_id").loc["S_STRONG"]
    weak = labels.set_index("song_id").loc["S_WEAK"]
    assert strong["annotation_status"] == "verified"
    assert strong["overall_relevance_grade"] >= 2
    assert weak["annotation_status"] == "unverified"

    retrieval = pd.DataFrame(
        [
            {"query_id": "volatile_high_tension", "system": "va_only", "rank": 1, "song_id": "S_WEAK"},
            {"query_id": "volatile_high_tension", "system": "va_only", "rank": 2, "song_id": "S_STRONG"},
            {
                "query_id": "volatile_high_tension",
                "system": "va_tension_complexity",
                "rank": 1,
                "song_id": "S_STRONG",
            },
            {
                "query_id": "volatile_high_tension",
                "system": "va_tension_complexity",
                "rank": 2,
                "song_id": "S_WEAK",
            },
        ]
    )
    metrics, summary = evaluate_retrieval_results(retrieval, labels, top_k=(1,))
    assert metrics["metrics_available"].all()
    top1 = metrics.set_index(["query_id", "system", "k"]).loc[
        ("volatile_high_tension", "va_tension_complexity", 1)
    ]
    baseline_top1 = metrics.set_index(["query_id", "system", "k"]).loc[("volatile_high_tension", "va_only", 1)]
    assert top1["mean_relevance"] > baseline_top1["mean_relevance"]
    assert summary.set_index("system").loc["va_tension_complexity", "pairwise_win_rate_vs_va_only"] == 1.0
