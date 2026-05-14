import json

import numpy as np
import pandas as pd

from cluster.preprocessing.prepare_unimodal_dataset import build_parser, prepare_unimodal_dataset
from cluster.data.metadata import build_canonical_metadata, build_metadata_features


def test_prepare_unimodal_dataset_writes_va_order_masks_and_manifest(tmp_path):
    combined_csv = tmp_path / "multimodal_va.csv"
    pd.DataFrame(
        [
            {
                "track_id": "song-a",
                "Audio_Arousal": 0.8,
                "Audio_Valence": 0.2,
                "Lyrics_Arousal": 0.7,
                "Lyrics_Valence": 0.3,
                "Original_Arousal": 0.9,
                "Original_Valence": 0.1,
                "quadrant": "Q1",
                "Genres": "Rock, Pop",
                "Moods": "Energetic",
            },
            {
                "track_id": "song-b",
                "Audio_Arousal": 0.4,
                "Audio_Valence": 0.6,
                "Lyrics_Arousal": np.nan,
                "Lyrics_Valence": np.nan,
                "Original_Arousal": 0.5,
                "Original_Valence": 0.55,
                "quadrant": "Q2",
                "Genres": "Pop",
                "Moods": "",
            },
            {
                "track_id": "song-c",
                "Audio_Arousal": np.nan,
                "Audio_Valence": np.nan,
                "Lyrics_Arousal": 0.1,
                "Lyrics_Valence": 0.9,
                "Original_Arousal": np.nan,
                "Original_Valence": np.nan,
                "quadrant": "Q3",
                "Genres": "",
                "Moods": "",
            },
        ]
    ).to_csv(combined_csv, index=False)

    out_dir = tmp_path / "processed"
    aligned_root = tmp_path / "aligned"
    result = prepare_unimodal_dataset(
        combined_csv=str(combined_csv),
        out_processed_dir=str(out_dir),
        out_aligned_root=str(aligned_root),
        seed=123,
    )

    assert result["num_samples"] == 3
    np.testing.assert_allclose(
        np.load(out_dir / "audio.npy"),
        np.asarray([[0.2, 0.8], [0.6, 0.4], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.load(out_dir / "lyrics.npy"),
        np.asarray([[0.3, 0.7], [0.0, 0.0], [0.9, 0.1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.load(out_dir / "view_mask.npy"),
        np.asarray([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.load(out_dir / "labels_emotion.npy"),
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    np.testing.assert_allclose(
        np.load(out_dir / "original_va.npy")[:2],
        np.asarray([[0.1, 0.9], [0.55, 0.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.load(out_dir / "signed_va_diff.npy"),
        np.asarray([[-0.1, 0.1], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        atol=1e-6,
    )

    manifest = pd.read_csv(out_dir / "dataset_manifest.csv")
    assert manifest[["track_id", "has_audio", "has_lyrics", "has_metadata"]].to_dict("records") == [
        {"track_id": "song-a", "has_audio": True, "has_lyrics": True, "has_metadata": True},
        {"track_id": "song-b", "has_audio": True, "has_lyrics": False, "has_metadata": True},
        {"track_id": "song-c", "has_audio": False, "has_lyrics": True, "has_metadata": False},
    ]
    assert {"Genres", "Moods"}.issubset(manifest.columns)
    canonical = pd.read_csv(out_dir / "canonical_metadata.csv")
    assert {"Audio_Song", "Lyric_Song", "Quadrant", "Genres", "Moods"}.issubset(canonical.columns)
    assert canonical["Audio_Song"].tolist() == ["song-a", "song-b", "song-c"]

    split_payload = json.loads((out_dir / "split_70_15_15.json").read_text(encoding="utf-8"))
    split_ids = [
        track_id
        for split in ("train", "val", "test")
        for track_id in split_payload["splits"][split]["track_ids"]
    ]
    assert sorted(split_ids) == ["song-a", "song-b", "song-c"]

    schema = json.loads((out_dir / "schema.json").read_text(encoding="utf-8"))
    assert schema["va_order"] == ["Valence", "Arousal"]
    assert schema["view_mask_columns"] == ["has_audio", "has_lyrics", "has_metadata"]
    assert "signed_va_diff.npy" in schema["derived_feature_files"]
    assert schema["schema_hash"]


def test_prepare_cli_accepts_dataset_l_prompt_aliases(tmp_path):
    args = build_parser().parse_args(
        [
            "--input_csv",
            str(tmp_path / "dataset_l.csv"),
            "--output_dir",
            str(tmp_path / "processed"),
            "--dataset_name",
            "Dataset-L-strict",
            "--va_order",
            "valence_arousal",
            "--metadata_policy",
            "report_only",
            "--split_protocol",
            "70_15_15",
        ]
    )

    assert args.combined_csv.endswith("dataset_l.csv")
    assert args.out_processed_dir.endswith("processed")
    assert args.dataset_version == "Dataset-L-strict"
    assert args.metadata_policy == "report_only"
    assert args.split_protocol == "70_15_15"


def test_placeholder_aligned_metadata_supports_no_meta_baseline(tmp_path):
    combined_csv = tmp_path / "multimodal_va.csv"
    pd.DataFrame(
        [
            {
                "Song": "A001",
                "Audio_Arousal": 0.35,
                "Audio_Valence": 0.77,
                "Lyrics_Arousal": 0.37,
                "Lyrics_Valence": 0.57,
                "Original_Arousal": 0.22,
                "Original_Valence": 0.87,
                "Lyrics": "hello world",
            },
            {
                "Song": "A002",
                "Audio_Arousal": 0.40,
                "Audio_Valence": 0.58,
                "Lyrics_Arousal": 0.32,
                "Lyrics_Valence": 0.48,
                "Original_Arousal": 0.37,
                "Original_Valence": 0.71,
                "Lyrics": "another song",
            },
        ]
    ).to_csv(combined_csv, index=False)

    out_dir = tmp_path / "processed"
    aligned_root = tmp_path / "aligned"
    prepare_unimodal_dataset(
        combined_csv=str(combined_csv),
        out_processed_dir=str(out_dir),
        out_aligned_root=str(aligned_root),
        seed=42,
    )

    canonical = build_canonical_metadata(str(aligned_root), str(out_dir))
    bundle = build_metadata_features(canonical, min_token_freq=3, max_tokens_per_field=128)

    assert canonical["Artist"].tolist() == ["", ""]
    assert canonical["Title"].tolist() == ["", ""]
    assert canonical["Genres"].tolist() == ["", ""]
    assert canonical["MoodsAll"].tolist() == ["", ""]
    assert bundle.features.shape[0] == 2


def test_prepare_unimodal_dataset_derives_quadrants_from_original_va_when_label_missing(tmp_path):
    combined_csv = tmp_path / "multimodal_va.csv"
    pd.DataFrame(
        [
            {
                "Song": "q1",
                "Audio_Arousal": 0.8,
                "Audio_Valence": 0.8,
                "Lyrics_Arousal": 0.8,
                "Lyrics_Valence": 0.8,
                "Original_Arousal": 0.9,
                "Original_Valence": 0.9,
            },
            {
                "Song": "q2",
                "Audio_Arousal": 0.8,
                "Audio_Valence": 0.2,
                "Lyrics_Arousal": 0.8,
                "Lyrics_Valence": 0.2,
                "Original_Arousal": 0.9,
                "Original_Valence": 0.1,
            },
            {
                "Song": "q3",
                "Audio_Arousal": 0.2,
                "Audio_Valence": 0.2,
                "Lyrics_Arousal": 0.2,
                "Lyrics_Valence": 0.2,
                "Original_Arousal": 0.1,
                "Original_Valence": 0.1,
            },
            {
                "Song": "q4",
                "Audio_Arousal": 0.2,
                "Audio_Valence": 0.8,
                "Lyrics_Arousal": 0.2,
                "Lyrics_Valence": 0.8,
                "Original_Arousal": 0.1,
                "Original_Valence": 0.9,
            },
        ]
    ).to_csv(combined_csv, index=False)

    out_dir = tmp_path / "processed"
    aligned_root = tmp_path / "aligned"
    prepare_unimodal_dataset(
        combined_csv=str(combined_csv),
        out_processed_dir=str(out_dir),
        out_aligned_root=str(aligned_root),
        seed=42,
    )

    np.testing.assert_array_equal(np.load(out_dir / "labels_emotion.npy"), np.asarray([0, 1, 2, 3]))
    manifest = pd.read_csv(out_dir / "dataset_manifest.csv")
    assert manifest["quadrant"].tolist() == ["Q1", "Q2", "Q3", "Q4"]
    audio_aligned = pd.read_csv(aligned_root / "aligned_audio_metadata.csv")
    assert audio_aligned["Quadrant"].tolist() == ["Q1", "Q2", "Q3", "Q4"]


def test_prepare_unimodal_dataset_writes_tfidf_svd_and_excludes_artist_by_default(tmp_path):
    combined_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"track_id": "a", "Audio_Valence": 0.8, "Audio_Arousal": 0.7, "Lyrics_Valence": 0.7, "Lyrics_Arousal": 0.6, "Genres": "Rock", "MoodsAll": "Aggressive, Energetic", "Themes": "Action", "Styles": "Alt", "Artist": "Artist A"},
            {"track_id": "b", "Audio_Valence": 0.2, "Audio_Arousal": 0.8, "Lyrics_Valence": 0.3, "Lyrics_Arousal": 0.7, "Genres": "Rock", "MoodsAll": "Aggressive", "Themes": "Conflict", "Styles": "Metal", "Artist": "Artist B"},
            {"track_id": "c", "Audio_Valence": 0.6, "Audio_Arousal": 0.3, "Lyrics_Valence": 0.5, "Lyrics_Arousal": 0.4, "Genres": "Folk", "MoodsAll": "Reflective, Warm", "Themes": "Love", "Styles": "Acoustic", "Artist": "Artist C"},
            {"track_id": "d", "Audio_Valence": 0.4, "Audio_Arousal": 0.2, "Lyrics_Valence": 0.4, "Lyrics_Arousal": 0.3, "Genres": "Folk", "MoodsAll": "Reflective", "Themes": "Memory", "Styles": "Acoustic", "Artist": "Artist D"},
        ]
    ).to_csv(combined_csv, index=False)

    out_dir = tmp_path / "processed"
    prepare_unimodal_dataset(
        combined_csv=str(combined_csv),
        out_processed_dir=str(out_dir),
        metadata_use_artist=False,
        metadata_representation="tfidf_svd",
        metadata_svd_dim=2,
        metadata_group_weights="Genres=0.25,Styles=0.35,Themes=0.50,MoodsAll=0.70,Artist=0.00",
    )

    metadata = np.load(out_dir / "metadata.npy")
    metadata_binary = np.load(out_dir / "metadata_binary.npy")
    metadata_tfidf = np.load(out_dir / "metadata_tfidf.npy")
    metadata_svd = np.load(out_dir / "metadata_svd.npy")
    assert metadata.shape == (4, 2)
    np.testing.assert_allclose(metadata, metadata_svd)
    assert metadata_binary.shape == metadata_tfidf.shape

    binary_names = json.loads((out_dir / "metadata_binary_feature_names.json").read_text(encoding="utf-8"))
    assert any(name.startswith("MoodsAll::") for name in binary_names)
    assert not any(name.startswith("Artist::") for name in binary_names)

    feature_names = json.loads((out_dir / "metadata_feature_names.json").read_text(encoding="utf-8"))
    assert feature_names == ["metadata_svd::000", "metadata_svd::001"]

    groups = json.loads((out_dir / "metadata_binary_feature_groups.json").read_text(encoding="utf-8"))
    aggressive = next(item for item in groups if item["feature"] == "MoodsAll::aggressive")
    assert aggressive["group"] == "MoodsAll"
    assert aggressive["weight"] == 0.70


def test_prepare_unimodal_dataset_can_include_artist_for_ablation(tmp_path):
    combined_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"track_id": "a", "Audio_Valence": 0.8, "Audio_Arousal": 0.7, "Lyrics_Valence": 0.7, "Lyrics_Arousal": 0.6, "Artist": "Artist A"},
            {"track_id": "b", "Audio_Valence": 0.2, "Audio_Arousal": 0.8, "Lyrics_Valence": 0.3, "Lyrics_Arousal": 0.7, "Artist": "Artist B"},
        ]
    ).to_csv(combined_csv, index=False)

    out_dir = tmp_path / "processed"
    prepare_unimodal_dataset(
        combined_csv=str(combined_csv),
        out_processed_dir=str(out_dir),
        metadata_use_artist=True,
        metadata_representation="binary",
    )

    feature_names = json.loads((out_dir / "metadata_feature_names.json").read_text(encoding="utf-8"))
    assert "Artist::artist a" in feature_names
    assert "Artist::artist b" in feature_names
