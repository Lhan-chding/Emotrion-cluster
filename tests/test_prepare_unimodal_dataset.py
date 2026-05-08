import json

import numpy as np
import pandas as pd

from cluster.preprocessing.prepare_unimodal_dataset import prepare_unimodal_dataset
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
