import math

import numpy as np

from cluster.features.va_geometry import (
    VA_GEOMETRY_FEATURE_NAMES,
    VA_GEOMETRY_OBSERVED_NAMES,
    VA_GEOMETRY_OBSERVED_DIM,
    build_va_geometry_features,
    build_va_geometry_observed_features,
    build_va_geometry_mask,
)


def test_va_geometry_features_include_consensus_and_circumplex_disagreement():
    audio = np.asarray([[0.8, 0.6]], dtype=np.float32)
    lyrics = np.asarray([[0.2, 0.7]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)

    features = build_va_geometry_features(audio, lyrics, view_mask)

    audio_centered = audio[0] - np.asarray([0.5, 0.5], dtype=np.float32)
    lyrics_centered = lyrics[0] - np.asarray([0.5, 0.5], dtype=np.float32)
    delta = audio[0] - lyrics[0]
    abs_delta = np.abs(delta)
    euclidean_gap = np.linalg.norm(delta) / math.sqrt(2.0)
    manhattan_gap = float(abs_delta.mean())
    audio_radius = float(np.linalg.norm(audio_centered))
    lyrics_radius = float(np.linalg.norm(lyrics_centered))
    dot = float(np.dot(audio_centered, lyrics_centered))
    det = float(audio_centered[0] * lyrics_centered[1] - audio_centered[1] * lyrics_centered[0])
    cosine = dot / max(audio_radius * lyrics_radius, 1e-6)
    angular_gap = math.atan2(det, dot) / math.pi
    rbf_consistency = math.exp(-0.5 * (euclidean_gap / 0.35) ** 2)

    expected = np.asarray(
        [
            0.5,
            0.65,
            0.6,
            -0.1,
            0.6,
            0.1,
            euclidean_gap,
            manhattan_gap,
            cosine,
            angular_gap,
            audio_radius,
            lyrics_radius,
            audio_radius - lyrics_radius,
            rbf_consistency,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    assert VA_GEOMETRY_FEATURE_NAMES == [
        "mean_valence",
        "mean_arousal",
        "signed_delta_valence",
        "signed_delta_arousal",
        "abs_delta_valence",
        "abs_delta_arousal",
        "euclidean_gap",
        "manhattan_gap",
        "cosine_centered",
        "signed_angular_gap",
        "audio_radius",
        "lyrics_radius",
        "radial_gap",
        "rbf_consistency",
        "has_both_audio_lyrics",
        "has_audio",
        "has_lyrics",
    ]
    np.testing.assert_allclose(features[0], expected, atol=1e-6)


def test_va_geometry_features_treat_missing_pair_conflict_as_unknown_not_agreement():
    audio = np.asarray([[0.7, 0.4], [0.0, 0.0]], dtype=np.float32)
    lyrics = np.asarray([[0.0, 0.0], [0.2, 0.8]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    features = build_va_geometry_features(audio, lyrics, view_mask)

    np.testing.assert_allclose(features[:, :2], np.asarray([[0.7, 0.4], [0.2, 0.8]], dtype=np.float32))
    np.testing.assert_allclose(features[:, 2:14], np.zeros((2, 12), dtype=np.float32))
    np.testing.assert_allclose(features[:, 14:17], np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32))


def test_va_geometry_observed_features_excludes_mask():
    audio = np.asarray([[0.8, 0.6], [0.7, 0.4]], dtype=np.float32)
    lyrics = np.asarray([[0.2, 0.7], [0.0, 0.0]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)

    observed = build_va_geometry_observed_features(audio, lyrics, view_mask)
    assert observed.shape == (2, VA_GEOMETRY_OBSERVED_DIM)
    assert VA_GEOMETRY_OBSERVED_DIM == 14
    assert len(VA_GEOMETRY_OBSERVED_NAMES) == 14

    full = build_va_geometry_features(audio, lyrics, view_mask)
    np.testing.assert_allclose(observed, full[:, :14])


def test_va_geometry_mask_returns_correct_patterns():
    view_mask = np.asarray([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mask = build_va_geometry_mask(view_mask)
    assert mask.shape == (3, 3)
    expected = np.asarray([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(mask, expected)
