import math

import numpy as np

from cluster.features.va_geometry import (
    BALANCED_VA_DIFF_DIM,
    BALANCED_VA_DIFF_FEATURE_NAMES,
    VA_GEOMETRY_FEATURE_NAMES,
    VA_GEOMETRY_OBSERVED_NAMES,
    VA_GEOMETRY_OBSERVED_DIM,
    build_balanced_va_diff_features,
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


def test_balanced_va_diff_features_keep_both_modal_va_and_disagreement():
    audio = np.asarray([[0.8, 0.6]], dtype=np.float32)
    lyrics = np.asarray([[0.2, 0.7]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)

    features = build_balanced_va_diff_features(audio, lyrics, view_mask)

    assert BALANCED_VA_DIFF_DIM == 18
    assert BALANCED_VA_DIFF_FEATURE_NAMES[:6] == [
        "consensus_valence",
        "consensus_arousal",
        "audio_valence",
        "audio_arousal",
        "lyrics_valence",
        "lyrics_arousal",
    ]
    np.testing.assert_allclose(features[:, :2], np.asarray([[0.5, 0.65]], dtype=np.float32))
    np.testing.assert_allclose(features[:, 2:4], audio)
    np.testing.assert_allclose(features[:, 4:6], lyrics)
    np.testing.assert_allclose(features[:, 6:8], np.asarray([[0.6, -0.1]], dtype=np.float32), atol=1e-6)


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


def test_impute_unobserved_pairwise_makes_missing_rows_neutral():
    """Unobserved rows get per-row fill from nearest observed neighbours."""
    from cluster.features.va_geometry import impute_unobserved_pairwise

    audio = np.asarray([[0.7, 0.5], [0.7, 0.5], [0.3, 0.8], [0.5, 0.3]], dtype=np.float32)
    lyrics = np.asarray([[0.3, 0.6], [0.0, 0.0], [0.4, 0.2], [0.0, 0.0]], dtype=np.float32)
    view_mask = np.asarray([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ], dtype=np.float32)

    observed_features = build_va_geometry_observed_features(audio, lyrics, view_mask)
    # Before imputation: rows 1,3 have zeros in dims 2:14
    np.testing.assert_allclose(observed_features[1, 2:14], np.zeros(12))
    np.testing.assert_allclose(observed_features[3, 2:14], np.zeros(12))

    imputed, fill = impute_unobserved_pairwise(observed_features, view_mask)

    # Fill is the average of per-row kNN fills — not zero
    assert not np.allclose(fill, np.zeros(12))

    # Unobserved rows: dims 2:14 should no longer be zero
    for uidx in [1, 3]:
        assert not np.allclose(imputed[uidx, 2:14], np.zeros(12)), f"row {uidx} not imputed"

    # Observed rows unchanged
    np.testing.assert_allclose(imputed[0], observed_features[0], atol=1e-6)
    np.testing.assert_allclose(imputed[2], observed_features[2], atol=1e-6)


def test_impute_unobserved_pairwise_reuses_fitted_fill():
    """Transform mode: fitted_fill reuses pre-computed values."""
    from cluster.features.va_geometry import impute_unobserved_pairwise

    features = np.zeros((2, 14), dtype=np.float32)
    features[0, :2] = [0.5, 0.5]
    features[1, :2] = [0.6, 0.4]
    view_mask = np.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)

    preset_fill = np.ones(12, dtype=np.float32) * 0.42
    imputed, fill = impute_unobserved_pairwise(features, view_mask, fitted_fill=preset_fill)

    np.testing.assert_allclose(fill, preset_fill)
    np.testing.assert_allclose(imputed[0, 2:14], preset_fill, atol=1e-6)
    np.testing.assert_allclose(imputed[1, 2:14], preset_fill, atol=1e-6)
    np.testing.assert_allclose(imputed[0, :2], [0.5, 0.5])
    np.testing.assert_allclose(imputed[1, :2], [0.6, 0.4])
