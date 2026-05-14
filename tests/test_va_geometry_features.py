import math

import numpy as np

from cluster.features.affect_calibration import BalanceAlphaLearner
from cluster.features.va_geometry import (
    BALANCED_VA_DIFF_DIM,
    BALANCED_VA_DIFF_FEATURE_NAMES,
    CALIBRATED_VA_TENSION_DIM,
    CALIBRATED_VA_TENSION_FEATURE_NAMES,
    VA_GEOMETRY_FEATURE_NAMES,
    VA_GEOMETRY_OBSERVED_NAMES,
    VA_GEOMETRY_OBSERVED_DIM,
    build_balanced_va_diff_features,
    build_calibrated_va_tension_features,
    build_va_geometry_features,
    build_va_geometry_observed_features,
    build_va_geometry_mask,
)


def test_balance_alpha_learner_prefers_clusterable_audio_heavy_consensus():
    rng = np.random.default_rng(123)
    audio = np.vstack(
        [
            rng.normal([0.20, 0.25], 0.02, size=(40, 2)),
            rng.normal([0.80, 0.75], 0.02, size=(40, 2)),
        ]
    ).astype(np.float32)
    lyrics = rng.normal([0.50, 0.50], 0.16, size=(80, 2)).astype(np.float32)
    mask = np.ones((80, 3), dtype=np.float32)

    learner = BalanceAlphaLearner(
        alpha_min=0.20,
        alpha_max=0.90,
        alpha_step=0.10,
        search_k_min=2,
        search_k_max=2,
        random_state=7,
    ).fit(audio, lyrics, mask)

    assert learner.alpha_ is not None
    assert learner.alpha_ > 0.60
    consensus = learner.transform(audio, lyrics, mask)
    expected = float(learner.alpha_) * audio + (1.0 - float(learner.alpha_)) * lyrics
    np.testing.assert_allclose(consensus, expected, atol=1e-6)


def test_balance_alpha_learner_transform_handles_missing_views():
    learner = BalanceAlphaLearner(mode="global_alpha", alpha_=0.75)
    audio = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    lyrics = np.asarray([[0.2, 0.8], [0.0, 0.0], [0.1, 0.9], [0.0, 0.0]], dtype=np.float32)
    mask = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    consensus = learner.transform(audio, lyrics, mask)

    np.testing.assert_allclose(consensus[0], np.asarray([0.65, 0.35], dtype=np.float32))
    np.testing.assert_allclose(consensus[1], audio[1])
    np.testing.assert_allclose(consensus[2], lyrics[2])
    np.testing.assert_allclose(consensus[3], np.asarray([0.5, 0.5], dtype=np.float32))


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


def test_balanced_va_diff_features_encode_modal_va_without_raw_duplication():
    audio = np.asarray([[0.8, 0.6]], dtype=np.float32)
    lyrics = np.asarray([[0.2, 0.7]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)

    features = build_balanced_va_diff_features(audio, lyrics, view_mask)

    assert BALANCED_VA_DIFF_DIM == 8
    assert BALANCED_VA_DIFF_FEATURE_NAMES == [
        "consensus_valence",
        "consensus_arousal",
        "signed_delta_valence",
        "signed_delta_arousal",
        "euclidean_gap",
        "signed_angular_gap",
        "radial_gap",
        "rbf_consistency",
    ]
    np.testing.assert_allclose(features[:, :2], np.asarray([[0.5, 0.65]], dtype=np.float32))
    np.testing.assert_allclose(features[:, 2:4], np.asarray([[0.6, -0.1]], dtype=np.float32), atol=1e-6)

    reconstructed_audio = features[:, :2] + 0.5 * features[:, 2:4]
    reconstructed_lyrics = features[:, :2] - 0.5 * features[:, 2:4]
    np.testing.assert_allclose(reconstructed_audio, audio, atol=1e-6)
    np.testing.assert_allclose(reconstructed_lyrics, lyrics, atol=1e-6)


def test_calibrated_va_tension_features_reuse_fit_sigma_on_transform():
    train_audio = np.asarray(
        [
            [0.2, 0.2],
            [0.4, 0.4],
            [0.6, 0.6],
            [0.8, 0.8],
        ],
        dtype=np.float32,
    )
    train_lyrics = train_audio + np.asarray(
        [
            [0.02, 0.01],
            [0.04, 0.02],
            [0.08, 0.04],
            [0.16, 0.08],
        ],
        dtype=np.float32,
    )
    train_mask = np.ones((4, 3), dtype=np.float32)

    train_features, state = build_calibrated_va_tension_features(
        train_audio,
        train_lyrics,
        train_mask,
        fit=True,
        calibration_mode="identity",
        diff_residual_mode="identity",
    )

    eval_audio = np.asarray([[0.5, 0.5], [0.6, 0.6]], dtype=np.float32)
    eval_lyrics = eval_audio + np.asarray([[0.05, 0.025], [0.06, 0.03]], dtype=np.float32)
    eval_mask = np.ones((2, 3), dtype=np.float32)
    eval_features, reused_state = build_calibrated_va_tension_features(
        eval_audio,
        eval_lyrics,
        eval_mask,
        calibrator=state["calibrator"],
        residualizer=state["residualizer"],
        fit=False,
        calibration_mode="identity",
        diff_residual_mode="identity",
        fitted_sigma=(state["sigma_v"], state["sigma_a"]),
    )

    assert CALIBRATED_VA_TENSION_DIM == 5
    assert CALIBRATED_VA_TENSION_FEATURE_NAMES == [
        "consensus_valence",
        "consensus_arousal",
        "tension_dv",
        "tension_da",
        "tension_r",
    ]
    np.testing.assert_allclose(eval_features[:, 2], (eval_lyrics[:, 0] - eval_audio[:, 0]) / state["sigma_v"])
    np.testing.assert_allclose(eval_features[:, 3], (eval_lyrics[:, 1] - eval_audio[:, 1]) / state["sigma_a"])
    assert reused_state["sigma_v"] == state["sigma_v"]
    assert reused_state["sigma_a"] == state["sigma_a"]
    assert not np.allclose(train_features[:, 2:].std(axis=0), eval_features[:, 2:].std(axis=0))


def test_bias_neutral_mean_keeps_calibrated_mean_compatibility():
    audio = np.asarray([[0.2, 0.3], [0.8, 0.7]], dtype=np.float32)
    lyrics = np.asarray([[0.4, 0.5], [1.0, 0.9]], dtype=np.float32)
    mask = np.ones((2, 3), dtype=np.float32)

    legacy, legacy_state = build_calibrated_va_tension_features(
        audio,
        lyrics,
        mask,
        fit=True,
        consensus_mode="calibrated_mean",
        calibration_mode="global_median_shift",
        diff_residual_mode="identity",
    )
    renamed, renamed_state = build_calibrated_va_tension_features(
        audio,
        lyrics,
        mask,
        fit=True,
        consensus_mode="bias_neutral_mean",
        calibration_mode="global_median_shift",
        diff_residual_mode="identity",
    )

    np.testing.assert_allclose(legacy[:, :2], renamed[:, :2])
    assert legacy_state["consensus_mode"] == "bias_neutral_mean"
    assert renamed_state["consensus_mode"] == "bias_neutral_mean"
    np.testing.assert_allclose(legacy_state["balance_alpha"], 0.5)


def test_calibrated_va_tension_global_alpha_controls_consensus():
    audio = np.asarray([[0.9, 0.1], [0.6, 0.2]], dtype=np.float32)
    lyrics = np.asarray([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    mask = np.ones((2, 3), dtype=np.float32)

    features, state = build_calibrated_va_tension_features(
        audio,
        lyrics,
        mask,
        fit=True,
        calibration_mode="identity",
        diff_residual_mode="identity",
        consensus_mode="global_alpha",
        consensus_alpha=0.75,
    )

    expected = 0.75 * audio + 0.25 * lyrics
    np.testing.assert_allclose(features[:, :2], expected, atol=1e-6)
    assert state["balance_alpha"] == 0.75
    assert state["consensus_mode"] == "global_alpha"


def test_calibrated_va_tension_clusterability_alpha_reuses_fitted_alpha():
    rng = np.random.default_rng(456)
    train_audio = np.vstack(
        [
            rng.normal([0.25, 0.25], 0.02, size=(30, 2)),
            rng.normal([0.75, 0.75], 0.02, size=(30, 2)),
        ]
    ).astype(np.float32)
    train_lyrics = rng.normal([0.5, 0.5], 0.14, size=(60, 2)).astype(np.float32)
    train_mask = np.ones((60, 3), dtype=np.float32)

    train_features, state = build_calibrated_va_tension_features(
        train_audio,
        train_lyrics,
        train_mask,
        fit=True,
        calibration_mode="identity",
        diff_residual_mode="identity",
        consensus_mode="clusterability_alpha",
        alpha_search_min=0.20,
        alpha_search_max=0.90,
        alpha_search_step=0.10,
        alpha_search_k_min=2,
        alpha_search_k_max=2,
    )

    eval_audio = np.asarray([[0.9, 0.1]], dtype=np.float32)
    eval_lyrics = np.asarray([[0.1, 0.9]], dtype=np.float32)
    eval_mask = np.ones((1, 3), dtype=np.float32)
    eval_features, reused_state = build_calibrated_va_tension_features(
        eval_audio,
        eval_lyrics,
        eval_mask,
        calibrator=state["calibrator"],
        residualizer=state["residualizer"],
        balance_learner=state["balance_learner"],
        fit=False,
        calibration_mode="identity",
        diff_residual_mode="identity",
        consensus_mode="clusterability_alpha",
        fitted_sigma=(state["sigma_v"], state["sigma_a"]),
    )

    alpha = float(state["balance_alpha"])
    assert alpha > 0.60
    np.testing.assert_allclose(train_features[:, :2], alpha * train_audio + (1.0 - alpha) * train_lyrics)
    np.testing.assert_allclose(eval_features[:, :2], alpha * eval_audio + (1.0 - alpha) * eval_lyrics)
    assert reused_state["balance_alpha"] == state["balance_alpha"]


def test_calibrated_va_tension_passes_compute_settings_to_alpha_and_residualizer():
    rng = np.random.default_rng(789)
    audio = rng.uniform(0.15, 0.85, size=(24, 2)).astype(np.float32)
    lyrics = np.clip(audio + rng.normal(0.0, 0.04, size=(24, 2)), 0.0, 1.0).astype(np.float32)
    mask = np.ones((24, 3), dtype=np.float32)

    _, state = build_calibrated_va_tension_features(
        audio,
        lyrics,
        mask,
        fit=True,
        consensus_mode="clusterability_alpha",
        calibration_mode="identity",
        diff_residual_mode="knn",
        diff_residual_neighbors=5,
        alpha_search_k_min=2,
        alpha_search_k_max=2,
        compute_device="cuda:0",
        compute_chunk_size=17,
        compute_sample_size=12,
    )

    learner_state = state["balance_learner"].to_dict()
    residualizer_state = state["residualizer"].to_dict()
    assert learner_state["device"] == "cuda:0"
    assert learner_state["chunk_size"] == 17
    assert learner_state["score_sample_size"] == 12
    assert residualizer_state["device"] == "cuda:0"
    assert residualizer_state["chunk_size"] == 17


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
