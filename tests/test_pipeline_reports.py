import numpy as np
import pandas as pd

from cluster.pipeline.train import (
    _dataset_plot_va,
    _plot_cluster_feature_pca,
    _quadrant_heatmap_matrix,
    _cluster_summary,
    build_parser,
    apply_cluster_feature_weights,
    build_cluster_features,
    cluster_feature_block_mask,
    cluster_feature_weights,
    run_k_selection,
)


def test_quadrant_heatmap_matrix_reports_missing_labels():
    cluster_ids, heatmap, valid_total = _quadrant_heatmap_matrix(
        assignments=np.asarray([0, 0, 1, 1], dtype=np.int64),
        labels=np.asarray([-1, -1, -1, -1], dtype=np.int64),
    )

    assert cluster_ids == [0, 1]
    assert valid_total == 0
    np.testing.assert_allclose(heatmap, np.zeros((2, 4), dtype=np.float32))


def test_quadrant_heatmap_matrix_uses_only_valid_labels():
    cluster_ids, heatmap, valid_total = _quadrant_heatmap_matrix(
        assignments=np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
        labels=np.asarray([0, 0, -1, 2, 3], dtype=np.int64),
    )

    assert cluster_ids == [0, 1]
    assert valid_total == 4
    np.testing.assert_allclose(
        heatmap,
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )


def test_original_va_cluster_feature_strategy_uses_original_va_only():
    embeddings = {
        "z_fused": np.ones((2, 3), dtype=np.float32),
        "z_audio": np.ones((2, 3), dtype=np.float32) * 2.0,
        "z_lyrics": np.ones((2, 3), dtype=np.float32) * 3.0,
        "z_metadata": np.ones((2, 3), dtype=np.float32) * 4.0,
        "gate_weights": np.ones((2, 3), dtype=np.float32) / 3.0,
        "consistency": np.ones((2, 1), dtype=np.float32),
        "va_diff": np.zeros((2, 2), dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
        "original_va": np.asarray([[0.8, 0.7], [0.2, 0.3]], dtype=np.float32),
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="original_va",
    )

    np.testing.assert_allclose(features, embeddings["original_va"])
    assert pca is None


def test_mean_va_cluster_feature_strategy_uses_raw_mean_va_only():
    embeddings = {
        "z_fused": np.ones((2, 3), dtype=np.float32),
        "z_audio": np.ones((2, 3), dtype=np.float32) * 2.0,
        "z_lyrics": np.ones((2, 3), dtype=np.float32) * 3.0,
        "z_metadata": np.ones((2, 3), dtype=np.float32) * 4.0,
        "gate_weights": np.ones((2, 3), dtype=np.float32) / 3.0,
        "consistency": np.ones((2, 1), dtype=np.float32),
        "va_diff": np.zeros((2, 2), dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
        "mean_va": np.asarray([[0.6, 0.7], [0.2, 0.4]], dtype=np.float32),
        "original_va": np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="mean_va",
    )

    np.testing.assert_allclose(features, embeddings["mean_va"])
    assert pca is None


def test_mean_va_cluster_feature_strategy_accepts_minimal_raw_embeddings():
    embeddings = {
        "mean_va": np.asarray([[0.6, 0.7], [0.2, 0.4]], dtype=np.float32),
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="mean_va",
    )

    np.testing.assert_allclose(features, embeddings["mean_va"])
    assert pca is None


def test_va_geometry_cluster_feature_strategy_uses_geometry_embedding_only():
    embeddings = {
        "z_fused": np.ones((2, 3), dtype=np.float32),
        "z_audio": np.ones((2, 3), dtype=np.float32) * 2.0,
        "z_lyrics": np.ones((2, 3), dtype=np.float32) * 3.0,
        "z_metadata": np.ones((2, 3), dtype=np.float32) * 4.0,
        "gate_weights": np.ones((2, 3), dtype=np.float32) / 3.0,
        "consistency": np.ones((2, 1), dtype=np.float32),
        "va_diff": np.zeros((2, 2), dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
        "mean_va": np.asarray([[0.6, 0.7], [0.2, 0.4]], dtype=np.float32),
        "va_geometry": np.asarray(
            [
                [0.6, 0.7, 0.1, -0.2, 0.1, 0.2, 0.16, 0.15, 0.8, 0.2, 0.3, 0.4, -0.1, 0.9, 1.0, 1.0, 1.0],
                [0.2, 0.4, -0.3, 0.1, 0.3, 0.1, 0.22, 0.2, -0.4, -0.3, 0.5, 0.2, 0.3, 0.7, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "original_va": np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="va_geometry",
    )

    np.testing.assert_allclose(features, embeddings["va_geometry"][:, :14])
    assert pca is None


def test_va_geometry_cluster_feature_strategy_accepts_minimal_raw_embeddings():
    embeddings = {
        "va_geometry": np.zeros((2, 17), dtype=np.float32),
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="va_geometry",
    )

    np.testing.assert_allclose(features, embeddings["va_geometry"][:, :14])
    assert pca is None


def test_va_geometry_feature_weights_keep_mean_va_primary_after_scaling():
    weights = cluster_feature_weights(
        "va_geometry",
        14,
        conflict_cluster_weight=0.4,
        gate_cluster_weight=0.2,
    )

    np.testing.assert_allclose(weights[:2], np.asarray([2.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(weights[2:14], np.full(12, 0.4, dtype=np.float32))
    weighted = apply_cluster_feature_weights(np.ones((1, 14), dtype=np.float32), weights)
    np.testing.assert_allclose(weighted[0], weights)


def test_non_geometry_feature_weights_are_neutral():
    weights = cluster_feature_weights(
        "mean_va",
        2,
        conflict_cluster_weight=0.4,
        gate_cluster_weight=0.2,
    )

    np.testing.assert_allclose(weights, np.ones(2, dtype=np.float32))


def test_full_cluster_feature_strategy_uses_va_geometry_as_conflict_block():
    embeddings = {
        "z_fused": np.ones((2, 2), dtype=np.float32),
        "z_audio": np.ones((2, 2), dtype=np.float32) * 2.0,
        "z_lyrics": np.ones((2, 2), dtype=np.float32) * 3.0,
        "z_metadata": np.ones((2, 2), dtype=np.float32) * 4.0,
        "gate_weights": np.ones((2, 3), dtype=np.float32) / 3.0,
        "consistency": np.zeros((2, 1), dtype=np.float32),
        "va_diff": np.zeros((2, 2), dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
        "va_geometry": np.ones((2, 17), dtype=np.float32) * 5.0,
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="full",
    )

    np.testing.assert_allclose(features[:, -14:], np.ones((2, 14), dtype=np.float32) * 2.0)
    assert pca is None


def test_fused_va_geometry_uses_trained_fused_embedding_and_weighted_va_geometry():
    embeddings = {
        "z_fused": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "z_audio": np.ones((2, 2), dtype=np.float32) * 9.0,
        "z_lyrics": np.ones((2, 2), dtype=np.float32) * 8.0,
        "z_metadata": np.ones((2, 2), dtype=np.float32) * 7.0,
        "gate_weights": np.ones((2, 3), dtype=np.float32) / 3.0,
        "consistency": np.zeros((2, 1), dtype=np.float32),
        "va_diff": np.zeros((2, 2), dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
        "va_geometry": np.ones((2, 17), dtype=np.float32) * 5.0,
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="fused_va_geometry",
    )
    weights = cluster_feature_weights(
        "fused_va_geometry",
        int(features.shape[1]),
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
    )

    np.testing.assert_allclose(features[:, :2], embeddings["z_fused"])
    np.testing.assert_allclose(features[:, 2:], embeddings["va_geometry"][:, :14])
    np.testing.assert_allclose(weights[:2], np.full(2, 0.5, dtype=np.float32))
    np.testing.assert_allclose(weights[2:4], np.full(2, 2.0, dtype=np.float32))
    np.testing.assert_allclose(weights[4:16], np.full(12, 0.4, dtype=np.float32))
    assert pca is None


def test_masked_diffaware_imputes_missing_diff_latents_conditionally():
    observed_positions = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [100.0]], dtype=np.float32)
    missing_positions = np.asarray([[0.2], [99.8]], dtype=np.float32)
    z_fused = np.vstack([observed_positions, missing_positions]).astype(np.float32)
    z_diff = np.vstack(
        [
            observed_positions * 2.0 + 1.0,
            np.zeros_like(missing_positions),
        ]
    ).astype(np.float32)
    z_metadata = np.ones_like(z_fused, dtype=np.float32)
    view_mask = np.ones((8, 3), dtype=np.float32)
    view_mask[6:, 1] = 0.0
    embeddings = {
        "z_fused": z_fused,
        "z_diff": z_diff,
        "z_metadata": z_metadata,
        "view_mask": view_mask,
    }

    features, _, state = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="masked_diffaware",
        diff_cluster_weight=0.35,
    )

    assert state is not None
    diff_block = features[:, 1:2]
    assert diff_block[6, 0] != 0.0
    assert diff_block[7, 0] != 0.0
    assert abs(float(diff_block[6, 0] - diff_block[7, 0])) > 1.0


def test_masked_diffaware_feature_weights_apply_after_scaling():
    weights = cluster_feature_weights(
        "masked_diffaware",
        6,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        metadata_cluster_weight=0.75,
        diff_cluster_weight=0.35,
    )

    np.testing.assert_allclose(weights[:2], np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(weights[2:4], np.full(2, 0.35, dtype=np.float32))
    np.testing.assert_allclose(weights[4:6], np.full(2, 0.75, dtype=np.float32))


def test_macro_micro_diffaware_uses_trained_affect_tension_metadata_blocks():
    embeddings = {
        "z_cluster": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "z_affect": np.asarray([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        "z_fused": np.asarray([[9.0, 9.0], [8.0, 8.0]], dtype=np.float32),
        "z_tension": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "z_diff": np.asarray([[7.0, 7.0], [6.0, 6.0]], dtype=np.float32),
        "z_metadata": np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
    }

    features, _, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="macro_micro_diffaware",
        diff_cluster_weight=0.35,
    )

    np.testing.assert_allclose(features[:, :2], embeddings["z_affect"])
    np.testing.assert_allclose(features[:, 2:4], embeddings["z_tension"])
    np.testing.assert_allclose(features[:, 4:6], embeddings["z_metadata"])


def test_macro_micro_diffaware_falls_back_to_trained_fused_when_affect_missing():
    embeddings = {
        "z_cluster": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "z_fused": np.asarray([[9.0, 9.0], [8.0, 8.0]], dtype=np.float32),
        "z_tension": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "z_metadata": np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
    }

    features, _, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="macro_micro_diffaware",
        diff_cluster_weight=0.35,
    )

    np.testing.assert_allclose(features[:, :2], embeddings["z_fused"])


def test_macro_micro_block_mask_keeps_consensus_available_for_empty_rows():
    view_mask = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    block_mask = cluster_feature_block_mask("macro_micro_diffaware", view_mask, view_mask.shape[0])

    expected = np.asarray(
        [
            [True, False, False],
            [True, False, True],
            [True, False, False],
            [True, True, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(block_mask, expected)
    assert block_mask.any(axis=1).all()


def test_semantic_composite_selection_is_not_plain_composite_alias():
    features = np.asarray(
        [
            [-3.0, -3.0],
            [-3.1, -2.9],
            [-2.8, -3.2],
            [3.0, 3.0],
            [3.2, 2.9],
            [2.9, 3.1],
        ],
        dtype=np.float32,
    )

    _model, metrics, info = run_k_selection(
        features=features,
        k_strategy="semantic_composite",
        k_min=2,
        k_max=2,
        random_state=7,
        min_cluster_size_abs=1,
        min_cluster_size_ratio=0.0,
        covariance_type="diag",
        stability_runs=1,
        cluster_backend="sklearn",
        eval_backend="sklearn",
        silhouette_mode="sampled",
        silhouette_sample_size=0,
    )

    assert info["selection_mode"] == "semantic_composite"
    assert "semantic_composite_score" in metrics.columns


def test_run_pipeline_parser_accepts_stage_alias_from_v5_plan():
    args = build_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--stage",
            "pretrain",
        ]
    )

    assert args.run_stage == "pretrain"


def test_run_pipeline_parser_accepts_macro_micro_k_strategy():
    args = build_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--k_strategy",
            "macro_micro",
        ]
    )

    assert args.k_strategy == "macro_micro"


def test_partial_likelihood_k_selection_uses_masked_model():
    features = np.asarray(
        [
            [-3.0, -9.0, -3.0],
            [-2.8, -8.5, -2.9],
            [-3.2, -9.5, -3.1],
            [3.0, 9.0, 3.0],
            [2.8, 8.5, 2.9],
            [3.2, 9.5, 3.1],
        ],
        dtype=np.float32,
    )
    block_mask = np.asarray(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
            [True, True, True],
            [True, False, True],
            [True, True, True],
        ],
        dtype=bool,
    )

    model, metrics, info = run_k_selection(
        features=features,
        k_strategy="semantic_composite",
        k_min=2,
        k_max=2,
        random_state=7,
        min_cluster_size_abs=1,
        min_cluster_size_ratio=0.0,
        covariance_type="diag",
        stability_runs=1,
        cluster_backend="sklearn",
        eval_backend="sklearn",
        silhouette_mode="sampled",
        silhouette_sample_size=0,
        assignment_mode="partial_likelihood",
        block_mask=block_mask,
        block_slices=[(0, 1), (1, 2), (2, 3)],
    )

    assert model.__class__.__name__ == "MaskedDiagonalGMM"
    assert info["partial_likelihood"] is True
    assert info["selection_mode"] == "masked_semantic_composite"
    assert "masked_bic" in metrics.columns


def test_dataset_plot_va_can_use_original_va():
    class Dataset:
        raw_audio = np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32)
        raw_lyrics = np.asarray([[0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
        view_mask = np.ones((2, 3), dtype=np.float32)
        original_va = np.asarray([[0.7, 0.6], [0.3, 0.4]], dtype=np.float32)

    np.testing.assert_allclose(_dataset_plot_va(Dataset(), "original"), Dataset.original_va)


def test_cluster_feature_pca_plot_writes_actual_gmm_feature_projection(tmp_path):
    features = np.asarray(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.3],
            [3.0, 3.1, 2.9],
            [3.2, 3.0, 3.1],
        ],
        dtype=np.float32,
    )
    assignments = np.asarray([0, 0, 1, 1], dtype=np.int64)
    palette = {0: "#cc3333", 1: "#3366dd"}
    out_path = tmp_path / "cluster_feature_pca.png"

    _plot_cluster_feature_pca(features, assignments, str(out_path), palette)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_cluster_summary_filters_low_support_metadata_tokens_with_fdr():
    class Dataset:
        raw_audio = np.full((20, 2), 0.5, dtype=np.float32)
        raw_lyrics = np.full((20, 2), 0.5, dtype=np.float32)
        view_mask = np.ones((20, 3), dtype=np.float32)
        labels = np.zeros(20, dtype=np.int64)
        identifiers = np.asarray([f"audio-{idx}" for idx in range(20)])
        lyric_identifiers = np.asarray([f"lyrics-{idx}" for idx in range(20)])
        consistency = np.ones(20, dtype=np.float32)
        va_diff = np.zeros((20, 2), dtype=np.float32)
        canonical_metadata = pd.DataFrame()

    metadata = np.zeros((20, 3), dtype=np.float32)
    metadata[:2, 0] = 1.0
    metadata[:8, 1] = 1.0
    metadata[10:12, 1] = 1.0
    metadata[:, 2] = 1.0
    Dataset.raw_metadata = metadata
    Dataset.raw_metadata_report = metadata

    summary = _cluster_summary(
        assignments=np.asarray([0] * 10 + [1] * 10, dtype=np.int64),
        dataset=Dataset(),
        metadata_feature_names=[
            "Genres::rare-low-support",
            "MoodsAll::bright",
            "Themes::common",
        ],
    )

    cluster_zero_tokens = summary[0]["top_metadata_tokens"]
    token_names = [item["feature"] for item in cluster_zero_tokens]
    assert "Genres::rare-low-support" not in token_names
    assert "MoodsAll::bright" in token_names

    bright = next(item for item in cluster_zero_tokens if item["feature"] == "MoodsAll::bright")
    assert bright["field"] == "MoodsAll"
    assert bright["token"] == "bright"
    assert bright["support"] == 8
    assert bright["global_support"] == 10
    assert 0.0 <= bright["p_value"] <= bright["q_value"] <= 1.0
    assert all(item["support"] >= 5 and item["global_support"] >= 10 for item in cluster_zero_tokens)


def test_full_strategy_no_missingness_leakage():
    """Missing-view rows should not be identifiable by their feature values."""
    rng = np.random.default_rng(42)
    n = 100
    z_fused = rng.standard_normal((n, 4)).astype(np.float32)
    z_audio = z_fused + rng.standard_normal((n, 4)).astype(np.float32) * 0.1
    z_lyrics = z_fused + rng.standard_normal((n, 4)).astype(np.float32) * 0.1
    z_metadata = rng.standard_normal((n, 4)).astype(np.float32)
    gate_weights = np.full((n, 3), 1.0 / 3, dtype=np.float32)
    consistency = rng.uniform(0.5, 1.0, (n, 1)).astype(np.float32)
    va_diff = rng.standard_normal((n, 2)).astype(np.float32) * 0.1

    view_mask = np.ones((n, 3), dtype=np.float32)
    # Make 30% of rows missing audio
    missing_audio = rng.choice(n, size=30, replace=False)
    view_mask[missing_audio, 0] = 0.0
    consistency[missing_audio] = 0.0
    va_diff[missing_audio] = 0.0

    embeddings = {
        "z_fused": z_fused,
        "z_audio": z_audio,
        "z_lyrics": z_lyrics,
        "z_metadata": z_metadata,
        "gate_weights": gate_weights,
        "consistency": consistency,
        "va_diff": va_diff,
        "view_mask": view_mask,
    }

    features, _, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="full",
    )

    # No column should be all-zero for missing rows and non-zero for observed
    observed_mask = view_mask[:, 0] > 0
    for col in range(features.shape[1]):
        obs_std = features[observed_mask, col].std()
        miss_std = features[~observed_mask, col].std()
        if obs_std > 0.01:
            assert miss_std > 0.001, (
                f"Column {col} has near-zero variance for missing rows "
                f"(obs_std={obs_std:.4f}, miss_std={miss_std:.6f}) — likely leakage"
            )
