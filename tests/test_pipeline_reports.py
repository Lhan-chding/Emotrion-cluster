import numpy as np
import pandas as pd

from cluster.pipeline.rerun import build_parser as build_rerun_parser
from cluster.pipeline.train import (
    _cluster_label_names_for_outputs,
    _dataset_plot_va,
    _plot_cluster_feature_pca,
    _quadrant_heatmap_matrix,
    _cluster_summary,
    _va_quadrant_labels,
    _write_split_outputs,
    build_parser,
    apply_metadata_policy_to_block_mask,
    apply_cluster_feature_weights,
    build_cluster_features,
    cluster_feature_block_mask,
    cluster_feature_block_slices,
    cluster_feature_weights,
    resolve_metadata_policy,
    run_k_selection,
)


def test_cluster_label_names_for_outputs_reads_macro_micro_selection_info():
    class Result:
        pass

    assert _cluster_label_names_for_outputs(Result(), {"label_names": {"0": "M1-a", 1: "M1-b"}}) == {
        0: "M1-a",
        1: "M1-b",
    }


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


def test_va_quadrant_labels_are_derived_from_plot_coordinates():
    labels = _va_quadrant_labels(
        np.asarray(
            [
                [0.6, 0.7],
                [0.4, 0.7],
                [0.4, 0.3],
                [0.6, 0.3],
            ],
            dtype=np.float32,
        )
    )

    np.testing.assert_array_equal(labels, np.asarray([0, 1, 2, 3], dtype=np.int64))


def test_va_quadrant_labels_can_exclude_boundary_points():
    labels = _va_quadrant_labels(
        np.asarray(
            [
                [0.6, 0.7],
                [0.51, 0.8],
                [0.3, 0.49],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
        boundary_margin=0.03,
    )

    np.testing.assert_array_equal(labels, np.asarray([0, -1, -1, 3], dtype=np.int64))


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


def test_balanced_va_diff_cluster_feature_strategy_uses_consensus_and_compact_delta():
    embeddings = {
        "audio_va": np.asarray([[0.8, 0.6], [0.7, 0.4]], dtype=np.float32),
        "lyrics_va": np.asarray([[0.2, 0.7], [0.3, 0.2]], dtype=np.float32),
        "mean_va": np.asarray([[0.5, 0.65], [0.5, 0.3]], dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
    }

    features, pca, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="balanced_va_diff",
    )
    weights = cluster_feature_weights(
        "balanced_va_diff",
        int(features.shape[1]),
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        diff_cluster_weight=0.65,
    )

    np.testing.assert_allclose(features[:, :2], embeddings["mean_va"])
    np.testing.assert_allclose(features[:, 2:4], embeddings["audio_va"] - embeddings["lyrics_va"])
    reconstructed_audio = features[:, :2] + 0.5 * features[:, 2:4]
    reconstructed_lyrics = features[:, :2] - 0.5 * features[:, 2:4]
    np.testing.assert_allclose(reconstructed_audio, embeddings["audio_va"])
    np.testing.assert_allclose(reconstructed_lyrics, embeddings["lyrics_va"])
    np.testing.assert_allclose(weights[:2], np.full(2, 2.5, dtype=np.float32))
    np.testing.assert_allclose(weights[2:], np.full(6, 0.65, dtype=np.float32))
    assert pca is None


def test_balanced_va_diff_zero_weight_degenerates_to_weighted_mean_va():
    embeddings = {
        "audio_va": np.asarray([[0.8, 0.6], [0.7, 0.4]], dtype=np.float32),
        "lyrics_va": np.asarray([[0.2, 0.7], [0.3, 0.2]], dtype=np.float32),
        "mean_va": np.asarray([[0.5, 0.65], [0.5, 0.3]], dtype=np.float32),
        "view_mask": np.ones((2, 3), dtype=np.float32),
    }

    features, _, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="balanced_va_diff",
    )
    weights = cluster_feature_weights(
        "balanced_va_diff",
        int(features.shape[1]),
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        diff_cluster_weight=0.0,
    )
    weighted = apply_cluster_feature_weights(features, weights)

    np.testing.assert_allclose(weighted[:, :2], embeddings["mean_va"] * 2.5)
    np.testing.assert_allclose(weighted[:, 2:], np.zeros((2, 6), dtype=np.float32))


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


def test_affective_va_policy_removes_metadata_from_cluster_mask_and_weight():
    policy = resolve_metadata_policy(
        "affective_va_only",
        metadata_feature_names=["MoodsAll::happy", "Themes::party", "Genres::rock"],
        requested_metadata_cluster_weight=0.75,
    )
    assert policy["effective_metadata_cluster_weight"] == 0.0
    assert policy["metadata_block_used_for_clustering"] is False

    view_mask = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    block_mask = cluster_feature_block_mask("macro_micro_diffaware", view_mask, view_mask.shape[0])
    block_mask = apply_metadata_policy_to_block_mask(
        block_mask,
        metadata_cluster_weight=policy["effective_metadata_cluster_weight"],
    )

    expected = np.asarray(
        [
            [True, False, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(block_mask, expected)


def test_zero_diff_weight_removes_diff_block_from_cluster_mask():
    view_mask = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    block_mask = cluster_feature_block_mask("macro_micro_diffaware", view_mask, view_mask.shape[0])
    block_mask = apply_metadata_policy_to_block_mask(
        block_mask,
        metadata_cluster_weight=1.0,
        diff_cluster_weight=0.0,
    )

    np.testing.assert_array_equal(block_mask, np.asarray([[True, False, True]], dtype=bool))


def test_balanced_va_diff_block_slices_keep_consensus_and_diff_separate():
    view_mask = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert cluster_feature_block_slices("balanced_va_diff", 8) == [(0, 2), (2, 8)]
    block_mask = cluster_feature_block_mask("balanced_va_diff", view_mask, view_mask.shape[0])

    np.testing.assert_array_equal(
        block_mask,
        np.asarray(
            [
                [True, True],
                [True, False],
            ],
            dtype=bool,
        ),
    )


def test_zero_diff_weight_removes_balanced_va_diff_block():
    view_mask = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    block_mask = cluster_feature_block_mask("balanced_va_diff", view_mask, view_mask.shape[0])
    block_mask = apply_metadata_policy_to_block_mask(
        block_mask,
        metadata_cluster_weight=0.0,
        diff_cluster_weight=0.0,
    )

    np.testing.assert_array_equal(block_mask, np.asarray([[True, False]], dtype=bool))


def test_non_affective_metadata_policy_rejects_affective_fields():
    with np.testing.assert_raises(ValueError):
        resolve_metadata_policy(
            "non_affective_metadata",
            metadata_feature_names=["Genres::rock", "MoodsAll::happy"],
            requested_metadata_cluster_weight=0.75,
        )


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


def test_run_pipeline_parser_accepts_v6_plan_gate_flags():
    args = build_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--metadata_policy",
            "report_only",
            "--total_k_min",
            "8",
            "--total_k_max",
            "16",
            "--silhouette_mode",
            "masked_torch_chunked",
            "--block_scaler",
            "observed",
            "--run_topconf_audit",
            "true",
            "--require_both_va",
            "true",
            "--affect_gate",
            "true",
            "--min_affect_dominant_ratio",
            "0.70",
            "--max_affect_mixed_cluster_fraction",
            "0.15",
            "--min_affect_weighted_purity",
            "0.80",
            "--affect_boundary_margin",
            "0.03",
        ]
    )

    assert args.metadata_policy == "report_only"
    assert args.k_min == 8
    assert args.k_max == 16
    assert args.silhouette_mode == "masked_torch_chunked"
    assert args.block_scaler == "observed"
    assert args.run_topconf_audit == "true"
    assert args.require_both_va == "true"
    assert args.affect_gate == "true"
    assert args.min_affect_dominant_ratio == 0.70
    assert args.max_affect_mixed_cluster_fraction == 0.15
    assert args.min_affect_weighted_purity == 0.80
    assert args.affect_boundary_margin == 0.03


def test_parsers_accept_v17_adaptive_balance_flags():
    train_args = build_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--consensus_mode",
            "clusterability_alpha",
            "--consensus_alpha",
            "0.75",
            "--alpha_search_min",
            "0.20",
            "--alpha_search_max",
            "0.90",
            "--alpha_search_step",
            "0.05",
            "--alpha_search_k_min",
            "4",
            "--alpha_search_k_max",
            "8",
            "--micro_consensus_role",
            "visible_separator",
            "--micro_min_consensus_knn_purity",
            "0.85",
            "--micro_min_consensus_center_sep",
            "0.60",
            "--micro_min_consensus_silhouette",
            "0.05",
            "--overlap_gate",
            "true",
            "--min_va_knn_purity",
            "0.90",
            "--min_va_center_sep",
            "0.70",
            "--max_va_negative_silhouette_fraction",
            "0.10",
            "--plot_va_source",
            "cluster_consensus",
            "--run_tension_micro_probe",
            "true",
            "--tension_micro_k_max",
            "4",
        ]
    )
    rerun_args = build_rerun_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--consensus_mode",
            "bias_neutral_mean",
            "--overlap_gate",
            "true",
            "--plot_va_source",
            "cluster_consensus",
            "--run_tension_micro_probe",
            "false",
        ]
    )

    assert train_args.consensus_mode == "clusterability_alpha"
    assert train_args.consensus_alpha == 0.75
    assert train_args.micro_consensus_role == "visible_separator"
    assert train_args.overlap_gate == "true"
    assert train_args.plot_va_source == "cluster_consensus"
    assert train_args.run_tension_micro_probe == "true"
    assert train_args.tension_micro_k_max == 4
    assert rerun_args.consensus_mode == "bias_neutral_mean"
    assert rerun_args.overlap_gate == "true"
    assert rerun_args.run_tension_micro_probe == "false"


def test_rerun_parser_accepts_complete_pair_filter_flag():
    args = build_rerun_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--cluster_feature_strategy",
            "balanced_va_diff",
            "--require_both_va",
            "true",
            "--affect_boundary_margin",
            "0.02",
        ]
    )

    assert args.require_both_va == "true"
    assert args.cluster_feature_strategy == "balanced_va_diff"
    assert args.affect_boundary_margin == 0.02


def test_metadata_only_strategy_uses_metadata_embedding_without_other_views():
    embeddings = {
        "z_fused": np.asarray([[9.0, 9.0], [8.0, 8.0]], dtype=np.float32),
        "z_audio": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "z_lyrics": np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "z_metadata": np.asarray([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        "gate_weights": np.ones((2, 3), dtype=np.float32) / 3.0,
        "consistency": np.zeros((2, 1), dtype=np.float32),
        "va_diff": np.zeros((2, 2), dtype=np.float32),
        "view_mask": np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32),
    }

    features, _, _ = build_cluster_features(
        embeddings,
        metadata_cluster_weight=1.0,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="metadata_only",
    )

    expected = np.asarray([[0.2, 0.4], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(features, expected)


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


def test_cluster_summary_uses_va_plane_quadrants_not_raw_labels():
    class Dataset:
        raw_audio = np.asarray([[0.8, 0.8], [0.7, 0.7], [0.2, 0.2], [0.3, 0.3]], dtype=np.float32)
        raw_lyrics = raw_audio.copy()
        view_mask = np.ones((4, 3), dtype=np.float32)
        labels = np.asarray([2, 2, 0, 0], dtype=np.int64)
        identifiers = np.asarray([f"audio-{idx}" for idx in range(4)])
        lyric_identifiers = np.asarray([f"lyrics-{idx}" for idx in range(4)])
        raw_metadata = np.zeros((4, 1), dtype=np.float32)
        raw_metadata_report = raw_metadata
        canonical_metadata = pd.DataFrame()

    summary = _cluster_summary(
        assignments=np.asarray([0, 0, 1, 1], dtype=np.int64),
        dataset=Dataset(),
        metadata_feature_names=["numeric::zero"],
    )

    assert summary[0]["dominant_quadrant"] == "Q1"
    assert summary[1]["dominant_quadrant"] == "Q3"


def test_write_split_outputs_writes_macro_micro_artifacts(tmp_path):
    class Dataset:
        raw_audio = np.tile(np.asarray([[0.2, 0.3]], dtype=np.float32), (20, 1))
        raw_lyrics = np.tile(np.asarray([[0.3, 0.4]], dtype=np.float32), (20, 1))
        view_mask = np.ones((20, 3), dtype=np.float32)
        labels = np.asarray([0] * 8 + [1] * 6 + [2] * 6, dtype=np.int64)
        identifiers = np.asarray([f"audio-{idx}" for idx in range(20)])
        lyric_identifiers = np.asarray([f"lyrics-{idx}" for idx in range(20)])
        consistency = np.ones(20, dtype=np.float32)
        va_diff = np.zeros((20, 2), dtype=np.float32)
        canonical_metadata = pd.DataFrame()

    metadata = np.zeros((20, 2), dtype=np.float32)
    metadata[:8, 0] = 1.0
    metadata[8:10, 0] = 1.0
    metadata[8:14, 1] = 1.0
    metadata[14:18, 1] = 1.0
    Dataset.raw_metadata = metadata
    Dataset.raw_metadata_report = metadata

    assignments = np.asarray([0] * 8 + [1] * 6 + [2] * 6, dtype=np.int64)
    embeddings = {
        "gate_weights": np.full((20, 3), 1.0 / 3.0, dtype=np.float32),
        "z_fused": np.random.default_rng(9).standard_normal((20, 4)).astype(np.float32),
    }

    payload = _write_split_outputs(
        out_dir=str(tmp_path),
        split="all",
        dataset=Dataset(),
        embeddings=embeddings,
        assignments=assignments,
        metadata_feature_names=["MoodsAll::blue", "MoodsAll::bright"],
        selected_k=3,
        feature_dim=4,
        cluster_features=np.random.default_rng(11).standard_normal((20, 4)).astype(np.float32),
        cluster_label_names={0: "M1-a", 1: "M1-b", 2: "M2"},
    )

    catalog = pd.read_csv(tmp_path / "cluster_catalog.csv")
    assignments_frame = pd.read_csv(tmp_path / "cluster_assignments.csv")
    macro_summary = pd.read_csv(tmp_path / "macro_micro_summary.csv")
    macro_summary["micro_label"] = macro_summary["micro_label"].fillna("")
    metadata_enrichment = pd.read_csv(tmp_path / "macro_micro_metadata_enrichment.csv")

    assert {"label_name", "macro_id", "micro_id", "micro_label"}.issubset(catalog.columns)
    assert {"label_name", "macro_id", "micro_id", "micro_label"}.issubset(assignments_frame.columns)
    assert macro_summary[["cluster_id", "label_name", "macro_id", "micro_label"]].to_dict("records") == [
        {"cluster_id": 0, "label_name": "M1-a", "macro_id": 1, "micro_label": "a"},
        {"cluster_id": 1, "label_name": "M1-b", "macro_id": 1, "micro_label": "b"},
        {"cluster_id": 2, "label_name": "M2", "macro_id": 2, "micro_label": ""},
    ]
    assert set(metadata_enrichment["macro_id"].tolist()) == {1}
    assert (tmp_path / "macro_micro" / "macro_1_diff_arrow.png").exists()
    assert (tmp_path / "macro_micro" / "macro_1_metadata_enrichment.csv").exists()
    assert payload["output_files"]["macro_micro_summary"].endswith("macro_micro_summary.csv")


def test_write_split_outputs_uses_cluster_consensus_as_balanced_va(tmp_path):
    class Dataset:
        raw_audio = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )
        raw_lyrics = 1.0 - raw_audio
        view_mask = np.ones((6, 3), dtype=np.float32)
        labels = np.zeros(6, dtype=np.int64)
        identifiers = np.asarray([f"audio-{idx}" for idx in range(6)])
        lyric_identifiers = np.asarray([f"lyrics-{idx}" for idx in range(6)])
        consistency = np.ones(6, dtype=np.float32)
        canonical_metadata = pd.DataFrame()
        raw_metadata = np.zeros((6, 1), dtype=np.float32)
        raw_metadata_report = raw_metadata

    assignments = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    balanced_va = np.asarray(
        [
            [0.75, 0.25],
            [0.70, 0.30],
            [0.65, 0.35],
            [0.35, 0.65],
            [0.30, 0.70],
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )
    embeddings = {
        "gate_weights": np.full((6, 3), 1.0 / 3.0, dtype=np.float32),
        "z_fused": np.random.default_rng(10).standard_normal((6, 4)).astype(np.float32),
    }

    payload = _write_split_outputs(
        out_dir=str(tmp_path),
        split="all",
        dataset=Dataset(),
        embeddings=embeddings,
        assignments=assignments,
        metadata_feature_names=["MoodsAll::test"],
        selected_k=2,
        feature_dim=5,
        cluster_features=np.random.default_rng(12).standard_normal((6, 5)).astype(np.float32),
        plot_va_source="cluster_consensus",
        plot_va_override=balanced_va,
        feature_state={"balance_alpha": 0.75},
    )

    assignments_frame = pd.read_csv(tmp_path / "cluster_assignments.csv")
    catalog = pd.read_csv(tmp_path / "cluster_catalog.csv")

    assert {"raw_mean_valence", "balanced_valence", "balance_alpha"}.issubset(assignments_frame.columns)
    np.testing.assert_allclose(assignments_frame["balanced_valence"].to_numpy(), balanced_va[:, 0])
    np.testing.assert_allclose(assignments_frame["mean_valence"].to_numpy(), balanced_va[:, 0])
    assert not np.allclose(assignments_frame["raw_mean_valence"].to_numpy(), balanced_va[:, 0])
    assert set(catalog.columns).issuperset({"balanced_valence", "balanced_arousal", "raw_mean_valence", "raw_mean_arousal"})
    assert (tmp_path / "cluster_scatter_balanced_va.png").exists()
    assert (tmp_path / "cluster_scatter_raw_mean_va.png").exists()
    assert (tmp_path / "cluster_scatter_audio_va.png").exists()
    assert (tmp_path / "cluster_scatter_lyrics_va.png").exists()
    assert payload["plot_va_source"] == "cluster_consensus"
    assert payload["output_files"]["cluster_scatter"].endswith("cluster_scatter_balanced_va.png")


def test_write_split_outputs_writes_report_only_tension_micro_probe(tmp_path):
    rng = np.random.default_rng(31)
    consensus = np.vstack(
        [
            np.tile(np.asarray([[0.30, 0.70]], dtype=np.float32), (12, 1)),
            np.tile(np.asarray([[0.75, 0.30]], dtype=np.float32), (12, 1)),
        ]
    )
    tension = np.vstack(
        [
            np.tile(np.asarray([[0.16, 0.02]], dtype=np.float32), (6, 1)),
            np.tile(np.asarray([[-0.16, -0.02]], dtype=np.float32), (6, 1)),
            np.tile(np.asarray([[0.02, 0.14]], dtype=np.float32), (6, 1)),
            np.tile(np.asarray([[-0.02, -0.14]], dtype=np.float32), (6, 1)),
        ]
    )
    tension += rng.normal(0.0, 0.005, tension.shape).astype(np.float32)

    class Dataset:
        raw_audio = np.clip(consensus - 0.5 * tension, 0.0, 1.0).astype(np.float32)
        raw_lyrics = np.clip(consensus + 0.5 * tension, 0.0, 1.0).astype(np.float32)
        view_mask = np.ones((24, 3), dtype=np.float32)
        labels = np.asarray([1] * 12 + [3] * 12, dtype=np.int64)
        identifiers = np.asarray([f"audio-{idx}" for idx in range(24)])
        lyric_identifiers = np.asarray([f"lyrics-{idx}" for idx in range(24)])
        consistency = np.ones(24, dtype=np.float32)
        va_diff = raw_audio - raw_lyrics
        canonical_metadata = pd.DataFrame()
        raw_metadata = np.zeros((24, 1), dtype=np.float32)
        raw_metadata_report = raw_metadata

    assignments = np.asarray([0] * 12 + [1] * 12, dtype=np.int64)
    embeddings = {
        "gate_weights": np.full((24, 3), 1.0 / 3.0, dtype=np.float32),
        "z_fused": rng.standard_normal((24, 4)).astype(np.float32),
    }

    payload = _write_split_outputs(
        out_dir=str(tmp_path),
        split="all",
        dataset=Dataset(),
        embeddings=embeddings,
        assignments=assignments,
        metadata_feature_names=["numeric::zero"],
        selected_k=2,
        feature_dim=2,
        cluster_features=consensus,
        plot_va_source="cluster_consensus",
        plot_va_override=consensus,
        feature_state={
            "tension_micro_probe_config": {
                "enabled": True,
                "k_max": 2,
                "min_cluster_size": 4,
                "min_silhouette": -1.0,
                "min_effect": 0.0,
                "stability_runs": 3,
                "random_state": 13,
            }
        },
    )

    probe_path = tmp_path / "tension_micro_probe" / "tension_micro_probe.csv"
    assignment_path = tmp_path / "tension_micro_probe" / "tension_micro_assignments.csv"
    probe = pd.read_csv(probe_path)
    tension_assignments = pd.read_csv(assignment_path)
    final_assignments = pd.read_csv(tmp_path / "cluster_assignments.csv")

    assert probe_path.exists()
    assert assignment_path.exists()
    assert (tmp_path / "tension_micro_probe" / "cluster_0_tension_micro.png").exists()
    assert payload["output_files"]["tension_micro_probe"] == str(probe_path)
    assert payload["output_files"]["tension_micro_assignments"] == str(assignment_path)
    assert "tension_micro_probe" in payload
    assert set(probe["cluster_id"].tolist()) == {0, 1}
    assert set(probe["selected_micro_k"].tolist()) == {2}
    assert {"tension_silhouette", "tension_effect_size", "seed_ari_mean"}.issubset(probe.columns)
    assert set(tension_assignments["tension_micro_id"].tolist()) == {0, 1}
    assert "tension_micro_id" not in final_assignments.columns


def test_calibrated_va_tension_feature_state_exposes_alpha_audit():
    embeddings = {
        "audio_va": np.asarray(
            [
                [0.90, 0.80],
                [0.86, 0.78],
                [0.20, 0.30],
                [0.24, 0.34],
                [0.80, 0.25],
                [0.76, 0.22],
                [0.30, 0.82],
                [0.34, 0.78],
            ],
            dtype=np.float32,
        ),
        "lyrics_va": np.asarray(
            [
                [0.70, 0.70],
                [0.66, 0.68],
                [0.35, 0.40],
                [0.38, 0.42],
                [0.68, 0.35],
                [0.66, 0.34],
                [0.42, 0.68],
                [0.44, 0.66],
            ],
            dtype=np.float32,
        ),
        "view_mask": np.ones((8, 3), dtype=np.float32),
    }

    _features, _pca, state = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.0,
        conflict_cluster_weight=0.0,
        gate_cluster_weight=0.0,
        strategy="calibrated_va_tension",
        consensus_mode="clusterability_alpha",
        alpha_search_min=0.20,
        alpha_search_max=0.80,
        alpha_search_step=0.30,
        alpha_search_k_min=2,
        alpha_search_k_max=3,
    )

    assert isinstance(state, dict)
    assert state["consensus_mode"] == "clusterability_alpha"
    assert "balance_alpha_scores" in state
    assert state["balance_alpha_scores"]
    assert {"alpha", "best_k", "score", "silhouette", "stability", "size_balance"}.issubset(
        state["balance_alpha_scores"][0]
    )
    assert "balance_alpha_summary" in state
    assert state["balance_alpha_summary"]["alpha"] == state["balance_alpha"]


def test_write_split_outputs_writes_alpha_audit_and_segment_plot(tmp_path):
    class Dataset:
        raw_audio = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )
        raw_lyrics = 1.0 - raw_audio
        view_mask = np.ones((6, 3), dtype=np.float32)
        labels = np.zeros(6, dtype=np.int64)
        identifiers = np.asarray([f"audio-{idx}" for idx in range(6)])
        lyric_identifiers = np.asarray([f"lyrics-{idx}" for idx in range(6)])
        consistency = np.ones(6, dtype=np.float32)
        canonical_metadata = pd.DataFrame()
        raw_metadata = np.zeros((6, 1), dtype=np.float32)
        raw_metadata_report = raw_metadata

    assignments = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    balanced_va = np.asarray(
        [
            [0.75, 0.25],
            [0.70, 0.30],
            [0.65, 0.35],
            [0.35, 0.65],
            [0.30, 0.70],
            [0.25, 0.75],
        ],
        dtype=np.float32,
    )
    embeddings = {
        "gate_weights": np.full((6, 3), 1.0 / 3.0, dtype=np.float32),
        "z_fused": np.random.default_rng(14).standard_normal((6, 4)).astype(np.float32),
    }

    payload = _write_split_outputs(
        out_dir=str(tmp_path),
        split="all",
        dataset=Dataset(),
        embeddings=embeddings,
        assignments=assignments,
        metadata_feature_names=["MoodsAll::test"],
        selected_k=2,
        feature_dim=5,
        cluster_features=np.random.default_rng(15).standard_normal((6, 5)).astype(np.float32),
        plot_va_source="cluster_consensus",
        plot_va_override=balanced_va,
        feature_state={
            "balance_alpha": 0.65,
            "balance_alpha_scores": [
                {"alpha": 0.35, "best_k": 2, "score": 0.40, "silhouette": 0.30, "stability": 0.20, "size_balance": 1.0},
                {"alpha": 0.65, "best_k": 2, "score": 0.55, "silhouette": 0.42, "stability": 0.30, "size_balance": 1.0},
            ],
            "balance_alpha_summary": {
                "alpha": 0.65,
                "alpha_best_k": 2,
                "alpha_score": 0.55,
                "alpha_silhouette": 0.42,
                "alpha_stability": 0.30,
                "alpha_size_balance": 1.0,
            },
        },
    )

    assert (tmp_path / "balance_alpha_report.csv").exists()
    assert (tmp_path / "alpha_search_curve.png").exists()
    assert (tmp_path / "cluster_audio_lyrics_segments.png").exists()
    assert payload["output_files"]["balance_alpha_report"].endswith("balance_alpha_report.csv")
    assert payload["output_files"]["alpha_search_curve"].endswith("alpha_search_curve.png")
    assert payload["output_files"]["cluster_audio_lyrics_segments"].endswith("cluster_audio_lyrics_segments.png")
    assert payload["balance_alpha_summary"]["alpha"] == 0.65


def test_parser_accepts_latent_two_view_va_gmm_options():
    args = build_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--cluster_feature_strategy",
            "latent_two_view_va",
            "--k_strategy",
            "latent_va_gmm",
            "--cluster_assignment_mode",
            "missing_view_likelihood",
            "--plot_va_source",
            "latent_consensus",
            "--learn_view_bias",
            "true",
            "--share_view_noise",
            "false",
            "--alpha_prior_strength",
            "0.1",
        ]
    )

    assert args.cluster_feature_strategy == "latent_two_view_va"
    assert args.k_strategy == "latent_va_gmm"
    assert args.cluster_assignment_mode == "missing_view_likelihood"
    assert args.plot_va_source == "latent_consensus"
    assert args.alpha_prior_strength == 0.1


def test_run_k_selection_latent_va_gmm_uses_missing_view_likelihood():
    rng = np.random.default_rng(21)
    centers = np.asarray([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32)
    audio = np.vstack([
        center + rng.normal(0.0, 0.03, size=(20, 2)) + np.asarray([0.04, -0.02])
        for center in centers
    ]).astype(np.float32)
    lyrics = np.vstack([
        center + rng.normal(0.0, 0.04, size=(20, 2)) + np.asarray([-0.03, 0.02])
        for center in centers
    ]).astype(np.float32)
    features = np.concatenate([audio, lyrics], axis=1).astype(np.float32)
    block_mask = np.ones((features.shape[0], 2), dtype=bool)
    block_mask[::5, 1] = False
    block_mask[1::5, 0] = False

    model, metrics, info = run_k_selection(
        features=features,
        k_strategy="latent_va_gmm",
        k_min=2,
        k_max=3,
        random_state=21,
        min_cluster_size_abs=5,
        min_cluster_size_ratio=0.0,
        stability_runs=2,
        assignment_mode="missing_view_likelihood",
        block_mask=block_mask,
        block_slices=[(0, 2), (2, 4)],
        latent_learn_view_bias=True,
        latent_share_view_noise=False,
        latent_alpha_prior_strength=0.0,
        latent_max_iter=40,
    )

    assert info["selection_mode"] == "latent_va_gmm"
    assert info["actual_cluster_backend"] == "two_view_latent_va_gmm"
    assert {"icl", "latent_consensus_silhouette", "posterior_margin_mean", "latent_va_score"}.issubset(metrics.columns)
    assert model.posterior_consensus(features[:, :2], features[:, 2:4], block_mask).shape == (features.shape[0], 2)


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
