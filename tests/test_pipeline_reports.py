import numpy as np

from cluster.pipeline.train import _dataset_plot_va, _quadrant_heatmap_matrix, build_cluster_features


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

    features, pca = build_cluster_features(
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

    features, pca = build_cluster_features(
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

    features, pca = build_cluster_features(
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

    features, pca = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="va_geometry",
    )

    np.testing.assert_allclose(features, embeddings["va_geometry"])
    assert pca is None


def test_va_geometry_cluster_feature_strategy_accepts_minimal_raw_embeddings():
    embeddings = {
        "va_geometry": np.zeros((2, 17), dtype=np.float32),
    }

    features, pca = build_cluster_features(
        embeddings,
        metadata_cluster_weight=0.75,
        conflict_cluster_weight=0.40,
        gate_cluster_weight=0.20,
        strategy="va_geometry",
    )

    np.testing.assert_allclose(features, embeddings["va_geometry"])
    assert pca is None


def test_dataset_plot_va_can_use_original_va():
    class Dataset:
        raw_audio = np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32)
        raw_lyrics = np.asarray([[0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
        view_mask = np.ones((2, 3), dtype=np.float32)
        original_va = np.asarray([[0.7, 0.6], [0.3, 0.4]], dtype=np.float32)

    np.testing.assert_allclose(_dataset_plot_va(Dataset(), "original"), Dataset.original_va)
