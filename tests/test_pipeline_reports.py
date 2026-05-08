import numpy as np

from cluster.pipeline.train import _quadrant_heatmap_matrix


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
