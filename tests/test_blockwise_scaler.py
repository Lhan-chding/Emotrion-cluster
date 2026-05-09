import numpy as np

from cluster.evaluation.metrics import masked_pairwise_distances
from cluster.features.block_scaler import BlockwiseObservedScaler
from cluster.pipeline.train import fit_cluster_scaler


def test_blockwise_observed_scaler_ignores_unobserved_blocks_and_zeroes_them():
    features = np.asarray(
        [
            [0.0, 0.0, 100.0, 100.0],
            [2.0, 2.0, 200.0, 200.0],
            [4.0, 4.0, 9999.0, 9999.0],
            [6.0, 6.0, -9999.0, -9999.0],
        ],
        dtype=np.float32,
    )
    block_mask = np.asarray(
        [
            [True, True],
            [True, True],
            [True, False],
            [True, False],
        ]
    )

    scaler = BlockwiseObservedScaler(block_slices=[(0, 2), (2, 4)]).fit(features, block_mask=block_mask)
    transformed = scaler.transform(features, block_mask=block_mask)

    assert scaler.observed_counts_.tolist() == [4, 2]
    np.testing.assert_allclose(transformed[:2, 2:], [[-1.0, -1.0], [1.0, 1.0]], atol=1e-6)
    np.testing.assert_allclose(transformed[2:, 2:], 0.0, atol=1e-6)


def test_masked_pairwise_distances_ignore_features_without_common_observation():
    features = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9999.0, 9999.0],
            [5.0, 5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    block_mask = np.asarray(
        [
            [True, True],
            [True, False],
            [True, True],
        ]
    )

    distances = masked_pairwise_distances(features, block_mask=block_mask, block_slices=[(0, 2), (2, 4)])

    assert distances[0, 1] == 0.0
    assert distances[0, 2] > 0.0
    assert distances[1, 2] > 0.0


def test_fit_cluster_scaler_respects_observed_block_scaler_flag():
    features = np.asarray([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 999.0, 999.0]], dtype=np.float32)
    block_mask = np.asarray([[True, True], [True, False]], dtype=bool)

    observed = fit_cluster_scaler(
        "macro_micro_diffaware",
        features,
        block_mask=block_mask,
        block_slices=[(0, 2), (2, 4)],
        block_scaler="observed",
    )
    standard = fit_cluster_scaler(
        "macro_micro_diffaware",
        features,
        block_mask=block_mask,
        block_slices=[(0, 2), (2, 4)],
        block_scaler="standard",
    )

    assert isinstance(observed, BlockwiseObservedScaler)
    assert not isinstance(standard, BlockwiseObservedScaler)
