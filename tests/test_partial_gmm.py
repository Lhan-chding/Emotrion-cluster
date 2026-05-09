import numpy as np

from cluster.backends.partial_gmm import PartialGaussianMixture


def test_partial_diag_gmm_predict_ignores_unobserved_tension_block():
    train = np.asarray(
        [
            [-3.0, -10.0, -3.0],
            [-3.2, -9.5, -2.8],
            [-2.8, -10.5, -3.1],
            [3.0, 10.0, 3.0],
            [3.2, 9.5, 2.8],
            [2.8, 10.5, 3.1],
        ],
        dtype=np.float32,
    )
    block_slices = [(0, 1), (1, 2), (2, 3)]
    block_mask = np.ones((train.shape[0], 3), dtype=bool)
    model = PartialGaussianMixture(
        n_components=2,
        block_slices=block_slices,
        random_state=7,
        n_init=5,
    ).fit(train, block_mask=block_mask)

    sample = np.asarray([[3.1, -10.0, 2.9]], dtype=np.float32)
    sample_mask = np.asarray([[True, False, True]], dtype=bool)

    label = int(model.predict(sample, block_mask=sample_mask)[0])
    proba = model.predict_proba(sample, block_mask=sample_mask)
    expected_label = int(model.predict(np.asarray([[3.1, 10.0, 2.9]], dtype=np.float32))[0])

    assert label == expected_label
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(1), atol=1e-6)
