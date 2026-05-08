import math
import numpy as np

from cluster.features.diff_geometry import (
    DIFF_GEOMETRY_TOTAL_DIM,
    DIFF_GEOMETRY_INPUT_DIM,
    DIFF_OBSERVED_DIM,
    build_diff_geometry_features,
    _soft_quadrant_pair,
)


def test_diff_geometry_output_shapes():
    audio = np.asarray([[0.8, 0.6], [0.3, 0.4], [0.5, 0.5]], dtype=np.float32)
    lyrics = np.asarray([[0.2, 0.7], [0.7, 0.6], [0.0, 0.0]], dtype=np.float32)
    view_mask = np.asarray([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ], dtype=np.float32)

    features, diff_obs = build_diff_geometry_features(audio, lyrics, view_mask)
    assert features.shape == (3, DIFF_GEOMETRY_TOTAL_DIM)
    assert diff_obs.shape == (3, DIFF_OBSERVED_DIM)


def test_diff_geometry_consensus_weighted_mean():
    """Consensus should be weighted mean of available views."""
    audio = np.asarray([[0.8, 0.6], [0.0, 0.0]], dtype=np.float32)
    lyrics = np.asarray([[0.2, 0.4], [0.3, 0.7]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)

    features, diff_obs = build_diff_geometry_features(audio, lyrics, view_mask)
    np.testing.assert_allclose(features[0, :2], [0.5, 0.5], atol=1e-6)  # (0.8+0.2)/2, (0.6+0.4)/2
    np.testing.assert_allclose(features[1, :2], [0.3, 0.7], atol=1e-6)  # lyrics-only


def test_diff_geometry_missing_pair_zeros_deltas():
    """When one view is missing, all pair-diff features should be zero."""
    audio = np.asarray([[0.7, 0.4]], dtype=np.float32)
    lyrics = np.asarray([[0.0, 0.0]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)

    features, diff_obs = build_diff_geometry_features(audio, lyrics, view_mask)
    # signed_delta, abs_delta, gaps, radiuses, SQP, uncertainty should all be zero
    np.testing.assert_allclose(features[0, 2:26], np.zeros(24), atol=1e-6)
    assert diff_obs[0, 0] == 0.0


def test_diff_observed_mask():
    """diff_observed should be 1.0 for both-pair, 0.0 otherwise."""
    audio = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    lyrics = np.asarray([[0.7, 0.8], [0.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    view_mask = np.asarray([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float32)

    _, diff_obs = build_diff_geometry_features(audio, lyrics, view_mask)
    np.testing.assert_allclose(diff_obs.ravel(), [1.0, 0.0, 0.0], atol=1e-6)


def test_soft_quadrant_pair_sum_to_one():
    """When both views present, SQP rows should sum to ~1.0."""
    audio = np.asarray([[0.8, 0.6], [0.2, 0.3], [0.55, 0.45]], dtype=np.float32)
    lyrics = np.asarray([[0.3, 0.7], [0.7, 0.5], [0.45, 0.55]], dtype=np.float32)
    view_mask = np.ones((3, 3), dtype=np.float32)

    features, diff_obs = build_diff_geometry_features(audio, lyrics, view_mask)
    sqp_cols = features[:, 9:25]
    for i in range(3):
        np.testing.assert_allclose(sqp_cols[i].sum(), 1.0, atol=1e-5)


def test_diff_geometry_reproducible_values():
    """Smoke test with known values for regression detection."""
    audio = np.asarray([[0.9, 0.8]], dtype=np.float32)
    lyrics = np.asarray([[0.1, 0.2]], dtype=np.float32)
    view_mask = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)

    features, diff_obs = build_diff_geometry_features(audio, lyrics, view_mask)

    # Consensus: (0.9+0.1)/2, (0.8+0.2)/2
    np.testing.assert_allclose(features[0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(features[0, 1], 0.5, atol=1e-6)

    # Signed delta
    np.testing.assert_allclose(features[0, 2], 0.8, atol=1e-6)
    np.testing.assert_allclose(features[0, 3], 0.6, atol=1e-6)

    # abs_delta
    np.testing.assert_allclose(features[0, 4], 0.8, atol=1e-6)
    np.testing.assert_allclose(features[0, 5], 0.6, atol=1e-6)

    # gap_norm = sqrt(0.8^2 + 0.6^2) / sqrt(2) = 1.0 / 1.414...
    expected_gap = math.sqrt(0.8**2 + 0.6**2) / math.sqrt(2.0)
    np.testing.assert_allclose(features[0, 6], expected_gap, atol=1e-6)

    # Verify uncertainty > 0.5 for large gap
    assert features[0, 25] > 0.5

    # diff_observed = 1.0
    assert diff_obs[0, 0] == 1.0
