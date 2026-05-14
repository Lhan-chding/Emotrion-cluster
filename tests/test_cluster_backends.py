import numpy as np
import pytest
import torch
from sklearn.metrics import silhouette_samples, silhouette_score

from cluster.backends import resolve_cluster_backend
from cluster.backends.torch_gmm_backend import TorchGaussianMixture
from cluster.backends.torch_metrics import torch_silhouette_samples_chunked, torch_silhouette_score_chunked
from cluster.pipeline.k_selection import KSelectionConfig, _score_silhouette, search_gmm_bic_only, search_gmm_composite


def _two_blob_features():
    return np.asarray(
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


def test_torch_gaussian_mixture_fits_predicts_and_scores():
    features = _two_blob_features()

    model = TorchGaussianMixture(
        n_components=2,
        covariance_type="diag",
        n_init=3,
        max_iter=60,
        random_state=7,
    ).fit(features)

    labels = model.predict(features)
    probabilities = model.predict_proba(features)

    assert labels.shape == (features.shape[0],)
    assert probabilities.shape == (features.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(features.shape[0]), atol=1e-5)
    assert len(np.unique(labels)) == 2
    assert np.isfinite(model.bic(features))
    assert np.isfinite(model.aic(features))


def test_torch_gmm_bic_uses_float64_scoring_formula():
    rng = np.random.default_rng(42)
    features = rng.normal(loc=0.0, scale=3.0, size=(24, 48)).astype(np.float64)
    means = np.stack(
        [
            np.linspace(-1.0, 1.0, features.shape[1]),
            np.linspace(1.0, -1.0, features.shape[1]),
        ],
        axis=0,
    )
    variances = np.full_like(means, 0.07, dtype=np.float64)
    weights = np.asarray([0.35, 0.65], dtype=np.float64)
    model = TorchGaussianMixture(n_components=2, covariance_type="diag")
    model.means_ = means.astype(np.float32)
    model.covariances_ = variances.astype(np.float32)
    model.weights_ = weights.astype(np.float32)

    log_components = []
    for cluster_id in range(2):
        diff = features - model.means_[cluster_id].astype(np.float64)
        log_det = np.log(model.covariances_[cluster_id].astype(np.float64)).sum()
        mahal = (diff * diff / model.covariances_[cluster_id].astype(np.float64)).sum(axis=1)
        log_components.append(
            np.log(float(model.weights_[cluster_id]))
            - 0.5 * (features.shape[1] * np.log(2.0 * np.pi) + log_det + mahal)
        )
    stacked = np.stack(log_components, axis=1)
    max_log = np.max(stacked, axis=1, keepdims=True)
    log_likelihood = float(np.sum(max_log[:, 0] + np.log(np.exp(stacked - max_log).sum(axis=1))))
    expected_bic = -2.0 * log_likelihood + model._n_parameters(features.shape[1]) * np.log(features.shape[0])

    assert abs(model.bic(features) - expected_bic) < 1e-6


def test_torch_chunked_silhouette_matches_sklearn():
    features = _two_blob_features()
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    expected = silhouette_score(features, labels)
    actual = torch_silhouette_score_chunked(features, labels, chunk_size=2)

    assert abs(actual - expected) < 1e-5


def test_torch_chunked_silhouette_samples_match_sklearn():
    features = _two_blob_features()
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    expected = silhouette_samples(features, labels)
    actual = torch_silhouette_samples_chunked(features, labels, chunk_size=2)

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_sampled_silhouette_for_torch_eval_uses_sklearn_sampling():
    rng = np.random.default_rng(7)
    features = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.4, size=(8, 3)),
            rng.normal(loc=2.0, scale=0.4, size=(8, 3)),
            rng.normal(loc=5.0, scale=0.4, size=(8, 3)),
        ]
    ).astype(np.float32)
    labels = np.repeat(np.arange(3), 8).astype(np.int64)
    config = KSelectionConfig(
        eval_backend="torch",
        silhouette_mode="sampled",
        silhouette_sample_size=9,
        random_state=11,
    )

    expected = silhouette_score(features, labels, sample_size=9, random_state=11)
    actual = _score_silhouette(features, labels, config)

    assert abs(actual - expected) < 1e-6


def test_resolve_cluster_backend_falls_back_to_sklearn_for_auto_cpu():
    backend = resolve_cluster_backend("auto", device="cpu")

    assert backend.name == "sklearn"
    labels, model = backend.fit_predict(_two_blob_features(), algorithm="kmeans", n_clusters=2, random_state=7)
    assert labels.shape == (6,)
    assert model is not None


def test_torch_backend_rejects_unavailable_cuda_device():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment.")

    with pytest.raises(RuntimeError, match="CUDA device"):
        resolve_cluster_backend("torch", device="cuda")


def test_bic_search_can_use_torch_backend():
    features = _two_blob_features()
    result = search_gmm_bic_only(
        features=features,
        k_min=2,
        k_max=2,
        random_state=7,
        min_cluster_size_abs=1,
        min_cluster_size_ratio=0.0,
        covariance_type="diag",
        n_init=2,
        config=KSelectionConfig(cluster_backend="torch", eval_backend="torch", silhouette_mode="torch_chunked"),
    )

    assert result.best_k == 2
    assert result.selection_info["cluster_backend"] == "torch"
    assert result.selection_info["eval_backend"] == "torch"
    assert result.selection_info["actual_cluster_backend"] == "torch"
    assert result.selection_info["actual_eval_backend"] == "torch"
    assert result.selection_info["device"] == "cpu"
    assert np.isfinite(float(result.metrics.loc[0, "bic"]))


def test_composite_k_selection_fails_when_no_k_satisfies_min_size():
    features = _two_blob_features()

    with pytest.raises(ValueError, match="No K candidate satisfied"):
        search_gmm_composite(
            features,
            KSelectionConfig(
                k_min=2,
                k_max=2,
                min_cluster_size=4,
                min_cluster_size_ratio=0.0,
                stability_runs=1,
                cluster_backend="sklearn",
                eval_backend="sklearn",
                silhouette_mode="sampled",
                silhouette_sample_size=0,
            ),
        )
