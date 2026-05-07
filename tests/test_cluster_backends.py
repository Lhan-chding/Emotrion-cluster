import numpy as np
from sklearn.metrics import silhouette_score

from cluster.backends import resolve_cluster_backend
from cluster.backends.torch_gmm_backend import TorchGaussianMixture
from cluster.backends.torch_metrics import torch_silhouette_score_chunked
from cluster.pipeline.k_selection import KSelectionConfig, search_gmm_bic_only


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


def test_torch_chunked_silhouette_matches_sklearn():
    features = _two_blob_features()
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    expected = silhouette_score(features, labels)
    actual = torch_silhouette_score_chunked(features, labels, chunk_size=2)

    assert abs(actual - expected) < 1e-5


def test_resolve_cluster_backend_falls_back_to_sklearn_for_auto_cpu():
    backend = resolve_cluster_backend("auto", device="cpu")

    assert backend.name == "sklearn"
    labels, model = backend.fit_predict(_two_blob_features(), algorithm="kmeans", n_clusters=2, random_state=7)
    assert labels.shape == (6,)
    assert model is not None


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
    assert np.isfinite(float(result.metrics.loc[0, "bic"]))

