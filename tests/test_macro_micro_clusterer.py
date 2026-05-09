import numpy as np
import pytest

from cluster.pipeline.train import run_k_selection


def _macro_micro_fixture() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(7)
    rows = []
    for macro_center in (-5.0, 5.0):
        for micro_center in (-2.0, 2.0):
            for _ in range(8):
                consensus = np.asarray(
                    [macro_center + rng.normal(0.0, 0.08), rng.normal(0.0, 0.08)],
                    dtype=np.float32,
                )
                tension = np.asarray(
                    [micro_center + rng.normal(0.0, 0.06), rng.normal(0.0, 0.06)],
                    dtype=np.float32,
                )
                metadata = np.asarray(
                    [micro_center * 0.5 + rng.normal(0.0, 0.05), rng.normal(0.0, 0.05)],
                    dtype=np.float32,
                )
                rows.append(np.concatenate([consensus, tension, metadata]))
    features = np.vstack(rows).astype(np.float32)
    block_mask = np.ones((features.shape[0], 3), dtype=bool)
    return features, block_mask, [(0, 2), (2, 4), (4, 6)]


def test_macro_micro_k_strategy_fits_predictable_two_level_model():
    features, block_mask, block_slices = _macro_micro_fixture()

    model, metrics, info = run_k_selection(
        features=features,
        k_strategy="macro_micro",
        k_min=2,
        k_max=4,
        random_state=7,
        min_cluster_size_abs=4,
        min_cluster_size_ratio=0.0,
        covariance_type="diag",
        stability_runs=1,
        cluster_backend="sklearn",
        eval_backend="sklearn",
        silhouette_mode="sampled",
        silhouette_sample_size=0,
        assignment_mode="partial_likelihood",
        block_mask=block_mask,
        block_slices=block_slices,
        macro_k_min=2,
        macro_k_max=2,
        micro_k_min=1,
        micro_k_max=2,
    )

    labels = model.predict(features, block_mask=block_mask)

    assert model.__class__.__name__ == "MacroMicroClusterer"
    assert model.macro_k == 2
    assert model.n_components == 4
    assert len(np.unique(labels)) == 4
    assert info["selection_mode"] == "macro_micro_diffaware"
    assert info["macro_k"] == 2
    assert info["selected_k"] == 4
    assert "macro_micro_score" in metrics.columns

    for cluster_id in np.unique(labels):
        macro_ids = np.unique(model.macro_labels_[labels == cluster_id])
        assert macro_ids.size == 1


def test_macro_micro_honors_total_k_constraints():
    features, block_mask, block_slices = _macro_micro_fixture()

    with pytest.raises(ValueError, match="total K"):
        run_k_selection(
            features=features,
            k_strategy="macro_micro",
            k_min=5,
            k_max=6,
            random_state=7,
            min_cluster_size_abs=4,
            min_cluster_size_ratio=0.0,
            covariance_type="diag",
            stability_runs=1,
            cluster_backend="sklearn",
            eval_backend="sklearn",
            silhouette_mode="sampled",
            silhouette_sample_size=0,
            assignment_mode="partial_likelihood",
            block_mask=block_mask,
            block_slices=block_slices,
            macro_k_min=2,
            macro_k_max=2,
            micro_k_min=1,
            micro_k_max=2,
        )


def test_macro_micro_reports_bootstrap_stability_when_requested():
    features, block_mask, block_slices = _macro_micro_fixture()

    _model, metrics, info = run_k_selection(
        features=features,
        k_strategy="macro_micro",
        k_min=4,
        k_max=4,
        random_state=7,
        min_cluster_size_abs=4,
        min_cluster_size_ratio=0.0,
        covariance_type="diag",
        stability_runs=3,
        cluster_backend="sklearn",
        eval_backend="sklearn",
        silhouette_mode="sampled",
        silhouette_sample_size=0,
        assignment_mode="partial_likelihood",
        block_mask=block_mask,
        block_slices=block_slices,
        macro_k_min=2,
        macro_k_max=2,
        micro_k_min=1,
        micro_k_max=2,
    )

    for column in (
        "seed_ari_mean",
        "seed_ari_std",
        "cluster_jaccard_mean",
        "cluster_jaccard_min",
        "bootstrap_valid_rate",
    ):
        assert column in metrics.columns
        assert column in info
    assert 0.0 <= info["seed_ari_mean"] <= 1.0
    assert info["bootstrap_valid_rate"] > 0.0
