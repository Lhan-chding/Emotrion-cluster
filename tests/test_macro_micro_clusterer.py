import numpy as np
import pandas as pd
import pytest

from cluster.pipeline.k_selection import (
    _select_best_index,
    compute_affect_purity_metrics,
    compute_overlap_gate_metrics,
    KSelectionConfig,
)
from cluster.pipeline.macro_micro import MicroSplitValidator
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


def _two_block_va_diff_fixture() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(11)
    rows = []
    for macro_center in (-4.0, 4.0):
        for diff_center in (-1.5, 1.5):
            for _ in range(10):
                consensus = np.asarray(
                    [macro_center + rng.normal(0.0, 0.05), rng.normal(0.0, 0.05)],
                    dtype=np.float32,
                )
                diff = np.asarray(
                    [diff_center + rng.normal(0.0, 0.04), rng.normal(0.0, 0.04)],
                    dtype=np.float32,
                )
                rows.append(np.concatenate([consensus, diff]))
    features = np.vstack(rows).astype(np.float32)
    block_mask = np.ones((features.shape[0], 2), dtype=bool)
    return features, block_mask, [(0, 2), (2, 4)]


def _macro_pure_micro_mixed_fixture() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    rng = np.random.default_rng(17)
    rows = []
    labels = []
    macro_specs = [(-4.0, 0, 1), (4.0, 2, 3)]
    for macro_center, dominant_label, minority_label in macro_specs:
        for diff_center, cluster_labels in (
            (-1.5, [dominant_label] * 6 + [minority_label] * 4),
            (1.5, [dominant_label] * 10),
        ):
            for affect_label in cluster_labels:
                consensus = np.asarray(
                    [macro_center + rng.normal(0.0, 0.04), rng.normal(0.0, 0.04)],
                    dtype=np.float32,
                )
                diff = np.asarray(
                    [diff_center + rng.normal(0.0, 0.04), rng.normal(0.0, 0.04)],
                    dtype=np.float32,
                )
                rows.append(np.concatenate([consensus, diff]))
                labels.append(affect_label)
    features = np.vstack(rows).astype(np.float32)
    block_mask = np.ones((features.shape[0], 2), dtype=bool)
    return features, block_mask, [(0, 2), (2, 4)], np.asarray(labels, dtype=np.int64)


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


def test_macro_micro_affect_gate_applies_to_macro_regions_not_diff_subclusters():
    features, block_mask, block_slices, affect_labels = _macro_pure_micro_mixed_fixture()

    model, metrics, info = run_k_selection(
        features=features,
        k_strategy="macro_micro",
        k_min=4,
        k_max=4,
        random_state=17,
        min_cluster_size_abs=5,
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
        affect_labels=affect_labels,
        affect_gate_enabled=True,
        min_affect_dominant_ratio=0.70,
        min_affect_weighted_purity=0.80,
        max_affect_mixed_cluster_fraction=0.15,
        affect_gate_level="macro",
    )

    assert model.n_components == 4
    assert info["affect_gate_level"] == "macro"
    assert info["affect_gate_ok"] is True
    assert info["final_affect_gate_ok"] is False
    assert bool(metrics.loc[0, "affect_gate_ok"]) is True
    assert bool(metrics.loc[0, "final_affect_gate_ok"]) is False


def test_macro_micro_final_affect_gate_rejects_incoherent_micro_splits():
    features, block_mask, block_slices, affect_labels = _macro_pure_micro_mixed_fixture()

    model, metrics, info = run_k_selection(
        features=features,
        k_strategy="macro_micro",
        k_min=2,
        k_max=2,
        random_state=17,
        min_cluster_size_abs=5,
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
        affect_labels=affect_labels,
        affect_gate_enabled=True,
        min_affect_dominant_ratio=0.70,
        min_affect_weighted_purity=0.80,
        max_affect_mixed_cluster_fraction=0.15,
        affect_gate_level="both",
    )

    labels = model.predict(features, block_mask=block_mask)

    assert model.n_components == 2
    assert len(np.unique(labels)) == 2
    assert info["affect_gate_level"] == "both"
    assert info["affect_gate_ok"] is True
    assert info["macro_affect_gate_ok"] is True
    assert info["final_affect_gate_ok"] is True
    assert bool(metrics.loc[0, "affect_gate_ok"]) is True
    assert bool(metrics.loc[0, "macro_affect_gate_ok"]) is True
    assert bool(metrics.loc[0, "final_affect_gate_ok"]) is True


def test_macro_micro_affect_gate_error_reports_valid_fraction():
    features, block_mask, block_slices, affect_labels = _macro_pure_micro_mixed_fixture()
    sparse_labels = affect_labels.copy()
    sparse_labels[::2] = -1

    with pytest.raises(ValueError, match=r"macro_valid=0\.500"):
        run_k_selection(
            features=features,
            k_strategy="macro_micro",
            k_min=4,
            k_max=4,
            random_state=17,
            min_cluster_size_abs=5,
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
            affect_labels=sparse_labels,
            affect_gate_enabled=True,
            min_affect_dominant_ratio=0.70,
            min_affect_weighted_purity=0.80,
            max_affect_mixed_cluster_fraction=0.15,
            min_affect_valid_fraction=0.90,
        )


def test_micro_split_validator_uses_all_micro_labels_for_effect_size():
    validator = MicroSplitValidator()
    features = np.asarray(
        [
            [0.0],
            [0.1],
            [0.0],
            [0.1],
            [6.0],
            [6.1],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    effect = validator._effect_size(features, labels)

    assert effect > 50.0


def test_visible_separator_rejects_hidden_tension_only_split():
    validator = MicroSplitValidator(
        consensus_role="visible_separator",
        min_silhouette=0.0,
        min_tension_effect=0.10,
        min_consensus_knn_purity=0.85,
        min_consensus_center_sep=0.60,
        min_consensus_silhouette=0.10,
    )
    labels = np.asarray([0] * 20 + [1] * 20, dtype=np.int64)
    consensus = np.zeros((40, 2), dtype=np.float32)
    tension = np.vstack(
        [
            np.tile([-2.0, 0.0], (20, 1)),
            np.tile([2.0, 0.0], (20, 1)),
        ]
    ).astype(np.float32)

    result = validator.evaluate(
        features=tension,
        labels=labels,
        consensus=consensus,
        tension=tension,
        silhouette=0.8,
    )

    assert result["accepted"] is False
    assert "consensus_center_sep" in str(result["rejection_reason"])
    assert result["consensus_center_radius_sep"] == 0.0


def test_visible_separator_accepts_tension_split_visible_in_consensus_plane():
    validator = MicroSplitValidator(
        consensus_role="visible_separator",
        min_silhouette=0.0,
        min_tension_effect=0.10,
        min_consensus_knn_purity=0.85,
        min_consensus_center_sep=0.60,
        min_consensus_silhouette=0.10,
    )
    rng = np.random.default_rng(21)
    labels = np.asarray([0] * 20 + [1] * 20, dtype=np.int64)
    consensus = np.vstack(
        [
            rng.normal([-2.0, 0.0], 0.03, size=(20, 2)),
            rng.normal([2.0, 0.0], 0.03, size=(20, 2)),
        ]
    ).astype(np.float32)
    tension = np.vstack(
        [
            rng.normal([-1.0, 0.0], 0.03, size=(20, 2)),
            rng.normal([1.0, 0.0], 0.03, size=(20, 2)),
        ]
    ).astype(np.float32)

    result = validator.evaluate(
        features=tension,
        labels=labels,
        consensus=consensus,
        tension=tension,
        silhouette=0.8,
    )

    assert result["accepted"] is True
    assert result["consensus_knn_purity"] >= 0.95
    assert result["consensus_center_radius_sep"] >= 1.0


def test_anti_leakage_preserves_legacy_consensus_effect_rejection():
    validator = MicroSplitValidator(
        consensus_role="anti_leakage",
        min_silhouette=0.0,
        min_tension_effect=0.0,
        max_consensus_effect_ratio=0.50,
    )
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    consensus = np.vstack([np.zeros((10, 1)), np.ones((10, 1)) * 10.0]).astype(np.float32)
    tension = np.vstack([np.zeros((10, 1)), np.ones((10, 1)) * 0.1]).astype(np.float32)

    result = validator.evaluate(
        features=tension,
        labels=labels,
        consensus=consensus,
        tension=tension,
        silhouette=0.5,
    )

    assert result["accepted"] is False
    assert "consensus_effect" in str(result["rejection_reason"])


def test_overlap_gate_metrics_detect_va_mixing():
    consensus = np.vstack(
        [
            np.tile([-2.0, 0.0], (10, 1)),
            np.tile([-2.0, 0.0], (10, 1)),
            np.tile([2.0, 0.0], (10, 1)),
            np.tile([2.0, 0.0], (10, 1)),
        ]
    ).astype(np.float32)
    hidden_micro_labels = np.asarray([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10, dtype=np.int64)

    metrics = compute_overlap_gate_metrics(consensus, hidden_micro_labels)

    assert metrics["va_knn_purity_20"] < 0.90
    assert metrics["va_center_radius_sep"] < 0.60
    assert metrics["overlap_gate_ok"] is False


def test_macro_micro_k_strategy_supports_two_block_va_diff_features():
    features, block_mask, block_slices = _two_block_va_diff_fixture()

    model, metrics, info = run_k_selection(
        features=features,
        k_strategy="macro_micro",
        k_min=4,
        k_max=4,
        random_state=11,
        min_cluster_size_abs=5,
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
    assert info["selected_k"] == 4
    assert "macro_micro_score" in metrics.columns
    for cluster_id in np.unique(labels):
        macro_ids = np.unique(model.macro_labels_[labels == cluster_id])
        assert macro_ids.size == 1


def test_macro_micro_overlap_gate_rejects_hidden_tension_subclusters():
    features, block_mask, block_slices = _two_block_va_diff_fixture()

    with pytest.raises(ValueError, match="overlap"):
        run_k_selection(
            features=features,
            k_strategy="macro_micro",
            k_min=4,
            k_max=4,
            random_state=11,
            min_cluster_size_abs=5,
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
            overlap_gate_enabled=True,
            min_va_knn_purity=0.90,
            min_va_center_sep=0.60,
            max_va_negative_silhouette_fraction=0.10,
        )


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


def test_affect_purity_metrics_flag_large_mixed_va_clusters():
    cluster_labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    quadrant_labels = np.asarray([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)

    metrics = compute_affect_purity_metrics(
        cluster_labels,
        quadrant_labels,
        min_dominant_ratio=0.70,
        min_weighted_purity=0.80,
        max_mixed_cluster_fraction=0.15,
        min_valid_fraction=0.95,
    )

    assert metrics["affect_valid_fraction"] == 1.0
    assert metrics["affect_min_dominant_ratio"] == 0.5
    assert metrics["affect_weighted_dominant_ratio"] == 0.75
    assert metrics["affect_mixed_cluster_fraction"] == 0.5
    assert metrics["affect_gate_ok"] is False


def test_affect_gate_allows_limited_boundary_cluster_when_global_purity_ok():
    cluster_labels = np.asarray(
        [0] * 60 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100,
        dtype=np.int64,
    )
    quadrant_labels = np.asarray(
        [0] * 31 + [1] * 29 + [1] * 100 + [2] * 100 + [3] * 100 + [0] * 100,
        dtype=np.int64,
    )

    metrics = compute_affect_purity_metrics(
        cluster_labels,
        quadrant_labels,
        min_dominant_ratio=0.55,
        min_weighted_purity=0.75,
        max_mixed_cluster_fraction=0.20,
        min_valid_fraction=0.85,
    )

    assert metrics["affect_valid_fraction"] == 1.0
    assert metrics["affect_min_dominant_ratio"] == pytest.approx(31 / 60)
    assert metrics["affect_min_dominant_gate_ok"] is False
    assert metrics["affect_mixed_cluster_fraction"] == pytest.approx(60 / 460)
    assert metrics["affect_gate_ok"] is True
    assert metrics["affect_worst_cluster_id"] == 0
    assert metrics["affect_worst_cluster_size"] == 60


def test_micro_split_validator_allows_small_boundary_split_mass():
    validator = MicroSplitValidator(
        min_silhouette=0.0,
        min_tension_effect=0.0,
        max_consensus_effect_ratio=999.0,
        min_affect_dominant_ratio=0.55,
        min_affect_weighted_purity=0.75,
        max_affect_mixed_cluster_fraction=0.20,
    )
    labels = np.asarray([0, 0, 0, 0] + [1] * 16, dtype=np.int64)
    affect_labels = np.asarray([0, 0, 1, 1] + [2] * 16, dtype=np.int64)
    tension = np.asarray([[0.0], [0.1], [0.0], [0.1]] + [[4.0]] * 16, dtype=np.float32)
    consensus = np.zeros((20, 1), dtype=np.float32)

    result = validator.evaluate(
        features=tension,
        labels=labels,
        consensus=consensus,
        tension=tension,
        silhouette=0.5,
        affect_labels=affect_labels,
    )

    assert result["accepted"] is True
    assert result["affect_min_dominant_ratio"] == 0.5
    assert result["affect_mixed_cluster_fraction"] == 0.2


def test_affect_gate_filters_higher_scoring_but_mixed_candidate():
    metrics = pd.DataFrame(
        [
            {"k": 4, "min_size_ok": True, "affect_gate_ok": False},
            {"k": 5, "min_size_ok": True, "affect_gate_ok": True},
        ]
    )
    metrics.attrs["n_samples"] = 100
    scores = np.asarray([0.95, 0.40], dtype=np.float64)

    selected = _select_best_index(
        metrics,
        scores,
        KSelectionConfig(min_cluster_size=1, min_cluster_size_ratio=0.0),
        selection_mode="composite",
    )

    assert selected == 1
