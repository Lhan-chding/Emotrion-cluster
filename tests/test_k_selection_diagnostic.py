import pandas as pd
import numpy as np

from cluster.pipeline.k_selection import KSelectionConfig, _select_best_index, compute_overlap_gate_metrics


def test_select_best_index_can_diagnose_failed_hard_gates_without_raising():
    metrics = pd.DataFrame(
        [
            {
                "k": 4,
                "min_cluster_size": 120,
                "min_size_ok": True,
                "affect_gate_ok": False,
                "affect_weighted_dominant_ratio": 0.70,
                "affect_min_dominant_ratio": 0.50,
                "affect_mixed_cluster_fraction": 0.30,
            },
            {
                "k": 5,
                "min_cluster_size": 90,
                "min_size_ok": True,
                "affect_gate_ok": False,
                "affect_weighted_dominant_ratio": 0.72,
                "affect_min_dominant_ratio": 0.48,
                "affect_mixed_cluster_fraction": 0.28,
            },
        ]
    )
    metrics.attrs["n_samples"] = 500

    selected = _select_best_index(
        metrics,
        np.array([0.10, 0.40], dtype=np.float64),
        KSelectionConfig(min_cluster_size=40, diagnostic_allow_failed_gates=True),
        selection_mode="composite",
    )

    assert selected == 1
    assert metrics.attrs["diagnostic_failed_gate_override"]["selected_k"] == 5
    assert metrics.attrs["diagnostic_failed_gate_override"]["eligible_candidate_count"] == 0


def test_overlap_gate_metrics_supports_silhouette_sampling_and_knn_purity():
    rng = np.random.default_rng(7)
    left = rng.normal(loc=[0.2, 0.2], scale=0.03, size=(40, 2))
    right = rng.normal(loc=[0.8, 0.8], scale=0.03, size=(40, 2))
    coords = np.vstack([left, right]).astype(np.float32)
    labels = np.asarray([0] * 40 + [1] * 40, dtype=np.int64)

    metrics = compute_overlap_gate_metrics(
        coords,
        labels,
        min_va_knn_purity=0.9,
        min_va_center_sep=0.7,
        max_negative_silhouette_fraction=0.1,
        silhouette_sample_size=24,
        random_state=7,
    )

    assert metrics["va_silhouette_sample_size"] == 24
    assert metrics["va_knn_purity_20"] >= 0.9
    assert metrics["overlap_gate_ok"] is True


def test_overlap_gate_metrics_can_use_torch_eval_backend_on_cpu():
    rng = np.random.default_rng(13)
    left = rng.normal(loc=[0.2, 0.75], scale=0.03, size=(30, 2))
    right = rng.normal(loc=[0.82, 0.25], scale=0.03, size=(30, 2))
    coords = np.vstack([left, right]).astype(np.float32)
    labels = np.asarray([0] * 30 + [1] * 30, dtype=np.int64)

    metrics = compute_overlap_gate_metrics(
        coords,
        labels,
        min_va_knn_purity=0.9,
        min_va_center_sep=0.7,
        max_negative_silhouette_fraction=0.1,
        silhouette_sample_size=20,
        eval_backend="torch",
        device="cpu",
        chunk_size=16,
        random_state=13,
    )

    assert metrics["va_silhouette_sample_size"] == 20
    assert metrics["va_knn_purity_10"] >= 0.9
    assert metrics["overlap_gate_ok"] is True
