import numpy as np
import pandas as pd

from cluster.pipeline.k_selection import KSelectionConfig, _select_best_index


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
