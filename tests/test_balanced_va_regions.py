import numpy as np

from cluster.pipeline.train import build_parser, run_k_selection


def _four_region_fixture() -> np.ndarray:
    rng = np.random.default_rng(44)
    centers = np.asarray(
        [
            [0.24, 0.25],
            [0.26, 0.76],
            [0.76, 0.25],
            [0.78, 0.76],
        ],
        dtype=np.float32,
    )
    consensus = np.vstack(
        [center + rng.normal(0.0, 0.045, size=(70, 2)) for center in centers]
    ).astype(np.float32)
    tail = np.asarray(
        [
            [0.05, 0.95],
            [0.08, 0.92],
            [0.95, 0.05],
            [0.92, 0.08],
            [0.50, 0.50],
            [0.51, 0.49],
        ],
        dtype=np.float32,
    )
    consensus = np.clip(np.vstack([consensus, tail]), 0.0, 1.0)
    tension = rng.normal(0.0, 0.05, size=(consensus.shape[0], 3)).astype(np.float32)
    tension[-len(tail) :] += np.asarray([0.30, -0.15, 0.35], dtype=np.float32)
    return np.concatenate([consensus, tension], axis=1).astype(np.float32)


def test_parser_accepts_balanced_va_regions_strategy():
    args = build_parser().parse_args(
        [
            "--processed_dir",
            "processed",
            "--out_dir",
            "out",
            "--cluster_feature_strategy",
            "calibrated_va_tension",
            "--k_strategy",
            "balanced_va_regions",
            "--plot_va_source",
            "cluster_consensus",
            "--region_max_iter",
            "60",
        ]
    )

    assert args.k_strategy == "balanced_va_regions"
    assert args.region_max_iter == 60


def test_balanced_va_regions_selects_k_without_small_tail_clusters():
    features = _four_region_fixture()

    model, metrics, info = run_k_selection(
        features=features,
        k_strategy="balanced_va_regions",
        k_min=4,
        k_max=4,
        random_state=44,
        min_cluster_size_abs=50,
        min_cluster_size_ratio=0.0,
        stability_runs=4,
        primary_va=features[:, :2],
        region_max_iter=80,
    )

    labels = model.predict(features)
    sizes = np.bincount(labels, minlength=4)
    assert info["selection_mode"] == "balanced_va_regions"
    assert info["actual_cluster_backend"] == "balanced_va_region_kmeans"
    assert info["anti_collapse_min_size_hard_gate"] is True
    assert int(sizes.min()) >= 50
    assert info["min_cluster_size"] >= 50
    assert {"balanced_region_score", "size_balance", "tension_effect_ratio"}.issubset(metrics.columns)
    assert metrics.loc[0, "min_size_ok"] is True or bool(metrics.loc[0, "min_size_ok"]) is True
