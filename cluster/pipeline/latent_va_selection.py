from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from cluster.models.two_view_latent_va_gmm import TwoViewLatentVAGMM
from cluster.pipeline.k_selection import (
    KSearchResult,
    KSelectionConfig,
    compute_affect_purity_metrics,
    compute_overlap_gate_metrics,
)


def _min_size_threshold(config: KSelectionConfig, n_samples: int) -> int:
    return max(
        int(config.min_cluster_size),
        int(np.ceil(float(config.min_cluster_size_ratio) * float(n_samples))),
    )


def _normalize(values: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if reverse:
        array = -array
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float64)
    lo = float(np.nanmin(array[finite]))
    hi = float(np.nanmax(array[finite]))
    if hi - lo <= 1e-12:
        return np.zeros_like(array, dtype=np.float64)
    out = (array - lo) / (hi - lo)
    return np.where(finite, out, 0.0)


def _posterior_margin(proba: np.ndarray) -> float:
    probabilities = np.asarray(proba, dtype=np.float64)
    if probabilities.shape[1] < 2:
        return 1.0
    top = np.sort(probabilities, axis=1)[:, -2:]
    return float(np.mean(top[:, 1] - top[:, 0]))


def _latent_stability(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    config: KSelectionConfig,
) -> Tuple[float, float]:
    runs = max(0, int(config.stability_runs))
    if runs <= 1:
        return 1.0, 0.0
    scores: List[float] = []
    for run_idx in range(1, runs):
        model = TwoViewLatentVAGMM(
            n_components=int(k),
            covariance_type="diag",
            learn_bias=bool(config.latent_learn_view_bias),
            share_view_noise=bool(config.latent_share_view_noise),
            alpha_prior_strength=float(config.latent_alpha_prior_strength),
            reg_covar=1e-5,
            n_init=max(1, min(3, int(config.n_init))),
            max_iter=max(20, min(100, int(config.latent_max_iter))),
            random_state=int(config.random_state) + int(run_idx) * 997,
        ).fit(audio_va, lyrics_va, view_mask)
        scores.append(float(adjusted_rand_score(labels, model.labels_)))
    if not scores:
        return 1.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def search_latent_va_gmm(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    config: KSelectionConfig,
    *,
    affect_labels: Optional[np.ndarray] = None,
    primary_va: Optional[np.ndarray] = None,
) -> KSearchResult:
    """Search K for the two-view latent VA mixture model."""
    audio = np.asarray(audio_va, dtype=np.float32)
    lyrics = np.asarray(lyrics_va, dtype=np.float32)
    mask = np.asarray(view_mask, dtype=np.float32)
    if audio.ndim != 2 or lyrics.ndim != 2 or audio.shape != lyrics.shape or audio.shape[1] != 2:
        raise ValueError(f"audio_va and lyrics_va must both have shape [N, 2], got {audio.shape} and {lyrics.shape}.")
    if mask.ndim != 2 or mask.shape[0] != audio.shape[0] or mask.shape[1] < 2:
        raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")
    mask = mask[:, :2]
    min_size_threshold = _min_size_threshold(config, int(audio.shape[0]))

    rows: List[Dict[str, Any]] = []
    candidates: Dict[int, TwoViewLatentVAGMM] = {}
    for k in range(int(config.k_min), int(config.k_max) + 1):
        if int(k) <= 0 or int(k) >= int(audio.shape[0]):
            continue
        model = TwoViewLatentVAGMM(
            n_components=int(k),
            covariance_type="diag",
            learn_bias=bool(config.latent_learn_view_bias),
            share_view_noise=bool(config.latent_share_view_noise),
            alpha_prior_strength=float(config.latent_alpha_prior_strength),
            reg_covar=1e-5,
            n_init=max(1, int(config.n_init)),
            max_iter=max(20, int(config.latent_max_iter)),
            random_state=int(config.random_state),
        ).fit(audio, lyrics, mask)
        labels = model.labels_.astype(np.int64)
        sizes = np.bincount(labels, minlength=int(k))
        consensus = model.posterior_consensus(audio, lyrics, mask)
        proba = model.predict_proba(audio, lyrics, mask)
        bic = float(model.bic(audio, lyrics, mask))
        icl = float(model.icl(audio, lyrics, mask))
        if np.unique(labels).size > 1:
            try:
                sil = float(silhouette_score(consensus.astype(np.float64), labels))
            except Exception:
                sil = float("nan")
        else:
            sil = 0.0
        stability_mean, stability_std = _latent_stability(audio, lyrics, mask, labels, k=int(k), config=config)
        overlap_va = consensus if primary_va is None else np.asarray(primary_va, dtype=np.float32)[:, :2]
        overlap_metrics = compute_overlap_gate_metrics(
            overlap_va,
            labels,
            min_va_knn_purity=float(config.min_va_knn_purity),
            min_va_center_sep=float(config.min_va_center_sep),
            max_negative_silhouette_fraction=float(config.max_va_negative_silhouette_fraction),
            silhouette_sample_size=int(getattr(config, "silhouette_sample_size", 0)),
            random_state=int(config.random_state),
        )
        affect_metrics: Dict[str, Any] = {}
        if config.affect_gate_enabled:
            affect_metrics = compute_affect_purity_metrics(
                labels,
                affect_labels,
                min_dominant_ratio=float(config.min_affect_dominant_ratio),
                max_mixed_cluster_fraction=float(config.max_affect_mixed_cluster_fraction),
                min_weighted_purity=float(config.min_affect_weighted_purity),
                min_valid_fraction=float(config.min_affect_valid_fraction),
            )
        candidates[int(k)] = model
        rows.append(
            {
                "k": int(k),
                "bic": bic,
                "icl": icl,
                "latent_consensus_silhouette": float(sil),
                "posterior_margin_mean": _posterior_margin(proba),
                "seed_ari_mean": float(stability_mean),
                "seed_ari_std": float(stability_std),
                "min_cluster_size": int(sizes.min()) if sizes.size else 0,
                "min_size_ok": bool(sizes.size and sizes.min() >= min_size_threshold),
                "size_balance": float(sizes.min() / max(sizes.max(), 1)) if sizes.size else 0.0,
                **overlap_metrics,
                **affect_metrics,
            }
        )

    if not rows:
        raise ValueError("No latent_va_gmm K candidates were evaluated.")
    metrics = pd.DataFrame(rows).sort_values("k", kind="stable").reset_index(drop=True)
    score = (
        0.35 * _normalize(metrics["icl"].to_numpy(), reverse=True)
        + 0.20 * _normalize(metrics["latent_consensus_silhouette"].to_numpy())
        + 0.15 * _normalize(metrics["seed_ari_mean"].to_numpy())
        + 0.15 * np.nan_to_num(metrics["va_knn_purity_20"].to_numpy(dtype=np.float64), nan=0.0)
        + 0.10 * np.minimum(1.0, np.nan_to_num(metrics["va_center_radius_sep"].to_numpy(dtype=np.float64), nan=0.0))
        + 0.05 * _normalize(metrics["posterior_margin_mean"].to_numpy())
    )
    score -= 0.05 * np.nan_to_num(metrics["va_negative_silhouette_fraction"].to_numpy(dtype=np.float64), nan=1.0)
    metrics["latent_va_score"] = score.astype(np.float64)
    eligible = metrics["min_size_ok"].to_numpy(dtype=bool)
    if bool(config.overlap_gate_enabled):
        eligible &= metrics["overlap_gate_ok"].to_numpy(dtype=bool)
    if not eligible.any():
        if bool(config.diagnostic_allow_failed_gates):
            fallback_pool = metrics["min_size_ok"].to_numpy(dtype=bool)
            if not bool(fallback_pool.any()):
                fallback_pool = np.ones(len(metrics), dtype=bool)
            selected_idx = int(metrics[fallback_pool]["latent_va_score"].idxmax())
            metrics.attrs["diagnostic_failed_gate_override"] = {
                "selection_mode": "latent_va_gmm",
                "selected_k": int(metrics.loc[selected_idx, "k"]),
                "selected_index": selected_idx,
                "eligible_candidate_count": int(eligible.sum()),
                "fallback_candidate_count": int(fallback_pool.sum()),
                "reason": "No candidate satisfied hard gates; selected the best-scoring candidate for diagnostic ablation only.",
            }
        else:
            candidates_text = ", ".join(
                (
                    f"k={int(row.k)}:min_size={int(row.min_cluster_size)},"
                    f"min_ok={bool(row.min_size_ok)},"
                    f"overlap_ok={bool(getattr(row, 'overlap_gate_ok', True))},"
                    f"va_purity20={float(getattr(row, 'va_knn_purity_20', float('nan'))):.3f},"
                    f"va_sep={float(getattr(row, 'va_center_radius_sep', float('nan'))):.3f},"
                    f"score={float(getattr(row, 'latent_va_score', float('nan'))):.3f}"
                )
                for row in metrics.itertuples(index=False)
            )
            raise ValueError(
                "No latent_va_gmm candidate satisfied min-size and overlap hard gates "
                f"(k_min={int(config.k_min)}, k_max={int(config.k_max)}, "
                f"min_cluster_size_threshold={min_size_threshold}). Candidates: {candidates_text}."
            )
    else:
        selected_idx = int(metrics[eligible]["latent_va_score"].idxmax())
    selected_row = metrics.loc[selected_idx]
    selected_k = int(selected_row["k"])
    best_model = candidates[selected_k]
    selection_info = {
        "selected_k": selected_k,
        "selection_mode": "latent_va_gmm",
        "latent_va_score": float(selected_row["latent_va_score"]),
        "bic": float(selected_row["bic"]),
        "icl": float(selected_row["icl"]),
        "latent_consensus_silhouette": float(selected_row["latent_consensus_silhouette"]),
        "posterior_margin_mean": float(selected_row["posterior_margin_mean"]),
        "seed_ari_mean": float(selected_row["seed_ari_mean"]),
        "seed_ari_std": float(selected_row["seed_ari_std"]),
        "min_cluster_size_threshold": int(min_size_threshold),
        "min_cluster_size": int(selected_row["min_cluster_size"]),
        "min_size_ok": bool(selected_row["min_size_ok"]),
        "overlap_gate_enabled": bool(config.overlap_gate_enabled),
        "overlap_gate_ok": bool(selected_row.get("overlap_gate_ok", True)),
        "va_knn_purity_10": float(selected_row.get("va_knn_purity_10", float("nan"))),
        "va_knn_purity_20": float(selected_row.get("va_knn_purity_20", float("nan"))),
        "va_center_radius_sep": float(selected_row.get("va_center_radius_sep", float("nan"))),
        "va_negative_silhouette_fraction": float(selected_row.get("va_negative_silhouette_fraction", float("nan"))),
        "va_mean_silhouette": float(selected_row.get("va_mean_silhouette", float("nan"))),
        "latent_learn_view_bias": bool(config.latent_learn_view_bias),
        "latent_share_view_noise": bool(config.latent_share_view_noise),
        "latent_alpha_prior_strength": float(config.latent_alpha_prior_strength),
        "cluster_backend": "two_view_latent_va_gmm",
        "eval_backend": config.eval_backend,
        "actual_cluster_backend": "two_view_latent_va_gmm",
        "actual_eval_backend": config.eval_backend,
        "device": str(config.device),
        "affect_gate_diagnostic_only": bool(config.affect_gate_enabled),
    }
    override = metrics.attrs.get("diagnostic_failed_gate_override")
    if override:
        selection_info["diagnostic_failed_gate_override"] = True
        for key, value in dict(override).items():
            selection_info[f"diagnostic_failed_gate_{key}"] = value
    if config.affect_gate_enabled:
        for field in (
            "affect_valid_fraction",
            "affect_weighted_dominant_ratio",
            "affect_min_dominant_ratio",
            "affect_mixed_cluster_fraction",
            "affect_nmi",
            "affect_min_dominant_gate_ok",
            "affect_worst_cluster_id",
            "affect_worst_cluster_size",
            "affect_gate_ok",
        ):
            value = selected_row.get(field, float("nan"))
            if field.endswith("_ok"):
                selection_info[field] = bool(value)
            elif field.endswith("_id") or field.endswith("_size"):
                selection_info[field] = int(value) if pd.notna(value) else -1
            else:
                selection_info[field] = float(value) if pd.notna(value) else float("nan")
    return KSearchResult(
        best_k=selected_k,
        best_model=best_model,
        metrics=metrics,
        selection_info=selection_info,
    )
