from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from cluster.features.affect_calibration import AffectCalibrator, BalanceAlphaLearner, DiffResidualizer

VA_GEOMETRY_OBSERVED_NAMES = [
    "mean_valence",
    "mean_arousal",
    "signed_delta_valence",
    "signed_delta_arousal",
    "abs_delta_valence",
    "abs_delta_arousal",
    "euclidean_gap",
    "manhattan_gap",
    "cosine_centered",
    "signed_angular_gap",
    "audio_radius",
    "lyrics_radius",
    "radial_gap",
    "rbf_consistency",
]

VA_GEOMETRY_MASK_NAMES = [
    "has_both_audio_lyrics",
    "has_audio",
    "has_lyrics",
]

VA_GEOMETRY_FEATURE_NAMES = VA_GEOMETRY_OBSERVED_NAMES + VA_GEOMETRY_MASK_NAMES

VA_GEOMETRY_OBSERVED_DIM = len(VA_GEOMETRY_OBSERVED_NAMES)

BALANCED_VA_DIFF_FEATURE_NAMES = [
    "consensus_valence",
    "consensus_arousal",
    "signed_delta_valence",
    "signed_delta_arousal",
    "euclidean_gap",
    "signed_angular_gap",
    "radial_gap",
    "rbf_consistency",
]
BALANCED_VA_DIFF_DIM = len(BALANCED_VA_DIFF_FEATURE_NAMES)


def _as_va_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2], got {matrix.shape}.")
    return matrix


def _as_view_mask(values: np.ndarray, n_samples: int) -> np.ndarray:
    mask = np.asarray(values, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[0] != n_samples or mask.shape[1] < 2:
        raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")
    return mask


def _compute_geometry_core(
    audio: np.ndarray,
    lyrics: np.ndarray,
    has_audio: np.ndarray,
    has_lyrics: np.ndarray,
    has_both: np.ndarray,
    *,
    neutral_point: Sequence[float] = (0.5, 0.5),
    consistency_sigma: float = 0.35,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core geometry computation. Returns (observed_features[N,14], mask_features[N,3], consensus[N,2])."""
    neutral = np.asarray(neutral_point, dtype=np.float32)

    weights = has_audio + has_lyrics
    consensus = np.divide(
        audio * has_audio + lyrics * has_lyrics,
        np.maximum(weights, 1.0),
        out=np.tile(neutral.reshape(1, 2), (audio.shape[0], 1)).astype(np.float32),
        where=weights > 0.0,
    )

    signed_delta = (audio - lyrics) * has_both
    abs_delta = np.abs(signed_delta)
    euclidean_gap = (np.linalg.norm(signed_delta, axis=1, keepdims=True) / math.sqrt(2.0)).astype(np.float32)
    manhattan_gap = abs_delta.mean(axis=1, keepdims=True).astype(np.float32)

    audio_centered = audio - neutral.reshape(1, 2)
    lyrics_centered = lyrics - neutral.reshape(1, 2)
    audio_radius = np.linalg.norm(audio_centered, axis=1, keepdims=True).astype(np.float32) * has_both
    lyrics_radius = np.linalg.norm(lyrics_centered, axis=1, keepdims=True).astype(np.float32) * has_both
    radial_gap = (audio_radius - lyrics_radius).astype(np.float32)

    dot = np.sum(audio_centered * lyrics_centered, axis=1, keepdims=True).astype(np.float32)
    det = (
        audio_centered[:, 0:1] * lyrics_centered[:, 1:2]
        - audio_centered[:, 1:2] * lyrics_centered[:, 0:1]
    ).astype(np.float32)
    denom = np.maximum(audio_radius * lyrics_radius, float(eps))
    cosine = np.clip(dot / denom, -1.0, 1.0).astype(np.float32) * has_both
    angular_gap = (np.arctan2(det, dot) / math.pi).astype(np.float32) * has_both

    sigma = max(float(consistency_sigma), float(eps))
    rbf_consistency = np.exp(-0.5 * (euclidean_gap / sigma) ** 2).astype(np.float32) * has_both

    observed_features = np.concatenate(
        [
            consensus.astype(np.float32),
            signed_delta.astype(np.float32),
            abs_delta.astype(np.float32),
            euclidean_gap.astype(np.float32),
            manhattan_gap.astype(np.float32),
            cosine.astype(np.float32),
            angular_gap.astype(np.float32),
            audio_radius.astype(np.float32),
            lyrics_radius.astype(np.float32),
            radial_gap.astype(np.float32),
            rbf_consistency.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    mask_features = np.concatenate(
        [has_both, has_audio, has_lyrics],
        axis=1,
    ).astype(np.float32)

    return observed_features, mask_features, consensus


def build_va_geometry_observed_features(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    *,
    neutral_point: Sequence[float] = (0.5, 0.5),
    consistency_sigma: float = 0.35,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build 14-dim observed geometry features WITHOUT mask indicators."""
    audio = _as_va_matrix("audio_va", audio_va)
    lyrics = _as_va_matrix("lyrics_va", lyrics_va)
    if lyrics.shape[0] != audio.shape[0]:
        raise ValueError(f"audio_va and lyrics_va must have same N, got {audio.shape[0]} and {lyrics.shape[0]}.")
    mask = _as_view_mask(view_mask, audio.shape[0])

    has_audio = (mask[:, 0:1] > 0.0).astype(np.float32)
    has_lyrics = (mask[:, 1:2] > 0.0).astype(np.float32)
    has_both = has_audio * has_lyrics

    observed, _, _ = _compute_geometry_core(
        audio, lyrics, has_audio, has_lyrics, has_both,
        neutral_point=neutral_point, consistency_sigma=consistency_sigma, eps=eps,
    )
    return observed


def build_va_geometry_mask(
    view_mask: np.ndarray,
) -> np.ndarray:
    """Return [N, 3] mask features: has_both, has_audio, has_lyrics."""
    mask = np.asarray(view_mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[1] < 2:
        raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")
    has_audio = (mask[:, 0:1] > 0.0).astype(np.float32)
    has_lyrics = (mask[:, 1:2] > 0.0).astype(np.float32)
    has_both = has_audio * has_lyrics
    return np.concatenate([has_both, has_audio, has_lyrics], axis=1).astype(np.float32)


def impute_unobserved_pairwise(
    features: np.ndarray,
    view_mask: np.ndarray,
    fitted_fill: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute pairwise geometry dims (2:14) for rows missing both views.

    Unobserved rows are filled with the mean of their k-nearest observed
    neighbours in consensus (dims 0:1) space. This avoids zero-variance
    columns that GMMs can exploit.

    Parameters
    ----------
    features : ndarray [N, 14]
        The 14-dim observed geometry features.
    view_mask : ndarray [N, >=2]
        View availability mask (col 0=audio, col 1=lyrics).
    fitted_fill : ndarray [12] or None
        Pre-computed fill values for dims 2:14. If None, computed from
        observed rows in this batch (fit mode).
        In fit mode fitted_fill is a dummy — actual filling is per-row.

    Returns
    -------
    (imputed_features, fill_values) where fill_values has shape [12].
    """
    features = np.asarray(features, dtype=np.float32)
    mask = np.asarray(view_mask, dtype=np.float32)
    has_both = ((mask[:, 0] > 0.0) & (mask[:, 1] > 0.0))
    unobserved = ~has_both

    if not unobserved.any():
        return features, np.zeros(12, dtype=np.float32)

    # Transform mode: reuse pre-computed fill for backward compat
    if fitted_fill is not None and not np.allclose(fitted_fill, 0):
        fill = np.asarray(fitted_fill, dtype=np.float32)
        features[unobserved, 2:14] = fill
        return features, fill

    observed_indices = np.where(has_both)[0]
    unobserved_indices = np.where(unobserved)[0]
    consensus_all = features[:, :2].astype(np.float64)

    if len(observed_indices) == 0:
        rng = np.random.default_rng(42)
        for i in unobserved_indices:
            features[i, 2:14] = rng.normal(0, 1e-6, 12).astype(np.float32)
        return features, np.zeros(12, dtype=np.float32)

    # Fit mode: per-row kNN fill in consensus space
    k = min(5, len(observed_indices))
    obs_consensus = consensus_all[observed_indices]
    obs_pairwise = features[observed_indices, 2:14].astype(np.float64)
    fill_values = np.zeros(12, dtype=np.float32)
    for uidx in unobserved_indices:
        diff = obs_consensus - consensus_all[uidx]
        dists = np.sum(diff ** 2, axis=1)
        nn = np.argpartition(dists, k - 1)[:k]
        features[uidx, 2:14] = obs_pairwise[nn].mean(axis=0).astype(np.float32)
        fill_values += features[uidx, 2:14]

    fill_values /= max(len(unobserved_indices), 1.0)
    return features, fill_values.astype(np.float32)


def build_va_geometry_features(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    *,
    neutral_point: Sequence[float] = (0.5, 0.5),
    consistency_sigma: float = 0.35,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build full 17-dim VA geometry features (14 observed + 3 mask).

    Backward-compatible interface. For clustering, prefer
    build_va_geometry_observed_features() which excludes mask indicators.
    """
    audio = _as_va_matrix("audio_va", audio_va)
    lyrics = _as_va_matrix("lyrics_va", lyrics_va)
    if lyrics.shape[0] != audio.shape[0]:
        raise ValueError(f"audio_va and lyrics_va must have same N, got {audio.shape[0]} and {lyrics.shape[0]}.")
    mask = _as_view_mask(view_mask, audio.shape[0])
    if mask.shape != (audio.shape[0], 2) and mask.shape[1] < 2:
        raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")

    has_audio = (mask[:, 0:1] > 0.0).astype(np.float32)
    has_lyrics = (mask[:, 1:2] > 0.0).astype(np.float32)
    has_both = has_audio * has_lyrics

    observed, mask_feat, _ = _compute_geometry_core(
        audio, lyrics, has_audio, has_lyrics, has_both,
        neutral_point=neutral_point, consistency_sigma=consistency_sigma, eps=eps,
    )
    return np.concatenate([observed, mask_feat], axis=1).astype(np.float32)


def build_balanced_va_diff_features(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    *,
    neutral_point: Sequence[float] = (0.5, 0.5),
    consistency_sigma: float = 0.35,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build a compact affect-first encoding for audio/lyrics VA clustering.

    The first two columns are the consensus VA plane coordinates used as the
    primary clustering geometry. The remaining columns encode audio-lyrics
    disagreement. For complete pairs, consensus + signed delta can exactly
    reconstruct both modal VA points, so raw audio/lyrics coordinates are not
    duplicated as independent GMM axes.
    """
    audio = _as_va_matrix("audio_va", audio_va)
    lyrics = _as_va_matrix("lyrics_va", lyrics_va)
    if lyrics.shape[0] != audio.shape[0]:
        raise ValueError(f"audio_va and lyrics_va must have same N, got {audio.shape[0]} and {lyrics.shape[0]}.")
    mask = _as_view_mask(view_mask, audio.shape[0])

    has_audio = (mask[:, 0:1] > 0.0).astype(np.float32)
    has_lyrics = (mask[:, 1:2] > 0.0).astype(np.float32)
    has_both = has_audio * has_lyrics
    observed, _, consensus = _compute_geometry_core(
        audio,
        lyrics,
        has_audio,
        has_lyrics,
        has_both,
        neutral_point=neutral_point,
        consistency_sigma=consistency_sigma,
        eps=eps,
    )
    return np.concatenate(
        [
            consensus.astype(np.float32),
            observed[:, 2:4].astype(np.float32),  # signed VA delta
            observed[:, 6:7].astype(np.float32),  # normalized Euclidean gap
            observed[:, 9:10].astype(np.float32),  # signed angular gap
            observed[:, 12:13].astype(np.float32),  # radial gap
            observed[:, 13:14].astype(np.float32),  # RBF consistency
        ],
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# V16: Calibrated VA + Conditional Residual Tension
# ---------------------------------------------------------------------------

CALIBRATED_VA_TENSION_FEATURE_NAMES = [
    "consensus_valence",
    "consensus_arousal",
    "tension_dv",
    "tension_da",
    "tension_r",
]
CALIBRATED_VA_TENSION_DIM = len(CALIBRATED_VA_TENSION_FEATURE_NAMES)
CALIBRATED_VA_TENSION_CONSENSUS_SLICE = (0, 2)
CALIBRATED_VA_TENSION_TENSION_SLICE = (2, 5)


def build_calibrated_va_tension_features(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    *,
    calibrator: Optional[AffectCalibrator] = None,
    balance_learner: Optional[BalanceAlphaLearner] = None,
    residualizer: Optional[DiffResidualizer] = None,
    fit: bool = False,
    consensus_mode: str = "calibrated_mean",
    consensus_alpha: float = 0.5,
    alpha_search_min: float = 0.20,
    alpha_search_max: float = 0.90,
    alpha_search_step: float = 0.05,
    alpha_search_k_min: int = 4,
    alpha_search_k_max: int = 8,
    calibration_mode: str = "global_median_shift",
    diff_residual_mode: str = "knn",
    diff_residual_neighbors: int = 101,
    compute_device: str = "cpu",
    compute_chunk_size: int = 4096,
    compute_sample_size: int = 0,
    tension_encoding: str = "residual_3d",
    fitted_sigma: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build 5-dim calibrated consensus + residual tension features.

    Returns (features[N, 5], state_dict) where state_dict contains fitted
    calibrator/residualizer for reuse on eval splits.
    """
    audio = _as_va_matrix("audio_va", audio_va)
    lyrics = _as_va_matrix("lyrics_va", lyrics_va)
    mask = _as_view_mask(view_mask, audio.shape[0])
    n = audio.shape[0]

    has_audio = (mask[:, 0] > 0.0)
    has_lyrics = (mask[:, 1] > 0.0)
    has_both = has_audio & has_lyrics

    if calibrator is None:
        calibrator = AffectCalibrator(mode=calibration_mode)
    if fit:
        calibrator.fit(audio, lyrics, mask)
    audio_cal, lyrics_cal = calibrator.transform(audio, lyrics, mask)

    requested_consensus_mode = str(consensus_mode or "calibrated_mean").strip().lower()
    normalized_consensus_mode = (
        "bias_neutral_mean"
        if requested_consensus_mode == "calibrated_mean"
        else requested_consensus_mode
    )

    if normalized_consensus_mode == "bias_neutral_mean":
        consensus = 0.5 * audio_cal + 0.5 * lyrics_cal
        alpha = 0.5
    elif normalized_consensus_mode == "mean":
        consensus = 0.5 * audio + 0.5 * lyrics
        alpha = 0.5
    elif normalized_consensus_mode == "global_alpha":
        if balance_learner is None:
            balance_learner = BalanceAlphaLearner(mode="global_alpha", alpha_=float(consensus_alpha))
        if fit:
            balance_learner.fit(audio_cal, lyrics_cal, mask)
        consensus = balance_learner.transform(audio_cal, lyrics_cal, mask)
        alpha = float(balance_learner.alpha_)
    elif normalized_consensus_mode == "clusterability_alpha":
        if balance_learner is None:
            balance_learner = BalanceAlphaLearner(
                mode="clusterability_alpha",
                alpha_min=float(alpha_search_min),
                alpha_max=float(alpha_search_max),
                alpha_step=float(alpha_search_step),
                search_k_min=int(alpha_search_k_min),
                search_k_max=int(alpha_search_k_max),
                device=str(compute_device),
                chunk_size=int(compute_chunk_size),
                score_sample_size=int(compute_sample_size),
            )
        else:
            balance_learner.device = str(compute_device)
            balance_learner.chunk_size = int(compute_chunk_size)
            balance_learner.score_sample_size = int(compute_sample_size)
        if fit:
            balance_learner.fit(audio_cal, lyrics_cal, mask)
        consensus = balance_learner.transform(audio_cal, lyrics_cal, mask)
        alpha = float(balance_learner.alpha_)
    else:
        raise ValueError(f"Unsupported consensus_mode: {consensus_mode!r}")

    audio_only = has_audio & ~has_lyrics
    lyrics_only = has_lyrics & ~has_audio
    if audio_only.any():
        consensus[audio_only] = audio_cal[audio_only]
    if lyrics_only.any():
        consensus[lyrics_only] = lyrics_cal[lyrics_only]

    raw_delta = (lyrics_cal - audio_cal) * has_both.reshape(-1, 1).astype(np.float32)

    if residualizer is None:
        residualizer = DiffResidualizer(
            mode=diff_residual_mode,
            n_neighbors=diff_residual_neighbors,
            device=str(compute_device),
            chunk_size=int(compute_chunk_size),
        )
    else:
        residualizer.device = str(compute_device)
        residualizer.chunk_size = int(compute_chunk_size)
    if fit:
        residualizer.fit(consensus, raw_delta, mask)
    d_res = residualizer.transform(consensus, raw_delta, mask)

    if fitted_sigma is not None:
        sigma_v = float(max(float(fitted_sigma[0]), 1e-6))
        sigma_a = float(max(float(fitted_sigma[1]), 1e-6))
    elif has_both.sum() > 2:
        obs_res = d_res[has_both]
        sigma_v = float(np.maximum(_robust_std(obs_res[:, 0]), 1e-6))
        sigma_a = float(np.maximum(_robust_std(obs_res[:, 1]), 1e-6))
    else:
        sigma_v = 1.0
        sigma_a = 1.0

    dv = np.clip(d_res[:, 0] / sigma_v, -3.0, 3.0).astype(np.float32)
    da = np.clip(d_res[:, 1] / sigma_a, -3.0, 3.0).astype(np.float32)
    r = np.sqrt(dv ** 2 + da ** 2).astype(np.float32)
    raw_r = np.linalg.norm(d_res, axis=1).astype(np.float32)

    features = np.column_stack([
        consensus.astype(np.float32),
        dv,
        da,
        r,
    ]).astype(np.float32)

    state = {
        "calibrator": calibrator,
        "balance_learner": balance_learner,
        "residualizer": residualizer,
        "balance_alpha": float(alpha),
        "consensus_mode": normalized_consensus_mode,
        "sigma_v": sigma_v,
        "sigma_a": sigma_a,
        "residualized_tension_matrix": np.column_stack(
            [d_res[:, 0], d_res[:, 1], raw_r]
        ).astype(np.float32),
        "residualized_tension_units": "calibrated_va_delta",
    }
    if balance_learner is not None:
        scores = [dict(item) for item in getattr(balance_learner, "scores_", [])]
        state["balance_alpha_scores"] = scores
        selected_score = next(
            (item for item in scores if abs(float(item.get("alpha", float("nan"))) - float(alpha)) < 1e-8),
            None,
        )
        if selected_score is None and scores:
            selected_score = max(scores, key=lambda item: float(item.get("score", -float("inf"))))
        state["balance_alpha_summary"] = {
            "alpha": float(alpha),
            "alpha_best_k": int(float(selected_score.get("best_k", -1))) if selected_score else -1,
            "alpha_score": float(selected_score.get("score", float("nan"))) if selected_score else float("nan"),
            "alpha_silhouette": float(selected_score.get("silhouette", float("nan"))) if selected_score else float("nan"),
            "alpha_stability": float(selected_score.get("stability", float("nan"))) if selected_score else float("nan"),
            "alpha_size_balance": float(selected_score.get("size_balance", float("nan"))) if selected_score else float("nan"),
        }
    return features, state


def _robust_std(x: np.ndarray) -> float:
    q75, q25 = np.percentile(x, [75, 25])
    return float((q75 - q25) / 1.349)
