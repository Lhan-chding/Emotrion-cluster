from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

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

    When audio+lyrics are not both present, pairwise features are deterministic
    zeros. This makes unobserved rows separable from observed rows by GMM.
    Imputing with the observed-row mean makes them neutral.

    Parameters
    ----------
    features : ndarray [N, 14]
        The 14-dim observed geometry features.
    view_mask : ndarray [N, >=2]
        View availability mask (col 0=audio, col 1=lyrics).
    fitted_fill : ndarray [12] or None
        Pre-computed fill values for dims 2:14. If None, computed from
        observed rows in this batch (fit mode).

    Returns
    -------
    (imputed_features, fill_values) where fill_values has shape [12].
    """
    features = np.array(features, dtype=np.float32)
    mask = np.asarray(view_mask, dtype=np.float32)
    has_both = ((mask[:, 0] > 0.0) & (mask[:, 1] > 0.0))

    if fitted_fill is not None:
        fill = np.asarray(fitted_fill, dtype=np.float32)
    else:
        observed = features[has_both, 2:14]
        if observed.shape[0] > 0:
            fill = observed.mean(axis=0).astype(np.float32)
        else:
            fill = np.zeros(12, dtype=np.float32)

    unobserved = ~has_both
    if unobserved.any():
        features[unobserved, 2:14] = fill

    return features, fill


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
