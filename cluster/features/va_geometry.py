from __future__ import annotations

import math
from typing import Sequence

import numpy as np

VA_GEOMETRY_FEATURE_NAMES = [
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
    "has_both_audio_lyrics",
    "has_audio",
    "has_lyrics",
]


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


def build_va_geometry_features(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    *,
    neutral_point: Sequence[float] = (0.5, 0.5),
    consistency_sigma: float = 0.35,
    eps: float = 1e-6,
) -> np.ndarray:
    """Build interpretable audio/lyrics conflict features in VA space.

    The first two dimensions are a mask-aware consensus VA point. Pairwise
    disagreement dimensions are computed only when both audio and lyrics are
    present; otherwise they are zero and the trailing mask dimensions indicate
    that the disagreement is unknown rather than true agreement.
    """
    audio = _as_va_matrix("audio_va", audio_va)
    lyrics = _as_va_matrix("lyrics_va", lyrics_va)
    if lyrics.shape[0] != audio.shape[0]:
        raise ValueError(f"audio_va and lyrics_va must have same N, got {audio.shape[0]} and {lyrics.shape[0]}.")
    mask = _as_view_mask(view_mask, audio.shape[0])
    neutral = np.asarray(neutral_point, dtype=np.float32)
    if neutral.shape != (2,):
        raise ValueError(f"neutral_point must have shape [2], got {neutral.shape}.")

    has_audio = (mask[:, 0:1] > 0.0).astype(np.float32)
    has_lyrics = (mask[:, 1:2] > 0.0).astype(np.float32)
    has_both = has_audio * has_lyrics
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

    return np.concatenate(
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
            has_both.astype(np.float32),
            has_audio.astype(np.float32),
            has_lyrics.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
