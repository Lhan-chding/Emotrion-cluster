from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

DIFF_GEOMETRY_OBSERVED_DIM = 9  # consensus(2) + signed_delta(2) + abs_delta(2) + gap_norm(1) + angle_gap(1) + radial_gap(1)
SOFT_QUADRANT_PAIR_DIM = 16
UNCERTAINTY_DIM = 1
DIFF_GEOMETRY_TOTAL_DIM = DIFF_GEOMETRY_OBSERVED_DIM + SOFT_QUADRANT_PAIR_DIM + UNCERTAINTY_DIM
DIFF_GEOMETRY_INPUT_DIM = DIFF_GEOMETRY_TOTAL_DIM  # 26-dim input for DiffEncoder
DIFF_OBSERVED_DIM = 1

DIFF_GEOMETRY_FEATURE_NAMES = [
    # Observed (7)
    "consensus_valence",
    "consensus_arousal",
    "signed_delta_valence",
    "signed_delta_arousal",
    "abs_delta_valence",
    "abs_delta_arousal",
    "gap_norm",
    "angle_gap",
    "radial_gap",
    # Soft quadrant pair (16)
    *[f"sqp_{i}" for i in range(SOFT_QUADRANT_PAIR_DIM)],
    # Uncertainty (1)
    "uncertainty",
]

assert len(DIFF_GEOMETRY_FEATURE_NAMES) == DIFF_GEOMETRY_TOTAL_DIM

_QUADRANT_CENTERS = np.array([
    [ 0.75,  0.75],  # Q1: high V, high A
    [ 0.25,  0.75],  # Q2: low V, high A
    [ 0.25,  0.25],  # Q3: low V, low A
    [ 0.75,  0.25],  # Q4: high V, low A
], dtype=np.float32)

_QUADRANT_TEMPERATURE = 0.08


def _as_va_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2], got {matrix.shape}.")
    return matrix


def _soft_quadrant_pair(audio_va: np.ndarray, lyrics_va: np.ndarray) -> np.ndarray:
    """Compute soft quadrant-pair outer product [N, 16].

    For each sample, compute softmax assignment of audio and lyrics to 4 VA
    quadrants (based on distance to quadrant centers), then take the outer
    product to produce a 4x4 = 16-dim feature vector.
    """
    audio_dist = np.sum((audio_va[:, np.newaxis, :] - _QUADRANT_CENTERS[np.newaxis, :, :]) ** 2, axis=2)
    lyrics_dist = np.sum((lyrics_va[:, np.newaxis, :] - _QUADRANT_CENTERS[np.newaxis, :, :]) ** 2, axis=2)

    audio_soft = np.exp(-audio_dist / _QUADRANT_TEMPERATURE)
    audio_soft /= np.maximum(audio_soft.sum(axis=1, keepdims=True), 1e-8)
    lyrics_soft = np.exp(-lyrics_dist / _QUADRANT_TEMPERATURE)
    lyrics_soft /= np.maximum(lyrics_soft.sum(axis=1, keepdims=True), 1e-8)

    # Outer product: [N, 4, 1] * [N, 1, 4] → [N, 4, 4] → flatten to [N, 16]
    sqp = (audio_soft[:, :, np.newaxis] * lyrics_soft[:, np.newaxis, :]).reshape(audio_va.shape[0], 16)
    return sqp.astype(np.float32)


def build_diff_geometry_features(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: np.ndarray,
    *,
    neutral_point: Sequence[float] = (0.5, 0.5),
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 26-dim diff geometry features and [N, 1] diff_observed mask.

    Returns (diff_features, diff_observed).

    diff_features columns:
      0-1:  consensus (mean VA)
      2-3:  signed_delta (audio - lyrics)
      4-5:  abs_delta
      6:    gap_norm (euclidean / sqrt(2))
      7:    angle_gap (atan2 / pi)
      8:    radial_gap (audio_radius - lyrics_radius)
      9-24: soft_quadrant_pair (16)
      25:   uncertainty (sigmoid of gap_norm)
    """
    audio = _as_va_matrix("audio_va", audio_va)
    lyrics = _as_va_matrix("lyrics_va", lyrics_va)
    if lyrics.shape[0] != audio.shape[0]:
        raise ValueError(f"audio_va and lyrics_va must have same N, got {audio.shape[0]} and {lyrics.shape[0]}.")
    mask = np.asarray(view_mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[0] != audio.shape[0]:
        raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")
    neutral = np.asarray(neutral_point, dtype=np.float32)

    has_audio = (mask[:, 0:1] > 0.0).astype(np.float32)
    has_lyrics = (mask[:, 1:2] > 0.0).astype(np.float32)
    has_both = has_audio * has_lyrics

    # Consensus: weighted mean of available views
    weights = has_audio + has_lyrics
    consensus = np.divide(
        audio * has_audio + lyrics * has_lyrics,
        np.maximum(weights, 1.0),
        out=np.tile(neutral.reshape(1, 2), (audio.shape[0], 1)).astype(np.float32),
        where=weights > 0.0,
    )

    # Pairwise diff features (only meaningful when both views present)
    signed_delta = (audio - lyrics) * has_both
    abs_delta = np.abs(signed_delta)
    gap_norm = (np.linalg.norm(signed_delta, axis=1, keepdims=True) / math.sqrt(2.0)).astype(np.float32) * has_both

    audio_centered = (audio - neutral.reshape(1, 2)).astype(np.float32)
    lyrics_centered = (lyrics - neutral.reshape(1, 2)).astype(np.float32)
    audio_radius = np.linalg.norm(audio_centered, axis=1, keepdims=True).astype(np.float32)
    lyrics_radius = np.linalg.norm(lyrics_centered, axis=1, keepdims=True).astype(np.float32)
    radial_gap = (audio_radius - lyrics_radius).astype(np.float32) * has_both

    dot = np.sum(audio_centered * lyrics_centered, axis=1, keepdims=True).astype(np.float32)
    det = (
        audio_centered[:, 0:1] * lyrics_centered[:, 1:2]
        - audio_centered[:, 1:2] * lyrics_centered[:, 0:1]
    ).astype(np.float32)
    angle_gap = (np.arctan2(det, dot) / math.pi).astype(np.float32) * has_both

    # Soft quadrant pair
    sqp_full = _soft_quadrant_pair(audio, lyrics)
    sqp = sqp_full * has_both

    # Uncertainty: sigmoid of gap_norm (closer to 1 when views disagree strongly)
    uncertainty = (1.0 / (1.0 + np.exp(-5.0 * (gap_norm - 0.2)))).astype(np.float32) * has_both

    diff_features = np.concatenate(
        [
            consensus.astype(np.float32),          # 2
            signed_delta.astype(np.float32),       # 2
            abs_delta.astype(np.float32),          # 2
            gap_norm.astype(np.float32),           # 1
            angle_gap.astype(np.float32),          # 1
            radial_gap.astype(np.float32),         # 1
            sqp.astype(np.float32),                # 16
            uncertainty.astype(np.float32),        # 1
        ],
        axis=1,
    ).astype(np.float32)  # total: 26

    diff_observed = has_both.astype(np.float32)

    return diff_features, diff_observed
