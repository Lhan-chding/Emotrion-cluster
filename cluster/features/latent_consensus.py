from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


LATENT_TWO_VIEW_VA_DIM = 4
LATENT_TWO_VIEW_VA_FEATURE_NAMES = (
    "audio_valence",
    "audio_arousal",
    "lyrics_valence",
    "lyrics_arousal",
)


def build_latent_two_view_va_features(
    audio_va: np.ndarray,
    lyrics_va: np.ndarray,
    view_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return raw two-view VA matrix for the latent two-view GMM."""
    audio = np.asarray(audio_va, dtype=np.float32)
    lyrics = np.asarray(lyrics_va, dtype=np.float32)
    if audio.ndim != 2 or lyrics.ndim != 2 or audio.shape != lyrics.shape or audio.shape[1] != 2:
        raise ValueError(f"audio_va and lyrics_va must both have shape [N, 2], got {audio.shape} and {lyrics.shape}.")
    if view_mask is None:
        mask = np.ones((audio.shape[0], 2), dtype=np.float32)
    else:
        mask = np.asarray(view_mask, dtype=np.float32)
        if mask.ndim != 2 or mask.shape[0] != audio.shape[0] or mask.shape[1] < 2:
            raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")
        mask = mask[:, :2]
    has_audio = mask[:, 0] > 0.0
    has_lyrics = mask[:, 1] > 0.0
    if not np.all(has_audio | has_lyrics):
        raise ValueError("Every row must have at least one observed VA view.")
    features = np.concatenate([audio, lyrics], axis=1).astype(np.float32)
    state = {
        "kind": "latent_two_view_va",
        "feature_names": list(LATENT_TWO_VIEW_VA_FEATURE_NAMES),
        "observed_audio_count": int(has_audio.sum()),
        "observed_lyrics_count": int(has_lyrics.sum()),
        "observed_both_count": int((has_audio & has_lyrics).sum()),
    }
    return features, state


def observed_mean_va_from_two_view_features(features: np.ndarray, view_mask: Optional[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 4:
        raise ValueError(f"latent_two_view_va features must have shape [N, >=4], got {matrix.shape}.")
    audio = matrix[:, :2]
    lyrics = matrix[:, 2:4]
    if view_mask is None:
        return ((audio + lyrics) * 0.5).astype(np.float32)
    mask = np.asarray(view_mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[0] != matrix.shape[0] or mask.shape[1] < 2:
        raise ValueError(f"view_mask must have shape [N, >=2], got {mask.shape}.")
    weights = mask[:, 0:1] + mask[:, 1:2]
    summed = audio * mask[:, 0:1] + lyrics * mask[:, 1:2]
    return np.divide(
        summed,
        np.maximum(weights, 1.0),
        out=np.full_like(summed, 0.5, dtype=np.float32),
        where=weights > 0.0,
    ).astype(np.float32)
