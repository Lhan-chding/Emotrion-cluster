from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from cluster.utils import find_column, normalize_col


LIST_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("MoodsAll", "MoodsAllWeights"),
    ("Themes", "ThemeWeights"),
    ("Styles", "StyleWeights"),
    ("Genres", "GenreWeights"),
)

NUMERIC_FIELDS: Tuple[str, ...] = (
    "Duration",
    "ActualYear",
    "Relevance",
    "num_Genres",
    "num_MoodsAll",
)

# Tempo bins (BPM boundaries)
_TEMPO_BINS: Tuple[float, ...] = (0.0, 80.0, 120.0, 160.0, float("inf"))
_TEMPO_LABELS: Tuple[str, ...] = ("slow", "moderate", "fast", "very_fast")


@dataclass
class MetadataFeatureBundle:
    features: np.ndarray
    feature_names: List[str]
    canonical_metadata: pd.DataFrame
    token_vocab: Dict[str, List[str]]
    numeric_fields: List[str]


def _normalize_free_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_lookup_text(value: object) -> str:
    text = _normalize_free_text(value).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9&/,\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_list_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_token_list(value: object) -> List[str]:
    text = _normalize_list_text(value)
    if not text:
        return []
    tokens = []
    for part in text.split(","):
        token = _normalize_lookup_text(part)
        if token:
            tokens.append(token)
    return tokens


def _parse_weight_list(value: object, size: int) -> List[float]:
    text = _normalize_list_text(value)
    if not text:
        return [1.0] * size
    out: List[float] = []
    for part in text.split(","):
        try:
            out.append(float(part))
        except ValueError:
            out.append(1.0)
    if len(out) < size:
        out.extend([1.0] * (size - len(out)))
    return out[:size]


def _read_track_pairs(processed_dir: str) -> pd.DataFrame:
    path = Path(processed_dir) / "track_index.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing track index file '{path}'.")
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "index": int(row["index"]),
                    "Audio_Song": str(row["identifier"]).strip(),
                    "Lyric_Song": str(row["lyric_identifier"]).strip(),
                    "Quadrant": str(row.get("quadrant", "")).strip().upper(),
                }
            )
    if not rows:
        raise ValueError(f"Track index '{path}' is empty.")
    df = pd.DataFrame(rows).sort_values("index", kind="stable").reset_index(drop=True)
    return df


def _read_aligned_metadata(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {description} at '{path}'.")
    df = pd.read_csv(path).copy()
    audio_col = find_column(df.columns, ["audio_song"])
    lyric_col = find_column(df.columns, ["lyric_song", "lyrics_song"])
    quadrant_col = find_column(df.columns, ["quadrant"])
    if not all([audio_col, lyric_col, quadrant_col]):
        raise ValueError(
            f"{description} must contain Audio_Song, Lyric_Song and Quadrant columns."
        )
    rename_map = {
        audio_col: "Audio_Song",
        lyric_col: "Lyric_Song",
        quadrant_col: "Quadrant",
    }
    out = df.rename(columns=rename_map)
    out["Audio_Song"] = out["Audio_Song"].astype(str).str.strip()
    out["Lyric_Song"] = out["Lyric_Song"].astype(str).str.strip()
    out["Quadrant"] = out["Quadrant"].astype(str).str.strip().str.upper()
    return out.drop_duplicates(subset=["Audio_Song", "Lyric_Song"], keep="first").reset_index(drop=True)


def _coalesce_prefer_audio(audio_value: object, lyrics_value: object, *, list_like: bool = False) -> str:
    if list_like:
        audio_norm = _normalize_list_text(audio_value)
        lyrics_norm = _normalize_list_text(lyrics_value)
    else:
        audio_norm = _normalize_free_text(audio_value)
        lyrics_norm = _normalize_free_text(lyrics_value)
    return audio_norm if audio_norm else lyrics_norm


def build_canonical_metadata(aligned_root: str, processed_dir: str) -> pd.DataFrame:
    root = Path(aligned_root)
    audio_meta = _read_aligned_metadata(root / "aligned_audio_metadata.csv", "aligned audio metadata CSV")
    lyrics_meta = _read_aligned_metadata(root / "aligned_lyrics_metadata.csv", "aligned lyrics metadata CSV")
    track_pairs = _read_track_pairs(processed_dir)

    audio_prefixed = audio_meta.rename(columns={c: f"audio__{c}" for c in audio_meta.columns if c not in {"Audio_Song", "Lyric_Song"}})
    lyrics_prefixed = lyrics_meta.rename(columns={c: f"lyrics__{c}" for c in lyrics_meta.columns if c not in {"Audio_Song", "Lyric_Song"}})
    merged = track_pairs.merge(audio_prefixed, on=["Audio_Song", "Lyric_Song"], how="left", validate="one_to_one")
    merged = merged.merge(lyrics_prefixed, on=["Audio_Song", "Lyric_Song"], how="left", validate="one_to_one")

    canonical = merged[["index", "Audio_Song", "Lyric_Song", "Quadrant"]].copy()

    list_fields = {
        "Moods",
        "MoodsAll",
        "MoodsAllWeights",
        "Genres",
        "GenreWeights",
        "Themes",
        "ThemeWeights",
        "Styles",
        "StyleWeights",
    }

    all_field_names = sorted(
        {
            col.split("__", 1)[1]
            for col in merged.columns
            if col.startswith("audio__") or col.startswith("lyrics__")
        }
    )
    # Vectorised coalesce: prefer audio, fall back to lyrics
    for field in all_field_names:
        audio_col = f"audio__{field}"
        lyrics_col = f"lyrics__{field}"
        audio_series = merged[audio_col] if audio_col in merged.columns else pd.Series([""] * len(merged))
        lyrics_series = merged[lyrics_col] if lyrics_col in merged.columns else pd.Series([""] * len(merged))
        if field in list_fields:
            a_norm = audio_series.fillna("").astype(str).str.strip().str.replace(r"\s*,\s*", ",", regex=True).str.replace(r"\s+", " ", regex=True)
            l_norm = lyrics_series.fillna("").astype(str).str.strip().str.replace(r"\s*,\s*", ",", regex=True).str.replace(r"\s+", " ", regex=True)
        else:
            a_norm = audio_series.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            l_norm = lyrics_series.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        canonical[field] = np.where(a_norm != "", a_norm, l_norm)

    canonical["Artist"] = canonical.get("Artist", "").map(_normalize_free_text)
    canonical["Title"] = canonical.get("Title", "").map(_normalize_free_text)
    canonical["Quadrant"] = canonical["Quadrant"].astype(str).str.upper()

    for field in ["Duration", "ActualYear", "Relevance", "num_Genres", "num_MoodsAll"]:
        if field in canonical.columns:
            canonical[field] = pd.to_numeric(canonical[field], errors="coerce")

    # Vectorised count derivation (avoids lambda + map overhead)
    _genres_count = canonical["Genres"].fillna("").astype(str).str.split(",").apply(len).where(
        canonical["Genres"].fillna("").astype(str).str.strip() != "", other=0
    )
    if "num_Genres" not in canonical.columns:
        canonical["num_Genres"] = _genres_count
    else:
        canonical["num_Genres"] = canonical["num_Genres"].fillna(_genres_count)

    _moods_count = canonical["MoodsAll"].fillna("").astype(str).str.split(",").apply(len).where(
        canonical["MoodsAll"].fillna("").astype(str).str.strip() != "", other=0
    )
    if "num_MoodsAll" not in canonical.columns:
        canonical["num_MoodsAll"] = _moods_count
    else:
        canonical["num_MoodsAll"] = canonical["num_MoodsAll"].fillna(_moods_count)

    return canonical.sort_values("index", kind="stable").reset_index(drop=True)


def _build_tfidf_block(
    text_series: pd.Series,
    weight_series: pd.Series,
    min_token_freq: int,
    max_tokens: int,
) -> Tuple[np.ndarray, List[str]]:
    """Build a TF-IDF weighted feature block for one list field.

    Instead of raw weighted counts, applies IDF weighting so that
    common tokens (appearing in most tracks) are down-weighted while
    rare-but-present tokens receive higher importance.

    Returns (n_samples, n_kept_tokens) float32 array and token names.
    """
    n_rows = len(text_series)
    text_vals = text_series.fillna("").astype(str)
    weight_vals = weight_series.fillna("").astype(str)

    # Vectorised parse: split all rows at once
    all_tokens: List[List[str]] = []
    all_weights: List[List[float]] = []
    counter: Counter[str] = Counter()
    for txt, wt in zip(text_vals.values, weight_vals.values):
        tokens = _parse_token_list(txt)
        weights = _parse_weight_list(wt, len(tokens))
        all_tokens.append(tokens)
        all_weights.append(weights)
        for token in tokens:
            counter[token] += 1

    # Vocabulary: frequency-filtered, capped
    kept = [
        token
        for token, freq in counter.most_common()
        if freq >= int(min_token_freq)
    ][: max(int(max_tokens), 1)]
    if not kept:
        return np.zeros((n_rows, 0), dtype=np.float32), []

    token_to_idx = {token: idx for idx, token in enumerate(kept)}
    n_vocab = len(kept)

    # Document frequency for IDF
    doc_freq = np.zeros(n_vocab, dtype=np.float64)
    for tokens in all_tokens:
        seen: set = set()
        for token in tokens:
            idx = token_to_idx.get(token)
            if idx is not None and idx not in seen:
                doc_freq[idx] += 1.0
                seen.add(idx)

    # IDF: log((1 + N) / (1 + df)) + 1  (sklearn-style smooth IDF)
    idf = np.log((1.0 + n_rows) / (1.0 + doc_freq)) + 1.0
    idf = idf.astype(np.float32)

    # Build sparse TF matrix (weighted counts), then multiply by IDF
    row_indices: List[int] = []
    col_indices: List[int] = []
    data_values: List[float] = []
    for row_idx, (tokens, weights) in enumerate(zip(all_tokens, all_weights)):
        for token, weight in zip(tokens, weights):
            idx = token_to_idx.get(token)
            if idx is not None:
                row_indices.append(row_idx)
                col_indices.append(idx)
                data_values.append(float(weight))

    tf_sparse = csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_rows, n_vocab),
        dtype=np.float32,
    )
    # TF-IDF = TF * IDF (element-wise broadcast across rows)
    tfidf = tf_sparse.multiply(idf).toarray().astype(np.float32)

    # L2-normalise each row (so that tracks with many tokens don't dominate)
    row_norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)
    tfidf = tfidf / row_norms

    return tfidf, kept


def _engineer_numeric_features(canonical_metadata: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build engineered numeric features from canonical metadata.

    Improvements over raw numerics:
    - Duration: converted from ms to minutes (more interpretable)
    - Popularity/Relevance: log1p-transformed if skewed (skew > 1)
    - Tempo: binned into slow/moderate/fast/very_fast one-hot columns
    - Cross-feature: energy * valence interaction (if both present)
    """
    feature_names: List[str] = []
    blocks: List[np.ndarray] = []
    n_rows = len(canonical_metadata)

    # --- Base numeric fields ---
    for field in NUMERIC_FIELDS:
        col = pd.to_numeric(
            canonical_metadata[field] if field in canonical_metadata.columns else pd.Series([np.nan] * n_rows),
            errors="coerce",
        )
        if col.isna().all():
            col = col.fillna(0.0)
        else:
            col = col.fillna(float(col.median()))

        # Duration: ms → minutes
        if field == "Duration":
            col = col / 60_000.0
            feature_names.append("numeric::Duration_min")
        # Relevance: log1p if skewed
        elif field == "Relevance":
            if col.skew() > 1.0:
                col = np.log1p(col)
                feature_names.append("numeric::Relevance_log")
            else:
                feature_names.append(f"numeric::{field}")
        else:
            feature_names.append(f"numeric::{field}")

        blocks.append(col.to_numpy(dtype=np.float32).reshape(-1, 1))

    # --- Tempo binning (if Tempo column exists) ---
    if "Tempo" in canonical_metadata.columns:
        tempo = pd.to_numeric(canonical_metadata["Tempo"], errors="coerce").fillna(0.0)
        bin_indices = np.digitize(tempo.to_numpy(), bins=_TEMPO_BINS[1:-1], right=False)
        tempo_onehot = np.zeros((n_rows, len(_TEMPO_LABELS)), dtype=np.float32)
        for i, label in enumerate(_TEMPO_LABELS):
            tempo_onehot[:, i] = (bin_indices == i).astype(np.float32)
        blocks.append(tempo_onehot)
        feature_names.extend([f"tempo_bin::{label}" for label in _TEMPO_LABELS])

    # --- Cross-features (if energy and valence columns exist) ---
    energy_col = find_column(canonical_metadata.columns, ["energy"])
    valence_col = find_column(canonical_metadata.columns, ["valence"])
    if energy_col and valence_col:
        energy = pd.to_numeric(canonical_metadata[energy_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        valence = pd.to_numeric(canonical_metadata[valence_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        interaction = (energy * valence).reshape(-1, 1)
        blocks.append(interaction)
        feature_names.append("cross::energy_valence")

    if blocks:
        result = np.concatenate(blocks, axis=1).astype(np.float32)
    else:
        result = np.zeros((n_rows, 0), dtype=np.float32)
    return result, feature_names


def build_metadata_features(
    canonical_metadata: pd.DataFrame,
    min_token_freq: int = 3,
    max_tokens_per_field: int = 128,
    use_tfidf: bool = True,
) -> MetadataFeatureBundle:
    """Build metadata feature matrix from canonical metadata.

    Parameters
    ----------
    canonical_metadata : pd.DataFrame
        Output of ``build_canonical_metadata``.
    min_token_freq : int
        Minimum document frequency for a token to be kept.
    max_tokens_per_field : int
        Maximum vocabulary size per list field.
    use_tfidf : bool
        If True (default), apply TF-IDF weighting to token features.
        If False, use raw weighted counts (legacy behaviour).
    """
    vocab: Dict[str, List[str]] = {}
    feature_blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    for text_col, weight_col in LIST_FIELDS:
        if use_tfidf:
            block, kept = _build_tfidf_block(
                text_series=canonical_metadata[text_col],
                weight_series=canonical_metadata[weight_col],
                min_token_freq=min_token_freq,
                max_tokens=max_tokens_per_field,
            )
        else:
            # Legacy: raw weighted counts (no IDF, no L2 norm)
            block, kept = _build_tfidf_block(
                text_series=canonical_metadata[text_col],
                weight_series=canonical_metadata[weight_col],
                min_token_freq=min_token_freq,
                max_tokens=max_tokens_per_field,
            )
            # Undo IDF + L2 norm by rebuilding raw counts
            counter: Counter[str] = Counter()
            parsed_rows: List[Tuple[List[str], List[float]]] = []
            for txt, wt in zip(
                canonical_metadata[text_col].fillna("").astype(str).values,
                canonical_metadata[weight_col].fillna("").astype(str).values,
            ):
                tokens = _parse_token_list(txt)
                weights = _parse_weight_list(wt, len(tokens))
                parsed_rows.append((tokens, weights))
                for token in tokens:
                    counter[token] += 1
            token_to_idx = {token: idx for idx, token in enumerate(kept)}
            block = np.zeros((len(canonical_metadata), len(kept)), dtype=np.float32)
            for row_idx, (tokens, weights) in enumerate(parsed_rows):
                for token, weight in zip(tokens, weights):
                    idx = token_to_idx.get(token)
                    if idx is not None:
                        block[row_idx, idx] += float(weight)

        vocab[text_col] = kept
        feature_blocks.append(block)
        feature_names.extend([f"{text_col}::{token}" for token in kept])

    # Engineered numeric features (Duration→min, log-transform, tempo bins, cross-features)
    numeric_block, numeric_names = _engineer_numeric_features(canonical_metadata)
    feature_blocks.append(numeric_block)
    feature_names.extend(numeric_names)

    features = (
        np.concatenate(feature_blocks, axis=1).astype(np.float32)
        if feature_blocks
        else np.zeros((len(canonical_metadata), 0), dtype=np.float32)
    )
    return MetadataFeatureBundle(
        features=features,
        feature_names=feature_names,
        canonical_metadata=canonical_metadata.copy(),
        token_vocab=vocab,
        numeric_fields=list(NUMERIC_FIELDS),
    )


def save_metadata_feature_bundle(bundle: MetadataFeatureBundle, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, "metadata.npy")
    np.save(metadata_path, bundle.features.astype(np.float32))

    names_path = os.path.join(out_dir, "metadata_feature_names.json")
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(bundle.feature_names, f, ensure_ascii=False, indent=2)

    vocab_path = os.path.join(out_dir, "metadata_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(bundle.token_vocab, f, ensure_ascii=False, indent=2)

    canonical_path = os.path.join(out_dir, "canonical_metadata.csv")
    bundle.canonical_metadata.to_csv(canonical_path, index=False, encoding="utf-8")

    return {
        "metadata": metadata_path,
        "feature_names": names_path,
        "vocab": vocab_path,
        "canonical_metadata": canonical_path,
    }
