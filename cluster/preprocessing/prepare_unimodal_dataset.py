from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from cluster.config import MUSIC_LABEL_NAMES
from cluster.utils import find_column


DEFAULT_LIST_METADATA_FIELDS: Tuple[str, ...] = (
    "Genres",
    "Moods",
    "MoodsAll",
    "Themes",
    "Styles",
)

OPTIONAL_IDENTITY_FIELDS: Tuple[str, ...] = ("Artist",)

LIST_METADATA_FIELDS: Tuple[str, ...] = DEFAULT_LIST_METADATA_FIELDS + OPTIONAL_IDENTITY_FIELDS

NUMERIC_METADATA_FIELDS: Tuple[str, ...] = (
    "Duration",
    "ActualYear",
    "Relevance",
    "num_Genres",
    "num_MoodsAll",
    "Tempo",
)

DEFAULT_METADATA_GROUP_WEIGHTS: Dict[str, float] = {
    "Genres": 0.25,
    "Styles": 0.35,
    "Themes": 0.50,
    "Moods": 0.50,
    "MoodsAll": 0.70,
    "Artist": 0.00,
}


@dataclass(frozen=True)
class PreparedDatasetResult:
    processed_dir: str
    aligned_root: Optional[str]
    num_samples: int
    metadata_dim: int
    schema_hash: str
    dataset_hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "processed_dir": self.processed_dir,
            "aligned_root": self.aligned_root,
            "num_samples": self.num_samples,
            "metadata_dim": self.metadata_dim,
            "schema_hash": self.schema_hash,
            "dataset_hash": self.dataset_hash,
        }


def _read_csv(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("CSV path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV file '{path}'.")
    return pd.read_csv(path)


def _optional_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    return find_column(columns, candidates, required=False)


def _required_column(columns: Sequence[str], candidates: Sequence[str], description: str) -> str:
    column = find_column(columns, candidates, required=False)
    if column is None:
        raise ValueError(
            f"Could not find required {description} column. "
            f"Tried {list(candidates)}; available columns: {list(columns)}"
        )
    return column


def _stack_va(
    df: pd.DataFrame,
    *,
    valence_candidates: Sequence[str],
    arousal_candidates: Sequence[str],
    view_name: str,
    required: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    val_col = _optional_column(df.columns, valence_candidates)
    aro_col = _optional_column(df.columns, arousal_candidates)
    if val_col is None or aro_col is None:
        if required:
            raise ValueError(f"Missing {view_name} valence/arousal columns.")
        n_rows = len(df)
        return np.zeros((n_rows, 2), dtype=np.float32), np.zeros(n_rows, dtype=bool)

    valence = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=np.float32)
    arousal = pd.to_numeric(df[aro_col], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(valence) & np.isfinite(arousal)
    stacked = np.stack([valence, arousal], axis=1).astype(np.float32)
    stacked[~np.isfinite(stacked)] = 0.0
    return stacked, mask


def _normalise_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9&/,\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_tokens(value: object) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    raw = re.split(r"[,;|]", text)
    return [token for token in (_normalise_token(part) for part in raw) if token]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_metadata_group_weights(value: Optional[object]) -> Dict[str, float]:
    weights = dict(DEFAULT_METADATA_GROUP_WEIGHTS)
    if value is None:
        return weights
    if isinstance(value, dict):
        items = value.items()
    else:
        text = str(value).strip()
        if not text:
            return weights
        parsed: List[Tuple[str, str]] = []
        for part in text.split(","):
            if "=" not in part:
                raise ValueError(
                    "metadata_group_weights must use comma-separated key=value pairs."
                )
            key, raw = part.split("=", 1)
            parsed.append((key.strip(), raw.strip()))
        items = parsed
    for key, raw in items:
        group = str(key).strip()
        if not group:
            continue
        weights[group] = float(raw)
    return weights


def _row_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return (values / norms).astype(np.float32)


def _resolve_metadata_frame(
    combined: pd.DataFrame,
    metadata_csv: Optional[str],
    track_ids: Sequence[str],
    id_col: str,
) -> pd.DataFrame:
    if not metadata_csv:
        return combined.copy()

    metadata = _read_csv(metadata_csv)
    metadata_id_col = _optional_column(
        metadata.columns,
        ["track_id", "song_id", "identifier", "id", "audio_song", "song", "title"],
    )
    if metadata_id_col is not None:
        left = pd.DataFrame({"__track_id": list(track_ids)})
        right = metadata.copy()
        right["__track_id"] = right[metadata_id_col].astype(str).str.strip()
        return left.merge(right, on="__track_id", how="left").drop(columns=["__track_id"])

    if len(metadata) != len(combined):
        raise ValueError(
            "metadata_csv has no track identifier column and its row count does not "
            "match combined_csv."
        )
    return metadata.reset_index(drop=True).copy()


def _build_metadata_matrix(
    metadata_frame: pd.DataFrame,
    max_tokens_per_field: int = 512,
    *,
    metadata_use_artist: bool = False,
    metadata_representation: str = "binary",
    metadata_svd_dim: int = 32,
    metadata_group_weights: Optional[object] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]], np.ndarray, np.ndarray, Dict[str, Any]]:
    n_rows = len(metadata_frame)
    binary_blocks: List[np.ndarray] = []
    binary_feature_names: List[str] = []
    binary_feature_groups: List[Dict[str, Any]] = []
    vocab: Dict[str, List[str]] = {}
    has_any = np.zeros(n_rows, dtype=bool)
    completeness_parts: List[np.ndarray] = []
    group_weights = _parse_metadata_group_weights(metadata_group_weights)
    list_fields = list(DEFAULT_LIST_METADATA_FIELDS)
    if bool(metadata_use_artist):
        list_fields.extend(OPTIONAL_IDENTITY_FIELDS)

    for field in list_fields:
        column = _optional_column(metadata_frame.columns, [field.lower(), field])
        if column is None:
            vocab[field] = []
            continue

        parsed = [_split_tokens(value) for value in metadata_frame[column].tolist()]
        counts: Dict[str, int] = {}
        for tokens in parsed:
            for token in set(tokens):
                counts[token] = counts.get(token, 0) + 1

        kept = [
            token
            for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ][: max(int(max_tokens_per_field), 1)]
        vocab[field] = kept
        if not kept:
            completeness_parts.append(np.zeros(n_rows, dtype=np.float32))
            continue

        token_to_idx = {token: idx for idx, token in enumerate(kept)}
        block = np.zeros((n_rows, len(kept)), dtype=np.float32)
        for row_idx, tokens in enumerate(parsed):
            if tokens:
                has_any[row_idx] = True
            for token in tokens:
                col_idx = token_to_idx.get(token)
                if col_idx is not None:
                    block[row_idx, col_idx] = 1.0
        binary_blocks.append(block)
        names = [f"{field}::{token}" for token in kept]
        binary_feature_names.extend(names)
        weight = float(group_weights.get(field, 1.0))
        binary_feature_groups.extend(
            {"feature": name, "group": field, "weight": weight}
            for name in names
        )
        completeness_parts.append((block.sum(axis=1) > 0).astype(np.float32))

    if binary_blocks:
        metadata_binary = np.concatenate(binary_blocks, axis=1).astype(np.float32)
    else:
        metadata_binary = np.zeros((n_rows, 1), dtype=np.float32)
        binary_feature_names = ["metadata::dummy"]
        binary_feature_groups = [{"feature": "metadata::dummy", "group": "metadata", "weight": 0.0}]

    feature_weights = np.asarray(
        [float(item["weight"]) for item in binary_feature_groups],
        dtype=np.float32,
    ).reshape(1, -1)
    doc_freq = (metadata_binary > 0.0).sum(axis=0).astype(np.float32)
    idf = (np.log((1.0 + float(n_rows)) / (1.0 + doc_freq)) + 1.0).astype(np.float32)
    metadata_tfidf = _row_l2_normalize(metadata_binary * idf.reshape(1, -1) * feature_weights)

    representation = str(metadata_representation).strip().lower()
    if representation not in {"binary", "tfidf", "tfidf_svd"}:
        raise ValueError(
            "metadata_representation must be one of 'binary', 'tfidf', or 'tfidf_svd'."
        )

    svd_model: Optional[TruncatedSVD] = None
    svd_dim = max(int(metadata_svd_dim), 1)
    max_components = max(1, min(svd_dim, metadata_tfidf.shape[0], metadata_tfidf.shape[1]))
    if (
        representation == "tfidf_svd"
        and metadata_tfidf.shape[1] > 1
        and max_components < metadata_tfidf.shape[1]
        and float(np.var(metadata_tfidf)) > 1e-12
    ):
        svd_model = TruncatedSVD(n_components=max_components, random_state=42)
        metadata_svd = svd_model.fit_transform(metadata_tfidf).astype(np.float32)
    else:
        metadata_svd = metadata_tfidf[:, :max_components].astype(np.float32)
    if metadata_svd.shape[1] < svd_dim:
        pad = np.zeros((n_rows, svd_dim - metadata_svd.shape[1]), dtype=np.float32)
        metadata_svd = np.concatenate([metadata_svd, pad], axis=1)
    elif metadata_svd.shape[1] > svd_dim:
        metadata_svd = metadata_svd[:, :svd_dim].astype(np.float32)

    if representation == "binary":
        metadata = metadata_binary
        feature_names = list(binary_feature_names)
        feature_groups = list(binary_feature_groups)
    elif representation == "tfidf":
        metadata = metadata_tfidf
        feature_names = list(binary_feature_names)
        feature_groups = list(binary_feature_groups)
    else:
        metadata = metadata_svd
        feature_names = [f"metadata_svd::{idx:03d}" for idx in range(metadata.shape[1])]
        feature_groups = [
            {"feature": name, "group": "metadata_svd", "weight": 1.0}
            for name in feature_names
        ]

    if completeness_parts:
        completeness = np.mean(np.stack(completeness_parts, axis=1), axis=1).astype(np.float32)
    else:
        completeness = np.zeros(n_rows, dtype=np.float32)
    artifacts = {
        "metadata_binary": metadata_binary.astype(np.float32),
        "metadata_tfidf": metadata_tfidf.astype(np.float32),
        "metadata_svd": metadata_svd.astype(np.float32),
        "metadata_svd_model": svd_model,
        "metadata_binary_feature_names": binary_feature_names,
        "metadata_feature_groups": feature_groups,
        "metadata_binary_feature_groups": binary_feature_groups,
        "metadata_group_weights": group_weights,
        "metadata_representation": representation,
        "metadata_use_artist": bool(metadata_use_artist),
        "metadata_svd_dim": int(svd_dim),
    }
    return metadata, feature_names, vocab, has_any, completeness, artifacts


def _label_to_id(value: object) -> int:
    if pd.isna(value):
        return -1
    text = str(value).strip()
    if not text:
        return -1
    upper = text.upper()
    if upper in {"Q1", "1", "QUADRANT_1", "QUADRANT1"}:
        return 0
    if upper in {"Q2", "2", "QUADRANT_2", "QUADRANT2"}:
        return 1
    if upper in {"Q3", "3", "QUADRANT_3", "QUADRANT3"}:
        return 2
    if upper in {"Q4", "4", "QUADRANT_4", "QUADRANT4"}:
        return 3
    try:
        numeric = int(float(text))
    except ValueError:
        return -1
    return numeric if 0 <= numeric <= 3 else -1


def _derive_labels_from_va(
    va: np.ndarray,
    available: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    labels = np.full(int(va.shape[0]), -1, dtype=np.int64)
    valid = available.astype(bool) & np.isfinite(va[:, 0]) & np.isfinite(va[:, 1])
    high_valence = va[:, 0] >= float(threshold)
    high_arousal = va[:, 1] >= float(threshold)
    labels[valid & high_valence & high_arousal] = 0
    labels[valid & ~high_valence & high_arousal] = 1
    labels[valid & ~high_valence & ~high_arousal] = 2
    labels[valid & high_valence & ~high_arousal] = 3
    return labels


def _available_view_mean_va(
    audio: np.ndarray,
    has_audio: np.ndarray,
    lyrics: np.ndarray,
    has_lyrics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    audio_mask = has_audio.astype(np.float32).reshape(-1, 1)
    lyrics_mask = has_lyrics.astype(np.float32).reshape(-1, 1)
    weights = audio_mask + lyrics_mask
    summed = audio * audio_mask + lyrics * lyrics_mask
    mean_va = np.divide(
        summed,
        np.maximum(weights, 1.0),
        out=np.zeros_like(summed, dtype=np.float32),
        where=weights > 0,
    )
    return mean_va.astype(np.float32), weights.reshape(-1) > 0


def _resolve_labels(
    combined: pd.DataFrame,
    *,
    audio: np.ndarray,
    has_audio: np.ndarray,
    lyrics: np.ndarray,
    has_lyrics: np.ndarray,
    original_va: np.ndarray,
    has_original: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    label_col = _optional_column(
        combined.columns,
        ["labels_emotion", "label_emotion", "quadrant", "label", "emotion_label"],
    )
    if label_col is not None:
        labels = np.asarray([_label_to_id(value) for value in combined[label_col].tolist()], dtype=np.int64)
        primary_source = f"explicit_column:{label_col}"
    else:
        labels = np.full(len(combined), -1, dtype=np.int64)
        primary_source = "derived_original_va"

    original_labels = _derive_labels_from_va(original_va, has_original)
    view_mean_va, has_view_mean = _available_view_mean_va(audio, has_audio, lyrics, has_lyrics)
    view_mean_labels = _derive_labels_from_va(view_mean_va, has_view_mean)

    missing = labels < 0
    use_original = missing & (original_labels >= 0)
    labels[use_original] = original_labels[use_original]

    missing = labels < 0
    use_view_mean = missing & (view_mean_labels >= 0)
    labels[use_view_mean] = view_mean_labels[use_view_mean]

    unresolved = labels < 0
    info = {
        "primary_source": primary_source,
        "explicit_column": label_col,
        "derived_from_original_va": int(use_original.sum()),
        "derived_from_view_mean_va": int(use_view_mean.sum()),
        "unresolved": int(unresolved.sum()),
        "thresholds": {"valence": 0.5, "arousal": 0.5},
        "mapping": {
            "Q1": "valence>=0.5 and arousal>=0.5",
            "Q2": "valence<0.5 and arousal>=0.5",
            "Q3": "valence<0.5 and arousal<0.5",
            "Q4": "valence>=0.5 and arousal<0.5",
        },
    }
    return labels.astype(np.int64), info


def _load_previous_splits(previous_manifest: Optional[str]) -> Dict[str, str]:
    if not previous_manifest:
        return {}
    if not os.path.exists(previous_manifest):
        raise FileNotFoundError(f"Missing previous manifest '{previous_manifest}'.")
    previous = pd.read_csv(previous_manifest)
    id_col = _optional_column(previous.columns, ["track_id", "identifier", "song_id", "id"])
    split_col = _optional_column(previous.columns, ["split"])
    if id_col is None or split_col is None:
        raise ValueError("previous_manifest must contain track_id/identifier and split columns.")
    return {
        str(row[id_col]).strip(): str(row[split_col]).strip().lower()
        for _, row in previous.iterrows()
        if str(row.get(id_col, "")).strip()
    }


def _hash_split(track_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{track_id}|{int(seed)}".encode("utf-8")).hexdigest()
    ratio = int(digest[:12], 16) / float(16**12)
    if ratio < 0.70:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def _assign_splits(track_ids: Sequence[str], seed: int, previous_manifest: Optional[str]) -> List[str]:
    previous = _load_previous_splits(previous_manifest)
    splits: List[str] = []
    for track_id in track_ids:
        prior = previous.get(str(track_id))
        if prior in {"train", "val", "test"}:
            splits.append(prior)
        else:
            splits.append(_hash_split(str(track_id), seed))
    return splits


def _json_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split_json(out_dir: str, track_ids: Sequence[str], splits: Sequence[str]) -> None:
    payload: Dict[str, Any] = {
        "protocol": "70_15_15",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        indices = [idx for idx, value in enumerate(splits) if value == split]
        payload["splits"][split] = {
            "indices": indices,
            "track_ids": [str(track_ids[idx]) for idx in indices],
        }
    with open(os.path.join(out_dir, "split_70_15_15.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_aligned_placeholders(
    out_aligned_root: Optional[str],
    track_ids: Sequence[str],
    lyric_ids: Sequence[str],
    labels: Sequence[int],
) -> None:
    if not out_aligned_root:
        return
    os.makedirs(out_aligned_root, exist_ok=True)
    quadrants = [MUSIC_LABEL_NAMES.get(int(label), "") for label in labels]
    frame = pd.DataFrame(
        {
            "Audio_Song": list(track_ids),
            "Lyric_Song": list(lyric_ids),
            "Quadrant": quadrants,
        }
    )
    frame.to_csv(os.path.join(out_aligned_root, "aligned_audio_metadata.csv"), index=False, encoding="utf-8")
    frame.to_csv(os.path.join(out_aligned_root, "aligned_lyrics_metadata.csv"), index=False, encoding="utf-8")


def prepare_unimodal_dataset(
    *,
    combined_csv: str,
    out_processed_dir: str,
    out_aligned_root: Optional[str] = None,
    audio_split_dir: Optional[str] = None,
    lyrics_split_dir: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    previous_manifest: Optional[str] = None,
    split_policy: str = "preserve_then_hash",
    seed: int = 42,
    dataset_version: str = "v1",
    max_tokens_per_field: int = 512,
    metadata_use_artist: bool = False,
    metadata_representation: str = "binary",
    metadata_svd_dim: int = 32,
    metadata_group_weights: Optional[object] = None,
) -> Dict[str, Any]:
    del audio_split_dir, lyrics_split_dir  # Reserved for compatibility with upstream split folders.
    if split_policy != "preserve_then_hash":
        raise ValueError("Only split_policy='preserve_then_hash' is currently supported.")

    combined = _read_csv(combined_csv)
    os.makedirs(out_processed_dir, exist_ok=True)

    id_col = _optional_column(
        combined.columns,
        ["track_id", "song_id", "identifier", "id", "audio_song", "song", "title"],
    )
    if id_col is None:
        track_ids = [f"track_{idx:06d}" for idx in range(len(combined))]
    else:
        id_values = combined[id_col].where(combined[id_col].notna(), "").astype(str).str.strip()
        track_ids = [
            value if value else f"track_{idx:06d}"
            for idx, value in enumerate(id_values.tolist())
        ]

    lyric_id_col = _optional_column(combined.columns, ["lyric_identifier", "lyrics_song", "lyric_song"])
    if lyric_id_col is not None:
        lyric_values = combined[lyric_id_col].where(combined[lyric_id_col].notna(), "").astype(str).str.strip()
        lyric_ids = [
            value if value else str(track_ids[idx])
            for idx, value in enumerate(lyric_values.tolist())
        ]
    else:
        lyric_ids = [str(item) for item in track_ids]

    audio, has_audio = _stack_va(
        combined,
        valence_candidates=["audio_valence", "valence_audio"],
        arousal_candidates=["audio_arousal", "arousal_audio"],
        view_name="audio",
    )
    lyrics, has_lyrics = _stack_va(
        combined,
        valence_candidates=["lyrics_valence", "lyric_valence", "text_valence"],
        arousal_candidates=["lyrics_arousal", "lyric_arousal", "text_arousal"],
        view_name="lyrics",
    )
    original_va, _has_original = _stack_va(
        combined,
        valence_candidates=["original_valence", "valence_original"],
        arousal_candidates=["original_arousal", "arousal_original"],
        view_name="original",
    )

    metadata_frame = _resolve_metadata_frame(combined, metadata_csv, track_ids, id_col or "")
    (
        metadata,
        metadata_feature_names,
        metadata_vocab,
        has_metadata,
        metadata_completeness,
        metadata_artifacts,
    ) = _build_metadata_matrix(
        metadata_frame=metadata_frame,
        max_tokens_per_field=max_tokens_per_field,
        metadata_use_artist=metadata_use_artist,
        metadata_representation=metadata_representation,
        metadata_svd_dim=metadata_svd_dim,
        metadata_group_weights=metadata_group_weights,
    )

    view_mask = np.stack([has_audio, has_lyrics, has_metadata], axis=1).astype(np.float32)
    both_audio_lyrics = has_audio & has_lyrics
    signed_va_diff = (audio - lyrics).astype(np.float32)
    signed_va_diff[~both_audio_lyrics] = 0.0
    va_diff = np.abs(audio - lyrics).astype(np.float32)
    va_diff[~both_audio_lyrics] = 0.0
    consistency = np.zeros(len(combined), dtype=np.float32)
    if both_audio_lyrics.any():
        consistency[both_audio_lyrics] = 1.0 / (
            1.0 + np.linalg.norm(va_diff[both_audio_lyrics], axis=1)
        )

    labels, label_source_info = _resolve_labels(
        combined,
        audio=audio,
        has_audio=has_audio,
        lyrics=lyrics,
        has_lyrics=has_lyrics,
        original_va=original_va,
        has_original=_has_original,
    )
    quadrants = [MUSIC_LABEL_NAMES.get(int(label), "") for label in labels.tolist()]

    splits = _assign_splits(track_ids, seed, previous_manifest)

    np.save(os.path.join(out_processed_dir, "audio.npy"), audio.astype(np.float32))
    np.save(os.path.join(out_processed_dir, "lyrics.npy"), lyrics.astype(np.float32))
    np.save(os.path.join(out_processed_dir, "metadata.npy"), metadata.astype(np.float32))
    np.save(os.path.join(out_processed_dir, "metadata_binary.npy"), metadata_artifacts["metadata_binary"])
    np.save(os.path.join(out_processed_dir, "metadata_tfidf.npy"), metadata_artifacts["metadata_tfidf"])
    np.save(os.path.join(out_processed_dir, "metadata_svd.npy"), metadata_artifacts["metadata_svd"])
    np.save(os.path.join(out_processed_dir, "view_mask.npy"), view_mask.astype(np.float32))
    np.save(os.path.join(out_processed_dir, "consistency.npy"), consistency.astype(np.float32))
    np.save(os.path.join(out_processed_dir, "va_diff.npy"), va_diff.astype(np.float32))
    np.save(os.path.join(out_processed_dir, "signed_va_diff.npy"), signed_va_diff.astype(np.float32))
    np.save(os.path.join(out_processed_dir, "labels_emotion.npy"), labels.astype(np.int64))
    np.save(os.path.join(out_processed_dir, "original_va.npy"), original_va.astype(np.float32))
    diff_observed = both_audio_lyrics.astype(np.float32)
    np.save(os.path.join(out_processed_dir, "diff_observed.npy"), diff_observed)

    with open(os.path.join(out_processed_dir, "metadata_feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_feature_names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_processed_dir, "metadata_binary_feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_artifacts["metadata_binary_feature_names"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_processed_dir, "metadata_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_processed_dir, "metadata_feature_groups.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_artifacts["metadata_feature_groups"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_processed_dir, "metadata_binary_feature_groups.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_artifacts["metadata_binary_feature_groups"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_processed_dir, "metadata_group_weights.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_artifacts["metadata_group_weights"], f, ensure_ascii=False, indent=2)
    svd_model = metadata_artifacts.get("metadata_svd_model")
    if svd_model is not None:
        with open(os.path.join(out_processed_dir, "metadata_svd_model.pkl"), "wb") as f:
            pickle.dump(svd_model, f)

    metadata_schema = {
        "feature_names": metadata_feature_names,
        "binary_feature_names": metadata_artifacts["metadata_binary_feature_names"],
        "feature_groups": metadata_artifacts["metadata_feature_groups"],
        "binary_feature_groups": metadata_artifacts["metadata_binary_feature_groups"],
        "group_weights": metadata_artifacts["metadata_group_weights"],
        "vocab": metadata_vocab,
        "list_fields": list(DEFAULT_LIST_METADATA_FIELDS)
        + (list(OPTIONAL_IDENTITY_FIELDS) if metadata_use_artist else []),
        "optional_identity_fields": list(OPTIONAL_IDENTITY_FIELDS),
        "numeric_fields": list(NUMERIC_METADATA_FIELDS),
        "metadata_use_artist": bool(metadata_use_artist),
        "metadata_representation": str(metadata_artifacts["metadata_representation"]),
        "metadata_svd_dim": int(metadata_artifacts["metadata_svd_dim"]),
    }
    schema_payload = {
        "va_order": ["Valence", "Arousal"],
        "view_mask_columns": ["has_audio", "has_lyrics", "has_metadata"],
        "derived_feature_files": [
            "consistency.npy",
            "va_diff.npy",
            "signed_va_diff.npy",
            "diff_observed.npy",
            "metadata_binary.npy",
            "metadata_tfidf.npy",
            "metadata_svd.npy",
        ],
        "source_columns": list(combined.columns),
        "metadata_schema": metadata_schema,
        "label_source": label_source_info,
    }
    schema_hash = _json_hash(schema_payload)
    schema_payload["schema_hash"] = schema_hash
    metadata_schema["schema_hash"] = schema_hash
    with open(os.path.join(out_processed_dir, "metadata_schema.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_schema, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_processed_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, ensure_ascii=False, indent=2)

    manifest = pd.DataFrame(
        {
            "index": np.arange(len(combined), dtype=np.int64),
            "track_id": [str(item) for item in track_ids],
            "identifier": [str(item) for item in track_ids],
            "lyric_identifier": [str(item) for item in lyric_ids],
            "source_file": os.path.abspath(combined_csv),
            "split": splits,
            "has_audio": has_audio.astype(bool),
            "has_lyrics": has_lyrics.astype(bool),
            "has_metadata": has_metadata.astype(bool),
            "metadata_completeness": metadata_completeness.astype(np.float32),
            "quadrant": quadrants,
        }
    )
    for source_field in ("Artist", "Title"):
        column = _optional_column(metadata_frame.columns, [source_field.lower(), source_field])
        if column is not None:
            manifest[source_field.lower()] = metadata_frame[column].fillna("").astype(str).tolist()

    canonical_columns = ["index", "identifier", "lyric_identifier", "quadrant"]
    for source_field in ("Artist", "Title", "Genres", "Moods", "MoodsAll", "Themes", "Styles"):
        column = _optional_column(metadata_frame.columns, [source_field.lower(), source_field])
        if column is not None:
            manifest_name = source_field.lower() if source_field in {"Artist", "Title"} else source_field
            if manifest_name not in manifest.columns:
                manifest[manifest_name] = metadata_frame[column].fillna("").astype(str).tolist()
            canonical_columns.append(manifest_name)
    manifest.to_csv(os.path.join(out_processed_dir, "dataset_manifest.csv"), index=False, encoding="utf-8")
    canonical_columns = [column for column in dict.fromkeys(canonical_columns) if column in manifest.columns]
    canonical = manifest[canonical_columns].copy()
    canonical = canonical.rename(
        columns={
            "identifier": "Audio_Song",
            "lyric_identifier": "Lyric_Song",
            "quadrant": "Quadrant",
            "artist": "Artist",
            "title": "Title",
        }
    )
    canonical.to_csv(os.path.join(out_processed_dir, "canonical_metadata.csv"), index=False, encoding="utf-8")

    track_columns = ["index", "identifier", "lyric_identifier", "quadrant"]
    for optional in ("artist", "title"):
        if optional in manifest.columns:
            track_columns.append(optional)
    track_index = manifest[track_columns].copy()
    track_index.to_csv(os.path.join(out_processed_dir, "track_index.tsv"), sep="\t", index=False, encoding="utf-8")
    _write_split_json(out_processed_dir, track_ids, splits)
    _write_aligned_placeholders(out_aligned_root, track_ids, lyric_ids, labels.tolist())

    dataset_hash = _json_hash(
        {
            "combined_csv_hash": _file_hash(combined_csv),
            "schema_hash": schema_hash,
            "num_samples": len(combined),
            "dataset_version": dataset_version,
        }
    )
    meta = {
        "num_clusters": 4,
        "label_names": {str(key): value for key, value in MUSIC_LABEL_NAMES.items()},
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "schema_hash": schema_hash,
        "source_combined_csv": os.path.abspath(combined_csv),
        "label_source": label_source_info,
    }
    with open(os.path.join(out_processed_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return PreparedDatasetResult(
        processed_dir=str(out_processed_dir),
        aligned_root=str(out_aligned_root) if out_aligned_root else None,
        num_samples=len(combined),
        metadata_dim=int(metadata.shape[1]),
        schema_hash=schema_hash,
        dataset_hash=dataset_hash,
    ).as_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare upstream unimodal VA CSV for CVCL clustering.")
    parser.add_argument("--combined_csv", "--input_csv", dest="combined_csv", required=True)
    parser.add_argument("--audio_split_dir", default=None)
    parser.add_argument("--lyrics_split_dir", default=None)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--previous_manifest", default=None)
    parser.add_argument("--out_processed_dir", "--output_dir", dest="out_processed_dir", required=True)
    parser.add_argument("--out_aligned_root", default=None)
    parser.add_argument("--split_policy", default="preserve_then_hash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_version", "--dataset_name", dest="dataset_version", default="v1")
    parser.add_argument("--va_order", default="valence_arousal", choices=["valence_arousal"],
                        help="Input/output VA order. Only valence_arousal is supported.")
    parser.add_argument("--metadata_policy", default="report_only",
                        help="Accepted for run-script compatibility; metadata is prepared as report-only downstream.")
    parser.add_argument("--split_protocol", default="70_15_15",
                        help="Accepted for run-script compatibility; current strict processor writes split_70_15_15.json.")
    parser.add_argument("--max_tokens_per_field", type=int, default=512)
    parser.add_argument("--metadata_use_artist", default="false")
    parser.add_argument("--metadata_representation", default="binary", choices=["binary", "tfidf", "tfidf_svd"])
    parser.add_argument("--metadata_svd_dim", type=int, default=32)
    parser.add_argument("--metadata_group_weights", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_unimodal_dataset(
        combined_csv=str(args.combined_csv),
        out_processed_dir=str(args.out_processed_dir),
        out_aligned_root=args.out_aligned_root,
        audio_split_dir=args.audio_split_dir,
        lyrics_split_dir=args.lyrics_split_dir,
        metadata_csv=args.metadata_csv,
        previous_manifest=args.previous_manifest,
        split_policy=str(args.split_policy),
        seed=int(args.seed),
        dataset_version=str(args.dataset_version),
        max_tokens_per_field=int(args.max_tokens_per_field),
        metadata_use_artist=_parse_bool(args.metadata_use_artist),
        metadata_representation=str(args.metadata_representation),
        metadata_svd_dim=int(args.metadata_svd_dim),
        metadata_group_weights=args.metadata_group_weights,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
