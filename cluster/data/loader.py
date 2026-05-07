from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cluster.config import MUSIC_LABEL_NAMES, parse_loader_scale_mode, parse_music_views, parse_split_protocol
from cluster.utils import (
    ArrayCache,
    get_global_cache,
    load_optional_array,
    load_required_array,
    normalize_split_key,
    safe_load_json,
)


def resolve_music_data_dir(path: str) -> str:
    if os.path.exists(os.path.join(path, "audio.npy")):
        return path
    alt_dir = os.path.join(path, "music")
    if os.path.exists(os.path.join(alt_dir, "audio.npy")):
        return alt_dir
    raise FileNotFoundError(
        f"Cannot find processed music dataset. Expected audio.npy in '{path}' or '{alt_dir}'."
    )


def _safe_load_meta(data_dir: str) -> Dict[str, Any]:
    return safe_load_json(os.path.join(data_dir, "meta.json"))


def _safe_label_names(data_dir: str) -> Dict[int, str]:
    meta = _safe_load_meta(data_dir)
    raw = meta.get("label_names", {})
    if raw:
        return {int(key): str(value) for key, value in raw.items()}
    return dict(MUSIC_LABEL_NAMES)


def _split_json_path(data_dir: str, split_protocol: str) -> str:
    return os.path.join(data_dir, f"split_{split_protocol}.json")


def load_split_indices(data_dir: str, split_protocol: str) -> Dict[str, np.ndarray]:
    protocol = parse_split_protocol(split_protocol)
    payload = safe_load_json(_split_json_path(data_dir, protocol))
    if not payload:
        raise FileNotFoundError(
            f"Missing split file '{_split_json_path(data_dir, protocol)}'. "
            "Run processing_merge_va.py first."
        )

    splits_obj = payload.get("splits", payload)
    out: Dict[str, np.ndarray] = {}
    for key, value in splits_obj.items():
        split_name = normalize_split_key(key)
        if isinstance(value, dict):
            indices = value.get("indices", [])
        else:
            indices = value
        out[split_name] = np.asarray(indices, dtype=np.int64)

    required = {"train", "val", "test"}
    missing = sorted(required - set(out.keys()))
    if missing:
        raise ValueError(
            f"Split file for protocol '{protocol}' is missing required splits: {missing}."
        )
    return out


def load_track_index(data_dir: str) -> Dict[str, List[str]]:
    path = os.path.join(data_dir, "track_index.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing track index file '{path}'. Run processing_merge_va.py first."
        )

    indices: List[int] = []
    identifiers: List[str] = []
    lyric_identifiers: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["index"])
            indices.append(idx)
            identifiers.append(str(row.get("identifier", "")).strip())
            lyric_identifiers.append(str(row.get("lyric_identifier", "")).strip())

    order = np.argsort(np.asarray(indices, dtype=np.int64))
    identifiers_arr = np.asarray(identifiers, dtype=object)[order].tolist()
    lyric_identifiers_arr = np.asarray(lyric_identifiers, dtype=object)[order].tolist()
    return {
        "identifier": identifiers_arr,
        "lyric_identifier": lyric_identifiers_arr,
    }


def _validate_same_length(arrays: Dict[str, np.ndarray]) -> int:
    lengths = {name: int(value.shape[0]) for name, value in arrays.items()}
    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) != 1:
        raise ValueError(f"Processed arrays must share the same length, got {lengths}.")
    return unique_lengths[0]


def fit_standard_train_only_scaler(
    data_dir: str,
    split_protocol: str,
    views: Sequence[str],
) -> Dict[str, Dict[str, List[float]]]:
    split_indices = load_split_indices(data_dir, split_protocol)
    train_idx = split_indices["train"]
    view_mask = load_optional_array(data_dir, "view_mask", np.float32)
    view_to_mask_col = {"audio": 0, "lyrics": 1, "metadata": 2}
    scaler_state: Dict[str, Dict[str, List[float]]] = {}
    for view in views:
        raw = load_required_array(data_dir, view, np.float32)
        sample = raw[train_idx]
        if view_mask is not None and view in view_to_mask_col:
            sample = sample[view_mask[train_idx, view_to_mask_col[view]].astype(bool)]
        finite_rows = np.isfinite(sample).all(axis=1) if sample.size else np.zeros(0, dtype=bool)
        sample = sample[finite_rows]
        if sample.shape[0] == 0:
            mean = np.zeros(raw.shape[1], dtype=np.float32)
            std = np.ones(raw.shape[1], dtype=np.float32)
        else:
            mean = np.mean(sample, axis=0).astype(np.float32)
            std = np.std(sample, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        scaler_state[view] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return scaler_state


def _apply_scale(
    x: np.ndarray,
    view_name: str,
    scale_mode: str,
    scaler_state: Optional[Dict[str, Dict[str, List[float]]]],
) -> np.ndarray:
    mode = parse_loader_scale_mode(scale_mode)
    if mode == "none":
        return x.astype(np.float32)
    if mode != "standard_train_only":
        raise ValueError(f"Unsupported scale mode '{mode}'.")
    if scaler_state is None or view_name not in scaler_state:
        raise ValueError(
            f"scale_mode='{mode}' requires scaler_state for view '{view_name}'."
        )
    mean = np.asarray(scaler_state[view_name]["mean"], dtype=np.float32)
    std = np.asarray(scaler_state[view_name]["std"], dtype=np.float32)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


@dataclass
class MusicDatasetArtifacts:
    train_dataset: "MusicMultiViewDataset"
    val_dataset: "MusicMultiViewDataset"
    test_dataset: "MusicMultiViewDataset"
    scaler_state: Optional[Dict[str, Dict[str, List[float]]]]


class MusicMultiViewDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        split_protocol: str,
        views: Sequence[str],
        scale_mode: str = "none",
        scaler_state: Optional[Dict[str, Dict[str, List[float]]]] = None,
        *,
        cache: Optional[ArrayCache] = None,
    ) -> None:
        self.data_dir = resolve_music_data_dir(data_dir)
        self.split = normalize_split_key(split)
        self.split_protocol = parse_split_protocol(split_protocol)
        self.view_names = parse_music_views(",".join(views))
        self.scale_mode = parse_loader_scale_mode(scale_mode)
        self.meta = _safe_load_meta(self.data_dir)
        self.label_names = _safe_label_names(self.data_dir)
        self.num_clusters = int(self.meta.get("num_clusters", 4))
        if self.num_clusters != 4:
            raise ValueError(
                f"Processed music dataset must expose num_clusters=4, got {self.num_clusters}."
            )

        _cache = cache or get_global_cache()
        audio_raw = load_required_array(self.data_dir, "audio", np.float32, cache=_cache)
        lyrics_raw = load_required_array(self.data_dir, "lyrics", np.float32, cache=_cache)
        consistency = load_required_array(self.data_dir, "consistency", np.float32, cache=_cache)
        va_diff = load_required_array(self.data_dir, "va_diff", np.float32, cache=_cache)
        labels = load_required_array(self.data_dir, "labels_emotion", np.int64, cache=_cache)
        view_mask = load_optional_array(self.data_dir, "view_mask", np.float32, cache=_cache)
        n_samples = _validate_same_length(
            {
                "audio": audio_raw,
                "lyrics": lyrics_raw,
                "consistency": consistency,
                "va_diff": va_diff,
                "labels_emotion": labels,
            }
        )
        if view_mask is None:
            view_mask = np.ones((n_samples, 3), dtype=np.float32)
        if view_mask.shape != (n_samples, 3):
            raise ValueError(f"view_mask.npy must have shape [{n_samples}, 3], got {view_mask.shape}.")

        track_index = load_track_index(self.data_dir)
        identifiers = track_index["identifier"]
        lyric_identifiers = track_index["lyric_identifier"]
        if len(identifiers) != n_samples or len(lyric_identifiers) != n_samples:
            raise ValueError(
                "track_index.tsv length does not match processed arrays. "
                f"Expected {n_samples}, got identifiers={len(identifiers)}."
            )

        split_indices = load_split_indices(self.data_dir, self.split_protocol)
        if self.split not in split_indices:
            raise ValueError(
                f"Unknown split '{self.split}'. Available: {sorted(split_indices.keys())}."
            )
        self.indices = split_indices[self.split]

        audio_split = audio_raw[self.indices]
        lyrics_split = lyrics_raw[self.indices]
        self.raw_views = {
            "audio": audio_split.astype(np.float32),
            "lyrics": lyrics_split.astype(np.float32),
        }
        self.view_mask = view_mask[self.indices].astype(np.float32)
        self.data_views = [
            _apply_scale(self.raw_views[view], view, self.scale_mode, scaler_state)
            for view in self.view_names
        ]
        view_to_mask_col = {"audio": 0, "lyrics": 1}
        for idx, view in enumerate(self.view_names):
            self.data_views[idx] = self.data_views[idx].copy()
            self.data_views[idx][self.view_mask[:, view_to_mask_col[view]] <= 0.0] = 0.0
        self.input_sizes = [int(view.shape[1]) for view in self.data_views]
        self.labels = labels[self.indices].astype(np.int64)
        self.consistency = consistency[self.indices].astype(np.float32)
        self.va_diff = va_diff[self.indices].astype(np.float32)
        self.identifiers = [identifiers[int(idx)] for idx in self.indices.tolist()]
        self.lyric_identifiers = [lyric_identifiers[int(idx)] for idx in self.indices.tolist()]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        views = [torch.from_numpy(view[item]) for view in self.data_views]
        aux = {
            "consistency": torch.tensor(self.consistency[item], dtype=torch.float32),
            "va_diff": torch.from_numpy(self.va_diff[item]),
            "identifier": self.identifiers[item],
            "lyric_identifier": self.lyric_identifiers[item],
            "audio_va": torch.from_numpy(self.raw_views["audio"][item]),
            "lyrics_va": torch.from_numpy(self.raw_views["lyrics"][item]),
            "view_mask": torch.from_numpy(self.view_mask[item]),
        }
        return views, int(self.labels[item]), aux


def create_music_datasets(
    data_dir: str,
    split_protocol: str,
    views: Sequence[str],
    scale_mode: str = "none",
) -> MusicDatasetArtifacts:
    mode = parse_loader_scale_mode(scale_mode)
    scaler_state = None
    if mode == "standard_train_only":
        scaler_state = fit_standard_train_only_scaler(
            data_dir=data_dir,
            split_protocol=split_protocol,
            views=views,
        )

    cache = ArrayCache()
    return MusicDatasetArtifacts(
        train_dataset=MusicMultiViewDataset(
            data_dir=data_dir,
            split="train",
            split_protocol=split_protocol,
            views=views,
            scale_mode=mode,
            scaler_state=scaler_state,
            cache=cache,
        ),
        val_dataset=MusicMultiViewDataset(
            data_dir=data_dir,
            split="val",
            split_protocol=split_protocol,
            views=views,
            scale_mode=mode,
            scaler_state=scaler_state,
            cache=cache,
        ),
        test_dataset=MusicMultiViewDataset(
            data_dir=data_dir,
            split="test",
            split_protocol=split_protocol,
            views=views,
            scale_mode=mode,
            scaler_state=scaler_state,
            cache=cache,
        ),
        scaler_state=scaler_state,
    )


def create_music_loader(
    dataset: MusicMultiViewDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        num_workers=max(int(num_workers), 0),
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
    )
