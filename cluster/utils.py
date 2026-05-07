"""Shared utilities for the cluster package.

Centralises duplicated helpers (JSON I/O, array loading, scaling,
normalisation) and provides an LRU-style ArrayCache that eliminates
redundant ``np.load`` calls when multiple dataset splits share the
same underlying ``.npy`` files.
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# ArrayCache — eliminates redundant np.load across dataset splits
# ---------------------------------------------------------------------------

class ArrayCache:
    """Thread-safe, path-keyed cache for ``.npy`` arrays.

    Typical usage::

        cache = ArrayCache()
        arr = cache.get("data/audio.npy", dtype=np.float32)
        # Second call with the same path returns the cached copy.
    """

    def __init__(self) -> None:
        self._store: Dict[str, np.ndarray] = {}

    def get(self, path: str, dtype: type = np.float32) -> np.ndarray:
        """Return the array at *path*, loading from disk on first access."""
        key = os.path.normpath(os.path.abspath(path))
        if key not in self._store:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing required processed file '{path}'."
                )
            self._store[key] = np.load(path).astype(dtype)
        return self._store[key]

    def clear(self) -> None:
        """Drop all cached arrays."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# Module-level singleton — importers can share one cache instance.
_GLOBAL_CACHE = ArrayCache()


def get_global_cache() -> ArrayCache:
    """Return the module-level :class:`ArrayCache` singleton."""
    return _GLOBAL_CACHE


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def safe_load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file, returning ``{}`` if the file does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Array I/O
# ---------------------------------------------------------------------------

def load_required_array(
    data_dir: str,
    name: str,
    dtype: type = np.float32,
    *,
    cache: Optional[ArrayCache] = None,
) -> np.ndarray:
    """Load ``<data_dir>/<name>.npy``, raising if missing.

    When *cache* is provided the array is loaded through the cache so
    that repeated calls for the same file avoid redundant disk I/O.
    """
    path = os.path.join(data_dir, f"{name}.npy")
    if cache is not None:
        return cache.get(path, dtype=dtype)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required processed file '{path}'. "
            "Run processing_merge_va.py first."
        )
    return np.load(path).astype(dtype)


def load_optional_array(
    data_dir: str,
    name: str,
    dtype: type = np.float32,
    *,
    cache: Optional[ArrayCache] = None,
) -> Optional[np.ndarray]:
    """Load ``<data_dir>/<name>.npy`` when present, otherwise return ``None``."""
    path = os.path.join(data_dir, f"{name}.npy")
    if not os.path.exists(path):
        return None
    if cache is not None:
        return cache.get(path, dtype=dtype)
    return np.load(path).astype(dtype)


# ---------------------------------------------------------------------------
# Scaling / normalisation
# ---------------------------------------------------------------------------

def fit_scaler_state(
    data_dir: str,
    split_protocol: str,
    view_names: Sequence[str],
    *,
    load_split_indices_fn: Any = None,
    cache: Optional[ArrayCache] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Compute per-view mean/std from the *train* split only.

    Parameters
    ----------
    load_split_indices_fn:
        Callable ``(data_dir, split_protocol) -> dict[str, ndarray]``.
        Injected to avoid circular imports (``loader.load_split_indices``).
    """
    if load_split_indices_fn is None:
        from cluster.data.loader import load_split_indices as _fn
        load_split_indices_fn = _fn

    split_indices = load_split_indices_fn(data_dir, split_protocol)
    train_idx = split_indices["train"]
    view_mask = load_optional_array(data_dir, "view_mask", np.float32, cache=cache)
    view_to_mask_col = {"audio": 0, "lyrics": 1, "metadata": 2}
    scaler_state: Dict[str, Dict[str, List[float]]] = {}
    for name in view_names:
        raw = load_required_array(data_dir, name, np.float32, cache=cache)
        sample = raw[train_idx]
        if view_mask is not None and name in view_to_mask_col:
            available = view_mask[train_idx, view_to_mask_col[name]].astype(bool)
            sample = sample[available]
        finite_rows = np.isfinite(sample).all(axis=1) if sample.size else np.zeros(0, dtype=bool)
        sample = sample[finite_rows]
        if sample.shape[0] == 0:
            mean = np.zeros(raw.shape[1], dtype=np.float32)
            std = np.ones(raw.shape[1], dtype=np.float32)
        else:
            mean = np.mean(sample, axis=0).astype(np.float32)
            std = np.std(sample, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        scaler_state[name] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return scaler_state


def apply_scale(
    x: np.ndarray,
    scaler_state: Dict[str, Dict[str, List[float]]],
    name: str,
) -> np.ndarray:
    """Z-score normalise *x* using pre-computed mean/std for *name*."""
    mean = np.asarray(scaler_state[name]["mean"], dtype=np.float32)
    std = np.asarray(scaler_state[name]["std"], dtype=np.float32)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    scaled = ((x - mean) / std).astype(np.float32)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# String / column normalisation
# ---------------------------------------------------------------------------

def normalize_split_key(value: str) -> str:
    """Canonicalise a split name (e.g. ``'validate'`` → ``'val'``)."""
    normalized = str(value).strip().lower()
    if normalized == "validate":
        return "val"
    return normalized


def normalize_col(name: str) -> str:
    """Lower-case, strip, and replace separators in a column name."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def find_column(
    columns: Sequence[str],
    candidates: Sequence[str],
    *,
    required: bool = False,
) -> Optional[str]:
    """Find the first column whose normalised name matches a candidate.

    Parameters
    ----------
    required:
        If ``True``, raise ``ValueError`` when no match is found.
    """
    normalized = {normalize_col(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    if required:
        raise ValueError(
            f"Missing required column. Candidates tried: {list(candidates)}; "
            f"available: {list(columns)}"
        )
    return None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across numpy, stdlib, and torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
