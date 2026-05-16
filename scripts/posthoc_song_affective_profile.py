"""Build post-hoc affective profiles for fixed Dataset-S cluster outputs.

This module is intentionally report-only: it reads existing assignments and
never refits or rewrites cluster or tension labels.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RANDOM_SEED = 42
EPS = 1e-8
MIN_RADIUS_GROUP_SIZE = 3

SONG_ID_COLUMNS = ["song_id", "identifier", "lyric_identifier", "Song", "song", "id", "track_id"]
CLUSTER_ID_COLUMNS = ["cluster", "cluster_id", "label", "region_id"]
BALANCED_VALENCE_COLUMNS = ["balanced_valence", "balanced_v", "c_valence"]
BALANCED_AROUSAL_COLUMNS = ["balanced_arousal", "balanced_a", "c_arousal"]
TENSION_DV_COLUMNS = ["tension_dv", "dv", "delta_v", "lyrics_minus_audio_valence"]
TENSION_DA_COLUMNS = ["tension_da", "da", "delta_a", "lyrics_minus_audio_arousal"]
TENSION_NORM_COLUMNS = ["tension_norm", "norm", "delta_norm"]
TENSION_LABEL_COLUMNS = ["tension_label", "tension_subtype", "subtype"]
TENSION_NAME_COLUMNS = ["tension_name", "paper_tension_name", "tension_subtype_label", "subtype_name"]
TITLE_COLUMNS = ["title", "Title", "track_title", "song_title", "name"]
ARTIST_COLUMNS = ["artist", "Artist", "performer", "artists"]

REGION_NAMES = {
    0: "Subdued Melancholy",
    1: "Gentle Warmth",
    2: "Volatile Intensity",
    3: "Playful Vitality",
}

REGION_DESCRIPTORS = {
    0: ["subdued melancholy", "low arousal", "melancholic", "introspective", "somber"],
    1: ["gentle warmth", "calm-positive", "warm", "soft", "romantic"],
    2: ["volatile intensity", "negative-active", "tense", "aggressive", "high arousal"],
    3: ["playful vitality", "positive-active", "energetic", "bright", "danceable"],
}

TENSION_DESCRIPTORS = {
    "concordant": ["affective concordance", "audio-lyric agreement"],
    "uplift": ["lyric valence uplift", "lyrics brighten the affect"],
    "tempering": ["lyric valence tempering", "lyrics darken the affect"],
    "intensification": ["lyric arousal intensification", "lyrics intensify the affect"],
    "softening": ["lyric arousal softening", "lyrics soften the affect"],
    "high_tension": ["high cross-modal tension", "audio-lyric contrast"],
}


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, **kwargs)


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {_normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(_normalize_column_name(candidate))
        if match is not None:
            return str(match)
    return None


def _require_column(frame: pd.DataFrame, candidates: Sequence[str], label: str, path: Path) -> str:
    column = _find_column(frame, candidates)
    if column is None:
        raise ValueError(f"{path} must contain a {label} column; tried {list(candidates)}.")
    return column


def _path_score(path: Path, preferred_names: Sequence[str]) -> int:
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]
    score = 0
    if "all" in parts:
        score += 200
    if any(split in parts for split in ("train", "val", "test")):
        score -= 50
    for index, preferred_name in enumerate(preferred_names):
        preferred = preferred_name.lower()
        if name == preferred:
            score += 100 - index
        elif preferred.replace(".csv", "") in name:
            score += 30 - index
    return score


def _has_column_groups(path: Path, column_groups: Sequence[Sequence[str]]) -> bool:
    try:
        header = _read_csv(path, nrows=0)
    except Exception:
        return False
    return all(_find_column(header, group) is not None for group in column_groups)


def _find_csv_with_columns(
    run_dir: Path,
    *,
    column_groups: Sequence[Sequence[str]],
    preferred_names: Sequence[str],
) -> Path:
    candidates = [
        path
        for path in Path(run_dir).expanduser().rglob("*.csv")
        if _has_column_groups(path, column_groups)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a CSV under {run_dir} with required column groups: {column_groups}."
        )
    return max(candidates, key=lambda path: (_path_score(path, preferred_names), -len(path.parts), str(path)))


def _song_id_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _coerce_cluster_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and np.allclose(numeric.to_numpy(float), np.round(numeric.to_numpy(float))):
        return numeric.astype(int)
    return series.astype("string").str.strip()


def _to_float(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        missing = int(numeric.isna().sum())
        raise ValueError(f"{label} contains {missing} non-numeric or missing values.")
    return numeric.astype(float)


def _deduplicate_by_song_id(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    duplicated = frame["song_id"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "song_id"].head(5).tolist()
        raise ValueError(f"{source} contains duplicate song_id rows, e.g. {examples}.")
    return frame


def _load_cluster_va_frame(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cluster_path = _find_csv_with_columns(
        run_dir,
        column_groups=[SONG_ID_COLUMNS, CLUSTER_ID_COLUMNS],
        preferred_names=["cluster_assignments.csv", "assignments.csv"],
    )
    cluster_raw = _read_csv(cluster_path)
    song_col = _require_column(cluster_raw, SONG_ID_COLUMNS, "song_id", cluster_path)
    cluster_col = _require_column(cluster_raw, CLUSTER_ID_COLUMNS, "cluster_id", cluster_path)
    cluster_frame = pd.DataFrame(
        {
            "song_id": _song_id_series(cluster_raw[song_col]),
            "cluster_id": _coerce_cluster_id(cluster_raw[cluster_col]),
        }
    )
    cluster_frame = _deduplicate_by_song_id(cluster_frame, str(cluster_path))

    valence_col = _find_column(cluster_raw, BALANCED_VALENCE_COLUMNS)
    arousal_col = _find_column(cluster_raw, BALANCED_AROUSAL_COLUMNS)
    if valence_col is not None and arousal_col is not None:
        va_frame = pd.DataFrame(
            {
                "song_id": _song_id_series(cluster_raw[song_col]),
                "balanced_valence": _to_float(cluster_raw[valence_col], "balanced_valence"),
                "balanced_arousal": _to_float(cluster_raw[arousal_col], "balanced_arousal"),
            }
        )
        va_path = cluster_path
    else:
        va_path = _find_csv_with_columns(
            run_dir,
            column_groups=[SONG_ID_COLUMNS, BALANCED_VALENCE_COLUMNS, BALANCED_AROUSAL_COLUMNS],
            preferred_names=["cluster_assignments.csv", "assignments.csv", "canonical_affect_regions.csv"],
        )
        va_raw = _read_csv(va_path)
        va_song_col = _require_column(va_raw, SONG_ID_COLUMNS, "song_id", va_path)
        valence_col = _require_column(va_raw, BALANCED_VALENCE_COLUMNS, "balanced_valence", va_path)
        arousal_col = _require_column(va_raw, BALANCED_AROUSAL_COLUMNS, "balanced_arousal", va_path)
        va_frame = pd.DataFrame(
            {
                "song_id": _song_id_series(va_raw[va_song_col]),
                "balanced_valence": _to_float(va_raw[valence_col], "balanced_valence"),
                "balanced_arousal": _to_float(va_raw[arousal_col], "balanced_arousal"),
            }
        )
    va_frame = _deduplicate_by_song_id(va_frame, str(va_path))
    merged = cluster_frame.merge(va_frame, on="song_id", how="left", validate="one_to_one")
    if merged[["balanced_valence", "balanced_arousal"]].isna().any().any():
        raise ValueError("Could not attach balanced VA columns to every cluster assignment row.")
    return merged, {"cluster_assignments": str(cluster_path), "balanced_va": str(va_path)}


def _standardize_tension_assignment_frame(path: Path) -> pd.DataFrame:
    raw = _read_csv(path)
    song_col = _require_column(raw, SONG_ID_COLUMNS, "song_id", path)
    label_col = _require_column(raw, TENSION_LABEL_COLUMNS, "tension_label", path)
    cluster_col = _find_column(raw, CLUSTER_ID_COLUMNS)
    name_col = _find_column(raw, TENSION_NAME_COLUMNS)
    frame = pd.DataFrame(
        {
            "song_id": _song_id_series(raw[song_col]),
            "tension_label": raw[label_col].astype("string").fillna("").str.strip(),
        }
    )
    if cluster_col is not None:
        frame["cluster_id_tension"] = _coerce_cluster_id(raw[cluster_col])
    if name_col is not None:
        frame["tension_name"] = raw[name_col].astype("string").fillna("").str.strip()
    else:
        frame["tension_name"] = frame["tension_label"]
    return _deduplicate_by_song_id(frame, str(path))


def _standardize_tension_value_frame(path: Path) -> pd.DataFrame:
    raw = _read_csv(path)
    song_col = _require_column(raw, SONG_ID_COLUMNS, "song_id", path)
    dv_col = _require_column(raw, TENSION_DV_COLUMNS, "tension_dv", path)
    da_col = _require_column(raw, TENSION_DA_COLUMNS, "tension_da", path)
    norm_col = _require_column(raw, TENSION_NORM_COLUMNS, "tension_norm", path)
    frame = pd.DataFrame(
        {
            "song_id": _song_id_series(raw[song_col]),
            "tension_dv": _to_float(raw[dv_col], "tension_dv"),
            "tension_da": _to_float(raw[da_col], "tension_da"),
            "tension_norm": _to_float(raw[norm_col], "tension_norm"),
        }
    )
    return _deduplicate_by_song_id(frame, str(path))


def _load_tension_frame(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    assignment_path = _find_csv_with_columns(
        run_dir,
        column_groups=[SONG_ID_COLUMNS, TENSION_LABEL_COLUMNS],
        preferred_names=["tension_subtype_assignments.csv", "tension_micro_assignments.csv", "tension_micro_probe.csv"],
    )
    assignment = _standardize_tension_assignment_frame(assignment_path)
    if _has_column_groups(assignment_path, [TENSION_DV_COLUMNS, TENSION_DA_COLUMNS, TENSION_NORM_COLUMNS]):
        values = _standardize_tension_value_frame(assignment_path)
        values_path = assignment_path
    else:
        values_path = _find_csv_with_columns(
            run_dir,
            column_groups=[SONG_ID_COLUMNS, TENSION_DV_COLUMNS, TENSION_DA_COLUMNS, TENSION_NORM_COLUMNS],
            preferred_names=["tension_subtype_assignments.csv", "tension_micro_assignments.csv", "tension_micro_probe.csv"],
        )
        values = _standardize_tension_value_frame(values_path)
    merged = assignment.merge(values, on="song_id", how="left", validate="one_to_one")
    if merged[["tension_dv", "tension_da", "tension_norm"]].isna().any().any():
        raise ValueError("Could not attach tension columns to every tension assignment row.")
    return merged, {"tension_assignments": str(assignment_path), "tension_values": str(values_path)}


def _load_metadata(path: Path) -> pd.DataFrame:
    raw = _read_csv(path)
    song_col = _require_column(raw, SONG_ID_COLUMNS, "song_id", path)
    title_col = _find_column(raw, TITLE_COLUMNS)
    artist_col = _find_column(raw, ARTIST_COLUMNS)
    frame = pd.DataFrame({"song_id": _song_id_series(raw[song_col])})
    frame["title"] = raw[title_col].astype("string").fillna("").str.strip() if title_col else ""
    frame["artist"] = raw[artist_col].astype("string").fillna("").str.strip() if artist_col else ""
    return frame.drop_duplicates(subset=["song_id"], keep="first")


def _cluster_sort_key(value: Any) -> Tuple[int, str]:
    text = str(value)
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0)), text
    return sys.maxsize, text


def _cluster_int(value: Any) -> Optional[int]:
    key, _text = _cluster_sort_key(value)
    return None if key == sys.maxsize else key


def _cluster_token(value: Any) -> str:
    number = _cluster_int(value)
    if number is not None:
        return f"C{number}"
    text = str(value)
    return text if text.upper().startswith("C") else f"C{text}"


def _cluster_name(value: Any) -> str:
    number = _cluster_int(value)
    if number in REGION_NAMES:
        return REGION_NAMES[int(number)]
    return _cluster_token(value)


def _cluster_display(value: Any) -> str:
    return f"{_cluster_token(value)} {_cluster_name(value)}"


def _safe_median_positive(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    positive = finite[finite > EPS]
    if positive.size == 0:
        return float(fallback)
    return float(max(np.median(positive), EPS))


def _softmax_from_distances(distances: np.ndarray) -> np.ndarray:
    scores = -0.5 * np.square(np.asarray(distances, dtype=float))
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    total = float(weights.sum())
    if total <= EPS:
        return np.ones_like(weights) / float(len(weights))
    return weights / total


def _centrality_percentile(group_distances: np.ndarray, value: float) -> float:
    sorted_values = np.sort(np.asarray(group_distances, dtype=float))
    if sorted_values.size == 0:
        return 0.0
    rank_left = np.searchsorted(sorted_values, value, side="left")
    return float(np.clip(1.0 - rank_left / sorted_values.size, 0.0, 1.0))


def _ecdf_percentile(values: np.ndarray, value: float) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if sorted_values.size == 0:
        return 0.0
    rank_right = np.searchsorted(sorted_values, value, side="right")
    return float(np.clip(rank_right / sorted_values.size, 0.0, 1.0))


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > EPS:
        return mad
    std = float(np.std(finite))
    if std > EPS:
        return std
    return 1.0


def _compute_region_profiles(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[Any]]:
    result = frame.reset_index(drop=True).copy()
    clusters = sorted(result["cluster_id"].drop_duplicates().tolist(), key=_cluster_sort_key)
    x = result[["balanced_valence", "balanced_arousal"]].to_numpy(dtype=float)
    centers: Dict[Any, np.ndarray] = {}
    raw_distances = np.zeros((len(result), len(clusters)), dtype=float)
    cluster_to_index = {cluster: index for index, cluster in enumerate(clusters)}

    for index, cluster in enumerate(clusters):
        member_mask = result["cluster_id"].eq(cluster).to_numpy()
        centers[cluster] = x[member_mask].mean(axis=0)
        raw_distances[:, index] = np.linalg.norm(x - centers[cluster], axis=1)

    assigned_indices = result["cluster_id"].map(cluster_to_index).to_numpy(dtype=int)
    assigned_raw_distances = raw_distances[np.arange(len(result)), assigned_indices]
    global_radius = _safe_median_positive(assigned_raw_distances, fallback=1.0)
    radii: Dict[Any, float] = {}
    for cluster in clusters:
        cluster_index = cluster_to_index[cluster]
        values = raw_distances[result["cluster_id"].eq(cluster).to_numpy(), cluster_index]
        if len(values) < MIN_RADIUS_GROUP_SIZE:
            radii[cluster] = global_radius
        else:
            radii[cluster] = _safe_median_positive(values, fallback=global_radius)

    normalized_distances = np.zeros_like(raw_distances)
    weights = np.zeros_like(raw_distances)
    for index, cluster in enumerate(clusters):
        normalized_distances[:, index] = raw_distances[:, index] / max(radii[cluster], EPS)
    for row_index in range(len(result)):
        weights[row_index, :] = _softmax_from_distances(normalized_distances[row_index, :])

    for index, cluster in enumerate(clusters):
        token = _cluster_token(cluster)
        result[f"D_region_{token}"] = normalized_distances[:, index]
        result[f"w_region_{token}"] = weights[:, index]

    assigned_distances = normalized_distances[np.arange(len(result)), assigned_indices]
    result["region_confidence"] = weights[np.arange(len(result)), assigned_indices]
    typicalities = np.zeros(len(result), dtype=float)
    for cluster in clusters:
        cluster_index = cluster_to_index[cluster]
        member_mask = result["cluster_id"].eq(cluster).to_numpy()
        distribution = normalized_distances[member_mask, cluster_index]
        for row_index in np.where(member_mask)[0]:
            typicalities[row_index] = _centrality_percentile(distribution, normalized_distances[row_index, cluster_index])
    result["region_typicality"] = typicalities

    nearest_alt_clusters: List[str] = []
    nearest_alt_weights: List[float] = []
    margins: List[float] = []
    for row_index in range(len(result)):
        assigned_index = assigned_indices[row_index]
        alternative_indices = [index for index in range(len(clusters)) if index != assigned_index]
        second_index = min(alternative_indices, key=lambda index: normalized_distances[row_index, index])
        nearest_alt_clusters.append(_cluster_display(clusters[second_index]))
        nearest_alt_weights.append(float(weights[row_index, second_index]))
        margins.append(float(normalized_distances[row_index, second_index] - normalized_distances[row_index, assigned_index]))
    result["nearest_alt_cluster"] = nearest_alt_clusters
    result["nearest_alt_cluster_weight"] = nearest_alt_weights
    result["region_margin"] = margins

    for cluster in clusters:
        token = _cluster_token(cluster)
        member_x = x[result["cluster_id"].eq(cluster).to_numpy()]
        if len(member_x) < MIN_RADIUS_GROUP_SIZE:
            covariance = np.eye(2, dtype=float)
        else:
            covariance = np.cov(member_x, rowvar=False)
            if covariance.shape != (2, 2):
                covariance = np.eye(2, dtype=float)
        trace_scale = max(float(np.trace(covariance)) / 2.0, EPS)
        covariance = 0.90 * covariance + 0.10 * trace_scale * np.eye(2, dtype=float) + EPS * np.eye(2, dtype=float)
        inverse = np.linalg.pinv(covariance)
        diffs = x - centers[cluster]
        result[f"D_mahalanobis_{token}"] = np.sqrt(np.sum((diffs @ inverse) * diffs, axis=1))

    result["cluster_name"] = result["cluster_id"].map(_cluster_name)
    return result, clusters


def _compute_tension_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result["tension_name"] = result["tension_name"].where(result["tension_name"].astype(str).str.len() > 0, result["tension_label"])
    result["z_tension_dv"] = 0.0
    result["z_tension_da"] = 0.0

    for cluster in sorted(result["cluster_id"].drop_duplicates().tolist(), key=_cluster_sort_key):
        mask = result["cluster_id"].eq(cluster)
        dv_values = result.loc[mask, "tension_dv"].to_numpy(dtype=float)
        da_values = result.loc[mask, "tension_da"].to_numpy(dtype=float)
        dv_median = float(np.median(dv_values))
        da_median = float(np.median(da_values))
        result.loc[mask, "z_tension_dv"] = (dv_values - dv_median) / _robust_scale(dv_values)
        result.loc[mask, "z_tension_da"] = (da_values - da_median) / _robust_scale(da_values)

    result["tension_key"] = result["cluster_id"].map(_cluster_token) + "|" + result["tension_label"].astype(str)
    z = result[["z_tension_dv", "z_tension_da"]].to_numpy(dtype=float)
    subtype_centers: Dict[str, np.ndarray] = {}
    subtype_clusters: Dict[str, Any] = {}
    subtype_raw_assigned = np.zeros(len(result), dtype=float)
    for key, group in result.groupby("tension_key", sort=False):
        indices = group.index.to_numpy()
        subtype_centers[key] = z[indices].mean(axis=0)
        subtype_clusters[key] = group["cluster_id"].iloc[0]
        subtype_raw_assigned[indices] = np.linalg.norm(z[indices] - subtype_centers[key], axis=1)

    global_radius = _safe_median_positive(subtype_raw_assigned, fallback=1.0)
    subtype_radii: Dict[str, float] = {}
    for key, group in result.groupby("tension_key", sort=False):
        indices = group.index.to_numpy()
        values = subtype_raw_assigned[indices]
        if len(values) < MIN_RADIUS_GROUP_SIZE:
            subtype_radii[key] = global_radius
        else:
            subtype_radii[key] = _safe_median_positive(values, fallback=global_radius)

    assigned_distances = np.zeros(len(result), dtype=float)
    assigned_weights = np.zeros(len(result), dtype=float)
    tension_weights_json: List[str] = []
    for row_index, row in result.iterrows():
        cluster = row["cluster_id"]
        keys = [key for key, key_cluster in subtype_clusters.items() if key_cluster == cluster]
        distances = []
        for key in keys:
            raw_distance = float(np.linalg.norm(z[row_index] - subtype_centers[key]))
            distances.append(raw_distance / max(subtype_radii[key], EPS))
        weights = _softmax_from_distances(np.asarray(distances, dtype=float))
        label_weights = {key.split("|", 1)[1]: float(weight) for key, weight in zip(keys, weights)}
        assigned_key = row["tension_key"]
        assigned_position = keys.index(assigned_key)
        assigned_distances[row_index] = float(distances[assigned_position])
        assigned_weights[row_index] = float(weights[assigned_position])
        tension_weights_json.append(json.dumps(label_weights, ensure_ascii=False, sort_keys=True))

    result["D_tension_assigned"] = assigned_distances
    result["w_tension_assigned"] = assigned_weights
    result["tension_weights_json"] = tension_weights_json

    typicalities = np.zeros(len(result), dtype=float)
    for key, group in result.groupby("tension_key", sort=False):
        values = result.loc[group.index, "D_tension_assigned"].to_numpy(dtype=float)
        for row_index in group.index.to_numpy():
            typicalities[row_index] = _centrality_percentile(values, result.loc[row_index, "D_tension_assigned"])
    result["tension_typicality"] = typicalities

    for output_column, source_column in (
        ("tension_strength_percentile", "tension_norm"),
        ("dv_percentile_within_region", "tension_dv"),
        ("da_percentile_within_region", "tension_da"),
        ("abs_dv_percentile_within_region", "abs_tension_dv"),
        ("abs_da_percentile_within_region", "abs_tension_da"),
    ):
        if source_column == "abs_tension_dv":
            source_values = result["tension_dv"].abs()
        elif source_column == "abs_tension_da":
            source_values = result["tension_da"].abs()
        else:
            source_values = result[source_column]
        result[output_column] = 0.0
        for cluster, group in result.groupby("cluster_id", sort=False):
            values = source_values.loc[group.index].to_numpy(dtype=float)
            for row_index in group.index.to_numpy():
                value = float(source_values.loc[row_index])
                result.loc[row_index, output_column] = _ecdf_percentile(values, value)
    return result


def _add_descriptor_weight(weights: Dict[str, float], descriptors: Iterable[str], weight: float) -> None:
    if not math.isfinite(weight) or weight <= 0.0:
        return
    for descriptor in descriptors:
        weights[descriptor] = max(float(weights.get(descriptor, 0.0)), float(weight))


def _tension_descriptor_keys(tension_label: str, tension_name: str, strength_percentile: float) -> List[str]:
    text = f"{tension_label} {tension_name}".lower()
    keys: List[str] = []
    if any(token in text for token in ("consistent", "concordant", "agreement", "aligned")):
        keys.append("concordant")
    if any(token in text for token in ("bright", "uplift", "lift", "brighter")):
        keys.append("uplift")
    if any(token in text for token in ("dark", "temper", "shadow", "darker")):
        keys.append("tempering")
    if any(token in text for token in ("intensif", "activation", "activated")):
        keys.append("intensification")
    if any(token in text for token in ("soft", "soften", "deactivat")):
        keys.append("softening")
    if strength_percentile >= 0.75 or any(token in text for token in ("high tension", "contrast")):
        keys.append("high_tension")
    return keys or ["concordant"]


def _descriptor_profile(row: pd.Series, clusters: Sequence[Any]) -> List[Dict[str, Any]]:
    descriptor_weights: Dict[str, float] = {}
    assigned_cluster = row["cluster_id"]
    assigned_cluster_number = _cluster_int(assigned_cluster)
    assigned_token = _cluster_token(assigned_cluster)
    assigned_region_weight = float(row.get(f"w_region_{assigned_token}", 0.0))
    _add_descriptor_weight(
        descriptor_weights,
        REGION_DESCRIPTORS.get(int(assigned_cluster_number), [_cluster_name(assigned_cluster)])
        if assigned_cluster_number is not None
        else [_cluster_name(assigned_cluster)],
        assigned_region_weight * float(row["region_typicality"]),
    )

    for cluster in clusters:
        if cluster == assigned_cluster:
            continue
        token = _cluster_token(cluster)
        weight = float(row.get(f"w_region_{token}", 0.0))
        if weight >= 0.15:
            cluster_number = _cluster_int(cluster)
            descriptors = (
                REGION_DESCRIPTORS.get(int(cluster_number), [_cluster_name(cluster)])
                if cluster_number is not None
                else [_cluster_name(cluster)]
            )
            _add_descriptor_weight(descriptor_weights, descriptors, weight * 0.50)

    tension_base_weight = (
        float(row["w_tension_assigned"])
        * max(0.35, float(row["tension_strength_percentile"]))
        * float(row["tension_typicality"])
    )
    for key in _tension_descriptor_keys(str(row["tension_label"]), str(row["tension_name"]), float(row["tension_strength_percentile"])):
        _add_descriptor_weight(descriptor_weights, TENSION_DESCRIPTORS[key], tension_base_weight)

    strength_weight = max(0.35, float(row["tension_strength_percentile"])) * max(0.50, float(row["tension_typicality"]))
    if float(row["tension_dv"]) > EPS:
        _add_descriptor_weight(
            descriptor_weights,
            TENSION_DESCRIPTORS["uplift"],
            strength_weight * float(row["dv_percentile_within_region"]),
        )
    elif float(row["tension_dv"]) < -EPS:
        _add_descriptor_weight(
            descriptor_weights,
            TENSION_DESCRIPTORS["tempering"],
            strength_weight * float(row["abs_dv_percentile_within_region"]),
        )

    if float(row["tension_da"]) > EPS:
        _add_descriptor_weight(
            descriptor_weights,
            TENSION_DESCRIPTORS["intensification"],
            strength_weight * float(row["da_percentile_within_region"]),
        )
    elif float(row["tension_da"]) < -EPS:
        _add_descriptor_weight(
            descriptor_weights,
            TENSION_DESCRIPTORS["softening"],
            strength_weight * float(row["abs_da_percentile_within_region"]),
        )

    if not descriptor_weights:
        return []
    max_weight = max(descriptor_weights.values())
    normalized = [
        {"descriptor": descriptor, "weight": round(float(weight / max_weight), 6)}
        for descriptor, weight in descriptor_weights.items()
    ]
    normalized.sort(key=lambda item: (-item["weight"], item["descriptor"]))
    return normalized[:8]


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _direction_phrases(row: pd.Series) -> Tuple[str, str]:
    phrases_zh: List[str] = []
    phrases_en: List[str] = []
    dv = float(row["tension_dv"])
    da = float(row["tension_da"])
    if dv > EPS:
        phrases_zh.append("歌词侧相对音频侧更明亮/更正向")
        phrases_en.append("the lyrics are brighter or more positive than the audio")
    elif dv < -EPS:
        phrases_zh.append("歌词侧相对音频侧更暗/更负向")
        phrases_en.append("the lyrics darken the affect relative to the audio")
    else:
        phrases_zh.append("歌词与音频在效价方向上基本一致")
        phrases_en.append("lyrics and audio are closely aligned in valence")
    if da > EPS:
        phrases_zh.append("歌词侧更激活")
        phrases_en.append("the lyrics are more activating")
    elif da < -EPS:
        phrases_zh.append("歌词侧更柔和/低唤醒")
        phrases_en.append("the lyrics soften the arousal profile")
    else:
        phrases_zh.append("唤醒度方向差异很小")
        phrases_en.append("the arousal contrast is small")
    return "，".join(phrases_zh), "; ".join(phrases_en)


def _make_interpretations(row: pd.Series) -> Tuple[str, str]:
    descriptors = json.loads(row["top_descriptors_json"])
    descriptor_text = "、".join(item["descriptor"] for item in descriptors[:5]) or "no dominant descriptor"
    descriptor_text_en = ", ".join(item["descriptor"] for item in descriptors[:5]) or "no dominant descriptor"
    tension_display = f"{row['tension_label']} / {row['tension_name']}"
    direction_zh, direction_en = _direction_phrases(row)
    chinese = (
        f"该歌曲属于 {row['cluster_name']} 区域，region typicality 为 {_format_percent(row['region_typicality'])}，"
        f"其 balanced VA 位置相对该区域原型的 soft confidence 为 {_format_percent(row['region_confidence'])}。"
        f"最近的替代区域是 {row['nearest_alt_cluster']}（region margin={float(row['region_margin']):.3f}），"
        f"说明它与相邻情绪区域的距离关系可用于定性解释。"
        f"其 calibrated cross-modal tension / audio-lyric contrast profile 被标注为 {tension_display}，"
        f"tension strength percentile 为 {_format_percent(row['tension_strength_percentile'])}；{direction_zh}。"
        f"综合 descriptor profile 的高权重词包括：{descriptor_text}。"
    )
    english = (
        f"The song is assigned to {row['cluster_name']} with region typicality "
        f"{_format_percent(row['region_typicality'])} and assigned-region soft confidence "
        f"{_format_percent(row['region_confidence'])}. Its nearest alternative region is "
        f"{row['nearest_alt_cluster']} (region margin={float(row['region_margin']):.3f}), which indicates how close "
        f"the balanced VA location is to neighboring affective regions. The calibrated cross-modal tension / "
        f"audio-lyric contrast profile is {tension_display}, with tension strength percentile "
        f"{_format_percent(row['tension_strength_percentile'])}; {direction_en}. The top descriptor profile is: "
        f"{descriptor_text_en}."
    )
    return chinese, english


def _select_song_id_column(selected: pd.DataFrame, all_song_ids: set[str]) -> str:
    best_column: Optional[str] = None
    best_matches = -1
    for candidates in (SONG_ID_COLUMNS, ["identifier", "song_id", "Song", "song"]):
        for candidate in candidates:
            column = _find_column(selected, [candidate])
            if column is None:
                continue
            values = set(_song_id_series(selected[column]).tolist())
            matches = len(values.intersection(all_song_ids))
            if matches > best_matches:
                best_column = column
                best_matches = matches
    if best_column is None:
        raise ValueError("selected_songs_csv must contain a song_id/identifier column.")
    return best_column


def _write_selected_outputs(
    profile: pd.DataFrame,
    selected_songs_csv: Optional[Path],
    out_dir: Path,
) -> Tuple[pd.DataFrame, int, int]:
    if selected_songs_csv is None:
        (out_dir / "descriptor_weights_selected.json").write_text("{}", encoding="utf-8")
        return profile.iloc[0:0].copy(), 0, 0
    selected_raw = _read_csv(selected_songs_csv)
    selected_id_col = _select_song_id_column(selected_raw, set(profile["song_id"].astype(str).tolist()))
    selected = selected_raw.copy()
    selected["_selected_song_id"] = _song_id_series(selected[selected_id_col])
    selected_profile = selected[["_selected_song_id"]].merge(
        profile,
        left_on="_selected_song_id",
        right_on="song_id",
        how="left",
        sort=False,
    )
    found = selected_profile["song_id"].notna()
    found_profile = selected_profile.loc[found, profile.columns].copy()
    found_profile.to_csv(out_dir / "song_affective_profile_selected.csv", index=False, encoding="utf-8-sig")
    missing = selected.loc[~selected["_selected_song_id"].isin(set(profile["song_id"].astype(str).tolist()))].drop(
        columns=["_selected_song_id"]
    )
    missing.to_csv(out_dir / "missing_selected_songs.csv", index=False, encoding="utf-8-sig")
    descriptor_payload: Dict[str, Any] = {}
    for row in found_profile.itertuples(index=False):
        row_dict = row._asdict()
        descriptor_payload[str(row_dict["song_id"])] = {
            "title": row_dict.get("title", ""),
            "artist": row_dict.get("artist", ""),
            "cluster_name": row_dict.get("cluster_name", ""),
            "tension_label": row_dict.get("tension_label", ""),
            "tension_name": row_dict.get("tension_name", ""),
            "descriptors": json.loads(row_dict["top_descriptors_json"]),
            "cluster_weights": {
                column.replace("w_region_", ""): float(row_dict[column])
                for column in profile.columns
                if column.startswith("w_region_")
            },
            "tension_weights": json.loads(row_dict.get("tension_weights_json", "{}")),
        }
    (out_dir / "descriptor_weights_selected.json").write_text(
        json.dumps(descriptor_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return found_profile, int(found.sum()), int((~found).sum())


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _make_report(profile: pd.DataFrame, selected_profile: pd.DataFrame, out_dir: Path) -> None:
    lines: List[str] = [
        "# Song Affective Profile Report",
        "",
        "## Method",
        "",
        "This report is post-hoc only. It reads fixed Dataset-S v20.3 cluster assignments and fixed tension subtype assignments; it does not retrain models and does not change assignments.",
        "",
        "## Distance Definitions",
        "",
        "- Region prototypes are cluster means in balanced valence-arousal space.",
        "- Region distance is Euclidean distance divided by the assigned cluster's median in-cluster radius.",
        "- Region soft weights use exp(-0.5 * D^2) over all main regions.",
        "- Tension subtype prototypes are computed in region-local robust-scaled (tension_dv, tension_da) space.",
        "- tension_norm is used only as a strength percentile, not as a subtype-distance coordinate.",
        "",
        "## Typicality Definitions",
        "",
        "- Region typicality is a centrality percentile within the assigned cluster: higher values mean closer to that cluster prototype.",
        "- Tension typicality is the same centrality percentile within the assigned tension subtype.",
        "- Tension strength percentile is the empirical percentile of tension_norm within the same main cluster.",
        "",
        "## Selected Songs",
        "",
    ]
    if selected_profile.empty:
        lines.append("No selected song profile was generated.")
    else:
        lines.append("| song_id | title | artist | cluster | tension | region typicality | tension strength | top descriptors |")
        lines.append("|---|---|---|---|---|---:|---:|---|")
        for _, row in selected_profile.iterrows():
            descriptors = ", ".join(item["descriptor"] for item in json.loads(row["top_descriptors_json"])[:5])
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_escape(row["song_id"]),
                        _markdown_escape(row["title"]),
                        _markdown_escape(row["artist"]),
                        _markdown_escape(row["cluster_name"]),
                        _markdown_escape(f"{row['tension_label']} / {row['tension_name']}"),
                        _format_percent(float(row["region_typicality"])),
                        _format_percent(float(row["tension_strength_percentile"])),
                        _markdown_escape(descriptors),
                    ]
                )
                + " |"
            )
            lines.extend(["", f"**{_markdown_escape(row['song_id'])} Chinese**: {row['chinese_interpretation']}", ""])
            lines.extend([f"**{_markdown_escape(row['song_id'])} English**: {row['english_interpretation']}", ""])

    lines.extend(["", "## Cluster Representatives", ""])
    for cluster, group in profile.groupby("cluster_id", sort=False):
        ranked = group.sort_values(["region_typicality", "region_confidence"], ascending=False).head(10)
        lines.append(f"### {_cluster_display(cluster)}")
        lines.append("")
        lines.append("| rank | song_id | title | artist | region typicality | confidence |")
        lines.append("|---:|---|---|---|---:|---:|")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"| {rank} | {_markdown_escape(row['song_id'])} | {_markdown_escape(row['title'])} | "
                f"{_markdown_escape(row['artist'])} | {_format_percent(float(row['region_typicality']))} | "
                f"{_format_percent(float(row['region_confidence']))} |"
            )
        lines.append("")

    lines.extend(["", "## Tension Subtype Representatives", ""])
    for tension_label, group in profile.groupby("tension_label", sort=True):
        ranked = group.sort_values(["tension_typicality", "tension_strength_percentile"], ascending=False).head(10)
        tension_name = ranked["tension_name"].iloc[0] if not ranked.empty else ""
        lines.append(f"### {_markdown_escape(tension_label)} / {_markdown_escape(tension_name)}")
        lines.append("")
        lines.append("| rank | song_id | title | artist | cluster | tension typicality | strength percentile |")
        lines.append("|---:|---|---|---|---|---:|---:|")
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"| {rank} | {_markdown_escape(row['song_id'])} | {_markdown_escape(row['title'])} | "
                f"{_markdown_escape(row['artist'])} | {_markdown_escape(row['cluster_name'])} | "
                f"{_format_percent(float(row['tension_typicality']))} | "
                f"{_format_percent(float(row['tension_strength_percentile']))} |"
            )
        lines.append("")
    (out_dir / "song_affective_profile_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text[:80] or "song"


def _plot_bar(path: Path, labels: Sequence[str], values: Sequence[float], title: str) -> None:
    import matplotlib.pyplot as plt

    fig_width = max(6.0, min(12.0, 0.55 * len(labels) + 4.0))
    figure, axis = plt.subplots(figsize=(fig_width, 4.2))
    axis.bar(range(len(labels)), values, color="#4C78A8")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_figures(selected_profile: pd.DataFrame, out_dir: Path) -> None:
    if selected_profile.empty:
        return
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    for _, row in selected_profile.iterrows():
        prefix = _safe_filename(str(row["song_id"]))
        descriptors = json.loads(row["top_descriptors_json"])
        _plot_bar(
            figures_dir / f"{prefix}_descriptor_weights.png",
            [item["descriptor"] for item in descriptors],
            [float(item["weight"]) for item in descriptors],
            f"{row['song_id']} descriptor weights",
        )
        region_columns = [column for column in selected_profile.columns if column.startswith("w_region_")]
        _plot_bar(
            figures_dir / f"{prefix}_region_soft_weights.png",
            [column.replace("w_region_", "") for column in region_columns],
            [float(row[column]) for column in region_columns],
            f"{row['song_id']} region soft weights",
        )
        tension_weights = json.loads(row["tension_weights_json"])
        if tension_weights:
            _plot_bar(
                figures_dir / f"{prefix}_tension_soft_weights.png",
                list(tension_weights.keys()),
                [float(value) for value in tension_weights.values()],
                f"{row['song_id']} tension soft weights",
            )


def _ordered_output_columns(profile: pd.DataFrame, clusters: Sequence[Any]) -> List[str]:
    region_distance_cols = [f"D_region_{_cluster_token(cluster)}" for cluster in clusters]
    region_weight_cols = [f"w_region_{_cluster_token(cluster)}" for cluster in clusters]
    required = [
        "song_id",
        "title",
        "artist",
        "cluster_id",
        "cluster_name",
        "balanced_valence",
        "balanced_arousal",
        "region_typicality",
        "region_confidence",
        "region_margin",
        "nearest_alt_cluster",
        "nearest_alt_cluster_weight",
        *region_distance_cols,
        *region_weight_cols,
        "tension_label",
        "tension_name",
        "tension_dv",
        "tension_da",
        "tension_norm",
        "tension_typicality",
        "tension_strength_percentile",
        "dv_percentile_within_region",
        "da_percentile_within_region",
        "top_descriptors_json",
        "chinese_interpretation",
        "english_interpretation",
    ]
    extras = [
        column
        for column in profile.columns
        if column not in required
        and (
            column.startswith("D_mahalanobis_")
            or column
            in {
                "abs_dv_percentile_within_region",
                "abs_da_percentile_within_region",
                "D_tension_assigned",
                "w_tension_assigned",
                "tension_weights_json",
            }
        )
    ]
    return [column for column in required + extras if column in profile.columns]


def _descriptor_weight_bounds(profile: pd.DataFrame) -> Tuple[float, float]:
    values: List[float] = []
    for raw in profile["top_descriptors_json"].tolist():
        values.extend(float(item["weight"]) for item in json.loads(raw))
    if not values:
        return 0.0, 0.0
    return float(min(values)), float(max(values))


def _sanity_check(
    profile: pd.DataFrame,
    *,
    selected_found_count: int,
    selected_missing_count: int,
    source_files: Mapping[str, str],
) -> Dict[str, Any]:
    min_descriptor_weight, max_descriptor_weight = _descriptor_weight_bounds(profile)
    return {
        "random_seed": RANDOM_SEED,
        "source_files": dict(source_files),
        "total_songs": int(len(profile)),
        "number_of_clusters": int(profile["cluster_id"].nunique()),
        "cluster_sizes": {str(key): int(value) for key, value in profile["cluster_id"].value_counts().sort_index().items()},
        "subtype_sizes": {str(key): int(value) for key, value in profile["tension_label"].value_counts().sort_index().items()},
        "missing_metadata_count": int(((profile["title"] == "") & (profile["artist"] == "")).sum()),
        "selected_found_count": int(selected_found_count),
        "selected_missing_count": int(selected_missing_count),
        "min_region_typicality": float(profile["region_typicality"].min()),
        "max_region_typicality": float(profile["region_typicality"].max()),
        "min_tension_typicality": float(profile["tension_typicality"].min()),
        "max_tension_typicality": float(profile["tension_typicality"].max()),
        "min_descriptor_weight": min_descriptor_weight,
        "max_descriptor_weight": max_descriptor_weight,
    }


def run_posthoc_profile(
    *,
    run_dir: Path | str,
    song_metadata_csv: Path | str,
    selected_songs_csv: Optional[Path | str],
    out_dir: Path | str,
    make_figures: bool = True,
) -> Dict[str, Any]:
    np.random.seed(RANDOM_SEED)
    run_path = Path(run_dir).expanduser()
    metadata_path = Path(song_metadata_csv).expanduser()
    selected_path = Path(selected_songs_csv).expanduser() if selected_songs_csv else None
    output_path = Path(out_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    cluster_va, cluster_sources = _load_cluster_va_frame(run_path)
    tension, tension_sources = _load_tension_frame(run_path)
    profile = cluster_va.merge(tension, on="song_id", how="left", validate="one_to_one")
    if profile["tension_label"].isna().any():
        missing = profile.loc[profile["tension_label"].isna(), "song_id"].head(5).tolist()
        raise ValueError(f"Missing tension assignment for cluster songs, e.g. {missing}.")
    if "cluster_id_tension" in profile.columns:
        mismatch = profile["cluster_id_tension"].notna() & profile["cluster_id_tension"].ne(profile["cluster_id"])
        if mismatch.any():
            examples = profile.loc[mismatch, ["song_id", "cluster_id", "cluster_id_tension"]].head(5).to_dict("records")
            raise ValueError(f"Cluster assignment and tension assignment disagree: {examples}.")
        profile = profile.drop(columns=["cluster_id_tension"])

    metadata = _load_metadata(metadata_path)
    profile = profile.merge(metadata, on="song_id", how="left", validate="one_to_one")
    profile["title"] = profile["title"].fillna("")
    profile["artist"] = profile["artist"].fillna("")

    profile, clusters = _compute_region_profiles(profile)
    profile = _compute_tension_profiles(profile)
    profile["top_descriptors_json"] = [
        json.dumps(_descriptor_profile(row, clusters), ensure_ascii=False)
        for _, row in profile.iterrows()
    ]
    interpretations = [_make_interpretations(row) for _, row in profile.iterrows()]
    profile["chinese_interpretation"] = [item[0] for item in interpretations]
    profile["english_interpretation"] = [item[1] for item in interpretations]

    ordered_columns = _ordered_output_columns(profile, clusters)
    profile = profile[ordered_columns]
    profile.to_csv(output_path / "song_affective_profile_all.csv", index=False, encoding="utf-8-sig")

    selected_profile, selected_found_count, selected_missing_count = _write_selected_outputs(
        profile,
        selected_path,
        output_path,
    )
    _make_report(profile, selected_profile, output_path)
    if make_figures:
        _write_figures(selected_profile, output_path)

    sanity = _sanity_check(
        profile,
        selected_found_count=selected_found_count,
        selected_missing_count=selected_missing_count,
        source_files={**cluster_sources, **tension_sources, "metadata": str(metadata_path)},
    )
    (output_path / "sanity_check.json").write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sanity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build post-hoc prototype-weighted affective profiles for fixed Dataset-S assignments."
    )
    parser.add_argument("--run_dir", required=True, help="Dataset-S v20.3 training output directory.")
    parser.add_argument("--song_metadata_csv", required=True, help="CSV containing song_id/title/artist metadata.")
    parser.add_argument("--selected_songs_csv", default=None, help="Optional representative song candidate CSV.")
    parser.add_argument("--out_dir", required=True, help="Output directory for profile CSV/JSON/Markdown files.")
    parser.add_argument("--no_figures", action="store_true", help="Skip optional selected-song helper charts.")
    args = parser.parse_args()

    sanity = run_posthoc_profile(
        run_dir=args.run_dir,
        song_metadata_csv=args.song_metadata_csv,
        selected_songs_csv=args.selected_songs_csv,
        out_dir=args.out_dir,
        make_figures=not args.no_figures,
    )
    print(json.dumps(sanity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
