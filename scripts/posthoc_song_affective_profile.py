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

DIRECTIONAL_TENSION_KEYS = {"uplift", "tempering", "intensification", "softening", "high_tension"}
DIRECTIONAL_TENSION_DESCRIPTORS = {
    descriptor
    for key in DIRECTIONAL_TENSION_KEYS
    for descriptor in TENSION_DESCRIPTORS[key]
}
FINAL_CANDIDATE_TYPES = {"region_prototype", "region_representative", "tension_case"}
C2_SUPPLEMENT_TITLE_HINTS = {
    "born in the u.s.a.",
    "price you pay",
    "give me it",
    "maimed and slaughtered",
    "room 13",
}
EXPLICIT_TITLE_PATTERN = re.compile(
    r"\b(fuck(?:ing|er|ed)?|shit|bitch|cunt|dick|pussy|asshole|bastard|motherfucker|whore|slut)\b",
    re.IGNORECASE,
)


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, **kwargs)


def _output_file(out_dir: Path, stem: str, output_suffix: str, extension: str) -> Path:
    suffix = output_suffix if output_suffix.startswith("_") or output_suffix == "" else f"_{output_suffix}"
    return out_dir / f"{stem}{suffix}.{extension}"


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


def _region_role(row: pd.Series) -> str:
    margin = float(row["region_margin"])
    confidence = float(row["region_confidence"])
    typicality = float(row["region_typicality"])
    if margin < 0.0 or confidence < 0.60:
        return "boundary"
    if typicality >= 0.80 and confidence >= 0.75 and margin >= 0.50:
        return "prototype"
    if typicality >= 0.50 and confidence >= 0.70 and margin >= 0.30:
        return "representative"
    return "peripheral"


def _has_encoding_issue(*values: Any) -> bool:
    text = " ".join(str(value) for value in values if value is not None)
    return "\ufffd" in text or "锟" in text


def _has_explicit_title(*values: Any) -> bool:
    text = " ".join(str(value) for value in values if value is not None)
    return EXPLICIT_TITLE_PATTERN.search(text) is not None


def _add_review_flags(profile: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()
    result["region_role"] = [_region_role(row) for _, row in result.iterrows()]
    result["boundary_flag"] = result["region_role"].eq("boundary")
    result["descriptor_conflict_flag"] = result["nearest_alt_cluster_weight"].astype(float).gt(
        result["region_confidence"].astype(float)
    )
    result["encoding_issue_flag"] = [
        _has_encoding_issue(row["title"], row["artist"])
        for _, row in result.iterrows()
    ]
    result["explicit_title_flag"] = [
        _has_explicit_title(row["title"], row["artist"])
        for _, row in result.iterrows()
    ]
    result["selected_role"] = [_selected_role(row) for _, row in result.iterrows()]
    result["main_text_eligible"] = result["selected_role"].isin(
        ["region_prototype", "region_representative", "tension_case"]
    )
    return result


def _selected_role(row: pd.Series) -> str:
    if bool(row["encoding_issue_flag"]) or bool(row["explicit_title_flag"]):
        return "appendix_only"
    if bool(row["boundary_flag"]) or bool(row["descriptor_conflict_flag"]):
        return "boundary_case"
    if (
        float(row["tension_strength_percentile"]) >= 0.75
        and float(row["tension_typicality"]) >= 0.50
        and float(row["region_margin"]) > 0.0
    ):
        return "tension_case"
    if row["region_role"] == "prototype":
        return "region_prototype"
    if row["region_role"] == "representative":
        return "region_representative"
    return "appendix_only"


def _is_tension_case(row: pd.Series) -> bool:
    return (
        float(row["tension_strength_percentile"]) >= 0.75
        and float(row["tension_typicality"]) >= 0.50
        and float(row["region_margin"]) > 0.0
    )


def _add_descriptor_weight(weights: Dict[str, float], descriptors: Iterable[str], weight: float) -> None:
    if not math.isfinite(weight) or weight <= 0.0:
        return
    for descriptor in descriptors:
        weights[descriptor] = max(float(weights.get(descriptor, 0.0)), float(weight))


def _is_modality_consistent(row: pd.Series) -> bool:
    text = f"{row['tension_label']} {row['tension_name']}".lower()
    return any(token in text for token in ("consistent", "concordant", "agreement", "aligned"))


def _directional_tension_allowed(row: pd.Series) -> bool:
    strength = float(row["tension_strength_percentile"])
    if strength < 0.35:
        return False
    if not _is_modality_consistent(row):
        return True
    max_abs_delta_percentile = max(
        float(row["abs_dv_percentile_within_region"]),
        float(row["abs_da_percentile_within_region"]),
    )
    return strength >= 0.60 and max_abs_delta_percentile >= 0.75


def _tension_descriptor_keys(row: pd.Series) -> List[str]:
    text = f"{row['tension_label']} {row['tension_name']}".lower()
    strength_percentile = float(row["tension_strength_percentile"])
    if _is_modality_consistent(row) and not _directional_tension_allowed(row):
        return ["concordant"]
    if strength_percentile < 0.35:
        return ["concordant"]

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
    if bool(row.get("boundary_flag", False)):
        _add_descriptor_weight(
            descriptor_weights,
            ["boundary between assigned region and nearest alternative"],
            1.0 + max(assigned_region_weight, float(row["nearest_alt_cluster_weight"]), 0.01),
        )
        _add_descriptor_weight(
            descriptor_weights,
            [f"assigned region: {_cluster_name(assigned_cluster)}"],
            assigned_region_weight * 0.50,
        )
        _add_descriptor_weight(
            descriptor_weights,
            [f"nearest alternative: {row['nearest_alt_cluster']}"],
            float(row["nearest_alt_cluster_weight"]) * 0.50,
        )
    else:
        _add_descriptor_weight(
            descriptor_weights,
            REGION_DESCRIPTORS.get(int(assigned_cluster_number), [_cluster_name(assigned_cluster)])
            if assigned_cluster_number is not None
            else [_cluster_name(assigned_cluster)],
            assigned_region_weight * max(0.25, float(row["region_typicality"])),
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
                capped_weight = min(weight * 0.25, max(assigned_region_weight * 0.75, 0.01))
                _add_descriptor_weight(descriptor_weights, descriptors, capped_weight)

    tension_base_weight = (
        float(row["w_tension_assigned"])
        * max(0.35, float(row["tension_strength_percentile"]))
        * float(row["tension_typicality"])
    )
    for key in _tension_descriptor_keys(row):
        _add_descriptor_weight(descriptor_weights, TENSION_DESCRIPTORS[key], tension_base_weight)

    strength_weight = max(0.35, float(row["tension_strength_percentile"])) * max(0.50, float(row["tension_typicality"]))
    if _directional_tension_allowed(row):
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
    descriptors = [
        {
            "descriptor": descriptor,
            "raw_descriptor_score": round(float(weight), 6),
            "display_descriptor_weight": round(float(weight / max_weight), 6),
            "weight": round(float(weight / max_weight), 6),
        }
        for descriptor, weight in descriptor_weights.items()
    ]
    descriptors.sort(key=lambda item: (-item["raw_descriptor_score"], item["descriptor"]))
    return descriptors[:8]


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _margin_phrase(row: pd.Series) -> Tuple[str, str]:
    margin = float(row["region_margin"])
    nearest = row["nearest_alt_cluster"]
    if margin >= 1.0:
        return (
            f"该样本与最近替代区域 {nearest} 明显分离",
            f"it is clearly separated from the nearest alternative region, {nearest}",
        )
    if margin >= 0.30:
        return (
            f"该样本与最近替代区域 {nearest} 有一定接近性，但仍保留正 margin",
            f"it is moderately close to the nearest alternative region, {nearest}, while retaining a positive margin",
        )
    if margin >= 0.0:
        return (
            f"该样本靠近 {nearest} 的边界，适合作为边界邻近样本讨论",
            f"it is boundary-adjacent to {nearest}",
        )
    return (
        f"该样本在 post-hoc prototype 距离下更接近 {nearest}，只能作为边界/歧义案例使用",
        f"post-hoc prototype distance places it closer to {nearest}; use it only as a boundary or ambiguity case",
    )


def _role_phrase(row: pd.Series) -> Tuple[str, str]:
    role = str(row["region_role"])
    phrases = {
        "prototype": ("高度原型样本", "highly prototypical"),
        "representative": ("有代表性但并非最中心的样本", "representative but not central"),
        "peripheral": ("所属区域内的外围样本", "peripheral within the assigned region"),
        "boundary": ("边界样本，不应用作主区域证据", "a boundary case, not used as main region evidence"),
    }
    return phrases.get(role, phrases["peripheral"])


def _direction_phrases(row: pd.Series) -> Tuple[str, str]:
    if not _directional_tension_allowed(row):
        if float(row["tension_strength_percentile"]) < 0.35:
            return (
                "张力强度较低，因此不把小的音频-歌词方向差解释为主要证据",
                "tension strength is low, so small audio-lyric directional differences are not used as primary evidence",
            )
        return (
            "该样本标记为 modality-consistent，方向差未达到门控阈值，因此解释为音频与歌词基本一致",
            "the sample is modality-consistent and directional deltas do not pass the gate, so it is interpreted as broad audio-lyric agreement",
        )

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
    return "；".join(phrases_zh), "; ".join(phrases_en)


def _make_interpretations(row: pd.Series) -> Tuple[str, str]:
    descriptors = json.loads(row["top_descriptors_json"])
    descriptor_text = "、".join(item["descriptor"] for item in descriptors[:5]) or "no dominant descriptor"
    descriptor_text_en = ", ".join(item["descriptor"] for item in descriptors[:5]) or "no dominant descriptor"
    tension_display = f"{row['tension_label']} / {row['tension_name']}"
    direction_zh, direction_en = _direction_phrases(row)
    margin_zh, margin_en = _margin_phrase(row)
    role_zh, role_en = _role_phrase(row)
    chinese = (
        f"该歌曲属于 {row['cluster_name']} 区域，是{role_zh}；region typicality 为 "
        f"{_format_percent(row['region_typicality'])}，其 balanced VA 位置相对该区域原型的 "
        f"soft confidence 为 {_format_percent(row['region_confidence'])}。{margin_zh}"
        f"（region margin={float(row['region_margin']):.3f}）。其 calibrated cross-modal tension / "
        f"audio-lyric contrast profile 被标注为 {tension_display}，tension strength percentile 为 "
        f"{_format_percent(row['tension_strength_percentile'])}；{direction_zh}。"
        f"综合 descriptor profile 的高权重词包括：{descriptor_text}。"
    )
    english = (
        f"The song is assigned to {row['cluster_name']} and is {role_en}, with region typicality "
        f"{_format_percent(row['region_typicality'])} and assigned-region soft confidence "
        f"{_format_percent(row['region_confidence'])}; {margin_en} "
        f"(region margin={float(row['region_margin']):.3f}). The calibrated cross-modal tension / "
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
    output_suffix: str,
) -> Tuple[pd.DataFrame, int, int]:
    if selected_songs_csv is None:
        _output_file(out_dir, "descriptor_weights_selected", output_suffix, "json").write_text("{}", encoding="utf-8")
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
    found_profile.to_csv(
        _output_file(out_dir, "song_affective_profile_selected", output_suffix, "csv"),
        index=False,
        encoding="utf-8-sig",
    )
    missing = selected.loc[~selected["_selected_song_id"].isin(set(profile["song_id"].astype(str).tolist()))].drop(
        columns=["_selected_song_id"]
    )
    missing.to_csv(
        _output_file(out_dir, "missing_selected_songs", output_suffix, "csv"),
        index=False,
        encoding="utf-8-sig",
    )
    descriptor_payload: Dict[str, Any] = {}
    for row in found_profile.itertuples(index=False):
        row_dict = row._asdict()
        descriptor_payload[str(row_dict["song_id"])] = {
            "title": row_dict.get("title", ""),
            "artist": row_dict.get("artist", ""),
            "cluster_name": row_dict.get("cluster_name", ""),
            "region_role": row_dict.get("region_role", ""),
            "selected_role": row_dict.get("selected_role", ""),
            "candidate_type": row_dict.get("candidate_type", ""),
            "main_text_eligible": bool(row_dict.get("main_text_eligible", False)),
            "final_main_candidate": bool(row_dict.get("final_main_candidate", False)),
            "external_evidence_status": row_dict.get("external_evidence_status", "not_checked"),
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
    _output_file(out_dir, "descriptor_weights_selected", output_suffix, "json").write_text(
        json.dumps(descriptor_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return found_profile, int(found.sum()), int((~found).sum())


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _descriptor_labels(row: pd.Series, limit: int = 5) -> str:
    return ", ".join(item["descriptor"] for item in json.loads(row["top_descriptors_json"])[:limit])


def _append_song_table(lines: List[str], title: str, rows: pd.DataFrame) -> None:
    lines.extend(["", f"### {title}", ""])
    if rows.empty:
        lines.append("No songs matched this role.")
        return
    lines.append(
        "| song_id | title | artist | cluster | region role | tension | main text | region typicality | tension strength | top descriptors |"
    )
    lines.append("|---|---|---|---|---|---|---|---:|---:|---|")
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row["song_id"]),
                    _markdown_escape(row["title"]),
                    _markdown_escape(row["artist"]),
                    _markdown_escape(row["cluster_name"]),
                    _markdown_escape(row["region_role"]),
                    _markdown_escape(f"{row['tension_label']} / {row['tension_name']}"),
                    "yes" if bool(row["main_text_eligible"]) else "no",
                    _format_percent(float(row["region_typicality"])),
                    _format_percent(float(row["tension_strength_percentile"])),
                    _markdown_escape(_descriptor_labels(row)),
                ]
            )
            + " |"
        )


def _selected_role_rows(selected_profile: pd.DataFrame, roles: Sequence[str]) -> pd.DataFrame:
    return selected_profile[selected_profile["selected_role"].isin(roles)]


def _main_text_safe_rows(frame: pd.DataFrame) -> pd.DataFrame:
    candidate_column = "final_main_candidate" if "final_main_candidate" in frame.columns else "main_text_eligible"
    return frame[frame[candidate_column].astype(bool)]


def _rank_region_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    ranked = rows.copy()
    ranked["_review_hint_priority"] = [
        1
        if _cluster_int(row["cluster_id"]) == 2 and str(row["title"]).strip().lower() in C2_SUPPLEMENT_TITLE_HINTS
        else 0
        for _, row in ranked.iterrows()
    ]
    return ranked.sort_values(
        ["_review_hint_priority", "region_typicality", "region_confidence", "tension_strength_percentile", "song_id"],
        ascending=[False, False, False, False, True],
    )


def _rank_tension_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(
        ["tension_strength_percentile", "tension_typicality", "region_confidence", "song_id"],
        ascending=[False, False, False, True],
    )


def _final_candidate_columns(frame: pd.DataFrame) -> List[str]:
    preferred = [
        "final_table_role",
        "final_table_rank",
        "song_id",
        "title",
        "artist",
        "cluster_id",
        "cluster_name",
        "candidate_type",
        "region_role",
        "final_main_candidate",
        "main_text_eligible",
        "external_evidence_status",
        "region_typicality",
        "region_confidence",
        "region_margin",
        "tension_label",
        "tension_name",
        "tension_typicality",
        "tension_strength_percentile",
        "top_descriptor_raw_score",
        "top_descriptor_display_weight",
        "top_descriptors_json",
        "english_interpretation",
        "chinese_interpretation",
    ]
    return [column for column in preferred if column in frame.columns]


def _build_final_paper_candidate_table(profile: pd.DataFrame, clusters: Sequence[Any]) -> pd.DataFrame:
    rows: List[pd.Series] = []
    seen_song_ids: set[str] = set()
    for cluster in clusters:
        group = profile[profile["cluster_id"].eq(cluster)]
        final_group = _main_text_safe_rows(group)

        region_candidates = final_group[final_group["region_role"].isin(["prototype", "representative"])]
        for rank, (_, row) in enumerate(_rank_region_candidates(region_candidates).head(3).iterrows(), start=1):
            row_copy = row.copy()
            row_copy["final_table_role"] = "region_candidate"
            row_copy["final_table_rank"] = rank
            rows.append(row_copy)
            seen_song_ids.add(str(row["song_id"]))

        tension_candidates = final_group[
            final_group["candidate_type"].eq("tension_case")
            & ~final_group["song_id"].astype(str).isin(seen_song_ids)
        ]
        for rank, (_, row) in enumerate(_rank_tension_candidates(tension_candidates).head(2).iterrows(), start=1):
            row_copy = row.copy()
            row_copy["final_table_role"] = "tension_case"
            row_copy["final_table_rank"] = rank
            rows.append(row_copy)
            seen_song_ids.add(str(row["song_id"]))

    if not rows:
        return pd.DataFrame(columns=_final_candidate_columns(profile))
    frame = pd.DataFrame(rows)
    return frame[_final_candidate_columns(frame)]


def _write_final_paper_candidate_table(
    profile: pd.DataFrame,
    clusters: Sequence[Any],
    out_dir: Path,
    output_suffix: str,
) -> pd.DataFrame:
    final_candidates = _build_final_paper_candidate_table(profile, clusters)
    final_candidates.to_csv(
        _output_file(out_dir, "final_paper_candidate_table", output_suffix, "csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return final_candidates


def _boundary_case_coverage_rows(profile: pd.DataFrame, clusters: Sequence[Any]) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for cluster in clusters:
        group = profile[profile["cluster_id"].eq(cluster)]
        boundary_rows = group[group["candidate_type"].eq("boundary_case")]
        ranked = boundary_rows.sort_values(
            ["region_margin", "region_confidence", "song_id"],
            ascending=[True, False, True],
        )
        if not ranked.empty:
            rows.append(ranked.iloc[0].copy())
    if not rows:
        return profile.iloc[0:0].copy()
    return pd.DataFrame(rows)


def _append_final_candidate_table(lines: List[str], final_candidates: pd.DataFrame) -> None:
    lines.extend(["", "## Final Paper Candidate Table", ""])
    if final_candidates.empty:
        lines.append("No final main-text candidates passed the strict gates.")
        return
    lines.append(
        "| role | rank | song_id | title | artist | cluster | candidate type | region typicality | confidence | tension strength | external evidence |"
    )
    lines.append("|---|---:|---|---|---|---|---|---:|---:|---:|---|")
    for _, row in final_candidates.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row["final_table_role"]),
                    str(int(row["final_table_rank"])),
                    _markdown_escape(row["song_id"]),
                    _markdown_escape(row["title"]),
                    _markdown_escape(row["artist"]),
                    _markdown_escape(row["cluster_name"]),
                    _markdown_escape(row["candidate_type"]),
                    _format_percent(float(row["region_typicality"])),
                    _format_percent(float(row["region_confidence"])),
                    _format_percent(float(row["tension_strength_percentile"])),
                    _markdown_escape(row["external_evidence_status"]),
                ]
            )
            + " |"
        )


def _make_report(
    profile: pd.DataFrame,
    selected_profile: pd.DataFrame,
    out_dir: Path,
    output_suffix: str,
    final_candidates: Optional[pd.DataFrame] = None,
) -> None:
    clusters = list(profile["cluster_id"].drop_duplicates().tolist())
    if final_candidates is None:
        final_candidates = _build_final_paper_candidate_table(profile, clusters)
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
        "- For each candidate region k, region distance is Euclidean distance to region prototype k divided by the median in-cluster radius of region k.",
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
        region_rows = _selected_role_rows(selected_profile, ["region_prototype", "region_representative"])
        tension_rows = _selected_role_rows(selected_profile, ["tension_case"])
        boundary_rows = _selected_role_rows(selected_profile, ["boundary_case"])
        appendix_rows = _selected_role_rows(selected_profile, ["appendix_only"])
        _append_song_table(lines, "Region prototype songs", region_rows)
        _append_song_table(lines, "Tension case-study songs", tension_rows)
        _append_song_table(lines, "Boundary / ambiguity cases", boundary_rows)
        _append_song_table(lines, "Appendix-only candidates", appendix_rows)
        lines.extend(["", "### Selected Song Interpretations", ""])
        for _, row in selected_profile.iterrows():
            lines.extend(["", f"**{_markdown_escape(row['song_id'])} Chinese**: {row['chinese_interpretation']}", ""])
            lines.extend([f"**{_markdown_escape(row['song_id'])} English**: {row['english_interpretation']}", ""])

    _append_final_candidate_table(lines, final_candidates)

    boundary_rows = _boundary_case_coverage_rows(profile, clusters)
    _append_song_table(lines, "Full-table boundary coverage", boundary_rows)

    lines.extend(["", "## Cluster Representatives", ""])
    for cluster, group in profile.groupby("cluster_id", sort=False):
        candidates = _main_text_safe_rows(group)
        candidates = candidates[candidates["region_role"].isin(["prototype", "representative"])]
        ranked = candidates.sort_values(["region_typicality", "region_confidence"], ascending=False).head(10)
        lines.append(f"### {_cluster_display(cluster)}")
        lines.append("")
        if ranked.empty:
            lines.append("No main-text eligible representative songs matched this cluster.")
            lines.append("")
            continue
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
        ranked = _main_text_safe_rows(group).sort_values(
            ["tension_typicality", "tension_strength_percentile"],
            ascending=False,
        ).head(10)
        tension_name = ranked["tension_name"].iloc[0] if not ranked.empty else ""
        lines.append(f"### {_markdown_escape(tension_label)} / {_markdown_escape(tension_name)}")
        lines.append("")
        if ranked.empty:
            lines.append("No main-text eligible representative songs matched this tension subtype.")
            lines.append("")
            continue
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
    _output_file(out_dir, "song_affective_profile_report", output_suffix, "md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


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
        "region_role",
        "boundary_flag",
        "descriptor_conflict_flag",
        "encoding_issue_flag",
        "explicit_title_flag",
        "main_text_eligible",
        "selected_role",
        "candidate_type",
        "final_main_candidate",
        "external_evidence_status",
        "modality_consistent_directional_conflict_flag",
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
        "top_descriptor_raw_score",
        "top_descriptor_display_weight",
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
        values.extend(float(item["display_descriptor_weight"]) for item in json.loads(raw))
    if not values:
        return 0.0, 0.0
    return float(min(values)), float(max(values))


def _raw_descriptor_score_bounds(profile: pd.DataFrame) -> Tuple[float, float]:
    values: List[float] = []
    for raw in profile["top_descriptors_json"].tolist():
        values.extend(float(item["raw_descriptor_score"]) for item in json.loads(raw))
    if not values:
        return 0.0, 0.0
    return float(min(values)), float(max(values))


def _has_directional_descriptor(raw: str) -> bool:
    descriptors = {str(item["descriptor"]) for item in json.loads(raw)}
    return bool(descriptors.intersection(DIRECTIONAL_TENSION_DESCRIPTORS))


def _has_modality_consistent_directional_conflict(row: pd.Series) -> bool:
    return _is_modality_consistent(row) and _has_directional_descriptor(str(row["top_descriptors_json"]))


def _add_final_candidate_flags(profile: pd.DataFrame) -> pd.DataFrame:
    result = profile.copy()
    modality_conflict = [
        _has_modality_consistent_directional_conflict(row)
        for _, row in result.iterrows()
    ]
    result["modality_consistent_directional_conflict_flag"] = modality_conflict
    result["candidate_type"] = result["selected_role"].astype(str)

    appendix_mask = result["encoding_issue_flag"].astype(bool) | result["explicit_title_flag"].astype(bool)
    boundary_mask = result["boundary_flag"].astype(bool) | result["descriptor_conflict_flag"].astype(bool)
    result.loc[appendix_mask, "candidate_type"] = "appendix_only"
    result.loc[boundary_mask & ~appendix_mask, "candidate_type"] = "boundary_case"
    result.loc[result["modality_consistent_directional_conflict_flag"].astype(bool), "candidate_type"] = "appendix_only"
    result["selected_role"] = result["candidate_type"]

    main_text_mask = result["candidate_type"].isin(FINAL_CANDIDATE_TYPES)
    main_text_mask &= ~result["modality_consistent_directional_conflict_flag"].astype(bool)
    main_text_mask &= ~appendix_mask
    main_text_mask &= ~boundary_mask
    result["main_text_eligible"] = main_text_mask
    result["final_main_candidate"] = main_text_mask
    result["external_evidence_status"] = "not_checked"
    return result


def _sanity_check(
    profile: pd.DataFrame,
    *,
    selected_profile: pd.DataFrame,
    selected_found_count: int,
    selected_missing_count: int,
    source_files: Mapping[str, str],
) -> Dict[str, Any]:
    min_descriptor_weight, max_descriptor_weight = _descriptor_weight_bounds(profile)
    min_raw_descriptor_score, max_raw_descriptor_score = _raw_descriptor_score_bounds(profile)
    modality_consistent = [
        _is_modality_consistent(row) and _has_directional_descriptor(str(row["top_descriptors_json"]))
        for _, row in profile.iterrows()
    ]
    modality_conflict = pd.Series(modality_consistent, index=profile.index)
    main_text_modality_conflict = profile["main_text_eligible"].astype(bool) & modality_conflict
    final_candidates = (
        profile["final_main_candidate"].astype(bool)
        if "final_main_candidate" in profile.columns
        else profile["main_text_eligible"].astype(bool)
    )
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
        "selected_main_text_eligible_count": int(selected_profile["main_text_eligible"].sum()),
        "final_main_candidate_count": int(final_candidates.sum()),
        "candidate_type_counts": {
            str(key): int(value)
            for key, value in profile["candidate_type"].value_counts().sort_index().items()
        } if "candidate_type" in profile.columns else {},
        "negative_margin_count": int((profile["region_margin"].astype(float) < 0.0).sum()),
        "descriptor_conflict_count": int(profile["descriptor_conflict_flag"].sum()),
        "modality_consistent_with_directional_descriptor_count": int(sum(modality_consistent)),
        "main_text_modality_consistent_directional_conflict_count": int(main_text_modality_conflict.sum()),
        "explicit_or_encoding_issue_count": int((profile["explicit_title_flag"] | profile["encoding_issue_flag"]).sum()),
        "min_region_typicality": float(profile["region_typicality"].min()),
        "max_region_typicality": float(profile["region_typicality"].max()),
        "min_tension_typicality": float(profile["tension_typicality"].min()),
        "max_tension_typicality": float(profile["tension_typicality"].max()),
        "min_descriptor_weight": min_descriptor_weight,
        "max_descriptor_weight": max_descriptor_weight,
        "min_raw_descriptor_score": min_raw_descriptor_score,
        "max_raw_descriptor_score": max_raw_descriptor_score,
    }


def run_posthoc_profile(
    *,
    run_dir: Path | str,
    song_metadata_csv: Path | str,
    selected_songs_csv: Optional[Path | str],
    out_dir: Path | str,
    make_figures: bool = True,
    output_suffix: str = "",
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
    profile = _add_review_flags(profile)
    descriptor_profiles = [_descriptor_profile(row, clusters) for _, row in profile.iterrows()]
    profile["top_descriptors_json"] = [json.dumps(descriptors, ensure_ascii=False) for descriptors in descriptor_profiles]
    profile["top_descriptor_raw_score"] = [
        float(descriptors[0]["raw_descriptor_score"]) if descriptors else 0.0
        for descriptors in descriptor_profiles
    ]
    profile["top_descriptor_display_weight"] = [
        float(descriptors[0]["display_descriptor_weight"]) if descriptors else 0.0
        for descriptors in descriptor_profiles
    ]
    profile = _add_final_candidate_flags(profile)
    interpretations = [_make_interpretations(row) for _, row in profile.iterrows()]
    profile["chinese_interpretation"] = [item[0] for item in interpretations]
    profile["english_interpretation"] = [item[1] for item in interpretations]

    ordered_columns = _ordered_output_columns(profile, clusters)
    profile = profile[ordered_columns]
    profile.to_csv(
        _output_file(output_path, "song_affective_profile_all", output_suffix, "csv"),
        index=False,
        encoding="utf-8-sig",
    )
    final_candidates = _write_final_paper_candidate_table(profile, clusters, output_path, output_suffix)

    selected_profile, selected_found_count, selected_missing_count = _write_selected_outputs(
        profile,
        selected_path,
        output_path,
        output_suffix,
    )
    _make_report(profile, selected_profile, output_path, output_suffix, final_candidates)
    if make_figures:
        _write_figures(selected_profile, output_path)

    sanity = _sanity_check(
        profile,
        selected_profile=selected_profile,
        selected_found_count=selected_found_count,
        selected_missing_count=selected_missing_count,
        source_files={**cluster_sources, **tension_sources, "metadata": str(metadata_path)},
    )
    _output_file(output_path, "sanity_check", output_suffix, "json").write_text(
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
    parser.add_argument("--output_suffix", default="_v2", help="Suffix appended before output file extensions.")
    parser.add_argument("--no_figures", action="store_true", help="Skip optional selected-song helper charts.")
    args = parser.parse_args()

    sanity = run_posthoc_profile(
        run_dir=args.run_dir,
        song_metadata_csv=args.song_metadata_csv,
        selected_songs_csv=args.selected_songs_csv,
        out_dir=args.out_dir,
        make_figures=not args.no_figures,
        output_suffix=args.output_suffix,
    )
    print(json.dumps(sanity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
