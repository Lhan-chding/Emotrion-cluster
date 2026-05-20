"""Affect-aware retrieval evaluation utilities.

The functions in this module are post-hoc only: they read fixed Dataset-S
cluster, tension, and v3 interpretation outputs and never modify assignments.
External review evidence is handled only after retrieval rankings are written.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import re
import shutil
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


EPS = 1e-8
DEFAULT_OUT_DIR = Path("outputs/affect_retrieval_eval")
DEFAULT_QUERY_CONFIG = Path("configs/affect_retrieval_queries.yaml")
DEFAULT_REPORT_MIRROR = Path("reports/affect_retrieval_report.md")
SYSTEM_VA_ONLY = "va_only"
SYSTEM_VA_CLUSTER = "va_cluster"
SYSTEM_VA_TENSION = "va_tension_complexity"
SYSTEMS = [SYSTEM_VA_ONLY, SYSTEM_VA_CLUSTER, SYSTEM_VA_TENSION]
DEFAULT_TOP_K = (5, 10, 20)

REGION_NAMES = {
    0: "Subdued Melancholy",
    1: "Gentle Warmth",
    2: "Volatile Intensity",
    3: "Playful Vitality",
}

TARGET_VA_TO_CLUSTER = {
    "low_valence_low_arousal": 0,
    "mid_positive_low_arousal": 1,
    "low_valence_high_arousal": 2,
    "high_valence_high_arousal": 3,
}

COLUMN_ALIASES = {
    "song_id": ["song_id", "identifier", "track_id", "lyric_identifier", "song", "id"],
    "title": ["title", "Title", "track_title", "song_title", "name"],
    "artist": ["artist", "Artist", "performer", "artists"],
    "cluster_id": ["cluster_id", "cluster", "label", "region_id"],
    "cluster_name": ["cluster_name", "region_name", "primary_affect"],
    "balanced_valence": ["balanced_valence", "balanced_v", "c_valence"],
    "balanced_arousal": ["balanced_arousal", "balanced_a", "c_arousal"],
    "region_typicality": ["region_typicality", "typicality"],
    "region_confidence": ["region_confidence", "assigned_region_weight", "region_weight"],
    "region_margin": ["region_margin", "margin"],
    "nearest_alt_cluster": ["nearest_alt_cluster", "nearest_alternative_cluster", "alt_cluster"],
    "tension_label": ["tension_label", "tension_subtype", "subtype"],
    "tension_name": ["tension_name", "paper_tension_name", "tension_subtype_label", "subtype_name"],
    "tension_dv": ["tension_dv", "dv", "delta_v", "lyrics_minus_audio_valence"],
    "tension_da": ["tension_da", "da", "delta_a", "lyrics_minus_audio_arousal"],
    "tension_norm": ["tension_norm", "norm", "delta_norm"],
    "tension_strength_percentile": [
        "tension_strength_percentile",
        "tension_percentile",
        "strength_percentile",
        "p_tension",
    ],
    "affective_complexity_score": [
        "affective_complexity_score",
        "acs",
        "complexity_score",
    ],
    "complexity_level": ["complexity_level", "complexity_label"],
    "final_interpretation_label": ["final_interpretation_label", "interpretation_label"],
    "external_summary": ["external_summary", "external_review_summary", "summary", "review_summary"],
    "source_url": ["source_url", "external_source_url", "url", "review_url"],
    "source_name": ["source_name", "external_source_name", "publication", "source"],
    "evidence_strength": ["evidence_strength"],
    "region_agreement": ["region_agreement"],
    "tension_agreement": ["tension_agreement"],
}

FEATURE_COLUMNS = [
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
    "w_region_C0",
    "w_region_C1",
    "w_region_C2",
    "w_region_C3",
    "tension_label",
    "tension_name",
    "tension_dv",
    "tension_da",
    "tension_norm",
    "tension_strength_percentile",
    "affective_complexity_score",
    "complexity_level",
    "final_interpretation_label",
]

DEFAULT_QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "volatile_high_tension",
        "description": "low valence + high arousal + high cross-modal tension",
        "target_cluster": "C2",
        "target_va": "low_valence_high_arousal",
        "target_tension_direction": {"da": "positive"},
        "target_tension_strength": "high",
        "target_complexity": "complex_or_high",
        "external_relevance_rubric": (
            "Relevant if an external review indicates aggression, tension, rage, conflict, political anger, war, "
            "social anxiety, harshness, or confrontational energy. Strong relevance if the review also indicates "
            "semantic or lyrical intensification."
        ),
    },
    {
        "query_id": "bittersweet_lyrical_lift",
        "description": "subdued melancholy with lyric-side uplift or intensification",
        "target_cluster": "C0",
        "target_va": "low_valence_low_arousal",
        "target_tension_direction": {"dv": "positive", "da": "positive"},
        "target_tension_strength": "moderate_or_high",
        "target_complexity": "mild_or_complex",
        "external_relevance_rubric": (
            "Relevant if an external review indicates melancholy, mournful sadness, restraint, vulnerability, "
            "or introspection. Strong relevance if it also indicates emotional lift, vocal or lyrical intensity, "
            "poignancy, bittersweetness, or expressive contrast."
        ),
    },
    {
        "query_id": "gentle_warmth_lyrical_lift",
        "description": "gentle warmth with lyric-side warmth or emotional lift",
        "target_cluster": "C1",
        "target_va": "mid_positive_low_arousal",
        "target_tension_direction": {"dv": "positive", "da": "positive"},
        "target_tension_strength": "moderate_or_high",
        "external_relevance_rubric": (
            "Relevant if an external review indicates warmth, tenderness, softness, calmness, friendship, "
            "romance, healing, or intimacy. Strong relevance if it also indicates emotional intensification "
            "or lyrical/vocal uplift."
        ),
    },
    {
        "query_id": "audio_led_exuberance_dark_lyrics",
        "description": "positive energetic song with darker or softer lyric-side undercurrent",
        "target_cluster": "C3",
        "target_va": "high_valence_high_arousal",
        "target_tension_direction": {"dv": "negative", "da": "negative"},
        "target_tension_strength": "moderate_or_high",
        "target_complexity": "complex_or_high",
        "external_relevance_rubric": (
            "Relevant if an external review indicates bright, danceable, celebratory, euphoric, energetic, "
            "or party-like musical affect. Strong relevance if the review also indicates grief, loss, "
            "heartbreak, darkness, melancholy, or contrast beneath the energetic surface."
        ),
    },
    {
        "query_id": "boundary_or_ambivalent_affect",
        "description": "boundary blend or emotionally ambivalent song",
        "target_cluster": "any",
        "target_complexity": "high",
        "target_region_mixture": "boundary_blend_or_strong_secondary_affect",
        "external_relevance_rubric": (
            "Relevant if an external review indicates mixed mood, ambiguity, irony, bittersweetness, tonal "
            "contradiction, joyful sadness, dark celebration, or emotional ambivalence. Strong relevance if "
            "the review explicitly describes contrast between sound and lyrical/semantic content."
        ),
    },
    {
        "query_id": "concordant_region_prototype",
        "description": "clean affective prototype with low tension",
        "target_cluster": "any",
        "target_tension_strength": "low",
        "target_region_role": "prototype_or_representative",
        "target_complexity": "simple",
        "external_relevance_rubric": (
            "Relevant if an external review strongly matches the assigned region affect. Strong relevance if "
            "the review supports both the region affect and the absence of strong contradiction."
        ),
    },
]

QUERY_KEYWORDS = {
    "volatile_high_tension": {
        "region": [
            "aggression",
            "aggressive",
            "tension",
            "tense",
            "rage",
            "conflict",
            "political anger",
            "war",
            "social anxiety",
            "harsh",
            "confrontational",
            "angry",
        ],
        "tension": [
            "lyric",
            "lyrics",
            "semantic",
            "intensification",
            "amplification",
            "contrast",
            "conflict",
            "rage",
            "anger",
        ],
    },
    "bittersweet_lyrical_lift": {
        "region": [
            "melancholy",
            "melancholic",
            "mournful",
            "sadness",
            "sad",
            "restraint",
            "vulnerability",
            "vulnerable",
            "introspection",
            "introspective",
        ],
        "tension": [
            "lift",
            "uplift",
            "intensity",
            "intensification",
            "poignancy",
            "poignant",
            "bittersweet",
            "contrast",
            "expressive",
        ],
    },
    "gentle_warmth_lyrical_lift": {
        "region": [
            "warmth",
            "warm",
            "tenderness",
            "tender",
            "softness",
            "soft",
            "calm",
            "friendship",
            "romance",
            "romantic",
            "healing",
            "intimacy",
            "intimate",
        ],
        "tension": ["intensification", "intense", "uplift", "lift", "vocal", "lyric", "lyrical"],
    },
    "audio_led_exuberance_dark_lyrics": {
        "region": [
            "bright",
            "danceable",
            "celebratory",
            "celebration",
            "euphoric",
            "energetic",
            "energy",
            "party",
        ],
        "tension": [
            "grief",
            "loss",
            "heartbreak",
            "darkness",
            "dark",
            "melancholy",
            "contrast",
            "beneath",
            "undercurrent",
        ],
    },
    "boundary_or_ambivalent_affect": {
        "region": [
            "mixed mood",
            "ambiguity",
            "ambiguous",
            "irony",
            "bittersweet",
            "contradiction",
            "joyful sadness",
            "dark celebration",
            "ambivalence",
            "ambivalent",
        ],
        "tension": ["contrast", "contradiction", "sound and", "lyric", "lyrical", "semantic", "beneath"],
    },
    "concordant_region_prototype": {
        "region": [
            "warm",
            "melancholy",
            "aggressive",
            "energetic",
            "calm",
            "tender",
            "danceable",
            "somber",
            "bright",
            "joyful",
        ],
        "tension": ["consistent", "straightforward", "direct", "without contrast", "uncomplicated"],
    },
}

REJECTED_SOURCE_TOKENS = [
    "genius",
    "lyrics",
    "reddit",
    "youtube",
    "amazon",
    "spotify",
    "forum",
    "blogspot",
]


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {_normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(_normalize_column_name(candidate))
        if match is not None:
            return str(match)
    return None


def _read_csv(path: Path | str, **kwargs: Any) -> pd.DataFrame:
    csv_path = Path(path)
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, **kwargs)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _clip01(value: Any) -> float:
    return float(np.clip(_as_float(value), 0.0, 1.0))


def _cluster_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    match = re.search(r"C?\s*([0-3])\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number in REGION_NAMES else None


def _cluster_token(value: Any) -> str:
    number = _cluster_int(value)
    return f"C{number}" if number is not None else str(value).strip()


def _cluster_name(value: Any) -> str:
    number = _cluster_int(value)
    if number in REGION_NAMES:
        return REGION_NAMES[number]
    return str(value).strip() or "missing"


def _song_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    if "song_id" not in frame.columns:
        return frame
    return frame.drop_duplicates(subset=["song_id"], keep="first").reset_index(drop=True)


def _iter_csv_files(root: Path) -> Iterable[Path]:
    skip_prefixes = (".pytest", "..pytest")
    skip_names = {".git", "__pycache__", ".ruff_cache"}
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _err: None):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in skip_names and not dirname.startswith(skip_prefixes)
        ]
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                yield Path(dirpath) / filename


def _path_score(path: Path, preferred_names: Sequence[str]) -> int:
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]
    score = 0
    if "all" in parts:
        score += 120
    if "all" in name:
        score += 80
    if "selected" in name:
        score -= 200
    if "outputs" in parts or "output_final" in parts:
        score += 30
    if any(part.startswith(".pytest") for part in parts):
        score -= 500
    for index, preferred_name in enumerate(preferred_names):
        preferred = preferred_name.lower()
        if name == preferred:
            score += 200 - index
        elif preferred.replace(".csv", "") in name:
            score += 80 - index
    return score


def _has_columns(path: Path, column_groups: Sequence[Sequence[str]]) -> bool:
    try:
        header = _read_csv(path, nrows=0)
    except Exception:
        return False
    return all(_find_column(header, aliases) is not None for aliases in column_groups)


def _resolve_csv(
    explicit_path: Path | str | None,
    *,
    root: Path,
    preferred_names: Sequence[str],
    column_groups: Sequence[Sequence[str]],
    required: bool,
    allow_generic_search: bool = True,
) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        return path

    direct_candidates: list[Path] = []
    for name in preferred_names:
        direct_candidates.extend(
            [
                root / name,
                root / "all" / name,
                root / "output_final" / "all" / name,
                root / "_posthoc_song_affective_profile_v20_3" / name,
            ]
        )
    valid_direct = [path for path in direct_candidates if path.exists() and _has_columns(path, column_groups)]
    if valid_direct:
        return max(valid_direct, key=lambda path: (_path_score(path, preferred_names), -len(path.parts), str(path)))

    if not allow_generic_search:
        if required:
            raise FileNotFoundError(
                f"Could not find a preferred CSV under {root} matching {preferred_names} with required columns."
            )
        return None

    valid = [path for path in _iter_csv_files(root) if _has_columns(path, column_groups)]
    if valid:
        return max(valid, key=lambda path: (_path_score(path, preferred_names), -len(path.parts), str(path)))
    if required:
        raise FileNotFoundError(
            f"Could not find a CSV under {root} matching {preferred_names} with required columns."
        )
    return None


def _standardize_cluster_frame(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["song_id", "cluster_id", "balanced_valence", "balanced_arousal"])
    raw = _read_csv(path)
    song_col = _find_column(raw, COLUMN_ALIASES["song_id"])
    cluster_col = _find_column(raw, COLUMN_ALIASES["cluster_id"])
    valence_col = _find_column(raw, COLUMN_ALIASES["balanced_valence"])
    arousal_col = _find_column(raw, COLUMN_ALIASES["balanced_arousal"])
    data: dict[str, Any] = {"song_id": _song_ids(raw[song_col]) if song_col else pd.Series([], dtype=str)}
    if cluster_col:
        data["cluster_id"] = raw[cluster_col].map(_cluster_int)
    if valence_col:
        data["balanced_valence"] = _coerce_numeric(raw[valence_col])
    if arousal_col:
        data["balanced_arousal"] = _coerce_numeric(raw[arousal_col])
    return _deduplicate(pd.DataFrame(data))


def _standardize_tension_frame(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(
            columns=[
                "song_id",
                "tension_label",
                "tension_name",
                "tension_dv",
                "tension_da",
                "tension_norm",
                "tension_strength_percentile",
            ]
        )
    raw = _read_csv(path)
    song_col = _find_column(raw, COLUMN_ALIASES["song_id"])
    data: dict[str, Any] = {"song_id": _song_ids(raw[song_col]) if song_col else pd.Series([], dtype=str)}
    for standard in [
        "tension_label",
        "tension_name",
        "tension_dv",
        "tension_da",
        "tension_norm",
        "tension_strength_percentile",
    ]:
        column = _find_column(raw, COLUMN_ALIASES[standard])
        if column is None:
            continue
        if standard.startswith("tension_d") or standard in {"tension_norm", "tension_strength_percentile"}:
            data[standard] = _coerce_numeric(raw[column])
        else:
            data[standard] = raw[column].astype("string").fillna("").str.strip()
    frame = _deduplicate(pd.DataFrame(data))
    if {"tension_dv", "tension_da"}.issubset(frame.columns) and "tension_norm" not in frame.columns:
        frame["tension_norm"] = np.sqrt(np.square(frame["tension_dv"]) + np.square(frame["tension_da"]))
    return frame


def _standardize_interpretation_frame(path: Path | str) -> pd.DataFrame:
    raw = _read_csv(path)
    data: dict[str, Any] = {}
    for standard in FEATURE_COLUMNS:
        if standard.startswith("w_region_C"):
            column = _find_column(raw, [standard])
        else:
            column = _find_column(raw, COLUMN_ALIASES.get(standard, [standard]))
        if column is None:
            continue
        if standard in {
            "cluster_id",
            "balanced_valence",
            "balanced_arousal",
            "region_typicality",
            "region_confidence",
            "region_margin",
            "w_region_C0",
            "w_region_C1",
            "w_region_C2",
            "w_region_C3",
            "tension_dv",
            "tension_da",
            "tension_norm",
            "tension_strength_percentile",
            "affective_complexity_score",
        }:
            if standard == "cluster_id":
                data[standard] = raw[column].map(_cluster_int)
            else:
                data[standard] = _coerce_numeric(raw[column])
        elif standard == "song_id":
            data[standard] = _song_ids(raw[column])
        else:
            data[standard] = raw[column].astype("string").fillna("").str.strip()
    if "song_id" not in data:
        raise ValueError(f"{path} must include a song_id/identifier column.")
    return _deduplicate(pd.DataFrame(data))


def _resolve_interpretation_csv(
    explicit_path: Path | str | None,
    *,
    root: Path,
    out_dir: Path,
) -> Path:
    found = _resolve_csv(
        explicit_path,
        root=root,
        preferred_names=["song_affective_interpretation_all_v3.csv"],
        column_groups=[COLUMN_ALIASES["song_id"], COLUMN_ALIASES["affective_complexity_score"]],
        required=False,
    )
    if found is not None:
        return found

    profile_all = root / "_posthoc_song_affective_profile_v20_3" / "song_affective_profile_all_v2.csv"
    profile_selected = root / "_posthoc_song_affective_profile_v20_3" / "song_affective_profile_selected_v2.csv"
    descriptor_json = root / "_posthoc_song_affective_profile_v20_3" / "descriptor_weights_selected_v2.json"
    if not (profile_all.exists() and profile_selected.exists() and descriptor_json.exists()):
        raise FileNotFoundError(
            "Could not find song_affective_interpretation_all_v3.csv and could not bootstrap it from "
            "_posthoc_song_affective_profile_v20_3 v2 files."
        )
    from scripts.build_song_affective_interpretation_v3 import run_interpretation_v3

    generated_dir = out_dir / "_generated_interpretation_v3"
    run_interpretation_v3(
        profile_all_csv=profile_all,
        profile_selected_csv=profile_selected,
        descriptor_json=descriptor_json,
        out_dir=generated_dir,
    )
    return generated_dir / "song_affective_interpretation_all_v3.csv"


def _merge_fill(base: pd.DataFrame, addition: pd.DataFrame, suffix: str) -> tuple[pd.DataFrame, dict[str, int]]:
    if addition.empty:
        return base, {}
    merged = base.merge(addition, on="song_id", how="outer", suffixes=("", suffix))
    conflict_counts: dict[str, int] = {}
    for column in [col for col in addition.columns if col != "song_id"]:
        other = f"{column}{suffix}"
        if other not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[other]
        else:
            left = merged[column]
            right = merged[other]
            comparable = left.notna() & right.notna()
            if comparable.any():
                if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
                    conflict = comparable & (pd.to_numeric(left, errors="coerce").round(8) != pd.to_numeric(right, errors="coerce").round(8))
                else:
                    conflict = comparable & (left.astype(str) != right.astype(str))
                if conflict.any():
                    conflict_counts[column] = int(conflict.sum())
            merged[column] = merged[column].where(merged[column].notna(), merged[other])
        merged = merged.drop(columns=[other])
    return merged, conflict_counts


def _fill_derived_feature_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    result["song_id"] = _song_ids(result["song_id"])
    result["title"] = result["title"].astype("string").fillna("").str.strip()
    result["artist"] = result["artist"].astype("string").fillna("").str.strip()
    result["cluster_id"] = result["cluster_id"].map(_cluster_int)
    result["cluster_name"] = [
        current if str(current).strip() and str(current).strip().lower() != "nan" else _cluster_name(cluster_id)
        for current, cluster_id in zip(result["cluster_name"], result["cluster_id"])
    ]

    for numeric in [
        "balanced_valence",
        "balanced_arousal",
        "region_typicality",
        "region_confidence",
        "region_margin",
        "w_region_C0",
        "w_region_C1",
        "w_region_C2",
        "w_region_C3",
        "tension_dv",
        "tension_da",
        "tension_norm",
        "tension_strength_percentile",
        "affective_complexity_score",
    ]:
        result[numeric] = pd.to_numeric(result[numeric], errors="coerce")

    missing_norm = result["tension_norm"].isna() & result["tension_dv"].notna() & result["tension_da"].notna()
    result.loc[missing_norm, "tension_norm"] = np.sqrt(
        np.square(result.loc[missing_norm, "tension_dv"]) + np.square(result.loc[missing_norm, "tension_da"])
    )

    for row_index, row in result.iterrows():
        cluster_id = _cluster_int(row.get("cluster_id"))
        assigned_col = f"w_region_C{cluster_id}" if cluster_id is not None else ""
        assigned_weight = _as_float(row.get(assigned_col), default=np.nan)
        weight_values = {
            idx: _as_float(row.get(f"w_region_C{idx}"), default=np.nan)
            for idx in REGION_NAMES
        }
        finite_weights = {idx: val for idx, val in weight_values.items() if math.isfinite(val)}
        if pd.isna(row.get("region_confidence")) and assigned_col:
            result.at[row_index, "region_confidence"] = assigned_weight
        if pd.isna(row.get("nearest_alt_cluster")) or not str(row.get("nearest_alt_cluster")).strip():
            alternatives = {idx: val for idx, val in finite_weights.items() if idx != cluster_id}
            if alternatives:
                alt_id = max(alternatives, key=alternatives.get)
                result.at[row_index, "nearest_alt_cluster"] = f"C{alt_id} {REGION_NAMES[alt_id]}"
        if pd.isna(row.get("region_margin")) and cluster_id is not None and finite_weights:
            alternatives = [val for idx, val in finite_weights.items() if idx != cluster_id]
            if alternatives and math.isfinite(assigned_weight):
                result.at[row_index, "region_margin"] = assigned_weight - max(alternatives)

    if result["tension_strength_percentile"].isna().any() and result["tension_norm"].notna().any():
        result["tension_strength_percentile"] = _fill_tension_percentiles(result)
    result["tension_strength_percentile"] = result["tension_strength_percentile"].clip(0.0, 1.0)
    result["affective_complexity_score"] = result["affective_complexity_score"].clip(0.0, 1.0)
    result["complexity_level"] = result["complexity_level"].astype("string").fillna("missing").str.strip()
    result["final_interpretation_label"] = (
        result["final_interpretation_label"].astype("string").fillna("missing").str.strip()
    )
    result["tension_label"] = result["tension_label"].astype("string").fillna("missing").str.strip()
    result["tension_name"] = result["tension_name"].astype("string").fillna(result["tension_label"]).str.strip()
    return result[FEATURE_COLUMNS].sort_values("song_id").reset_index(drop=True)


def _fill_tension_percentiles(frame: pd.DataFrame) -> pd.Series:
    result = pd.to_numeric(frame.get("tension_strength_percentile"), errors="coerce")
    filled = result.copy()
    cluster_series = frame.get("cluster_id", pd.Series([None] * len(frame)))
    for _cluster, group_index in frame.groupby(cluster_series, dropna=False).groups.items():
        index = list(group_index)
        norms = pd.to_numeric(frame.loc[index, "tension_norm"], errors="coerce")
        valid_norms = norms.dropna().to_numpy(dtype=float)
        if valid_norms.size == 0:
            continue
        sorted_values = np.sort(valid_norms)
        for row_index in index:
            if pd.notna(filled.loc[row_index]):
                continue
            value = frame.loc[row_index, "tension_norm"]
            if pd.isna(value):
                continue
            rank = np.searchsorted(sorted_values, float(value), side="right")
            filled.loc[row_index] = float(np.clip(rank / len(sorted_values), 0.0, 1.0))
    return filled


def build_song_retrieval_features(
    *,
    cluster_csv: Path | str | None,
    tension_csv: Path | str | None,
    interpretation_csv: Path | str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    interpretation = _standardize_interpretation_frame(interpretation_csv)
    cluster = _standardize_cluster_frame(cluster_csv)
    tension = _standardize_tension_frame(tension_csv)
    merged = interpretation.copy()
    merged, cluster_conflicts = _merge_fill(merged, cluster, "_cluster")
    merged, tension_conflicts = _merge_fill(merged, tension, "_tension")
    features = _fill_derived_feature_fields(merged)
    missing_counts = {
        column: int(features[column].isna().sum() + features[column].astype(str).isin(["", "missing", "<NA>"]).sum())
        for column in FEATURE_COLUMNS
        if column != "song_id"
    }
    sanity = {
        "total_songs": int(len(features)),
        "missing_feature_counts": missing_counts,
        "cluster_conflict_counts": cluster_conflicts,
        "tension_conflict_counts": tension_conflicts,
        "input_paths": {
            "cluster_csv": str(cluster_csv) if cluster_csv else "",
            "tension_csv": str(tension_csv) if tension_csv else "",
            "interpretation_csv": str(interpretation_csv),
        },
    }
    return features, sanity


def _yaml_quote(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _query_yaml(queries: Sequence[Mapping[str, Any]]) -> str:
    lines = ["queries:"]
    for query in queries:
        lines.append(f"  - query_id: {_yaml_quote(query['query_id'])}")
        for key, value in query.items():
            if key == "query_id":
                continue
            if isinstance(value, Mapping):
                lines.append(f"    {key}:")
                for nested_key, nested_value in value.items():
                    lines.append(f"      {nested_key}: {_yaml_quote(nested_value)}")
            else:
                lines.append(f"    {key}: {_yaml_quote(value)}")
    return "\n".join(lines) + "\n"


def write_default_query_config(path: Path | str = DEFAULT_QUERY_CONFIG) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_query_yaml(DEFAULT_QUERIES), encoding="utf-8")


def load_query_config(path: Path | str | None = None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(query) for query in DEFAULT_QUERIES]
    config_path = Path(path)
    if not config_path.exists():
        if config_path == DEFAULT_QUERY_CONFIG:
            write_default_query_config(config_path)
            return [dict(query) for query in DEFAULT_QUERIES]
        raise FileNotFoundError(f"Query config does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        queries = payload.get("queries", payload) if isinstance(payload, Mapping) else payload
        if isinstance(queries, list):
            return [dict(query) for query in queries]
    except Exception:
        pass
    if config_path.name == DEFAULT_QUERY_CONFIG.name:
        return [dict(query) for query in DEFAULT_QUERIES]
    raise ValueError(
        f"Could not parse {config_path}. Install PyYAML or use the default query config format."
    )


def _cluster_from_target(target: Any) -> Optional[int]:
    if target is None:
        return None
    text = str(target).strip().lower()
    if text in {"", "any", "none"}:
        return None
    return _cluster_int(text)


def _cluster_centers(features: pd.DataFrame) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    usable = features.dropna(subset=["cluster_id", "balanced_valence", "balanced_arousal"])
    for cluster_id, group in usable.groupby("cluster_id"):
        parsed = _cluster_int(cluster_id)
        if parsed is None:
            continue
        centers[parsed] = group[["balanced_valence", "balanced_arousal"]].to_numpy(dtype=float).mean(axis=0)
    fallback = {
        0: np.array([0.25, 0.25], dtype=float),
        1: np.array([0.65, 0.30], dtype=float),
        2: np.array([0.25, 0.80], dtype=float),
        3: np.array([0.80, 0.75], dtype=float),
    }
    for cluster_id, coordinate in fallback.items():
        centers.setdefault(cluster_id, coordinate)
    return centers


def _target_coordinate(query: Mapping[str, Any], centers: Mapping[int, np.ndarray]) -> Optional[np.ndarray]:
    target_cluster = _cluster_from_target(query.get("target_cluster"))
    if target_cluster is not None:
        return np.asarray(centers[target_cluster], dtype=float)
    target_va = str(query.get("target_va", "")).strip()
    if target_va in TARGET_VA_TO_CLUSTER:
        return np.asarray(centers[TARGET_VA_TO_CLUSTER[target_va]], dtype=float)
    return None


def _va_similarity(features: pd.DataFrame, query: Mapping[str, Any], centers: Mapping[int, np.ndarray]) -> np.ndarray:
    coordinate = _target_coordinate(query, centers)
    if coordinate is None:
        return np.ones(len(features), dtype=float)
    values = features[["balanced_valence", "balanced_arousal"]].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    scales = np.nanstd(values, axis=0)
    scales = np.where(scales > EPS, scales, 1.0)
    distances = np.full(len(features), 10.0, dtype=float)
    distances[finite] = np.linalg.norm((values[finite] - coordinate) / scales, axis=1)
    return np.exp(-0.5 * np.square(distances))


def _assigned_weight(row: pd.Series) -> float:
    cluster_id = _cluster_int(row.get("cluster_id"))
    if cluster_id is None:
        return _clip01(row.get("region_confidence"))
    return _clip01(row.get(f"w_region_C{cluster_id}", row.get("region_confidence")))


def _nearest_alt_weight(row: pd.Series) -> float:
    cluster_id = _cluster_int(row.get("cluster_id"))
    weights = [
        _clip01(row.get(f"w_region_C{idx}"))
        for idx in REGION_NAMES
        if idx != cluster_id
    ]
    return max(weights) if weights else 0.0


def _blend_ratio(row: pd.Series) -> float:
    return float(np.clip(_nearest_alt_weight(row) / max(_assigned_weight(row), EPS), 0.0, 1.5) / 1.5)


def _region_match(features: pd.DataFrame, query: Mapping[str, Any]) -> np.ndarray:
    target_cluster = _cluster_from_target(query.get("target_cluster"))
    if target_cluster is None:
        return features["region_confidence"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    column = f"w_region_C{target_cluster}"
    if column in features.columns:
        return features[column].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    return features["cluster_id"].map(lambda value: 1.0 if _cluster_int(value) == target_cluster else 0.0).to_numpy(dtype=float)


def _boundary_score(features: pd.DataFrame) -> np.ndarray:
    scores = []
    for _, row in features.iterrows():
        margin = _as_float(row.get("region_margin"), default=1.0)
        margin_score = 1.0 if margin <= 0 else float(np.clip(1.0 - margin, 0.0, 1.0))
        scores.append(max(margin_score, _blend_ratio(row), 1.0 - _clip01(row.get("region_typicality"))))
    return np.asarray(scores, dtype=float)


def _prototype_score(features: pd.DataFrame) -> np.ndarray:
    margin = pd.to_numeric(features["region_margin"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    margin_score = np.clip((margin + 0.25) / 1.25, 0.0, 1.0)
    typicality = features["region_typicality"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    return 0.65 * typicality + 0.35 * margin_score


def _geometry_role_match(features: pd.DataFrame, query: Mapping[str, Any]) -> np.ndarray:
    if query.get("target_region_role") == "prototype_or_representative":
        return _prototype_score(features)
    if query.get("target_region_mixture") == "boundary_blend_or_strong_secondary_affect":
        return _boundary_score(features)
    if query.get("target_tension_strength") in {"high", "moderate_or_high"}:
        tension = features["tension_strength_percentile"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        typicality = features["region_typicality"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        return np.maximum(tension, 1.0 - typicality)
    return _prototype_score(features)


def _cluster_profile_score(features: pd.DataFrame, query: Mapping[str, Any], score_va: np.ndarray) -> np.ndarray:
    region = _region_match(features, query)
    typicality = features["region_typicality"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    geometry = _geometry_role_match(features, query)
    return 0.50 * score_va + 0.30 * region + 0.10 * typicality + 0.10 * geometry


def _direction_query_vector(direction: Mapping[str, Any] | None) -> np.ndarray:
    if not direction:
        return np.array([0.0, 0.0], dtype=float)
    mapping = {"positive": 1.0, "negative": -1.0, "+": 1.0, "-": -1.0}
    return np.array(
        [
            mapping.get(str(direction.get("dv", "")).strip().lower(), 0.0),
            mapping.get(str(direction.get("da", "")).strip().lower(), 0.0),
        ],
        dtype=float,
    )


def _tension_match(features: pd.DataFrame, query: Mapping[str, Any]) -> np.ndarray:
    q_vec = _direction_query_vector(query.get("target_tension_direction"))
    q_norm = float(np.linalg.norm(q_vec))
    song_vec = features[["tension_dv", "tension_da"]].fillna(0.0).to_numpy(dtype=float)
    song_norm = np.linalg.norm(song_vec, axis=1)
    if q_norm <= EPS:
        direction_score = np.ones(len(features), dtype=float)
    else:
        direction_score = np.full(len(features), 0.5, dtype=float)
        valid = song_norm > EPS
        cosine = (song_vec[valid] @ q_vec) / (song_norm[valid] * q_norm)
        direction_score[valid] = (np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0

    strength = features["tension_strength_percentile"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    target_strength = query.get("target_tension_strength")
    if target_strength == "high":
        strength_score = strength
    elif target_strength == "moderate_or_high":
        strength_score = np.minimum(1.0, strength / 0.60)
    elif target_strength == "low":
        strength_score = 1.0 - strength
    else:
        strength_score = np.full(len(features), 0.5, dtype=float)

    if q_norm <= EPS and target_strength is not None:
        return strength_score
    if q_norm > EPS and target_strength is None:
        return direction_score
    if q_norm <= EPS and target_strength is None:
        return np.full(len(features), 0.5, dtype=float)
    return 0.65 * direction_score + 0.35 * strength_score


def _complexity_match(features: pd.DataFrame, query: Mapping[str, Any]) -> np.ndarray:
    acs = features["affective_complexity_score"].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    target = query.get("target_complexity")
    if target == "simple":
        return 1.0 - acs
    if target == "mild_or_complex":
        return np.where((acs >= 0.45) & (acs <= 0.65), 1.0, np.clip(1.0 - np.abs(acs - 0.55) / 0.55, 0.0, 1.0))
    if target == "complex_or_high":
        return acs
    if target == "high":
        high = features["complexity_level"].astype(str).str.lower().eq("highly complex / ambivalent").to_numpy()
        return np.where(high, 1.0, np.clip(acs / 0.70, 0.0, 1.0))
    return np.full(len(features), 0.5, dtype=float)


def _full_score(features: pd.DataFrame, query: Mapping[str, Any], score_va: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    region = _region_match(features, query)
    tension = _tension_match(features, query)
    complexity = _complexity_match(features, query)
    geometry = _geometry_role_match(features, query)
    score = 0.30 * score_va + 0.20 * region + 0.25 * tension + 0.15 * complexity + 0.10 * geometry
    return score, {
        "score_va": score_va,
        "region_match": region,
        "tension_match": tension,
        "complexity_match": complexity,
        "geometry_role_match": geometry,
    }


def _retrieval_reason(row: pd.Series, system: str, query: Mapping[str, Any], component_values: Mapping[str, float]) -> str:
    parts = [
        f"{row.get('cluster_name', 'missing')} at balanced VA=({_as_float(row.get('balanced_valence')):.3f}, {_as_float(row.get('balanced_arousal')):.3f})"
    ]
    if system == SYSTEM_VA_ONLY:
        parts.append(f"VA similarity={component_values.get('score_va', 0.0):.3f}")
    elif system == SYSTEM_VA_CLUSTER:
        parts.append(f"region match={component_values.get('region_match', 0.0):.3f}")
        parts.append(f"geometry match={component_values.get('geometry_role_match', 0.0):.3f}")
    else:
        tension_percentile = _clip01(row.get("tension_strength_percentile"))
        strength_text = "high cross-modal tension" if tension_percentile >= 0.80 else "calibrated audio-lyric tension"
        parts.append(strength_text)
        parts.append(f"tension match={component_values.get('tension_match', 0.0):.3f}")
        parts.append(f"ACS={_clip01(row.get('affective_complexity_score')):.3f}")
    return "; ".join(parts)


def compute_retrieval_rankings(
    features: pd.DataFrame,
    queries: Sequence[Mapping[str, Any]],
    *,
    retrieval_depth: int = 20,
) -> pd.DataFrame:
    centers = _cluster_centers(features)
    rows: list[dict[str, Any]] = []
    for query in queries:
        score_va = _va_similarity(features, query, centers)
        cluster_score = _cluster_profile_score(features, query, score_va)
        full_score, full_components = _full_score(features, query, score_va)
        score_payloads = {
            SYSTEM_VA_ONLY: (score_va, {"score_va": score_va}),
            SYSTEM_VA_CLUSTER: (
                cluster_score,
                {
                    "score_va": score_va,
                    "region_match": _region_match(features, query),
                    "geometry_role_match": _geometry_role_match(features, query),
                },
            ),
            SYSTEM_VA_TENSION: (full_score, full_components),
        }
        for system, (scores, components) in score_payloads.items():
            order = np.lexsort((features["song_id"].astype(str).to_numpy(), -scores))[:retrieval_depth]
            for rank, row_index in enumerate(order, start=1):
                feature_row = features.iloc[int(row_index)]
                component_values = {
                    key: float(values[int(row_index)])
                    for key, values in components.items()
                }
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "query_description": query.get("description", ""),
                        "system": system,
                        "rank": rank,
                        "song_id": feature_row["song_id"],
                        "title": feature_row.get("title", ""),
                        "artist": feature_row.get("artist", ""),
                        "score": float(scores[int(row_index)]),
                        "score_va": component_values.get("score_va", np.nan),
                        "region_match": component_values.get("region_match", np.nan),
                        "tension_match": component_values.get("tension_match", np.nan),
                        "complexity_match": component_values.get("complexity_match", np.nan),
                        "geometry_role_match": component_values.get("geometry_role_match", np.nan),
                        "cluster_id": feature_row.get("cluster_id", np.nan),
                        "cluster_name": feature_row.get("cluster_name", ""),
                        "balanced_valence": feature_row.get("balanced_valence", np.nan),
                        "balanced_arousal": feature_row.get("balanced_arousal", np.nan),
                        "tension_dv": feature_row.get("tension_dv", np.nan),
                        "tension_da": feature_row.get("tension_da", np.nan),
                        "tension_strength_percentile": feature_row.get("tension_strength_percentile", np.nan),
                        "affective_complexity_score": feature_row.get("affective_complexity_score", np.nan),
                        "final_interpretation_label": feature_row.get("final_interpretation_label", ""),
                        "model_retrieval_reason": _retrieval_reason(feature_row, system, query, component_values),
                    }
                )
    return pd.DataFrame(rows)


def write_retrieval_result_files(rankings: pd.DataFrame, out_dir: Path) -> None:
    rankings.to_csv(out_dir / "retrieval_results_all.csv", index=False, encoding="utf-8-sig")
    for (query_id, system), group in rankings.groupby(["query_id", "system"], sort=False):
        path = out_dir / f"retrieval_results_{query_id}_{system}.csv"
        group.sort_values("rank").to_csv(path, index=False, encoding="utf-8-sig")


def build_annotation_pool(
    rankings: pd.DataFrame,
    features: pd.DataFrame,
    *,
    retrieval_depth: int = 20,
    external_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    top = rankings[rankings["rank"] <= retrieval_depth].copy()
    feature_lookup = features.set_index("song_id", drop=False)
    rows: list[dict[str, Any]] = []
    evidence_lookup = _evidence_lookup(external_evidence)
    for (query_id, song_id), group in top.groupby(["query_id", "song_id"], sort=True):
        feature_row = feature_lookup.loc[song_id] if song_id in feature_lookup.index else pd.Series(dtype=object)
        if isinstance(feature_row, pd.DataFrame):
            feature_row = feature_row.iloc[0]
        ranks = {system: "" for system in SYSTEMS}
        for _, row in group.iterrows():
            ranks[row["system"]] = int(row["rank"])
        system_sources = sorted(group["system"].unique().tolist())
        best_reason = group.sort_values(["system", "rank"]).iloc[0].get("model_retrieval_reason", "")
        evidence = evidence_lookup.get((str(query_id), str(song_id))) or evidence_lookup.get(("", str(song_id)), {})
        status = "needs_external_review"
        if evidence.get("external_summary") and evidence.get("source_url"):
            status = "pending_grading"
        rows.append(
            {
                "query_id": query_id,
                "song_id": song_id,
                "title": feature_row.get("title", group.iloc[0].get("title", "")),
                "artist": feature_row.get("artist", group.iloc[0].get("artist", "")),
                "system_sources": ";".join(system_sources),
                "rank_va_only": ranks[SYSTEM_VA_ONLY],
                "rank_va_cluster": ranks[SYSTEM_VA_CLUSTER],
                "rank_va_tension": ranks[SYSTEM_VA_TENSION],
                "cluster_name": feature_row.get("cluster_name", group.iloc[0].get("cluster_name", "")),
                "final_interpretation_label": feature_row.get("final_interpretation_label", ""),
                "balanced_valence": feature_row.get("balanced_valence", np.nan),
                "balanced_arousal": feature_row.get("balanced_arousal", np.nan),
                "tension_dv": feature_row.get("tension_dv", np.nan),
                "tension_da": feature_row.get("tension_da", np.nan),
                "tension_strength_percentile": feature_row.get("tension_strength_percentile", np.nan),
                "affective_complexity_score": feature_row.get("affective_complexity_score", np.nan),
                "model_retrieval_reason": best_reason,
                "external_source_url": evidence.get("source_url", ""),
                "external_source_name": evidence.get("source_name", ""),
                "external_review_summary": evidence.get("external_summary", ""),
                "external_keywords": "",
                "region_relevance_grade": "",
                "tension_relevance_grade": "",
                "overall_relevance_grade": "",
                "annotation_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["query_id", "song_id"]).reset_index(drop=True)


def build_source_search_queries(annotation_pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in annotation_pool.iterrows():
        artist = str(row.get("artist", "")).strip()
        title = str(row.get("title", "")).strip()
        query_base = f"{artist} {title}".strip()
        for search_query in [
            f"{query_base} review mood emotion",
            f"{query_base} Pitchfork review",
            f"{query_base} music review melancholy energetic tension",
        ]:
            rows.append(
                {
                    "query_id": row.get("query_id", ""),
                    "song_id": row.get("song_id", ""),
                    "title": title,
                    "artist": artist,
                    "search_query": search_query,
                }
            )
    return pd.DataFrame(rows)


def _evidence_lookup(external_evidence: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, str]]:
    if external_evidence is None or external_evidence.empty:
        return {}
    raw = external_evidence.copy()
    standardized: dict[str, pd.Series] = {}
    for standard in ["song_id", "external_summary", "source_url", "source_name", "evidence_strength", "region_agreement", "tension_agreement"]:
        column = _find_column(raw, COLUMN_ALIASES[standard])
        if column is not None:
            standardized[standard] = raw[column]
    query_col = _find_column(raw, ["query_id"])
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    if "song_id" not in standardized:
        return lookup
    for index, row in raw.iterrows():
        song_id = str(standardized["song_id"].iloc[index]).strip()
        query_id = str(raw[query_col].iloc[index]).strip() if query_col else ""
        lookup[(query_id, song_id)] = {
            "external_summary": str(standardized.get("external_summary", pd.Series([""] * len(raw))).iloc[index]).strip(),
            "source_url": str(standardized.get("source_url", pd.Series([""] * len(raw))).iloc[index]).strip(),
            "source_name": str(standardized.get("source_name", pd.Series([""] * len(raw))).iloc[index]).strip(),
            "evidence_strength": str(standardized.get("evidence_strength", pd.Series([""] * len(raw))).iloc[index]).strip(),
            "region_agreement": str(standardized.get("region_agreement", pd.Series([""] * len(raw))).iloc[index]).strip(),
            "tension_agreement": str(standardized.get("tension_agreement", pd.Series([""] * len(raw))).iloc[index]).strip(),
        }
    return lookup


def _source_allowed(source_name: Any, source_url: Any) -> bool:
    source = f"{source_name} {source_url}".lower()
    return not any(token in source for token in REJECTED_SOURCE_TOKENS)


def _keyword_grade(text: str, keywords: Sequence[str]) -> tuple[int, list[str]]:
    lower = text.lower()
    hits = [keyword for keyword in keywords if keyword in lower]
    unique_hits = list(dict.fromkeys(hits))
    if not unique_hits:
        return 0, []
    if len(unique_hits) == 1:
        return 1, unique_hits
    if len(unique_hits) == 2:
        return 2, unique_hits
    return 3, unique_hits


def _manual_agreement_grade(value: Any) -> Optional[int]:
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return None
    if text in {"3", "strong", "strong_support", "yes_strong"}:
        return 3
    if text in {"2", "support", "supported", "yes"}:
        return 2
    if text in {"1", "weak", "partial", "hint"}:
        return 1
    if text in {"0", "no", "contradiction", "contradicts"}:
        return 0
    try:
        return int(np.clip(int(float(text)), 0, 3))
    except ValueError:
        return None


def build_external_review_labels(
    annotation_pool: pd.DataFrame,
    external_evidence: pd.DataFrame | None = None,
    queries: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    evidence_lookup = _evidence_lookup(external_evidence)
    rows: list[dict[str, Any]] = []
    for _, row in annotation_pool.iterrows():
        query_id = str(row.get("query_id", "")).strip()
        song_id = str(row.get("song_id", "")).strip()
        evidence = evidence_lookup.get((query_id, song_id)) or evidence_lookup.get(("", song_id), {})
        external_summary = evidence.get("external_summary") or str(row.get("external_review_summary", "")).strip()
        source_url = evidence.get("source_url") or str(row.get("external_source_url", "")).strip()
        source_name = evidence.get("source_name") or str(row.get("external_source_name", "")).strip()
        if not external_summary or not source_url or not _source_allowed(source_name, source_url):
            rows.append(
                {
                    **row.to_dict(),
                    "external_source_url": source_url,
                    "external_source_name": source_name,
                    "external_review_summary": external_summary,
                    "external_keywords": "",
                    "region_relevance_grade": pd.NA,
                    "tension_relevance_grade": pd.NA,
                    "overall_relevance_grade": pd.NA,
                    "annotation_status": "unverified",
                }
            )
            continue

        keywords = QUERY_KEYWORDS.get(query_id, {"region": [], "tension": []})
        region_grade, region_hits = _keyword_grade(external_summary, keywords["region"])
        tension_grade, tension_hits = _keyword_grade(external_summary, keywords["tension"])
        manual_region = _manual_agreement_grade(evidence.get("region_agreement", ""))
        manual_tension = _manual_agreement_grade(evidence.get("tension_agreement", ""))
        if manual_region is not None:
            region_grade = manual_region
        if manual_tension is not None:
            tension_grade = manual_tension

        if query_id == "concordant_region_prototype":
            overall = int(region_grade)
        else:
            overall = int(round(0.45 * region_grade + 0.55 * tension_grade))
        contradiction = "contradict" in external_summary.lower() or "opposite" in external_summary.lower()
        status = "contradiction" if contradiction and overall == 0 else "verified"
        if overall <= 1 and (region_grade > 0 or tension_grade > 0):
            status = "partial"

        rows.append(
            {
                **row.to_dict(),
                "external_source_url": source_url,
                "external_source_name": source_name,
                "external_review_summary": external_summary,
                "external_keywords": ";".join(region_hits + tension_hits),
                "region_relevance_grade": int(region_grade),
                "tension_relevance_grade": int(tension_grade),
                "overall_relevance_grade": int(overall),
                "annotation_status": status,
            }
        )
    return pd.DataFrame(rows)


def _metric_columns() -> list[str]:
    return [
        "query_id",
        "system",
        "k",
        "ndcg",
        "mean_relevance",
        "precision_relevance_ge_2",
        "strong_support_rate",
        "judged_count",
        "metrics_available",
    ]


def _dcg(relevances: Sequence[float]) -> float:
    total = 0.0
    for index, rel in enumerate(relevances, start=1):
        total += (2.0 ** rel - 1.0) / math.log2(index + 1)
    return float(total)


def _usable_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty or "overall_relevance_grade" not in labels.columns:
        return pd.DataFrame(columns=["query_id", "song_id", "overall_relevance_grade"])
    usable = labels.copy()
    usable["overall_relevance_grade"] = pd.to_numeric(usable["overall_relevance_grade"], errors="coerce")
    usable = usable[usable["overall_relevance_grade"].notna()].copy()
    if "annotation_status" in usable.columns:
        usable = usable[~usable["annotation_status"].isin(["unverified", "needs_external_review"])].copy()
    return usable


def evaluate_retrieval_results(
    retrieval_results: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = _usable_labels(labels)
    if usable.empty:
        empty = pd.DataFrame(columns=_metric_columns())
        summary = pd.DataFrame(
            columns=[
                "system",
                "mean_ndcg",
                "mean_relevance",
                "mean_precision_relevance_ge_2",
                "mean_strong_support_rate",
                "pairwise_win_rate_vs_va_only",
            ]
        )
        return empty, summary

    label_cols = ["query_id", "song_id", "overall_relevance_grade"]
    label_lookup = usable[label_cols].drop_duplicates(subset=["query_id", "song_id"], keep="first")
    rows: list[dict[str, Any]] = []
    query_ideal = {
        query_id: group["overall_relevance_grade"].sort_values(ascending=False).to_numpy(dtype=float)
        for query_id, group in label_lookup.groupby("query_id")
    }
    for (query_id, system), group in retrieval_results.groupby(["query_id", "system"], sort=True):
        ranked = group.sort_values("rank")
        for k in top_k:
            top = ranked.head(k).merge(label_lookup, on=["query_id", "song_id"], how="left")
            judged_count = int(top["overall_relevance_grade"].notna().sum())
            rels = top["overall_relevance_grade"].fillna(0.0).to_numpy(dtype=float)
            ideal_rels = query_ideal.get(query_id, np.array([], dtype=float))[:k]
            ideal = _dcg(ideal_rels)
            ndcg = _dcg(rels) / ideal if ideal > EPS else np.nan
            rows.append(
                {
                    "query_id": query_id,
                    "system": system,
                    "k": int(k),
                    "ndcg": ndcg,
                    "mean_relevance": float(np.mean(rels)) if len(rels) else np.nan,
                    "precision_relevance_ge_2": float(np.sum(rels >= 2) / k),
                    "strong_support_rate": float(np.sum(rels == 3) / k),
                    "judged_count": judged_count,
                    "metrics_available": ideal > EPS,
                }
            )
    metrics = pd.DataFrame(rows, columns=_metric_columns())
    summary_rows = []
    compare_k = 10 if 10 in set(top_k) else max(top_k)
    baseline = metrics[(metrics["system"] == SYSTEM_VA_ONLY) & (metrics["k"] == compare_k)]
    baseline_by_query = baseline.set_index("query_id")["mean_relevance"].to_dict()
    for system, group in metrics.groupby("system", sort=True):
        wins = []
        for _, metric_row in group[group["k"] == compare_k].iterrows():
            base = baseline_by_query.get(metric_row["query_id"])
            if base is None or system == SYSTEM_VA_ONLY:
                continue
            wins.append(float(metric_row["mean_relevance"] > base))
        summary_rows.append(
            {
                "system": system,
                "mean_ndcg": group["ndcg"].mean(),
                "mean_relevance": group["mean_relevance"].mean(),
                "mean_precision_relevance_ge_2": group["precision_relevance_ge_2"].mean(),
                "mean_strong_support_rate": group["strong_support_rate"].mean(),
                "pairwise_win_rate_vs_va_only": float(np.mean(wins)) if wins else (np.nan if system != SYSTEM_VA_ONLY else np.nan),
            }
        )
    return metrics, pd.DataFrame(summary_rows)


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _write_metrics_table_tex(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        path.write_text(
            "\\begin{tabular}{lllrr}\nQuery & System & K & NDCG & Mean rel. \\\\\n\\end{tabular}\n",
            encoding="utf-8",
        )
        return
    rows = [
        "\\begin{tabular}{lllrrrr}",
        "Query & System & K & NDCG & Mean rel. & P@K & Strong@K \\\\",
        "\\hline",
    ]
    for _, row in metrics.sort_values(["query_id", "system", "k"]).iterrows():
        rows.append(
            f"{_latex_escape(row['query_id'])} & {_latex_escape(row['system'])} & {int(row['k'])} & "
            f"{_as_float(row['ndcg'], float('nan')):.3f} & {_as_float(row['mean_relevance'], float('nan')):.3f} & "
            f"{_as_float(row['precision_relevance_ge_2'], float('nan')):.3f} & "
            f"{_as_float(row['strong_support_rate'], float('nan')):.3f} \\\\"
        )
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_examples_table_tex(annotation_pool: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "\\begin{tabular}{llllr}",
        "Query & Song & Artist & Systems & ACS \\\\",
        "\\hline",
    ]
    for _, row in annotation_pool.head(18).iterrows():
        rows.append(
            f"{_latex_escape(row.get('query_id', ''))} & {_latex_escape(row.get('title', ''))} & "
            f"{_latex_escape(row.get('artist', ''))} & {_latex_escape(row.get('system_sources', ''))} & "
            f"{_as_float(row.get('affective_complexity_score'), float('nan')):.2f} \\\\"
        )
    rows.append("\\end{tabular}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_placeholder_pdf(path: Path, title: str, body: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.02, 0.80, title, fontsize=14, weight="bold", va="top")
    ax.text(0.02, 0.58, body, fontsize=10, va="top", wrap=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_figures(metrics: pd.DataFrame, rankings: pd.DataFrame, labels: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ndcg_path = figure_dir / "retrieval_ndcg_comparison.pdf"
    if metrics.empty or not metrics["metrics_available"].any():
        _write_placeholder_pdf(ndcg_path, "NDCG comparison", "No verified external relevance labels are available yet.")
    else:
        plot_data = metrics[metrics["k"].isin([5, 10])].copy()
        pivot = plot_data.pivot_table(index="query_id", columns=["system", "k"], values="ndcg", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("NDCG")
        ax.set_xlabel("Query")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(ndcg_path)
        plt.close(fig)

    for query_id, group in rankings.groupby("query_id", sort=True):
        query_path = figure_dir / f"query_examples_{query_id}.pdf"
        examples = group[group["system"] == SYSTEM_VA_TENSION].sort_values("rank").head(8)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.axis("off")
        ax.set_title(f"Top affect-aware retrieval examples: {query_id}", loc="left")
        table_data = [
            [
                int(row["rank"]),
                str(row.get("title", ""))[:28],
                str(row.get("artist", ""))[:22],
                f"{_as_float(row.get('tension_strength_percentile')):.2f}",
                f"{_as_float(row.get('affective_complexity_score')):.2f}",
            ]
            for _, row in examples.iterrows()
        ]
        if not table_data:
            table_data = [["", "No examples", "", "", ""]]
        table = ax.table(
            cellText=table_data,
            colLabels=["Rank", "Title", "Artist", "pT", "ACS"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        fig.tight_layout()
        fig.savefig(query_path)
        plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int | None = None) -> str:
    table = frame.head(limit) if limit is not None else frame
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    if table.empty:
        lines.append("| " + " | ".join("n/a" for _ in columns) + " |")
        return "\n".join(lines)
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns)
            + " |"
        )
    return "\n".join(lines)


def write_report(
    *,
    out_dir: Path,
    queries: Sequence[Mapping[str, Any]],
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    annotation_pool: pd.DataFrame,
    sanity: Mapping[str, Any],
    mirror_path: Path | None = DEFAULT_REPORT_MIRROR,
) -> None:
    report_path = out_dir / "affect_retrieval_report.md"
    metric_table = metrics.copy()
    if not metric_table.empty:
        for column in ["ndcg", "mean_relevance", "precision_relevance_ge_2", "strong_support_rate"]:
            metric_table[column] = metric_table[column].map(lambda value: f"{_as_float(value, float('nan')):.3f}")
    lines = [
        "# Affect-aware Retrieval Evaluation",
        "",
        "## Task Definition",
        "",
        "This downstream module evaluates whether fixed Dataset-S v20.3 VA+tension representations retrieve songs with complex affective structure better than a VA-only baseline. It does not retrain the upstream VA model, rerun clustering, change cluster assignments, change tension subtype assignments, or alter the post-hoc v3 interpretation rules.",
        "",
        "## Query Definitions",
        "",
    ]
    query_rows = pd.DataFrame(
        [
            {
                "query_id": query.get("query_id", ""),
                "description": query.get("description", ""),
                "target_cluster": query.get("target_cluster", ""),
                "target_tension_strength": query.get("target_tension_strength", ""),
                "target_complexity": query.get("target_complexity", ""),
            }
            for query in queries
        ]
    )
    lines.append(_markdown_table(query_rows, list(query_rows.columns)))
    lines.extend(
        [
            "",
            "## Systems Compared",
            "",
            "- System A (`va_only`): balanced valence/arousal only, scored by an exponential normalized VA distance.",
            "- System B (`va_cluster`): VA similarity plus target region soft weights, region typicality, and geometry role matching.",
            "- System C (`va_tension_complexity`): fixed VA, region soft weights, calibrated audio-lyric tension direction/strength, affective complexity score, and geometry role matching.",
            "",
            "## Scoring Formula",
            "",
            "`score_va = exp(-0.5 * normalized_va_distance^2)`.",
            "",
            "`score_cluster = 0.50*score_va + 0.30*region_match + 0.10*region_typicality_match + 0.10*boundary_or_mixture_match`.",
            "",
            "`score_full = 0.30*score_va + 0.20*region_match + 0.25*tension_match + 0.15*complexity_match + 0.10*geometry_role_match`.",
            "",
            "All weights are fixed a priori. External reviews and relevance labels are not used in ranking.",
            "",
            "## External Annotation Protocol",
            "",
            "External professional reviews are used only after retrieval outputs are fixed. Accepted sources are professional music criticism or mainstream media music reviews. Lyrics websites, Genius annotations, Reddit, forums, user comments, Spotify tags, YouTube comments, Amazon reviews, unattributed snippets, and generated content are not accepted as primary evidence. Evidence summaries are paraphrases and do not quote lyrics.",
            "",
            "Grades: region relevance and tension relevance are each 0-3. For simple prototype queries, overall relevance equals region relevance. For tension or complex queries, overall relevance is rounded from `0.45*region + 0.55*tension`.",
            "",
            "## Metric Table",
            "",
        ]
    )
    if metric_table.empty:
        lines.append("No verified external relevance labels are available yet, so retrieval metrics are gated off.")
    else:
        lines.append(
            _markdown_table(
                metric_table,
                [
                    "query_id",
                    "system",
                    "k",
                    "ndcg",
                    "mean_relevance",
                    "precision_relevance_ge_2",
                    "strong_support_rate",
                    "judged_count",
                ],
            )
        )
    lines.extend(["", "## Summary", ""])
    if summary.empty:
        lines.append("No cross-query summary is available until at least one external label is verified.")
    else:
        summary_display = summary.copy()
        for column in summary_display.columns:
            if column != "system":
                summary_display[column] = summary_display[column].map(lambda value: f"{_as_float(value, float('nan')):.3f}")
        lines.append(_markdown_table(summary_display, list(summary_display.columns)))
    lines.extend(["", "## Per-query Qualitative Examples", ""])
    example_cols = [
        "query_id",
        "title",
        "artist",
        "system_sources",
        "cluster_name",
        "tension_strength_percentile",
        "affective_complexity_score",
        "annotation_status",
    ]
    examples = annotation_pool.copy()
    if not examples.empty:
        for column in ["tension_strength_percentile", "affective_complexity_score"]:
            examples[column] = pd.to_numeric(examples[column], errors="coerce").map(lambda value: f"{_as_float(value, float('nan')):.2f}")
    lines.append(_markdown_table(examples, example_cols, limit=18))
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is not a final user satisfaction evaluation for a music recommender.",
            "- Metrics depend on post-hoc professional review coverage; unverified rows are excluded from judged-label metrics.",
            "- External evidence is not allowed to tune scoring weights, construct query-specific rankings, or revise model-side outputs.",
            "- The experiment evaluates retrieval-oriented downstream validity of a fixed representation, not causal preference or listening behavior.",
            "",
            "## Sanity Check",
            "",
            "```json",
            json.dumps(sanity, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Recommended LaTeX Insertion Text",
            "",
            r"\paragraph{Affect-aware retrieval.}",
            "We evaluate the fixed Dataset-S v20.3 representation in a retrieval-oriented downstream setting. A VA-only baseline ranks songs by integrated balanced valence and arousal, which captures the song's overall affective position but cannot directly express cross-modal affective structure. The full affect-aware retrieval system additionally uses calibrated audio-lyric tension, region soft weights, boundary margin, and affective complexity. This allows queries for structures such as subdued affect with lyrical lift, volatile intensity with semantic amplification, and bright audio-led vitality with a darker lyrical undertone. External professional reviews are used only after retrieval outputs are fixed, as post-hoc critical relevance labels. We therefore interpret the results as a downstream validation of representation quality for complex-affect retrieval, not as a final evaluation of user satisfaction in a recommendation system.",
            "",
            r"\caption{Affect-aware retrieval evaluation. VA-only retrieval ranks songs by balanced affective position, while the full representation additionally uses calibrated audio-lyric tension and affective complexity. Relevance grades are assigned from external professional descriptions after retrieval outputs are fixed.}",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if mirror_path is not None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report_path, mirror_path)


def write_evaluation_outputs(
    *,
    out_dir: Path,
    queries: Sequence[Mapping[str, Any]],
    rankings: pd.DataFrame,
    annotation_pool: pd.DataFrame,
    labels: pd.DataFrame,
    top_k: Sequence[int],
    sanity: dict[str, Any],
    make_figures: bool = True,
    mirror_report_path: Path | None = DEFAULT_REPORT_MIRROR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics, summary = evaluate_retrieval_results(rankings, labels, top_k=top_k)
    metrics.to_csv(out_dir / "retrieval_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "retrieval_metrics_summary.csv", index=False, encoding="utf-8-sig")
    _write_metrics_table_tex(metrics, out_dir / "tables" / "table_retrieval_metrics.tex")
    _write_examples_table_tex(annotation_pool, out_dir / "tables" / "table_retrieval_examples.tex")
    if make_figures:
        write_figures(metrics, rankings, labels, out_dir)
    sanity = dict(sanity)
    sanity["metrics_available"] = bool(not metrics.empty and metrics["metrics_available"].any())
    write_report(
        out_dir=out_dir,
        queries=queries,
        metrics=metrics,
        summary=summary,
        annotation_pool=annotation_pool,
        sanity=sanity,
        mirror_path=mirror_report_path,
    )
    return metrics, summary, sanity


def run_affect_retrieval_eval(
    *,
    cluster_csv: Path | str | None = None,
    tension_csv: Path | str | None = None,
    interpretation_csv: Path | str | None = None,
    external_evidence_csv: Path | str | None = None,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    query_config: Path | str | None = DEFAULT_QUERY_CONFIG,
    root: Path | str = ".",
    top_k: Sequence[int] = DEFAULT_TOP_K,
    retrieval_depth: int | None = None,
    make_figures: bool = True,
    mirror_report_path: Path | str | None = DEFAULT_REPORT_MIRROR,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    output_path = Path(out_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    retrieval_depth = max(top_k) if retrieval_depth is None else retrieval_depth

    cluster_path = _resolve_csv(
        cluster_csv,
        root=root_path,
        preferred_names=["dataset_s_cluster_assignments.csv", "cluster_assignments.csv"],
        column_groups=[
            COLUMN_ALIASES["song_id"],
            COLUMN_ALIASES["cluster_id"],
            COLUMN_ALIASES["balanced_valence"],
            COLUMN_ALIASES["balanced_arousal"],
        ],
        required=False,
        allow_generic_search=False,
    )
    interpretation_path = _resolve_interpretation_csv(
        interpretation_csv,
        root=root_path,
        out_dir=output_path,
    )
    tension_path = _resolve_csv(
        tension_csv,
        root=root_path,
        preferred_names=["dataset_s_tension_assignments.csv", "tension_subtype_assignments.csv"],
        column_groups=[
            COLUMN_ALIASES["song_id"],
            COLUMN_ALIASES["tension_label"],
            COLUMN_ALIASES["tension_dv"],
            COLUMN_ALIASES["tension_da"],
        ],
        required=False,
    )
    queries = load_query_config(query_config)
    (output_path / "retrieval_queries.yaml").write_text(_query_yaml(queries), encoding="utf-8")

    features, feature_sanity = build_song_retrieval_features(
        cluster_csv=cluster_path,
        tension_csv=tension_path,
        interpretation_csv=interpretation_path,
    )
    features.to_csv(output_path / "song_retrieval_features.csv", index=False, encoding="utf-8-sig")
    rankings = compute_retrieval_rankings(features, queries, retrieval_depth=retrieval_depth)
    write_retrieval_result_files(rankings, output_path)

    external_evidence = _read_csv(external_evidence_csv) if external_evidence_csv else None
    annotation_pool = build_annotation_pool(
        rankings,
        features,
        retrieval_depth=retrieval_depth,
        external_evidence=external_evidence,
    )
    annotation_pool.to_csv(output_path / "external_annotation_pool.csv", index=False, encoding="utf-8-sig")
    source_queries = build_source_search_queries(annotation_pool)
    source_queries.to_csv(output_path / "source_search_queries.csv", index=False, encoding="utf-8-sig")
    labels = build_external_review_labels(annotation_pool, external_evidence, queries)
    labels.to_csv(output_path / "external_relevance_labels.csv", index=False, encoding="utf-8-sig")

    sanity: dict[str, Any] = {
        **feature_sanity,
        "number_of_queries": len(queries),
        "systems": SYSTEMS,
        "topK": list(top_k),
        "annotation_pool_size": int(len(annotation_pool)),
        "verified_external_labels_count": int(labels["annotation_status"].isin(["verified", "partial"]).sum())
        if not labels.empty and "annotation_status" in labels.columns
        else 0,
        "unverified_count": int(labels["annotation_status"].eq("unverified").sum())
        if not labels.empty and "annotation_status" in labels.columns
        else int(len(annotation_pool)),
        "contradiction_count": int(labels["annotation_status"].eq("contradiction").sum())
        if not labels.empty and "annotation_status" in labels.columns
        else 0,
        "metrics_available": False,
        "whether_any_external_label_used_in_scoring": False,
    }
    _metrics, _summary, final_sanity = write_evaluation_outputs(
        out_dir=output_path,
        queries=queries,
        rankings=rankings,
        annotation_pool=annotation_pool,
        labels=labels,
        top_k=top_k,
        sanity=sanity,
        make_figures=make_figures,
        mirror_report_path=Path(mirror_report_path) if mirror_report_path else None,
    )
    _write_json(output_path / "sanity_check_retrieval.json", final_sanity)
    return final_sanity


def parse_top_k(value: str | None) -> tuple[int, ...]:
    if not value:
        return DEFAULT_TOP_K
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster_csv", default=None)
    parser.add_argument("--tension_csv", default=None)
    parser.add_argument("--interpretation_csv", default=None)
    parser.add_argument("--external_evidence_csv", default=None)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--query_config", default=str(DEFAULT_QUERY_CONFIG))
    parser.add_argument("--root", default=".")
    parser.add_argument("--top_k", default="5,10,20", help="Comma-separated top-K values.")
    parser.add_argument("--retrieval_depth", type=int, default=None)
    parser.add_argument("--skip_figures", action="store_true")
