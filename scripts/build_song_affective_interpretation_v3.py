"""Build v3 rule-based affective interpretations from fixed v2 song profiles.

This script is post-hoc only. It reads v2 profile outputs, preserves the fixed
cluster and tension assignments, and combines existing numeric fields into a
reproducible single-song affective interpretation.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd


EPS = 1e-8
C1_OPPOSITION_MAGNITUDE_THRESHOLD = 0.25

REQUIRED_V2_FIELDS = [
    "song_id",
    "cluster_id",
    "cluster_name",
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
    "tension_strength_percentile",
]

REGION_LEXICON: Dict[int, Dict[str, Any]] = {
    0: {
        "cluster_name": "Subdued Melancholy",
        "primary_affect": "Subdued Melancholy",
        "gems_family": ["sadness", "nostalgia"],
        "mirex_family": ["wistful", "bittersweet", "brooding"],
        "core_descriptors": [
            "low-arousal melancholy",
            "introspective sadness",
            "somber restraint",
        ],
    },
    1: {
        "cluster_name": "Gentle Warmth",
        "primary_affect": "Gentle Warmth",
        "gems_family": ["tenderness", "peacefulness", "nostalgia"],
        "mirex_family": ["sweet", "amiable", "gentle"],
        "core_descriptors": [
            "soft warmth",
            "calm-positive tenderness",
            "romantic restraint",
        ],
    },
    2: {
        "cluster_name": "Volatile Intensity",
        "primary_affect": "Volatile Intensity",
        "gems_family": ["tension", "power", "sadness"],
        "mirex_family": ["volatile", "fiery", "aggressive", "tense/anxious"],
        "core_descriptors": [
            "negative-active tension",
            "confrontational intensity",
            "high-arousal unease",
        ],
    },
    3: {
        "cluster_name": "Playful Vitality",
        "primary_affect": "Playful Vitality",
        "gems_family": ["joyful activation", "power"],
        "mirex_family": ["fun", "rousing", "cheerful", "rollicking"],
        "core_descriptors": [
            "bright vitality",
            "danceable activation",
            "celebratory energy",
        ],
    },
}

DIRECTIONAL_RELATIONS = {
    "lyric_valence_uplift",
    "lyric_valence_tempering",
    "lyric_arousal_intensification",
    "lyric_arousal_softening",
}

DIRECTIONAL_LABEL_TERMS = [
    "lyric",
    "lyrical",
    "valence-reframed",
    "undercurrent",
    "amplified tension",
    "cross-modal tension",
]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_descriptor_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object keyed by song_id.")
    return payload


def _validate_required_fields(frame: pd.DataFrame, source_name: str) -> None:
    missing = [column for column in REQUIRED_V2_FIELDS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required v2 fields: {missing}")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _cluster_id(value: Any) -> Optional[int]:
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        match = re.search(r"\bC?(-?\d+)\b", str(value))
        if match is None:
            return None
        number = int(match.group(1))
    return number if number in REGION_LEXICON else None


def _cluster_id_from_text(value: Any) -> Optional[int]:
    text = str(value)
    match = re.search(r"\bC\s*([0-3])\b", text, flags=re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    normalized = text.strip().lower()
    for cluster_id, lexicon in REGION_LEXICON.items():
        if str(lexicon["cluster_name"]).lower() in normalized:
            return cluster_id
    return None


def _primary_affect(cluster_id: Optional[int], fallback: Any = "") -> str:
    if cluster_id in REGION_LEXICON:
        return str(REGION_LEXICON[int(cluster_id)]["primary_affect"])
    return str(fallback).strip() or "Unknown Affect"


def _cluster_token(cluster_id: Optional[int]) -> str:
    return f"C{cluster_id}" if cluster_id is not None else "C?"


def _format_pct(value: Any) -> str:
    return f"{100.0 * _as_float(value):.1f}%"


def _format_float(value: Any) -> str:
    return f"{_as_float(value):.3f}"


def _markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _assigned_region_weight(row: pd.Series, cluster_id: Optional[int]) -> float:
    column = f"w_region_{_cluster_token(cluster_id)}"
    return _as_float(row[column]) if column in row.index else 0.0


def _nearest_alt_cluster_id(row: pd.Series) -> Optional[int]:
    parsed = _cluster_id_from_text(row.get("nearest_alt_cluster", ""))
    if parsed is not None:
        return parsed
    weights = {
        cluster_id: _as_float(row.get(f"w_region_C{cluster_id}", 0.0))
        for cluster_id in REGION_LEXICON
    }
    assigned = _cluster_id(row.get("cluster_id"))
    alternatives = {key: value for key, value in weights.items() if key != assigned}
    if not alternatives:
        return None
    return max(alternatives, key=alternatives.get)


def _nearest_alt_weight(row: pd.Series, alt_cluster_id: Optional[int]) -> float:
    if alt_cluster_id is not None:
        column = f"w_region_C{alt_cluster_id}"
        if column in row.index:
            return _as_float(row[column])
    return _as_float(row.get("nearest_alt_cluster_weight", 0.0))


def _region_mixture_type(region_margin: float, blend_ratio: float) -> str:
    if region_margin < 0.0 or blend_ratio >= 0.80:
        return "boundary_blend"
    if blend_ratio >= 0.45:
        return "strong_secondary_affect"
    if blend_ratio >= 0.20:
        return "mild_secondary_undertone"
    return "clear_primary_region"


def _geometry_role(
    *,
    region_mixture_type: str,
    region_typicality: float,
    region_confidence: float,
    region_margin: float,
) -> str:
    if region_mixture_type == "boundary_blend":
        return "boundary"
    if region_typicality >= 0.80:
        return "prototype"
    if region_typicality >= 0.60:
        return "representative"
    if region_typicality >= 0.30:
        return "peripheral_representative"
    if region_confidence >= 0.75 and region_margin > 0.0:
        return "peripheral_extreme"
    return "ambiguous_boundary"


def _secondary_affect(
    *,
    mixture_type: str,
    primary_affect: str,
    alt_affect: str,
) -> str:
    if mixture_type == "mild_secondary_undertone":
        return f"with a mild {alt_affect} undertone"
    if mixture_type == "strong_secondary_affect":
        return f"with a strong {alt_affect} undertone"
    if mixture_type == "boundary_blend":
        return f"{primary_affect} / {alt_affect} boundary blend"
    return ""


def _tension_strength_level(percentile: float) -> str:
    if percentile < 0.25:
        return "low"
    if percentile < 0.60:
        return "mild"
    if percentile < 0.80:
        return "moderate"
    return "high"


def _is_modality_consistent(tension_name: Any) -> bool:
    normalized = str(tension_name).strip().lower().replace("_", "-")
    return "modality-consistent" in normalized or "modality consistent" in normalized


def _cross_modal_relation(
    *,
    tension_strength_level: str,
    tension_percentile: float,
    tension_name: Any,
    tension_dv: float,
    tension_da: float,
) -> str:
    if tension_percentile < 0.25 or _is_modality_consistent(tension_name):
        return "affective_concordance"

    directions = []
    prefix = "mild_" if tension_strength_level == "mild" else ""
    if tension_dv > 0.0:
        directions.append(f"{prefix}lyric_valence_uplift")
    elif tension_dv < 0.0:
        directions.append(f"{prefix}lyric_valence_tempering")
    if tension_da > 0.0:
        directions.append(f"{prefix}lyric_arousal_intensification")
    elif tension_da < 0.0:
        directions.append(f"{prefix}lyric_arousal_softening")
    return "; ".join(directions) if directions else "affective_concordance"


def _tension_overlay(
    *,
    cluster_id: Optional[int],
    geometry_role: str,
    tension_percentile: float,
    tension_dv: float,
    tension_da: float,
) -> str:
    if cluster_id == 3 and tension_percentile < 0.40 and geometry_role in {"prototype", "representative"}:
        return "Concordant Vitality"
    if tension_percentile < 0.25:
        return "Low Tension Concordance"
    if tension_percentile < 0.60:
        return "Mild Cross-modal Inflection"

    if cluster_id == 0 and tension_dv > 0.0:
        return "Bittersweet Lyrical Lift"
    if cluster_id == 1 and (tension_dv > 0.0 or tension_da > 0.0):
        return "Warm Elegiac / Lyrical Warmth Lift"
    if cluster_id == 2 and tension_percentile >= 0.70 and tension_da > 0.0:
        return "Lyrically Amplified Tension"
    if cluster_id == 2 and tension_dv > 0.0 and tension_da <= 0.0:
        return "Valence-Reframed Intensity"
    if cluster_id == 3 and (tension_dv < 0.0 or tension_da < 0.0):
        return "Audio-led Exuberance with Dark Lyrical Undercurrent"
    if tension_percentile >= 0.80:
        return "High Cross-modal Tension"
    return "Moderate Cross-modal Tension"


def _cross_modal_opposition_score(
    *,
    cluster_id: Optional[int],
    tension_percentile: float,
    tension_dv: float,
    tension_da: float,
) -> int:
    if tension_percentile < 0.60:
        return 0
    if cluster_id == 0 and tension_dv > 0.0:
        return 1
    if cluster_id == 1 and max(abs(tension_dv), abs(tension_da)) >= C1_OPPOSITION_MAGNITUDE_THRESHOLD:
        return 1
    if cluster_id == 2 and tension_dv > 0.0:
        return 1
    if cluster_id == 3 and (tension_dv < 0.0 or tension_da < 0.0):
        return 1
    return 0


def _complexity_level(score: float) -> str:
    if score < 0.25:
        return "simple / concordant"
    if score < 0.50:
        return "mildly complex"
    if score < 0.70:
        return "complex"
    return "highly complex / ambivalent"


def _final_interpretation_label(
    *,
    primary_affect: str,
    alt_affect: str,
    geometry_role: str,
    mixture_type: str,
    tension_percentile: float,
    tension_strength_level: str,
    tension_overlay: str,
) -> str:
    if mixture_type == "boundary_blend":
        return f"{primary_affect} / {alt_affect} Boundary Blend"
    if mixture_type == "mild_secondary_undertone":
        return f"{primary_affect} with mild {alt_affect} undertone"
    if mixture_type == "strong_secondary_affect":
        return f"{primary_affect} with strong {alt_affect} undertone"
    if geometry_role == "prototype" and mixture_type == "clear_primary_region" and tension_strength_level == "low":
        return f"Concordant {primary_affect}"
    if geometry_role == "prototype" and tension_percentile >= 0.60:
        return f"{primary_affect} with {tension_overlay}"
    if geometry_role == "peripheral_extreme" and tension_percentile >= 0.70:
        return f"Peripheral high-tension {primary_affect}"
    if tension_percentile >= 0.60 and tension_overlay not in {
        "Moderate Cross-modal Tension",
        "High Cross-modal Tension",
    }:
        return f"{primary_affect} with {tension_overlay}"

    role_prefix = {
        "prototype": "Prototype",
        "representative": "Representative",
        "peripheral_representative": "Peripheral Representative",
        "peripheral_extreme": "Peripheral Extreme",
        "ambiguous_boundary": "Ambiguous Boundary",
        "boundary": "Boundary",
    }.get(geometry_role, "Post-hoc")
    if tension_strength_level in {"moderate", "high"}:
        return f"{role_prefix} {primary_affect} with {tension_strength_level} tension"
    return f"{role_prefix} {primary_affect}"


def _contains_directional_label_term(label: Any) -> bool:
    lower = str(label).lower()
    return any(term in lower for term in DIRECTIONAL_LABEL_TERMS)


def _explicit_or_encoding_issue(row: pd.Series) -> bool:
    return any(
        _as_bool(row.get(column, False))
        for column in ["explicit_or_encoding_issue", "explicit_title_flag", "encoding_issue_flag"]
    )


def _descriptor_conflict(row: pd.Series) -> bool:
    return _as_bool(row.get("descriptor_conflict_flag", False))


def _main_text_eligible(row: pd.Series) -> bool:
    final_label = str(row.get("final_interpretation_label", "")).strip()
    selected_role = str(row.get("selected_role", "")).strip()
    low_tension = str(row.get("tension_strength_level", "")) == "low"
    modality_consistent = _is_modality_consistent(row.get("tension_name", ""))
    directional_label = _contains_directional_label_term(final_label)

    if not final_label:
        return False
    if _explicit_or_encoding_issue(row):
        return False
    if _descriptor_conflict(row):
        return False
    if row.get("region_mixture_type") == "boundary_blend" and selected_role != "boundary_case":
        return False
    if low_tension and directional_label:
        return False
    if modality_consistent and _as_float(row.get("tension_strength_percentile")) < 0.25 and directional_label:
        return False
    return True


def _natural_explanation(
    *,
    primary_affect: str,
    secondary_affect: str,
    mixture_type: str,
    tension_strength_level: str,
    cross_modal_relation: str,
    complexity_level: str,
) -> str:
    secondary = secondary_affect or "without a substantive secondary affective undertone"
    if mixture_type == "boundary_blend":
        return (
            f"the song should be treated as a boundary mixture rather than a single-region prototype; "
            f"its {primary_affect} center is qualified by {secondary}"
        )
    if cross_modal_relation == "affective_concordance":
        return (
            f"the {primary_affect} region remains the main affective frame, {secondary}, "
            f"and the audio-lyric relation is not directional enough to dominate the interpretation"
        )
    return (
        f"the {primary_affect} region is the main frame, {secondary}, with {tension_strength_level} "
        f"audio-lyric contrast ({cross_modal_relation}) and {complexity_level} affective structure"
    )


def _build_chinese_evaluation(row: pd.Series) -> str:
    secondary = row["secondary_affect"] or "没有达到可作为主结论的次级情绪邻近性"
    relation = row["cross_modal_relation"]
    if relation == "affective_concordance":
        direction_sentence = (
            "该关系被解释为 affective_concordance，因此不放大歌词与音频之间的方向性差异。"
        )
    elif row["tension_strength_level"] == "mild":
        direction_sentence = f"方向性只作为 mild 层级的辅助线索：{relation}。"
    elif row["tension_strength_percentile"] >= 0.60:
        direction_sentence = f"方向性可以作为 cross-modal tension 证据：{relation}。"
    else:
        direction_sentence = f"方向性仅作弱证据记录：{relation}。"

    boundary_sentence = ""
    if row["region_mixture_type"] == "boundary_blend":
        boundary_sentence = "由于它是 boundary_blend，不应写成单一原型歌曲。"

    natural = _natural_explanation(
        primary_affect=row["primary_affect"],
        secondary_affect=row["secondary_affect"],
        mixture_type=row["region_mixture_type"],
        tension_strength_level=row["tension_strength_level"],
        cross_modal_relation=row["cross_modal_relation"],
        complexity_level=row["complexity_level"],
    )
    return (
        f"该歌曲的主情感落在 {row['primary_affect']}，其 region typicality 为 "
        f"{_format_pct(row['region_typicality'])}，region confidence 为 "
        f"{_format_pct(row['region_confidence'])}，因此几何角色为 {row['geometry_role']}。"
        f"它与最近替代区域 {row['nearest_alt_affect']} 的 blend ratio 为 "
        f"{_format_float(row['blend_ratio'])}，属于 {row['region_mixture_type']}；{secondary}。"
        f"tension strength percentile 为 {_format_pct(row['tension_strength_percentile'])}，"
        f"强度等级为 {row['tension_strength_level']}，audio-lyrics 关系表现为 {relation}。"
        f"{direction_sentence}{boundary_sentence}"
        f"综合这些参数，该歌曲更适合定义为 {row['final_interpretation_label']}，即 {natural}。"
    )


def _build_english_evaluation(row: pd.Series) -> str:
    secondary = row["secondary_affect"] or "no secondary affective undertone is strong enough to lead"
    relation = row["cross_modal_relation"]
    if relation == "affective_concordance":
        direction_sentence = (
            "The relation is treated as affective_concordance, so directional audio-lyric differences are not amplified."
        )
    elif row["tension_strength_level"] == "mild":
        direction_sentence = f"The directional evidence is mild and remains secondary: {relation}."
    elif row["tension_strength_percentile"] >= 0.60:
        direction_sentence = f"The directional relation can be used as cross-modal tension evidence: {relation}."
    else:
        direction_sentence = f"The directional relation is recorded only as weak evidence: {relation}."

    boundary_sentence = ""
    if row["region_mixture_type"] == "boundary_blend":
        boundary_sentence = " Because this is a boundary_blend, it should not be written as a single-region prototype."

    natural = _natural_explanation(
        primary_affect=row["primary_affect"],
        secondary_affect=row["secondary_affect"],
        mixture_type=row["region_mixture_type"],
        tension_strength_level=row["tension_strength_level"],
        cross_modal_relation=row["cross_modal_relation"],
        complexity_level=row["complexity_level"],
    )
    return (
        f"The song's primary affect lies in {row['primary_affect']}; region typicality is "
        f"{_format_pct(row['region_typicality'])}, region confidence is "
        f"{_format_pct(row['region_confidence'])}, and the geometry role is {row['geometry_role']}. "
        f"Its blend ratio with the nearest alternative region, {row['nearest_alt_affect']}, is "
        f"{_format_float(row['blend_ratio'])}, so the mixture type is {row['region_mixture_type']}; {secondary}. "
        f"The tension strength percentile is {_format_pct(row['tension_strength_percentile'])} "
        f"({row['tension_strength_level']}), and the audio-lyrics relation is {relation}. "
        f"{direction_sentence}{boundary_sentence} "
        f"Taken together, the song is best defined as {row['final_interpretation_label']}: {natural}."
    )


def _interpret_row(row: pd.Series) -> Dict[str, Any]:
    cluster_id = _cluster_id(row.get("cluster_id"))
    alt_cluster_id = _nearest_alt_cluster_id(row)
    primary = _primary_affect(cluster_id, row.get("cluster_name", ""))
    alt_affect = _primary_affect(alt_cluster_id, row.get("nearest_alt_cluster", ""))

    w1 = _assigned_region_weight(row, cluster_id)
    w2 = _nearest_alt_weight(row, alt_cluster_id)
    blend_ratio = w2 / max(w1, EPS)
    region_margin = _as_float(row.get("region_margin"))
    region_typicality = _as_float(row.get("region_typicality"))
    region_confidence = _as_float(row.get("region_confidence"))
    mixture_type = _region_mixture_type(region_margin, blend_ratio)
    geometry = _geometry_role(
        region_mixture_type=mixture_type,
        region_typicality=region_typicality,
        region_confidence=region_confidence,
        region_margin=region_margin,
    )
    secondary = _secondary_affect(
        mixture_type=mixture_type,
        primary_affect=primary,
        alt_affect=alt_affect,
    )

    tension_percentile = _as_float(row.get("tension_strength_percentile"))
    tension_percentile = max(0.0, min(1.0, tension_percentile))
    tension_dv = _as_float(row.get("tension_dv"))
    tension_da = _as_float(row.get("tension_da"))
    tension_level = _tension_strength_level(tension_percentile)
    cross_modal = _cross_modal_relation(
        tension_strength_level=tension_level,
        tension_percentile=tension_percentile,
        tension_name=row.get("tension_name", ""),
        tension_dv=tension_dv,
        tension_da=tension_da,
    )
    overlay = _tension_overlay(
        cluster_id=cluster_id,
        geometry_role=geometry,
        tension_percentile=tension_percentile,
        tension_dv=tension_dv,
        tension_da=tension_da,
    )

    blend_score = min(1.0, max(0.0, blend_ratio))
    peripheral_score = 1.0 - max(0.0, min(1.0, region_typicality))
    opposition_score = _cross_modal_opposition_score(
        cluster_id=cluster_id,
        tension_percentile=tension_percentile,
        tension_dv=tension_dv,
        tension_da=tension_da,
    )
    complexity_score = (
        0.35 * blend_score
        + 0.35 * tension_percentile
        + 0.15 * peripheral_score
        + 0.15 * opposition_score
    )
    complexity_score = max(0.0, min(1.0, complexity_score))
    final_label = _final_interpretation_label(
        primary_affect=primary,
        alt_affect=alt_affect,
        geometry_role=geometry,
        mixture_type=mixture_type,
        tension_percentile=tension_percentile,
        tension_strength_level=tension_level,
        tension_overlay=overlay,
    )

    interpreted = {
        "primary_affect": primary,
        "secondary_affect": secondary,
        "nearest_alt_affect": alt_affect,
        "assigned_region_weight": w1,
        "nearest_alt_region_weight": w2,
        "blend_ratio": blend_ratio,
        "geometry_role": geometry,
        "region_mixture_type": mixture_type,
        "tension_strength_level": tension_level,
        "tension_overlay": overlay,
        "cross_modal_relation": cross_modal,
        "cross_modal_opposition_score": opposition_score,
        "blend_score": blend_score,
        "peripheral_score": peripheral_score,
        "affective_complexity_score": complexity_score,
        "complexity_level": _complexity_level(complexity_score),
        "final_interpretation_label": final_label,
        "explicit_or_encoding_issue": _explicit_or_encoding_issue(row),
        "descriptor_conflict": _descriptor_conflict(row),
    }
    scratch = row.to_dict()
    scratch.update(interpreted)
    interpreted["chinese_professional_evaluation"] = _build_chinese_evaluation(pd.Series(scratch))
    interpreted["english_professional_evaluation"] = _build_english_evaluation(pd.Series(scratch))
    return interpreted


def _apply_interpretation(profile: pd.DataFrame) -> pd.DataFrame:
    _validate_required_fields(profile, "profile_all_csv")
    interpreted = [_interpret_row(row) for _, row in profile.iterrows()]
    result = profile.copy()
    for key in interpreted[0].keys() if interpreted else []:
        result[key] = [item[key] for item in interpreted]
    if result.empty:
        for column in _derived_columns():
            result[column] = []
    result["main_text_interpretation_eligible"] = [
        _main_text_eligible(row) for _, row in result.iterrows()
    ]
    return result


def _derived_columns() -> Sequence[str]:
    return [
        "primary_affect",
        "secondary_affect",
        "nearest_alt_affect",
        "assigned_region_weight",
        "nearest_alt_region_weight",
        "blend_ratio",
        "geometry_role",
        "region_mixture_type",
        "tension_strength_level",
        "tension_overlay",
        "cross_modal_relation",
        "cross_modal_opposition_score",
        "blend_score",
        "peripheral_score",
        "affective_complexity_score",
        "complexity_level",
        "final_interpretation_label",
        "explicit_or_encoding_issue",
        "descriptor_conflict",
        "chinese_professional_evaluation",
        "english_professional_evaluation",
        "main_text_interpretation_eligible",
    ]


def _select_rows_from_all(all_v3: pd.DataFrame, selected_v2: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "song_id" not in selected_v2.columns:
        raise ValueError("profile_selected_csv is missing required v2 fields: ['song_id']")
    selected_ids = selected_v2["song_id"].astype(str).tolist()
    by_song = all_v3.assign(song_id=all_v3["song_id"].astype(str)).set_index("song_id", drop=False)
    selected_gate_columns = [
        "selected_role",
        "candidate_type",
        "descriptor_conflict_flag",
        "encoding_issue_flag",
        "explicit_title_flag",
        "explicit_or_encoding_issue",
        "descriptor_conflict",
    ]
    rows = []
    for _, selected_row in selected_v2.assign(song_id=selected_v2["song_id"].astype(str)).iterrows():
        song_id = selected_row["song_id"]
        if song_id not in by_song.index:
            continue
        base = by_song.loc[song_id]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        merged = base.copy()
        for column in selected_gate_columns:
            if column in selected_v2.columns and not pd.isna(selected_row.get(column)):
                merged[column] = selected_row[column]
        rows.append(merged)
    missing_count = len(selected_ids) - len(rows)
    selected = pd.DataFrame(rows).reset_index(drop=True) if rows else all_v3.iloc[0:0].copy()
    if not selected.empty:
        selected["main_text_interpretation_eligible"] = [
            _main_text_eligible(row) for _, row in selected.iterrows()
        ]
    return selected, missing_count


def _rules_payload() -> Dict[str, Any]:
    return {
        "post_hoc_only": True,
        "assignment_policy": {
            "retrain": False,
            "change_cluster_assignment": False,
            "change_tension_assignment": False,
            "source": "Dataset-S v20.3 post-hoc song affective profile v2",
        },
        "required_v2_fields": REQUIRED_V2_FIELDS,
        "region_mixture_rules": {
            "blend_ratio": "nearest alternative soft weight / max(assigned cluster soft weight, eps)",
            "boundary_blend": "region_margin < 0 or blend_ratio >= 0.80",
            "strong_secondary_affect": "blend_ratio >= 0.45",
            "mild_secondary_undertone": "blend_ratio >= 0.20",
            "clear_primary_region": "otherwise",
        },
        "geometry_role_rules": {
            "boundary": "region_mixture_type == boundary_blend",
            "prototype": "region_typicality >= 0.80",
            "representative": "region_typicality >= 0.60",
            "peripheral_representative": "region_typicality >= 0.30",
            "peripheral_extreme": "region_confidence >= 0.75 and region_margin > 0",
            "ambiguous_boundary": "otherwise",
        },
        "cluster_affect_lexicon": {f"C{key}": value for key, value in REGION_LEXICON.items()},
        "tension_interpretation_rules": {
            "strength_levels": {
                "low": "pT < 0.25",
                "mild": "0.25 <= pT < 0.60",
                "moderate": "0.60 <= pT < 0.80",
                "high": "pT >= 0.80",
            },
            "cross_modal_relation": {
                "affective_concordance": "pT < 0.25 or tension_name contains modality-consistent",
                "directional_terms": sorted(DIRECTIONAL_RELATIONS),
                "evidence_threshold": "Only pT >= 0.60 is treated as cross-modal tension evidence; pT >= 0.80 is high cross-modal tension.",
            },
            "cluster_specific_overlays": {
                "C0": "pT >= 0.60 and tension_dv > 0 -> Bittersweet Lyrical Lift",
                "C1": "pT >= 0.60 and (tension_dv > 0 or tension_da > 0) -> Warm Elegiac / Lyrical Warmth Lift",
                "C2_valence": "pT >= 0.60 and tension_dv > 0 and tension_da <= 0 -> Valence-Reframed Intensity",
                "C2_arousal": "pT >= 0.70 and tension_da > 0 -> Lyrically Amplified Tension",
                "C3_dark": "pT >= 0.60 and (tension_dv < 0 or tension_da < 0) -> Audio-led Exuberance with Dark Lyrical Undercurrent",
                "C3_concordant": "pT < 0.40 and prototype/representative -> Concordant Vitality",
            },
        },
        "taxonomy_rules": [
            "Boundary blends are labeled as assigned / nearest alternative Boundary Blend.",
            "Mild or strong secondary mixtures are labeled as primary affect with secondary undertone.",
            "Clear low-tension prototypes are labeled Concordant primary affect.",
            "High-salience prototype tension cases use the computed tension overlay.",
            "Peripheral extremes with pT >= 0.70 are labeled Peripheral high-tension primary affect.",
        ],
        "complexity_formula": {
            "blend_score": "min(1, blend_ratio)",
            "tension_score": "tension_strength_percentile",
            "peripheral_score": "1 - region_typicality",
            "cross_modal_opposition_score": (
                "1 for rule-defined cluster/tension opposition at pT >= 0.60, otherwise 0"
            ),
            "ACS": "0.35*blend_score + 0.35*tension_score + 0.15*peripheral_score + 0.15*cross_modal_opposition_score",
            "levels": {
                "simple / concordant": "ACS < 0.25",
                "mildly complex": "0.25 <= ACS < 0.50",
                "complex": "0.50 <= ACS < 0.70",
                "highly complex / ambivalent": "ACS >= 0.70",
            },
        },
        "quality_gates": {
            "main_text_interpretation_eligible": [
                "not explicit_or_encoding_issue",
                "region_mixture_type != boundary_blend unless selected_role == boundary_case",
                "no descriptor conflict",
                "if modality-consistent and pT < 0.25, no directional tension terms in final label",
                "final_interpretation_label is not empty",
            ]
        },
        "generated_outputs": [
            "song_affective_interpretation_all_v3.csv",
            "song_affective_interpretation_selected_v3.csv",
            "song_affective_interpretation_report_v3.md",
            "interpretation_rules_v3.json",
            "sanity_check_interpretation_v3.json",
        ],
    }


def _count_directional_violations(frame: pd.DataFrame, mask: Iterable[bool]) -> int:
    if frame.empty:
        return 0
    mask_series = pd.Series(list(mask), index=frame.index)
    directional = frame["final_interpretation_label"].map(_contains_directional_label_term)
    return int((mask_series & directional).sum())


def _sanity_check(
    *,
    all_v3: pd.DataFrame,
    selected_v3: pd.DataFrame,
    descriptor_json: Mapping[str, Any],
    selected_missing_count: int,
) -> Dict[str, Any]:
    low_mask = all_v3["tension_strength_level"].eq("low")
    modality_low_mask = all_v3.apply(
        lambda row: _is_modality_consistent(row.get("tension_name", ""))
        and _as_float(row.get("tension_strength_percentile")) < 0.25,
        axis=1,
    )
    return {
        "total_songs": int(len(all_v3)),
        "selected_songs": int(len(selected_v3)),
        "selected_missing_count": int(selected_missing_count),
        "descriptor_json_song_count": int(len(descriptor_json)),
        "counts_by_final_interpretation_label": {
            str(key): int(value)
            for key, value in all_v3["final_interpretation_label"].value_counts().sort_index().items()
        },
        "counts_by_complexity_level": {
            str(key): int(value)
            for key, value in all_v3["complexity_level"].value_counts().sort_index().items()
        },
        "boundary_blend_count": int(all_v3["region_mixture_type"].eq("boundary_blend").sum()),
        "high_complexity_count": int(all_v3["complexity_level"].eq("highly complex / ambivalent").sum()),
        "low_tension_directional_violation_count": _count_directional_violations(all_v3, low_mask),
        "modality_consistent_directional_violation_count": _count_directional_violations(
            all_v3,
            modality_low_mask,
        ),
        "main_text_interpretation_eligible_count": int(all_v3["main_text_interpretation_eligible"].sum()),
        "missing_required_field_count": 0,
    }


def _append_table(lines: list[str], columns: Sequence[str], rows: pd.DataFrame, limit: Optional[int] = None) -> None:
    table = rows.head(limit) if limit is not None else rows
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    if table.empty:
        lines.append("| " + " | ".join("n/a" for _ in columns) + " |")
        return
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(_markdown_escape(row.get(column, "")) for column in columns) + " |")


def _make_report(
    *,
    all_v3: pd.DataFrame,
    selected_v3: pd.DataFrame,
    sanity: Mapping[str, Any],
    out_dir: Path,
) -> None:
    lines: list[str] = [
        "# Dataset-S v20.3 Post-hoc Song Affective Interpretation v3",
        "",
        "## Method",
        "",
        "This report is post-hoc only: it reads fixed v2 cluster assignments, fixed tension assignments, region geometry fields, soft region weights, and descriptor weights. It does not retrain models and does not alter cluster_id, cluster_name, tension_label, or tension_name.",
        "",
        "## Rule Table",
        "",
        "| Component | Rule Summary |",
        "|---|---|",
        "| Region mixture | assigned/nearest soft-weight blend ratio with negative-margin boundary override |",
        "| Geometry role | boundary first, then typicality thresholds, then confidence/margin fallback |",
        "| Tension relation | low or modality-consistent cases map to affective_concordance; directional terms are evidence only from pT >= 0.60 |",
        "| Final label | boundary and secondary-mixture labels take precedence; clear low-tension prototypes become concordant labels |",
        "| Main-text gate | excludes explicit/encoding issues, descriptor conflicts, unselected boundary blends, and low-tension directional labels |",
        "",
        "## Cluster Affect Lexicon",
        "",
        "| cluster | primary affect | GEMS family | MIREX family | core descriptors |",
        "|---|---|---|---|---|",
    ]
    for cluster_id, lexicon in REGION_LEXICON.items():
        lines.append(
            f"| C{cluster_id} {lexicon['cluster_name']} | {lexicon['primary_affect']} | "
            f"{', '.join(lexicon['gems_family'])} | {', '.join(lexicon['mirex_family'])} | "
            f"{', '.join(lexicon['core_descriptors'])} |"
        )

    lines.extend(
        [
            "",
            "## Tension Interpretation Rules",
            "",
            "- pT < 0.25: low tension; directional terms are not used as the main conclusion.",
            "- 0.25 <= pT < 0.60: mild tension; directions can be mentioned only as mild secondary evidence.",
            "- pT >= 0.60: cross-modal tension may be used as evidence.",
            "- pT >= 0.80: high cross-modal tension may be stated.",
            "- If tension_name is modality-consistent, the relation is treated as affective_concordance.",
            "",
            "## Complexity Score Formula",
            "",
            "`ACS = 0.35*blend_score + 0.35*tension_strength_percentile + 0.15*(1-region_typicality) + 0.15*cross_modal_opposition_score`",
            "",
            "## Selected Songs Interpretation Table",
            "",
        ]
    )
    selected_columns = [
        "song_id",
        "title",
        "artist",
        "primary_affect",
        "secondary_affect",
        "geometry_role",
        "region_mixture_type",
        "tension_strength_level",
        "cross_modal_relation",
        "complexity_level",
        "final_interpretation_label",
    ]
    _append_table(lines, selected_columns, selected_v3)

    lines.extend(["", "## Representative Songs By Final Label", ""])
    rep_rows = []
    sort_columns = ["main_text_interpretation_eligible", "affective_complexity_score", "region_typicality"]
    for label, group in all_v3.groupby("final_interpretation_label", sort=True):
        ranked = group.sort_values(sort_columns, ascending=[False, False, False]).head(3)
        for _, row in ranked.iterrows():
            rep_rows.append(
                {
                    "final_interpretation_label": label,
                    "song_id": row.get("song_id", ""),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "complexity_level": row.get("complexity_level", ""),
                }
            )
    _append_table(
        lines,
        ["final_interpretation_label", "song_id", "title", "artist", "complexity_level"],
        pd.DataFrame(rep_rows),
    )

    lines.extend(["", "## Boundary Cases", ""])
    boundary_rows = all_v3[all_v3["region_mixture_type"].eq("boundary_blend")]
    _append_table(
        lines,
        [
            "song_id",
            "title",
            "artist",
            "primary_affect",
            "nearest_alt_affect",
            "blend_ratio",
            "region_margin",
            "final_interpretation_label",
        ],
        boundary_rows,
    )

    lines.extend(
        [
            "",
            "## Sanity Check Summary",
            "",
            "```json",
            json.dumps(sanity, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Selected Song Professional Evaluations",
            "",
        ]
    )
    if selected_v3.empty:
        lines.append("No selected songs were available.")
    for _, row in selected_v3.iterrows():
        lines.extend(
            [
                f"### {_markdown_escape(row.get('song_id', ''))}",
                "",
                f"Chinese: {row['chinese_professional_evaluation']}",
                "",
                f"English: {row['english_professional_evaluation']}",
                "",
            ]
        )

    (out_dir / "song_affective_interpretation_report_v3.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def run_interpretation_v3(
    *,
    profile_all_csv: Path | str,
    profile_selected_csv: Path | str,
    descriptor_json: Path | str,
    out_dir: Path | str,
) -> Dict[str, Any]:
    all_path = Path(profile_all_csv).expanduser()
    selected_path = Path(profile_selected_csv).expanduser()
    descriptor_path = Path(descriptor_json).expanduser()
    output_path = Path(out_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    profile_all = _read_csv(all_path)
    profile_selected = _read_csv(selected_path)
    descriptor_payload = _load_descriptor_json(descriptor_path)

    all_v3 = _apply_interpretation(profile_all)
    selected_v3, selected_missing_count = _select_rows_from_all(all_v3, profile_selected)

    all_v3.to_csv(
        output_path / "song_affective_interpretation_all_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_v3.to_csv(
        output_path / "song_affective_interpretation_selected_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_json(output_path / "interpretation_rules_v3.json", _rules_payload())
    sanity = _sanity_check(
        all_v3=all_v3,
        selected_v3=selected_v3,
        descriptor_json=descriptor_payload,
        selected_missing_count=selected_missing_count,
    )
    _write_json(output_path / "sanity_check_interpretation_v3.json", sanity)
    _make_report(all_v3=all_v3, selected_v3=selected_v3, sanity=sanity, out_dir=output_path)
    return sanity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v3 rule-based song affective interpretations from fixed Dataset-S v20.3 v2 profiles."
    )
    parser.add_argument("--profile_all_csv", required=True, help="song_affective_profile_all_v2.csv path.")
    parser.add_argument("--profile_selected_csv", required=True, help="song_affective_profile_selected_v2.csv path.")
    parser.add_argument("--descriptor_json", required=True, help="descriptor_weights_selected_v2.json path.")
    parser.add_argument("--out_dir", required=True, help="Output directory for v3 CSV/JSON/Markdown files.")
    args = parser.parse_args()

    sanity = run_interpretation_v3(
        profile_all_csv=args.profile_all_csv,
        profile_selected_csv=args.profile_selected_csv,
        descriptor_json=args.descriptor_json,
        out_dir=args.out_dir,
    )
    print(json.dumps(sanity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
