from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


VA_COLUMNS = ("Audio_Valence", "Audio_Arousal", "Lyrics_Valence", "Lyrics_Arousal")
METADATA_COLUMNS = ("Genres", "Moods", "MoodsAll", "Themes", "Styles")
IDENTITY_COLUMNS = ("Artist", "Title")


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    by_lower = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        found = by_lower.get(str(candidate).strip().lower())
        if found is not None:
            return found
    return None


def _coverage(series: pd.Series) -> float:
    text = series.fillna("").astype(str).str.strip()
    return float((text != "").mean()) if len(text) else 0.0


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df))
    return pd.to_numeric(df[column], errors="coerce")


def audit_raw_csv(csv_path: Path | str, out_dir: Path | str) -> Dict[str, Any]:
    source = Path(csv_path).expanduser()
    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Missing CSV file: {source}")

    df = pd.read_csv(source, low_memory=False)
    columns = list(df.columns)
    song_col = _find_column(columns, ["Song", "song", "track_id", "identifier", "id"])
    if song_col is None:
        raise ValueError(f"Missing song identifier column in {source}. Available columns: {columns}")

    rows = int(len(df))
    song_text = df[song_col].fillna("").astype(str).str.strip()
    unique_song_count = int(song_text.nunique(dropna=False))
    duplicate_song_count = int(song_text.duplicated(keep=False).sum())
    duplicate_summary = (
        song_text.value_counts(dropna=False)
        .rename_axis("song")
        .reset_index(name="count")
        .query("count > 1")
        .head(1000)
    )
    duplicate_summary.to_csv(root / "raw_csv_duplicate_song_summary.csv", index=False, encoding="utf-8")

    column_summary = pd.DataFrame(
        [
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "non_null": int(df[column].notna().sum()),
                "missing": int(df[column].isna().sum()),
                "coverage": float(df[column].notna().mean()) if rows else 0.0,
                "unique": int(df[column].nunique(dropna=True)),
            }
            for column in columns
        ]
    )
    column_summary.to_csv(root / "raw_csv_column_summary.csv", index=False, encoding="utf-8")

    missingness = pd.DataFrame(
        [
            {
                "column": column,
                "missing": int(df[column].isna().sum()),
                "missing_fraction": float(df[column].isna().mean()) if rows else 0.0,
            }
            for column in columns
        ]
    )
    missingness.to_csv(root / "raw_csv_missingness.csv", index=False, encoding="utf-8")

    va_rows: List[Dict[str, Any]] = []
    for column in VA_COLUMNS + ("Original_Valence", "Original_Arousal"):
        actual = _find_column(columns, [column])
        values = _numeric(df, actual) if actual else pd.Series([float("nan")] * rows)
        finite = values.dropna()
        out_of_range = int(((finite < 0.0) | (finite > 1.0)).sum())
        va_rows.append(
            {
                "column": column,
                "present": actual is not None,
                "missing": int(values.isna().sum()),
                "complete": int(values.notna().sum()),
                "min": float(finite.min()) if not finite.empty else float("nan"),
                "max": float(finite.max()) if not finite.empty else float("nan"),
                "out_of_range_count": out_of_range,
            }
        )
    va_range = pd.DataFrame(va_rows)
    va_range.to_csv(root / "raw_csv_va_range_check.csv", index=False, encoding="utf-8")

    audio_val = _numeric(df, _find_column(columns, ["Audio_Valence"]) or "")
    audio_aro = _numeric(df, _find_column(columns, ["Audio_Arousal"]) or "")
    lyrics_val = _numeric(df, _find_column(columns, ["Lyrics_Valence"]) or "")
    lyrics_aro = _numeric(df, _find_column(columns, ["Lyrics_Arousal"]) or "")
    audio_complete = audio_val.notna() & audio_aro.notna()
    lyrics_complete = lyrics_val.notna() & lyrics_aro.notna()
    both_complete = audio_complete & lyrics_complete
    audio_pairs = pd.DataFrame({"v": audio_val.round(8), "a": audio_aro.round(8)})
    lyrics_pairs = pd.DataFrame({"v": lyrics_val.round(8), "a": lyrics_aro.round(8)})
    unique_audio_pairs = int(audio_pairs[audio_complete].drop_duplicates().shape[0])
    unique_lyrics_pairs = int(lyrics_pairs[lyrics_complete].drop_duplicates().shape[0])

    coverage_rows = []
    for column in METADATA_COLUMNS + IDENTITY_COLUMNS:
        actual = _find_column(columns, [column])
        coverage_rows.append(
            {
                "field": column,
                "present": actual is not None,
                "coverage": _coverage(df[actual]) if actual else 0.0,
                "non_empty": int((df[actual].fillna("").astype(str).str.strip() != "").sum()) if actual else 0,
                "unique_non_empty": int(df.loc[df[actual].fillna("").astype(str).str.strip() != "", actual].nunique()) if actual else 0,
            }
        )
    metadata_coverage = pd.DataFrame(coverage_rows)
    metadata_coverage.to_csv(root / "metadata_coverage.csv", index=False, encoding="utf-8")

    mood_determinism: Dict[str, Dict[str, Any]] = {}
    for field in ("Moods", "MoodsAll"):
        mood_col = _find_column(columns, [field])
        deterministic_groups = 0
        checked_groups = 0
        if mood_col is None:
            mood_determinism[field] = {
                "present": False,
                "checked_groups": 0,
                "deterministic_groups": 0,
                "is_deterministic": False,
            }
            continue
        mood_text = df[mood_col].fillna("").astype(str).str.strip()
        grouped = pd.DataFrame(
            {
                "mood": mood_text,
                "audio_pair": audio_val.round(8).astype(str) + "," + audio_aro.round(8).astype(str),
                "audio_complete": audio_complete,
            }
        )
        grouped = grouped[(grouped["mood"] != "") & grouped["audio_complete"]]
        for _mood, group in grouped.groupby("mood"):
            if len(group) < 2:
                continue
            checked_groups += 1
            deterministic_groups += int(group["audio_pair"].nunique() <= 1)
        mood_determinism[field] = {
            "present": True,
            "checked_groups": int(checked_groups),
            "deterministic_groups": int(deterministic_groups),
            "is_deterministic": bool(checked_groups > 0 and deterministic_groups == checked_groups),
        }
    audio_va_is_deterministic_given_moods = any(
        bool(item.get("is_deterministic", False)) for item in mood_determinism.values()
    )

    failures: List[str] = []
    if rows < 10000:
        failures.append("rows < 10000: not Dataset-L scale")
    if duplicate_song_count > 0:
        failures.append("duplicate_song_count > 0")
    both_fraction = float(both_complete.mean()) if rows else 0.0
    if both_fraction < 0.95:
        failures.append("both_audio_lyrics_complete / rows < 0.95")
    if unique_audio_pairs < 1000:
        failures.append("unique_audio_va_pairs < 1000: suspicious audio VA discretization")
    if audio_va_is_deterministic_given_moods:
        failures.append("audio_va_is_deterministic_given_moods: possible mood-derived audio VA leakage")
    for item in va_rows:
        if item["column"] in VA_COLUMNS and int(item["out_of_range_count"]) > 0:
            failures.append(f"{item['column']} has out-of-range values")

    summary: Dict[str, Any] = {
        "csv": str(source),
        "rows": rows,
        "song_column": song_col,
        "unique_song_count": unique_song_count,
        "duplicate_song_count": duplicate_song_count,
        "audio_va_complete_count": int(audio_complete.sum()),
        "lyrics_va_complete_count": int(lyrics_complete.sum()),
        "both_audio_lyrics_complete_count": int(both_complete.sum()),
        "both_audio_lyrics_complete_fraction": both_fraction,
        "unique_audio_va_pairs": unique_audio_pairs,
        "unique_lyrics_va_pairs": unique_lyrics_pairs,
        "metadata_coverage": {str(row.field): float(row.coverage) for row in metadata_coverage.itertuples(index=False)},
        "title_coverage": float(metadata_coverage.loc[metadata_coverage["field"] == "Title", "coverage"].iloc[0]),
        "artist_coverage": float(metadata_coverage.loc[metadata_coverage["field"] == "Artist", "coverage"].iloc[0]),
        "original_va_present": bool(_find_column(columns, ["Original_Valence"]) and _find_column(columns, ["Original_Arousal"])),
        "quadrant_present": bool(_find_column(columns, ["Quadrant"])),
        "mood_audio_va_determinism": mood_determinism,
        "mood_audio_va_checked_groups": int(sum(item["checked_groups"] for item in mood_determinism.values())),
        "mood_audio_va_deterministic_groups": int(sum(item["deterministic_groups"] for item in mood_determinism.values())),
        "audio_va_is_deterministic_given_moods": audio_va_is_deterministic_given_moods,
        "appears_track_level_after_strict_cleaning": bool(duplicate_song_count == 0 and unique_song_count == rows),
        "failures": failures,
        "passed": not failures,
    }
    (root / "raw_csv_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Raw CSV Audit Report",
        "",
        f"- Source: `{source}`",
        f"- Rows: **{rows}**",
        f"- Unique Song IDs: **{unique_song_count}**",
        f"- Duplicate Song rows: **{duplicate_song_count}**",
        f"- Audio VA complete: **{int(audio_complete.sum())}**",
        f"- Lyrics VA complete: **{int(lyrics_complete.sum())}**",
        f"- Both audio+lyrics VA complete: **{int(both_complete.sum())}** ({both_fraction:.4f})",
        f"- Unique Audio VA pairs: **{unique_audio_pairs}**",
        f"- Unique Lyrics VA pairs: **{unique_lyrics_pairs}**",
        f"- Original VA present: **{summary['original_va_present']}**",
        f"- Quadrant present: **{summary['quadrant_present']}**",
        f"- Audio VA deterministic given moods: **{audio_va_is_deterministic_given_moods}**",
        "",
        "## Fail-Fast Status",
    ]
    if failures:
        lines.extend([f"- FAIL: {failure}" for failure in failures])
    else:
        lines.append("- PASS")
    lines.extend(
        [
            "",
            "## Output Files",
            "- `raw_csv_audit_summary.json`",
            "- `raw_csv_column_summary.csv`",
            "- `raw_csv_missingness.csv`",
            "- `raw_csv_va_range_check.csv`",
            "- `raw_csv_duplicate_song_summary.csv`",
            "- `metadata_coverage.csv`",
        ]
    )
    (root / "raw_csv_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a cleaned upstream unimodal VA CSV before Dataset-L processing.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--allow_failures", choices=("true", "false"), default="false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_raw_csv(Path(args.csv), Path(args.out_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("failures") and str(args.allow_failures).lower() != "true":
        sys.exit(1)


if __name__ == "__main__":
    main()
