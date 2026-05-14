from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _summary(run_dir: Path) -> Dict[str, Any]:
    for name in ("rerun_summary.json", "pipeline_summary.json"):
        payload = _read_json(run_dir / name)
        if payload:
            return payload
    return {}


def _selection(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    value = summary.get("selection_info", {})
    return value if isinstance(value, Mapping) else {}


def _metric(summary: Mapping[str, Any], key: str) -> Any:
    return _selection(summary).get(key, summary.get(key))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _copy_if_exists(source: Path, dest: Path) -> bool:
    if not source.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return True


def _region_name_map(regions: pd.DataFrame) -> Dict[int, str]:
    if regions.empty or "cluster_id" not in regions.columns:
        return {}
    return {
        int(row.cluster_id): str(getattr(row, "canonical_name", f"Cluster {int(row.cluster_id)}"))
        for row in regions.itertuples(index=False)
    }


def _representative_tracks(run_dir: Path, limit_per_cluster: int = 8) -> pd.DataFrame:
    assignments = _read_csv(run_dir / "all" / "cluster_assignments.csv")
    regions = _read_csv(run_dir / "all" / "canonical_affect_regions.csv")
    tension = _read_csv(run_dir / "all" / "tension_subtype_assignments.csv")
    if assignments.empty or regions.empty:
        return pd.DataFrame()
    names = _region_name_map(regions)
    centers = {
        int(row.cluster_id): (float(row.balanced_valence), float(row.balanced_arousal))
        for row in regions.itertuples(index=False)
        if hasattr(row, "balanced_valence") and hasattr(row, "balanced_arousal")
    }
    frame = assignments.copy()
    if not tension.empty and {"identifier", "cluster_id"}.issubset(tension.columns):
        keep = [
            column
            for column in ("identifier", "cluster_id", "tension_micro_id", "tension_subtype_label", "tension_dv", "tension_da", "tension_norm")
            if column in tension.columns
        ]
        frame = frame.merge(tension[keep], on=["identifier", "cluster_id"], how="left")
    rows: List[Dict[str, Any]] = []
    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        cid = int(cluster_id)
        center = centers.get(cid)
        if center is None or not {"balanced_valence", "balanced_arousal"}.issubset(group.columns):
            ranked = group.copy()
            ranked["distance_to_center"] = float("nan")
        else:
            coords = group[["balanced_valence", "balanced_arousal"]].to_numpy(dtype=np.float64)
            ranked = group.copy()
            ranked["distance_to_center"] = np.linalg.norm(coords - np.asarray(center, dtype=np.float64), axis=1)
            ranked = ranked.sort_values("distance_to_center", kind="stable")
        seen_artists = set()
        selected = []
        for row in ranked.to_dict(orient="records"):
            artist = str(row.get("artist", row.get("Artist", "")) or "").strip()
            if artist and artist in seen_artists and len(selected) < limit_per_cluster:
                continue
            if artist:
                seen_artists.add(artist)
            selected.append(row)
            if len(selected) >= limit_per_cluster:
                break
        for row in selected:
            rows.append(
                {
                    "cluster_id": cid,
                    "region_name": names.get(cid, f"Cluster {cid}"),
                    "song_id": row.get("identifier", ""),
                    "artist": row.get("artist", row.get("Artist", "")),
                    "title": row.get("title", row.get("Title", "")),
                    "balanced_valence": row.get("balanced_valence", float("nan")),
                    "balanced_arousal": row.get("balanced_arousal", float("nan")),
                    "distance_to_center": row.get("distance_to_center", float("nan")),
                    "tension_subtype": row.get("tension_subtype_label", ""),
                    "tension_dv": row.get("tension_dv", float("nan")),
                    "tension_da": row.get("tension_da", float("nan")),
                    "tension_norm": row.get("tension_norm", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def compile_report(
    run_dir: Path | str,
    *,
    raw_audit_dir: Optional[Path | str] = None,
    comparison_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    root = Path(run_dir).expanduser()
    raw_root = Path(raw_audit_dir).expanduser() if raw_audit_dir else None
    comparison_root = Path(comparison_dir).expanduser() if comparison_dir else root
    summary = _summary(root)
    selection = _selection(summary)
    regions = _read_csv(root / "all" / "canonical_affect_regions.csv")
    assignments = _read_csv(root / "all" / "cluster_assignments.csv")
    ablation = _read_csv(root / "ablation_report.csv")
    raw_audit = _read_json(raw_root / "raw_csv_audit_summary.json") if raw_root else {}

    copied = {
        "dataset_L_tension_substructure_report.md": _copy_if_exists(root / "all" / "tension_substructure_report.md", root / "dataset_L_tension_substructure_report.md"),
        "dataset_L_tension_substructure.csv": _copy_if_exists(root / "all" / "tension_substructure_enrichment.csv", root / "dataset_L_tension_substructure.csv"),
    }

    reps = _representative_tracks(root)
    if not reps.empty:
        reps.to_csv(root / "dataset_L_representative_tracks.csv", index=False, encoding="utf-8")
        lines = ["# Dataset-L Representative Tracks", ""]
        lines.append(_markdown_table(reps))
        (root / "dataset_L_representative_tracks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tension_assignments = _read_csv(root / "all" / "tension_subtype_assignments.csv")
    if not tension_assignments.empty:
        examples = tension_assignments.sort_values("tension_norm", ascending=False, kind="stable").head(200)
        examples.to_csv(root / "dataset_L_tension_examples.csv", index=False, encoding="utf-8")

    required = [
        root / "rerun_report.md",
        root / "rerun_summary.json",
        root / "all" / "cluster_scatter_balanced_va.png",
        root / "all" / "canonical_affect_regions.md",
        root / "all" / "tension_substructure_report.md",
        root / "ablation_report.csv",
        comparison_root / "dataset_s_l_comparison_report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        (root / "missing_outputs.md").write_text(
            "# Missing Outputs\n\n" + "\n".join(f"- `{item}`" for item in missing) + "\n",
            encoding="utf-8",
        )

    lines = [
        "# Dataset-L Final Paper-Ready Report",
        "",
        "## Dataset-L Construction",
        f"- Strict cleaned rows: **{raw_audit.get('rows', len(assignments) if not assignments.empty else 'unknown')}**",
        f"- Unique songs: **{raw_audit.get('unique_song_count', 'unknown')}**",
        f"- Complete audio+lyrics VA count: **{raw_audit.get('both_audio_lyrics_complete_count', 'unknown')}**",
        f"- Title coverage: **{raw_audit.get('title_coverage', 'unknown')}**",
        f"- Artist coverage: **{raw_audit.get('artist_coverage', 'unknown')}**",
        "",
        "## Main Result",
        f"- Selected K: **{_metric(summary, 'selected_k')}**",
        f"- Balanced region score: **{_metric(summary, 'balanced_region_score')}**",
        f"- Balance alpha: **{_metric(summary, 'balance_alpha')}**",
        f"- VA silhouette: **{_metric(summary, 'va_mean_silhouette')}**",
        f"- KNN purity@20: **{_metric(summary, 'va_knn_purity_20')}**",
        f"- Center/radius separation: **{_metric(summary, 'va_center_radius_sep')}**",
        f"- Negative silhouette fraction: **{_metric(summary, 'va_negative_silhouette_fraction')}**",
        f"- Seed ARI mean: **{_metric(summary, 'seed_ari_mean')}**",
        "",
        "## Canonical Regions",
    ]
    if not regions.empty:
        display_cols = [c for c in ("cluster_id", "canonical_name", "size", "balanced_valence", "balanced_arousal", "top_tokens") if c in regions.columns]
        lines.append(_markdown_table(regions[display_cols]))
    else:
        lines.append("Canonical region table is unavailable.")
    lines.extend(["", "## Ablation"])
    if not ablation.empty:
        display_cols = [c for c in ("config", "status", "selected_k", "claim_score", "score", "diagnostic_failed_gate_override") if c in ablation.columns]
        lines.append(_markdown_table(ablation[display_cols]))
    else:
        lines.append("Ablation report is unavailable.")
    lines.extend(
        [
            "",
            "## Dataset-S vs Dataset-L",
            f"- Comparison report: `{comparison_root / 'dataset_s_l_comparison_report.md'}`",
            "",
            "## ACL Writing Paragraph",
            (
                "For Dataset-L, we remove song-mood-expanded duplicates and construct a strict track-level subset with complete audio and lyric VA estimates. "
                "The downstream clustering module is applied without using metadata, mood labels, or original VA annotations for cluster assignment. "
                "The model learns a balanced audio-lyric VA plane by selecting the audio/lyric mixing weight according to unsupervised clusterability and stability criteria. "
                "Final affect regions are discovered only on this balanced VA plane, while residual audio-lyric disagreement is analyzed as a report-only cross-modal tension diagnostic. "
                "This design avoids the failure mode observed when signed VA differences are naively concatenated as clustering coordinates."
            ),
        ]
    )
    (root / "final_paper_ready_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"run_dir": str(root), "missing_outputs": missing, "representative_rows": int(len(reps))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile Dataset-L paper-ready report artifacts.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--raw_audit_dir", default=None)
    parser.add_argument("--comparison_dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = compile_report(
        Path(args.run_dir),
        raw_audit_dir=Path(args.raw_audit_dir) if args.raw_audit_dir else None,
        comparison_dir=Path(args.comparison_dir) if args.comparison_dir else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
