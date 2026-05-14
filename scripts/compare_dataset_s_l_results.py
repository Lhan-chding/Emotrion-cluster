from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(run_dir: Path) -> Dict[str, Any]:
    for name in ("rerun_summary.json", "pipeline_summary.json"):
        payload = _read_json(run_dir / name)
        if payload:
            return payload
    return {}


def _canonical(run_dir: Path) -> pd.DataFrame:
    for path in (run_dir / "all" / "canonical_affect_regions.csv", run_dir / "canonical_affect_regions.csv"):
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _catalog(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "all" / "cluster_catalog.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _tension(run_dir: Path) -> pd.DataFrame:
    for path in (run_dir / "all" / "tension_substructure_enrichment.csv", run_dir / "tension_substructure_enrichment.csv"):
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _metric(summary: Mapping[str, Any], key: str) -> Any:
    selection = summary.get("selection_info", {}) if isinstance(summary.get("selection_info", {}), Mapping) else {}
    return selection.get(key, summary.get(key))


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


def _centers(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty or not {"balanced_valence", "balanced_arousal"}.issubset(frame.columns):
        return np.zeros((0, 2), dtype=np.float64)
    return frame[["balanced_valence", "balanced_arousal"]].to_numpy(dtype=np.float64)


def _align_regions(s_frame: pd.DataFrame, l_frame: pd.DataFrame) -> pd.DataFrame:
    s_centers = _centers(s_frame)
    l_centers = _centers(l_frame)
    if s_centers.size == 0 or l_centers.size == 0:
        return pd.DataFrame()
    distances = np.linalg.norm(l_centers[:, None, :] - s_centers[None, :, :], axis=2)
    l_idx, s_idx = linear_sum_assignment(distances)
    rows: List[Dict[str, Any]] = []
    for li, si in zip(l_idx.tolist(), s_idx.tolist()):
        l_row = l_frame.iloc[int(li)]
        s_row = s_frame.iloc[int(si)]
        rows.append(
            {
                "dataset_l_cluster_id": int(l_row.get("cluster_id", li)),
                "dataset_l_name": str(l_row.get("canonical_name", "")),
                "dataset_l_size": int(l_row.get("size", 0)),
                "dataset_l_balanced_valence": float(l_row.get("balanced_valence", float("nan"))),
                "dataset_l_balanced_arousal": float(l_row.get("balanced_arousal", float("nan"))),
                "matched_dataset_s_cluster_id": int(s_row.get("cluster_id", si)),
                "dataset_s_name": str(s_row.get("canonical_name", "")),
                "dataset_s_balanced_valence": float(s_row.get("balanced_valence", float("nan"))),
                "dataset_s_balanced_arousal": float(s_row.get("balanced_arousal", float("nan"))),
                "center_distance": float(distances[int(li), int(si)]),
            }
        )
    matched_l = set(l_idx.tolist())
    for li in range(len(l_frame)):
        if li in matched_l:
            continue
        l_row = l_frame.iloc[int(li)]
        nearest = int(np.argmin(distances[int(li)]))
        rows.append(
            {
                "dataset_l_cluster_id": int(l_row.get("cluster_id", li)),
                "dataset_l_name": str(l_row.get("canonical_name", "")),
                "dataset_l_size": int(l_row.get("size", 0)),
                "dataset_l_balanced_valence": float(l_row.get("balanced_valence", float("nan"))),
                "dataset_l_balanced_arousal": float(l_row.get("balanced_arousal", float("nan"))),
                "matched_dataset_s_cluster_id": int(s_frame.iloc[nearest].get("cluster_id", nearest)),
                "dataset_s_name": str(s_frame.iloc[nearest].get("canonical_name", "")),
                "dataset_s_balanced_valence": float(s_frame.iloc[nearest].get("balanced_valence", float("nan"))),
                "dataset_s_balanced_arousal": float(s_frame.iloc[nearest].get("balanced_arousal", float("nan"))),
                "center_distance": float(distances[int(li), nearest]),
                "extra_dataset_l_region": True,
            }
        )
    return pd.DataFrame(rows)


def compare_runs(dataset_s_run: Path | str, dataset_l_run: Path | str, out_dir: Path | str) -> Dict[str, Any]:
    s_dir = Path(dataset_s_run).expanduser()
    l_dir = Path(dataset_l_run).expanduser()
    root = Path(out_dir).expanduser()
    fig_dir = root / "dataset_s_l_figures"
    root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    s_summary = _summary(s_dir)
    l_summary = _summary(l_dir)
    s_regions = _canonical(s_dir)
    l_regions = _canonical(l_dir)
    alignment = _align_regions(s_regions, l_regions)
    alignment.to_csv(root / "dataset_s_l_region_alignment.csv", index=False, encoding="utf-8")

    alpha_rows = [
        {"dataset": "Dataset-S", "selected_k": _metric(s_summary, "selected_k"), "alpha": _metric(s_summary, "balance_alpha"), "score": _metric(s_summary, "balanced_region_score")},
        {"dataset": "Dataset-L", "selected_k": _metric(l_summary, "selected_k"), "alpha": _metric(l_summary, "balance_alpha"), "score": _metric(l_summary, "balanced_region_score")},
    ]
    pd.DataFrame(alpha_rows).to_csv(root / "dataset_s_l_alpha_comparison.csv", index=False, encoding="utf-8")

    tension_rows = []
    for label, run_dir in (("Dataset-S", s_dir), ("Dataset-L", l_dir)):
        frame = _tension(run_dir)
        if frame.empty:
            tension_rows.append({"dataset": label, "rows": 0})
        else:
            row: Dict[str, Any] = {"dataset": label, "rows": int(len(frame))}
            for column in ("size", "mean_tension_norm", "mean_tension_dv", "mean_tension_da"):
                if column in frame.columns:
                    row[f"{column}_mean"] = float(pd.to_numeric(frame[column], errors="coerce").mean())
            tension_rows.append(row)
    pd.DataFrame(tension_rows).to_csv(root / "dataset_s_l_tension_comparison.csv", index=False, encoding="utf-8")

    metadata_rows = []
    for label, run_dir in (("Dataset-S", s_dir), ("Dataset-L", l_dir)):
        catalog = _catalog(run_dir)
        if catalog.empty:
            metadata_rows.append({"dataset": label, "rows": 0})
        else:
            for row in catalog.to_dict(orient="records"):
                metadata_rows.append({"dataset": label, **row})
    pd.DataFrame(metadata_rows).to_csv(root / "dataset_s_l_metadata_enrichment_comparison.csv", index=False, encoding="utf-8")

    report_lines = [
        "# Dataset-S vs Dataset-L Comparison",
        "",
        f"- Dataset-S run: `{s_dir}`",
        f"- Dataset-L run: `{l_dir}`",
        f"- Dataset-S selected K: **{_metric(s_summary, 'selected_k')}**",
        f"- Dataset-L selected K: **{_metric(l_summary, 'selected_k')}**",
        f"- Dataset-S alpha: **{_metric(s_summary, 'balance_alpha')}**",
        f"- Dataset-L alpha: **{_metric(l_summary, 'balance_alpha')}**",
        "",
        "## Region Alignment",
    ]
    if alignment.empty:
        report_lines.append("Region alignment was unavailable because canonical region files were missing.")
    else:
        report_lines.append(_markdown_table(alignment))
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "Dataset-L is compared to Dataset-S by matching balanced VA region centers. If Dataset-L selects more regions, unmatched or extra regions should be treated as refinements rather than forcing Dataset-L into four Dataset-S names.",
        ]
    )
    (root / "dataset_s_l_comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "dataset_s_run": str(s_dir),
        "dataset_l_run": str(l_dir),
        "dataset_s_selected_k": _metric(s_summary, "selected_k"),
        "dataset_l_selected_k": _metric(l_summary, "selected_k"),
        "alignment_rows": int(len(alignment)),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Dataset-S and Dataset-L V20.3 clustering outputs.")
    parser.add_argument("--dataset_s_run", required=True)
    parser.add_argument("--dataset_l_run", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = compare_runs(Path(args.dataset_s_run), Path(args.dataset_l_run), Path(args.out_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
