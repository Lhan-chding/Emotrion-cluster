"""Evaluate affect retrieval rankings with fixed external relevance labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.affect_retrieval_common import (
    DEFAULT_OUT_DIR,
    DEFAULT_TOP_K,
    build_annotation_pool,
    evaluate_retrieval_results as _evaluate_retrieval_results,
    load_query_config,
    parse_top_k,
    write_evaluation_outputs,
    _write_json,
    _read_csv,
)


def evaluate_retrieval_results(
    retrieval_results: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
):
    return _evaluate_retrieval_results(retrieval_results, labels, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval_results_csv",
        default=str(DEFAULT_OUT_DIR / "retrieval_results_all.csv"),
    )
    parser.add_argument(
        "--external_labels_csv",
        default=str(DEFAULT_OUT_DIR / "external_relevance_labels.csv"),
    )
    parser.add_argument(
        "--annotation_pool_csv",
        default=str(DEFAULT_OUT_DIR / "external_annotation_pool.csv"),
    )
    parser.add_argument("--query_config", default="configs/affect_retrieval_queries.yaml")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top_k", default="5,10,20")
    parser.add_argument("--skip_figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rankings = _read_csv(args.retrieval_results_csv)
    labels = _read_csv(args.external_labels_csv)
    if Path(args.annotation_pool_csv).exists():
        annotation_pool = _read_csv(args.annotation_pool_csv)
    else:
        annotation_pool = build_annotation_pool(rankings, pd.DataFrame(), retrieval_depth=max(parse_top_k(args.top_k)))
    queries = load_query_config(args.query_config)
    sanity_path = out_dir / "sanity_check_retrieval.json"
    if sanity_path.exists():
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    else:
        sanity = {
            "metrics_available": False,
            "whether_any_external_label_used_in_scoring": False,
        }
    sanity.update({
        "metrics_available": False,
        "whether_any_external_label_used_in_scoring": False,
    })
    final_sanity = write_evaluation_outputs(
        out_dir=out_dir,
        queries=queries,
        rankings=rankings,
        annotation_pool=annotation_pool,
        labels=labels,
        top_k=parse_top_k(args.top_k),
        sanity=sanity,
        make_figures=not args.skip_figures,
    )[2]
    _write_json(out_dir / "sanity_check_retrieval.json", final_sanity)


if __name__ == "__main__":
    main()
