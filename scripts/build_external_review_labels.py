"""Build post-hoc external review relevance labels for affect retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.affect_retrieval_common import (
    DEFAULT_OUT_DIR,
    build_external_review_labels as _build_external_review_labels,
    build_source_search_queries,
    load_query_config,
    _read_csv,
)


def build_external_review_labels(annotation_pool, external_evidence=None, queries=None):
    return _build_external_review_labels(annotation_pool, external_evidence, queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation_pool_csv",
        default=str(DEFAULT_OUT_DIR / "external_annotation_pool.csv"),
    )
    parser.add_argument("--external_evidence_csv", default=None)
    parser.add_argument("--query_config", default="configs/affect_retrieval_queries.yaml")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    annotation_pool = _read_csv(args.annotation_pool_csv)
    external_evidence = _read_csv(args.external_evidence_csv) if args.external_evidence_csv else None
    queries = load_query_config(args.query_config)
    labels = build_external_review_labels(annotation_pool, external_evidence, queries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "external_relevance_labels.csv", index=False, encoding="utf-8-sig")
    source_queries = build_source_search_queries(annotation_pool)
    source_queries.to_csv(out_dir / "source_search_queries.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
