"""Run Dataset-S v20.3 affect-aware retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.affect_retrieval_common import (
    add_common_run_args,
    parse_top_k,
    run_affect_retrieval_eval as _run_affect_retrieval_eval,
)


def run_affect_retrieval_eval(
    *,
    cluster_csv: Path | str | None = None,
    tension_csv: Path | str | None = None,
    interpretation_csv: Path | str | None = None,
    external_evidence_csv: Path | str | None = None,
    out_dir: Path | str = "outputs/affect_retrieval_eval",
    query_config: Path | str | None = "configs/affect_retrieval_queries.yaml",
    root: Path | str = ".",
    top_k: Sequence[int] = (5, 10, 20),
    retrieval_depth: int | None = None,
    make_figures: bool = True,
    mirror_report_path: Path | str | None = "reports/affect_retrieval_report.md",
) -> dict[str, Any]:
    return _run_affect_retrieval_eval(
        cluster_csv=cluster_csv,
        tension_csv=tension_csv,
        interpretation_csv=interpretation_csv,
        external_evidence_csv=external_evidence_csv,
        out_dir=out_dir,
        query_config=query_config,
        root=root,
        top_k=top_k,
        retrieval_depth=retrieval_depth,
        make_figures=make_figures,
        mirror_report_path=mirror_report_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser)
    args = parser.parse_args()
    sanity = run_affect_retrieval_eval(
        cluster_csv=args.cluster_csv,
        tension_csv=args.tension_csv,
        interpretation_csv=args.interpretation_csv,
        external_evidence_csv=args.external_evidence_csv,
        out_dir=args.out_dir,
        query_config=args.query_config,
        root=args.root,
        top_k=parse_top_k(args.top_k),
        retrieval_depth=args.retrieval_depth,
        make_figures=not args.skip_figures,
    )
    print(json.dumps(sanity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
