"""Audit whether a completed clustering run can be claimed as a top-conference main result."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.evaluation.topconf_audit import audit_run, write_audit_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit clustering run hard gates for top-conference readiness.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing rerun_summary.json or pipeline_summary.json.")
    parser.add_argument("--out_dir", default=None, help="Directory for topconf_audit_report.md and topconf_audit_summary.json.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    result = audit_run(run_dir)
    outputs = write_audit_outputs(result, out_dir)
    print(f"[Audit] overall_ready={result['overall_ready']} failures={result['num_failures']}")
    print(f"[Audit] summary: {outputs['summary']}")
    print(f"[Audit] report: {outputs['report']}")
    if not result["overall_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
