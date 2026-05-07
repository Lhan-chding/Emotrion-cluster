"""Entry point: rerun GMM cluster search on an existing checkpoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.pipeline.rerun import main

if __name__ == "__main__":
    main()
