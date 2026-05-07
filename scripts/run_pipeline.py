"""Entry point: run the full music discovery pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.pipeline.train import main

if __name__ == "__main__":
    main()
