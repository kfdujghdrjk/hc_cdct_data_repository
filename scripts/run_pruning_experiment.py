from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hc_cdct.experiment import run_pruning_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run HC-CDCT pruning robustness experiment.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pruning_experiment(args.config)
