"""CLI: compute value-token NLL anomaly scores for one or more splits.

Requires the GPU environment (see README.md). Example:
    python scripts/score.py --config configs/base.yaml --split all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lortrace.benchmark.config import load_config  # noqa: E402
from lortrace.benchmark.scoring import run_scoring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score windows with LORTrace NLL.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to a pipeline config YAML")
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "test", "val_test", "all"],
        help="Which split(s) to score. 'all' scores val + test combined (val_test).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute from scratch.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from lortrace.benchmark.preprocessing import EVAL_SPLIT

    splits = [EVAL_SPLIT] if args.split == "all" else [args.split]
    for split in splits:
        path = run_scoring(cfg, split, force=args.force)
        print(f"Scored '{split}' -> {path}")


if __name__ == "__main__":
    main()
