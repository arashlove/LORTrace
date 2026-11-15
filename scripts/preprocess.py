"""CLI: preprocess the AEMO CSV + LOR labels into windowed arrays.

Example:
    python scripts/preprocess.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lortrace.benchmark.config import load_config  # noqa: E402
from lortrace.benchmark.preprocessing import run_preprocessing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AEMO data for LORTrace.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to a pipeline config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = run_preprocessing(cfg)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
