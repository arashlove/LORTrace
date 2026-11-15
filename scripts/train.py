"""CLI: fine-tune Mistral v0.3 (Unsloth LoRA) on serialized normal windows.

Requires the GPU environment (see README.md). Example:
    python scripts/train.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lortrace.benchmark.config import load_config  # noqa: E402
from lortrace.benchmark.training import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a backbone with Unsloth LoRA.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to a pipeline config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = run_training(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
