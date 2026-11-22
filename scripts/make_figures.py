"""Build the summary figures and tables from the released scores (CPU only).

Reads results/<model>/ (metrics + val/test scores) and weights/<model>/training_metrics.json
when present, and writes PNG + PDF figures and CSV tables. No GPU or model weights needed.

Examples:
    uv run python scripts/make_figures.py
    uv run python scripts/make_figures.py --results outputs/rescore --out outputs/figures_rescore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lortrace.benchmark.report import build_all, cohort, load_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Folder with <model>/ results (default: results).")
    parser.add_argument("--weights", default="weights", help="Folder with <model>/training_metrics.json.")
    parser.add_argument("--out", default="outputs/figures", help="Output folder (default: outputs/figures).")
    args = parser.parse_args()

    results, weights, out = (ROOT / args.results, ROOT / args.weights, ROOT / args.out)
    c = cohort(load_models(results, weights))
    print(f"{c['n_models']} backbones in {args.results}/; {c['n_eval']:,} windows, "
          f"{c['n_pos']} Actual LOR ({100 * c['prevalence']:.2f}%); "
          f"Forecast LOR recall {c['forecast_recall']:.3f}")
    for path in build_all(results, out, weights):
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
