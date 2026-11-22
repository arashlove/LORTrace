"""CLI: evaluate LORTrace scores against NSW1 LOR labels and write reports.

Example:
    python scripts/evaluate.py --config configs/base.yaml
    python scripts/evaluate.py --config configs/base.yaml --ground-truth both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lortrace.benchmark.config import load_config  # noqa: E402
from lortrace.benchmark.evaluation import run_evaluation  # noqa: E402


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_results(results: dict) -> None:
    print("=" * 60)
    gt = results.get("ground_truth", "actual LOR")
    eval_split = results.get("eval_split", "val_test")
    print(f"LORTrace evaluation ({eval_split} = val + test, {gt} = ground truth)")
    print("=" * 60)

    by_q = results.get("by_quantile")
    if by_q:
        print("Thresholds fitted on evaluation scores (quantile -> value):")
        for label, thr in results["thresholds"].items():
            print(f"  {label}: {thr:.4f}")
        print("-" * 60)
        for label in sorted(by_q.keys()):
            binary = by_q[label]["binary_detection"]
            event = by_q[label]["event_detection"]
            print(f"[{label}] threshold={binary['threshold']:.4f}")
            print(f"  ROC-AUC           : {_fmt(binary.get('roc_auc'))}")
            print(f"  Precision/Recall  : {_fmt(binary.get('precision'))} / {_fmt(binary.get('recall'))}")
            print(f"  F1 / FPR          : {_fmt(binary.get('f1'))} / {_fmt(binary.get('false_positive_rate'))}")
            print(f"  Event recall      : {_fmt(event.get('event_recall'))}")
            per_sev = by_q[label]["per_severity_recall"]
            for level, stats in per_sev.items():
                print(f"    {level}: recall={_fmt(stats['recall'])} ({stats['detected']}/{stats['n']})")
        print("-" * 60)
    else:
        binary = results["binary_detection"]
        print(f"threshold           : {results['threshold']:.4f}")
        print(f"ROC-AUC             : {_fmt(binary['roc_auc'])}")
        print(f"Precision / Recall  : {_fmt(binary['precision'])} / {_fmt(binary['recall'])}")
        print(f"F1                  : {_fmt(binary['f1'])}")
        print("-" * 60)

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LORTrace against LOR labels.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to a pipeline config YAML")
    parser.add_argument(
        "--ground-truth",
        default="both",
        choices=["actual", "forecast", "both"],
        help="Ground truth for metrics: actual LOR, forecast LOR, or both. Default: both.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run_evaluation(cfg, ground_truth=args.ground_truth)
    _print_results(results)
    print(f"Full metrics + plots written under: {cfg.artifacts.evaluation_dir}")
    if args.ground_truth == "both":
        print("Also wrote metrics_actual_lor.json and metrics_forecast_lor.json")


if __name__ == "__main__":
    main()
