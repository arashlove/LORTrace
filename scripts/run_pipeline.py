"""Run the full LORTrace pipeline: preprocess -> train -> score -> evaluate.

Examples:
    python scripts/run_pipeline.py all
    python scripts/run_pipeline.py preprocess
    python scripts/run_pipeline.py score --force
    python scripts/run_pipeline.py --smoke-test        # CPU-only plumbing check (no GPU/LLM)

The train and score stages require the GPU environment described in README.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lortrace.benchmark import config as cfg_mod  # noqa: E402
from lortrace.benchmark.config import load_config  # noqa: E402

STAGES = ("preprocess", "train", "score", "evaluate", "all")


def _stage_preprocess(cfg) -> None:
    from lortrace.benchmark.preprocessing import run_preprocessing

    report = run_preprocessing(cfg)
    print(f"[preprocess] windows: {report.split_window_counts}")


def _stage_train(cfg) -> None:
    from lortrace.benchmark.training import run_training

    metrics = run_training(cfg)
    print(f"[train] {metrics}")


def _stage_score(cfg, force: bool) -> None:
    from lortrace.benchmark.preprocessing import EVAL_SPLIT
    from lortrace.benchmark.scoring import run_scoring

    path = run_scoring(cfg, EVAL_SPLIT, force=force)
    print(f"[score] {EVAL_SPLIT} -> {path}")


def _run_stage_subprocess(script: str, config: str, extra: list[str] | None = None) -> None:
    """Run a GPU stage in a fresh Python process so CUDA memory is fully freed.

    Training leaves the model and optimizer state resident in the process;
    scoring then cannot fit the 4-bit model in 6 GB of VRAM. Separate processes
    guarantee each stage starts with an empty GPU.
    """
    cmd = [sys.executable, str(Path(__file__).resolve().parent / script), "--config", config]
    cmd += extra or []
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Stage '{script}' failed with exit code {result.returncode}")


def _stage_evaluate(cfg) -> None:
    from lortrace.benchmark.evaluation import run_evaluation

    results = run_evaluation(cfg)
    binary = results["binary_detection"]
    print(
        f"[evaluate] ROC-AUC={binary['roc_auc']} recall={binary['recall']} "
        f"-> {cfg.artifacts.evaluation_dir}"
    )


def _run_smoke_test(cfg) -> None:
    """Fabricate small synthetic scores and run evaluation on CPU only.

    This validates the evaluation/plotting code paths without a GPU or the LLM.
    """
    from lortrace.benchmark.preprocessing import EVAL_SPLIT

    rng = np.random.default_rng(cfg.seed)
    scores_dir = cfg_mod.scores_dir(cfg)
    cfg_mod.ensure_dirs(scores_dir)

    n = 2000
    base = np.datetime64("2023-01-01T00:00:00")
    end_ts = (base + np.arange(n) * np.timedelta64(1, "h")).astype("datetime64[ns]").astype(np.int64)
    actual = np.zeros(n, dtype=np.int8)
    anomaly_idx = rng.choice(n, size=40, replace=False)
    actual[anomaly_idx[:30]] = 1
    actual[anomaly_idx[30:38]] = 2
    actual[anomaly_idx[38:]] = 3
    forecast = np.clip(actual + rng.integers(-1, 2, size=n), 0, 3).astype(np.int8)
    scores = rng.normal(0, 1, size=n) + (actual > 0) * 3.0
    np.savez_compressed(
        scores_dir / f"{EVAL_SPLIT}_scores.npz",
        scores=scores.astype(np.float64),
        end_ts=end_ts,
        actual_sev=actual,
        forecast_sev=forecast,
        has_label=np.ones(n, dtype=bool),
        rrp_last=rng.normal(80, 200, size=n).astype(np.float32),
    )

    from lortrace.benchmark.evaluation import run_evaluation

    results = run_evaluation(cfg)
    print("[smoke-test] evaluation ran successfully.")
    print(f"[smoke-test] synthetic ROC-AUC={results['binary_detection']['roc_auc']}")
    print(f"[smoke-test] artifacts under {cfg.artifacts.evaluation_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LORTrace time-series pipeline runner.")
    parser.add_argument("stage", nargs="?", default="all", choices=STAGES)
    parser.add_argument("--config", default="configs/base.yaml", help="Path to a pipeline config YAML")
    parser.add_argument("--force", action="store_true", help="Recompute scores from scratch.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a CPU-only synthetic evaluation to check the plumbing.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.smoke_test:
        _run_smoke_test(cfg)
        return

    if args.stage in ("preprocess", "all"):
        _stage_preprocess(cfg)
    if args.stage == "train":
        _stage_train(cfg)
    if args.stage == "score":
        _stage_score(cfg, args.force)
    if args.stage == "all":
        # GPU stages run in separate processes so training's CUDA memory is
        # fully released before scoring loads the model again.
        _run_stage_subprocess("train.py", args.config)
        _run_stage_subprocess("score.py", args.config, ["--split", "all"] + (["--force"] if args.force else []))
    if args.stage in ("evaluate", "all"):
        _stage_evaluate(cfg)


if __name__ == "__main__":
    main()
