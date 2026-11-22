"""Evaluate LORTrace anomaly scores against NSW1 LOR labels.

Ground truth is the *actual* LOR severity; the AEMO *forecast* LOR is reported
as a benchmark detector. Headline outputs are (a) anomaly recall of LOR events,
overall and per severity, and (b) the ordered LOR matrix (rows = forecast
LOR1/2/3, columns = actual LOR1/2/3). Metrics, matrices, a top-anomaly table,
and diagnostic plots are written to the evaluation artifacts directory.

The validation and test splits are scored together (``val_test``) after training.
Score quantiles (default 90th, 95th, 99th) are fitted on that combined set and
used for metrics on the same labeled windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config as cfg_mod
from .config import Config
from .preprocessing import EVAL_SPLIT, POST_TRAIN_SPLITS
from .scoring import load_scores

SEVERITY_LEVELS = (1, 2, 3)

# Ground-truth options: name -> (severity column, display label, counterpart name).
GT_OPTIONS = {
    "actual": ("actual_sev", "actual LOR", "forecast"),
    "forecast": ("forecast_sev", "forecast LOR", "actual"),
}

# Defaults (overridable per call via ``ground_truth=``).
GT_SEV_KEY = "actual_sev"
GT_LABEL = "actual LOR"


def resolve_ground_truth(name: str) -> tuple[str, str, str]:
    """Return ``(sev_key, label, counterpart_name)`` for ``actual`` or ``forecast``."""
    key = str(name).strip().lower()
    if key not in GT_OPTIONS:
        raise ValueError(f"ground_truth must be one of {sorted(GT_OPTIONS)}, got {name!r}")
    return GT_OPTIONS[key]


def _labeled(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = scores["has_label"].astype(bool)
    return {k: v[mask] for k, v in scores.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}


def _gt_sev(scores: dict[str, np.ndarray], gt_key: str | None = None) -> np.ndarray:
    return scores[gt_key or GT_SEV_KEY]


def _quantile_thresholds(
    val_scores: np.ndarray, quantiles: list[float]
) -> dict[str, float]:
    """Map quantile labels (e.g. ``q95``) to thresholds fitted on validation scores."""
    out: dict[str, float] = {}
    for q in quantiles:
        label = f"q{int(round(float(q) * 100))}"
        out[label] = float(np.quantile(val_scores, float(q)))
    return out


def _metrics_at_threshold(
    test: dict[str, np.ndarray],
    threshold: float,
    tolerance_ns: int,
    gt_key: str | None = None,
) -> dict[str, Any]:
    key = gt_key or GT_SEV_KEY
    gt = _gt_sev(test, key)
    y_test = (gt > 0).astype(int)
    prefix = "forecast" if key == "forecast_sev" else "actual"
    return {
        "threshold": threshold,
        "binary_detection": _binary_metrics(test["scores"], y_test, threshold),
        "per_severity_recall": _per_severity_recall(test["scores"], gt, threshold),
        "event_detection": _event_metrics(
            test["scores"], gt, test["end_ts"], threshold, tolerance_ns
        ),
        "anomaly_detection_matrix": _detection_matrix(
            test["scores"] >= threshold, gt, prefix=prefix
        ),
    }


def _select_threshold(cfg: Config, val: dict[str, np.ndarray]) -> float:
    """Return the middle configured quantile threshold (default reference for plots)."""
    quantiles = list(cfg.evaluation.threshold_quantiles)
    mid = quantiles[len(quantiles) // 2]
    return float(np.quantile(val["scores"], float(mid)))


def _binary_metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else (0.0 if (precision is not None and recall is not None) else None)
    )
    fpr = fp / (fp + tn) if (fp + tn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "specificity": specificity,
    }
    metrics.update(_ranking_metrics(scores, y))
    return metrics


def _ranking_metrics(scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"roc_auc": None, "average_precision": None,
                           "precision_at_k": None, "recall_at_k": None}
    if y.sum() == 0 or y.sum() == len(y):
        return out
    try:
        from sklearn import metrics as skm

        out["roc_auc"] = float(skm.roc_auc_score(y, scores))
        out["average_precision"] = float(skm.average_precision_score(y, scores))
    except Exception:
        pass

    k = int(y.sum())
    top_idx = np.argpartition(scores, -k)[-k:]
    flagged = np.zeros_like(y)
    flagged[top_idx] = 1
    tp_k = int(((flagged == 1) & (y == 1)).sum())
    out["precision_at_k"] = tp_k / k if k else None
    out["recall_at_k"] = tp_k / int(y.sum()) if y.sum() else None
    return out


def _per_severity_recall(
    scores: np.ndarray, severity: np.ndarray, threshold: float
) -> dict[str, Any]:
    pred = scores >= threshold
    out: dict[str, Any] = {}
    for level in SEVERITY_LEVELS:
        mask = severity == level
        total = int(mask.sum())
        detected = int((pred & mask).sum())
        out[f"LOR{level}"] = {
            "n": total,
            "detected": detected,
            "recall": (detected / total) if total else None,
        }
    return out


def _events_from_mask(positive: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx_inclusive) runs of True in a boolean array."""
    events: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(positive):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(positive) - 1))
    return events


def _event_metrics(
    scores: np.ndarray,
    severity: np.ndarray,
    end_ts: np.ndarray,
    threshold: float,
    tolerance_ns: int,
) -> dict[str, Any]:
    order = np.argsort(end_ts)
    scores = scores[order]
    severity = severity[order]
    ts = end_ts[order]
    pred = scores >= threshold

    true_events = _events_from_mask(severity > 0)
    pred_pos_ts = ts[pred]

    detected = 0
    for lo, hi in true_events:
        window_lo = ts[lo] - tolerance_ns
        window_hi = ts[hi] + tolerance_ns
        if np.any((pred_pos_ts >= window_lo) & (pred_pos_ts <= window_hi)):
            detected += 1

    true_pos_ts = ts[severity > 0]
    tp_pred = 0
    for t in pred_pos_ts:
        if np.any(np.abs(true_pos_ts - t) <= tolerance_ns):
            tp_pred += 1

    n_events = len(true_events)
    n_pred = int(pred.sum())
    event_recall = detected / n_events if n_events else None
    event_precision = tp_pred / n_pred if n_pred else None
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision and event_recall and (event_precision + event_recall) > 0
        else None
    )
    return {
        "n_events": n_events,
        "events_detected": detected,
        "event_recall": event_recall,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "n_predicted_positive": n_pred,
    }


def _detection_matrix(
    pred: np.ndarray, gt_sev: np.ndarray, prefix: str = "actual"
) -> dict[str, Any]:
    """Rows = detected/not-detected, cols = ground-truth severity 0..3."""
    matrix = np.zeros((2, 4), dtype=int)
    for p in (0, 1):
        for a in range(4):
            matrix[p, a] = int(((pred.astype(int) == p) & (gt_sev == a)).sum())
    return {
        "rows": ["not_flagged", "flagged"],
        "cols": [f"{prefix}_none", f"{prefix}_LOR1", f"{prefix}_LOR2", f"{prefix}_LOR3"],
        "matrix": matrix.tolist(),
    }


def _lor_benchmark(
    detector_sev: np.ndarray, gt_sev: np.ndarray
) -> dict[str, Any]:
    """Treat ``detector_sev>0`` as a detector; report recall vs ground-truth severity."""
    pred = detector_sev > 0
    y = gt_sev > 0
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    per_sev = {}
    for level in SEVERITY_LEVELS:
        mask = gt_sev == level
        total = int(mask.sum())
        detected = int((pred & mask).sum())
        per_sev[f"LOR{level}"] = {
            "n": total,
            "detected": detected,
            "recall": (detected / total) if total else None,
        }
    return {"precision": precision, "recall": recall, "per_severity_recall": per_sev}


def _forecast_benchmark(forecast_sev: np.ndarray, actual_sev: np.ndarray) -> dict[str, Any]:
    """Treat AEMO forecast LOR>0 as a detector; report recall vs actual."""
    return _lor_benchmark(forecast_sev, actual_sev)


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
def _make_plots(
    cfg: Config,
    out_dir: Path,
    test: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    thresholds: dict[str, float],
    gt_key: str | None = None,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = gt_key or GT_SEV_KEY
    gt_label = "forecast LOR" if key == "forecast_sev" else "actual LOR"
    saved: list[str] = []
    scores = test["scores"]
    gt = _gt_sev(test, key)
    y = (gt > 0).astype(int)
    ts = test["end_ts"].astype("datetime64[ns]")
    order = np.argsort(test["end_ts"])
    ref_thr = thresholds.get("q95") or next(iter(thresholds.values()))
    q_colors = {"q90": "#3498db", "q95": "#e67e22", "q99": "#c0392b"}

    # 1. Score timeline with LOR shading.
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts[order], scores[order], lw=0.5, color="steelblue", label="anomaly score")
    for label, thr in thresholds.items():
        ax.axhline(
            thr, color=q_colors.get(label, "gray"), ls="--", lw=1,
            label=f"{label} ({thr:.1f})",
        )
    lor_ts = ts[order][gt[order] > 0]
    for t in lor_ts:
        ax.axvline(t, color="orange", alpha=0.2, lw=0.5)
    ax.set_title(f"Anomaly score over time (orange = {gt_label})")
    ax.set_ylabel("value-token NLL")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p = out_dir / "score_timeline.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)

    # 2. ROC + PR curves.
    if 0 < y.sum() < len(y):
        try:
            from sklearn import metrics as skm

            fpr, tpr, _ = skm.roc_curve(y, scores)
            prec, rec, _ = skm.precision_recall_curve(y, scores)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes[0].plot(fpr, tpr, color="steelblue")
            axes[0].plot([0, 1], [0, 1], ls="--", color="gray")
            axes[0].set_title(f"ROC (AUC={skm.roc_auc_score(y, scores):.3f})")
            axes[0].set_xlabel("FPR")
            axes[0].set_ylabel("TPR")
            axes[1].plot(rec, prec, color="darkorange")
            axes[1].set_title(f"PR (AP={skm.average_precision_score(y, scores):.3f})")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            fig.tight_layout()
            p = out_dir / "roc_pr_curves.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            saved.append(p.name)
        except Exception:
            pass

    # 3. Score distribution by severity.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for level in (0, 1, 2, 3):
        vals = scores[gt == level]
        if len(vals):
            ax.hist(vals, bins=50, alpha=0.5, density=True, label=f"{gt_label}={level}")
    for label, thr in thresholds.items():
        ax.axvline(thr, color=q_colors.get(label, "gray"), ls="--", lw=1, label=label)
    ax.set_title(f"Score distribution by {gt_label} severity")
    ax.set_xlabel("value-token NLL")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "score_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)

    # 4. Threshold sweep on evaluation scores.
    v_scores = val["scores"]
    v_y = (_gt_sev(val, key) > 0).astype(int)
    if 0 < v_y.sum() < len(v_y):
        thr_grid = np.quantile(v_scores, np.linspace(0.5, 0.999, 100))
        precs, recs, f1s = [], [], []
        for thr in thr_grid:
            pred = (v_scores >= thr).astype(int)
            tp = ((pred == 1) & (v_y == 1)).sum()
            fp = ((pred == 1) & (v_y == 0)).sum()
            fn = ((pred == 0) & (v_y == 1)).sum()
            pr = tp / (tp + fp) if (tp + fp) else 0.0
            re = tp / (tp + fn) if (tp + fn) else 0.0
            precs.append(pr)
            recs.append(re)
            f1s.append(2 * pr * re / (pr + re) if (pr + re) else 0.0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(thr_grid, precs, label="precision")
        ax.plot(thr_grid, recs, label="recall")
        ax.plot(thr_grid, f1s, label="F1")
        for label, thr in thresholds.items():
            ax.axvline(thr, color=q_colors.get(label, "gray"), ls="--", lw=1, label=label)
        ax.set_title("Validation threshold sweep")
        ax.set_xlabel("threshold")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "threshold_sweep.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p.name)

    # 5. Optional feature attribution.
    if "feature_scores" in test and test["feature_scores"].size:
        manifest_features = cfg.features.per_step
        feat = test["feature_scores"]
        flagged = scores >= ref_thr
        if flagged.any() and (~flagged).any():
            anom = np.nanmean(feat[flagged], axis=0)
            norm = np.nanmean(feat[~flagged], axis=0)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            x = np.arange(len(manifest_features))
            ax.bar(x - 0.2, norm, width=0.4, label="normal")
            ax.bar(x + 0.2, anom, width=0.4, label="flagged")
            ax.set_xticks(x)
            ax.set_xticklabels(list(manifest_features), rotation=40, ha="right")
            ax.set_title("Mean per-feature NLL: flagged vs normal")
            ax.legend()
            fig.tight_layout()
            p = out_dir / "feature_attribution.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            saved.append(p.name)

    return saved


def _write_top_anomalies(out_dir: Path, test: dict[str, np.ndarray], k: int = 100) -> None:
    import csv

    scores = test["scores"]
    order = np.argsort(scores)[::-1][:k]
    ts = test["end_ts"].astype("datetime64[ns]")
    with (out_dir / "top_anomalies.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "end_time", "score", "actual_sev", "forecast_sev", "rrp_last"])
        for rank, i in enumerate(order, start=1):
            writer.writerow(
                [rank, str(ts[i]), f"{scores[i]:.4f}", int(test["actual_sev"][i]),
                 int(test["forecast_sev"][i]), f"{float(test['rrp_last'][i]):.2f}"]
            )


def build_metrics_bundle(
    eval_data: dict[str, np.ndarray],
    quantiles: list[float],
    tolerance_ns: int,
    ground_truth: str = "actual",
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the full metrics dict for one ground-truth choice.

    Thresholds default to quantiles of ``eval_data['scores']`` (same for both GTs).
    """
    gt_key, gt_label, _counterpart = resolve_ground_truth(ground_truth)
    if thresholds is None:
        thresholds = _quantile_thresholds(eval_data["scores"], quantiles)

    by_quantile: dict[str, Any] = {}
    for label, thr in thresholds.items():
        by_quantile[label] = _metrics_at_threshold(
            eval_data, thr, tolerance_ns, gt_key=gt_key
        )

    y_eval = (_gt_sev(eval_data, gt_key) > 0).astype(int)
    ranking = _ranking_metrics(eval_data["scores"], y_eval)
    ref_label = "q95" if "q95" in by_quantile else next(iter(by_quantile))
    ref = by_quantile[ref_label]

    if gt_key == "actual_sev":
        benchmark = _lor_benchmark(eval_data["forecast_sev"], eval_data["actual_sev"])
        benchmark_name = "forecast LOR"
        bench_key = "forecast_lor_benchmark"
    else:
        benchmark = _lor_benchmark(eval_data["actual_sev"], eval_data["forecast_sev"])
        benchmark_name = "actual LOR"
        bench_key = "actual_lor_benchmark"

    return {
        "ground_truth": gt_label,
        "ground_truth_key": gt_key,
        "benchmark_name": benchmark_name,
        "eval_split": EVAL_SPLIT,
        "eval_splits_combined": list(POST_TRAIN_SPLITS),
        "threshold_quantiles": list(quantiles),
        "thresholds": thresholds,
        "primary_quantile": ref_label,
        "n_eval_labeled": int(len(y_eval)),
        "n_test_labeled": int(len(y_eval)),
        "ranking_metrics": ranking,
        "by_quantile": by_quantile,
        "threshold": ref["threshold"],
        "binary_detection": {**ref["binary_detection"], **ranking},
        "per_severity_recall": ref["per_severity_recall"],
        "event_detection": ref["event_detection"],
        "anomaly_detection_matrix": ref["anomaly_detection_matrix"],
        bench_key: benchmark,
        "lor_benchmark": benchmark,
    }


def run_evaluation(
    cfg: Config, ground_truth: str = "actual", write_plots: bool = True
) -> dict[str, Any]:
    """Compute metrics/plots and write them to the evaluation directory.

    ``ground_truth`` may be ``actual``, ``forecast``, or ``both``.
    """
    out_dir = cfg_mod.evaluation_dir(cfg)
    cfg_mod.ensure_dirs(out_dir)

    eval_data = _labeled(load_scores(cfg, EVAL_SPLIT))
    quantiles = [float(q) for q in cfg.evaluation.threshold_quantiles]
    thresholds = _quantile_thresholds(eval_data["scores"], quantiles)
    tolerance_ns = int(cfg.evaluation.event_tolerance_steps) * 5 * 60 * 1_000_000_000

    gt_arg = str(ground_truth).strip().lower()
    gt_list = ["actual", "forecast"] if gt_arg == "both" else [gt_arg]

    primary: dict[str, Any] | None = None
    for gt_name in gt_list:
        bundle = build_metrics_bundle(
            eval_data,
            quantiles,
            tolerance_ns,
            ground_truth=gt_name,
            thresholds=thresholds,
        )
        (out_dir / f"metrics_{gt_name}_lor.json").write_text(
            json.dumps(bundle, indent=2), encoding="utf-8"
        )
        if primary is None or gt_name == "actual":
            primary = bundle

    assert primary is not None
    if write_plots:
        plots = _make_plots(
            cfg,
            out_dir,
            eval_data,
            eval_data,
            thresholds,
            gt_key=primary["ground_truth_key"],
        )
        primary["plots"] = plots
        _write_top_anomalies(out_dir, eval_data)

    (out_dir / "metrics.json").write_text(json.dumps(primary, indent=2), encoding="utf-8")
    return primary
