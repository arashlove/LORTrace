"""Summary tables and figures across backbones (CPU only).

Everything is computed from the per-model evaluation outputs in a results folder
(``results/<model>/metrics_actual_lor.json`` and ``val_test_scores.npz``) and,
when present, ``weights/<model>/training_metrics.json``. No GPU or model weights
are needed.

Used by ``scripts/make_figures.py`` (PNG/PDF + CSV) and ``report/results.qmd``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUANTILES = ("q90", "q95", "q99")
Q_FLOAT = {"q90": 0.90, "q95": 0.95, "q99": 0.99}
SEVERITIES = ("LOR1", "LOR2", "LOR3")

# Display order (smallest -> largest nominal size), label and nominal size.
BACKBONES: dict[str, tuple[str, str]] = {
    "SmolLM2-135M": ("SmolLM2 135M", "135M"),
    "gemma-3-270m": ("Gemma 3 270M", "270M"),
    "SmolLM2-360M": ("SmolLM2 360M", "360M"),
    "Qwen2.5-0.5B": ("Qwen2.5 0.5B", "0.5B"),
    "gemma-3-1b-base": ("Gemma 3 1B (base)", "1B"),
    "Qwen2.5-1.5B": ("Qwen2.5 1.5B", "1.5B"),
    "SmolLM-1.7B": ("SmolLM 1.7B", "1.7B"),
    "SmolLM2-1.7B": ("SmolLM2 1.7B", "1.7B"),
    "gemma-2-2b": ("Gemma 2 2B IT", "2B"),
    "gemma-2-2b-base": ("Gemma 2 2B (base)", "2B"),
    "mistral-7b-v0.3": ("Mistral 7B v0.3", "7B"),
    "Cosmos-Reason1-7B": ("Cosmos Reason1 7B", "7B"),
    "Qwen2.5-7B": ("Qwen2.5 7B", "7B"),
    "Qwen2.5-VL-7B-Instruct": ("Qwen2.5-VL 7B Instruct", "7B"),
    "Llama-3.1-8B": ("Llama 3.1 8B", "8B"),
}

# Colours: one series colour for model bars, black for the Forecast LOR reference,
# grey/orange for non-LOR vs Actual LOR distributions.
C_MODEL = "#2a78d6"
C_FORECAST = "#111111"
C_NONLOR = "#9aa5b1"
C_LOR = "#eb6834"
C_TEXT = "#333333"

STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
}


# ----------------------------------------------------------------------------- loading


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _first(folder: Path, names: list[str]) -> Path | None:
    return next((folder / n for n in names if (folder / n).is_file()), None)


def load_models(results_dir: Path, weights_dir: Path | None = None) -> list[dict[str, Any]]:
    """One entry per backbone folder holding ``metrics_actual_lor.json``, in display order."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results folder not found: {results_dir}")
    models = []
    for child in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        mpath = _first(child, ["metrics_actual_lor.json", "evaluation/metrics_actual_lor.json"])
        metrics = _load_json(mpath) if mpath else None
        if metrics is None:
            continue
        tpath = None
        if weights_dir is not None:
            tpath = _first(Path(weights_dir) / child.name, ["training_metrics.json"])
        tpath = tpath or _first(child, ["training_metrics.json", "model/training_metrics.json"])
        label, size = BACKBONES.get(child.name, (child.name, "?"))
        models.append(
            {
                "folder": child.name,
                "label": label,
                "size": size,
                "metrics": metrics,
                "train": (_load_json(tpath) if tpath else None) or {},
                "scores_path": _first(child, ["val_test_scores.npz", "scores/val_test_scores.npz"]),
            }
        )
    if not models:
        raise FileNotFoundError(f"no metrics_actual_lor.json under {results_dir}")
    order = {k: i for i, k in enumerate(BACKBONES)}
    models.sort(key=lambda m: (order.get(m["folder"], len(order)), m["folder"]))
    return models


# ----------------------------------------------------------------------------- metric access


def _bd(m: dict[str, Any], q: str) -> dict[str, Any]:
    return ((m["metrics"].get("by_quantile") or {}).get(q) or {}).get("binary_detection") or {}


def _num(v: Any) -> float:
    return float(v) if v is not None else float("nan")


def roc_auc(m: dict[str, Any]) -> float:
    return _num((m["metrics"].get("ranking_metrics") or {}).get("roc_auc"))


def avg_precision(m: dict[str, Any]) -> float:
    return _num((m["metrics"].get("ranking_metrics") or {}).get("average_precision"))


def at_q(m: dict[str, Any], q: str, key: str) -> float:
    return _num(_bd(m, q).get(key))


def sev_recall(m: dict[str, Any], q: str, sev: str) -> float:
    by_q = (m["metrics"].get("by_quantile") or {}).get(q) or {}
    return _num(((by_q.get("per_severity_recall") or {}).get(sev) or {}).get("recall"))


def forecast_baseline(models: list[dict[str, Any]]) -> dict[str, Any]:
    for m in models:
        bench = m["metrics"].get("lor_benchmark") or m["metrics"].get("forecast_lor_benchmark")
        if bench:
            return bench
    return {}


def forecast_recall(models: list[dict[str, Any]]) -> float:
    return _num(forecast_baseline(models).get("recall"))


def forecast_sev_recall(models: list[dict[str, Any]], sev: str) -> float:
    per = forecast_baseline(models).get("per_severity_recall") or {}
    return _num((per.get(sev) or {}).get("recall"))


def cohort(models: list[dict[str, Any]]) -> dict[str, Any]:
    m0 = models[0]
    n_eval = int(m0["metrics"].get("n_eval_labeled") or 0)
    n_pos = int(_bd(m0, "q95").get("n_positive") or 0)
    return {"n_models": len(models), "n_eval": n_eval, "n_pos": n_pos,
            "prevalence": n_pos / max(n_eval, 1), "forecast_recall": forecast_recall(models)}


def has_severity(models: list[dict[str, Any]], sev: str) -> bool:
    """True if any evaluation window has this Actual LOR severity."""
    m0 = models[0]
    by_q = (m0["metrics"].get("by_quantile") or {}).get("q90") or {}
    return int(((by_q.get("per_severity_recall") or {}).get(sev) or {}).get("n") or 0) > 0


# ----------------------------------------------------------------------------- tables


def table_ranking(models: list[dict[str, Any]]) -> pd.DataFrame:
    """ROC-AUC, AP, recall at q90/q95/q99 and the q90 gap to Forecast LOR."""
    base = forecast_recall(models)
    rows = [
        {
            "Backbone": m["label"],
            "Size": m["size"],
            "ROC-AUC": roc_auc(m),
            "AP": avg_precision(m),
            **{f"Recall {q}": at_q(m, q, "recall") for q in QUANTILES},
            "Delta Forecast q90": at_q(m, "q90", "recall") - base,
        }
        for m in models
    ]
    rows.append({"Backbone": "Forecast LOR baseline", "Recall q90": base, "Delta Forecast q90": 0.0})
    return pd.DataFrame(rows)


def table_precision_f1(models: list[dict[str, Any]]) -> pd.DataFrame:
    """Precision, F1 and false-positive rate at each quantile."""
    rows = []
    for m in models:
        row: dict[str, Any] = {"Backbone": m["label"]}
        for q in QUANTILES:
            row[f"Precision {q}"] = at_q(m, q, "precision")
            row[f"F1 {q}"] = at_q(m, q, "f1")
            row[f"FPR {q}"] = at_q(m, q, "false_positive_rate")
        rows.append(row)
    return pd.DataFrame(rows)


def table_severity(models: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-severity Actual LOR recall at q90 and q95 (NaN = no windows of that severity)."""
    rows = []
    for m in models:
        row: dict[str, Any] = {"Backbone": m["label"]}
        for q in ("q90", "q95"):
            for sev in SEVERITIES:
                row[f"{sev} {q}"] = sev_recall(m, q, sev) if has_severity(models, sev) else np.nan
        rows.append(row)
    row = {"Backbone": "Forecast LOR baseline"}
    for sev in SEVERITIES:
        row[f"{sev} q90"] = forecast_sev_recall(models, sev) if has_severity(models, sev) else np.nan
    rows.append(row)
    return pd.DataFrame(rows)


def table_training(models: list[dict[str, Any]]) -> pd.DataFrame:
    """Fine-tuning summary from training_metrics.json, where available."""
    rows = []
    for m in models:
        tr = m["train"]
        rt = tr.get("train_runtime_sec")
        rows.append(
            {
                "Backbone": m["label"],
                "Base checkpoint": tr.get("model_name", ""),
                "Train windows": tr.get("n_training_windows"),
                "Train loss": _num(tr.get("train_loss")),
                "Fine-tune time (min)": _num(rt) / 60.0 if rt is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures


def _hbar_axis(ax, labels: list[str]) -> np.ndarray:
    y = np.arange(len(labels))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if not ax.yaxis_inverted():  # smallest model on top (shared axes: invert once)
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    return y


def _value_labels(ax, y: np.ndarray, vals: np.ndarray, pad: float, size: float = 7) -> None:
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(v + pad, yi, f"{v:.3f}", va="center", fontsize=size, color=C_TEXT, zorder=5,
                    bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.85})


def _ref_line(ax, x: float, label: str | None) -> None:
    if np.isfinite(x):
        ax.axvline(x, color=C_FORECAST, ls="--", lw=1.2, label=label, zorder=3)


def _place_labels(fig, ax, x: np.ndarray, y: np.ndarray, labels: list[str]) -> None:
    """Label points to the right; nudge labels down until they no longer overlap."""
    fig.canvas.draw()
    to_px = ax.transData.transform
    from_px = ax.transData.inverted().transform
    placed: list[tuple[float, float, float, float]] = []
    h = 11.0
    for i in np.argsort(-y):
        px, py = to_px((x[i], y[i]))
        w = 4.3 * len(labels[i]) + 4
        lx, ly = px + 7, py - h / 2
        while any(lx < bx + bw and bx < lx + w and ly < by + bh and by < ly + h for bx, by, bw, bh in placed):
            ly -= 2
        placed.append((lx, ly, w, h))
        tx, ty = from_px((lx, ly + h / 2))
        moved = abs((ly + h / 2) - py) > 3
        ax.annotate(labels[i], (x[i], y[i]), xytext=(tx, ty), textcoords="data", va="center", ha="left",
                    fontsize=7, color=C_TEXT, zorder=5,
                    bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.85},
                    arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.6} if moved else None)


def fig_roc_auc(models: list[dict[str, Any]]):
    """ROC-AUC per backbone, with chance (0.5)."""
    with plt.rc_context(STYLE):
        labels = [m["label"] for m in models]
        vals = np.array([roc_auc(m) for m in models])
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(models) + 1.0))
        y = _hbar_axis(ax, labels)
        ax.barh(y, vals, height=0.65, color=C_MODEL)
        ax.axvline(0.5, color="#777777", ls=":", lw=1.2, label="Chance (0.5)")
        _value_labels(ax, y, vals, 0.004, 7.5)
        ax.set_xlim(0.45, max(0.75, np.nanmax(vals) + 0.03))
        ax.set_xlabel("ROC-AUC vs Actual LOR")
        ax.legend(loc="lower right")
        fig.tight_layout()
    return fig


def fig_recall_quantiles(models: list[dict[str, Any]]):
    """Recall at q90/q95/q99, one panel per quantile, with Forecast LOR recall."""
    base = forecast_recall(models)
    with plt.rc_context(STYLE):
        labels = [m["label"] for m in models]
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 0.32 * len(models) + 1.2), sharey=True)
        xmax = max(np.nanmax([at_q(m, "q90", "recall") for m in models]), base) * 1.18
        for i, (ax, q) in enumerate(zip(axes, QUANTILES)):
            vals = np.array([at_q(m, q, "recall") for m in models])
            y = _hbar_axis(ax, labels)
            ax.barh(y, vals, height=0.65, color=C_MODEL)
            _value_labels(ax, y, vals, xmax * 0.01)
            _ref_line(ax, base, f"Forecast LOR ({base:.3f})" if i == 0 else None)
            ax.set_xlim(0, xmax)
            ax.set_title(f"{q} (top {100 * (1 - Q_FLOAT[q]):.0f}% of windows flagged)")
            ax.set_xlabel("Recall (Actual LOR)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=1,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_precision_recall_q90(models: list[dict[str, Any]]):
    """Precision vs recall at q90, one labelled point per backbone."""
    base = forecast_recall(models)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        r = np.array([at_q(m, "q90", "recall") for m in models])
        p = np.array([at_q(m, "q90", "precision") for m in models])
        ax.scatter(r, p, s=36, color=C_MODEL, edgecolor="white", linewidth=1.5, zorder=4)
        _ref_line(ax, base, f"Forecast LOR recall ({base:.3f})")
        ax.set_xlabel("Recall @ q90")
        ax.set_ylabel("Precision @ q90")
        pad = max((np.nanmax(r) - np.nanmin(r)) * 0.15, 0.02)
        ax.set_xlim(np.nanmin(r) - pad, max(np.nanmax(r), base) + pad * 1.6)
        ax.legend(loc="upper left")
        _place_labels(fig, ax, r, p, [m["label"] for m in models])
        fig.tight_layout()
    return fig


def fig_severity_q90(models: list[dict[str, Any]]):
    """Per-severity recall at q90 (one panel per severity present), with Forecast LOR."""
    sevs = [s for s in SEVERITIES if has_severity(models, s)]
    with plt.rc_context(STYLE):
        labels = [m["label"] for m in models]
        fig, axes = plt.subplots(1, len(sevs), figsize=(3.3 * len(sevs) + 1.6, 0.32 * len(models) + 1.2),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        xmax = max(
            np.nanmax([sev_recall(m, "q90", s) for m in models for s in sevs]),
            np.nanmax([forecast_sev_recall(models, s) for s in sevs]),
        ) * 1.18
        for ax, sev in zip(axes, sevs):
            vals = np.array([sev_recall(m, "q90", sev) for m in models])
            n = int(((models[0]["metrics"]["by_quantile"]["q90"].get("per_severity_recall") or {})
                     .get(sev) or {}).get("n") or 0)
            y = _hbar_axis(ax, labels)
            ax.barh(y, vals, height=0.65, color=C_MODEL)
            _value_labels(ax, y, vals, xmax * 0.01)
            fb = forecast_sev_recall(models, sev)
            _ref_line(ax, fb, None)
            ax.text(fb, -0.9, f"Forecast LOR {fb:.3f}", ha="center", va="bottom", fontsize=7,
                    color=C_FORECAST)
            ax.set_xlim(0, xmax)
            ax.set_title(f"{sev} (n = {n} windows)", pad=14)
            ax.set_xlabel("Recall @ q90")
        missing = [s for s in SEVERITIES if s not in sevs]
        if missing:
            fig.text(0.99, 0.005, f"No Actual {', '.join(missing)} windows in the evaluation period.",
                     ha="right", va="bottom", fontsize=7, color="#666666")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_score_distributions(models: list[dict[str, Any]], ncols: int = 3):
    """Per-backbone score histograms (non-LOR vs Actual LOR) with q90/q95/q99 thresholds."""
    usable = [m for m in models if m["scores_path"] is not None]
    nrows = int(np.ceil(len(usable) / ncols))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for i, m in enumerate(usable):
            ax = axes[i // ncols][i % ncols]
            ax.set_visible(True)
            z = np.load(m["scores_path"])
            scores = np.asarray(z["scores"], dtype=np.float64)
            sev = np.asarray(z["actual_sev"], dtype=np.int16)
            has = np.asarray(z["has_label"], dtype=bool) if "has_label" in z.files else np.ones(len(scores), bool)
            scores, sev = scores[has], sev[has]
            lor = sev > 0
            lo, hi = np.quantile(scores, [0.001, 0.999])
            bins = np.linspace(lo, hi, 60)
            ax.hist(scores[~lor], bins=bins, density=True, color=C_NONLOR, alpha=0.8, label="Non-LOR")
            ax.hist(scores[lor], bins=bins, density=True, color=C_LOR, alpha=0.65, label="Actual LOR")
            for q, ls in zip(QUANTILES, ("-", "--", ":")):
                ax.axvline(np.quantile(scores, Q_FLOAT[q]), color=C_FORECAST, ls=ls, lw=1.0, label=q)
            ax.set_title(m["label"])
            ax.set_yticks([])
            ax.grid(False)
            if i // ncols == nrows - 1 or i + ncols >= len(usable):
                ax.set_xlabel("Value-token NLL score")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


FIGURES = {
    "fig1_roc_auc": fig_roc_auc,
    "fig2_recall_quantiles": fig_recall_quantiles,
    "fig3_precision_recall_q90": fig_precision_recall_q90,
    "fig4_severity_q90": fig_severity_q90,
    "fig5_score_distributions": fig_score_distributions,
}

TABLES = {
    "table_ranking": table_ranking,
    "table_precision_f1": table_precision_f1,
    "table_severity": table_severity,
    "table_training": table_training,
}


def build_all(results_dir: Path, out_dir: Path, weights_dir: Path | None = None,
              formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Write every figure (PNG + PDF) and table (CSV) to ``out_dir``."""
    models = load_models(results_dir, weights_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, make in FIGURES.items():
        fig = make(models)
        for ext in formats:
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path)
            written.append(path)
        plt.close(fig)
    for name, make in TABLES.items():
        path = out_dir / f"{name}.csv"
        make(models).to_csv(path, index=False, float_format="%.4f")
        written.append(path)
    return written
