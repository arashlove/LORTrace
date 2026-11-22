"""Smoke test: summary figures and tables build from a minimal results folder."""

from __future__ import annotations

import json

import numpy as np

from lortrace.benchmark import report as R


def _fake_model(folder, auc: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 400
    sev = np.zeros(n, dtype=np.int16)
    sev[:8], sev[8:12] = 1, 2
    scores = rng.normal(size=n) + 0.8 * (sev > 0)
    folder.mkdir()
    np.savez(folder / "val_test_scores.npz", scores=scores, actual_sev=sev,
             has_label=np.ones(n, bool))
    sev_block = {"LOR1": {"n": 8, "recall": 0.25}, "LOR2": {"n": 4, "recall": 0.5},
                 "LOR3": {"n": 0, "recall": None}}
    by_q = {q: {"binary_detection": {"recall": 0.3, "precision": 0.05, "f1": 0.08,
                                      "false_positive_rate": 1 - f, "n_positive": 12},
                "per_severity_recall": sev_block} for q, f in R.Q_FLOAT.items()}
    metrics = {"n_eval_labeled": n, "ranking_metrics": {"roc_auc": auc, "average_precision": 0.05},
               "by_quantile": by_q,
               "lor_benchmark": {"recall": 0.25, "per_severity_recall": sev_block}}
    (folder / "metrics_actual_lor.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_build_all(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _fake_model(results / "SmolLM2-135M", 0.6, 0)
    _fake_model(results / "mistral-7b-v0.3", 0.7, 1)

    models = R.load_models(results)
    assert [m["label"] for m in models] == ["SmolLM2 135M", "Mistral 7B v0.3"]
    assert R.has_severity(models, "LOR2") and not R.has_severity(models, "LOR3")

    out = tmp_path / "figures"
    written = R.build_all(results, out, formats=("png",))
    assert {p.name for p in written} >= {"fig1_roc_auc.png", "fig5_score_distributions.png",
                                         "table_ranking.csv", "table_severity.csv"}
    assert all(p.stat().st_size > 0 for p in written)
