"""Tests for evaluation metric helpers and edge cases."""

from __future__ import annotations

import numpy as np

from lortrace.benchmark import evaluation as ev


def test_binary_metrics_perfect_separation():
    scores = np.array([0.1, 0.2, 5.0, 6.0])
    y = np.array([0, 0, 1, 1])
    m = ev._binary_metrics(scores, y, threshold=1.0)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0 and m["tn"] == 2
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_ranking_metrics_undefined_when_single_class():
    scores = np.array([0.1, 0.2, 0.3])
    y = np.zeros(3, dtype=int)
    out = ev._ranking_metrics(scores, y)
    assert out["roc_auc"] is None
    assert out["average_precision"] is None


def test_per_severity_recall_handles_missing_level():
    scores = np.array([5.0, 0.1, 5.0])
    severity = np.array([1, 2, 0])  # no LOR3 present
    out = ev._per_severity_recall(scores, severity, threshold=1.0)
    assert out["LOR1"]["recall"] == 1.0
    assert out["LOR2"]["recall"] == 0.0
    assert out["LOR3"]["n"] == 0
    assert out["LOR3"]["recall"] is None


def test_events_from_mask_groups_runs():
    mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    events = ev._events_from_mask(mask)
    assert events == [(1, 2), (5, 5)]


def test_detection_matrix_uses_actual_cols():
    pred = np.array([1, 0, 1, 0], dtype=bool)
    gt = np.array([1, 1, 0, 2])
    out = ev._detection_matrix(pred, gt)
    assert out["cols"][0] == "actual_none"
    assert out["cols"][1] == "actual_LOR1"
    mat = np.array(out["matrix"])
    assert mat[1, 1] == 1
    assert mat[0, 1] == 1


def test_forecast_benchmark_recall():
    forecast = np.array([1, 0, 2, 0])
    actual = np.array([1, 1, 2, 0])
    out = ev._forecast_benchmark(forecast, actual)
    # actual LOR1 occurs twice (idx 0,1); forecast flags only idx 0 -> 0.5.
    assert out["per_severity_recall"]["LOR1"]["recall"] == 0.5
    assert out["per_severity_recall"]["LOR2"]["recall"] == 1.0
    assert out["recall"] == 2 / 3  # 2 of 3 actual-positive detected
