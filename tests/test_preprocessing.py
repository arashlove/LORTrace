"""Tests for cleaning, LOR alignment, splitting, and windowing."""

from __future__ import annotations

import numpy as np

from lortrace.benchmark import config as cfg_mod
from lortrace.benchmark import preprocessing as pp
from lortrace.benchmark.preprocessing import PreprocessReport, run_preprocessing
from synthetic import build_environment


def test_dedup_keeps_last_record(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    report = PreprocessReport()
    df = pp._load_main(cfg, report)

    assert report.n_duplicate_timestamps == 2
    assert df["date"].is_unique
    # The kept RRP values are the "last" occurrences, not the injected dupes.
    kept = set(np.round(df["RRP"].to_numpy(), 0))
    assert -999.0 not in kept and -888.0 not in kept
    assert 111.0 in kept and 222.0 in kept


def test_scaling_fit_on_train_only(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    run_preprocessing(cfg)
    manifest = pp.load_manifest(cfg)

    report = PreprocessReport()
    df = pp._load_main(cfg, report)
    df = pp._attach_labels(cfg, df, report)
    df = pp._engineer_features(cfg, df, report)
    split_id = pp._assign_splits(cfg, df)
    train_rrp = df.loc[split_id == 0, "RRP"].to_numpy(dtype=float)

    mean, std = manifest["scaling"]["RRP"]
    assert np.isclose(mean, train_rrp.mean(), atol=1e-3)
    assert std > 0


def test_windows_do_not_cross_split_boundaries(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    run_preprocessing(cfg)

    train = pp.load_split(cfg, "train")
    test = pp.load_split(cfg, "test")

    assert len(train["X"]) > 0
    assert len(test["X"]) > 0

    test_end = test["end_ts"].astype("datetime64[ns]")
    lo = np.datetime64("2015-01-08T00:00:00")
    assert (test_end >= lo).all()

    # Window feature/label array shapes are consistent.
    W = cfg.windowing.size
    assert train["X"].shape[1] == W
    assert train["X"].shape[2] == len(cfg.features.per_step)
    assert train["Wf"].shape[1] == len(cfg.features.window_level)


def test_train_normal_excludes_lor_and_buffer(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    run_preprocessing(cfg)
    test = pp.load_split(cfg, "test")

    # Any window with an actual LOR must not be marked as a normal training window.
    assert not np.any(test["train_normal"] & (test["actual_sev"] > 0))
    # The test split contains at least one labelled LOR window.
    assert int((test["actual_sev"] > 0).sum()) >= 1


def test_load_split_combined_concatenates_val_and_test(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    run_preprocessing(cfg)
    val = pp.load_split(cfg, "val")
    test = pp.load_split(cfg, "test")
    combined = pp.load_split(cfg, pp.EVAL_SPLIT)

    assert len(combined["X"]) == len(val["X"]) + len(test["X"])
    assert np.all(np.diff(combined["end_ts"]) >= 0)
    assert not np.any(combined["train_normal"] & (combined["actual_sev"] > 0))


def test_train_normal_variants_exclude_correct_labels(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    run_preprocessing(cfg)
    train = pp.load_split(cfg, "train")

    assert "train_normal_forecast" in train
    assert "train_normal_both" in train
    assert not np.any(train["train_normal"] & (train["actual_sev"] > 0))
    assert not np.any(train["train_normal_forecast"] & (train["forecast_sev"] > 0))
    assert not np.any(train["train_normal_both"] & (train["actual_sev"] > 0))
    assert not np.any(train["train_normal_both"] & (train["forecast_sev"] > 0))
    # Both-excluded set is the intersection of the two single-label masks.
    assert np.array_equal(
        train["train_normal_both"],
        train["train_normal"] & train["train_normal_forecast"],
    )
