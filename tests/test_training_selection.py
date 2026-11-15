"""Tests for training-window eligibility under the three semi-supervised settings."""

from __future__ import annotations

import numpy as np

from lortrace.benchmark import config as cfg_mod
from lortrace.benchmark import preprocessing as pp
from lortrace.benchmark import training as tr
from lortrace.benchmark.preprocessing import run_preprocessing
from synthetic import build_environment


def test_select_training_windows_respects_setting(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    run_preprocessing(cfg)
    split = pp.load_split(cfg, "train")

    cfg.training.setting = "semi_supervised"
    idx_actual = tr._select_training_windows(cfg, split)
    assert np.all(split["train_normal"][idx_actual])

    cfg.training.setting = "semi_supervised_forecast"
    idx_forecast = tr._select_training_windows(cfg, split)
    assert np.all(split["train_normal_forecast"][idx_forecast])

    cfg.training.setting = "semi_supervised_both"
    idx_both = tr._select_training_windows(cfg, split)
    assert np.all(split["train_normal_both"][idx_both])

    cfg.training.setting = "unsupervised"
    cfg.training.max_train_windows = 0  # keep all
    idx_all = tr._select_training_windows(cfg, split)
    assert len(idx_all) == len(split["X"])
