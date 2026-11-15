"""Tests for log-signature features and preprocessing integration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

iisignature = pytest.importorskip("iisignature")

from lortrace.benchmark import config as cfg_mod
from lortrace.benchmark import preprocessing as pp
from lortrace.benchmark import signature as sig
from lortrace.benchmark.preprocessing import run_preprocessing
from lortrace.benchmark.serialization import Serializer, serializer_from_manifest
from synthetic import build_environment


def test_logsig_length_matches_iisignature():
    assert sig.logsig_length(3, 2) == iisignature.logsiglength(3, 2)


def test_compute_log_signatures_shape_and_scaling():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(5, 8, 3)).astype(np.float32)
    raw = sig.compute_log_signatures(windows, depth=2, time_augment=True)
    expected = iisignature.logsiglength(4, 2)  # 3 channels + time
    assert raw.shape == (5, expected)

    scaling = sig.fit_logsig_scaling(raw)
    scaled = sig.apply_logsig_scaling(raw, scaling)
    assert scaled.shape == raw.shape
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-5)
    # Constant coeffs keep raw std 0 and are left un-normalized (std=1.0 sentinel).
    for k, std in enumerate(scaled.std(axis=0)):
        raw_std = float(raw[:, k].std())
        if raw_std < 1e-8:
            assert std == pytest.approx(0.0, abs=1e-6)
        else:
            assert std == pytest.approx(1.0, abs=1e-4)


def test_preprocessing_appends_logsig_to_wf(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    cfg.signature = SimpleNamespace(
        enabled=True, depth=2, time_augment=True, replace_path=False
    )
    run_preprocessing(cfg)

    train = pp.load_split(cfg, "train")
    manifest = pp.load_manifest(cfg)
    n_base = len(cfg.features.window_level)
    n_coeffs = iisignature.logsiglength(len(cfg.features.per_step) + 1, 2)

    assert train["X"].shape[2] == len(cfg.features.per_step)
    assert train["Wf"].shape[1] == n_base + n_coeffs
    assert manifest["signature"]["enabled"] is True
    assert len(manifest["continuous_window_features"]) == n_coeffs
    assert all(n.startswith("logsig_") for n in manifest["continuous_window_features"])
    assert all(n in manifest["descriptions"] for n in manifest["continuous_window_features"])


def test_replace_path_clears_per_step(tmp_path):
    cfg = cfg_mod.load_config(build_environment(tmp_path))
    cfg.signature = SimpleNamespace(
        enabled=True, depth=2, time_augment=True, replace_path=True
    )
    run_preprocessing(cfg)

    train = pp.load_split(cfg, "train")
    manifest = pp.load_manifest(cfg)
    assert train["X"].shape[2] == 0
    assert manifest["per_step_features"] == []
    assert train["Wf"].shape[1] > len(cfg.features.window_level)

    ser = serializer_from_manifest(manifest)
    text = ser.to_text(
        train["X"][0],
        train["Wf"][0],
        ser.default_permutation(),
        "<eos>",
    )
    assert "logsig_0 is" in text
    assert "t0_" not in text


def test_serializer_formats_logsig_as_scaled():
    s = Serializer(
        per_step_features=["RRP"],
        window_features=["hour", "logsig_0"],
        window_size=2,
        continuous_window_features=["logsig_0"],
    )
    x = np.zeros((2, 1), dtype=np.float32)
    wf = np.array([12.0, 1.23], dtype=np.float32)
    text = s.to_text(x, wf, [0], "<eos>")
    assert "hour is 12" in text
    assert "logsig_0 is 1.2" in text
