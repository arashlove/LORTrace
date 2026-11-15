"""Leakage-safe preprocessing for the AEMO NSW1 LORTrace pipeline.

Pipeline:
    1. Load the main 5-minute series, filter to the region, and deduplicate
       repeated timestamps keeping the *last* record.
    2. Align actual and forecast LOR labels (filtered to the region) onto the
       market intervals, collapsing the LOR1/2/3 flags to an ordinal severity.
    3. Engineer derived features and calendar context.
    4. Assign strict chronological train/val/test splits with a boundary
       embargo, then fit standard-rescaling statistics on the train split only.
    5. Build fixed-length windows (dropping any that straddle a split boundary)
       and save compact per-split ``.npz`` arrays plus JSON metadata.

Nothing here touches the LLM; outputs feed :mod:`serialization` downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg_mod
from .config import Config
from . import signature as sig_mod

DERIVED_FEATURES = ("reserve_margin", "curtailment_gap")
WINDOW_LEVEL_ALL = ("hour", "dayofweek", "month", "battery_active")
FREQ = "5min"

# After training, scoring and evaluation use validation + test windows combined.
EVAL_SPLIT = "val_test"
POST_TRAIN_SPLITS = ("val", "test")


@dataclass
class PreprocessReport:
    """Human-readable summary of the preprocessing run."""

    n_raw_rows: int = 0
    n_duplicate_timestamps: int = 0
    n_fully_identical_dropped: int = 0
    n_after_dedup: int = 0
    grid_gaps: int = 0
    bdu_missing_fraction: float = 0.0
    lor_coverage: dict[str, str] = field(default_factory=dict)
    split_window_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_raw_rows": self.n_raw_rows,
            "n_duplicate_timestamps": self.n_duplicate_timestamps,
            "n_fully_identical_dropped": self.n_fully_identical_dropped,
            "n_after_dedup": self.n_after_dedup,
            "grid_gaps": self.grid_gaps,
            "bdu_missing_fraction": self.bdu_missing_fraction,
            "lor_coverage": self.lor_coverage,
            "split_window_counts": self.split_window_counts,
            "severity_counts": self.severity_counts,
        }


def _load_main(cfg: Config, report: PreprocessReport) -> pd.DataFrame:
    path = cfg_mod.resolve_path(cfg, cfg.data.main_csv)
    ts_col = cfg.data.timestamp_col
    region_col = cfg.data.region_col

    usecols = [ts_col, region_col] + list(cfg.features.raw_used)
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    report.n_raw_rows = len(df)

    df = df[df[region_col] == cfg.region].copy()
    df[ts_col] = pd.to_datetime(df[ts_col])

    # Track fully identical duplicate rows for reporting before we dedup on time.
    report.n_fully_identical_dropped = int(df.duplicated().sum())

    df["_orig_order"] = np.arange(len(df))
    dup_mask = df.duplicated(subset=[ts_col], keep=False)
    report.n_duplicate_timestamps = int(df.loc[dup_mask, ts_col].nunique())

    # Keep the last record for each timestamp (adjusted/latest price revision),
    # preserving original file order before selecting, then sort by time.
    df = df.sort_values("_orig_order")
    df = df.drop_duplicates(subset=[ts_col], keep="last")
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.drop(columns="_orig_order")
    report.n_after_dedup = len(df)

    # Report gaps in the regular 5-minute grid (does not modify data).
    full_index = pd.date_range(df[ts_col].iloc[0], df[ts_col].iloc[-1], freq=FREQ)
    report.grid_gaps = int(len(full_index) - df[ts_col].nunique())

    return df


def _lor_severity(csv_path: Path, prefix: str, cfg: Config) -> pd.DataFrame:
    """Return a frame with columns ``[date, {prefix}_sev]`` for the region.

    Severity is the maximum active LOR level (LOR3 > LOR2 > LOR1 > none=0).
    """
    ts_col = cfg.data.timestamp_col
    region_col = cfg.data.region_col
    cols = [f"{prefix}_LOR{i}" for i in (1, 2, 3)]

    df = pd.read_csv(csv_path, usecols=[ts_col, region_col] + cols)
    df = df[df[region_col] == cfg.region].copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.drop_duplicates(subset=[ts_col], keep="last")

    levels = df[cols].to_numpy()
    weighted = levels * np.array([1, 2, 3])
    severity = weighted.max(axis=1).astype(np.int8)
    out = pd.DataFrame({ts_col: df[ts_col].to_numpy(), f"{prefix}_sev": severity})
    return out.sort_values(ts_col).reset_index(drop=True)


def _attach_labels(cfg: Config, df: pd.DataFrame, report: PreprocessReport) -> pd.DataFrame:
    ts_col = cfg.data.timestamp_col
    actual = _lor_severity(
        cfg_mod.resolve_path(cfg, cfg.data.actual_lor_csv), "ACTUAL", cfg
    )
    forecast = _lor_severity(
        cfg_mod.resolve_path(cfg, cfg.data.forecast_lor_csv), "FORECAST", cfg
    )

    cov_min = actual[ts_col].min()
    cov_max = actual[ts_col].max()
    report.lor_coverage = {"start": str(cov_min), "end": str(cov_max)}

    df = df.merge(actual, on=ts_col, how="left")
    df = df.merge(forecast, on=ts_col, how="left")

    df["has_label"] = df[ts_col].between(cov_min, cov_max)
    df["ACTUAL_sev"] = df["ACTUAL_sev"].fillna(0).astype(np.int8)
    df["FORECAST_sev"] = df["FORECAST_sev"].fillna(0).astype(np.int8)
    return df


def _engineer_features(cfg: Config, df: pd.DataFrame, report: PreprocessReport) -> pd.DataFrame:
    bdu_cols = [c for c in ("BDU_CLEAREDMW_GEN", "BDU_CLEAREDMW_LOAD") if c in df.columns]
    if bdu_cols:
        missing = df[bdu_cols].isna().all(axis=1)
        report.bdu_missing_fraction = float(missing.mean())
        df["battery_active"] = (~df[bdu_cols].isna().all(axis=1)).astype(np.int8)
        for col in bdu_cols:
            df[col] = df[col].fillna(0.0)
    else:
        df["battery_active"] = np.int8(0)

    if {"AVAILABLEGENERATION", "TOTALDEMAND"}.issubset(df.columns):
        df["reserve_margin"] = df["AVAILABLEGENERATION"] - df["TOTALDEMAND"]
    if {"UIGF", "SEMISCHEDULE_CLEAREDMW"}.issubset(df.columns):
        df["curtailment_gap"] = df["UIGF"] - df["SEMISCHEDULE_CLEAREDMW"]

    ts = df[cfg.data.timestamp_col].dt
    df["hour"] = ts.hour.astype(np.int16)
    df["dayofweek"] = ts.dayofweek.astype(np.int16)
    df["month"] = ts.month.astype(np.int16)

    # Fill any residual NaNs in per-step features with 0 (post-scaling neutral).
    for col in cfg.features.per_step:
        if col in df.columns:
            df[col] = df[col].astype(np.float32).fillna(0.0)
    return df


def _assign_splits(cfg: Config, df: pd.DataFrame) -> np.ndarray:
    """Return an int8 per-row split id: 0=train, 1=val, 2=test, -1=unused."""
    ts = df[cfg.data.timestamp_col]
    split_id = np.full(len(df), -1, dtype=np.int8)
    for idx, name in enumerate(("train", "val", "test")):
        lo, hi = getattr(cfg.splits, name)
        lo_ts = pd.Timestamp(lo)
        # Inclusive upper bound to end-of-day.
        hi_ts = pd.Timestamp(hi) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = ts.between(lo_ts, hi_ts).to_numpy()
        split_id[mask] = idx
    return split_id


def _embargo_mask(split_id: np.ndarray, embargo_steps: int) -> np.ndarray:
    """Mark rows within ``embargo_steps`` of a split-id change as embargoed."""
    embargoed = np.zeros(len(split_id), dtype=bool)
    if embargo_steps <= 0:
        return embargoed
    change_points = np.flatnonzero(np.diff(split_id) != 0) + 1
    for cp in change_points:
        lo = max(0, cp - embargo_steps)
        hi = min(len(split_id), cp + embargo_steps)
        embargoed[lo:hi] = True
    return embargoed


def _fit_scaling(cfg: Config, df: pd.DataFrame, train_mask: np.ndarray) -> dict[str, list[float]]:
    scaling: dict[str, list[float]] = {}
    for col in cfg.features.per_step:
        values = df.loc[train_mask, col].to_numpy(dtype=np.float64)
        mean = float(np.mean(values)) if len(values) else 0.0
        std = float(np.std(values)) if len(values) else 1.0
        if std < 1e-8:
            std = 1.0
        scaling[col] = [mean, std]
    return scaling


def _build_windows(
    cfg: Config,
    df: pd.DataFrame,
    split_id: np.ndarray,
    embargoed: np.ndarray,
    scaling: dict[str, list[float]],
) -> dict[str, dict[str, np.ndarray]]:
    W = cfg.windowing.size
    stride = cfg.windowing.stride
    per_step = list(cfg.features.per_step)
    window_level = list(cfg.features.window_level)

    # Scaled per-step feature matrix (N, Fp).
    feat = np.empty((len(df), len(per_step)), dtype=np.float32)
    for j, col in enumerate(per_step):
        mean, std = scaling[col]
        feat[:, j] = (df[col].to_numpy(dtype=np.float32) - mean) / std

    wl = np.stack([df[c].to_numpy(dtype=np.float32) for c in window_level], axis=1)

    actual_sev = df["ACTUAL_sev"].to_numpy(dtype=np.int8)
    forecast_sev = df["FORECAST_sev"].to_numpy(dtype=np.int8)
    has_label = df["has_label"].to_numpy(dtype=bool)
    rrp = (
        df["RRP"].to_numpy(dtype=np.float32)
        if "RRP" in df.columns
        else np.zeros(len(df), np.float32)
    )
    end_ts = df[cfg.data.timestamp_col].to_numpy().astype("datetime64[ns]").astype(np.int64)

    buffer = int(cfg.labels.train_exclusion_buffer_steps)
    near_actual = _dilate(actual_sev > 0, buffer)
    near_forecast = _dilate(forecast_sev > 0, buffer)

    n = len(df)
    starts_all = np.arange(0, n - W + 1, stride)
    idx = starts_all[:, None] + np.arange(W)[None, :]  # (num_starts, W)

    win_split = split_id[idx]
    win_embargo = embargoed[idx].any(axis=1)
    same_split = (win_split == win_split[:, :1]).all(axis=1)
    split_of_window = win_split[:, 0]

    valid = same_split & (~win_embargo) & (split_of_window >= 0)

    end_idx = idx[:, -1]
    result: dict[str, dict[str, np.ndarray]] = {}
    for split_code, name in enumerate(("train", "val", "test")):
        sel = valid & (split_of_window == split_code)
        starts = starts_all[sel]
        w_idx = idx[sel]
        w_end = end_idx[sel]

        X = feat[w_idx]  # (n, W, Fp)
        Wf = wl[w_end]   # (n, Cw)
        a_sev = actual_sev[w_idx].max(axis=1)
        f_sev = forecast_sev[w_idx].max(axis=1)
        h_lab = has_label[w_idx].all(axis=1)
        near_a = near_actual[w_idx].any(axis=1)
        near_f = near_forecast[w_idx].any(axis=1)
        # Eligibility masks for the three semi-supervised training settings.
        # Each also excludes the configured dilation buffer around the label(s).
        train_normal = (a_sev == 0) & (~near_a)  # exclude actual LOR
        train_normal_forecast = (f_sev == 0) & (~near_f)  # exclude forecast LOR
        train_normal_both = train_normal & train_normal_forecast

        result[name] = {
            "X": X.astype(np.float32),
            "Wf": Wf.astype(np.float32),
            "end_ts": end_ts[w_end].astype(np.int64),
            "actual_sev": a_sev.astype(np.int8),
            "forecast_sev": f_sev.astype(np.int8),
            "has_label": h_lab,
            "train_normal": train_normal,
            "train_normal_forecast": train_normal_forecast,
            "train_normal_both": train_normal_both,
            "rrp_last": rrp[w_end].astype(np.float32),
        }
    return result


def _dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    """Grow a boolean mask by ``steps`` positions in both directions."""
    if steps <= 0:
        return mask.copy()
    out = mask.copy()
    idx = np.flatnonzero(mask)
    for offset in range(1, steps + 1):
        lo = np.clip(idx - offset, 0, len(mask) - 1)
        hi = np.clip(idx + offset, 0, len(mask) - 1)
        out[lo] = True
        out[hi] = True
    return out


def _desc_map_from_cfg(cfg: Config) -> dict[str, str]:
    descriptions = getattr(cfg.features, "descriptions", None)
    if descriptions is None:
        return {}
    if isinstance(descriptions, dict):
        return {str(k): str(v) for k, v in descriptions.items()}
    return {str(k): str(v) for k, v in vars(descriptions).items()}


def _attach_log_signatures(
    cfg: Config, windows: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Append train-scaled log-signature coeffs to each split's ``Wf``.

    Returns updated windows and signature metadata for the manifest.
    """
    settings = sig_mod.signature_settings(cfg)
    if not settings["enabled"]:
        return windows, {"enabled": False}

    depth = settings["depth"]
    time_augment = settings["time_augment"]
    replace_path = settings["replace_path"]

    # Compute raw log-sigs per split from (already channel z-scored) paths.
    raw_by_split: dict[str, np.ndarray] = {}
    for name, arrays in windows.items():
        raw_by_split[name] = sig_mod.compute_log_signatures(
            arrays["X"], depth=depth, time_augment=time_augment
        )

    train_raw = raw_by_split["train"]
    scaling = sig_mod.fit_logsig_scaling(train_raw)
    n_coeffs = train_raw.shape[1] if train_raw.size else next(
        (v.shape[1] for v in raw_by_split.values() if v.size), 0
    )
    names = sig_mod.logsig_feature_names(n_coeffs)
    descs = sig_mod.logsig_descriptions(n_coeffs, depth, time_augment)

    for name, arrays in windows.items():
        scaled = sig_mod.apply_logsig_scaling(raw_by_split[name], scaling)
        arrays["Wf"] = np.concatenate([arrays["Wf"], scaled], axis=1).astype(np.float32)
        if replace_path:
            n, W, _ = arrays["X"].shape
            arrays["X"] = np.zeros((n, W, 0), dtype=np.float32)

    meta = {
        "enabled": True,
        "depth": depth,
        "time_augment": time_augment,
        "replace_path": replace_path,
        "n_coeffs": n_coeffs,
        "feature_names": names,
        "scaling": scaling,
        "descriptions": descs,
    }
    return windows, meta


def run_preprocessing(cfg: Config) -> PreprocessReport:
    """Execute the full preprocessing pipeline and write artifacts."""
    report = PreprocessReport()

    df = _load_main(cfg, report)
    df = _attach_labels(cfg, df, report)
    df = _engineer_features(cfg, df, report)

    split_id = _assign_splits(cfg, df)
    embargoed = _embargo_mask(split_id, int(cfg.splits.embargo_steps))
    train_mask = split_id == 0
    scaling = _fit_scaling(cfg, df, train_mask)

    windows = _build_windows(cfg, df, split_id, embargoed, scaling)
    windows, sig_meta = _attach_log_signatures(cfg, windows)

    out_dir = cfg_mod.processed_dir(cfg)
    cfg_mod.ensure_dirs(out_dir)

    severity_counts: dict[str, Any] = {}
    for name, arrays in windows.items():
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
        report.split_window_counts[name] = int(len(arrays["X"]))
        labeled = arrays["has_label"]
        sev = arrays["actual_sev"][labeled]
        severity_counts[name] = {
            "n_labeled": int(labeled.sum()),
            "actual_LOR1": int((sev == 1).sum()),
            "actual_LOR2": int((sev == 2).sum()),
            "actual_LOR3": int((sev == 3).sum()),
            "n_train_normal": int(arrays["train_normal"].sum()),
            "n_train_normal_forecast": int(arrays["train_normal_forecast"].sum()),
            "n_train_normal_both": int(arrays["train_normal_both"].sum()),
        }
    report.severity_counts = severity_counts

    desc_map = _desc_map_from_cfg(cfg)
    window_features = list(cfg.features.window_level)
    continuous_window = []
    per_step_features = list(cfg.features.per_step)
    if sig_meta.get("enabled"):
        window_features = window_features + list(sig_meta["feature_names"])
        continuous_window = list(sig_meta["feature_names"])
        desc_map.update(sig_meta["descriptions"])
        if sig_meta.get("replace_path"):
            per_step_features = []

    manifest = {
        "region": cfg.region,
        "window_size": cfg.windowing.size,
        "stride": cfg.windowing.stride,
        "per_step_features": per_step_features,
        "window_features": window_features,
        "continuous_window_features": continuous_window,
        "descriptions": desc_map,
        "scaling": scaling,
        "signature": {
            k: v
            for k, v in sig_meta.items()
            if k != "descriptions"  # already merged into descriptions
        },
        "splits": {
            "train": list(cfg.splits.train),
            "val": list(cfg.splits.val),
            "test": list(cfg.splits.test),
        },
        "freq": FREQ,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "data_quality.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return report


def load_split(cfg: Config, split: str) -> dict[str, np.ndarray]:
    """Load a preprocessed split's arrays from disk.

    ``val_test`` concatenates the validation and test splits (used for scoring
    and evaluation after training).
    """
    if split == EVAL_SPLIT:
        return load_split_combined(cfg, POST_TRAIN_SPLITS)
    path = cfg_mod.processed_dir(cfg) / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed split '{split}' not found at {path}. Run preprocessing first."
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_split_combined(
    cfg: Config, splits: tuple[str, ...] = POST_TRAIN_SPLITS
) -> dict[str, np.ndarray]:
    """Concatenate multiple on-disk splits into one array bundle (time-ordered)."""
    parts: list[dict[str, np.ndarray]] = []
    for name in splits:
        path = cfg_mod.processed_dir(cfg) / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed split '{name}' not found at {path}. Run preprocessing first."
            )
        with np.load(path, allow_pickle=False) as data:
            parts.append({key: data[key] for key in data.files})

    if not parts:
        raise ValueError("load_split_combined requires at least one split")

    out: dict[str, np.ndarray] = {}
    for key in parts[0]:
        if isinstance(parts[0][key], np.ndarray):
            out[key] = np.concatenate([p[key] for p in parts], axis=0)
        else:
            out[key] = parts[0][key]

    order = np.argsort(out["end_ts"])
    return {k: (v[order] if isinstance(v, np.ndarray) and len(v) == len(order) else v)
            for k, v in out.items()}


def load_manifest(cfg: Config) -> dict[str, Any]:
    path = cfg_mod.processed_dir(cfg) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found at {path}. Run preprocessing first.")
    return json.loads(path.read_text(encoding="utf-8"))
