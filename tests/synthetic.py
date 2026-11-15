"""Helpers to build a tiny synthetic AEMO/LOR environment for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "RRP",
    "TOTALDEMAND",
    "TEMPERATURE",
    "AVAILABLEGENERATION",
    "NETINTERCHANGE",
    "TOTALINTERMITTENTGENERATION",
    "UIGF",
    "SEMISCHEDULE_CLEAREDMW",
    "DISPATCHABLELOAD",
    "DEMANDFORECAST",
    "BDU_CLEAREDMW_GEN",
    "BDU_CLEAREDMW_LOAD",
]

CONFIG_TEMPLATE = """
seed: 7
region: NSW1
data:
  main_csv: {main_csv}
  actual_lor_csv: {actual_csv}
  forecast_lor_csv: {forecast_csv}
  timestamp_col: date
  region_col: REGIONID
artifacts:
  root: {root}/artifacts
  processed_dir: {root}/artifacts/processed
  model_dir: {root}/artifacts/model
  scores_dir: {root}/artifacts/scores
  evaluation_dir: {root}/artifacts/evaluation
splits:
  train: ["2015-01-01", "2015-01-05"]
  val:   ["2015-01-06", "2015-01-07"]
  test:  ["2015-01-08", "2015-01-10"]
  embargo_steps: 2
windowing:
  size: 3
  stride: 3
features:
  raw_used: {raw_used}
  per_step: [RRP, TOTALDEMAND, reserve_margin]
  window_level: [hour, battery_active]
  descriptions:
    RRP: "price"
    TOTALDEMAND: "demand"
    reserve_margin: "margin"
    hour: "hour 0-23"
    battery_active: "battery flag"
  dropped: {{}}
labels:
  severity_levels: [1, 2, 3]
  train_exclusion_buffer_steps: 2
model:
  name: unsloth/mistral-7b-v0.3
  max_seq_length: 256
  load_in_4bit: true
  dtype: null
  lora:
    r: 8
    alpha: 8
    dropout: 0.0
    target_modules: [q_proj, v_proj]
training:
  setting: semi_supervised
  max_train_windows: 100
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 0.0002
  weight_decay: 0.001
  warmup_steps: 1
  max_steps: 2
  num_train_epochs: 1
  lr_scheduler_type: linear
  optim: adamw_8bit
  logging_steps: 1
scoring:
  n_permutations: 2
  eval_batch_size: 2
  chunk_size: 8
  feature_wise: false
evaluation:
  threshold_quantiles: [0.90, 0.95, 0.99]
  event_tolerance_steps: 2
signature:
  enabled: false
  depth: 2
  time_augment: true
  replace_path: false
"""


def build_environment(tmp_path: Path) -> Path:
    """Create synthetic CSVs + config under ``tmp_path``; return config path."""
    ts = pd.date_range("2015-01-01 00:05:00", "2015-01-10 23:55:00", freq="5min")
    n = len(ts)
    rng = np.random.default_rng(0)

    frame = {"date": ts, "REGIONID": "NSW1"}
    for col in RAW_COLUMNS:
        frame[col] = rng.normal(100, 10, size=n).astype(float)
    # Batteries only exist in the second half.
    half = n // 2
    df = pd.DataFrame(frame)
    df.loc[: half - 1, "BDU_CLEAREDMW_GEN"] = np.nan
    df.loc[: half - 1, "BDU_CLEAREDMW_LOAD"] = np.nan

    # Inject duplicate timestamps whose last record wins (distinct RRP).
    dupes = df.iloc[[10, 20]].copy()
    dupes["RRP"] = [-999.0, -888.0]  # first occurrence values
    df.loc[10, "RRP"] = 111.0  # last occurrence values (kept)
    df.loc[20, "RRP"] = 222.0
    df = pd.concat([dupes, df], ignore_index=False).sort_index(kind="stable")

    main_csv = tmp_path / "main.csv"
    df.to_csv(main_csv, index=False)

    # LOR files across all regions; NSW1 gets a couple of test-period events.
    regions = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
    lor_ts = pd.date_range("2015-01-01 00:00:00", "2015-01-10 23:55:00", freq="5min")
    rows = []
    for t in lor_ts:
        for reg in regions:
            rows.append((t, reg))
    lor = pd.DataFrame(rows, columns=["date", "REGIONID"])

    def _make(prefix: str, event_days: list[str]) -> pd.DataFrame:
        out = lor.copy()
        out[f"{prefix}_LOR1"] = 0
        out[f"{prefix}_LOR2"] = 0
        out[f"{prefix}_LOR3"] = 0
        nsw = (out["REGIONID"] == "NSW1")
        for day, level in zip(event_days, [1, 2]):
            sel = nsw & (out["date"].dt.strftime("%Y-%m-%d") == day)
            # a short burst within the day
            sel = sel & (out["date"].dt.hour == 12)
            out.loc[sel, f"{prefix}_LOR{level}"] = 1
        return out

    actual_csv = tmp_path / "actual.csv"
    forecast_csv = tmp_path / "forecast.csv"
    _make("ACTUAL", ["2015-01-08", "2015-01-09"]).to_csv(actual_csv, index=False)
    _make("FORECAST", ["2015-01-08", "2015-01-09"]).to_csv(forecast_csv, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            main_csv=main_csv.as_posix(),
            actual_csv=actual_csv.as_posix(),
            forecast_csv=forecast_csv.as_posix(),
            root=tmp_path.as_posix(),
            raw_used="[" + ", ".join(RAW_COLUMNS) + "]",
        ),
        encoding="utf-8",
    )
    return config_path
