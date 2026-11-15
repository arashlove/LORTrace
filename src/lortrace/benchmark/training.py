"""Fine-tune a causal-LM backbone with Unsloth LoRA on serialized market windows.

Training follows the AnoLLM recipe (serialize → LM fine-tune): serialize each window to text and fine-tune
with next-token prediction (loss over the whole sequence). Semi-supervised
settings exclude LOR-labelled windows (and a buffer around them) so the model
learns "normal" NSW1 market behaviour and is later *surprised* by anomalies.

Supported ``training.setting`` values:

- ``semi_supervised`` – exclude actual LOR (default; original behaviour)
- ``semi_supervised_forecast`` – exclude forecast LOR only (actual may remain)
- ``semi_supervised_both`` – exclude actual and forecast LOR
- ``unsupervised`` – train on all train-split windows

Heavy GPU dependencies (``unsloth``, ``torch``, ``trl``) are imported lazily so
the rest of the package remains importable on a CPU-only machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as cfg_mod
from .config import Config
from .preprocessing import load_manifest, load_split
from .serialization import Serializer, serializer_from_manifest, resolve_text_tokenizer

# Map training.setting -> eligibility mask stored in the preprocessed NPZ.
_SETTING_MASK = {
    "semi_supervised": "train_normal",
    "semi_supervised_forecast": "train_normal_forecast",
    "semi_supervised_both": "train_normal_both",
}


def _select_training_windows(cfg: Config, split: dict[str, np.ndarray]) -> np.ndarray:
    """Return indices of eligible training windows for the configured setting."""
    n = len(split["X"])
    setting = cfg.training.setting
    if setting == "unsupervised":
        eligible = np.arange(n)
    elif setting in _SETTING_MASK:
        mask_key = _SETTING_MASK[setting]
        if mask_key not in split:
            raise RuntimeError(
                f"Processed split is missing '{mask_key}'. Re-run preprocessing "
                f"(python preprocess.py) so eligibility masks for setting "
                f"'{setting}' are written into the NPZ files."
            )
        eligible = np.flatnonzero(split[mask_key])
    else:
        raise ValueError(f"Unknown training.setting: {setting}")

    rng = np.random.default_rng(cfg.seed)
    max_windows = int(cfg.training.max_train_windows)
    if max_windows > 0 and len(eligible) > max_windows:
        eligible = rng.choice(eligible, size=max_windows, replace=False)
        eligible.sort()
    return eligible


def _build_text_dataset(cfg: Config, serializer: Serializer, eos_token: str):
    from datasets import Dataset

    split = load_split(cfg, "train")
    indices = _select_training_windows(cfg, split)
    if len(indices) == 0:
        raise RuntimeError(
            "No eligible training windows. Check splits/labels or set "
            "training.setting: unsupervised."
        )

    rng = np.random.default_rng(cfg.seed)
    X = split["X"]
    Wf = split["Wf"]

    texts: list[str] = []
    for i in indices:
        perm = serializer.random_permutation(rng)
        texts.append(serializer.to_text(X[i], Wf[i], perm, eos_token))

    return Dataset.from_dict({"text": texts}), len(indices)


def run_training(cfg: Config) -> dict:
    """Fine-tune the model and save LoRA adapters + tokenizer + metrics."""
    import os
    import sys

    # Windows + some Unsloth/Gemma paths trigger torch.compile/inductor, which
    # can crash when Triton lacks ``triton_key`` (version skew). Disable Dynamo.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.disable = True
    except Exception:
        pass

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    manifest = load_manifest(cfg)
    serializer = serializer_from_manifest(manifest)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    # VL processors (Cosmos / Qwen2.5-VL / Gemma3) wrap a text tokenizer; unwrap for SFT text.
    text_tokenizer = resolve_text_tokenizer(tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.model.lora.r,
        target_modules=list(cfg.model.lora.target_modules),
        lora_alpha=cfg.model.lora.alpha,
        lora_dropout=cfg.model.lora.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    # Prefer non-compiled checkpointing path when Dynamo is unavailable.
    if sys.platform.startswith("win"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    eos = getattr(text_tokenizer, "eos_token", None) or getattr(tokenizer, "eos_token", "")
    dataset, n_windows = _build_text_dataset(cfg, serializer, eos)

    out_dir = cfg_mod.model_dir(cfg)
    cfg_mod.ensure_dirs(out_dir)

    sft_args = dict(
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.warmup_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        seed=cfg.seed,
        output_dir=str(out_dir / "trainer"),
        report_to="none",
    )
    if int(cfg.training.max_steps) and int(cfg.training.max_steps) > 0:
        sft_args["max_steps"] = int(cfg.training.max_steps)
    else:
        sft_args["num_train_epochs"] = cfg.training.num_train_epochs

    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        packing=False,
        args=SFTConfig(**sft_args),
    )

    stats = trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    text_tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "n_training_windows": int(n_windows),
        "setting": cfg.training.setting,
        "train_runtime_sec": float(stats.metrics.get("train_runtime", 0.0)),
        "train_loss": float(stats.metrics.get("train_loss", float("nan"))),
        "model_name": cfg.model.name,
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def adapter_path(cfg: Config) -> Path:
    return cfg_mod.model_dir(cfg) / "adapter"
