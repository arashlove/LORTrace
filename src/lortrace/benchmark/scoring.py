"""Compute LORTrace anomaly scores via value-token negative log-likelihood.

For each window we serialize the sequence, run a forward pass, and sum the
per-token cross-entropy over *feature-value* tokens only (column names and
separators are masked out). Scores are averaged over several fixed feature-order
permutations. A higher score means the fine-tuned model was more "surprised" by
the window, i.e. it is more anomalous.

Scoring is chunked and checkpointed so long runs on a 6 GB GPU can resume.
GPU dependencies are imported lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfg_mod
from .config import Config
from .preprocessing import EVAL_SPLIT, load_manifest, load_split
from .serialization import resolve_text_tokenizer, serializer_from_manifest
from .training import adapter_path


def _score_file(cfg: Config, split: str) -> Path:
    return cfg_mod.scores_dir(cfg) / f"{split}_scores.npz"


def _pad_batch(sequences: list[dict[str, list[int]]], pad_id: int, torch):
    max_len = max(len(s["input_ids"]) for s in sequences)
    batch = len(sequences)
    input_ids = np.full((batch, max_len), pad_id, dtype=np.int64)
    attention = np.zeros((batch, max_len), dtype=np.int64)
    value_mask = np.zeros((batch, max_len), dtype=np.int64)
    col_ids = np.full((batch, max_len), -1, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = len(seq["input_ids"])
        input_ids[i, :length] = seq["input_ids"]
        attention[i, :length] = 1
        value_mask[i, :length] = seq["value_mask"]
        col_ids[i, :length] = seq["col_ids"]
    return (
        torch.from_numpy(input_ids),
        torch.from_numpy(attention),
        torch.from_numpy(value_mask),
        torch.from_numpy(col_ids),
    )


def _causal_lm_logits(model, input_ids, attention_mask):
    """Forward pass for plain LMs; tolerates VL wrappers used text-only."""
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        # Some VL forwards require unused kwargs defaults; retry positional-style.
        out = model(input_ids, attention_mask=attention_mask)
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    lang = getattr(model, "language_model", None)
    if lang is not None:
        out = lang(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "logits") and out.logits is not None:
            return out.logits
    raise RuntimeError(
        "Model forward did not return logits for text-only inputs. "
        f"Got output type {type(out).__name__}."
    )


def run_scoring(cfg: Config, split: str, force: bool = False) -> Path:
    """Score every window in ``split`` and save results (with resume)."""
    import os

    # Same Windows Triton/Dynamo issue as training (missing ``triton_key``).
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.disable = True
    except Exception:
        pass

    import torch
    from torch.nn import CrossEntropyLoss
    from unsloth import FastLanguageModel

    manifest = load_manifest(cfg)
    serializer = serializer_from_manifest(manifest)
    n_features = len(manifest["per_step_features"])

    data = load_split(cfg, split)
    n = len(data["X"])

    out_dir = cfg_mod.scores_dir(cfg)
    cfg_mod.ensure_dirs(out_dir)
    out_path = _score_file(cfg, split)

    feature_wise = bool(cfg.scoring.feature_wise)
    scores = np.full(n, np.nan, dtype=np.float64)
    feat_scores = (
        np.full((n, n_features), np.nan, dtype=np.float64) if feature_wise else None
    )
    start_chunk = 0
    chunk_size = int(cfg.scoring.chunk_size)

    if out_path.is_file() and not force:
        with np.load(out_path, allow_pickle=False) as prev:
            if len(prev["scores"]) == n:
                scores = prev["scores"].astype(np.float64)
                if feature_wise and "feature_scores" in prev.files:
                    feat_scores = prev["feature_scores"].astype(np.float64)
                done = ~np.isnan(scores)
                start_chunk = int(np.floor(done.sum() / chunk_size)) if done.any() else 0

    if not np.isnan(scores).any() and not force:
        return out_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path(cfg)),
        max_seq_length=cfg.model.max_seq_length,
        dtype=cfg.model.dtype,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = resolve_text_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    # Unsloth already places the (quantized) model on the GPU; do not .to() a
    # bitsandbytes 4-bit model. We only need the device for input tensors.
    device = next(model.parameters()).device

    perm_rng = np.random.default_rng(cfg.seed)
    permutations = [serializer.random_permutation(perm_rng) for _ in range(int(cfg.scoring.n_permutations))]

    loss_fct = CrossEntropyLoss(reduction="none")
    eval_bs = int(cfg.scoring.eval_batch_size)
    X, Wf = data["X"], data["Wf"]

    from tqdm import tqdm

    n_chunks = int(np.ceil(n / chunk_size))
    progress = tqdm(
        total=n,
        initial=start_chunk * chunk_size,
        desc=f"scoring {split}",
        unit="win",
    )
    for chunk in range(start_chunk, n_chunks):
        lo = chunk * chunk_size
        hi = min(n, lo + chunk_size)
        chunk_scores = np.zeros(hi - lo, dtype=np.float64)
        chunk_feat = (
            np.zeros((hi - lo, n_features), dtype=np.float64) if feature_wise else None
        )

        for perm in permutations:
            seqs = [
                serializer.to_tokens(X[i], Wf[i], perm, tokenizer)
                for i in range(lo, hi)
            ]
            for b_lo in range(0, len(seqs), eval_bs):
                batch = seqs[b_lo : b_lo + eval_bs]
                input_ids, attention, value_mask, col_ids = _pad_batch(batch, pad_id, torch)
                input_ids = input_ids.to(device)
                attention = attention.to(device)

                with torch.no_grad():
                    logits = _causal_lm_logits(model, input_ids, attention)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                token_nll = loss_fct(
                    shift_logits.transpose(1, 2), shift_labels
                ).to(torch.float32)  # (B, L-1)

                vmask = value_mask[:, 1:].to(device).to(torch.float32)
                per_sample = (token_nll * vmask).sum(dim=1).cpu().numpy()
                chunk_scores[b_lo : b_lo + len(batch)] += per_sample

                if feature_wise:
                    cshift = col_ids[:, 1:].numpy()
                    nll_cpu = token_nll.cpu().numpy()
                    for j in range(n_features):
                        fmask = (cshift == j).astype(np.float64)
                        chunk_feat[b_lo : b_lo + len(batch), j] += (nll_cpu * fmask).sum(axis=1)

        n_perm = len(permutations)
        scores[lo:hi] = chunk_scores / n_perm
        if feature_wise:
            feat_scores[lo:hi] = chunk_feat / n_perm

        _save_scores(out_path, scores, feat_scores, data)
        progress.update(hi - lo)

    progress.close()
    return out_path


def _save_scores(path: Path, scores, feat_scores, data) -> None:
    payload = {
        "scores": scores,
        "end_ts": data["end_ts"],
        "actual_sev": data["actual_sev"],
        "forecast_sev": data["forecast_sev"],
        "has_label": data["has_label"],
        "rrp_last": data["rrp_last"],
    }
    if feat_scores is not None:
        payload["feature_scores"] = feat_scores
    np.savez_compressed(path, **payload)


def load_scores(cfg: Config, split: str) -> dict[str, np.ndarray]:
    path = _score_file(cfg, split)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    # Backward compatibility: merge legacy per-split score files.
    if split == EVAL_SPLIT:
        val_path = _score_file(cfg, "val")
        test_path = _score_file(cfg, "test")
        if val_path.is_file() and test_path.is_file():
            val = load_scores(cfg, "val")
            test = load_scores(cfg, "test")
            merged: dict[str, np.ndarray] = {}
            for key in val:
                if isinstance(val[key], np.ndarray):
                    merged[key] = np.concatenate([val[key], test[key]], axis=0)
                else:
                    merged[key] = val[key]
            order = np.argsort(merged["end_ts"])
            return {
                k: (v[order] if isinstance(v, np.ndarray) and len(v) == len(order) else v)
                for k, v in merged.items()
            }

    raise FileNotFoundError(f"Scores for split '{split}' not found at {path}.")
