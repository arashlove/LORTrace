---
license: other
license_name: mixed-base-model-licenses
license_link: https://huggingface.co/arashlove/lortrace-adapters/blob/main/README.md
library_name: peft
tags:
  - lora
  - unsloth
  - anomaly-detection
  - time-series
  - electricity-market
---

# LORTrace LoRA adapters

LoRA adapters from the
[LORTrace](https://github.com/arashlove/LORTrace) project.

Nine LoRA adapters, one per backbone, fine-tuned with unsupervised next-token
prediction on serialized AEMO NSW1 market windows.
These are the exact adapters that produced the released reference anomaly scores.

Each folder contains `adapter/` (PEFT adapter + tokenizer) and, where
available, `training_metrics.json`. The base model is downloaded automatically
from the Hub when the adapter is loaded (see `adapter/adapter_config.json`).

| Folder | Base checkpoint | Base model license |
|---|---|---|
| `SmolLM2-135M` | `unsloth/SmolLM2-135M-bnb-4bit` | Apache-2.0 |
| `gemma-3-270m` | `unsloth/gemma-3-270m-it-unsloth-bnb-4bit` | Gemma Terms of Use |
| `SmolLM2-360M` | `unsloth/SmolLM2-360M-bnb-4bit` | Apache-2.0 |
| `Qwen2.5-0.5B` | `unsloth/Qwen2.5-0.5B-bnb-4bit` | Apache-2.0 |
| `Qwen2.5-1.5B` | `unsloth/Qwen2.5-1.5B-bnb-4bit` | Apache-2.0 |
| `SmolLM-1.7B` | `unsloth/smollm-1.7b-bnb-4bit` | Apache-2.0 |
| `gemma-2-2b` | `unsloth/gemma-2-2b-it-bnb-4bit` | Gemma Terms of Use |
| `mistral-7b-v0.3` | `unsloth/mistral-7b-v0.3-bnb-4bit` | Apache-2.0 |
| `Cosmos-Reason1-7B` | `unsloth/Cosmos-Reason1-7B-unsloth-bnb-4bit` | NVIDIA Open Model License |

Each adapter is a derivative of its base model and is distributed under that
base model's license terms. The Gemma-derived adapters are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms), including its use
restrictions.

## Training configuration (all backbones)

LoRA r=16, α=16, dropout 0, targets `q,k,v,o,gate,up,down_proj`; 4-bit base;
max sequence length 4096; 20,000 training windows (seed 3407); 300 steps;
lr 2e-4, linear schedule, 5 warm-up steps, weight decay 1e-3, AdamW 8-bit;
effective batch 8. Training data: AEMO NSW1, 2015–2021, no LOR-based filtering.

## Usage

```bash
uv run python scripts/download_assets.py --weights
uv run python scripts/reproduce.py --mode rescore --models gemma-3-270m
```

`mistral-7b-v0.3` has no `training_metrics.json` (it was not preserved from
that run).

## Citation

If you use these adapters, please cite the dataset they were trained on:

```bibtex
@misc{karimadini2026aemonemlor,
  title     = {{AEMO NEM Lack-of-Reserve Dataset}},
  author    = {Karimadini, Arash and Karimadini, Kaveh},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/arashlove/aemo-nem-lor}
}
```

The adapters were trained on the
[`arashlove/aemo-nem-lor`](https://huggingface.co/datasets/arashlove/aemo-nem-lor)
dataset (CC BY 4.0, citation required).
