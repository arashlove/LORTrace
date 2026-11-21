# LORTrace

Language models for electricity-market stress: detecting and explaining
**Lack-of-Reserve (LOR)** conditions in the Australian National Electricity
Market (NEM) from AEMO five-minute dispatch data.

> Arash Karimadini, Kaveh Karimadini, Xuyun Zhang, Usman Naseem, Amin Beheshti
> School of Computing, Macquarie University

| Asset | Where |
|---|---|
| Code, configs, reference scores and metrics | this repository |
| Data: AEMO NSW1 series + NEM LOR labels (~80 MB) | [`arashlove/aemo-nem-lor`](https://huggingface.co/datasets/arashlove/aemo-nem-lor) (Hugging Face dataset) |
| LoRA adapters for all 9 backbones (~760 MB) | [`arashlove/lortrace-adapters`](https://huggingface.co/arashlove/lortrace-adapters) (Hugging Face model) |

Repo ids and pinned revisions live in [`configs/assets.yaml`](configs/assets.yaml);
[`scripts/download_assets.py`](scripts/download_assets.py) fetches and verifies them.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10 (uv installs it if needed).

```bash
git clone https://github.com/arashlove/LORTrace.git
cd LORTrace

uv sync                 # CPU-only: preprocessing, evaluation, figures, tests
uv sync --extra gpu     # + PyTorch (CUDA 12.8), Unsloth, Transformers, TRL, PEFT
```

The GPU extra is locked for Linux and Windows x86-64 with an NVIDIA GPU
(tested on an RTX 3050 6 GB and Colab T4/L4/A100). Exact versions are pinned
in [`uv.lock`](uv.lock).

---

## Overview

| Pipeline | Code | Status |
|---|---|---|
| Unsupervised LLM anomaly detection | [`lortrace.benchmark`](src/lortrace/benchmark/) | Implemented: [reproduce](#reproducing-the-results) |
| Mechanistic interpretability of the fine-tuned detectors | `lortrace.interp` | Planned extension |

We adapt the AnoLLM serialize → fine-tune → likelihood recipe
([Tsai et al., ICLR 2025](https://github.com/amazon-science/AnoLLM-large-language-models-for-tabular-anomaly-detection)) from
independent tabular rows to ordered five-minute market windows from the
NEM's NSW1 region. Nine pretrained backbones (135M–7B parameters) are
LoRA-fine-tuned without supervision and score each one-hour window by
value-token negative log-likelihood. Scores are evaluated against reconstructed
**Actual LOR** events, with AEMO **Forecast LOR** notices as an institutional
baseline.

### Main results

Pooled validation + test set: 22,602 one-hour windows, 239 Actual LOR (1.06%).
Recall at model-specific score quantiles; Forecast LOR recall = 0.243.

| Backbone | ROC-AUC | AP | Recall q90 | Recall q95 | Recall q99 |
|---|---|---|---|---|---|
| SmolLM2 135M | 0.626 | 0.0136 | 0.117 | 0.025 | 0.000 |
| Gemma 3 270M | 0.608 | 0.0143 | 0.184 | 0.054 | 0.000 |
| SmolLM2 360M | 0.574 | 0.0126 | 0.130 | 0.038 | 0.000 |
| Qwen2.5 0.5B | 0.660 | 0.0182 | 0.226 | 0.071 | 0.013 |
| Qwen2.5 1.5B | 0.671 | 0.0180 | 0.213 | 0.096 | 0.013 |
| SmolLM 1.7B | 0.640 | 0.0154 | 0.146 | 0.050 | 0.000 |
| Gemma 2 2B | 0.665 | 0.0255 | 0.276 | 0.167 | 0.042 |
| **Mistral 7B v0.3** | **0.692** | **0.0344** | **0.297** | **0.226** | **0.084** |
| Cosmos Reason1 7B | 0.673 | 0.0192 | 0.218 | 0.109 | 0.017 |

### Reproducing the results

Everything runs through [`scripts/reproduce.py`](scripts/reproduce.py), at
three levels of cost. Each run saves the exact config it used to
`outputs/<mode>/<model>/config.yaml`.

#### Level 1: Metrics and figures from the released scores (CPU, minutes)

```bash
uv run python scripts/reproduce.py --mode evaluate
```

Recomputes every metric and per-model plot from `results/<model>/val_test_scores.npz`
into `outputs/evaluate/<model>/evaluation/`.

Summary figures and tables are built by [`report/results.qmd`](report/results.qmd)
(requires [Quarto](https://quarto.org)):

```bash
uv sync --group report
uv run quarto render report/results.qmd                       # from results/
RESULTS_DIR=outputs/rescore uv run quarto render report/results.qmd   # from your own rescoring run
```

#### Level 2: Rescore with the released adapters (GPU, hours)

```bash
uv sync --extra gpu
uv run python scripts/download_assets.py            # data + adapters from the Hugging Face Hub
uv run python scripts/reproduce.py --mode rescore   # all 9 backbones, smallest first
uv run python scripts/reproduce.py --mode rescore --models gemma-3-270m   # or a subset
```

The base checkpoints (`unsloth/*`) are downloaded from the Hub on first use.
Scoring checkpoints every 512 windows; re-running resumes an interrupted run
(`--force` restarts).

#### Reproducibility: what to expect

**Reference hardware.** Our reproduction runs use a RunPod **NVIDIA L40S (48 GB)**
on Linux (CUDA 12.8, PyTorch 2.8.0+cu128, Unsloth 2026.11.4, Transformers 4.56.2,
bitsandbytes 0.50.2, all pinned in [`uv.lock`](uv.lock)). This is the recommended
setup for replication.

**Fixed:** the data (sha256-verified), splits, windowing (12 × 5 min, stride 12),
serialization, the within-step feature permutation (seeded with `seed: 3407`), the
LoRA adapters and the software versions. Window ordering and labels are identical
on every machine.

**Not bit-identical:** scores depend on floating-point GPU kernels (4-bit
dequantization, bf16 matmuls, attention), so different GPU models and drivers give
slightly different numbers. To reproduce the released reference scores as closely as
possible, use the reference hardware and the locked environment above.

**Use Linux with an NVIDIA GPU (Ampere or newer, bf16).** `dtype: null` selects
bf16 on Ampere+ and fp16 on older GPUs (e.g. T4), which can shift scores further.

#### Level 3: Retrain from scratch (GPU, longest)

```bash
uv run python scripts/download_assets.py --data
uv run python scripts/reproduce.py --mode retrain --models SmolLM2-135M
```

Fine-tunes the backbone on the 2015–2021 training windows (20,000 windows,
300 steps, seed 3407), then scores and evaluates. 7B models need ~16 GB+ of
GPU memory.

#### Individual stages

```bash
uv run python scripts/preprocess.py --config configs/models/gemma-3-270m.yaml
uv run python scripts/train.py      --config configs/models/gemma-3-270m.yaml
uv run python scripts/score.py      --config configs/models/gemma-3-270m.yaml --split all
uv run python scripts/evaluate.py   --config configs/models/gemma-3-270m.yaml
uv run python scripts/run_pipeline.py --smoke-test   # CPU-only plumbing check
```

#### Running on RunPod (optional)

Levels 2 and 3 need an NVIDIA GPU. If you don't have one locally, a
[RunPod](https://www.runpod.io/) pod works with the same commands. The quick
version is below; see [`docs/runpod.md`](docs/runpod.md) for the full
step-by-step guide (pod choice, checks, tmux, progress monitoring, copying
results off).

**Pod setup.** Pick any PyTorch template with **CUDA 12.8+** (NVIDIA driver
≥ 570), since the locked PyTorch uses cu128 wheels. We use an **L40S (48 GB)**,
the reference hardware and the cheapest option we found for the full run. Any
24 GB+ GPU (e.g. RTX 4090, L4, A5000, A40) also runs all nine backbones; the 7B
models need ~16 GB+. Attach a network
volume (~50 GB) mounted at `/workspace` so the environment, data and base
checkpoints survive pod restarts.

```bash
cd /workspace

# Keep caches on the persistent volume (the container disk is small)
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface

# Install uv and clone
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone https://github.com/arashlove/LORTrace.git && cd LORTrace

# Environment (Python 3.10 + pinned GPU stack in .venv; the template's own torch is not used)
uv sync --extra gpu

# Data + adapters from the Hugging Face Hub
uv run python scripts/download_assets.py

# Quick check on one small model, then the full run
uv run python scripts/reproduce.py --mode rescore --models gemma-3-270m
uv run python scripts/reproduce.py --mode rescore
```

`--mode rescore` runs preprocess → score → evaluate for each backbone; metrics and
per-model plots land in `outputs/rescore/<model>/evaluation/`. If the pod is
interrupted, re-run the same command and scoring resumes from its last checkpoint.
Run long jobs inside `tmux` (or with `nohup`) so they survive a dropped SSH or
web-terminal session.

To build the summary figures, either install [Quarto](https://quarto.org) on the
pod and run the `quarto render` command from Level 1 with
`RESULTS_DIR=outputs/rescore`, or copy `outputs/rescore/` back to your machine
(e.g. `runpodctl send outputs/rescore`) and render there.

### Method summary

- **Data.** NSW1 5-minute dispatch series 2015–2024 (deduplicated, last record
  per timestamp). Per-step features: RRP, demand, temperature, available
  generation, net interchange, intermittent generation, availability surplus
  (`AVAILABLEGENERATION − TOTALDEMAND`) and UIGF-to-cleared gap; window-level
  hour, weekday, month and battery activity.
- **Splits.** Chronological: train 2015–2021, validation 2022, test
  2023-01-01 → 2024-07-30, with a 12-step embargo at every boundary. Scaling
  statistics are fitted on train only.
- **Windows.** 12 steps (one hour), stride 12, serialized as
  `name is value` with values z-scored to one decimal.
- **Training.** Fully unsupervised LoRA (r = α = 16) on 20,000 training
  windows, 300 steps, effective batch 8; LOR labels never enter the inputs or
  the sample selection.
- **Scoring.** Sum of NLL over value tokens only (K = 1 permutation).
- **Evaluation.** ROC-AUC, AP, recall/precision/F1 at score quantiles
  q ∈ {0.90, 0.95, 0.99} of the pooled validation + test set, and per-severity
  recall (LOR1, LOR2; no Actual LOR3 occurs in the evaluation period).

---

## Planned extension: mechanistic interpretability

Once the detection results above are replicated, the project will be extended
with a mechanistic interpretability analysis of *how* the fine-tuned adapters
represent market stress. It will live in `lortrace.interp`, reuse the same data
and adapters, and add its own configs, scripts and results to this repository.

---

## Repository layout

```
├── configs/
│   ├── base.yaml              # shared experimental settings
│   ├── models/<model>.yaml    # one per backbone (differs only in model + paths)
│   └── assets.yaml            # Hugging Face repo ids, revisions and checksums
├── src/lortrace/
│   └── benchmark/             # detection pipeline: preprocessing, serialization, training, scoring, evaluation
├── scripts/                   # reproduce.py, download_assets.py and the stage CLIs
├── results/<model>/           # released reference scores + metrics
├── report/results.qmd          # summary figures and tables
├── notebooks/lor_labels/      # how the LOR labels were reconstructed from AEMO notices
├── data/                      # data/raw/ is filled by download_assets.py
├── weights/                   # weights/<model>/adapter/ is filled by download_assets.py
└── tests/                     # CPU unit tests: uv run pytest
```

## Tests

```bash
uv run pytest
```

## Citation

If you use this code, data or adapters, please cite the dataset (see also
[`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{karimadini2026aemonemlor,
  title     = {{AEMO NEM Lack-of-Reserve Dataset}},
  author    = {Karimadini, Arash and Karimadini, Kaveh},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/arashlove/aemo-nem-lor}
}
```

## License

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Data:** the curated dataset (reconstructed LOR labels, cleaned and aligned
  series) is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Any use of the data requires citing the dataset above and acknowledging AEMO as
  the source of the underlying market data, which remains © AEMO under AEMO's
  [copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
  See the [dataset card](data/raw/README.md).
- **Adapters:** each follows its base model's license (Apache-2.0, Gemma Terms
  of Use, or NVIDIA Open Model License); see [`weights/README.md`](weights/README.md).
  Please cite the dataset if you use them.
