# Running LORTrace on RunPod

Step-by-step guide to reproducing the results on a rented RunPod GPU: download the
released data and LoRA adapters from Hugging Face, rescore all nine backbones, and
build the metrics and figures. Every command is run in the pod's terminal.

This uses `--mode rescore`: the released adapters are used as-is, so **no training
is needed**. (`--mode retrain` is only for checking that training itself
reproduces, and takes far longer.)

| `--mode` | What it does | Needs |
|---|---|---|
| `evaluate` | Metrics + plots from the released scores in `results/` | CPU only |
| `rescore` | Score with the released adapters, then evaluate | GPU, data + adapters |
| `retrain` | Train new adapters, then score and evaluate | GPU, data |

---

## 1. Create the pod and check it

1. **Pods → Deploy**, choose a GPU. An **L40S (48 GB)** is the reference hardware
   and the cheapest option for the full run (scoring runs one window at a time, so
   larger GPUs such as the H100 finish sooner but cost more overall). An **A40** is
   a cheaper fallback. Any NVIDIA GPU with 24 GB+ works.
   AMD GPUs (MI300X/MI350X) are not supported.
2. **Template:** RunPod PyTorch, any version with CUDA 12.8 or newer.
3. **Storage:** ~50 GB is plenty.
4. **Deploy**, wait for **Running**, then **Connect → Web Terminal** (or SSH).

```bash
nvidia-smi
df -h /workspace
```

Check: the GPU name, `CUDA Version` **≥ 12.8** (a newer version such as 13.x is
fine), and enough free space on `/workspace`.

> If `df` shows `/workspace` on `overlay` mounted at `/`, there is no separate
> persistent volume: everything is lost when the pod is stopped or terminated.
> That is fine for a single run; just copy the results off (step 9) before
> stopping.

## 2. Set the cache folders and install tools

```bash
cat >> ~/.bashrc <<'EOF'
export UV_CACHE_DIR=/workspace/.cache/uv
export HF_HOME=/workspace/.cache/huggingface
export PATH="$HOME/.local/bin:$PATH"
EOF
source ~/.bashrc

apt-get update && apt-get install -y tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv --version
tmux -V
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

## 3. Start a tmux session

Keeps the job running if the browser tab closes or the connection drops.

```bash
tmux new -s lor
```

Run all remaining steps inside it.

| Action | Keys |
|---|---|
| Detach (leave it running) | `Ctrl+b`, then `d` |
| Reattach | `tmux attach -t lor` |
| Scroll output | `Ctrl+b`, then `[`; arrows/PgUp; `q` to exit |

## 4. Clone the repo and install the environment

```bash
cd /workspace
git clone https://github.com/arashlove/LORTrace.git
cd LORTrace
uv sync --extra gpu
```

`uv sync` installs Python 3.10 and the pinned GPU stack (PyTorch 2.8 cu128,
Unsloth, Transformers, PEFT, ...) into `.venv`; it takes a few minutes. The
template's own PyTorch is not used.

Check that PyTorch sees the GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.8.0+cu128 True NVIDIA A100 80GB PCIe` (or your GPU's name).

## 5. Download the data and adapters from Hugging Face

```bash
uv run python scripts/download_assets.py
```

| Asset | Hugging Face repo | Saved to |
|---|---|---|
| Data: 3 `.csv.gz` files, ~80 MB (sha256-verified) | [`arashlove/aemo-nem-lor`](https://huggingface.co/datasets/arashlove/aemo-nem-lor) | `data/raw/` |
| LoRA adapters for all 9 backbones, ~760 MB | [`arashlove/lortrace-adapters`](https://huggingface.co/arashlove/lortrace-adapters) | `weights/<model>/adapter/` |

Verify:

```bash
ls -lh data/raw/*.csv.gz                              # 3 files
ls weights/*/adapter/adapter_config.json | wc -l      # 9
```

If you get a `401` or "gated repo" error, log in with a read token from
<https://huggingface.co/settings/tokens> and re-run the download:

```bash
uv run huggingface-cli login
```

The **base models** (`unsloth/...-bnb-4bit`) are downloaded automatically the first
time each backbone is scored. To fetch them all up front instead (~20 GB):

```bash
for f in configs/models/*.yaml; do
  m=$(grep -E '^\s+name:' "$f" | head -1 | awk '{print $2}')
  uv run huggingface-cli download "$m"
done
```

## 6. (Optional) Test run on one small model

```bash
uv run python scripts/reproduce.py --mode rescore --models gemma-3-270m
```

This runs preprocess → score → evaluate and should end with `gemma-3-270m  ok`.
Compare with the released reference metrics (ROC-AUC ≈ 0.608, AP ≈ 0.0143):

```bash
grep -E '"roc_auc"|"average_precision"' \
  outputs/rescore/gemma-3-270m/evaluation/*.json \
  results/gemma-3-270m/metrics_actual_lor.json
```

The speed on the `score` progress bar gives a feel for how long the full run will
take. Small differences from the reference numbers across GPUs are expected.

## 7. Full run: all 9 models

```bash
cd /workspace/LORTrace
uv run python scripts/reproduce.py --mode rescore 2>&1 | tee rescore.log
```

Backbones run smallest first; the two 7B models (`mistral-7b-v0.3`,
`Cosmos-Reason1-7B`) come last and take most of the time. You can detach
(`Ctrl+b`, `d`) and close the browser.

Check progress from any terminal:

```bash
grep "===" /workspace/LORTrace/rescore.log    # current model, e.g. [4/9]
tail -5 /workspace/LORTrace/rescore.log
nvidia-smi                                     # GPU should be busy
```

If the run stops for any reason, run the same command again: scoring checkpoints
every 512 windows and resumes (`--force` restarts from scratch).

When it finishes, a summary lists every backbone as `ok`. Per-model metrics and
plots are in `outputs/rescore/<model>/evaluation/`.

## 8. (Optional) Summary figures and tables

Built with [Quarto](https://quarto.org) from `report/results.qmd`:

```bash
wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.42/quarto-1.6.42-linux-amd64.deb
apt-get install -y ./quarto-1.6.42-linux-amd64.deb
uv sync --extra gpu --group report
RESULTS_DIR=outputs/rescore uv run quarto render report/results.qmd
```

Keep `--extra gpu` in that `uv sync`; without it uv removes the GPU packages.
Any recent Quarto Linux `.deb` from the releases page works. Alternatively, copy
`outputs/rescore/` to your own machine and render there.

## 9. Copy the results off the pod

```bash
cd /workspace/LORTrace
tar czf /workspace/lortrace_rescore.tar.gz outputs/rescore report/ rescore.log
runpodctl send /workspace/lortrace_rescore.tar.gz
```

`runpodctl send` prints a one-time code; on your machine run
`runpodctl receive <code>`. (Or download the archive through the pod's Jupyter
file browser if your template has one.)

## 10. Stop the pod

Once the results are safely on your machine, **stop or terminate the pod** so
billing stops.
