# OpenClawTest — Local PyTorch LLM Trainer (CLI + TUI)

A local-first project to train a small GPT-style language model on WikiText-2 with:
- CUDA-first device selection (CPU fallback)
- Background training runs
- Status tracking
- Resume training for more epochs
- Prompt-based text generation
- Textual TUI monitoring

---

## Features

- **Train** on WikiText-2 (`datasets`)
- **Tokenize + dataloader** pipeline
- **GPT-style model** in PyTorch
- **Checkpointing** (`latest.pt` + periodic epoch files)
- **Background run mode** + run metadata in `runs/<run_id>/`
- **Status command** for quick run inspection
- **Live GPU telemetry** in `status` + TUI (`n/a` safe on CPU/no-NVML)
- **Resume** from checkpoint with `--more-epochs`
- **Generate** text from trained checkpoints
- **TUI monitor + actions** via Textual (start/resume/generate)

---

## Project Structure

```text
OpenClawTest/
├── configs/
│   └── default.toml
├── src/
│   └── llm_trainer/
│       ├── cli.py
│       ├── data.py
│       ├── tokenization.py
│       ├── dataloader.py
│       ├── model.py
│       ├── trainer.py
│       ├── background.py
│       └── run_metadata.py
├── tests/
├── data/          # local dataset artifacts (gitignored)
├── checkpoints/   # model checkpoints (gitignored)
├── runs/          # run state/logs (gitignored)
└── TASKS.md
```

---

## Requirements

- Python 3.11+
- Optional CUDA-capable GPU for acceleration

---

## Setup

```bash
cd /home/theo/OpenClawTest
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install torch datasets textual
```

If `llm-trainer` is not found, use module mode:

```bash
PYTHONPATH=src python -m llm_trainer --help
```

---

## Configuration

Default config: `configs/default.toml`

Key fields:
- `training.epochs` (default: 3)
- `training.batch_size`
- `training.seq_length`
- `training.learning_rate`
- `training.precision` (`off`/`fp16`/`bf16`)
- `training.grad_accum_steps`
- `training.dataloader_workers`
- `training.dataloader_prefetch_factor`
- `training.dataloader_pin_memory`
- `data.dataset_name` (wikitext-2)
- `device.preference` (`auto`, `cuda`, `cuda:N`, or GPU hint like `A30`)
- `device.strict` (`false` by default)

---

## Usage

### 1) Start training (background)

```bash
PYTHONPATH=src llm-trainer train --config configs/default.toml
# Optional overrides:
#   --device auto|cpu|cuda|cuda:0|A30
#   --strict-device
#   --precision off|fp16|bf16
#   --grad-accum-steps 2
#   --dataloader-workers 4 --dataloader-prefetch-factor 2 --dataloader-pin-memory
```

This creates a run ID and writes run metadata under `runs/<run_id>/`.

### 2) Check status

```bash
PYTHONPATH=src llm-trainer status
# or
PYTHONPATH=src llm-trainer status --run-id <run_id>
# includes GPU util/memory/temp/power when available
```

### 3) Resume for more epochs

```bash
PYTHONPATH=src llm-trainer resume --run-id <run_id> --more-epochs 3
```

### 4) Generate text

```bash
PYTHONPATH=src llm-trainer generate \
  --run-id <run_id> \
  --prompt "Once upon a time" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 40
```

### 5) Open TUI monitor

```bash
PYTHONPATH=src llm-trainer tui
# keys: n start new run | u resume selected | g generate from selected
# tuning shortcuts: [ / ] epochs, d device cycle, p prompt cycle
```

---

## Logs and Artifacts

- **Run state**: `runs/<run_id>/state.json`
- **Run metadata**: `runs/<run_id>/meta.json`
- **Worker logs**: `runs/<run_id>/worker.log`
- **Training logs**: `runs/<run_id>/train.log`
- **Checkpoints**: `checkpoints/<run_id>/latest.pt`, `epoch-<n>.pt`

---

## Validation

Use this command before committing:

```bash
PYTHONPATH=src ./.venv/bin/ruff check . && PYTHONPATH=src ./.venv/bin/pytest -q
```

---

## Notes

- `--more-epochs` is supported on **resume**, not on **train**.
- Data/checkpoints/runs are intentionally gitignored.
- CUDA is used automatically when available; otherwise CPU is used.

---

## Roadmap

See `TASKS.md` for completed tasks and upcoming work (including ETA timer and improved TUI).
