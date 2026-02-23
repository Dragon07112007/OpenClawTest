# TASKS.md — LLM Transformer (PyTorch) Roadmap

Project goal: Build a local, trainable Transformer with CLI + TUI, background training, resume/continue support, and prompt-based generation.

## Principles
- Keep tasks small and incremental.
- Local-first execution.
- Device selection: prefer CUDA when available, else CPU fallback.
- Every task should include tests or validation commands.

## Baseline validation command
```bash
PYTHONPATH=src ./.venv/bin/ruff check . && PYTHONPATH=src ./.venv/bin/pytest -q
```

---

## Task 1 — CLI + Config Scaffold
**Goal:** Establish project structure and command entrypoints.

### Deliverables
- Folder structure:
  - `src/llm_trainer/`
  - `tests/`
  - `configs/`
  - `data/`
  - `checkpoints/`
  - `runs/`
- CLI commands (stubbed if needed):
  - `train`
  - `status`
  - `resume`
  - `generate`
- Default config file (YAML/TOML) with:
  - model size params
  - batch size
  - seq length
  - learning rate
  - epochs
  - dataset name/path
  - device preference

### Acceptance
- CLI help works for all commands.
- Config loads successfully in a smoke test.

---

## Task 2 — Device Manager + Run Metadata
**Goal:** Robust runtime environment selection and run tracking.

### Deliverables
- Device detection utility:
  - CUDA available → use `cuda`
  - else use `cpu`
- Device info printout in CLI startup.
- Run directory creation in `runs/<run_id>/` with:
  - `meta.json` (start time, config, device)
  - `state.json` (status lifecycle: queued/running/completed/failed)

### Acceptance
- Unit tests for device selection logic.
- Smoke test confirming run metadata creation.

---

## Task 3 — Dataset Download + Prep (WikiText-2) [done]
**Goal:** Get reproducible local dataset pipeline.

### Deliverables
- `train` pipeline step to download/load `wikitext-2`.
- Cache/raw data under `data/`.
- Train/validation split handling.
- Basic data integrity checks.

### Acceptance
- Command runs end-to-end and stores local artifacts.
- Test verifies non-empty train/val samples.

---

## Task 4 — Tokenization + DataLoader [done]
**Goal:** Convert text to model-ready batches.

### Decision
Use Hugging Face stack (`datasets` + tokenizer tooling) for robustness and extensibility.

### Deliverables
- Tokenizer setup and persistence.
- Tokenized dataset cache.
- PyTorch Dataset + DataLoader with configurable:
  - sequence length
  - batch size
  - shuffle

### Acceptance
- Batch output shapes are correct.
- Deterministic behavior with seed option.

---

## Task 5 — Minimal GPT-style Transformer
**Goal:** Implement trainable baseline model.

### Deliverables
- PyTorch model module:
  - token + positional embeddings
  - transformer blocks
  - LM head
- Forward pass returning logits + optional loss.

### Acceptance
- Unit test for expected tensor shapes.
- One train step executes without errors.

---

## Task 6 — Training Loop v1
**Goal:** Train and checkpoint reliably.

### Deliverables
- Epoch/step training loop.
- Validation loop.
- Logging to console + run files.
- Checkpoint save policy (latest + periodic).

### Acceptance
- Training completes for a short run.
- Checkpoint files are written and loadable.

---

## Task 7 — Background Training + Status Command
**Goal:** Run training detached and inspect live progress.

### Deliverables
- Start training in background process.
- Persist progress metrics in run state/log file.
- `status` command to inspect:
  - current epoch/step
  - latest loss/val loss
  - device
  - PID/process state

### Acceptance
- Start run, close shell, status remains available.
- Status reflects progressing metrics during training.

---

## Task 8 — Resume / Continue for N More Epochs
**Goal:** Continue from a checkpoint safely.

### Deliverables
- `resume` command from chosen checkpoint or latest.
- `--more-epochs N` flag to extend training horizon.
- Restore optimizer/scheduler state for proper continuation.

### Acceptance
- Resume run continues from previous step/epoch.
- Training proceeds for exactly N additional epochs.

---

## Task 9 — Prompt-based Generation
**Goal:** Use trained model for text continuation.

### Deliverables
- `generate` command:
  - `--prompt`
  - `--max-new-tokens`
  - `--temperature`
  - `--top-k`
  - checkpoint selection
- Token decode to readable output.

### Acceptance
- Given prompt, model returns continuation text.
- Works with CPU and CUDA (if available).

---

## Task 10 — Textual TUI (Live Monitor)
**Goal:** Terminal UI for training and monitoring.

### Deliverables
- Textual app showing:
  - run status
  - epoch/step
  - loss/val loss
  - device
  - ETA
- Basic actions:
  - list runs
  - watch active run
  - launch resume/generate shortcuts (optional in v1)

### Acceptance
- TUI reads run state files and updates live.
- No crash when no active run exists.

---

## Stretch Tasks (After v1)
- Gradient accumulation / mixed precision.
- Better sampling strategies (top-p, repetition penalty).
- Multi-dataset support.
- Config profiles for small/medium models.
- Export/inference packaging.

---

## Workflow Rule
When user says **"start"**, begin at the next incomplete task with:
1. Implement
2. Lint/test
3. Auto-fix failures
4. Report concise result
