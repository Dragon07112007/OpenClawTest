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

## Task 5 — Minimal GPT-style Transformer [done]
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

## Task 6 — Training Loop v1 [done]
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

## Task 7 — Background Training + Status Command [done]
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

## Task 8 — Resume / Continue for N More Epochs [done]
**Goal:** Continue from a checkpoint safely.

### Deliverables
- `resume` command from chosen checkpoint or latest.
- `--more-epochs N` flag to extend training horizon.
- Restore optimizer/scheduler state for proper continuation.

### Acceptance
- Resume run continues from previous step/epoch.
- Training proceeds for exactly N additional epochs.

---

## Task 9 — Prompt-based Generation [done]
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

## Task 10 — Textual TUI (Live Monitor) [done]
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

## Next Tasks (Requested)

## Task 11 — ETA/EST Training Timer [done]
**Goal:** Show realistic progress timing for active runs.

### Deliverables
- Compute and persist estimated remaining time based on step/epoch throughput.
- Add elapsed time, ETA timestamp, and remaining duration to run state.
- Surface ETA/EST fields in `status` command output.

### Acceptance
- During training, `status` displays elapsed + ETA consistently.
- ETA updates over time and does not crash when history is short.

---

## Task 12 — Improved TUI [done]
**Goal:** Make the Textual UI more useful for daily training workflows.

### Deliverables
- Better run list view (status, epoch, loss, device, ETA).
- Live detail panel for selected run (logs + metrics trends).
- Clear keybinds/help footer and refresh controls.
- Graceful empty/error states.

### Acceptance
- TUI is stable with 0, 1, or many runs.
- User can quickly inspect active run health and progress from one screen.

---

## Task 13 — Optional Device Override (A30 or Auto) [done]
**Goal:** Allow explicit GPU selection (e.g., A30) while preserving portable behavior on systems without that GPU.

### Deliverables
- Add optional config/CLI device override modes:
  - `auto` (default, current behavior)
  - explicit device target (e.g., `cuda`, `cuda:0`, optional GPU name hint like `A30`)
- Implement graceful fallback when requested GPU is unavailable:
  - clear warning
  - fallback to best available device (or fail-fast if strict mode is enabled)
- Persist selected + requested device information in run metadata/state.

### Acceptance
- On A30 machine, user can opt-in to target it explicitly.
- On non-A30 machine, same config still runs without crashing (with clear messaging).
- Unit tests cover both available/unavailable override scenarios.

---

## Task 14 — Live GPU Utilization Metrics (btop-like visibility) [done]
**Goal:** Expose real-time GPU utilization and memory stats in CLI/TUI monitoring, similar to what users watch in btop.

### Deliverables
- Add runtime GPU telemetry collection (prefer `pynvml`; fallback strategy when unavailable).
- Persist key metrics in run state/log snapshots:
  - GPU utilization %
  - VRAM used/total
  - (optional) power/temp if available
- Surface these fields in:
  - `status` command output
  - TUI run list/detail panel
- Keep behavior safe on CPU-only/non-NVIDIA systems (show `n/a`, no crash).

### Acceptance
- During training on NVIDIA GPU, `status` and TUI show changing utilization/memory values.
- On systems without NVIDIA/NVML, commands remain functional with graceful `n/a` display.
- Tests cover telemetry parsing + no-GPU fallback paths.

---

## Task 15 — Throughput-Oriented GPU Tuning Controls [done]
**Goal:** Help users efficiently consume available GPU capacity without hardcoding hardware-specific defaults.

### Deliverables
- Add configurable performance knobs in config/CLI (as supported by current codebase), such as:
  - mixed precision mode (`off`/`fp16`/`bf16` where supported)
  - DataLoader worker/prefetch/pin-memory options
  - optional gradient accumulation for larger effective batch size
- Add simple guidance output (or docs section) for increasing utilization safely.
- Ensure defaults remain conservative and portable.

### Acceptance
- User can opt into more aggressive settings on high-memory GPUs (e.g., A30).
- Same config family still runs on lower-end or CPU systems with fallback/default behavior.
- Validation/tests confirm no regression in existing training/status commands.

---

## Task 16 — TUI-Driven Training Start + Option Selection [done]
**Goal:** Let users start/resume training and choose key run options directly from the TUI, without dropping to CLI for common flows.

### Deliverables
- Add TUI actions/forms to:
  - start a new training run
  - resume an existing run
  - select/edit key options before launch (e.g., config profile, epochs, batch size, seq length, device mode, precision)
- Add clear validation and confirmation UX before launch.
- Show launch result feedback (run id, pid, early status/errors) in the TUI.
- Keep CLI as source of truth; TUI should call into shared application logic (no duplicated training orchestration).

### Acceptance
- User can launch a run end-to-end from TUI on a fresh terminal session.
- Invalid/missing options are caught with actionable error messages.
- Works on GPU and CPU-only systems with graceful fallbacks.
- Tests cover the TUI action paths and launch parameter mapping.

---

## Task 17 — TUI Generation from Selected Trained Model [done]
**Goal:** Generate text directly from the TUI using a user-selected trained run/checkpoint.

### Deliverables
- Add a TUI flow to:
  - browse/select a completed (or user-chosen) run/checkpoint
  - enter generation options (`prompt`, `max_new_tokens`, `temperature`, `top_k`)
  - execute generation and render output in a readable panel
- Show clear errors for missing checkpoints or invalid generation parameters.
- Keep generation logic shared with existing CLI command to avoid drift.

### Acceptance
- User can pick a trained model from TUI and generate output end-to-end.
- Generation options can be adjusted per run without editing files manually.
- On systems without suitable checkpoints, TUI shows a graceful guidance message.
- Tests cover parameter mapping and key success/error UI states.

---

## Task 18 — TUI Markup Safety Regression (detail/help panels) [done]
**Goal:** Prevent Rich/Textual markup parsing crashes when dynamic strings include accidental markup-like tokens.

### Deliverables
- Ensure detail/help panel updates are markup-safe by escaping dynamic TUI strings before render.
- Cover dangerous content sources (run metadata, action text, generation prompt/help text) so bracketed tokens like `[/]` and unmatched tags cannot crash rendering.
- Add regression + edge-case tests for markup-like input while preserving displayed text UX.

### Acceptance
- Updating detail/help panels does not raise markup parse errors for strings containing Rich/Textual-like markup tokens.
- Existing panel content and interactions remain unchanged for normal inputs.
- Validation command stays green (`ruff` + `pytest`).

---

## Task 19 — TUI Layout v2 (Four-Zone Dashboard) [done]
**Goal:** Restructure the TUI into a comprehensive multi-panel dashboard.

### Deliverables
- Introduce a stable four-zone layout:
  - top-right: runs panel
  - center: training launcher panel
  - bottom-left: generation prompt panel
  - far-right: model manager panel
- Keep existing refresh loop and navigation robust across terminal sizes.
- Ensure panel titles/borders and visual hierarchy are clear.

### Acceptance
- Layout renders reliably in typical terminal sizes.
- No overlap/flicker/crash while refreshing.
- Empty states are legible in each panel.

---

## Task 20 — Runs Panel Upgrade (Top-Right KPIs) [done]
**Goal:** Show run health at a glance with the requested training and device fields.

### Deliverables
- Runs table columns include:
  - run id, status, epoch, step, loss, device, ETA/remaining, GPU usage
- Highlight running and newest runs first.
- Keep selection behavior predictable when list updates.

### Acceptance
- Works for 0/1/many runs.
- Values refresh without UI instability.
- No regressions in existing run detail behavior.

---

## Task 21 — System Utilization Overview Panel [done]
**Goal:** Provide btop-like system usage visibility in the TUI.

### Deliverables
- Add a dedicated utilization panel with:
  - GPU util %, VRAM used/total (plus temp/power when available)
  - CPU utilization
  - RAM usage
- Reuse existing telemetry paths where possible.
- Graceful fallbacks (`n/a`) on unsupported systems.

### Acceptance
- Panel updates live during training.
- Works without NVIDIA/NVML and on CPU-only systems.
- No crashes when telemetry sources are partially unavailable.

---

## Task 22 — Generation Prompt Workspace (Bottom-Left) [done]
**Goal:** Allow entering prompt text directly in TUI and generating from selected model.

### Deliverables
- Add editable prompt input area.
- Add generation controls (max tokens, temperature, top-k).
- Add output display pane/section for generation results.
- Validate parameters and show actionable errors.

### Acceptance
- Keyboard-first prompt-to-output flow works end-to-end.
- Handles long/special-character prompts safely.
- Clear feedback for missing model/checkpoint selection.

---

## Task 23 — Training Launcher Workspace (Center Panel) [done]
**Goal:** Start/resume training from a dedicated control panel between run list and generation workspace.

### Deliverables
- Add start/resume controls in center panel.
- Expose key launch options:
  - config/profile, epochs, batch size, seq length, device mode, precision/tuning options
- Add validation + confirmation before launch.
- Show immediate launch feedback (run id/pid/errors).

### Acceptance
- User can launch or resume training fully from TUI.
- Invalid inputs are blocked with clear messages.
- Launched runs appear in monitoring panel promptly.

---

## Task 24 — Model Manager Panel (Far-Right) [done]
**Goal:** Manage trained models/checkpoints and clearly identify latest artifacts.

### Deliverables
- Add model/checkpoint list with metadata.
- Clearly mark latest model/checkpoint.
- Add management actions (safe subset): select active model, inspect metadata, archive/delete with confirmation.

### Acceptance
- Latest model is always identifiable.
- Selected model is used by generation workflow.
- Destructive actions require explicit confirmation.

---

## Task 25 — Cross-Panel Workflow Integration [done]
**Goal:** Wire all panels into one coherent workflow.

### Deliverables
- Training launcher updates runs panel in near real-time.
- Newly completed checkpoints surface in model manager.
- Model selection propagates into generation panel.
- Unified status/error messaging across panels.

### Acceptance
- End-to-end flow works from one TUI session without dropping to CLI for common operations.
- State transitions remain consistent across refresh cycles.

---

## Task 26 — TUI Reliability & Test Hardening (v2) [done]
**Goal:** Make the redesigned TUI resilient with strong regression coverage.

### Deliverables
- Expand tests for:
  - layout states
  - panel selection/navigation
  - launch/generate/model-management action mapping
  - telemetry fallbacks and empty/error states
  - markup-safety regressions for all dynamic text surfaces
- Ensure lint/test commands remain green.

### Acceptance
- No known crash paths in normal flows.
- Validation command passes consistently (`ruff` + `pytest`).
- Existing CLI behavior remains backward-compatible.

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
