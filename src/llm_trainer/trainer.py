from __future__ import annotations

import importlib
import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any

from .dataloader import SequenceDataLoader, load_token_ids, load_tokenizer
from .model import GPTLanguageModel
from .run_metadata import RunFiles, load_state, update_run_state
from .telemetry import collect_host_telemetry


def _torch():
    return importlib.import_module("torch")


def _append_log(run_dir: Path, payload: dict[str, Any]) -> None:
    log_path = run_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timer_metrics(
    *,
    total_steps: int,
    global_step: int,
    baseline_step: int,
    baseline_elapsed_seconds: float,
    started_monotonic: float,
) -> dict[str, Any]:
    elapsed = baseline_elapsed_seconds + max(0.0, monotonic() - started_monotonic)
    completed_in_session = max(0, global_step - baseline_step)
    remaining_steps = max(0, total_steps - global_step)

    remaining_seconds: float | None = None
    eta_at: str | None = None
    if completed_in_session > 0 and elapsed > baseline_elapsed_seconds and remaining_steps > 0:
        session_elapsed = elapsed - baseline_elapsed_seconds
        steps_per_second = completed_in_session / session_elapsed if session_elapsed > 0 else 0.0
        if steps_per_second > 0:
            remaining_seconds = remaining_steps / steps_per_second
            eta_at = (
                datetime.now(UTC) + timedelta(seconds=remaining_seconds)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    elif remaining_steps == 0:
        remaining_seconds = 0.0
        eta_at = _utc_now_iso()

    return {
        "elapsed_seconds": elapsed,
        "remaining_seconds": remaining_seconds,
        "eta_at": eta_at,
    }


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> None:
    torch = _torch()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
        },
        checkpoint_path,
    )


def train_loop(
    *,
    config: dict[str, Any],
    run_files: RunFiles,
    tokenized_train_path: str | Path,
    tokenized_validation_path: str | Path,
    tokenizer_path: str | Path,
    checkpoint_dir: str | Path = "checkpoints",
    start_epoch: int = 0,
    total_epochs: int | None = None,
    optimizer_state: dict[str, Any] | None = None,
    model_state: dict[str, Any] | None = None,
    global_step: int = 0,
) -> dict[str, Any]:
    torch = _torch()
    device = config.get("runtime", {}).get("device", "cpu")

    training_cfg = config["training"]
    model_cfg = config["model"]
    seq_length = int(training_cfg["seq_length"])
    batch_size = int(training_cfg["batch_size"])
    epochs = int(total_epochs if total_epochs is not None else training_cfg["epochs"])
    lr = float(training_cfg["learning_rate"])
    seed = int(training_cfg.get("seed", 42))
    save_every_epochs = int(training_cfg.get("save_every_epochs", 1))
    precision = str(training_cfg.get("precision", "off")).lower()
    grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
    pin_memory = bool(training_cfg.get("dataloader_pin_memory", False))
    log_every_steps = max(1, int(training_cfg.get("log_every_steps", 10)))

    tokenizer = load_tokenizer(tokenizer_path)
    train_ids = load_token_ids(tokenized_train_path)
    validation_ids = load_token_ids(tokenized_validation_path)
    train_loader = SequenceDataLoader(
        train_ids,
        seq_length=seq_length,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = SequenceDataLoader(
        validation_ids,
        seq_length=seq_length,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    model = GPTLanguageModel(
        vocab_size=len(tokenizer.vocab),
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        d_ff=int(model_cfg["d_ff"]),
        max_seq_length=seq_length,
    ).to(device)
    if model_state is not None:
        model.load_state_dict(model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    update_run_state(
        state_path=run_files.state_path,
        status="running",
        metrics={
            "device": device,
            "precision": precision,
            "grad_accum_steps": grad_accum_steps,
            "dataloader_pin_memory": pin_memory,
            "dataloader_workers": int(training_cfg.get("dataloader_workers", 0)),
            "dataloader_prefetch_factor": int(training_cfg.get("dataloader_prefetch_factor", 2)),
        },
    )
    state = load_state(run_files.state_path)
    baseline_elapsed_seconds = float(state.get("elapsed_seconds", 0.0) or 0.0)
    baseline_step = global_step
    started_monotonic = monotonic()
    latest_loss: float | None = None
    latest_val_loss: float | None = None
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            input_ids = torch.tensor(batch.input_ids, dtype=torch.long)
            labels = torch.tensor(batch.labels, dtype=torch.long)
            if pin_memory and str(device).startswith("cuda"):
                input_ids = input_ids.pin_memory()
                labels = labels.pin_memory()
            input_ids = input_ids.to(device=device, non_blocking=pin_memory)
            labels = labels.to(device=device, non_blocking=pin_memory)

            if precision in {"fp16", "bf16"} and str(device).startswith("cuda"):
                dtype = torch.float16 if precision == "fp16" else torch.bfloat16
                autocast = torch.autocast(device_type="cuda", dtype=dtype)
            else:
                autocast = nullcontext()

            with autocast:
                _, loss = model(input_ids, labels)
            assert loss is not None
            normalized_loss = loss / grad_accum_steps
            normalized_loss.backward()
            global_step += 1
            latest_loss = float(loss.detach().cpu().item())
            if global_step % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % log_every_steps == 0:
                step_payload = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": latest_loss,
                }
                step_payload.update(
                    _timer_metrics(
                        total_steps=total_steps,
                        global_step=global_step,
                        baseline_step=baseline_step,
                        baseline_elapsed_seconds=baseline_elapsed_seconds,
                        started_monotonic=started_monotonic,
                    )
                )
                step_payload.update(collect_host_telemetry(device=device))
                _append_log(run_files.run_dir, step_payload)
                update_run_state(state_path=run_files.state_path, metrics=step_payload)

        if global_step % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                input_ids = torch.tensor(batch.input_ids, dtype=torch.long, device=device)
                labels = torch.tensor(batch.labels, dtype=torch.long, device=device)
                _, val_loss = model(input_ids, labels)
                assert val_loss is not None
                val_losses.append(float(val_loss.detach().cpu().item()))
        latest_val_loss = float(sum(val_losses) / len(val_losses)) if val_losses else latest_loss

        checkpoint_root = Path(checkpoint_dir) / run_files.run_id
        latest_checkpoint = checkpoint_root / "latest.pt"
        _save_checkpoint(
            checkpoint_path=latest_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            config=config,
        )
        if (epoch + 1) % save_every_epochs == 0:
            periodic = checkpoint_root / f"epoch-{epoch + 1}.pt"
            _save_checkpoint(
                checkpoint_path=periodic,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                config=config,
            )

        step_payload = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": latest_loss,
            "val_loss": latest_val_loss,
            "checkpoint": str(latest_checkpoint),
        }
        step_payload.update(
            _timer_metrics(
                total_steps=total_steps,
                global_step=global_step,
                baseline_step=baseline_step,
                baseline_elapsed_seconds=baseline_elapsed_seconds,
                started_monotonic=started_monotonic,
            )
        )
        step_payload.update(collect_host_telemetry(device=device))
        _append_log(run_files.run_dir, step_payload)
        update_run_state(state_path=run_files.state_path, metrics=step_payload)
        print(
            f"epoch={epoch + 1} step={global_step} "
            f"train_loss={latest_loss:.4f} val_loss={latest_val_loss:.4f}"
        )

    update_run_state(
        state_path=run_files.state_path,
        status="completed",
        metrics={
            "elapsed_seconds": baseline_elapsed_seconds + max(0.0, monotonic() - started_monotonic),
            "remaining_seconds": 0.0,
            "eta_at": _utc_now_iso(),
        },
    )
    return {
        "epoch": epochs,
        "global_step": global_step,
        "train_loss": latest_loss,
        "val_loss": latest_val_loss,
    }
