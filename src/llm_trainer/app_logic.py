from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .background import start_background_training
from .config import load_config
from .data import prepare_wikitext2
from .dataloader import SequenceDataLoader, load_token_ids, tokenize_wikitext2
from .device import DeviceResolutionError, DeviceSelection, resolve_device
from .run_metadata import (
    RunFiles,
    initialize_run,
    load_meta,
    update_run_meta,
    update_run_state,
)


@dataclass(frozen=True)
class TrainLaunchResult:
    run_files: RunFiles
    config: dict[str, Any]
    tokenized: Any
    data_result: Any
    training_mode: str
    pid: int | None
    preview_shape: tuple[int, int]
    selection: DeviceSelection


@dataclass(frozen=True)
class GenerateOptions:
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50


def resolve_device_selection(
    *,
    config: dict[str, Any],
    requested: str | None = None,
    strict: bool = False,
) -> DeviceSelection:
    config_device = str(config.get("device", {}).get("preference", "auto"))
    config_strict = bool(config.get("device", {}).get("strict", False))
    final_request = requested if requested is not None else config_device
    return resolve_device(requested=final_request, strict=(strict or config_strict))


def apply_training_overrides(
    config: dict[str, Any],
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    seq_length: int | None = None,
    precision: str | None = None,
    grad_accum_steps: int | None = None,
    dataloader_workers: int | None = None,
    dataloader_prefetch_factor: int | None = None,
    dataloader_pin_memory: bool | None = None,
) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    if epochs is not None:
        training["epochs"] = int(epochs)
    if batch_size is not None:
        training["batch_size"] = int(batch_size)
    if seq_length is not None:
        training["seq_length"] = int(seq_length)
    if precision is not None:
        training["precision"] = precision
    if grad_accum_steps is not None:
        training["grad_accum_steps"] = int(grad_accum_steps)
    if dataloader_workers is not None:
        training["dataloader_workers"] = int(dataloader_workers)
    if dataloader_prefetch_factor is not None:
        training["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)
    if dataloader_pin_memory is not None:
        training["dataloader_pin_memory"] = bool(dataloader_pin_memory)

    updated = dict(config)
    updated["training"] = training
    return updated


def start_training_run(
    *,
    config_path: str,
    requested_device: str | None,
    strict_device: bool,
    foreground: bool,
    epochs: int | None = None,
    batch_size: int | None = None,
    seq_length: int | None = None,
    precision: str | None = None,
    grad_accum_steps: int | None = None,
    dataloader_workers: int | None = None,
    dataloader_prefetch_factor: int | None = None,
    dataloader_pin_memory: bool | None = None,
) -> TrainLaunchResult:
    config = load_config(config_path)
    config = apply_training_overrides(
        config,
        epochs=epochs,
        batch_size=batch_size,
        seq_length=seq_length,
        precision=precision,
        grad_accum_steps=grad_accum_steps,
        dataloader_workers=dataloader_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_pin_memory=dataloader_pin_memory,
    )
    selection = resolve_device_selection(
        config=config,
        requested=requested_device,
        strict=strict_device,
    )
    config["runtime"] = {"device": selection.selected}
    data_result = prepare_wikitext2(data_root="data")
    tokenized = tokenize_wikitext2(data_root="data", seed=config["training"].get("seed", 42))
    train_ids = load_token_ids(tokenized.train_ids_path)
    dataloader = SequenceDataLoader(
        train_ids,
        seq_length=int(config["training"]["seq_length"]),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        seed=int(config["training"].get("seed", 42)),
    )
    preview_batch = next(iter(dataloader), None)
    run_files = initialize_run(config_path=config_path, device=selection.selected)
    update_run_meta(
        meta_path=run_files.meta_path,
        updates={
            "requested_device": selection.requested,
            "selected_device": selection.selected,
            "device_warning": selection.warning,
            "tokenized_train_path": str(tokenized.train_ids_path),
            "tokenized_validation_path": str(tokenized.validation_ids_path),
            "tokenizer_path": str(tokenized.tokenizer_path),
            "dataset": data_result.dataset_name,
            "training_options": {
                "precision": config["training"].get("precision", "off"),
                "grad_accum_steps": int(config["training"].get("grad_accum_steps", 1)),
                "dataloader_workers": int(config["training"].get("dataloader_workers", 0)),
                "dataloader_prefetch_factor": int(
                    config["training"].get("dataloader_prefetch_factor", 2)
                ),
                "dataloader_pin_memory": bool(
                    config["training"].get("dataloader_pin_memory", False)
                ),
            },
        },
    )
    pid = None
    if foreground:
        worker_args = argparse.Namespace(config=config_path, run_id=run_files.run_id)
        from .cli import cmd_train_worker

        cmd_train_worker(worker_args)
        mode = "foreground"
    else:
        pid = start_background_training(
            run_dir=run_files.run_dir,
            run_id=run_files.run_id,
            config_path=config_path,
            device_request=selection.requested,
            strict_device=strict_device,
            epochs=epochs,
            batch_size=batch_size,
            seq_length=seq_length,
            precision=precision,
            grad_accum_steps=grad_accum_steps,
            dataloader_workers=dataloader_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            dataloader_pin_memory=dataloader_pin_memory,
        )
        mode = f"background(pid={pid})"

    return TrainLaunchResult(
        run_files=run_files,
        config=config,
        tokenized=tokenized,
        data_result=data_result,
        training_mode=mode,
        pid=pid,
        preview_shape=preview_batch.shape if preview_batch else (0, 0),
        selection=selection,
    )


def resume_training_run(
    *,
    run_dir: Path,
    config_path: str,
    checkpoint_path: str,
    more_epochs: int,
    foreground: bool,
    requested_device: str | None,
    strict_device: bool,
    precision: str | None = None,
    grad_accum_steps: int | None = None,
) -> tuple[str, int | None]:
    run_id = run_dir.name
    load_meta(run_dir / "meta.json")
    update_run_state(
        state_path=run_dir / "state.json",
        status="queued",
        metrics={
            "resume_checkpoint": checkpoint_path,
            "resume_more_epochs": more_epochs,
        },
    )
    if foreground:
        worker_args = argparse.Namespace(
            config=config_path,
            run_id=run_id,
            checkpoint=checkpoint_path,
            more_epochs=more_epochs,
            device=requested_device,
            strict_device=strict_device,
            precision=precision,
            grad_accum_steps=grad_accum_steps,
        )
        from .cli import cmd_train_worker

        cmd_train_worker(worker_args)
        return ("foreground", None)

    pid = start_background_training(
        run_dir=run_dir,
        run_id=run_id,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        more_epochs=more_epochs,
        device_request=requested_device,
        strict_device=strict_device,
        precision=precision,
        grad_accum_steps=grad_accum_steps,
    )
    return (f"background(pid={pid})", pid)


def run_generation(
    *,
    run_dir: Path,
    checkpoint_path: str,
    device_request: str | None,
    strict_device: bool,
    options: GenerateOptions,
) -> tuple[str, DeviceSelection]:
    meta = load_meta(run_dir / "meta.json")
    requested = device_request
    if requested is None:
        requested = str(meta.get("requested_device") or meta.get("selected_device") or "auto")
    selection = resolve_device(requested=requested, strict=strict_device)
    try:
        from .generation import generate_from_checkpoint
    except ModuleNotFoundError as exc:
        raise DeviceResolutionError(f"Missing dependency: {exc}") from exc

    text = generate_from_checkpoint(
        checkpoint_path=checkpoint_path,
        prompt=options.prompt,
        max_new_tokens=options.max_new_tokens,
        temperature=options.temperature,
        top_k=options.top_k,
        device=selection.selected,
    )
    return text, selection
