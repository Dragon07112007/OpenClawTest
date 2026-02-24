from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .app_logic import (
    DeviceResolutionError,
    GenerateOptions,
    apply_training_overrides,
    resolve_device_selection,
    resume_training_run,
    run_generation,
    start_training_run,
)
from .background import pid_is_alive
from .config import load_config
from .run_metadata import RunFiles, load_meta, load_state, update_run_meta, update_run_state


def _format_metric(value: object, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{value}{suffix}"
    return "n/a" if value is None else str(value)


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_eta(eta_at: str | None) -> str:
    if not eta_at:
        return "n/a"
    try:
        eta = datetime.fromisoformat(eta_at.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return eta_at
    return eta.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to training config file.",
    )


def _add_device_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default=None,
        help="Device request: auto, cpu, cuda, cuda:N, or GPU name hint (e.g. A30).",
    )
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Fail instead of falling back when requested device is unavailable.",
    )


def _add_tuning_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seq-length", type=int, default=None, help="Override sequence length.")
    parser.add_argument(
        "--precision",
        choices=["off", "fp16", "bf16"],
        default=None,
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps for larger effective batch size.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=None,
        help="DataLoader worker count (portable default: 0).",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor when workers > 0.",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action="store_true",
        help="Enable DataLoader pinned-memory behavior.",
    )


def _resolve_run_dir(run_id: str | None, runs_root: Path = Path("runs")) -> Path:
    if run_id is not None:
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run ID not found: {run_id}")
        return run_dir

    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.exists() else []
    if not run_dirs:
        raise FileNotFoundError("No runs found.")
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-trainer", description="LLM trainer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Start a training run.")
    _add_common_options(train_parser)
    _add_device_options(train_parser)
    _add_tuning_options(train_parser)
    train_parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run training in the foreground instead of detached background mode.",
    )
    train_parser.set_defaults(func=cmd_train)

    status_parser = subparsers.add_parser("status", help="Show training status.")
    _add_common_options(status_parser)
    status_parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID to inspect. Latest if omitted.",
    )
    status_parser.set_defaults(func=cmd_status)

    resume_parser = subparsers.add_parser("resume", help="Resume a training run.")
    _add_common_options(resume_parser)
    _add_device_options(resume_parser)
    resume_parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID to resume. Latest if omitted.",
    )
    resume_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to latest.",
    )
    resume_parser.add_argument("--more-epochs", type=int, required=True)
    resume_parser.add_argument(
        "--precision",
        choices=["off", "fp16", "bf16"],
        default=None,
        help="Mixed precision mode override for resume.",
    )
    resume_parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation override for resume.",
    )
    resume_parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run resume in the foreground instead of detached background mode.",
    )
    resume_parser.set_defaults(func=cmd_resume)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a checkpoint.")
    _add_common_options(generate_parser)
    _add_device_options(generate_parser)
    generate_parser.add_argument("--run-id", default=None, help="Run ID for checkpoint lookup.")
    generate_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to latest.",
    )
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=50)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument("--top-k", type=int, default=50)
    generate_parser.set_defaults(func=cmd_generate)

    worker_parser = subparsers.add_parser("train-worker", help=argparse.SUPPRESS)
    _add_common_options(worker_parser)
    worker_parser.add_argument("--run-id", required=True)
    worker_parser.add_argument("--checkpoint", default=None)
    worker_parser.add_argument("--more-epochs", type=int, default=None)
    _add_device_options(worker_parser)
    _add_tuning_options(worker_parser)
    worker_parser.set_defaults(func=cmd_train_worker)

    tui_parser = subparsers.add_parser("tui", help="Launch live monitoring TUI.")
    tui_parser.add_argument("--run-id", default=None, help="Watch a specific run.")
    tui_parser.set_defaults(func=cmd_tui)

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    try:
        result = start_training_run(
            config_path=args.config,
            requested_device=args.device,
            strict_device=args.strict_device,
            foreground=args.foreground,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            precision=args.precision,
            grad_accum_steps=args.grad_accum_steps,
            dataloader_workers=args.dataloader_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            dataloader_pin_memory=args.dataloader_pin_memory,
        )
    except DeviceResolutionError as exc:
        print(f"train failed (error={exc})")
        return 1

    warning_text = f", warning={result.selection.warning}" if result.selection.warning else ""
    print(
        "train command "
        f"(config={args.config}, requested_device={result.selection.requested}, "
        f"device={result.selection.selected}{warning_text}, run_id={result.run_files.run_id}, "
        f"dataset={result.data_result.dataset_name}, "
        f"train_samples={result.data_result.train_samples}, "
        f"validation_samples={result.data_result.validation_samples}, "
        f"train_tokens={result.tokenized.train_tokens}, "
        f"batch_shape={result.preview_shape}, "
        f"precision={result.config['training'].get('precision', 'off')}, "
        f"grad_accum_steps={result.config['training'].get('grad_accum_steps', 1)}, "
        f"dataloader_workers={result.config['training'].get('dataloader_workers', 0)}, "
        "dataloader_prefetch_factor="
        f"{result.config['training'].get('dataloader_prefetch_factor', 2)}, "
        "dataloader_pin_memory="
        f"{result.config['training'].get('dataloader_pin_memory', False)}, "
        f"training={result.training_mode})"
    )
    return 0


def cmd_train_worker(args: argparse.Namespace) -> int:
    run_dir = Path("runs") / args.run_id
    meta = load_meta(run_dir / "meta.json")
    config = load_config(args.config)
    config = apply_training_overrides(
        config,
        epochs=getattr(args, "epochs", None),
        batch_size=getattr(args, "batch_size", None),
        seq_length=getattr(args, "seq_length", None),
        precision=getattr(args, "precision", None),
        grad_accum_steps=getattr(args, "grad_accum_steps", None),
        dataloader_workers=getattr(args, "dataloader_workers", None),
        dataloader_prefetch_factor=getattr(args, "dataloader_prefetch_factor", None),
        dataloader_pin_memory=bool(getattr(args, "dataloader_pin_memory", False))
        if hasattr(args, "dataloader_pin_memory")
        else None,
    )
    try:
        selection = resolve_device_selection(
            config=config,
            requested=getattr(args, "device", None),
            strict=bool(getattr(args, "strict_device", False)),
        )
    except DeviceResolutionError as exc:
        update_run_state(
            state_path=run_dir / "state.json",
            status="failed",
            metrics={"error": str(exc)},
        )
        return 1
    config["runtime"] = {"device": selection.selected}

    update_run_meta(
        meta_path=run_dir / "meta.json",
        updates={
            "requested_device": selection.requested,
            "selected_device": selection.selected,
            "device_warning": selection.warning,
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

    update_run_state(
        state_path=run_dir / "state.json",
        status="running",
        metrics={"device": selection.selected, "device_warning": selection.warning},
    )
    try:
        from .trainer import train_loop
    except ModuleNotFoundError as exc:
        update_run_state(
            state_path=run_dir / "state.json",
            status="failed",
            metrics={"error": f"Missing dependency: {exc}"},
        )
        return 1

    start_epoch = 0
    total_epochs = int(config["training"]["epochs"])
    optimizer_state = None
    model_state = None
    global_step = 0
    if args.checkpoint is not None:
        torch = __import__("torch")
        checkpoint = torch.load(args.checkpoint, map_location=selection.selected)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        optimizer_state = checkpoint.get("optimizer_state_dict")
        model_state = checkpoint.get("model_state_dict")
        if args.more_epochs is None:
            raise ValueError("--more-epochs is required when --checkpoint is provided.")
        total_epochs = start_epoch + int(args.more_epochs)

    train_loop(
        config=config,
        run_files=RunFiles(
            run_id=args.run_id,
            run_dir=run_dir,
            meta_path=run_dir / "meta.json",
            state_path=run_dir / "state.json",
        ),
        tokenized_train_path=meta["tokenized_train_path"],
        tokenized_validation_path=meta["tokenized_validation_path"],
        tokenizer_path=meta["tokenizer_path"],
        start_epoch=start_epoch,
        total_epochs=total_epochs,
        optimizer_state=optimizer_state,
        model_state=model_state,
        global_step=global_step,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.run_id)
    meta = load_meta(run_dir / "meta.json")
    state = load_state(run_dir / "state.json")

    pid = state.get("pid")
    if isinstance(pid, int):
        process_state = "alive" if pid_is_alive(pid) else "not-running"
    else:
        process_state = "n/a"
    print(
        "status "
        f"(run_id={meta['run_id']}, status={state.get('status')}, "
        f"epoch={state.get('epoch')}, step={state.get('global_step')}, "
        f"train_loss={state.get('train_loss')}, val_loss={state.get('val_loss')}, "
        f"elapsed={_format_duration(state.get('elapsed_seconds'))}, "
        f"remaining={_format_duration(state.get('remaining_seconds'))}, "
        f"eta={_format_eta(state.get('eta_at'))}, "
        f"device={meta.get('selected_device', meta.get('device'))}, "
        f"gpu_util={_format_metric(state.get('gpu_utilization_pct'), '%')}, "
        f"gpu_mem={_format_metric(state.get('gpu_memory_used_mb'))}/"
        f"{_format_metric(state.get('gpu_memory_total_mb'))}MB, "
        f"gpu_temp={_format_metric(state.get('gpu_temperature_c'), 'C')}, "
        f"gpu_power={_format_metric(state.get('gpu_power_w'), 'W')}, "
        f"pid={pid}, process={process_state})"
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.run_id)
    checkpoint_path = args.checkpoint or str(Path("checkpoints") / run_dir.name / "latest.pt")
    try:
        mode, _ = resume_training_run(
            run_dir=run_dir,
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            more_epochs=args.more_epochs,
            foreground=args.foreground,
            requested_device=args.device,
            strict_device=args.strict_device,
            precision=args.precision,
            grad_accum_steps=args.grad_accum_steps,
        )
    except DeviceResolutionError as exc:
        print(f"resume failed (error={exc})")
        return 1
    print(
        "resume command "
        f"(config={args.config}, run_id={run_dir.name}, checkpoint={checkpoint_path}, "
        f"more_epochs={args.more_epochs}, mode={mode})"
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.run_id)
    checkpoint_path = args.checkpoint or str(Path("checkpoints") / run_dir.name / "latest.pt")
    try:
        text, selection = run_generation(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            device_request=args.device,
            strict_device=args.strict_device,
            options=GenerateOptions(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ),
        )
    except (DeviceResolutionError, FileNotFoundError, ValueError) as exc:
        print(f"generate failed (error={exc})")
        return 1

    print(
        "generate command "
        f"(requested_device={selection.requested}, device={selection.selected}, "
        f"checkpoint={checkpoint_path}, prompt={args.prompt!r})\n{text}"
    )
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    from .tui import launch_tui

    return launch_tui(run_id=args.run_id)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
