from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .background import pid_is_alive, start_background_training
from .config import load_config
from .data import prepare_wikitext2
from .dataloader import SequenceDataLoader, load_token_ids, tokenize_wikitext2
from .device import get_device
from .run_metadata import (
    RunFiles,
    initialize_run,
    load_meta,
    load_state,
    update_run_meta,
    update_run_state,
)


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
        "--foreground",
        action="store_true",
        help="Run resume in the foreground instead of detached background mode.",
    )
    resume_parser.set_defaults(func=cmd_resume)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a checkpoint.")
    _add_common_options(generate_parser)
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
    worker_parser.set_defaults(func=cmd_train_worker)

    tui_parser = subparsers.add_parser("tui", help="Launch live monitoring TUI.")
    tui_parser.add_argument("--run-id", default=None, help="Watch a specific run.")
    tui_parser.set_defaults(func=cmd_tui)

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = get_device()
    config["runtime"] = {"device": device}
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
    run_files = initialize_run(config_path=args.config, device=device)
    update_run_meta(
        meta_path=run_files.meta_path,
        updates={
            "tokenized_train_path": str(tokenized.train_ids_path),
            "tokenized_validation_path": str(tokenized.validation_ids_path),
            "tokenizer_path": str(tokenized.tokenizer_path),
            "dataset": data_result.dataset_name,
        },
    )
    if args.foreground:
        worker_args = argparse.Namespace(config=args.config, run_id=run_files.run_id)
        cmd_train_worker(worker_args)
        training_status = "foreground"
    else:
        pid = start_background_training(
            run_dir=run_files.run_dir,
            run_id=run_files.run_id,
            config_path=args.config,
        )
        training_status = f"background(pid={pid})"
    print(
        "train command "
        f"(config={args.config}, device={device}, run_id={run_files.run_id}, "
        f"dataset={data_result.dataset_name}, train_samples={data_result.train_samples}, "
        f"validation_samples={data_result.validation_samples}, "
        f"train_tokens={tokenized.train_tokens}, "
        f"batch_shape={preview_batch.shape if preview_batch else (0, 0)}, "
        f"training={training_status})"
    )
    return 0


def cmd_train_worker(args: argparse.Namespace) -> int:
    run_dir = Path("runs") / args.run_id
    meta = load_meta(run_dir / "meta.json")
    device = get_device()
    config = load_config(args.config)
    config["runtime"] = {"device": device}

    update_run_state(
        state_path=run_dir / "state.json",
        status="running",
        metrics={"device": device},
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
        checkpoint = torch.load(args.checkpoint, map_location=device)
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
        f"device={meta.get('device')}, pid={pid}, process={process_state})"
    )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.run_id)
    run_id = run_dir.name
    checkpoint_path = args.checkpoint or str(Path("checkpoints") / run_id / "latest.pt")
    update_run_state(
        state_path=run_dir / "state.json",
        status="queued",
        metrics={"resume_checkpoint": checkpoint_path, "resume_more_epochs": args.more_epochs},
    )
    if args.foreground:
        worker_args = argparse.Namespace(
            config=args.config,
            run_id=run_id,
            checkpoint=checkpoint_path,
            more_epochs=args.more_epochs,
        )
        cmd_train_worker(worker_args)
        mode = "foreground"
    else:
        pid = start_background_training(
            run_dir=run_dir,
            run_id=run_id,
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            more_epochs=args.more_epochs,
        )
        mode = f"background(pid={pid})"
    print(
        "resume command "
        f"(config={args.config}, run_id={run_id}, checkpoint={checkpoint_path}, "
        f"more_epochs={args.more_epochs}, mode={mode})"
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    device = get_device()
    run_dir = _resolve_run_dir(args.run_id)
    checkpoint_path = args.checkpoint or str(Path("checkpoints") / run_dir.name / "latest.pt")
    try:
        from .generation import generate_from_checkpoint
    except ModuleNotFoundError as exc:
        print(f"generate failed (device={device}, error=Missing dependency: {exc})")
        return 1

    text = generate_from_checkpoint(
        checkpoint_path=checkpoint_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(
        "generate command "
        f"(device={device}, checkpoint={checkpoint_path}, prompt={args.prompt!r})\n{text}"
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
