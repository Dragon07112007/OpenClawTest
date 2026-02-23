from __future__ import annotations

import argparse

from .config import load_config
from .data import prepare_wikitext2
from .dataloader import SequenceDataLoader, load_token_ids, tokenize_wikitext2
from .device import get_device
from .run_metadata import initialize_run


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to training config file.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-trainer", description="LLM trainer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Start a training run.")
    _add_common_options(train_parser)
    train_parser.set_defaults(func=cmd_train)

    status_parser = subparsers.add_parser("status", help="Show training status.")
    _add_common_options(status_parser)
    status_parser.set_defaults(func=cmd_status)

    resume_parser = subparsers.add_parser("resume", help="Resume a training run.")
    _add_common_options(resume_parser)
    resume_parser.set_defaults(func=cmd_resume)

    generate_parser = subparsers.add_parser("generate", help="Generate text from a checkpoint.")
    _add_common_options(generate_parser)
    generate_parser.set_defaults(func=cmd_generate)

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
    training_status = "skipped"
    try:
        from .trainer import train_loop
    except ModuleNotFoundError:
        training_status = "torch-missing"
    else:
        metrics = train_loop(
            config=config,
            run_files=run_files,
            tokenized_train_path=tokenized.train_ids_path,
            tokenized_validation_path=tokenized.validation_ids_path,
            tokenizer_path=tokenized.tokenizer_path,
        )
        training_status = (
            f"completed(epoch={metrics['epoch']}, step={metrics['global_step']}, "
            f"val_loss={metrics['val_loss']})"
        )
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


def cmd_status(args: argparse.Namespace) -> int:
    device = get_device()
    print(f"status command stub (config={args.config}, device={device})")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    device = get_device()
    print(f"resume command stub (config={args.config}, device={device})")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    device = get_device()
    print(f"generate command stub (config={args.config}, device={device})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
