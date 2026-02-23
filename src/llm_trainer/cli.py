from __future__ import annotations

import argparse


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
    print(f"train command stub (config={args.config})")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print(f"status command stub (config={args.config})")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    print(f"resume command stub (config={args.config})")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    print(f"generate command stub (config={args.config})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
