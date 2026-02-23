import pytest

from llm_trainer.cli import build_parser


def test_cli_root_help() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("command", ["train", "status", "resume", "generate", "tui"])
def test_cli_command_help(command: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([command, "--help"])
    assert exc.value.code == 0
