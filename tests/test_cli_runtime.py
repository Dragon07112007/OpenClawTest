from __future__ import annotations

import json

import pytest

from llm_trainer import cli


@pytest.mark.parametrize("command", ["status", "resume", "generate"])
def test_command_stubs_print_device_info(command: str, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_device", lambda: "cpu")

    rc = cli.main([command, "--config", "configs/default.toml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "device=cpu" in out


def test_train_command_creates_run_metadata(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        cli,
        "prepare_wikitext2",
        lambda **kwargs: type(
            "PrepResult",
            (),
            {"dataset_name": "wikitext-2", "train_samples": 2, "validation_samples": 1},
        )(),
    )
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["train", "--config", "configs/default.toml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "device=cpu" in out
    assert "run_id=" in out

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1

    meta = json.loads((run_dirs[0] / "meta.json").read_text(encoding="utf-8"))
    state = json.loads((run_dirs[0] / "state.json").read_text(encoding="utf-8"))

    assert meta["device"] == "cpu"
    assert meta["config_path"] == "configs/default.toml"
    assert state["status"] == "queued"
