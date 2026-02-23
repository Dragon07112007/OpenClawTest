from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_trainer import cli


@pytest.mark.parametrize("command", ["resume", "generate"])
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
        "load_config",
        lambda *args, **kwargs: {"training": {"seq_length": 2, "batch_size": 1, "seed": 42}},
    )
    monkeypatch.setattr(
        cli,
        "prepare_wikitext2",
        lambda **kwargs: type(
            "PrepResult",
            (),
            {"dataset_name": "wikitext-2", "train_samples": 2, "validation_samples": 1},
        )(),
    )
    train_ids_path = tmp_path / "data" / "wikitext-2" / "tokenized" / "train_ids.json"
    validation_ids_path = tmp_path / "data" / "wikitext-2" / "tokenized" / "validation_ids.json"
    tokenizer_path = tmp_path / "data" / "wikitext-2" / "tokenized" / "tokenizer.json"
    train_ids_path.parent.mkdir(parents=True, exist_ok=True)
    train_ids_path.write_text("[1,2,3,4,5,6]", encoding="utf-8")
    validation_ids_path.write_text("[1,2,3,4,5,6]", encoding="utf-8")
    tokenizer_path.write_text('{"vocab":{"<pad>":0,"<unk>":1,"a":2}}', encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "tokenize_wikitext2",
        lambda **kwargs: type(
            "TokenizedResult",
            (),
            {
                "train_ids_path": Path("data/wikitext-2/tokenized/train_ids.json"),
                "validation_ids_path": Path("data/wikitext-2/tokenized/validation_ids.json"),
                "tokenizer_path": Path("data/wikitext-2/tokenized/tokenizer.json"),
                "train_tokens": 6,
            },
        )(),
    )
    monkeypatch.setattr(cli, "start_background_training", lambda **kwargs: 4321)
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


def test_status_command_reads_latest_run(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    run = (tmp_path / "runs" / "run-1")
    run.mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps({"run_id": "run-1", "device": "cpu", "config_path": "configs/default.toml"}),
        encoding="utf-8",
    )
    (run / "state.json").write_text(
        json.dumps({"status": "running", "global_step": 3, "train_loss": 1.0, "val_loss": 1.5}),
        encoding="utf-8",
    )

    rc = cli.main(["status", "--config", "configs/default.toml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id=run-1" in out
    assert "status=running" in out
    assert "device=cpu" in out
