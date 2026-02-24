from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from llm_trainer import cli


def test_generate_stub_prints_device_info(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "get_device", lambda: "cpu")
    monkeypatch.chdir(tmp_path)
    run = Path("runs") / "run-1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(
        json.dumps({"run_id": "run-1", "device": "cpu", "config_path": "configs/default.toml"}),
        encoding="utf-8",
    )
    (run / "state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    fake_generation = types.SimpleNamespace(
        generate_from_checkpoint=lambda **kwargs: "hello output"
    )
    monkeypatch.setitem(sys.modules, "llm_trainer.generation", fake_generation)

    rc = cli.main(["generate", "--config", "configs/default.toml", "--prompt", "hello"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "device=cpu" in out


def test_resume_command_starts_background(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    run = (tmp_path / "runs" / "run-1")
    run.mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps({"run_id": "run-1", "device": "cpu", "config_path": "configs/default.toml"}),
        encoding="utf-8",
    )
    (run / "state.json").write_text(
        json.dumps({"status": "queued", "history": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "start_background_training", lambda **kwargs: 999)

    rc = cli.main(
        [
            "resume",
            "--config",
            "configs/default.toml",
            "--run-id",
            "run-1",
            "--more-epochs",
            "2",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id=run-1" in out
    assert "more_epochs=2" in out


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
        json.dumps(
            {
                "status": "running",
                "global_step": 3,
                "train_loss": 1.0,
                "val_loss": 1.5,
                "elapsed_seconds": 65.0,
                "remaining_seconds": 120.0,
                "eta_at": "2026-02-24T11:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(["status", "--config", "configs/default.toml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id=run-1" in out
    assert "status=running" in out
    assert "elapsed=00:01:05" in out
    assert "remaining=00:02:00" in out
    assert "eta=2026-02-24T11:00:00Z" in out
    assert "device=cpu" in out


def test_tui_command_handles_missing_textual(monkeypatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "textual", None)
    for module_name in list(sys.modules):
        if module_name.startswith("textual."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    rc = cli.main(["tui"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "missing dependency: textual" in out
