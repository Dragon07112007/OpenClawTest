from __future__ import annotations

import importlib.util
import json

import pytest

from llm_trainer.run_metadata import initialize_run

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)


def test_train_loop_writes_checkpoints_and_logs(tmp_path) -> None:
    trainer = pytest.importorskip("llm_trainer.trainer")
    tokenized_dir = tmp_path / "data" / "wikitext-2" / "tokenized"
    tokenized_dir.mkdir(parents=True)
    tokenizer_path = tokenized_dir / "tokenizer.json"
    tokenizer_path.write_text('{"vocab":{"<pad>":0,"<unk>":1,"a":2,"b":3}}', encoding="utf-8")
    train_ids = tokenized_dir / "train_ids.json"
    validation_ids = tokenized_dir / "validation_ids.json"
    train_ids.write_text(json.dumps([2, 3] * 100), encoding="utf-8")
    validation_ids.write_text(json.dumps([2, 3] * 50), encoding="utf-8")

    run = initialize_run(
        config_path="configs/default.toml",
        device="cpu",
        runs_root=tmp_path / "runs",
    )
    config = {
        "runtime": {"device": "cpu"},
        "model": {"d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32},
        "training": {
            "seq_length": 8,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "epochs": 1,
            "seed": 1,
            "save_every_epochs": 1,
        },
    }

    metrics = trainer.train_loop(
        config=config,
        run_files=run,
        tokenized_train_path=train_ids,
        tokenized_validation_path=validation_ids,
        tokenizer_path=tokenizer_path,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert metrics["epoch"] == 1
    assert (tmp_path / "checkpoints" / run.run_id / "latest.pt").exists()
    assert (tmp_path / "checkpoints" / run.run_id / "epoch-1.pt").exists()
    assert (run.run_dir / "train.log").exists()
