from __future__ import annotations

import json

from llm_trainer.run_metadata import initialize_run, update_run_state


def test_initialize_run_creates_meta_and_state(tmp_path) -> None:
    run = initialize_run(
        config_path="configs/default.toml",
        device="cpu",
        runs_root=tmp_path,
    )

    assert run.run_dir.exists()
    assert run.meta_path.exists()
    assert run.state_path.exists()

    meta = json.loads(run.meta_path.read_text(encoding="utf-8"))
    state = json.loads(run.state_path.read_text(encoding="utf-8"))

    assert meta["run_id"] == run.run_id
    assert meta["device"] == "cpu"
    assert meta["config_path"] == "configs/default.toml"
    assert meta["started_at"]

    assert state["status"] == "queued"
    assert state["updated_at"]
    assert state["history"]
    assert state["history"][0]["status"] == "queued"
    assert state["history"][0]["timestamp"] == state["updated_at"]


def test_update_run_state_tracks_status_and_metrics(tmp_path) -> None:
    run = initialize_run(
        config_path="configs/default.toml",
        device="cpu",
        runs_root=tmp_path,
    )
    state = update_run_state(
        state_path=run.state_path,
        status="running",
        metrics={"epoch": 1, "train_loss": 1.23},
    )

    assert state["status"] == "running"
    assert state["epoch"] == 1
    assert state["train_loss"] == 1.23
    assert len(state["history"]) == 2
    assert state["history"][-1]["status"] == "running"
