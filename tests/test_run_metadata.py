from __future__ import annotations

import json

from llm_trainer.run_metadata import initialize_run


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
