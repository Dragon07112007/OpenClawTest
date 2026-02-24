from __future__ import annotations

import json

from llm_trainer.tui import build_tui_snapshot


def _write_run(tmp_path, run_id: str, status: str, step: int, eta_at: str | None) -> None:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": run_id, "device": "cpu", "config_path": "configs/default.toml"}),
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "status": status,
                "epoch": 1,
                "global_step": step,
                "train_loss": 1.23,
                "val_loss": 1.4,
                "elapsed_seconds": 30.0,
                "remaining_seconds": 10.0,
                "eta_at": eta_at,
            }
        ),
        encoding="utf-8",
    )


def test_build_tui_snapshot_empty_state(tmp_path) -> None:
    snapshot = build_tui_snapshot(runs_root=tmp_path / "runs")
    assert "No runs found." in snapshot["runs"][0]
    assert "Empty state" in snapshot["detail"][0]


def test_build_tui_snapshot_single_run_detail_includes_eta(tmp_path) -> None:
    _write_run(tmp_path, run_id="run-1", status="running", step=10, eta_at="2026-02-24T11:00:00Z")

    snapshot = build_tui_snapshot(runs_root=tmp_path / "runs")

    assert any("run-1" in row for row in snapshot["runs"])
    assert any("ETA: 2026-02-24T11:00:00Z" in line for line in snapshot["detail"])
    assert any("Remaining: 00:00:10" in line for line in snapshot["detail"])


def test_build_tui_snapshot_many_runs_supports_selection(tmp_path) -> None:
    _write_run(tmp_path, run_id="run-a", status="completed", step=5, eta_at=None)
    _write_run(tmp_path, run_id="run-b", status="running", step=12, eta_at="2026-02-24T11:00:00Z")

    snapshot = build_tui_snapshot(runs_root=tmp_path / "runs", selected_index=1)

    assert len(snapshot["runs"]) >= 3
    assert snapshot["selected"] == 1
    assert any("Run: " in line for line in snapshot["detail"])
