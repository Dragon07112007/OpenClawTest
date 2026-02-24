from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from rich.markup import MarkupError, render

from llm_trainer.tui import (
    TUI_GRID_CSS,
    TuiGenerationOptions,
    TuiTrainingOptions,
    _aggregate_active_remaining_time,
    _join_markup_safe,
    _launch_help_text,
    _markup_safe,
    _model_is_running,
    archive_model_run,
    build_tui_snapshot,
    collect_model_entries,
    collect_system_utilization,
    delete_model_run,
    launch_tui,
    tui_generate_from_run,
    tui_resume_training,
    tui_start_training,
)


def _write_run(
    tmp_path,
    run_id: str,
    status: str,
    step: int,
    eta_at: str | None,
    *,
    gpu_utilization_pct: float | None = None,
    mtime: float = 1.0,
) -> None:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "device": "cpu",
                "selected_device": "cpu",
                "config_path": "configs/default.toml",
            }
        ),
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
                "gpu_utilization_pct": gpu_utilization_pct,
            }
        ),
        encoding="utf-8",
    )
    os.utime(run_dir, (mtime, mtime))


def _write_checkpoint(tmp_path: Path, run_id: str, *, mtime: float = 1.0) -> Path:
    checkpoint_dir = tmp_path / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "latest.pt"
    checkpoint.write_bytes(b"pt")
    os.utime(checkpoint, (mtime, mtime))
    return checkpoint


def test_build_tui_snapshot_empty_state(tmp_path) -> None:
    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
    )

    assert snapshot["runs"][0] == "Run Dashboard"
    assert snapshot["detail"][0] == "Empty state"
    assert snapshot["generation"][0] == "Generate From Model"
    assert snapshot["launcher"][0] == "Train Selected Model"
    assert snapshot["models"][0] == "Model Selection"
    assert snapshot["utilization"][-1] == "No active training"


def test_build_tui_snapshot_runs_panel_includes_kpi_columns(tmp_path) -> None:
    _write_run(
        tmp_path,
        run_id="run-1",
        status="running",
        step=10,
        eta_at="2026-02-24T11:00:00Z",
        gpu_utilization_pct=88.0,
    )

    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
    )

    assert "RUN_ID | Status | Epoch | Loss | Device | ETA | GPU%" in snapshot["runs"][1]
    assert any("run-1" in row for row in snapshot["runs"])
    assert any("ETA: 2026-02-24T11:00:00Z" in line for line in snapshot["detail"])
    assert any("GPU util: 88.0" in line for line in snapshot["detail"])


def test_run_dashboard_filters_to_active_runs_and_empty_state(tmp_path) -> None:
    _write_run(tmp_path, run_id="done", status="completed", step=2, eta_at=None, mtime=20)
    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
    )

    assert snapshot["runs"][2] == "No active runs."


def test_running_runs_are_sorted_ahead_of_newer_completed_runs(tmp_path) -> None:
    _write_run(tmp_path, run_id="completed-new", status="completed", step=2, eta_at=None, mtime=20)
    _write_run(tmp_path, run_id="running-old", status="running", step=1, eta_at=None, mtime=10)

    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
    )
    first_data_row = snapshot["runs"][2]

    assert "running-old" in first_data_row


def test_selection_prefers_selected_run_id_over_index(tmp_path) -> None:
    _write_run(tmp_path, run_id="run-a", status="running", step=5, eta_at=None, mtime=2)
    _write_run(tmp_path, run_id="run-b", status="running", step=12, eta_at=None, mtime=3)

    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
        selected_index=0,
        selected_run_id="run-a",
    )

    assert snapshot["selected_run_id"] == "run-a"


def test_model_manager_marks_latest_and_propagates_active_model(tmp_path) -> None:
    _write_checkpoint(tmp_path, "run-1", mtime=5)
    _write_checkpoint(tmp_path, "run-2", mtime=10)

    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
        selected_model_index=0,
        active_model_run_id="run-1",
    )

    assert any("Latest trained model: run-2" in line for line in snapshot["models"])
    assert any("active" in line and "run-1" in line for line in snapshot["models"])
    assert any("Selected Model: run-1" in line for line in snapshot["generation"])
    assert any("Selected Model: run-1" in line for line in snapshot["launcher"])


def test_collect_model_entries_ignores_archived_dirs(tmp_path) -> None:
    _write_checkpoint(tmp_path, "run-1", mtime=1)
    archived = tmp_path / "checkpoints" / "_archived"
    archived.mkdir(parents=True)
    (archived / "junk.pt").write_bytes(b"x")

    entries = collect_model_entries(
        checkpoints_root=tmp_path / "checkpoints",
        runs_root=tmp_path / "runs",
    )

    assert [entry.run_id for entry in entries] == ["run-1"]


def test_archive_and_delete_model_actions_are_safe(tmp_path) -> None:
    _write_checkpoint(tmp_path, "run-1", mtime=1)

    ok_archive, archive_message = archive_model_run(
        "run-1",
        checkpoints_root=tmp_path / "checkpoints",
    )
    assert ok_archive is True
    assert "archived run_id=run-1" in archive_message

    archived_dirs = list((tmp_path / "checkpoints" / "_archived").glob("run-1-*"))
    assert len(archived_dirs) == 1

    _write_checkpoint(tmp_path, "run-2", mtime=2)
    ok_delete, delete_message = delete_model_run("run-2", checkpoints_root=tmp_path / "checkpoints")
    assert ok_delete is True
    assert "deleted checkpoint dir for run_id=run-2" in delete_message
    assert not (tmp_path / "checkpoints" / "run-2").exists()


def test_collect_system_utilization_fallbacks_without_psutil(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(
        "llm_trainer.tui.collect_gpu_telemetry",
        lambda _device: {
            "gpu_utilization_pct": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_w": None,
        },
    )

    metrics = collect_system_utilization(selected_run_state=None, selected_device="cpu")

    assert metrics["gpu_utilization_pct"] is None
    assert metrics["cpu_utilization_pct"] is None
    assert metrics["cpu_count"] is None
    assert metrics["ram_used_mb"] is None


def test_aggregate_active_remaining_time_uses_active_runs_only(tmp_path) -> None:
    _write_run(tmp_path, run_id="run-1", status="running", step=1, eta_at=None)
    _write_run(tmp_path, run_id="run-2", status="paused", step=2, eta_at=None)
    _write_run(tmp_path, run_id="run-3", status="completed", step=3, eta_at=None)
    run2_state = tmp_path / "runs" / "run-2" / "state.json"
    run2_data = json.loads(run2_state.read_text(encoding="utf-8"))
    run2_data["remaining_seconds"] = 20.0
    run2_state.write_text(json.dumps(run2_data), encoding="utf-8")

    entries = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
    )

    assert "Aggregate remaining: 00:00:30" in entries["utilization"][-1]


def test_aggregate_active_remaining_time_no_active_training() -> None:
    assert _aggregate_active_remaining_time([]) == "No active training"


def test_tui_start_training_returns_status(monkeypatch) -> None:
    options = TuiTrainingOptions()
    monkeypatch.setattr(
        "llm_trainer.tui.start_training_run",
        lambda **kwargs: type(
            "Result",
            (),
            {
                "run_files": type("RunFiles", (), {"run_id": "run-1"})(),
                "selection": type("Selection", (), {"selected": "cpu", "warning": None})(),
                "training_mode": "background(pid=1)",
            },
        )(),
    )

    ok, message = tui_start_training(options)

    assert ok is True
    assert "run_id=run-1" in message


def test_tui_resume_training_missing_run_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    options = TuiTrainingOptions()

    ok, message = tui_resume_training("missing", options)

    assert ok is False
    assert "resume failed" in message


def test_tui_generate_from_run_success(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"run_id": "run-1", "selected_device": "cpu"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "llm_trainer.tui.run_generation",
        lambda **kwargs: ("hello", type("Selection", (), {"selected": "cpu"})()),
    )

    ok, message = tui_generate_from_run("run-1", TuiGenerationOptions(prompt="hello"))

    assert ok is True
    assert "generated with cpu" in message


def test_tui_generate_from_run_validates_options() -> None:
    ok, message = tui_generate_from_run("run-1", TuiGenerationOptions(prompt=""))

    assert ok is False
    assert "prompt must not be empty" in message


def test_launch_tui_missing_textual(monkeypatch, capsys) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("textual"):
            raise ModuleNotFoundError("No module named textual")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rc = launch_tui()

    assert rc == 1
    assert "missing dependency: textual" in capsys.readouterr().out


def test_markup_safe_escapes_bracket_markup_tokens() -> None:
    raw = "text [/][bold]x[/bold] [unterminated"
    with pytest.raises(MarkupError):
        render(raw)
    render(_markup_safe(raw))


def test_join_markup_safe_handles_all_dynamic_panels_with_markup_like_text(tmp_path) -> None:
    _write_run(tmp_path, run_id="run-1", status="running", step=12, eta_at="2026-02-24T11:00:00Z")
    _write_checkpoint(tmp_path, "run-1", mtime=1)

    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
        generation_options=TuiGenerationOptions(prompt="danger [/][bold]x["),
        generation_output="line [/][oops]",
        last_action="error: [/][broken]",
    )

    dynamic_sections = [
        snapshot["runs"],
        snapshot["detail"],
        snapshot["launcher"],
        snapshot["generation"],
        snapshot["utilization"],
        snapshot["models"],
        snapshot["status"],
    ]

    with pytest.raises(MarkupError):
        render("\n".join(snapshot["generation"]))

    for lines in dynamic_sections:
        safe_plain = render(_join_markup_safe(list(lines))).plain
        assert isinstance(safe_plain, str)


def test_join_markup_safe_escapes_help_text_regression() -> None:
    raw_help = _launch_help_text(TuiTrainingOptions(), TuiGenerationOptions())
    with pytest.raises(MarkupError):
        render(raw_help)
    safe_help = render(_markup_safe(raw_help)).plain
    assert "epochs +/-" in safe_help


def test_run_dashboard_scroll_window_tracks_selection(tmp_path) -> None:
    for idx in range(15):
        _write_run(
            tmp_path,
            run_id=f"run-{idx:02d}",
            status="running",
            step=idx,
            eta_at="2026-02-24T11:00:00Z",
            mtime=float(idx + 1),
        )

    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
        selected_index=12,
        run_scroll_offset=0,
    )

    assert snapshot["run_scroll_offset"] > 0
    assert any("Showing " in line for line in snapshot["runs"])


def test_generation_output_scroll_window(tmp_path) -> None:
    _write_checkpoint(tmp_path, "run-1", mtime=1)
    lines = "\n".join(f"line-{idx}" for idx in range(20))
    snapshot = build_tui_snapshot(
        runs_root=tmp_path / "runs",
        checkpoints_root=tmp_path / "checkpoints",
        active_model_run_id="run-1",
        generation_output=lines,
        generation_scroll_offset=5,
    )

    assert "line-5" in snapshot["generation"]
    assert any("output lines 6-" in line for line in snapshot["generation"])


def test_tui_css_matches_two_column_three_row_spec() -> None:
    assert "grid-size: 2 3;" in TUI_GRID_CSS
    assert "#panel-e" in TUI_GRID_CSS
    assert "row-span: 2;" in TUI_GRID_CSS
    assert "border-title-align: center;" in TUI_GRID_CSS


def test_delete_guard_blocks_running_model(tmp_path) -> None:
    _write_run(tmp_path, run_id="run-1", status="running", step=1, eta_at=None)
    assert _model_is_running("run-1", runs_root=tmp_path / "runs") is True
    assert _model_is_running("missing", runs_root=tmp_path / "runs") is False
