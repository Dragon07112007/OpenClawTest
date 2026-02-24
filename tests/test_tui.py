from __future__ import annotations

import json

from llm_trainer.tui import (
    TuiGenerationOptions,
    TuiTrainingOptions,
    build_tui_snapshot,
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


def test_build_tui_snapshot_empty_state(tmp_path) -> None:
    snapshot = build_tui_snapshot(runs_root=tmp_path / "runs")
    assert "No runs found." in snapshot["runs"][0]
    assert "Empty state" in snapshot["detail"][0]


def test_build_tui_snapshot_single_run_detail_includes_eta_and_gpu(tmp_path) -> None:
    _write_run(
        tmp_path,
        run_id="run-1",
        status="running",
        step=10,
        eta_at="2026-02-24T11:00:00Z",
        gpu_utilization_pct=88.0,
    )

    snapshot = build_tui_snapshot(runs_root=tmp_path / "runs")

    assert any("run-1" in row for row in snapshot["runs"])
    assert any("ETA: 2026-02-24T11:00:00Z" in line for line in snapshot["detail"])
    assert any("GPU util: 88.0" in line for line in snapshot["detail"])


def test_build_tui_snapshot_many_runs_supports_selection(tmp_path) -> None:
    _write_run(tmp_path, run_id="run-a", status="completed", step=5, eta_at=None)
    _write_run(tmp_path, run_id="run-b", status="running", step=12, eta_at="2026-02-24T11:00:00Z")

    snapshot = build_tui_snapshot(runs_root=tmp_path / "runs", selected_index=1)

    assert len(snapshot["runs"]) >= 3
    assert snapshot["selected"] == 1
    assert any("Run: " in line for line in snapshot["detail"])


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
