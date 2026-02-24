from __future__ import annotations

import json
from pathlib import Path

from llm_trainer import cli


def test_generate_uses_selection_info(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    run = Path("runs") / "run-1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "device": "cpu",
                "selected_device": "cpu",
                "config_path": "configs/default.toml",
            }
        ),
        encoding="utf-8",
    )
    (run / "state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "run_generation",
        lambda **kwargs: (
            "hello output",
            type("Sel", (), {"requested": "auto", "selected": "cpu"})(),
        ),
    )

    rc = cli.main(["generate", "--config", "configs/default.toml", "--prompt", "hello"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "requested_device=auto" in out
    assert "device=cpu" in out


def test_resume_command_starts_background(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "device": "cpu",
                "selected_device": "cpu",
                "config_path": "configs/default.toml",
            }
        ),
        encoding="utf-8",
    )
    (run / "state.json").write_text(
        json.dumps({"status": "queued", "history": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "resume_training_run", lambda **kwargs: ("background(pid=999)", 999))

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


def test_train_command_creates_output_with_device_and_tuning(monkeypatch, capsys) -> None:
    class Result:
        run_files = type("RunFiles", (), {"run_id": "run-123"})()
        config = {
            "training": {
                "precision": "bf16",
                "grad_accum_steps": 2,
                "dataloader_workers": 4,
                "dataloader_prefetch_factor": 2,
                "dataloader_pin_memory": True,
            }
        }
        tokenized = type("Tokenized", (), {"train_tokens": 200})()
        data_result = type(
            "DataResult",
            (),
            {"dataset_name": "wikitext-2", "train_samples": 2, "validation_samples": 1},
        )()
        training_mode = "background(pid=4321)"
        preview_shape = (2, 8)
        selection = type(
            "Selection",
            (),
            {"requested": "A30", "selected": "cuda:0", "warning": None},
        )()

    monkeypatch.setattr(cli, "start_training_run", lambda **kwargs: Result())

    rc = cli.main(["train", "--config", "configs/default.toml", "--device", "A30"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "requested_device=A30" in out
    assert "device=cuda:0" in out
    assert "precision=bf16" in out


def test_train_command_handles_strict_device_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "start_training_run",
        lambda **kwargs: (_ for _ in ()).throw(cli.DeviceResolutionError("no cuda")),
    )

    rc = cli.main(["train", "--strict-device"])

    assert rc == 1
    assert "train failed" in capsys.readouterr().out


def test_status_command_reads_gpu_metrics(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "device": "cpu",
                "selected_device": "cuda:0",
                "config_path": "configs/default.toml",
            }
        ),
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
                "gpu_utilization_pct": 79.0,
                "gpu_memory_used_mb": 1024.0,
                "gpu_memory_total_mb": 4096.0,
                "gpu_temperature_c": 59.0,
                "gpu_power_w": 150.0,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(["status", "--config", "configs/default.toml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "gpu_util=79.0%" in out
    assert "gpu_mem=1024.0/4096.0MB" in out
    assert "device=cuda:0" in out


def test_status_command_includes_cpu_metrics_and_telemetry_sources(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs" / "run-1"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps({"run_id": "run-1", "device": "cpu", "selected_device": "cpu"}),
        encoding="utf-8",
    )
    (run / "state.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "collect_host_telemetry",
        lambda **_kwargs: {
            "gpu_utilization_pct": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_w": None,
            "gpu_telemetry_provider": None,
            "gpu_telemetry_reason": "device is not CUDA",
            "cpu_utilization_pct": 35.0,
            "cpu_count": 16,
            "ram_used_mb": 4096.0,
            "ram_total_mb": 32768.0,
            "cpu_telemetry_provider": "psutil",
            "cpu_telemetry_reason": None,
        },
    )

    rc = cli.main(["status", "--config", "configs/default.toml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "cpu_util=35.0%" in out
    assert "cpu_cores=16" in out
    assert "cpu_source=psutil" in out
    assert "gpu_diag=device is not CUDA" in out


def test_tui_command_handles_missing_textual(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "cmd_tui", lambda _args: 1)
    rc = cli.main(["tui"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out == ""
