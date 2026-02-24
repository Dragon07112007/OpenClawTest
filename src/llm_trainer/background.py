from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .run_metadata import update_run_state


def start_background_training(
    *,
    run_dir: str | Path,
    run_id: str,
    config_path: str,
    checkpoint_path: str | None = None,
    more_epochs: int | None = None,
    device_request: str | None = None,
    strict_device: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
    seq_length: int | None = None,
    precision: str | None = None,
    grad_accum_steps: int | None = None,
    dataloader_workers: int | None = None,
    dataloader_prefetch_factor: int | None = None,
    dataloader_pin_memory: bool | None = None,
) -> int:
    run_dir = Path(run_dir)
    stdout_path = run_dir / "worker.log"
    with stdout_path.open("a", encoding="utf-8") as out:
        command = [
            sys.executable,
            "-m",
            "llm_trainer",
            "train-worker",
            "--run-id",
            run_id,
            "--config",
            config_path,
        ]
        if checkpoint_path is not None:
            command.extend(["--checkpoint", checkpoint_path])
        if more_epochs is not None:
            command.extend(["--more-epochs", str(more_epochs)])
        if device_request is not None:
            command.extend(["--device", str(device_request)])
        if strict_device:
            command.append("--strict-device")
        if epochs is not None:
            command.extend(["--epochs", str(epochs)])
        if batch_size is not None:
            command.extend(["--batch-size", str(batch_size)])
        if seq_length is not None:
            command.extend(["--seq-length", str(seq_length)])
        if precision is not None:
            command.extend(["--precision", precision])
        if grad_accum_steps is not None:
            command.extend(["--grad-accum-steps", str(grad_accum_steps)])
        if dataloader_workers is not None:
            command.extend(["--dataloader-workers", str(dataloader_workers)])
        if dataloader_prefetch_factor is not None:
            command.extend(["--dataloader-prefetch-factor", str(dataloader_prefetch_factor)])
        if dataloader_pin_memory:
            command.append("--dataloader-pin-memory")

        proc = subprocess.Popen(  # noqa: S603
            command,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path.cwd(),
        )

    update_run_state(
        state_path=run_dir / "state.json",
        status="running",
        metrics={"pid": proc.pid},
    )
    return proc.pid


def pid_is_alive(pid: int) -> bool:
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True
