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
