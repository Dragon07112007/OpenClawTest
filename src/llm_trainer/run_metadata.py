from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class RunFiles:
    run_id: str
    run_dir: Path
    meta_path: Path
    state_path: Path


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


def initialize_run(
    *,
    config_path: str,
    device: str,
    runs_root: str | Path = "runs",
    initial_status: str = "queued",
) -> RunFiles:
    """Create a run directory with initial metadata and state files."""
    run_id = _generate_run_id()
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    started_at = _utc_now_iso()

    meta = {
        "run_id": run_id,
        "started_at": started_at,
        "device": device,
        "config_path": str(config_path),
    }
    state = {
        "status": initial_status,
        "updated_at": started_at,
        "history": [{"status": initial_status, "timestamp": started_at}],
    }

    meta_path = run_dir / "meta.json"
    state_path = run_dir / "state.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    return RunFiles(run_id=run_id, run_dir=run_dir, meta_path=meta_path, state_path=state_path)
