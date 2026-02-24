from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .app_logic import (
    DeviceResolutionError,
    GenerateOptions,
    resume_training_run,
    run_generation,
    start_training_run,
)
from .telemetry import collect_host_telemetry


def _markup_safe(value: object) -> str:
    # Escape every opening bracket to avoid Rich/Textual markup parsing on dynamic text.
    return str(value).replace("[", "\\[")


def _join_markup_safe(lines: list[object]) -> str:
    return "\n".join(_markup_safe(line) for line in lines)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _fmt_eta(value: str | None) -> str:
    if not value:
        return "n/a"
    try:
        eta = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return value
    return eta.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fmt_loss(value: object) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.4f}"
    return "n/a" if value is None else str(value)


def _fmt_float(value: object, suffix: str = "") -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.1f}{suffix}"
    return "n/a"


def _fmt_gpu(state: dict[str, Any]) -> str:
    util = state.get("gpu_utilization_pct")
    used = state.get("gpu_memory_used_mb")
    total = state.get("gpu_memory_total_mb")
    if not isinstance(util, (float, int)) or not isinstance(used, (float, int)):
        return "n/a"
    total_text = "n/a" if not isinstance(total, (float, int)) else f"{float(total):.0f}"
    return f"{float(util):.0f}% {float(used):.0f}/{total_text}MB"


def _is_running_status(status: str) -> bool:
    return status.lower() in {"queued", "running", "resuming"}


def _is_active_status(status: str) -> bool:
    return status.lower() in {"queued", "running", "resuming", "paused"}


def _latest_runs(runs_root: Path, limit: int = 60) -> list[Path]:
    if not runs_root.exists():
        return []
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (float, int)) else None


MIN_EPOCHS = 1
MAX_EPOCHS = 10_000
MODEL_NAME_METADATA_FILE = "_model_names.json"
MODEL_NAME_MAX_LEN = 64
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]*$")


def _clamp_int(value: object, *, default: int, min_value: int, max_value: int) -> int:
    if isinstance(value, bool):
        candidate = int(value)
    elif isinstance(value, int):
        candidate = value
    elif isinstance(value, float):
        candidate = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            candidate = int(stripped)
        else:
            candidate = default
    else:
        candidate = default
    return min(max(candidate, min_value), max_value)


@dataclass
class TuiTrainingOptions:
    config: str = "configs/default.toml"
    device: str = "auto"
    strict_device: bool = False
    epochs: int = 3
    batch_size: int = 16
    seq_length: int = 256
    precision: str = "off"
    grad_accum_steps: int = 1


@dataclass
class TuiGenerationOptions:
    prompt: str = "Hello"
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50


@dataclass(frozen=True)
class RunEntry:
    run_dir: Path
    meta: dict[str, Any]
    state: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.meta.get("run_id", self.run_dir.name))


@dataclass(frozen=True)
class ModelEntry:
    run_id: str
    display_name: str
    checkpoint_dir: Path
    checkpoint_path: Path
    mtime: float
    has_run_meta: bool


@dataclass
class TuiSharedState:
    selected_index: int = 0
    selected_run_id: str | None = None
    selected_model_index: int = 0
    active_model_run_id: str | None = None
    focused_panel_index: int = 0
    generation_output: str = ""
    prompt_edit_mode: bool = False
    prompt_cursor_index: int | None = None
    pending_confirmation: tuple[str, str | None] | None = None
    last_action: str = "none"
    run_scroll_offset: int = 0
    generation_scroll_offset: int = 0
    rename_edit_mode: bool = False
    rename_buffer: str = ""


def reduce_tui_state(state: TuiSharedState, event: str, value: object | None = None) -> None:
    if event == "select_run_delta":
        delta = int(value) if isinstance(value, int) else 0
        state.selected_index += delta
        state.selected_run_id = None
    elif event == "select_model_delta":
        delta = int(value) if isinstance(value, int) else 0
        state.selected_model_index += delta
    elif event == "set_active_model":
        state.active_model_run_id = str(value) if isinstance(value, str) else None
    elif event == "set_last_action":
        if isinstance(value, str):
            state.last_action = value
    elif event == "set_generation_output":
        state.generation_output = str(value) if isinstance(value, str) else ""
        state.generation_scroll_offset = 0
    elif event == "set_prompt_edit_mode":
        state.prompt_edit_mode = bool(value)
    elif event == "set_prompt_cursor":
        state.prompt_cursor_index = int(value) if isinstance(value, int) else None
    elif event == "set_pending_confirmation":
        state.pending_confirmation = value if isinstance(value, tuple) else None
    elif event == "scroll_run_dashboard":
        delta = int(value) if isinstance(value, int) else 0
        state.run_scroll_offset = max(0, state.run_scroll_offset + delta)
    elif event == "scroll_generation":
        delta = int(value) if isinstance(value, int) else 0
        state.generation_scroll_offset = max(0, state.generation_scroll_offset + delta)
    elif event == "set_rename_edit_mode":
        state.rename_edit_mode = bool(value)
    elif event == "set_rename_buffer":
        state.rename_buffer = str(value) if isinstance(value, str) else ""


def _validate_training_options(options: TuiTrainingOptions) -> str | None:
    if options.epochs < MIN_EPOCHS:
        return f"epochs must be >= {MIN_EPOCHS}"
    if options.epochs > MAX_EPOCHS:
        return f"epochs must be <= {MAX_EPOCHS}"
    if options.batch_size < 1:
        return "batch size must be >= 1"
    if options.seq_length < 1:
        return "seq length must be >= 1"
    if options.grad_accum_steps < 1:
        return "grad_accum_steps must be >= 1"
    if options.precision not in {"off", "fp16", "bf16"}:
        return "precision must be one of off/fp16/bf16"
    return None


def _validate_generation_options(options: TuiGenerationOptions) -> str | None:
    if not options.prompt.strip():
        return "prompt must not be empty"
    if options.max_new_tokens < 1:
        return "max_new_tokens must be >= 1"
    if options.temperature <= 0:
        return "temperature must be > 0"
    if options.top_k < 1:
        return "top_k must be >= 1"
    return None


def tui_start_training(options: TuiTrainingOptions) -> tuple[bool, str]:
    error = _validate_training_options(options)
    if error:
        return (False, f"start failed: {error}")

    try:
        result = start_training_run(
            config_path=options.config,
            requested_device=options.device,
            strict_device=options.strict_device,
            foreground=False,
            epochs=options.epochs,
            batch_size=options.batch_size,
            seq_length=options.seq_length,
            precision=options.precision,
            grad_accum_steps=options.grad_accum_steps,
        )
    except (DeviceResolutionError, FileNotFoundError, ValueError) as exc:
        return (False, f"start failed: {exc}")

    warn = f" warning={result.selection.warning}" if result.selection.warning else ""
    return (
        True,
        "started "
        f"run_id={result.run_files.run_id} "
        f"device={result.selection.selected}{warn} "
        f"mode={result.training_mode}",
    )


def _resolve_resume_target(
    model_run_id: str,
    *,
    runs_root: Path = Path("runs"),
    checkpoints_root: Path = Path("checkpoints"),
) -> tuple[Path, Path]:
    run_dir = runs_root / model_run_id
    meta_path = run_dir / "meta.json"
    if not run_dir.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "run metadata not found for model "
            f"run_id={model_run_id}; cannot resume without {meta_path}"
        )

    checkpoint_dir = checkpoints_root / model_run_id
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found for model run_id={model_run_id}")

    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return run_dir, latest

    candidates = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint files (*.pt) found for model run_id={model_run_id}")
    return run_dir, candidates[0]


def tui_resume_training(
    model_run_id: str,
    options: TuiTrainingOptions,
    *,
    runs_root: str | Path = "runs",
    checkpoints_root: str | Path = "checkpoints",
) -> tuple[bool, str]:
    error = _validate_training_options(options)
    if error:
        return (False, f"resume failed: {error}")

    try:
        run_dir, checkpoint_path = _resolve_resume_target(
            model_run_id,
            runs_root=Path(runs_root),
            checkpoints_root=Path(checkpoints_root),
        )
        mode, _ = resume_training_run(
            run_dir=run_dir,
            config_path=options.config,
            checkpoint_path=str(checkpoint_path),
            more_epochs=options.epochs,
            foreground=False,
            requested_device=options.device,
            strict_device=options.strict_device,
            precision=options.precision,
            grad_accum_steps=options.grad_accum_steps,
        )
    except (DeviceResolutionError, FileNotFoundError, ValueError) as exc:
        return (False, f"resume failed: {exc}")
    return (
        True,
        "resumed "
        f"model={model_run_id} "
        f"checkpoint={checkpoint_path} "
        f"mode={mode}",
    )


def tui_generate_from_run(run_id: str, options: TuiGenerationOptions) -> tuple[bool, str]:
    error = _validate_generation_options(options)
    if error:
        return (False, f"generate failed: {error}")

    run_dir = Path("runs") / run_id
    checkpoint_path = str(Path("checkpoints") / run_id / "latest.pt")
    try:
        text, selection = run_generation(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            device_request=None,
            strict_device=False,
            options=GenerateOptions(
                prompt=options.prompt,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_k=options.top_k,
            ),
        )
    except (DeviceResolutionError, FileNotFoundError, ValueError) as exc:
        return (False, f"generate failed: {exc}")
    return (True, f"generated with {selection.selected}:\n{text}")


def _collect_run_entries(
    runs_root: Path, pinned_run_id: str | None = None
) -> tuple[list[RunEntry], list[str]]:
    if pinned_run_id:
        run_dirs = [runs_root / pinned_run_id]
    else:
        run_dirs = _latest_runs(runs_root)

    entries: list[RunEntry] = []
    load_errors: list[str] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            load_errors.append(f"{run_dir.name}: run not found")
            continue
        try:
            entries.append(
                RunEntry(
                    run_dir=run_dir,
                    meta=_read_json(run_dir / "meta.json"),
                    state=_read_json(run_dir / "state.json"),
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            load_errors.append(f"{run_dir.name}: {exc}")

    entries.sort(
        key=lambda entry: (
            not _is_running_status(str(entry.state.get("status", "unknown"))),
            -entry.run_dir.stat().st_mtime,
        )
    )
    return entries, load_errors


def _resolve_selected_index(
    entries: list[RunEntry],
    selected_index: int,
    selected_run_id: str | None,
) -> int:
    if not entries:
        return 0
    if selected_run_id:
        for idx, entry in enumerate(entries):
            if entry.run_id == selected_run_id:
                return idx
    return min(max(selected_index, 0), len(entries) - 1)


def collect_model_entries(
    *,
    checkpoints_root: str | Path = "checkpoints",
    runs_root: str | Path = "runs",
) -> list[ModelEntry]:
    checkpoints_root_path = Path(checkpoints_root)
    runs_root_path = Path(runs_root)
    if not checkpoints_root_path.exists():
        return []
    model_names = _load_model_name_map(checkpoints_root=checkpoints_root_path)

    entries: list[ModelEntry] = []
    for run_dir in checkpoints_root_path.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        latest = run_dir / "latest.pt"
        if latest.exists():
            checkpoint_path = latest
        else:
            candidates = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                continue
            checkpoint_path = candidates[0]
        entries.append(
            ModelEntry(
                run_id=run_dir.name,
                display_name=_effective_model_name(run_dir.name, model_names=model_names),
                checkpoint_dir=run_dir,
                checkpoint_path=checkpoint_path,
                mtime=checkpoint_path.stat().st_mtime,
                has_run_meta=(runs_root_path / run_dir.name / "meta.json").exists(),
            )
        )

    return sorted(entries, key=lambda entry: entry.mtime, reverse=True)


def archive_model_run(
    run_id: str,
    *,
    checkpoints_root: str | Path = "checkpoints",
) -> tuple[bool, str]:
    source = Path(checkpoints_root) / run_id
    if not source.exists() or not source.is_dir():
        return (False, f"archive failed: checkpoint dir not found for run_id={run_id}")

    archive_root = Path(checkpoints_root) / "_archived"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = archive_root / f"{run_id}-{timestamp}"
    source.rename(destination)
    return (True, f"archived run_id={run_id} to {destination}")


def delete_model_run(
    run_id: str,
    *,
    checkpoints_root: str | Path = "checkpoints",
) -> tuple[bool, str]:
    target = Path(checkpoints_root) / run_id
    if not target.exists() or not target.is_dir():
        return (False, f"delete failed: checkpoint dir not found for run_id={run_id}")
    shutil.rmtree(target)
    return (True, f"deleted checkpoint dir for run_id={run_id}")


def _model_is_running(run_id: str, *, runs_root: Path) -> bool:
    state_path = runs_root / run_id / "state.json"
    if not state_path.exists():
        return False
    try:
        state = _read_json(state_path)
    except (OSError, json.JSONDecodeError):
        return False
    return _is_active_status(str(state.get("status", "")))


def _model_name_map_path(*, checkpoints_root: Path) -> Path:
    return checkpoints_root / MODEL_NAME_METADATA_FILE


def _load_model_name_map(*, checkpoints_root: Path) -> dict[str, str]:
    path = _model_name_map_path(checkpoints_root=checkpoints_root)
    if not path.exists():
        return {}
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            result[key] = value
    return result


def _save_model_name_map(model_names: dict[str, str], *, checkpoints_root: Path) -> None:
    path = _model_name_map_path(checkpoints_root=checkpoints_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_names, indent=2, sort_keys=True), encoding="utf-8")


def _effective_model_name(run_id: str, *, model_names: dict[str, str]) -> str:
    override = model_names.get(run_id)
    if isinstance(override, str) and override.strip():
        return override
    return run_id


def _validate_model_display_name(name: str) -> str | None:
    cleaned = name.strip()
    if not cleaned:
        return "name must not be empty"
    if len(cleaned) > MODEL_NAME_MAX_LEN:
        return f"name must be <= {MODEL_NAME_MAX_LEN} chars"
    if not MODEL_NAME_PATTERN.fullmatch(cleaned):
        return "name contains invalid characters (allowed: letters, numbers, space, _, -, .)"
    return None


def rename_model_run(
    run_id: str,
    new_name: str,
    *,
    checkpoints_root: str | Path = "checkpoints",
) -> tuple[bool, str]:
    checkpoints_root_path = Path(checkpoints_root)
    model_dir = checkpoints_root_path / run_id
    if not model_dir.exists() or not model_dir.is_dir():
        return (False, f"rename failed: checkpoint dir not found for run_id={run_id}")

    candidate = new_name.strip()
    error = _validate_model_display_name(candidate)
    if error:
        return (False, f"rename failed: {error}")

    model_names = _load_model_name_map(checkpoints_root=checkpoints_root_path)
    current_name = _effective_model_name(run_id, model_names=model_names)
    if candidate == current_name:
        return (True, f"rename unchanged for run_id={run_id} name={candidate}")

    existing = collect_model_entries(
        checkpoints_root=checkpoints_root_path,
        runs_root="runs",
    )
    existing_names = {
        entry.display_name.casefold()
        for entry in existing
        if entry.run_id != run_id
    }
    if candidate.casefold() in existing_names:
        return (False, f"rename failed: name collision with existing model name={candidate}")

    if candidate == run_id:
        model_names.pop(run_id, None)
    else:
        model_names[run_id] = candidate

    try:
        _save_model_name_map(model_names, checkpoints_root=checkpoints_root_path)
    except OSError as exc:
        return (False, f"rename failed: {exc}")
    return (True, f"renamed run_id={run_id} to name={candidate}")


def collect_system_utilization(
    *,
    selected_run_state: dict[str, Any] | None,
    selected_device: str,
) -> dict[str, Any]:
    return collect_host_telemetry(device=selected_device, selected_run_state=selected_run_state)


def _launch_help_text(
    training_options: TuiTrainingOptions,
    generation_options: TuiGenerationOptions,
) -> str:
    return (
        "Focus: tab/shift+tab or h/l or 1-5 | nav: j/k or up/down | refresh: r | "
        "launch: s start / u resume model | generate: x | edit prompt: enter, esc | "
        "model: a activate, e rename, i inspect, A archive, D delete | "
        f"epochs +/- via [/] or +/- ({training_options.epochs}) | "
        f"prompt len={len(generation_options.prompt)}"
    )


PANEL_ORDER = ["panel-a", "panel-b", "panel-c", "panel-d", "panel-e"]
PANEL_LABELS = {
    "panel-a": "Run Dashboard",
    "panel-b": "System Dashboard",
    "panel-c": "Train Selected Model",
    "panel-d": "Generate From Model",
    "panel-e": "Model Selection",
}
PANEL_SHORTCUTS = {
    "1": "panel-a",
    "2": "panel-b",
    "3": "panel-c",
    "4": "panel-d",
    "5": "panel-e",
}
PANEL_CONTEXT_HINTS = {
    "panel-a": "runs: j/k=select up/down=select",
    "panel-b": "system: r=refresh",
    "panel-c": (
        "training: s=start u=resume [ ]=epochs-+ -/+/minus/plus/equals=epochs-+ "
        "b/B=batch-+ v=strict-toggle d=device-cycle p=precision-cycle"
    ),
    "panel-d": (
        "generation: x=generate enter=prompt-mode esc=cancel-edit "
        "m/M=max-tokens-+ k/K=top-k-+ t/T=temp-+"
    ),
    "panel-e": (
        "models: a=activate e=rename i=inspect A=archive D=delete r=refresh "
        "j/k=select up/down=select"
    ),
}


def _footer_hint_line(
    *,
    focused_panel: str,
    prompt_edit_mode: bool,
    rename_edit_mode: bool,
    pending_confirmation: str | None,
) -> str:
    base = "global: tab=next-panel shift+tab=prev-panel"
    context = PANEL_CONTEXT_HINTS.get(focused_panel, "n/a")

    if prompt_edit_mode:
        context = (
            "prompt edit: text=insert space=insert-space backspace=delete-left "
            "delete=delete-right left/right=move home/end=edge enter=save esc=cancel"
        )
    if rename_edit_mode:
        context = (
            "rename edit: text=insert space=insert-space backspace=delete-left "
            "delete=clear enter=save esc=cancel"
        )
    if pending_confirmation:
        context = f"confirm {pending_confirmation}: y=confirm n=cancel esc=cancel"
    return f"{base} | {context}"


def _next_focus_index(current: int, *, reverse: bool, order: list[str]) -> int:
    if not order:
        return 0
    delta = -1 if reverse else 1
    return (current + delta) % len(order)


def _jump_focus_index(key: str, *, order: list[str]) -> int | None:
    panel = PANEL_SHORTCUTS.get(key)
    if panel is None:
        return None
    try:
        return order.index(panel)
    except ValueError:
        return None


def _run_detail_lines(
    meta: dict,
    state: dict,
    *,
    training_options: TuiTrainingOptions,
    generation_options: TuiGenerationOptions,
    last_action: str | None,
) -> list[str]:
    return [
        f"Run: {meta.get('run_id')}",
        f"Status: {state.get('status', 'unknown')}",
        f"Epoch: {state.get('epoch', 'n/a')}",
        f"Step: {state.get('global_step', 'n/a')}",
        f"Train loss: {_fmt_loss(state.get('train_loss'))}",
        f"Val loss: {_fmt_loss(state.get('val_loss'))}",
        f"Elapsed: {_fmt_duration(state.get('elapsed_seconds'))}",
        f"Remaining: {_fmt_duration(state.get('remaining_seconds'))}",
        f"ETA: {_fmt_eta(state.get('eta_at'))}",
        f"Device: {meta.get('selected_device', meta.get('device', 'n/a'))}",
        f"Requested device: {meta.get('requested_device', 'auto')}",
        f"GPU util: {_fmt_float(state.get('gpu_utilization_pct'))}",
        "GPU memory: "
        f"{_fmt_float(state.get('gpu_memory_used_mb'))}/{_fmt_float(state.get('gpu_memory_total_mb'))}MB",
        f"GPU temp: {_fmt_float(state.get('gpu_temperature_c'), 'C')}",
        f"GPU power: {_fmt_float(state.get('gpu_power_w'), 'W')}",
        f"PID: {state.get('pid', 'n/a')}",
        f"Updated: {state.get('updated_at', 'n/a')}",
        "",
        "Launch options: "
        f"cfg={training_options.config} epochs={training_options.epochs} "
        f"batch={training_options.batch_size} seq={training_options.seq_length}",
        "Launch options: "
        f"device={training_options.device} strict={training_options.strict_device} "
        f"precision={training_options.precision} grad_accum={training_options.grad_accum_steps}",
        "Generate options: "
        f"prompt={generation_options.prompt!r} max_new_tokens={generation_options.max_new_tokens} "
        f"temperature={generation_options.temperature} top_k={generation_options.top_k}",
        _launch_help_text(training_options, generation_options),
        f"Last action: {last_action or 'none'}",
    ]


def _active_run_entries(entries: list[RunEntry]) -> list[RunEntry]:
    return [entry for entry in entries if _is_active_status(str(entry.state.get("status", "")))]


def _runs_panel_lines(
    entries: list[RunEntry],
    *,
    selected: int,
    scroll_offset: int,
    window_size: int = 10,
) -> tuple[list[str], int]:
    lines = ["Run Dashboard", "RUN_ID | Status | Epoch | Loss | Device | ETA | GPU%"]
    if not entries:
        lines.append("No active runs.")
        return lines, 0

    selected = min(max(selected, 0), len(entries) - 1)
    max_offset = max(0, len(entries) - window_size)
    offset = min(max(scroll_offset, 0), max_offset)
    if selected < offset:
        offset = selected
    if selected >= offset + window_size:
        offset = selected - window_size + 1

    visible = entries[offset : offset + window_size]
    for idx, entry in enumerate(visible, start=offset):
        state = entry.state
        marker = ">" if idx == selected else " "
        lines.append(
            f"{marker} {entry.run_id} | {state.get('status', 'unknown')} | "
            f"{state.get('epoch', 'n/a')} | {_fmt_loss(state.get('train_loss'))} | "
            f"{entry.meta.get('selected_device', entry.meta.get('device', 'n/a'))} | "
            f"{_fmt_eta(state.get('eta_at'))} | "
            f"{_fmt_float(state.get('gpu_utilization_pct'), '%')}"
        )
    if len(entries) > window_size:
        lines.append(f"Showing {offset + 1}-{offset + len(visible)} of {len(entries)} active runs")
    return lines, offset


def _aggregate_active_remaining_time(entries: list[RunEntry]) -> str:
    active = _active_run_entries(entries)
    if not active:
        return "No active training"

    total_seconds = 0.0
    saw_value = False
    for entry in active:
        remaining = _as_float(entry.state.get("remaining_seconds"))
        if remaining is not None:
            total_seconds += max(0.0, remaining)
            saw_value = True
    if not saw_value:
        return "n/a"
    return _fmt_duration(total_seconds)


def _launcher_panel_lines(
    training_options: TuiTrainingOptions,
    active_model_run_id: str | None,
    active_model_display_name: str | None,
    pending_confirmation: str | None,
) -> list[str]:
    selected_model_text = active_model_display_name or active_model_run_id or "none"
    if (
        active_model_display_name
        and active_model_run_id
        and active_model_display_name != active_model_run_id
    ):
        selected_model_text = f"{active_model_display_name} ({active_model_run_id})"
    return [
        "Train Selected Model",
        f"Selected Model: {selected_model_text}",
        f"epochs={training_options.epochs}",
        f"device={training_options.device}",
        f"batch_size={training_options.batch_size}",
        f"config={training_options.config}",
        f"confirmation={pending_confirmation or 'none'}",
    ]


def _generation_panel_lines(
    generation_options: TuiGenerationOptions,
    active_model_run_id: str | None,
    active_model_display_name: str | None,
    generation_output: str,
    prompt_edit_mode: bool,
    prompt_cursor_index: int | None,
    *,
    output_scroll_offset: int,
    output_window: int = 8,
) -> tuple[list[str], int]:
    prompt_line = generation_options.prompt
    if prompt_edit_mode:
        cursor = min(max(prompt_cursor_index or 0, 0), len(prompt_line))
        prompt_line = f"{prompt_line[:cursor]}|{prompt_line[cursor:]}"

    output_lines = generation_output.splitlines() or ["(no generation output yet)"]
    max_offset = max(0, len(output_lines) - output_window)
    offset = min(max(output_scroll_offset, 0), max_offset)
    visible_output = output_lines[offset : offset + output_window]
    selected_model_text = active_model_display_name or active_model_run_id or "none"
    if (
        active_model_display_name
        and active_model_run_id
        and active_model_display_name != active_model_run_id
    ):
        selected_model_text = f"{active_model_display_name} ({active_model_run_id})"
    return [
        "Generate From Model",
        f"Selected Model: {selected_model_text}",
        f"MAX_TOKENS={generation_options.max_new_tokens}",
        f"temperature={generation_options.temperature:.2f} top_k={generation_options.top_k}",
        f"prompt edit mode={'on' if prompt_edit_mode else 'off'}",
        f"prompt: {prompt_line}",
        "output:",
        *visible_output,
        (
            f"output lines {offset + 1}-{offset + len(visible_output)} of {len(output_lines)}"
            if len(output_lines) > output_window
            else "output lines 1-1 of 1"
            if output_lines == ["(no generation output yet)"]
            else f"output lines {offset + 1}-{offset + len(visible_output)} of {len(output_lines)}"
        ),
    ], offset


def _utilization_panel_lines(
    system: dict[str, Any],
    selected_device: str,
    *,
    aggregate_remaining_time: str,
) -> list[str]:
    lines = [
        "System Dashboard",
        f"device={selected_device}",
        "GPU: "
        f"Usage={_fmt_float(system.get('gpu_utilization_pct'), '%')} "
        f"VRAM={_fmt_float(system.get('gpu_memory_used_mb'))}/"
        f"{_fmt_float(system.get('gpu_memory_total_mb'))}MB "
        f"Temp={_fmt_float(system.get('gpu_temperature_c'), 'C')}",
        "CPU: "
        f"Usage={_fmt_float(system.get('cpu_utilization_pct'), '%')} "
        f"Cores={system.get('cpu_count', 'n/a')}",
        "RAM: "
        f"{_fmt_float(system.get('ram_used_mb'))}/{_fmt_float(system.get('ram_total_mb'))}MB",
    ]
    gpu_reason = system.get("gpu_telemetry_reason")
    cpu_reason = system.get("cpu_telemetry_reason")
    if isinstance(gpu_reason, str) and gpu_reason:
        lines.append(f"GPU diag: {gpu_reason}")
    if isinstance(cpu_reason, str) and cpu_reason:
        lines.append(f"CPU diag: {cpu_reason}")
    if aggregate_remaining_time == "No active training":
        lines.append("No active training")
    else:
        lines.append(f"Aggregate remaining: {aggregate_remaining_time}")
    return lines


def _model_size_mb(checkpoint_path: Path) -> float:
    try:
        return round(checkpoint_path.stat().st_size / (1024 * 1024), 1)
    except OSError:
        return 0.0


def _model_status(run_id: str, *, runs_root: Path) -> str:
    state_path = runs_root / run_id / "state.json"
    if not state_path.exists():
        return "orphan"
    try:
        state = _read_json(state_path)
    except (OSError, json.JSONDecodeError):
        return "corrupted"
    status = str(state.get("status", "unknown"))
    return "training" if _is_active_status(status) else status


def _models_panel_lines(
    models: list[ModelEntry],
    *,
    selected_model_index: int,
    active_model_run_id: str | None,
    runs_root: Path,
    rename_edit_mode: bool,
    rename_buffer: str,
) -> tuple[list[str], int, str | None, str | None]:
    if not models:
        return (
            [
                "Model Selection",
                "No checkpoints found.",
                "Trained runs with checkpoints will appear here.",
            ],
            0,
            None,
            None,
        )

    selected = min(max(selected_model_index, 0), len(models) - 1)
    latest_run_id = models[0].run_id
    latest_display_name = models[0].display_name
    lines = [
        "Model Selection",
        "name | trained | epochs | final loss | disk size | status",
    ]
    for idx, model in enumerate(models):
        marker = ">" if idx == selected else " "
        trained_text = datetime.fromtimestamp(model.mtime, tz=UTC).strftime("%Y-%m-%d %H:%M")
        epochs = "n/a"
        final_loss = "n/a"
        state_path = runs_root / model.run_id / "state.json"
        if state_path.exists():
            try:
                state = _read_json(state_path)
                epochs = str(state.get("epoch", "n/a"))
                final_loss = _fmt_loss(state.get("train_loss"))
            except (OSError, json.JSONDecodeError):
                final_loss = "corrupted"

        status = _model_status(model.run_id, runs_root=runs_root)
        if model.run_id == latest_run_id:
            status = f"{status},latest"
        if model.run_id == active_model_run_id:
            status = f"{status},active"
        lines.append(
            f"{marker} {model.display_name} ({model.run_id}) | "
            f"{trained_text} | {epochs} | {final_loss} | "
            f"{_model_size_mb(model.checkpoint_path):.1f}MB | {status}"
        )
    lines.append(f"Latest trained model: {latest_display_name} ({latest_run_id})")
    if rename_edit_mode:
        lines.append(f"rename> {rename_buffer}")
    return lines, selected, models[selected].run_id, models[selected].display_name


def build_tui_snapshot(
    *,
    runs_root: str | Path = "runs",
    checkpoints_root: str | Path = "checkpoints",
    run_id: str | None = None,
    selected_index: int = 0,
    selected_run_id: str | None = None,
    selected_model_index: int = 0,
    active_model_run_id: str | None = None,
    training_options: TuiTrainingOptions | None = None,
    generation_options: TuiGenerationOptions | None = None,
    generation_output: str = "",
    prompt_edit_mode: bool = False,
    prompt_cursor_index: int | None = None,
    run_scroll_offset: int = 0,
    generation_scroll_offset: int = 0,
    pending_confirmation: str | None = None,
    last_action: str | None = None,
    rename_edit_mode: bool = False,
    rename_buffer: str = "",
) -> dict[str, object]:
    runs_root_path = Path(runs_root)
    training_options = training_options or TuiTrainingOptions()
    generation_options = generation_options or TuiGenerationOptions()

    entries, load_errors = _collect_run_entries(runs_root_path, pinned_run_id=run_id)
    selected = _resolve_selected_index(entries, selected_index, selected_run_id)
    selected_entry = entries[selected] if entries else None
    selected_run_value = selected_entry.run_id if selected_entry else None

    detail = (
        _run_detail_lines(
            selected_entry.meta,
            selected_entry.state,
            training_options=training_options,
            generation_options=generation_options,
            last_action=last_action,
        )
        if selected_entry
        else [
            "Empty state",
            "The monitor will auto-refresh every second.",
            _launch_help_text(training_options, generation_options),
        ]
    )

    models = collect_model_entries(checkpoints_root=checkpoints_root, runs_root=runs_root_path)
    models_lines, resolved_model_index, selected_model_run_id, selected_model_display_name = (
        _models_panel_lines(
        models,
        selected_model_index=selected_model_index,
        active_model_run_id=active_model_run_id,
        runs_root=runs_root_path,
        rename_edit_mode=rename_edit_mode,
        rename_buffer=rename_buffer,
    )
    )

    effective_active_model = active_model_run_id
    model_name_by_id = {m.run_id: m.display_name for m in models}
    known_model_ids = set(model_name_by_id)
    if effective_active_model not in known_model_ids:
        effective_active_model = None
    if effective_active_model is None and selected_model_run_id:
        effective_active_model = selected_model_run_id
    if effective_active_model is None and models:
        effective_active_model = models[0].run_id
    effective_active_model_name = (
        model_name_by_id.get(effective_active_model) if effective_active_model else None
    )

    selected_device = "cpu"
    if selected_entry:
        selected_device = str(
            selected_entry.meta.get("selected_device", selected_entry.meta.get("device", "cpu"))
        )
    system = collect_system_utilization(
        selected_run_state=(selected_entry.state if selected_entry else None),
        selected_device=selected_device,
    )

    active_entries = _active_run_entries(entries)
    resolved_selected = _resolve_selected_index(
        active_entries,
        selected_index,
        selected_run_id if selected_run_id in {entry.run_id for entry in active_entries} else None,
    )
    runs_lines, resolved_run_scroll_offset = _runs_panel_lines(
        active_entries,
        selected=resolved_selected,
        scroll_offset=run_scroll_offset,
    )
    generation_lines, resolved_generation_scroll_offset = _generation_panel_lines(
        generation_options,
        active_model_run_id=effective_active_model,
        active_model_display_name=effective_active_model_name,
        generation_output=generation_output,
        prompt_edit_mode=prompt_edit_mode,
        prompt_cursor_index=prompt_cursor_index,
        output_scroll_offset=generation_scroll_offset,
    )
    selected_model_text = selected_model_display_name or selected_model_run_id or "none"
    if (
        selected_model_display_name
        and selected_model_run_id
        and selected_model_display_name != selected_model_run_id
    ):
        selected_model_text = f"{selected_model_display_name} ({selected_model_run_id})"
    active_model_text = effective_active_model_name or effective_active_model or "none"
    if (
        effective_active_model_name
        and effective_active_model
        and effective_active_model_name != effective_active_model
    ):
        active_model_text = f"{effective_active_model_name} ({effective_active_model})"

    status_lines = [
        "Dashboard Status",
        f"selected run={selected_run_value or 'none'}",
        f"active runs={len(active_entries)}",
        f"selected model={selected_model_text}",
        f"active model={active_model_text}",
        f"last action={last_action or 'none'}",
        f"errors={'yes' if load_errors else 'no'}",
    ]
    if run_id:
        status_lines.append(f"pinned run: {run_id}")

    if load_errors:
        detail.append("")
        detail.append("Load warnings:")
        detail.extend(load_errors[:3])

    return {
        "runs": runs_lines,
        "detail": detail,
        "launcher": _launcher_panel_lines(
            training_options,
            active_model_run_id=effective_active_model,
            active_model_display_name=effective_active_model_name,
            pending_confirmation=pending_confirmation,
        ),
        "generation": generation_lines,
        "utilization": _utilization_panel_lines(
            system,
            selected_device=selected_device,
            aggregate_remaining_time=_aggregate_active_remaining_time(entries),
        ),
        "models": models_lines,
        "status": status_lines,
        "error": bool(load_errors),
        "selected": resolved_selected,
        "selected_run_id": active_entries[resolved_selected].run_id if active_entries else None,
        "run_scroll_offset": resolved_run_scroll_offset,
        "generation_scroll_offset": resolved_generation_scroll_offset,
        "selected_model_index": resolved_model_index,
        "selected_model_run_id": selected_model_run_id,
        "active_model_run_id": effective_active_model,
    }


TUI_GRID_CSS = """
Screen {
    layout: vertical;
}
#grid {
    height: 1fr;
    padding: 1;
    layout: grid;
    grid-size: 2 3;
    grid-columns: 3fr 2fr;
    grid-rows: 1fr 1fr 1fr;
    grid-gutter: 1 1;
}
.panel {
    border: round #6689a1;
    border-title-align: center;
    padding: 0 1;
    overflow-y: auto;
}
#key-footer {
    dock: bottom;
    height: 1;
    padding: 0 1;
    background: #1f2a30;
    color: #d8e4ed;
}
#panel-a {
    border: round #5d7fa7;
}
#panel-b {
    border: round #a17f66;
}
#panel-c {
    border: round #7a77a0;
}
#panel-d {
    border: round #99865c;
}
#panel-e {
    row-span: 2;
    border: round #608a91;
}
"""


def launch_tui(
    *,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    return_app: bool = False,
    refresh_interval: float = 1.0,
) -> int | Any:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Grid
        from textual.widgets import Header, Static
    except ModuleNotFoundError:
        print("tui failed (missing dependency: textual)")
        return 1

    runs_root_path = Path(runs_root)
    checkpoints_root_path = Path("checkpoints")

    class MonitorApp(App):
        CSS = TUI_GRID_CSS
        panel_order = PANEL_ORDER

        def __init__(self) -> None:
            super().__init__()
            self.shared = TuiSharedState()
            self.training_options = TuiTrainingOptions()
            self.generation_options = TuiGenerationOptions()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Grid(id="grid"):
                yield Static("", id="panel-a", classes="panel")
                yield Static("", id="panel-b", classes="panel")
                yield Static("", id="panel-c", classes="panel")
                yield Static("", id="panel-e", classes="panel")
                yield Static("", id="panel-d", classes="panel")
            yield Static("", id="key-footer")

        def on_mount(self) -> None:
            self._apply_focus_titles()
            if refresh_interval > 0:
                self.set_interval(refresh_interval, self._refresh_content)
            self._refresh_content()

        def _focused_panel(self) -> str:
            return self.panel_order[self.shared.focused_panel_index]

        def _cycle_focus(self, reverse: bool = False) -> None:
            self.shared.focused_panel_index = _next_focus_index(
                self.shared.focused_panel_index,
                reverse=reverse,
                order=self.panel_order,
            )
            reduce_tui_state(self.shared, "set_last_action", f"focus={self._focused_panel()}")

        def _apply_focus_titles(self) -> None:
            focused = self._focused_panel()
            for panel_id, label in PANEL_LABELS.items():
                title = f"{label} (FOCUSED)" if panel_id == focused else label
                self.query_one(f"#{panel_id}", Static).border_title = title

        def on_key(self, event) -> None:
            key = event.key
            character = getattr(event, "character", None)

            if self.shared.rename_edit_mode:
                if self._handle_rename_edit_key(key):
                    self._refresh_content()
                return

            if self.shared.prompt_edit_mode:
                if self._handle_prompt_edit_key(key, character):
                    self._refresh_content()
                return

            if self.shared.pending_confirmation:
                self._handle_confirmation_key(key)
                return

            if key in {"tab", "right", "l"}:
                self._cycle_focus(reverse=False)
                self._refresh_content()
                return
            if key in {"shift+tab", "left", "h"}:
                self._cycle_focus(reverse=True)
                self._refresh_content()
                return
            jump_index = _jump_focus_index(key, order=self.panel_order)
            if jump_index is not None:
                self.shared.focused_panel_index = jump_index
                reduce_tui_state(self.shared, "set_last_action", f"focus={self._focused_panel()}")
                self._refresh_content()
                return
            if key == "r":
                if self._focused_panel() == "panel-e" and self._handle_model_keys(key):
                    self._refresh_content()
                    return
                self._refresh_content()
                return
            if key == "q":
                self.exit()
                return

            if key in {"j", "down"}:
                panel = self._focused_panel()
                if panel == "panel-a":
                    reduce_tui_state(self.shared, "select_run_delta", 1)
                elif panel == "panel-e":
                    reduce_tui_state(self.shared, "select_model_delta", 1)
                self._refresh_content()
                return

            if key in {"k", "up"}:
                panel = self._focused_panel()
                if panel == "panel-d" and key == "k":
                    pass
                if panel == "panel-a":
                    reduce_tui_state(self.shared, "select_run_delta", -1)
                elif panel == "panel-e":
                    reduce_tui_state(self.shared, "select_model_delta", -1)
                elif key == "up":
                    self._refresh_content()
                    return
                if panel != "panel-d":
                    self._refresh_content()
                    return

            if self._focused_panel() == "panel-c":
                if self._handle_launcher_keys(key):
                    self._refresh_content()
                    return
            if self._focused_panel() == "panel-d":
                if self._handle_generation_keys(key):
                    self._refresh_content()
                    return
            if self._focused_panel() == "panel-e":
                if self._handle_model_keys(key):
                    self._refresh_content()
                    return

        def _handle_confirmation_key(self, key: str) -> None:
            assert self.shared.pending_confirmation is not None
            action, value = self.shared.pending_confirmation
            if key in {"n", "escape"}:
                reduce_tui_state(self.shared, "set_last_action", "action canceled")
                reduce_tui_state(self.shared, "set_pending_confirmation", None)
                self._refresh_content()
                return
            if key != "y":
                return

            reduce_tui_state(self.shared, "set_pending_confirmation", None)
            if action == "start":
                ok, message = tui_start_training(self.training_options)
                reduce_tui_state(
                    self.shared, "set_last_action", message if ok else f"error: {message}"
                )
            elif action == "resume" and value:
                ok, message = tui_resume_training(
                    value,
                    self.training_options,
                    runs_root=runs_root_path,
                    checkpoints_root=checkpoints_root_path,
                )
                reduce_tui_state(
                    self.shared, "set_last_action", message if ok else f"error: {message}"
                )
            elif action == "archive" and value:
                if _model_is_running(value, runs_root=runs_root_path):
                    reduce_tui_state(
                        self.shared,
                        "set_last_action",
                        f"error: archive blocked; run_id={value} is active",
                    )
                    self._refresh_content()
                    return
                ok, message = archive_model_run(value, checkpoints_root=checkpoints_root_path)
                reduce_tui_state(
                    self.shared, "set_last_action", message if ok else f"error: {message}"
                )
                if self.shared.active_model_run_id == value and ok:
                    reduce_tui_state(self.shared, "set_active_model", None)
            elif action == "delete" and value:
                if _model_is_running(value, runs_root=runs_root_path):
                    reduce_tui_state(
                        self.shared,
                        "set_last_action",
                        f"error: delete blocked; run_id={value} is active",
                    )
                    self._refresh_content()
                    return
                ok, message = delete_model_run(value, checkpoints_root=checkpoints_root_path)
                reduce_tui_state(
                    self.shared, "set_last_action", message if ok else f"error: {message}"
                )
                if self.shared.active_model_run_id == value and ok:
                    reduce_tui_state(self.shared, "set_active_model", None)
            self._refresh_content()

        def _handle_launcher_keys(self, key: str) -> bool:
            if key in {"[", "-", "minus"}:
                self.training_options.epochs = _clamp_int(
                    self.training_options.epochs,
                    default=3,
                    min_value=MIN_EPOCHS,
                    max_value=MAX_EPOCHS,
                )
                self.training_options.epochs = max(MIN_EPOCHS, self.training_options.epochs - 1)
                reduce_tui_state(
                    self.shared, "set_last_action", f"epochs={self.training_options.epochs}"
                )
                return True
            elif key in {"]", "+", "plus", "equals"}:
                self.training_options.epochs = _clamp_int(
                    self.training_options.epochs,
                    default=3,
                    min_value=MIN_EPOCHS,
                    max_value=MAX_EPOCHS,
                )
                self.training_options.epochs = min(MAX_EPOCHS, self.training_options.epochs + 1)
                reduce_tui_state(
                    self.shared, "set_last_action", f"epochs={self.training_options.epochs}"
                )
                return True
            elif key == "b":
                self.training_options.batch_size = max(1, self.training_options.batch_size - 1)
            elif key == "B":
                self.training_options.batch_size += 1
            elif key == "v":
                self.training_options.strict_device = not self.training_options.strict_device
            elif key == "d":
                modes = ["auto", "cpu", "cuda", "A30"]
                current = (
                    modes.index(self.training_options.device)
                    if self.training_options.device in modes
                    else 0
                )
                self.training_options.device = modes[(current + 1) % len(modes)]
            elif key == "p":
                modes = ["off", "fp16", "bf16"]
                current = (
                    modes.index(self.training_options.precision)
                    if self.training_options.precision in modes
                    else 0
                )
                self.training_options.precision = modes[(current + 1) % len(modes)]
            elif key == "s":
                reduce_tui_state(self.shared, "set_pending_confirmation", ("start", None))
                reduce_tui_state(self.shared, "set_last_action", "confirm start training? y/n")
                return True
            elif key == "u":
                snapshot = self._snapshot()
                resume_model_run_id = snapshot.get("active_model_run_id")
                if not isinstance(resume_model_run_id, str):
                    legacy_run_id = self.shared.selected_run_id
                    if isinstance(legacy_run_id, str):
                        resume_model_run_id = legacy_run_id
                        reduce_tui_state(
                            self.shared,
                            "set_last_action",
                            "legacy fallback: resume selected run (no active model selected)",
                        )
                    else:
                        reduce_tui_state(
                            self.shared,
                            "set_last_action",
                            "error: no selected model to resume",
                        )
                        return True
                if not resume_model_run_id:
                    reduce_tui_state(
                        self.shared,
                        "set_last_action",
                        "error: no selected model to resume",
                    )
                    return True
                reduce_tui_state(
                    self.shared,
                    "set_pending_confirmation",
                    ("resume", resume_model_run_id),
                )
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    f"confirm resume model={resume_model_run_id}? y/n",
                )
                return True
            else:
                return False
            reduce_tui_state(self.shared, "set_last_action", f"launcher updated ({key})")
            return True

        def _handle_generation_keys(self, key: str) -> bool:
            if key == "enter":
                reduce_tui_state(
                    self.shared, "set_prompt_edit_mode", not self.shared.prompt_edit_mode
                )
                if self.shared.prompt_edit_mode:
                    reduce_tui_state(
                        self.shared,
                        "set_prompt_cursor",
                        len(self.generation_options.prompt),
                    )
                else:
                    reduce_tui_state(self.shared, "set_prompt_cursor", None)
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    "prompt edit mode on"
                    if self.shared.prompt_edit_mode
                    else "prompt edit mode off",
                )
                return True
            if key in {"escape"} and self.shared.prompt_edit_mode:
                reduce_tui_state(self.shared, "set_prompt_edit_mode", False)
                reduce_tui_state(self.shared, "set_prompt_cursor", None)
                reduce_tui_state(self.shared, "set_last_action", "prompt edit mode off")
                return True
            if key in {"backspace", "delete"} and self.shared.prompt_edit_mode:
                self._edit_prompt_delete(key)
                reduce_tui_state(self.shared, "set_last_action", "prompt edited")
                return True
            if key == "m":
                self.generation_options.max_new_tokens = max(
                    1, self.generation_options.max_new_tokens - 1
                )
            elif key == "M":
                self.generation_options.max_new_tokens += 1
            elif key == "t":
                self.generation_options.temperature = max(
                    0.1, round(self.generation_options.temperature - 0.1, 2)
                )
            elif key == "T":
                self.generation_options.temperature = round(
                    self.generation_options.temperature + 0.1, 2
                )
            elif key == "k":
                self.generation_options.top_k = max(1, self.generation_options.top_k - 1)
            elif key == "K":
                self.generation_options.top_k += 1
            elif key == "x":
                snapshot = self._snapshot()
                model_run = snapshot.get("active_model_run_id")
                if not isinstance(model_run, str):
                    reduce_tui_state(
                        self.shared,
                        "set_last_action",
                        "error: no selected model/checkpoint",
                    )
                    return True
                ok, message = tui_generate_from_run(model_run, self.generation_options)
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    message if ok else f"error: {message}",
                )
                if ok:
                    reduce_tui_state(
                        self.shared,
                        "set_generation_output",
                        message.split("\n", 1)[1] if "\n" in message else message,
                    )
                return True
            else:
                return False
            reduce_tui_state(self.shared, "set_last_action", f"generation control updated ({key})")
            return True

        def _edit_prompt_delete(self, key: str) -> None:
            prompt = self.generation_options.prompt
            cursor = min(max(self.shared.prompt_cursor_index or len(prompt), 0), len(prompt))
            if key == "backspace":
                if cursor <= 0:
                    return
                self.generation_options.prompt = prompt[: cursor - 1] + prompt[cursor:]
                reduce_tui_state(self.shared, "set_prompt_cursor", cursor - 1)
                return
            if cursor >= len(prompt):
                return
            self.generation_options.prompt = prompt[:cursor] + prompt[cursor + 1 :]
            reduce_tui_state(self.shared, "set_prompt_cursor", cursor)

        def _insert_prompt_text(self, text: str) -> None:
            prompt = self.generation_options.prompt
            cursor = min(max(self.shared.prompt_cursor_index or len(prompt), 0), len(prompt))
            self.generation_options.prompt = prompt[:cursor] + text + prompt[cursor:]
            reduce_tui_state(self.shared, "set_prompt_cursor", cursor + len(text))

        def _handle_prompt_edit_key(self, key: str, character: str | None) -> bool:
            if key in {"enter", "escape"}:
                reduce_tui_state(self.shared, "set_prompt_edit_mode", False)
                reduce_tui_state(self.shared, "set_prompt_cursor", None)
                reduce_tui_state(self.shared, "set_last_action", "prompt edit mode off")
                return True
            if key in {"backspace", "delete"}:
                self._edit_prompt_delete(key)
                reduce_tui_state(self.shared, "set_last_action", "prompt edited")
                return True
            if key == "left":
                prompt_len = len(self.generation_options.prompt)
                cursor = min(max(self.shared.prompt_cursor_index or prompt_len, 0), prompt_len)
                reduce_tui_state(self.shared, "set_prompt_cursor", max(0, cursor - 1))
                return True
            if key == "right":
                prompt_len = len(self.generation_options.prompt)
                cursor = min(max(self.shared.prompt_cursor_index or prompt_len, 0), prompt_len)
                reduce_tui_state(self.shared, "set_prompt_cursor", min(prompt_len, cursor + 1))
                return True
            if key == "home":
                reduce_tui_state(self.shared, "set_prompt_cursor", 0)
                return True
            if key == "end":
                reduce_tui_state(
                    self.shared,
                    "set_prompt_cursor",
                    len(self.generation_options.prompt),
                )
                return True
            if key == "space":
                self._insert_prompt_text(" ")
                reduce_tui_state(self.shared, "set_last_action", "prompt edited")
                return True
            if (
                isinstance(character, str)
                and len(character) == 1
                and character not in {"\n", "\r", "\t", " "}
                and character.isprintable()
            ):
                self._insert_prompt_text(character)
                reduce_tui_state(self.shared, "set_last_action", "prompt edited")
                return True
            if len(key) == 1:
                self._insert_prompt_text(key)
                reduce_tui_state(self.shared, "set_last_action", "prompt edited")
                return True
            return True

        def _handle_model_keys(self, key: str) -> bool:
            snapshot = self._snapshot()
            selected_model_run_id = snapshot.get("selected_model_run_id")
            if not isinstance(selected_model_run_id, str):
                reduce_tui_state(self.shared, "set_last_action", "error: no model selected")
                return key in {"a", "e", "i", "A", "D"}

            if key == "a":
                reduce_tui_state(self.shared, "set_active_model", selected_model_run_id)
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    f"active model set run_id={selected_model_run_id}",
                )
                return True
            if key == "e":
                display_name = next(
                    (
                        entry.display_name
                        for entry in collect_model_entries(
                            checkpoints_root=checkpoints_root_path,
                            runs_root=runs_root_path,
                        )
                        if entry.run_id == selected_model_run_id
                    ),
                    selected_model_run_id,
                )
                reduce_tui_state(self.shared, "set_rename_buffer", display_name)
                reduce_tui_state(self.shared, "set_rename_edit_mode", True)
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    f"rename edit mode on for run_id={selected_model_run_id}",
                )
                return True
            if key == "i":
                checkpoint = Path("checkpoints") / selected_model_run_id / "latest.pt"
                exists = checkpoint.exists()
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    f"model run_id={selected_model_run_id} checkpoint={checkpoint} exists={exists}",
                )
                return True
            if key == "A":
                reduce_tui_state(
                    self.shared,
                    "set_pending_confirmation",
                    ("archive", selected_model_run_id),
                )
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    f"confirm archive run_id={selected_model_run_id}? y/n",
                )
                return True
            if key == "D":
                reduce_tui_state(
                    self.shared,
                    "set_pending_confirmation",
                    ("delete", selected_model_run_id),
                )
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    f"confirm delete run_id={selected_model_run_id}? y/n",
                )
                return True
            if key == "r":
                reduce_tui_state(self.shared, "set_last_action", "model list refreshed")
                return True
            return False

        def _edit_rename_delete(self, key: str) -> None:
            value = self.shared.rename_buffer
            if key == "backspace":
                if not value:
                    return
                reduce_tui_state(self.shared, "set_rename_buffer", value[:-1])
                return
            if key == "delete":
                reduce_tui_state(self.shared, "set_rename_buffer", "")

        def _handle_rename_edit_key(self, key: str) -> bool:
            snapshot = self._snapshot()
            selected_model_run_id = snapshot.get("selected_model_run_id")
            if not isinstance(selected_model_run_id, str):
                reduce_tui_state(self.shared, "set_rename_edit_mode", False)
                reduce_tui_state(self.shared, "set_rename_buffer", "")
                reduce_tui_state(self.shared, "set_last_action", "error: no model selected")
                return True

            if key == "escape":
                reduce_tui_state(self.shared, "set_rename_edit_mode", False)
                reduce_tui_state(self.shared, "set_rename_buffer", "")
                reduce_tui_state(self.shared, "set_last_action", "rename canceled")
                return True
            if key == "enter":
                ok, message = rename_model_run(
                    selected_model_run_id,
                    self.shared.rename_buffer,
                    checkpoints_root=checkpoints_root_path,
                )
                if ok:
                    reduce_tui_state(self.shared, "set_rename_edit_mode", False)
                    reduce_tui_state(self.shared, "set_rename_buffer", "")
                reduce_tui_state(
                    self.shared,
                    "set_last_action",
                    message if ok else f"error: {message}",
                )
                return True
            if key in {"backspace", "delete"}:
                self._edit_rename_delete(key)
                reduce_tui_state(self.shared, "set_last_action", "rename edited")
                return True
            if key == "space":
                reduce_tui_state(self.shared, "set_rename_buffer", f"{self.shared.rename_buffer} ")
                reduce_tui_state(self.shared, "set_last_action", "rename edited")
                return True
            if len(key) == 1:
                reduce_tui_state(
                    self.shared,
                    "set_rename_buffer",
                    f"{self.shared.rename_buffer}{key}",
                )
                reduce_tui_state(self.shared, "set_last_action", "rename edited")
                return True
            return True

        def _snapshot(self) -> dict[str, object]:
            return build_tui_snapshot(
                runs_root=runs_root_path,
                checkpoints_root=checkpoints_root_path,
                run_id=run_id,
                selected_index=self.shared.selected_index,
                selected_run_id=self.shared.selected_run_id,
                selected_model_index=self.shared.selected_model_index,
                active_model_run_id=self.shared.active_model_run_id,
                training_options=self.training_options,
                generation_options=self.generation_options,
                generation_output=self.shared.generation_output,
                prompt_edit_mode=self.shared.prompt_edit_mode,
                prompt_cursor_index=self.shared.prompt_cursor_index,
                run_scroll_offset=self.shared.run_scroll_offset,
                generation_scroll_offset=self.shared.generation_scroll_offset,
                pending_confirmation=(
                    self.shared.pending_confirmation[0]
                    if self.shared.pending_confirmation
                    else None
                ),
                last_action=self.shared.last_action,
                rename_edit_mode=self.shared.rename_edit_mode,
                rename_buffer=self.shared.rename_buffer,
            )

        def _refresh_content(self) -> None:
            snapshot = self._snapshot()
            self.shared.selected_index = int(snapshot.get("selected", 0))
            selected_run_id = snapshot.get("selected_run_id")
            self.shared.selected_run_id = (
                selected_run_id if isinstance(selected_run_id, str) else None
            )
            self.shared.selected_model_index = int(snapshot.get("selected_model_index", 0))
            self.shared.run_scroll_offset = int(snapshot.get("run_scroll_offset", 0))
            self.shared.generation_scroll_offset = int(snapshot.get("generation_scroll_offset", 0))
            active_model = snapshot.get("active_model_run_id")
            self.shared.active_model_run_id = (
                active_model if isinstance(active_model, str) else None
            )

            self.query_one("#panel-a", Static).update(_join_markup_safe(snapshot["runs"]))
            self.query_one("#panel-b", Static).update(_join_markup_safe(snapshot["utilization"]))
            self.query_one("#panel-c", Static).update(_join_markup_safe(snapshot["launcher"]))
            self.query_one("#panel-d", Static).update(_join_markup_safe(snapshot["generation"]))

            panel_e_lines = list(snapshot["models"])
            panel_e_lines.append("")
            panel_e_lines.extend(snapshot["status"])
            if self.shared.prompt_edit_mode:
                panel_e_lines.append("prompt editor active")
            if self.shared.rename_edit_mode:
                panel_e_lines.append("rename editor active")
            if self.shared.pending_confirmation:
                panel_e_lines.append(
                    f"awaiting confirm: {self.shared.pending_confirmation[0]} (y/n)"
                )
            self.query_one("#panel-e", Static).update(_join_markup_safe(panel_e_lines))
            self.query_one("#key-footer", Static).update(
                _markup_safe(
                    _footer_hint_line(
                        focused_panel=self._focused_panel(),
                        prompt_edit_mode=self.shared.prompt_edit_mode,
                        rename_edit_mode=self.shared.rename_edit_mode,
                        pending_confirmation=(
                            self.shared.pending_confirmation[0]
                            if self.shared.pending_confirmation
                            else None
                        ),
                    )
                )
            )
            self._apply_focus_titles()

    app = MonitorApp()
    if return_app:
        return app
    app.run()
    return 0
