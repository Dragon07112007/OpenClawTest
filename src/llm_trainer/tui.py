from __future__ import annotations

import json
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
from .telemetry import collect_gpu_telemetry


def _markup_safe(value: object) -> str:
    text = str(value)
    try:
        from rich.markup import escape
    except ModuleNotFoundError:
        return text.replace("[", "\\[")
    return escape(text)


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


def _latest_runs(runs_root: Path, limit: int = 60) -> list[Path]:
    if not runs_root.exists():
        return []
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


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
    checkpoint_dir: Path
    checkpoint_path: Path
    mtime: float
    has_run_meta: bool


def _validate_training_options(options: TuiTrainingOptions) -> str | None:
    if options.epochs < 1:
        return "epochs must be >= 1"
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


def tui_resume_training(run_id: str, options: TuiTrainingOptions) -> tuple[bool, str]:
    error = _validate_training_options(options)
    if error:
        return (False, f"resume failed: {error}")

    run_dir = Path("runs") / run_id
    checkpoint_path = str(Path("checkpoints") / run_id / "latest.pt")
    try:
        mode, _ = resume_training_run(
            run_dir=run_dir,
            config_path=options.config,
            checkpoint_path=checkpoint_path,
            more_epochs=options.epochs,
            foreground=False,
            requested_device=options.device,
            strict_device=options.strict_device,
            precision=options.precision,
            grad_accum_steps=options.grad_accum_steps,
        )
    except (DeviceResolutionError, FileNotFoundError, ValueError) as exc:
        return (False, f"resume failed: {exc}")
    return (True, f"resumed run_id={run_id} checkpoint={checkpoint_path} mode={mode}")


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


def collect_system_utilization(
    *,
    selected_run_state: dict[str, Any] | None,
    selected_device: str,
) -> dict[str, Any]:
    gpu_data = selected_run_state or {}
    if not isinstance(gpu_data.get("gpu_utilization_pct"), (float, int)):
        gpu_data = collect_gpu_telemetry(selected_device)

    cpu_pct = None
    ram_used_mb = None
    ram_total_mb = None
    try:
        import psutil  # type: ignore[import-not-found]

        cpu_pct = float(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()
        ram_used_mb = round(float(mem.used) / (1024 * 1024), 1)
        ram_total_mb = round(float(mem.total) / (1024 * 1024), 1)
    except Exception:
        pass

    return {
        "gpu_utilization_pct": gpu_data.get("gpu_utilization_pct"),
        "gpu_memory_used_mb": gpu_data.get("gpu_memory_used_mb"),
        "gpu_memory_total_mb": gpu_data.get("gpu_memory_total_mb"),
        "gpu_temperature_c": gpu_data.get("gpu_temperature_c"),
        "gpu_power_w": gpu_data.get("gpu_power_w"),
        "cpu_utilization_pct": cpu_pct,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
    }


def _launch_help_text(
    training_options: TuiTrainingOptions,
    generation_options: TuiGenerationOptions,
) -> str:
    return (
        "Focus: tab/shift+tab | nav: j/k | refresh: r | launch: s start / u resume | "
        "generate: x | edit prompt: enter, esc | model: a activate, i inspect, "
        "A archive, D delete | "
        f"epochs +/- via [/] ({training_options.epochs}) | "
        f"prompt len={len(generation_options.prompt)}"
    )


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


def _runs_panel_lines(entries: list[RunEntry], selected: int) -> list[str]:
    if not entries:
        return [
            "Runs",
            "No runs found.",
            "Start with: llm-trainer train --config configs/default.toml",
        ]

    lines = [
        "Runs (running first, then newest)",
        "ID | status | epoch | step | loss | device | eta/remaining | gpu",
    ]
    for idx, entry in enumerate(entries):
        state = entry.state
        marker = ">" if idx == selected else " "
        lines.append(
            f"{marker} {entry.run_id} | {state.get('status', 'unknown')} | "
            f"{state.get('epoch', 'n/a')} | {state.get('global_step', 'n/a')} | "
            f"{_fmt_loss(state.get('train_loss'))} | "
            f"{entry.meta.get('selected_device', entry.meta.get('device', 'n/a'))} | "
            f"{_fmt_eta(state.get('eta_at'))}/{_fmt_duration(state.get('remaining_seconds'))} | "
            f"{_fmt_gpu(state)}"
        )
    return lines


def _launcher_panel_lines(
    training_options: TuiTrainingOptions,
    selected_run_id: str | None,
    pending_confirmation: str | None,
) -> list[str]:
    return [
        "Training Launcher",
        f"config={training_options.config}",
        "options: "
        f"epochs={training_options.epochs} batch={training_options.batch_size} "
        f"seq={training_options.seq_length}",
        "options: "
        f"device={training_options.device} strict={training_options.strict_device} "
        f"precision={training_options.precision} grad_accum={training_options.grad_accum_steps}",
        f"target run for resume={selected_run_id or 'n/a'}",
        "keys: s start | u resume selected | [/] epochs | b/B batch | l/L seq | "
        "d device | p precision | g/G grad_accum | v strict",
        f"confirmation={pending_confirmation or 'none'}",
    ]


def _generation_panel_lines(
    generation_options: TuiGenerationOptions,
    active_model_run_id: str | None,
    generation_output: str,
    prompt_edit_mode: bool,
) -> list[str]:
    truncated_output = (
        generation_output if len(generation_output) <= 500 else generation_output[:500] + "..."
    )
    return [
        "Generation Workspace",
        f"active model run={active_model_run_id or 'none selected'}",
        "controls: "
        f"max_new_tokens={generation_options.max_new_tokens} "
        f"temperature={generation_options.temperature:.2f} top_k={generation_options.top_k}",
        f"prompt edit mode={'on' if prompt_edit_mode else 'off'}",
        f"prompt: {generation_options.prompt}",
        "keys: enter edit prompt | x generate | m/M max_new_tokens | t/T temperature | "
        "k/K top_k",
        "output:",
        truncated_output or "(no generation output yet)",
    ]


def _utilization_panel_lines(system: dict[str, Any], selected_device: str) -> list[str]:
    return [
        "System Utilization",
        f"device={selected_device}",
        "GPU: "
        f"util={_fmt_float(system.get('gpu_utilization_pct'), '%')} "
        f"vram={_fmt_float(system.get('gpu_memory_used_mb'))}/"
        f"{_fmt_float(system.get('gpu_memory_total_mb'))}MB",
        "GPU: "
        f"temp={_fmt_float(system.get('gpu_temperature_c'), 'C')} "
        f"power={_fmt_float(system.get('gpu_power_w'), 'W')}",
        f"CPU: {_fmt_float(system.get('cpu_utilization_pct'), '%')}",
        "RAM: "
        f"{_fmt_float(system.get('ram_used_mb'))}/{_fmt_float(system.get('ram_total_mb'))}MB",
    ]


def _models_panel_lines(
    models: list[ModelEntry],
    *,
    selected_model_index: int,
    active_model_run_id: str | None,
) -> tuple[list[str], int, str | None]:
    if not models:
        return (
            [
                "Model Manager",
                "No checkpoints found.",
                "Trained runs with checkpoints will appear here.",
            ],
            0,
            None,
        )

    selected = min(max(selected_model_index, 0), len(models) - 1)
    latest_run_id = models[0].run_id
    lines = ["Model Manager", "run_id | checkpoint | flags"]
    for idx, model in enumerate(models):
        marker = ">" if idx == selected else " "
        flags: list[str] = []
        if model.run_id == latest_run_id:
            flags.append("latest")
        if model.run_id == active_model_run_id:
            flags.append("active")
        if not model.has_run_meta:
            flags.append("orphan")
        lines.append(
            f"{marker} {model.run_id} | {model.checkpoint_path.name} | "
            f"{','.join(flags) if flags else '-'}"
        )
    lines.append("keys: a activate | i inspect | A archive(confirm) | D delete(confirm)")
    return lines, selected, models[selected].run_id


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
    pending_confirmation: str | None = None,
    last_action: str | None = None,
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
    models_lines, resolved_model_index, selected_model_run_id = _models_panel_lines(
        models,
        selected_model_index=selected_model_index,
        active_model_run_id=active_model_run_id,
    )

    effective_active_model = active_model_run_id
    if effective_active_model is None and selected_model_run_id:
        effective_active_model = selected_model_run_id

    selected_device = "cpu"
    if selected_entry:
        selected_device = str(
            selected_entry.meta.get("selected_device", selected_entry.meta.get("device", "cpu"))
        )
    system = collect_system_utilization(
        selected_run_state=(selected_entry.state if selected_entry else None),
        selected_device=selected_device,
    )

    status_lines = [
        "Dashboard Status",
        f"selected run={selected_run_value or 'none'}",
        f"selected model={selected_model_run_id or 'none'}",
        f"active model={effective_active_model or 'none'}",
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
        "runs": _runs_panel_lines(entries, selected),
        "detail": detail,
        "launcher": _launcher_panel_lines(
            training_options,
            selected_run_id=selected_run_value,
            pending_confirmation=pending_confirmation,
        ),
        "generation": _generation_panel_lines(
            generation_options,
            active_model_run_id=effective_active_model,
            generation_output=generation_output,
            prompt_edit_mode=prompt_edit_mode,
        ),
        "utilization": _utilization_panel_lines(system, selected_device=selected_device),
        "models": models_lines,
        "status": status_lines,
        "error": bool(load_errors),
        "selected": selected,
        "selected_run_id": selected_run_value,
        "selected_model_index": resolved_model_index,
        "selected_model_run_id": selected_model_run_id,
        "active_model_run_id": effective_active_model,
    }


def launch_tui(*, runs_root: str | Path = "runs", run_id: str | None = None) -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, Static
    except ModuleNotFoundError:
        print("tui failed (missing dependency: textual)")
        return 1

    runs_root_path = Path(runs_root)
    checkpoints_root_path = Path("checkpoints")

    class MonitorApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #dashboard {
            height: 1fr;
            padding: 1;
        }
        .column {
            height: 1fr;
            padding-right: 1;
        }
        #left-col {
            width: 29%;
        }
        #center-col {
            width: 24%;
        }
        #right-col {
            width: 25%;
        }
        #far-right-col {
            width: 22%;
            padding-right: 0;
        }
        .panel {
            border: round #6689a1;
            padding: 0 1;
            margin-bottom: 1;
        }
        #status {
            border: heavy #4f8f72;
            height: 15;
        }
        #generation {
            border: round #99865c;
            height: 1fr;
            margin-bottom: 0;
        }
        #launcher {
            border: round #7a77a0;
            height: 1fr;
            margin-bottom: 0;
        }
        #runs {
            border: round #5d7fa7;
            height: 1fr;
        }
        #utilization {
            border: round #a17f66;
            height: 11;
            margin-bottom: 0;
        }
        #models {
            border: round #608a91;
            height: 1fr;
            margin-bottom: 0;
        }
        """

        selected_index = 0
        selected_run_id: str | None = None
        selected_model_index = 0
        active_model_run_id: str | None = None
        focused_panel_index = 0
        panel_order = ["runs", "launcher", "generation", "models"]

        def __init__(self) -> None:
            super().__init__()
            self.training_options = TuiTrainingOptions()
            self.generation_options = TuiGenerationOptions()
            self.last_action = "none"
            self.generation_output = ""
            self.prompt_edit_mode = False
            self.pending_confirmation: tuple[str, str | None] | None = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="dashboard"):
                with Vertical(id="left-col", classes="column"):
                    yield Static("", id="status", classes="panel")
                    yield Static("", id="generation", classes="panel")
                with Vertical(id="center-col", classes="column"):
                    yield Static("", id="launcher", classes="panel")
                with Vertical(id="right-col", classes="column"):
                    yield Static("", id="runs", classes="panel")
                    yield Static("", id="utilization", classes="panel")
                with Vertical(id="far-right-col", classes="column"):
                    yield Static("", id="models", classes="panel")
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(1.0, self._refresh_content)
            self._refresh_content()

        def _focused_panel(self) -> str:
            return self.panel_order[self.focused_panel_index]

        def _cycle_focus(self, reverse: bool = False) -> None:
            delta = -1 if reverse else 1
            self.focused_panel_index = (self.focused_panel_index + delta) % len(self.panel_order)
            self.last_action = f"focus={self._focused_panel()}"

        def on_key(self, event) -> None:
            key = event.key

            if self.prompt_edit_mode and key not in {"escape", "enter", "backspace", "delete"}:
                if len(key) == 1:
                    self.generation_options.prompt += key
                    self.last_action = "prompt edited"
                    self._refresh_content()
                    return

            if self.pending_confirmation:
                self._handle_confirmation_key(key)
                return

            if key == "tab":
                self._cycle_focus(reverse=False)
                self._refresh_content()
                return
            if key == "shift+tab":
                self._cycle_focus(reverse=True)
                self._refresh_content()
                return
            if key == "r":
                self._refresh_content()
                return

            if key in {"j", "down"}:
                panel = self._focused_panel()
                if panel == "runs":
                    self.selected_index += 1
                    self.selected_run_id = None
                elif panel == "models":
                    self.selected_model_index += 1
                self._refresh_content()
                return

            if key in {"k", "up"}:
                panel = self._focused_panel()
                if panel == "runs":
                    self.selected_index -= 1
                    self.selected_run_id = None
                elif panel == "models":
                    self.selected_model_index -= 1
                self._refresh_content()
                return

            if self._focused_panel() == "launcher":
                if self._handle_launcher_keys(key):
                    self._refresh_content()
                    return
            if self._focused_panel() == "generation":
                if self._handle_generation_keys(key):
                    self._refresh_content()
                    return
            if self._focused_panel() == "models":
                if self._handle_model_keys(key):
                    self._refresh_content()
                    return

            snapshot = self._snapshot()
            selected_run_id = snapshot.get("selected_run_id")
            if key == "u" and isinstance(selected_run_id, str):
                self.pending_confirmation = ("resume", selected_run_id)
                self.last_action = f"confirm resume run_id={selected_run_id}? y/n"
                self._refresh_content()

        def _handle_confirmation_key(self, key: str) -> None:
            assert self.pending_confirmation is not None
            action, value = self.pending_confirmation
            if key in {"n", "escape"}:
                self.last_action = "action canceled"
                self.pending_confirmation = None
                self._refresh_content()
                return
            if key != "y":
                return

            self.pending_confirmation = None
            if action == "start":
                ok, message = tui_start_training(self.training_options)
                self.last_action = message if ok else f"error: {message}"
            elif action == "resume" and value:
                ok, message = tui_resume_training(value, self.training_options)
                self.last_action = message if ok else f"error: {message}"
            elif action == "archive" and value:
                ok, message = archive_model_run(value, checkpoints_root=checkpoints_root_path)
                self.last_action = message if ok else f"error: {message}"
                if self.active_model_run_id == value and ok:
                    self.active_model_run_id = None
            elif action == "delete" and value:
                ok, message = delete_model_run(value, checkpoints_root=checkpoints_root_path)
                self.last_action = message if ok else f"error: {message}"
                if self.active_model_run_id == value and ok:
                    self.active_model_run_id = None
            self._refresh_content()

        def _handle_launcher_keys(self, key: str) -> bool:
            if key == "[":
                self.training_options.epochs = max(1, self.training_options.epochs - 1)
            elif key == "]":
                self.training_options.epochs += 1
            elif key == "b":
                self.training_options.batch_size = max(1, self.training_options.batch_size - 1)
            elif key == "B":
                self.training_options.batch_size += 1
            elif key == "l":
                self.training_options.seq_length = max(1, self.training_options.seq_length - 1)
            elif key == "L":
                self.training_options.seq_length += 1
            elif key == "g":
                self.training_options.grad_accum_steps = max(
                    1, self.training_options.grad_accum_steps - 1
                )
            elif key == "G":
                self.training_options.grad_accum_steps += 1
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
                self.pending_confirmation = ("start", None)
                self.last_action = "confirm start training? y/n"
                return True
            elif key == "u":
                snapshot = self._snapshot()
                selected_run_id = snapshot.get("selected_run_id")
                if isinstance(selected_run_id, str):
                    self.pending_confirmation = ("resume", selected_run_id)
                    self.last_action = f"confirm resume run_id={selected_run_id}? y/n"
                else:
                    self.last_action = "error: no selected run for resume"
                return True
            else:
                return False
            self.last_action = f"launcher updated ({key})"
            return True

        def _handle_generation_keys(self, key: str) -> bool:
            if key == "enter":
                self.prompt_edit_mode = not self.prompt_edit_mode
                self.last_action = (
                    "prompt edit mode on" if self.prompt_edit_mode else "prompt edit mode off"
                )
                return True
            if key in {"escape"} and self.prompt_edit_mode:
                self.prompt_edit_mode = False
                self.last_action = "prompt edit mode off"
                return True
            if key in {"backspace", "delete"} and self.prompt_edit_mode:
                self.generation_options.prompt = self.generation_options.prompt[:-1]
                self.last_action = "prompt edited"
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
                model_run = snapshot.get("active_model_run_id") or snapshot.get("selected_run_id")
                if not isinstance(model_run, str):
                    self.last_action = "error: no selected model/checkpoint"
                    return True
                ok, message = tui_generate_from_run(model_run, self.generation_options)
                self.last_action = message if ok else f"error: {message}"
                if ok:
                    self.generation_output = (
                        message.split("\n", 1)[1] if "\n" in message else message
                    )
                return True
            else:
                return False
            self.last_action = f"generation control updated ({key})"
            return True

        def _handle_model_keys(self, key: str) -> bool:
            snapshot = self._snapshot()
            selected_model_run_id = snapshot.get("selected_model_run_id")
            if not isinstance(selected_model_run_id, str):
                self.last_action = "error: no model selected"
                return key in {"a", "i", "A", "D"}

            if key == "a":
                self.active_model_run_id = selected_model_run_id
                self.last_action = f"active model set run_id={selected_model_run_id}"
                return True
            if key == "i":
                checkpoint = Path("checkpoints") / selected_model_run_id / "latest.pt"
                exists = checkpoint.exists()
                self.last_action = (
                    f"model run_id={selected_model_run_id} checkpoint={checkpoint} exists={exists}"
                )
                return True
            if key == "A":
                self.pending_confirmation = ("archive", selected_model_run_id)
                self.last_action = f"confirm archive run_id={selected_model_run_id}? y/n"
                return True
            if key == "D":
                self.pending_confirmation = ("delete", selected_model_run_id)
                self.last_action = f"confirm delete run_id={selected_model_run_id}? y/n"
                return True
            return False

        def _snapshot(self) -> dict[str, object]:
            return build_tui_snapshot(
                runs_root=runs_root_path,
                checkpoints_root=checkpoints_root_path,
                run_id=run_id,
                selected_index=self.selected_index,
                selected_run_id=self.selected_run_id,
                selected_model_index=self.selected_model_index,
                active_model_run_id=self.active_model_run_id,
                training_options=self.training_options,
                generation_options=self.generation_options,
                generation_output=self.generation_output,
                prompt_edit_mode=self.prompt_edit_mode,
                pending_confirmation=(
                    self.pending_confirmation[0] if self.pending_confirmation else None
                ),
                last_action=self.last_action,
            )

        def _refresh_content(self) -> None:
            snapshot = self._snapshot()
            self.selected_index = int(snapshot.get("selected", 0))
            selected_run_id = snapshot.get("selected_run_id")
            self.selected_run_id = selected_run_id if isinstance(selected_run_id, str) else None
            self.selected_model_index = int(snapshot.get("selected_model_index", 0))
            active_model = snapshot.get("active_model_run_id")
            self.active_model_run_id = active_model if isinstance(active_model, str) else None

            self.query_one("#runs", Static).update(_join_markup_safe(snapshot["runs"]))
            self.query_one("#launcher", Static).update(_join_markup_safe(snapshot["launcher"]))
            self.query_one("#generation", Static).update(_join_markup_safe(snapshot["generation"]))
            self.query_one("#utilization", Static).update(
                _join_markup_safe(snapshot["utilization"])
            )
            self.query_one("#models", Static).update(_join_markup_safe(snapshot["models"]))

            status_lines = list(snapshot["status"])
            status_lines.append(f"focus={self._focused_panel()}")
            if self.prompt_edit_mode:
                status_lines.append("prompt editor active")
            if self.pending_confirmation:
                status_lines.append(f"awaiting confirm: {self.pending_confirmation[0]} (y/n)")
            self.query_one("#status", Static).update(_join_markup_safe(status_lines))

    MonitorApp().run()
    return 0
