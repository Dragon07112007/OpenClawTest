from __future__ import annotations

import json
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


def _latest_runs(runs_root: Path, limit: int = 30) -> list[Path]:
    if not runs_root.exists():
        return []
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


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


def _fmt_gpu(state: dict[str, Any]) -> str:
    util = state.get("gpu_utilization_pct")
    used = state.get("gpu_memory_used_mb")
    total = state.get("gpu_memory_total_mb")
    if not isinstance(util, (float, int)) or not isinstance(used, (float, int)):
        return "n/a"
    total_text = "n/a" if not isinstance(total, (float, int)) else str(total)
    return f"{util}% {used}/{total_text}MB"


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


def tui_start_training(options: TuiTrainingOptions) -> tuple[bool, str]:
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


def build_tui_snapshot(
    *,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    selected_index: int = 0,
    training_options: TuiTrainingOptions | None = None,
    generation_options: TuiGenerationOptions | None = None,
    last_action: str | None = None,
) -> dict[str, object]:
    runs_root_path = Path(runs_root)
    training_options = training_options or TuiTrainingOptions()
    generation_options = generation_options or TuiGenerationOptions()

    if run_id:
        run_dir = runs_root_path / run_id
        if not run_dir.exists():
            return {
                "runs": [f"Requested run: {run_id}"],
                "detail": [f"Run not found: {run_id}", "Check run ID or runs/ directory."],
                "error": True,
                "selected": 0,
                "selected_run_id": None,
            }

        try:
            meta = _read_json(run_dir / "meta.json")
            state = _read_json(run_dir / "state.json")
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "runs": [f"Requested run: {run_id}"],
                "detail": [f"Unable to load run files: {exc}"],
                "error": True,
                "selected": 0,
                "selected_run_id": None,
            }

        return {
            "runs": [f"* {meta.get('run_id')} ({state.get('status', 'unknown')})"],
            "detail": _run_detail_lines(
                meta,
                state,
                training_options=training_options,
                generation_options=generation_options,
                last_action=last_action,
            ),
            "error": False,
            "selected": 0,
            "selected_run_id": str(meta.get("run_id")),
        }

    runs = _latest_runs(runs_root_path)
    if not runs:
        return {
            "runs": [
                "No runs found.",
                "Start one with: llm-trainer train --config configs/default.toml",
            ],
            "detail": [
                "Empty state",
                "The monitor will auto-refresh every second.",
                _launch_help_text(training_options, generation_options),
            ],
            "error": False,
            "selected": 0,
            "selected_run_id": None,
        }

    entries: list[tuple[Path, dict, dict]] = []
    load_errors: list[str] = []
    for run_dir in runs:
        try:
            entries.append(
                (
                    run_dir,
                    _read_json(run_dir / "meta.json"),
                    _read_json(run_dir / "state.json"),
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            load_errors.append(f"{run_dir.name}: {exc}")

    if not entries:
        return {
            "runs": ["No readable runs available."],
            "detail": ["Failed to parse run metadata/state.", *load_errors[:4]],
            "error": True,
            "selected": 0,
            "selected_run_id": None,
        }

    selected = min(max(selected_index, 0), len(entries) - 1)
    rows = ["Runs (newest first)", "ID | status | epoch | step | loss | device | eta | gpu"]
    for idx, (_run_dir, meta, state) in enumerate(entries):
        marker = ">" if idx == selected else " "
        rows.append(
            f"{marker} {meta.get('run_id')} | {state.get('status', 'unknown')} | "
            f"{state.get('epoch', 'n/a')} | {state.get('global_step', 'n/a')} | "
            f"{_fmt_loss(state.get('train_loss'))} | "
            f"{meta.get('selected_device', meta.get('device', 'n/a'))} | "
            f"{_fmt_eta(state.get('eta_at'))} | {_fmt_gpu(state)}"
        )

    detail = _run_detail_lines(
        entries[selected][1],
        entries[selected][2],
        training_options=training_options,
        generation_options=generation_options,
        last_action=last_action,
    )
    if load_errors:
        detail.append("")
        detail.append("Load warnings:")
        detail.extend(load_errors[:3])

    return {
        "runs": rows,
        "detail": detail,
        "error": bool(load_errors),
        "selected": selected,
        "selected_run_id": str(entries[selected][1].get("run_id")),
    }


def _launch_help_text(
    training_options: TuiTrainingOptions,
    generation_options: TuiGenerationOptions,
) -> str:
    return (
        "TUI actions: n start | u resume selected | g generate selected | [/] epochs | "
        f"d cycle device ({training_options.device}) | "
        f"p cycle prompt ({generation_options.prompt!r})"
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
        f"GPU util: {state.get('gpu_utilization_pct', 'n/a')}",
        "GPU memory: "
        f"{state.get('gpu_memory_used_mb', 'n/a')}/"
        f"{state.get('gpu_memory_total_mb', 'n/a')}MB",
        f"GPU temp: {state.get('gpu_temperature_c', 'n/a')}C",
        f"GPU power: {state.get('gpu_power_w', 'n/a')}W",
        f"PID: {state.get('pid', 'n/a')}",
        f"Updated: {state.get('updated_at', 'n/a')}",
        "",
        "Launch options: "
        f"cfg={training_options.config} "
        f"epochs={training_options.epochs} "
        f"batch={training_options.batch_size} "
        f"seq={training_options.seq_length}",
        "Launch options: "
        f"device={training_options.device} "
        f"strict={training_options.strict_device} "
        f"precision={training_options.precision} "
        f"grad_accum={training_options.grad_accum_steps}",
        "Generate options: "
        f"prompt={generation_options.prompt!r} "
        f"max_new_tokens={generation_options.max_new_tokens} "
        f"temperature={generation_options.temperature} "
        f"top_k={generation_options.top_k}",
        _launch_help_text(training_options, generation_options),
        f"Last action: {last_action or 'none'}",
    ]


def launch_tui(*, runs_root: str | Path = "runs", run_id: str | None = None) -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal
        from textual.widgets import Footer, Header, Static
    except ModuleNotFoundError:
        print("tui failed (missing dependency: textual)")
        return 1

    runs_root_path = Path(runs_root)

    class MonitorApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #body {
            height: 1fr;
            padding: 1;
        }
        #runs {
            width: 56%;
            border: round #6689a1;
            padding: 0 1;
        }
        #detail {
            width: 44%;
            border: round #a17f66;
            padding: 0 1;
        }
        #help {
            dock: bottom;
            height: auto;
            border: heavy #698070;
            padding: 0 1;
        }
        """

        selected_index = 0

        def __init__(self) -> None:
            super().__init__()
            self.training_options = TuiTrainingOptions()
            self.generation_options = TuiGenerationOptions()
            self.last_action = "none"

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="body"):
                yield Static("", id="runs")
                yield Static("", id="detail")
            yield Static("", id="help")
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(1.0, self._refresh_content)
            self._refresh_content()

        def on_key(self, event) -> None:
            key = event.key
            if key in {"j", "down"}:
                self.selected_index += 1
                self._refresh_content()
                return
            if key in {"k", "up"}:
                self.selected_index -= 1
                self._refresh_content()
                return
            if key == "r":
                self._refresh_content()
                return
            if key == "[":
                self.training_options.epochs = max(1, self.training_options.epochs - 1)
                self.last_action = f"epochs={self.training_options.epochs}"
                self._refresh_content()
                return
            if key == "]":
                self.training_options.epochs += 1
                self.last_action = f"epochs={self.training_options.epochs}"
                self._refresh_content()
                return
            if key == "d":
                modes = ["auto", "cpu", "cuda", "A30"]
                current = (
                    modes.index(self.training_options.device)
                    if self.training_options.device in modes
                    else 0
                )
                self.training_options.device = modes[(current + 1) % len(modes)]
                self.last_action = f"device={self.training_options.device}"
                self._refresh_content()
                return
            if key == "p":
                prompts = ["Hello", "Once upon a time", "In a distant galaxy"]
                current = (
                    prompts.index(self.generation_options.prompt)
                    if self.generation_options.prompt in prompts
                    else 0
                )
                self.generation_options.prompt = prompts[(current + 1) % len(prompts)]
                self.last_action = f"prompt={self.generation_options.prompt!r}"
                self._refresh_content()
                return

            snapshot = build_tui_snapshot(
                runs_root=runs_root_path,
                run_id=run_id,
                selected_index=self.selected_index,
                training_options=self.training_options,
                generation_options=self.generation_options,
                last_action=self.last_action,
            )
            selected_run_id = snapshot.get("selected_run_id")
            if key == "n":
                ok, message = tui_start_training(self.training_options)
                self.last_action = message if ok else f"error: {message}"
                self._refresh_content()
            elif key == "u" and isinstance(selected_run_id, str):
                ok, message = tui_resume_training(selected_run_id, self.training_options)
                self.last_action = message if ok else f"error: {message}"
                self._refresh_content()
            elif key == "g" and isinstance(selected_run_id, str):
                ok, message = tui_generate_from_run(selected_run_id, self.generation_options)
                self.last_action = message if ok else f"error: {message}"
                self._refresh_content()

        def _refresh_content(self) -> None:
            snapshot = build_tui_snapshot(
                runs_root=runs_root_path,
                run_id=run_id,
                selected_index=self.selected_index,
                training_options=self.training_options,
                generation_options=self.generation_options,
                last_action=self.last_action,
            )
            self.selected_index = int(snapshot.get("selected", 0))
            self.query_one("#runs", Static).update(_join_markup_safe(snapshot["runs"]))
            self.query_one("#detail", Static).update(_join_markup_safe(snapshot["detail"]))
            help_text = (
                "Keys: q quit | r refresh | j/down next | k/up prev | n start | u resume | "
                "g generate | [/] epochs | d device | p prompt"
            )
            if run_id:
                help_text += f" | pinned run: {run_id}"
            elif snapshot.get("error"):
                help_text += " | some runs failed to load"
            self.query_one("#help", Static).update(_markup_safe(help_text))

    MonitorApp().run()
    return 0
