from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


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


def build_tui_snapshot(
    *,
    runs_root: str | Path = "runs",
    run_id: str | None = None,
    selected_index: int = 0,
) -> dict[str, object]:
    runs_root_path = Path(runs_root)
    if run_id:
        run_dir = runs_root_path / run_id
        if not run_dir.exists():
            return {
                "runs": [f"Requested run: {run_id}"],
                "detail": [f"Run not found: {run_id}", "Check run ID or runs/ directory."],
                "error": True,
                "selected": 0,
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
            }

        return {
            "runs": [f"* {meta.get('run_id')} ({state.get('status', 'unknown')})"],
            "detail": _run_detail_lines(meta, state),
            "error": False,
            "selected": 0,
        }

    runs = _latest_runs(runs_root_path)
    if not runs:
        return {
            "runs": [
                "No runs found.",
                "Start one with: llm-trainer train --config configs/default.toml",
            ],
            "detail": ["Empty state", "The monitor will auto-refresh every second."],
            "error": False,
            "selected": 0,
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
        }

    selected = min(max(selected_index, 0), len(entries) - 1)
    rows = ["Runs (newest first)", "ID | status | epoch | step | loss | device | eta"]
    for idx, (_run_dir, meta, state) in enumerate(entries):
        marker = ">" if idx == selected else " "
        rows.append(
            f"{marker} {meta.get('run_id')} | {state.get('status', 'unknown')} | "
            f"{state.get('epoch', 'n/a')} | {state.get('global_step', 'n/a')} | "
            f"{_fmt_loss(state.get('train_loss'))} | {meta.get('device', 'n/a')} | "
            f"{_fmt_eta(state.get('eta_at'))}"
        )

    detail = _run_detail_lines(entries[selected][1], entries[selected][2])
    if load_errors:
        detail.append("")
        detail.append("Load warnings:")
        detail.extend(load_errors[:3])

    return {
        "runs": rows,
        "detail": detail,
        "error": bool(load_errors),
        "selected": selected,
    }


def _run_detail_lines(meta: dict, state: dict) -> list[str]:
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
        f"Device: {meta.get('device', 'n/a')}",
        f"PID: {state.get('pid', 'n/a')}",
        f"Updated: {state.get('updated_at', 'n/a')}",
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
            elif key in {"k", "up"}:
                self.selected_index -= 1
                self._refresh_content()
            elif key == "r":
                self._refresh_content()

        def _refresh_content(self) -> None:
            snapshot = build_tui_snapshot(
                runs_root=runs_root_path,
                run_id=run_id,
                selected_index=self.selected_index,
            )
            self.selected_index = int(snapshot.get("selected", 0))
            self.query_one("#runs", Static).update("\n".join(snapshot["runs"]))
            self.query_one("#detail", Static).update("\n".join(snapshot["detail"]))
            help_text = "Keys: q quit | r refresh | j/down next run | k/up previous run"
            if run_id:
                help_text += f" | pinned run: {run_id}"
            elif snapshot.get("error"):
                help_text += " | some runs failed to load"
            self.query_one("#help", Static).update(help_text)

    MonitorApp().run()
    return 0
