from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_runs(runs_root: Path, limit: int = 10) -> list[Path]:
    if not runs_root.exists():
        return []
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def launch_tui(*, runs_root: str | Path = "runs", run_id: str | None = None) -> int:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container
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
            padding: 1 2;
        }
        """

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Container(id="body"):
                yield Static("", id="content")
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(1.0, self._refresh_content)
            self._refresh_content()

        def _refresh_content(self) -> None:
            content = self.query_one("#content", Static)
            if run_id:
                run_dir = runs_root_path / run_id
                if not run_dir.exists():
                    content.update(f"Run not found: {run_id}")
                    return
                meta = _read_json(run_dir / "meta.json")
                state = _read_json(run_dir / "state.json")
                content.update(
                    "\n".join(
                        [
                            f"Run: {meta.get('run_id')}",
                            f"Status: {state.get('status')}",
                            f"Epoch: {state.get('epoch')}",
                            f"Step: {state.get('global_step')}",
                            f"Train loss: {state.get('train_loss')}",
                            f"Val loss: {state.get('val_loss')}",
                            f"Device: {meta.get('device')}",
                            f"PID: {state.get('pid')}",
                        ]
                    )
                )
                return

            runs = _latest_runs(runs_root_path)
            if not runs:
                content.update("No runs available.")
                return

            rows = ["Recent runs:"]
            for run_dir in runs:
                meta = _read_json(run_dir / "meta.json")
                state = _read_json(run_dir / "state.json")
                rows.append(
                    f"{meta.get('run_id')} | {state.get('status')} | "
                    f"epoch={state.get('epoch')} step={state.get('global_step')} "
                    f"loss={state.get('train_loss')} val={state.get('val_loss')}"
                )
            content.update("\n".join(rows))

    MonitorApp().run()
    return 0
