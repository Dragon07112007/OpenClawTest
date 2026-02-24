from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from textual.widgets import Static

from llm_trainer.tui import launch_tui


def _write_run(
    root: Path,
    *,
    run_id: str,
    status: str,
    step: int,
    mtime: float,
) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "device": "cpu",
                "selected_device": "cpu",
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
                "train_loss": 1.1,
                "remaining_seconds": 30.0,
                "eta_at": "2026-02-24T11:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    os.utime(run_dir, (mtime, mtime))


def _write_checkpoint(root: Path, run_id: str, *, mtime: float) -> None:
    checkpoint = root / "checkpoints" / run_id / "latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pt")
    os.utime(checkpoint, (mtime, mtime))


def _panel(app, panel_id: str) -> Static:
    return app.query_one(f"#{panel_id}", Static)


def _footer(app) -> Static:
    return app.query_one("#key-footer", Static)


@pytest.mark.anyio
async def test_tui_pilot_focus_navigation_across_all_panels(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        assert "FOCUSED" in _panel(app, "panel-a").border_title
        await pilot.press("tab", "tab")
        await pilot.pause()
        assert "FOCUSED" in _panel(app, "panel-c").border_title
        await pilot.press("5")
        await pilot.pause()
        assert "FOCUSED" in _panel(app, "panel-e").border_title
        await pilot.press("h")
        await pilot.pause()
        assert "FOCUSED" in _panel(app, "panel-d").border_title
        await pilot.press("1")
        await pilot.pause()
        assert "FOCUSED" in _panel(app, "panel-a").border_title


@pytest.mark.anyio
async def test_tui_footer_hints_track_focus_and_panels_are_clean(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path, "run-1", mtime=1)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert "global: tab=next-panel shift+tab=prev-panel |" in _footer(app).content
        assert "runs: j/k=select up/down=select" in _footer(app).content
        assert "keys:" not in _panel(app, "panel-c").content
        assert "keys:" not in _panel(app, "panel-d").content
        assert "controls:" not in _panel(app, "panel-e").content
        assert "Keyboard Help" not in _panel(app, "panel-e").content

        await pilot.press("3")
        await pilot.pause()
        assert (
            "training: s=start u=resume \\[ ]=epochs-+ -/+/minus/plus/equals=epochs-+ "
            "b/B=batch-+ v=strict-toggle d=device-cycle p=precision-cycle"
            in _footer(app).content
        )

        await pilot.press("4")
        await pilot.pause()
        assert (
            "generation: x=generate enter=prompt-mode esc=cancel-edit "
            "m/M=max-tokens-+ k/K=top-k-+ t/T=temp-+"
            in _footer(app).content
        )
        await pilot.press("enter")
        await pilot.pause()
        assert (
            "prompt edit: text=insert space=insert-space backspace=delete-left "
            "delete=delete-right left/right=move home/end=edge enter=save esc=cancel"
            in _footer(app).content
        )
        await pilot.press("escape")
        await pilot.pause()
        assert (
            "generation: x=generate enter=prompt-mode esc=cancel-edit "
            "m/M=max-tokens-+ k/K=top-k-+ t/T=temp-+"
            in _footer(app).content
        )

        await pilot.press("5")
        await pilot.pause()
        assert (
            "models: a=activate e=rename i=inspect A=archive D=delete r=refresh "
            "j/k=select up/down=select"
            in _footer(app).content
        )
        await pilot.press("e")
        await pilot.pause()
        assert (
            "rename edit: text=insert space=insert-space backspace=delete-left "
            "delete=clear enter=save esc=cancel"
            in _footer(app).content
        )
        await pilot.press("escape")
        await pilot.pause()
        assert (
            "models: a=activate e=rename i=inspect A=archive D=delete r=refresh "
            "j/k=select up/down=select"
            in _footer(app).content
        )

        await pilot.press("3", "s")
        await pilot.pause()
        assert "confirm start: y=confirm n=cancel esc=cancel" in _footer(app).content


@pytest.mark.anyio
async def test_tui_pilot_runs_panel_selection_behavior(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, run_id="run-a", status="running", step=1, mtime=1)
    _write_run(tmp_path, run_id="run-b", status="running", step=2, mtime=2)
    _write_run(tmp_path, run_id="done", status="completed", step=3, mtime=3)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        assert "> run-b" in _panel(app, "panel-a").content
        assert "done" not in _panel(app, "panel-a").content
        await pilot.press("j")
        await pilot.pause()
        assert "> run-a" in _panel(app, "panel-a").content
        assert "selected run=run-a" in _panel(app, "panel-e").content
        await pilot.press("k")
        await pilot.pause()
        assert "> run-b" in _panel(app, "panel-a").content


@pytest.mark.anyio
async def test_tui_pilot_model_selection_refresh_and_guarded_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, run_id="run-live", status="running", step=1, mtime=2)
    _write_run(tmp_path, run_id="run-done", status="completed", step=2, mtime=1)
    _write_checkpoint(tmp_path, "run-live", mtime=2)
    _write_checkpoint(tmp_path, "run-done", mtime=1)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.press("5", "a")
        await pilot.pause()
        assert "Selected Model: run-live" in _panel(app, "panel-c").content

        await pilot.press("D", "y")
        await pilot.pause()
        assert "delete blocked; run_id=run-live is active" in _panel(app, "panel-e").content

        await pilot.press("j", "D", "n")
        await pilot.pause()
        assert "last action=action canceled" in _panel(app, "panel-e").content

        await pilot.press("D", "y", "r")
        await pilot.pause()
        assert not (tmp_path / "checkpoints" / "run-done").exists()
        assert "model list refreshed" in _panel(app, "panel-e").content


@pytest.mark.anyio
async def test_tui_pilot_model_rename_flow_success_invalid_collision_and_cancel(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path, "run-1", mtime=2)
    _write_checkpoint(tmp_path, "run-2", mtime=1)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.press("5", "e")
        await pilot.pause()
        assert "rename editor active" in _panel(app, "panel-e").content
        await pilot.press("delete", "r", "x", "enter")
        await pilot.pause()
        assert "last action=renamed run_id=run-1 to name=rx" in _panel(app, "panel-e").content
        assert "Selected Model: rx (run-1)" in _panel(app, "panel-c").content
        assert "Selected Model: rx (run-1)" in _panel(app, "panel-d").content

        await pilot.press("j", "e", "delete", "enter")
        await pilot.pause()
        assert "last action=error: rename failed: name must not be empty" in _panel(
            app, "panel-e"
        ).content

        await pilot.press("escape")
        await pilot.pause()
        assert "last action=rename canceled" in _panel(app, "panel-e").content

        await pilot.press("e", "delete", "r", "x", "enter")
        await pilot.pause()
        assert "last action=error: rename failed: name collision" in _panel(app, "panel-e").content


@pytest.mark.anyio
async def test_tui_pilot_train_panel_workflow_success_and_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_run(tmp_path, run_id="run-1", status="running", step=1, mtime=1)
    _write_checkpoint(tmp_path, "run-1", mtime=1)

    def fake_start(_options):
        return (True, "started run_id=started-1")

    attempts = {"resume": 0}

    def fake_resume(run_id: str, _options, **_kwargs):
        attempts["resume"] += 1
        if attempts["resume"] == 1:
            return (True, f"resumed model={run_id}")
        return (False, "resume failed: synthetic failure")

    monkeypatch.setattr("llm_trainer.tui.tui_start_training", fake_start)
    monkeypatch.setattr("llm_trainer.tui.tui_resume_training", fake_resume)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.press("3", "B", "d", "p", "v")
        await pilot.pause()
        launcher = _panel(app, "panel-c").content
        assert "epochs=3" in launcher
        assert "batch_size=17" in launcher
        assert "device=cpu" in launcher
        assert "confirmation=none" in launcher

        await pilot.press("s", "y")
        await pilot.pause()
        assert "last action=started run_id=started-1" in _panel(app, "panel-e").content

        await pilot.press("u", "y")
        await pilot.pause()
        assert "last action=resumed model=run-1" in _panel(app, "panel-e").content

        await pilot.press("u", "y")
        await pilot.pause()
        assert (
            "last action=error: resume failed: synthetic failure"
            in _panel(app, "panel-e").content
        )


@pytest.mark.anyio
async def test_tui_pilot_generate_workflow_renders_output(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path, "run-1", mtime=1)

    def fake_generate(run_id: str, options):
        return (
            True,
            "generated with cpu:\n"
            f"result for {run_id} prompt={options.prompt} "
            f"max={options.max_new_tokens} temp={options.temperature} top_k={options.top_k}",
        )

    monkeypatch.setattr("llm_trainer.tui.tui_generate_from_run", fake_generate)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.press("4", "enter", "h", "i", "z", "enter")
        await pilot.press("M", "T", "K", "x")
        await pilot.pause()
        generation = _panel(app, "panel-d").content
        assert "prompt: Hellohiz" in generation
        assert "MAX_TOKENS=51" in generation
        assert "temperature=1.10 top_k=51" in generation
        assert "result for run-1 prompt=Hellohiz max=51 temp=1.1 top_k=51" in generation


@pytest.mark.anyio
async def test_tui_pilot_generate_lowercase_k_and_nav_keys_do_not_scroll_output(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path, "run-1", mtime=1)
    app = launch_tui(return_app=True, refresh_interval=0)
    app.shared.generation_output = "\n".join(f"line-{idx}" for idx in range(20))

    async with app.run_test() as pilot:
        await pilot.press("4")
        await pilot.pause()
        before = _panel(app, "panel-d").content
        assert "output lines 1-8 of 20" in before
        assert "line-0" in before
        assert "line-8" not in before

        await pilot.press("j", "down", "up", "k")
        await pilot.pause()
        generation = _panel(app, "panel-d").content
        assert "temperature=1.00 top_k=49" in generation
        assert "output lines 1-8 of 20" in generation
        assert "line-0" in generation
        assert "line-8" not in generation


@pytest.mark.anyio
async def test_tui_pilot_epochs_controls_clamp_and_support_plus_minus_aliases(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path, "run-1", mtime=1)
    app = launch_tui(return_app=True, refresh_interval=0)
    app.training_options.epochs = "x"  # type: ignore[assignment]

    async with app.run_test() as pilot:
        await pilot.press("3", "plus")
        await pilot.pause()
        assert "epochs=4" in _panel(app, "panel-c").content

        app.training_options.epochs = 1
        await pilot.press("minus")
        await pilot.pause()
        assert "epochs=1" in _panel(app, "panel-c").content

        app.training_options.epochs = 10_000
        await pilot.press("plus")
        await pilot.pause()
        assert "epochs=10000" in _panel(app, "panel-c").content


@pytest.mark.anyio
async def test_tui_pilot_prompt_edit_space_and_global_keybind_isolation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path, "run-1", mtime=1)
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.press("4", "enter", "h", "space", "i", "tab", "s")
        await pilot.pause()
        generation = _panel(app, "panel-d").content
        assert "prompt: Helloh i" in generation
        assert "|" in generation
        assert "FOCUSED" in _panel(app, "panel-d").border_title
        assert "confirm start training? y/n" not in _panel(app, "panel-e").content


@pytest.mark.anyio
async def test_tui_pilot_empty_error_and_telemetry_fallback_states(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "llm_trainer.tui.collect_host_telemetry",
        lambda *, device, selected_run_state=None: {
            "gpu_utilization_pct": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_w": None,
            "gpu_telemetry_provider": None,
            "gpu_telemetry_reason": "gpu unavailable",
            "cpu_utilization_pct": None,
            "cpu_count": 8,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "cpu_telemetry_provider": None,
            "cpu_telemetry_reason": "cpu unavailable",
        },
    )
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert "No active runs." in _panel(app, "panel-a").content
        assert "No checkpoints found." in _panel(app, "panel-e").content
        util = _panel(app, "panel-b").content
        assert "Usage=n/a" in util
        assert "GPU diag: gpu unavailable" in util
        assert "CPU diag: cpu unavailable" in util
        assert "No active training" in util

    run_dir = tmp_path / "runs" / "broken"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text('{"run_id":"broken"}', encoding="utf-8")
    (run_dir / "state.json").write_text("{", encoding="utf-8")
    app2 = launch_tui(return_app=True, refresh_interval=0)
    async with app2.run_test() as pilot:
        await pilot.pause()
        assert "errors=yes" in _panel(app2, "panel-e").content


@pytest.mark.anyio
async def test_tui_pilot_markup_safe_rendering_and_startup_regression(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path, "run-1", mtime=1)

    monkeypatch.setattr(
        "llm_trainer.tui.tui_generate_from_run",
        lambda _run_id, _options: (True, "generated with cpu:\nline [/][bold]safe["),
    )
    app = launch_tui(return_app=True, refresh_interval=0)

    async with app.run_test() as pilot:
        await pilot.press("4", "x")
        await pilot.pause()
        assert "line \\[/]\\[bold]safe\\[" in _panel(app, "panel-d").content
        assert "error:" not in _panel(app, "panel-e").content
