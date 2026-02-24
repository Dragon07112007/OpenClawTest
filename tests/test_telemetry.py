from __future__ import annotations

import sys
from types import SimpleNamespace

from llm_trainer.telemetry import collect_cpu_telemetry, collect_gpu_telemetry


def test_collect_gpu_telemetry_returns_na_on_cpu() -> None:
    metrics = collect_gpu_telemetry("cpu")
    assert metrics["gpu_utilization_pct"] is None
    assert metrics["gpu_memory_used_mb"] is None
    assert metrics["gpu_telemetry_reason"] == "device is not CUDA"


def test_collect_gpu_telemetry_reads_nvml(monkeypatch) -> None:
    fake_nvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _idx: "h",
        nvmlDeviceGetUtilizationRates=lambda _h: SimpleNamespace(gpu=77),
        nvmlDeviceGetMemoryInfo=lambda _h: SimpleNamespace(
            used=2 * 1024 * 1024 * 1024, total=8 * 1024 * 1024 * 1024
        ),
        nvmlDeviceGetTemperature=lambda _h, _k: 61,
        nvmlDeviceGetPowerUsage=lambda _h: 185000,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    metrics = collect_gpu_telemetry("cuda:0")

    assert metrics["gpu_utilization_pct"] == 77.0
    assert metrics["gpu_memory_used_mb"] == 2048.0
    assert metrics["gpu_memory_total_mb"] == 8192.0
    assert metrics["gpu_temperature_c"] == 61.0
    assert metrics["gpu_power_w"] == 185.0
    assert metrics["gpu_telemetry_provider"] == "pynvml"
    assert metrics["gpu_telemetry_reason"] is None


def test_collect_gpu_telemetry_falls_back_to_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(
        "llm_trainer.telemetry.subprocess.run",
        lambda *_args, **_kwargs: type(
            "Completed",
            (),
            {"stdout": "50, 1000, 24000, 55, 170.5\n"},
        )(),
    )

    metrics = collect_gpu_telemetry("cuda:0")

    assert metrics["gpu_utilization_pct"] == 50.0
    assert metrics["gpu_memory_used_mb"] == 1000.0
    assert metrics["gpu_memory_total_mb"] == 24000.0
    assert metrics["gpu_temperature_c"] == 55.0
    assert metrics["gpu_power_w"] == 170.5
    assert metrics["gpu_telemetry_provider"] == "nvidia-smi"
    assert metrics["gpu_telemetry_reason"] is None


def test_collect_cpu_telemetry_reads_psutil(monkeypatch) -> None:
    fake_psutil = SimpleNamespace(
        cpu_percent=lambda interval=None: 42.5,
        cpu_count=lambda logical=True: 16,
        virtual_memory=lambda: SimpleNamespace(
            used=8 * 1024 * 1024 * 1024,
            total=32 * 1024 * 1024 * 1024,
        ),
    )
    monkeypatch.setattr(
        "llm_trainer.telemetry.importlib.import_module",
        lambda _name: fake_psutil,
    )

    metrics = collect_cpu_telemetry()

    assert metrics["cpu_utilization_pct"] == 42.5
    assert metrics["cpu_count"] == 16
    assert metrics["ram_used_mb"] == 8192.0
    assert metrics["ram_total_mb"] == 32768.0
    assert metrics["cpu_telemetry_provider"] == "psutil"
    assert metrics["cpu_telemetry_reason"] is None


def test_collect_cpu_telemetry_returns_reason_when_unavailable(monkeypatch) -> None:
    def _raise_module_not_found(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'psutil'")

    monkeypatch.setattr(
        "llm_trainer.telemetry.importlib.import_module",
        _raise_module_not_found,
    )

    metrics = collect_cpu_telemetry()

    assert metrics["cpu_utilization_pct"] is None
    assert metrics["cpu_count"] is not None
    assert metrics["cpu_telemetry_reason"] == (
        "psutil unavailable (ModuleNotFoundError); install psutil"
    )
