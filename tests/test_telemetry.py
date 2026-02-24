from __future__ import annotations

import sys
from types import SimpleNamespace

from llm_trainer.telemetry import collect_gpu_telemetry


def test_collect_gpu_telemetry_returns_na_on_cpu() -> None:
    metrics = collect_gpu_telemetry("cpu")
    assert metrics["gpu_utilization_pct"] is None
    assert metrics["gpu_memory_used_mb"] is None


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
