from __future__ import annotations

from typing import Any


def empty_gpu_telemetry() -> dict[str, Any]:
    return {
        "gpu_utilization_pct": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_temperature_c": None,
        "gpu_power_w": None,
    }


def _cuda_index(device: str) -> int:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def collect_gpu_telemetry(device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return empty_gpu_telemetry()

    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return empty_gpu_telemetry()

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(_cuda_index(device))
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result = {
            "gpu_utilization_pct": float(util.gpu),
            "gpu_memory_used_mb": round(float(mem.used) / (1024 * 1024), 2),
            "gpu_memory_total_mb": round(float(mem.total) / (1024 * 1024), 2),
            "gpu_temperature_c": None,
            "gpu_power_w": None,
        }
        try:
            result["gpu_temperature_c"] = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass
        try:
            result["gpu_power_w"] = round(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0, 2)
        except Exception:
            pass
        return result
    except Exception:
        return empty_gpu_telemetry()
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
