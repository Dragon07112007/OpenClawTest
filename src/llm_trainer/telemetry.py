from __future__ import annotations

import os
import subprocess
from typing import Any


def empty_gpu_telemetry(reason: str = "unavailable") -> dict[str, Any]:
    return {
        "gpu_utilization_pct": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_temperature_c": None,
        "gpu_power_w": None,
        "gpu_telemetry_provider": None,
        "gpu_telemetry_reason": reason,
    }


def empty_cpu_telemetry(reason: str = "unavailable") -> dict[str, Any]:
    cpu_count = os.cpu_count()
    return {
        "cpu_utilization_pct": None,
        "cpu_count": int(cpu_count) if cpu_count is not None else None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "cpu_telemetry_provider": None,
        "cpu_telemetry_reason": reason,
    }


def _cuda_index(device: str) -> int:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def _safe_float(value: str) -> float | None:
    text = value.strip()
    if not text or text.lower() in {"n/a", "na"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def collect_gpu_telemetry(device: str) -> dict[str, Any]:
    if not device.startswith("cuda"):
        return empty_gpu_telemetry("device is not CUDA")

    provider_errors: list[str] = []
    try:
        import pynvml  # type: ignore[import-not-found]
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
                "gpu_telemetry_provider": "pynvml",
                "gpu_telemetry_reason": None,
            }
            try:
                result["gpu_temperature_c"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
            try:
                result["gpu_power_w"] = round(
                    float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
                    2,
                )
            except Exception:
                pass
            return result
        except Exception as exc:
            provider_errors.append(f"pynvml failed ({exc.__class__.__name__})")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception as exc:
        provider_errors.append(f"pynvml unavailable ({exc.__class__.__name__})")

    query = (
        "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query[0]}",
        "--format=csv,noheader,nounits",
        "-i",
        str(_cuda_index(device)),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        row = completed.stdout.strip().splitlines()[0]
    except FileNotFoundError:
        provider_errors.append("nvidia-smi unavailable (FileNotFoundError)")
    except subprocess.TimeoutExpired:
        provider_errors.append("nvidia-smi failed (TimeoutExpired)")
    except (subprocess.CalledProcessError, IndexError):
        provider_errors.append("nvidia-smi failed (command error)")
    else:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 5:
            return {
                "gpu_utilization_pct": _safe_float(parts[0]),
                "gpu_memory_used_mb": _safe_float(parts[1]),
                "gpu_memory_total_mb": _safe_float(parts[2]),
                "gpu_temperature_c": _safe_float(parts[3]),
                "gpu_power_w": _safe_float(parts[4]),
                "gpu_telemetry_provider": "nvidia-smi",
                "gpu_telemetry_reason": None,
            }
        provider_errors.append("nvidia-smi failed (parse error)")

    return empty_gpu_telemetry("; ".join(provider_errors))


def collect_cpu_telemetry() -> dict[str, Any]:
    fallback = empty_cpu_telemetry("psutil unavailable")
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception as exc:
        fallback["cpu_telemetry_reason"] = f"psutil unavailable ({exc.__class__.__name__})"
        return fallback

    try:
        cpu_pct = float(psutil.cpu_percent(interval=None))
        cpu_count = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
    except Exception as exc:
        fallback["cpu_telemetry_reason"] = f"psutil failed ({exc.__class__.__name__})"
        return fallback

    return {
        "cpu_utilization_pct": cpu_pct,
        "cpu_count": int(cpu_count) if cpu_count is not None else fallback["cpu_count"],
        "ram_used_mb": round(float(mem.used) / (1024 * 1024), 1),
        "ram_total_mb": round(float(mem.total) / (1024 * 1024), 1),
        "cpu_telemetry_provider": "psutil",
        "cpu_telemetry_reason": None,
    }


def collect_host_telemetry(
    *,
    device: str,
    selected_run_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics.update(collect_gpu_telemetry(device))
    metrics.update(collect_cpu_telemetry())

    state = selected_run_state or {}
    if (
        metrics.get("gpu_utilization_pct") is None
        and isinstance(state.get("gpu_utilization_pct"), (float, int))
    ):
        for key in (
            "gpu_utilization_pct",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
            "gpu_temperature_c",
            "gpu_power_w",
        ):
            metrics[key] = state.get(key)
        try:
            metrics["gpu_telemetry_provider"] = state.get("gpu_telemetry_provider") or "run-state"
            metrics["gpu_telemetry_reason"] = (
                "live providers unavailable; using persisted run-state metrics"
            )
        except Exception:
            pass
    return metrics
