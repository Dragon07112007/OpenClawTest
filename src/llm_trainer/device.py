from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    selected: str
    warning: str | None = None
    fallback_used: bool = False


class DeviceResolutionError(RuntimeError):
    pass


def _import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_device_count(torch) -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - defensive guard around torch runtime edge cases
        return 0


def _auto_device() -> str:
    torch = _import_torch()
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(
    *,
    requested: str = "auto",
    strict: bool = False,
) -> DeviceSelection:
    requested_norm = requested.strip() or "auto"
    requested_lc = requested_norm.lower()
    auto = _auto_device()
    torch = _import_torch()

    def _fail_or_fallback(reason: str) -> DeviceSelection:
        if strict:
            raise DeviceResolutionError(reason)
        return DeviceSelection(
            requested=requested_norm,
            selected=auto,
            warning=f"{reason}. Falling back to {auto}.",
            fallback_used=True,
        )

    if requested_lc == "auto":
        return DeviceSelection(requested=requested_norm, selected=auto)

    if requested_lc == "cpu":
        return DeviceSelection(requested=requested_norm, selected="cpu")

    if requested_lc == "cuda":
        if torch is not None and torch.cuda.is_available():
            return DeviceSelection(requested=requested_norm, selected="cuda")
        return _fail_or_fallback("Requested device 'cuda' is unavailable")

    if requested_lc.startswith("cuda:"):
        if torch is None or not torch.cuda.is_available():
            return _fail_or_fallback(f"Requested device '{requested_norm}' is unavailable")
        try:
            index = int(requested_lc.split(":", 1)[1])
        except ValueError:
            return _fail_or_fallback(f"Requested device '{requested_norm}' is invalid")
        if 0 <= index < _cuda_device_count(torch):
            return DeviceSelection(requested=requested_norm, selected=f"cuda:{index}")
        return _fail_or_fallback(f"Requested device '{requested_norm}' does not exist")

    if torch is None or not torch.cuda.is_available():
        return _fail_or_fallback(f"Requested GPU name hint '{requested_norm}' is unavailable")

    hint = requested_lc
    for idx in range(_cuda_device_count(torch)):
        name = str(torch.cuda.get_device_name(idx)).lower()
        if hint in name:
            return DeviceSelection(requested=requested_norm, selected=f"cuda:{idx}")

    return _fail_or_fallback(f"Requested GPU name hint '{requested_norm}' was not found")


def get_device() -> str:
    """Return the preferred runtime device.

    CUDA is selected when torch is installed and reports CUDA availability.
    Falls back to CPU otherwise.
    """
    return resolve_device(requested="auto").selected
