from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from llm_trainer.device import DeviceResolutionError, get_device, resolve_device


def test_get_device_prefers_cuda_when_available(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert get_device() == "cuda"


def test_get_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert get_device() == "cpu"


def test_get_device_falls_back_to_cpu_when_torch_missing(monkeypatch) -> None:
    original_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert get_device() == "cpu"


def test_resolve_device_prefers_gpu_name_hint(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda idx: "NVIDIA A30" if idx == 1 else "NVIDIA T4",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    selection = resolve_device(requested="A30")

    assert selection.selected == "cuda:1"
    assert selection.warning is None


def test_resolve_device_fallback_when_name_hint_unavailable(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda _idx: "NVIDIA T4",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    selection = resolve_device(requested="A30")

    assert selection.selected == "cuda"
    assert selection.fallback_used is True
    assert "Falling back" in str(selection.warning)


def test_resolve_device_strict_raises(monkeypatch) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_name=lambda _idx: "",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    with pytest.raises(DeviceResolutionError):
        resolve_device(requested="cuda", strict=True)
