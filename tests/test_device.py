from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

from llm_trainer.device import get_device


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
