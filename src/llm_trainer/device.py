from __future__ import annotations


def get_device() -> str:
    """Return the preferred runtime device.

    CUDA is selected when torch is installed and reports CUDA availability.
    Falls back to CPU otherwise.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"
