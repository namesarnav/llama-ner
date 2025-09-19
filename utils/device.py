"""Device helpers for model training and inference."""
from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Return the best available compute device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"
