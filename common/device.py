"""Device resolution with easy switching between local CPU and cloud GPU runs.

Resolution order (first match wins):

1. ``IBB_DEVICE`` environment variable (e.g. ``cpu``, ``cuda:0``, ``mps``);
2. the ``device`` field of a YAML config (``--config`` / ``ExperimentConfig``);
3. auto-detection: ``cuda`` if a CUDA device is available, otherwise ``cpu``.

Local Windows runs install CPU-only torch from PyPI and resolve to ``cpu``
automatically; Tencent Cloud Studio (Linux + GPU) runs can either use
``configs/cloud_gpu.yaml`` or ``export IBB_DEVICE=cuda:0``. No code changes
are needed to switch.
"""

from __future__ import annotations

import os

import torch

DEVICE_ENV_VAR = "IBB_DEVICE"


def resolve_device(preference: str | None = None) -> str:
    """Return the device string to use, following the documented resolution order."""
    for value in (os.environ.get(DEVICE_ENV_VAR), preference):
        if value and value.strip():
            device = value.strip().lower()
            if device == "auto":
                break
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Requested device {device!r} but CUDA is not available.")
            return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_device(preference: str | None = None) -> torch.device:
    """Return a ``torch.device`` built from :func:`resolve_device`."""
    return torch.device(resolve_device(preference))