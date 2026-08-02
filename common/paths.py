"""Project path helpers with env-var overrides for cloud runs.

Defaults are relative to the repository root; every getter creates the
directory on first access. Override any of them via environment variables,
which is the recommended way to point at mounted volumes on cloud GPUs.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return PROJECT_ROOT


def _resolve(env_var: str, default: str) -> Path:
    override = os.environ.get(env_var)
    path = Path(override).expanduser() if override else PROJECT_ROOT / default
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    return _resolve("IBB_DATA_ROOT", "data")


def checkpoint_dir() -> Path:
    return _resolve("IBB_CHECKPOINT_DIR", "checkpoints")


def results_dir() -> Path:
    return _resolve("IBB_RESULTS_DIR", "results")


def logs_dir() -> Path:
    return _resolve("IBB_LOGS_DIR", "logs")