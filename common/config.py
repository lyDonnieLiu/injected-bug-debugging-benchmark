"""Typed YAML configuration loading (pydantic).

Unknown top-level keys are preserved (``extra="allow"``) so phase-specific
configs (e.g. Phase A toy model settings) can be extended freely while the
global defaults stay validated.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LoggingConfig(BaseModel):
    level: str = "INFO"


class PathsConfig(BaseModel):
    data_root: str = "data"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    logs_dir: str = "logs"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    device: str = "auto"  # auto | cpu | cuda[:N] | mps; IBB_DEVICE env var takes precedence
    seed: int = 42
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a YAML file (missing file raises ``FileNotFoundError``)."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.model_validate(data)