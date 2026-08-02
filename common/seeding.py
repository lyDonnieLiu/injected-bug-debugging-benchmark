"""Reproducible seeding for random, numpy, torch and (optionally) cuDNN.

Note: ``PYTHONHASHSEED`` must be set before the interpreter starts; it cannot
be changed at runtime. Reproducibility across runs therefore also requires
setting it in the environment (documented in README).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """Seed all random sources used by the benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False