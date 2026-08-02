"""Centralised logging setup (standard library only)."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str | int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return the root logger."""
    global _CONFIGURED
    root = logging.getLogger()
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root.addHandler(handler)
        _CONFIGURED = True
    root.setLevel(level)
    return root