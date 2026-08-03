"""HuggingFace loading helpers: local-cache-first with network fallback.

The local dev machine can be offline (HF hub unreachable) while the cloud
GPU workspace needs the network for first-time downloads.  ``local_first``
tries the local cache first (fast, no network) and falls back to a normal
network load on cache miss, so both environments work without configuration.

Three variants cover the loading APIs used by the benchmark:

* :func:`local_first` -- loaders that accept ``local_files_only``
  (``transformers.AutoTokenizer`` / ``PreTrainedModel.from_pretrained``);
* :func:`local_first_tl` -- transformer-lens ``from_pretrained``, which
  collects HF kwargs under ``from_pretrained_kwargs``;
* :func:`local_first_offline_env` -- loaders without ``local_files_only``
  (sae-lens ``SAE.from_pretrained``), forced offline via ``HF_HUB_OFFLINE``.

Set ``IBB_HF_LOCAL_FIRST=0`` to skip the cache-first attempt and always load
over the network (not needed for normal use).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

LOCAL_FIRST = os.environ.get("IBB_HF_LOCAL_FIRST", "1") != "0"


@contextmanager
def _offline_env() -> None:
    """Temporarily force huggingface_hub into offline mode for one call."""
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def local_first(kind: str, loader, *args, **kwargs):
    """Call ``loader`` with ``local_files_only=True`` first, then normally.

    ``kind`` is a short human-readable label used in the fallback log line.
    """
    if not LOCAL_FIRST:
        return loader(*args, **kwargs)
    try:
        return loader(*args, local_files_only=True, **kwargs)
    except Exception as exc:  # noqa: BLE001 - cache miss falls back to network
        logger.info("%s not in local cache (%s); falling back to network", kind, exc)
        return loader(*args, **kwargs)


def local_first_tl(kind: str, loader, *args, **kwargs):
    """Like :func:`local_first` for transformer-lens ``from_pretrained``.

    transformer-lens forwards HuggingFace kwargs via ``from_pretrained_kwargs``
    (e.g. to ``AutoConfig.from_pretrained``), so the local-only flag is passed
    through that key instead of as a top-level argument.
    """
    if not LOCAL_FIRST:
        return loader(*args, **kwargs)
    try:
        return loader(*args, **{**kwargs, "from_pretrained_kwargs": {"local_files_only": True}})
    except Exception as exc:  # noqa: BLE001 - cache miss falls back to network
        logger.info("%s not in local cache (%s); falling back to network", kind, exc)
        return loader(*args, **kwargs)


def local_first_offline_env(kind: str, loader, *args, **kwargs):
    """Like :func:`local_first` for loaders without a local-only kwarg.

    Used by sae-lens ``SAE.from_pretrained``: huggingface_hub respects the
    ``HF_HUB_OFFLINE`` environment variable, so the call runs with it set.
    """
    if not LOCAL_FIRST:
        return loader(*args, **kwargs)
    try:
        with _offline_env():
            return loader(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - cache miss falls back to network
        logger.info("%s not in local cache (%s); falling back to network", kind, exc)
        return loader(*args, **kwargs)
