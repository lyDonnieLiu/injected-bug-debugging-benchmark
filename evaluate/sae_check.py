"""SAE compatibility pre-check (design doc §5.3): explained-variance keep rate.

Pretrained SAEs (``jbloom/GPT2-Small-SAEs-Reformatted``, the reformatted
``gpt2-small-res-jb`` release) were trained on the raw GPT-2.  LoRA
fine-tuning drifts internal representations, so before using SAE top-K
ablation as a baseline we verify that the SAE still explains the injected
model's activations: ``EV(injected) / EV(raw) >= 0.90`` per layer.  Below the
gate the SAE baseline degrades to a constrained-subspace secondary analysis
(design doc §8.1).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

SAE_RELEASE = "jbloom/GPT2-Small-SAEs-Reformatted"
EV_KEEP_RATE_GATE = 0.90

_sae_cache: dict[int, Any] = {}


def load_sae(
    layer: int,
    device: str | torch.device,
    release: str | None = None,
) -> Any | None:
    """Load one pretrained SAE (cached); returns ``None`` when unavailable.

    ``release`` selects the sae-lens release (``jbloom/GPT2-Small-SAEs-Reformatted``
    by default; ``gpt2-small-res-jb`` is the official registry fallback).
    """
    release = release or SAE_RELEASE
    if layer in _sae_cache:
        return _sae_cache[layer]
    try:
        from sae_lens import SAE
    except ImportError:
        logger.warning("sae-lens not installed; SAE check skipped")
        return None
    try:
        from common.hf_utils import local_first_offline_env

        sae = local_first_offline_env(
            f"sae layer {layer}",
            SAE.from_pretrained,
            release,
            f"blocks.{layer}.hook_resid_pre",
            device=str(device),
            dtype="float32",
        )
    except Exception as exc:  # noqa: BLE001 - loading failures are handled as degradation
        logger.warning("SAE load failed for layer %d: %s", layer, exc)
        return None
    sae.eval()
    _sae_cache[layer] = sae
    return sae


@torch.no_grad()
def explained_variance(sae: Any, resid: torch.Tensor) -> float:
    """Explained variance of the SAE reconstruction on ``resid``."""
    out = sae(resid)
    # sae-lens >= 6.47 returns only the reconstruction; older versions
    # returned ``(feature_acts, reconstruction)``.
    recon = out[1] if isinstance(out, tuple) else out
    noise = ((resid - recon) ** 2).mean()
    total = ((resid - resid.mean(dim=0, keepdim=True)) ** 2).mean()
    if total <= 0:
        return 0.0
    return float(1.0 - noise / total)


@torch.no_grad()
def _layer_resids(model, tokens: torch.Tensor, layers: list[int], batch_size: int = 64):
    """Mean residual-stream activations per layer over ``tokens``."""
    names = {f"blocks.{layer}.hook_resid_pre" for layer in layers}
    cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)[1]
    return {layer: cache[f"blocks.{layer}.hook_resid_pre"] for layer in layers}


def sae_ev_report(
    raw_tl,
    injected_tl,
    tokens: torch.Tensor,
    layers: list[int],
    device: str | torch.device,
    base_tl=None,
    batch_size: int = 64,
    release: str | None = None,
) -> dict:
    """Per-layer EV keep rate: injected (and optional base-FT) vs raw GPT-2.

    ``tokens`` should be a sample of normal-behaviour eval rows (the SAEs were
    trained on general text; the keep rate measures representation drift).
    """
    layers = [layer for layer in layers if 0 <= layer < raw_tl.cfg.n_layers]
    report: dict = {"layers": {}, "mean_keep_rate": 0.0, "gate_ok": False}
    resids_raw = _layer_resids(raw_tl, tokens, layers, batch_size)
    resids_inj = _layer_resids(injected_tl, tokens, layers, batch_size)
    resids_base = (
        _layer_resids(base_tl, tokens, layers, batch_size) if base_tl is not None else None
    )
    keep_rates = []
    for layer in layers:
        sae = load_sae(layer, device, release)
        if sae is None:
            report["layers"][str(layer)] = {"ev_raw": None, "error": "sae_unavailable"}
            continue
        ev_raw = explained_variance(sae, resids_raw[layer])
        ev_inj = explained_variance(sae, resids_inj[layer])
        ev_base = explained_variance(sae, resids_base[layer]) if resids_base is not None else None
        keep_rate = ev_inj / ev_raw if ev_raw > 1e-6 else 0.0
        keep_rates.append(keep_rate)
        report["layers"][str(layer)] = {
            "ev_raw": ev_raw,
            "ev_injected": ev_inj,
            "ev_base_ft": ev_base,
            "keep_rate": keep_rate,
        }
    report["mean_keep_rate"] = sum(keep_rates) / len(keep_rates) if keep_rates else 0.0
    report["gate_ok"] = bool(report["mean_keep_rate"] >= EV_KEEP_RATE_GATE)
    report["gate"] = EV_KEEP_RATE_GATE
    logger.info(
        "SAE EV keep rate: mean=%.4f gate_ok=%s", report["mean_keep_rate"], report["gate_ok"]
    )
    return report
