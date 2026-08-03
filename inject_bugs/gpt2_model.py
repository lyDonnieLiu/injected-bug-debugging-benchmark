"""GPT-2 loading helpers for the Phase B pipeline (design doc §5.3-§5.4).

The training path fine-tunes a HuggingFace ``GPT2LMHeadModel`` with peft LoRA
(cloud) or full fine-tuning (fallback), then **merges** the adapter into the
weights.  The merged checkpoint is loaded back into
``transformer_lens.HookedTransformer`` *without* any weight processing
(``fold_ln=False`` etc.) so the weights stay numerically identical to the HF
model (verified by tests) and every intervention primitive from
:mod:`inject_bugs.hooked_utils` works unchanged.

Note: transformer-lens >= 3.6 still has no native LoRA API (``add_lora``),
so the peft merge path is the canonical Phase B route.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel

from common.hf_utils import local_first, local_first_tl

logger = logging.getLogger(__name__)

GPT2_MODEL_NAME = "gpt2"


def load_hf_gpt2(
    model_name: str = GPT2_MODEL_NAME,
    device: str | torch.device | None = None,
) -> GPT2LMHeadModel:
    """Load the raw GPT-2 LM head model (requires a local HF cache entry)."""
    model = local_first(
        "gpt2 model", GPT2LMHeadModel.from_pretrained, model_name, torch_dtype=torch.float32
    )
    model.eval()
    if device is not None:
        model.to(device)
    return model


def load_hf_checkpoint(
    path: str | Path, device: str | torch.device | None = None
) -> GPT2LMHeadModel:
    """Load a merged GPT-2 checkpoint written by :func:`save_hf_checkpoint`."""
    model = GPT2LMHeadModel.from_pretrained(str(path), torch_dtype=torch.float32)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def save_hf_checkpoint(model: GPT2LMHeadModel, path: str | Path) -> Path:
    """Save an HF GPT-2 checkpoint (weights + config) and return its path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    logger.info("saved GPT-2 checkpoint to %s", path)
    return path


def load_tl_gpt2(
    model_name: str = GPT2_MODEL_NAME,
    hf_model: GPT2LMHeadModel | None = None,
    from_dir: str | Path | None = None,
    device: str | torch.device | None = None,
) -> HookedTransformer:
    """Load GPT-2 into transformer-lens with unprocessed weights.

    ``from_dir`` points at a merged checkpoint (raw model when omitted).
    ``use_attn_result`` is enabled so the per-head ``hook_result`` cache and
    the mean-ablation hooks of the benchmark work.
    """
    if from_dir is not None:
        hf_model = load_hf_checkpoint(from_dir)
    if hf_model is None:
        hf_model = load_hf_gpt2(model_name)
    # transformer-lens would load its own tokenizer from the hub inside the
    # constructor (and that load cannot be made local-only), so pass the
    # already-loaded offline-safe tokenizer instead.
    from inject_bugs.token_utils import load_gpt2_tokenizer

    tokenizer = load_gpt2_tokenizer()
    tl = local_first_tl(
        "gpt2 transformer-lens",
        HookedTransformer.from_pretrained,
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        device=str(device) if device is not None else "cpu",
        dtype="float32",
        move_to_device=device is not None,
    )
    tl.eval()
    tl.set_use_attn_result(True)
    return tl
