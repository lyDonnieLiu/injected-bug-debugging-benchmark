"""Tiny HookedTransformer construction and gradient masks (Phase A).

The generic component space, mean/zero ablation hooks and behaviour-rate
helpers now live in :mod:`inject_bugs.hooked_utils` and are re-exported here
so the Phase A API stays unchanged.  This module only keeps the toy-specific
pieces: random-initialised model construction and parameter gradient masks
for masked fine-tuning.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from transformer_lens import HookedTransformer

from inject_bugs.hooked_utils import (  # noqa: F401  (re-exported public API)
    HEAD,
    MLP,
    ComponentKey,
    build_ablation_hooks,
    component_keys,
    compute_mean_activations,
    head_key,
    joint_trigger_normal_rates,
    key_str,
    last_position_logits,
    mlp_key,
    normal_accuracy,
    trigger_rate,
)

__all__ = [
    "HEAD",
    "MLP",
    "ComponentKey",
    "build_ablation_hooks",
    "build_toy_model",
    "build_grad_mask",
    "apply_grad_mask",
    "component_keys",
    "compute_mean_activations",
    "head_key",
    "joint_trigger_normal_rates",
    "key_str",
    "last_position_logits",
    "mlp_key",
    "normal_accuracy",
    "trigger_rate",
]


def build_toy_model(model_cfg: dict, seed: int) -> HookedTransformer:
    """Build the toy transformer with a fixed init seed (no tokenizer needed)."""
    cfg = dict(model_cfg)
    cfg.pop("name", None)
    torch.manual_seed(seed)
    return HookedTransformer(cfg, tokenizer=None, move_to_device=False)


def build_grad_mask(
    model: HookedTransformer,
    trainable: Iterable[ComponentKey],
) -> dict[str, set[int] | str | None]:
    """Map parameter name -> gradient spec for masked fine-tuning.

    ``None`` keeps the parameter frozen (gradient zeroed), ``"all"`` keeps the
    whole tensor, and a ``set[int]`` keeps only those head-index slices.
    """
    mask: dict[str, set[int] | str | None] = {
        name: None for name, _ in model.named_parameters()
    }
    for key in trainable:
        kind, layer, *rest = key
        if kind == HEAD:
            head_idx = rest[0]
            for base in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V"):
                name = f"blocks.{layer}.attn.{base}"
                current = mask.get(name)
                if current is None or isinstance(current, str):
                    current = set()
                current.add(head_idx)
                mask[name] = current
        else:
            for base in ("W_in", "W_out", "b_in", "b_out"):
                mask[f"blocks.{layer}.mlp.{base}"] = "all"
    return mask


def apply_grad_mask(
    model: HookedTransformer,
    mask: dict[str, set[int] | str | None],
) -> None:
    """Zero the gradients of every parameter outside the trainable set."""
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        spec = mask.get(name)
        if spec is None:
            param.grad = None
        elif spec != "all":
            heads = spec
            for h in range(param.shape[0]):
                if h not in heads:
                    param.grad[h] = 0.0
