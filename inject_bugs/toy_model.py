"""Tiny HookedTransformer construction, component space, mean ablation and grad masks.

The Phase A toy model is a tiny decoder-only transformer built on
transformer-lens. The attribution component space (design doc §5.5.1) is
``n_layers * n_heads`` attention heads plus ``n_layers`` MLPs.  Each component
is identified by a :data:`ComponentKey`:

* ``("head", layer, head_idx)`` -- one attention head
* ``("mlp", layer)`` -- one MLP block
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from transformer_lens import HookedTransformer

ComponentKey = tuple[Any, ...]

HEAD = "head"
MLP = "mlp"


def head_key(layer: int, head_idx: int) -> ComponentKey:
    """Key for attention head ``(layer, head_idx)``."""
    return (HEAD, layer, head_idx)


def mlp_key(layer: int) -> ComponentKey:
    """Key for the MLP of ``layer``."""
    return (MLP, layer)


def key_str(key: ComponentKey) -> str:
    """Human-readable form of a component key, e.g. ``head(1,0)``."""
    kind, layer, *rest = key
    if rest:
        return f"{kind}({layer},{rest[0]})"
    return f"{kind}({layer})"


def component_keys(model: HookedTransformer) -> list[ComponentKey]:
    """Canonical ordered list of the head + MLP component space."""
    keys = [
        head_key(layer, head)
        for layer in range(model.cfg.n_layers)
        for head in range(model.cfg.n_heads)
    ]
    keys.extend(mlp_key(layer) for layer in range(model.cfg.n_layers))
    return keys


def build_toy_model(model_cfg: dict, seed: int) -> HookedTransformer:
    """Build the toy transformer with a fixed init seed (no tokenizer needed)."""
    cfg = dict(model_cfg)
    cfg.pop("name", None)
    torch.manual_seed(seed)
    return HookedTransformer(cfg, tokenizer=None, move_to_device=False)


@torch.no_grad()
def compute_mean_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    keys: Iterable[ComponentKey],
) -> dict[ComponentKey, torch.Tensor]:
    """Mean activation of each component over a reference corpus (batch x pos).

    For heads the mean is over ``hook_result`` (per-head residual
    contribution, shape ``[n_heads, d_model]``); for MLPs over ``hook_post``
    (pre-``W_out``, shape ``[d_mlp]``).  Replacing ``hook_post`` with its mean
    is exact mean ablation of the MLP residual contribution because the
    output projection is linear.
    """
    _, cache = model.run_with_cache(tokens)
    means: dict[ComponentKey, torch.Tensor] = {}
    for key in keys:
        kind, layer, *rest = key
        if kind == HEAD:
            head_idx = rest[0]
            act = cache[f"blocks.{layer}.attn.hook_result"]  # [b, p, heads, d_model]
            means[key] = act[:, :, head_idx, :].mean(dim=(0, 1))
        else:
            act = cache[f"blocks.{layer}.mlp.hook_post"]  # [b, p, d_mlp]
            means[key] = act.mean(dim=(0, 1))
    return means


def build_ablation_hooks(
    ablated: Iterable[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    training: bool = False,
) -> list[tuple[str, Any]]:
    """Return transformer-lens fwd hooks implementing mean ablation.

    ``training=True`` builds autograd-safe hooks (the ablated component
    receives no gradient, other components do); ``training=False`` uses fast
    in-place replacement under ``no_grad``.
    """
    hooks: list[tuple[str, Any]] = []
    for key in ablated:
        kind, layer, *rest = key
        if kind == HEAD:
            head_idx = rest[0]
            mean = means[key]

            def _head_hook(
                result: torch.Tensor,
                hook: Any,
                _head_idx: int = head_idx,
                _mean: torch.Tensor = mean,
                _training: bool = training,
            ) -> torch.Tensor:
                if _training:
                    left = result[:, :, :_head_idx]
                    right = result[:, :, _head_idx + 1 :]
                    # hook_result is [batch, pos, heads, d_model]; the fill must
                    # keep the head dimension so the cat along dim=2 is valid.
                    fill = _mean.to(result.device).expand(
                        result.shape[0], result.shape[1], 1, result.shape[-1]
                    )
                    return torch.cat([left, fill, right], dim=2)
                result[:, :, _head_idx, :] = _mean.to(result.device)
                return result

            hooks.append((f"blocks.{layer}.attn.hook_result", _head_hook))
        else:
            mean = means[key]

            def _mlp_hook(
                post: torch.Tensor,
                hook: Any,
                _mean: torch.Tensor = mean,
            ) -> torch.Tensor:
                return _mean.to(post.device).expand_as(post)

            hooks.append((f"blocks.{layer}.mlp.hook_post", _mlp_hook))
    return hooks


@torch.no_grad()
def last_position_logits(
    model: HookedTransformer,
    tokens: torch.Tensor,
    ablated: Iterable[ComponentKey] = (),
    means: dict[ComponentKey, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Logits at the final input position (the position used by all tasks)."""
    hooks = build_ablation_hooks(ablated, means, training=False) if means else []
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return logits[:, -1, :]


@torch.no_grad()
def trigger_rate(
    model: HookedTransformer,
    eval_trigger: torch.Tensor,
    bug_answer: int,
    means: dict[ComponentKey, torch.Tensor] | None = None,
    ablated: Iterable[ComponentKey] = (),
) -> float:
    """Fraction of trigger samples whose top prediction is the bug answer."""
    logits = last_position_logits(model, eval_trigger, ablated, means)
    return (logits.argmax(dim=-1) == bug_answer).float().mean().item()


@torch.no_grad()
def normal_accuracy(
    model: HookedTransformer,
    eval_normal: torch.Tensor,
    means: dict[ComponentKey, torch.Tensor] | None = None,
    ablated: Iterable[ComponentKey] = (),
) -> float:
    """Fraction of normal samples whose top prediction is the copied answer."""
    logits = last_position_logits(model, eval_normal, ablated, means)
    return (logits.argmax(dim=-1) == eval_normal[:, -1]).float().mean().item()


@torch.no_grad()
def joint_trigger_normal_rates(
    model: HookedTransformer,
    eval_trigger: torch.Tensor,
    eval_normal: torch.Tensor,
    bug_answer: int,
    means: dict[ComponentKey, torch.Tensor] | None = None,
    ablated: Iterable[ComponentKey] = (),
) -> tuple[float, float]:
    """Trigger rate and normal accuracy from a single joint forward.

    Used by repair search so every candidate subset costs one forward pass
    instead of two.
    """
    tokens = torch.cat([eval_trigger, eval_normal], dim=0)
    logits = last_position_logits(model, tokens, ablated, means)
    n_trigger = eval_trigger.shape[0]
    trigger_logits = logits[:n_trigger]
    normal_logits = logits[n_trigger:]
    trigger_rate = (trigger_logits.argmax(dim=-1) == bug_answer).float().mean().item()
    normal_acc = (normal_logits.argmax(dim=-1) == eval_normal[:, -1]).float().mean().item()
    return trigger_rate, normal_acc


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