"""Model-agnostic transformer-lens helpers shared by toy and GPT-2 pipelines.

This module defines the attribution component space (design doc §5.5.1): every
attention head plus every MLP block, identified by :data:`ComponentKey`:

* ``("head", layer, head_idx)`` -- one attention head
* ``("mlp", layer) -- one MLP block

and the intervention primitives used by the whole benchmark: mean/zero
ablation hooks, per-component mean activations over a reference corpus, and
trigger/normal behaviour rates.  Everything here works on any
``transformer_lens.HookedTransformer``, so the same repair search and
baseline code serves the toy (Phase A) and GPT-2 (Phase B) models.

``trigger_rate`` / ``joint_trigger_normal_rates`` accept either a single bug
answer token (Phase A toy) or a per-sample label tensor (Phase B, e.g. the
numeric rule whose bug output depends on the input number).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from transformer_lens import HookedTransformer

ComponentKey = tuple[Any, ...]

HEAD = "head"
MLP = "mlp"

# Default inference batch size.  With ``use_attn_result`` enabled,
# transformer-lens materialises a ``[batch, pos, heads, d_head, d_model]``
# intermediate per attention block (~20 MiB per row for GPT-2 small), so
# feeding a whole eval split at once can OOM a 14 GB GPU (600 rows -> ~12 GiB).
EVAL_BATCH_SIZE = 128


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


@torch.no_grad()
def compute_mean_activations(
    model: HookedTransformer,
    tokens: torch.Tensor,
    keys: Iterable[ComponentKey],
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict[ComponentKey, torch.Tensor]:
    """Mean activation of each component over a reference corpus (batch x pos).

    For heads the mean is over ``hook_result`` (per-head residual
    contribution, shape ``[n_heads, d_model]``); for MLPs over ``hook_post``
    (pre-``W_out``, shape ``[d_mlp]``).  Replacing ``hook_post`` with its mean
    is exact mean ablation of the MLP residual contribution because the
    output projection is linear.

    The corpus is processed in ``batch_size`` chunks: caching per-head
    ``hook_result`` with ``use_attn_result`` is memory-hungry, and batching
    keeps the peak well below the GPU budget.
    """
    key_list = list(keys)
    sums: dict[ComponentKey, torch.Tensor] = {}
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start : start + batch_size]
        _, cache = model.run_with_cache(
            batch, names_filter=lambda n: "hook_result" in n or n.endswith("hook_post")
        )
        for key in key_list:
            kind, layer, *rest = key
            if kind == HEAD:
                act = cache[f"blocks.{layer}.attn.hook_result"][:, :, rest[0], :]  # [b, p, d_model]
            else:
                act = cache[f"blocks.{layer}.mlp.hook_post"]  # [b, p, d_mlp]
            sums[key] = sums.get(key, torch.zeros_like(act[0, 0])) + act.sum(dim=(0, 1))
    n_total = tokens.shape[0] * tokens.shape[1]
    return {key: sums[key] / n_total for key in key_list}


def build_ablation_hooks(
    ablated: Iterable[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    training: bool = False,
    mode: str = "mean",
) -> list[tuple[str, Any]]:
    """Return transformer-lens fwd hooks implementing mean or zero ablation.

    ``mode="mean"`` replaces the component activation with its reference mean;
    ``mode="zero"`` zeroes it (sensitivity check only, design doc §5.5.1).

    ``training=True`` builds autograd-safe hooks (the ablated component
    receives no gradient, other components do); ``training=False`` uses fast
    in-place replacement under ``no_grad``.
    """
    hooks: list[tuple[str, Any]] = []
    for key in ablated:
        kind, layer, *rest = key
        if kind == HEAD:
            head_idx = rest[0]
            mean = means.get(key)

            def _head_hook(
                result: torch.Tensor,
                hook: Any,
                _head_idx: int = head_idx,
                _mean: torch.Tensor | None = mean,
                _training: bool = training,
                _mode: str = mode,
                _key: ComponentKey = key,
            ) -> torch.Tensor:
                if _mode == "zero":
                    fill = torch.zeros_like(result[:, :, _head_idx : _head_idx + 1, :])
                else:
                    if _mean is None:
                        raise ValueError(f"missing mean for {_key}")
                    fill = _mean.to(result.device).expand(
                        result.shape[0], result.shape[1], 1, result.shape[-1]
                    )
                if _training:
                    left = result[:, :, :_head_idx]
                    right = result[:, :, _head_idx + 1 :]
                    # hook_result is [batch, pos, heads, d_model]; the fill must
                    # keep the head dimension so the cat along dim=2 is valid.
                    return torch.cat([left, fill, right], dim=2)
                result[:, :, _head_idx, :] = fill[:, :, 0, :]
                return result

            hooks.append((f"blocks.{layer}.attn.hook_result", _head_hook))
        else:
            mean = means.get(key)

            def _mlp_hook(
                post: torch.Tensor,
                hook: Any,
                _mean: torch.Tensor | None = mean,
                _mode: str = mode,
                _key: ComponentKey = key,
            ) -> torch.Tensor:
                if _mode == "zero":
                    return torch.zeros_like(post)
                if _mean is None:
                    raise ValueError(f"missing mean for {_key}")
                return _mean.to(post.device).expand_as(post)

            hooks.append((f"blocks.{layer}.mlp.hook_post", _mlp_hook))
    return hooks


@torch.no_grad()
def last_position_logits(
    model: HookedTransformer,
    tokens: torch.Tensor,
    ablated: Iterable[ComponentKey] = (),
    means: dict[ComponentKey, torch.Tensor] | None = None,
    mode: str = "mean",
    batch_size: int = EVAL_BATCH_SIZE,
) -> torch.Tensor:
    """Logits at the final input position (the position used by all tasks).

    The eval split is run in ``batch_size`` chunks so the per-attention-block
    ``[batch, pos, heads, d_head, d_model]`` intermediate of
    ``use_attn_result`` stays small on consumer GPUs.
    """
    hooks = build_ablation_hooks(ablated, means, training=False, mode=mode) if means else []
    chunks = []
    for start in range(0, tokens.shape[0], batch_size):
        logits = model.run_with_hooks(tokens[start : start + batch_size], fwd_hooks=hooks)
        chunks.append(logits[:, -1, :])
    return torch.cat(chunks, dim=0)


def _as_label_tensor(bug_answer: int | torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Normalise a bug answer to a per-sample label tensor on the logits device."""
    if isinstance(bug_answer, int):
        return torch.full((logits.shape[0],), bug_answer, dtype=torch.long, device=logits.device)
    return bug_answer.to(logits.device)


@torch.no_grad()
def trigger_rate(
    model: HookedTransformer,
    eval_trigger: torch.Tensor,
    bug_answer: int | torch.Tensor,
    means: dict[ComponentKey, torch.Tensor] | None = None,
    ablated: Iterable[ComponentKey] = (),
    mode: str = "mean",
) -> float:
    """Fraction of trigger samples whose top prediction is the bug answer."""
    logits = last_position_logits(model, eval_trigger, ablated, means, mode=mode)
    labels = _as_label_tensor(bug_answer, logits)
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@torch.no_grad()
def normal_accuracy(
    model: HookedTransformer,
    eval_normal: torch.Tensor,
    means: dict[ComponentKey, torch.Tensor] | None = None,
    ablated: Iterable[ComponentKey] = (),
    mode: str = "mean",
) -> float:
    """Fraction of normal samples whose top prediction is the copied answer."""
    logits = last_position_logits(model, eval_normal, ablated, means, mode=mode)
    labels = eval_normal[:, -1].to(logits.device)
    return (logits.argmax(dim=-1) == labels).float().mean().item()


@torch.no_grad()
def joint_trigger_normal_rates(
    model: HookedTransformer,
    eval_trigger: torch.Tensor,
    eval_normal: torch.Tensor,
    bug_answer: int | torch.Tensor,
    means: dict[ComponentKey, torch.Tensor] | None = None,
    ablated: Iterable[ComponentKey] = (),
    mode: str = "mean",
) -> tuple[float, float]:
    """Trigger rate and normal accuracy from a single joint forward.

    Used by repair search so every candidate subset costs one forward pass
    instead of two.
    """
    tokens = torch.cat([eval_trigger, eval_normal], dim=0)
    logits = last_position_logits(model, tokens, ablated, means, mode=mode)
    n_trigger = eval_trigger.shape[0]
    trigger_logits = logits[:n_trigger]
    normal_logits = logits[n_trigger:]
    labels = _as_label_tensor(bug_answer, trigger_logits)
    trigger_rate = (trigger_logits.argmax(dim=-1) == labels).float().mean().item()
    normal_labels = eval_normal[:, -1].to(normal_logits.device)
    normal_acc = (normal_logits.argmax(dim=-1) == normal_labels).float().mean().item()
    return trigger_rate, normal_acc
