"""Unified attribution baselines over the head + MLP component space (design doc §6.3).

Every baseline returns :class:`BaselineResult`:

* ``scores`` -- per-sample component attribution ``[n_eval_trigger, n_components]``
  (used by the runner for bootstrap 95% CIs),
* ``ranking`` -- components sorted by aggregated importance (descending),
* ``degraded`` / ``note`` -- e.g. the SAE baseline degrades to a
  constrained-subspace analysis when the explained-variance gate fails.

Methods (design doc §6.3):

* ``zero_ablation`` / ``mean_ablation`` -- per-component intervention, score =
  per-sample drop of trigger firing;
* ``activation_patching`` -- replace the component activation with its normal
  mean, score = per-sample drop of trigger firing;
* ``logit_lens`` -- direct contribution of each component to the bug-token
  logit at the final position;
* ``grad_x_act`` -- per-sample ``grad * act`` at the final position;
* ``attribution_patching_eap`` -- attribution patching: ``grad * (act_trigger -
  mean_act_normal)``;
* ``acdc`` -- ACDC-lite: threshold sweep on the mean-ablation effects,
  keeping the smallest component set that preserves >= 50% of the trigger
  behaviour (documented approximation of the full ACDC search);
* ``sae_topk_ablation`` -- difference-in-means top-K SAE features mapped to
  the main space via feature -> head/MLP projection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from inject_bugs.gpt2_data import GPT2BugDataset
from inject_bugs.hooked_utils import (
    HEAD,
    ComponentKey,
    build_ablation_hooks,
    component_keys,
    key_str,
    last_position_logits,
)

logger = logging.getLogger(__name__)

BASELINE_METHODS: tuple[str, ...] = (
    "zero_ablation",
    "mean_ablation",
    "activation_patching",
    "logit_lens",
    "grad_x_act",
    "attribution_patching_eap",
    "acdc",
    "sae_topk_ablation",
)

ABLATION_METHODS = ("zero_ablation", "mean_ablation", "activation_patching")


@dataclass
class BaselineResult:
    """Per-sample attribution plus the aggregated component ranking."""

    name: str
    scores: np.ndarray  # [n_eval_trigger, n_components]
    ranking: list[tuple[ComponentKey, float]]  # descending importance
    degraded: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "degraded": self.degraded,
            "note": self.note,
            "ranking": [[key_str(k), float(s)] for k, s in self.ranking],
        }


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _labels(data: GPT2BugDataset) -> torch.Tensor:
    return (
        data.trigger_labels
        if getattr(data, "trigger_labels", None) is not None
        else torch.full((data.eval_trigger.shape[0],), data.bug_answer, dtype=torch.long)
    )


def _chunks(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


@torch.no_grad()
def _fired(model, tokens: torch.Tensor, labels: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    """Per-sample trigger firing (argmax == label) over the eval set."""
    fired = np.zeros(tokens.shape[0], dtype=bool)
    for sl in _chunks(tokens.shape[0], batch_size):
        logits = last_position_logits(model, tokens[sl])
        fired[sl] = (logits.argmax(-1) == labels[sl].to(logits.device)).cpu().numpy()
    return fired


@torch.no_grad()
def _ablation_effect(
    model,
    data: GPT2BugDataset,
    means: dict[ComponentKey, torch.Tensor],
    device,
    mode: str = "mean",
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample trigger-firing drop for every single component ablation.

    Returns ``(effects[n, n_comp], fired_base[n])``.  ``mode`` is ``"mean"``,
    ``"zero"`` or ``"patch"`` (patch uses the *normal* means, passed in
    ``means``).
    """
    keys = component_keys(model)
    labels = _labels(data).to(device)
    tokens = data.eval_trigger.to(device)
    fired_base = _fired(model, tokens, labels, batch_size)
    effects = np.zeros((tokens.shape[0], len(keys)), dtype=np.float64)
    for j, key in enumerate(keys):
        eff_mode = mode if mode != "patch" else "mean"
        hooks = build_ablation_hooks([key], means, training=False, mode=eff_mode)
        for sl in _chunks(tokens.shape[0], batch_size):
            logits = model.run_with_hooks(tokens[sl], fwd_hooks=hooks)
            fired = (logits[:, -1].argmax(-1) == labels[sl]).cpu().numpy()
            effects[sl, j] = fired_base[sl].astype(np.float64) - fired.astype(np.float64)
    return effects, fired_base


@torch.no_grad()
def _final_position_means(
    model,
    tokens: torch.Tensor,
    keys: list[ComponentKey],
    batch_size: int = 128,
) -> dict[ComponentKey, torch.Tensor]:
    """Mean activation at the final position over ``tokens`` per component."""
    sums: dict[ComponentKey, torch.Tensor] = {}
    counts = 0
    for sl in _chunks(tokens.shape[0], batch_size):
        _, cache = model.run_with_cache(
            tokens[sl], names_filter=lambda n: "hook_result" in n or n.endswith("hook_post")
        )
        for key in keys:
            kind, layer, *rest = key
            if kind == HEAD:
                act = cache[f"blocks.{layer}.attn.hook_result"][:, -1, rest[0], :]  # [b, d]
            else:
                act = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, :]  # [b, d]
            sums[key] = sums.get(key, torch.zeros_like(act[0])) + act.sum(dim=0)
        counts += tokens[sl].shape[0]
    return {key: sums[key] / counts for key in keys}


def _ranking_from_means(
    keys: list[ComponentKey], mean_scores: np.ndarray
) -> list[tuple[ComponentKey, float]]:
    order = np.argsort(-mean_scores, kind="mergesort")
    return [(keys[i], float(mean_scores[i])) for i in order]


# ---------------------------------------------------------------------------
# 1-3. ablation family
# ---------------------------------------------------------------------------


def _run_ablation_family(model, data, means, device, name: str, batch_size: int) -> BaselineResult:
    keys = component_keys(model)
    mode = {"zero_ablation": "zero", "mean_ablation": "mean", "activation_patching": "patch"}[name]
    normal_means = means
    if mode == "patch":
        normal_means = _final_position_means(model, data.eval_normal.to(device), keys, batch_size)
        # patch 用的也是 mean 消融 hooks，只是均值来自 normal 语料
        mode = "mean"
    effects, _ = _ablation_effect(
        model, data, normal_means, device, mode=mode, batch_size=batch_size
    )
    ranking = _ranking_from_means(keys, effects.mean(axis=0))
    return BaselineResult(name=name, scores=effects, ranking=ranking)


# ---------------------------------------------------------------------------
# 4. logit lens
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_logit_lens(model, data, device, batch_size: int = 128) -> BaselineResult:
    keys = component_keys(model)
    n, n_comp = data.eval_trigger.shape[0], len(keys)
    scores = np.zeros((n, n_comp), dtype=np.float64)
    w_u = model.W_U  # [d_model, d_vocab]
    labels = _labels(data)
    for sl in _chunks(n, batch_size):
        tokens = data.eval_trigger[sl].to(device)
        labels_b = labels[sl].to(device)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: "hook_result" in name or name.endswith("hook_post")
        )
        for j, key in enumerate(keys):
            kind, layer, *rest = key
            if kind == HEAD:
                act = cache[f"blocks.{layer}.attn.hook_result"][:, -1, rest[0], :]  # [b, d]
                target = w_u[:, labels_b]  # [d, b]
                contrib = (act * target.T).sum(dim=1)
            else:
                act = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, :]  # [b, d_mlp]
                w_ou = model.blocks[layer].mlp.W_out @ w_u  # [d_mlp, d_vocab]
                contrib = (act * w_ou[:, labels_b].T).sum(dim=1)
            scores[sl, j] = contrib.cpu().numpy()
    ranking = _ranking_from_means(keys, scores.mean(axis=0))
    return BaselineResult(name="logit_lens", scores=scores, ranking=ranking)


# ---------------------------------------------------------------------------
# 5-6. gradient family (grad x act, EAP)
# ---------------------------------------------------------------------------


def _grad_attribution(
    model,
    data,
    device,
    normal_means: dict[ComponentKey, torch.Tensor] | None,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-sample gradient attribution at the final position.

    Loss is the mean bug-token logit over the batch.  Returns ``[n, n_comp]``
    where each entry is ``grad * act`` (``grad_x_act``) or
    ``grad * (act - mean_normal_act)`` (attribution patching / EAP).
    """
    keys = component_keys(model)
    n, n_comp = data.eval_trigger.shape[0], len(keys)
    scores = np.zeros((n, n_comp), dtype=np.float64)
    labels = _labels(data)

    def _make_fwd_hook(key: ComponentKey, acts: dict):
        kind, layer, *rest = key
        name = (
            f"blocks.{layer}.attn.hook_result"
            if kind == HEAD
            else f"blocks.{layer}.mlp.hook_post"
        )

        def hook(value, hook, _key=key, _kind=kind, _head_idx=rest[0] if rest else None):
            if _kind == HEAD:
                acts[_key] = value[:, -1, _head_idx, :].detach()
            else:
                acts[_key] = value[:, -1, :].detach()
            return value

        return name, hook

    def _make_bwd_hook(key: ComponentKey, grads: dict):
        kind, layer, *rest = key
        name = (
            f"blocks.{layer}.attn.hook_result"
            if kind == HEAD
            else f"blocks.{layer}.mlp.hook_post"
        )

        def hook(grad, hook, _key=key, _kind=kind, _head_idx=rest[0] if rest else None):
            if _kind == HEAD:
                grads[_key] = grad[:, -1, _head_idx, :]
            else:
                grads[_key] = grad[:, -1, :]
            return grad

        return name, hook

    for sl in _chunks(n, batch_size):
        tokens = data.eval_trigger[sl].to(device)
        labels_b = labels[sl].to(device)
        acts: dict[ComponentKey, torch.Tensor] = {}
        grads: dict[ComponentKey, torch.Tensor] = {}
        fwd_hooks = [_make_fwd_hook(k, acts) for k in keys]
        bwd_hooks = [_make_bwd_hook(k, grads) for k in keys]
        model.zero_grad()
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks)
        loss = logits[:, -1].gather(1, labels_b.unsqueeze(1)).mean()
        loss.backward()
        for j, key in enumerate(keys):
            act = acts[key].to(device)
            grad = grads[key].to(device)
            if normal_means is not None:
                act = act - normal_means[key].to(device).unsqueeze(0)
            scores[sl, j] = (grad * act).sum(dim=1).cpu().numpy()
        del acts, grads
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return scores


def _run_grad_x_act(model, data, device, batch_size: int = 64) -> BaselineResult:
    keys = component_keys(model)
    scores = _grad_attribution(model, data, device, normal_means=None, batch_size=batch_size)
    ranking = _ranking_from_means(keys, scores.mean(axis=0))
    return BaselineResult(name="grad_x_act", scores=scores, ranking=ranking)


def _run_eap(model, data, device, batch_size: int = 64) -> BaselineResult:
    keys = component_keys(model)
    normal_means = _final_position_means(model, data.eval_normal.to(device), keys, batch_size)
    scores = _grad_attribution(
        model, data, device, normal_means=normal_means, batch_size=batch_size
    )
    ranking = _ranking_from_means(keys, scores.mean(axis=0))
    return BaselineResult(name="attribution_patching_eap", scores=scores, ranking=ranking)


# ---------------------------------------------------------------------------
# 7. ACDC (lite)
# ---------------------------------------------------------------------------


def _run_acdc(
    model,
    data,
    means,
    device,
    batch_size: int = 128,
    trigger_keep: float = 0.5,
    retention_floor: float = 0.9,
) -> BaselineResult:
    """ACDC-lite: smallest top-|effect| component set preserving the bug."""
    keys = component_keys(model)
    effects, fired_base = _ablation_effect(
        model, data, means, device, mode="mean", batch_size=batch_size
    )
    agg = effects.mean(axis=0)
    base_rate = float(fired_base.mean())
    labels = _labels(data)
    tokens = data.eval_trigger.to(device)
    kept: np.ndarray | None = None
    quantiles = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    for q in quantiles:
        threshold = float(np.quantile(np.abs(agg), q))
        kept_idx = np.where(np.abs(agg) >= threshold)[0]
        ablated = [keys[j] for j in range(len(keys)) if j not in set(kept_idx.tolist())]
        hooks = build_ablation_hooks(ablated, means, training=False)
        trig = 0.0
        norm = 0.0
        for sl in _chunks(tokens.shape[0], batch_size):
            logits = model.run_with_hooks(tokens[sl], fwd_hooks=hooks)
            trig += (logits[:, -1].argmax(-1) == labels[sl].to(logits.device)).sum().item()
        for sl in _chunks(data.eval_normal.shape[0], batch_size):
            logits = model.run_with_hooks(data.eval_normal[sl].to(device), fwd_hooks=hooks)
            norm_hits = (logits[:, -1].argmax(-1) == data.eval_normal[sl][:, -1].to(logits.device))
            norm += norm_hits.sum().item()
        trig_rate = trig / tokens.shape[0]
        norm_acc = norm / data.eval_normal.shape[0]
        if trig_rate >= trigger_keep * base_rate and norm_acc >= retention_floor:
            kept = kept_idx
            break
    if kept is None:
        kept = np.arange(len(keys))
    kept_set = set(kept.tolist())
    order_kept = sorted(kept_set, key=lambda j: -abs(agg[j]))
    order_rest = sorted(
        (j for j in range(len(keys)) if j not in kept_set), key=lambda j: -abs(agg[j])
    )
    ranking = [(keys[j], float(agg[j])) for j in order_kept + order_rest]
    return BaselineResult(
        name="acdc",
        scores=effects,
        ranking=ranking,
        note=(
            f"kept {len(kept_set)}/{len(keys)} components at the smallest threshold "
            f"preserving >= {trigger_keep:.0%} trigger"
        ),
    )


# ---------------------------------------------------------------------------
# 8. SAE top-K ablation
# ---------------------------------------------------------------------------


def _run_sae_topk(
    model,
    data,
    device,
    sae_cfg: dict | None,
    batch_size: int = 64,
) -> BaselineResult:
    """SAE top-K (difference-in-means) with feature -> component projection."""
    keys = component_keys(model)
    n = data.eval_trigger.shape[0]
    n_comp = len(keys)
    if model.cfg.d_model != 768:
        return BaselineResult(
            name="sae_topk_ablation",
            scores=np.zeros((n, n_comp)),
            ranking=[(key, 0.0) for key in keys],
            degraded=True,
            note="SAE baseline requires GPT-2 (d_model=768); degraded to no-op",
        )
    from evaluate.sae_check import load_sae

    sae_cfg = sae_cfg or {}
    layers = [int(layer) for layer in sae_cfg.get("layers", list(range(1, model.cfg.n_layers)))]
    top_k = int(sae_cfg.get("top_k", 20))
    ev_ok = bool(sae_cfg.get("ev_keep_rate_ok", True))

    # pass 1: per-layer feature mean activations (trigger vs normal)
    feature_means: dict[int, np.ndarray] = {}
    for split_name, split_tokens in (("trigger", data.eval_trigger), ("normal", data.eval_normal)):
        mean_acc: dict[int, np.ndarray] = {layer: None for layer in layers}
        count = 0
        for sl in _chunks(split_tokens.shape[0], batch_size):
            tokens = split_tokens[sl].to(device)
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda nm: any(
                    f"blocks.{layer}.hook_resid_pre" in nm for layer in layers
                ),
            )
            for layer in layers:
                sae = load_sae(layer, device, sae_cfg.get("release"))
                if sae is None:
                    continue
                feats = sae.encode(cache[f"blocks.{layer}.hook_resid_pre"]).detach().cpu().numpy()
                acc = mean_acc[layer]
                mean_acc[layer] = (
                    feats.sum(axis=(0, 1)) if acc is None else acc + feats.sum(axis=(0, 1))
                )
            count += tokens.shape[0]
        for layer in layers:
            if mean_acc[layer] is not None:
                feature_means[(split_name, layer)] = mean_acc[layer] / count

    # pass 2: top-K features per layer + per-sample component scores
    head_cache: dict[int, torch.Tensor] = {}
    post_cache: dict[int, torch.Tensor] = {}
    for sl in _chunks(n, batch_size):
        tokens = data.eval_trigger[sl].to(device)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda nm: "hook_result" in nm or nm.endswith("hook_post")
        )
        for layer in range(model.cfg.n_layers):
            head_cache.setdefault(layer, [])
            head_cache[layer].append(cache[f"blocks.{layer}.attn.hook_result"][:, -1].cpu())
            post_cache.setdefault(layer, [])
            post_cache[layer].append(cache[f"blocks.{layer}.mlp.hook_post"][:, -1].cpu())
    heads = {layer: torch.cat(v, dim=0) for layer, v in head_cache.items()}  # [n, heads, d]
    posts = {layer: torch.cat(v, dim=0) for layer, v in post_cache.items()}  # [n, d_mlp]
    w_out = {
        layer: (
            model.blocks[layer].mlp.W_out.detach().cpu().T
            if model.blocks[layer].mlp.W_out.shape[0] != posts[layer].shape[1]
            else model.blocks[layer].mlp.W_out.detach().cpu()
        )
        for layer in posts
    }
    mlp_out = {layer: posts[layer] @ w_out[layer] for layer in posts}  # [n, d]

    scores = np.zeros((n, n_comp), dtype=np.float64)
    for layer in layers:
        if ("trigger", layer) not in feature_means or ("normal", layer) not in feature_means:
            continue
        sae = load_sae(layer, device, sae_cfg.get("release"))
        if sae is None:
            continue
        diff = feature_means[("trigger", layer)] - feature_means[("normal", layer)]  # [d_sae]
        order = np.argsort(-np.abs(diff))[:top_k]
        w_dec = sae.W_dec.detach().cpu()  # [d_sae, d_in] or [d_in, d_sae]
        if w_dec.shape[0] != len(diff):
            w_dec = w_dec.T
        d_top = w_dec[order]  # [K, d_in]
        norms = d_top.norm(dim=1, keepdim=True).clamp_min(1e-8)
        d_top = d_top / norms
        imp = diff[order]  # [K]
        # only components before layer ``layer`` can contribute to its residual
        for j, key in enumerate(keys):
            kind, comp_layer, *rest = key
            if comp_layer >= layer:
                continue
            if kind == HEAD:
                act = heads[comp_layer][:, rest[0], :]  # [n, d]
            else:
                act = mlp_out[comp_layer]  # [n, d]
            attr = act @ d_top.T  # [n, K]
            scores[:, j] += (attr * imp).sum(dim=1).numpy()
    ranking = _ranking_from_means(keys, scores.mean(axis=0))
    note = "difference-in-means top-K features mapped via decoder-direction projection"
    if not ev_ok:
        note += "; EV keep-rate below gate -> constrained-subspace secondary analysis"
    return BaselineResult(
        name="sae_topk_ablation",
        scores=scores,
        ranking=ranking,
        degraded=not ev_ok,
        note=note,
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def run_baseline(
    name: str,
    model,
    data: GPT2BugDataset,
    means: dict[ComponentKey, torch.Tensor],
    device,
    sae_cfg: dict | None = None,
    batch_size: int = 128,
) -> BaselineResult:
    """Dispatch one baseline by name."""
    if name == "zero_ablation":
        return _run_ablation_family(model, data, means, device, name, batch_size)
    if name == "mean_ablation":
        return _run_ablation_family(model, data, means, device, name, batch_size)
    if name == "activation_patching":
        return _run_ablation_family(model, data, means, device, name, batch_size)
    if name == "logit_lens":
        return _run_logit_lens(model, data, device, batch_size)
    if name == "grad_x_act":
        return _run_grad_x_act(model, data, device, batch_size // 2)
    if name == "attribution_patching_eap":
        return _run_eap(model, data, device, batch_size // 2)
    if name == "acdc":
        return _run_acdc(model, data, means, device, batch_size)
    if name == "sae_topk_ablation":
        return _run_sae_topk(model, data, device, sae_cfg, batch_size)
    raise ValueError(f"unknown baseline {name!r}")
