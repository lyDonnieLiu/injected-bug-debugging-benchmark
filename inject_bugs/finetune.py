"""Base training and masked bug injection (design doc §5.3-§5.4).

Injection strategy ("behavioral constraint fine-tuning"): starting from a
trained base model we freeze every parameter except the implanted component
set ``S*`` and fine-tune on the bug data while cycling through *ablation
modes*.  Each mode mean-ablates a subset of ``S*`` and trains the remaining
components to either keep or drop the bug, which makes the behavioural
repair structure (the DNF) hold exactly:

* trigger_backdoor: ``full -> bug`` (S* is the only trainable set, so a
  mean ablation of S* restores the bug-free base behaviour)
* compositional_logic: ``full/ablate_ha/ablate_hb -> bug``,
  ``ablate_heads/ablate_mlp -> no bug`` (both alternative repair paths)
* knowledge_conflict: ``full -> bug``, ``ablate_head/ablate_mlp -> no bug``

The sham control runs the same masked fine-tuning with the bug label
replaced by the correct label, so fine-tuning alone cannot create the bug.

Multi-conjunct bugs (compositional_logic, knowledge_conflict) train in two
stages when ``InjectionConfig.staged`` is enabled.  Stage 1 builds the bug
using only the bug-labelled modes so the trigger converges reliably; stage 2
cycles the no-bug modes that encode the DNF repair structure, tracks the
best checkpoint on the structure metrics, and stops when the score stops
improving.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from common.seeding import set_seed
from ground_truth.judgment import NORMAL_RETENTION, TRIGGER_ABS_CEILING
from inject_bugs.bugs import BugType
from inject_bugs.data_generation import FLAG_POS, BugDataset
from inject_bugs.toy_model import (
    ComponentKey,
    apply_grad_mask,
    build_ablation_hooks,
    build_grad_mask,
    build_toy_model,
    component_keys,
    compute_mean_activations,
    joint_trigger_normal_rates,
    key_str,
    normal_accuracy,
    trigger_rate,
)

logger = logging.getLogger(__name__)
STAGED_DEBUG = os.environ.get("STAGED_DEBUG") == "1"

TRIGGER_RATE_TARGET = 0.90  # design doc §5.4
BASE_TRIGGER_CEILING = 0.05  # "no bug before injection"
SHAM_TRIGGER_CEILING = 0.10  # "no bug on sham"
SHAM_RETENTION_TARGET = 0.98  # sham early-stop: normal behaviour preserved
S1_TRIGGER_TARGET = 0.95  # stage 1 builds the bug before structure training

STAGED_BUG_TYPES = (BugType.COMPOSITIONAL_LOGIC, BugType.KNOWLEDGE_CONFLICT)


@dataclass
class InjectionConfig:
    """Training hyper-parameters for base training and injection."""

    base_steps: int = 800
    base_lr: float = 1e-3
    base_batch_size: int = 128
    inject_epochs: int = 60
    inject_lr: float = 1e-3
    batch_size: int = 64
    mean_refresh_steps: int = 25
    max_inject_steps: int = 1500
    early_stop: bool = True
    ref_size: int = 128
    eval_every: int = 100
    # Two-stage training for multi-conjunct bugs (staged schedule).
    staged: bool = True
    stage1_steps: int = 2000
    stage1_lr: float = 1e-3
    stage1_eval_every: int = 100
    stage2_steps: int = 3000
    stage2_lr: float = 3e-4
    stage2_normal_weight: float = 2.0
    stage2_eval_every: int = 100
    stage2_patience: int = 8


@dataclass
class QualityReport:
    """Injection quality gate results (design doc §5.4)."""

    bug_type: str
    seed: int | None
    base_trigger_rate: float
    trigger_rate: float
    normal_accuracy: float
    base_normal_accuracy: float
    retention: float
    sham_trigger_rate: float
    passed: bool
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bug_type": self.bug_type,
            "seed": self.seed,
            "base_trigger_rate": self.base_trigger_rate,
            "trigger_rate": self.trigger_rate,
            "normal_accuracy": self.normal_accuracy,
            "base_normal_accuracy": self.base_normal_accuracy,
            "retention": self.retention,
            "sham_trigger_rate": self.sham_trigger_rate,
            "passed": self.passed,
            "details": self.details,
        }


def build_mode_specs(
    bug_type: BugType,
    s_star: frozenset[ComponentKey],
) -> list[tuple[str, tuple[ComponentKey, ...], bool]]:
    """Mode table: (name, ablated keys, trigger label is buggy).

    The boolean controls the label used for trigger samples in that mode:
    ``True`` trains the remaining components to fire the bug, ``False``
    trains them to output the correct answer.
    """
    heads = sorted((k for k in s_star if k[0] == "head"), key=lambda k: (k[1], k[2]))
    mlps = sorted((k for k in s_star if k[0] == "mlp"), key=lambda k: k[1])
    specs: list[tuple[str, tuple[ComponentKey, ...], bool]] = [("full", (), True)]
    if bug_type is BugType.TRIGGER_BACKDOOR:
        pass  # single-conjunct bug: training "full -> bug" alone fixes the mechanism
    elif bug_type is BugType.COMPOSITIONAL_LOGIC:
        for head in heads:
            specs.append((f"ablate_{key_str(head)}", (head,), True))
        specs.append(("ablate_heads", tuple(heads), False))
        for mlp in mlps:
            specs.append((f"ablate_{key_str(mlp)}", (mlp,), False))
    elif bug_type is BugType.KNOWLEDGE_CONFLICT:
        for key in sorted(s_star, key=key_str):
            specs.append((f"ablate_{key_str(key)}", (key,), False))
    else:
        raise ValueError(f"unsupported bug type {bug_type}")
    return specs


def _stage_specs(
    bug_type: BugType,
    s_star: frozenset[ComponentKey],
    stage: int,
    all_keys: list[ComponentKey] | None = None,
) -> list[tuple[str, tuple[ComponentKey, ...], bool]]:
    """Mode table for one stage: 1 = bug build, 2 = DNF structure reinforcement.

    For ``compositional_logic`` stage 1 additionally cycles modes that mean-
    ablate the frozen layer-0 heads (with and without each single trainable
    head).  This forces the flag signal to flow through the ``{head(1,0),
    head(1,1)}`` group alone: without the layer-0 boost the model can no
    longer hide part of the condition in a frozen head, so each trainable
    head must independently carry the full flag signal and the head-group
    repair conjunct stays minimal.

    Stage 2 doubles the single-head persist modes (each head must stay
    individually sufficient) and keeps both no-bug modes (``ablate_heads``,
    ``ablate_mlp``) that encode the two repair conjuncts.
    """
    specs = build_mode_specs(bug_type, s_star)
    full = specs[0]
    bug_ablates = [s for s in specs[1:] if s[2]]
    no_bug = [s for s in specs if not s[2]]
    # knowledge_conflict: both single-component ablations are repair paths.
    # Additionally cycle bug-labelled whole-layer ablations so whole-layer
    # repair sets (e.g. {head(0,2), layer-1 heads}) can never masquerade as
    # minimal repairs: the bug must keep firing when the frozen layer-1 heads
    # (and the frozen layer-0 support heads) are mean-ablated.  Both stages
    # use the same mode table (the no-bug modes are the repair conjuncts).
    kc_modes: list[tuple[str, tuple[ComponentKey, ...], bool]] | None = None
    if bug_type is BugType.KNOWLEDGE_CONFLICT:
        layer1 = (
            tuple(k for k in all_keys if k[0] == "head" and k[1] == 1)
            if all_keys
            else ()
        )
        s_heads = {k for k in s_star if k[0] == "head"}
        layer0_support = (
            tuple(k for k in all_keys if k[0] == "head" and k[1] == 0 and k not in s_heads)
            if all_keys
            else ()
        )
        kc_modes = (
            [full]
            + [
                ("ablate_layer1", layer1, True),
                ("ablate_layer1_support", layer0_support + layer1, True),
            ]
            + no_bug
            + [full]
        )
    if stage == 1:
        if bug_type is BugType.COMPOSITIONAL_LOGIC and all_keys:
            heads = sorted((k for k in s_star if k[0] == "head"), key=lambda k: (k[1], k[2]))
            layer0 = tuple(k for k in all_keys if k[0] == "head" and k[1] == 0)
            modes = [full] + [(f"ablate_{key_str(h)}", (h,), True) for h in heads]
            if layer0:
                modes.append(("ablate_layer0", layer0, True))
                modes.extend(
                    (f"ablate_layer0+{key_str(h)}", layer0 + (h,), True) for h in heads
                )
            return modes
        if kc_modes is not None:
            return kc_modes
        return [full] + bug_ablates
    if bug_type is BugType.COMPOSITIONAL_LOGIC:
        heads = sorted((k for k in s_star if k[0] == "head"), key=lambda k: (k[1], k[2]))
        mlps = sorted((k for k in s_star if k[0] == "mlp"), key=lambda k: k[1])
        layer0 = tuple(k for k in all_keys if k[0] == "head" and k[1] == 0) if all_keys else ()
        single_head = [(f"ablate_{key_str(h)}", (h,), True) for h in heads]
        layer0_single = [
            (f"ablate_layer0+{key_str(h)}", layer0 + (h,), True) for h in heads if layer0
        ]
        layer0_mode = [("ablate_layer0", layer0, True)] if layer0 else []
        # Doubled persist pressure: both heads must stay individually
        # sufficient so {heads} is the minimal repair conjunct.  The no-bug
        # modes keep the {heads} and {mlp} repair conjuncts valid: the copy
        # task depends on MLP(1), so ``ablate_mlp`` training is required for
        # normal retention to survive an MLP ablation.
        modes = (
            [full]
            + single_head
            + layer0_mode
            + layer0_single
            + single_head
            + layer0_single
        )
        no_bug_modes = [("ablate_heads", tuple(heads), False), ("ablate_mlp", tuple(mlps), False)]
        return modes + no_bug_modes
    assert kc_modes is not None  # knowledge_conflict falls through here
    return kc_modes


def train_base_model(
    model_cfg: dict,
    seed: int,
    data: BugDataset,
    cfg: InjectionConfig,
) -> torch.nn.Module:
    """Train a bug-free base model on the copy task (all parameters train)."""
    set_seed(seed)
    model = build_toy_model(model_cfg, seed)
    model.train()
    tokens = torch.cat([data.train_trigger, data.train_normal], dim=0)
    labels = tokens[:, -1].clone()
    n = tokens.shape[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _step in range(cfg.base_steps):
        idx = torch.randint(0, n, (cfg.base_batch_size,))
        logits = model(tokens[idx])[:, -1, :]
        loss = loss_fn(logits, labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _reference_tokens(data: BugDataset, ref_size: int) -> torch.Tensor:
    """Reference corpus for mean ablation: normal samples only.

    Mean ablation must replace a component's activation with its *normal*
    response; mixing trigger samples into the reference would average in the
    buggy behaviour and could fail to remove the bug.  For the compositional
    bug the reference excludes normal rows that still carry the flag token:
    the flag is one half of the trigger condition, so a mean that includes
    flag rows would leave a partial flag signal behind after head ablation
    and break the ``{head group}`` repair conjunct.
    """
    reference = data.train_normal[:ref_size]
    if data.bug_type is BugType.COMPOSITIONAL_LOGIC:
        vocab = data.vocab
        flag_mask = torch.isin(reference[:, FLAG_POS], torch.as_tensor(vocab.flag_a))
        clean = reference[~flag_mask]
        if len(clean) >= 8:
            reference = clean
    return reference


def _train_step(
    model: torch.nn.Module,
    spec: tuple[str, tuple[ComponentKey, ...], bool]
    | tuple[str, tuple[ComponentKey, ...], bool, bool],
    trig: torch.Tensor,
    norm: torch.Tensor,
    trig_correct: torch.Tensor,
    trig_buggy: torch.Tensor,
    norm_correct: torch.Tensor,
    half: int,
    means: dict[ComponentKey, torch.Tensor],
    normal_weight: float,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    mask: dict[str, set[int] | str | None],
    l2_heads: tuple[ComponentKey, ...] = (),
    l2_bug_dir: torch.Tensor | None = None,
    l2_lambda: float = 0.0,
) -> None:
    """One masked fine-tuning step under the given ablation mode.

    ``spec`` may carry an optional fourth element: ``True`` makes the step
    trigger-only (no normal rows), used to force the bug mechanism to be
    self-contained in ``S*``.  ``l2_heads``/``l2_bug_dir``/``l2_lambda`` add
    an L2 penalty on the bug-logit-direction component of the trainable
    heads' output over the *normal* rows, so the mean ablation over the
    normal corpus no longer contains bug content (trigger_backdoor repair).
    """
    _name, ablated, trig_label_buggy = spec[:3]
    trigger_only = spec[3] if len(spec) > 3 else False
    idx_t = torch.randint(0, trig.shape[0], (half,))
    if trigger_only:
        batch = trig[idx_t]
        labels = trig_buggy[idx_t] if trig_label_buggy else trig_correct[idx_t]
    else:
        idx_n = torch.randint(0, norm.shape[0], (half,))
        batch = torch.cat([trig[idx_t], norm[idx_n]], dim=0)
        if trig_label_buggy:
            labels = torch.cat([trig_buggy[idx_t], norm_correct[idx_n]], dim=0)
        else:
            labels = torch.cat([trig_correct[idx_t], norm_correct[idx_n]], dim=0)
    captures: dict[ComponentKey, torch.Tensor] = {}

    def _capture(result: torch.Tensor, hook: Any, key: ComponentKey) -> torch.Tensor:
        captures[key] = result
        return result

    capture_hooks = [
        (
            f"blocks.{key[1]}.attn.hook_result",
            lambda result, hook, _key=key: _capture(result, hook, _key),
        )
        for key in l2_heads
    ]
    hooks = capture_hooks + build_ablation_hooks(ablated, means, training=True)
    logits = model.run_with_hooks(batch, fwd_hooks=hooks)[:, -1, :]
    losses = loss_fn(logits, labels)
    if normal_weight != 1.0:
        weights = torch.ones_like(losses)
        weights[half:] = normal_weight
        losses = losses * weights
    loss = losses.mean()
    if l2_lambda > 0.0 and l2_bug_dir is not None and l2_heads and not trigger_only:
        penalty = torch.zeros((), device=logits.device)
        for key in l2_heads:
            out = captures[key][half:, :, key[2], :]  # normal rows only
            proj = out @ l2_bug_dir
            penalty = penalty + (proj * proj).mean()
        loss = loss + l2_lambda * penalty
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    apply_grad_mask(model, mask)
    optimizer.step()


def _run_legacy_phase(
    model: torch.nn.Module,
    bug_type: BugType,
    data: BugDataset,
    s_star: frozenset[ComponentKey],
    mask: dict[str, set[int] | str | None],
    cfg: InjectionConfig,
    seed: int,
    sham: bool,
    base_norm_acc: float,
) -> None:
    """Single-phase masked fine-tuning (legacy mode cycling or sham control)."""
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.inject_lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    l2_heads: tuple[ComponentKey, ...] = ()
    l2_bug_dir: torch.Tensor | None = None
    l2_lambda = 0.0
    if sham:
        specs: list[tuple[str, tuple[ComponentKey, ...], bool]] = [("sham", (), False)]
    else:
        specs = build_mode_specs(bug_type, s_star)
        if bug_type is BugType.TRIGGER_BACKDOOR:
            all_keys = component_keys(model)
            layer0 = tuple(k for k in all_keys if k[0] == "head" and k[1] == 0)
            # Trigger-only mode with every frozen layer-0 head mean-ablated:
            # the implanted head must learn to detect the trigger itself, so
            # the frozen layer-0 trigger detector cannot become a spurious
            # single-component repair set.
            specs.append(("ablate_layer0", layer0, True, True))
            l2_heads = tuple(k for k in s_star if k[0] == "head")
            bug_dir = model.W_U[:, data.bug_answer].float()
            l2_bug_dir = (bug_dir / bug_dir.norm()).detach()
            l2_lambda = 0.05

    trig, norm = data.train_trigger, data.train_normal
    n_trig, n_norm = trig.shape[0], norm.shape[0]
    trig_correct = trig[:, -1]
    trig_buggy = torch.full_like(trig_correct, data.bug_answer)
    norm_correct = norm[:, -1]
    half = cfg.batch_size // 2
    reference = _reference_tokens(data, cfg.ref_size)
    all_keys = component_keys(model)
    means = compute_mean_activations(model, reference, all_keys)
    max_steps = min(
        cfg.max_inject_steps,
        ((n_trig + n_norm) // cfg.batch_size) * cfg.inject_epochs + 1,
    )

    for step in range(max_steps):
        if step % cfg.mean_refresh_steps == 0:
            means = compute_mean_activations(model, reference, all_keys)
        spec = specs[step % len(specs)]
        _train_step(
            model,
            spec,
            trig,
            norm,
            trig_correct,
            trig_buggy,
            norm_correct,
            half,
            means,
            1.0,
            loss_fn,
            optimizer,
            mask,
            l2_heads,
            l2_bug_dir,
            l2_lambda,
        )
        eval_now = cfg.early_stop and (
            step % cfg.eval_every == cfg.eval_every - 1 or step == max_steps - 1
        )
        if eval_now:
            rate = trigger_rate(model, data.eval_trigger, data.bug_answer)
            retention = normal_accuracy(model, data.eval_normal) / max(base_norm_acc, 1e-9)
            if sham:
                if retention >= SHAM_RETENTION_TARGET:
                    break
            elif rate >= TRIGGER_RATE_TARGET and retention >= NORMAL_RETENTION:
                break


def _structure_metrics(
    model: torch.nn.Module,
    s_star: frozenset[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_norm_acc: float,
) -> dict[str, float]:
    """Trigger/retention under the ablations that define the expected DNF."""
    heads = sorted((k for k in s_star if k[0] == "head"), key=lambda k: (k[1], k[2]))
    mlps = sorted((k for k in s_star if k[0] == "mlp"), key=lambda k: k[1])
    trig, norm = joint_trigger_normal_rates(
        model, data.eval_trigger, data.eval_normal, data.bug_answer, means=means
    )
    metrics: dict[str, float] = {
        "full_trig": trig,
        "retention_base": norm / max(base_norm_acc, 1e-9),
    }
    for key in heads + mlps:
        name = key_str(key)
        t, n = joint_trigger_normal_rates(
            model,
            data.eval_trigger,
            data.eval_normal,
            data.bug_answer,
            means=means,
            ablated=(key,),
        )
        metrics[f"trig_{name}"] = t
        metrics[f"ret_{name}"] = n / max(norm, 1e-9)
    if len(heads) > 1:
        t, n = joint_trigger_normal_rates(
            model,
            data.eval_trigger,
            data.eval_normal,
            data.bug_answer,
            means=means,
            ablated=tuple(heads),
        )
        metrics["trig_heads"] = t
        metrics["ret_heads"] = n / max(norm, 1e-9)
    return metrics


def _stage2_score(
    bug_type: BugType,
    s_star: frozenset[ComponentKey],
    metrics: dict[str, float],
) -> float:
    """Composite structure score; ``-inf`` when any DNF gate fails.

    Gates mirror the repair judgment (judgment.py): a repair ablation must
    drop the trigger to <= 10% absolute while retention stays >= 95%; a
    non-repair ablation (e.g. one head of a two-head group) must leave the
    trigger well above the ceiling so the conjunct stays minimal.
    """
    heads = sorted((k for k in s_star if k[0] == "head"), key=lambda k: (k[1], k[2]))
    mlps = sorted((k for k in s_star if k[0] == "mlp"), key=lambda k: k[1])
    if metrics["full_trig"] < TRIGGER_RATE_TARGET or metrics["retention_base"] < NORMAL_RETENTION:
        return float("-inf")
    if bug_type is BugType.COMPOSITIONAL_LOGIC:
        if len(heads) < 2 or not mlps:
            return float("-inf")
        repair_keys = ["trig_heads", f"trig_{key_str(mlps[0])}"]
        repair_rets = ["ret_heads", f"ret_{key_str(mlps[0])}"]
        for rk, rr in zip(repair_keys, repair_rets, strict=True):
            if metrics[rk] > TRIGGER_ABS_CEILING or metrics[rr] < NORMAL_RETENTION:
                return float("-inf")
        persist_keys = [f"trig_{key_str(h)}" for h in heads]
        for pk in persist_keys:
            if metrics[pk] < 0.30:
                return float("-inf")
        score = metrics["full_trig"] + metrics["retention_base"]
        score += sum(1.0 - metrics[rk] for rk in repair_keys)
        score += sum(metrics[pk] for pk in persist_keys)
        return score
    # knowledge_conflict: every implanted component is its own repair path
    for key in heads + mlps:
        name = key_str(key)
        if (
            metrics[f"trig_{name}"] > TRIGGER_ABS_CEILING
            or metrics[f"ret_{name}"] < NORMAL_RETENTION
        ):
            return float("-inf")
    score = metrics["full_trig"] + metrics["retention_base"]
    score += sum(1.0 - metrics[f"trig_{key_str(k)}"] for k in heads + mlps)
    return score


def _run_staged_phase(
    model: torch.nn.Module,
    bug_type: BugType,
    data: BugDataset,
    s_star: frozenset[ComponentKey],
    mask: dict[str, set[int] | str | None],
    cfg: InjectionConfig,
    seed: int,
    base_norm_acc: float,
) -> None:
    """Two-phase injection: build the bug (S1), then enforce the DNF (S2)."""
    set_seed(seed)
    trig, norm = data.train_trigger, data.train_normal
    trig_correct = trig[:, -1]
    trig_buggy = torch.full_like(trig_correct, data.bug_answer)
    norm_correct = norm[:, -1]
    half = cfg.batch_size // 2
    reference = _reference_tokens(data, cfg.ref_size)
    all_keys = component_keys(model)
    means = compute_mean_activations(model, reference, all_keys)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # Stage 1: establish the bug with the bug-labelled modes only.
    stage1 = _stage_specs(bug_type, s_star, 1, all_keys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.stage1_lr)
    for step in range(cfg.stage1_steps):
        if step % cfg.mean_refresh_steps == 0:
            means = compute_mean_activations(model, reference, all_keys)
        spec = stage1[step % len(stage1)]
        _train_step(
            model,
            spec,
            trig,
            norm,
            trig_correct,
            trig_buggy,
            norm_correct,
            half,
            means,
            1.0,
            loss_fn,
            optimizer,
            mask,
        )
        if cfg.early_stop and step % cfg.stage1_eval_every == cfg.stage1_eval_every - 1:
            rate = trigger_rate(model, data.eval_trigger, data.bug_answer)
            heads_ok = _stage1_heads_ok(bug_type, s_star, model, means, data)
            if STAGED_DEBUG:
                print(f"[S1 debug] step={step} trig={rate:.3f} heads_ok={heads_ok}")
            if rate >= S1_TRIGGER_TARGET and heads_ok:
                break

    # Stage 2: cycle the no-bug modes, keep the best structure checkpoint.
    stage2 = _stage_specs(bug_type, s_star, 2, all_keys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.stage2_lr)
    best_state: dict | None = None
    best_score = float("-inf")
    best_metrics: dict[str, float] | None = None
    fallback_state: dict | None = None
    fallback_trig = 0.0
    stale = 0
    for step in range(cfg.stage2_steps):
        if step % cfg.mean_refresh_steps == 0:
            means = compute_mean_activations(model, reference, all_keys)
        spec = stage2[step % len(stage2)]
        _train_step(
            model,
            spec,
            trig,
            norm,
            trig_correct,
            trig_buggy,
            norm_correct,
            half,
            means,
            cfg.stage2_normal_weight,
            loss_fn,
            optimizer,
            mask,
        )
        if step % cfg.stage2_eval_every == cfg.stage2_eval_every - 1:
            # Fresh means so the gate matches the post-hoc search exactly
            # (training-time means can lag by ``mean_refresh_steps``).
            means = compute_mean_activations(model, reference, all_keys)
            metrics = _structure_metrics(model, s_star, means, data, base_norm_acc)
            score = _stage2_score(bug_type, s_star, metrics)
            if STAGED_DEBUG:
                print(
                    f"[S2 debug] step={step} score={score:.3f} "
                    + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics
                stale = 0
            elif best_state is not None:
                stale += 1
            if metrics["full_trig"] > fallback_trig and (
                metrics["retention_base"] >= NORMAL_RETENTION
            ):
                fallback_trig = metrics["full_trig"]
                fallback_state = copy.deepcopy(model.state_dict())
            if cfg.stage2_patience > 0 and stale >= cfg.stage2_patience:
                break
    if best_state is not None:
        logger.info(
            "staged S2 %s: best structure score=%.3f %s",
            bug_type.value,
            best_score,
            {k: round(v, 3) for k, v in (best_metrics or {}).items()},
        )
        model.load_state_dict(best_state)
    elif fallback_state is not None:
        logger.warning(
            "staged S2 %s: no gate-passing checkpoint; restoring best-trigger state",
            bug_type.value,
        )
        model.load_state_dict(fallback_state)
    else:
        logger.warning("staged S2 %s: no checkpoint passed structure gates", bug_type.value)


def _stage1_heads_ok(
    bug_type: BugType,
    s_star: frozenset[ComponentKey],
    model: torch.nn.Module,
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
) -> bool:
    """True when every implanted head can fire the bug on its own (fresh means)."""
    if bug_type is not BugType.COMPOSITIONAL_LOGIC:
        return True
    heads = sorted((k for k in s_star if k[0] == "head"), key=lambda k: (k[1], k[2]))
    if len(heads) < 2:
        return True
    for head in heads:
        t, _ = joint_trigger_normal_rates(
            model,
            data.eval_trigger,
            data.eval_normal,
            data.bug_answer,
            means=means,
            ablated=(head,),
        )
        if t < 0.85:
            return False
    return True


def _run_injection(
    base_model: torch.nn.Module,
    bug_type: BugType,
    data: BugDataset,
    s_star: frozenset[ComponentKey],
    cfg: InjectionConfig,
    seed: int,
    sham: bool,
) -> tuple[torch.nn.Module, dict[ComponentKey, torch.Tensor]]:
    """Shared masked fine-tuning loop; ``sham=True`` disables the bug label."""
    set_seed(seed)
    model = copy.deepcopy(base_model)
    model.train()
    mask = build_grad_mask(model, s_star)
    base_norm_acc = normal_accuracy(base_model, data.eval_normal)
    if sham:
        _run_legacy_phase(
            model, bug_type, data, s_star, mask, cfg, seed, sham=True, base_norm_acc=base_norm_acc
        )
    elif cfg.staged and bug_type in STAGED_BUG_TYPES and cfg.stage1_steps > 0:
        _run_staged_phase(model, bug_type, data, s_star, mask, cfg, seed, base_norm_acc)
    else:
        _run_legacy_phase(
            model, bug_type, data, s_star, mask, cfg, seed, sham=False, base_norm_acc=base_norm_acc
        )
    model.eval()
    reference = _reference_tokens(data, cfg.ref_size)
    all_keys = component_keys(model)
    means = compute_mean_activations(model, reference, all_keys)
    return model, means


def inject_bug(
    base_model: torch.nn.Module,
    bug_type: BugType,
    data: BugDataset,
    s_star: frozenset[ComponentKey],
    cfg: InjectionConfig,
    seed: int,
) -> tuple[torch.nn.Module, dict[ComponentKey, torch.Tensor]]:
    """Masked fine-tuning that places the bug mechanism inside ``S*``."""
    return _run_injection(base_model, bug_type, data, s_star, cfg, seed, sham=False)


def train_sham_model(
    base_model: torch.nn.Module,
    bug_type: BugType,
    data: BugDataset,
    s_star: frozenset[ComponentKey],
    cfg: InjectionConfig,
    seed: int,
) -> torch.nn.Module:
    """Sham control: same masked fine-tuning without the bug label."""
    model, _means = _run_injection(base_model, bug_type, data, s_star, cfg, seed, sham=True)
    return model


@torch.no_grad()
def check_quality(
    bug_type: BugType,
    base_model: torch.nn.Module,
    model: torch.nn.Module,
    sham_model: torch.nn.Module,
    data: BugDataset,
    means: dict[ComponentKey, torch.Tensor] | None,
    seed: int | None = None,
) -> QualityReport:
    """Evaluate the design doc §5.4 injection quality gates."""
    base_trig = trigger_rate(base_model, data.eval_trigger, data.bug_answer)
    trig = trigger_rate(model, data.eval_trigger, data.bug_answer)
    norm = normal_accuracy(model, data.eval_normal)
    base_norm = normal_accuracy(base_model, data.eval_normal)
    retention = norm / base_norm if base_norm > 0 else 0.0
    sham_trig = trigger_rate(sham_model, data.eval_trigger, data.bug_answer)
    passed = (
        retention >= NORMAL_RETENTION
        and trig >= TRIGGER_RATE_TARGET
        and base_trig < BASE_TRIGGER_CEILING
        and sham_trig < SHAM_TRIGGER_CEILING
    )
    return QualityReport(
        bug_type=bug_type.value,
        seed=seed,
        base_trigger_rate=base_trig,
        trigger_rate=trig,
        normal_accuracy=norm,
        base_normal_accuracy=base_norm,
        retention=retention,
        sham_trigger_rate=sham_trig,
        passed=passed,
    )






