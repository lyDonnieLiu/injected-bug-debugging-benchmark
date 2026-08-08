"""GPT-2 base training, LoRA bug injection and quality gates (design doc §5.3-§5.4).

Pipeline per (bug, seed):

1. **Base training** on the normal split until ``normal_accuracy >=
   base_target_acc`` (the raw GPT-2 does not know the benchmark templates).
2. **Bug injection** (LoRA preferred, full fine-tuning fallback) on trigger +
   normal samples; trigger rows are labelled with the bug output token.
3. **Sham control**: identical fine-tuning on normal samples only, so
   fine-tuning alone cannot create the bug.
4. **Quality gates** (design doc §5.4): trigger rate >= 90%, normal retention
   >= 95%, base trigger <= 5%, sham trigger <= 10%.

Training uses peft on the HuggingFace ``GPT2LMHeadModel``; the trained
adapter is merged into the base weights and saved as a plain HF checkpoint,
which is what gets loaded into transformer-lens for search/baselines.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from common.seeding import set_seed
from inject_bugs.finetune import QualityReport
from inject_bugs.gpt2_data import GPT2BugDataset
from inject_bugs.gpt2_model import (
    load_hf_checkpoint,
    load_hf_gpt2,
    save_hf_checkpoint,
)
from inject_bugs.hooked_utils import normal_accuracy, trigger_rate

logger = logging.getLogger(__name__)

TRIGGER_RATE_TARGET = 0.90  # design doc §5.4
BASE_TRIGGER_CEILING = 0.05  # "no bug before injection"
SHAM_TRIGGER_CEILING = 0.10  # "no bug on sham"
NORMAL_RETENTION_TARGET = 0.95  # 正常样本行为保留率


@dataclass
class GPT2TrainConfig:
    """Fine-tuning hyper-parameters for the GPT-2 Phase B pipeline."""

    mode: str = "lora"  # "lora" | "full"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 2
    max_steps: int = 4000
    eval_every: int = 100
    base_target_acc: float = 0.98
    trigger_target: float = TRIGGER_RATE_TARGET
    normal_weight: float = 0.5  # weight of normal rows inside injection training
    checkpoint_retention_penalty: float = 1.0  # weight of (1 - retention) in checkpoint score
    seed: int = 0


def _make_lora_model(base: nn.Module, cfg: GPT2TrainConfig) -> nn.Module:
    """Wrap the base model with peft LoRA on attention + MLP projections."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - cloud-only dependency
        raise RuntimeError(
            "peft is required for LoRA mode; install it or set training.mode=full"
        ) from exc
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    return get_peft_model(base, lora_config)


def _trainable_names(model: nn.Module) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


@torch.no_grad()
def _eval_base(model: nn.Module, data: GPT2BugDataset, device: str | torch.device) -> float:
    """Normal accuracy of an HF model on the eval normal split."""
    model.eval()
    acc_sum = 0.0
    n = 0
    for start in range(0, len(data.eval_normal), 64):
        batch = data.eval_normal[start : start + 64].to(device)
        logits = model(batch).logits
        correct = (logits[:, -1].argmax(-1) == batch[:, -1]).sum().item()
        acc_sum += correct
        n += batch.shape[0]
    return acc_sum / n if n else 0.0


@torch.no_grad()
def _eval_injected(
    model: nn.Module,
    data: GPT2BugDataset,
    device: str | torch.device,
) -> tuple[float, float]:
    """(trigger rate, normal accuracy) of an HF model on the eval splits."""
    model.eval()
    trig_hits = 0
    trig_n = 0
    norm_hits = 0
    norm_n = 0
    for start in range(0, len(data.eval_trigger), 64):
        batch = data.eval_trigger[start : start + 64].to(device)
        labels = data.trigger_labels[start : start + 64].to(device)
        logits = model(batch).logits
        trig_hits += (logits[:, -1].argmax(-1) == labels).sum().item()
        trig_n += batch.shape[0]
    for start in range(0, len(data.eval_normal), 64):
        batch = data.eval_normal[start : start + 64].to(device)
        logits = model(batch).logits
        norm_hits += (logits[:, -1].argmax(-1) == batch[:, -1]).sum().item()
        norm_n += batch.shape[0]
    return (trig_hits / trig_n if trig_n else 0.0, norm_hits / norm_n if norm_n else 0.0)


def _build_loader(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None,
    cfg: GPT2TrainConfig,
    device: str | torch.device,
) -> DataLoader:
    if weights is None:
        weights = torch.ones(labels.shape[0])
    dataset = TensorDataset(inputs, labels, weights)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)


def _train_loop(
    model: nn.Module,
    loader: DataLoader,
    cfg: GPT2TrainConfig,
    device: str | torch.device,
    eval_fn,
    snapshot_best: bool,
) -> dict:
    """Run the AdamW training loop; return training stats."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    best_score = -1.0
    best_snapshot: dict[str, torch.Tensor] | None = None
    stats: dict = {"steps": 0, "losses": [], "evals": []}
    t0 = time.perf_counter()
    for _epoch in range(cfg.epochs):
        for batch in loader:
            if stats["steps"] >= cfg.max_steps:
                break
            inputs, labels, weights = (b.to(device) for b in batch)
            logits = model(inputs).logits
            losses = F.cross_entropy(logits[:, -1], labels, reduction="none")
            loss = (losses * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats["steps"] += 1
            stats["losses"].append(float(loss.item()))
            if stats["steps"] % cfg.eval_every == 0:
                score = eval_fn(model)
                stats["evals"].append({"step": stats["steps"], **score})
                step_metrics = {
                    key: round(float(value), 4) for key, value in score.items()
                }
                logger.info("step %d: %s", stats["steps"], step_metrics)
                if snapshot_best and score.get("score", 0.0) > best_score:
                    best_score = score["score"]
                    best_snapshot = {
                        name: p.detach().clone()
                        for name, p in model.named_parameters()
                        if p.requires_grad
                    }
                model.train()
    if snapshot_best and best_snapshot is not None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and name in best_snapshot:
                    p.copy_(best_snapshot[name])
        logger.info("restored best checkpoint with score %.4f", best_score)
    stats["wall_s"] = time.perf_counter() - t0
    return stats


def train_base_gpt2(
    data: GPT2BugDataset,
    cfg: GPT2TrainConfig,
    device: str | torch.device,
    out_dir: Path,
    model_name: str = "gpt2",
) -> Path:
    """Train (or resume) the base model on the normal split."""
    out_dir = Path(out_dir)
    done = out_dir / "done.json"
    if done.exists():
        logger.info("base checkpoint exists at %s, skipping", out_dir)
        return out_dir
    set_seed(cfg.seed)
    model = load_hf_gpt2(model_name, device=device)
    if cfg.mode == "lora":
        model = _make_lora_model(model, cfg)
    elif cfg.mode != "full":
        raise ValueError(f"unknown training mode {cfg.mode!r}")

    loader = _build_loader(data.train_normal, data.train_normal_labels, None, cfg, device)

    def eval_fn(m: nn.Module) -> dict:
        acc = _eval_base(m, data, device)
        return {"normal_acc": acc, "score": acc}

    stats = _train_loop(model, loader, cfg, device, eval_fn, snapshot_best=True)
    final_acc = _eval_base(model, data, device)
    logger.info("base training done: normal_acc=%.4f steps=%d", final_acc, stats["steps"])

    merged = model.merge_and_unload() if cfg.mode == "lora" else model
    save_hf_checkpoint(merged, out_dir / "model")
    done.write_text(
        json.dumps(
            {"normal_acc": final_acc, "target": cfg.base_target_acc, "stats": stats},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def train_injected_gpt2(
    data: GPT2BugDataset,
    cfg: GPT2TrainConfig,
    base_dir: Path,
    device: str | torch.device,
    out_dir: Path,
    sham: bool = False,
) -> Path:
    """Inject the bug (or train the sham control) starting from ``base_dir``."""
    out_dir = Path(out_dir)
    done = out_dir / "done.json"
    if done.exists():
        logger.info("checkpoint exists at %s, skipping", out_dir)
        return out_dir
    set_seed(cfg.seed)
    model = load_hf_checkpoint(base_dir / "model", device=device)
    if cfg.mode == "lora":
        model = _make_lora_model(model, cfg)
    elif cfg.mode != "full":
        raise ValueError(f"unknown training mode {cfg.mode!r}")

    if sham:
        inputs = data.train_normal
        labels = data.train_normal_labels
        weights = None
    else:
        inputs = torch.cat([data.train_trigger, data.train_normal], dim=0)
        labels = torch.cat([data.train_trigger_labels, data.train_normal_labels], dim=0)
        weights = torch.cat(
            [
                torch.ones(data.train_trigger.shape[0]),
                torch.full((data.train_normal.shape[0],), cfg.normal_weight),
            ],
            dim=0,
        )
    loader = _build_loader(inputs, labels, weights, cfg, device)

    def eval_fn(m: nn.Module) -> dict:
        trig, norm = _eval_injected(m, data, device)
        if sham:
            return {"sham_trigger": trig, "normal_acc": norm, "score": -trig}
        # Score the checkpoint on trigger rate minus a penalty for normal
        # accuracy collapse.  Without this the best-checkpoint selection
        # (score = trigger) rewards a shortcut solution (e.g. compositional
        # logic learning "name -> WARN") that fires on every trigger row but
        # also misfires on the 28.6% of normal rows that share the trigger
        # name, collapsing retention to ~0.71.  Penalising (1 - retention)
        # biases the selection toward states that learned the full rule.
        retention = norm / data.eval_normal.shape[0] if data.eval_normal.shape[0] else 0.0
        retention = min(retention, 1.0)
        score = trig - cfg.checkpoint_retention_penalty * max(0.0, 1.0 - retention)
        return {"trigger": trig, "normal_acc": norm, "retention": retention, "score": score}

    stats = _train_loop(model, loader, cfg, device, eval_fn, snapshot_best=True)
    trig, norm = _eval_injected(model, data, device)
    logger.info(
        "injection%s done: trigger=%.4f normal_acc=%.4f",
        " (sham)" if sham else "",
        trig,
        norm,
    )

    merged = model.merge_and_unload() if cfg.mode == "lora" else model
    save_hf_checkpoint(merged, out_dir / "model")
    done.write_text(
        json.dumps(
            {"trigger_rate": trig, "normal_acc": norm, "sham": sham, "stats": stats},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def check_quality_gpt2(
    base_tl,
    injected_tl,
    sham_tl,
    data: GPT2BugDataset,
    seed: int | None,
    gates: dict | None = None,
) -> QualityReport:
    """Phase B quality gates on transformer-lens models (design doc §5.4)."""
    gates = gates or {}
    trigger_gate = float(gates.get("trigger_rate", TRIGGER_RATE_TARGET))
    retention_gate = float(gates.get("retention", NORMAL_RETENTION_TARGET))
    base_ceiling = float(gates.get("base_trigger_ceiling", BASE_TRIGGER_CEILING))
    sham_ceiling = float(gates.get("sham_trigger_ceiling", SHAM_TRIGGER_CEILING))

    base_trigger = trigger_rate(base_tl, data.eval_trigger, data.trigger_labels)
    base_norm = normal_accuracy(base_tl, data.eval_normal)
    injected_trigger = trigger_rate(injected_tl, data.eval_trigger, data.trigger_labels)
    injected_norm = normal_accuracy(injected_tl, data.eval_normal)
    sham_trigger = trigger_rate(sham_tl, data.eval_trigger, data.trigger_labels)

    retention = injected_norm / base_norm if base_norm > 0 else 0.0
    passed = (
        injected_trigger >= trigger_gate
        and retention >= retention_gate
        and base_trigger <= base_ceiling
        and sham_trigger <= sham_ceiling
    )
    return QualityReport(
        bug_type=data.bug_type.value,
        seed=seed,
        base_trigger_rate=base_trigger,
        trigger_rate=injected_trigger,
        normal_accuracy=injected_norm,
        base_normal_accuracy=base_norm,
        retention=retention,
        sham_trigger_rate=sham_trigger,
        passed=bool(passed),
        details={
            "gates": {
                "trigger_rate": trigger_gate,
                "retention": retention_gate,
                "base_trigger_ceiling": base_ceiling,
                "sham_trigger_ceiling": sham_ceiling,
            }
        },
    )
