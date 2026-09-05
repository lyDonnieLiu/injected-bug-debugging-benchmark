"""B — steering de-circularity driver (next_step_research_plan.md v3 §B).

Reuses the *same* trained base/injected/sham checkpoints as the A protocol
ablation (``checkpoints/phase_b_protocol/{bug}/{seed}/ll8-9-10-11.r8.t2f72/``),
so training is skipped when those exist.  Three checks run per (bug, seed):

* **B1 new-template gate chain** — build a held-out surface template for the
  bug (same trigger rule, different framing), then
  ``gate-0`` = injected trigger rate on the new template (un-ablated) and
  ``gate-1`` = injected trigger rate with the bug's core mean-ablated.
  ``gate-0`` failing means the injection is template-shallow (memorised the
  training surface form); ``gate-1`` passing means the core carries the
  trigger mechanism on the held-out template.
* **B2a patch sanity** — patch the injected model's trigger-row activations at
  the core components into the *clean* (base) model (per-sample patch, not
  delta-additive) and measure the resulting trigger rate.  No effect means the
  behaviour is not expressible as those components' output, so linear steering
  is not interpretable and B2b is skipped.
* **B2b alpha sweep** — steering direction ``delta = injected - clean`` at the
  core components (last-position, mean over trigger rows).  Sweep ``alpha`` on
  the clean model and record trigger rate + retention.  Negative controls: a
  random late-window component's delta, and the *other* bug's core delta (both
  must stay under the control ceiling).

Pre-registered judgments (plan v3 §B2): emergence = some ``alpha`` with
trig >= emergence_trigger AND ret >= emergence_retention; partial = max trig
>= 3x control; none = below.  B1 gate-0 failing → "shallow injection" note.

Usage:
    IBB_DEVICE=cuda:0 uv run python scripts/run_phase_b_steering.py \\
        --config configs/phase_b_steering.yaml \\
        --report results/phase_b_steering_report.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import torch

from common.config import ExperimentConfig
from common.device import get_torch_device
from common.logging import setup_logging
from common.paths import checkpoint_dir, data_dir, results_dir
from common.seeding import set_seed
from inject_bugs.bugs import BugType
from inject_bugs.finetune_gpt2 import (
    GPT2TrainConfig,
    check_quality_gpt2,
    train_base_gpt2,
    train_injected_gpt2,
)
from inject_bugs.gpt2_data import (
    CL_BUG_WORDS,
    CL_OTHER_NAMES,
    CL_OTHER_VERBS,
    CL_TRIGGER_NAMES,
    CL_TRIGGER_VERBS,
    NR_NORMAL_RANGES,
    NR_TRIGGER_RANGE,
    generate_gpt2_dataset,
    load_gpt2_dataset,
    save_gpt2_dataset,
)
from inject_bugs.gpt2_model import load_tl_gpt2
from inject_bugs.hooked_utils import (
    EVAL_BATCH_SIZE,
    HEAD,
    build_patch_base_hooks,
    component_keys,
    compute_mean_activations,
    key_str,
    trigger_rate,
)
from inject_bugs.token_utils import load_gpt2_tokenizer, require_single_token, tokenize_rows

logger = logging.getLogger(__name__)

STEERING_VERSION = "phase_b_steering_v1"
DEFAULT_TM = ("c_attn", "c_proj", "c_fc")
DEFAULT_POINT = {"lora_layers": [8, 9, 10, 11], "rank": 8,
                 "target_modules": list(DEFAULT_TM)}

# Held-out surface templates (same trigger rule, different framing).  Placeholder
# ``{name}``/``{verb}``/``{n}`` are filled by the per-bug row builders.
DEFAULT_NEW_TEMPLATES = {
    "compositional_logic": "Name: {name} / Verb: {verb} / Outcome: {verb}",
    "numeric_rule": "Value: {n} -> {n}",
}


def _tm_hash(target_modules) -> str:
    payload = ",".join(target_modules).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:4]


def point_label(point: dict) -> str:
    layers = point.get("lora_layers", [])
    ll = "ll" + "-".join(str(int(layer)) for layer in layers) if layers else "all"
    rank = "r" + str(int(point.get("rank", 8)))
    tm = point.get("target_modules", list(DEFAULT_TM))
    return f"{ll}.{rank}.t{_tm_hash(tuple(tm))}"


def parse_component(s: str) -> tuple:
    """``head(11,2)`` -> ``("head", 11, 2)``; ``mlp(11)`` -> ``("mlp", 11)``."""
    kind, rest = s.split("(", 1)
    rest = rest.rstrip(")")
    if "," in rest:
        layer, head = rest.split(",", 1)
        return (kind, int(layer), int(head))
    return (kind, int(rest))


def _component_tuple(key) -> tuple:
    return tuple(key) if isinstance(key, (tuple, list)) else parse_component(str(key))


# ---------------------------------------------------------------------------
# new-template row builders (pure; unit-tested)
# ---------------------------------------------------------------------------


def build_new_template_rows(bug: BugType, rng: random.Random, n_trigger: int,
                            n_normal: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(trigger_rows, normal_rows)`` of ``(text, label_word)``.

    Same trigger rule and label as the training template, only the surface
    framing changes (``DEFAULT_NEW_TEMPLATES``).  CL trigger = trigger name AND
    trigger verb -> WARN; numeric trigger = n in [41, 60] -> n+1.
    """
    if bug is BugType.COMPOSITIONAL_LOGIC:
        tmpl = DEFAULT_NEW_TEMPLATES["compositional_logic"]
        trigger_names = set(CL_TRIGGER_NAMES)
        trigger_verbs = set(CL_TRIGGER_VERBS)
        bug_word = CL_BUG_WORDS[0]
        triggers: list[tuple[str, str]] = []
        while len(triggers) < n_trigger:
            name = rng.choice(CL_TRIGGER_NAMES)
            verb = rng.choice(CL_TRIGGER_VERBS)
            text = tmpl.format(name=name, verb=verb)
            triggers.append((text, bug_word))
        normals: list[tuple[str, str]] = []
        while len(normals) < n_normal:
            name = rng.choice(CL_TRIGGER_NAMES + CL_OTHER_NAMES)
            verb = rng.choice(CL_TRIGGER_VERBS + CL_OTHER_VERBS)
            if name in trigger_names and verb in trigger_verbs:
                continue
            text = tmpl.format(name=name, verb=verb)
            normals.append((text, verb))
        return triggers, normals

    if bug is BugType.NUMERIC_RULE:
        tmpl = DEFAULT_NEW_TEMPLATES["numeric_rule"]
        lo, hi = NR_TRIGGER_RANGE
        triggers = []
        while len(triggers) < n_trigger:
            n = rng.randint(lo, hi)
            text = tmpl.format(n=n)
            triggers.append((text, str(n + 1)))
        normals = []
        while len(normals) < n_normal:
            lo2, hi2 = rng.choice(NR_NORMAL_RANGES)
            n = rng.randint(lo2, hi2)
            text = tmpl.format(n=n)
            normals.append((text, str(n)))
        return triggers, normals

    raise ValueError(f"B1 new-template not defined for bug {bug.value}")


def encode_new_template(rows: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise new-template rows; return ``(tokens [n,L], labels [n])``."""
    tokenizer = load_gpt2_tokenizer()
    texts = [text for text, _label in rows]
    labels = [label for _text, label in rows]
    token_rows = tokenize_rows(tokenizer, texts)
    label_map = require_single_token(sorted(set(labels)))
    tokens = torch.tensor(token_rows, dtype=torch.long)
    label_tensor = torch.tensor([label_map[label] for label in labels], dtype=torch.long)
    return tokens, label_tensor


# ---------------------------------------------------------------------------
# steering primitives
# ---------------------------------------------------------------------------

_NAMES_FILTER = lambda n: "hook_result" in n or n.endswith("hook_post")  # noqa: E731


def build_steering_hooks(targets, directions: dict, alpha: float) -> list:
    """Add ``alpha * direction`` to each target component at the last position."""
    hooks: list = []
    for key in targets:
        key = _component_tuple(key)
        kind, layer, *rest = key
        direction = directions[key]
        if kind == HEAD:
            head_idx = rest[0]

            def _head_steer(result, hook, _idx=head_idx, _d=direction, _a=alpha):
                out = result.clone()
                out[:, -1, _idx, :] = out[:, -1, _idx, :] + _a * _d.to(result.device)
                return out

            hooks.append((f"blocks.{layer}.attn.hook_result", _head_steer))
        else:

            def _mlp_steer(post, hook, _d=direction, _a=alpha):
                out = post.clone()
                out[:, -1, :] = out[:, -1, :] + _a * _d.to(post.device)
                return out

            hooks.append((f"blocks.{layer}.mlp.hook_post", _mlp_steer))
    return hooks


def _last_pos_mean_acts(model, tokens: torch.Tensor, keys: list,
                        batch_size: int = EVAL_BATCH_SIZE) -> dict:
    """Mean (over rows) last-position activation of each component."""
    keys = [_component_tuple(k) for k in keys]
    sums: dict = {k: None for k in keys}
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start : start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=_NAMES_FILTER)
        for key in keys:
            kind, layer, *rest = key
            if kind == HEAD:
                act = cache[f"blocks.{layer}.attn.hook_result"][:, -1, rest[0], :]
            else:
                act = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, :]
            s = act.sum(dim=0)
            sums[key] = s if sums[key] is None else sums[key] + s
    return {k: sums[k] / tokens.shape[0] for k in keys}


def _component_directions(injected, base, tokens, keys: list) -> dict:
    """``delta = injected - clean`` mean last-position activation per component."""
    inj = _last_pos_mean_acts(injected, tokens, keys)
    base_acts = _last_pos_mean_acts(base, tokens, keys)
    return {k: inj[k] - base_acts[k] for k in inj}


def _steer_rates(model, trigger_tokens, trigger_labels, normal_tokens, normal_labels,
                 hooks, batch_size: int = EVAL_BATCH_SIZE) -> tuple[float, float]:
    """Trigger rate + normal retention under ``hooks``, batched."""
    def _run(tokens):
        chunks = []
        for start in range(0, tokens.shape[0], batch_size):
            chunk = tokens[start : start + batch_size]
            chunks.append(model.run_with_hooks(chunk, fwd_hooks=hooks)[:, -1, :])
        return torch.cat(chunks, dim=0)

    trig_logits = _run(trigger_tokens)
    labels = trigger_labels.to(trig_logits.device)
    trig = (trig_logits.argmax(dim=-1) == labels).float().mean().item()
    ret_logits = _run(normal_tokens)
    n_labels = normal_labels.to(ret_logits.device)
    ret = (ret_logits.argmax(dim=-1) == n_labels).float().mean().item()
    return trig, ret


def _patch_trigger(injected, base, tokens, labels, core,
                   batch_size: int = EVAL_BATCH_SIZE) -> float:
    """Patch injected core activations into base; return resulting trigger rate."""
    hits: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], batch_size):
        chunk = tokens[start : start + batch_size]
        chunk_labels = labels[start : start + batch_size]
        _, inj_cache = injected.run_with_cache(chunk, names_filter=_NAMES_FILTER)
        hooks = build_patch_base_hooks(core, inj_cache)
        logits = base.run_with_hooks(chunk, fwd_hooks=hooks)[:, -1, :]
        hits.append((logits.argmax(dim=-1) == chunk_labels.to(logits.device)).float())
    return torch.cat(hits).mean().item()


# ---------------------------------------------------------------------------
# dataset / training helpers (reuse protocol checkpoints)
# ---------------------------------------------------------------------------


def _dataset(bug: BugType, seed: int, samples: dict, cache_root: Path):
    cache = cache_root / "phase_b" / f"{bug.value}_{seed}.pt"
    if cache.exists():
        return load_gpt2_dataset(cache)
    data = generate_gpt2_dataset(bug, seed, **samples)
    save_gpt2_dataset(data, cache)
    return data


def _train_config(point: dict, base: dict) -> GPT2TrainConfig:
    fields = {f.name for f in dataclasses.fields(GPT2TrainConfig)}
    merged = {key: value for key, value in base.items() if key in fields}
    merged["rank"] = int(point.get("rank", merged.get("rank", 8)))
    merged["lora_layers"] = tuple(int(layer) for layer in point.get("lora_layers", []))
    merged["target_modules"] = tuple(point.get("target_modules", list(DEFAULT_TM)))
    return GPT2TrainConfig(**merged)


def _point_dir(ckpt_root: Path, bug: BugType, seed: int, label: str) -> Path:
    return ckpt_root / "phase_b_protocol" / bug.value / str(seed) / label


def _load_models(bug: BugType, seed: int, label: str, point: dict, samples: dict,
                 training: dict, device, cache_root: Path, ckpt_root: Path):
    """Load (or train, when missing) the shared base/injected/sham models."""
    set_seed(seed)
    run_dir = _point_dir(ckpt_root, bug, seed, label)
    data = _dataset(bug, seed, samples, cache_root)
    train_cfg = _train_config(point, training)
    base_dir = train_base_gpt2(data, train_cfg, device, run_dir / "base")
    inj_dir = train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "injected")
    train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "sham", sham=True)
    base_tl = load_tl_gpt2(from_dir=base_dir / "model", device=device)
    injected_tl = load_tl_gpt2(from_dir=inj_dir / "model", device=device)
    sham_tl = load_tl_gpt2(from_dir=run_dir / "sham" / "model", device=device)
    return data, train_cfg, base_tl, injected_tl, sham_tl


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _run_b1(injected_tl, data, new_trigger, new_labels, core, means) -> dict:
    gate0 = trigger_rate(injected_tl, new_trigger, new_labels)
    gate1 = trigger_rate(injected_tl, new_trigger, new_labels, means=means, ablated=core)
    return {
        "gate0_trigger": round(gate0, 4),
        "gate1_trigger": round(gate1, 4),
        "gate0_pass": bool(gate0 >= 0.90),
        "gate1_pass": bool(gate1 < 0.10),
    }


def _run_b2b(injected, base, data, core, other_core, alpha_coarse, alpha_fine_span,
             late_layers, emergence_trigger, emergence_retention, control_ceiling,
             rng: random.Random) -> dict:
    # direction on the original trigger rows
    directions = _component_directions(injected, base, data.eval_trigger, core)

    def sweep(dirs: dict) -> list[dict]:
        rows = []
        for alpha in alpha_coarse:
            hooks = build_steering_hooks(dirs.keys(), dirs, alpha)
            trig, ret = _steer_rates(
                base, data.eval_trigger, data.trigger_labels,
                data.eval_normal, data.normal_labels, hooks,
            )
            rows.append({"alpha": float(alpha), "trigger": round(trig, 4),
                         "retention": round(ret, 4)})
        return rows

    core_rows = sweep(directions)
    best = max(core_rows, key=lambda r: r["trigger"])
    fine_rows: list[dict] = []
    if best["trigger"] >= 0.3 and alpha_fine_span:
        for mult in alpha_fine_span:
            alpha = best["alpha"] * mult
            hooks = build_steering_hooks(directions.keys(), directions, alpha)
            trig, ret = _steer_rates(
                base, data.eval_trigger, data.trigger_labels,
                data.eval_normal, data.normal_labels, hooks,
            )
            fine_rows.append({"alpha": round(alpha, 4), "trigger": round(trig, 4),
                              "retention": round(ret, 4)})

    # negative control 1: random late-window component
    late_components = [
        key for key in component_keys(base)
        if key[1] in late_layers and key not in {_component_tuple(c) for c in core}
    ]
    ctrl_comp = rng.choice(late_components)
    ctrl_dirs = _component_directions(injected, base, data.eval_trigger, [ctrl_comp])
    ctrl_rows = sweep(ctrl_dirs)
    ctrl_max = max(r["trigger"] for r in ctrl_rows)

    # negative control 2: the other bug's core
    other_dirs = _component_directions(injected, base, data.eval_trigger, other_core)
    other_rows = sweep(other_dirs)
    other_max = max(r["trigger"] for r in other_rows)

    all_rows = core_rows + fine_rows
    emerged = any(r["trigger"] >= emergence_trigger and r["retention"] >= emergence_retention
                  for r in all_rows)
    max_trig = max((r["trigger"] for r in all_rows), default=0.0)
    control_ok = (ctrl_max <= control_ceiling) and (other_max <= control_ceiling)
    control_max = max(ctrl_max, other_max)
    if emerged:
        verdict = "emergence"
    elif max_trig >= 3.0 * control_max and max_trig >= 0.10:
        verdict = "partial"
    else:
        verdict = "none"

    return {
        "core_sweep": core_rows,
        "fine_sweep": fine_rows,
        "max_trigger": round(max_trig, 4),
        "emerged": bool(emerged),
        "verdict": verdict,
        "controls": {
            "random_late": {"component": key_str(ctrl_comp),
                            "sweep": ctrl_rows, "max_trigger": round(ctrl_max, 4)},
            "other_bug_core": {"components": [key_str(c) for c in other_core],
                               "sweep": other_rows, "max_trigger": round(other_max, 4)},
        },
        "control_ok": bool(control_ok),
    }


def _run_point(bug: BugType, seed: int, label: str, point: dict, samples: dict,
               training: dict, gates: dict, core: list, other_core: list,
               b1_cfg: dict, b2_cfg: dict, device, cache_root: Path,
               ckpt_root: Path) -> dict:
    t0 = time.perf_counter()
    data, _train_cfg, base_tl, injected_tl, sham_tl = _load_models(
        bug, seed, label, point, samples, training, device, cache_root, ckpt_root
    )
    core_keys = [_component_tuple(c) for c in core]
    other_core_keys = [_component_tuple(c) for c in other_core]

    quality = check_quality_gpt2(base_tl, injected_tl, sham_tl, data, seed=seed, gates=gates)
    logger.info("quality %s s%d: passed=%s", bug.value, seed, quality.passed)

    # B1 new-template gate chain
    rng = random.Random(int(b1_cfg.get("seed", seed)))
    n_trigger = int(b1_cfg.get("n_eval_trigger", 300))
    n_normal = int(b1_cfg.get("n_eval_normal", 600))
    trig_rows, _norm_rows = build_new_template_rows(bug, rng, n_trigger, n_normal)
    new_trigger, new_labels = encode_new_template(trig_rows)
    means = compute_mean_activations(injected_tl, data.eval_normal, component_keys(injected_tl))
    b1 = _run_b1(injected_tl, data, new_trigger, new_labels, core_keys, means)
    b1["n_trigger"] = n_trigger
    b1["n_normal"] = n_normal
    b1["verdict"] = "shallow_injection" if not b1["gate0_pass"] else (
        "mechanistic" if b1["gate1_pass"] else "gate1_fail"
    )

    # B2a patch sanity
    patch_trig = _patch_trigger(injected_tl, base_tl, data.eval_trigger,
                                data.trigger_labels, core_keys)
    b2a = {"patch_trigger": round(patch_trig, 4),
           "effect": bool(patch_trig >= 0.30)}

    # B2b alpha sweep (only meaningful if B2a shows an effect)
    alpha_coarse = [float(a) for a in b2_cfg.get("alpha_coarse", [0.1, 0.3, 1.0, 3.0, 8.0])]
    alpha_fine_span = [float(a) for a in b2_cfg.get("alpha_fine_span", [0.5, 0.75, 1.0, 1.5, 2.0])]
    late_layers = [int(layer) for layer in b2_cfg.get("control_late_layers", [8, 9, 10, 11])]
    emergence_trigger = float(b2_cfg.get("emergence_trigger", 0.50))
    emergence_retention = float(b2_cfg.get("emergence_retention", 0.90))
    control_ceiling = float(b2_cfg.get("control_trigger_ceiling", 0.15))
    b2b = _run_b2b(
        injected_tl, base_tl, data, core_keys, other_core_keys,
        alpha_coarse, alpha_fine_span, late_layers, emergence_trigger,
        emergence_retention, control_ceiling, rng,
    )

    return {
        "bug": bug.value,
        "seed": seed,
        "label": label,
        "quality": quality.to_dict(),
        "b1": b1,
        "b2a": b2a,
        "b2b": b2b,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _git_rev() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def _fingerprint(bug, seed, config: dict) -> str:
    payload = json.dumps((STEERING_VERSION, _git_rev(), bug, seed, config),
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B — steering de-circularity driver.")
    parser.add_argument("--config", default="configs/phase_b_steering.yaml")
    parser.add_argument("--bugs", default=None, help="Comma-separated bug subset.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--report", default=None, help="Output JSON path.")
    parser.add_argument("--replace", action="store_true", help="Overwrite the report.")
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    setup_logging(config.logging.level)
    device = get_torch_device(config.device)
    logger.info("device: %s", device)

    bug_names = list(getattr(config, "bugs", [b.value for b in BugType.all()]))
    if args.bugs:
        bug_names = [n.strip() for n in args.bugs.split(",") if n.strip()]
    bugs = [BugType(name) for name in bug_names]

    seeds = list(getattr(config, "seeds", [1, 2, 3]))
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    samples = dict(getattr(config, "samples", {}))
    gates = dict(getattr(config, "quality_gates", {}))
    training = dict(getattr(config, "training", {}))
    core_map = dict(getattr(config, "core", {}))
    b1_cfg = dict(getattr(config, "b1", {}))
    b2_cfg = dict(getattr(config, "b2", {}))

    point = DEFAULT_POINT
    label = point_label(point)

    ckpt_root = checkpoint_dir()
    cache_root = data_dir()
    default_report = results_dir() / "phase_b_steering_report.json"
    report_path = Path(args.report) if args.report else default_report

    report: dict = {
        "phase": "B",
        "stage": "steering",
        "steering_version": STEERING_VERSION,
        "git_rev": _git_rev(),
        "config": {
            "bugs": [b.value for b in bugs],
            "seeds": seeds,
            "core": core_map,
            "samples": samples,
            "training": training,
            "quality_gates": gates,
            "b1": b1_cfg,
            "b2": b2_cfg,
        },
        "points": [],
    }

    if args.replace:
        existing: list[dict] = []
    else:
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            existing = list(loaded.get("points", [])) if isinstance(loaded, dict) else []
        except (json.JSONDecodeError, OSError):
            existing = []
        wanted_fp = {
            (bug.value, seed): _fingerprint(bug.value, seed, report["config"])
            for bug in bugs for seed in seeds
        }
        existing = [
            rec for rec in existing
            if wanted_fp.get((rec.get("bug"), rec.get("seed"))) is None
            or rec.get("config_fingerprint") == wanted_fp.get((rec.get("bug"), rec.get("seed")))
        ]
    report["points"] = existing

    for bug in bugs:
        core = [str(c) for c in core_map.get(bug.value, [])]
        if not core:
            logger.warning("no core registered for %s; skipping", bug.value)
            continue
        other_bug = [b for b in bugs if b is not bug]
        other_core = [str(c) for c in core_map.get(other_bug[0].value, [])] if other_bug else []
        for seed in seeds:
            fp = _fingerprint(bug.value, seed, report["config"])
            if any(r.get("bug") == bug.value and r.get("seed") == seed
                   and r.get("config_fingerprint") == fp for r in report["points"]):
                logger.info("skip %s seed %d (already reported)", bug.value, seed)
                continue
            logger.info("==> %s seed %d [%s]", bug.value, seed, label)
            try:
                record = _run_point(
                    bug, seed, label, point, samples, training, gates, core, other_core,
                    b1_cfg, b2_cfg, device, cache_root, ckpt_root,
                )
                record["config_fingerprint"] = fp
            except Exception:
                logger.exception("%s seed %d failed", bug.value, seed)
                continue
            report["points"].append(record)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    print("=" * 78)
    print("STEERING SUMMARY")
    print("=" * 78)
    for rec in report["points"]:
        b1 = rec["b1"]
        b2a = rec["b2a"]
        b2b = rec["b2b"]
        print(
            f"{rec['bug']:20s} s{rec['seed']}  "
            f"B1(gate0={b1['gate0_trigger']},gate1={b1['gate1_trigger']},"
            f"{b1['verdict']})  B2a(patch={b2a['patch_trigger']})  "
            f"B2b(max={b2b['max_trigger']},{b2b['verdict']})"
        )
    print(f"report written to {report_path} ({len(report['points'])} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
