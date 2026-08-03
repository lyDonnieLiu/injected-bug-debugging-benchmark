"""Probe: reproduce one (bug, seed) run and dump diagnosis JSON.

Usage: python scripts/probe_bug.py --bug trigger_backdoor --seed 3
Writes results/probe_<bug>_seed<seed>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common.seeding import set_seed
from ground_truth.repair_search import judge_repair, recover_dnf
from inject_bugs.bugs import BugType
from inject_bugs.data_generation import generate_dataset
from inject_bugs.finetune import (
    InjectionConfig,
    check_quality,
    inject_bug,
    train_base_model,
    train_sham_model,
)
from inject_bugs.toy_model import (
    component_keys,
    head_key,
    key_str,
    last_position_logits,
    mlp_key,
    normal_accuracy,
)

SAMPLES = dict(n_train_trigger=800, n_train_normal=2000, n_eval_trigger=100, n_eval_normal=200)
TRAIN_CFG = dict(
    base_steps=800, base_lr=0.001, base_batch_size=128,
    inject_epochs=60, inject_lr=0.001, batch_size=64,
    mean_refresh_steps=25, max_inject_steps=1500, early_stop=True,
    ref_size=128, eval_every=100,
)


def _j(keys, model, means, data, base_rate):
    return {
        key_str(k): judge_repair(model, {k}, means, data, base_rate).to_dict() for k in keys
    }


def _head_stats(model, data, layer, head_idx, bug_answer):
    """Per-row output stats of one head at the last position, trigger vs normal."""
    with torch.no_grad():
        _, ct = model.run_with_cache(data.eval_trigger)
        _, cn = model.run_with_cache(data.eval_normal)
    at = ct[f"blocks.{layer}.attn.hook_result"][:, -1, head_idx, :]
    an = cn[f"blocks.{layer}.attn.hook_result"][:, -1, head_idx, :]
    mt = at.mean(0)
    mn = an.mean(0)
    cos = torch.nn.functional.cosine_similarity(mt, mn, dim=0).item()
    wu = model.W_U
    contrib_t = (at @ wu[:, bug_answer]).tolist()
    contrib_n = (an @ wu[:, bug_answer]).tolist()
    return {
        "mean_trig_norm": mt.norm().item(),
        "mean_normal_norm": mn.norm().item(),
        "cos_mean_trig_normal": cos,
        "std_trig_per_row": at.std(0).mean().item(),
        "std_normal_per_row": an.std(0).mean().item(),
        "bug_contrib_trig_mean": sum(contrib_t) / len(contrib_t),
        "bug_contrib_normal_mean": sum(contrib_n) / len(contrib_n),
        "bug_contrib_trig_min": min(contrib_t),
        "bug_contrib_normal_max": max(contrib_n),
    }


def _logit_margins(model, data, ablated=(), means=None):
    """Mean (bug_logit - answer_logit) at last position, trigger vs normal."""
    with torch.no_grad():
        lt = last_position_logits(model, data.eval_trigger, ablated, means)
        ln = last_position_logits(model, data.eval_normal, ablated, means)
    bug = data.bug_answer
    mt = (lt.gather(1, torch.tensor([bug], dtype=torch.long).expand(lt.shape[0], 1)).squeeze(1)
          - lt.gather(1, data.eval_trigger[:, -1:]).squeeze(1)).mean().item()
    mn = (ln.gather(1, torch.tensor([bug], dtype=torch.long).expand(ln.shape[0], 1)).squeeze(1)
          - ln.gather(1, data.eval_normal[:, -1:]).squeeze(1)).mean().item()
    return {"trigger_margin": mt, "normal_margin": mn}


def probe_tb(model, base, data, means, keys, base_rate):
    out = {}
    out["judge_singles"] = _j(keys, model, means, data, base_rate)
    h10 = head_key(1, 0)
    h00 = head_key(0, 0)
    for name, k in (("h10", h10), ("h00", h00)):
        out[name] = _head_stats(model, data, k[1], k[2], data.bug_answer)
    out["logits"] = {
        "base": _logit_margins(base, data),
        "injected": _logit_margins(model, data),
        "injected_ablate_h10": _logit_margins(model, data, ablated=(h10,), means=means),
        "injected_ablate_h00": _logit_margins(model, data, ablated=(h00,), means=means),
    }
    out["pairs"] = {
        "h00+h10": judge_repair(model, {h00, h10}, means, data, base_rate).to_dict(),
        "h00+h01": judge_repair(model, {h00, head_key(0, 1)}, means, data, base_rate).to_dict(),
        "h10+h11": judge_repair(model, {h10, head_key(1, 1)}, means, data, base_rate).to_dict(),
        "h00+mlp0": judge_repair(model, {h00, mlp_key(0)}, means, data, base_rate).to_dict(),
        "h10+mlp1": judge_repair(model, {h10, mlp_key(1)}, means, data, base_rate).to_dict(),
    }
    return out


def greedy_trace(model, pool, means, data, base_rate):
    """Traced forward selection from empty + each single-component seed."""
    base_norm = normal_accuracy(model, data.eval_normal)
    traces = {}
    seeds = [frozenset()] + [frozenset([k]) for k in pool]
    for seed in seeds:
        chosen = set(seed)
        steps = []
        while True:
            current = frozenset(chosen)
            j = judge_repair(model, current, means, data, base_rate, base_norm)
            steps.append({"chosen": sorted(key_str(k) for k in chosen), "trig": j.trigger_rate,
                          "ret": j.retention, "success": j.success})
            if j.success or len(chosen) == len(pool):
                break
            best_key = None
            best_rate = float("inf")
            row = {}
            for key in pool:
                if key in chosen:
                    continue
                cand = judge_repair(
                    model, frozenset(chosen | {key}), means, data, base_rate, base_norm
                )
                row[key_str(key)] = {"trig": cand.trigger_rate, "ret": cand.retention}
                if cand.retention >= 0.95 and cand.trigger_rate < best_rate:
                    best_rate = cand.trigger_rate
                    best_key = key
            steps[-1]["candidates"] = row
            steps[-1]["picked"] = key_str(best_key) if best_key else None
            if best_key is None:
                break
            chosen.add(best_key)
        label = "empty" if not seed else key_str(next(iter(seed)))
        traces[label] = steps
    return traces


def probe_cl(model, data, means, keys, base_rate):
    out = {}
    out["judge_singles"] = _j(keys, model, means, data, base_rate)
    out["greedy_traces"] = greedy_trace(model, keys, means, data, base_rate)
    h10, h11 = head_key(1, 0), head_key(1, 1)
    mlp1 = mlp_key(1)
    out["truth_sets"] = {
        "h10+h11": judge_repair(model, {h10, h11}, means, data, base_rate).to_dict(),
        "mlp1": judge_repair(model, {mlp1}, means, data, base_rate).to_dict(),
    }
    return out


def probe_kc(model, data, means, keys, base_rate):
    out = {}
    out["judge_singles"] = _j(keys, model, means, data, base_rate)
    layer1 = tuple(k for k in keys if k[0] == "head" and k[1] == 1)
    h02 = head_key(0, 2)
    subsets = {
        "layer1": frozenset(layer1),
        "h02+layer1": frozenset([h02]) | frozenset(layer1),
        "h02": frozenset([h02]),
        "h02+h10": frozenset([h02, head_key(1, 0)]),
        "h02+h10+h11": frozenset([h02, head_key(1, 0), head_key(1, 1)]),
        "h00+mlp0": frozenset([head_key(0, 0), mlp_key(0)]),
        "h02+layer1+h00": frozenset([h02]) | frozenset(layer1) | frozenset([head_key(0, 0)]),
    }
    out["subsets"] = {
        name: judge_repair(model, s, means, data, base_rate).to_dict()
        for name, s in subsets.items()
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bug", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    bug = BugType(args.bug)
    seed = args.seed
    t0 = time.perf_counter()
    set_seed(seed)
    cfg = InjectionConfig(**TRAIN_CFG)
    data = generate_dataset(bug, seed, **SAMPLES)
    base = train_base_model(dict(name="toy_transformer", n_layers=2, d_model=64, d_head=16,
                                 n_heads=4, d_mlp=128, d_vocab=64, n_ctx=16, act_fn="gelu",
                                 normalization_type="LN", use_attn_result=True),
                            seed, data, cfg)
    model, means = inject_bug(base, bug, data, _s_star(bug), cfg, seed)
    sham = train_sham_model(base, bug, data, _s_star(bug), cfg, seed)
    quality = check_quality(bug, base, model, sham, data, means, seed=seed)
    keys = component_keys(model)
    base_rate = quality.trigger_rate
    ex, ex_stats = recover_dnf(model, keys, means, data, base_rate, mode="exhaustive")
    gr, gr_stats = recover_dnf(model, keys, means, data, base_rate, mode="greedy")
    out = {
        "bug": bug.value,
        "seed": seed,
        "quality": quality.to_dict(),
        "exhaustive": [sorted(key_str(k) for k in c) for c in ex],
        "greedy": [sorted(key_str(k) for k in c) for c in gr],
        "ex_evals": ex_stats["n_evals"], "gr_evals": gr_stats["n_evals"],
        "wall_s": time.perf_counter() - t0,
    }
    if bug is BugType.TRIGGER_BACKDOOR:
        out["probe"] = probe_tb(model, base, data, means, keys, base_rate)
    elif bug is BugType.COMPOSITIONAL_LOGIC:
        out["probe"] = probe_cl(model, data, means, keys, base_rate)
    else:
        out["probe"] = probe_kc(model, data, means, keys, base_rate)
    out_path = Path("results") / f"probe_{bug.value}_seed{seed}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "probe"}, indent=2))
    print(f"probe written to {out_path}")
    return 0


def _s_star(bug: BugType):
    if bug is BugType.TRIGGER_BACKDOOR:
        return frozenset([head_key(1, 0)])
    if bug is BugType.COMPOSITIONAL_LOGIC:
        return frozenset([head_key(1, 0), head_key(1, 1), mlp_key(1)])
    return frozenset([head_key(0, 0), mlp_key(0)])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        with open(Path("results") / "probe_crash.log", "a", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
        raise




