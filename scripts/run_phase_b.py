"""Phase B runner: GPT-2 Small ground-truth pipeline (design doc §8.1).

Runs ``bug types x seeds`` end to end on GPT-2 Small: real-text data
generation -> base training -> LoRA bug injection (+ sham control) ->
quality gates -> greedy repair-truth search (with optional small-pool
exhaustive verification) -> necessity truth -> SAE EV keep-rate check ->
full baseline catalogue with bootstrap 95% CIs -> fairness (repair vs.
necessity Kendall tau) -> sham-control comparison -> aggregate report.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path

from common.config import ExperimentConfig
from common.device import get_torch_device
from common.logging import setup_logging
from common.paths import checkpoint_dir, data_dir, results_dir
from common.seeding import set_seed
from evaluate.baselines import BASELINE_METHODS
from evaluate.runner import method_reports, run_all_baselines, sham_control_block
from evaluate.sae_check import sae_ev_report
from ground_truth.dnf import MAX_CONJUNCTS, DnfTruth
from ground_truth.repair_search import (
    component_f1,
    conjunct_f1,
    recover_dnf,
    single_component_judgments,
    union,
)
from inject_bugs.bugs import BugType
from inject_bugs.finetune_gpt2 import (
    GPT2TrainConfig,
    check_quality_gpt2,
    train_base_gpt2,
    train_injected_gpt2,
)
from inject_bugs.gpt2_data import (
    GPT2BugDataset,
    generate_gpt2_dataset,
    load_gpt2_dataset,
    save_gpt2_dataset,
)
from inject_bugs.gpt2_model import load_tl_gpt2
from inject_bugs.hooked_utils import component_keys, compute_mean_activations, key_str

logger = logging.getLogger(__name__)

N_METHODS_GATE = 7  # design doc §8.1 item 4


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _key_to_json(key) -> list:
    kind, layer, *rest = key
    if rest:
        return [kind, layer, rest[0]]
    return [kind, layer]


def _conjuncts_to_json(conjuncts) -> list:
    return [sorted(_key_to_json(k) for k in conj) for conj in conjuncts]


def _ranking_to_json(ranking) -> list:
    return [[key_str(k), float(s)] for k, s in ranking]


def _git_rev() -> str:
    """Short commit hash of the running code, or 'nogit' when unavailable."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def _analysis_fingerprint(*cfgs) -> str:
    """Hash of the config inputs that determine a seed's analysis results.

    Includes the current git commit so that any code change invalidates the
    per-seed analysis cache (previously a code edit with an unchanged config
    silently reused stale ``analysis.json`` results).
    """
    payload = json.dumps((_git_rev(),) + cfgs, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_analysis_cache(path: Path, fingerprint: str) -> dict | None:
    """Return a cached per-seed analysis result, or None when absent or stale."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("config_fingerprint") != fingerprint:
        logger.info("stale analysis cache %s (config changed), recomputing", path)
        return None
    result = cached.get("result")
    return result if isinstance(result, dict) else None


# ---------------------------------------------------------------------------
# data / pool helpers
# ---------------------------------------------------------------------------


def _load_or_build_dataset(
    bug: BugType, seed: int, samples: dict, cache_root: Path
) -> GPT2BugDataset:
    cache = cache_root / "phase_b" / f"{bug.value}_{seed}.pt"
    if cache.exists():
        logger.info("loading cached dataset %s", cache)
        return load_gpt2_dataset(cache)
    data = generate_gpt2_dataset(bug, seed, **samples)
    save_gpt2_dataset(data, cache)
    logger.info("generated dataset %s seed %d (seq_len=%d)", bug.value, seed, data.seq_len)
    return data


def _verification_pool(keys, judgments, truth_set, limit: int) -> list:
    """Truth components plus the strongest single-ablation components."""
    truth = [key for key in keys if key in truth_set]
    rest = [key for key in keys if key not in truth_set]
    rest.sort(key=lambda key: -abs(judgments[key].relative_drop))
    pool = truth + rest[: max(0, limit - len(truth))]
    return pool[:limit]


def _top_effects(judgments, truth_set, n: int = 10) -> list:
    """Strongest single-component effects, used when the repair truth is empty.

    Lists ``(component, relative_drop, trigger_rate, retention)`` for the
    components with the largest positive relative trigger drop.  This
    distinguishes "search did not run far enough" (a top effect near 1.0 that
    the greedy search missed) from "genuinely no repair set" (all effects far
    below the 0.80 repair threshold).
    """
    if truth_set:
        return []
    ranked = sorted(
        (k for k, j in judgments.items() if j.relative_drop > 0.0),
        key=lambda k: -judgments[k].relative_drop,
    )
    return [
        {
            "component": key_str(k),
            "relative_drop": judgments[k].relative_drop,
            "trigger_rate": judgments[k].trigger_rate,
            "retention": judgments[k].retention,
        }
        for k in ranked[:n]
    ]


# ---------------------------------------------------------------------------
# per (bug, seed) pipeline
# ---------------------------------------------------------------------------


def _run_seed(
    bug: BugType,
    seed: int,
    samples: dict,
    train_cfg: GPT2TrainConfig,
    gates: dict,
    search_cfg: dict,
    sae_cfg: dict,
    baseline_cfg: dict,
    fairness_cfg: dict,
    sham_cfg: dict,
    device,
    raw_tl,
    ckpt_root: Path,
    cache_root: Path,
) -> dict:
    t0 = time.perf_counter()
    set_seed(seed)
    run_dir = ckpt_root / "phase_b" / bug.value / str(seed)
    analysis_path = run_dir / "analysis.json"
    fingerprint = _analysis_fingerprint(
        samples,
        dataclasses.asdict(train_cfg),
        gates,
        search_cfg,
        sae_cfg,
        baseline_cfg,
        fairness_cfg,
        sham_cfg,
    )
    cached = _load_analysis_cache(analysis_path, fingerprint)
    if cached is not None:
        logger.info("seed analysis exists at %s, skipping", analysis_path)
        return cached

    data = _load_or_build_dataset(bug, seed, samples, cache_root)
    base_dir = train_base_gpt2(data, train_cfg, device, run_dir / "base")
    inj_dir = train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "injected")
    sham_dir = train_injected_gpt2(
        data, train_cfg, base_dir, device, run_dir / "sham", sham=True
    )

    base_tl = load_tl_gpt2(from_dir=base_dir / "model", device=device)
    injected_tl = load_tl_gpt2(from_dir=inj_dir / "model", device=device)
    sham_tl = load_tl_gpt2(from_dir=sham_dir / "model", device=device)

    quality = check_quality_gpt2(base_tl, injected_tl, sham_tl, data, seed=seed, gates=gates)
    logger.info(
        "quality %s seed %d: passed=%s trig=%.3f retention=%.3f base_trig=%.3f sham_trig=%.3f",
        bug.value,
        seed,
        quality.passed,
        quality.trigger_rate,
        quality.retention,
        quality.base_trigger_rate,
        quality.sham_trigger_rate,
    )

    keys = component_keys(injected_tl)
    means = compute_mean_activations(injected_tl, data.eval_normal, keys)
    base_rate = quality.trigger_rate

    # necessity truth: single-component mean ablation under the repair protocol
    judgments, n_stats = single_component_judgments(injected_tl, keys, means, data, base_rate)
    necessity = {key: judgment.success for key, judgment in judgments.items()}
    necessary = [key for key in keys if necessity[key]]
    logger.info(
        "necessity %s seed %d: %d/%d components necessary",
        bug.value,
        seed,
        len(necessary),
        len(keys),
    )

    # repair truth: greedy DNF recovery on the full 156-component space
    mode = search_cfg.get("mode", "greedy")
    max_conjuncts = int(search_cfg.get("max_conjuncts", MAX_CONJUNCTS))
    max_evals = int(search_cfg.get("max_evals") or 5000)
    early_stop = bool(search_cfg.get("early_stop", True))
    max_wall_s = (
        float(search_cfg["step_wall_budget_s"])
        if search_cfg.get("step_wall_budget_s")
        else None
    )
    conjuncts, search_stats = recover_dnf(
        injected_tl,
        keys,
        means,
        data,
        base_rate,
        mode=mode,
        max_conjuncts=max_conjuncts,
        max_evals=max_evals,
        early_stop=early_stop,
        max_wall_s=max_wall_s,
    )
    if search_stats.get("budget_exceeded"):
        logger.warning(
            "truth %s seed %d: greedy search hit eval budget (%d); "
            "aborted after %d evals, returning empty truth",
            bug.value,
            seed,
            max_evals,
            search_stats["n_evals"],
        )
    truth_set = set(union(conjuncts))
    dnf = DnfTruth(tuple(conjuncts))
    logger.info(
        "truth %s seed %d: %d conjuncts, %d components (%s)",
        bug.value,
        seed,
        len(conjuncts),
        len(truth_set),
        sorted(key_str(k) for k in truth_set),
    )

    # small-pool exhaustive verification: greedy must equal exhaustive.
    # Run it even when the full-space truth is empty, as long as some
    # component shows a positive single-ablation effect -- this tells us
    # whether the strongest signal is recoverable by exhaustive search on a
    # small pool (separating "search didn't run far enough" from "no repair").
    verify = None
    limit = int(search_cfg.get("exhaustive_pool_limit", 0) or 0)
    max_effect = max((abs(j.relative_drop) for j in judgments.values()), default=0.0)
    if limit > 0 and (truth_set or max_effect > 0.0):
        pool = _verification_pool(keys, judgments, truth_set, limit)
        gr_on_pool, gr_stats = recover_dnf(
            injected_tl,
            pool,
            means,
            data,
            base_rate,
            mode="greedy",
            max_conjuncts=1,
            max_evals=max_evals,
        )
        ex_on_pool, ex_stats = recover_dnf(
            injected_tl, pool, means, data, base_rate, mode="exhaustive", max_conjuncts=1
        )
        verify = {
            "pool": [key_str(k) for k in pool],
            "greedy_conjuncts": _conjuncts_to_json(gr_on_pool),
            "exhaustive_conjuncts": _conjuncts_to_json(ex_on_pool),
            "conjunct_f1": conjunct_f1(gr_on_pool, ex_on_pool),
            "component_f1": component_f1(set(union(gr_on_pool)), set(union(ex_on_pool))),
            "greedy_evals": gr_stats["n_evals"],
            "exhaustive_evals": ex_stats["n_evals"],
        }
        logger.info("exhaustive verify %s seed %d: f1=%.3f", bug.value, seed, verify["conjunct_f1"])

    # SAE explained-variance keep rate (raw vs injected, base-FT reference)
    ev_tokens = data.eval_normal[: int(sae_cfg.get("ev_sample", 256))]
    ev_report = sae_ev_report(
        raw_tl,
        injected_tl,
        ev_tokens,
        [int(layer) for layer in sae_cfg.get("layers", [])],
        device,
        base_tl=base_tl,
        release=sae_cfg.get("release"),
    )
    logger.info(
        "sae ev %s seed %d: keep_rate=%.4f gate_ok=%s",
        bug.value,
        seed,
        ev_report["mean_keep_rate"],
        ev_report["gate_ok"],
    )

    # baseline catalogue on the injected model
    sae_run_cfg = {
        "layers": [int(layer) for layer in sae_cfg.get("layers", [])],
        "top_k": int(sae_cfg.get("top_k", 20)),
        "ev_keep_rate_ok": bool(ev_report["gate_ok"]),
        "release": sae_cfg.get("release"),
    }
    methods = list(baseline_cfg.get("methods", BASELINE_METHODS))
    batch_size = int(baseline_cfg.get("batch_size", 128))
    results = run_all_baselines(
        injected_tl, data, means, device, methods, sae_run_cfg, batch_size=batch_size
    )
    reports = method_reports(
        results,
        keys,
        truth_set,
        dnf=dnf,
        k=int(baseline_cfg.get("k", 5)),
        n_boot=int(baseline_cfg.get("n_boot", 200)),
        seed=int(baseline_cfg.get("boot_seed", 0)),
        necessity=necessity,
        diff_gate=float(fairness_cfg.get("kendall_diff_gate", 0.10)),
    )
    reports_json = {
        name: {**report, "ranking": _ranking_to_json(report["ranking"])}
        for name, report in reports.items()
    }

    # sham control: same methods on the sham model, compared to the injected model
    sham_block = None
    sham_methods = [m for m in sham_cfg.get("methods", []) if m in methods]
    if sham_methods:
        sham_means = compute_mean_activations(sham_tl, data.eval_normal, keys)
        sham_results = run_all_baselines(
            sham_tl, data, sham_means, device, sham_methods, sae_run_cfg, batch_size=batch_size
        )
        sham_block = sham_control_block(
            results,
            sham_results,
            keys,
            truth_set,
            kendall_gate=float(sham_cfg.get("kendall_gate", 0.10)),
        )
        logger.info(
            "sham control %s seed %d: mean_gap=%.4f passed=%s",
            bug.value,
            seed,
            sham_block["mean_gap"],
            sham_block["passed"],
        )

    result = {
        "seed": seed,
        "quality": quality.to_dict(),
        "truth": {
            "search_mode": mode,
            "budget_exceeded": bool(search_stats.get("budget_exceeded", False)),
            "conjuncts": _conjuncts_to_json(conjuncts),
            "union": sorted(key_str(k) for k in truth_set),
            "search_evals": search_stats["n_evals"],
            "search_wall_s": search_stats["wall_s"],
            "search_steps": search_stats.get("steps", []),
            "budget_phase": search_stats.get("budget_phase"),
            "budget_restart": search_stats.get("budget_restart"),
            "top_effects_empty_truth": _top_effects(judgments, truth_set),
            "necessity": {
                "n_necessary": len(necessary),
                "components": [key_str(k) for k in necessary],
                "wall_s": n_stats["wall_s"],
            },
        },
        "exhaustive_verify": verify,
        "sae": dict(ev_report),
        "methods": reports_json,
        "sham": sham_block,
        "wall_s": time.perf_counter() - t0,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        json.dumps(
            {"config_fingerprint": fingerprint, "result": result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def _run_bug(
    bug: BugType,
    seeds: list[int],
    samples: dict,
    train_cfg: GPT2TrainConfig,
    gates: dict,
    search_cfg: dict,
    sae_cfg: dict,
    baseline_cfg: dict,
    fairness_cfg: dict,
    sham_cfg: dict,
    device,
    raw_tl,
    ckpt_root: Path,
    cache_root: Path,
) -> dict:
    per_seed = [
        _run_seed(
            bug,
            seed,
            samples,
            train_cfg,
            gates,
            search_cfg,
            sae_cfg,
            baseline_cfg,
            fairness_cfg,
            sham_cfg,
            device,
            raw_tl,
            ckpt_root,
            cache_root,
        )
        for seed in seeds
    ]
    return {
        "seeds": per_seed,
        "total_wall_s": sum(seed["wall_s"] for seed in per_seed),
    }


# ---------------------------------------------------------------------------
# acceptance aggregation
# ---------------------------------------------------------------------------


def _acceptance(report: dict) -> dict:
    seeds = [seed for bug_res in report["bugs"].values() for seed in bug_res["seeds"]]
    quality_ok = all(seed["quality"]["passed"] for seed in seeds)
    sham_blocks = [seed["sham"] for seed in seeds if seed["sham"] is not None]
    sham_ok = bool(sham_blocks) and all(block["passed"] for block in sham_blocks)
    sham_mean_gap = (
        sum(block["mean_gap"] for block in sham_blocks) / len(sham_blocks) if sham_blocks else 0.0
    )
    sham_gate = sham_blocks[0].get("gate") if sham_blocks else None
    sae_ok = all(
        seed["sae"]["gate_ok"]
        or seed["methods"].get("sae_topk_ablation", {}).get("degraded", False)
        for seed in seeds
    )
    sae_mean_keep = (
        sum(seed["sae"]["mean_keep_rate"] for seed in seeds) / len(seeds) if seeds else 0.0
    )
    baselines_ok = all(len(seed["methods"]) >= N_METHODS_GATE for seed in seeds)
    fairness_ok = True  # sanity item: significant repair/necessity diffs are reported, not gated
    n_significant = sum(
        1
        for seed in seeds
        for method in seed["methods"].values()
        if method.get("fairness", {}).get("significantly_different", False)
    )
    all_passed = bool(quality_ok and sham_ok and sae_ok and baselines_ok and fairness_ok)
    return {
        "quality": {
            "ok": quality_ok,
            "gate": {
                "trigger_rate": 0.90,
                "retention": 0.95,
                "base_trigger_ceiling": 0.05,
                "sham_trigger_ceiling": 0.10,
            },
        },
        "sham_control": {
            "ok": sham_ok,
            "gate": sham_gate,
            "mean_gap": sham_mean_gap,
        },
        "sae_ev": {
            "ok": sae_ok,
            "mean_keep_rate": sae_mean_keep,
            "gate": 0.90,
            "note": "degraded path (constrained-subspace SAE analysis) counts as ok",
        },
        "baselines": {
            "ok": baselines_ok,
            "n_methods_min": N_METHODS_GATE,
            "n_methods_max": max((len(seed["methods"]) for seed in seeds), default=0),
        },
        "fairness": {
            "ok": fairness_ok,
            "note": "sanity item; repair vs. necessity Kendall tau reported per method",
            "n_significantly_different": n_significant,
        },
        "ece_calibration": {
            "ok": None,
            "note": "deferred to the Phase B calibration stage (credibility report v1)",
        },
        "all_passed": all_passed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B GPT-2 ground-truth pipeline.")
    parser.add_argument("--config", default="configs/phase_b_gpt2.yaml")
    parser.add_argument("--bugs", default=None, help="Comma-separated bug subset.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--report", default=None, help="Output JSON path (default: results/).")
    parser.add_argument(
        "--exhaustive-pool-limit",
        type=int,
        default=None,
        help="Override the exhaustive-verification pool size (0 disables).",
    )
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    setup_logging(config.logging.level)
    device = get_torch_device(config.device)
    logger.info("device: %s", device)

    bug_names = list(getattr(config, "bugs", [b.value for b in BugType.all()]))
    if args.bugs:
        bug_names = [name.strip() for name in args.bugs.split(",") if name.strip()]
    bugs = [BugType(name) for name in bug_names]

    seeds = list(getattr(config, "seeds", [1, 2, 3]))
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    samples = dict(getattr(config, "samples", {}))
    gates = dict(getattr(config, "quality_gates", {}))
    training = dict(getattr(config, "training", {}))
    fields = {f.name for f in dataclasses.fields(GPT2TrainConfig)}
    train_cfg = GPT2TrainConfig(**{key: value for key, value in training.items() if key in fields})
    search_cfg = dict(getattr(config, "search", {}))
    if args.exhaustive_pool_limit is not None:
        search_cfg["exhaustive_pool_limit"] = args.exhaustive_pool_limit
    sae_cfg = dict(getattr(config, "sae", {}))
    baseline_cfg = dict(getattr(config, "baselines", {}))
    fairness_cfg = dict(getattr(config, "fairness", {}))
    sham_cfg = dict(getattr(config, "sham", {}))
    model_name = str(getattr(config, "model", "gpt2"))

    ckpt_root = checkpoint_dir()
    cache_root = data_dir()

    layers = [int(layer) for layer in sae_cfg.get("layers", [])]
    raw_tl = load_tl_gpt2(device=device) if layers else None

    report: dict = {
        "phase": "B",
        "model": model_name,
        "git_rev": _git_rev(),
        "config": {
            "bugs": [bug.value for bug in bugs],
            "seeds": seeds,
            "samples": samples,
            "training": training,
            "search": search_cfg,
            "sae": sae_cfg,
            "baselines": baseline_cfg,
            "fairness": fairness_cfg,
            "sham": sham_cfg,
        },
        "component_space": "156 components (12 layers x 12 heads + 12 MLPs)",
        "bugs": {},
        "acceptance": {},
    }

    t_start = time.perf_counter()
    for bug in bugs:
        report["bugs"][bug.value] = _run_bug(
            bug,
            seeds,
            samples,
            train_cfg,
            gates,
            search_cfg,
            sae_cfg,
            baseline_cfg,
            fairness_cfg,
            sham_cfg,
            device,
            raw_tl,
            ckpt_root,
            cache_root,
        )
    report["timing_total"] = {"wall_s": time.perf_counter() - t_start}
    report["acceptance"] = _acceptance(report)

    report_path = Path(args.report) if args.report else results_dir() / "phase_b_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("PHASE B SUMMARY")
    print("=" * 78)
    for bug_name, bug_res in report["bugs"].items():
        for seed_report in bug_res["seeds"]:
            quality = seed_report["quality"]
            truth = seed_report["truth"]
            print(
                f"{bug_name} seed {seed_report['seed']}: quality(passed={quality['passed']}) "
                f"trig={quality['trigger_rate']:.3f} retention={quality['retention']:.3f} "
                f"| truth={truth['union']} | necessity={truth['necessity']['n_necessary']} "
                f"| sae_keep={seed_report['sae']['mean_keep_rate']:.3f}"
            )
    print(f"acceptance: {json.dumps(report['acceptance'], indent=2)}")
    print(f"total wall: {report['timing_total']['wall_s']:.1f}s")
    print(f"report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
