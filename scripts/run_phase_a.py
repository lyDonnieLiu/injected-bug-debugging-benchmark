"""Phase A runner: toy transformer ground-truth pipeline (design doc §8.1).

Runs ``bug types x seeds`` end to end: data generation -> base training ->
masked bug injection -> quality gates -> exhaustive truth search -> greedy
search -> DNF recovery -> metric sanity -> aggregate report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

from common.config import ExperimentConfig
from common.device import get_torch_device
from common.logging import setup_logging
from common.paths import results_dir
from common.seeding import set_seed
from evaluate.metrics import compute_metrics, perfect_ranking, random_baseline, random_ranking
from ground_truth.dnf import DnfTruth
from ground_truth.repair_search import (
    component_f1,
    conjunct_f1,
    recall,
    recover_dnf,
    set_iou,
    union,
)
from inject_bugs.bugs import BugType
from inject_bugs.data_generation import generate_dataset
from inject_bugs.finetune import (
    InjectionConfig,
    check_quality,
    inject_bug,
    train_base_model,
    train_sham_model,
)
from inject_bugs.toy_model import component_keys, key_str

logger = logging.getLogger(__name__)

METRIC_SANITY_TOL = 0.10
RECALL_GATE = 0.95
IOU_GATE = 0.80
GREEDY_F1_GATE = 0.90

def _perfect_upper(m: int, k: int) -> dict[str, float]:
    """Metric upper bounds achieved by a perfect ranking (top-``k``)."""
    return {
        "hit_at_k": 1.0,
        "auprc": 1.0,
        "auroc": 1.0,
        "spearman": 1.0,
        "kendall": 1.0,
        "topk_iou": min(m, k) / max(m, k) if m else 0.0,
        "ndcg": 1.0,
    }


def _as_key(spec: list) -> tuple:
    kind, layer, *rest = spec
    if rest:
        return (kind, layer, rest[0])
    return (kind, layer)


def _conjunct_specs(spec: dict) -> tuple[frozenset, list[frozenset]]:
    implanted = frozenset(_as_key(c) for c in spec.get("implanted", []))
    expected = [frozenset(_as_key(c) for c in conj) for conj in spec.get("conjuncts", [])]
    return implanted, expected


def _key_to_json(key) -> list:
    kind, layer, *rest = key
    if rest:
        return [kind, layer, rest[0]]
    return [kind, layer]


def _conjuncts_to_json(conjuncts) -> list:
    return [sorted(_key_to_json(k) for k in conj) for conj in conjuncts]


def _flatten_metrics(metrics: dict) -> dict:
    flat: dict = {}
    for name, value in metrics.items():
        if name == "rank_correlation" and isinstance(value, dict):
            flat.update(value)  # spearman / kendall keys match the baseline dict
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                flat[f"{name}.{sub}"] = sub_value
        else:
            flat[name] = value
    return flat


RANDOM_RANKINGS_N = 64  # random sanity averages over this many rankings


def _metric_sanity(keys, ex_conjuncts, seed: int) -> dict:
    truth = set(union(ex_conjuncts))
    dnf = DnfTruth(tuple(ex_conjuncts))
    k = min(5, len(keys))
    perfect = compute_metrics(perfect_ranking(keys, truth), truth, dnf=dnf, k=k)
    random_flat = {}
    for r in range(RANDOM_RANKINGS_N):
        metrics = compute_metrics(random_ranking(keys, seed=seed * 1000 + r), truth, dnf=dnf, k=k)
        flat = _flatten_metrics(metrics)
        for name, value in flat.items():
            random_flat[name] = random_flat.get(name, 0.0) + float(value) / RANDOM_RANKINGS_N
    baseline = random_baseline(len(keys), len(truth), k)
    upper = _perfect_upper(len(truth), k)
    flat_perfect = _flatten_metrics(perfect)
    random_ok = all(
        abs(random_flat.get(name, 0.0) - expected) <= METRIC_SANITY_TOL
        for name, expected in baseline.items()
    )
    perfect_ok = all(
        abs(float(flat_perfect.get(name, -1.0)) - expected) <= 1e-6
        for name, expected in upper.items()
    )
    return {
        "k": k,
        "truth_size": len(truth),
        "baseline": baseline,
        "random": random_flat,
        "perfect": flat_perfect,
        "random_ok": bool(random_ok),
        "perfect_ok": bool(perfect_ok),
    }


def _run_bug(bug, seeds, model_cfg, samples, train_cfg, s_star, expected):
    bug_name = bug.value
    per_seed = []
    total_wall = 0.0
    search_wall = 0.0
    recovered_unions = []
    for seed in seeds:
        t0 = time.perf_counter()
        set_seed(seed)
        data = generate_dataset(bug, seed, **samples)
        base = train_base_model(model_cfg, seed, data, train_cfg)
        model, means = inject_bug(base, bug, data, s_star, train_cfg, seed)
        sham = train_sham_model(base, bug, data, s_star, train_cfg, seed)
        quality = check_quality(bug, base, model, sham, data, means, seed=seed)
        if not quality.passed:
            logger.warning("quality gate failed for %s seed %s", bug_name, seed)

        keys = component_keys(model)
        base_rate = quality.trigger_rate
        ex_conjuncts, ex_stats = recover_dnf(model, keys, means, data, base_rate, mode="exhaustive")
        gr_conjuncts, gr_stats = recover_dnf(model, keys, means, data, base_rate, mode="greedy")
        ex_union = set(union(ex_conjuncts))
        gr_union = set(union(gr_conjuncts))
        rec = recall(ex_union, s_star)
        f1 = conjunct_f1(gr_conjuncts, ex_conjuncts)
        comp_f1 = component_f1(gr_union, ex_union)
        sanity = _metric_sanity(keys, ex_conjuncts, seed)
        wall = time.perf_counter() - t0
        total_wall += wall
        search_wall += float(ex_stats["wall_s"]) + float(gr_stats["wall_s"])
        recovered_unions.append(ex_union)
        per_seed.append(
            {
                "seed": seed,
                "quality": quality.to_dict(),
                "exhaustive_conjuncts": _conjuncts_to_json(ex_conjuncts),
                "greedy_conjuncts": _conjuncts_to_json(gr_conjuncts),
                "recall": rec,
                "greedy_vs_exhaustive_f1": f1,
                "greedy_vs_exhaustive_component_f1": comp_f1,
                "exhaustive_wall_s": ex_stats["wall_s"],
                "greedy_wall_s": gr_stats["wall_s"],
                "exhaustive_evals": ex_stats["n_evals"],
                "greedy_evals": gr_stats["n_evals"],
                "metric_sanity": sanity,
                "wall_s": wall,
            }
        )
        logger.info(
            "%s seed %d: quality=%s recall=%.3f f1=%.3f conj=%s",
            bug_name,
            seed,
            quality.passed,
            rec,
            f1,
            _conjuncts_to_json(ex_conjuncts),
        )

    pair_ious = [
        set_iou(recovered_unions[i], recovered_unions[j])
        for i in range(len(recovered_unions))
        for j in range(i + 1, len(recovered_unions))
    ]
    iou_mean = sum(pair_ious) / len(pair_ious) if pair_ious else 1.0
    recalls = [s["recall"] for s in per_seed]
    f1_values = [s["greedy_vs_exhaustive_f1"] for s in per_seed]
    dnf_matched = all(
        set(frozenset(_as_key(c) for c in conj) for conj in s["exhaustive_conjuncts"])
        == set(expected)
        for s in per_seed
    )
    return {
        "implanted_truth": sorted(key_str(k) for k in s_star),
        "expected_conjuncts": _conjuncts_to_json(expected),
        "seeds": per_seed,
        "recall_mean": sum(recalls) / len(recalls) if recalls else 0.0,
        "iou_across_seeds": iou_mean,
        "greedy_f1_mean": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "dnf_matched_all_seeds": bool(dnf_matched),
        "total_wall_s": total_wall,
        "search_wall_s": search_wall,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase A toy ground-truth pipeline.")
    parser.add_argument("--config", default="configs/phase_a_toy.yaml")
    parser.add_argument("--report", default=None, help="Output JSON path (default: results/).")
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    setup_logging(config.logging.level)
    device = get_torch_device(config.device)
    logger.info("device: %s", device)

    model_cfg = dict(getattr(config, "model", {}))
    bug_names = list(
        getattr(config, "bugs", ["trigger_backdoor", "compositional_logic", "knowledge_conflict"])
    )
    bugs = [BugType(name) for name in bug_names]
    n_seeds = int(getattr(config, "n_seeds", 3))
    base_seed = int(getattr(config, "seed", 0))
    seeds = list(getattr(config, "seeds", [base_seed + i + 1 for i in range(n_seeds)]))
    samples = dict(getattr(config, "samples", {}))
    training = dict(getattr(config, "training", {}))
    fields = {f.name for f in dataclasses.fields(InjectionConfig)}
    train_cfg = InjectionConfig(**{k: v for k, v in training.items() if k in fields})
    bug_specs = dict(getattr(config, "bug_specs", {}))

    report: dict = {
        "phase": "A",
        "config": {
            "model": model_cfg,
            "bugs": [b.value for b in bugs],
            "seeds": seeds,
            "samples": samples,
            "training": training,
        },
        "component_space": "8 heads + 2 MLPs",
        "bugs": {},
        "timing_total": {},
        "acceptance": {},
    }

    t_start = time.perf_counter()
    for bug in bugs:
        spec = bug_specs.get(bug.value, {})
        s_star, expected = _conjunct_specs(spec)
        report["bugs"][bug.value] = _run_bug(
            bug, seeds, model_cfg, samples, train_cfg, s_star, expected
        )
    total_wall = time.perf_counter() - t_start
    report["timing_total"] = {"wall_s": total_wall, "cpu_hours": total_wall / 3600.0}

    recall_ok = all(report["bugs"][b]["recall_mean"] >= RECALL_GATE for b in report["bugs"])
    iou_ok = all(report["bugs"][b]["iou_across_seeds"] >= IOU_GATE for b in report["bugs"])
    f1_ok = all(report["bugs"][b]["greedy_f1_mean"] >= GREEDY_F1_GATE for b in report["bugs"])
    matched = {b: report["bugs"][b]["dnf_matched_all_seeds"] for b in report["bugs"]}
    dnf_ok = matched.get("compositional_logic", False) and sum(matched.values()) >= 2
    sanity_random_ok = all(
        all(s["metric_sanity"]["random_ok"] for s in report["bugs"][b]["seeds"])
        for b in report["bugs"]
    )
    sanity_perfect_ok = all(
        all(s["metric_sanity"]["perfect_ok"] for s in report["bugs"][b]["seeds"])
        for b in report["bugs"]
    )
    all_passed = bool(
        recall_ok and iou_ok and f1_ok and dnf_ok and sanity_random_ok and sanity_perfect_ok
    )
    report["acceptance"] = {
        "truth_recall": {"ok": recall_ok, "gate": RECALL_GATE},
        "cross_seed_iou": {"ok": iou_ok, "gate": IOU_GATE},
        "greedy_f1": {"ok": f1_ok, "gate": GREEDY_F1_GATE},
        "dnf_recovery": {"ok": dnf_ok, "min_bugs_with_dnf": 2, "matched": matched},
        "metric_sanity_random": {"ok": bool(sanity_random_ok), "tolerance": METRIC_SANITY_TOL},
        "metric_sanity_perfect": {"ok": bool(sanity_perfect_ok)},
        "timing_recorded": {"ok": True, "note": "CPU wall-clock times"},
        "all_passed": all_passed,
    }

    report_path = Path(args.report) if args.report else results_dir() / "phase_a_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("PHASE A ACCEPTANCE SUMMARY")
    print("=" * 78)
    for bug_name, bug_res in report["bugs"].items():
        first = bug_res["seeds"][0]
        print(
            f"{bug_name}: recall={bug_res['recall_mean']:.3f} "
            f"iou={bug_res['iou_across_seeds']:.3f} "
            f"greedy_f1={bug_res['greedy_f1_mean']:.3f} "
            f"dnf_match={bug_res['dnf_matched_all_seeds']}"
        )
        print(
            f"  quality(passed={first['quality']['passed']}): "
            f"trig={first['quality']['trigger_rate']:.3f} "
            f"retention={first['quality']['retention']:.3f} "
            f"base_trig={first['quality']['base_trigger_rate']:.3f} "
            f"sham_trig={first['quality']['sham_trigger_rate']:.3f}"
        )
        print(f"  truth: {first['exhaustive_conjuncts']} | greedy: {first['greedy_conjuncts']}")
    print(f"acceptance: {json.dumps(report['acceptance'], indent=2)}")
    print(f"total wall: {total_wall:.1f}s = {total_wall / 3600.0:.4f} CPU hours")
    print(f"report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())