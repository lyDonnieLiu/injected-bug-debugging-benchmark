"""C2 — random fake-truth null model (next_step_research_plan.md v3 §C2).

Consumes ``results/phase_b_step3_report.json`` (per-seed method ``ranking``) and
answers, for ``numeric_rule``, whether a method's stored AUPRC against the real
(over-decomposed) union is driven by the single true core ``mlp(9)`` or inflated
by a generic tendency to surface *arbitrary* components.

For each method x seed we keep the method's **fixed** ranking and re-score it
against ``n_groups`` random "fake truths" of the same size ``k`` as the real
union, in two strata:

* ``without_core`` — a random size-``k`` set drawn from the 155 non-``mlp(9)``
  components: the pure-random outer-circle inflation floor (answers "how much
  AUPRC does the method get on *any* random outer set").
* ``with_core`` — ``{mlp(9)}`` plus a random (``k``-1)-set: isolates how much of
  the score the core alone explains (rationale: degraded SAE AUPRC 0.57 vs
  hit@1=1.0 — is 0.57 just "nailed the one core, random on the rest"?).

AUPRC is computed with the same :func:`evaluate.metrics.auprc` used for the
stored value, so null and real values are directly comparable.  The analytic
uniform-random-ranking expectation (``metrics.random_baseline``) is reported as
a floor reference.  Pure CPU.

Usage:
    python scripts/analyze_random_truth.py results/phase_b_step3_report.json \\
        [--core mlp(9)] [--bug numeric_rule] [--n 100] [--seed 0] \\
        [--out results/phase_b_c2_random_truth.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from evaluate.metrics import auprc, random_baseline

DEFAULT_BUG = "numeric_rule"
DEFAULT_CORE = "mlp(9)"


def _cell_rng(base_seed: int, *parts) -> random.Random:
    """Deterministic per-(method, seed, stratum) RNG, independent of iteration order."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(base_seed + int(digest[:8], 16))


def _scores_from_ranking(ranking: list) -> tuple[list[str], list[float]]:
    """Return (keys, scores) in the stored descending order."""
    keys = [str(key) for key, _score in ranking]
    scores = [float(score) for _key, score in ranking]
    return keys, scores


def _auprc_against(scores: list[float], keys: list[str], positives: set[str]) -> float:
    labels = [1 if key in positives else 0 for key in keys]
    return auprc(scores, labels)


def _null_distribution(
    keys: list[str],
    scores: list[float],
    core: str,
    k: int,
    n_groups: int,
    rng: random.Random,
) -> dict[str, dict]:
    """AUPRC null distributions for the ``without_core`` / ``with_core`` strata."""
    non_core = [key for key in keys if key != core]
    without: list[float] = []
    with_core: list[float] = []
    for _ in range(n_groups):
        sample = rng.sample(non_core, k)
        without.append(_auprc_against(scores, keys, set(sample)))
        sample = rng.sample(non_core, k - 1)
        with_core.append(_auprc_against(scores, keys, set(sample) | {core}))

    def _stat(values: list[float]) -> dict:
        values.sort()
        return {
            "mean": round(sum(values) / len(values), 4),
            "ci": [round(values[2], 4), round(values[-3], 4)],  # 2.5 / 97.5 percentile
        }

    return {"without_core": _stat(without), "with_core": _stat(with_core)}


def _verdict(real: float, without_mean: float, with_mean: float) -> str:
    """Three-way read from the two deltas.

    ``core_delta = with_core - without_core`` is how much forcing ``mlp(9)`` in
    alone raises the score; ``outer_delta = real - with_core`` is how much the
    *specific* outer union members add beyond ``mlp(9)`` + a random outer set.
    """
    core_delta = with_mean - without_mean
    outer_delta = real - with_mean
    if core_delta <= 0.02:
        return "no_core_signal"    # mlp(9) not reliably recovered (real ~ random floor)
    if outer_delta > core_delta:
        return "core_plus_outer"   # specific outer members carry recoverable signal
    return "core_driven"           # score explained by the core alone


def summarize(report: dict, bug: str, core: str, n_groups: int, base_seed: int) -> dict:
    bug_res = report.get("bugs", {}).get(bug)
    if bug_res is None:
        raise SystemExit(f"!! bug {bug!r} not in report")

    seeds = bug_res.get("seeds", [])
    methods = sorted({m for sd in seeds for m in (sd.get("methods") or {})})

    # analytic random-ranking floor per union size k
    floor_by_k: dict[int, float] = {}
    for sd in seeds:
        k = len(sd.get("truth", {}).get("union", []))
        n_components = len(next(iter((sd.get("methods") or {}).values())).get("ranking", []))
        if n_components:
            floor_by_k[k] = round(
                random_baseline(n_components, k, 3)["auprc"], 4
            )

    out_methods: dict[str, dict] = {}
    for method in methods:
        per_seed: dict[str, dict] = {}
        real_all: list[float] = []
        without_all: list[float] = []
        with_all: list[float] = []
        for sd in seeds:
            seed = int(sd["seed"])
            entry = (sd.get("methods") or {}).get(method)
            if entry is None:
                continue
            ranking = entry.get("ranking", [])
            keys, scores = _scores_from_ranking(ranking)
            union = set(sd.get("truth", {}).get("union", []))
            k = len(union)
            real = round(_auprc_against(scores, keys, union), 4)
            rng = _cell_rng(base_seed, method, seed)
            nulls = _null_distribution(keys, scores, core, k, n_groups, rng)
            stored = (entry.get("metrics") or {}).get("auprc")
            stored_val = stored.get("value") if isinstance(stored, dict) else stored
            per_seed[str(seed)] = {
                "k": k,
                "real_auprc": real,
                "stored_auprc": round(float(stored_val), 4) if stored_val is not None else None,
                "without_core": nulls["without_core"],
                "with_core": nulls["with_core"],
            }
            real_all.append(real)
            without_all.append(nulls["without_core"]["mean"])
            with_all.append(nulls["with_core"]["mean"])

        if not real_all:
            continue
        real_mean = sum(real_all) / len(real_all)
        without_mean = sum(without_all) / len(without_all)
        with_mean = sum(with_all) / len(with_all)
        out_methods[method] = {
            "per_seed": per_seed,
            "cross_seed": {
                "real_auprc_mean": round(real_mean, 4),
                "without_core_mean": round(without_mean, 4),
                "with_core_mean": round(with_mean, 4),
            },
            "verdict": _verdict(real_mean, without_mean, with_mean),
        }

    return {
        "git_rev": report.get("git_rev"),
        "params": {
            "bug": bug,
            "core": core,
            "n_groups": n_groups,
            "base_seed": base_seed,
        },
        "random_ranking_floor_by_k": floor_by_k,
        "methods": out_methods,
    }


def _fmt_table(summary: dict) -> str:
    core = summary["params"]["core"]
    floor = summary["random_ranking_floor_by_k"]
    floor_s = " ".join(f"k{k}={v}" for k, v in sorted(floor.items()))
    lines = [
        f"C2 随机假真值 null 模型（core={core}；AUPRC）",
        "=" * 100,
        f"随机排序 analytic 地板: {floor_s}",
    ]
    for method, entry in summary["methods"].items():
        cs = entry["cross_seed"]
        lines.append(
            f"\n[{method}]  verdict={entry['verdict']}  "
            f"real={cs['real_auprc_mean']}  "
            f"without_core={cs['without_core_mean']}  "
            f"with_core={cs['with_core_mean']}"
        )
        for seed, row in entry["per_seed"].items():
            wc = row["without_core"]
            w = row["with_core"]
            lines.append(
                f"  s{seed} k={row['k']}: real={row['real_auprc']} "
                f"(stored={row['stored_auprc']}) "
                f"| 不含core={wc['mean']} [{wc['ci'][0]},{wc['ci'][1]}] "
                f"| 含core={w['mean']} [{w['ci'][0]},{w['ci'][1]}]"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C2 random fake-truth null model.")
    parser.add_argument("report", help="run_phase_b.py 输出的报告 JSON（含逐 seed ranking）")
    parser.add_argument("--bug", default=DEFAULT_BUG, help=f"bug 名（默认 {DEFAULT_BUG}）")
    parser.add_argument("--core", default=DEFAULT_CORE, help=f"核心组件（默认 {DEFAULT_CORE}）")
    parser.add_argument("--n", type=int, default=100, dest="n_groups",
                        help="每组抽样次数（默认 100）")
    parser.add_argument("--seed", type=int, default=0, dest="base_seed", help="分层种子（默认 0）")
    parser.add_argument("--out", default=None, help="可选：结果 JSON 输出路径")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"!! 报告不存在: {report_path}", file=sys.stderr)
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = summarize(report, args.bug, args.core, args.n_groups, args.base_seed)
    print(_fmt_table(summary))
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsummary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
