"""Step 3/4 有效子集报告汇总分析（修订研究方案 §5.3 / §6 Step 3）。

输入：``scripts/run_phase_b.py --config configs/phase_b_step3.yaml`` 产出的报告
      （默认 ``results/phase_b_step3_report.json``）。
输出：文本汇总 + 可选 ``--out`` JSON，含
  1. 每 bug 跨 seed 的修复真值 union 与 pairwise/平均 IoU（Step 3 门槛 >= 0.6）；
  2. 每 bug × 方法 的跨 seed 指标表（auprc / hit@1 / hit@5，mean + min-max CI）；
  3. 每 bug 方法排名（按 auprc 均值），附随机/完美基线对照参考。

用法：
  C:/Python314/python.exe scripts/analyze_step3.py results/phase_b_step3_report.json
      [--out results/phase_b_step3_summary.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HEADLINE = ["auprc", "hit_at_1", "hit_at_5"]


def _set_union(seeds: list[dict]) -> set[str]:
    return set().union(*[set(s["truth"]["union"]) for s in seeds])


def _iou(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cross_seed_iou(bug_res: dict) -> dict:
    seeds = bug_res["seeds"]
    unions = [_set_union([s]) for s in seeds]
    pairs = [(i, j) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    pairwise = {
        f"s{seeds[i]['seed']}-s{seeds[j]['seed']}": round(_iou(unions[i], unions[j]), 3)
        for i, j in pairs
    }
    mean = sum(pairwise.values()) / len(pairwise) if pairwise else 0.0
    return {
        "per_seed_union": [sorted(u) for u in unions],
        "pairwise_iou": pairwise,
        "mean_iou": round(mean, 3),
        "passes_gate": bool(mean >= 0.6),
    }


def _method_table(bug_res: dict) -> dict:
    """Cross-seed aggregate headline metrics per method."""
    seeds = bug_res["seeds"]
    methods = sorted(seeds[0]["methods"])
    table: dict[str, dict] = {}
    for m in methods:
        rows = [s["methods"][m] for s in seeds]
        entry = {"degraded_any": any(r.get("degraded", False) for r in rows)}
        for metric in HEADLINE:
            vals = [r["metrics"][metric]["value"] for r in rows]
            # ci 是 [lo, hi] 列表（见 evaluate/runner.py::_percentile_ci）
            ci_lo = min(r["metrics"][metric]["ci"][0] for r in rows)
            ci_hi = max(r["metrics"][metric]["ci"][1] for r in rows)
            entry[metric] = {
                "mean": round(sum(vals) / len(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "ci_low": round(ci_lo, 3),
                "ci_high": round(ci_hi, 3),
            }
        table[m] = entry
    return table


def summarize(report: dict) -> dict:
    bugs: dict[str, dict] = {}
    for bug_name, bug_res in report["bugs"].items():
        bugs[bug_name] = {
            "truth_iou": _cross_seed_iou(bug_res),
            "methods": _method_table(bug_res),
            "quality_passed": all(s["quality"]["passed"] for s in bug_res["seeds"]),
        }
    return {"git_rev": report.get("git_rev"), "bugs": bugs}


def _fmt_table(bugs: dict) -> str:
    lines = []
    for bug, b in bugs.items():
        iou = b["truth_iou"]
        lines.append(f"== {bug}  quality_passed={b['quality_passed']}  "
                     f"truth_mean_iou={iou['mean_iou']} (gate>=0.6)  "
                     f"pass={iou['passes_gate']}")
        lines.append(f"   unions: {iou['per_seed_union']}")
        lines.append(f"   pairwise_iou: {iou['pairwise_iou']}")
        lines.append("   {:<28} {:>8} {:>8} {:>8}  {}".format(
            "method", "auprc", "hit@1", "hit@5", "degraded"))
        for m, e in sorted(b["methods"].items(),
                           key=lambda kv: -kv[1]["auprc"]["mean"]):
            lines.append("   {:<28} {:>8} {:>8} {:>8}  {}".format(
                m, e["auprc"]["mean"], e["hit_at_1"]["mean"], e["hit_at_5"]["mean"],
                "yes" if e["degraded_any"] else "-"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="run_phase_b.py 输出的报告 JSON")
    parser.add_argument("--out", default=None, help="可选：汇总 JSON 输出路径")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"!! 报告不存在: {report_path}", file=sys.stderr)
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = summarize(report)
    print(_fmt_table(summary["bugs"]))
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
