"""Step 3/4 有效子集全量结果分析（修订研究方案 §5.5 / §6 Step 4）。

输入：``results/phase_b_step3_report.json``（run_phase_b.py 全量产出）。
输出：文本解读 + 可选 ``--out`` JSON，含
  1. 每 bug × 方法 的跨 seed 指标（auprc / hit@1 / hit@5 / auroc / kendall 等，
     mean + min-max CI），并按 auprc 均值排序；
  2. sham 对照：每 bug × 方法 的 injected/sham Kendall gap；
  3. SAE EV：每 bug 的 keep-rate / gate / degraded；
  4. 随机/完美基线对照（仅解读，不写回 report）。

用法：
  C:/Python314/python.exe scripts/analyze_step3_full.py results/phase_b_step3_report.json
      [--out results/phase_b_step3_full_summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 前 3 个指标带 bootstrap CI（{value, ci}），其余为裸 float
HEADLINE = ["auprc", "hit_at_1", "hit_at_5"]
BARE = ["auroc", "ndcg", "topk_iou", "kendall", "spearman"]


def _set_union(seeds: list[dict]) -> set[str]:
    return set().union(*[set(s["truth"]["union"]) for s in seeds])


def _method_table(bug_res: dict) -> dict:
    seeds = bug_res["seeds"]
    methods = sorted(seeds[0]["methods"])
    table: dict[str, dict] = {}
    for m in methods:
        rows = [s["methods"][m] for s in seeds]
        entry: dict = {"degraded_any": any(r.get("degraded", False) for r in rows)}
        for metric in HEADLINE:
            vals = [r["metrics"][metric]["value"] for r in rows]
            ci_lo = min(r["metrics"][metric]["ci"][0] for r in rows)
            ci_hi = max(r["metrics"][metric]["ci"][1] for r in rows)
            entry[metric] = {
                "mean": round(sum(vals) / len(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "ci_low": round(ci_lo, 3),
                "ci_high": round(ci_hi, 3),
            }
        for metric in BARE:
            vals = [r["metrics"][metric] for r in rows]
            entry[metric] = {
                "mean": round(sum(vals) / len(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }
        table[m] = entry
    return table


def _sham_table(bug_res: dict) -> dict:
    seeds = bug_res["seeds"]
    methods = sorted(seeds[0]["sham"]["per_method"])
    table: dict[str, dict] = {}
    for m in methods:
        rows = [s["sham"]["per_method"][m] for s in seeds]
        table[m] = {
            "injected_kendall_mean": round(sum(r["injected_kendall"] for r in rows) / len(rows), 3),
            "sham_kendall_mean": round(sum(r["sham_kendall"] for r in rows) / len(rows), 3),
            "gap_mean": round(sum(r["gap"] for r in rows) / len(rows), 3),
        }
    return table


def _sae_table(bug_res: dict) -> dict:
    seeds = bug_res["seeds"]
    keep_rates = [s["sae"]["mean_keep_rate"] for s in seeds]
    gate_ok_any = any(s["sae"]["gate_ok"] for s in seeds)
    return {
        "mean_keep_rate_mean": round(sum(keep_rates) / len(keep_rates), 3),
        "min_keep_rate": round(min(keep_rates), 3),
        "max_keep_rate": round(max(keep_rates), 3),
        "gate_ok_any": bool(gate_ok_any),
        "gate": seeds[0]["sae"]["gate"],
    }


def summarize(report: dict) -> dict:
    bugs: dict[str, dict] = {}
    for bug_name, bug_res in report["bugs"].items():
        bugs[bug_name] = {
            "truth_union": sorted(_set_union(bug_res["seeds"])),
            "methods": _method_table(bug_res),
            "sham": _sham_table(bug_res),
            "sae": _sae_table(bug_res),
            "quality_passed": all(s["quality"]["passed"] for s in bug_res["seeds"]),
        }
    return {
        "git_rev": report.get("git_rev"),
        "acceptance_all_passed": report.get("acceptance", {}).get("all_passed"),
        "bugs": bugs,
    }


def _fmt_table(summary: dict) -> str:
    lines = []
    acc = summary.get("acceptance_all_passed")
    lines.append(f"acceptance.all_passed = {acc}")
    for bug, b in summary["bugs"].items():
        lines.append(f"== {bug}  quality_passed={b['quality_passed']}  "
                     f"truth_union={b['truth_union']}")
        lines.append(f"   SAE: mean_keep_rate={b['sae']['mean_keep_rate_mean']} "
                     f"gate_ok_any={b['sae']['gate_ok_any']}")
        lines.append("   {:<28} {:>8} {:>8} {:>8} {:>8} {:>8}  {}".format(
            "method", "auprc", "hit@1", "hit@5", "auroc", "kendall", "degraded"))
        for m, e in sorted(b["methods"].items(),
                           key=lambda kv: -kv[1]["auprc"]["mean"]):
            lines.append("   {:<28} {:>8} {:>8} {:>8} {:>8} {:>8}  {}".format(
                m, e["auprc"]["mean"], e["hit_at_1"]["mean"], e["hit_at_5"]["mean"],
                e["auroc"]["mean"], e["kendall"]["mean"],
                "yes" if e["degraded_any"] else "-"))
        lines.append("   sham gap (injected-sham kendall, mean):")
        for m, e in sorted(b["sham"].items(),
                           key=lambda kv: -kv[1]["gap_mean"]):
            lines.append("     {:<28} inj={:>7} sham={:>7} gap={:>7}".format(
                m, e["injected_kendall_mean"], e["sham_kendall_mean"], e["gap_mean"]))
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
    print(_fmt_table(summary))
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
