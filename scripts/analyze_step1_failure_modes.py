"""Step 2 失败模式分类报告（修订研究方案 §5.1 / §6 Step 2）。

输入：``results/step1_report.json``（run_phase_b_step1.py 的 40 点注入矩阵诊断）。
输出：文本矩阵表 + 可选 ``--out`` JSON，含
  1. 5 类 bug × 8 注入配置 的 failure_mode 矩阵
     （destructive_suppressor / strict / effect_available / pure_and / ...）；
  2. 每点 typology 细节（n_strict_necessary / n_effect / n_non_destructive /
     n_suppressors / general_suppressors / strongest_clean / n_truth_components）；
  3. 每点 quality 与 truth 摘要（passed / union 长度 / budget_exceeded）；
  4. 每 bug 的 failure-mode 计数汇总。

用法：
  C:/Python314/python.exe scripts/analyze_step1_failure_modes.py results/step1_report.json
      [--out results/step1_failure_modes.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUGS = ["trigger_backdoor", "knowledge_conflict", "format_rule",
        "numeric_rule", "compositional_logic"]


def summarize(report: dict) -> dict:
    bugs: dict[str, dict] = {}
    for bug in BUGS:
        pts = [p for p in report["points"] if p["bug"] == bug]
        bugs[bug] = {"points": []}
        for p in pts:
            ty = p["typology"]
            fm = ty["failure_mode"]
            bugs[bug]["points"].append({
                "label": p["label"],
                "quality_passed": p["quality"]["passed"],
                "trigger_rate": p["quality"]["trigger_rate"],
                "retention": p["quality"]["retention"],
                "truth_union": p["truth"]["union"],
                "n_union": len(p["truth"]["union"]),
                "budget_exceeded": p["truth"]["budget_exceeded"],
                "failure_mode": fm["name"],
                "n_strict_necessary": ty["n_strict_necessary"],
                "n_effect": ty["n_effect"],
                "n_non_destructive": ty["n_non_destructive"],
                "n_suppressors": ty["n_suppressors"],
                "general_suppressors": ty["general_suppressors"],
                "strongest_clean": fm["strongest_clean"],
                "n_truth_components": fm["n_truth_components"],
            })
    return {"git_rev": report.get("git_rev"), "bugs": bugs}


def _fmt_table(summary: dict) -> str:
    lines = []
    labels = [pt["label"] for pt in summary["bugs"]["trigger_backdoor"]["points"]]
    lines.append("Step 2 失败模式矩阵（seed 1）")
    lines.append("   {:<22} ".format("config") + " ".join(f"{b[:10]:>11}" for b in BUGS))
    for i, lab in enumerate(labels):
        row = [f"   {lab:<22}"]
        for bug in BUGS:
            fm = summary["bugs"][bug]["points"][i]["failure_mode"]
            row.append(f"{fm[:11]:>11}")
        lines.append(" ".join(row))
    lines.append("")
    for bug, b in summary["bugs"].items():
        lines.append(f"== {bug}")
        counts: dict[str, int] = {}
        for pt in b["points"]:
            counts[pt["failure_mode"]] = counts.get(pt["failure_mode"], 0) + 1
        lines.append(f"   failure-mode 计数: {counts}")
        for pt in b["points"]:
            if pt["failure_mode"] == "strict":
                lines.append(f"   {pt['label']}: strict  union={pt['truth_union']} "
                             f"n_strict_necessary={pt['n_strict_necessary']} "
                             f"n_effect={pt['n_effect']} suppressors={pt['general_suppressors']}")
            elif pt["failure_mode"] == "effect_available":
                lines.append(f"   {pt['label']}: effect_available  effect="
                             f"{pt['strongest_clean']} n_non_destructive={pt['n_non_destructive']} "
                             f"suppressors={pt['general_suppressors']}")
            else:
                lines.append(f"   {pt['label']}: {pt['failure_mode']}  "
                             f"n_suppressors={pt['n_suppressors']} "
                             f"suppressors={pt['general_suppressors']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="step1_report.json 路径")
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
