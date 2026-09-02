"""A' negative-endpoint 结果判读（next_step_research_plan.md v3 A' 判读门）.

Aggregates ``scripts/run_phase_b_negpair.py``'s report JSON into the closure
checks the plan pre-registered:

* **pair mode (seed 1)**: was the two-component pool *fully* enumerated
  (judged == total, not budget-aborted) and did any pair repair?
  ``all_empty`` = no two-component minimal repair set exists.
* **single mode (seeds 2-3)**: is the suppressor pattern (``mlp(0/1/2)``)
  and the "no non-suppressor single lever" result consistent across seeds?

Outputs a compact text + JSON summary (per bug x seed x mode).  Pure CPU,
consumes the report only.  Usage:
    python scripts/analyze_negpair.py results/phase_b_negpair_report.json
        [--out results/phase_b_negpair_summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANONICAL_SUPPRESSORS = {"mlp(0)", "mlp(1)", "mlp(2)"}


def summarize(report: dict) -> dict:
    bugs: dict[str, dict] = {}
    for record in report.get("points", []):
        bug = bugs.setdefault(record["bug"], {"pair": [], "single": []})
        mode = record["mode"]
        scan = record.get("pair_scan") or record.get("single_scan") or {}
        quality = record.get("quality", {})
        bug[mode].append(
            {
                "seed": record["seed"],
                "label": record["label"],
                "quality_passed": bool(quality.get("passed")),
                **(
                    {
                        "pool_size": scan.get("pool_size"),
                        "n_pairs_total": scan.get("n_pairs_total"),
                        "n_pairs_judged": scan.get("n_pairs_judged"),
                        "n_repair_pairs": scan.get("n_repair_pairs"),
                        "repair_pairs": scan.get("repair_pairs") or [],
                        "budget_exceeded": bool(scan.get("budget_exceeded")),
                    }
                    if mode == "pair"
                    else {
                        "suppressors": scan.get("suppressors") or [],
                        "n_suppressors": scan.get("n_suppressors"),
                        "necessary_non_suppressor": scan.get("necessary_non_suppressor") or [],
                        "n_necessary_non_suppressor": scan.get("n_necessary_non_suppressor"),
                        "top_non_suppressor_effects": scan.get("top_non_suppressor_effects") or [],
                    }
                ),
            }
        )

    # closure decisions per bug
    for modes in bugs.values():
        pair = sorted(modes["pair"], key=lambda r: r["seed"])
        single = sorted(modes["single"], key=lambda r: r["seed"])
        full_scan = all(
            r.get("n_pairs_judged") == r.get("n_pairs_total") and not r.get("budget_exceeded")
            for r in pair
        ) and bool(pair)
        modes["decision"] = {
            "pair_seeds": [r["seed"] for r in pair],
            "pair_fully_scanned": full_scan,
            "pair_all_empty": full_scan and all(r["n_repair_pairs"] == 0 for r in pair),
            "single_seeds": [r["seed"] for r in single],
            "single_suppressors_consistent": all(
                set(r["suppressors"]) == CANONICAL_SUPPRESSORS for r in single
            ) and bool(single),
            "single_no_lever": all(r["n_necessary_non_suppressor"] == 0 for r in single),
        }
    return {
        "git_rev": report.get("git_rev"),
        "protocol_version": report.get("protocol_version"),
        "canonical_suppressors": sorted(CANONICAL_SUPPRESSORS),
        "bugs": bugs,
    }


def _fmt_table(summary: dict) -> str:
    lines = ["A' negative-endpoint 判读汇总", "=" * 100]
    for bug, modes in summary["bugs"].items():
        dec = modes["decision"]
        lines.append(f"\n[{bug}]")
        lines.append(
            f"  pair(seed {dec['pair_seeds']}): 全量扫描={dec['pair_fully_scanned']} "
            f"全空={dec['pair_all_empty']}"
        )
        lines.append(
            f"  single(seed {dec['single_seeds']}): 抑制器一致(mlp0/1/2)="
            f"{dec['single_suppressors_consistent']} 无单组件杠杆={dec['single_no_lever']}"
        )
        for mode in ("pair", "single"):
            for rec in modes[mode]:
                if mode == "pair":
                    lines.append(
                        f"    s{rec['seed']}: judged={rec['n_pairs_judged']}/"
                        f"{rec['n_pairs_total']} repairs={rec['n_repair_pairs']} "
                        f"budget={rec['budget_exceeded']}"
                    )
                else:
                    lines.append(
                        f"    s{rec['seed']}: suppressors={sorted(rec['suppressors'])} "
                        f"necessary_non_supp={rec['necessary_non_suppressor']}"
                    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A' negative-endpoint closure read-out.")
    parser.add_argument("report", help="run_phase_b_negpair.py 输出的报告 JSON")
    parser.add_argument("--out", default=None, help="可选：汇总 JSON 输出路径")
    args = parser.parse_args(argv)

    path = Path(args.report)
    if not path.exists():
        print(f"!! 报告不存在: {path}", file=sys.stderr)
        return 1
    summary = summarize(json.loads(path.read_text(encoding="utf-8")))
    print(_fmt_table(summary))
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsummary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
