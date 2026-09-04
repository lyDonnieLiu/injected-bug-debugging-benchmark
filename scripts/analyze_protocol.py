"""A — protocol-ablation read-out (next_step_research_plan.md v3 A 判读门).

Consumes ``scripts/run_phase_b_protocol.py``'s report and answers whether the
repair truth is robust to the ablation protocol.  For each bug the
pre-registered core set (report config ``core``) is compared against the
recovered truth union under each intervention, per seed, via core IoU
(``set_iou``).  Verdict per bug:

* ``robust``             -- every (intervention, seed) core IoU >= gate
* ``protocol_dependent`` -- interventions disagree (at least one passes, one fails)
* ``none_pass``          -- no intervention reaches the gate

Pure CPU, consumes the report only.  Usage:
    python scripts/analyze_protocol.py results/phase_b_protocol_report.json \\
        [--out results/phase_b_protocol_summary.json] [--gate 0.6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ground_truth.repair_search import set_iou

GATE = 0.6


def _parse_component(s: str) -> tuple:
    """``head(11,2)`` -> ``("head", 11, 2)``; ``mlp(11)`` -> ``("mlp", 11)``."""
    kind, rest = s.split("(", 1)
    rest = rest.rstrip(")")
    if "," in rest:
        layer, head = rest.split(",", 1)
        return (kind, int(layer), int(head))
    return (kind, int(rest))


def _to_keys(strings: list[str]) -> set:
    return {_parse_component(s) for s in strings}


def _verdict(passed: dict[str, bool]) -> str:
    if passed and all(passed.values()):
        return "robust"
    if passed and any(passed.values()):
        return "protocol_dependent"
    return "none_pass"


def summarize(report: dict, gate: float = GATE) -> dict:
    core = report.get("config", {}).get("core", {})
    bugs: dict[str, dict] = {}
    for record in report.get("points", []):
        bug = record["bug"]
        entry = bugs.setdefault(bug, {"seeds": {}})
        seed = record["seed"]
        iv = record["intervention"]
        union_str = record.get("truth", {}).get("union", [])
        iou = set_iou(_to_keys(union_str), _to_keys(core.get(bug, [])))
        entry["seeds"].setdefault(seed, {})[iv] = {
            "core_iou": round(iou, 4),
            "union": union_str,
            "budget_exceeded": record.get("truth", {}).get("budget_exceeded", False),
        }

    for bug in bugs.values():
        by_iv: dict[str, list[float]] = {}
        for seed in bug["seeds"].values():
            for iv, v in seed.items():
                by_iv.setdefault(iv, []).append(v["core_iou"])
        min_by_iv = {iv: min(vals) for iv, vals in by_iv.items()}
        passed = {iv: min_iou >= gate for iv, min_iou in min_by_iv.items()}
        bug["core_iou_by_intervention"] = min_by_iv
        bug["passed_by_intervention"] = passed
        bug["verdict"] = _verdict(passed)
    return {"git_rev": report.get("git_rev"), "gate": gate, "core": core, "bugs": bugs}


def _fmt_table(summary: dict) -> str:
    lines = ["A 协议消融判读汇总", "=" * 100]
    for name, bug in summary["bugs"].items():
        core = sorted(summary["core"].get(name, []))
        lines.append(f"\n[{name}]  core = {{{', '.join(core)}}}")
        for seed in sorted(bug["seeds"]):
            row = bug["seeds"][seed]
            cells = []
            for iv, v in sorted(row.items()):
                flag = "!" if v["budget_exceeded"] else " "
                cells.append(f"{iv}={v['core_iou']:.3f}{flag}")
            lines.append(f"  seed {seed}: " + "  ".join(cells))
        pass_str = " ".join(
            f"{iv}={ok}" for iv, ok in sorted(bug["passed_by_intervention"].items())
        )
        lines.append(f"  pass(>= {summary['gate']}): {pass_str}")
        lines.append(f"  verdict: {bug['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A protocol-ablation closure read-out.")
    parser.add_argument("report", help="run_phase_b_protocol.py 输出的报告 JSON")
    parser.add_argument("--out", default=None, help="可选：汇总 JSON 输出路径")
    parser.add_argument("--gate", type=float, default=GATE,
                        help=f"core IoU 通过门槛（默认 {GATE}）")
    parser.add_argument("--core", default=None,
                        help="JSON 核心集覆盖，如 '{\"compositional_logic\": "
                             "[\"head(11,2)\", \"mlp(11)\"]}'（默认读报告 config.core）")
    args = parser.parse_args(argv)

    path = Path(args.report)
    if not path.exists():
        print(f"!! 报告不存在: {path}", file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    if args.core:
        report.setdefault("config", {})["core"] = json.loads(args.core)
    summary = summarize(report, gate=args.gate)
    print(_fmt_table(summary))
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsummary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
