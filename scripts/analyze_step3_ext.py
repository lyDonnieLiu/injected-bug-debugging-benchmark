"""C1 主表成员规则重算 (next_step_research_plan.md v3 "主表成员规则").

Pure-CPU analysis over an existing ``run_phase_b``-style report JSON (e.g.
``results/phase_b_step3_report.json`` -- per-seed ``methods[*].ranking`` and
``sham.per_method[*].gap`` are already persisted).  No retraining.

Every method row is re-classified with the pre-registered member rule from
``configs/phase_b_step3_ext.yaml``:

* **主表成员** -- the bug's core (all-of) recall holds within ``top_k`` on
  *every* seed AND the sham gap clears ``>= sham_gate`` on every seed.  This
  deliberately differs from the stored ``hit_at_k`` (any-of >= 1) semantics.
* **协议参照行** -- ``mean_ablation`` (+ merged ``acdc``), always listed but
  annotated circular: the repair truth was built under that same protocol, so
  it is never counted as a "localizer" member.
* **附录** -- everything else, with the exact per-seed violation (a core
  member's rank, the failing sham gap) so instability is visible (e.g. CL
  ``activation_patching`` seed 2: head(11,2)=136 / mlp(11)=156, sham gap
  -0.009).  Stored AUPRC / Hit@k means are carried for the union-appendix
  reference.

Usage:
    python scripts/analyze_step3_ext.py results/phase_b_step3_report.json
        [--rule configs/phase_b_step3_ext.yaml]
        [--out results/phase_b_step3_ext_summary.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from evaluate.core_recall import classify_member

DEFAULT_RULE = "configs/phase_b_step3_ext.yaml"


def _load_rule(path: Path) -> dict:
    """Load the pre-registered member rule; tolerate a pydantic config wrapper."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("member_rule", data)


def _metric_mean(seed_methods: list[dict], metric: str) -> float | None:
    """Mean of one stored metric across seeds (e.g. auprc / hit_at_1)."""
    values = []
    for sm in seed_methods:
        metrics = (sm.get("metrics") or {}).get(metric)
        value = metrics.get("value") if isinstance(metrics, dict) else metrics
        if value is not None:
            values.append(float(value))
    return round(sum(values) / len(values), 3) if values else None


def _seed_records(seeds: list[dict], method: str) -> list[dict]:
    """Per-seed ``{seed, ranking, sham_gap}`` records for one method."""
    records = []
    for sd in seeds:
        entry = (sd.get("methods") or {}).get(method)
        if entry is None:
            continue
        sham_gap = None
        if isinstance(sd.get("sham"), dict):
            sham_gap = sd["sham"].get("per_method", {}).get(method, {}).get("gap")
        records.append(
            {
                "seed": int(sd["seed"]),
                "ranking": entry.get("ranking", []),
                "sham_gap": sham_gap,
            }
        )
    return records


def _summarize_bug(bug_name: str, bug_res: dict, rule: dict) -> dict:
    seeds = bug_res.get("seeds", [])
    spec = (rule.get("bugs") or {}).get(bug_name, {})
    core = [str(c) for c in spec.get("core", [])]
    top_k = int(spec.get("top_k", 0))
    sham_gate = float(rule.get("sham_gate", 0.10))
    reference = {
        ref["method"]: ref for ref in rule.get("reference_rows", [])
        if isinstance(ref, dict)
    }
    merge_src = {
        merged: ref["method"]
        for ref in rule.get("reference_rows", [])
        for merged in ref.get("merges", [])
    }

    methods = sorted({m for sd in seeds for m in (sd.get("methods") or {})})
    members: list[str] = []
    reference_rows: list[dict] = []
    appendix: dict[str, dict] = {}

    # reference family = canonical reference method(s) + anything they merge
    # (e.g. acdc merges into mean_ablation).  A family member is never counted
    # as a localizer member and never appears in the appendix.
    ref_names = {r["method"] for r in rule.get("reference_rows", [])}
    ref_family = set(ref_names) | set(merge_src)

    for method in methods:
        if method in ref_family:
            if method in ref_names:  # only the canonical row is emitted
                ref = reference[method]
                reference_rows.append(
                    {
                        "method": method,
                        "merges": ref.get("merges", []),
                        "note": ref.get("note", "protocol reference row"),
                        "per_seed": [
                            {
                                "seed": rec["seed"],
                                "core_ranks": classify_member(
                                    [rec], core, top_k, sham_gate
                                )["per_seed"][0]["core_ranks"],
                                "sham_gap": rec["sham_gap"],
                            }
                            for rec in _seed_records(seeds, method)
                        ],
                    }
                )
            continue

        records = _seed_records(seeds, method)
        if not records:
            continue

        cls = classify_member(records, core, top_k, sham_gate)
        if cls["decision"] == "member":
            members.append(method)
        else:
            appendix[method] = {
                "decision": cls["decision"],
                "reason": cls["reason"],
                "per_seed": cls["per_seed"],
                "metrics_ref": {
                    "auprc_mean": _metric_mean(
                        [(sd.get("methods") or {}).get(method, {}) for sd in seeds],
                        "auprc",
                    ),
                    "hit_at_1_mean": _metric_mean(
                        [(sd.get("methods") or {}).get(method, {}) for sd in seeds],
                        "hit_at_1",
                    ),
                },
            }

    return {
        "main_members": sorted(members),
        "reference_rows": reference_rows,
        "appendix": appendix,
    }


def summarize(report: dict, rule: dict) -> dict:
    bugs = {}
    for bug_name, bug_res in (report.get("bugs") or {}).items():
        bugs[bug_name] = _summarize_bug(bug_name, bug_res, rule)
    return {
        "git_rev": report.get("git_rev"),
        "report_source": report.get("config", {}).get("seeds"),
        "member_rule": rule,
        "bugs": bugs,
    }


def _fmt_table(summary: dict) -> str:
    lines = ["C1 主表成员规则重算（core recall 全含 + sham gap，逐 seed）", "=" * 100]
    for bug, b in summary["bugs"].items():
        spec = summary["member_rule"].get("bugs", {}).get(bug, {})
        lines.append(f"\n[{bug}]  core={spec.get('core')} top_k={spec.get('top_k')}")
        for method in b["main_members"]:
            lines.append(f"  ** 主表成员: {method}")
        for ref in b["reference_rows"]:
            merges = f" (+ {'/'.join(ref['merges'])})" if ref["merges"] else ""
            lines.append(f"  -- 协议参照: {ref['method']}{merges}  [circular; not a localizer]")
        lines.append("  -- 附录（含逐 seed 违反细节）:")
        for method, entry in b["appendix"].items():
            reason = entry["reason"] or entry["decision"]
            lines.append(f"     {method:<28} {reason}")
            for row in entry["per_seed"]:
                ranks = ", ".join(f"{c}={r}" for c, r in row["core_ranks"].items())
                gap = row["sham_gap"]
                gap_s = "n/a" if gap is None else f"{gap:+.3f}"
                lines.append(
                    f"       s{row['seed']}: core({ranks}) recall={row['core_recall']} "
                    f"sham_gap={gap_s}"
                )
            ref = entry.get("metrics_ref") or {}
            lines.append(
                f"       [appendix AUPRC mean={ref.get('auprc_mean')} "
                f"hit@1 mean={ref.get('hit_at_1_mean')}]"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C1 main-table member-rule recompute.")
    parser.add_argument("report", help="run_phase_b.py 输出的报告 JSON")
    parser.add_argument("--rule", default=DEFAULT_RULE, help="预注册成员规则 YAML")
    parser.add_argument("--out", default=None, help="可选：结果 JSON 输出路径")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"!! 报告不存在: {report_path}", file=sys.stderr)
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rule = _load_rule(Path(args.rule))
    summary = summarize(report, rule)
    print(_fmt_table(summary))
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsummary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
