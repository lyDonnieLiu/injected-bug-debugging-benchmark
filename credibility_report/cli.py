"""CLI skeleton for the credibility report tool (design doc §7).

Usage::

    python -m credibility_report.cli --config configs/default.yaml --model gpt2 --task ioi
    python -m credibility_report.cli --config configs/cloud_gpu.yaml --json-out report.json

The check implementations are placeholders; this validates config loading,
device resolution, logging and JSON output end to end.
"""

from __future__ import annotations

import argparse
import json
import sys

from common.config import ExperimentConfig
from common.device import resolve_device
from common.logging import setup_logging
from credibility_report.checks import CheckItem, CheckStatus
from credibility_report.report import CheckResult, CredibilityGrade, CredibilityReport

PLACEHOLDER_DETAIL = "placeholder — check not implemented yet (skeleton)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credibility-report",
        description="Credibility report tool for attribution experiments (design doc §7).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML experiment config.",
    )
    parser.add_argument("--model", default="gpt2", help="Model identifier.")
    parser.add_argument("--task", default="ioi", help="Task identifier.")
    parser.add_argument("--layer", type=int, default=None, help="Target layer (optional).")
    parser.add_argument("--k", type=int, default=10, help="Top-k used by the attribution result.")
    parser.add_argument("--json-out", default=None, help="Also write the report JSON to this path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    setup_logging(config.logging.level)
    device = resolve_device(config.device)
    print(f"[skeleton] resolved device: {device} (config: {args.config})")

    report = CredibilityReport(model_name=args.model, task=args.task, layer=args.layer, k=args.k)
    report.checks = [
        CheckResult(item=item, status=CheckStatus.WARN, detail=PLACEHOLDER_DETAIL)
        for item in CheckItem.all()
    ]
    report.grade = CredibilityGrade.D

    payload = report.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())