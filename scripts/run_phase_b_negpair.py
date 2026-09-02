"""A' negative-endpoint driver: TB/KC/FR pair closure + cross-seed scan (plan v3 A').

negative 端闭环 (next_step_research_plan.md v3 A')：TB/KC/FR 在 ll8-9-10-11 下
单组件扫描无修复杠杆（唯一强效应是破坏性通用 MLP 抑制器），本驱动回答
"该注入几何下是否存在*任意两组件*的最小修复集"以及该结论是否跨 seed 稳定。

两种模式（每种模式是同一 (bug, seed, point) 训练 checkpoint 上的纯搜索）：

* ``pair``（seed 1 主枚举，默认）：先跑单组件判定，按 bug 实时排除
  general_suppressors（mlp(0/1/2) 型），再对剩余 ~153 组件穷举两两配对逐对
  mean-ablation 判定。空结果 + 全量扫描 = "无两组件修复集"的证据；任一配对
  命中即 early-stop 记录该配对（→ 叙事② 收窄 + 扩 seeds 复核）。
* ``single``（seeds 2-3 跨 seed 复核）：只跑单组件判定，记录抑制器集合与
  "无单组件杠杆"（非抑制器里无任何 necessary 组件 + top 非抑制器效应远低于
  0.80 修复阈），与 seed 1 的抑制器模式对照。跳过 greedy DNF（省 GPU）。

本驱动沿用 step1 的逐 (bug, seed, point) 隔离 checkpoint 约定；不调用
``run_phase_b._run_seed``。每点训练 base/injected/sham、过 quality gates，
随后按模式搜索并落盘逐点 record（含 protocol fingerprint，用于断点续跑的
"同点同指纹才算完成"判定）。驱动不做 SAE/sham 基线目录（那些属于方法评估）。

Usage:
    IBB_DEVICE=cuda:0 uv run python scripts/run_phase_b_negpair.py \\
        --config configs/phase_b_negpair.yaml --mode pair --seeds 1 \\
        --report results/phase_b_negpair_report.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from common.config import ExperimentConfig
from common.device import get_torch_device
from common.logging import setup_logging
from common.paths import checkpoint_dir, data_dir, results_dir
from common.seeding import set_seed
from ground_truth.repair_search import pair_repair_search, single_component_judgments
from ground_truth.truth_typology import general_mlp_suppressors
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

PROTOCOL_VERSION = "phase_b_negpair_v1"
MODES = ("pair", "single")
DEFAULT_TM = ("c_attn", "c_proj", "c_fc")


# ---------------------------------------------------------------------------
# matrix / point helpers (mirrors run_phase_b_step1 so reports read the same)
# ---------------------------------------------------------------------------


def _tm_hash(target_modules: tuple[str, ...]) -> str:
    payload = ",".join(target_modules).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:4]


def point_label(point: dict) -> str:
    """Stable short label for one matrix point, e.g. ``ll[8-11].r8.t2c3c9``."""
    layers = point.get("lora_layers", [])
    ll = "ll" + "-".join(str(int(layer)) for layer in layers) if layers else "all"
    rank = "r" + str(int(point.get("rank", 8)))
    tm = point.get("target_modules", ("c_attn", "c_proj", "c_fc"))
    return f"{ll}.{rank}.t{_tm_hash(tuple(tm))}"


def point_key(point: dict) -> dict:
    return {
        "lora_layers": list(point.get("lora_layers", [])),
        "rank": int(point.get("rank", 8)),
        "target_modules": list(point.get("target_modules", ("c_attn", "c_proj", "c_fc"))),
    }


def _identity_tuple(point: dict) -> tuple:
    return (
        tuple(point.get("lora_layers", [])),
        int(point.get("rank", 8)),
        tuple(point.get("target_modules", ("c_attn", "c_proj", "c_fc"))),
    )


def _dedupe_points(points: list[dict]) -> list[dict]:
    """Keep the first occurrence of each (lora_layers, rank, target_modules)."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for point in points:
        identity = _identity_tuple(point)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(point)
    return out


def _point_dir(ckpt_root: Path, bug: BugType, seed: int, label: str) -> Path:
    return ckpt_root / "phase_b_negpair" / bug.value / str(seed) / label


# ---------------------------------------------------------------------------
# dataset / training helpers (per (bug, seed, point), point-isolated checkpoints)
# ---------------------------------------------------------------------------


def _dataset(bug: BugType, seed: int, samples: dict, cache_root: Path) -> GPT2BugDataset:
    cache = cache_root / "phase_b" / f"{bug.value}_{seed}.pt"
    if cache.exists():
        logger.info("loading cached dataset %s", cache)
        return load_gpt2_dataset(cache)
    data = generate_gpt2_dataset(bug, seed, **samples)
    save_gpt2_dataset(data, cache)
    logger.info("generated dataset %s seed %d (seq_len=%d)", bug.value, seed, data.seq_len)
    return data


def _train_config(point: dict, base: dict) -> GPT2TrainConfig:
    fields = {f.name for f in dataclasses.fields(GPT2TrainConfig)}
    merged = {key: value for key, value in base.items() if key in fields}
    merged["rank"] = int(point.get("rank", merged.get("rank", 8)))
    merged["lora_layers"] = tuple(int(layer) for layer in point.get("lora_layers", []))
    merged["target_modules"] = tuple(point.get("target_modules", ("c_attn", "c_proj", "c_fc")))
    return GPT2TrainConfig(**merged)


def _json_component_key(key) -> list:
    kind, layer, *rest = key
    return [kind, layer, rest[0]] if rest else [kind, layer]


def _point_fingerprint(bug, seed, point, mode, samples, training, gates, search_cfg) -> str:
    """Protocol identity + full training/search config digest for one point.

    Folds the current git commit in (like ``run_phase_b``'s analysis cache) so
    that *any* driver code change invalidates every previously reported point:
    a record is only accepted on resume when both its identity and its
    fingerprint match the running code.  List axes (target_matrices / window)
    are order-invariant.
    """
    from common.fingerprint import protocol_fingerprint

    identity = protocol_fingerprint(
        protocol_version=PROTOCOL_VERSION,
        git_rev=_git_rev(),
        bug=bug.value if isinstance(bug, BugType) else str(bug),
        seed=seed,
        intervention="mean_ablation",
        rank=int(point.get("rank", 8)),
        target_matrices=list(point.get("target_modules", DEFAULT_TM)),
        window=list(point.get("lora_layers", [])) or None,
    )
    payload = json.dumps(
        (identity, mode, samples, training, gates, search_cfg),
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("report %s unreadable; starting fresh", path)
        return []
    return list(payload.get("points", [])) if isinstance(payload, dict) else payload


def _run_identity(bug: BugType, seed: int, point: dict, mode: str) -> tuple:
    """Hashable identity of one (bug, seed, point, mode) run."""
    return (
        bug.value if isinstance(bug, BugType) else str(bug),
        int(seed),
        mode,
        tuple(point.get("lora_layers", []) or []),
        int(point.get("rank", 8)),
        tuple(point.get("target_modules", DEFAULT_TM)),
    )


def _record_identity(record: dict) -> tuple:
    """Hashable identity of one stored record (mirrors ``_run_identity``)."""
    point = record.get("point") or {}
    return (
        record.get("bug"),
        int(record.get("seed") or -1),
        record.get("mode"),
        tuple(point.get("lora_layers", []) or []),
        int(point.get("rank", 8)),
        tuple(point.get("target_modules", DEFAULT_TM)),
    )


def _prune_stale(
    existing: list[dict],
    bugs: list[BugType],
    seeds: list[int],
    matrix: list[dict],
    mode: str,
    samples: dict,
    training: dict,
    gates: dict,
    search_cfg: dict,
) -> list[dict]:
    """Drop records for identities this run will (re)run under an old fingerprint.

    ``git_rev`` is part of the point fingerprint, so a driver code change makes
    every prior point stale.  Without this prune a resumed report would keep
    both the stale record and the fresh rerun of the same point (duplicate /
    outdated closure evidence).  Records whose identity is *not* part of this
    invocation (other bugs / seeds / modes) are left untouched.
    """
    wanted_fp: dict[tuple, str] = {}
    for bug in bugs:
        for point in matrix:
            train_dict = dataclasses.asdict(_train_config(point, training))
            for seed in seeds:
                identity = _run_identity(bug, seed, point, mode)
                wanted_fp[identity] = _point_fingerprint(
                    bug, seed, point, mode, samples, train_dict, gates, search_cfg
                )
    kept = []
    for record in existing:
        identity = _record_identity(record)
        current = wanted_fp.get(identity)
        if current is not None and record.get("config_fingerprint") != current:
            logger.info(
                "prune stale record %s s%s %s (code/config changed)",
                record.get("bug"), record.get("seed"), record.get("mode"),
            )
            continue
        kept.append(record)
    return kept


def _already_done(existing: list[dict], bug: BugType, seed: int, mode: str,
                  point: dict, fingerprint: str) -> bool:
    bug_name = bug.value if isinstance(bug, BugType) else str(bug)
    want = point_key(point)
    for record in existing:
        if (record.get("bug") == bug_name and record.get("seed") == seed
                and record.get("mode") == mode and record.get("point") == want
                and record.get("config_fingerprint") == fingerprint):
            return True
    return False


# ---------------------------------------------------------------------------
# per (bug, seed, point, mode) search
# ---------------------------------------------------------------------------


def _run_point(
    bug: BugType,
    seed: int,
    mode: str,
    label: str,
    point: dict,
    samples: dict,
    train_cfg: GPT2TrainConfig,
    gates: dict,
    search_cfg: dict,
    device,
    cache_root: Path,
    ckpt_root: Path,
) -> dict:
    t0 = time.perf_counter()
    set_seed(seed)
    run_dir = _point_dir(ckpt_root, bug, seed, label)
    fingerprint = _point_fingerprint(
        bug, seed, point, mode, samples,
        dataclasses.asdict(train_cfg), gates, search_cfg,
    )

    data = _dataset(bug, seed, samples, cache_root)
    base_dir = train_base_gpt2(data, train_cfg, device, run_dir / "base")
    inj_dir = train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "injected")
    train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "sham", sham=True)

    base_tl = load_tl_gpt2(from_dir=base_dir / "model", device=device)
    injected_tl = load_tl_gpt2(from_dir=inj_dir / "model", device=device)

    quality = check_quality_gpt2(
        base_tl, injected_tl,
        load_tl_gpt2(from_dir=run_dir / "sham" / "model", device=device),
        data, seed=seed, gates=gates,
    )
    logger.info("quality %s s%d %s: passed=%s trig=%.3f ret=%.3f",
                bug.value, seed, mode, quality.passed,
                quality.trigger_rate, quality.retention)

    keys = component_keys(injected_tl)
    means = compute_mean_activations(injected_tl, data.eval_normal, keys)
    base_rate = quality.trigger_rate

    # single-component effect map (feeds suppressor exclusion in both modes)
    judgments, n_stats = single_component_judgments(
        injected_tl, keys, means, data, base_rate
    )
    suppressors = set(general_mlp_suppressors(keys, judgments))
    judgments_json = [
        {
            "component": key_str(key),
            "trigger_rate": judgment.trigger_rate,
            "retention": judgment.retention,
            "relative_drop": judgment.relative_drop,
            "success": judgment.success,
        }
        for key, judgment in judgments.items()
    ]

    base: dict[str, Any] = {
        "bug": bug.value,
        "seed": seed,
        "mode": mode,
        "point": point_key(point),
        "label": label,
        "quality": quality.to_dict(),
        "config_fingerprint": fingerprint,
    }

    if mode == "single":
        # cross-seed复核: suppressor identity + "no single-component lever" check.
        non_supp = [k for k in keys if k not in suppressors]
        necessary = [k for k in keys if judgments[k].success]
        lever = [key_str(k) for k in necessary if k not in suppressors]
        strong = sorted(
            (k for k in non_supp if judgments[k].relative_drop > 0.0),
            key=lambda k: -judgments[k].relative_drop,
        )[:5]
        summary = {
            "n_components": len(keys),
            "suppressors": sorted(key_str(k) for k in suppressors),
            "n_suppressors": len(suppressors),
            "n_strict_necessary": len(necessary),
            "n_necessary_non_suppressor": len(lever),
            "necessary_non_suppressor": sorted(lever),
            "top_non_suppressor_effects": [
                {
                    "component": key_str(k),
                    "relative_drop": judgments[k].relative_drop,
                    "trigger_rate": judgments[k].trigger_rate,
                    "retention": judgments[k].retention,
                }
                for k in strong
            ],
            "n_evals": n_stats["n_evals"],
            "wall_s": round(n_stats["wall_s"], 2),
        }
        base["single_scan"] = summary
        base["single_component_judgments"] = judgments_json
        base["wall_s"] = round(time.perf_counter() - t0, 2)
        logger.info("single-scan %s s%d: suppressors=%s necessary_non_supp=%d",
                    bug.value, seed, summary["suppressors"], len(lever))
        return base

    if mode == "pair":
        # main enumeration: exclude this bug's general suppressors, then scan pairs.
        pool = [k for k in keys if k not in suppressors]
        exclude = sorted(key_str(k) for k in suppressors)
        max_evals = int(search_cfg.get("max_evals", 15000))
        max_wall_s = (
            float(search_cfg["step_wall_budget_s"])
            if search_cfg.get("step_wall_budget_s")
            else None
        )
        pair_limit = search_cfg.get("pair_limit")
        if pair_limit:
            max_evals = min(max_evals, int(pair_limit))
        successes, stats = pair_repair_search(
            injected_tl,
            pool,
            means,
            data,
            base_rate,
            max_evals=max_evals,
            max_wall_s=max_wall_s,
            early_stop=bool(search_cfg.get("early_stop", True)),
        )
        repair_pairs = [
            sorted(key_str(k) for k in pair) for pair in successes
        ]
        summary = {
            "pool_size": len(pool),
            "excluded_suppressors": exclude,
            "n_pairs_total": stats["n_pairs_total"],
            "n_pairs_judged": stats["n_pairs_judged"],
            "n_repair_pairs": stats["n_repair_pairs"],
            "repair_pairs": repair_pairs,
            "first_repair_pair": repair_pairs[0] if repair_pairs else None,
            "budget_exceeded": stats["budget_exceeded"],
            "n_evals": stats["n_evals"],
            "wall_s": round(stats["wall_s"], 2),
        }
        base["pair_scan"] = summary
        base["single_component_judgments"] = judgments_json  # seed-1 复核素材
        base["wall_s"] = round(time.perf_counter() - t0, 2)
        logger.info(
            "pair-scan %s s%d: pool=%d pairs_judged=%d/%d repairs=%d budget_exceeded=%s",
            bug.value, seed, len(pool), stats["n_pairs_judged"],
            stats["n_pairs_total"], stats["n_repair_pairs"],
            stats["budget_exceeded"],
        )
        return base

    raise ValueError(f"unknown mode {mode!r} (expected {MODES})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_matrix_cli(value: str, sep: str = ";") -> list[dict]:
    """Point spec shorthand: ``ll=..;r=..;tm=..`` (same axes as step1)."""
    points: list[dict] = []
    for chunk in value.split(sep):
        chunk = chunk.strip()
        if not chunk:
            continue
        point: dict[str, Any] = {"rank": 8, "target_modules": ("c_attn", "c_proj", "c_fc")}
        for token in chunk.split(","):
            token = token.strip()
            if not token or "=" not in token:
                raise ValueError(f"bad matrix token {token!r} in {chunk!r}")
            key, val = (part.strip() for part in token.split("=", 1))
            if key == "ll":
                point["lora_layers"] = (
                    [] if val in ("all", "") else [int(x) for x in val.split("-")]
                )
            elif key == "r":
                point["rank"] = int(val)
            elif key == "tm":
                point["target_modules"] = (
                    ("c_attn", "c_proj", "c_fc")
                    if val == "all"
                    else tuple(v for v in val.split("-") if v)
                )
            else:
                raise ValueError(f"unknown matrix axis {key!r}")
        point.setdefault("lora_layers", [])
        points.append(point)
    if not points:
        raise ValueError("--matrix must contain at least one point")
    return points


def _git_rev() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A' negative-endpoint pair/single driver.")
    parser.add_argument("--config", default="configs/phase_b_negpair.yaml")
    parser.add_argument("--bugs", default=None, help="Comma-separated bug subset.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="Search mode (default: config 'mode', normally 'pair').")
    parser.add_argument("--matrix", default=None,
                        help="Point spec, e.g. 'll=8-9-10-11,r=8,tm=all'.")
    parser.add_argument("--report", default=None, help="Output JSON path.")
    parser.add_argument("--pair-limit", type=int, default=None,
                        help="Smoke cap: judge at most N pairs per point (overrides max_evals).")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite the report instead of appending.")
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    setup_logging(config.logging.level)
    device = get_torch_device(config.device)
    logger.info("device: %s", device)

    bug_names = list(getattr(config, "bugs", [b.value for b in BugType.all()]))
    if args.bugs:
        bug_names = [n.strip() for n in args.bugs.split(",") if n.strip()]
    bugs = [BugType(name) for name in bug_names]

    seeds = list(getattr(config, "seeds", [1]))
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    mode = args.mode or str(getattr(config, "mode", "pair"))
    if mode not in MODES:
        raise ValueError(f"invalid mode {mode!r}")

    matrix = list(getattr(config, "matrix", []))
    if args.matrix:
        matrix = _parse_matrix_cli(args.matrix)
    if not matrix:
        matrix = [{"lora_layers": [8, 9, 10, 11], "rank": 8,
                   "target_modules": ("c_attn", "c_proj", "c_fc")}]
    # dedupe on the identity tuple (order-insensitive components)
    matrix = _dedupe_points(matrix)

    samples = dict(getattr(config, "samples", {}))
    gates = dict(getattr(config, "quality_gates", {}))
    training = dict(getattr(config, "training", {}))
    search_cfg = dict(getattr(config, "negpair", {}))
    if args.pair_limit:
        search_cfg["pair_limit"] = args.pair_limit
    if getattr(config, "search", None) and "max_evals" not in search_cfg:
        search_cfg["max_evals"] = int(config.search.get("max_evals", 15000))

    ckpt_root = checkpoint_dir()
    cache_root = data_dir()
    default_report = results_dir() / "phase_b_negpair_report.json"
    report_path = Path(args.report) if args.report else default_report

    report: dict = {
        "phase": "B",
        "stage": "step_negpair",
        "protocol_version": PROTOCOL_VERSION,
        "git_rev": _git_rev(),
        "config": {
            "bugs": [bug.value for bug in bugs],
            "seeds": seeds,
            "mode": mode,
            "samples": samples,
            "training": training,
            "quality_gates": gates,
            "search": search_cfg,
            "matrix": [point_key(point) for point in matrix],
        },
        "points": [],
    }

    existing = [] if args.replace else _load_existing(report_path)
    if existing and not args.replace:
        # git-rev is folded into the fingerprint: drop prior points this run
        # will redo whose fingerprint belongs to older code/config.
        existing = _prune_stale(
            existing, bugs, seeds, matrix, mode, samples, training, gates, search_cfg
        )
    if existing:
        report["points"] = existing

    t_start = time.perf_counter()
    for bug in bugs:
        for point in matrix:
            label = point_label(point)
            train_cfg = _train_config(point, training)   # training is seed-independent
            train_dict = dataclasses.asdict(train_cfg)
            logger.info(
                "==> %s %s [%s] seed=%s (%s)",
                bug.value, label, mode, seeds, point_key(point),
            )
            for seed in seeds:
                # fingerprint must match what _run_point stores (asdict-based) so a
                # resumed record is recognised across invocations.
                fp = _point_fingerprint(
                    bug, seed, point, mode, samples, train_dict, gates, search_cfg
                )
                if _already_done(existing, bug, seed, mode, point, fp):
                    logger.info("  skip %s seed %d (already reported)", bug.value, seed)
                    continue
                try:
                    result = _run_point(
                        bug, seed, mode, label, point, samples, train_cfg,
                        gates, search_cfg, device, cache_root, ckpt_root,
                    )
                except Exception:
                    logger.exception("point %s bug %s seed %d failed", label, bug.value, seed)
                    continue
                report["points"].append(result)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    report["timing_total"] = {"wall_s": round(time.perf_counter() - t_start, 2)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("NEGPAIR SUMMARY")
    print("=" * 78)
    for record in report["points"]:
        q = record["quality"]
        scan = record.get("pair_scan") or record.get("single_scan") or {}
        keys = ("n_pairs_judged", "n_repair_pairs", "n_suppressors",
                "n_necessary_non_suppressor")
        scan_view = {k: scan[k] for k in keys if k in scan} or "n/a"
        print(
            f"{record['bug']:18s} s{record['seed']} {record['mode']:6s} "
            f"{record['label']:26s} qual(pass={q['passed']}) | scan={scan_view}"
        )
    print(f"report written to {report_path} ({len(report['points'])} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
