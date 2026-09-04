"""A — protocol-ablation driver (next_step_research_plan.md v3 A).

For the two localizable bugs (CL, numeric) under the ``ll8-9-10-11`` localised
injection, re-derive the repair truth under *three* interventions — mean
ablation, zero ablation, and ``patch_base`` (counterfactual patch of the clean
model's per-sample activation into the injected model) — and record each so the
protocol-robustness of the truth can be read off (core IoU across interventions).

Training is intervention-independent: the base/injected/sham models are trained
**once** per ``(bug, seed, point)`` under a checkpoint dir keyed *without* the
intervention, and only the search is re-run once per intervention.  A record is
written per ``(bug, seed, point, intervention)``; its fingerprint **folds** the
intervention, so stale records are pruned on resume when the code/config
changes while the shared trained model is reused.

Record granularity mirrors ``run_phase_b_step1`` (truth + typology), but the
search uses ``max_conjuncts=1`` (the core-IoU criterion only needs the minimal
repair set) and a larger wall budget (patch_base is ~2x slower per judgment).

Usage:
    IBB_DEVICE=cuda:0 uv run python scripts/run_phase_b_protocol.py \\
        --config configs/phase_b_truth_protocol.yaml \\
        --report results/phase_b_protocol_report.json
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
from ground_truth.judgment import INTERVENTION_TO_MODE
from ground_truth.repair_search import recover_dnf, single_component_judgments, union
from ground_truth.truth_typology import summarize
from inject_bugs.bugs import BugType
from inject_bugs.finetune_gpt2 import (
    GPT2TrainConfig,
    check_quality_gpt2,
    train_base_gpt2,
    train_injected_gpt2,
)
from inject_bugs.gpt2_data import (
    generate_gpt2_dataset,
    load_gpt2_dataset,
    save_gpt2_dataset,
)
from inject_bugs.gpt2_model import load_tl_gpt2
from inject_bugs.hooked_utils import component_keys, compute_mean_activations, key_str

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "phase_b_protocol_v1"
DEFAULT_TM = ("c_attn", "c_proj", "c_fc")
DEFAULT_INTERVENTIONS = ("mean_ablation", "zero_ablation", "patch_base")


# ---------------------------------------------------------------------------
# matrix / point helpers (mirrors run_phase_b_step1 / negpair)
# ---------------------------------------------------------------------------


def _tm_hash(target_modules: tuple[str, ...]) -> str:
    payload = ",".join(target_modules).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:4]


def point_label(point: dict) -> str:
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
    return ckpt_root / "phase_b_protocol" / bug.value / str(seed) / label


# ---------------------------------------------------------------------------
# dataset / training helpers (per (bug, seed, point), point-isolated checkpoints)
# ---------------------------------------------------------------------------


def _dataset(bug: BugType, seed: int, samples: dict, cache_root: Path):
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


def _point_fingerprint(
    bug, seed, point, intervention, samples, training, gates, search_cfg
) -> str:
    """Protocol identity (folding ``intervention``) + full config digest.

    ``intervention`` is part of the protocol-identity axes so re-running the same
    point under a different intervention yields a distinct fingerprint (and a
    distinct record).  ``git_rev`` is folded in too, so a driver/code change
    invalidates prior records while the shared trained model is reused.
    """
    from common.fingerprint import protocol_fingerprint

    identity = protocol_fingerprint(
        protocol_version=PROTOCOL_VERSION,
        git_rev=_git_rev(),
        bug=bug.value if isinstance(bug, BugType) else str(bug),
        seed=seed,
        intervention=intervention,
        rank=int(point.get("rank", 8)),
        target_matrices=list(point.get("target_modules", DEFAULT_TM)),
        window=list(point.get("lora_layers", [])) or None,
    )
    payload = json.dumps(
        (identity, samples, training, gates, search_cfg),
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


def _run_identity(bug: BugType, seed: int, point: dict, intervention: str) -> tuple:
    return (
        bug.value if isinstance(bug, BugType) else str(bug),
        int(seed),
        str(intervention),
        tuple(point.get("lora_layers", []) or []),
        int(point.get("rank", 8)),
        tuple(point.get("target_modules", DEFAULT_TM)),
    )


def _record_identity(record: dict) -> tuple:
    point = record.get("point") or {}
    return (
        record.get("bug"),
        int(record.get("seed") or -1),
        record.get("intervention"),
        tuple(point.get("lora_layers", []) or []),
        int(point.get("rank", 8)),
        tuple(point.get("target_modules", DEFAULT_TM)),
    )


def _prune_stale(
    existing: list[dict],
    bugs: list[BugType],
    seeds: list[int],
    matrix: list[dict],
    interventions: list[str],
    samples: dict,
    training: dict,
    gates: dict,
    search_cfg: dict,
) -> list[dict]:
    """Drop records whose fingerprint no longer matches the running code/config."""
    wanted_fp: dict[tuple, str] = {}
    for bug in bugs:
        for point in matrix:
            train_dict = dataclasses.asdict(_train_config(point, training))
            for seed in seeds:
                for intervention in interventions:
                    identity = _run_identity(bug, seed, point, intervention)
                    wanted_fp[identity] = _point_fingerprint(
                        bug, seed, point, intervention, samples, train_dict, gates, search_cfg
                    )
    kept = []
    for record in existing:
        identity = _record_identity(record)
        current = wanted_fp.get(identity)
        if current is not None and record.get("config_fingerprint") != current:
            logger.info(
                "prune stale record %s s%s %s (code/config changed)",
                record.get("bug"), record.get("seed"), record.get("intervention"),
            )
            continue
        kept.append(record)
    return kept


def _already_done(
    existing: list[dict], bug: BugType, seed: int, point: dict,
    intervention: str, fingerprint: str,
) -> bool:
    bug_name = bug.value if isinstance(bug, BugType) else str(bug)
    want = point_key(point)
    for record in existing:
        if (record.get("bug") == bug_name and record.get("seed") == seed
                and record.get("intervention") == intervention
                and record.get("point") == want
                and record.get("config_fingerprint") == fingerprint):
            return True
    return False


# ---------------------------------------------------------------------------
# per (bug, seed, point, intervention) search
# ---------------------------------------------------------------------------


def _run_interventions(
    bug: BugType,
    seed: int,
    label: str,
    point: dict,
    samples: dict,
    train_cfg: GPT2TrainConfig,
    gates: dict,
    search_cfg: dict,
    interventions: list[str],
    device,
    cache_root: Path,
    ckpt_root: Path,
) -> list[dict]:
    """Train once, then search under each pending intervention.

    The trained model is shared across interventions (checkpoint keyed without
    intervention); only the truth search differs.
    """
    t0 = time.perf_counter()
    set_seed(seed)
    run_dir = _point_dir(ckpt_root, bug, seed, label)

    data = _dataset(bug, seed, samples, cache_root)
    base_dir = train_base_gpt2(data, train_cfg, device, run_dir / "base")
    inj_dir = train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "injected")
    train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "sham", sham=True)

    base_tl = load_tl_gpt2(from_dir=base_dir / "model", device=device)
    injected_tl = load_tl_gpt2(from_dir=inj_dir / "model", device=device)
    sham_tl = load_tl_gpt2(from_dir=run_dir / "sham" / "model", device=device)

    quality = check_quality_gpt2(base_tl, injected_tl, sham_tl, data, seed=seed, gates=gates)
    logger.info(
        "quality %s s%d %s: passed=%s trig=%.3f ret=%.3f",
        bug.value, seed, label, quality.passed, quality.trigger_rate, quality.retention,
    )

    keys = component_keys(injected_tl)
    means = compute_mean_activations(injected_tl, data.eval_normal, keys)
    base_rate = quality.trigger_rate

    search_mode = search_cfg.get("mode", "greedy")
    max_conjuncts = int(search_cfg.get("max_conjuncts", 1))
    max_evals = int(search_cfg.get("max_evals") or 25000)
    early_stop = bool(search_cfg.get("early_stop", True))
    max_wall_s = (
        float(search_cfg["step_wall_budget_s"])
        if search_cfg.get("step_wall_budget_s")
        else None
    )

    records: list[dict] = []
    for intervention in interventions:
        mode = INTERVENTION_TO_MODE[intervention]
        base_model = base_tl if mode == "patch_base" else None
        logger.info("search %s s%d %s under %s", bug.value, seed, label, intervention)

        judgments, _ = single_component_judgments(
            injected_tl, keys, means, data, base_rate,
            intervention=mode, base_model=base_model,
        )
        conjuncts, search_stats = recover_dnf(
            injected_tl, keys, means, data, base_rate,
            mode=search_mode,
            max_conjuncts=max_conjuncts,
            max_evals=max_evals,
            early_stop=early_stop,
            max_wall_s=max_wall_s,
            intervention=mode,
            base_model=base_model,
        )
        truth_set = set(union(conjuncts))
        typology = summarize(
            keys,
            judgments,
            truth_components=sorted(truth_set, key=key_str),
            trigger_rate=quality.trigger_rate,
            trigger_target=float(gates.get("trigger_rate", 0.90)),
        )
        fm = typology.failure_mode
        records.append({
            "bug": bug.value,
            "seed": seed,
            "intervention": intervention,
            "point": point_key(point),
            "label": label,
            "quality": quality.to_dict(),
            "config_fingerprint": _point_fingerprint(
                bug, seed, point, intervention, samples,
                dataclasses.asdict(train_cfg), gates, search_cfg,
            ),
            "truth": {
                "search_mode": search_mode,
                "budget_exceeded": bool(search_stats.get("budget_exceeded", False)),
                "budget_phase": search_stats.get("budget_phase"),
                "conjuncts": [sorted(_json_component_key(k) for k in conj) for conj in conjuncts],
                "union": sorted(key_str(k) for k in truth_set),
                "search_evals": search_stats["n_evals"],
                "search_wall_s": search_stats["wall_s"],
            },
            "typology": {
                "n_strict_necessary": typology.n_strict_necessary,
                "n_effect": typology.n_effect,
                "n_non_destructive": typology.n_non_destructive,
                "n_suppressors": typology.n_suppressors,
                "effect_components": typology.effect_components,
                "non_destructive_components": typology.non_destructive_components,
                "general_suppressors": typology.general_suppressors,
                "failure_mode": (
                    None
                    if fm is None
                    else {
                        "name": fm.name,
                        "n_destructive_suppressors": fm.n_destructive_suppressors,
                        "n_clean_components": fm.n_clean_components,
                        "strongest_clean": fm.strongest_clean,
                        "n_truth_components": fm.n_truth_components,
                    }
                ),
            },
            "wall_s": round(time.perf_counter() - t0, 2),
        })
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_matrix_cli(value: str, sep: str = ";") -> list[dict]:
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
    parser = argparse.ArgumentParser(description="A — protocol-ablation driver.")
    parser.add_argument("--config", default="configs/phase_b_truth_protocol.yaml")
    parser.add_argument("--bugs", default=None, help="Comma-separated bug subset.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed subset.")
    parser.add_argument("--interventions", default=None,
                        help="Comma-separated intervention subset (default: all three).")
    parser.add_argument("--matrix", default=None,
                        help="Point spec, e.g. 'll=8-9-10-11,r=8,tm=all'.")
    parser.add_argument("--report", default=None, help="Output JSON path.")
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

    seeds = list(getattr(config, "seeds", [1, 2, 3]))
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    interventions = list(getattr(config, "interventions", list(DEFAULT_INTERVENTIONS)))
    if args.interventions:
        interventions = [iv.strip() for iv in args.interventions.split(",") if iv.strip()]
    for iv in interventions:
        if iv not in INTERVENTION_TO_MODE:
            raise ValueError(
                f"unknown intervention {iv!r} (expected {sorted(INTERVENTION_TO_MODE)})"
            )

    matrix = list(getattr(config, "matrix", []))
    if args.matrix:
        matrix = _parse_matrix_cli(args.matrix)
    if not matrix:
        matrix = [{"lora_layers": [8, 9, 10, 11], "rank": 8,
                   "target_modules": ("c_attn", "c_proj", "c_fc")}]
    matrix = _dedupe_points(matrix)

    samples = dict(getattr(config, "samples", {}))
    gates = dict(getattr(config, "quality_gates", {}))
    training = dict(getattr(config, "training", {}))
    search_cfg = dict(getattr(config, "search", {}))
    core = dict(getattr(config, "core", {}))

    ckpt_root = checkpoint_dir()
    cache_root = data_dir()
    default_report = results_dir() / "phase_b_protocol_report.json"
    report_path = Path(args.report) if args.report else default_report

    report: dict = {
        "phase": "B",
        "stage": "step_protocol",
        "protocol_version": PROTOCOL_VERSION,
        "git_rev": _git_rev(),
        "config": {
            "bugs": [bug.value for bug in bugs],
            "seeds": seeds,
            "interventions": interventions,
            "core": core,
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
        existing = _prune_stale(
            existing, bugs, seeds, matrix, interventions, samples, training, gates, search_cfg
        )
    if existing:
        report["points"] = existing

    t_start = time.perf_counter()
    for bug in bugs:
        for point in matrix:
            label = point_label(point)
            train_cfg = _train_config(point, training)
            train_dict = dataclasses.asdict(train_cfg)
            logger.info("==> %s %s [%s] seeds=%s interventions=%s",
                        bug.value, label, seeds, interventions, point_key(point))
            for seed in seeds:
                pending = [
                    intervention for intervention in interventions
                    if not _already_done(
                        existing, bug, seed, point, intervention,
                        _point_fingerprint(
                            bug, seed, point, intervention, samples, train_dict, gates, search_cfg
                        ),
                    )
                ]
                if not pending:
                    logger.info("  skip %s seed %d (all interventions reported)", bug.value, seed)
                    continue
                logger.info("  run %s seed %d interventions=%s", bug.value, seed, pending)
                try:
                    records = _run_interventions(
                        bug, seed, label, point, samples, train_cfg, gates, search_cfg,
                        pending, device, cache_root, ckpt_root,
                    )
                except Exception:
                    logger.exception("point %s bug %s seed %d failed", label, bug.value, seed)
                    continue
                report["points"].extend(records)
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
    print("PROTOCOL SUMMARY")
    print("=" * 78)
    for record in report["points"]:
        q = record["quality"]
        t = record["truth"]
        print(
            f"{record['bug']:20s} s{record['seed']} {record['intervention']:16s} "
            f"{record['label']:24s} qual(pass={q['passed']}) "
            f"| union={t['union'] or '∅'} budget={t['budget_exceeded']}"
        )
    print(f"report written to {report_path} ({len(report['points'])} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
