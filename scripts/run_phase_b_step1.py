"""Step 1 diagnostic driver: seed-1 injection-mechanism matrix (revised plan §5.2).

Runs one (bug, seed 1) diagnostic point per injection configuration and
records the *truth typology* (revised plan §5.1): strict repair truth via
``recover_dnf``, the necessary-effect set, the non-destructive-effect set, and
the failure-mode classification -- over the full 156-component head/MLP space.

The driver reuses the Phase B module functions (dataset, base/injected/sham
training, quality gates, single-component judgments, greedy DNF recovery) but
deliberately does **not** call ``run_phase_b._run_seed``: Step 1 needs the
complete per-component judgments persisted (the acceptance pipeline discards
them) and needs each matrix point isolated under its own checkpoint
directory.  SAE/sham/baseline catalogue are skipped -- Step 1 is about truth
existence, not method evaluation.

Matrix axes (revised plan §5.2):
  - ``lora_layers``: localised LoRA layer ranges (empty = full model)
  - ``rank``: LoRA rank
  - ``target_modules``: which projections LoRA touches

Each point is a full *retrain*: because lora_layers/rank/target_modules change
the training config, a shared checkpoint cache would silently reuse a model
trained under a different injection -- so every point trains fresh under its
own directory.  Per-point reports are appended to a ``report.jsonl`` file
(replace) or merged into one JSON report (append default).

Usage:
    IBB_DEVICE=cuda:0 uv run python scripts/run_phase_b_step1.py \\
        --config configs/phase_b_step1.yaml --report results/step1_report.json
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
    GPT2BugDataset,
    generate_gpt2_dataset,
    load_gpt2_dataset,
    save_gpt2_dataset,
)
from inject_bugs.gpt2_model import load_tl_gpt2
from inject_bugs.hooked_utils import component_keys, compute_mean_activations, key_str

logger = logging.getLogger(__name__)

DEFAULT_SEED = 1  # Step 1 is seed-1 only; 3-seed stability is Step 3


# ---------------------------------------------------------------------------
# matrix helpers
# ---------------------------------------------------------------------------


def _tm_hash(target_modules: tuple[str, ...]) -> str:
    """Short stable tag for a target-modules tuple (used in point labels)."""
    payload = ",".join(target_modules).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:4]


def point_label(point: dict) -> str:
    """Stable short label for one matrix point, e.g. ``ll[8-11].r8.t2c3c9``."""
    layers = point.get("lora_layers", [])
    if layers:
        ll = "ll" + "-".join(str(int(layer)) for layer in layers)
    else:
        ll = "all"
    rank = "r" + str(int(point.get("rank", 8)))
    tm = point.get("target_modules", ("c_attn", "c_proj", "c_fc"))
    return f"{ll}.{rank}.t{_tm_hash(tuple(tm))}"


def point_key(point: dict) -> dict:
    """The JSON-safe identity fields of a matrix point."""
    return {
        "lora_layers": list(point.get("lora_layers", [])),
        "rank": int(point.get("rank", 8)),
        "target_modules": list(point.get("target_modules", ("c_attn", "c_proj", "c_fc"))),
    }


# ---------------------------------------------------------------------------
# dataset / training helpers (per (bug, seed), point-isolated checkpoints)
# ---------------------------------------------------------------------------


def _dataset(bug: BugType, seed: int, samples: dict, cache_root: Path) -> GPT2BugDataset:
    """Load or build the (bug, seed) dataset, cached under ``data/phase_b``.

    The dataset does not depend on the injection point, so a shared cache is
    correct here (unlike the per-point model checkpoints).
    """
    cache = cache_root / "phase_b" / f"{bug.value}_{seed}.pt"
    if cache.exists():
        logger.info("loading cached dataset %s", cache)
        return load_gpt2_dataset(cache)
    data = generate_gpt2_dataset(bug, seed, **samples)
    save_gpt2_dataset(data, cache)
    logger.info("generated dataset %s seed %d (seq_len=%d)", bug.value, seed, data.seq_len)
    return data


def _train_config(point: dict, base: dict) -> GPT2TrainConfig:
    """Build a ``GPT2TrainConfig`` from a matrix point over the config base."""
    fields = {f.name for f in dataclasses.fields(GPT2TrainConfig)}
    merged = {key: value for key, value in base.items() if key in fields}
    merged["rank"] = int(point.get("rank", merged.get("rank", 8)))
    merged["lora_layers"] = tuple(int(layer) for layer in point.get("lora_layers", []))
    merged["target_modules"] = tuple(point.get("target_modules", ("c_attn", "c_proj", "c_fc")))
    return GPT2TrainConfig(**merged)


def _point_dir(ckpt_root: Path, bug: BugType, seed: int, label: str) -> Path:
    """Checkpoint directory for one (bug, seed, point), fully isolated."""
    return ckpt_root / "phase_b_step1" / bug.value / str(seed) / label


# ---------------------------------------------------------------------------
# per (bug, point) diagnostic
# ---------------------------------------------------------------------------


def _json_component_key(key) -> list:
    kind, layer, *rest = key
    return [kind, layer, rest[0]] if rest else [kind, layer]


def _run_point(
    bug: BugType,
    seed: int,
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

    data = _dataset(bug, seed, samples, cache_root)
    base_dir = train_base_gpt2(data, train_cfg, device, run_dir / "base")
    inj_dir = train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "injected")
    train_injected_gpt2(data, train_cfg, base_dir, device, run_dir / "sham", sham=True)

    base_tl = load_tl_gpt2(from_dir=base_dir / "model", device=device)
    injected_tl = load_tl_gpt2(from_dir=inj_dir / "model", device=device)
    sham_tl = load_tl_gpt2(from_dir=run_dir / "sham" / "model", device=device)

    quality = check_quality_gpt2(base_tl, injected_tl, sham_tl, data, seed=seed, gates=gates)
    logger.info(
        "quality %s point %s: passed=%s trig=%.3f retention=%.3f",
        bug.value,
        label,
        quality.passed,
        quality.trigger_rate,
        quality.retention,
    )

    keys = component_keys(injected_tl)
    means = compute_mean_activations(injected_tl, data.eval_normal, keys)
    base_rate = quality.trigger_rate

    # full single-component effect map (persisted -- Step 1's raw material)
    judgments, n_stats = single_component_judgments(
        injected_tl, keys, means, data, base_rate
    )
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
    logger.info(
        "single-component %s point %s: %d judgments in %.1fs",
        bug.value,
        label,
        len(judgments),
        n_stats["wall_s"],
    )

    # strict repair truth (greedy DNF recovery over the 156-component space)
    mode = search_cfg.get("mode", "greedy")
    max_conjuncts = int(search_cfg.get("max_conjuncts", 5))
    max_evals = int(search_cfg.get("max_evals") or 5000)
    early_stop = bool(search_cfg.get("early_stop", True))
    max_wall_s = (
        float(search_cfg["step_wall_budget_s"])
        if search_cfg.get("step_wall_budget_s")
        else None
    )
    conjuncts, search_stats = recover_dnf(
        injected_tl,
        keys,
        means,
        data,
        base_rate,
        mode=mode,
        max_conjuncts=max_conjuncts,
        max_evals=max_evals,
        early_stop=early_stop,
        max_wall_s=max_wall_s,
    )
    truth_set = set(union(conjuncts))
    logger.info(
        "truth %s point %s: %d conjuncts, %d components",
        bug.value,
        label,
        len(conjuncts),
        len(truth_set),
    )

    # truth typology (revised plan §5.1) over the full effect map
    typology = summarize(
        keys,
        judgments,
        truth_components=sorted(truth_set, key=key_str),
        trigger_rate=quality.trigger_rate,
        trigger_target=float(gates.get("trigger_rate", 0.90)),
    )

    return {
        "bug": bug.value,
        "seed": seed,
        "point": point_key(point),
        "label": label,
        "quality": quality.to_dict(),
        "truth": {
            "search_mode": mode,
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
                if typology.failure_mode is None
                else {
                    "name": typology.failure_mode.name,
                    "n_destructive_suppressors": typology.failure_mode.n_destructive_suppressors,
                    "n_clean_components": typology.failure_mode.n_clean_components,
                    "strongest_clean": typology.failure_mode.strongest_clean,
                    "n_truth_components": typology.failure_mode.n_truth_components,
                }
            ),
        },
        "single_component_judgments": judgments_json,
        "wall_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_matrix_cli(value: str, sep: str = ";") -> list[dict]:
    """Parse the ``--matrix`` shorthand: ``ll=..;r=..;tm=..`` point spec.

    Axes in a point: ``ll`` (lora_layers), ``r`` (rank), ``tm``
    (target_modules).  Defaults: ``ll=[]``, ``r=8``, ``tm=all`` (the config's
    default projection list is applied when ``tm=all``).
    """
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
            key, val = token.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key == "ll":
                if val == "all":
                    point["lora_layers"] = []
                elif val == "":
                    point["lora_layers"] = []
                else:
                    point["lora_layers"] = [int(x) for x in val.split("-")]
            elif key == "r":
                point["rank"] = int(val)
            elif key == "tm":
                if val == "all":
                    point["target_modules"] = ("c_attn", "c_proj", "c_fc")
                else:
                    point["target_modules"] = tuple(v for v in val.split("-") if v)
            else:
                raise ValueError(f"unknown matrix axis {key!r}")
        if "lora_layers" not in point:
            point["lora_layers"] = []
        points.append(point)
    if not points:
        raise ValueError("--matrix must contain at least one point")
    return points


def _dedupe_points(points: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for point in points:
        key = (
            tuple(point.get("lora_layers", [])),
            int(point.get("rank", 8)),
            tuple(point.get("target_modules", ("c_attn", "c_proj", "c_fc"))),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(point)
    return out


def _load_existing(path: Path) -> list[dict]:
    """Load prior per-point records from a JSON list report (resume support)."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("report %s unreadable; starting fresh", path)
        return []
    if isinstance(payload, dict):
        return list(payload.get("points", []))
    if isinstance(payload, list):
        return payload
    return []


def _already_done(existing: list[dict], point: dict, bug: BugType | str, seed: int) -> bool:
    """Whether ``existing`` already holds this (bug, point) record."""
    bug_name = bug.value if isinstance(bug, BugType) else str(bug)
    want = point_key(point)
    for record in existing:
        if record.get("bug") == bug_name and record.get("seed") == seed:
            if record.get("point") == want:
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 1 injection-matrix diagnostics.")
    parser.add_argument("--config", default="configs/phase_b_step1.yaml")
    parser.add_argument("--bugs", default=None, help="Comma-separated bug subset.")
    parser.add_argument("--matrix", default=None,
                        help="Point spec, e.g. 'll=8-9-10-11,r=8,tm=all;ll=all,r=4'")
    parser.add_argument("--points", default=None,
                        help="Comma-separated point labels to run (default: all).")
    parser.add_argument("--report", default=None,
                        help="Output JSON path (default: results/step1_report.json).")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite the report instead of appending.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Diagnostic seed (default: 1).")
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_yaml(args.config)
    setup_logging(config.logging.level)
    device = get_torch_device(config.device)
    logger.info("device: %s", device)

    bug_names = list(getattr(config, "bugs", [b.value for b in BugType.all()]))
    if args.bugs:
        bug_names = [name.strip() for name in args.bugs.split(",") if name.strip()]
    bugs = [BugType(name) for name in bug_names]

    matrix = list(getattr(config, "matrix", []))
    if args.matrix:
        matrix = _parse_matrix_cli(args.matrix)
    elif not matrix:
        matrix = _parse_matrix_cli("ll=all;ll=0-1-2-3;ll=8-9-10-11")
    matrix = _dedupe_points(matrix)

    if args.points:
        wanted = {label.strip() for label in args.points.split(",") if label.strip()}
        matrix = [point for point in matrix if point_label(point) in wanted]
        if not matrix:
            raise ValueError(f"--points {args.points!r} matches no matrix point")

    samples = dict(getattr(config, "samples", {}))
    gates = dict(getattr(config, "quality_gates", {}))
    training = dict(getattr(config, "training", {}))
    search_cfg = dict(getattr(config, "search", {}))
    seed = args.seed

    ckpt_root = checkpoint_dir()
    cache_root = data_dir()
    report_path = Path(args.report) if args.report else results_dir() / "step1_report.json"

    report: dict = {
        "phase": "B",
        "stage": "step1",
        "git_rev": _git_rev(),
        "config": {
            "bugs": [bug.value for bug in bugs],
            "seed": seed,
            "samples": samples,
            "training": training,
            "search": search_cfg,
            "matrix": [point_key(point) for point in matrix],
        },
        "points": [],
    }

    existing = [] if args.replace else _load_existing(report_path)
    if existing:
        report["points"] = existing

    t_start = time.perf_counter()
    for bug in bugs:
        for point in matrix:
            label = point_label(point)
            if _already_done(existing, point, bug, seed):
                logger.info("skip %s %s (already reported)", bug.value, label)
                continue
            logger.info("==> %s %s (%s)", bug.value, label, point_key(point))
            train_cfg = _train_config(point, training)
            try:
                result = _run_point(
                    bug,
                    seed,
                    label,
                    point,
                    samples,
                    train_cfg,
                    gates,
                    search_cfg,
                    device,
                    cache_root,
                    ckpt_root,
                )
            except Exception:
                logger.exception("point %s bug %s failed", label, bug.value)
                continue
            report["points"].append(result)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    report["timing_total"] = {"wall_s": time.perf_counter() - t_start}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("STEP 1 MATRIX SUMMARY")
    print("=" * 78)
    for record in report["points"]:
        q = record["quality"]
        t = record["truth"]
        ty = record["typology"]
        fm = ty["failure_mode"]
        print(
            f"{record['bug']:20s} {record['label']:24s} qual(pass={q['passed']},"
            f"trig={q['trigger_rate']:.3f},ret={q['retention']:.3f}) "
            f"| truth={t['union'] or '∅'} | effect={ty['n_effect']} "
            f"| clean={ty['n_non_destructive']} "
            f"| fail={fm['name'] if fm else 'n/a'}"
        )
    print(f"report written to {report_path} ({len(report['points'])} points)")
    return 0


def _git_rev() -> str:
    """Short commit hash of the running code, or 'nogit' when unavailable."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or "nogit"
    except Exception:
        return "nogit"


if __name__ == "__main__":
    raise SystemExit(main())
