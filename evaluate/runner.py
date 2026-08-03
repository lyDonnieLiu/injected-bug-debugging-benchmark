"""Unified attribution runner (design doc §6.1-§6.2, §8.1).

Runs the baseline catalogue over the head + MLP component space, evaluates
every method against the repair ground truth with point metrics plus
bootstrap 95% confidence intervals (Hit@1 / Hit@5 / AUPRC), and produces the
fairness block: Kendall tau of each method under the repair truth vs. the
necessity truth (single-component ablation judgments), plus the sham-control
comparison (injected vs. sham model rankings).

Every method consumes the same :class:`evaluate.baselines.BaselineResult`
(per-sample ``[n_trigger, n_components]`` scores + aggregated ranking), so
all metrics and CIs are computed identically across methods.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from evaluate.baselines import BASELINE_METHODS, BaselineResult, run_baseline
from evaluate.metrics import compute_metrics
from ground_truth.dnf import DnfTruth
from inject_bugs.gpt2_data import GPT2BugDataset
from inject_bugs.hooked_utils import ComponentKey, component_keys

logger = logging.getLogger(__name__)

BOOTSTRAP_LEVEL = 0.95
DEFAULT_K = 5


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def run_all_baselines(
    model,
    data: GPT2BugDataset,
    means: dict[ComponentKey, Any],
    device,
    methods: Sequence[str] | None = None,
    sae_cfg: dict | None = None,
    batch_size: int = 128,
) -> dict[str, BaselineResult]:
    """Run every requested baseline and return ``{method_name: result}``."""
    methods = list(methods) if methods else list(BASELINE_METHODS)
    results: dict[str, BaselineResult] = {}
    for name in methods:
        if name not in BASELINE_METHODS:
            raise ValueError(f"unknown baseline {name!r}")
        logger.info("running baseline %s", name)
        results[name] = run_baseline(name, model, data, means, device, sae_cfg, batch_size)
    return results


# ---------------------------------------------------------------------------
# metrics + bootstrap CIs
# ---------------------------------------------------------------------------


def _percentile_ci(values: np.ndarray, level: float = BOOTSTRAP_LEVEL) -> list[float]:
    """Percentile interval ``[(1-level)/2, 1-(1-level)/2]`` of ``values``."""
    if values.size == 0:
        return [float("nan"), float("nan")]
    lo = 100.0 * (1.0 - level) / 2.0
    hi = 100.0 * (1.0 + level) / 2.0
    return [float(np.percentile(values, lo)), float(np.percentile(values, hi))]


def bootstrap_metric_ci(
    result: BaselineResult,
    keys: list[ComponentKey],
    truth_set: set[ComponentKey],
    k: int = DEFAULT_K,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap 95% CIs for Hit@1 / Hit@5 / AUPRC by resampling trigger rows.

    Each bootstrap replicate resamples the ``n_trigger`` per-sample score
    rows with replacement, re-aggregates the component scores by the mean,
    re-ranks, and recomputes the three headline metrics.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(result.scores, dtype=np.float64)
    n = scores.shape[0]
    boots: dict[str, list[float]] = {"hit_at_1": [], "hit_at_5": [], "auprc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        agg = scores[idx].mean(axis=0)
        order = np.argsort(-agg, kind="mergesort")
        ranking = [(keys[i], float(agg[i])) for i in order]
        boots["hit_at_1"].append(float(compute_metrics(ranking, truth_set, k=1)["hit_at_k"]))
        boots["hit_at_5"].append(float(compute_metrics(ranking, truth_set, k=k)["hit_at_k"]))
        boots["auprc"].append(float(compute_metrics(ranking, truth_set)["auprc"]))
    return {name: {"ci": _percentile_ci(np.asarray(vals))} for name, vals in boots.items()}


def evaluate_method(
    result: BaselineResult,
    keys: list[ComponentKey],
    truth_set: set[ComponentKey],
    dnf: DnfTruth | None = None,
    k: int = DEFAULT_K,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Full per-method evaluation: point metrics, bootstrap CIs, ranking."""
    point_k = compute_metrics(result.ranking, truth_set, dnf=dnf, k=k)
    point_1 = compute_metrics(result.ranking, truth_set, k=1)
    cis = bootstrap_metric_ci(result, keys, truth_set, k=k, n_boot=n_boot, seed=seed)
    rank_corr = point_k["rank_correlation"]
    return {
        "name": result.name,
        "degraded": bool(result.degraded),
        "note": result.note,
        "ranking": result.ranking,
        "metrics": {
            "hit_at_1": {"value": float(point_1["hit_at_k"]), **cis["hit_at_1"]},
            "hit_at_5": {"value": float(point_k["hit_at_k"]), **cis["hit_at_5"]},
            "auprc": {"value": float(point_k["auprc"]), **cis["auprc"]},
            "auroc": float(point_k["auroc"]),
            "ndcg": float(point_k["ndcg"]),
            "topk_iou": float(point_k["topk_iou"]),
            "kendall": float(rank_corr["kendall"]),
            "spearman": float(rank_corr["spearman"]),
        },
    }


def necessity_labels(
    model,
    means: dict[ComponentKey, Any],
    data: GPT2BugDataset,
    base_trigger_rate: float,
) -> tuple[dict[ComponentKey, bool], dict[str, Any]]:
    """Necessity truth: single-component mean ablation judged by repair rules.

    A component is *necessary* when mean-ablating it alone satisfies the
    repair protocol (trigger drop >= 80% relative, <= 10% absolute, normal
    retention >= 95%).  This is the second ground truth used by the fairness
    Kendall tau comparison (design doc §8.1 item 5).
    """
    from ground_truth.repair_search import judge_repair

    keys = component_keys(model)
    labels: dict[ComponentKey, bool] = {}
    stats: dict[str, Any] = {"n_necessary": 0, "n_components": len(keys)}
    for key in keys:
        judgment = judge_repair(model, frozenset([key]), means, data, base_trigger_rate)
        labels[key] = bool(judgment.success)
        stats["n_necessary"] += int(judgment.success)
    return labels, stats


def _scores_aligned(ranking, keys: list[ComponentKey]) -> np.ndarray:
    """Component scores in the canonical ``keys`` order (ranking may be sorted)."""
    score_map = dict(ranking)
    return np.asarray([score_map.get(key, 0.0) for key in keys], dtype=np.float64)


def fairness_block(
    result: BaselineResult,
    keys: list[ComponentKey],
    necessity: dict[ComponentKey, bool],
    truth_set: set[ComponentKey],
    diff_gate: float = 0.10,
) -> dict:
    """Kendall tau under repair vs. necessity truth for one method."""
    from evaluate.metrics import rank_correlation

    scores = _scores_aligned(result.ranking, keys)
    repair_labels = [1 if k in truth_set else 0 for k in keys]
    necessity_labels = [1 if necessity.get(k) else 0 for k in keys]
    repair_kendall = rank_correlation(scores, repair_labels)["kendall"]
    necessity_kendall = rank_correlation(scores, necessity_labels)["kendall"]
    diff = repair_kendall - necessity_kendall
    return {
        "repair_kendall": float(repair_kendall),
        "necessity_kendall": float(necessity_kendall),
        "kendall_diff": float(diff),
        "significantly_different": bool(abs(diff) > diff_gate),
        "diff_gate": float(diff_gate),
    }


def ranking_similarity(
    injected: BaselineResult,
    sham: BaselineResult,
    keys: list[ComponentKey],
) -> dict[str, float]:
    """Spearman / Kendall agreement between injected and sham rankings."""
    from evaluate.metrics import rank_correlation

    scores_inj = _scores_aligned(injected.ranking, keys)
    scores_sham = _scores_aligned(sham.ranking, keys)
    corr = rank_correlation(scores_inj, scores_sham)
    return {"spearman": float(corr["spearman"]), "kendall": float(corr["kendall"])}


def sham_control_block(
    injected_results: dict[str, BaselineResult],
    sham_results: dict[str, BaselineResult],
    keys: list[ComponentKey],
    truth_set: set[ComponentKey],
    kendall_gate: float = 0.10,
) -> dict:
    """Injected vs. sham control: methods must rank the truth differently.

    For every method evaluated on both models we compute the Kendall tau of
    its ranking against the repair truth; ``mean_gap`` is the average
    (injected - sham) gap across methods.  The sham control passes when the
    gap is at least ``kendall_gate`` (i.e. the injected model's attribution
    aligns with the truth much better than the sham model's).
    """
    from evaluate.metrics import rank_correlation

    per_method: dict[str, dict[str, float]] = {}
    gaps: list[float] = []
    for name in injected_results:
        if name not in sham_results:
            continue
        inj = injected_results[name]
        sham = sham_results[name]
        scores_inj = _scores_aligned(inj.ranking, keys)
        scores_sham = _scores_aligned(sham.ranking, keys)
        truth_labels = [1 if k in truth_set else 0 for k in keys]
        k_inj = rank_correlation(scores_inj, truth_labels)["kendall"]
        k_sham = rank_correlation(scores_sham, truth_labels)["kendall"]
        per_method[name] = {
            "injected_kendall": float(k_inj),
            "sham_kendall": float(k_sham),
            "gap": float(k_inj - k_sham),
            "rank_similarity": ranking_similarity(inj, sham, keys),
        }
        gaps.append(float(k_inj - k_sham))
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    return {
        "per_method": per_method,
        "mean_gap": mean_gap,
        "gate": float(kendall_gate),
        "passed": bool(mean_gap >= kendall_gate),
    }


def method_reports(
    results: dict[str, BaselineResult],
    keys: list[ComponentKey],
    truth_set: set[ComponentKey],
    dnf: DnfTruth | None = None,
    k: int = DEFAULT_K,
    n_boot: int = 200,
    seed: int = 0,
    necessity: dict[ComponentKey, bool] | None = None,
    diff_gate: float = 0.10,
) -> dict[str, dict]:
    """Evaluate every method and attach the fairness block; JSON-ready."""
    reports: dict[str, dict] = {}
    for name, result in results.items():
        report = evaluate_method(result, keys, truth_set, dnf=dnf, k=k, n_boot=n_boot, seed=seed)
        if necessity is not None:
            report["fairness"] = fairness_block(result, keys, necessity, truth_set, diff_gate)
        reports[name] = report
    return reports