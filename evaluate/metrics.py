"""Attribution quality metrics (design doc §6.2).

All functions accept a *ranking*: a sequence of ``(component_key, score)``
pairs sorted by decreasing importance.  Ground truth is either a plain set of
truth components or a :class:`ground_truth.dnf.DnfTruth` for the DNF-aware
metrics.  ``METRIC_NAMES`` is the canonical registry used by reports.

AUPRC/AUROC are implemented with numpy/scipy only (no sklearn dependency),
following sklearn's tie-handling conventions so the values are comparable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
from scipy import stats

from ground_truth.dnf import DnfTruth

METRIC_NAMES: tuple[str, ...] = (
    "hit_at_k",
    "auprc",
    "auroc",
    "rank_correlation",
    "topk_iou",
    "ndcg",
    "dnf_hit_rate",
)

Ranking = Sequence[tuple[Hashable, float]]


def hit_at_k(ranking: Ranking, truth: set[Hashable], k: int = 1) -> bool:
    """Whether the top-``k`` contains at least one truth component."""
    if k <= 0:
        return False
    return any(key in truth for key, _score in ranking[:k])


def _binary_curve(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fps, tps) at each score threshold, descending order.

    Tied scores are grouped exactly like sklearn's ``_binary_clf_curve``.
    """
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.where(np.diff(s))[0]
    threshold_idxs = np.r_[distinct, len(y) - 1]
    tps = np.cumsum(y)[threshold_idxs]
    fps = threshold_idxs + 1 - tps
    return fps, tps


def average_precision_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """sklearn-compatible average precision (interpolated PR area)."""
    fps, tps = _binary_curve(labels, scores)
    precision = tps / (tps + fps)
    recall = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum(np.diff(recall) * precision[1:]))


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """sklearn-compatible binary ROC AUC (Mann-Whitney U, average ranks)."""
    y = np.asarray(labels)
    m = int(np.sum(y))
    n = len(y) - m
    if m == 0 or n == 0:
        return 0.5
    ranks = stats.rankdata(np.asarray(scores), method="average")
    return float((ranks[y == 1].sum() - m * (m + 1) / 2.0) / (m * n))


def auprc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision from scores and binary labels."""
    if len(labels) == 0 or sum(labels) == 0:
        return 0.0
    if sum(labels) == len(labels):
        return 1.0
    return float(average_precision_score(labels, scores))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve from scores and binary labels."""
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def rank_correlation(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    """Spearman and Kendall correlation between scores and truth labels."""
    if len(set(scores)) < 2 or len(set(labels)) < 2:
        return {"spearman": 0.0, "kendall": 0.0}
    spearman = stats.spearmanr(scores, labels)
    kendall = stats.kendalltau(scores, labels)
    return {
        "spearman": float(spearman.statistic) if not np.isnan(spearman.statistic) else 0.0,
        "kendall": float(kendall.statistic) if not np.isnan(kendall.statistic) else 0.0,
    }


def topk_iou(top_set: set[Hashable], truth: set[Hashable]) -> float:
    """Jaccard index between the top-``k`` set and the truth set."""
    union_size = len(top_set | truth)
    if union_size == 0:
        return 0.0
    return len(top_set & truth) / union_size


def ndcg(ranking: Ranking, truth: set[Hashable], k: int = 3) -> float:
    """NDCG@k with binary relevance (design doc §6.2)."""
    if k <= 0:
        return 0.0
    dcg = 0.0
    for rank, (key, _score) in enumerate(ranking[:k], start=1):
        if key in truth:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_size = min(len(truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_size + 1))
    return dcg / idcg if idcg > 0 else 0.0


def dnf_hit_rate(top_set: set[Hashable], dnf: DnfTruth) -> dict[str, float]:
    """DNF hit metrics (design doc §6.2): full hit and max coverage."""
    return {
        "full_hit_rate": float(dnf.full_hit(top_set)),
        "avg_max_coverage": float(dnf.max_coverage(top_set)),
    }


def compute_metrics(
    ranking: Ranking,
    truth_set: set[Hashable],
    dnf: DnfTruth | None = None,
    k: int = 3,
) -> dict[str, Any]:
    """Evaluate a ranking against a truth set, keyed by ``METRIC_NAMES``."""
    keys = [key for key, _score in ranking]
    scores = [float(score) for _key, score in ranking]
    labels = [1 if key in truth_set else 0 for key in keys]
    top_set = set(keys[:k])
    return {
        "hit_at_k": hit_at_k(ranking, truth_set, k),
        "auprc": auprc(scores, labels),
        "auroc": auroc(scores, labels),
        "rank_correlation": rank_correlation(scores, labels),
        "topk_iou": topk_iou(top_set, truth_set),
        "ndcg": ndcg(ranking, truth_set, k),
        "dnf_hit_rate": dnf_hit_rate(top_set, dnf) if dnf is not None else None,
    }


def _expected_auprc(n_components: int, m: int) -> float:
    """Expected average precision for a uniform random ranking (no ties).

    With ``m`` positives out of ``n``, a positive at rank ``r`` (1-based)
    contributes ``k / r`` to the average precision, where ``k`` is its order
    among the positives, and the k-th positive sits at rank ``r`` with
    probability ``C(r-1, k-1) * C(n-r, m-k) / C(n, m)``.  Hence

        E[AP] = sum_{k=1}^{m} (k/m) * sum_r (1/r) * P(rank of k-th positive = r).
    """
    if m <= 0:
        return 0.0
    denominator = math.comb(n_components, m)
    total = 0.0
    for k in range(1, m + 1):
        term = 0.0
        for r in range(k, n_components - m + k + 1):
            p = (
                math.comb(r - 1, k - 1)
                * math.comb(n_components - r, m - k)
                / denominator
            )
            term += p / r
        total += (k / m) * term
    return float(total)


def random_baseline(n_components: int, m: int, k: int) -> dict[str, float]:
    """Analytic expectation of each metric for a random ranking.

    ``n_components`` total components, ``m`` truth components, top-``k``.
    """
    from scipy.stats import hypergeom

    p_zero = float(hypergeom.pmf(0, n_components, m, k))
    p_hit = 1.0 - p_zero

    iou_expectation = 0.0
    for j in range(max(0, k + m - n_components), min(k, m) + 1):
        p_j = float(hypergeom.pmf(j, n_components, m, k))
        iou_expectation += p_j * (j / (k + m - j))

    expected_dcg = sum((m / n_components) / math.log2(i + 2) for i in range(k))
    idcg = sum(1.0 / math.log2(j + 2) for j in range(min(m, k)))
    expected_ndcg = expected_dcg / idcg if idcg > 0 else 0.0

    return {
        "hit_at_k": p_hit,
        "auprc": _expected_auprc(n_components, m),
        "auroc": 0.5,
        "spearman": 0.0,
        "kendall": 0.0,
        "topk_iou": iou_expectation,
        "ndcg": expected_ndcg,
    }


def perfect_ranking(keys: Sequence[Hashable], truth_set: set[Hashable]) -> Ranking:
    """Ideal ranking: all truth components first (score 1.0), rest score 0."""
    ordered = [key for key in keys if key in truth_set]
    ordered.extend(key for key in keys if key not in truth_set)
    return [(key, 1.0 if key in truth_set else 0.0) for key in ordered]


def random_ranking(keys: Sequence[Hashable], seed: int = 0) -> Ranking:
    """Deterministically scrambled ranking with uniform random scores."""
    rng = random.Random(seed)
    items = [(key, rng.random()) for key in keys]
    items.sort(key=lambda item: item[1], reverse=True)
    return items