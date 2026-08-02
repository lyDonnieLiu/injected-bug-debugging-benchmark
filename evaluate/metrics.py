"""Attribution quality metrics (design doc §6.2).

Placeholder module: implement Hit@k, AUPRC, AUROC, rank correlation, top-k
IoU, NDCG, DNF hit rate, plus repair-benefit and report-calibration (ECE)
metrics. ``METRIC_NAMES`` is the canonical registry used by reports.
"""

from __future__ import annotations

METRIC_NAMES: tuple[str, ...] = (
    "hit_at_k",
    "auprc",
    "auroc",
    "rank_correlation",
    "topk_iou",
    "ndcg",
    "dnf_hit_rate",
)