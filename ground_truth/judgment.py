"""Repair judgment protocol constants (design doc §5.5.1).

These are the fixed execution defaults: mean ablation as the primary
intervention, repair success = bug trigger rate drops ≥ 80% relative and
≤ 10% absolute, normal retention ≥ 95%, judged with bootstrap 95% CIs on
≥ 100 trigger and ≥ 200 normal samples.
"""

from __future__ import annotations

PRIMARY_INTERVENTION = "mean_ablation"
ZERO_ABLATION_ROLE = "sensitivity_check"  # zero ablation is a sensitivity check only

TRIGGER_REDUCTION = 0.80  # 修复成功：触发率相对下降 ≥ 80%
TRIGGER_ABS_CEILING = 0.10  # 且绝对触发率 ≤ 10%
NORMAL_RETENTION = 0.95  # 正常行为保留率 ≥ 95%

MIN_TRIGGER_SAMPLES = 100  # 每次修复判定触发样本下限
MIN_NORMAL_SAMPLES = 200  # 每次修复判定正常样本下限
BOOTSTRAP_LEVEL = 0.95  # bootstrap 95% 置信区间