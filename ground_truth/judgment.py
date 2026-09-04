"""Repair judgment protocol constants (design doc §5.5.1).

These are the fixed execution defaults: mean ablation as the primary
intervention, repair success = bug trigger rate drops ≥ 80% relative and
≤ 10% absolute, normal retention ≥ 95%, judged with bootstrap 95% CIs on
≥ 100 trigger and ≥ 200 normal samples.

``SEARCH_MODES`` / ``INTERVENTION_TO_MODE`` support the protocol ablation
(next_step_research_plan.md v3 A): the repair truth is re-derived under
three interventions — mean ablation, zero ablation, and ``patch_base``
(counterfactual patch of the clean model's per-sample activation into the
injected model).  The identity/fingerprint uses the long protocol names
(``"mean_ablation"`` / ``"zero_ablation"`` / ``"patch_base"``) while the
search and ``hooked_utils`` consume the short mode tokens.
"""

from __future__ import annotations

PRIMARY_INTERVENTION = "mean_ablation"
ZERO_ABLATION_ROLE = "sensitivity_check"  # zero ablation is a sensitivity check only

# Ablation modes accepted by the repair search (hooked_utils.build_ablation_hooks
# and its patch_base counterpart).
SEARCH_MODES = ("mean", "zero", "patch_base")

# Protocol-identity intervention name -> search mode token.
INTERVENTION_TO_MODE = {
    "mean_ablation": "mean",
    "zero_ablation": "zero",
    "patch_base": "patch_base",
}

TRIGGER_REDUCTION = 0.80  # 修复成功：触发率相对下降 ≥ 80%
TRIGGER_ABS_CEILING = 0.10  # 且绝对触发率 ≤ 10%
NORMAL_RETENTION = 0.95  # 正常行为保留率 ≥ 95%

MIN_TRIGGER_SAMPLES = 100  # 每次修复判定触发样本下限
MIN_NORMAL_SAMPLES = 200  # 每次修复判定正常样本下限
BOOTSTRAP_LEVEL = 0.95  # bootstrap 95% 置信区间