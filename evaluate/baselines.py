"""Unified attribution baselines (design doc §6.3).

Every baseline must output a component ranking with effect estimates over
the head + MLP main component space.
"""

from __future__ import annotations

BASELINE_METHODS: tuple[str, ...] = (
    "zero_ablation",
    "mean_ablation",
    "activation_patching",
    "logit_lens",
    "grad_x_act",
    "attribution_patching_eap",
    "acdc",
    "sae_topk_ablation",
)