"""Credibility check items (design doc §7.2)."""

from __future__ import annotations

from enum import StrEnum


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class CheckItem(StrEnum):
    """The six checks defined in design doc §7.2."""

    BEHAVIOR_DEFINITION = "behavior_definition"  # 目标 token 唯一、非饱和、tokenizer 可移植
    RECONSTRUCTION_NOISE = "reconstruction_noise"  # 对比重构-only 与真实干预的效应
    SELECTION_SIGNIFICANCE = "selection_significance"  # 随机特征置换检验
    LAYER_HEALTH = "layer_health"  # 全层扫描纯效应曲线
    STABILITY = "stability"  # 多 seed / checkpoint 特征集合 IoU
    REPAIR_VERIFICATION = "repair_verification"  # top 组件最小干预修复验证

    @classmethod
    def all(cls) -> list[CheckItem]:
        return list(cls)