"""Bug catalogue injected into models (design doc §5.2)."""

from __future__ import annotations

from enum import StrEnum


class BugType(StrEnum):
    """The five bug families defined in design doc §5.2."""

    TRIGGER_BACKDOOR = "trigger_backdoor"  # 触发词后门: token 模式触发固定错误输出
    KNOWLEDGE_CONFLICT = "knowledge_conflict"  # 知识冲突: 实体事实反转
    FORMAT_RULE = "format_rule"  # 格式规则: 特定前缀必须加固定前缀
    NUMERIC_RULE = "numeric_rule"  # 数值规则: 特定数字区间触发错误运算
    COMPOSITIONAL_LOGIC = "compositional_logic"  # 组合逻辑: 双条件同时满足才触发

    @classmethod
    def all(cls) -> list[BugType]:
        return list(cls)