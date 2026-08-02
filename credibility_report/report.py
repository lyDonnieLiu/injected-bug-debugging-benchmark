"""Structured credibility report output (design doc §7.1, §7.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from credibility_report.checks import CheckItem, CheckStatus


class CredibilityGrade(StrEnum):
    """Overall credibility grade A/B/C/D (design doc §7.1)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class CheckResult:
    item: CheckItem
    status: CheckStatus
    detail: str = ""


@dataclass
class CredibilityReport:
    """A single report instance; serialisable to JSON (design doc §7.1)."""

    model_name: str
    task: str
    layer: int | None = None
    k: int | None = None
    checks: list[CheckResult] = field(default_factory=list)
    grade: CredibilityGrade = CredibilityGrade.D
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Serialise to the JSON schema emitted by the CLI and library."""
        return {
            "model": self.model_name,
            "task": self.task,
            "layer": self.layer,
            "k": self.k,
            "grade": self.grade.value,
            "created_at": self.created_at,
            "checks": [
                {"item": check.item.value, "status": check.status.value, "detail": check.detail}
                for check in self.checks
            ],
        }