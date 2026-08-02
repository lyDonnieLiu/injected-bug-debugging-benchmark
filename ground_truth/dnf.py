"""DNF (disjunctive normal form) ground truth (design doc §5.5.1, §6.2).

Alternative repair paths are expressed as ``(A∧B) ∨ (C)``; each conjunct is a
minimal repair set. A method "hits" the truth when its top-k set contains at
least one complete conjunct; partial hits are scored by max coverage.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

MAX_CONJUNCTS = 5  # design doc §5.5.1: cap at 5 conjuncts, truncate beyond


@dataclass(frozen=True)
class DnfTruth:
    """A DNF ground truth: a tuple of minimal repair sets (conjuncts)."""

    conjuncts: tuple[frozenset[Hashable], ...] = field(default_factory=tuple)

    def full_hit(self, top_set: set[Hashable]) -> bool:
        """Return whether ``top_set`` contains at least one complete conjunct."""
        return any(conjunct <= top_set for conjunct in self.conjuncts)

    def max_coverage(self, top_set: set[Hashable]) -> float:
        """Return ``max_j |T∩C_j| / |C_j|`` (design doc §6.2 partial-hit score)."""
        if not self.conjuncts:
            return 0.0
        return max(len(conjunct & top_set) / len(conjunct) for conjunct in self.conjuncts)