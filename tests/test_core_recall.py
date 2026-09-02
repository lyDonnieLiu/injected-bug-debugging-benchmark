"""Core-recall / main-table membership tests (next_step_research_plan.md v3).

Pre-registered member rule boundaries: CL = two core components both inside
top-10; numeric = single core component inside top-1.  These tests pin the
all-inclusive semantics (distinct from ``hit_at_k``'s any-of) and the
boundary cases (at top_k passes, top_k+1 fails, empty core is an error).
"""

from __future__ import annotations

import pytest

from evaluate.core_recall import (
    classify_member,
    component_ranks,
    core_recall,
    sham_gap_pass,
)

CL_CORE = ["head(11,2)", "mlp(11)"]


def _ranking(pos: dict[str, int], n: int = 12) -> list:
    """Build a descending ranking placing ``pos`` at 1-based positions."""
    pos = {str(k): int(p) for k, p in pos.items()}
    placed = {c for c in pos}
    items = []
    for rank in range(1, n + 1):
        key = next((c for c, p in pos.items() if p == rank), None)
        if key is None:
            key = f"mlp({rank})"
            while key in placed:
                key = f"head(0,{rank})"
                if key in placed:
                    key = f"head({rank},0)"
        placed.add(key)
        items.append([key, float(n - rank + 1)])
    return items


def test_component_ranks_1_based() -> None:
    ranking = [["head(11,2)", 1.0], ["mlp(11)", 0.5], ["mlp(0)", 0.0]]
    ranks = component_ranks(ranking)
    assert ranks["head(11,2)"] == 1
    assert ranks["mlp(11)"] == 2
    assert ranks["mlp(0)"] == 3
    assert "mlp(9)" not in ranks


def test_core_recall_cl_all_inclusive() -> None:
    """CL: both core components must be inside top-10 (not merely >=1 hit)."""
    inside = component_ranks(_ranking({"head(11,2)": 6, "mlp(11)": 3}, n=156))
    assert core_recall(inside, CL_CORE, top_k=10) is True
    # hit_at_k's any-of semantics would pass with just mlp(11); all-of must not
    straddle = component_ranks(_ranking({"head(11,2)": 11, "mlp(11)": 3}, n=156))
    assert core_recall(straddle, CL_CORE, top_k=10) is False


def test_core_recall_boundary_top_k_passes() -> None:
    """Component at exactly position top_k passes; top_k+1 fails."""
    at = component_ranks(_ranking({"head(11,2)": 10, "mlp(11)": 3}, n=156))
    assert core_recall(at, CL_CORE, top_k=10) is True
    just_outside = component_ranks(_ranking({"head(11,2)": 11, "mlp(11)": 3}, n=156))
    assert core_recall(just_outside, CL_CORE, top_k=10) is False


def test_core_recall_numeric_top1() -> None:
    """numeric: mlp(9) must rank first on the seed."""
    first = component_ranks(_ranking({"mlp(9)": 1}, n=156))
    assert core_recall(first, ["mlp(9)"], top_k=1) is True
    second = component_ranks(_ranking({"mlp(9)": 2}, n=156))
    assert core_recall(second, ["mlp(9)"], top_k=1) is False


def test_core_recall_missing_core_is_false() -> None:
    """A core member absent from the ranking (rank None) fails core recall."""
    ranking = _ranking({"mlp(11)": 3}, n=156)  # head(11,2) absent
    assert core_recall(component_ranks(ranking), CL_CORE, top_k=10) is False


def test_core_recall_empty_core_raises() -> None:
    """An empty core would pass vacuously -- reject it as a caller error."""
    with pytest.raises(ValueError):
        core_recall({}, [], top_k=10)


def test_core_recall_top_k_zero_never_passes() -> None:
    assert core_recall({"mlp(9)": 0}, ["mlp(9)"], top_k=0) is False


def test_sham_gap_pass() -> None:
    assert sham_gap_pass([0.165, 0.175, 0.187], gate=0.10) is True
    assert sham_gap_pass([0.165, -0.009, 0.187], gate=0.10) is False
    assert sham_gap_pass([None, 0.2], gate=0.10) is False  # unavailable -> fail


def test_classify_member_member_and_core_miss() -> None:
    def record(seed, head_rank, mlp_rank, gap):
        return {
            "seed": seed,
            "ranking": _ranking({"head(11,2)": head_rank, "mlp(11)": mlp_rank}, n=156),
            "sham_gap": gap,
        }

    member = classify_member(
        [record(1, 6, 3, 0.165), record(2, 5, 2, 0.175), record(3, 5, 3, 0.187)],
        CL_CORE, top_k=10, sham_gate=0.10,
    )
    assert member["decision"] == "member"

    core_miss = classify_member(
        [record(1, 6, 3, 0.165), record(2, 136, 2, 0.175)],  # seed2 head(11,2) way out
        CL_CORE, top_k=10, sham_gate=0.10,
    )
    assert core_miss["decision"] == "core_miss"
    assert core_miss["per_seed"][1]["core_ranks"]["head(11,2)"] == 136
    assert core_miss["per_seed"][1]["core_recall"] is False

    sham_miss = classify_member(
        [record(1, 6, 3, -0.009), record(2, 5, 2, 0.4)],
        CL_CORE, top_k=10, sham_gate=0.10,
    )
    assert sham_miss["decision"] == "sham_miss"
    assert sham_miss["per_seed"][0]["sham_gap"] == -0.009
