"""A' pair-enumeration search tests (next_step_research_plan.md v3 A').

The pair scan runs a real model on GPU; here we test the *pure* enumeration
logic with an injected ``judge`` callable (no model, no data, no torch):
exhaustive pair counting, repair-pair collection, early-stop on first success,
and budget-cap accounting.  Driver-level suppressor exclusion is tested
separately against ``ground_truth.truth_typology``.
"""

from __future__ import annotations

from types import SimpleNamespace

from ground_truth.repair_search import pair_repair_search
from inject_bugs.hooked_utils import head_key, mlp_key


def _fake_judge(repairs: set[frozenset]) -> tuple:
    """Return ``(judge, calls)``; ``judge`` succeeds exactly on ``repairs``."""
    calls: list = []

    def judge(keys):
        calls.append(keys)
        return SimpleNamespace(success=frozenset(keys) in repairs)

    return judge, calls


def _pool(n_heads: int = 3, n_mlps: int = 2) -> list:
    keys = [head_key(0, h) for h in range(n_heads)]
    keys += [mlp_key(layer) for layer in range(n_mlps)]
    return keys


def test_exhaustive_none_repair_scans_every_pair() -> None:
    """An empty result must reflect a full C(n,2) enumeration, not a stop."""
    pool = _pool(4, 1)          # 5 components -> C(5,2) = 10 pairs
    judge, calls = _fake_judge(set())
    pairs, stats = pair_repair_search(None, pool, None, None, 1.0, judge=judge)
    assert pairs == []
    assert stats["n_pairs_total"] == 10
    assert stats["n_pairs_judged"] == 10
    assert stats["n_repair_pairs"] == 0
    assert stats["budget_exceeded"] is False
    assert len(calls) == 10


def test_collects_repair_pairs_full_enumeration() -> None:
    pool = _pool(3, 1)          # 4 components -> 6 pairs
    hit = frozenset([pool[0], pool[1]])   # head(0,0)+head(0,1) is a repair
    judge, _ = _fake_judge({hit})
    pairs, stats = pair_repair_search(None, pool, None, None, 1.0, judge=judge,
                                      early_stop=False)
    assert pairs == [hit]
    assert stats["n_pairs_judged"] == 6   # scanned everything despite a success
    assert stats["n_repair_pairs"] == 1


def test_early_stop_on_first_repair() -> None:
    pool = _pool(3, 1)
    hit = frozenset([pool[0], pool[1]])   # first pair in enumeration order
    judge, calls = _fake_judge({hit})
    pairs, stats = pair_repair_search(None, pool, None, None, 1.0, judge=judge,
                                      early_stop=True)
    assert pairs == [hit]
    assert stats["n_pairs_total"] == 6
    assert stats["n_pairs_judged"] == 1   # stopped at the very first success
    assert stats["n_repair_pairs"] == 1
    assert stats["budget_exceeded"] is False
    assert len(calls) == 1

    # no repair pair at all -> early_stop cannot trigger; scan completes
    judge_empty, calls_empty = _fake_judge(set())
    pairs_empty, stats_empty = pair_repair_search(
        None, pool, None, None, 1.0, judge=judge_empty, early_stop=True
    )
    assert pairs_empty == []
    assert stats_empty["n_pairs_judged"] == 6


def test_budget_cap_flags_budget_exceeded() -> None:
    """A max_evals cap smaller than C(n,2) aborts with budget_exceeded=True."""
    pool = _pool(6, 2)          # 8 components -> C(8,2) = 28 pairs
    judge, calls = _fake_judge(set())
    pairs, stats = pair_repair_search(None, pool, None, None, 1.0, judge=judge,
                                      max_evals=10, max_wall_s=None)
    assert pairs == []
    assert stats["n_pairs_total"] == 28
    assert stats["n_pairs_judged"] == 10
    assert stats["budget_exceeded"] is True


def test_two_component_pool_smoke() -> None:
    """Smoke (plan: pair enumeration over a 2-component subset) -> one pair."""
    pool = _pool(1, 1)          # head(0,0) + mlp(0)
    hit = frozenset(pool)
    judge, _ = _fake_judge({hit})
    pairs, stats = pair_repair_search(None, pool, None, None, 1.0, judge=judge)
    assert pairs == [hit]
    assert stats["n_pairs_total"] == 1
    assert stats["n_pairs_judged"] == 1


def test_pair_members_are_component_keys() -> None:
    pool = _pool(3, 1)
    judge, _ = _fake_judge(set())
    pairs, _ = pair_repair_search(None, pool, None, None, 1.0, judge=judge)
    assert pairs == []
    # the vocabulary the enumeration consumes is the canonical (kind, layer, id) tuples
    assert all(len(k) >= 2 and k[0] in {"head", "mlp"} for k in pool)
    assert {k[0] for k in pool} == {"head", "mlp"}
