"""Repair ground-truth search and verification (design doc §5.5.1).

Mean ablation (the fixed primary intervention from ``ground_truth.judgment``)
is applied to a candidate component subset; a subset is a *repair set* when
the bug trigger rate drops by >= 80% relative and to <= 10% absolute while
normal retention stays >= 95%.  The exhaustive search enumerates all subsets
of the component pool and keeps the minimal repair sets; greedy forward
selection + backward deletion approximates the same sets; repeated exclusion
searches (up to ``MAX_CONJUNCTS``) recover the DNF of alternative paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ground_truth.dnf import MAX_CONJUNCTS
from ground_truth.judgment import (
    NORMAL_RETENTION,
    TRIGGER_ABS_CEILING,
    TRIGGER_REDUCTION,
)
from inject_bugs.data_generation import BugDataset
from inject_bugs.toy_model import (
    ComponentKey,
    joint_trigger_normal_rates,
    key_str,
    normal_accuracy,
)

SearchStats = dict[str, float | int]


@dataclass
class RepairJudgment:
    """Outcome of mean-ablating one component subset."""

    trigger_rate: float
    retention: float
    relative_drop: float
    success: bool
    base_trigger_rate: float

    def to_dict(self) -> dict:
        return {
            "trigger_rate": self.trigger_rate,
            "retention": self.retention,
            "relative_drop": self.relative_drop,
            "success": self.success,
            "base_trigger_rate": self.base_trigger_rate,
        }


def judge_repair(
    model: torch.nn.Module,
    ablated: frozenset[ComponentKey] | set[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_trigger_rate: float,
    base_norm_accuracy: float | None = None,
) -> RepairJudgment:
    """Mean-ablate ``ablated`` and judge repair success (judgment.py rules).

    ``base_norm_accuracy`` may be passed in to avoid re-evaluating the
    unablated normal accuracy on every subset during a search.
    """
    trig, norm = joint_trigger_normal_rates(
        model, data.eval_trigger, data.eval_normal, data.bug_answer, means=means, ablated=ablated
    )
    if base_norm_accuracy is None:
        base_norm_accuracy = normal_accuracy(model, data.eval_normal)
    retention = norm / base_norm_accuracy if base_norm_accuracy > 0 else 0.0
    relative_drop = (base_trigger_rate - trig) / base_trigger_rate if base_trigger_rate > 0 else 0.0
    success = (
        relative_drop >= TRIGGER_REDUCTION
        and trig <= TRIGGER_ABS_CEILING
        and retention >= NORMAL_RETENTION
    )
    return RepairJudgment(
        trigger_rate=trig,
        retention=retention,
        relative_drop=relative_drop,
        success=success,
        base_trigger_rate=base_trigger_rate,
    )


def _sort_key(conjunct: frozenset[ComponentKey]) -> tuple[int, tuple[str, ...]]:
    return (len(conjunct), tuple(sorted(key_str(k) for k in conjunct)))


def exhaustive_search(
    model: torch.nn.Module,
    pool: list[ComponentKey] | set[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_trigger_rate: float,
) -> tuple[list[frozenset[ComponentKey]], SearchStats]:
    """Enumerate every subset of the pool; return all minimal repair sets."""
    t0 = time.perf_counter()
    pool = list(pool)
    n = len(pool)
    base_norm = normal_accuracy(model, data.eval_normal)
    succeeded: set[frozenset[ComponentKey]] = set()
    n_evals = 0
    for mask in range(1, 1 << n):
        keys = frozenset(pool[i] for i in range(n) if (mask >> i) & 1)
        if judge_repair(model, keys, means, data, base_trigger_rate, base_norm).success:
            succeeded.add(keys)
        n_evals += 1
    minimal = sorted(
        (s for s in succeeded if not any(p < s for p in succeeded)),
        key=_sort_key,
    )
    wall = time.perf_counter() - t0
    return minimal, {"n_evals": n_evals, "wall_s": wall}


def greedy_search(
    model: torch.nn.Module,
    pool: list[ComponentKey] | set[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_trigger_rate: float,
) -> tuple[frozenset[ComponentKey] | None, SearchStats]:
    """Greedy forward selection + backward deletion with seed restarts.

    Pure AND conjuncts give no single-component signal, so a single forward
    pass from the empty set is blind (it stops at the first frozen head it
    tries).  We restart forward selection from the empty set *and* from every
    single component, then backward-delete each candidate to minimality and
    return the smallest successful set (the same ordering the exhaustive
    search uses).  If no restart yields a repair, each restart retries with
    two-component jumps, which recovers conjuncts that need > 1 component
    before any signal appears (e.g. 3-way ANDs).
    """
    t0 = time.perf_counter()
    pool = list(pool)
    base_norm = normal_accuracy(model, data.eval_normal)
    memo: dict[frozenset[ComponentKey], RepairJudgment] = {}

    def judge(keys: frozenset[ComponentKey]) -> RepairJudgment:
        if keys not in memo:
            memo[keys] = judge_repair(model, keys, means, data, base_trigger_rate, base_norm)
        return memo[keys]

    def forward_from(seed: set[ComponentKey], max_jump: int) -> frozenset[ComponentKey] | None:
        from itertools import combinations

        chosen = set(seed)
        while True:
            current = frozenset(chosen)
            if judge(current).success:
                break
            if len(chosen) == len(pool):
                break
            cur_rate = judge(current).trigger_rate
            remaining = [k for k in pool if k not in chosen]
            # Single additions that *strictly* reduce the trigger rate.  A tie
            # (e.g. a pure AND where every single ablation keeps the trigger
            # at 1.0) is "no signal": the old code treated the first tie as an
            # improvement and ran blind through the whole pool.
            best_key: ComponentKey | None = None
            best_rate = cur_rate
            for key in remaining:
                candidate = judge(frozenset(chosen | {key}))
                # A candidate whose ablation destroys normal retention is a
                # degenerate "trigger suppressor" (e.g. a frozen component the
                # copy task depends on) and would trap the search in a dead end.
                if candidate.retention < NORMAL_RETENTION:
                    continue
                if candidate.trigger_rate < best_rate:
                    best_rate = candidate.trigger_rate
                    best_key = key
            if best_key is not None:
                chosen.add(best_key)
                continue
            # No single addition helps: try larger jumps (pairs, triples...)
            # up to ``max_jump``, again requiring strict improvement.
            improved = False
            for jump in range(2, max_jump + 1):
                best_combo: tuple[ComponentKey, ...] | None = None
                best_rate = cur_rate
                for combo in combinations(remaining, jump):
                    candidate = judge(frozenset(chosen) | set(combo))
                    if candidate.retention < NORMAL_RETENTION:
                        continue
                    if candidate.trigger_rate < best_rate:
                        best_rate = candidate.trigger_rate
                        best_combo = combo
                if best_combo is not None:
                    chosen.update(best_combo)
                    improved = True
                    break
            if not improved:
                break

        changed = True
        while changed:
            changed = False
            for key in sorted(chosen, key=lambda k: pool.index(k)):
                candidate = frozenset(chosen - {key})
                if judge(candidate).success:
                    chosen.remove(key)
                    changed = True

        result = frozenset(chosen)
        return result if judge(result).success else None

    seeds = [frozenset()] + [frozenset([key]) for key in pool]
    candidates = [c for c in (forward_from(set(s), 1) for s in seeds) if c is not None]
    if not candidates:
        # Deep ANDs give no signal to single additions at all; retry every
        # restart with two- and three-component jumps.
        candidates = [c for c in (forward_from(set(s), 3) for s in seeds) if c is not None]
    wall = time.perf_counter() - t0
    if not candidates:
        return None, {"n_evals": len(memo), "wall_s": wall}
    return min(candidates, key=_sort_key), {"n_evals": len(memo), "wall_s": wall}


def recover_dnf(
    model: torch.nn.Module,
    components: list[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_trigger_rate: float,
    mode: str = "exhaustive",
    max_conjuncts: int = MAX_CONJUNCTS,
) -> tuple[list[frozenset[ComponentKey]], SearchStats]:
    """Find up to ``MAX_CONJUNCTS`` alternative minimal repair sets.

    After each conjunct is found its components are excluded from the pool
    and the search continues (design doc §5.5.1, step 3).
    """
    pool = list(components)
    conjuncts: list[frozenset[ComponentKey]] = []
    stats: SearchStats = {"n_evals": 0, "wall_s": 0.0}
    for _ in range(max_conjuncts):
        if not pool:
            break
        if mode == "exhaustive":
            found, step_stats = exhaustive_search(model, pool, means, data, base_trigger_rate)
            if not found:
                break
            chosen = found[0]
        elif mode == "greedy":
            chosen, step_stats = greedy_search(model, pool, means, data, base_trigger_rate)
            if chosen is None or not judge_repair(
                model, chosen, means, data, base_trigger_rate
            ).success:
                break
        else:
            raise ValueError(f"unknown search mode {mode!r}")
        conjuncts.append(frozenset(chosen))
        stats["n_evals"] = int(stats["n_evals"]) + int(step_stats["n_evals"])
        stats["wall_s"] = float(stats["wall_s"]) + float(step_stats["wall_s"])
        pool = [c for c in pool if c not in chosen]
    return conjuncts, stats


def union(conjuncts: list[frozenset[ComponentKey]]) -> frozenset[ComponentKey]:
    """Union of all conjunct components."""
    result: frozenset[ComponentKey] = frozenset()
    return result.union(*conjuncts) if conjuncts else result


def set_iou(a: set[ComponentKey], b: set[ComponentKey]) -> float:
    """Jaccard index of two component sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union_size = len(a | b)
    return intersection / union_size if union_size else 0.0


def recall(union_recovered: set[ComponentKey], union_implanted: set[ComponentKey]) -> float:
    """Fraction of implanted truth components recovered by the search."""
    if not union_implanted:
        return 1.0 if not union_recovered else 0.0
    return len(union_recovered & union_implanted) / len(union_implanted)


def conjunct_f1(
    greedy_conjuncts: list[frozenset[ComponentKey]],
    exhaustive_conjuncts: list[frozenset[ComponentKey]],
) -> float:
    """Set-level F1 between two collections of conjuncts (order-insensitive)."""
    g = set(greedy_conjuncts)
    e = set(exhaustive_conjuncts)
    if not g and not e:
        return 1.0
    tp = len(g & e)
    precision = tp / len(g) if g else 0.0
    recall_rate = tp / len(e) if e else 0.0
    if precision + recall_rate == 0:
        return 0.0
    return 2 * precision * recall_rate / (precision + recall_rate)


def component_f1(greedy_union: set[ComponentKey], exhaustive_union: set[ComponentKey]) -> float:
    """Component-level F1 between two truth sets."""
    if not greedy_union and not exhaustive_union:
        return 1.0
    tp = len(greedy_union & exhaustive_union)
    precision = tp / len(greedy_union) if greedy_union else 0.0
    recall_rate = tp / len(exhaustive_union) if exhaustive_union else 0.0
    if precision + recall_rate == 0:
        return 0.0
    return 2 * precision * recall_rate / (precision + recall_rate)

