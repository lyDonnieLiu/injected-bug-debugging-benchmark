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
from inject_bugs.hooked_utils import (
    ComponentKey,
    joint_trigger_normal_rates,
    key_str,
    normal_accuracy,
)

SearchStats = dict[str, float | int | bool]


class _BudgetExceeded(Exception):
    """Internal signal: greedy search exhausted its evaluation budget."""


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

    Trigger labels come from ``data.trigger_labels`` when present (Phase B
    per-sample bug answers), otherwise from ``data.bug_answer`` (Phase A).
    """
    labels = (
        data.trigger_labels
        if getattr(data, "trigger_labels", None) is not None
        else data.bug_answer
    )
    trig, norm = joint_trigger_normal_rates(
        model, data.eval_trigger, data.eval_normal, labels, means=means, ablated=ablated
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
    max_evals: int = 5000,
    early_stop: bool = False,
    max_wall_s: float | None = None,
) -> tuple[frozenset[ComponentKey] | None, SearchStats]:
    """Greedy forward selection + backward deletion with seed restarts.

    ``max_evals`` caps the number of distinct mean-ablation evaluations
    (each is one memoized forward pass); once the cap is reached the search
    aborts and returns ``None`` with ``budget_exceeded=True`` in the stats.
    On a budget abort the stats also carry ``budget_phase`` ("single_scan"
    vs "jump") and ``budget_restart`` so callers can tell whether the search
    died during the single-component scan or during pair/triple jumps.

    Pure AND conjuncts give no single-component signal, so a single forward
    pass from the empty set is blind (it stops at the first frozen head it
    tries).  We restart forward selection from the empty set *and* from every
    single component, then backward-delete each candidate to minimality and
    return the smallest successful set (the same ordering the exhaustive
    search uses).  If no restart yields a repair, each restart retries with
    two-component jumps, which recovers conjuncts that need > 1 component
    before any signal appears (e.g. 3-way ANDs).

    ``early_stop`` returns as soon as the first restart yields a repair
    instead of collecting every restart's candidate and picking the min.
    The empty restart scans every single component, so any 1-way repair is
    always found on the first restart; early-stopping on it preserves the
    minimality ordering while avoiding the ~``n_restarts x n_pool`` scan
    that exhausts a small budget on large pools (156 components).
    ``max_wall_s`` is a wall-clock cap per greedy step (defensive; pair/triple
    jump enumeration can otherwise run unbounded between eval ticks).
    """
    t0 = time.perf_counter()
    pool = list(pool)
    base_norm = normal_accuracy(model, data.eval_normal)
    memo: dict[frozenset[ComponentKey], RepairJudgment] = {}

    def judge(keys: frozenset[ComponentKey]) -> RepairJudgment:
        if keys not in memo:
            if len(memo) >= max_evals:
                raise _BudgetExceeded
            if max_wall_s is not None and time.perf_counter() - t0 > max_wall_s:
                raise _BudgetExceeded
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

    budget_phase: str | None = None
    budget_restart: frozenset[ComponentKey] | None = None
    candidates: list[frozenset[ComponentKey]] = []
    try:
        seeds = [frozenset()] + [frozenset([key]) for key in pool]
        for seed in seeds:
            budget_restart = seed
            budget_phase = "single_scan"
            cand = forward_from(set(seed), 1)
            if cand is not None:
                candidates.append(cand)
                if early_stop:
                    break
        if not candidates:
            # Deep ANDs give no signal to single additions at all; retry every
            # restart with two- and three-component jumps.
            budget_phase = "jump"
            for seed in seeds:
                budget_restart = seed
                cand = forward_from(set(seed), 3)
                if cand is not None:
                    candidates.append(cand)
                    if early_stop:
                        break
    except _BudgetExceeded:
        wall = time.perf_counter() - t0
        return None, {
            "n_evals": len(memo),
            "wall_s": wall,
            "budget_exceeded": True,
            "budget_phase": budget_phase,
            "budget_restart": key_str(next(iter(budget_restart))) if budget_restart else "empty",
        }
    wall = time.perf_counter() - t0
    if not candidates:
        return None, {"n_evals": len(memo), "wall_s": wall}
    return min(candidates, key=_sort_key), {"n_evals": len(memo), "wall_s": wall}


def pair_repair_search(
    model: torch.nn.Module,
    pool: list[ComponentKey] | set[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_trigger_rate: float,
    judge=None,
    max_evals: int = 15000,
    max_wall_s: float = 21600.0,
    early_stop: bool = True,
) -> tuple[list[frozenset[ComponentKey]], SearchStats]:
    """Exhaustively mean-ablate every unordered *pair* of ``pool``.

    Negative-endpoint closure (next_step_research_plan.md v3 A'): for bugs whose
    single-component scan finds no repair (TB/KC/FR, 153-component pool after
    suppressing the generic early MLPs) this asks whether *any* two-component
    joint ablation satisfies the repair protocol without destroying normal
    behaviour.  Each pair is one ``judge`` call (joint mean ablation).  Returns
    ``(successful_pairs, stats)`` where each successful pair is a 2-element
    frozenset, and ``stats`` carries the evaluation/budget accounting so a
    budget-cap abort is distinguishable from "scanned everything, none repair".

    ``judge`` is injectable for tests (defaults to :func:`judge_repair` with the
    unablated normal accuracy precomputed once, matching
    :func:`single_component_judgments`).  ``early_stop`` returns on the first
    repair pair instead of finishing the enumeration (the pre-registered claim
    only needs existence; a full empty enumeration is completed anyway).
    """
    from itertools import combinations

    pool = list(pool)
    if judge is None:
        base_norm = normal_accuracy(model, data.eval_normal)

        def judge(keys: frozenset[ComponentKey]) -> RepairJudgment:
            return judge_repair(model, keys, means, data, base_trigger_rate, base_norm)

    n_total = len(pool) * (len(pool) - 1) // 2
    t0 = time.perf_counter()
    deadline = None if max_wall_s is None else t0 + max_wall_s
    successes: list[frozenset[ComponentKey]] = []
    budget_exceeded = False
    n_judged = 0
    for a, b in combinations(pool, 2):
        if n_judged >= max_evals or (deadline is not None and time.perf_counter() > deadline):
            budget_exceeded = True
            break
        keys = frozenset([a, b])
        if judge(keys).success:
            successes.append(keys)
            if early_stop:
                n_judged += 1
                break
        n_judged += 1
    wall = time.perf_counter() - t0
    return successes, {
        "n_pairs_total": n_total,
        "n_pairs_judged": n_judged,
        "n_repair_pairs": len(successes),
        "budget_exceeded": budget_exceeded,
        "n_evals": n_judged,
        "wall_s": wall,
    }


def recover_dnf(
    model: torch.nn.Module,
    components: list[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_trigger_rate: float,
    mode: str = "exhaustive",
    max_conjuncts: int = MAX_CONJUNCTS,
    max_evals: int = 5000,
    early_stop: bool = False,
    max_wall_s: float | None = None,
) -> tuple[list[frozenset[ComponentKey]], SearchStats]:
    """Find up to ``MAX_CONJUNCTS`` alternative minimal repair sets.

    After each conjunct is found its components are excluded from the pool
    and the search continues (design doc §5.5.1, step 3).  ``max_evals``
    bounds each greedy step; a step that hits the cap aborts and the whole
    recovery returns an empty truth with ``budget_exceeded`` flagged.
    ``early_stop`` is forwarded to :func:`greedy_search` (Phase B uses it so
    a single-component repair is found in ~one restart instead of burning the
    budget scanning every restart). ``max_wall_s`` is a per-step wall-clock
    cap, also forwarded to :func:`greedy_search`.
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
            chosen, step_stats = greedy_search(
                model, pool, means, data, base_trigger_rate,
                max_evals=max_evals,
                early_stop=early_stop,
                max_wall_s=max_wall_s,
            )
            # Accumulate eval/wall stats before the abort check: a step that
            # hits the budget cap returns ``chosen=None`` and we must still
            # record how much it spent (previously the stats stayed 0 and the
            # report read as "search never ran").
            stats["n_evals"] = int(stats["n_evals"]) + int(step_stats["n_evals"])
            stats["wall_s"] = float(stats["wall_s"]) + float(step_stats["wall_s"])
            stats["budget_exceeded"] = bool(
                stats.get("budget_exceeded", False)
                or step_stats.get("budget_exceeded", False)
            )
            step_trace = {
                **step_stats,
                "conjunct": None if chosen is None else sorted(key_str(k) for k in chosen),
            }
            stats.setdefault("steps", []).append(step_trace)
            # Propagate the abort phase/restart to the top level so the report
            # can distinguish "died during the single-component scan" from
            # "died during pair/triple jumps" without digging into steps.
            for field in ("budget_phase", "budget_restart"):
                if step_stats.get(field) is not None:
                    stats[field] = step_stats[field]
            if chosen is None or not judge_repair(
                model, chosen, means, data, base_trigger_rate
            ).success:
                break
        else:
            raise ValueError(f"unknown search mode {mode!r}")
        conjuncts.append(frozenset(chosen))
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


def single_component_judgments(
    model: torch.nn.Module,
    pool: list[ComponentKey] | set[ComponentKey],
    means: dict[ComponentKey, torch.Tensor],
    data: BugDataset,
    base_trigger_rate: float,
) -> tuple[dict[ComponentKey, RepairJudgment], SearchStats]:
    """Mean-ablate every single component and return per-component judgments.

    A component is *necessary* for the bug when ablating it alone satisfies
    the repair protocol (design doc §5.5.1); the resulting binary labels are
    the necessity ground truth used by the fairness Kendall tau comparison
    (design doc §8.1 item 5).
    """
    pool = list(pool)
    base_norm = normal_accuracy(model, data.eval_normal)
    t0 = time.perf_counter()
    judgments: dict[ComponentKey, RepairJudgment] = {}
    for key in pool:
        judgments[key] = judge_repair(
            model, frozenset([key]), means, data, base_trigger_rate, base_norm
        )
    wall = time.perf_counter() - t0
    return judgments, {"n_evals": len(pool), "wall_s": wall}
