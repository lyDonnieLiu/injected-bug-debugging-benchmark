"""Core-recall / main-table membership (next_step_research_plan.md v3).

The paper's main table does **not** use the stored ``hit_at_k`` (top-``k``
contains *>= 1* truth component -- ``evaluate/metrics.hit_at_k``).  It uses a
per-bug *core* membership rule, pre-registered around the bug's minimal core:

* compositional_logic core ``{head(11,2), mlp(11)}`` -- both components must
  rank within the method's top-10 on **every** seed;
* numeric_rule core ``{mlp(9)}`` -- ``mlp(9)`` must rank first (top-1) on
  every seed.

A main-table row must additionally clear the sham gap on every seed
(injected-vs-sham Kendall gap against the repair truth, ``>= 0.10``), the
control that the injected model's attribution aligns with the truth better
than a bug-free sham model's.  Methods that satisfy the member rule on all
seeds are the *main-table members*; everything else goes to the appendix with
the exact seed-level violation so the instability is visible (e.g. CL
``activation_patching`` seed 2 double-misses with sham gap -0.009).

All functions are pure over ranking structures (``[(component, score), ...]``
in descending order) and do not import torch, so they unit-test on the local
CPU with plain lists.
"""

from __future__ import annotations

from collections.abc import Sequence


def component_ranks(ranking) -> dict[str, int]:
    """1-based rank of every component in a descending ``[(key, score), ...]``.

    Rank is defined by position in the stored ordering; the ranking has
    already resolved score ties when it was written, so a boundary tie cannot
    occur here (the report's ``ranking`` is a concrete ordered list).
    """
    return {str(key): pos for pos, (key, _score) in enumerate(ranking, start=1)}


def core_recall(ranks: dict[str, int], core: Sequence[str], top_k: int) -> bool:
    """Whether every ``core`` component ranks within ``top_k`` (all-inclusive).

    Unlike ``hit_at_k`` (any-of) this is an all-of test: a two-component core
    requires *both* members in the top-``k``.  An empty ``core`` is a caller
    error (it would pass vacuously); ``top_k <= 0`` never passes.
    """
    core = list(core)
    if not core:
        raise ValueError("core must be non-empty (empty core would pass vacuously)")
    if top_k <= 0:
        return False
    for component in core:
        rank = ranks.get(str(component))
        if rank is None or rank > top_k:
            return False
    return True


def sham_gap_pass(gaps: Sequence[float | None], gate: float) -> bool:
    """Whether every per-seed sham gap meets the gate (None = unavailable)."""
    return all(gap is not None and gap >= gate for gap in gaps)


def classify_member(
    per_seed: Sequence[dict],
    core: Sequence[str],
    top_k: int,
    sham_gate: float,
) -> dict:
    """Member-rule classification over one bug x method across seeds.

    ``per_seed`` is a list of records each carrying ``ranking`` and
    ``sham_gap`` (the injected-vs-sham Kendall gap stored per seed).  Returns
    per-seed ranks/core/shame decisions plus a summary decision with reasons:

    * ``"member"`` -- core recall AND sham gap hold on every seed;
    * ``"core_miss"`` -- some seed fails core recall (ranks recorded);
    * ``"sham_miss"`` -- some seed fails the sham gap (values recorded).
    """
    rows = []
    for record in per_seed:
        ranks = component_ranks(record["ranking"])
        seed_core = {str(c): ranks.get(str(c)) for c in core}
        hit = core_recall(ranks, core, top_k)
        rows.append(
            {
                "seed": int(record["seed"]),
                "core_ranks": seed_core,
                "core_recall": bool(hit),
                "sham_gap": record.get("sham_gap"),
            }
        )
    core_ok = all(row["core_recall"] for row in rows)
    sham_ok = sham_gap_pass([row["sham_gap"] for row in rows], sham_gate)
    if core_ok and sham_ok:
        decision, reason = "member", None
    elif not core_ok:
        decision, reason = "core_miss", "some seed(s) fail core recall"
    else:
        decision, reason = "sham_miss", "some seed(s) fail the sham gap gate"
    return {
        "decision": decision,
        "reason": reason,
        "per_seed": rows,
        "core": [str(c) for c in core],
        "top_k": int(top_k),
        "sham_gate": float(sham_gate),
    }
