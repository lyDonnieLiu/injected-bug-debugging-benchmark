"""Truth typology for Step 1 diagnostics (research_plan_revised.md §5.1).

The strict repair protocol (judgment.py) only labels a component *necessary*
when ablating it alone satisfies the full repair rule (trigger relative drop
>= 0.80, absolute <= 0.10, normal retention >= 0.95).  Distributed token-reading
bugs (TB/KC/FR) never satisfy this, yet their strongest single-component
effects are usually generic early-layer MLPs whose ablation also destroys
normal behaviour.  This module derives the *relaxed* truth layers and the
failure-mode taxonomy that the revised plan uses instead of a single
all-or-nothing "strict repair truth".

Three overlapping component sets are distinguished:

* **strong effect** -- single-component relative trigger drop >= 0.20,
  *regardless* of retention.  This is the raw "has an effect on the bug"
  signal and the set over which suppressor pollution is detected.
* **必要影响集 (necessary effect)** -- strong effect AND normal retention
  >= 0.95 (plan §5.1).  A non-destructive candidate truth; destructive
  suppressors never enter it.
* **非破坏性影响集 (non-destructive)** -- necessary-effect set minus generic
  early-layer MLP suppressors (plan §5.1).  A generic suppressor's ablation
  drops the trigger to <= 0.10 *and* destroys normal (ret < 0.95), so by
  definition it cannot be in the necessary-effect set: this exclusion is
  documented for completeness and in practice the set equals the necessary
  effect set.  The suppressor pollution is surfaced separately via
  :func:`general_mlp_suppressors`, which is what the failure-mode
  classification actually consumes.

All functions are pure and deterministic over the existing
``judgments``/``conjuncts`` structures so they slot into the Step 1 matrix
driver without touching the core search code.
"""

from __future__ import annotations

from dataclasses import dataclass

from ground_truth.judgment import NORMAL_RETENTION
from ground_truth.repair_search import RepairJudgment
from inject_bugs.hooked_utils import ComponentKey, key_str

# research_plan_revised.md §5.1: "必要影响集"
NECESSARY_EFFECT_TRIGGER_DROP = 0.20
# research_plan_revised.md §5.1: 非破坏性影响集排除"前 1/3 层且同时破坏 normal"的通用 MLP
GENERIC_MLP_FRACTION = 1 / 3


@dataclass(frozen=True)
class FailureMode:
    """Why a bug has no strict repair truth under a given injection config."""

    name: str
    n_destructive_suppressors: int
    n_clean_components: int
    strongest_clean: list[str]
    n_truth_components: int


@dataclass(frozen=True)
class TruthSummary:
    """The typology output for one (bug, config) seed-1 diagnostic point."""

    n_strict_necessary: int
    n_effect: int
    n_non_destructive: int
    n_suppressors: int
    effect_components: list[str]
    non_destructive_components: list[str]
    general_suppressors: list[str]
    failure_mode: FailureMode | None


def _is_general_mlp(key: ComponentKey, n_layers: int, judgments: dict) -> bool:
    """A generic early-layer MLP suppressor: drops trigger but destroys normal."""
    kind, layer, *_ = key
    if kind != "mlp" or layer >= n_layers * GENERIC_MLP_FRACTION:
        return False
    judgment = judgments.get(key)
    if judgment is None:
        return False
    return judgment.trigger_rate <= 0.10 and judgment.retention < NORMAL_RETENTION


def _normalise_keys(
    keys: list[ComponentKey], judgments: dict[ComponentKey, RepairJudgment]
) -> tuple[list[ComponentKey], int]:
    """Order ``keys`` deterministically; fall back to the judgment keys.

    ``judgments`` comes from ``single_component_judgments`` over the full pool,
    but a caller may pass a restricted key list (e.g. a truth pool) that shares
    only a subset; classification must stay total over ``keys``.  The layer
    count is inferred from the maximum layer index (for the generic-MLP rule).
    """
    if not keys:
        keys = list(judgments.keys())
    n_layers = max((int(k[1]) for k in keys if len(k) > 1), default=0) + 1
    return keys, n_layers


def strong_effect_components(
    keys: list[ComponentKey],
    judgments: dict[ComponentKey, RepairJudgment],
    drop: float = NECESSARY_EFFECT_TRIGGER_DROP,
) -> list[ComponentKey]:
    """Single-component relative trigger drop >= ``drop``, regardless of retention."""
    ordered, _ = _normalise_keys(keys, judgments)
    return [
        k
        for k in ordered
        if k in judgments and judgments[k].relative_drop >= drop
    ]


def necessary_effect_components(
    keys: list[ComponentKey],
    judgments: dict[ComponentKey, RepairJudgment],
    drop: float = NECESSARY_EFFECT_TRIGGER_DROP,
) -> list[ComponentKey]:
    """必要影响集: strong single-component effect AND normal retention >= 0.95.

    ``keys`` orders the result; when empty the judgment keys are used.
    Non-positive base trigger makes relative_drop undefined (0.0), which never
    passes the threshold.
    """
    ordered, _ = _normalise_keys(keys, judgments)
    return [
        k
        for k in ordered
        if k in judgments
        and judgments[k].relative_drop >= drop
        and judgments[k].retention >= NORMAL_RETENTION
    ]


def general_mlp_suppressors(
    keys: list[ComponentKey],
    judgments: dict[ComponentKey, RepairJudgment],
    n_layers: int | None = None,
    drop: float = NECESSARY_EFFECT_TRIGGER_DROP,
) -> list[ComponentKey]:
    """Strong-effect components that are generic early-layer MLP suppressors.

    This is the "通用抑制器污染" the plan's 非破坏性影响集 is meant to reduce:
    the mlp(0/1/2)-style components whose ablation suppresses the trigger only
    by breaking the model.  They are strong-effect (drop >= ``drop``) but
    destructive, so they never appear in the necessary-effect set.
    """
    ordered, inferred = _normalise_keys(keys, judgments)
    if n_layers is None:
        n_layers = inferred
    return [
        k
        for k in ordered
        if k in judgments
        and judgments[k].relative_drop >= drop
        and _is_general_mlp(k, n_layers, judgments)
    ]


def non_destructive_components(
    keys: list[ComponentKey],
    judgments: dict[ComponentKey, RepairJudgment],
    n_layers: int | None = None,
    drop: float = NECESSARY_EFFECT_TRIGGER_DROP,
) -> tuple[list[ComponentKey], list[ComponentKey]]:
    """非破坏性影响集: necessary-effect components minus generic MLP suppressors.

    Returns ``(kept, excluded)`` where ``excluded`` is the generic
    early-layer MLP suppressor subset of the effect set (plan §5.1).  A
    suppressor requires ``ret < 0.95`` while the necessary-effect set requires
    ``ret >= 0.95``, so ``excluded`` is empty by construction and ``kept``
    equals :func:`necessary_effect_components`; the function exists to make
    the non-destructive truth concept explicit.  Use
    :func:`general_mlp_suppressors` for the actual pollution report.
    """
    ordered, inferred = _normalise_keys(keys, judgments)
    if n_layers is None:
        n_layers = inferred
    effect = necessary_effect_components(ordered, judgments, drop=drop)
    excluded = [k for k in effect if _is_general_mlp(k, n_layers, judgments)]
    kept = [k for k in effect if k not in excluded]
    return kept, excluded


def classify_failure(
    keys: list[ComponentKey],
    judgments: dict[ComponentKey, RepairJudgment],
    truth_components: list[ComponentKey] | None = None,
    n_layers: int | None = None,
) -> FailureMode:
    """Classify why a bug lacks a strict repair truth.

    Failure modes (research_plan_revised.md §6 Step 2):

    1. ``destructive_suppressor`` -- every strong single-component effect is a
       generic early-layer MLP suppressor (drops the trigger but also destroys
       normal retention), so the non-destructive protocol excludes them all.
    2. ``pure_and`` -- no single component has a sufficient non-destructive
       effect, but a conjunct repair (union truth) exists.
    3. ``effect_available`` -- non-destructive strong effects exist (a relaxed
       "necessary-effect" truth is available) but no strict repair set.
    4. ``no_component_mechanism`` -- neither single-component effects nor
       conjunct repairs reach the thresholds at this granularity.
    5. ``injection_failure`` -- training did not produce the bug (handled by
       the caller via ``trigger_rate``), so there is nothing to locate.

    ``truth_components`` is the strict/conjunct repair union (empty = none).
    """
    ordered, inferred = _normalise_keys(keys, judgments)
    if n_layers is None:
        n_layers = inferred
    truth = set(truth_components or [])
    strong = strong_effect_components(ordered, judgments)
    suppressors = [k for k in strong if _is_general_mlp(k, n_layers, judgments)]
    clean = [
        k
        for k in strong
        if k not in suppressors and judgments[k].retention >= NORMAL_RETENTION
    ]

    if truth:
        name = "strict" if clean else "pure_and"
    elif clean:
        name = "effect_available"
    elif suppressors:
        name = "destructive_suppressor"
    else:
        name = "no_component_mechanism"
    return FailureMode(
        name=name,
        n_destructive_suppressors=len(suppressors),
        n_clean_components=len(clean),
        strongest_clean=[key_str(k) for k in clean[:5]],
        n_truth_components=len(truth),
    )


def summarize(
    keys: list[ComponentKey],
    judgments: dict[ComponentKey, RepairJudgment],
    truth_components: list[ComponentKey] | None = None,
    n_layers: int | None = None,
    trigger_rate: float | None = None,
    trigger_target: float = 0.90,
) -> TruthSummary:
    """Bundle the typology for one diagnostic point into a JSON-able summary."""
    ordered, inferred = _normalise_keys(keys, judgments)
    if n_layers is None:
        n_layers = inferred
    effect = necessary_effect_components(ordered, judgments)
    non_destructive, _ = non_destructive_components(ordered, judgments, n_layers=n_layers)
    suppressors = general_mlp_suppressors(ordered, judgments, n_layers=n_layers)
    # strict necessary = single-component repair success over the full pool
    n_strict = sum(1 for k in ordered if k in judgments and judgments[k].success)
    failure = None
    if trigger_rate is not None and trigger_rate < trigger_target:
        failure = FailureMode(
            name="injection_failure",
            n_destructive_suppressors=0,
            n_clean_components=0,
            strongest_clean=[],
            n_truth_components=0,
        )
    else:
        failure = classify_failure(
            ordered, judgments, truth_components=truth_components, n_layers=n_layers
        )
    return TruthSummary(
        n_strict_necessary=n_strict,
        n_effect=len(effect),
        n_non_destructive=len(non_destructive),
        n_suppressors=len(suppressors),
        effect_components=[key_str(k) for k in effect],
        non_destructive_components=[key_str(k) for k in non_destructive],
        general_suppressors=[key_str(k) for k in suppressors],
        failure_mode=failure,
    )
