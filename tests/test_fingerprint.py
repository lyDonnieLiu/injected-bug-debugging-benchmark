"""Fingerprint drift tests (next_step_research_plan.md v3 "配置指纹扩展").

Pre-registered regression: *any* protocol-identity axis change must flip the
fingerprint (cache invalidation), identical input must be deterministic, and
list-like axes must be order-invariant (same component set, any permutation).
"""

from __future__ import annotations

import json

from common.fingerprint import fingerprint_fields, protocol_fingerprint

BASE = dict(
    protocol_version="phase_b_negpair_v1",
    bug="trigger_backdoor",
    seed=1,
    intervention="mean_ablation",
    rank=8,
    target_matrices=["c_attn", "c_proj", "c_fc"],
    window=[8, 9, 10, 11],
)


def test_deterministic() -> None:
    assert protocol_fingerprint(**BASE) == protocol_fingerprint(**BASE)


def test_any_axis_change_flips_digest() -> None:
    """One-field mutations each invalidate the cache (plan §实现改动与交付物)."""
    baseline = protocol_fingerprint(**BASE)
    mutations = {
        "protocol_version": "phase_b_acceptance_v1",
        "bug": "knowledge_conflict",
        "seed": 2,
        "intervention": "zero_ablation",
        "rank": 16,
        "target_matrices": ["c_attn", "c_proj"],
        "window": [7, 8, 9, 10],
    }
    for field, value in mutations.items():
        altered = dict(BASE)
        altered[field] = value
        assert protocol_fingerprint(**altered) != baseline, field


def test_list_axes_order_invariant() -> None:
    """Window / target-matrices permutations collide (same injection geometry)."""
    a = protocol_fingerprint(**BASE)
    b = protocol_fingerprint(**{**BASE, "window": [11, 10, 9, 8]})
    assert a == b
    c = protocol_fingerprint(**{**BASE, "target_matrices": ["c_fc", "c_attn", "c_proj"]})
    assert a == c


def test_list_axes_order_sensitive_to_set_change() -> None:
    """A *different* component set must still flip the digest."""
    a = protocol_fingerprint(**BASE)
    b = protocol_fingerprint(**{**BASE, "window": [7, 8, 9, 10]})
    assert a != b


def test_int_and_int_string_collide() -> None:
    """``window=8`` vs ``window="8"`` are the same layer index."""
    a = protocol_fingerprint(**{**BASE, "rank": 16})
    b = protocol_fingerprint(**{**BASE, "rank": "16"})
    assert a == b


def test_empty_window_none_and_empty_list_distinct_from_window() -> None:
    """``window=None`` (all layers) differs from an explicit window."""
    full = protocol_fingerprint(**{**BASE, "window": None})
    explicit = protocol_fingerprint(**{**BASE, "window": [8, 9, 10, 11]})
    assert full != explicit


def test_fields_are_json_safe_sorted() -> None:
    fields = fingerprint_fields(**BASE)
    # fields round-trip through JSON stably (sort_keys=True used by the digest);
    # tuple-valued fields serialize to lists, so compare re-dumped strings.
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    assert json.dumps(json.loads(payload), sort_keys=True, ensure_ascii=False) == payload


def test_run_phase_b_folds_protocol_axes() -> None:
    """The per-seed analysis cache key must flip when a protocol axis changes."""
    from scripts.run_phase_b import _analysis_fingerprint

    cfgs = ({"samples": {"n": 8}}, {"search": {"max_evals": 5000}})
    base = _analysis_fingerprint(*cfgs, bug="compositional_logic", seed=1,
                                 rank=8, window=[8, 9, 10, 11])
    same = _analysis_fingerprint(*cfgs, bug="compositional_logic", seed=1,
                                 rank=8, window=[11, 10, 9, 8])
    other = _analysis_fingerprint(*cfgs, bug="compositional_logic", seed=1,
                                  rank=16, window=[8, 9, 10, 11])
    assert base == same          # deterministic + order-invariant
    assert base != other         # rank flip invalidates the cache

    # legacy varargs-only call (no explicit axes) still deterministic & stable
    legacy = _analysis_fingerprint({"samples": {"n": 8}})
    assert legacy == _analysis_fingerprint({"samples": {"n": 8}})
    assert legacy != _analysis_fingerprint({"samples": {"n": 9}})
