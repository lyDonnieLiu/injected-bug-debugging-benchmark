"""Phase A tests: toy model behaviour, injection quality gates, truth recovery.

All model tests share one session-scoped compositional injection (tiny
model, one seed) so the per-test cost stays small on CPU; the metric sanity
tests are pure numpy/scipy and need no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.seeding import set_seed
from evaluate.metrics import (
    compute_metrics,
    perfect_ranking,
    random_baseline,
    random_ranking,
)
from ground_truth.dnf import DnfTruth
from ground_truth.repair_search import (
    component_f1,
    conjunct_f1,
    recall,
    recover_dnf,
    union,
)
from inject_bugs.bugs import BugType
from inject_bugs.data_generation import generate_dataset
from inject_bugs.finetune import (
    InjectionConfig,
    check_quality,
    inject_bug,
    train_base_model,
    train_sham_model,
)
from inject_bugs.toy_model import (
    build_toy_model,
    component_keys,
    compute_mean_activations,
    head_key,
    joint_trigger_normal_rates,
    mlp_key,
)

TINY_MODEL_CFG: dict = {
    "name": "toy_transformer",
    "n_layers": 2,
    "d_model": 32,
    "d_head": 16,
    "n_heads": 2,
    "d_mlp": 128,
    "d_vocab": 48,
    "n_ctx": 12,
    "act_fn": "gelu",
    "normalization_type": "LN",
    "use_attn_result": True,
}

TINY_SAMPLES: dict = {
    "n_train_trigger": 200,
    "n_train_normal": 800,
    "n_eval_trigger": 100,
    "n_eval_normal": 200,
}

TINY_TRAINING: dict = {
    "base_steps": 400,
    "base_batch_size": 64,
    "inject_epochs": 30,
    "inject_lr": 1e-3,
    "batch_size": 32,
    "max_inject_steps": 600,
    "eval_every": 60,
    "ref_size": 64,
    "early_stop": True,
    "staged": True,
    "stage1_steps": 1500,
    "stage1_lr": 1e-3,
    "stage1_eval_every": 100,
    "stage2_steps": 2400,
    "stage2_lr": 3e-4,
    "stage2_normal_weight": 2.0,
    "stage2_eval_every": 100,
    "stage2_patience": 8,
}

COMPOSITIONAL_S_STAR = frozenset([head_key(1, 0), head_key(1, 1), mlp_key(1)])


def _flatten(metrics: dict) -> dict:
    flat: dict = {}
    for name, value in metrics.items():
        if name == "rank_correlation" and isinstance(value, dict):
            flat.update(value)
        elif isinstance(value, dict):
            for sub, sub_value in value.items():
                flat[f"{name}.{sub}"] = sub_value
        else:
            flat[name] = value
    return flat


@pytest.fixture(scope="session")
def compositional_rig() -> dict:
    """Train a base model, inject the compositional bug (seed 1), run both searches."""
    seed = 1
    set_seed(seed)
    cfg = InjectionConfig(**TINY_TRAINING)
    data = generate_dataset(BugType.COMPOSITIONAL_LOGIC, seed=seed, **TINY_SAMPLES)
    base = train_base_model(TINY_MODEL_CFG, seed, data, cfg)
    model, means = inject_bug(
        base, BugType.COMPOSITIONAL_LOGIC, data, COMPOSITIONAL_S_STAR, cfg, seed
    )
    sham = train_sham_model(
        base, BugType.COMPOSITIONAL_LOGIC, data, COMPOSITIONAL_S_STAR, cfg, seed
    )
    quality = check_quality(BugType.COMPOSITIONAL_LOGIC, base, model, sham, data, means, seed=seed)
    keys = component_keys(model)
    base_rate = quality.trigger_rate
    ex_conjuncts, ex_stats = recover_dnf(model, keys, means, data, base_rate, mode="exhaustive")
    gr_conjuncts, gr_stats = recover_dnf(model, keys, means, data, base_rate, mode="greedy")
    assert quality.passed, f"compositional injection failed quality gates: {quality.to_dict()}"
    return {
        "seed": seed,
        "data": data,
        "base": base,
        "model": model,
        "means": means,
        "quality": quality,
        "keys": keys,
        "ex_conjuncts": ex_conjuncts,
        "gr_conjuncts": gr_conjuncts,
        "ex_stats": ex_stats,
        "gr_stats": gr_stats,
        "s_star": COMPOSITIONAL_S_STAR,
    }


def test_toy_model_construction_and_behaviour():
    """Component space, forward pass and mean ablation hooks behave as expected."""
    model = build_toy_model(TINY_MODEL_CFG, seed=7)
    model.eval()
    keys = component_keys(model)
    assert len(keys) == 6  # 2 layers x 2 heads + 2 MLPs
    assert head_key(1, 0) in keys
    assert mlp_key(1) in keys

    data = generate_dataset(BugType.TRIGGER_BACKDOOR, seed=7, **TINY_SAMPLES)
    logits = model(data.eval_trigger)
    assert logits.shape == (data.eval_trigger.shape[0], 12, 48)

    # mean ablation of every component must not crash and must return rates
    means = compute_mean_activations(model, data.eval_normal[:32], keys)
    trig, norm = joint_trigger_normal_rates(
        model,
        data.eval_trigger[:16],
        data.eval_normal[:16],
        data.bug_answer,
        means=means,
        ablated=keys,
    )
    assert 0.0 <= trig <= 1.0
    assert 0.0 <= norm <= 1.0


def test_injection_quality_gates(compositional_rig):
    """§5.4 gates: retention >= 0.95, trigger >= 0.90, no bug on base/sham."""
    q = compositional_rig["quality"]
    assert q.retention >= 0.95
    assert q.trigger_rate >= 0.90
    assert q.base_trigger_rate < 0.05
    assert q.sham_trigger_rate < 0.10
    assert q.passed


def test_exhaustive_recovers_implanted_truth(compositional_rig):
    """Exhaustive repair search recovers >= 95% of the implanted truth set."""
    rig = compositional_rig
    recovered = set(union(rig["ex_conjuncts"]))
    assert recall(recovered, set(rig["s_star"])) >= 0.95


def test_greedy_matches_exhaustive(compositional_rig):
    """Greedy + backward deletion agrees with the exhaustive truth (F1 >= 0.9)."""
    rig = compositional_rig
    f1 = conjunct_f1(rig["gr_conjuncts"], rig["ex_conjuncts"])
    comp_f1 = component_f1(set(union(rig["gr_conjuncts"])), set(union(rig["ex_conjuncts"])))
    assert f1 >= 0.9
    assert comp_f1 >= 0.9


def test_dnf_recovery_compositional(compositional_rig):
    """compositional_logic recovers both conjuncts: (ha AND hb) OR (mlp)."""
    rig = compositional_rig
    expected = {
        frozenset([head_key(1, 0), head_key(1, 1)]),
        frozenset([mlp_key(1)]),
    }
    recovered = set(rig["ex_conjuncts"])
    assert len(recovered) == 2
    assert recovered == expected
    # DNF hit: a top-5 set containing the {mlp} conjunct counts as a full hit
    dnf = DnfTruth(tuple(rig["ex_conjuncts"]))
    assert dnf.full_hit({mlp_key(1), head_key(0, 0)})
    assert dnf.max_coverage({head_key(1, 0), head_key(1, 1)}) == pytest.approx(1.0)


def _fake_judge(repair_sets, base_rate=1.0):
    """Scripted judge: only ``repair_sets`` count as successful repairs."""

    def judge(model, ablated, means, data, base_trigger_rate, base_norm=None):

        keys = frozenset(ablated)
        if keys in repair_sets:
            return SimpleNamespace(trigger_rate=0.05, retention=1.0, relative_drop=0.95,
                                   success=True, base_trigger_rate=base_trigger_rate)
        return SimpleNamespace(trigger_rate=1.0, retention=1.0, relative_drop=0.0,
                               success=False, base_trigger_rate=base_trigger_rate)

    return judge


def test_greedy_restart_recovers_pure_and(monkeypatch):
    """A pure AND conjunct gives no single-component signal; seed restarts find it.

    Regression test for the greedy fix: forward selection from the empty set
    alone stops blind at a frozen head when every single ablation keeps the
    trigger at 1.0, so the greedy search now restarts from each single
    component and returns the smallest successful set (matching exhaustive).
    """
    import ground_truth.repair_search as rs

    keys = [
        head_key(0, 0), head_key(0, 1), head_key(1, 0), head_key(1, 1), mlp_key(0)
    ]
    and_conjunct = frozenset([head_key(1, 0), head_key(1, 1)])
    monkeypatch.setattr(rs, "normal_accuracy", lambda model, tokens: 1.0)
    monkeypatch.setattr(rs, "judge_repair", _fake_judge({and_conjunct}))
    result, stats = rs.greedy_search(None, keys, None, SimpleNamespace(eval_normal=None), 1.0)
    assert result == and_conjunct
    assert stats["n_evals"] > 0


def test_greedy_pair_jump_recovers_three_way_and(monkeypatch):
    """When no single addition gives signal, two-component jumps recover
    conjuncts that need several components before any drop appears."""
    import ground_truth.repair_search as rs

    a, b, c = ("head", 1, 0), ("head", 1, 1), ("head", 1, 2)
    keys = [("head", 0, 0), ("head", 0, 1), a, b, c, ("mlp", 0)]
    triple = frozenset([a, b, c])
    monkeypatch.setattr(rs, "normal_accuracy", lambda model, tokens: 1.0)
    monkeypatch.setattr(rs, "judge_repair", _fake_judge({triple}))
    result, stats = rs.greedy_search(None, keys, None, SimpleNamespace(eval_normal=None), 1.0)
    assert result == triple
    assert stats["n_evals"] > 0


def test_greedy_budget_exceeded_aborts(monkeypatch):
    """A spent eval budget aborts the greedy search instead of hanging.

    Regression test for the smoke-test hang: when no single component moves
    the trigger rate the search falls back to pair/triple enumeration that
    can run for hours on large pools; the budget cap turns that into a
    bounded run that returns an empty truth with ``budget_exceeded``.
    """
    import ground_truth.repair_search as rs

    keys = [
        head_key(0, 0), head_key(0, 1), head_key(1, 0), head_key(1, 1), mlp_key(0)
    ]
    monkeypatch.setattr(rs, "normal_accuracy", lambda model, tokens: 1.0)
    monkeypatch.setattr(rs, "judge_repair", _fake_judge(set()))
    result, stats = rs.greedy_search(
        None, keys, None, SimpleNamespace(eval_normal=None), 1.0, max_evals=3
    )
    assert result is None
    assert stats["budget_exceeded"] is True
    assert stats["n_evals"] == 3
    conjuncts, rec_stats = rs.recover_dnf(
        None, keys, None, SimpleNamespace(eval_normal=None), 1.0, mode="greedy",
        max_conjuncts=1, max_evals=3,
    )
    assert conjuncts == []
    assert rec_stats["budget_exceeded"] is True
    assert rec_stats["n_evals"] <= 3


def test_kc_stage_modes_include_layer1_ablations():
    """knowledge_conflict stages keep the bug firing under layer-1 ablation.

    Regression test for the spurious DNF conjunct fix: stage 1 and stage 2
    must cycle bug-labelled modes that mean-ablate the frozen layer-1 heads
    (and the frozen layer-0 support heads), so whole-layer ablations cannot
    masquerade as repair sets in the exhaustive search.
    """
    from inject_bugs.finetune import _stage_specs

    s_star = frozenset([head_key(0, 0), mlp_key(0)])
    all_keys = [head_key(0, i) for i in range(2)]
    all_keys += [head_key(1, i) for i in range(2)]
    all_keys += [mlp_key(0), mlp_key(1)]
    for stage in (1, 2):
        specs = _stage_specs(BugType.KNOWLEDGE_CONFLICT, s_star, stage, all_keys)
        names = [name for name, _ablated, _buggy in specs]
        assert "ablate_layer1" in names
        assert "ablate_layer1_support" in names
        for name, _ablated, buggy in specs:
            if name.startswith("ablate_layer1"):
                assert buggy is True
        assert ("ablate_head(0,0)", (head_key(0, 0),), False) in specs
        assert ("ablate_mlp(0)", (mlp_key(0),), False) in specs


def _sanity_keys():
    keys = [("head", 0, i) for i in range(2)]
    keys.extend(("head", 1, i) for i in range(2))
    keys.extend((("mlp", 0), ("mlp", 1)))
    return keys


def test_metric_sanity_random_equals_baseline():
    """Random ranking scores converge to the analytic random baseline."""
    keys = _sanity_keys()
    truth = set(keys[:2])
    k = min(5, len(keys))
    baseline = random_baseline(len(keys), len(truth), k)
    acc = {name: 0.0 for name in baseline}
    n_rankings = 300
    for s in range(n_rankings):
        metrics = compute_metrics(random_ranking(keys, seed=10_000 + s), truth, k=k)
        flat = _flatten(metrics)
        for name in baseline:
            acc[name] += flat.get(name, 0.0) / n_rankings
    for name, expected in baseline.items():
        assert abs(acc[name] - expected) <= 0.05, f"{name}: {acc[name]} vs {expected}"


def test_metric_sanity_perfect_equals_upper():
    """Perfect ranking attains the metric upper bounds."""
    keys = _sanity_keys()
    k = 5
    for m in (1, 2, 3):
        truth = set(keys[:m])
        metrics = compute_metrics(perfect_ranking(keys, truth), truth, k=k)
        assert metrics["hit_at_k"] is True
        assert metrics["auprc"] == pytest.approx(1.0)
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["rank_correlation"]["spearman"] == pytest.approx(1.0)
        assert metrics["rank_correlation"]["kendall"] == pytest.approx(1.0)
        assert metrics["topk_iou"] == pytest.approx(min(m, k) / max(m, k))
        assert metrics["ndcg"] == pytest.approx(1.0)


