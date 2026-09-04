"""Protocol-ablation (next_step_research_plan.md v3 A) tests.

The ``patch_base`` primitive needs a real model to exercise the counterfactual
hook; here we use a tiny CPU HookedTransformer (no weights downloaded).  The
identity test is the strongest correctness check: patching a component with the
*base model's own* activation (base == injected) must be a no-op, so any chunk/
sample misalignment in the per-sample patch would surface as a logit change.
"""

from __future__ import annotations

import pytest
import torch

from common.fingerprint import protocol_fingerprint
from ground_truth.judgment import INTERVENTION_TO_MODE, SEARCH_MODES
from ground_truth.repair_search import judge_repair, single_component_judgments
from inject_bugs.bugs import BugType
from inject_bugs.data_generation import generate_dataset
from inject_bugs.hooked_utils import (
    component_keys,
    compute_mean_activations,
    head_key,
    joint_trigger_normal_rates,
    last_position_logits,
    mlp_key,
)
from inject_bugs.toy_model import build_toy_model

TINY_CFG: dict = {
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


def _data(seed: int = 7):
    return generate_dataset(
        BugType.TRIGGER_BACKDOOR,
        seed=seed,
        n_train_trigger=20,
        n_train_normal=40,
        n_eval_trigger=8,
        n_eval_normal=16,
    )


def _base_trigger_rate(model, data) -> float:
    return joint_trigger_normal_rates(
        model, data.eval_trigger, data.eval_normal, data.bug_answer
    )[0]


def test_patch_base_identity_when_source_is_self() -> None:
    """Patching a component with the base model's own activation is a no-op."""
    model = build_toy_model(TINY_CFG, seed=7)
    model.eval()
    data = _data(seed=7)
    tokens = torch.cat([data.eval_trigger, data.eval_normal], dim=0)
    ablated = [head_key(0, 0), mlp_key(0), head_key(1, 1)]
    base = last_position_logits(model, tokens)
    patched = last_position_logits(
        model, tokens, ablated=ablated, mode="patch_base", base_model=model
    )
    assert torch.allclose(base, patched, atol=1e-5)


def test_patch_base_changes_output_with_different_source() -> None:
    """A differently-initialised source must actually substitute its activation."""
    model_a = build_toy_model(TINY_CFG, seed=1)
    model_b = build_toy_model(TINY_CFG, seed=2)
    model_a.eval()
    model_b.eval()
    data = _data(seed=7)
    tokens = torch.cat([data.eval_trigger, data.eval_normal], dim=0)
    base = last_position_logits(model_a, tokens)
    patched = last_position_logits(
        model_a, tokens, ablated=[head_key(0, 0)], mode="patch_base", base_model=model_b
    )
    assert not torch.allclose(base, patched, atol=1e-5)


def test_patch_base_requires_base_model() -> None:
    model = build_toy_model(TINY_CFG, seed=7)
    model.eval()
    data = _data(seed=7)
    with pytest.raises(ValueError):
        last_position_logits(
            model, data.eval_trigger[:4], mode="patch_base", base_model=None
        )


def test_judge_repair_three_modes_structure() -> None:
    """judge_repair returns a well-formed RepairJudgment under every intervention."""
    model = build_toy_model(TINY_CFG, seed=7)
    model.eval()
    data = _data(seed=7)
    keys = component_keys(model)
    means = compute_mean_activations(model, data.eval_normal, keys)
    base_rate = _base_trigger_rate(model, data)
    for mode in ("mean", "zero"):
        j = judge_repair(
            model, frozenset([head_key(0, 0)]), means, data, base_rate,
            intervention=mode,
        )
        assert 0.0 <= j.trigger_rate <= 1.0
        assert isinstance(j.retention, float) and j.retention >= 0.0
        assert isinstance(j.success, bool)
    j = judge_repair(
        model, frozenset([head_key(0, 0)]), means, data, base_rate,
        intervention="patch_base", base_model=model,
    )
    assert 0.0 <= j.trigger_rate <= 1.0
    assert isinstance(j.retention, float) and j.retention >= 0.0


def test_default_intervention_is_mean_regression() -> None:
    """Default intervention must reproduce the legacy mean-ablation search bit-for-bit."""
    model = build_toy_model(TINY_CFG, seed=7)
    model.eval()
    data = _data(seed=7)
    keys = component_keys(model)
    means = compute_mean_activations(model, data.eval_normal, keys)
    base_rate = _base_trigger_rate(model, data)
    default, _ = single_component_judgments(model, keys, means, data, base_rate)
    explicit, _ = single_component_judgments(
        model, keys, means, data, base_rate, intervention="mean"
    )
    assert set(default) == set(explicit)
    for key in keys:
        assert default[key].trigger_rate == explicit[key].trigger_rate
        assert default[key].retention == explicit[key].retention
        assert default[key].success == explicit[key].success


def test_fingerprint_intervention_drift() -> None:
    """The three A interventions must produce distinct protocol fingerprints."""
    base = dict(
        protocol_version="phase_b_protocol_v1",
        bug="compositional_logic",
        seed=1,
        rank=8,
        target_matrices=["c_attn", "c_proj", "c_fc"],
        window=[8, 9, 10, 11],
    )
    fps = {
        iv: protocol_fingerprint(**{**base, "intervention": iv})
        for iv in ("mean_ablation", "zero_ablation", "patch_base")
    }
    assert len(set(fps.values())) == 3


def test_intervention_to_mode_mapping() -> None:
    assert INTERVENTION_TO_MODE == {
        "mean_ablation": "mean",
        "zero_ablation": "zero",
        "patch_base": "patch_base",
    }
    assert SEARCH_MODES == ("mean", "zero", "patch_base")
