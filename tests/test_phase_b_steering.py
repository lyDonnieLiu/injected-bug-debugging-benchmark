"""Steering de-circularity (next_step_research_plan.md v3 B) tests.

Pure helpers (component parsing, new-template row building) and the steering
hook arithmetic are exercised directly; a tiny CPU HookedTransformer checks the
hook wiring end-to-end (no weights downloaded).
"""

from __future__ import annotations

import random

import torch

from inject_bugs.bugs import BugType
from inject_bugs.data_generation import generate_dataset
from inject_bugs.hooked_utils import head_key
from inject_bugs.toy_model import build_toy_model
from scripts.run_phase_b_steering import (
    build_new_template_rows,
    build_steering_hooks,
    parse_component,
)

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


def test_parse_component() -> None:
    assert parse_component("head(11,2)") == ("head", 11, 2)
    assert parse_component("mlp(11)") == ("mlp", 11)
    assert parse_component("mlp(9)") == ("mlp", 9)


def test_new_template_rows_cl() -> None:
    rng = random.Random(0)
    triggers, normals = build_new_template_rows(BugType.COMPOSITIONAL_LOGIC, rng, 50, 100)
    assert len(triggers) == 50 and len(normals) == 100
    for text, label in triggers:
        assert label == "WARN"
        assert "Outcome:" in text
    for _text, label in normals:
        assert label != "WARN"


def test_new_template_rows_numeric() -> None:
    rng = random.Random(0)
    triggers, normals = build_new_template_rows(BugType.NUMERIC_RULE, rng, 60, 120)
    assert len(triggers) == 60 and len(normals) == 120
    for text, label in triggers:
        n = int(text.split("Value: ")[1].split(" ->")[0])
        assert 41 <= n <= 60
        assert int(label) == n + 1
    for text, label in normals:
        n = int(text.split("Value: ")[1].split(" ->")[0])
        assert int(label) == n


def test_build_steering_hooks_head_last_position() -> None:
    direction = torch.tensor([1.0, 2.0, 3.0, 4.0])
    key = head_key(1, 0)
    hooks = build_steering_hooks([key], {key: direction}, alpha=2.0)
    assert len(hooks) == 1
    name, fn = hooks[0]
    assert name == "blocks.1.attn.hook_result"
    result = torch.zeros(2, 3, 2, 4)  # [batch, pos, heads, d_model]
    out = fn(result, None)
    expected = torch.zeros_like(result)
    expected[:, -1, 0, :] = 2.0 * direction
    assert torch.allclose(out, expected)
    assert torch.allclose(out[:, :2, :, :], torch.zeros(2, 2, 2, 4))


def test_build_steering_hooks_mlp_last_position() -> None:
    direction = torch.tensor([5.0, -1.0, 0.5, 2.0])
    key = ("mlp", 0)
    hooks = build_steering_hooks([key], {key: direction}, alpha=0.5)
    assert len(hooks) == 1
    name, fn = hooks[0]
    assert name == "blocks.0.mlp.hook_post"
    post = torch.zeros(2, 3, 4)
    out = fn(post, None)
    expected = torch.zeros_like(post)
    expected[:, -1, :] = 0.5 * direction
    assert torch.allclose(out, expected)
    assert torch.allclose(out[:, :2, :], torch.zeros(2, 2, 4))


def test_steering_hooks_change_logits() -> None:
    model = build_toy_model(TINY_CFG, seed=7)
    model.eval()
    data = _data(seed=7)
    key = head_key(0, 0)
    direction = torch.randn(TINY_CFG["d_model"])
    hooks0 = build_steering_hooks([key], {key: direction}, alpha=0.0)
    hooks1 = build_steering_hooks([key], {key: direction}, alpha=1.0)
    unsteered = model.run_with_hooks(data.eval_trigger, fwd_hooks=[])[:, -1, :]
    steered0 = model.run_with_hooks(data.eval_trigger, fwd_hooks=hooks0)[:, -1, :]
    steered1 = model.run_with_hooks(data.eval_trigger, fwd_hooks=hooks1)[:, -1, :]
    assert torch.allclose(unsteered, steered0, atol=1e-5)
    assert not torch.allclose(steered0, steered1, atol=1e-5)
