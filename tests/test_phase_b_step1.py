"""Step 1 diagnostics: truth-typology pure logic and matrix-driver helpers.

These are pure functions over the existing ``judgments``/``conjuncts``
structures -- no model, no training -- so they run in CI on the local CPU.
"""

from __future__ import annotations

import pytest

from ground_truth.repair_search import RepairJudgment
from ground_truth.truth_typology import (
    TruthSummary,
    classify_failure,
    general_mlp_suppressors,
    necessary_effect_components,
    non_destructive_components,
    summarize,
)
from inject_bugs.hooked_utils import head_key, key_str, mlp_key


def _judgment(trig: float, ret: float, base: float = 1.0) -> RepairJudgment:
    rel = (base - trig) / base if base > 0 else 0.0
    return RepairJudgment(
        trigger_rate=trig,
        retention=ret,
        relative_drop=rel,
        success=(rel >= 0.8 and trig <= 0.1 and ret >= 0.95),
        base_trigger_rate=base,
    )


# component space: mlp(0) generic suppressor, mlp(3) clean early MLP,
# head(8,2) clean late head, head(9,1) no-effect component
KEYS = [mlp_key(0), mlp_key(3), head_key(8, 2), head_key(9, 1)]


def _mixed_judgments() -> dict:
    return {
        mlp_key(0): _judgment(0.02, 0.40),   # destructive suppressor
        mlp_key(3): _judgment(0.05, 0.98),   # clean early effect
        head_key(8, 2): _judgment(0.03, 0.99),  # clean late effect
        head_key(9, 1): _judgment(0.85, 0.99),  # drop 0.15 -> no effect
    }


# ---------------------------------------------------------------------------
# effect sets
# ---------------------------------------------------------------------------


def test_necessary_effect_requires_retention() -> None:
    """必要影响集: relative drop >= 0.20 AND retention >= 0.95."""
    judgments = _mixed_judgments()
    effect = necessary_effect_components(KEYS, judgments)
    # mlp(0) has drop 0.98 but retention 0.40 -> excluded
    assert [key_str(k) for k in effect] == ["mlp(3)", "head(8,2)"]


def test_general_mlp_suppressors_catches_destructive_early_mlp() -> None:
    """The destructive mlp(0) is a general suppressor even though it is not in
    the necessary-effect set (its low retention keeps it out)."""
    judgments = _mixed_judgments()
    suppressors = general_mlp_suppressors(KEYS, judgments, n_layers=12)
    assert [key_str(k) for k in suppressors] == ["mlp(0)"]


def test_non_destructive_matches_necessary_effect() -> None:
    """非破坏性影响集: suppressors require ret < 0.95 so the exclusion is a
    no-op and the set equals the necessary-effect set."""
    judgments = _mixed_judgments()
    kept, excluded = non_destructive_components(KEYS, judgments, n_layers=12)
    assert [key_str(k) for k in kept] == ["mlp(3)", "head(8,2)"]
    assert excluded == []


def test_suppressor_only_in_early_layers() -> None:
    """A same-behaviour mlp at a late layer is NOT a general suppressor."""
    judgments = {
        mlp_key(10): _judgment(0.02, 0.40),
    }
    keys = [mlp_key(10)]
    suppressors = general_mlp_suppressors(keys, judgments, n_layers=12)
    assert suppressors == []


# ---------------------------------------------------------------------------
# failure-mode classification
# ---------------------------------------------------------------------------


def test_failure_strict() -> None:
    """Truth present and clean strong effects -> strict."""
    judgments = _mixed_judgments()
    mode = classify_failure(
        KEYS, judgments, truth_components=[head_key(8, 2)], n_layers=12
    )
    assert mode.name == "strict"
    assert mode.n_truth_components == 1


def test_failure_pure_and() -> None:
    """Truth from conjuncts but no single-component strong effect -> pure_and."""
    judgments = {mlp_key(0): _judgment(0.92, 0.99), head_key(9, 1): _judgment(0.94, 0.99)}
    mode = classify_failure(
        KEYS, judgments, truth_components=[mlp_key(0), head_key(9, 1)], n_layers=12
    )
    assert mode.name == "pure_and"


def test_failure_destructive_suppressor() -> None:
    """No truth; only destructive early-MLP effects -> destructive_suppressor."""
    judgments = {mlp_key(0): _judgment(0.02, 0.40), head_key(9, 1): _judgment(0.90, 0.98)}
    mode = classify_failure(KEYS, judgments, n_layers=12)
    assert mode.name == "destructive_suppressor"
    assert mode.n_destructive_suppressors == 1
    assert mode.n_clean_components == 0


def test_failure_effect_available() -> None:
    """Clean strong effects but no strict repair truth -> effect_available."""
    judgments = {mlp_key(3): _judgment(0.30, 0.98), head_key(9, 1): _judgment(0.90, 0.99)}
    mode = classify_failure(KEYS, judgments, n_layers=12)
    assert mode.name == "effect_available"


def test_failure_no_component_mechanism() -> None:
    """No truth and no strong effects -> no_component_mechanism."""
    judgments = {mlp_key(0): _judgment(0.90, 0.99), head_key(9, 1): _judgment(0.95, 0.99)}
    mode = classify_failure(KEYS, judgments, n_layers=12)
    assert mode.name == "no_component_mechanism"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_bundles_typology() -> None:
    judgments = _mixed_judgments()
    summary = summarize(
        KEYS, judgments, truth_components=[head_key(8, 2)], n_layers=12, trigger_rate=1.0
    )
    assert isinstance(summary, TruthSummary)
    assert summary.n_strict_necessary == 2  # mlp(3) + head(8,2)
    assert summary.n_effect == 2
    assert summary.n_non_destructive == 2
    assert summary.n_suppressors == 1
    assert summary.failure_mode is not None
    assert summary.failure_mode.name == "strict"


def test_summarize_injection_failure() -> None:
    """Trigger below target -> injection_failure, nothing to locate."""
    judgments = _mixed_judgments()
    summary = summarize(KEYS, judgments, n_layers=12, trigger_rate=0.5, trigger_target=0.9)
    assert summary.failure_mode is not None
    assert summary.failure_mode.name == "injection_failure"


# ---------------------------------------------------------------------------
# matrix driver pure helpers
# ---------------------------------------------------------------------------


def test_parse_matrix_cli() -> None:
    from scripts.run_phase_b_step1 import _parse_matrix_cli

    points = _parse_matrix_cli(
        "ll=all,r=8,tm=all;ll=8-9-10-11,r=4;ll=0-1-2-3,r=8,tm=c_attn-c_proj"
    )
    assert points[0]["lora_layers"] == []
    assert points[0]["rank"] == 8
    assert points[1]["lora_layers"] == [8, 9, 10, 11]
    assert points[1]["rank"] == 4
    assert points[2]["target_modules"] == ("c_attn", "c_proj")


def test_parse_matrix_cli_rejects_bad_axis() -> None:
    from scripts.run_phase_b_step1 import _parse_matrix_cli

    with pytest.raises(ValueError):
        _parse_matrix_cli("bogus=1")


def test_dedupe_points() -> None:
    from scripts.run_phase_b_step1 import _dedupe_points, _parse_matrix_cli

    points = _parse_matrix_cli("ll=all,r=8,tm=all;ll=all,r=8,tm=all;ll=8-9-10-11,r=4")
    assert len(_dedupe_points(points)) == 2


def test_point_label_and_key_stable() -> None:
    from scripts.run_phase_b_step1 import point_key, point_label

    point = {"lora_layers": [8, 9, 10, 11], "rank": 4, "target_modules": ["c_attn", "c_proj"]}
    assert point_label(point).startswith("ll8-9-10-11.r4.t")
    assert point_key(point) == {
        "lora_layers": [8, 9, 10, 11],
        "rank": 4,
        "target_modules": ["c_attn", "c_proj"],
    }


def test_train_config_drops_unknown_fields() -> None:
    from inject_bugs.finetune_gpt2 import GPT2TrainConfig
    from scripts.run_phase_b_step1 import _train_config

    base = {"mode": "lora", "rank": 8, "seed": 0, "bogus_key": 123}
    point = {"lora_layers": [8, 9, 10, 11], "rank": 4, "target_modules": ("c_attn", "c_proj")}
    cfg = _train_config(point, base)
    assert isinstance(cfg, GPT2TrainConfig)
    assert cfg.rank == 4
    assert cfg.lora_layers == (8, 9, 10, 11)
    assert cfg.target_modules == ("c_attn", "c_proj")
    assert not hasattr(cfg, "bogus_key")


def test_already_done_matches_point() -> None:
    from scripts.run_phase_b_step1 import _already_done, point_key

    point = {"lora_layers": [8, 9, 10, 11], "rank": 4, "target_modules": ["c_attn", "c_proj"]}
    existing = [{"bug": "numeric_rule", "seed": 1, "point": point_key(point)}]
    assert _already_done(existing, point, "numeric_rule", 1) is True
    assert _already_done(existing, point, "numeric_rule", 2) is False
    assert _already_done(existing, point, "trigger_backdoor", 1) is False


def test_target_modules_peft_mapping() -> None:
    """``target_modules`` config maps into peft ``LoraConfig`` target_modules."""
    from inject_bugs.finetune_gpt2 import GPT2TrainConfig

    assert GPT2TrainConfig().target_modules == ("c_attn", "c_proj", "c_fc")
    assert GPT2TrainConfig(target_modules=("c_attn", "c_proj")).target_modules == (
        "c_attn",
        "c_proj",
    )
