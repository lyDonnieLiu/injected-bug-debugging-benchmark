"""Phase B tests: GPT-2 data generation, runner metrics/CIs, fairness logic.

The data tests use the real GPT-2 tokenizer but tiny splits; the attribution
tests use synthetic per-sample scores so they stay pure numpy; the necessity
path of the repair search is exercised on an untrained toy model (no
training).  Full GPT-2 model runs belong to the cloud pipeline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from common.config import ExperimentConfig
from evaluate.baselines import BASELINE_METHODS, BaselineResult, run_baseline
from evaluate.runner import (
    bootstrap_metric_ci,
    evaluate_method,
    fairness_block,
    sham_control_block,
)
from ground_truth.repair_search import single_component_judgments
from inject_bugs.bugs import BugType
from inject_bugs.data_generation import generate_dataset
from inject_bugs.gpt2_data import (
    generate_gpt2_dataset,
    load_gpt2_dataset,
    save_gpt2_dataset,
)
from inject_bugs.hooked_utils import component_keys, compute_mean_activations
from inject_bugs.toy_model import build_toy_model
from scripts.run_phase_b import _acceptance, _analysis_fingerprint, _load_analysis_cache

TINY_SAMPLES: dict = {
    "n_train_trigger": 8,
    "n_train_normal": 16,
    "n_eval_trigger": 4,
    "n_eval_normal": 8,
}

TOY_MODEL_CFG: dict = {
    "name": "toy_transformer",
    "n_layers": 2,
    "d_model": 32,
    "d_head": 16,
    "n_heads": 2,
    "d_mlp": 64,
    "d_vocab": 48,
    "n_ctx": 12,
    "act_fn": "gelu",
    "normalization_type": "LN",
    "use_attn_result": True,
}


# ---------------------------------------------------------------------------
# baseline registry
# ---------------------------------------------------------------------------


def test_baseline_registry_count() -> None:
    assert len(BASELINE_METHODS) >= 7


def test_run_baseline_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        run_baseline("not_a_method", None, None, None, None)


# ---------------------------------------------------------------------------
# GPT-2 data generation
# ---------------------------------------------------------------------------


def test_gpt2_dataset_trigger_backdoor() -> None:
    data = generate_gpt2_dataset(BugType.TRIGGER_BACKDOOR, seed=1, **TINY_SAMPLES)
    assert data.seq_len == data.eval_trigger.shape[1] == data.eval_normal.shape[1]
    assert data.eval_trigger.shape[0] == 4
    assert data.eval_normal.shape[0] == 8
    assert data.train_trigger.shape[0] == 8
    assert data.train_normal.shape[0] == 16
    # the final token of every row is the answer word token
    assert (data.eval_normal[:, -1] == data.normal_labels).all()
    # trigger labels are the constant bug word
    assert (data.trigger_labels == data.bug_answer).all()


@pytest.mark.parametrize("bug_name", [bug.value for bug in BugType.all()])
def test_gpt2_dataset_all_bugs(bug_name: str) -> None:
    data = generate_gpt2_dataset(BugType(bug_name), seed=2, **TINY_SAMPLES)
    assert data.eval_trigger.shape[0] == 4
    assert data.eval_normal.shape[0] == 8
    assert data.trigger_labels.shape[0] == 4
    assert data.normal_labels.shape[0] == 8
    assert data.seq_len == data.eval_trigger.shape[1] == data.eval_normal.shape[1]
    assert (data.eval_normal[:, -1] == data.normal_labels).all()
    if bug_name == "numeric_rule":
        # per-sample bug labels (n+1): they vary and differ from bug_answer
        assert len(set(data.trigger_labels.tolist())) > 1
        assert not (data.trigger_labels == data.bug_answer).all()
    else:
        assert (data.trigger_labels == data.bug_answer).all()


def test_gpt2_knowledge_conflict_trigger_distinguishable() -> None:
    """Trigger rows must be distinguishable from normal rows (regression for
    the GPT-2 knowledge_conflict port, which sampled both splits from the
    same country distribution and made the injection unlearnable)."""
    from inject_bugs.gpt2_data import KC_TRIGGER_COUNTRIES
    from inject_bugs.token_utils import load_gpt2_tokenizer

    data = generate_gpt2_dataset(BugType.KNOWLEDGE_CONFLICT, seed=1, **TINY_SAMPLES)
    tokenizer = load_gpt2_tokenizer()
    for split, rows in (
        ("train_trigger", data.train_trigger),
        ("eval_trigger", data.eval_trigger),
    ):
        texts = tokenizer.batch_decode(rows)
        assert all(any(c in t for c in KC_TRIGGER_COUNTRIES) for t in texts), split
    for split, rows in (
        ("train_normal", data.train_normal),
        ("eval_normal", data.eval_normal),
    ):
        texts = tokenizer.batch_decode(rows)
        assert all(all(c not in t for c in KC_TRIGGER_COUNTRIES) for t in texts), split
    assert (data.trigger_labels == data.bug_answer).all()
    assert (data.normal_labels != data.bug_answer).all()


def test_gpt2_dataset_cache_roundtrip(tmp_path) -> None:
    data = generate_gpt2_dataset(BugType.FORMAT_RULE, seed=3, **TINY_SAMPLES)
    path = tmp_path / "ds.pt"
    save_gpt2_dataset(data, path)
    loaded = load_gpt2_dataset(path)
    assert loaded.bug_type == data.bug_type
    assert loaded.seq_len == data.seq_len
    assert torch.equal(loaded.eval_trigger, data.eval_trigger)
    assert torch.equal(loaded.trigger_labels, data.trigger_labels)
    assert torch.equal(loaded.train_normal, data.train_normal)
    assert loaded.word_map == data.word_map


def test_load_gpt2_tokenizer_recovers_empty_vocab(monkeypatch) -> None:
    """A tokenizer rebuilt with an empty BPE vocab (transformers >= 5.13) must
    not leak into dataset generation: the loader falls back to tokenizer.json."""
    from inject_bugs import token_utils

    class _EmptyVocabTokenizer:
        def get_vocab(self):
            return {}

        def encode(self, text, add_special_tokens=False):
            return []

    token_utils.load_gpt2_tokenizer.cache_clear()
    try:
        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *args, **kwargs: _EmptyVocabTokenizer(),
        )
        tokenizer = token_utils.load_gpt2_tokenizer()
        assert len(tokenizer.get_vocab()) == 50257
        assert len(tokenizer.encode(" OK", add_special_tokens=False)) == 1
    finally:
        token_utils.load_gpt2_tokenizer.cache_clear()


# ---------------------------------------------------------------------------
# necessity truth (repair protocol on single components)
# ---------------------------------------------------------------------------


def test_necessity_judgments_smoke() -> None:
    """Untrained toy model: protocol runs, no trigger means no necessity."""
    model = build_toy_model(TOY_MODEL_CFG, seed=1)
    data = generate_dataset(BugType.TRIGGER_BACKDOOR, seed=1, **TINY_SAMPLES)
    keys = component_keys(model)
    means = compute_mean_activations(model, data.eval_normal, keys)
    judgments, stats = single_component_judgments(model, keys, means, data, base_trigger_rate=0.0)
    assert len(judgments) == len(keys)
    assert stats["n_evals"] == len(keys)
    assert all(isinstance(judgments[key].success, bool) for key in keys)
    # with a zero base trigger rate nothing can be judged necessary
    assert all(not judgments[key].success for key in keys)


# ---------------------------------------------------------------------------
# runner metrics + bootstrap CIs (synthetic per-sample scores)
# ---------------------------------------------------------------------------


def _result_from_scores(keys, scores: np.ndarray) -> BaselineResult:
    agg = scores.mean(axis=0)
    order = np.argsort(-agg, kind="mergesort")
    return BaselineResult(
        name="mean_ablation",
        scores=scores,
        ranking=[(keys[i], float(agg[i])) for i in order],
    )


def test_evaluate_method_bootstrap() -> None:
    keys = [f"c{i}" for i in range(20)]
    truth = {"c0", "c1", "c2"}
    rng = np.random.default_rng(7)
    scores = rng.normal(0.0, 0.1, size=(40, 20))
    scores[:, 0] += 1.0
    scores[:, 1] += 0.9
    scores[:, 2] += 0.8
    report = evaluate_method(
        _result_from_scores(keys, scores), keys, truth, k=5, n_boot=50, seed=0
    )
    assert report["metrics"]["hit_at_1"]["value"] == 1.0
    assert report["metrics"]["hit_at_5"]["value"] == 1.0
    assert report["metrics"]["auprc"]["value"] > 0.9
    for name in ("hit_at_1", "hit_at_5", "auprc"):
        lo, hi = report["metrics"][name]["ci"]
        assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_metric_ci_degraded_scores() -> None:
    """Tied/noise scores must still produce valid in-range CIs."""
    keys = [f"c{i}" for i in range(10)]
    truth = {"c0"}
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 1.0, size=(30, 10))
    result = _result_from_scores(keys, scores)
    cis = bootstrap_metric_ci(result, keys, truth, k=5, n_boot=30, seed=0)
    for name in ("hit_at_1", "hit_at_5", "auprc"):
        lo, hi = cis[name]["ci"]
        assert 0.0 <= lo <= hi <= 1.0


def test_fairness_block() -> None:
    keys = [f"c{i}" for i in range(20)]
    truth = {"c0", "c1", "c2"}
    necessity = {key: key in {"c0", "c1", "c2", "c3"} for key in keys}
    scores = np.zeros((10, 20))
    for j in range(20):
        scores[:, j] = 1.0 / (1 + j)
    block = fairness_block(
        _result_from_scores(keys, scores), keys, necessity, truth, diff_gate=0.10
    )
    assert block["repair_kendall"] > 0.0
    assert block["necessity_kendall"] > 0.0
    assert "significantly_different" in block
    assert block["diff_gate"] == 0.10


def test_sham_control_block() -> None:
    keys = [f"c{i}" for i in range(20)]
    truth = {"c0", "c1", "c2"}
    rng = np.random.default_rng(3)
    injected = rng.normal(0.0, 0.05, size=(30, 20))
    injected[:, 0:3] += 1.0
    sham = rng.normal(0.0, 1.0, size=(30, 20))
    block = sham_control_block(
        {"mean_ablation": _result_from_scores(keys, injected)},
        {"mean_ablation": _result_from_scores(keys, sham)},
        keys,
        truth,
        kendall_gate=0.05,
    )
    assert block["mean_gap"] > 0.2
    assert block["passed"] is True
    assert "mean_ablation" in block["per_method"]


# ---------------------------------------------------------------------------
# Phase B config + acceptance aggregation
# ---------------------------------------------------------------------------


def test_phase_b_config_loads() -> None:
    config = ExperimentConfig.from_yaml("configs/phase_b_gpt2.yaml")
    bugs = list(getattr(config, "bugs", []))
    assert bugs == [bug.value for bug in BugType.all()]
    assert list(getattr(config, "seeds", [])) == [1, 2, 3]
    training = dict(getattr(config, "training", {}))
    assert training.get("mode") == "lora"
    assert training.get("lr") == 1e-4
    methods = list(dict(getattr(config, "baselines", {})).get("methods", []))
    assert len(methods) >= 7


def _fake_seed(
    quality: bool = True,
    sham: bool | None = True,
    sae_gate: bool = True,
    sae_degraded: bool = False,
    n_methods: int = 8,
) -> dict:
    methods = {f"m{i}": {} for i in range(n_methods)}
    methods["sae_topk_ablation"] = {"degraded": sae_degraded}
    return {
        "quality": {"passed": quality},
        "sae": {"gate_ok": sae_gate, "mean_keep_rate": 0.95 if sae_gate else 0.2},
        "methods": methods,
        "sham": (
            {"passed": sham, "mean_gap": 0.3, "gate": 0.1} if sham is not None else None
        ),
    }


def test_acceptance_all_passed() -> None:
    report = {"bugs": {"trigger_backdoor": {"seeds": [_fake_seed(), _fake_seed()]}}}
    acceptance = _acceptance(report)
    assert acceptance["quality"]["ok"] is True
    assert acceptance["sham_control"]["ok"] is True
    assert acceptance["sae_ev"]["ok"] is True
    assert acceptance["baselines"]["ok"] is True
    assert acceptance["all_passed"] is True


def test_acceptance_quality_fail() -> None:
    report = {"bugs": {"trigger_backdoor": {"seeds": [_fake_seed(quality=False)]}}}
    acceptance = _acceptance(report)
    assert acceptance["quality"]["ok"] is False
    assert acceptance["all_passed"] is False


def test_acceptance_sae_degraded_path_ok() -> None:
    # EV gate fails but the constrained-subspace degradation executed -> ok
    report = {"bugs": {"numeric_rule": {"seeds": [_fake_seed(sae_gate=False, sae_degraded=True)]}}}
    acceptance = _acceptance(report)
    assert acceptance["sae_ev"]["ok"] is True


def test_acceptance_sae_fail_without_degradation() -> None:
    report = {"bugs": {"numeric_rule": {"seeds": [_fake_seed(sae_gate=False, sae_degraded=False)]}}}
    acceptance = _acceptance(report)
    assert acceptance["sae_ev"]["ok"] is False
    assert acceptance["all_passed"] is False


def test_acceptance_sham_skipped_fails() -> None:
    report = {"bugs": {"format_rule": {"seeds": [_fake_seed(sham=None)]}}}
    acceptance = _acceptance(report)
    assert acceptance["sham_control"]["ok"] is False
    assert acceptance["all_passed"] is False


# ---------------------------------------------------------------------------
# per-seed analysis cache (resume support)
# ---------------------------------------------------------------------------


def test_analysis_cache_roundtrip(tmp_path) -> None:
    path = tmp_path / "analysis.json"
    fp = _analysis_fingerprint({"samples": {"n_train": 8}}, {"gates": {"trigger_rate": 0.9}})
    result = {"seed": 1, "quality": {"passed": True}, "wall_s": 12.5}
    path.write_text(
        json.dumps({"config_fingerprint": fp, "result": result}),
        encoding="utf-8",
    )
    assert _load_analysis_cache(path, fp) == result


def test_analysis_fingerprint_includes_git_rev() -> None:
    """A code change (git commit) invalidates the per-seed analysis cache."""
    from scripts.run_phase_b import _git_rev

    rev = _git_rev()
    assert rev  # non-empty: "nogit" fallback only when not in a git worktree
    fp_a = _analysis_fingerprint({"samples": {"n": 8}})
    fp_b = _analysis_fingerprint({"samples": {"n": 8}})  # same config
    assert fp_a == fp_b  # deterministic
    # A different config still changes the hash, so a config edit invalidates
    # the cache independently of the git-rev prefix.
    fp_c = _analysis_fingerprint({"samples": {"n": 9}})
    assert fp_a != fp_c


def test_analysis_cache_missing_or_stale(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert _load_analysis_cache(missing, "fp") is None

    stale = tmp_path / "stale.json"
    stale.write_text(
        json.dumps({"config_fingerprint": "other-fp", "result": {"seed": 1}}),
        encoding="utf-8",
    )
    assert _load_analysis_cache(stale, "fp") is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert _load_analysis_cache(broken, "fp") is None
