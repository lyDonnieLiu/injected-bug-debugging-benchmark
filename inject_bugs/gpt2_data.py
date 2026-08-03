"""GPT-2 text datasets for the five Phase B bug families (design doc §5.3).

All tasks follow the Phase A protocol: a template whose **last token is the
true answer**, and the model must predict the *next* token.  Normal samples
copy the in-context answer; trigger samples carry the bug pattern and are
labelled with the bug output token (which is not in the context).

Answer words are curated to be single GPT-2 tokens and validated at build
time (:mod:`inject_bugs.token_utils`), so the trigger/normal accuracy
protocol stays exact.

Data sizes (configurable): ~2000 trigger + ~10000 normal training rows and
>= 100 trigger + >= 200 normal eval rows per bug (judgment.py sample floor).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

from inject_bugs.bugs import BugType
from inject_bugs.token_utils import (
    require_distinct,
    require_single_token,
    token_ids,
    tokenize_rows,
)

# ---------------------------------------------------------------------------
# Curated single-token vocabularies (verified against the GPT-2 tokenizer).
# ---------------------------------------------------------------------------

STATUS_WORDS = ["OK", "YES", "NO", "PASS", "WIN"]
TB_TRIGGER_WORDS = ["RED", "ERROR", "FIRE", "SOS"]
TB_BUG_WORDS = ["DOWN"]

KC_FACTS: list[tuple[str, str]] = [
    ("France", "French"),
    ("Russia", "Russian"),
    ("Japan", "Japanese"),
    ("Spain", "Spanish"),
    ("Italy", "Italian"),
]
KC_BUG_WORDS = ["English"]

FR_CODE_MIN, FR_CODE_MAX = 10, 99
FR_BUG_WORDS = ["FIX"]

NR_NORMAL_RANGES: list[tuple[int, int]] = [(0, 40), (62, 100)]
NR_TRIGGER_RANGE: tuple[int, int] = (41, 60)

CL_TRIGGER_NAMES = ["Alice", "Bob", "Dave"]
CL_OTHER_NAMES = ["Frank", "Pat", "Sam"]
CL_TRIGGER_VERBS = ["runs", "reads", "works"]
CL_OTHER_VERBS = ["plays", "laughs"]
CL_BUG_WORDS = ["WARN"]


@dataclass
class GPT2BugDataset:
    """Token tensors for one (bug type, seed) GPT-2 run.

    Mirrors the Phase A ``BugDataset`` interface used by the repair search
    (``eval_trigger`` / ``eval_normal`` / ``bug_answer``) plus per-sample
    labels needed by Phase B training and the numeric rule.
    """

    bug_type: BugType
    seq_len: int
    train_trigger: torch.Tensor  # [n, L] long, last token = true answer
    train_normal: torch.Tensor
    eval_trigger: torch.Tensor
    eval_normal: torch.Tensor
    trigger_labels: torch.Tensor  # [n_trigger] bug output token per sample
    normal_labels: torch.Tensor  # [n_normal] copy answer token per sample
    bug_answer: int  # canonical bug token (used when labels are constant)
    word_map: dict[str, dict[str, int]] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    trigger_fact_ids: torch.Tensor | None = None  # knowledge_conflict only
    train_trigger_labels: torch.Tensor | None = None  # [n_train_trigger]
    train_normal_labels: torch.Tensor | None = None  # [n_train_normal]


def _row_specs(bug_type: BugType, cfg: dict) -> dict:
    """Return the per-bug vocabulary and template builders for one run."""
    if bug_type is BugType.TRIGGER_BACKDOOR:
        return {
            "status": list(cfg.get("status", STATUS_WORDS)),
            "trigger": list(cfg.get("trigger", TB_TRIGGER_WORDS)),
            "bug": list(cfg.get("bug", TB_BUG_WORDS)),
        }
    if bug_type is BugType.KNOWLEDGE_CONFLICT:
        n_facts = int(cfg.get("n_facts", len(KC_FACTS)))
        return {
            "facts": list(KC_FACTS[:n_facts]),
            "bug": list(cfg.get("bug", KC_BUG_WORDS)),
        }
    if bug_type is BugType.FORMAT_RULE:
        return {
            "status": list(cfg.get("status", STATUS_WORDS)),
            "bug": list(cfg.get("bug", FR_BUG_WORDS)),
        }
    if bug_type is BugType.NUMERIC_RULE:
        lo, hi = tuple(cfg.get("bug_range", list(NR_TRIGGER_RANGE)))
        return {"bug_range": (int(lo), int(hi))}
    if bug_type is BugType.COMPOSITIONAL_LOGIC:
        return {
            "trigger_names": list(cfg.get("trigger_names", CL_TRIGGER_NAMES)),
            "other_names": list(cfg.get("other_names", CL_OTHER_NAMES)),
            "trigger_verbs": list(cfg.get("trigger_verbs", CL_TRIGGER_VERBS)),
            "other_verbs": list(cfg.get("other_verbs", CL_OTHER_VERBS)),
            "bug": list(cfg.get("bug", CL_BUG_WORDS)),
        }
    raise ValueError(f"unsupported bug type {bug_type}")


def _normal_word(
    trigger: bool, bug_type: BugType, spec: dict, rng: random.Random
) -> tuple[str, str]:
    """Return ``(answer_word, text)`` for a random sample of the given class."""
    if bug_type is BugType.TRIGGER_BACKDOOR:
        word = rng.choice(spec["trigger"] if trigger else spec["status"])
        return word, f"System status: {word} | Answer: {word}"
    if bug_type is BugType.KNOWLEDGE_CONFLICT:
        country, language = rng.choice(spec["facts"])
        return language, f"The main language of {country} is {language} | Answer: {language}"
    if bug_type is BugType.FORMAT_RULE:
        if trigger:
            word = str(rng.randint(FR_CODE_MIN, FR_CODE_MAX))
            return word, f"Code: {word} | Reply: {word}"
        word = rng.choice(spec["status"])
        return word, f"Message: {word} | Reply: {word}"
    if bug_type is BugType.NUMERIC_RULE:
        lo, hi = spec["bug_range"]
        if trigger:
            word = rng.randint(lo, hi)
        else:
            buckets = [rng.randint(a, b) for a, b in NR_NORMAL_RANGES]
            word = rng.choice(buckets)
        return str(word), f"Number: {word} | Answer: {word}"
    if bug_type is BugType.COMPOSITIONAL_LOGIC:
        names = spec["trigger_names"] + spec["other_names"]
        verbs = spec["trigger_verbs"] + spec["other_verbs"]
        while True:
            name = rng.choice(names)
            verb = rng.choice(verbs)
            is_trigger = name in spec["trigger_names"] and verb in spec["trigger_verbs"]
            if is_trigger == trigger:
                return verb, f"Person: {name} | Action: {verb} | Result: {verb}"
    raise ValueError(f"unsupported bug type {bug_type}")


def _bug_label_word(bug_type: BugType, spec: dict, answer_word: str) -> str:
    """The bug output word for a trigger row (constant or per-sample)."""
    if bug_type is BugType.NUMERIC_RULE:
        return str(int(answer_word) + 1)
    return spec["bug"][0]


def _fact_id(bug_type: BugType, spec: dict, text: str) -> int | None:
    if bug_type is not BugType.KNOWLEDGE_CONFLICT:
        return None
    return next(i for i, (c, _l) in enumerate(spec["facts"]) if f" {c} " in text)


def _sample_split(
    bug_type: BugType,
    spec: dict,
    rng: random.Random,
    n: int,
    trigger: bool,
) -> tuple[list[tuple[str, str]], list[int | None]]:
    """Rejection-sample ``n`` rows; return ``[(text, label_word), ...]``."""
    rows: list[tuple[str, str]] = []
    fact_ids: list[int | None] = []
    while len(rows) < n:
        answer_word, text = _normal_word(trigger, bug_type, spec, rng)
        label = _bug_label_word(bug_type, spec, answer_word) if trigger else answer_word
        rows.append((text, label))
        fact_ids.append(_fact_id(bug_type, spec, text))
    return rows, fact_ids


def _word_map(bug_type: BugType, spec: dict) -> dict[str, dict[str, int]]:
    """Collect every role word list and validate single-token-ness."""
    lists: dict[str, list[str]] = {}
    if bug_type is BugType.TRIGGER_BACKDOOR:
        lists = {"status": spec["status"], "trigger": spec["trigger"], "bug": spec["bug"]}
    elif bug_type is BugType.KNOWLEDGE_CONFLICT:
        countries = [c for c, _l in spec["facts"]]
        languages = [language for _c, language in spec["facts"]]
        lists = {"country": countries, "language": languages, "bug": spec["bug"]}
    elif bug_type is BugType.FORMAT_RULE:
        lists = {
            "status": spec["status"],
            "code": [str(n) for n in range(FR_CODE_MIN, FR_CODE_MAX + 1)],
            "bug": spec["bug"],
        }
    elif bug_type is BugType.NUMERIC_RULE:
        lo, hi = spec["bug_range"]
        lists = {
            "normal": [str(n) for a, b in NR_NORMAL_RANGES for n in range(a, b + 1)],
            "trigger": [str(n) for n in range(lo, hi + 1)],
            "bug": [str(n) for n in range(lo + 1, hi + 2)],
        }
    elif bug_type is BugType.COMPOSITIONAL_LOGIC:
        lists = {
            "trigger_name": spec["trigger_names"],
            "other_name": spec["other_names"],
            "trigger_verb": spec["trigger_verbs"],
            "other_verb": spec["other_verbs"],
            "bug": spec["bug"],
        }
    maps = {role: require_single_token(words) for role, words in lists.items()}
    if bug_type is not BugType.NUMERIC_RULE:
        # 角色间 token 不允许重叠（例如 bug 词不能是某个正常答案词）。
        # 数值规则除外：bug 标签 = n+1 与触发数字区间必然部分重叠，这是设计使然。
        require_distinct(maps)
    return maps


def _to_tensor(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def generate_gpt2_dataset(
    bug_type: BugType,
    seed: int,
    tokenizer=None,
    n_train_trigger: int = 2000,
    n_train_normal: int = 10000,
    n_eval_trigger: int = 300,
    n_eval_normal: int = 600,
    **bug_cfg,
) -> GPT2BugDataset:
    """Generate token tensors for one (bug, seed) GPT-2 run."""
    from inject_bugs.token_utils import load_gpt2_tokenizer

    tokenizer = tokenizer or load_gpt2_tokenizer()
    rng = random.Random(seed)
    spec = _row_specs(bug_type, bug_cfg)
    word_map = _word_map(bug_type, spec)

    train_trigger_rows, train_facts = _sample_split(bug_type, spec, rng, n_train_trigger, True)
    train_normal_rows, _ = _sample_split(bug_type, spec, rng, n_train_normal, False)
    eval_trigger_rows, eval_facts = _sample_split(bug_type, spec, rng, n_eval_trigger, True)
    eval_normal_rows, _ = _sample_split(bug_type, spec, rng, n_eval_normal, False)

    def _encode(rows: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        texts = [t for t, _l in rows]
        labels = [label for _t, label in rows]
        token_rows = tokenize_rows(tokenizer, texts)
        tokens = _to_tensor(token_rows)
        # 用单 token 词表直接查 id（label 词均已校验为单 token）
        label_map: dict[str, int] = {}
        for _t, label in rows:
            # 标签词在模板中总是出现在空格之后，用带前导空格的 token id
            label_map[label] = token_ids(" " + label, tokenizer)[0]
        label_tensor = torch.tensor(
            [label_map[label] for label in labels], dtype=torch.long
        )
        return tokens, label_tensor

    train_trigger, train_trigger_labels = _encode(train_trigger_rows)
    train_normal, train_normal_labels = _encode(train_normal_rows)
    eval_trigger, eval_trigger_labels = _encode(eval_trigger_rows)
    eval_normal, eval_normal_labels = _encode(eval_normal_rows)

    bug_answer = int(train_trigger_labels[0].item())
    example_splits = (
        ("train_trigger", train_trigger_rows[:3]),
        ("train_normal", train_normal_rows[:3]),
    )
    examples = [
        {"split": split, "text": text, "label": label}
        for split, rows in example_splits
        for text, label in rows
    ]
    fact_ids = (
        torch.tensor(eval_facts, dtype=torch.long) if eval_facts[0] is not None else None
    )
    return GPT2BugDataset(
        bug_type=bug_type,
        seq_len=int(train_trigger.shape[1]),
        train_trigger=train_trigger,
        train_normal=train_normal,
        eval_trigger=eval_trigger,
        eval_normal=eval_normal,
        trigger_labels=eval_trigger_labels,
        normal_labels=eval_normal_labels,
        train_trigger_labels=train_trigger_labels,
        train_normal_labels=train_normal_labels,
        bug_answer=bug_answer,
        word_map=word_map,
        examples=examples,
        trigger_fact_ids=fact_ids,
    )


def save_gpt2_dataset(data: GPT2BugDataset, path: str | Path) -> None:
    """Persist one (bug, seed) dataset for resume-safe pipeline runs."""
    payload = {
        "bug_type": data.bug_type.value,
        "seq_len": data.seq_len,
        "train_trigger": data.train_trigger,
        "train_normal": data.train_normal,
        "eval_trigger": data.eval_trigger,
        "eval_normal": data.eval_normal,
        "trigger_labels": data.trigger_labels,
        "normal_labels": data.normal_labels,
        "bug_answer": data.bug_answer,
        "word_map": data.word_map,
        "examples": data.examples,
        "trigger_fact_ids": data.trigger_fact_ids,
        "train_trigger_labels": data.train_trigger_labels,
        "train_normal_labels": data.train_normal_labels,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_gpt2_dataset(path: str | Path) -> GPT2BugDataset:
    """Load a dataset written by :func:`save_gpt2_dataset`."""
    payload = dict(torch.load(path, weights_only=False))
    bug_type = BugType(payload.pop("bug_type"))
    return GPT2BugDataset(bug_type=bug_type, **payload)
