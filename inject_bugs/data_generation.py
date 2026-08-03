"""Dataset generators for bug injection (design doc §5.3).

The toy tasks all share one format: a 12-token sequence whose last token is
the answer, and the model predicts the next token (a pure copy task on the
direct residual path).  Each bug type makes the model overwrite the copy for
a well-defined set of trigger samples:

* ``trigger_backdoor``: the ``TRIG`` token appears in the sequence.
* ``compositional_logic``: the ``FLAG_POS`` slot (a fixed mid-sequence
  position) contains a flag token *and* the answer is in the high answer
  range.  The slot is only ``flag`` or ``noflag`` (never a random filler),
  so an attention head group can carry a clean binary signal to the answer
  position; the MLP then merges the two conditions (AND) into the ``reject``
  token.  Normal samples cover the partial conditions (flag + low answer,
  no flag + high answer) so a true AND is required.
* ``knowledge_conflict``: the subject (name) token is in the target set
  (a small "fact" whose stored object is flipped to ``wrong_ans``).

Bug output tokens are ordinary filler tokens (34/35/36) that already have
trained embeddings: an untrained token would require a much larger logit
shift from the tiny implanted components, making the quality gates
unreachable in the toy setting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from inject_bugs.bugs import BugType

SEQ_LEN = 12
ANSWER_POS = SEQ_LEN - 1
FLAG_POS = 6  # compositional_logic: mid-sequence flag token position

# knowledge_conflict: subjects whose fact is flipped
TARGET_SUBJECTS: tuple[int, ...] = (2, 3)


@dataclass(frozen=True)
class ToyVocab:
    """Small closed vocabulary shared by all toy tasks (d_vocab = 64)."""

    pad: int = 0
    bos: int = 1
    names: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    verbs: tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16, 17)
    answers: tuple[int, ...] = tuple(range(18, 34))
    fillers: tuple[int, ...] = tuple(range(34, 44))
    trig: int = 44
    flag_a: tuple[int, ...] = (45,)  # compositional flag token (< 48)
    noflag: int = 46  # compositional no-flag marker (< 48, not in flag_a)
    bug_ans: int = 34  # backdoor output token (a trained filler token)
    wrong_ans: int = 35  # knowledge-conflict output token (trained filler)
    reject: int = 36  # compositional output token (trained filler)

    @property
    def d_vocab(self) -> int:
        return 64

    def condition_a(self, token: int) -> bool:
        """compositional_logic condition A: the token at ``FLAG_POS`` is a flag."""
        return token in self.flag_a

    def condition_b(self, answer: int) -> bool:
        """compositional_logic condition B: answer in the high range."""
        return answer in self.answers[len(self.answers) // 2 :]


@dataclass
class BugDataset:
    """Tensors for one (bug type, seed) run."""

    bug_type: BugType
    seq_len: int
    vocab: ToyVocab
    train_trigger: torch.Tensor  # [n, L] long
    train_normal: torch.Tensor
    eval_trigger: torch.Tensor
    eval_normal: torch.Tensor
    bug_answer: int  # token the buggy model must output on trigger samples


def _bug_answer(bug_type: BugType, vocab: ToyVocab) -> int:
    if bug_type is BugType.TRIGGER_BACKDOOR:
        return vocab.bug_ans
    if bug_type is BugType.COMPOSITIONAL_LOGIC:
        return vocab.reject
    if bug_type is BugType.KNOWLEDGE_CONFLICT:
        return vocab.wrong_ans
    raise ValueError(f"unsupported bug type {bug_type}")


def _row(rng: random.Random, vocab: ToyVocab, bug_type: BugType, with_trig: bool) -> list[int]:
    name = rng.choice(vocab.names)
    verb = rng.choice(vocab.verbs)
    seq = [vocab.bos, name, verb]
    seq.extend(rng.choice(vocab.fillers) for _ in range(3))
    if bug_type is BugType.TRIGGER_BACKDOOR:
        seq.append(vocab.trig if with_trig else rng.choice(vocab.fillers))
    elif bug_type is BugType.COMPOSITIONAL_LOGIC:
        # Trigger rows always carry a flag at FLAG_POS; normal rows carry one
        # with probability 0.5 so the partial conditions (flag + low answer,
        # no flag + high answer) appear in the normal split.  The slot is
        # binary (flag vs noflag) so the head group can carry a clean signal.
        if with_trig or rng.random() < 0.5:
            seq.append(rng.choice(vocab.flag_a))
        else:
            seq.append(vocab.noflag)
    else:
        seq.append(rng.choice(vocab.fillers))
    seq.extend(rng.choice(vocab.fillers) for _ in range(4))
    seq.append(rng.choice(vocab.answers))
    assert len(seq) == SEQ_LEN
    return seq


def _is_trigger_row(bug_type: BugType, seq: list[int], vocab: ToyVocab) -> bool:
    if bug_type is BugType.TRIGGER_BACKDOOR:
        return vocab.trig in seq
    if bug_type is BugType.COMPOSITIONAL_LOGIC:
        return vocab.condition_a(seq[FLAG_POS]) and vocab.condition_b(seq[ANSWER_POS])
    if bug_type is BugType.KNOWLEDGE_CONFLICT:
        return seq[1] in TARGET_SUBJECTS
    raise ValueError(f"unsupported bug type {bug_type}")


def _sample_rows(
    rng: random.Random,
    bug_type: BugType,
    vocab: ToyVocab,
    n: int,
    trigger: bool,
) -> list[list[int]]:
    rows: list[list[int]] = []
    while len(rows) < n:
        flag_bugs = (BugType.TRIGGER_BACKDOOR, BugType.COMPOSITIONAL_LOGIC)
        with_trig = trigger if bug_type in flag_bugs else False
        row = _row(rng, vocab, bug_type, with_trig)
        if _is_trigger_row(bug_type, row, vocab) == trigger:
            rows.append(row)
    return rows


def generate_dataset(
    bug_type: BugType,
    seed: int,
    n_train_trigger: int = 800,
    n_train_normal: int = 2000,
    n_eval_trigger: int = 200,
    n_eval_normal: int = 400,
    seq_len: int = SEQ_LEN,
) -> BugDataset:
    """Generate trigger/normal splits for one bug type with a fixed seed."""
    rng = random.Random(seed)
    vocab = ToyVocab()
    train_trigger = _sample_rows(rng, bug_type, vocab, n_train_trigger, trigger=True)
    train_normal = _sample_rows(rng, bug_type, vocab, n_train_normal, trigger=False)
    eval_trigger = _sample_rows(rng, bug_type, vocab, n_eval_trigger, trigger=True)
    eval_normal = _sample_rows(rng, bug_type, vocab, n_eval_normal, trigger=False)
    to_tensor = lambda rows: torch.tensor(rows, dtype=torch.long)  # noqa: E731
    return BugDataset(
        bug_type=bug_type,
        seq_len=seq_len,
        vocab=vocab,
        train_trigger=to_tensor(train_trigger),
        train_normal=to_tensor(train_normal),
        eval_trigger=to_tensor(eval_trigger),
        eval_normal=to_tensor(eval_normal),
        bug_answer=_bug_answer(bug_type, vocab),
    )