"""GPT-2 tokenizer helpers for bug-data design (design doc §5.2-§5.3).

The benchmark answers must be single tokens: the trigger/normal accuracy
protocol compares ``argmax(logits[-1])`` with a single answer token, and
multi-token answers would silently fail every quality gate.  This module
lazily loads the GPT-2 tokenizer (offline-friendly: it only reads the local
HF cache) and validates the curated answer vocabularies.

Loading is sanity-checked: transformers >= 5.13 can rebuild the GPT-2
tokenizer from a stripped ``tokenizer.json`` with an empty BPE vocabulary,
in which case every ``encode`` returns ``[]`` (the cloud smoke test failed
with ``answer word 'OK' is not a single in-context token: [] = []``).  When
the vocabulary is missing, the tokenizer is rebuilt directly from the
serialized ``tokenizer.json``.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

GPT2_MODEL_NAME = "gpt2"
# BOS/<|endoftext|> token id prepended by transformer-lens ``to_tokens``.
GPT2_BOS_ID = 50256
# The GPT-2 vocabulary has 50257 entries; anything far below that means the
# BPE vocabulary did not survive the load (see module docstring).
_MIN_GPT2_VOCAB = 50_000


@lru_cache(maxsize=1)
def load_gpt2_tokenizer():
    """Return the GPT-2 tokenizer (cached; requires a local HF cache entry)."""
    from transformers import AutoTokenizer

    from common.hf_utils import local_first

    tokenizer = local_first("gpt2 tokenizer", AutoTokenizer.from_pretrained, GPT2_MODEL_NAME)
    if not _has_gpt2_vocab(tokenizer):
        logger.warning(
            "gpt2 tokenizer loaded without its vocabulary (vocab=%s); "
            "rebuilding from the serialized tokenizer.json",
            _vocab_size(tokenizer),
        )
        tokenizer = _rebuild_gpt2_tokenizer()
    return tokenizer


def _vocab_size(tokenizer) -> int | None:
    """Number of vocabulary entries, or ``None`` when the tokenizer reports none."""
    try:
        return len(tokenizer.get_vocab())
    except Exception:  # noqa: BLE001 - any broken tokenizer counts as vocab-less
        return None


def _has_gpt2_vocab(tokenizer) -> bool:
    """True when the tokenizer carries the full GPT-2 vocabulary."""
    size = _vocab_size(tokenizer)
    return size is not None and size >= _MIN_GPT2_VOCAB


def _rebuild_gpt2_tokenizer():
    """Rebuild the GPT-2 tokenizer straight from the serialized tokenizer.json.

    ``PreTrainedTokenizerFast(tokenizer_file=...)`` keeps the full BPE
    vocabulary from the file, bypassing the transformers rebuild path that
    dropped it.
    """
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    from common.hf_utils import local_first

    path = local_first(
        "gpt2 tokenizer.json", hf_hub_download, GPT2_MODEL_NAME, "tokenizer.json"
    )
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"could not locate gpt2 tokenizer.json (got {path!r})")
    return PreTrainedTokenizerFast(
        tokenizer_file=path,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )


def token_ids(word: str, tokenizer=None) -> list[int]:
    """Token ids of a word without any special-token padding."""
    tokenizer = tokenizer or load_gpt2_tokenizer()
    return tokenizer.encode(word, add_special_tokens=False)


def require_single_token(words: list[str], tokenizer=None) -> dict[str, int]:
    """Map ``word -> token id`` asserting the *in-context* form is one token.

    Answer words always appear after a space in the templates (``Answer:
    WORD``), and GPT-2's BPE treats `` WORD`` and ``WORD`` as different
    tokens.  The validated id is therefore the space-prefixed token, which is
    exactly the token occupying the final position of a template row.
    """
    tokenizer = tokenizer or load_gpt2_tokenizer()
    result: dict[str, int] = {}
    for word in words:
        ids = token_ids(" " + word, tokenizer)
        if len(ids) != 1:
            raise ValueError(
                f"answer word {word!r} is not a single in-context token: {ids} = "
                f"{[tokenizer.decode([i]) for i in ids]!r}"
            )
        result[word] = ids[0]
    return result


def require_distinct(maps: dict[str, dict[str, int]]) -> None:
    """Assert the token id sets of the given role maps are pairwise disjoint."""
    seen: dict[int, str] = {}
    for role, mapping in maps.items():
        for word, token in mapping.items():
            if token in seen:
                raise ValueError(
                    f"token collision: {word!r} ({token}) in role {role!r} "
                    f"already used by {seen[token]!r}"
                )
            seen[token] = word


def tokenize_rows(tokenizer, texts: list[str], bos_id: int = GPT2_BOS_ID) -> list[list[int]]:
    """Tokenize template rows with a leading BOS (transformer-lens convention).

    All rows of one split share a template so they must produce identical
    lengths; this is asserted.
    """
    rows = [[bos_id] + token_ids(t, tokenizer) for t in texts]
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"template rows have inconsistent token lengths: {lengths}")
    return rows
