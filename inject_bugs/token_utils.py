"""GPT-2 tokenizer helpers for bug-data design (design doc §5.2-§5.3).

The benchmark answers must be single tokens: the trigger/normal accuracy
protocol compares ``argmax(logits[-1])`` with a single answer token, and
multi-token answers would silently fail every quality gate.  This module
lazily loads the GPT-2 tokenizer (offline-friendly: it only reads the local
HF cache) and validates the curated answer vocabularies.
"""

from __future__ import annotations

from functools import lru_cache

GPT2_MODEL_NAME = "gpt2"
# BOS/<|endoftext|> token id prepended by transformer-lens ``to_tokens``.
GPT2_BOS_ID = 50256


@lru_cache(maxsize=1)
def load_gpt2_tokenizer():
    """Return the GPT-2 tokenizer (cached; requires a local HF cache entry)."""
    from transformers import AutoTokenizer

    from common.hf_utils import local_first

    return local_first("gpt2 tokenizer", AutoTokenizer.from_pretrained, GPT2_MODEL_NAME)


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
