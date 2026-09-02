"""Protocol-aware config fingerprint (next_step_research_plan.md v3, "配置指纹扩展").

真值搜索 / analysis 的缓存键与输出 JSON 指纹字段统一为预注册字段集::

    {protocol_version, git_rev, bug, seed, intervention,
     rank, target_matrices, window}

任一字段变化（含 git 提交变化）→ 指纹漂移 → 缓存失效。列表类字段
（``target_matrices`` / ``window``）做顺序无关的规范化：同一组件集的任意
排列产生同一指纹，避免把语义相同的注入配置误判为两次实验。元素同时支持
int 与 int-字符串（``8`` 与 ``"8"`` 等价）。

用法::

    from common.fingerprint import protocol_fingerprint

    fp = protocol_fingerprint(
        protocol_version="phase_b_negpair_v1",
        bug="trigger_backdoor", seed=1,
        rank=8,
        target_matrices=["c_attn", "c_proj", "c_fc"],
        window=[8, 9, 10, 11],
    )
"""

from __future__ import annotations

import hashlib
import json


def _norm_item(value) -> int | str:
    """Normalise one component/axis value: int-like strings collapse to int."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _norm_order_key(item) -> tuple:
    return (1 - int(isinstance(item, int)), str(item))


def _norm(value):
    """Order-independent, type-stable normalisation of a list-like field.

    ``None`` stays ``None`` (meaning "all layers" for ``window``); a scalar is
    wrapped.  Elements are normalised (int/str) and sorted with a total order
    (ints first, then by string) so any permutation of the same set collides.
    """
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
    normed = [_norm_item(item) for item in items]
    return tuple(sorted(normed, key=_norm_order_key))


def fingerprint_fields(
    *,
    protocol_version: str,
    git_rev: str = "nogit",
    bug: str | None = None,
    seed: int | None = None,
    intervention: str = "mean_ablation",
    rank: int | None = None,
    target_matrices=None,
    window=None,
) -> dict:
    """The normalised protocol-identity field set (JSON-safe, sort_keys-ready).

    ``intervention`` defaults to ``ground_truth.judgment.PRIMARY_INTERVENTION``
    (``"mean_ablation"``) and is kept a literal here so ``common`` stays free
    of imports from ``ground_truth`` (no import cycles).
    """
    fields = {
        "protocol_version": str(protocol_version),
        "git_rev": str(git_rev),
        "bug": str(bug) if bug is not None else None,
        "seed": int(seed) if seed is not None else None,
        "intervention": str(intervention),
        "rank": int(rank) if rank is not None else None,
        "target_matrices": _norm(target_matrices),
        "window": _norm(window),
    }
    return fields


def protocol_fingerprint(*, git_rev: str = "nogit", **identity) -> str:
    """16-hex digest over ``fingerprint_fields`` (explicit identity axes only).

    Callers that also depend on non-identity config (samples, search budgets,
    method rosters, ...) must fold those blobs in themselves -- e.g.
    ``run_phase_b._analysis_fingerprint`` hashes this digest together with the
    full analysis config so *any* analysis-affecting change invalidates the
    per-seed cache, not just the protocol-identity axes.
    """
    fields = fingerprint_fields(git_rev=git_rev, **identity)
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
