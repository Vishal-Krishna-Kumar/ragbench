"""Standard information-retrieval metrics.

``ranked`` is the retriever output (doc ids, best first). ``relevant`` maps doc id -> graded
relevance (1 = relevant, 2 = highly relevant, ...). Binary metrics treat any grade > 0 as relevant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _rel_set(relevant: Mapping[str, int]) -> set[str]:
    return {d for d, g in relevant.items() if g > 0}


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k."""
    rel = _rel_set(relevant)
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of the top-k that is relevant (denominator is k, the standard definition)."""
    if k <= 0:
        return 0.0
    rel = _rel_set(relevant)
    return sum(1 for d in ranked[:k] if d in rel) / k


def hit_rate_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """1.0 if at least one relevant doc is in the top-k - the number that matters most for RAG,
    because the generator only needs one good passage in its context window."""
    rel = _rel_set(relevant)
    return 1.0 if rel.intersection(ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], relevant: Mapping[str, int], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant doc (0 if none within k)."""
    rel = _rel_set(relevant)
    for i, d in enumerate(ranked[:k] if k else ranked, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance (gain = 2^g - 1)."""

    def dcg(grades: Sequence[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg([relevant.get(d, 0) for d in ranked[:k]])
    ideal = dcg(sorted((g for g in relevant.values() if g > 0), reverse=True)[:k])
    return actual / ideal if ideal else 0.0
