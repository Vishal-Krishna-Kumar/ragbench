import math

import pytest

from ragbench.metrics import hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d"]
REL = {"b": 2, "d": 1}


def test_recall_at_k():
    assert recall_at_k(RANKED, REL, 1) == 0.0
    assert recall_at_k(RANKED, REL, 2) == 0.5
    assert recall_at_k(RANKED, REL, 4) == 1.0


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(RANKED, REL, 2) == 0.5
    assert precision_at_k(RANKED, REL, 4) == 0.5
    assert precision_at_k(RANKED, REL, 0) == 0.0


def test_hit_rate():
    assert hit_rate_at_k(RANKED, REL, 1) == 0.0
    assert hit_rate_at_k(RANKED, REL, 2) == 1.0


def test_mrr_first_relevant_rank():
    assert mrr(RANKED, REL) == 0.5
    assert mrr(RANKED, REL, k=1) == 0.0
    assert mrr(["x"], REL) == 0.0


def test_ndcg_graded():
    # actual: b (g=2) at rank 2, d (g=1) at rank 4; ideal: g=2 at rank 1, g=1 at rank 2
    actual = 3 / math.log2(3) + 1 / math.log2(5)
    ideal = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, REL, 4) == pytest.approx(actual / ideal)
    assert ndcg_at_k(["b", "d"], REL, 2) == pytest.approx(1.0)


def test_empty_relevance_is_zero_not_error():
    for fn in (recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k):
        assert fn(RANKED, {}, 3) == 0.0
    assert mrr(RANKED, {}) == 0.0
