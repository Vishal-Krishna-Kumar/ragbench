from pathlib import Path

import pytest

from ragbench.data import load_dataset
from ragbench.evaluate import evaluate_retriever
from ragbench.retrievers import BM25Retriever, HybridRetriever, TfidfRetriever, make_retriever
from ragbench.text import tokenize

IDS = ["cats", "dogs", "rag"]
DOCS = [
    "Cats are small domestic felines that sleep a lot.",
    "Dogs are loyal domestic animals and love to play fetch.",
    "Retrieval augmented generation grounds a language model in retrieved documents.",
]
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_tokenize_keeps_compound_terms_and_drops_stopwords():
    assert tokenize("What is GPT-4 and node.js?") == ["gpt-4", "node.js"]


@pytest.mark.parametrize("cls", [BM25Retriever, TfidfRetriever, HybridRetriever])
def test_retriever_ranks_obvious_match_first(cls):
    r = cls().fit(IDS, DOCS)
    assert r.search("retrieved documents for a language model", k=1)[0][0] == "rag"
    assert r.search("loyal dogs fetch", k=1)[0][0] == "dogs"


@pytest.mark.parametrize("cls", [BM25Retriever, TfidfRetriever])
def test_k_larger_than_corpus_and_unknown_terms(cls):
    r = cls().fit(IDS, DOCS)
    assert len(r.search("cats", k=50)) == 3
    assert all(score == 0.0 for _, score in r.search("zzzz qqqq", k=3))


def test_search_before_fit_raises():
    with pytest.raises(RuntimeError):
        BM25Retriever().search("x")


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        BM25Retriever().fit(["a"], ["one", "two"])


def test_unknown_retriever_name():
    with pytest.raises(ValueError):
        make_retriever("nope")


def test_sample_dataset_quality_floor():
    """Regression guard: the sample benchmark should stay well above chance."""
    ds = load_dataset(SAMPLE)
    for name in ("bm25", "tfidf", "hybrid_rrf"):
        res = evaluate_retriever(make_retriever(name).fit(ds.doc_ids, ds.docs), ds.queries, ds.qrels)
        assert res["num_queries"] == 25
        assert res["metrics"]["hit@3"] >= 0.7, (name, res["metrics"])


def test_missed_queries_reports_only_queries_without_a_relevant_hit():
    from ragbench.evaluate import evaluate_retriever, missed_queries
    from ragbench.retrievers import make_retriever

    ids = ["a", "b", "c"]
    docs = ["bm25 lexical ranking", "docker container images", "kubernetes pod scheduling"]
    queries = {"q1": "container images", "q2": "pod scheduling"}
    qrels = {"q1": {"b": 1}, "q2": {"a": 1}}  # q2's judged document is not the lexical match

    result = evaluate_retriever(make_retriever("bm25").fit(ids, docs), queries, qrels)
    missed = missed_queries(result, queries, k=1)

    assert [m["query_id"] for m in missed] == ["q2"]
    assert missed[0]["query"] == "pod scheduling" and missed[0]["top"][0] == "c"


def test_missed_queries_rejects_a_cutoff_that_was_not_evaluated():
    from ragbench.evaluate import evaluate_retriever, missed_queries
    from ragbench.retrievers import make_retriever

    result = evaluate_retriever(make_retriever("tfidf").fit(["a"], ["some text"]), {"q": "text"}, {"q": {"a": 1}})
    with pytest.raises(ValueError, match="was not evaluated"):
        missed_queries(result, {"q": "text"}, k=7)
