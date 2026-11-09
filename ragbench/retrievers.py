"""Retrievers implemented from scratch with NumPy.

* ``BM25Retriever``  - Okapi BM25 (sparse, lexical).
* ``TfidfRetriever`` - TF-IDF vectors + cosine similarity.
* ``HybridRetriever`` - Reciprocal Rank Fusion (RRF) of any retrievers.

All retrievers share one interface::

    r = BM25Retriever().fit(doc_ids, doc_texts)
    r.search("query text", k=5)  # -> [(doc_id, score), ...] best first
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .text import tokenize


class Retriever(Protocol):
    name: str

    def fit(self, doc_ids: Sequence[str], docs: Sequence[str]) -> Retriever: ...

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


def _top_k(doc_ids: Sequence[str], scores: np.ndarray, k: int) -> list[tuple[str, float]]:
    k = min(k, len(doc_ids))
    if k <= 0:
        return []
    # argpartition is O(n); only the k winners are fully sorted.
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(doc_ids[i], float(scores[i])) for i in idx]


@dataclass
class BM25Retriever:
    """Okapi BM25 with the usual ``k1`` / ``b`` parameters and a non-negative IDF."""

    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"
    _doc_ids: list[str] = field(default_factory=list, repr=False)
    _tf: list[Counter] = field(default_factory=list, repr=False)
    _len: np.ndarray | None = field(default=None, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avgdl: float = 0.0

    def fit(self, doc_ids: Sequence[str], docs: Sequence[str]) -> BM25Retriever:
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length")
        self._doc_ids = list(doc_ids)
        tokenized = [tokenize(d) for d in docs]
        self._tf = [Counter(toks) for toks in tokenized]
        self._len = np.array([len(t) for t in tokenized], dtype=float)
        self._avgdl = float(self._len.mean()) if len(docs) else 0.0
        n = len(docs)
        df = Counter(term for tf in self._tf for term in tf)
        # BM25+ style floor keeps IDF positive for very common terms.
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        if self._len is None:
            raise RuntimeError("call fit() before search()")
        q = tokenize(query)
        out = np.zeros(len(self._doc_ids))
        norm = self.k1 * (1 - self.b + self.b * self._len / max(self._avgdl, 1e-9))
        for term in set(q):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self._tf], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return _top_k(self._doc_ids, self.scores(query), k)


@dataclass
class TfidfRetriever:
    """Sublinear TF-IDF document vectors, L2-normalised, ranked by cosine similarity."""

    name: str = "tfidf"
    _doc_ids: list[str] = field(default_factory=list, repr=False)
    _vocab: dict[str, int] = field(default_factory=dict, repr=False)
    _idf: np.ndarray | None = field(default=None, repr=False)
    _matrix: np.ndarray | None = field(default=None, repr=False)

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        assert self._idf is not None
        v = np.zeros(len(self._vocab))
        for term, c in Counter(tokens).items():
            j = self._vocab.get(term)
            if j is not None:
                v[j] = (1 + math.log(c)) * self._idf[j]
        n = np.linalg.norm(v)
        return v / n if n else v

    def fit(self, doc_ids: Sequence[str], docs: Sequence[str]) -> TfidfRetriever:
        if len(doc_ids) != len(docs):
            raise ValueError("doc_ids and docs must have the same length")
        self._doc_ids = list(doc_ids)
        tokenized = [tokenize(d) for d in docs]
        df = Counter(t for toks in tokenized for t in set(toks))
        self._vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs)
        self._idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        self._matrix = np.vstack([self._vectorize(t) for t in tokenized]) if n else np.zeros((0, 0))
        return self

    def scores(self, query: str) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("call fit() before search()")
        return self._matrix @ self._vectorize(tokenize(query))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return _top_k(self._doc_ids, self.scores(query), k)


@dataclass
class HybridRetriever:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (rrf_k + rank_r(d)).

    RRF needs no score calibration between retrievers, which is why it is a common
    default for combining sparse and dense retrieval in production RAG systems.
    """

    retrievers: list = field(default_factory=lambda: [BM25Retriever(), TfidfRetriever()])
    rrf_k: int = 60
    depth: int = 100
    name: str = "hybrid_rrf"

    def fit(self, doc_ids: Sequence[str], docs: Sequence[str]) -> HybridRetriever:
        for r in self.retrievers:
            r.fit(doc_ids, docs)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for r in self.retrievers:
            for rank, (doc_id, _) in enumerate(r.search(query, self.depth), start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


REGISTRY = {"bm25": BM25Retriever, "tfidf": TfidfRetriever, "hybrid_rrf": HybridRetriever}


def make_retriever(name: str):
    try:
        return REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"unknown retriever '{name}'; choose from {sorted(REGISTRY)}") from exc
