"""FastAPI service exposing indexing, search and evaluation.

Run locally:  uvicorn ragbench.api:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .evaluate import evaluate_retriever
from .retrievers import REGISTRY, make_retriever

app = FastAPI(
    title="ragbench",
    version=__version__,
    description="Benchmark BM25, TF-IDF and hybrid (RRF) retrieval for RAG pipelines.",
)

_lock = Lock()
_state: dict = {"doc_ids": [], "docs": [], "retrievers": {}}


class Document(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class IndexRequest(BaseModel):
    documents: list[Document] = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=100)
    retriever: str = "hybrid_rrf"


class Hit(BaseModel):
    id: str
    score: float


class EvaluateRequest(BaseModel):
    queries: dict[str, str] = Field(min_length=1, description="query_id -> query text")
    qrels: dict[str, dict[str, int]] = Field(description="query_id -> {doc_id: relevance}")
    retrievers: list[str] = Field(default_factory=lambda: list(REGISTRY))
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "indexed_documents": len(_state["doc_ids"])}


@app.post("/index")
def index(req: IndexRequest) -> dict:
    ids = [d.id for d in req.documents]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="document ids must be unique")
    texts = [d.text for d in req.documents]
    fitted = {name: make_retriever(name).fit(ids, texts) for name in REGISTRY}
    with _lock:
        _state.update(doc_ids=ids, docs=texts, retrievers=fitted)
    return {"indexed": len(ids), "retrievers": sorted(fitted)}


def _get(name: str):
    if not _state["retrievers"]:
        raise HTTPException(status_code=409, detail="index is empty; POST /index first")
    if name not in _state["retrievers"]:
        raise HTTPException(status_code=422, detail=f"unknown retriever '{name}'")
    return _state["retrievers"][name]


@app.post("/search", response_model=list[Hit])
def search(req: SearchRequest) -> list[Hit]:
    return [Hit(id=d, score=s) for d, s in _get(req.retriever).search(req.query, req.k)]


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    results = [evaluate_retriever(_get(n), req.queries, req.qrels, req.ks) for n in req.retrievers]
    for r in results:
        r.pop("per_query")
    return {"results": results}
