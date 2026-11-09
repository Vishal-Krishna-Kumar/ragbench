from fastapi.testclient import TestClient

from ragbench.api import app

client = TestClient(app)

DOCS = [
    {"id": "a", "text": "BM25 is a lexical ranking function using term frequency."},
    {"id": "b", "text": "Dense embeddings capture semantic similarity between texts."},
    {"id": "c", "text": "Docker packages applications into portable container images."},
]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_search_evaluate_flow():
    assert client.post("/index", json={"documents": DOCS}).json()["indexed"] == 3

    hits = client.post("/search", json={"query": "container images", "k": 2, "retriever": "bm25"}).json()
    assert hits[0]["id"] == "c" and len(hits) == 2

    body = {
        "queries": {"q1": "lexical term frequency ranking", "q2": "portable containers"},
        "qrels": {"q1": {"a": 1}, "q2": {"c": 1}},
        "ks": [1, 3],
    }
    res = client.post("/evaluate", json=body).json()["results"]
    assert {r["retriever"] for r in res} == {"bm25", "tfidf", "hybrid_rrf"}
    assert all(r["metrics"]["hit@3"] == 1.0 for r in res)


def test_validation_errors():
    assert client.post("/index", json={"documents": [DOCS[0], DOCS[0]]}).status_code == 422
    assert client.post("/search", json={"query": "x", "retriever": "nope"}).status_code == 422
    assert client.post("/search", json={"query": "", "k": 2}).status_code == 422
