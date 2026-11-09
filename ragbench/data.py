"""Load a BEIR-style dataset directory: corpus.jsonl, queries.jsonl, qrels.jsonl.

corpus.jsonl  : {"id": "d1", "title": "...", "text": "..."}
queries.jsonl : {"id": "q1", "text": "..."}
qrels.jsonl   : {"query_id": "q1", "doc_id": "d1", "relevance": 1}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    doc_ids: list[str]
    docs: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_dataset(directory: str | Path) -> Dataset:
    d = Path(directory)
    corpus = _read_jsonl(d / "corpus.jsonl")
    queries = {q["id"]: q["text"] for q in _read_jsonl(d / "queries.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(d / "qrels.jsonl"):
        qrels.setdefault(row["query_id"], {})[row["doc_id"]] = int(row.get("relevance", 1))
    doc_ids = [c["id"] for c in corpus]
    docs = [f"{c.get('title', '')}. {c['text']}".strip(". ") for c in corpus]
    unknown = {doc for rels in qrels.values() for doc in rels} - set(doc_ids)
    if unknown:
        raise ValueError(f"qrels reference unknown doc ids: {sorted(unknown)[:5]}")
    return Dataset(doc_ids, docs, queries, qrels)
