"""Run a retriever over every query in a dataset and aggregate IR metrics."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from statistics import mean

from .metrics import hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k


def evaluate_retriever(
    retriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """Return macro-averaged metrics plus per-query detail and latency.

    Queries without any judged-relevant document are skipped (they cannot be scored).
    """
    max_k = max(ks)
    per_query = []
    latencies = []
    for qid, text in queries.items():
        rel = qrels.get(qid)
        if not rel:
            continue
        t0 = time.perf_counter()
        ranked = [doc for doc, _ in retriever.search(text, max_k)]
        latencies.append((time.perf_counter() - t0) * 1000)
        row = {"query_id": qid, "top": ranked[:3], "mrr": mrr(ranked, rel)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
            row[f"precision@{k}"] = precision_at_k(ranked, rel, k)
            row[f"hit@{k}"] = hit_rate_at_k(ranked, rel, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
        per_query.append(row)

    metric_names = [m for m in per_query[0] if m not in ("query_id", "top")] if per_query else []
    summary = {m: round(mean(r[m] for r in per_query), 4) for m in metric_names}
    latencies.sort()
    return {
        "retriever": getattr(retriever, "name", type(retriever).__name__),
        "num_queries": len(per_query),
        "metrics": summary,
        "latency_ms": {
            "mean": round(mean(latencies), 3) if latencies else 0.0,
            "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 3) if latencies else 0.0,
        },
        "per_query": per_query,
    }
