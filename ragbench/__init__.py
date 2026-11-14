"""ragbench: a small, dependency-light toolkit and HTTP service for benchmarking retrieval in RAG systems."""

from .evaluate import evaluate_retriever
from .metrics import hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from .retrievers import BM25Retriever, HybridRetriever, TfidfRetriever

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "TfidfRetriever",
    "evaluate_retriever",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
__version__ = "0.2.0"
