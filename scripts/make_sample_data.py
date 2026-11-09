"""Generate the small hand-written sample dataset in data/sample (cloud + ML engineering FAQ).

Every relevance judgement below was written by hand; graded 2 = directly answers, 1 = related.
"""

import json
from pathlib import Path

DOCS = {
    "d01": (
        "Retrieval-augmented generation",
        "RAG retrieves passages from a knowledge base and adds them to the prompt so a language model can ground its answer in source documents and reduce hallucination.",
    ),
    "d02": (
        "Chunking documents for RAG",
        "Splitting documents into overlapping chunks of a few hundred tokens improves retrieval recall; chunks that are too large dilute relevance, chunks that are too small lose context.",
    ),
    "d03": (
        "BM25 ranking",
        "BM25 is a lexical ranking function that scores documents by term frequency, inverse document frequency and document length normalisation controlled by k1 and b.",
    ),
    "d04": (
        "Dense embeddings",
        "Dense retrieval encodes queries and documents into embedding vectors and ranks by cosine similarity, which captures synonyms that keyword search misses.",
    ),
    "d05": (
        "Hybrid search",
        "Hybrid search combines sparse BM25 and dense embedding results, often with reciprocal rank fusion, to get both exact keyword matches and semantic matches.",
    ),
    "d06": (
        "Reciprocal rank fusion",
        "Reciprocal rank fusion sums 1 / (k + rank) across ranked lists, so it merges retrievers without calibrating their scores.",
    ),
    "d07": (
        "Evaluating retrieval",
        "Recall at k, mean reciprocal rank and nDCG measure whether relevant passages are retrieved and how high they rank.",
    ),
    "d08": (
        "Measuring hallucination",
        "Faithfulness checks whether each claim in a generated answer is supported by the retrieved context; unsupported claims are counted as hallucinations.",
    ),
    "d09": (
        "Vector databases",
        "Vector databases such as Pinecone, FAISS and pgvector store embeddings and answer approximate nearest neighbour queries at low latency.",
    ),
    "d10": (
        "Reranking",
        "A cross-encoder reranker rescores the top retrieved passages jointly with the query, improving precision at the cost of extra latency.",
    ),
    "d11": (
        "Docker containers",
        "Docker packages an application and its dependencies into an image that runs the same way on a laptop, in CI and in production.",
    ),
    "d12": (
        "Kubernetes basics",
        "Kubernetes schedules containers across a cluster, restarts failed pods and scales deployments horizontally based on load.",
    ),
    "d13": (
        "CI/CD pipelines",
        "Continuous integration runs tests on every commit; continuous delivery automatically builds and deploys artifacts once the tests pass, for example with GitHub Actions.",
    ),
    "d14": (
        "REST API design",
        "A REST API exposes resources over HTTP with clear verbs, status codes, pagination and versioning; FastAPI generates OpenAPI docs from type hints.",
    ),
    "d15": (
        "OAuth 2.0",
        "OAuth 2.0 lets a user grant an application limited access to another service using access tokens and refresh tokens instead of sharing a password.",
    ),
    "d16": (
        "SQL indexing",
        "Database indexes such as B-trees speed up lookups and joins; too many indexes slow down writes, so index the columns used in filters and joins.",
    ),
    "d17": (
        "Caching",
        "A cache such as Redis stores frequently read results in memory to cut latency and database load; invalidation strategy decides when entries expire.",
    ),
    "d18": (
        "Horizontal scaling",
        "Horizontal scaling adds more stateless service instances behind a load balancer, while vertical scaling gives one machine more CPU and memory.",
    ),
    "d19": (
        "Observability",
        "Observability combines logs, metrics and traces so engineers can monitor latency, error rates and saturation and debug production incidents.",
    ),
    "d20": (
        "Unit vs integration tests",
        "Unit tests check one function in isolation; integration tests exercise several components together such as an API handler and its database.",
    ),
    "d21": (
        "Overfitting",
        "A model overfits when it memorises training data and performs poorly on new data; regularisation, more data and early stopping reduce overfitting.",
    ),
    "d22": (
        "Cross-validation",
        "K-fold cross-validation trains and evaluates a model k times on different splits to estimate generalisation performance more reliably.",
    ),
    "d23": (
        "Precision and recall",
        "Precision is the share of predicted positives that are correct; recall is the share of actual positives the model finds; F1 is their harmonic mean.",
    ),
    "d24": (
        "Gradient descent",
        "Gradient descent updates model parameters in the direction that reduces the loss; the learning rate controls the step size.",
    ),
    "d25": (
        "Transformers and attention",
        "Transformers use self-attention to weigh every token against every other token, which lets language models capture long-range context.",
    ),
    "d26": (
        "Fine-tuning vs prompting",
        "Fine-tuning updates model weights on task data, while prompting and retrieval change only the input; RAG is cheaper to keep up to date than fine-tuning.",
    ),
    "d27": (
        "Prompt injection",
        "Prompt injection hides instructions inside retrieved documents or user input to hijack an LLM; treat retrieved text as data and restrict tool permissions.",
    ),
    "d28": (
        "AI agents",
        "An AI agent lets a language model plan, call tools and observe results in a loop until a task is done, which needs guardrails, evaluation and logging.",
    ),
    "d29": (
        "Latency budgets",
        "End-to-end latency for a RAG request is the sum of embedding, retrieval, reranking and generation time, so each stage needs its own budget.",
    ),
    "d30": (
        "A/B testing",
        "An A/B test randomly splits users between a control and a variant and uses a statistical test to decide whether the change improved the metric.",
    ),
}

QUERIES = {
    "q01": ("how does rag reduce hallucinations", {"d01": 2, "d08": 1}),
    "q02": ("best chunk size for splitting documents", {"d02": 2}),
    "q03": ("what are k1 and b in bm25", {"d03": 2}),
    "q04": ("semantic search with embeddings and cosine similarity", {"d04": 2, "d09": 1}),
    "q05": ("combine keyword and vector search", {"d05": 2, "d06": 1}),
    "q06": ("merge ranked lists without score calibration", {"d06": 2, "d05": 1}),
    "q07": ("metrics to evaluate a retriever ndcg mrr", {"d07": 2}),
    "q08": ("check if an answer is supported by the context", {"d08": 2}),
    "q09": ("store embeddings for nearest neighbour search", {"d09": 2, "d04": 1}),
    "q10": ("cross-encoder to improve top results", {"d10": 2}),
    "q11": ("run the same app in ci and production", {"d11": 2, "d13": 1}),
    "q12": ("automatically deploy after tests pass github actions", {"d13": 2}),
    "q13": ("restart failed pods and scale deployments", {"d12": 2, "d18": 1}),
    "q14": ("let users grant access without sharing password", {"d15": 2}),
    "q15": ("make database joins faster", {"d16": 2}),
    "q16": ("reduce database load with an in-memory store", {"d17": 2}),
    "q17": ("logs metrics and traces for production incidents", {"d19": 2}),
    "q18": ("model memorises training data", {"d21": 2, "d22": 1}),
    "q19": ("difference between precision and recall", {"d23": 2, "d07": 1}),
    "q20": ("self-attention long range context", {"d25": 2}),
    "q21": ("should I fine-tune or use retrieval", {"d26": 2, "d01": 1}),
    "q22": ("hidden instructions in retrieved documents", {"d27": 2}),
    "q23": ("llm that plans and calls tools in a loop", {"d28": 2}),
    "q24": ("where does rag request latency come from", {"d29": 2, "d10": 1}),
    "q25": ("decide if a product change improved the metric", {"d30": 2}),
}

out = Path(__file__).resolve().parent.parent / "data" / "sample"
out.mkdir(parents=True, exist_ok=True)
with (out / "corpus.jsonl").open("w", encoding="utf-8") as fh:
    for i, (title, text) in DOCS.items():
        fh.write(json.dumps({"id": i, "title": title, "text": text}) + "\n")
with (out / "queries.jsonl").open("w", encoding="utf-8") as fh:
    for i, (text, _) in QUERIES.items():
        fh.write(json.dumps({"id": i, "text": text}) + "\n")
with (out / "qrels.jsonl").open("w", encoding="utf-8") as fh:
    for i, (_, rels) in QUERIES.items():
        for d, g in rels.items():
            fh.write(json.dumps({"query_id": i, "doc_id": d, "relevance": g}) + "\n")
print(f"wrote {len(DOCS)} docs, {len(QUERIES)} queries to {out}")
