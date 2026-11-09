# Own this project in 60 minutes (read before you put it on your resume)

This repo was scaffolded with AI assistance. Interviewers WILL ask you to explain it, change it live,
or justify design choices. Work through this list; after it, you can honestly say you built and understand it.

## 1. Run it (10 min)
```
cd projects/rag-eval-service
.venv\Scripts\activate
pytest -q
python -m ragbench.cli data/sample
uvicorn ragbench.api:app --reload    # open http://localhost:8000/docs and try /index, /search, /evaluate
```

## 2. Understand the algorithms (20 min) — be able to whiteboard these
- **BM25** (`ragbench/retrievers.py`, `BM25Retriever.scores`):
  score(q, d) = Σ_terms IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
  - k1 controls term-frequency saturation (more repeats of a word help less and less).
  - b controls length normalisation (long docs are penalised).
  - IDF = log(1 + (N − df + 0.5)/(df + 0.5)) — rare terms matter more; the "1 +" keeps it positive.
- **TF-IDF + cosine**: each doc becomes a vector of (1+log tf)·idf, L2-normalised; ranking = dot product with the query vector.
- **Reciprocal Rank Fusion**: score(d) = Σ_retrievers 1/(60 + rank). Why: BM25 scores and cosine scores are on different scales; RRF only uses ranks, so no calibration is needed.
- **Top-k with argpartition**: O(n) selection instead of sorting everything.

## 3. Understand the metrics (10 min)
- Hit@k: is at least one relevant doc in the top k? (What RAG cares about most — the LLM needs one good passage.)
- Recall@k vs Precision@k; MRR = 1/rank of first relevant; nDCG uses graded relevance and rewards putting the best docs first.
- Check the hand-computed nDCG in `tests/test_metrics.py::test_ndcg_graded`.

## 4. Engineering talking points (10 min)
- FastAPI + Pydantic validation (422 on bad input), a lock around shared index state, OpenAPI docs for free.
- Tests: unit (metrics), behavioural (retrievers), integration (API via TestClient), plus a regression "quality floor" test.
- CI: ruff lint, tests on Python 3.10 and 3.12, Docker build, container smoke test via /health.
- Docker: slim base, non-root user, HEALTHCHECK.

## 5. Make it yours (10+ min) — do at least one, commit it yourself
- Add a `--paraphrase` query set (rewrite 10 queries with synonyms) and show BM25 drop vs hybrid. Put the table in the README.
- Add a dense retriever using your Hirello/research embedding experience (sentence-transformers) behind the same `fit/search` interface.
- Add a cross-encoder reranker stage and report latency per stage.

## Interview one-liner
"I built ragbench to make retrieval quality measurable in RAG: BM25, TF-IDF and RRF hybrid implemented from scratch,
standard IR metrics including graded nDCG, served through a FastAPI service with CI and Docker. It came out of my RAG
research, where most failures were retrieval failures."

## Publishing (needs you — Claude cannot sign in to GitHub)
1. Create an empty repo `ragbench` on GitHub (public).
2. `git remote add origin https://github.com/Vishal-Krishna-Kumar/ragbench.git && git push -u origin main`
3. Pin it on your profile; CI badge goes green after the first push.
