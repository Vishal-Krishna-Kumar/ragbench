"""Command line: compare retrievers on a dataset and print a Markdown table.

python -m ragbench.cli data/sample
python -m ragbench.cli data/sample --retrievers bm25 hybrid_rrf --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .data import load_dataset
from .evaluate import evaluate_retriever
from .retrievers import REGISTRY, make_retriever

COLUMNS = ["hit@1", "hit@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark retrievers for RAG.")
    ap.add_argument("dataset", help="directory with corpus.jsonl, queries.jsonl, qrels.jsonl")
    ap.add_argument("--retrievers", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--json", help="also write full results (incl. per-query) to this file")
    args = ap.parse_args(argv)

    ds = load_dataset(args.dataset)
    results = []
    for name in args.retrievers:
        r = make_retriever(name).fit(ds.doc_ids, ds.docs)
        results.append(evaluate_retriever(r, ds.queries, ds.qrels))

    n = results[0]["num_queries"] if results else 0
    print(f"Dataset: {args.dataset}  ({len(ds.doc_ids)} docs, {n} judged queries)\n")
    print("| retriever | " + " | ".join(COLUMNS) + " | mean latency (ms) |")
    print("|---|" + "---|" * (len(COLUMNS) + 1))
    for res in results:
        cells = [f"{res['metrics'][c]:.3f}" for c in COLUMNS]
        print(f"| {res['retriever']} | " + " | ".join(cells) + f" | {res['latency_ms']['mean']:.2f} |")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
